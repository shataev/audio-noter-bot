"""The conversations the owner can still reply to, in a file that outlives the bot.

The obvious implementation of "replying to a coach message continues the
conversation" is a dict in memory, and it is fine right up to the moment the
process restarts — which it does on every deploy. Then a conversation the owner
was in the middle of is gone, with no message saying so: his next reply either
disappears or starts a fresh thread that has forgotten everything, and both look
like the bot ignoring him. So the chains live in their own JSON file, written the
same atomic way the memory store is written.

A thread is addressed by ``chat_id:message_id`` — one key for *each* message the
coach sent in it, because a long answer is delivered as several Telegram
messages and replying to any of them means the same thing. The keys are the
thread's, not the other way round: one chain, several doors into it.

Three bounds keep the file from growing forever, and all three are deliberately
generous: this is a single-user bot, and the cost of forgetting a conversation
too eagerly is much higher than the cost of a file a few kilobytes larger.

Forgetting is not silent. When a thread is pruned its keys move to ``forgotten``,
so a reply that arrives against a conversation the file no longer holds can be
answered honestly instead of being taken for the first message of a new one with
no context — which would produce a confident answer to a question the coach
cannot see, the worst of the available failures.

Standard library only, and nothing here logs the text of a turn: a thread is
diary material.
"""

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .memory import now_stamp
from .store import write_text_atomically

logger = logging.getLogger(__name__)

VERSION = 1

# At most this many conversations, dropping the least recently touched.
MAX_THREADS = 50
# At most this many messages in one conversation, dropping the oldest. The
# beginning of a long exchange is what the model can most afford to lose.
MAX_TURNS = 40
# Nothing older than this, counted from the last message in the thread.
MAX_AGE_DAYS = 14
# How many keys of pruned threads stay answerable with "I no longer have that
# conversation". Cheap — a key is about twenty bytes — and the alternative is a
# reply to a two-week-old message silently becoming a new conversation.
MAX_FORGOTTEN = 500


class ThreadStoreError(RuntimeError):
    """The file exists and is not this store's document.

    Same rule as the memory store: a file that is not valid JSON, or whose top
    level is not an object, is never read as an empty one, because the next save
    would overwrite whatever it actually was.
    """


@dataclass(frozen=True)
class Turn:
    """One message in a chain. ``role`` is "user" or "assistant", as the API wants."""

    role: str
    text: str


@dataclass(frozen=True)
class Thread:
    """One conversation: how it was started, what was said, and its doors."""

    id: str
    mode: str
    turns: tuple[Turn, ...] = ()
    keys: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class Threads:
    """Every live conversation, plus the keys of the ones that have been dropped.

    Ordered least-recently-touched first, which is the order :func:`prune` drops
    them in. ``next_id`` is persisted rather than derived, for the same reason the
    memory list's is: an id computed as one past the highest can be handed out
    twice after a prune.
    """

    threads: tuple[Thread, ...] = ()
    forgotten: tuple[str, ...] = ()
    next_id: int = 1


def key(chat_id: int, message_id: int) -> str:
    """The address of one message the owner can reply to."""
    return f"{chat_id}:{message_id}"


def find(items: Threads, address: str) -> Thread | None:
    """The conversation that message belongs to, or ``None``."""
    return next((thread for thread in items.threads if address in thread.keys), None)


def was_forgotten(items: Threads, address: str) -> bool:
    """True for a message that was the coach's and whose conversation has been pruned."""
    return address in items.forgotten


def _parsed(stamp: str) -> datetime | None:
    """A stored timestamp, or ``None`` if it is not one.

    Unparseable is not treated as old. Ageing a thread out on a timestamp nobody
    can read is destroying data on the strength of a bug, and the count bound
    below still stops the file growing.
    """
    try:
        parsed = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _forget(forgotten: tuple[str, ...], dropped: Iterable[Thread]) -> tuple[str, ...]:
    keys = [address for thread in dropped for address in thread.keys]
    if not keys:
        return forgotten
    # Deduplicated keeping the newest mention, so a key cannot be pushed out of
    # the window by its own repeats.
    ordered = list(dict.fromkeys([*forgotten, *keys]))
    return tuple(ordered[-MAX_FORGOTTEN:])


def prune(items: Threads, *, now: datetime | None = None) -> Threads:
    """Apply all three bounds, moving the keys of anything dropped to ``forgotten``.

    Called on every load and on every change rather than on a schedule, so the
    bounds hold without anything having to remember to enforce them — including
    the age bound, which is the one that comes due while the bot is not running.
    """
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=MAX_AGE_DAYS)

    fresh: list[Thread] = []
    dropped: list[Thread] = []
    for thread in items.threads:
        touched = _parsed(thread.updated_at)
        if touched is not None and touched < cutoff:
            dropped.append(thread)
        else:
            fresh.append(thread)

    if len(fresh) > MAX_THREADS:
        dropped.extend(fresh[: len(fresh) - MAX_THREADS])
        fresh = fresh[len(fresh) - MAX_THREADS :]

    trimmed = tuple(
        replace(thread, turns=thread.turns[-MAX_TURNS:])
        if len(thread.turns) > MAX_TURNS
        else thread
        for thread in fresh
    )

    if dropped:
        logger.info("coach threads: dropped %d conversation(s) past the bounds", len(dropped))

    return Threads(
        threads=trimmed,
        forgotten=_forget(items.forgotten, dropped),
        next_id=items.next_id,
    )


def start(
    items: Threads,
    *,
    mode: str,
    turns: Sequence[Turn],
    keys: Sequence[str],
    now: datetime | None = None,
) -> tuple[Threads, Thread]:
    """Open a conversation on the messages that have just been sent."""
    stamp = now_stamp(now)
    thread = Thread(
        id=str(items.next_id),
        mode=mode,
        turns=tuple(turns),
        keys=tuple(dict.fromkeys(keys)),
        created_at=stamp,
        updated_at=stamp,
    )
    grown = Threads(
        threads=(*items.threads, thread),
        forgotten=items.forgotten,
        next_id=items.next_id + 1,
    )
    return prune(grown, now=now), thread


def extend(
    items: Threads,
    thread_id: str,
    *,
    turns: Sequence[Turn] = (),
    keys: Sequence[str] = (),
    now: datetime | None = None,
) -> tuple[Threads, Thread | None]:
    """Add to a conversation and move it to the back of the queue.

    Moving it is what makes the count bound mean "least recently used" rather than
    "oldest", so a conversation still being used cannot be pruned out from under
    the owner by fifty single-message ones.
    """
    index = next((i for i, thread in enumerate(items.threads) if thread.id == thread_id), None)
    if index is None:
        logger.info("coach threads: nothing to extend for thread %s", thread_id)
        return items, None

    existing = items.threads[index]
    updated = replace(
        existing,
        turns=(*existing.turns, *turns),
        keys=tuple(dict.fromkeys([*existing.keys, *keys])),
        updated_at=now_stamp(now),
    )
    rest = [*items.threads[:index], *items.threads[index + 1 :]]
    grown = Threads(
        threads=(*rest, updated),
        forgotten=items.forgotten,
        next_id=items.next_id,
    )
    pruned = prune(grown, now=now)
    # Looked up again rather than handed back directly: pruning can trim the
    # turns of the very thread just extended, and the caller must not be given a
    # version of it that is no longer what was written.
    return pruned, next((t for t in pruned.threads if t.id == thread_id), None)


def _turn_from_json(raw: object) -> Turn | None:
    if not isinstance(raw, dict):
        return None
    role = raw.get("role")
    text = raw.get("text")
    if role not in ("user", "assistant") or not isinstance(text, str) or not text.strip():
        return None
    return Turn(role=role, text=text)


def _strings(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, str) and item)


def _thread_from_json(raw: object) -> Thread | None:
    """One conversation, or ``None`` for an entry there is nothing to keep in.

    A thread with no keys is unreachable — nothing can ever reply into it — and a
    thread with no turns has nothing to say, so both are dropped rather than
    loaded into a list that is scanned on every message.
    """
    if not isinstance(raw, dict):
        return None
    thread_id = raw.get("id")
    if not isinstance(thread_id, str) or not thread_id:
        return None

    turns = tuple(turn for turn in map(_turn_from_json, raw.get("turns") or []) if turn)
    keys = _strings(raw.get("keys"))
    if not turns or not keys:
        return None

    mode = raw.get("mode")
    created_at = raw.get("created_at")
    created_at = created_at if isinstance(created_at, str) else ""
    updated_at = raw.get("updated_at")
    return Thread(
        id=thread_id,
        mode=mode if isinstance(mode, str) else "",
        turns=turns,
        keys=keys,
        created_at=created_at,
        updated_at=updated_at if isinstance(updated_at, str) else created_at,
    )


def _from_json(body: dict) -> Threads:
    raw_threads = body.get("threads")
    loaded: list[Thread] = []
    dropped = 0
    for raw in raw_threads if isinstance(raw_threads, list) else []:
        thread = _thread_from_json(raw)
        if thread is None:
            dropped += 1
        else:
            loaded.append(thread)

    # Raised past every id in the file so that a hand-edited counter cannot hand
    # out an id a live thread already has. Only ever raised.
    highest = max((int(t.id) for t in loaded if t.id.isdigit()), default=0)
    raw_next = body.get("next_id")
    next_id = raw_next if isinstance(raw_next, int) and not isinstance(raw_next, bool) else 1

    if dropped:
        logger.warning("coach threads: %d unusable conversation(s) dropped on load", dropped)

    return Threads(
        threads=tuple(loaded),
        forgotten=_strings(body.get("forgotten"))[-MAX_FORGOTTEN:],
        next_id=max(next_id, highest + 1),
    )


def _to_json(items: Threads) -> dict[str, object]:
    return {
        "version": VERSION,
        "next_id": items.next_id,
        "threads": [
            {
                "id": thread.id,
                "mode": thread.mode,
                "created_at": thread.created_at,
                "updated_at": thread.updated_at,
                "keys": list(thread.keys),
                "turns": [{"role": turn.role, "text": turn.text} for turn in thread.turns],
            }
            for thread in items.threads
        ],
        "forgotten": list(items.forgotten),
    }


class ThreadStore:
    """Reads and writes the conversation file. Holds no state of its own."""

    def __init__(self, path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self, *, now: datetime | None = None) -> Threads:
        """The stored conversations, with the bounds already applied.

        Pruning on the way in is what makes the age bound hold across a restart:
        a fortnight with the bot switched off has to expire the same threads a
        fortnight of it running would have.
        """
        try:
            raw_text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return Threads()

        if not raw_text.strip():
            logger.warning("coach threads: %s is empty, starting from nothing", self._path)
            return Threads()

        try:
            body = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise ThreadStoreError(f"{self._path} is not valid JSON: {error}") from error
        if not isinstance(body, dict):
            raise ThreadStoreError(f"{self._path} does not hold a JSON object")

        items = prune(_from_json(body), now=now)
        logger.info(
            "coach threads: loaded %d conversation(s) from %s",
            len(items.threads),
            self._path,
        )
        return items

    def save(self, items: Threads) -> None:
        text = json.dumps(_to_json(items), ensure_ascii=False, indent=2) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomically(self._path, text)
        logger.info(
            "coach threads: wrote %d conversation(s) to %s",
            len(items.threads),
            self._path,
        )
