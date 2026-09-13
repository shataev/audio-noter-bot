"""An accumulated list of facts the model edits one entry at a time.

The coach keeps two of these lists about its owner: an author profile grown from
every diary entry, and the standing behaviour rules he has given the bot. Both
are the same structure, and both are maintained the same way.

The obvious design — hand the model the whole list, ask for the whole list back —
is not the one used here. It grows the request without bound, it loses any fact
the model forgets to repeat, and it costs a full rewrite to fix one word. Instead
every fact carries a stable id and the model answers with *operations* addressed
to those ids: ``create``, ``modify``, ``delete``. A fact nobody mentions survives
by construction, and an operation aimed at an id that no longer exists is a
skipped no-op rather than a wrong edit.

Losing a known fact is the expensive error here. Failing to add a new one is
cheap, because the next diary entry offers it again. Every ambiguous case below
resolves towards keeping data: an unknown ``kind`` becomes ``other`` instead of
dropping the operation, a delete leaves a tombstone instead of a hole, and a
malformed payload leaves the list exactly as it was.

Standard library only, and nothing in here logs the text of a fact — see
``services/coach/__init__.py`` for why both of those are rules rather than
preferences.
"""

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# The two vocabularies for ``Fact.kind``. They are passed in rather than guessed
# from the contents of the list, because an empty rules list and an empty profile
# are indistinguishable and would then diverge on their first create.
PROFILE_KINDS = (
    "trait",
    "bias",
    "value",
    "pattern",
    "relation",
    "work",
    "body",
    "skill",
    "phase",
    "other",
)
RULE_KINDS = ("rule",)

# How many entry references one fact remembers. Provenance is worth keeping, a
# fact mentioned in four hundred entries is not worth keeping four hundred times.
MAX_SOURCES = 20
# How many deleted facts stay recoverable. Two hundred is far more than a year of
# ordinary corrections and still a file a human can open.
MAX_TOMBSTONES = 200

_ACTIONS = frozenset({"create", "modify", "delete"})
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Fact:
    """One short statement about the owner, and where it came from.

    ``created_at``, ``updated_at`` and ``sources`` are what this structure has
    that a list of strings does not, and they are not decoration: a fact can be
    shown with its provenance, a fact that has stopped being true can be aged out
    on evidence rather than on a guess, and a profile that has gone wrong can be
    debugged. A timestamp cannot be retrofitted onto facts that already exist, so
    they are collected from the first one.

    ``key`` is an identity handed over by whatever renders the list somewhere the
    owner can edit it — a Notion block id, today. It is what lets a bullet he has
    reworded by hand come back as the same fact rather than as a new one.
    """

    id: str
    text: str
    kind: str
    created_at: str
    updated_at: str
    sources: tuple[str, ...] = ()
    key: str | None = None


@dataclass(frozen=True)
class Tombstone:
    """A deleted fact, kept so that "why does it no longer know X" has an answer."""

    fact: Fact
    deleted_at: str
    reason: str | None = None


@dataclass(frozen=True)
class MemoryList:
    """One list: its live facts, its tombstones, and its id counter.

    The counter travels with the facts because it is worthless apart from them.
    It is never derived from the facts themselves — see :func:`_mint`.
    """

    facts: tuple[Fact, ...] = ()
    tombstones: tuple[Tombstone, ...] = ()
    next_id: int = 1


@dataclass(frozen=True)
class SkippedOp:
    """An operation that changed nothing, and why.

    Carries the id and the action but never the text: a skipped operation is the
    thing a caller most wants to log, and the text of a fact must not be logged.
    """

    action: str | None
    id: str | None
    reason: str


@dataclass(frozen=True)
class ApplyResult:
    """The new list, plus what happened to it.

    Callers report the change from this rather than by diffing two lists, which
    they cannot do accurately anyway: a modify that rewords a fact and a delete
    followed by a create look identical in a diff and mean different things.
    """

    items: MemoryList
    created: tuple[Fact, ...] = ()
    modified: tuple[Fact, ...] = ()
    deleted: tuple[Fact, ...] = ()
    skipped: tuple[SkippedOp, ...] = ()


@dataclass(frozen=True)
class RestoreResult:
    """The result of pulling one fact back out of the tombstones."""

    items: tuple[Fact, ...]
    tombstones: tuple[Tombstone, ...]
    restored: Fact | None


@dataclass(frozen=True)
class EditedItem:
    """One line of a list as the owner left it after editing it by hand."""

    key: str | None
    text: str


def now_stamp(now: datetime | None = None) -> str:
    """ISO-8601, UTC, second precision — the one timestamp format in this package.

    Second precision because these are diary-scale events and a microsecond tail
    is noise in a file a human reads. ``+00:00`` rather than ``Z`` because
    ``datetime.fromisoformat`` parses it on every version this project supports.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).isoformat(timespec="seconds")


def normalise(text: str) -> str:
    """The form two texts are compared in: case-folded, whitespace collapsed.

    Used for duplicate detection and for matching a hand-edited line back to the
    fact it came from. It is deliberately forgiving in both places, because the
    cost of a false match is a fact keeping its id when it might have earned a new
    one, and the cost of a missed match is a duplicate or a lost history.
    """
    return _WHITESPACE.sub(" ", text).strip().casefold()


def fallback_kind(kinds: Sequence[str]) -> str:
    """The kind a create falls back to when it names one this list does not have."""
    return "other" if "other" in kinds else kinds[0]


def _mint(next_id: int) -> tuple[str, int]:
    """Take the next id and hand back the counter that follows it.

    Ids are minted from a counter that is persisted, never computed as one past
    the highest id currently in the list. The computed version is the obvious
    implementation and it is wrong: delete the highest-numbered fact and the next
    create takes its number, so an operation from a request that was already in
    flight lands on a *different fact than the model was looking at*. Rare,
    silent, and it corrupts exactly the data this feature exists to keep.
    """
    return str(next_id), next_id + 1


def _clean_text(raw: object) -> str | None:
    """The text of an operation, or ``None`` if there isn't one worth storing."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    return text or None


def _clean_id(raw: object) -> str | None:
    """The id an operation addresses, tolerating a model that sent it as a number."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return str(raw)
    if isinstance(raw, str):
        return raw.strip() or None
    return None


def _clean_kind(raw: object, kinds: Sequence[str]) -> str | None:
    """A kind from this list's vocabulary, or ``None`` if it named something else."""
    if not isinstance(raw, str):
        return None
    kind = raw.strip().lower()
    return kind if kind in kinds else None


def _append_source(sources: tuple[str, ...], source: str | None) -> tuple[str, ...]:
    """Record one more entry as evidence for a fact, without repeating it."""
    if not source or source in sources:
        return sources
    return (*sources, source)[-MAX_SOURCES:]


def _entomb(
    tombstones: tuple[Tombstone, ...],
    fact: Fact,
    stamp: str,
    reason: str | None,
) -> tuple[Tombstone, ...]:
    """Move a fact out of the active list without losing it."""
    return (*tombstones, Tombstone(fact=fact, deleted_at=stamp, reason=reason))[-MAX_TOMBSTONES:]


def apply_ops(
    items: MemoryList,
    ops: object,
    *,
    source: str | None = None,
    now: datetime | None = None,
    kinds: Sequence[str] = PROFILE_KINDS,
) -> ApplyResult:
    """Apply the model's operations to a list and report what they did.

    ``ops`` is whatever came back from the model, parsed as JSON and not trusted
    any further than that: anything that is not a list of operation objects is
    skipped and counted, and the list is handed back untouched. ``source`` is a
    reference to the diary entry that produced these operations, recorded on every
    fact they create or change.
    """
    facts = list(items.facts)
    tombstones = items.tombstones
    next_id = items.next_id
    stamp = now_stamp(now)

    created: list[Fact] = []
    modified: list[Fact] = []
    deleted: list[Fact] = []
    skipped: list[SkippedOp] = []

    # A string is iterable, and iterating one would turn "create" into five
    # operations named "c", "r", "e"… Anything that is not a real sequence of
    # operations is one skipped payload, not a reason to touch the list.
    if not isinstance(ops, (list, tuple)):
        logger.info("coach memory: ignoring an ops payload of type %s", type(ops).__name__)
        return ApplyResult(
            items=items,
            skipped=(SkippedOp(action=None, id=None, reason="payload-not-a-list"),),
        )

    for op in ops:
        if not isinstance(op, dict):
            skipped.append(SkippedOp(action=None, id=None, reason="op-not-an-object"))
            continue

        raw_action = op.get("action", op.get("op"))
        action = raw_action.strip().lower() if isinstance(raw_action, str) else None
        op_id = _clean_id(op.get("id"))

        if action not in _ACTIONS:
            skipped.append(SkippedOp(action=action, id=op_id, reason="unknown-action"))
            continue

        if action == "create":
            text = _clean_text(op.get("text"))
            if text is None:
                skipped.append(SkippedOp(action=action, id=None, reason="blank-text"))
                continue

            # Exact repeats are this module's problem. Semantic near-duplicates
            # are the model's, and guessing at them here would merge two facts
            # that only look alike.
            wanted = normalise(text)
            if any(normalise(fact.text) == wanted for fact in facts):
                skipped.append(SkippedOp(action=action, id=None, reason="duplicate-text"))
                continue

            kind = _clean_kind(op.get("kind"), kinds)
            if kind is None:
                kind = fallback_kind(kinds)
                logger.info(
                    "coach memory: create names no known kind, filed as %s (kinds: %s)",
                    kind,
                    ", ".join(kinds),
                )

            new_id, next_id = _mint(next_id)
            fact = Fact(
                id=new_id,
                text=text,
                kind=kind,
                created_at=stamp,
                updated_at=stamp,
                sources=_append_source((), source),
            )
            facts.append(fact)
            created.append(fact)
            continue

        if op_id is None:
            skipped.append(SkippedOp(action=action, id=None, reason="missing-id"))
            continue

        index = next((i for i, fact in enumerate(facts) if fact.id == op_id), None)
        if index is None:
            # Never turned into a create: the model was looking at a list this id
            # was in, so a create here would be a guess at what it meant.
            skipped.append(SkippedOp(action=action, id=op_id, reason="unknown-id"))
            continue

        existing = facts[index]

        if action == "delete":
            reason = _clean_text(op.get("reason"))
            tombstones = _entomb(tombstones, existing, stamp, reason)
            del facts[index]
            deleted.append(existing)
            continue

        text = _clean_text(op.get("text"))
        if text is None and "text" in op:
            skipped.append(SkippedOp(action=action, id=op_id, reason="blank-text"))
            continue

        kind = _clean_kind(op.get("kind"), kinds)
        if kind is None and "kind" in op:
            # Unlike a create, a modify already has a kind worth keeping, so an
            # unrecognised one is ignored rather than flattened to ``other``.
            logger.info(
                "coach memory: modify of %s names no known kind, keeping %s",
                existing.id,
                existing.kind,
            )

        if text is None and kind is None:
            skipped.append(SkippedOp(action=action, id=op_id, reason="nothing-to-change"))
            continue

        updated = replace(
            existing,
            text=text if text is not None else existing.text,
            kind=kind if kind is not None else existing.kind,
            updated_at=stamp,
            sources=_append_source(existing.sources, source),
        )
        facts[index] = updated
        modified.append(updated)

    if skipped:
        logger.info(
            "coach memory: %d created, %d modified, %d deleted, %d skipped (%s)",
            len(created),
            len(modified),
            len(deleted),
            len(skipped),
            ", ".join(sorted({op.reason for op in skipped})),
        )

    return ApplyResult(
        items=MemoryList(facts=tuple(facts), tombstones=tombstones, next_id=next_id),
        created=tuple(created),
        modified=tuple(modified),
        deleted=tuple(deleted),
        skipped=tuple(skipped),
    )


def restore(
    items: Sequence[Fact],
    tombstones: Sequence[Tombstone],
    id: str,
) -> RestoreResult:
    """Pull one deleted fact back into the active list.

    Takes the two sequences rather than a :class:`MemoryList` because restoring
    mints nothing: the fact comes back under the id it already had, so the
    counter is not involved and is not something this function should be able to
    touch. An id that is not in the tombstones, or is already live, restores
    nothing and is not an error.
    """
    live = tuple(items)
    remaining = tuple(tombstones)

    if any(fact.id == id for fact in live):
        return RestoreResult(items=live, tombstones=remaining, restored=None)

    index = next((i for i, stone in enumerate(remaining) if stone.fact.id == id), None)
    if index is None:
        logger.info("coach memory: nothing to restore for id %s", id)
        return RestoreResult(items=live, tombstones=remaining, restored=None)

    fact = remaining[index].fact
    return RestoreResult(
        items=(*live, fact),
        tombstones=remaining[:index] + remaining[index + 1 :],
        restored=fact,
    )


def adopt(
    items: MemoryList,
    edited: Iterable[EditedItem],
    *,
    now: datetime | None = None,
    kinds: Sequence[str] = PROFILE_KINDS,
) -> ApplyResult:
    """Take the list as the owner left it after editing it by hand.

    A hand edit outranks what the model stored, so ``edited`` decides the contents
    and the order. What the stored list contributes is identity: each line is
    matched back to the fact it came from, first on ``key`` and then on its text,
    so that a reworded bullet keeps its id, its ``created_at`` and the entries
    that taught it. Only a line that matches nothing becomes a new fact, and a
    fact that has vanished from the list is tombstoned rather than dropped.
    """
    stamp = now_stamp(now)
    next_id = items.next_id
    tombstones = items.tombstones

    by_key = {fact.key: fact for fact in items.facts if fact.key}
    by_text: dict[str, Fact] = {}
    for fact in items.facts:
        by_text.setdefault(normalise(fact.text), fact)

    kept: list[Fact] = []
    created: list[Fact] = []
    modified: list[Fact] = []
    deleted: list[Fact] = []
    skipped: list[SkippedOp] = []
    claimed: set[str] = set()

    for item in edited:
        text = _clean_text(item.text)
        if text is None:
            skipped.append(SkippedOp(action="adopt", id=None, reason="blank-text"))
            continue

        key = item.key.strip() if isinstance(item.key, str) and item.key.strip() else None

        match = by_key.get(key) if key else None
        if match is None:
            match = by_text.get(normalise(text))
        if match is not None and match.id in claimed:
            match = None

        if match is None:
            new_id, next_id = _mint(next_id)
            fact = Fact(
                id=new_id,
                text=text,
                kind=fallback_kind(kinds),
                created_at=stamp,
                updated_at=stamp,
                key=key,
            )
            kept.append(fact)
            created.append(fact)
            continue

        claimed.add(match.id)
        # A line that arrives without a key keeps the one the fact already had:
        # the renderer not supplying an identity this time is no reason to throw
        # away the identity that made the next match possible.
        new_key = key or match.key
        changed = match.text != text or match.key != new_key
        fact = replace(
            match,
            text=text,
            key=new_key,
            updated_at=stamp if changed else match.updated_at,
        )
        kept.append(fact)
        if changed:
            modified.append(fact)

    for fact in items.facts:
        if fact.id not in claimed:
            tombstones = _entomb(tombstones, fact, stamp, "removed by hand")
            deleted.append(fact)

    logger.info(
        "coach memory: adopted a hand-edited list — %d kept, %d new, %d reworded, %d removed",
        len(kept),
        len(created),
        len(modified),
        len(deleted),
    )

    return ApplyResult(
        items=MemoryList(facts=tuple(kept), tombstones=tombstones, next_id=next_id),
        created=tuple(created),
        modified=tuple(modified),
        deleted=tuple(deleted),
        skipped=tuple(skipped),
    )
