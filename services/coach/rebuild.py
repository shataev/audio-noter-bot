"""Rebuilding the profile from a diary that was written before it existed.

The profile grows one saved entry at a time, which means it knows nothing about
the years of diary that were written before the feature shipped. This walks all
of it and folds it in — the same per-entry extraction, in sequence, each call
seeing the facts the calls before it produced.

It is deliberately the dullest thing in the package, because it is the only one
that runs unattended for a long time, costs real money per step, and rewrites
the file every other part of the coach depends on:

*Sequential, with a pause.* One entry, one call, then a wait. Concurrency here
would buy minutes and would mean a hundred requests at Notion and at the
provider from a bot whose normal load is one request every few hours — and it
would break the property the whole pass rests on, which is that entry *n* sees
what entries 1..n-1 taught.

*Every failure is one entry's failure.* An entry that cannot be read, a call
that comes back with nothing, a write that fails — each is counted and stepped
over. The accumulated profile is never rolled back by one bad entry.

*Except when they stop being one entry's failure.* A provider that is down, a
key that has expired, a Notion token that was rotated: those fail every call,
and a pass that keeps going through them burns one doomed request per remaining
entry. So consecutive failures trip a breaker and the pass stops with what it
has.

*Written after every entry.* An abort, a crash or a deploy in the middle of a
pass costs the entry in flight and nothing else.

The caller supplies the entries, the persistence and the progress reporting.
This module holds no state, touches no file and knows nothing about Telegram or
Notion — see ``diary.py`` for why the entries arrive as plain data.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass

from . import profile as coach_profile
from .diary import Entry, Unreadable
from .memory import MemoryList

logger = logging.getLogger(__name__)

# Between one entry and the next. Notion's own limit is around three requests a
# second and the provider's is far higher, so this is not about staying under a
# limit — it is about a background job that runs for an hour not being the
# loudest client either service has, and about leaving room for the entry the
# owner dictates while it runs.
PAUSE_SECONDS = 1.0

# Consecutive failures that stop the pass. Small enough that an outage costs a
# handful of requests rather than a diary's worth, large enough that it is not
# tripped by two odd entries in a row — which does happen, because an entry can
# be a single word.
CONSECUTIVE_FAILURES = 5

# Telegram allows an edit roughly once a second per chat and answers the rest
# with 429s that the library then sleeps off, which would pace the whole pass to
# the speed of its own progress bar. Five seconds is well clear of it and still
# reads as live.
PROGRESS_SECONDS = 5.0


@dataclass(frozen=True)
class Progress:
    """How far the pass has got. Counts only — never a fact, never diary text."""

    done: int
    learned: int
    skipped: int
    created: int
    modified: int
    deleted: int


@dataclass(frozen=True)
class Rebuild:
    """What the pass ended up with.

    ``profile`` is the accumulated list, which the caller has already been given
    the chance to persist after every entry — it is here so that a caller that
    wants to compare it with what it started from can.
    """

    profile: MemoryList
    done: int = 0
    learned: int = 0
    skipped: int = 0
    created: int = 0
    modified: int = 0
    deleted: int = 0
    aborted: bool = False


async def run(
    *,
    entries: AsyncIterable[Entry | Unreadable],
    profile: MemoryList,
    model: str,
    focus: str | None = None,
    persist: Callable[[MemoryList], Awaitable[None]] | None = None,
    progress: Callable[[Progress], Awaitable[None]] | None = None,
    learn: Callable[..., Awaitable[coach_profile.Learned]] | None = None,
    pause: float | None = None,
    breaker: int | None = None,
    progress_seconds: float | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Rebuild:
    """Fold every entry in ``entries`` into ``profile``, oldest first.

    Returns what it managed to do. Raises only if ``entries`` itself does: a
    walk that cannot even be started is the caller's problem to report, while
    everything that goes wrong inside one step is counted here and stepped over.
    """
    # Resolved here rather than as default arguments, so that these are the values
    # the module holds now — a default is bound once, at import, which puts the
    # constants above out of reach of anything that wants to change them.
    extract = learn if learn is not None else coach_profile.learn
    pause = PAUSE_SECONDS if pause is None else pause
    breaker = CONSECUTIVE_FAILURES if breaker is None else breaker
    progress_seconds = PROGRESS_SECONDS if progress_seconds is None else progress_seconds

    done = learned = skipped = 0
    created = modified = deleted = 0
    consecutive = 0
    aborted = False
    first = True
    last_progress = monotonic()

    async def report(force: bool = False) -> None:
        nonlocal last_progress
        if progress is None:
            return
        now = monotonic()
        if not force and now - last_progress < progress_seconds:
            return
        last_progress = now
        try:
            await progress(
                Progress(
                    done=done,
                    learned=learned,
                    skipped=skipped,
                    created=created,
                    modified=modified,
                    deleted=deleted,
                )
            )
        except Exception:
            # A progress bar that cannot be drawn is not a reason to abandon a
            # pass that is working. It is also the most likely thing in here to
            # fail, because it is the one part that talks to Telegram.
            logger.warning("coach rebuild: could not report progress", exc_info=True)

    walk = entries.__aiter__()
    async for item in walk:
        if not first:
            await sleep(pause)
        first = False

        if isinstance(item, Unreadable):
            skipped += 1
            consecutive += 1
            logger.warning("coach rebuild: could not read %s, skipping it", item.where)
        else:
            done += 1
            try:
                result = await extract(
                    profile=profile,
                    title=item.title,
                    text=item.text,
                    model=model,
                    source=item.source,
                    focus=focus,
                )
            except Exception:
                # ``learn`` promises to raise nothing, and this is what happens
                # if that promise is ever broken: one entry lost, not the pass.
                logger.exception("coach rebuild: the extraction raised, skipping one entry")
                skipped += 1
                consecutive += 1
            else:
                if result.changed is None:
                    consecutive = 0
                else:
                    profile = result.profile
                    created += len(result.changed.created)
                    modified += len(result.changed.modified)
                    deleted += len(result.changed.deleted)
                    learned += 1
                    if persist is not None:
                        try:
                            await persist(profile)
                            consecutive = 0
                        except Exception:
                            # Kept in ``profile`` on purpose: the next entry's
                            # write carries these facts too, so a write that
                            # failed once costs nothing if the next one works.
                            logger.exception("coach rebuild: could not write the profile")
                            skipped += 1
                            consecutive += 1
                    else:
                        consecutive = 0

        if breaker and consecutive >= breaker:
            aborted = True
            logger.error(
                "coach rebuild: %d failures in a row, stopping after %d entr(ies)",
                consecutive,
                done,
            )
            break

        await report()

    # An async generator that was abandoned half-way — which is what the breaker
    # does — is otherwise closed whenever the collector gets to it, and its
    # cleanup then runs against whatever loop is current at the time.
    closer = getattr(walk, "aclose", None)
    if closer is not None:
        await closer()

    await report(force=True)
    logger.info(
        "coach rebuild: %d entr(ies) read, %d taught something, %d skipped, "
        "%d created, %d modified, %d deleted, aborted=%s",
        done,
        learned,
        skipped,
        created,
        modified,
        deleted,
        aborted,
    )
    return Rebuild(
        profile=profile,
        done=done,
        learned=learned,
        skipped=skipped,
        created=created,
        modified=modified,
        deleted=deleted,
        aborted=aborted,
    )
