"""Keeping the coach's two lists and their two Notion pages in step.

The profile and the rules live in a JSON file on the server, which is the right
place for them to live: it is atomic, it is fast, and it is there whether Notion
is or not. It is also somewhere the owner cannot read without an ssh session,
which makes the one part of this bot that has an opinion about him the one part
he cannot check. So both lists are mirrored onto two ordinary Notion pages beside
the diary — and because they are ordinary pages, he can edit them.

Two rules decide everything below.

**A page the owner touched by hand outranks what the bot stored.** A bullet he
reworded, reordered, added or deleted is adopted before the next answer is
written. That is what :func:`services.coach.memory.adopt` is for, and matching is
on the block id Notion gives every bullet — not on the text, which is the exact
thing an edit changes.

**A read that did not succeed is not an empty page.** ``adopt([])`` tombstones
every fact in the list, which is correct for a page the owner cleared on purpose
and catastrophic for a timeout. The two are told apart by the only honest signal
there is: a failed read raises out of ``services/notion.py`` and never reaches
``adopt`` at all. There is a second case of the same shape — a page that has
never been written is empty for a reason that has nothing to do with intent — and
:meth:`MemorySync._adopted` guards that one too.

The file stays the source of truth for *availability*: Notion being down, slow or
strange costs a sync and nothing else, and every caller here carries on with what
it read locally. The page is the source of truth for *intent*.

Pulls are throttled, because a conversation's follow-up turns would otherwise ask
Notion once a turn. A caller that is about to *change* a list forces the pull
instead: a throttled copy is fine to answer from and not fine to write back over.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, replace

from services import notion
from services.coach import memory
from services.coach.store import MemoryStore, StoreData

logger = logging.getLogger(__name__)

# How long a pull is reused for. A coach conversation asks for the rules on every
# follow-up turn, and a reply typed ten seconds after the last one does not need
# Notion asked again. Tens of seconds is the whole fix for that burst.
#
# It is only the right default where the caller is *reading*. A caller that is
# about to rewrite a list and mirror it back passes force=True, because accepting
# a copy from up to this long ago would let it write over an edit made by hand in
# between — see the two forced pulls in bot.py. What that leaves unclosed is the
# span of one model call, which is not something a window can help with.
PULL_INTERVAL_SECONDS = 45.0

# How long a *failed* sync is left alone for. Longer than the interval above by a
# lot: a Notion outage or a page the owner moved fails slowly — the client retries
# and each attempt has a timeout — and paying that on every turn would turn a coach
# that answers from local state into a coach that answers a minute late.
FAILURE_BACKOFF_SECONDS = 300.0


@dataclass(frozen=True)
class PageIds:
    profile: str
    rules: str


class MemorySync:
    """The one serialised path between the memory file and its two pages.

    Every read-modify-write in here happens under the same lock ``bot.py`` holds
    for its own writes to the file, and the Notion requests happen inside it too.
    That is deliberate and it is the point: it costs a background profile pass a
    few seconds of waiting, and it buys the guarantee that there are never two
    writers on the same page, and never a save built on a document that moved.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        lock: asyncio.Lock,
        interval: float = PULL_INTERVAL_SECONDS,
        backoff: float = FAILURE_BACKOFF_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self._store = store
        self._lock = lock
        self._interval = interval
        self._backoff = backoff
        self._clock = clock
        self._pages: PageIds | None = None
        # Monotonic, because this is an elapsed-time question and the wall clock
        # on a small VPS moves when ntp corrects it.
        self._next_sync = 0.0

    # -- the two directions ------------------------------------------------- #

    async def pull(self, *, force: bool = False) -> StoreData:
        """Adopt both pages into the stored lists, and hand back what to use.

        Returns the stored document whatever happens, so a caller can use it
        without asking whether the sync worked: on a throttled call it is what is
        already on disk, on a failed one it is the same, and on a successful one
        it is the lists as the owner left them.

        Failures from Notion are swallowed; failures from the store are not. A
        store that cannot be read is not something to carry on past — it is what
        the caller's own error path is for, and answering with an empty rules list
        would be worse than not answering.
        """
        if not force and self._fresh():
            return self._store.load()

        async with self._lock:
            # Checked again inside the lock, and this is what collapses a burst:
            # the turns that queued behind the one doing the work find the answer
            # already fetched rather than each fetching it again.
            if not force and self._fresh():
                return self._store.load()

            data = self._store.load()
            try:
                pages = await self._page_ids()
                on_profile = await notion.read_bullets(pages.profile)
                on_rules = await notion.read_bullets(pages.rules)
            except Exception:
                self._defer(self._backoff)
                logger.warning(
                    "Could not read the memory pages; using the %d profile fact(s) and "
                    "%d rule(s) on disk",
                    len(data.profile.facts),
                    len(data.rules.facts),
                    exc_info=True,
                )
                return data

            profile, profile_moved = self._adopted(
                data.profile, on_profile, kinds=memory.PROFILE_KINDS, name="profile"
            )
            rules, rules_moved = self._adopted(
                data.rules, on_rules, kinds=memory.RULE_KINDS, name="rules"
            )
            if profile_moved or rules_moved:
                data = replace(data, profile=profile, rules=rules)
                self._store.save(data)

            self._defer(self._interval)
            return data

    async def push(self) -> StoreData:
        """Write both stored lists onto their pages, and record the ids that come back.

        Called after every change, and it is the half of this that gives a fact
        its ``key``: a bullet the bot has just written is the first thing that
        knows what block it is, and without that id the owner's next rewording of
        it would arrive as a stranger.

        Notion failing costs the mirror one round and nothing else — the lists on
        disk are untouched and are handed back exactly as they were.
        """
        async with self._lock:
            data = self._store.load()
            try:
                pages = await self._page_ids()
                profile_keys = await notion.write_bullets(
                    pages.profile, [fact.text for fact in data.profile.facts]
                )
                rules_keys = await notion.write_bullets(
                    pages.rules, [fact.text for fact in data.rules.facts]
                )
            except Exception:
                self._defer(self._backoff)
                logger.warning("Could not write the memory pages", exc_info=True)
                return data

            profile = _with_keys(data.profile, profile_keys)
            rules = _with_keys(data.rules, rules_keys)
            if profile != data.profile or rules != data.rules:
                data = replace(data, profile=profile, rules=rules)
                self._store.save(data)
                logger.info("Recorded the block ids of the memory pages")

            # The pages now say exactly what the file says, so the next read would
            # find nothing to adopt. Charging the caller for it anyway would mean
            # a save is always followed by a pointless round trip.
            self._defer(self._interval)
            return data

    # -- the parts the two directions share --------------------------------- #

    def _fresh(self) -> bool:
        return self._clock() < self._next_sync

    def _defer(self, seconds: float) -> None:
        self._next_sync = self._clock() + seconds

    async def _page_ids(self) -> PageIds:
        """The two page ids, found once and remembered.

        Looked up by title and created only if the lookup came back without them,
        so a restart adopts the pages that are already there rather than making a
        second pair. Nothing is cached unless both were resolved: half a cache is
        how the wrong page ends up being written to for the rest of the process.
        """
        if self._pages is not None:
            return self._pages

        parent = await notion.memory_parent_page_id()
        pages = PageIds(
            profile=await self._page(parent, notion.MEMORY_PROFILE_TITLE),
            rules=await self._page(parent, notion.MEMORY_RULES_TITLE),
        )
        self._pages = pages
        return pages

    async def _page(self, parent: str, title: str) -> str:
        found = await notion.find_child_page(parent, title)
        return found if found is not None else await notion.create_child_page(parent, title)

    def _adopted(
        self,
        items: memory.MemoryList,
        bullets: list[notion.Bullet],
        *,
        kinds,
        name: str,
    ) -> tuple[memory.MemoryList, bool]:
        """One list as the page has it, and whether that is a change.

        The guard is the reason this is not a one-liner. An empty page that no
        fact has ever been written to is a page the mirror has simply not run for
        yet — every fact still has ``key`` unset, because a key is what a push
        hands back — and adopting it would tombstone the lot on the first sync
        after a restart that happened at the wrong moment. An empty page that
        keyed facts were written to is a page the owner cleared, which is the
        documented way to reset a list, and that one is adopted.
        """
        if not bullets and not any(fact.key for fact in items.facts):
            if items.facts:
                logger.info(
                    "The %s page is empty and no fact has been mirrored yet; "
                    "keeping the %d on disk",
                    name,
                    len(items.facts),
                )
            return items, False

        result = memory.adopt(
            items,
            [memory.EditedItem(key=bullet.id, text=bullet.text) for bullet in bullets],
            kinds=kinds,
        )
        # Whether to write the file is a different question from what to tell the
        # owner changed, and it is the one question a comparison answers better
        # than the result does: reordering the bullets creates nothing, modifies
        # nothing and deletes nothing, and it is still an edit — the order is the
        # order the facts reach the prompt in, and it is his to decide.
        moved = result.items != items
        if moved:
            logger.info(
                "Adopted the %s page: %d new, %d reworded, %d removed by hand",
                name,
                len(result.created),
                len(result.modified),
                len(result.deleted),
            )
        return result.items, moved


def _with_keys(items: memory.MemoryList, keys: list[str]) -> memory.MemoryList:
    """The same list, with each fact carrying the block id its line was written to.

    Bookkeeping, not an edit: ``updated_at`` is left alone, because learning where
    a fact is rendered is not a change to what the fact says. A fact with no key
    to record — an append whose response did not name every block it created —
    keeps the key it had rather than losing it.
    """
    facts = tuple(
        replace(fact, key=keys[index])
        if index < len(keys) and keys[index] and fact.key != keys[index]
        else fact
        for index, fact in enumerate(items.facts)
    )
    return replace(items, facts=facts)
