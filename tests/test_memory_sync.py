"""The mirror between the coach's memory file and its two Notion pages.

Notion is mocked at ``services/notion.py``'s own boundary here — the requests
those functions send have their own tests in ``tests/test_notion_memory_pages.py``
— because what is being checked in this file is policy: which edit wins, what a
failed read is allowed to do, and how often Notion is asked at all.

One test matters more than the rest. ``adopt([])`` tombstones every fact in the
list, which is exactly right for a page the owner cleared on purpose and
catastrophic for a page that could not be read. If
``test_a_read_that_failed_never_clears_anything`` ever stops failing when that
guard is removed, this file has stopped being worth running.
"""

import os

import pytest

DUMMY_ENV = {
    "TELEGRAM_TOKEN": "test-token",
    "OPENAI_API_KEY": "test-key",
    "NOTION_TOKEN": "test-notion",
    "NOTION_DATABASE_ID": "test-db",
    "ALLOWED_USER_ID": "1",
    "TIMEZONE": "Europe/Moscow",
}
for _name, _value in DUMMY_ENV.items():
    os.environ.setdefault(_name, _value)

import asyncio  # noqa: E402
from dataclasses import replace  # noqa: E402

from notion_pages_fake import FakePages  # noqa: E402

from services import memory_sync, notion  # noqa: E402
from services.coach import memory  # noqa: E402
from services.coach.store import MemoryStore, StoreData  # noqa: E402

OLD = "2026-01-02T03:04:05+00:00"


def fact(id, text, *, key=None, kind="trait", sources=()):
    return memory.Fact(
        id=id, text=text, kind=kind, created_at=OLD, updated_at=OLD, sources=sources, key=key
    )


def listed(*facts, next_id=None):
    return memory.MemoryList(
        facts=tuple(facts), next_id=next_id if next_id is not None else len(facts) + 1
    )


class CountingStore(MemoryStore):
    """A real store that says how many times it was written."""

    def __init__(self, path):
        super().__init__(path)
        self.saves = 0

    def save(self, data):
        self.saves += 1
        super().save(data)


@pytest.fixture
def pages(monkeypatch):
    return FakePages().install(monkeypatch)


@pytest.fixture
def clock():
    now = [1000.0]
    return now


@pytest.fixture
def store(tmp_path):
    return CountingStore(tmp_path / "coach_memory.state.json")


def make_sync(store, clock, **kwargs):
    return memory_sync.MemorySync(store, lock=asyncio.Lock(), clock=lambda: clock[0], **kwargs)


# --------------------------------------------------------------------------- #
# Identity survives an edit
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_reworded_bullet_keeps_its_id_created_at_and_sources(store, pages, clock):
    """The edit this feature invites, and the one a text match would get wrong."""
    store.save(
        StoreData(
            profile=listed(
                fact("7", "Откладывает трудные разговоры.", key="b1", sources=("2026-01-01 · a",))
            )
        )
    )
    pages.put("profile-page", ("b1", "Откладывает трудный разговор, пока он не решится сам."))

    data = await make_sync(store, clock).pull()

    (kept,) = data.profile.facts
    assert kept.id == "7"
    assert kept.text == "Откладывает трудный разговор, пока он не решится сам."
    assert kept.created_at == OLD
    assert kept.sources == ("2026-01-01 · a",)
    assert data.profile.tombstones == ()


@pytest.mark.asyncio
async def test_a_reordered_page_is_adopted_without_inventing_anything(store, pages, clock):
    store.save(
        StoreData(profile=listed(fact("1", "первое", key="b1"), fact("2", "второе", key="b2")))
    )
    pages.put("profile-page", ("b2", "второе"), ("b1", "первое"))

    data = await make_sync(store, clock).pull()

    assert [(f.id, f.text) for f in data.profile.facts] == [("2", "второе"), ("1", "первое")]
    assert data.profile.tombstones == ()


@pytest.mark.asyncio
async def test_a_bullet_added_by_hand_becomes_a_fact(store, pages, clock):
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"))))
    pages.put("profile-page", ("b1", "первое"), ("b9", "Не любит звонить по телефону."))

    data = await make_sync(store, clock).pull()

    new = data.profile.facts[1]
    assert new.text == "Не любит звонить по телефону."
    assert new.id == "2", "a new fact takes the stored counter, not a guess at one"
    assert new.key == "b9", "the block he typed into is the identity it keeps"


@pytest.mark.asyncio
async def test_a_hand_written_rule_is_filed_as_a_rule(store, pages, clock):
    """RULE_KINDS, not the profile vocabulary the default would file it under."""
    store.save(StoreData(rules=listed(fact("1", "короче", kind="rule", key="r1"))))
    pages.put("rules-page", ("r1", "короче"), ("r2", "не начинай с приветствия"))

    data = await make_sync(store, clock).pull()

    assert [f.kind for f in data.rules.facts] == ["rule", "rule"]


@pytest.mark.asyncio
async def test_a_deleted_bullet_tombstones_its_fact(store, pages, clock):
    store.save(
        StoreData(profile=listed(fact("1", "первое", key="b1"), fact("2", "второе", key="b2")))
    )
    pages.put("profile-page", ("b1", "первое"))

    data = await make_sync(store, clock).pull()

    assert [f.id for f in data.profile.facts] == ["1"]
    assert [(t.fact.id, t.reason) for t in data.profile.tombstones] == [("2", "removed by hand")]


@pytest.mark.asyncio
async def test_an_emptied_page_clears_the_list(store, pages, clock):
    """The documented way to reset a list, so it has to actually work."""
    store.save(
        StoreData(profile=listed(fact("1", "первое", key="b1"), fact("2", "второе", key="b2")))
    )
    pages.put("profile-page")

    data = await make_sync(store, clock).pull()

    assert data.profile.facts == ()
    assert [t.fact.id for t in data.profile.tombstones] == ["1", "2"]


# --------------------------------------------------------------------------- #
# What an empty answer is not
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_read_that_failed_never_clears_anything(store, pages, clock):
    """The single most destructive mistake available here.

    A timeout, a permissions change, a page the owner moved: none of them are the
    owner clearing a page, and none of them may reach ``adopt``.
    """
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"))))
    pages.fails = notion.NotionError("Notion GET /blocks/profile-page/children failed")

    data = await make_sync(store, clock).pull()

    assert [f.id for f in data.profile.facts] == ["1"]
    assert data.profile.tombstones == ()
    assert store.load().profile.facts == data.profile.facts
    assert store.saves == 1, "the failed pull wrote nothing; the one save is the fixture's"


@pytest.mark.asyncio
async def test_a_page_that_was_never_written_does_not_look_emptied(store, pages, clock):
    """An empty page is only an instruction once something has been put on it.

    No fact carries a key, so nothing has ever been mirrored: the page is empty
    because the push has not run, not because he cleared it.
    """
    store.save(StoreData(profile=listed(fact("1", "первое"), fact("2", "второе"))))
    pages.put("profile-page")

    data = await make_sync(store, clock).pull()

    assert [f.id for f in data.profile.facts] == ["1", "2"]
    assert data.profile.tombstones == ()


@pytest.mark.asyncio
async def test_one_keyed_fact_is_enough_for_an_empty_page_to_mean_it(store, pages, clock):
    """The other side of the same guard: something was mirrored, so it was cleared."""
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"), fact("2", "второе"))))
    pages.put("profile-page")

    data = await make_sync(store, clock).pull()

    assert data.profile.facts == ()


@pytest.mark.asyncio
async def test_notion_being_down_still_hands_back_the_local_lists(store, pages, clock):
    """A coach answer is produced from the file. It is never the mirror's to break."""
    store.save(
        StoreData(
            profile=listed(fact("1", "первое", key="b1")),
            rules=listed(fact("1", "короче", kind="rule", key="r1")),
        )
    )
    pages.fails = TimeoutError("the connection went away")

    data = await make_sync(store, clock).pull()

    assert [f.text for f in data.rules.facts] == ["короче"]
    assert [f.text for f in data.profile.facts] == ["первое"]


@pytest.mark.asyncio
async def test_a_store_that_cannot_be_read_is_not_swallowed(store, pages, clock):
    """Unlike Notion. Answering with an empty rules list would be worse than not."""
    store.path.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(Exception):  # noqa: B017 - StoreError, and any subclass of it
        await make_sync(store, clock).pull()


# --------------------------------------------------------------------------- #
# Writing the pages, and the ids that come back
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_push_puts_both_lists_on_their_pages(store, pages, clock):
    store.save(
        StoreData(
            profile=listed(fact("1", "первое"), fact("2", "второе")),
            rules=listed(fact("1", "короче", kind="rule")),
        )
    )

    await make_sync(store, clock).push()

    assert pages.texts("profile-page") == ["первое", "второе"]
    assert pages.texts("rules-page") == ["короче"]


@pytest.mark.asyncio
async def test_a_push_records_the_block_id_of_every_line_it_wrote(store, pages, clock):
    """Without this the owner's first rewording of a new fact arrives as a stranger."""
    store.save(StoreData(profile=listed(fact("1", "первое"), fact("2", "второе"))))

    data = await make_sync(store, clock).push()

    assert [f.key for f in data.profile.facts] == ["new-block-1", "new-block-2"]
    assert [f.key for f in store.load().profile.facts] == ["new-block-1", "new-block-2"]


@pytest.mark.asyncio
async def test_recording_a_key_is_not_an_edit_to_the_fact(store, pages, clock):
    store.save(StoreData(profile=listed(fact("1", "первое"))))

    data = await make_sync(store, clock).push()

    assert data.profile.facts[0].updated_at == OLD
    assert data.profile.facts[0].created_at == OLD


@pytest.mark.asyncio
async def test_a_push_with_nothing_new_to_record_does_not_rewrite_the_file(store, pages, clock):
    store.save(StoreData(profile=listed(fact("1", "первое"))))
    sync = make_sync(store, clock)
    await sync.push()
    saves = store.saves

    clock[0] += memory_sync.PULL_INTERVAL_SECONDS + 1
    await sync.push()

    assert store.saves == saves


@pytest.mark.asyncio
async def test_notion_failing_a_push_leaves_the_lists_alone(store, pages, clock):
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"))))
    pages.fails = notion.NotionError("Notion PATCH /blocks/profile-page/children returned 502")

    data = await make_sync(store, clock).push()

    assert [(f.id, f.key) for f in data.profile.facts] == [("1", "b1")]
    assert store.saves == 1


@pytest.mark.asyncio
async def test_a_fact_the_bot_created_is_mirrored_and_then_keeps_its_identity(store, pages, clock):
    """The round trip the feature is for: bot writes it, owner rewords it, it is the same fact."""
    store.save(StoreData(profile=listed(fact("4", "Откладывает трудные разговоры."))))
    sync = make_sync(store, clock)

    await sync.push()
    key = store.load().profile.facts[0].key
    pages.put("profile-page", (key, "Откладывает всё трудное."))
    clock[0] += memory_sync.PULL_INTERVAL_SECONDS + 1
    data = await sync.pull()

    (kept,) = data.profile.facts
    assert kept.id == "4"
    assert kept.text == "Откладывает всё трудное."
    assert kept.created_at == OLD


# --------------------------------------------------------------------------- #
# How often Notion is asked
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_burst_of_follow_up_turns_costs_one_read(store, pages, clock):
    """Every turn of a conversation pulls. Ten seconds of them is one request."""
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"))))
    pages.put("profile-page", ("b1", "первое"))
    sync = make_sync(store, clock)

    for _ in range(5):
        await sync.pull()
        clock[0] += 3

    assert pages.reads == 2, "both pages, once"


@pytest.mark.asyncio
async def test_the_window_does_expire(store, pages, clock):
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"))))
    pages.put("profile-page", ("b1", "первое"))
    sync = make_sync(store, clock)

    await sync.pull()
    clock[0] += memory_sync.PULL_INTERVAL_SECONDS + 1
    await sync.pull()

    assert pages.reads == 4


@pytest.mark.asyncio
async def test_concurrent_turns_do_not_each_fetch_the_page(store, pages, clock):
    """The check inside the lock, not only the one before it."""
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"))))
    pages.put("profile-page", ("b1", "первое"))
    sync = make_sync(store, clock)

    await asyncio.gather(*(sync.pull() for _ in range(4)))

    assert pages.reads == 2


@pytest.mark.asyncio
async def test_a_forced_pull_ignores_the_window(store, pages, clock):
    """What /rules does: the owner is asking what the bot believes right now."""
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"))))
    pages.put("profile-page", ("b1", "первое"))
    sync = make_sync(store, clock)

    await sync.pull()
    await sync.pull(force=True)

    assert pages.reads == 4


@pytest.mark.asyncio
async def test_a_failure_is_left_alone_for_longer_than_a_success(store, pages, clock):
    """A Notion outage fails slowly, and a coach answer is not going to wait for it twice."""
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"))))
    pages.fails = notion.NotionError("Notion GET /databases/test-db failed")
    sync = make_sync(store, clock)

    await sync.pull()
    pages.fails = None
    pages.put("profile-page", ("b1", "первое"))
    clock[0] += memory_sync.PULL_INTERVAL_SECONDS + 1
    await sync.pull()

    assert pages.reads == 0, "still inside the backoff"

    clock[0] += memory_sync.FAILURE_BACKOFF_SECONDS
    await sync.pull()
    assert pages.reads == 2


@pytest.mark.asyncio
async def test_a_push_counts_as_a_read_for_the_window(store, pages, clock):
    """The page says exactly what the file says, so asking it again buys nothing."""
    store.save(StoreData(profile=listed(fact("1", "первое"))))
    sync = make_sync(store, clock)

    await sync.push()
    await sync.pull()

    assert pages.reads == 0


# --------------------------------------------------------------------------- #
# Finding the pages
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_pages_are_found_once_and_remembered(store, pages, clock):
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"))))
    sync = make_sync(store, clock)

    await sync.pull()
    clock[0] += memory_sync.PULL_INTERVAL_SECONDS + 1
    await sync.pull()

    assert pages.lookups == 1


@pytest.mark.asyncio
async def test_a_missing_page_is_created_and_an_existing_one_is_not(store, pages, clock):
    del pages.exists[notion.MEMORY_RULES_TITLE]
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"))))

    await make_sync(store, clock).pull()

    assert pages.created == [notion.MEMORY_RULES_TITLE]


@pytest.mark.asyncio
async def test_a_half_resolved_pair_is_not_remembered(store, pages, clock):
    """Caching one page and failing on the other is how the wrong page gets written to."""
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"))))

    async def find(parent, title):
        if title == notion.MEMORY_RULES_TITLE:
            raise notion.NotionError("Notion GET /blocks/parent-page/children failed")
        return "profile-page"

    sync = make_sync(store, clock)
    original, notion.find_child_page = notion.find_child_page, find
    try:
        await sync.pull()
    finally:
        notion.find_child_page = original

    clock[0] += memory_sync.FAILURE_BACKOFF_SECONDS + 1
    await sync.pull()

    assert pages.lookups == 2, "looked the pair up again rather than trusting half of it"


# --------------------------------------------------------------------------- #
# Long lists
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_list_far_longer_than_one_page_of_blocks_survives_a_round_trip(store, pages, clock):
    """A profile outgrows the hundred blocks one request carries; nothing may be lost."""
    facts = [fact(str(n), f"факт {n}") for n in range(1, 251)]
    store.save(StoreData(profile=listed(*facts)))
    sync = make_sync(store, clock)

    await sync.push()
    clock[0] += memory_sync.PULL_INTERVAL_SECONDS + 1
    data = await sync.pull()

    assert len(data.profile.facts) == 250
    assert [f.id for f in data.profile.facts] == [str(n) for n in range(1, 251)]
    assert all(f.key for f in data.profile.facts)
    assert data.profile.tombstones == ()


@pytest.mark.asyncio
async def test_a_fact_with_no_key_to_record_keeps_the_one_it_had(store, pages, clock, monkeypatch):
    """An append whose answer did not name every block it made is not a reason to forget."""
    store.save(StoreData(profile=listed(fact("1", "первое", key="b1"), fact("2", "второе"))))

    async def short(page_id, texts):
        return ["b1"]

    monkeypatch.setattr(notion, "write_bullets", short)
    data = await make_sync(store, clock).push()

    assert [f.key for f in data.profile.facts] == ["b1", None]


def test_replacing_a_key_leaves_everything_else_alone():
    items = listed(fact("1", "первое", key="b1"))

    updated = memory_sync._with_keys(items, ["b2"])

    assert updated.facts[0] == replace(items.facts[0], key="b2")
