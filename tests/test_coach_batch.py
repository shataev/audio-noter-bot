"""The retrospective profile rebuild: /memory, and the pass behind it.

Nothing here reaches Notion, Telegram or a provider — there are no credentials
on this machine. The diary is a list of fake Notion blocks, the extraction is a
function that records what it was handed, and Telegram is the same kind of
double the rest of the suite uses.

What is actually being pinned down is the set of promises that make it safe to
let a job rewrite the coach's memory unattended: nothing runs before the owner
confirms, a snapshot exists before the first call, one bad entry costs one
entry, a provider that is down stops the pass instead of being asked once per
remaining entry, and every entry that taught something is on disk before the
next one starts.
"""

import os

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

from dataclasses import replace
from datetime import date

import pytest
from notion_pages_fake import FakePages

import bot
from services import summary
from services.coach import diary as coach_diary
from services.coach import memory as coach_memory
from services.coach import profile as coach_profile
from services.coach import rebuild as coach_rebuild
from services.coach.store import MemoryStore

# --------------------------------------------------------------------------- #
# Fixtures shared by the two halves: the pure pass, and the handlers on top.
# --------------------------------------------------------------------------- #


def fact(fact_id, text="a fact", kind="trait"):
    return coach_memory.Fact(
        id=fact_id,
        text=text,
        kind=kind,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def profile_of(*facts, next_id=None):
    return coach_memory.MemoryList(
        facts=tuple(facts), next_id=next_id if next_id is not None else len(facts) + 1
    )


def entry(number):
    return coach_diary.Entry(
        source=f"2026-01-{number:02d} · Entry {number}",
        title=f"Entry {number}",
        text=f"text {number}",
    )


async def entries_from(items):
    for item in items:
        yield item


class Learner:
    """A stand-in for ``profile.learn`` that records what each call was shown.

    ``script`` is one instruction per entry: ``"teach"`` adds a fact, ``"quiet"``
    changes nothing, ``"raise"`` breaks the promise that ``learn`` never raises.
    The default is to teach on every entry.
    """

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls = []

    async def __call__(self, *, profile, title, text, model, source=None, focus=None, **kwargs):
        self.calls.append(
            {
                "profile": profile,
                "title": title,
                "text": text,
                "model": model,
                "source": source,
                "focus": focus,
            }
        )
        action = self.script.pop(0) if self.script else "teach"
        if action == "raise":
            raise RuntimeError("the provider said no")
        if action == "quiet":
            return coach_profile.Learned(profile=profile)

        new = fact(str(profile.next_id), text=f"learned from {title}")
        applied = coach_memory.ApplyResult(
            items=replace(profile, facts=(*profile.facts, new), next_id=profile.next_id + 1),
            created=(new,),
        )
        return coach_profile.Learned(profile=applied.items, changed=applied)


class Clock:
    """A monotonic clock the test drives, so throttling is tested without waiting."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


async def no_sleep(_seconds):
    return None


# --------------------------------------------------------------------------- #
# services/coach/rebuild.py — the pass itself.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_each_entry_is_shown_what_the_entries_before_it_taught():
    """The whole reason the pass is sequential rather than concurrent."""
    learn = Learner()

    result = await coach_rebuild.run(
        entries=entries_from([entry(1), entry(2), entry(3)]),
        profile=profile_of(),
        model="m",
        learn=learn,
        sleep=no_sleep,
    )

    seen = [len(call["profile"].facts) for call in learn.calls]
    assert seen == [0, 1, 2]
    assert [call["source"] for call in learn.calls] == [
        "2026-01-01 · Entry 1",
        "2026-01-02 · Entry 2",
        "2026-01-03 · Entry 3",
    ]
    assert len(result.profile.facts) == 3
    assert (result.done, result.learned, result.skipped) == (3, 3, 0)


@pytest.mark.asyncio
async def test_there_is_a_pause_between_entries_and_none_before_the_first():
    slept = []

    async def record(seconds):
        slept.append(seconds)

    await coach_rebuild.run(
        entries=entries_from([entry(1), entry(2), entry(3)]),
        profile=profile_of(),
        model="m",
        learn=Learner(),
        pause=0.25,
        sleep=record,
    )

    assert slept == [0.25, 0.25]


@pytest.mark.asyncio
async def test_the_focus_reaches_every_call():
    learn = Learner()

    await coach_rebuild.run(
        entries=entries_from([entry(1), entry(2)]),
        profile=profile_of(),
        model="m",
        focus="что с деньгами",
        learn=learn,
        sleep=no_sleep,
    )

    assert [call["focus"] for call in learn.calls] == ["что с деньгами", "что с деньгами"]


@pytest.mark.asyncio
async def test_the_profile_is_written_after_every_entry_that_taught_something():
    """A restart mid-pass must not cost the entries that ran before it."""
    written = []

    async def persist(profile):
        written.append(len(profile.facts))

    await coach_rebuild.run(
        entries=entries_from([entry(1), entry(2), entry(3)]),
        profile=profile_of(),
        model="m",
        learn=Learner(["teach", "quiet", "teach"]),
        persist=persist,
        sleep=no_sleep,
    )

    # One write per entry that changed something, never one write at the end.
    assert written == [1, 2]


@pytest.mark.asyncio
async def test_an_unreadable_day_is_counted_and_stepped_over():
    learn = Learner()

    result = await coach_rebuild.run(
        entries=entries_from([entry(1), coach_diary.Unreadable(where="page-2"), entry(3)]),
        profile=profile_of(),
        model="m",
        learn=learn,
        sleep=no_sleep,
    )

    assert [call["title"] for call in learn.calls] == ["Entry 1", "Entry 3"]
    assert (result.done, result.skipped) == (2, 1)
    assert len(result.profile.facts) == 2
    assert result.aborted is False


@pytest.mark.asyncio
async def test_an_extraction_that_raises_costs_one_entry_and_not_the_facts():
    result = await coach_rebuild.run(
        entries=entries_from([entry(1), entry(2), entry(3)]),
        profile=profile_of(fact("1", "an old fact")),
        model="m",
        learn=Learner(["teach", "raise", "teach"]),
        sleep=no_sleep,
    )

    assert result.skipped == 1
    assert result.aborted is False
    # The fact the pass started with, plus the two entries that worked.
    assert [item.text for item in result.profile.facts] == [
        "an old fact",
        "learned from Entry 1",
        "learned from Entry 3",
    ]


@pytest.mark.asyncio
async def test_a_write_that_fails_is_a_skip_and_the_next_write_carries_the_facts():
    written = []

    async def persist(profile):
        if len(written) == 0:
            written.append("failed")
            raise OSError("no room on the device")
        written.append(tuple(item.text for item in profile.facts))

    result = await coach_rebuild.run(
        entries=entries_from([entry(1), entry(2)]),
        profile=profile_of(),
        model="m",
        learn=Learner(),
        persist=persist,
        sleep=no_sleep,
    )

    assert result.skipped == 1
    assert written[1] == ("learned from Entry 1", "learned from Entry 2")


@pytest.mark.asyncio
async def test_consecutive_failures_stop_the_pass_instead_of_burning_a_call_each():
    learn = Learner(["raise"] * 10)

    result = await coach_rebuild.run(
        entries=entries_from([entry(number) for number in range(1, 21)]),
        profile=profile_of(),
        model="m",
        learn=learn,
        breaker=3,
        sleep=no_sleep,
    )

    assert result.aborted is True
    assert len(learn.calls) == 3
    assert result.skipped == 3


@pytest.mark.asyncio
async def test_scattered_failures_do_not_trip_the_breaker():
    """Three failures, never two in a row: an odd entry is not an outage."""
    result = await coach_rebuild.run(
        entries=entries_from([entry(number) for number in range(1, 7)]),
        profile=profile_of(),
        model="m",
        learn=Learner(["raise", "teach", "raise", "teach", "raise", "teach"]),
        breaker=2,
        sleep=no_sleep,
    )

    assert result.aborted is False
    assert (result.done, result.skipped, result.learned) == (6, 3, 3)


@pytest.mark.asyncio
async def test_an_entry_that_taught_nothing_resets_the_breaker():
    """A quiet entry is a working call, not a failure."""
    result = await coach_rebuild.run(
        entries=entries_from([entry(1), entry(2), entry(3), entry(4)]),
        profile=profile_of(),
        model="m",
        learn=Learner(["raise", "quiet", "raise", "teach"]),
        breaker=2,
        sleep=no_sleep,
    )

    assert result.aborted is False
    assert result.done == 4


@pytest.mark.asyncio
async def test_progress_is_throttled_and_the_last_one_always_arrives():
    clock = Clock()
    seen = []

    async def progress(state):
        seen.append(state.done)

    async def tick(_seconds):
        clock.now += 1.0

    result = await coach_rebuild.run(
        entries=entries_from([entry(number) for number in range(1, 21)]),
        profile=profile_of(),
        model="m",
        learn=Learner(),
        progress=progress,
        progress_seconds=5.0,
        sleep=tick,
        monotonic=clock,
    )

    # Twenty entries, one simulated second each: an unthrottled bar would be
    # twenty edits and a 429 from Telegram.
    assert len(seen) < 6
    assert seen[-1] == result.done == 20


@pytest.mark.asyncio
async def test_a_progress_bar_that_cannot_be_drawn_does_not_stop_the_pass():
    async def progress(_state):
        raise RuntimeError("Telegram said no")

    result = await coach_rebuild.run(
        entries=entries_from([entry(1), entry(2)]),
        profile=profile_of(),
        model="m",
        learn=Learner(),
        progress=progress,
        progress_seconds=0.0,
        sleep=no_sleep,
    )

    assert result.done == 2
    assert len(result.profile.facts) == 2


@pytest.mark.asyncio
async def test_an_empty_diary_is_not_an_error():
    result = await coach_rebuild.run(
        entries=entries_from([]),
        profile=profile_of(fact("1")),
        model="m",
        learn=Learner(),
        sleep=no_sleep,
    )

    assert (result.done, result.skipped, result.learned) == (0, 0, 0)
    assert len(result.profile.facts) == 1


# --------------------------------------------------------------------------- #
# The focus reaches the model as guidance, and never as a fact.
# --------------------------------------------------------------------------- #


def test_the_focus_is_rendered_into_the_extraction_prompt():
    with_focus = coach_profile.system_prompt((), "смотри на деньги")
    without = coach_profile.system_prompt(())

    assert "смотри на деньги" in with_focus
    assert "смотри на деньги" not in without


def test_no_focus_leaves_the_prompt_exactly_as_it_was():
    """An ordinary saved entry must not notice that the option exists."""
    facts = (fact("1", "he runs in the mornings"),)

    assert coach_profile.system_prompt(facts, None) == coach_profile.system_prompt(facts)
    assert coach_profile.system_prompt(facts, "   ") == coach_profile.system_prompt(facts)


# --------------------------------------------------------------------------- #
# services/summary.py — one page per day, split back into the entries on it.
# --------------------------------------------------------------------------- #


def heading(text):
    return {"type": "heading_3", "heading_3": {"rich_text": [{"plain_text": text}]}}


def paragraph(text):
    return {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": text}]}}


DIVIDER = {"type": "divider", "divider": {}}


def test_a_day_is_split_at_its_headings():
    blocks = [
        heading("Утро"),
        paragraph("встал рано"),
        paragraph("и сразу побежал"),
        DIVIDER,
        heading("Вечер"),
        paragraph("устал"),
    ]

    entries = summary.split_entries(blocks, date(2026, 5, 9))

    assert [(item.title, item.text) for item in entries] == [
        ("Утро", "встал рано\n\nи сразу побежал"),
        ("Вечер", "устал"),
    ]
    assert entries[0].source == "2026-05-09 · Утро"


def test_text_written_above_the_first_heading_is_kept_as_an_untitled_entry():
    """A page started by hand is still diary, and the pass is what would lose it."""
    entries = summary.split_entries(
        [paragraph("написал сам, без бота"), heading("A"), paragraph("b")]
    )

    assert [(item.title, item.text) for item in entries] == [
        ("", "написал сам, без бота"),
        ("A", "b"),
    ]


def test_an_empty_page_yields_no_entries():
    assert summary.split_entries([DIVIDER]) == []
    assert summary.split_entries([]) == []


def test_a_heading_with_no_text_under_it_is_still_an_entry():
    entries = summary.split_entries([heading("Заголовок")])

    assert [(item.title, item.text) for item in entries] == [("Заголовок", "")]


def test_the_day_comes_from_the_property_the_bot_sets():
    page = {"properties": {"Created": {"date": {"start": "2026-05-09"}}}}

    assert summary.page_day(page) == date(2026, 5, 9)


@pytest.mark.parametrize(
    "properties",
    [
        {},
        {"Created": {"date": None}},
        {"Created": {"date": {"start": ""}}},
        {"Created": {"date": {"start": "not-a-date"}}},
    ],
)
def test_a_page_with_no_usable_date_is_dated_none_rather_than_guessed_at(properties):
    assert summary.page_day({"properties": properties}) is None


@pytest.mark.asyncio
async def test_every_diary_page_is_listed_oldest_first_across_pages_of_results(monkeypatch):
    requests = []

    class Response:
        def __init__(self, body):
            self._body = body

        def json(self):
            return self._body

    async def fake_request(method, path, json=None, **kwargs):
        requests.append(json)
        if json.get("start_cursor") is None:
            return Response({"results": [{"id": "a"}], "has_more": True, "next_cursor": "c1"})
        return Response({"results": [{"id": "b"}], "has_more": False})

    monkeypatch.setattr(summary.notion, "_request", fake_request)

    pages = await summary.all_pages()

    assert [page["id"] for page in pages] == ["a", "b"]
    assert requests[0]["sorts"] == [{"property": "Created", "direction": "ascending"}]
    # No date filter at all: this is the whole diary, not a window on it.
    assert "filter" not in requests[0]
    assert requests[1]["start_cursor"] == "c1"


@pytest.mark.asyncio
async def test_a_server_that_keeps_handing_back_one_cursor_does_not_spin_forever(monkeypatch):
    class Response:
        def json(self):
            return {"results": [{"id": "a"}], "has_more": True, "next_cursor": "stuck"}

    calls = []

    async def fake_request(method, path, json=None, **kwargs):
        calls.append(json)
        return Response()

    monkeypatch.setattr(summary.notion, "_request", fake_request)

    pages = await summary.all_pages()

    assert len(calls) == 2
    assert len(pages) == 2


# --------------------------------------------------------------------------- #
# bot.py — the two steps the owner controls, and the pass they start.
#
# A small Telegram double rather than the one in tests/test_bot.py: two branches
# appending to that file is how the last merge conflict happened, and a conflict
# resolved by dropping somebody's test is the one outcome that must not happen.
# --------------------------------------------------------------------------- #


class Sent:
    def __init__(self, message_id, text, reply_markup, reply_to):
        self.message_id = message_id
        self.text = text
        self.reply_markup = reply_markup
        self.reply_to = reply_to


class FakeChat:
    def __init__(self, chat_id):
        self.id = chat_id


class FakeBot:
    def __init__(self, chat_id=1):
        self.chat_id = chat_id
        self.sent = []
        self.answered = []
        self._next_id = 100

    def _record(self, text, reply_markup=None, reply_to=None):
        self._next_id += 1
        message = Sent(self._next_id, text, reply_markup, reply_to)
        self.sent.append(message)
        return message

    def find(self, message_id):
        for message in reversed(self.sent):
            if message.message_id == message_id:
                return message
        raise AssertionError(f"no message {message_id}")

    async def send_message(
        self, chat_id, text, parse_mode=None, reply_markup=None, reply_parameters=None, **kwargs
    ):
        reply_to = reply_parameters.message_id if reply_parameters is not None else None
        return self._record(text, reply_markup, reply_to)

    async def edit_message_text(self, chat_id, message_id, text, **kwargs):
        message = self.find(message_id)
        message.text = text
        return message

    async def answer_callback_query(self, callback_query_id, text=None, **kwargs):
        self.answered.append(text)


class FakeMessage:
    def __init__(self, fake_bot, message_id=1, text=None, voice=None, reply_to_message=None):
        self._bot = fake_bot
        self.message_id = message_id
        self.text = text
        self.voice = voice
        self.chat = FakeChat(fake_bot.chat_id)
        self.reply_to_message = reply_to_message

    async def reply_text(self, text, parse_mode=None, reply_markup=None, **kwargs):
        sent = self._bot._record(text, reply_markup)
        return FakeMessage(self._bot, message_id=sent.message_id, text=text)


class FakeCallbackQuery:
    def __init__(self, fake_bot, data, message):
        self._bot = fake_bot
        self.data = data
        self.message = message

    async def answer(self, text=None, **kwargs):
        self._bot.answered.append(text)


class FakeUpdate:
    def __init__(self, fake_bot, message=None, callback_query=None):
        self.effective_message = message
        self.callback_query = callback_query
        self.effective_chat = FakeChat(fake_bot.chat_id)


class FakeApplication:
    def __init__(self):
        self.tasks = []

    def create_task(self, coroutine, update=None, *, name=None):
        self.tasks.append(coroutine)
        return coroutine

    async def run_tasks(self):
        while self.tasks:
            await self.tasks.pop(0)

    def close(self):
        for coroutine in self.tasks:
            coroutine.close()
        self.tasks.clear()


class FakeContext:
    def __init__(self, fake_bot):
        self.bot = fake_bot
        self.user_data = {}
        self.bot_data = {}
        self.args = []
        self.application = FakeApplication()


@pytest.fixture
def fake_bot():
    return FakeBot()


@pytest.fixture
def context(fake_bot):
    made = FakeContext(fake_bot)
    yield made
    made.application.close()


@pytest.fixture
def memory_state(tmp_path, monkeypatch):
    """A temp state directory, and the module's own dictionaries emptied."""
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(bot, "_thread_cache", None)
    monkeypatch.setattr(bot, "_memory_focus_prompts", {})
    monkeypatch.setattr(bot, "_memory_confirmations", {})
    monkeypatch.setattr(bot, "_rebuild_in_flight", False)
    # The pass waits a second between entries in earnest; a test that waited with
    # it would be minutes long and would prove nothing the pure tests above do not.
    monkeypatch.setattr(coach_rebuild, "PAUSE_SECONDS", 0)
    return tmp_path


def store_at(state_dir):
    return MemoryStore(os.path.join(state_dir, bot.COACH_MEMORY_FILE_NAME))


def seed_profile(state_dir, *facts):
    store = store_at(state_dir)
    store.save(replace(store.load(), profile=profile_of(*facts)))
    return store


def stub_diary(monkeypatch, days):
    """Points the walk at a diary of ``{page id: [(title, text), ...]}``."""

    async def fake_all_pages():
        return [
            {"id": page_id, "properties": {"Created": {"date": {"start": f"2026-01-0{index}"}}}}
            for index, page_id in enumerate(days, start=1)
        ]

    async def fake_read_page_entries(page):
        written = days[page["id"]]
        if written == "unreadable":
            raise RuntimeError("Notion said no")
        return [
            summary.DiaryEntry(day=summary.page_day(page), title=title, text=text)
            for title, text in written
        ]

    monkeypatch.setattr(bot, "all_pages", fake_all_pages)
    monkeypatch.setattr(bot, "read_page_entries", fake_read_page_entries)


def stub_learn(monkeypatch, script=None):
    learn = Learner(script)
    monkeypatch.setattr(coach_profile, "learn", learn)
    return learn


async def open_memory(context, fake_bot, monkeypatch, *, focus="деньги", voice=False):
    """Runs /memory and answers its focus prompt. Returns the confirmation message."""
    await bot.handle_memory(FakeUpdate(fake_bot, message=FakeMessage(fake_bot, 1)), context)
    prompt = fake_bot.sent[-1]

    if voice:

        async def fake_voice_to_text(_context, _message):
            return focus

        monkeypatch.setattr(bot, "_voice_to_text", fake_voice_to_text)
        answer = FakeMessage(
            fake_bot, 2, voice=object(), reply_to_message=FakeMessage(fake_bot, prompt.message_id)
        )
    else:
        answer = FakeMessage(
            fake_bot, 2, text=focus, reply_to_message=FakeMessage(fake_bot, prompt.message_id)
        )

    await bot.memory_focus_reply(FakeUpdate(fake_bot, message=answer), context)
    return fake_bot.sent[-1]


async def press(context, fake_bot, confirmation, data):
    message = FakeMessage(fake_bot, confirmation.message_id)
    update = FakeUpdate(
        fake_bot, message=message, callback_query=FakeCallbackQuery(fake_bot, data, message)
    )
    await bot.memory_callback(update, context)
    await context.application.run_tasks()


@pytest.mark.asyncio
async def test_memory_asks_what_the_pass_should_look_for_and_does_nothing_else(
    memory_state, context, fake_bot, monkeypatch
):
    learn = stub_learn(monkeypatch)
    seed_profile(memory_state, fact("1", "known"))

    await bot.handle_memory(FakeUpdate(fake_bot, message=FakeMessage(fake_bot, 1)), context)

    assert bot.MEMORY_FOCUS_PROMPT in fake_bot.sent[-1].text
    assert learn.calls == []
    assert store_at(memory_state).snapshots() == []


@pytest.mark.asyncio
async def test_the_focus_is_echoed_with_the_fact_count_and_still_nothing_runs(
    memory_state, context, fake_bot, monkeypatch
):
    learn = stub_learn(monkeypatch)
    seed_profile(memory_state, fact("1", "known"), fact("2", "also known"))

    confirmation = await open_memory(context, fake_bot, monkeypatch, focus="смотри на деньги")

    assert "смотри на деньги" in confirmation.text
    assert "2" in confirmation.text
    assert confirmation.reply_markup is not None
    # Two buttons, and nothing has happened behind them.
    assert [
        button.callback_data for row in confirmation.reply_markup.inline_keyboard for button in row
    ] == ["memory:run", "memory:cancel"]
    assert learn.calls == []
    assert store_at(memory_state).snapshots() == []


@pytest.mark.asyncio
async def test_a_dash_means_no_focus(memory_state, context, fake_bot, monkeypatch):
    stub_learn(monkeypatch)
    seed_profile(memory_state)

    confirmation = await open_memory(context, fake_bot, monkeypatch, focus="-")

    assert bot.MEMORY_NO_FOCUS in confirmation.text


@pytest.mark.asyncio
async def test_the_focus_is_never_stored_as_a_fact_and_never_saved_as_an_entry(
    memory_state, context, fake_bot, monkeypatch
):
    """It steers the pass and is gone. It is not diary, and it is not memory."""
    saved = []

    async def fake_save_entry(*args, **kwargs):  # pragma: no cover - must never be reached
        saved.append(args)

    monkeypatch.setattr(bot, "save_entry", fake_save_entry)
    learn = stub_learn(monkeypatch)
    seed_profile(memory_state, fact("1", "known"))
    stub_diary(monkeypatch, {"page-1": [("Entry", "text")]})

    confirmation = await open_memory(
        context, fake_bot, monkeypatch, focus="забудь всё про работу", voice=True
    )
    await press(context, fake_bot, confirmation, "memory:run")

    assert saved == []
    assert context.user_data == {}
    stored = store_at(memory_state).load().profile
    assert "забудь всё про работу" not in [item.text for item in stored.facts]
    # It did reach the model, which is the whole point of asking for it.
    assert learn.calls[0]["focus"] == "забудь всё про работу"


@pytest.mark.asyncio
async def test_cancel_runs_nothing_and_says_so(memory_state, context, fake_bot, monkeypatch):
    learn = stub_learn(monkeypatch)
    seed_profile(memory_state, fact("1", "known"))
    stub_diary(monkeypatch, {"page-1": [("Entry", "text")]})

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    await press(context, fake_bot, confirmation, "memory:cancel")

    assert fake_bot.find(confirmation.message_id).text == bot.MEMORY_CANCELLED
    assert learn.calls == []
    assert store_at(memory_state).snapshots() == []
    assert [item.text for item in store_at(memory_state).load().profile.facts] == ["known"]


@pytest.mark.asyncio
async def test_a_snapshot_exists_before_the_first_call(
    memory_state, context, fake_bot, monkeypatch
):
    """The only thing that makes a pass over the whole diary reversible."""
    seen = []
    store = seed_profile(memory_state, fact("1", "known"))

    learn = Learner()
    original = learn.__call__

    async def watching(**kwargs):
        seen.append([path.name for path in store.snapshots()])
        return await original(**kwargs)

    monkeypatch.setattr(coach_profile, "learn", watching)
    stub_diary(monkeypatch, {"page-1": [("A", "a"), ("B", "b")]})

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    await press(context, fake_bot, confirmation, "memory:run")

    assert seen[0] != []
    assert "before-rebuild" in seen[0][0]


@pytest.mark.asyncio
async def test_a_second_memory_while_one_is_running_is_refused_not_queued(
    memory_state, context, fake_bot, monkeypatch
):
    stub_learn(monkeypatch)
    seed_profile(memory_state)
    stub_diary(monkeypatch, {"page-1": [("A", "a")]})

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    # Press, but do not let the pass run: this is exactly the window a second
    # /memory arrives in.
    message = FakeMessage(fake_bot, confirmation.message_id)
    await bot.memory_callback(
        FakeUpdate(
            fake_bot,
            message=message,
            callback_query=FakeCallbackQuery(fake_bot, "memory:run", message),
        ),
        context,
    )

    await bot.handle_memory(FakeUpdate(fake_bot, message=FakeMessage(fake_bot, 9)), context)
    assert fake_bot.sent[-1].text == bot.MEMORY_IN_FLIGHT

    await context.application.run_tasks()
    # And once it has finished, /memory works again.
    await bot.handle_memory(FakeUpdate(fake_bot, message=FakeMessage(fake_bot, 10)), context)
    assert bot.MEMORY_FOCUS_PROMPT in fake_bot.sent[-1].text


@pytest.mark.asyncio
async def test_a_second_confirmation_pressed_while_one_runs_is_refused(
    memory_state, context, fake_bot, monkeypatch
):
    learn = stub_learn(monkeypatch)
    seed_profile(memory_state)
    stub_diary(monkeypatch, {"page-1": [("A", "a")]})

    first = await open_memory(context, fake_bot, monkeypatch)
    second = await open_memory(context, fake_bot, monkeypatch)

    for confirmation in (first, second):
        message = FakeMessage(fake_bot, confirmation.message_id)
        await bot.memory_callback(
            FakeUpdate(
                fake_bot,
                message=message,
                callback_query=FakeCallbackQuery(fake_bot, "memory:run", message),
            ),
            context,
        )

    assert bot.MEMORY_IN_FLIGHT in fake_bot.answered
    assert len(context.application.tasks) == 1
    await context.application.run_tasks()
    assert len(learn.calls) == 1


@pytest.mark.asyncio
async def test_a_confirmation_from_before_a_restart_asks_for_the_command_again(
    memory_state, context, fake_bot, monkeypatch
):
    learn = stub_learn(monkeypatch)
    seed_profile(memory_state)
    stub_diary(monkeypatch, {"page-1": [("A", "a")]})

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    bot._memory_confirmations.clear()  # what a restart looks like

    await press(context, fake_bot, confirmation, "memory:run")

    assert bot.MEMORY_GONE in fake_bot.answered
    assert learn.calls == []


@pytest.mark.asyncio
async def test_the_profile_is_on_disk_after_every_entry(
    memory_state, context, fake_bot, monkeypatch
):
    """A restart mid-pass loses the entry in flight and nothing else."""
    store = seed_profile(memory_state)
    on_disk = []

    learn = Learner()
    original = learn.__call__

    async def watching(**kwargs):
        on_disk.append(len(store.load().profile.facts))
        return await original(**kwargs)

    monkeypatch.setattr(coach_profile, "learn", watching)
    stub_diary(monkeypatch, {"page-1": [("A", "a"), ("B", "b"), ("C", "c")]})

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    await press(context, fake_bot, confirmation, "memory:run")

    # What was already written when each call started: nothing, then one, then two.
    assert on_disk == [0, 1, 2]
    assert len(store.load().profile.facts) == 3


@pytest.mark.asyncio
async def test_a_write_from_a_conversation_during_the_pass_is_not_rolled_back(
    memory_state, context, fake_bot, monkeypatch
):
    """The rules live in the same file, and the pass only owns the profile."""
    store = seed_profile(memory_state)
    learn = Learner()
    original = learn.__call__
    wrote_a_rule = []

    async def watching(**kwargs):
        if not wrote_a_rule:
            wrote_a_rule.append(True)
            store.save(replace(store.load(), rules=profile_of(fact("r1", "a rule"))))
        return await original(**kwargs)

    monkeypatch.setattr(coach_profile, "learn", watching)
    stub_diary(monkeypatch, {"page-1": [("A", "a"), ("B", "b")]})

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    await press(context, fake_bot, confirmation, "memory:run")

    stored = store.load()
    assert [item.text for item in stored.rules.facts] == ["a rule"]
    assert len(stored.profile.facts) == 2


@pytest.mark.asyncio
async def test_the_last_message_reports_the_delta_the_skips_and_how_to_undo_it(
    memory_state, context, fake_bot, monkeypatch
):
    store = seed_profile(memory_state, fact("1", "known"))
    stub_learn(monkeypatch, ["teach", "raise", "teach"])
    stub_diary(
        monkeypatch,
        {
            "page-1": [("A", "a"), ("B", "b")],
            "page-2": "unreadable",
            "page-3": [("C", "c")],
        },
    )

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    await press(context, fake_bot, confirmation, "memory:run")

    last = fake_bot.sent[-1].text
    assert bot.MEMORY_DONE_HEADER in last
    # One fact to start with, two entries that taught, one entry that failed and
    # one day that could not be read.
    assert "1" in last and "3" in last
    assert "пропущено: 2" in last
    snapshot = store.snapshots()[0]
    assert snapshot.name in last


@pytest.mark.asyncio
async def test_the_breaker_says_so_and_leaves_what_it_managed_to_learn(
    memory_state, context, fake_bot, monkeypatch
):
    store = seed_profile(memory_state)
    stub_learn(monkeypatch, ["teach"] + ["raise"] * 20)
    stub_diary(monkeypatch, {"page-1": [(f"E{n}", "text") for n in range(30)]})

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    await press(context, fake_bot, confirmation, "memory:run")

    aborted = bot.MEMORY_ABORTED.format(count=coach_rebuild.CONSECUTIVE_FAILURES)
    assert aborted in fake_bot.sent[-1].text
    assert len(store.load().profile.facts) == 1


@pytest.mark.asyncio
async def test_an_empty_diary_says_so_rather_than_reporting_a_delta_of_nothing(
    memory_state, context, fake_bot, monkeypatch
):
    seed_profile(memory_state, fact("1", "known"))
    stub_learn(monkeypatch)

    async def no_pages():
        return []

    monkeypatch.setattr(bot, "all_pages", no_pages)

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    await press(context, fake_bot, confirmation, "memory:run")

    assert fake_bot.sent[-1].text == bot.MEMORY_NO_ENTRIES


@pytest.mark.asyncio
async def test_a_diary_that_cannot_be_listed_is_reported_and_releases_the_guard(
    memory_state, context, fake_bot, monkeypatch
):
    seed_profile(memory_state)
    stub_learn(monkeypatch)

    async def broken():
        raise RuntimeError("Notion said no")

    monkeypatch.setattr(bot, "all_pages", broken)

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    await press(context, fake_bot, confirmation, "memory:run")

    assert fake_bot.sent[-1].text == bot.MEMORY_FAILED
    assert bot._rebuild_in_flight is False


@pytest.mark.asyncio
async def test_a_memory_file_that_cannot_be_read_stops_the_pass_before_it_starts(
    memory_state, context, fake_bot, monkeypatch
):
    learn = stub_learn(monkeypatch)
    stub_diary(monkeypatch, {"page-1": [("A", "a")]})
    confirmation = await open_memory(context, fake_bot, monkeypatch)

    broken = os.path.join(memory_state, bot.COACH_MEMORY_FILE_NAME)
    with open(broken, "w", encoding="utf-8") as handle:
        handle.write("this is not JSON")

    await press(context, fake_bot, confirmation, "memory:run")

    assert fake_bot.sent[-1].text == bot.MEMORY_UNAVAILABLE
    assert learn.calls == []
    assert bot._rebuild_in_flight is False


def test_the_focus_answer_is_routed_ahead_of_the_conversation(memory_state, monkeypatch):
    """A dictated focus must not reach the entry point that turns voice into a draft."""
    application = bot.build_application()
    handlers = application.handlers[0]
    callbacks = [getattr(handler, "callback", None) for handler in handlers]

    focus_at = callbacks.index(bot.memory_focus_reply)
    conversation_at = next(
        index
        for index, handler in enumerate(handlers)
        if handler.__class__.__name__ == "ConversationHandler"
    )

    assert focus_at < conversation_at


def test_the_filter_matches_only_a_reply_to_a_prompt_this_process_is_waiting_on(memory_state):
    fake = FakeBot()
    bot._memory_focus_prompts.clear()
    matching = FakeMessage(fake, 2, text="деньги", reply_to_message=FakeMessage(fake, 55))

    assert bot.MemoryFocusFilter().filter(matching) is False

    bot._memory_focus_prompts[(fake.chat_id, 55)] = None
    assert bot.MemoryFocusFilter().filter(matching) is True
    assert bot.MemoryFocusFilter().filter(FakeMessage(fake, 3, text="обычное сообщение")) is False


# --------------------------------------------------------------------------- #
# The rebuild and the two Notion memory pages
#
# Added when `feat-coach-notion-memory` merged this branch: both merged cleanly
# and the product did not. The mirror adopts whatever the profile page says on
# the next pull, so a rebuild that wrote the file and left the page alone would
# have been undone in full the next time the coach answered — the most expensive
# outcome either branch can produce, and one neither of them could have on its
# own.
# --------------------------------------------------------------------------- #


@pytest.fixture
def pages(monkeypatch):
    return FakePages().install(monkeypatch)


@pytest.mark.asyncio
async def test_what_a_rebuild_produced_reaches_the_page(
    memory_state, context, fake_bot, monkeypatch, pages
):
    """Or the next pull adopts the old page and the whole pass is thrown away."""
    seed_profile(memory_state, fact("1", "known"))
    stub_diary(monkeypatch, {"day-1": [("Заголовок", "тело")]})
    stub_learn(monkeypatch, ["teach"])

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    await press(context, fake_bot, confirmation, "memory:run")

    assert pages.texts("profile-page") == ["known", "learned from Заголовок"]


@pytest.mark.asyncio
async def test_a_rebuild_that_broke_half_way_still_leaves_the_page_agreeing(
    memory_state, context, fake_bot, monkeypatch, pages
):
    """Everything it learned before it stopped is on disk, so it is on the page."""
    seed_profile(memory_state, fact("1", "known"))
    stub_diary(monkeypatch, {"day-1": "unreadable"})
    stub_learn(monkeypatch)

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    await press(context, fake_bot, confirmation, "memory:run")

    assert pages.texts("profile-page") == ["known"]


@pytest.mark.asyncio
async def test_a_rebuild_starts_from_the_page_rather_than_from_the_file(
    memory_state, context, fake_bot, monkeypatch, pages
):
    """He reworded a fact in Notion, then asked for a rebuild. His wording stands."""
    store = seed_profile(memory_state, fact("1", "known"))
    await bot._memory_sync().push()
    key = store.load().profile.facts[0].key
    pages.put("profile-page", (key, "his own wording"))
    stub_diary(monkeypatch, {"day-1": [("Заголовок", "тело")]})
    stub_learn(monkeypatch, ["teach"])

    confirmation = await open_memory(context, fake_bot, monkeypatch)
    await press(context, fake_bot, confirmation, "memory:run")

    assert [item.text for item in store_at(memory_state).load().profile.facts] == [
        "his own wording",
        "learned from Заголовок",
    ]
