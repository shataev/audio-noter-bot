"""The one message a week the coach sends without being asked.

The interesting properties are all about *not* sending: silence on a week with
nothing in it, silence on a week that has already been spoken about — including
across the restart every deploy causes — and silence in the chat when something
goes wrong, with the reason in the log instead.

No credentials on this machine, so Notion is a dictionary of fake pages, the
provider is a recording double, and Telegram is a small stand-in of the same
shape the rest of the suite uses.
"""

import os

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import logging
from dataclasses import replace
from datetime import date, time

import pytest
from notion_pages_fake import FakePages

import bot
from config import WEEKDAYS
from services import summary
from services.ai import Completion
from services.coach import diary as coach_diary
from services.coach import memory as coach_memory
from services.coach import threads as coach_threads
from services.coach import weekly as coach_weekly
from services.coach.store import MemoryStore

ANSWER = "Ты третью неделю пишешь про одно и то же и каждый раз называешь это иначе."


def fact(fact_id, text="a fact", kind="trait"):
    return coach_memory.Fact(
        id=fact_id,
        text=text,
        kind=kind,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def profile_of(*facts):
    return coach_memory.MemoryList(facts=tuple(facts), next_id=len(facts) + 1)


def entry(number, text=None):
    return coach_diary.Entry(
        source=f"2026-09-0{number} · Entry {number}",
        title=f"Entry {number}",
        text=text or f"text {number}",
    )


class FakeChatClient:
    def __init__(self, text=ANSWER, fail=False):
        self.text = text
        self.fail = fail
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("the provider said no")
        return Completion(text=self.text, finish_reason="stop")


# --------------------------------------------------------------------------- #
# services/coach/weekly.py
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_week_and_the_profile_both_reach_the_model():
    client = FakeChatClient()

    text = await coach_weekly.session(
        profile=profile_of(fact("1", "он бегает по утрам")),
        entries=[entry(1), entry(2)],
        model="m",
        client=client,
    )

    assert text == ANSWER
    call = client.calls[0]
    assert "он бегает по утрам" in call["system"]
    assert "text 1" in call["messages"][0].content
    assert "text 2" in call["messages"][0].content
    # It reasons, and it is told not to write a recap.
    assert call["effort"] == "high"
    assert "пересказ" in call["system"]


@pytest.mark.asyncio
async def test_a_week_with_nothing_in_it_is_never_even_asked_about():
    client = FakeChatClient()

    said = await coach_weekly.session(profile=profile_of(), entries=[], model="m", client=client)

    assert said == ""
    assert client.calls == []


@pytest.mark.asyncio
async def test_a_provider_that_says_no_comes_back_as_nothing_to_send():
    text = await coach_weekly.session(
        profile=profile_of(),
        entries=[entry(1)],
        model="m",
        client=FakeChatClient(fail=True),
    )

    assert text == ""


@pytest.mark.asyncio
async def test_an_answer_with_no_text_in_it_comes_back_as_nothing_to_send():
    text = await coach_weekly.session(
        profile=profile_of(),
        entries=[entry(1)],
        model="m",
        client=FakeChatClient(text="   "),
    )

    assert text == ""


def test_a_week_is_addressed_by_its_iso_week_not_by_the_day_it_was_sent():
    # The Sunday that ends ISO week 37 of 2026, and the Monday that opens week 38.
    assert coach_weekly.week_key(date(2026, 9, 13)) == "2026-W37"
    assert coach_weekly.week_key(date(2026, 9, 14)) == "2026-W38"
    assert coach_weekly.week_key(date(2026, 1, 4)) == "2026-W01"


def test_a_week_that_has_been_spoken_about_is_written_down_and_read_back(tmp_path):
    path = tmp_path / "coach_weekly.state.json"

    assert coach_weekly.last_spoken(path) is None

    coach_weekly.remember(path, "2026-W37")
    assert coach_weekly.last_spoken(path) == "2026-W37"

    coach_weekly.remember(path, "2026-W38")
    assert coach_weekly.last_spoken(path) == "2026-W38"


@pytest.mark.parametrize("contents", ["", "not json at all", "[]", '{"last_week": 7}'])
def test_a_damaged_state_file_reads_as_nothing_recorded(tmp_path, contents):
    """Costing one extra message beats refusing to run the job at all."""
    path = tmp_path / "coach_weekly.state.json"
    path.write_text(contents, encoding="utf-8")

    assert coach_weekly.last_spoken(path) is None


# --------------------------------------------------------------------------- #
# bot.py — the scheduled job.
# --------------------------------------------------------------------------- #


class Sent:
    def __init__(self, message_id, text, reply_to):
        self.message_id = message_id
        self.text = text
        self.reply_to = reply_to


class FakeBot:
    def __init__(self, chat_id=1, fail=False):
        self.chat_id = chat_id
        self.sent = []
        self.fail = fail
        self._next_id = 100

    async def send_message(
        self, chat_id, text, parse_mode=None, reply_markup=None, reply_parameters=None, **kwargs
    ):
        if self.fail:
            raise RuntimeError("Telegram said no")
        self._next_id += 1
        message = Sent(
            self._next_id,
            text,
            reply_parameters.message_id if reply_parameters is not None else None,
        )
        self.sent.append(message)
        return message


class FakeContext:
    def __init__(self, fake_bot):
        self.bot = fake_bot
        self.bot_data = {}


@pytest.fixture
def weekly_state(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(bot, "_thread_cache", None)
    return tmp_path


@pytest.fixture
def fake_bot():
    return FakeBot()


@pytest.fixture
def context(fake_bot):
    return FakeContext(fake_bot)


def stub_week(monkeypatch, days):
    """A week of ``{page id: [(title, text), ...]}``, or ``"unreadable"`` for a bad page."""

    async def fake_get_week_pages():
        return [
            {"id": page_id, "properties": {"Created": {"date": {"start": f"2026-09-0{index}"}}}}
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

    monkeypatch.setattr(bot, "get_week_pages", fake_get_week_pages)
    monkeypatch.setattr(bot, "read_page_entries", fake_read_page_entries)


def stub_client(monkeypatch, **kwargs):
    client = FakeChatClient(**kwargs)
    monkeypatch.setattr(coach_weekly, "_client", client)
    return client


def stub_today(monkeypatch, day=date(2026, 9, 13)):
    monkeypatch.setattr(bot, "diary_today", lambda *args, **kwargs: day)


def weekly_path(state_dir):
    return os.path.join(state_dir, bot.COACH_WEEKLY_FILE_NAME)


@pytest.mark.asyncio
async def test_the_session_sends_one_message_and_writes_the_week_down(
    weekly_state, context, fake_bot, monkeypatch
):
    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": [("A", "a")], "page-2": [("B", "b")]})
    client = stub_client(monkeypatch)

    await bot.send_coach_weekly(context)

    assert [message.text for message in fake_bot.sent] == [ANSWER]
    assert coach_weekly.last_spoken(weekly_path(weekly_state)) == "2026-W37"
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_an_empty_week_gets_silence_and_stays_eligible(
    weekly_state, context, fake_bot, monkeypatch
):
    """No entries is not an occasion for a generated observation about nothing."""
    stub_today(monkeypatch)
    stub_week(monkeypatch, {})
    client = stub_client(monkeypatch)

    await bot.send_coach_weekly(context)

    assert fake_bot.sent == []
    assert client.calls == []
    # Nothing was said, so nothing is written down: an entry later this week
    # still deserves its message.
    assert coach_weekly.last_spoken(weekly_path(weekly_state)) is None


@pytest.mark.asyncio
async def test_a_week_of_empty_entries_is_an_empty_week(
    weekly_state, context, fake_bot, monkeypatch
):
    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": [("A", "   ")]})
    client = stub_client(monkeypatch)

    await bot.send_coach_weekly(context)

    assert fake_bot.sent == []
    assert client.calls == []


@pytest.mark.asyncio
async def test_the_same_week_is_never_spoken_about_twice_even_across_a_restart(
    weekly_state, context, fake_bot, monkeypatch
):
    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": [("A", "a")]})
    client = stub_client(monkeypatch)

    await bot.send_coach_weekly(context)
    # A deploy: the process is new, its caches are empty, only the files survive.
    monkeypatch.setattr(bot, "_thread_cache", None)
    await bot.send_coach_weekly(context)

    assert len(fake_bot.sent) == 1
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_the_next_week_is_spoken_about(weekly_state, context, fake_bot, monkeypatch):
    stub_week(monkeypatch, {"page-1": [("A", "a")]})
    stub_client(monkeypatch)

    stub_today(monkeypatch, date(2026, 9, 13))
    await bot.send_coach_weekly(context)
    stub_today(monkeypatch, date(2026, 9, 20))
    await bot.send_coach_weekly(context)

    assert len(fake_bot.sent) == 2
    assert coach_weekly.last_spoken(weekly_path(weekly_state)) == "2026-W38"


@pytest.mark.asyncio
async def test_the_message_is_repliable(weekly_state, context, fake_bot, monkeypatch):
    """It lands in a coach thread, so a reply to it continues a conversation."""
    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": [("A", "a")]})
    stub_client(monkeypatch)

    await bot.send_coach_weekly(context)

    sent = fake_bot.sent[0]
    assert bot._knows_coach_message(1, sent.message_id) is True
    thread = coach_threads.find(bot._load_threads(), coach_threads.key(1, sent.message_id))
    assert thread is not None
    assert thread.turns[-1].text == ANSWER
    assert bot.coach_prompts.mode_for(thread.mode) is not None


@pytest.mark.asyncio
async def test_every_message_of_a_long_answer_is_a_door_into_the_conversation(
    weekly_state, context, fake_bot, monkeypatch
):
    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": [("A", "a")]})
    stub_client(monkeypatch, text="\n\n".join(["абзац " + "х" * 3000] * 2))

    await bot.send_coach_weekly(context)

    assert len(fake_bot.sent) == 2
    for message in fake_bot.sent:
        assert bot._knows_coach_message(1, message.message_id) is True
    # The follow-ups hang off the first message rather than arriving loose.
    assert fake_bot.sent[1].reply_to == fake_bot.sent[0].message_id


@pytest.mark.asyncio
async def test_a_failure_is_silent_in_the_chat_and_loud_in_the_log(
    weekly_state, context, fake_bot, monkeypatch, caplog
):
    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": [("A", "a")]})
    stub_client(monkeypatch, fail=True)

    with caplog.at_level(logging.INFO):
        await bot.send_coach_weekly(context)

    assert fake_bot.sent == []
    assert coach_weekly.last_spoken(weekly_path(weekly_state)) is None
    assert any("weekly" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_a_send_that_fails_leaves_the_week_unspoken(weekly_state, context, monkeypatch):
    """Otherwise a Telegram outage costs the week its message for good."""
    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": [("A", "a")]})
    stub_client(monkeypatch)

    await bot.send_coach_weekly(FakeContext(FakeBot(fail=True)))

    assert coach_weekly.last_spoken(weekly_path(weekly_state)) is None


@pytest.mark.asyncio
async def test_one_unreadable_day_does_not_cost_the_week_its_message(
    weekly_state, context, fake_bot, monkeypatch
):
    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": "unreadable", "page-2": [("B", "b")]})
    client = stub_client(monkeypatch)

    await bot.send_coach_weekly(context)

    assert len(fake_bot.sent) == 1
    assert "b" in client.calls[0]["messages"][0].content


@pytest.mark.asyncio
async def test_the_profile_is_read_from_the_store(weekly_state, context, monkeypatch):
    store = MemoryStore(os.path.join(weekly_state, bot.COACH_MEMORY_FILE_NAME))
    store.save(replace(store.load(), profile=profile_of(fact("1", "он поздно ложится"))))
    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": [("A", "a")]})
    client = stub_client(monkeypatch)

    await bot.send_coach_weekly(context)

    assert "он поздно ложится" in client.calls[0]["system"]


@pytest.mark.asyncio
async def test_the_job_does_nothing_when_it_is_switched_off(
    weekly_state, context, fake_bot, monkeypatch
):
    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": [("A", "a")]})
    client = stub_client(monkeypatch)
    monkeypatch.setattr(bot.settings, "coach_weekly_enabled", False)

    await bot.send_coach_weekly(context)

    assert fake_bot.sent == []
    assert client.calls == []


# --------------------------------------------------------------------------- #
# The schedule itself.
# --------------------------------------------------------------------------- #


def jobs(application, name):
    return application.job_queue.jobs() and [
        job for job in application.job_queue.jobs() if job.name == name
    ]


def test_the_weekly_session_is_scheduled_on_the_configured_day_and_hour(weekly_state, monkeypatch):
    monkeypatch.setattr(bot.settings, "coach_weekly_enabled", True)
    monkeypatch.setattr(bot.settings, "coach_weekly_day", WEEKDAYS.index("wednesday"))
    monkeypatch.setattr(bot.settings, "coach_weekly_hour", 9)
    monkeypatch.setattr(bot.settings, "coach_weekly_minute", 30)

    application = bot.build_application()

    job = jobs(application, bot.COACH_WEEKLY_JOB)[0]
    trigger = job.job.trigger
    assert str(trigger.fields[trigger.FIELD_NAMES.index("day_of_week")]) == "wed"
    assert job.job.trigger.fields[trigger.FIELD_NAMES.index("hour")].expressions[0].first == 9


def test_switching_it_off_leaves_no_job_at_all(weekly_state, monkeypatch):
    monkeypatch.setattr(bot.settings, "coach_weekly_enabled", False)

    application = bot.build_application()

    assert not jobs(application, bot.COACH_WEEKLY_JOB)
    # And the two that were always there are untouched.
    assert jobs(application, bot.DAILY_SUMMARY_JOB)
    assert jobs(application, bot.WEEKLY_REPORT_JOB)


def test_the_default_day_and_hour_stay_clear_of_the_two_recaps(weekly_state, monkeypatch):
    """Both recaps go out at 21:00; a third message in the same minute is one message."""
    monkeypatch.setattr(bot.settings, "coach_weekly_enabled", True)

    application = bot.build_application()

    job = jobs(application, bot.COACH_WEEKLY_JOB)[0]
    trigger = job.job.trigger
    hour = trigger.fields[trigger.FIELD_NAMES.index("hour")].expressions[0].first
    assert time(hour) != time(21)


def test_the_weekday_setting_is_named_not_numbered():
    """`run_daily` numbers its days Sunday-first, which nobody should have to know."""
    assert WEEKDAYS[0] == "sunday"
    assert WEEKDAYS[bot.SUNDAY] == "sunday"
    assert len(WEEKDAYS) == 7


# --------------------------------------------------------------------------- #
# The weekly session and the Notion memory pages
#
# Added when `feat-coach-notion-memory` merged this branch. The profile the
# session reasons about is the one the owner can edit on a Notion page, and this
# message is an answer like any other: it reads the page first.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_profile_it_reasons_about_is_the_one_on_the_page(
    weekly_state, context, fake_bot, monkeypatch
):
    pages = FakePages().install(monkeypatch)
    # Seeded with the block id already recorded rather than pushed for it: a push
    # would count as a sync and the throttled pull below would reuse it, which is
    # the right production behaviour and would prove nothing here.
    store = MemoryStore(os.path.join(weekly_state, bot.COACH_MEMORY_FILE_NAME))
    mirrored = replace(fact("1", "what the bot wrote"), key="b1")
    store.save(replace(store.load(), profile=profile_of(mirrored)))
    pages.put("profile-page", ("b1", "what he corrected it to"))

    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": [("A", "a")]})
    client = stub_client(monkeypatch)

    await bot.send_coach_weekly(context)

    shown = client.calls[0]["system"] + client.calls[0]["messages"][0].content
    assert "what he corrected it to" in shown
    assert "what the bot wrote" not in shown


@pytest.mark.asyncio
async def test_notion_being_down_does_not_cost_the_week_its_message(
    weekly_state, context, fake_bot, monkeypatch
):
    pages = FakePages().install(monkeypatch)
    pages.fails = RuntimeError("Notion said no")
    store = MemoryStore(os.path.join(weekly_state, bot.COACH_MEMORY_FILE_NAME))
    store.save(replace(store.load(), profile=profile_of(fact("1", "what the bot wrote"))))

    stub_today(monkeypatch)
    stub_week(monkeypatch, {"page-1": [("A", "a")]})
    stub_client(monkeypatch)

    await bot.send_coach_weekly(context)

    assert [message.text for message in fake_bot.sent] == [ANSWER]
