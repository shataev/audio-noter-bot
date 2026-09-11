"""Unit tests for the bot's handlers and job scheduling.

There are no credentials on this machine: every Telegram, OpenAI and Notion call is
replaced by a stub. The environment variables below are what ``config`` reads at import
time, so they have to be in place before any project module is imported.
"""

import os

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import pytest

from telegram.constants import ParseMode
from telegram.error import BadRequest

import bot


# --------------------------------------------------------------------------- #
# A stub Telegram, standing in for the Bot API.
#
# `_check_parse` is the part that matters: it rejects a message body the way the
# API would, so a test fails on the content of the message rather than on an
# assertion about how it was built. The HTML branch is exact — Telegram's HTML
# parse mode accepts a fixed set of tags in well-formed XML, which is what
# ElementTree checks. The legacy-Markdown branch is a deliberate simplification:
# `*`, `_` and backtick open and close entities, and an odd number of them leaves
# one unterminated, which the API rejects with "Can't parse entities". It is only
# used to show what the old code sent; nothing in the bot uses Markdown any more.
# --------------------------------------------------------------------------- #

from xml.etree import ElementTree


def _check_parse(text, parse_mode):
    if parse_mode is None:
        return
    if parse_mode == ParseMode.HTML:
        try:
            ElementTree.fromstring(f"<root>{text}</root>")
        except ElementTree.ParseError as exc:
            raise BadRequest(f"Can't parse entities: {exc}") from exc
        return
    for marker in ("*", "_", "`"):
        if text.count(marker) % 2:
            raise BadRequest(f"Can't parse entities: unterminated entity for {marker!r}")


class Sent:
    def __init__(self, message_id, text, parse_mode, reply_markup):
        self.message_id = message_id
        self.text = text
        self.parse_mode = parse_mode
        self.reply_markup = reply_markup


class FakeChat:
    def __init__(self, chat_id):
        self.id = chat_id


class FakeBot:
    """Records what the bot sends and validates it the way the API would."""

    def __init__(self, chat_id=1):
        self.chat_id = chat_id
        self.sent = []
        self.deleted = []
        self.answered = []
        self.files = {}
        self._next_id = 100

    def _new_id(self):
        self._next_id += 1
        return self._next_id

    def _record(self, text, parse_mode, reply_markup):
        _check_parse(text, parse_mode)
        message = Sent(self._new_id(), text, parse_mode, reply_markup)
        self.sent.append(message)
        return message

    def find(self, message_id):
        for message in reversed(self.sent):
            if message.message_id == message_id:
                return message
        raise AssertionError(f"no message {message_id}")

    async def send_message(self, chat_id, text, parse_mode=None, reply_markup=None, **kwargs):
        return self._record(text, parse_mode, reply_markup)

    async def edit_message_text(self, chat_id, message_id, text, parse_mode=None, reply_markup=None, **kwargs):
        _check_parse(text, parse_mode)
        message = self.find(message_id)
        message.text = text
        message.parse_mode = parse_mode
        message.reply_markup = reply_markup
        return message

    async def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None, **kwargs):
        message = self.find(message_id)
        message.reply_markup = reply_markup
        return message

    async def delete_message(self, chat_id, message_id, **kwargs):
        self.deleted.append(message_id)

    async def get_file(self, file_id):
        return self.files[file_id]


class FakeMessage:
    def __init__(self, fake_bot, message_id=1, text=None, voice=None):
        self._bot = fake_bot
        self.message_id = message_id
        self.text = text
        self.voice = voice
        self.chat = FakeChat(fake_bot.chat_id)

    async def reply_text(self, text, parse_mode=None, reply_markup=None, **kwargs):
        sent = self._bot._record(text, parse_mode, reply_markup)
        return FakeMessage(self._bot, message_id=sent.message_id, text=text)


class FakeVoice:
    def __init__(self, file_id="voice-1"):
        self.file_id = file_id


class FakeCallbackQuery:
    def __init__(self, fake_bot, data, message):
        self._bot = fake_bot
        self.data = data
        self.message = message

    async def answer(self, text=None, **kwargs):
        self._bot.answered.append(text)

    async def edit_message_text(self, text, parse_mode=None, reply_markup=None, **kwargs):
        return await self._bot.edit_message_text(
            self._bot.chat_id, self.message.message_id, text,
            parse_mode=parse_mode, reply_markup=reply_markup,
        )

    async def edit_message_reply_markup(self, reply_markup=None, **kwargs):
        return await self._bot.edit_message_reply_markup(
            self._bot.chat_id, self.message.message_id, reply_markup=reply_markup,
        )


class FakeUpdate:
    def __init__(self, fake_bot, message=None, callback_query=None):
        self.effective_message = message
        self.callback_query = callback_query
        self.effective_chat = FakeChat(fake_bot.chat_id)


class FakeContext:
    def __init__(self, fake_bot):
        self.bot = fake_bot
        self.user_data = {}


class FakeFile:
    """Stands in for telegram.File; writes a byte into the destination path."""

    def __init__(self, fail=False):
        self.fail = fail

    async def download_to_drive(self, path):
        if self.fail:
            raise BadRequest("File is too big")
        with open(path, "wb") as handle:
            handle.write(b"ogg")


@pytest.fixture
def fake_bot():
    return FakeBot()


@pytest.fixture
def context(fake_bot):
    return FakeContext(fake_bot)


def voice_update(fake_bot, message_id=1):
    return FakeUpdate(fake_bot, message=FakeMessage(fake_bot, message_id, voice=FakeVoice()))


def text_update(fake_bot, text, message_id=50):
    return FakeUpdate(fake_bot, message=FakeMessage(fake_bot, message_id, text=text))


def callback_update(fake_bot, data, message_id):
    message = FakeMessage(fake_bot, message_id)
    return FakeUpdate(fake_bot, message=message, callback_query=FakeCallbackQuery(fake_bot, data, message))


async def stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, title, text, tags, download_fails=False):
    """Points the voice pipeline at stubs and at a temp dir the test can inspect."""
    monkeypatch.setattr(bot.tempfile, "tempdir", str(tmp_path))
    fake_bot.files["voice-1"] = FakeFile(fail=download_fails)

    async def fake_transcribe(path):
        return "raw transcription"

    async def fake_format(transcription):
        return title, text, tags

    monkeypatch.setattr(bot, "transcribe", fake_transcribe)
    monkeypatch.setattr(bot, "format_entry", fake_format)


def preview_bodies(fake_bot):
    return [message.text for message in fake_bot.sent]


# --------------------------------------------------------------------------- #
# Defect 1 — the weekly report fired on Saturday
# --------------------------------------------------------------------------- #


def _trigger_fields(job):
    return {field.name: str(field) for field in job.job.trigger.fields}


def test_weekly_report_is_scheduled_for_sunday(tmp_path, monkeypatch):
    """`days=(6,)` means Saturday in python-telegram-bot >= 20; Sunday is 0."""
    monkeypatch.setattr(bot, "STATE_FILE", str(tmp_path / "state.pickle"), raising=False)
    app = bot.build_application()

    jobs = {job.name: job for job in app.job_queue.jobs()}
    assert bot.WEEKLY_REPORT_JOB in jobs

    fields = _trigger_fields(jobs[bot.WEEKLY_REPORT_JOB])
    assert fields["day_of_week"] == "sun"
    assert fields["hour"] == "21"
    assert fields["minute"] == "0"


def test_daily_summary_runs_every_day(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "STATE_FILE", str(tmp_path / "state.pickle"), raising=False)
    app = bot.build_application()

    jobs = {job.name: job for job in app.job_queue.jobs()}
    fields = _trigger_fields(jobs[bot.DAILY_SUMMARY_JOB])
    assert fields["day_of_week"] == "sun,mon,tue,wed,thu,fri,sat"
    assert fields["hour"] == "21"


# --------------------------------------------------------------------------- #
# Defect 5 — formatting characters in generated text broke the message
# --------------------------------------------------------------------------- #

AWKWARD_TITLE = "Отчёт за 1*2 недели & R&D <итоги>"
AWKWARD_TAGS = ["c++_dev", "prod`ready"]
AWKWARD_REPORT = "Неделя была насыщенной.\n- 5*5 тренировок\n- R&D <итоги>"


@pytest.mark.asyncio
async def test_preview_survives_formatting_characters_in_title(tmp_path, monkeypatch, fake_bot, context):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, AWKWARD_TITLE, "тело записи", [])

    state = await bot.handle_voice(voice_update(fake_bot), context)

    assert state == bot.PREVIEW
    assert context.user_data["pending"]["title"] == AWKWARD_TITLE
    title_body = fake_bot.find(context.user_data["title_msg_id"]).text
    assert title_body == "<b>Отчёт за 1*2 недели &amp; R&amp;D &lt;итоги&gt;</b>"


@pytest.mark.asyncio
async def test_tags_typed_by_the_user_do_not_break_the_message(fake_bot, context):
    context.user_data.update({
        "pending": {"title": "Заголовок", "text": "тело", "tags": []},
        "title_msg_id": (await fake_bot.send_message(1, "<b>Заголовок</b>", ParseMode.HTML)).message_id,
        "text_msg_id": (await fake_bot.send_message(1, "тело")).message_id,
        "tags_msg_id": (await fake_bot.send_message(1, bot._tags_line([]), ParseMode.HTML)).message_id,
        "buttons_msg_id": (await fake_bot.send_message(1, "Actions:")).message_id,
        "edit_prompt_msg_id": (await fake_bot.send_message(1, "Send tags separated by commas:")).message_id,
    })

    state = await bot.receive_new_tags(text_update(fake_bot, ", ".join(AWKWARD_TAGS)), context)

    assert state == bot.PREVIEW
    tags_body = fake_bot.find(context.user_data["tags_msg_id"]).text
    assert tags_body == "<code>Daily</code> <code>c++_dev</code> <code>prod`ready</code>"


@pytest.mark.asyncio
async def test_weekly_report_with_an_unpaired_asterisk_is_delivered(monkeypatch, fake_bot, context):
    async def fake_report():
        return AWKWARD_REPORT

    monkeypatch.setattr(bot, "generate_weekly_report", fake_report)

    await bot.handle_weekly(text_update(fake_bot, "/weekly"), context)

    body = fake_bot.sent[-1].text
    assert body.startswith("<b>Weekly highlights</b>")
    assert "5*5 тренировок" in body
    assert "R&amp;D &lt;итоги&gt;" in body


@pytest.mark.asyncio
async def test_daily_summary_job_with_an_unpaired_asterisk_is_delivered(monkeypatch, fake_bot, context):
    async def fake_summary():
        return "День прошёл 3*4 раза лучше"

    monkeypatch.setattr(bot, "generate_daily_summary", fake_summary)

    await bot.send_daily_summary(context)

    body = fake_bot.sent[-1].text
    assert body == "<b>Daily summary</b>\n\nДень прошёл 3*4 раза лучше"


def test_render_escapes_every_interpolated_value():
    assert bot.render("<b>{a}</b> {b}", a="<x>", b="a & b") == "<b>&lt;x&gt;</b> a &amp; b"


# --------------------------------------------------------------------------- #
# Defects 6 and 7 — exception text in the chat, and a leaked temp file
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_failing_download_leaves_no_temp_file(tmp_path, monkeypatch, fake_bot, context):
    await stub_voice_pipeline(
        monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [], download_fails=True
    )

    state = await bot.handle_voice(voice_update(fake_bot), context)

    assert state == bot.ConversationHandler.END
    assert list(tmp_path.iterdir()) == []
    assert "pending" not in context.user_data


@pytest.mark.asyncio
async def test_failing_download_reports_plainly(tmp_path, monkeypatch, fake_bot, context):
    await stub_voice_pipeline(
        monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [], download_fails=True
    )

    await bot.handle_voice(voice_update(fake_bot), context)

    assert fake_bot.sent[-1].text == bot.DOWNLOAD_FAILED
    assert "File is too big" not in fake_bot.sent[-1].text


@pytest.mark.asyncio
async def test_failing_transcription_is_distinguishable_and_cleans_up(tmp_path, monkeypatch, fake_bot, context):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])

    async def boom(path):
        raise RuntimeError("openai said no: sk-secret-ish detail")

    monkeypatch.setattr(bot, "transcribe", boom)

    state = await bot.handle_voice(voice_update(fake_bot), context)

    assert state == bot.ConversationHandler.END
    assert list(tmp_path.iterdir()) == []
    assert fake_bot.sent[-1].text == bot.TRANSCRIBE_FAILED
    assert bot.TRANSCRIBE_FAILED != bot.DOWNLOAD_FAILED
    assert "sk-secret-ish detail" not in fake_bot.sent[-1].text


@pytest.mark.asyncio
async def test_successful_run_also_removes_the_temp_file(tmp_path, monkeypatch, fake_bot, context):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", ["sport"])

    state = await bot.handle_voice(voice_update(fake_bot), context)

    assert state == bot.PREVIEW
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Defect 2 — an abandoned preview wedged the bot until restart
# --------------------------------------------------------------------------- #

from datetime import datetime, timezone

from telegram import Chat, Message, Update, User, Voice


def _real_voice_update(update_id=1, message_id=7):
    """A genuine telegram.Update, so the ConversationHandler's own routing is exercised."""
    user = User(id=1, first_name="Owner", is_bot=False)
    chat = Chat(id=1, type=Chat.PRIVATE)
    message = Message(
        message_id=message_id,
        date=datetime.now(timezone.utc),
        chat=chat,
        from_user=user,
        voice=Voice(file_id="voice-1", file_unique_id="unique-1", duration=3),
    )
    return Update(update_id=update_id, message=message)


def _conversation_handler(app):
    for handler in app.handlers[0]:
        if isinstance(handler, bot.ConversationHandler):
            return handler
    raise AssertionError("no ConversationHandler registered")


def test_a_second_voice_message_during_a_preview_is_not_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "STATE_FILE", str(tmp_path / "state.pickle"), raising=False)
    conv = _conversation_handler(bot.build_application())
    update = _real_voice_update()

    # No conversation yet: the entry point matches, as it always did.
    assert conv.check_update(update) is not None

    # Now pretend a preview is already open and left untouched.
    key = (update.effective_chat.id, update.effective_user.id)
    conv._conversations[key] = bot.PREVIEW

    assert conv.check_update(update) is not None, (
        "a voice message sent during an open preview must be handled, not swallowed"
    )


@pytest.mark.parametrize("state_name", ["PREVIEW", "EDIT_TITLE", "EDIT_TEXT", "EDIT_TAGS"])
def test_a_voice_message_is_accepted_from_every_state(tmp_path, monkeypatch, state_name):
    monkeypatch.setattr(bot, "STATE_FILE", str(tmp_path / "state.pickle"), raising=False)
    conv = _conversation_handler(bot.build_application())
    update = _real_voice_update()

    key = (update.effective_chat.id, update.effective_user.id)
    conv._conversations[key] = getattr(bot, state_name)

    assert conv.check_update(update) is not None


def test_cancel_is_reachable_from_every_state(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "STATE_FILE", str(tmp_path / "state.pickle"), raising=False)
    conv = _conversation_handler(bot.build_application())

    assert any(
        getattr(handler, "commands", None) == frozenset({"cancel"})
        for handler in conv.fallbacks
    ), "/cancel must be a fallback so it works in every state"
    assert conv.conversation_timeout == bot.PREVIEW_TIMEOUT


@pytest.mark.asyncio
async def test_a_new_recording_takes_the_buttons_off_the_old_preview(tmp_path, monkeypatch, fake_bot, context):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Первая", "тело", [])
    await bot.handle_voice(voice_update(fake_bot, message_id=1), context)

    first_buttons_id = context.user_data["buttons_msg_id"]
    assert fake_bot.find(first_buttons_id).reply_markup is not None

    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Вторая", "тело", [])
    state = await bot.handle_voice(voice_update(fake_bot, message_id=2), context)

    assert state == bot.PREVIEW
    assert context.user_data["pending"]["title"] == "Вторая"
    assert context.user_data["buttons_msg_id"] != first_buttons_id

    retired = fake_bot.find(first_buttons_id)
    assert retired.reply_markup is None
    assert retired.text == bot.DRAFT_REPLACED


@pytest.mark.asyncio
async def test_cancel_discards_the_draft_and_its_messages(tmp_path, monkeypatch, fake_bot, context):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", ["sport"])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]
    content_ids = [context.user_data[key] for key in ("title_msg_id", "text_msg_id", "tags_msg_id")]

    state = await bot.handle_cancel(text_update(fake_bot, "/cancel"), context)

    assert state == bot.ConversationHandler.END
    assert "pending" not in context.user_data
    assert sorted(fake_bot.deleted) == sorted(content_ids)
    assert fake_bot.find(buttons_id).reply_markup is None
    assert fake_bot.find(buttons_id).text == bot.DRAFT_CANCELLED


@pytest.mark.asyncio
async def test_cancel_without_a_draft_says_so(fake_bot, context):
    state = await bot.handle_cancel(text_update(fake_bot, "/cancel"), context)

    assert state == bot.ConversationHandler.END
    assert fake_bot.sent[-1].text == bot.NOTHING_TO_CANCEL


@pytest.mark.asyncio
async def test_timeout_takes_the_buttons_off_the_stale_preview(tmp_path, monkeypatch, fake_bot, context):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]

    state = await bot.handle_preview_timeout(text_update(fake_bot, "anything"), context)

    assert state == bot.ConversationHandler.END
    assert "pending" not in context.user_data
    assert fake_bot.find(buttons_id).reply_markup is None
    assert fake_bot.find(buttons_id).text == bot.DRAFT_TIMED_OUT
