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

import asyncio
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree

import pytest
from telegram import CallbackQuery, Chat, Message, Update, User, Voice
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

    async def edit_message_text(
        self, chat_id, message_id, text, parse_mode=None, reply_markup=None, **kwargs
    ):
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

    async def answer_callback_query(self, callback_query_id, text=None, **kwargs):
        self.answered.append(text)
        return True

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
            self._bot.chat_id,
            self.message.message_id,
            text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )

    async def edit_message_reply_markup(self, reply_markup=None, **kwargs):
        return await self._bot.edit_message_reply_markup(
            self._bot.chat_id,
            self.message.message_id,
            reply_markup=reply_markup,
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
        # bot_data is where the transcription keywords live, and args is what a
        # CommandHandler fills in from the message. Both exist on the real
        # CallbackContext; a double without them hides a missing attribute.
        self.bot_data = {}
        self.args = []


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
    return FakeUpdate(
        fake_bot, message=message, callback_query=FakeCallbackQuery(fake_bot, data, message)
    )


async def stub_voice_pipeline(
    monkeypatch, tmp_path, fake_bot, title, text, tags, download_fails=False
):
    """Points the voice pipeline at stubs and at a temp dir the test can inspect."""
    monkeypatch.setattr(bot.tempfile, "tempdir", str(tmp_path))
    fake_bot.files["voice-1"] = FakeFile(fail=download_fails)

    transcribed_with: list[list[str]] = []

    async def fake_transcribe(path, keywords=None):
        transcribed_with.append(list(keywords or []))
        return "raw transcription"

    async def fake_format(transcription):
        return title, text, tags

    monkeypatch.setattr(bot, "transcribe", fake_transcribe)
    monkeypatch.setattr(bot, "format_entry", fake_format)
    return transcribed_with


def preview_bodies(fake_bot):
    return [message.text for message in fake_bot.sent]


# --------------------------------------------------------------------------- #
# Defect 1 — the weekly report fired on Saturday
# --------------------------------------------------------------------------- #


def _trigger_fields(job):
    return {field.name: str(field) for field in job.job.trigger.fields}


def test_weekly_report_is_scheduled_for_sunday(tmp_path, monkeypatch):
    """`days=(6,)` means Saturday in python-telegram-bot >= 20; Sunday is 0."""
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))
    app = bot.build_application()

    jobs = {job.name: job for job in app.job_queue.jobs()}
    assert bot.WEEKLY_REPORT_JOB in jobs

    fields = _trigger_fields(jobs[bot.WEEKLY_REPORT_JOB])
    assert fields["day_of_week"] == "sun"
    assert fields["hour"] == "21"
    assert fields["minute"] == "0"


def test_daily_summary_runs_every_day(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))
    app = bot.build_application()

    jobs = {job.name: job for job in app.job_queue.jobs()}
    fields = _trigger_fields(jobs[bot.DAILY_SUMMARY_JOB])
    assert fields["day_of_week"] == "sun,mon,tue,wed,thu,fri,sat"
    assert fields["hour"] == "21"


# --------------------------------------------------------------------------- #
# Defect 5 — formatting characters in generated text broke the message
# --------------------------------------------------------------------------- #

AWKWARD_TITLE = "Отчёт за 1*2 недели & R&D <итоги>"
AWKWARD_TAGS = ["c++_dev", "prod`ready", "R&D", "<draft>"]
AWKWARD_REPORT = "Неделя была насыщенной.\n- 5*5 тренировок\n- R&D <итоги>"


@pytest.mark.asyncio
async def test_preview_survives_formatting_characters_in_title(
    tmp_path, monkeypatch, fake_bot, context
):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, AWKWARD_TITLE, "тело записи", [])

    state = await bot.handle_voice(voice_update(fake_bot), context)

    assert state == bot.PREVIEW
    assert context.user_data["pending"]["title"] == AWKWARD_TITLE
    title_body = fake_bot.find(context.user_data["title_msg_id"]).text
    assert title_body == "<b>Отчёт за 1*2 недели &amp; R&amp;D &lt;итоги&gt;</b>"


@pytest.mark.asyncio
async def test_tags_typed_by_the_user_do_not_break_the_message(fake_bot, context):
    context.user_data.update(
        {
            "pending": {"title": "Заголовок", "text": "тело", "tags": []},
            "title_msg_id": (
                await fake_bot.send_message(1, "<b>Заголовок</b>", ParseMode.HTML)
            ).message_id,
            "text_msg_id": (await fake_bot.send_message(1, "тело")).message_id,
            "tags_msg_id": (
                await fake_bot.send_message(1, bot._tags_line([]), ParseMode.HTML)
            ).message_id,
            "buttons_msg_id": (await fake_bot.send_message(1, "Actions:")).message_id,
            "edit_prompt_msg_id": (
                await fake_bot.send_message(1, "Send tags separated by commas:")
            ).message_id,
        }
    )

    state = await bot.receive_new_tags(text_update(fake_bot, ", ".join(AWKWARD_TAGS)), context)

    assert state == bot.PREVIEW
    tags_body = fake_bot.find(context.user_data["tags_msg_id"]).text
    assert tags_body == (
        "<code>Daily</code> <code>c++_dev</code> <code>prod`ready</code>"
        " <code>R&amp;D</code> <code>&lt;draft&gt;</code>"
    )


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
async def test_daily_summary_job_with_an_unpaired_asterisk_is_delivered(
    monkeypatch, fake_bot, context
):
    async def fake_summary():
        return "День прошёл 3*4 раза лучше & <ярче>"

    monkeypatch.setattr(bot, "generate_daily_summary", fake_summary)

    await bot.send_daily_summary(context)

    body = fake_bot.sent[-1].text
    assert body == "<b>Daily summary</b>\n\nДень прошёл 3*4 раза лучше &amp; &lt;ярче&gt;"


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
async def test_failing_transcription_is_distinguishable_and_cleans_up(
    tmp_path, monkeypatch, fake_bot, context
):
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
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))
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
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))
    conv = _conversation_handler(bot.build_application())
    update = _real_voice_update()

    key = (update.effective_chat.id, update.effective_user.id)
    conv._conversations[key] = getattr(bot, state_name)

    assert conv.check_update(update) is not None


def test_cancel_is_reachable_from_every_state(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))
    conv = _conversation_handler(bot.build_application())

    assert any(
        getattr(handler, "commands", None) == frozenset({"cancel"}) for handler in conv.fallbacks
    ), "/cancel must be a fallback so it works in every state"
    assert conv.conversation_timeout == bot.PREVIEW_TIMEOUT


@pytest.mark.asyncio
async def test_a_new_recording_takes_the_buttons_off_the_old_preview(
    tmp_path, monkeypatch, fake_bot, context
):
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
async def test_a_failed_new_recording_leaves_the_old_draft_alone(
    tmp_path, monkeypatch, fake_bot, context
):
    """Replacing a preview is only earned by a recording that makes it all the way.

    The second recording is too large for the Bot API, so the download raises. The
    first draft has not been saved yet and a transcription and a formatting call have
    already been paid for; it has to survive, buttons and all.
    """
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Первая", "тело", ["sport"])
    await bot.handle_voice(voice_update(fake_bot, message_id=1), context)
    first = dict(context.user_data)

    await stub_voice_pipeline(
        monkeypatch, tmp_path, fake_bot, "Вторая", "тело", [], download_fails=True
    )
    state = await bot.handle_voice(voice_update(fake_bot, message_id=2), context)

    assert state == bot.PREVIEW, "the conversation must stay on the surviving preview"
    assert fake_bot.sent[-1].text == bot.DOWNLOAD_FAILED
    assert context.user_data["pending"] == first["pending"]
    assert context.user_data["buttons_msg_id"] == first["buttons_msg_id"]

    buttons = fake_bot.find(first["buttons_msg_id"])
    assert buttons.reply_markup is not None, "the surviving draft keeps working buttons"
    assert buttons.text != bot.DRAFT_REPLACED


@pytest.mark.asyncio
async def test_a_failed_first_recording_still_ends_the_conversation(
    tmp_path, monkeypatch, fake_bot, context
):
    """With no draft to protect there is nothing to stay open for."""
    await stub_voice_pipeline(
        monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [], download_fails=True
    )

    state = await bot.handle_voice(voice_update(fake_bot), context)

    assert state == bot.ConversationHandler.END
    assert "pending" not in context.user_data


@pytest.mark.asyncio
async def test_a_failed_recording_sent_mid_edit_leaves_the_edit_open(
    tmp_path, monkeypatch, fake_bot, context
):
    """Opening an edit takes the preview's keyboard off, so PREVIEW is not a way back.

    A recording that fails while the bot is waiting for a new title has to leave the
    conversation in EDIT_TITLE, with the prompt still standing and still answerable.
    """
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Первая", "тело", [])
    await bot.handle_voice(voice_update(fake_bot, message_id=1), context)
    buttons_id = context.user_data["buttons_msg_id"]
    await bot.edit_title_callback(callback_update(fake_bot, "edit_title", buttons_id), context)
    prompt_id = context.user_data["edit_prompt_msg_id"]
    assert fake_bot.find(buttons_id).reply_markup is None

    await stub_voice_pipeline(
        monkeypatch, tmp_path, fake_bot, "Вторая", "тело", [], download_fails=True
    )
    state = await bot.handle_voice(voice_update(fake_bot, message_id=2), context)

    assert state == bot.EDIT_TITLE
    assert context.user_data["edit_prompt_msg_id"] == prompt_id
    assert prompt_id not in fake_bot.deleted

    # And the prompt still works: the draft the user came back to is the first one.
    await bot.receive_new_title(text_update(fake_bot, "Исправленный"), context)
    assert context.user_data["pending"]["title"] == "Исправленный"


@pytest.mark.asyncio
async def test_a_finished_edit_is_not_somewhere_a_failed_recording_returns_to(
    tmp_path, monkeypatch, fake_bot, context
):
    """Once the edit is answered the prompt is gone; PREVIEW is the way back again."""
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Первая", "тело", [])
    await bot.handle_voice(voice_update(fake_bot, message_id=1), context)
    await bot.edit_title_callback(
        callback_update(fake_bot, "edit_title", context.user_data["buttons_msg_id"]), context
    )
    await bot.receive_new_title(text_update(fake_bot, "Исправленный"), context)

    await stub_voice_pipeline(
        monkeypatch, tmp_path, fake_bot, "Вторая", "тело", [], download_fails=True
    )
    state = await bot.handle_voice(voice_update(fake_bot, message_id=2), context)

    assert state == bot.PREVIEW


@pytest.mark.asyncio
async def test_a_new_recording_clears_the_replaced_drafts_edit_prompt(
    tmp_path, monkeypatch, fake_bot, context
):
    """Dictating over an open "send a new title" must not leave the prompt behind."""
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Первая", "тело", [])
    await bot.handle_voice(voice_update(fake_bot, message_id=1), context)
    await bot.edit_title_callback(
        callback_update(fake_bot, "edit_title", context.user_data["buttons_msg_id"]), context
    )
    prompt_id = context.user_data["edit_prompt_msg_id"]

    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Вторая", "тело", [])
    await bot.handle_voice(voice_update(fake_bot, message_id=2), context)

    assert prompt_id in fake_bot.deleted
    assert context.user_data.get("edit_prompt_msg_id") is None


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
async def test_timeout_takes_the_buttons_off_the_stale_preview(
    tmp_path, monkeypatch, fake_bot, context
):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]

    state = await bot.handle_preview_timeout(text_update(fake_bot, "anything"), context)

    assert state == bot.ConversationHandler.END
    assert "pending" not in context.user_data
    assert fake_bot.find(buttons_id).reply_markup is None
    assert fake_bot.find(buttons_id).text == bot.DRAFT_TIMED_OUT


# --------------------------------------------------------------------------- #
# Defect 3 — a restart left live buttons wired to nothing
# --------------------------------------------------------------------------- #


@pytest.fixture
def no_network(monkeypatch):
    """Nothing in these tests may reach Notion or OpenAI; there are no credentials."""
    calls = []

    async def fake_save_entry(title, text, tags, day=None):
        calls.append((title, text, tags, day))
        return False

    monkeypatch.setattr(bot, "save_entry", fake_save_entry)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data", ["save", "toggle_highlight", "edit_title", "edit_text", "edit_tags", "cancel"]
)
async def test_a_callback_for_a_missing_draft_does_not_raise(fake_bot, context, data, no_network):
    """This is the state the bot comes back in after a restart: buttons, no user_data."""
    buttons = await fake_bot.send_message(1, "Actions:", reply_markup=bot._preview_keyboard())
    update = callback_update(fake_bot, data, buttons.message_id)

    handler = {
        "save": bot.save_callback,
        "toggle_highlight": bot.toggle_highlight_callback,
        "edit_title": bot.edit_title_callback,
        "edit_text": bot.edit_text_callback,
        "edit_tags": bot.edit_tags_callback,
        "cancel": bot.cancel_callback,
    }[data]

    state = await handler(update, context)

    assert state == bot.ConversationHandler.END
    assert no_network == []
    assert fake_bot.find(buttons.message_id).reply_markup is None
    assert fake_bot.find(buttons.message_id).text == bot.DRAFT_GONE
    assert fake_bot.answered == [bot.DRAFT_GONE]


@pytest.mark.asyncio
async def test_a_missing_draft_does_not_write_an_empty_entry(fake_bot, context, no_network):
    """The old code fell back to empty strings and appended a blank entry to Notion."""
    buttons = await fake_bot.send_message(1, "Actions:", reply_markup=bot._preview_keyboard())

    await bot.save_callback(callback_update(fake_bot, "save", buttons.message_id), context)

    assert no_network == []


@pytest.mark.asyncio
async def test_a_text_reply_without_a_draft_does_not_raise(fake_bot, context):
    state = await bot.receive_new_title(text_update(fake_bot, "новый заголовок"), context)

    assert state == bot.ConversationHandler.END
    assert fake_bot.sent[-1].text == bot.DRAFT_GONE


def test_drafts_are_persisted_across_a_restart(tmp_path, monkeypatch):
    state_file = tmp_path / bot.STATE_FILE_NAME
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))

    app = bot.build_application()

    assert app.persistence is not None
    assert str(app.persistence.filepath) == str(state_file)
    assert app.persistence.store_data.user_data is True

    conv = _conversation_handler(app)
    assert conv.name == bot.CONVERSATION_NAME
    assert conv.persistent is True


def test_the_state_file_goes_where_the_service_may_write(tmp_path, monkeypatch):
    """The deployed unit makes everything but StateDirectory read-only.

    A relative path resolves against the working directory, which is read-only there.
    The resulting PermissionError never reaches the user — python-telegram-bot reports
    persistence failures with no update attached — so the draft would silently never be
    written. systemd exports the writable directory as STATE_DIRECTORY; use it.
    """
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))

    assert bot.state_file_path() == str(tmp_path / bot.STATE_FILE_NAME)
    assert os.path.isabs(bot.state_file_path())
    assert str(bot.build_application().persistence.filepath) == str(tmp_path / bot.STATE_FILE_NAME)


def test_the_state_file_falls_back_to_the_working_directory(monkeypatch):
    """Outside systemd there is no STATE_DIRECTORY, and the old behaviour is right."""
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)

    assert bot.state_file_path() == os.path.join(".", bot.STATE_FILE_NAME)


def test_a_global_error_handler_is_registered(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))
    app = bot.build_application()

    assert bot.handle_error in app.error_handlers


def test_a_stale_callback_handler_catches_what_the_conversation_does_not(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path))
    app = bot.build_application()

    handlers = app.handlers[0]
    conv_index = next(i for i, h in enumerate(handlers) if isinstance(h, bot.ConversationHandler))
    stale = [
        h
        for h in handlers
        if isinstance(h, bot.CallbackQueryHandler) and h.callback is bot._draft_missing
    ]

    assert len(stale) == 1
    # Only one handler per group runs, so the conversation has to be tried first.
    assert handlers.index(stale[0]) > conv_index


class ErrorContext:
    """Stands in for the context the error handler is called with."""

    def __init__(self, fake_bot, error):
        self.bot = fake_bot
        self.error = error


LEAKY_ERROR = RuntimeError('Notion PATCH pages error 400: {"message": "secret-ish body"}')


def _real_message_update(fake_bot, message_id=9):
    user = User(id=1, first_name="Owner", is_bot=False)
    chat = Chat(id=fake_bot.chat_id, type=Chat.PRIVATE)
    message = Message(
        message_id=message_id,
        date=datetime.now(timezone.utc),
        chat=chat,
        from_user=user,
        text="/weekly",
    )
    message.set_bot(fake_bot)
    update = Update(update_id=1, message=message)
    update.set_bot(fake_bot)
    return update


@pytest.mark.asyncio
async def test_error_handler_logs_the_traceback(fake_bot, caplog):
    with caplog.at_level("ERROR"):
        await bot.handle_error("not an update", ErrorContext(fake_bot, LEAKY_ERROR))

    assert "RuntimeError" in caplog.text
    assert "secret-ish body" in caplog.text
    assert fake_bot.sent == []


@pytest.mark.asyncio
async def test_error_handler_tells_the_user_without_quoting_the_exception(fake_bot, caplog):
    with caplog.at_level("ERROR"):
        await bot.handle_error(_real_message_update(fake_bot), ErrorContext(fake_bot, LEAKY_ERROR))

    assert fake_bot.sent[-1].text == bot.SOMETHING_BROKE
    assert "secret-ish body" not in fake_bot.sent[-1].text
    assert "400" not in fake_bot.sent[-1].text
    assert "secret-ish body" in caplog.text


def _real_callback_update(fake_bot, message_id=9):
    user = User(id=1, first_name="Owner", is_bot=False)
    chat = Chat(id=fake_bot.chat_id, type=Chat.PRIVATE)
    message = Message(
        message_id=message_id,
        date=datetime.now(timezone.utc),
        chat=chat,
        from_user=user,
        text="Actions:",
    )
    message.set_bot(fake_bot)
    query = CallbackQuery(
        id="cb-1", from_user=user, chat_instance="chat-instance", data="save", message=message
    )
    query.set_bot(fake_bot)
    update = Update(update_id=2, callback_query=query)
    update.set_bot(fake_bot)
    return update


@pytest.mark.asyncio
async def test_error_handler_tells_the_user_about_a_failed_press_exactly_once(fake_bot, caplog):
    """`effective_message` for a callback update is the message the button sits on.

    Answering the query and then replying to that message is one failure reported
    twice — a toast and a chat message for the same press.
    """
    with caplog.at_level("ERROR"):
        await bot.handle_error(_real_callback_update(fake_bot), ErrorContext(fake_bot, LEAKY_ERROR))

    assert fake_bot.answered == [bot.SOMETHING_BROKE]
    assert fake_bot.sent == [], "the toast is the whole notification; do not also post in the chat"
    assert "secret-ish body" in caplog.text


# --------------------------------------------------------------------------- #
# Defect 4 — a second Save press
# --------------------------------------------------------------------------- #


async def _open_preview(monkeypatch, tmp_path, fake_bot, context, title="Заголовок"):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, title, "тело", ["sport"])
    await bot.handle_voice(voice_update(fake_bot), context)
    return context.user_data["buttons_msg_id"]


@pytest.mark.asyncio
async def test_two_save_presses_write_one_entry(
    tmp_path, monkeypatch, fake_bot, context, no_network
):
    buttons_id = await _open_preview(monkeypatch, tmp_path, fake_bot, context)

    first = await bot.save_callback(callback_update(fake_bot, "save", buttons_id), context)
    second = await bot.save_callback(callback_update(fake_bot, "save", buttons_id), context)

    assert len(no_network) == 1
    assert no_network[0] == ("Заголовок", "тело", ["sport"], bot.diary_today())
    assert first == bot.ConversationHandler.END
    assert second == bot.ConversationHandler.END
    assert fake_bot.answered[-1] == bot.ALREADY_SAVED
    assert fake_bot.find(buttons_id).text == "✓ Saved to Notion"
    assert fake_bot.find(buttons_id).reply_markup is None


@pytest.mark.asyncio
async def test_a_press_during_the_notion_round_trip_is_refused(
    tmp_path, monkeypatch, fake_bot, context
):
    """Two presses that genuinely overlap: the guard is read and set with no await."""
    buttons_id = await _open_preview(monkeypatch, tmp_path, fake_bot, context)

    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def slow_save_entry(title, text, tags, day=None):
        calls.append((title, text, tags, day))
        started.set()
        await release.wait()
        return True

    monkeypatch.setattr(bot, "save_entry", slow_save_entry)

    first = asyncio.create_task(
        bot.save_callback(callback_update(fake_bot, "save", buttons_id), context)
    )
    await started.wait()

    # Bounded: a second press that is not refused blocks on the first save instead of
    # returning, so without the guard this fails rather than hanging the suite.
    try:
        second = await asyncio.wait_for(
            bot.save_callback(callback_update(fake_bot, "save", buttons_id), context),
            timeout=2,
        )
    except asyncio.TimeoutError:  # pragma: no cover - only reached without the guard
        release.set()
        await first
        pytest.fail("a second Save press entered the save path instead of being refused")

    assert second == bot.PREVIEW
    assert fake_bot.answered[-1] == bot.SAVE_IN_FLIGHT

    release.set()
    assert await first == bot.ConversationHandler.END
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_the_keyboard_is_gone_before_the_save_starts(
    tmp_path, monkeypatch, fake_bot, context
):
    buttons_id = await _open_preview(monkeypatch, tmp_path, fake_bot, context)
    markup_during_save = []

    async def inspect_save_entry(title, text, tags, day=None):
        markup_during_save.append(fake_bot.find(buttons_id).reply_markup)
        return True

    monkeypatch.setattr(bot, "save_entry", inspect_save_entry)

    await bot.save_callback(callback_update(fake_bot, "save", buttons_id), context)

    assert markup_during_save == [None]


@pytest.mark.asyncio
async def test_a_failed_save_can_be_retried(tmp_path, monkeypatch, fake_bot, context):
    buttons_id = await _open_preview(monkeypatch, tmp_path, fake_bot, context)
    attempts = []

    async def flaky_save_entry(title, text, tags, day=None):
        attempts.append((title, text, tags))
        if len(attempts) == 1:
            raise RuntimeError("Notion PATCH pages error 502: upstream")
        return True

    monkeypatch.setattr(bot, "save_entry", flaky_save_entry)

    state = await bot.save_callback(callback_update(fake_bot, "save", buttons_id), context)

    assert state == bot.PREVIEW
    assert context.user_data["pending"]["title"] == "Заголовок"
    assert context.user_data.get("saving") is False
    failed = fake_bot.find(buttons_id)
    assert failed.text == bot.NOTION_FAILED
    assert failed.reply_markup is not None, "Save must still be pressable after a failure"
    assert "502" not in failed.text

    state = await bot.save_callback(callback_update(fake_bot, "save", buttons_id), context)

    assert state == bot.ConversationHandler.END
    assert len(attempts) == 2
    assert fake_bot.find(buttons_id).text == "✓ Added to today's page"


@pytest.mark.asyncio
async def test_only_the_last_few_saved_previews_are_remembered(
    tmp_path, monkeypatch, fake_bot, context, no_network
):
    for _ in range(bot.SAVED_BUTTONS_REMEMBERED + 3):
        buttons_id = await _open_preview(monkeypatch, tmp_path, fake_bot, context)
        await bot.save_callback(callback_update(fake_bot, "save", buttons_id), context)

    assert len(context.user_data[bot.SAVED_BUTTONS_KEY]) == bot.SAVED_BUTTONS_REMEMBERED
    assert len(no_network) == bot.SAVED_BUTTONS_REMEMBERED + 3


@pytest.mark.asyncio
async def test_cancel_after_an_edit_does_not_chase_a_deleted_prompt(
    tmp_path, monkeypatch, fake_bot, context
):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)

    await bot.edit_title_callback(
        callback_update(fake_bot, "edit_title", context.user_data["buttons_msg_id"]), context
    )
    prompt_id = context.user_data["edit_prompt_msg_id"]
    await bot.receive_new_title(text_update(fake_bot, "Новый заголовок"), context)

    assert prompt_id in fake_bot.deleted
    assert "edit_prompt_msg_id" not in context.user_data

    fake_bot.deleted.clear()
    await bot.handle_cancel(text_update(fake_bot, "/cancel"), context)

    assert prompt_id not in fake_bot.deleted


@pytest.mark.asyncio
async def test_the_preview_offers_a_cancel_button(tmp_path, monkeypatch, fake_bot, context):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)

    keyboard = fake_bot.find(context.user_data["buttons_msg_id"]).reply_markup
    rows = [[button.callback_data for button in row] for row in keyboard.inline_keyboard]

    assert ["cancel"] in rows, "there is a cancel button"
    assert ["save", "cancel"] not in rows, "and it does not share a row with Save"
    assert rows[-1] == ["cancel"], "it is the last row, furthest from the edit buttons"


@pytest.mark.asyncio
async def test_the_cancel_button_discards_the_draft_like_the_command(
    tmp_path, monkeypatch, fake_bot, context
):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", ["sport"])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]
    content_ids = [context.user_data[key] for key in ("title_msg_id", "text_msg_id", "tags_msg_id")]

    state = await bot.cancel_callback(callback_update(fake_bot, "cancel", buttons_id), context)

    assert state == bot.ConversationHandler.END
    assert "pending" not in context.user_data
    assert sorted(fake_bot.deleted) == sorted(content_ids)
    assert fake_bot.find(buttons_id).reply_markup is None
    assert fake_bot.find(buttons_id).text == bot.DRAFT_CANCELLED


@pytest.mark.asyncio
async def test_the_cancel_button_deletes_an_open_edit_prompt(
    tmp_path, monkeypatch, fake_bot, context
):
    """Cancelling from an editing state must not leave "send a new title" behind."""
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]
    await bot.edit_title_callback(callback_update(fake_bot, "edit_title", buttons_id), context)
    prompt_id = context.user_data["edit_prompt_msg_id"]

    await bot.cancel_callback(callback_update(fake_bot, "cancel", buttons_id), context)

    assert prompt_id in fake_bot.deleted


@pytest.mark.asyncio
async def test_cancelling_twice_does_not_claim_the_second_press_lost_a_draft(
    tmp_path, monkeypatch, fake_bot, context
):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]

    await bot.cancel_callback(callback_update(fake_bot, "cancel", buttons_id), context)
    await bot.cancel_callback(callback_update(fake_bot, "cancel", buttons_id), context)

    # The first press already said "discarded"; the second must not overwrite that
    # with something that reads like a different, worse outcome.
    assert fake_bot.find(buttons_id).text in (bot.DRAFT_CANCELLED, bot.DRAFT_GONE)
    assert "pending" not in context.user_data


async def _keywords(fake_bot, context, *args):
    context.args = list(args)
    await bot.handle_keywords(text_update(fake_bot, "/keywords"), context)
    return fake_bot.sent[-1].text


@pytest.mark.asyncio
async def test_keywords_starts_empty_and_says_how_to_begin(fake_bot, context):
    assert await _keywords(fake_bot, context) == bot.KEYWORDS_EMPTY


@pytest.mark.asyncio
async def test_keywords_add_stores_several_at_once(fake_bot, context):
    reply = await _keywords(fake_bot, context, "add", "Кэт,", "Спур,", "бабулечки")

    assert context.bot_data[bot.KEYWORDS_KEY] == ["Кэт", "Спур", "бабулечки"]
    assert "Кэт" in reply and "Спур" in reply and "бабулечки" in reply


@pytest.mark.asyncio
async def test_keywords_add_does_not_repeat_a_word_in_another_case(fake_bot, context):
    await _keywords(fake_bot, context, "add", "Кэт")
    await _keywords(fake_bot, context, "add", "кэт,", "Спур")

    assert context.bot_data[bot.KEYWORDS_KEY] == ["Кэт", "Спур"]


@pytest.mark.asyncio
async def test_keywords_remove_drops_one_regardless_of_case(fake_bot, context):
    await _keywords(fake_bot, context, "add", "Кэт,", "Спур")

    await _keywords(fake_bot, context, "remove", "спур")

    assert context.bot_data[bot.KEYWORDS_KEY] == ["Кэт"]


@pytest.mark.asyncio
async def test_keywords_clear_empties_the_list(fake_bot, context):
    await _keywords(fake_bot, context, "add", "Кэт,", "Спур")

    reply = await _keywords(fake_bot, context, "clear")

    assert context.bot_data[bot.KEYWORDS_KEY] == []
    assert reply == bot.KEYWORDS_EMPTY


@pytest.mark.asyncio
async def test_keywords_with_an_unknown_verb_explains_itself(fake_bot, context):
    assert await _keywords(fake_bot, context, "делай") == bot.KEYWORDS_USAGE


@pytest.mark.asyncio
async def test_keywords_added_in_the_chat_reach_the_transcriber(
    tmp_path, monkeypatch, fake_bot, context
):
    """The whole point: a word heard wrong is fixed from the chat, not over ssh."""
    transcribed_with = await stub_voice_pipeline(
        monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", []
    )
    await _keywords(fake_bot, context, "add", "Кэт,", "Спур")

    await bot.handle_voice(voice_update(fake_bot), context)

    assert transcribed_with[-1] == ["Кэт", "Спур"]


@pytest.mark.asyncio
async def test_the_environment_seeds_the_list_and_the_chat_adds_to_it(
    tmp_path, monkeypatch, fake_bot, context
):
    monkeypatch.setattr(bot.settings, "transcription_keywords", "Паттайя, Кэт")
    transcribed_with = await stub_voice_pipeline(
        monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", []
    )
    await _keywords(fake_bot, context, "add", "кэт,", "Спур")

    await bot.handle_voice(voice_update(fake_bot), context)

    assert transcribed_with[-1] == ["Паттайя", "Кэт", "Спур"], "seeded first, no repeat"


def _keyboard_rows(fake_bot, message_id):
    markup = fake_bot.find(message_id).reply_markup
    return [[button.callback_data for button in row] for row in markup.inline_keyboard]


def _labels(fake_bot, message_id):
    markup = fake_bot.find(message_id).reply_markup
    return [button.text for row in markup.inline_keyboard for button in row]


@pytest.mark.asyncio
async def test_a_new_draft_is_filed_under_the_diary_today(tmp_path, monkeypatch, fake_bot, context):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)

    assert context.user_data["pending"]["date"] == bot.diary_today().isoformat()
    assert "📅 Today" in _labels(fake_bot, context.user_data["buttons_msg_id"])


@pytest.mark.asyncio
async def test_the_date_button_opens_a_picker_of_the_last_week(
    tmp_path, monkeypatch, fake_bot, context
):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]

    await bot.date_open_callback(callback_update(fake_bot, "date_open", buttons_id), context)

    offered = [d for row in _keyboard_rows(fake_bot, buttons_id) for d in row]
    assert len(offered) == bot.DATE_CHOICES + 1, "a week of days plus Back"
    assert offered[-1] == "date_back"
    assert offered[0] == f"date:{bot.diary_today().isoformat()}"
    labels = _labels(fake_bot, buttons_id)
    assert "• Today" in labels, "the chosen day is marked"
    assert "Yesterday" in labels


@pytest.mark.asyncio
async def test_choosing_a_day_files_the_draft_under_it_and_closes_the_picker(
    tmp_path, monkeypatch, fake_bot, context
):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]
    yesterday = bot.diary_today() - timedelta(days=1)

    await bot.date_open_callback(callback_update(fake_bot, "date_open", buttons_id), context)
    await bot.date_chosen_callback(
        callback_update(fake_bot, f"date:{yesterday.isoformat()}", buttons_id), context
    )

    assert context.user_data["pending"]["date"] == yesterday.isoformat()
    rows = _keyboard_rows(fake_bot, buttons_id)
    assert ["save"] in rows, "the action buttons are back"
    assert "📅 Yesterday" in _labels(fake_bot, buttons_id)


@pytest.mark.asyncio
async def test_back_closes_the_picker_without_changing_the_date(
    tmp_path, monkeypatch, fake_bot, context
):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]
    before = context.user_data["pending"]["date"]

    await bot.date_open_callback(callback_update(fake_bot, "date_open", buttons_id), context)
    await bot.date_back_callback(callback_update(fake_bot, "date_back", buttons_id), context)

    assert context.user_data["pending"]["date"] == before
    assert ["save"] in _keyboard_rows(fake_bot, buttons_id)


@pytest.mark.asyncio
async def test_the_chosen_day_is_what_gets_saved(
    tmp_path, monkeypatch, fake_bot, context, no_network
):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]
    chosen = bot.diary_today() - timedelta(days=3)

    await bot.date_chosen_callback(
        callback_update(fake_bot, f"date:{chosen.isoformat()}", buttons_id), context
    )
    await bot.save_callback(callback_update(fake_bot, "save", buttons_id), context)

    assert no_network[0][3] == chosen


@pytest.mark.asyncio
async def test_the_date_survives_an_edit_of_the_title(
    tmp_path, monkeypatch, fake_bot, context, no_network
):
    """Every editing path rebuilds the keyboard; none of them may reset the day."""
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]
    chosen = bot.diary_today() - timedelta(days=2)
    await bot.date_chosen_callback(
        callback_update(fake_bot, f"date:{chosen.isoformat()}", buttons_id), context
    )

    await bot.edit_title_callback(callback_update(fake_bot, "edit_title", buttons_id), context)
    await bot.receive_new_title(text_update(fake_bot, "Новый заголовок"), context)

    assert context.user_data["pending"]["date"] == chosen.isoformat()
    assert "📅" in " ".join(_labels(fake_bot, buttons_id))


@pytest.mark.asyncio
async def test_a_malformed_date_callback_changes_nothing(tmp_path, monkeypatch, fake_bot, context):
    await stub_voice_pipeline(monkeypatch, tmp_path, fake_bot, "Заголовок", "тело", [])
    await bot.handle_voice(voice_update(fake_bot), context)
    buttons_id = context.user_data["buttons_msg_id"]
    before = context.user_data["pending"]["date"]

    await bot.date_chosen_callback(
        callback_update(fake_bot, "date:not-a-date", buttons_id), context
    )

    assert context.user_data["pending"]["date"] == before
    assert ["save"] in _keyboard_rows(fake_bot, buttons_id)
