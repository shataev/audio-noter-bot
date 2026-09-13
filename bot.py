import asyncio
import html
import logging
import os
import tempfile
import weakref
import zoneinfo
from dataclasses import replace
from datetime import date, time, timedelta

from telegram import (
    Bot,
    ForceReply,
    Message,
    ReplyParameters,
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    PicklePersistence,
    TypeHandler,
    filters,
)

from config import ANTHROPIC, settings
from services.coach import conversation as coach
from services.coach import diary as coach_diary
from services.coach import memory as coach_memory
from services.coach import profile as coach_profile
from services.coach import prompts as coach_prompts
from services.coach import rebuild as coach_rebuild
from services.coach import threads as coach_threads
from services.coach import weekly as coach_weekly
from services.coach.store import MemoryStore
from services.formatter import MIN_KEPT, format_entry, measure_kept
from services.memory_sync import MemorySync
from services.notion import day_label, diary_today, get_week_pages, save_entry
from services.summary import (
    all_pages,
    generate_daily_summary,
    generate_weekly_report,
    read_page_entries,
)
from services.whisper import merge_keywords, transcribe

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

# httpx logs every request at INFO, and the Telegram API puts the bot token in
# the path: "POST https://api.telegram.org/bot<TOKEN>/getUpdates". Polling runs
# every few seconds, so at INFO the token is written to the journal thousands of
# times a day, where anyone who can read the unit's log can lift it — including
# the deploy account, whose whole point is that it cannot reach the credentials.
# WARNING keeps the failures and drops the successful-request line that carries
# the secret.
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

PREVIEW, EDIT_TITLE, EDIT_TEXT, EDIT_TAGS = range(4)

# `JobQueue.run_daily` numbers weekdays 0-6 as Sunday-Saturday, not Monday-Sunday.
# The mapping was flipped in python-telegram-bot 20.0, and the old reading of it is
# why the weekly report used to arrive on Saturday. Never pass a bare number here.
SUNDAY = 0

DAILY_SUMMARY_JOB = "daily_summary"
WEEKLY_REPORT_JOB = "weekly_report"
COACH_WEEKLY_JOB = "coach_weekly_session"

# Everything in user_data that belongs to one draft. Cleared together, so a draft can
# never survive half-way and leave message ids pointing at a preview that is gone.
DRAFT_KEYS = (
    "pending",
    "title_msg_id",
    "text_msg_id",
    "tags_msg_id",
    # The line that says the draft is holding the raw transcription, when it is.
    # Part of the preview, so it goes when the preview goes.
    "unformatted_msg_id",
    "buttons_msg_id",
    "edit_prompt_msg_id",
    "editing_state",
    "saving",
    # The coach is not part of the draft, but its in-flight flag is cleared with
    # one: a call still in the air when the entry is saved must not leave the
    # next draft's buttons refusing to answer.
    "coach_in_flight",
    # The conversations opened about this draft, by thread id. They are the
    # draft's: an answer about an entry that has been discarded is about nothing,
    # so it is discarded with it. Kept as ids rather than message ids because the
    # thread is what knows every message it was delivered in.
    "coach_thread_ids",
)

# Message ids of previews that were saved, so a press arriving from a client whose view
# has not caught up is answered honestly instead of being called a lost draft. Only a
# handful are worth keeping.
SAVED_BUTTONS_KEY = "saved_buttons"
SAVED_BUTTONS_REMEMBERED = 10

# Long enough that no realistic dictate-and-edit session gets cut off, short enough
# that a preview abandoned in the morning is not still live when the 21:00 jobs run.
PREVIEW_TIMEOUT = timedelta(minutes=30)

# Drafts live in user_data, which is in-memory, and `make deploy` restarts the bot on
# every push. Without somewhere to put them a preview from a minute earlier comes back
# with working buttons and nothing behind them. The file holds a draft, not a secret,
# but it is local runtime state and is gitignored.
STATE_FILE_NAME = "bot_state.pickle"
CONVERSATION_NAME = "preview_flow"

# The coach's two files, beside the draft pickle and under the same directory —
# the only one the unit is allowed to write to. Both end in .state.json, which
# .gitignore already covers, because neither is anybody's business but the owner's.
COACH_MEMORY_FILE_NAME = "coach_memory.state.json"
COACH_THREADS_FILE_NAME = "coach_threads.state.json"
# One line: the week the coach last opened a conversation about. On disk rather
# than in memory because the bot restarts on every deploy, and "never twice for
# the same week" has to survive that.
COACH_WEEKLY_FILE_NAME = "coach_weekly.state.json"

WELCOME_TEXT = """👋 Welcome to Noter!

I turn your voice messages into structured diary entries in Notion.

Send me a voice message and I'll:
• Transcribe it
• Format it into a title and clean text
• Let you review and edit before saving
• Save it to your Notion journal

Every day at 21:00 I'll send you a summary of the day.

Type /help to see detailed instructions."""

HELP_TEXT = """<b>How to use Noter</b>

1. Send a voice message — speak naturally, I'll handle the rest
2. Review the preview:
   • <b>Title</b> — short heading generated by AI
   • <b>Text</b> — cleaned up transcription
   • <b>Tags</b> — auto-extracted + <code>Daily</code> (always added)
3. Edit anything using the buttons
4. Press <b>✓ Save</b> to send to Notion

<b>Editing</b>
<b>✎ Title</b> — send a new title
<b>✎ Text</b> — send a new text
<b>✎ Tags</b> — send tags separated by commas: <code>sport, health, work</code>
<b>📅 Date</b> — file the entry under an earlier day; the picker offers the last week
<b>/cancel</b> — throw the current draft away, same as the ✕ Cancel button

<b>Misheard words</b>
<b>/keywords add Кэт, Спур</b> — tell the transcriber about names it keeps getting wrong. <b>/keywords</b> on its own shows the list.

<b>A second opinion</b>
The buttons under <b>✓ Save</b> send the entry to a model that thinks about it and answers in a new message. The draft is never touched — it is an opinion, not an edit.
🔥 <b>Разъёб</b> — blunt; 🧭 <b>Разбор</b> — structural; 🫂 <b>Поддержка</b> — kind.
Reply to its message to keep talking, by text or by voice. A voice reply inside a conversation is never turned into a diary entry.
Tell it to change how it answers — "don't start with a greeting" — and it writes the rule down itself. <b>/rules</b> shows what it has written.
Once a week it writes first: one thing it noticed, or one question. Reply to it like any other message of its own.

<b>What it remembers</b>
<b>/memory</b> rebuilds what it knows about you from everything already in the diary. It asks what to look for, then asks again before it starts — it costs money and it takes a while.

<b>Daily summary</b>
Every day at 21:00 I send a summary of all entries recorded that day. If there are none, I'll send a friendly nudge instead."""

# Shown to the user when something breaks. They say which step failed and what to do
# next; the exception itself belongs in the log, where it is readable and where it
# cannot end up quoting a third-party response body back into the chat.
DOWNLOAD_FAILED = "I couldn't download that voice message. Send it again and I'll retry."
TRANSCRIBE_FAILED = "I couldn't transcribe that voice message. Send it again and I'll retry."
FORMAT_FAILED = "I transcribed it but couldn't turn it into an entry. Send the voice message again."
PREVIEW_FAILED = "I couldn't show the preview for that entry. Send the voice message again."
NOTION_FAILED = "I couldn't reach Notion, so nothing was saved. Press Save to try again."

# Not a failure: nothing broke and the draft is fine. It says which of the two
# texts is on screen, because they are worth different things — the words are
# the entry, the punctuation is a convenience — and because the author is the
# only one who can decide whether to tidy it up before saving.
UNFORMATTED_NOTICE = (
    "The cleanup came back short, so this is your own words, unformatted."
)

# How far back the date picker goes. A week covers "I forgot to write this up on
# Sunday"; anything older is rare enough to be worth editing in Notion directly,
# and a longer list stops fitting on a phone screen.
DATE_CHOICES = 7

KEYWORDS_KEY = "transcription_keywords"

KEYWORDS_USAGE = (
    "<b>Transcription keywords</b>\n\n"
    "Words the transcriber should lean towards — names, places, anything it hears "
    "wrong the same way every time. They are hints: a word appears in a transcript "
    "only if it is actually in the audio.\n\n"
    "<code>/keywords</code> — show the list\n"
    "<code>/keywords add Кэт, Спур</code> — add one or more, comma separated\n"
    "<code>/keywords remove Спур</code> — drop one or more\n"
    "<code>/keywords clear</code> — empty the list"
)
KEYWORDS_EMPTY = "No keywords yet. <code>/keywords add Кэт, Спур</code> to start one."

DRAFT_REPLACED = "✕ Draft discarded — a newer recording replaced it."
DRAFT_CANCELLED = "✕ Draft discarded."
DRAFT_TIMED_OUT = (
    "✕ Draft discarded — the preview went unanswered for "
    f"{int(PREVIEW_TIMEOUT.total_seconds() // 60)} minutes."
)
NOTHING_TO_CANCEL = "There is no draft open right now."
DRAFT_GONE = "That draft is no longer available. Send a new voice message and I'll start over."
SOMETHING_BROKE = "Something went wrong on my side. It is in the log — please try that again."
SAVE_IN_FLIGHT = "Still saving — one moment."
ALREADY_SAVED = "Already saved."
SAVING_NOTICE = "Saving to Notion..."

# The coach. Every one of these is one line: a model that has nothing useful to
# say must not cost the owner more than the seconds he already waited, and above
# all must not cost him the draft — which is why none of them mention it except
# to say it is still there.
COACH_THINKING = "{label} — думаю..."
COACH_IN_FLIGHT = "Still thinking — one moment."
COACH_FAILED = "The coach didn't answer. Your draft is exactly as it was."
COACH_EMPTY = "The coach came back with nothing to say."
# Said to a reply aimed at a conversation that has been pruned. Honest and short,
# in the coach's own voice: silently starting a new conversation with none of the
# context would answer confidently and wrongly, which is far worse.
COACH_FORGOTTEN = "этот разговор уже не помню"
RULES_TITLE = "<b>Rules the coach follows</b> ({count})"
RULES_EMPTY = (
    "No rules yet. Tell the coach how you want it to answer — "
    "<i>«не начинай с приветствия»</i> — and it writes the rule down itself."
)
RULES_UNAVAILABLE = "I couldn't read the rules. It is in the log."

# The note the bot posts when a saved entry taught it something, and the two
# buttons under it. Both are Russian, like the coach's own messages: what they
# act on is a sentence the model wrote in Russian about the owner.
#
# The id is on every button because it is on every line of the note, so three
# facts changing at once still leaves each button pointing at a line the owner
# can read. A press costs one tap; correcting a wrong fact anywhere else costs a
# trip to another app, which is the same as never.
FACT_WRONG_LABEL = "✗ неверно {id}"
FACT_FIX_LABEL = "✎ поправить {id}"
FACT_DROPPED = "Забыл."
FACT_GONE = "Этого факта уже нет."
FACT_FIX_PROMPT = "Пришли новый текст факта {id}:"
FACT_FIXED = "Поправил."
FACT_EMPTY = "Пустой текст — оставил как было."
FACT_UNAVAILABLE = "Не смог обновить память. Это в логе."

# How many facts one note offers buttons for. A pass that changes more than this
# is not a pass worth hand-correcting one line at a time, and Telegram stops
# drawing a keyboard long before it stops accepting one.
MAX_CORRECTION_ROWS = 10

# /memory: the retrospective rebuild, in the language the memory notes are in.
MEMORY_OFF = "Коуч выключен: у провайдера нет ключа."
MEMORY_FOCUS_PROMPT = (
    "🧠 Пересборка памяти по всему дневнику.\n\n"
    "На что смотреть в этом проходе? Что важно, что оставить, что выкинуть.\n"
    "Можно текстом или голосом. «-» — без фокуса."
)
MEMORY_NO_FOCUS = "без фокуса"
MEMORY_CONFIRM = (
    "Фокус: {focus}\n"
    "Сейчас в памяти фактов: {count}\n\n"
    "Пройду по всему дневнику, запись за записью. Это долго и стоит денег. Погнали?"
)
MEMORY_RUN_LABEL = "🧠 Пересобрать"
MEMORY_CANCEL_LABEL = "Отмена"
MEMORY_CANCELLED = "Отменил. Память не тронута."
MEMORY_IN_FLIGHT = "Пересборка уже идёт."
MEMORY_GONE = "Это подтверждение уже неактуально — набери /memory заново."
MEMORY_UNAVAILABLE = "Не смог прочитать память. Это в логе."
MEMORY_FAILED = "Пересборка сорвалась. Это в логе; память в том виде, до которого дошла."
MEMORY_STARTED = "🧠 Пересобираю память..."
MEMORY_PROGRESS = (
    "🧠 Пересобираю память... день {page}/{pages}, записей {done}, "
    "+{created} / ~{modified} / −{deleted}"
)
MEMORY_NO_ENTRIES = "В дневнике нечего читать — память не тронута."
MEMORY_DONE_HEADER = "🧠 Пересборка закончена."
MEMORY_ABORTED = "🧠 Остановился: {count} ошибок подряд."
MEMORY_DONE_COUNTS = (
    "Записей прочитано: {done}, из них с фактами: {learned}, пропущено: {skipped}.\n"
    "Фактов было {before}, стало {after}: +{created} / ~{modified} / −{deleted}."
)
MEMORY_UNDO = "Откатить: вернуть на место {path}"
MEMORY_NO_SNAPSHOT = "Откатывать не к чему: до этого прохода памяти не было."

# Telegram rejects a message body longer than this.
TELEGRAM_TEXT_LIMIT = 4096

TITLE_TEMPLATE = "<b>{title}</b>"
TAG_TEMPLATE = "<code>{tag}</code>"
DAILY_SUMMARY_TEMPLATE = "<b>Daily summary</b>\n\n{summary}"
WEEKLY_REPORT_TEMPLATE = "<b>Weekly highlights</b>\n\n{report}"


def render(template: str, **values: object) -> str:
    """Builds a message body for ParseMode.HTML, escaping every interpolated value.

    Titles and tags come from gpt-4o-mini or from whatever the user typed, and the
    summaries are written entirely by the model. Any of them can contain characters
    that mean something to Telegram's parser, and an unbalanced one rejects the whole
    message. Assembling message bodies here — and nowhere else — is what keeps that
    from being reintroduced the next time a message is added to this bot.
    """
    escaped = {key: html.escape(str(value), quote=False) for key, value in values.items()}
    return template.format(**escaped)


def _title_body(title: str) -> str:
    return render(TITLE_TEMPLATE, title=title)


async def reply_html(message: Message, body: str, **kwargs: object) -> Message:
    """Replies with a body produced by render(); the parse mode is not a call-site choice."""
    return await message.reply_text(body, parse_mode=ParseMode.HTML, **kwargs)


async def send_html(bot: Bot, chat_id: int, body: str, **kwargs: object) -> Message:
    return await bot.send_message(chat_id=chat_id, text=body, parse_mode=ParseMode.HTML, **kwargs)


async def edit_html(bot: Bot, chat_id: int, message_id: int, body: str, **kwargs: object) -> None:
    await bot.edit_message_text(
        chat_id=chat_id,
        message_id=message_id,
        text=body,
        parse_mode=ParseMode.HTML,
        **kwargs,
    )


def _tags_line(tags: list[str]) -> str:
    all_tags = ["Daily"] + [t for t in tags if t != "Daily"]
    return " ".join(render(TAG_TEMPLATE, tag=t) for t in all_tags)


def _clear_draft(context: ContextTypes.DEFAULT_TYPE) -> dict:
    """Forgets the current draft and returns what was stored, message ids included."""
    return {key: context.user_data.pop(key, None) for key in DRAFT_KEYS}


async def _retire_preview(bot: Bot, chat_id: int, draft: dict, notice: str) -> bool:
    """Ends a draft on screen: the coach's messages about it, then the buttons.

    Editing the text of the buttons message and passing no markup is what removes the
    keyboard, so the dead buttons disappear rather than sitting there doing nothing.

    This is the one place all three endings meet — the ✕ button and /cancel, a
    newer recording replacing the preview, and the preview timing out — which is
    why the coach is wound up here rather than in any of them. What each of them
    deletes above this call differs on purpose; what a draft ending *means* does
    not.
    """
    await _end_coach_conversations(bot, chat_id, draft)

    buttons_msg_id = draft.get("buttons_msg_id")
    if buttons_msg_id is None:
        return False
    try:
        await edit_html(bot, chat_id, buttons_msg_id, notice, reply_markup=None)
    except Exception:
        logger.warning("Could not retire preview message %s", buttons_msg_id, exc_info=True)
    return True


async def _delete_messages(bot: Bot, chat_id: int, message_ids) -> None:
    for message_id in message_ids:
        if message_id is None:
            continue
        try:
            await bot.delete_message(chat_id, message_id)
        except Exception:
            logger.warning("Could not delete message %s", message_id, exc_info=True)


def _remember_saved(context: ContextTypes.DEFAULT_TYPE, buttons_msg_id: int | None) -> None:
    if buttons_msg_id is None:
        return
    saved = context.user_data.setdefault(SAVED_BUTTONS_KEY, [])
    saved.append(buttons_msg_id)
    del saved[:-SAVED_BUTTONS_REMEMBERED]


async def _draft_missing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Deals with an action aimed at a draft that is no longer there.

    A preview that outlived its draft is the one failure the user cannot see: the
    button stays live and pressing it used to raise a KeyError into a log nobody reads.
    Saying so and taking the buttons off makes the dead button go away.
    """
    query = update.callback_query
    if query is None:
        await update.effective_message.reply_text(DRAFT_GONE)
        return ConversationHandler.END

    message_id = query.message.message_id if query.message is not None else None
    if message_id is not None and message_id in context.user_data.get(SAVED_BUTTONS_KEY, []):
        # This preview was saved and already says so. A late press must not claim the
        # draft was lost, and must not write anything a second time.
        await query.answer(ALREADY_SAVED)
        return ConversationHandler.END

    await query.answer(DRAFT_GONE)
    try:
        await query.edit_message_text(DRAFT_GONE, reply_markup=None)
    except Exception:
        logger.warning("Could not take the buttons off a stale preview", exc_info=True)
    return ConversationHandler.END


def _draft_day(context: ContextTypes.DEFAULT_TYPE) -> date:
    """The date the open draft will be filed under; today's when nothing is set."""
    stored = (context.user_data.get("pending") or {}).get("date")
    return date.fromisoformat(stored) if stored else diary_today()


def _date_button_label(day: date) -> str:
    """What the date button says. Relative where that is clearer than a date."""
    today = diary_today()
    if day == today:
        return "📅 Today"
    if day == today - timedelta(days=1):
        return "📅 Yesterday"
    return f"📅 {day.strftime('%a %d %b')}"


def _date_picker_keyboard(chosen: date) -> InlineKeyboardMarkup:
    """The last DATE_CHOICES diary days, newest first, three to a row."""
    today = diary_today()
    buttons = []
    for offset in range(DATE_CHOICES):
        day = today - timedelta(days=offset)
        label = _date_button_label(day).removeprefix("📅 ")
        buttons.append(
            InlineKeyboardButton(
                f"• {label}" if day == chosen else label,
                callback_data=f"date:{day.isoformat()}",
            )
        )
    rows = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
    rows.append([InlineKeyboardButton("← Back", callback_data="date_back")])
    return InlineKeyboardMarkup(rows)


def _preview_keyboard(highlighted: bool = False, day: date | None = None) -> InlineKeyboardMarkup:
    highlight_btn = (
        InlineKeyboardButton("⭐ Highlighted", callback_data="toggle_highlight")
        if highlighted else
        InlineKeyboardButton("Mark as Highlight ⭐", callback_data="toggle_highlight")
    )
    rows = [
        [
            InlineKeyboardButton("✎ Title", callback_data="edit_title"),
            InlineKeyboardButton("✎ Text", callback_data="edit_text"),
            InlineKeyboardButton("✎ Tags", callback_data="edit_tags"),
        ],
        [InlineKeyboardButton(_date_button_label(day or diary_today()), callback_data="date_open")],
        [highlight_btn],
        [InlineKeyboardButton("✓ Save", callback_data="save")],
    ]

    # The coach sits below Save and above Cancel. Below Save because saving is
    # what this keyboard is for and a second opinion is optional; above Cancel
    # because Cancel keeps the bottom row to itself. Drawn only when the active
    # provider has a key: a button whose every press can only fail is worse than
    # no button at all.
    if coach_enabled():
        rows.append([
            InlineKeyboardButton(mode.label, callback_data=f"coach:{mode.key}")
            for mode in coach_prompts.MODES
        ])

    # Its own row, under Save rather than beside it. Discarding is the one
    # irreversible thing in this keyboard — the transcription and the
    # formatting have already been paid for and the messages are deleted —
    # so it does not sit a thumb's width from the button pressed every time.
    rows.append([InlineKeyboardButton("✕ Cancel", callback_data="cancel")])
    return InlineKeyboardMarkup(rows)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(WELCOME_TEXT)


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await reply_html(update.effective_message, HELP_TEXT)


def _discard_temp_file(path: str | None) -> None:
    """Removes a downloaded audio file, if there is one.

    Never raises. This runs on the failure path too, and a temp file we cannot delete
    must not replace the error we are already reporting.
    """
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        logger.warning("Could not remove temporary audio file %s", path, exc_info=True)


async def _voice_to_text(context: ContextTypes.DEFAULT_TYPE, message: Message) -> str | None:
    """Turns a voice message into a transcription, or says what failed and returns None.

    Shared by the draft flow and by a voice reply inside a coach conversation. The four
    steps and the two failure messages are the same for both; what differs is entirely
    what happens next — one builds a preview, the other continues a conversation and must
    never build one — so this hands back a transcription or nothing and lets the caller
    decide what that means.
    """
    await message.reply_text("Listening...")

    # The download lives inside the guarded region: get_file and download_to_drive can
    # both fail — a network blip, or a voice note too large for the Bot API — and the
    # file has already been created by then.
    tmp_path = None
    try:
        try:
            voice_file = await context.bot.get_file(message.voice.file_id)
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp_path = tmp.name
            await voice_file.download_to_drive(tmp_path)
        except Exception:
            logger.exception("Error downloading voice message")
            await message.reply_text(DOWNLOAD_FAILED)
            return None

        try:
            await message.reply_text("Transcribing...")
            return await transcribe(tmp_path, _keywords_for_transcription(context))
        except Exception:
            logger.exception("Error transcribing voice message")
            await message.reply_text(TRANSCRIBE_FAILED)
            return None
    finally:
        _discard_temp_file(tmp_path)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    message = update.effective_message

    # A recording sent while a preview is still open replaces it — but only once the
    # replacement exists. Everything below can fail, and a draft the user has not saved
    # yet must not be thrown away for a recording that never arrives: the old preview
    # stays live and the failed recording is the one that is lost. Which means failing
    # has to leave the conversation exactly where it found it — including mid-edit,
    # where the preview's keyboard is off and the prompt the user can still answer is
    # the only way back.
    if context.user_data.get("pending") is None:
        on_failure = ConversationHandler.END
    else:
        on_failure = context.user_data.get("editing_state", PREVIEW)

    transcription = await _voice_to_text(context, message)
    if transcription is None:
        return on_failure

    logger.info("Transcription: %s", transcription)

    try:
        await message.reply_text("Formatting...")
        title, text, tags = await format_entry(transcription)
    except Exception:
        logger.exception("Error formatting transcription")
        await message.reply_text(FORMAT_FAILED)
        return on_failure

    # The formatter is forbidden to remove anything and does not reliably hold to
    # it on a long entry. When the reply comes back materially shorter it is not a
    # cleaned-up version of what was said, so it is not used: the draft keeps the
    # transcription, which has no punctuation but has every word. Losing them is
    # the worst thing this bot can do, and it would happen with nothing on screen
    # to show for it — there is no copy of the transcription anywhere else.
    kept = measure_kept(transcription, text)
    unformatted = kept.too_little
    if unformatted:
        # Lengths and the ratio, never the text. What a diagnosis needs is how
        # much came back, and the journal on the server is readable by the deploy
        # account, so a diary entry has no business in it.
        logger.warning(
            "The formatter returned %d of %d spoken characters (%.2f), below %.2f: "
            "keeping the raw transcription",
            kept.kept,
            kept.spoken,
            kept.ratio,
            MIN_KEPT,
        )
        text = transcription

    try:
        title_msg = await reply_html(message, _title_body(title))
        text_msg = await message.reply_text(text)
        tags_msg = await reply_html(message, _tags_line(tags))
        # Under the entry and above the buttons: the buttons stay the last message,
        # which is what the author's thumb and every callback in here expect.
        unformatted_msg = await message.reply_text(UNFORMATTED_NOTICE) if unformatted else None
        buttons_msg = await message.reply_text(
            "Actions:", reply_markup=_preview_keyboard(highlighted=False, day=diary_today())
        )
    except Exception:
        logger.exception("Error sending the preview")
        await message.reply_text(PREVIEW_FAILED)
        return on_failure

    # The new preview is up, so the old one can go. Its message ids live under the same
    # keys that are about to be overwritten, so its buttons have to come off here or
    # they would end up driving the new draft. The editing prompt goes too: a "send a
    # new title" that belonged to the replaced draft answers to nothing now.
    previous = _clear_draft(context)
    if previous.get("pending") is not None:
        chat_id = update.effective_chat.id
        await _retire_preview(context.bot, chat_id, previous, DRAFT_REPLACED)
        await _delete_messages(context.bot, chat_id, [previous["edit_prompt_msg_id"]])

    # The date is decided when the draft is made, not when it is saved: dictating
    # at 23:58 and pressing Save at 00:01 must not file the entry under tomorrow.
    context.user_data["pending"] = {
        "title": title, "text": text, "tags": tags, "date": diary_today().isoformat(),
    }
    context.user_data["title_msg_id"] = title_msg.message_id
    context.user_data["text_msg_id"] = text_msg.message_id
    context.user_data["tags_msg_id"] = tags_msg.message_id
    context.user_data["unformatted_msg_id"] = (
        unformatted_msg.message_id if unformatted_msg is not None else None
    )
    context.user_data["buttons_msg_id"] = buttons_msg.message_id

    return PREVIEW


async def save_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    pending = context.user_data.get("pending")
    if pending is None:
        return await _draft_missing(update, context)

    query = update.callback_query

    # Read and set with no await in between, so two presses cannot both get past here
    # whatever the application's update concurrency is set to.
    if context.user_data.get("saving"):
        await query.answer(SAVE_IN_FLIGHT)
        return PREVIEW
    context.user_data["saving"] = True

    await query.answer()

    # The keyboard comes off before the round trip, not after it returns. Saving is two
    # HTTP requests when today's page already exists, and Telegram leaves a button live
    # until the message is edited.
    try:
        await query.edit_message_text(SAVING_NOTICE, reply_markup=None)
    except Exception:
        logger.warning("Could not take the buttons off the preview before saving", exc_info=True)

    try:
        updated = await save_entry(
            pending["title"], pending["text"], pending["tags"], _draft_day(context)
        )
    except Exception:
        logger.exception("Error saving to Notion")
        # A genuine failure keeps the draft and puts the keyboard back, so Save can be
        # pressed again rather than the whole note having to be dictated again.
        context.user_data["saving"] = False
        try:
            await query.edit_message_text(
                NOTION_FAILED,
                reply_markup=_preview_keyboard(
                    highlighted=pending["title"].startswith("⭐ "), day=_draft_day(context)
                ),
            )
        except Exception:
            logger.warning("Could not restore the preview keyboard after a failed save", exc_info=True)
        return PREVIEW

    status = "Added to today's page" if updated else "Saved to Notion"
    buttons_msg_id = context.user_data.get("buttons_msg_id")
    day = _draft_day(context)
    _remember_saved(context, buttons_msg_id)
    _clear_draft(context)
    await query.edit_message_text(f"✓ {status}", reply_markup=None)

    # Last, and in the background. The entry is in Notion and the preview says
    # so: everything the owner asked for has happened by this line, and what
    # follows it can only add.
    _learn_later(
        context,
        chat_id=update.effective_chat.id,
        reply_to=buttons_msg_id,
        title=pending["title"],
        text=pending["text"],
        day=day,
    )
    return ConversationHandler.END


async def toggle_highlight_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    pending = context.user_data.get("pending")
    if pending is None:
        return await _draft_missing(update, context)

    query = update.callback_query
    await query.answer()

    highlighted = pending["title"].startswith("⭐ ")
    if highlighted:
        pending["title"] = pending["title"][len("⭐ "):]
    else:
        pending["title"] = f"⭐ {pending['title']}"

    highlighted = not highlighted
    chat_id = update.effective_chat.id
    await edit_html(
        context.bot,
        chat_id,
        context.user_data["title_msg_id"],
        _title_body(pending["title"]),
    )
    await context.bot.edit_message_reply_markup(
        chat_id=chat_id,
        message_id=context.user_data["buttons_msg_id"],
        reply_markup=_preview_keyboard(highlighted=highlighted, day=_draft_day(context)),
    )
    return PREVIEW


async def date_open_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Swaps the action buttons for the day picker, in the same message."""
    if context.user_data.get("pending") is None:
        return await _draft_missing(update, context)

    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(
        reply_markup=_date_picker_keyboard(_draft_day(context))
    )
    return PREVIEW


async def date_chosen_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Files the draft under the chosen day and puts the action buttons back."""
    if context.user_data.get("pending") is None:
        return await _draft_missing(update, context)

    query = update.callback_query
    try:
        day = date.fromisoformat(query.data.split(":", 1)[1])
    except ValueError:
        # Only reachable from a button this bot did not draw. Nothing is changed,
        # and the keyboard goes back rather than leaving the picker open.
        logger.warning("Ignoring a date callback that is not a date: %r", query.data)
        day = _draft_day(context)
    else:
        context.user_data["pending"]["date"] = day.isoformat()

    await query.answer()
    await query.edit_message_reply_markup(
        reply_markup=_preview_keyboard(
            highlighted=context.user_data["pending"]["title"].startswith("⭐ "), day=day
        )
    )
    return PREVIEW


async def date_back_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Closes the picker without changing the date."""
    if context.user_data.get("pending") is None:
        return await _draft_missing(update, context)

    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(
        reply_markup=_preview_keyboard(
            highlighted=context.user_data["pending"]["title"].startswith("⭐ "),
            day=_draft_day(context),
        )
    )
    return PREVIEW


async def edit_title_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if context.user_data.get("pending") is None:
        return await _draft_missing(update, context)

    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    prompt = await query.message.reply_text("Send a new title:")
    context.user_data["edit_prompt_msg_id"] = prompt.message_id
    context.user_data["editing_state"] = EDIT_TITLE
    return EDIT_TITLE


async def edit_text_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if context.user_data.get("pending") is None:
        return await _draft_missing(update, context)

    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    prompt = await query.message.reply_text("Send a new text:")
    context.user_data["edit_prompt_msg_id"] = prompt.message_id
    context.user_data["editing_state"] = EDIT_TEXT
    return EDIT_TEXT


async def receive_new_title(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if context.user_data.get("pending") is None:
        return await _draft_missing(update, context)

    user_msg = update.effective_message
    context.user_data["pending"]["title"] = user_msg.text.strip()

    chat_id = update.effective_chat.id
    await edit_html(
        context.bot,
        chat_id,
        context.user_data["title_msg_id"],
        _title_body(context.user_data["pending"]["title"]),
    )
    await context.bot.edit_message_reply_markup(
        chat_id=chat_id,
        message_id=context.user_data["buttons_msg_id"],
        reply_markup=_preview_keyboard(
            highlighted=context.user_data["pending"]["title"].startswith("⭐ "),
            day=_draft_day(context),
        ),
    )
    # Popped, so /cancel does not later try to delete a prompt that is already gone,
    # and so an interrupted recording is not sent back to an edit that is finished.
    context.user_data.pop("editing_state", None)
    await _delete_messages(context.bot, chat_id, [
        context.user_data.pop("edit_prompt_msg_id", None),
        user_msg.message_id,
    ])
    return PREVIEW


async def receive_new_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if context.user_data.get("pending") is None:
        return await _draft_missing(update, context)

    user_msg = update.effective_message
    context.user_data["pending"]["text"] = user_msg.text.strip()

    chat_id = update.effective_chat.id
    await context.bot.edit_message_text(
        chat_id=chat_id,
        message_id=context.user_data["text_msg_id"],
        text=context.user_data["pending"]["text"],
    )
    await context.bot.edit_message_reply_markup(
        chat_id=chat_id,
        message_id=context.user_data["buttons_msg_id"],
        reply_markup=_preview_keyboard(
            highlighted=context.user_data["pending"]["title"].startswith("⭐ "),
            day=_draft_day(context),
        ),
    )
    # Popped, so /cancel does not later try to delete a prompt that is already gone,
    # and so an interrupted recording is not sent back to an edit that is finished.
    context.user_data.pop("editing_state", None)
    await _delete_messages(context.bot, chat_id, [
        context.user_data.pop("edit_prompt_msg_id", None),
        user_msg.message_id,
    ])
    return PREVIEW


async def edit_tags_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if context.user_data.get("pending") is None:
        return await _draft_missing(update, context)

    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    prompt = await query.message.reply_text("Send tags separated by commas:")
    context.user_data["edit_prompt_msg_id"] = prompt.message_id
    context.user_data["editing_state"] = EDIT_TAGS
    return EDIT_TAGS


async def receive_new_tags(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if context.user_data.get("pending") is None:
        return await _draft_missing(update, context)

    user_msg = update.effective_message
    new_tags = [t.strip() for t in user_msg.text.split(",") if t.strip()]
    context.user_data["pending"]["tags"] = new_tags

    chat_id = update.effective_chat.id
    await edit_html(
        context.bot,
        chat_id,
        context.user_data["tags_msg_id"],
        _tags_line(new_tags),
    )
    await context.bot.edit_message_reply_markup(
        chat_id=chat_id,
        message_id=context.user_data["buttons_msg_id"],
        reply_markup=_preview_keyboard(
            highlighted=context.user_data["pending"]["title"].startswith("⭐ "),
            day=_draft_day(context),
        ),
    )
    # Popped, so /cancel does not later try to delete a prompt that is already gone,
    # and so an interrupted recording is not sent back to an edit that is finished.
    context.user_data.pop("editing_state", None)
    await _delete_messages(context.bot, chat_id, [
        context.user_data.pop("edit_prompt_msg_id", None),
        user_msg.message_id,
    ])
    return PREVIEW


async def _discard_draft(context: ContextTypes.DEFAULT_TYPE, chat_id: int, draft: dict) -> bool:
    """Takes a cleared draft off the screen. True if the preview said so itself."""
    await _delete_messages(context.bot, chat_id, [
        draft["title_msg_id"],
        draft["text_msg_id"],
        draft["tags_msg_id"],
        draft["unformatted_msg_id"],
        draft["edit_prompt_msg_id"],
    ])
    return await _retire_preview(context.bot, chat_id, draft, DRAFT_CANCELLED)


async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    draft = _clear_draft(context)
    if draft.get("pending") is None:
        await update.effective_message.reply_text(NOTHING_TO_CANCEL)
        return ConversationHandler.END

    chat_id = update.effective_chat.id
    if not await _discard_draft(context, chat_id, draft):
        await update.effective_message.reply_text(DRAFT_CANCELLED)
    return ConversationHandler.END


async def cancel_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """The ✕ Cancel button. Same outcome as /cancel, reached without typing.

    /cancel stays: it is the way out when the preview has scrolled away, and it
    is the conversation's fallback, which a callback cannot be.
    """
    if context.user_data.get("pending") is None:
        return await _draft_missing(update, context)

    await update.callback_query.answer()
    draft = _clear_draft(context)
    await _discard_draft(context, update.effective_chat.id, draft)
    return ConversationHandler.END


async def handle_preview_timeout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    draft = _clear_draft(context)
    if draft.get("pending") is None:
        return ConversationHandler.END

    logger.info("Preview timed out, discarding draft")
    await _retire_preview(context.bot, update.effective_chat.id, draft, DRAFT_TIMED_OUT)
    return ConversationHandler.END


def _stored_keywords(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
    """The words added from the chat. bot_data rides the persistence already wired up."""
    return context.bot_data.setdefault(KEYWORDS_KEY, [])


def _keywords_for_transcription(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
    """What the environment seeded plus what has been added since."""
    return merge_keywords(settings.keywords, _stored_keywords(context))


def _keywords_list(words: list[str]) -> str:
    if not words:
        return KEYWORDS_EMPTY
    listed = "\n".join(render("• {word}", word=w) for w in words)
    return render("<b>Transcription keywords</b> ({count})", count=str(len(words))) + "\n" + listed


async def handle_keywords(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manage the transcription hints without touching the server.

    Misheard words are found while using the bot, one at a time, so the place to
    fix them is the same chat — not an environment variable behind an ssh login.
    TRANSCRIPTION_KEYWORDS still seeds a fresh install; this list adds to it.
    """
    words = _stored_keywords(context)
    args = context.args or []

    if not args:
        await reply_html(update.effective_message, _keywords_list(_keywords_for_transcription(context)))
        return

    verb, rest = args[0].lower(), " ".join(args[1:])
    given = [w.strip() for w in rest.split(",") if w.strip()]

    if verb == "add" and given:
        merged = merge_keywords(words, given)
        context.bot_data[KEYWORDS_KEY] = merged
        added = len(merged) - len(words)
        await reply_html(
            update.effective_message,
            render("Added {n}.", n=str(added)) + "\n" + _keywords_list(_keywords_for_transcription(context)),
        )
    elif verb == "remove" and given:
        dropped = {w.casefold() for w in given}
        context.bot_data[KEYWORDS_KEY] = [w for w in words if w.casefold() not in dropped]
        await reply_html(
            update.effective_message, _keywords_list(_keywords_for_transcription(context))
        )
    elif verb == "clear":
        context.bot_data[KEYWORDS_KEY] = []
        await reply_html(
            update.effective_message, _keywords_list(_keywords_for_transcription(context))
        )
    else:
        await reply_html(update.effective_message, KEYWORDS_USAGE)


# --------------------------------------------------------------------------- #
# The coach: a second opinion on the draft, and a conversation on top of it.
#
# Everything below obeys one rule that outranks the whole feature: nothing here
# may cost the owner an unsaved entry. So no function in this section writes to
# `pending` or to any of the preview's message ids, and every failure path ends
# in one message saying so and nothing else.
# --------------------------------------------------------------------------- #


def coach_enabled() -> bool:
    """Whether the mode buttons are drawn at all: only when the provider has a key."""
    if settings.ai_provider == ANTHROPIC:
        return bool(settings.anthropic_api_key)
    return bool(settings.openai_api_key)


def coach_memory_path() -> str:
    return os.path.join(os.environ.get("STATE_DIRECTORY", "."), COACH_MEMORY_FILE_NAME)


def coach_threads_path() -> str:
    return os.path.join(os.environ.get("STATE_DIRECTORY", "."), COACH_THREADS_FILE_NAME)


def _memory_store() -> MemoryStore:
    return MemoryStore(coach_memory_path())


# One lock over every write to the memory file. Two things write to it — a coach
# answer that carried a rules block, and the profile pass that runs in the
# background after a save — and they can be in the air at the same time: the pass
# starts when Save is pressed and the owner is free to open a conversation while
# it runs. The file is written atomically, so neither can see half of the other's
# document; what the lock stops is the read-modify-write around it, where the
# second writer saves a document built on a copy from before the first one wrote
# and silently drops it.
#
# Made per loop rather than once at import, because an asyncio.Lock binds itself
# to the loop it first has to wait on and raises "bound to a different event
# loop" in any other. The bot has exactly one loop and never notices; a test
# suite has one per test and would fail on the second contended case. Weak keys
# so that a loop that is finished takes its lock with it.
_store_locks: "weakref.WeakKeyDictionary[object, asyncio.Lock]" = weakref.WeakKeyDictionary()


def _store_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _store_locks.get(loop)
    if lock is None:
        lock = _store_locks[loop] = asyncio.Lock()
    return lock


# The mirror onto the two Notion pages the owner reads and edits. Per loop and
# per path for the same two reasons the lock above is — it holds that lock, and a
# test that points STATE_DIRECTORY somewhere else wants its own — and kept rather
# than rebuilt because the page ids it found and the moment it last read them are
# the whole of the throttle.
_syncs: "weakref.WeakKeyDictionary[object, tuple[str, MemorySync]]" = weakref.WeakKeyDictionary()


def _memory_sync() -> MemorySync:
    loop = asyncio.get_running_loop()
    cached = _syncs.get(loop)
    path = coach_memory_path()
    if cached is None or cached[0] != path:
        cached = (path, MemorySync(_memory_store(), lock=_store_lock()))
        _syncs[loop] = cached
    return cached[1]


async def _mirror_memory() -> None:
    """Put the stored lists on their pages. Never the reason a caller fails.

    Every call site has already done the thing that mattered — the rule is
    written, the profile is saved, the entry is in Notion — and a mirror that did
    not run costs the owner a page that is a few minutes behind, which the next
    change or the next pull corrects.
    """
    try:
        await _memory_sync().push()
    except Exception:
        logger.warning("Could not mirror the coach's memory onto Notion", exc_info=True)


# The conversations, read from their file once and kept in step on every write.
# Keyed by the path they were read from, so that pointing STATE_DIRECTORY
# somewhere else — which is what a test does — reads that directory rather than
# handing back the previous one's threads.
_thread_cache: tuple[str, coach_threads.Threads] | None = None


def _load_threads() -> coach_threads.Threads:
    global _thread_cache
    path = coach_threads_path()
    if _thread_cache is None or _thread_cache[0] != path:
        _thread_cache = (path, coach_threads.ThreadStore(path).load())
    return _thread_cache[1]


def _save_threads(items: coach_threads.Threads) -> None:
    global _thread_cache
    path = coach_threads_path()
    coach_threads.ThreadStore(path).save(items)
    _thread_cache = (path, items)


async def _end_coach_conversations(bot: Bot, chat_id: int, draft: dict) -> None:
    """Takes the coach's answers about a draft away with the draft.

    Called from `_retire_preview`, which is where all three endings meet. An
    answer left behind after a cancel discusses an entry that no longer exists,
    and worse, the thread behind it still holds the keys of its messages: a reply
    would carry on a conversation about something that was never written.

    A conversation several replies deep still goes. It is the same thread, opened
    about this draft, and a rule that tried to tell "still about the entry" from
    "a conversation of its own" would have to guess.

    Nothing in here can stop the cancel. A message the author deleted himself, or
    one older than the Bot API will delete, is a warning in the log — the draft is
    discarded either way, exactly as it was before any of this existed.
    """
    thread_ids = draft.get("coach_thread_ids") or []
    if not thread_ids:
        return

    try:
        items = _load_threads()
    except Exception:
        logger.warning("Could not read the coach conversations to end them", exc_info=True)
        return

    message_ids: list[int] = []
    for thread_id in thread_ids:
        thread = coach_threads.find_by_id(items, thread_id)
        if thread is None:
            continue
        message_ids += [
            message_id
            for message_id in (
                coach_threads.message_of(address, chat_id) for address in thread.keys
            )
            if message_id is not None
        ]
        items = coach_threads.drop(items, thread_id)

    logger.info(
        "Ending %d coach conversation(s) with the draft: %d message(s)",
        len(thread_ids),
        len(message_ids),
    )
    await _delete_messages(bot, chat_id, message_ids)

    # Written even if a delete failed. The keys have moved to `forgotten`, so a
    # reply to a message still on screen is answered with "I no longer have that
    # conversation" rather than starting a blank one about a discarded entry.
    try:
        _save_threads(items)
    except Exception:
        logger.warning("Could not drop the coach conversations of a draft", exc_info=True)


def _knows_coach_message(chat_id: int, message_id: int) -> bool:
    """Whether that message is one of the coach's — live conversation or pruned one.

    This decides routing, so it is deliberately total: a file that cannot be read
    means "not a coach message", and the reply falls through to the flow it would
    have reached before this feature existed. A broken conversation file must not
    be able to swallow a voice message that was meant to become a diary entry.
    """
    try:
        items = _load_threads()
    except Exception:
        logger.warning("Could not read the coach conversations", exc_info=True)
        return False

    address = coach_threads.key(chat_id, message_id)
    return (
        coach_threads.find(items, address) is not None
        or coach_threads.was_forgotten(items, address)
    )


class CoachReplyFilter(filters.MessageFilter):
    """Matches a message replying to something the coach said, and nothing else.

    A filter rather than a check inside the handler, because a handler cannot
    decline an update once it has been given one. Replying to the preview, or to
    any other message, has to go on reaching the handler it always did.
    """

    def filter(self, message: Message) -> bool:
        reply = message.reply_to_message
        if reply is None or message.chat is None:
            return False
        return _knows_coach_message(message.chat.id, reply.message_id)


def _pack(pieces: list[str], joiner: str, limit: int) -> list[str]:
    """Greedily refills `pieces` into as few groups of at most `limit` as it can."""
    packed: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{joiner}{piece}" if current else piece
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            packed.append(current)
        current = piece
    if current:
        packed.append(current)
    return packed


def _chunks(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Splits a long answer into messages, preferring the seams a reader would pick.

    Paragraphs first, then lines, then — only for a single line longer than a whole
    message, which no coach answer should ever be — mid-word. Cutting mid-sentence is
    what the naive fixed-width split does to every answer that overflows, and it reads
    like a bug even when the text is all there.
    """
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    for block in _pack(text.split("\n\n"), "\n\n", limit):
        if len(block) <= limit:
            chunks.append(block)
            continue
        for line in _pack(block.split("\n"), "\n", limit):
            if len(line) <= limit:
                chunks.append(line)
            else:
                chunks.extend(line[at : at + limit] for at in range(0, len(line), limit))
    return chunks


async def _send_plain(
    bot: Bot,
    chat_id: int,
    text: str,
    reply_to: int | None = None,
    reply_markup: object | None = None,
) -> Message:
    """Sends a coach message: plain text, never a parse mode.

    The coach is told to write no markdown, but what reaches here is a model's
    output about the owner's own day and it is not worth one unbalanced asterisk
    to render it nicely. Plain text cannot fail to parse — which matters twice
    over for the memory note, whose every line is a sentence the model wrote.

    `allow_sending_without_reply` so that a reply target which has since been
    deleted costs the reply threading and not the answer.
    """
    reply_parameters = (
        ReplyParameters(message_id=reply_to, allow_sending_without_reply=True)
        if reply_to is not None
        else None
    )
    return await bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_parameters=reply_parameters,
        reply_markup=reply_markup,
    )


async def _edit_plain(bot: Bot, chat_id: int, message_id: int, text: str) -> None:
    await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text)


async def _run_coach(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    reply_to: int,
    mode: coach_prompts.Mode,
    new_turns: list[coach_threads.Turn],
    thread: coach_threads.Thread | None = None,
) -> str | None:
    """One turn of the coach, from the waiting message to the recorded conversation.

    Returns the id of the conversation this turn belongs to, so a caller that has a
    draft can note that the conversation is the draft's and end it with it. ``None``
    means there is nothing to note: the turn produced no conversation, or the file
    it would have been written to could not be written.

    The same function for the first press and for the fifth reply: the only
    difference is whether there is already a thread to carry the earlier turns and
    to extend afterwards. That is what makes a rule the model writes down in its
    very first answer work exactly like one written in a long conversation.

    A call at high effort takes many seconds, so the waiting message goes out
    first and then becomes the answer. It is also the one message every failure
    path below edits, so there is never more than one new message for a turn that
    produced nothing.
    """
    notice = await _send_plain(
        context.bot, chat_id, COACH_THINKING.format(label=mode.label), reply_to=reply_to
    )

    previous = list(thread.turns) if thread is not None else []
    try:
        # The rules as the owner last left them, which may be as he left them on
        # the Notion page rather than as the bot wrote them. Throttled, so the
        # fifth reply in a conversation does not ask Notion again.
        stored = await _memory_sync().pull()
        answer = await coach.answer(
            mode=mode,
            rules=stored.rules,
            turns=[*previous, *new_turns],
            model=settings.coach_model,
        )
    except Exception:
        logger.exception("The coach call failed")
        await _edit_plain(context.bot, chat_id, notice.message_id, COACH_FAILED)
        return

    # Written before the answer is delivered: a rule the owner has just asked for
    # is worth more than the ordering, and a delivery that fails half-way must not
    # also lose it. `changed` is None for a reply with no rules block at all,
    # which is almost every reply, and then nothing is written.
    if answer.changed is not None:
        # Re-read inside the lock rather than writing back the document loaded
        # before the call: the profile pass from an entry saved a minute ago may
        # have written facts into it while this answer was being generated, and
        # `stored` no longer has them. Only the rules are this turn's to replace.
        try:
            async with _store_lock():
                _memory_store().save(replace(_memory_store().load(), rules=answer.rules))
        except Exception:
            logger.exception("Could not write the coach's rules")
        else:
            # Outside the lock: the mirror takes the same one, and an asyncio
            # lock is not reentrant.
            await _mirror_memory()

    if not answer.text:
        logger.warning("The coach answered with no visible text")
        await _edit_plain(context.bot, chat_id, notice.message_id, COACH_EMPTY)
        return

    # Every message of the answer becomes a door back into this conversation, so
    # replying to any part of a long one continues it.
    sent = [notice.message_id]
    try:
        parts = _chunks(answer.text)
        await _edit_plain(context.bot, chat_id, notice.message_id, parts[0])
        for part in parts[1:]:
            follow = await _send_plain(context.bot, chat_id, part, reply_to=notice.message_id)
            sent.append(follow.message_id)
    except Exception:
        # Recorded anyway, below: the owner has some of the answer in front of him
        # and a reply to it has to reach the conversation it belongs to.
        logger.warning("Could not deliver the whole coach answer", exc_info=True)

    exchange = [*new_turns, coach_threads.Turn(role="assistant", text=answer.text)]
    keys = [coach_threads.key(chat_id, message_id) for message_id in sent]
    try:
        items = _load_threads()
        if thread is None:
            items, recorded = coach_threads.start(items, mode=mode.key, turns=exchange, keys=keys)
        else:
            items, recorded = coach_threads.extend(
                items, thread.id, turns=exchange, keys=keys
            )
        _save_threads(items)
    except Exception:
        # The answer is already on screen. Losing the thread costs the next reply
        # its context, and that is worth a log line, not a message.
        logger.exception("Could not record the coach conversation")
        return None
    return recorded.id if recorded is not None else None


async def coach_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """A mode button on the preview. Opens a conversation about the draft.

    Returns to PREVIEW whatever happens: the draft is untouched by design, and a
    coach call that failed must leave the keyboard exactly where the owner left it.
    """
    pending = context.user_data.get("pending")
    if pending is None:
        return await _draft_missing(update, context)

    query = update.callback_query
    mode = coach_prompts.mode_for(query.data.split(":", 1)[1])
    if mode is None:
        # Only reachable from a button this bot did not draw, or from one drawn by
        # a version that had a mode this one does not.
        logger.warning("Ignoring a coach callback for an unknown mode: %r", query.data)
        await query.answer()
        return PREVIEW

    # Read and set with no await in between, so two presses cannot both get past
    # here whatever the application's update concurrency is set to. A model call
    # takes many seconds and an impatient second press is the normal case, not
    # the exceptional one.
    if context.user_data.get("coach_in_flight"):
        await query.answer(COACH_IN_FLIGHT)
        return PREVIEW
    context.user_data["coach_in_flight"] = True

    await query.answer()
    try:
        opened = await _run_coach(
            context,
            chat_id=update.effective_chat.id,
            reply_to=query.message.message_id,
            mode=mode,
            new_turns=[
                coach_threads.Turn(
                    role="user",
                    text=coach_prompts.entry_message(pending["title"], pending["text"]),
                )
            ],
        )
    finally:
        context.user_data["coach_in_flight"] = False

    # The conversation belongs to this draft, and goes when the draft does. Only
    # while the draft is still here: a call that was in the air when the author
    # cancelled has nothing left to belong to, and must not be handed to whatever
    # draft comes next. A reply into this conversation extends the same thread, so
    # the id noted here covers the whole of it however long it runs.
    if opened is not None and context.user_data.get("pending") is not None:
        context.user_data.setdefault("coach_thread_ids", []).append(opened)
    return PREVIEW


async def coach_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A reply to something the coach said, by text or by voice.

    This handler is registered ahead of the conversation so that it, and not the
    voice entry point, is the one handler in its group that takes the update.
    That is the whole mechanism behind "a voice reply must not become a diary
    entry": there is no path from here to a draft, rather than a draft that is
    built and then cleaned up.

    Returns None, not a conversation state: a coach reply is not part of the
    preview flow and must not move it. An open edit prompt stays open.
    """
    message = update.effective_message
    chat_id = update.effective_chat.id
    address = coach_threads.key(chat_id, message.reply_to_message.message_id)

    thread = coach_threads.find(_load_threads(), address)
    if thread is None:
        # The filter let it through, so this was the coach's message and its
        # conversation has been pruned. Say so rather than answer without it.
        await message.reply_text(COACH_FORGOTTEN)
        return

    mode = coach_prompts.mode_for(thread.mode) or coach_prompts.MODES[0]

    if context.user_data.get("coach_in_flight"):
        await message.reply_text(COACH_IN_FLIGHT)
        return
    context.user_data["coach_in_flight"] = True

    try:
        said = message.text
        if said is None:
            said = await _voice_to_text(context, message)
            if said is None:
                # Already reported by the transcription step. Nothing further
                # happens — and, in particular, no draft is created.
                return
        await _run_coach(
            context,
            chat_id=chat_id,
            reply_to=message.message_id,
            mode=mode,
            new_turns=[coach_threads.Turn(role="user", text=said)],
            thread=thread,
        )
    finally:
        context.user_data["coach_in_flight"] = False


async def handle_rules(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The standing instructions the coach has written down, with the ids it uses.

    The same numbering the model sees, so that "забудь правило 2" addresses the
    rule the owner is looking at.
    """
    try:
        # Forced rather than throttled: this command is the owner asking what the
        # bot believes right now, and he may have edited the page a moment ago.
        # A read that fails falls back to the file inside `pull`.
        rules = (await _memory_sync().pull(force=True)).rules.facts
    except Exception:
        logger.exception("Could not read the coach's rules")
        await update.effective_message.reply_text(RULES_UNAVAILABLE)
        return

    if not rules:
        await reply_html(update.effective_message, RULES_EMPTY)
        return

    lines = [render(RULES_TITLE, count=str(len(rules)))]
    lines += [render("{line}", line=coach_prompts.rule_line(fact)) for fact in rules]
    await reply_html(update.effective_message, "\n".join(lines))


# --------------------------------------------------------------------------- #
# The profile: what a saved entry taught the bot, and the two buttons that fix it.
#
# This runs after Save, never before it and never in its way. The entry is
# already in Notion by the time any of this starts, so the worst a failure here
# can cost is one entry's contribution to the profile — and the next entry offers
# the same facts again. That is the whole reason it is allowed to run unattended.
# --------------------------------------------------------------------------- #

# Which fact a "✎ поправить" prompt is waiting for an answer about, keyed by the
# chat and the prompt's own message id. In memory rather than in user_data
# because the filter that routes the reply runs before any handler and cannot see
# user_data; a restart in the seconds between the prompt and the answer costs the
# owner one more tap on the button, which is the cheapest failure in this file.
_fact_edit_prompts: dict[tuple[int, int], str] = {}
FACT_EDITS_REMEMBERED = 20


def _learn_later(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    reply_to: int | None,
    title: str,
    text: str,
    day: date,
) -> None:
    """Start the profile pass for an entry that has just been saved.

    Deliberately not awaited. A model call takes seconds and the entry is already
    in Notion: making Save wait for it would turn a working save into a bot that
    looks stuck, and a failing extraction into a save that looks broken.
    `Application.create_task` rather than a bare task so that a failure reaches
    the error handler and a pass in flight is awaited at shutdown instead of
    being dropped.
    """
    if not coach_enabled():
        return
    work = _learn_from_entry(
        context, chat_id=chat_id, reply_to=reply_to, title=title, text=text, day=day
    )
    try:
        context.application.create_task(work)
    except Exception:
        # The last line of the save path, and it must not be the one that breaks
        # it: the entry is already in Notion and the preview already says so, so
        # a pass that could not even be started is worth a log line and nothing
        # more. Closed explicitly, or it is reported against whatever runs next.
        work.close()
        logger.exception("Could not start the profile pass")


async def _learn_from_entry(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    reply_to: int | None,
    title: str,
    text: str,
    day: date,
) -> None:
    """Fold one saved entry into the profile, and say what that changed.

    The read before the call is not locked — a load is one atomic read of a file
    that is only ever replaced whole — but the commit is, and it checks that the
    profile is still the one the model was looking at. If it has moved, this pass
    is dropped rather than written: the owner correcting a fact by hand while the
    call was in the air is a correction that will not come back, and the facts
    this pass found will be offered again by the next entry.
    """
    try:
        # Forced rather than throttled. This pass is about to rewrite the list and
        # mirror it back, so reading a copy from up to a window ago would let a
        # background job put his own wording back on the page — and unlike a
        # conversation's follow-up turns, saves are minutes apart, so there is no
        # burst here for the window to collapse.
        profile = (await _memory_sync().pull(force=True)).profile
    except Exception:
        logger.exception("Could not read the profile; this entry teaches nothing")
        return

    learned = await coach_profile.learn(
        profile=profile,
        title=title,
        text=text,
        model=settings.profile_model,
        # Where the bot learned it. One page per day, so the day and the entry's
        # own title are what identifies the entry it came from.
        source=f"{day.isoformat()} · {title}",
    )
    if learned.changed is None:
        # The common case, and it is silent by design: an entry that taught
        # nothing must not cost the owner a message.
        return

    try:
        async with _store_lock():
            stored = _memory_store().load()
            if stored.profile != profile:
                logger.info("The profile moved while an entry was being read; dropping the pass")
                return
            _memory_store().save(replace(stored, profile=learned.profile))
    except Exception:
        logger.exception("Could not write the profile; this entry teaches nothing")
        return

    await _mirror_memory()
    await _post_memory_note(
        context,
        chat_id=chat_id,
        reply_to=reply_to,
        before=profile.facts,
        changed=learned.changed,
    )


def _corrections_keyboard(changed: coach_memory.ApplyResult) -> InlineKeyboardMarkup | None:
    """One row per fact the note offers to correct, in the order the note lists them.

    Only facts that are still there: a fact the pass deleted has nothing left to
    be wrong about.
    """
    rows = [
        [
            InlineKeyboardButton(
                FACT_WRONG_LABEL.format(id=fact.id), callback_data=f"fact:drop:{fact.id}"
            ),
            InlineKeyboardButton(
                FACT_FIX_LABEL.format(id=fact.id), callback_data=f"fact:fix:{fact.id}"
            ),
        ]
        for fact in (*changed.created, *changed.modified)[:MAX_CORRECTION_ROWS]
    ]
    return InlineKeyboardMarkup(rows) if rows else None


async def _post_memory_note(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    reply_to: int | None,
    before: tuple[coach_memory.Fact, ...],
    changed: coach_memory.ApplyResult,
) -> None:
    """Say what was learned, under the entry it was learned from."""
    body = coach_profile.note(about=coach_profile.change_lines(before, changed))
    if body is None:  # pragma: no cover - `changed` is never empty by the time we are here
        return

    keyboard = _corrections_keyboard(changed)
    try:
        parts = _chunks(body)
        for part in parts[:-1]:
            await _send_plain(context.bot, chat_id, part, reply_to=reply_to)
        # The buttons ride the last message, so they are under the whole list
        # rather than in the middle of it.
        await _send_plain(context.bot, chat_id, parts[-1], reply_to=reply_to, reply_markup=keyboard)
    except Exception:
        # The profile is already written, which is the part worth keeping. A note
        # that did not arrive costs the owner the chance to correct it now, not
        # the fact itself.
        logger.warning("Could not post the memory note", exc_info=True)


async def _apply_profile_op(op: dict) -> coach_memory.ApplyResult | None:
    """Apply one hand-made operation to the profile. None if the file was unreachable.

    Under the same lock as everything else that writes this file, and reading
    inside it: the owner pressing a button while a pass from the entry he saved
    a moment ago is still running is the ordinary case, not the exotic one.

    The pull comes first and outside the lock, because the id in the button is an
    id the owner is looking at in the chat, and the page may meanwhile have given
    that fact different words.
    """
    try:
        # Forced, for the same reason the pass above is: a press is about to
        # change the list, and a press is a human action, not a burst.
        await _memory_sync().pull(force=True)
        async with _store_lock():
            stored = _memory_store().load()
            applied = coach_memory.apply_ops(
                stored.profile, [op], kinds=coach_memory.PROFILE_KINDS
            )
            if not (applied.created or applied.modified or applied.deleted):
                # An id that is already gone. Nothing to write, and not an error:
                # see `fact_callback` for why this is the second press.
                return applied
            _memory_store().save(replace(stored, profile=applied.items))
    except Exception:
        logger.exception("Could not change the profile by hand")
        return None

    await _mirror_memory()
    return applied


def _without_fact(markup: InlineKeyboardMarkup | None, fact_id: str) -> InlineKeyboardMarkup | None:
    """The same keyboard with one fact's row taken out."""
    if markup is None:
        return None
    rows = [
        row
        for row in markup.inline_keyboard
        if not any((button.callback_data or "").rsplit(":", 1)[-1] == fact_id for button in row)
    ]
    return InlineKeyboardMarkup(rows) if rows else None


async def fact_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The two buttons under a memory note.

    Both are written to be harmless when pressed twice, or pressed on a fact that
    is already gone — from a second device, or from a note further up the chat.
    The id is addressed rather than a position, so the worst outcome of a stale
    press is that it matches nothing, which `apply_ops` reports as a skip and
    this answers with one line.
    """
    query = update.callback_query
    _, action, fact_id = query.data.split(":", 2)

    if action == "fix":
        prompt = await _send_plain(
            context.bot,
            update.effective_chat.id,
            FACT_FIX_PROMPT.format(id=fact_id),
            reply_to=query.message.message_id,
            reply_markup=ForceReply(selective=True),
        )
        _fact_edit_prompts[(update.effective_chat.id, prompt.message_id)] = fact_id
        for stale in list(_fact_edit_prompts)[:-FACT_EDITS_REMEMBERED]:
            del _fact_edit_prompts[stale]
        await query.answer()
        return

    applied = await _apply_profile_op(
        {"action": "delete", "id": fact_id, "reason": "marked wrong in the chat"}
    )
    if applied is None:
        await query.answer(FACT_UNAVAILABLE)
        return

    await query.answer(FACT_DROPPED if applied.deleted else FACT_GONE)
    # The row goes either way: the fact is not there any more, whichever press
    # removed it, and a button that can only say so again is noise.
    try:
        await query.edit_message_reply_markup(
            reply_markup=_without_fact(query.message.reply_markup, fact_id)
        )
    except Exception:
        logger.warning("Could not take a corrected fact off the note", exc_info=True)


class FactEditReplyFilter(filters.MessageFilter):
    """Matches the answer to a "✎ поправить" prompt, and nothing else.

    A filter rather than a check inside a handler, for the same reason the coach
    has one: a handler cannot decline an update once it has been given one, and
    everything this does not match has to go on reaching the handler it always
    did — including a new title typed into an open draft.
    """

    def filter(self, message: Message) -> bool:
        reply = message.reply_to_message
        if reply is None or message.chat is None:
            return False
        return (message.chat.id, reply.message_id) in _fact_edit_prompts


async def fact_edit_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The new wording for a fact, applied as a modify against its id."""
    message = update.effective_message
    chat_id = update.effective_chat.id
    fact_id = _fact_edit_prompts.pop((chat_id, message.reply_to_message.message_id), None)
    if fact_id is None:  # pragma: no cover - the filter has just said otherwise
        return

    text = (message.text or "").strip()
    if not text:
        await message.reply_text(FACT_EMPTY)
        return

    applied = await _apply_profile_op({"action": "modify", "id": fact_id, "text": text})
    if applied is None:
        await message.reply_text(FACT_UNAVAILABLE)
        return
    await message.reply_text(FACT_FIXED if applied.modified else FACT_GONE)


async def handle_weekly(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text("Generating weekly report...")
    try:
        report = await generate_weekly_report()
        if report:
            await reply_html(
                update.effective_message,
                render(WEEKLY_REPORT_TEMPLATE, report=report),
            )
        else:
            await update.effective_message.reply_text("No entries this week.")
    except Exception:
        logger.exception("Error generating weekly report")
        await update.effective_message.reply_text("Error generating weekly report.")


async def send_weekly_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Generating weekly report...")
    try:
        report = await generate_weekly_report()
        if report:
            await send_html(
                context.bot,
                settings.allowed_user_id,
                render(WEEKLY_REPORT_TEMPLATE, report=report),
            )
        else:
            await context.bot.send_message(
                chat_id=settings.allowed_user_id,
                text="No entries this week — next week is a fresh start!",
            )
    except Exception:
        logger.exception("Error generating weekly report")


async def send_daily_summary(context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Generating daily summary...")
    try:
        summary = await generate_daily_summary()
        if summary:
            await send_html(
                context.bot,
                settings.allowed_user_id,
                render(DAILY_SUMMARY_TEMPLATE, summary=summary),
            )
        else:
            await context.bot.send_message(
                chat_id=settings.allowed_user_id,
                text="Hey, how was your day? I'm sure you have something to be proud of!",
            )
    except Exception:
        logger.exception("Error generating daily summary")


# --------------------------------------------------------------------------- #
# /memory — rebuilding the profile from the diary that came before it.
#
# The profile only knows what it was told by entries saved after it shipped.
# This walks everything already written and folds it in, in two steps the owner
# controls: what the pass should be looking for, and an explicit go-ahead. It
# costs real money per entry and rewrites the file the whole coach reads, so it
# never starts from one tap.
#
# The pass itself is in services/coach/rebuild.py and knows nothing about
# Telegram or Notion. Everything here is the adapter: the two prompts, the walk
# over the diary, the progress bar, and the single-flight guard.
# --------------------------------------------------------------------------- #

# The prompt this bot is waiting on an answer to, keyed by the chat and the
# prompt's own message id — the same shape, and for the same reason, as
# `_fact_edit_prompts`: the filter that routes the answer runs before any handler
# and cannot see user_data. The value is the focus once it has been given.
_memory_focus_prompts: dict[tuple[int, int], None] = {}
_memory_confirmations: dict[tuple[int, int], str | None] = {}
MEMORY_PROMPTS_REMEMBERED = 5

# One pass at a time, for the whole process rather than per chat: it is one
# owner, one profile and one file. A second /memory while one is running is
# refused rather than queued — a queued rebuild is a rebuild that starts an hour
# later, against a profile the first one has already rewritten, which is not what
# anybody pressing the button a second time is asking for.
_rebuild_in_flight = False


def coach_weekly_path() -> str:
    return os.path.join(os.environ.get("STATE_DIRECTORY", "."), COACH_WEEKLY_FILE_NAME)


def _remember_prompt(prompts: dict, address: tuple[int, int], value: str | None) -> None:
    """Keep the newest few prompts and forget the rest.

    An unanswered prompt is abandoned, not cancelled, and without a bound the
    dictionary is a slow leak over the life of a process that is restarted only
    by a deploy.
    """
    prompts[address] = value
    for stale in list(prompts)[:-MEMORY_PROMPTS_REMEMBERED]:
        prompts.pop(stale, None)


def _focus_label(focus: str | None) -> str:
    return focus if focus else MEMORY_NO_FOCUS


async def handle_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Step one of the rebuild: ask what this pass should be looking for."""
    message = update.effective_message
    if not coach_enabled():
        await message.reply_text(MEMORY_OFF)
        return
    if _rebuild_in_flight:
        await message.reply_text(MEMORY_IN_FLIGHT)
        return

    prompt = await _send_plain(
        context.bot,
        update.effective_chat.id,
        MEMORY_FOCUS_PROMPT,
        reply_to=message.message_id,
        reply_markup=ForceReply(selective=True),
    )
    _remember_prompt(_memory_focus_prompts, (update.effective_chat.id, prompt.message_id), None)


class MemoryFocusFilter(filters.MessageFilter):
    """Matches the answer to the focus prompt, and nothing else.

    Ahead of the conversation, so that answering it by voice reaches this and not
    the entry point that would turn it into a draft. That is the whole reason the
    focus is asked for with a ForceReply: the answer is a reply to one known
    message, rather than "the next thing said in this chat", which would be a far
    wider net over a chat whose ordinary traffic is diary entries.
    """

    def filter(self, message: Message) -> bool:
        reply = message.reply_to_message
        if reply is None or message.chat is None:
            return False
        return (message.chat.id, reply.message_id) in _memory_focus_prompts


async def memory_focus_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Step two: echo the focus and the fact count, and wait to be told to go.

    What the owner says here is read as focus and as nothing else. It is never
    saved as a diary entry — there is no path from this handler to a draft — and
    it is never stored as a fact: it reaches the model as part of the extraction
    prompt for each entry and is gone when the pass ends.
    """
    message = update.effective_message
    chat_id = update.effective_chat.id
    address = (chat_id, message.reply_to_message.message_id)
    if address not in _memory_focus_prompts:  # pragma: no cover - the filter says otherwise
        return
    _memory_focus_prompts.pop(address, None)

    said = message.text
    if said is None:
        said = await _voice_to_text(context, message)
        if said is None:
            # Already reported by the transcription step, and deliberately the
            # end of it: no draft, and no half-built rebuild left waiting.
            return

    focus = said.strip()
    if focus in ("", "-", "—", "–"):
        focus = None

    try:
        count = len(_memory_store().load().profile.facts)
    except Exception:
        logger.exception("Could not read the profile before a rebuild")
        await message.reply_text(MEMORY_UNAVAILABLE)
        return

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(MEMORY_RUN_LABEL, callback_data="memory:run"),
                InlineKeyboardButton(MEMORY_CANCEL_LABEL, callback_data="memory:cancel"),
            ]
        ]
    )
    confirmation = await _send_plain(
        context.bot,
        chat_id,
        MEMORY_CONFIRM.format(focus=_focus_label(focus), count=count),
        reply_to=message.message_id,
        reply_markup=keyboard,
    )
    logger.info(
        "A profile rebuild is waiting to be confirmed: %d fact(s), focus of %d character(s)",
        count,
        len(focus or ""),
    )
    _remember_prompt(_memory_confirmations, (chat_id, confirmation.message_id), focus)


async def memory_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Step three: the go-ahead, or the cancel. Nothing has run before this point."""
    global _rebuild_in_flight

    query = update.callback_query
    chat_id = update.effective_chat.id
    address = (chat_id, query.message.message_id)

    if query.data.endswith(":cancel"):
        _memory_confirmations.pop(address, None)
        await query.answer()
        await _edit_plain(context.bot, chat_id, query.message.message_id, MEMORY_CANCELLED)
        return

    if address not in _memory_confirmations:
        # A confirmation from before a restart, or one that has already been
        # pressed. Either way there is no focus behind it and starting a pass
        # that ignores what was asked for is worse than asking again.
        await query.answer(MEMORY_GONE)
        return

    # Read and set with no await in between: two presses of the same button, or
    # a press on each of two confirmations, must not both get past here.
    if _rebuild_in_flight:
        await query.answer(MEMORY_IN_FLIGHT)
        return
    _rebuild_in_flight = True
    focus = _memory_confirmations.pop(address)

    await query.answer()
    work = _run_rebuild(context, chat_id=chat_id, focus=focus)
    try:
        # Not awaited. Updates are processed one at a time, so a pass awaited here
        # would leave the bot deaf for as long as it runs — which is the length of
        # the diary, times a model call each.
        context.application.create_task(work)
    except Exception:
        _rebuild_in_flight = False
        work.close()
        logger.exception("Could not start the profile rebuild")
        await _edit_plain(context.bot, chat_id, query.message.message_id, MEMORY_UNAVAILABLE)


async def _diary_entries(walked: dict):
    """Every entry in the diary, oldest first, as plain data for the coach package.

    A page that cannot be read becomes one `Unreadable` rather than an exception:
    a single unreachable day must not end a walk over a year of diary, and the
    pass counts it and steps over it. A failure to list the pages at all is
    different and is left to propagate — there is nothing to walk.
    """
    pages = await all_pages()
    walked["pages"] = len(pages)
    for index, page in enumerate(pages, start=1):
        walked["page"] = index
        where = page.get("id", "an unnamed page")
        try:
            entries = await read_page_entries(page)
        except Exception:
            logger.warning("Could not read the diary page %s", where, exc_info=True)
            yield coach_diary.Unreadable(where=where)
            continue
        for entry in entries:
            yield coach_diary.Entry(source=entry.source, title=entry.title, text=entry.text)


def _rebuild_report(
    result: coach_rebuild.Rebuild, *, before: int, after: int, snapshot, breaker: int
) -> str:
    """The last message of a pass: what changed, what was skipped, how to undo it."""
    lines = [
        MEMORY_ABORTED.format(count=breaker) if result.aborted else MEMORY_DONE_HEADER,
        MEMORY_DONE_COUNTS.format(
            done=result.done,
            learned=result.learned,
            skipped=result.skipped,
            before=before,
            after=after,
            created=result.created,
            modified=result.modified,
            deleted=result.deleted,
        ),
    ]
    lines.append(MEMORY_UNDO.format(path=snapshot) if snapshot else MEMORY_NO_SNAPSHOT)
    return "\n".join(lines)


async def _run_rebuild(
    context: ContextTypes.DEFAULT_TYPE, *, chat_id: int, focus: str | None
) -> None:
    """The pass itself: snapshot, walk, and one final message. Never raises."""
    global _rebuild_in_flight

    store = _memory_store()
    try:
        # Forced, like every other read that a write follows. A rebuild replaces
        # the whole profile, so starting from a copy that predates an edit made on
        # the Notion page would throw that edit away twice over: once here, and
        # again when the mirror writes this pass's result out.
        stored = await _memory_sync().pull(force=True)
        # Before the first call, not after the last: the snapshot is the only
        # thing that makes a pass over the whole diary a reversible decision, and
        # a snapshot taken at the end would be a copy of the damage.
        snapshot = store.snapshot("before-rebuild")
    except Exception:
        _rebuild_in_flight = False
        logger.exception("Could not snapshot the profile; the rebuild has not started")
        await _send_plain(context.bot, chat_id, MEMORY_UNAVAILABLE)
        return

    before = len(stored.profile.facts)
    notice = await _send_plain(context.bot, chat_id, MEMORY_STARTED)
    walked: dict[str, int] = {"page": 0, "pages": 0}

    async def persist(profile: coach_memory.MemoryList) -> None:
        # Under the same lock as every other write to this file, and re-reading
        # inside it, because the rules a conversation writes live in the same
        # document and must not be rolled back to what they were when the pass
        # started. Only the profile is this pass's to replace.
        async with _store_lock():
            store.save(replace(store.load(), profile=profile))

    async def progress(state: coach_rebuild.Progress) -> None:
        await _edit_plain(
            context.bot,
            chat_id,
            notice.message_id,
            MEMORY_PROGRESS.format(
                page=walked["page"],
                pages=walked["pages"],
                done=state.done,
                created=state.created,
                modified=state.modified,
                deleted=state.deleted,
            ),
        )

    try:
        result = await coach_rebuild.run(
            entries=_diary_entries(walked),
            profile=stored.profile,
            model=settings.profile_model,
            focus=focus,
            persist=persist,
            progress=progress,
        )
    except Exception:
        logger.exception("The profile rebuild failed")
        await _send_plain(context.bot, chat_id, MEMORY_FAILED)
        return
    finally:
        _rebuild_in_flight = False
        # Once, at the end, rather than after each entry: the pass writes the file
        # hundreds of times and the page only has to agree with it when it stops.
        # In the `finally` because a pass that broke half way still persisted
        # everything it had learned by then, and a page left disagreeing with the
        # file is a page the next pull would adopt — undoing the rebuild.
        await _mirror_memory()

    if result.done == 0 and result.skipped == 0:
        await _send_plain(context.bot, chat_id, MEMORY_NO_ENTRIES)
        return

    await _send_plain(
        context.bot,
        chat_id,
        _rebuild_report(
            result,
            before=before,
            after=len(result.profile.facts),
            snapshot=snapshot,
            breaker=coach_rebuild.CONSECUTIVE_FAILURES,
        ),
    )


# --------------------------------------------------------------------------- #
# The weekly session: the one time the coach speaks first.
# --------------------------------------------------------------------------- #

# Which persona carries the conversation if the owner replies. The weekly message
# is written by its own prompt, but a reply to it lands in an ordinary coach
# thread and something has to answer it — the reflective one rather than the
# first in the list, which is the roast.
COACH_WEEKLY_MODE = "breakdown"


async def _week_entries() -> list:
    """This week's diary, entry by entry, as plain data for the coach package.

    The same pages the Sunday report reads. A page that cannot be read is dropped
    with a log line rather than failing the week: an observation drawn from six
    days out of seven is worth more than silence.
    """
    entries = []
    for page in await get_week_pages():
        try:
            entries.extend(await read_page_entries(page))
        except Exception:
            logger.warning("Could not read a diary page for the weekly session", exc_info=True)
    return [
        coach_diary.Entry(source=entry.source, title=entry.title, text=entry.text)
        for entry in entries
        if entry.text.strip()
    ]


async def send_coach_weekly(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Once a week: read the week and the profile, and send one thing.

    Silent in the chat whatever goes wrong, and loud in the log. This is the one
    message the owner did not ask for, and a version of it that sometimes arrives
    as an apology for itself is worse than one that is occasionally missing.
    """
    if not (coach_enabled() and settings.coach_weekly_enabled):
        return

    path = coach_weekly_path()
    week = coach_weekly.week_key(diary_today())
    try:
        if coach_weekly.last_spoken(path) == week:
            # A deploy restarts the bot, and a restart re-schedules the job. The
            # week that has already been spoken about is on disk for exactly this.
            logger.info("coach weekly: %s has already been spoken about", week)
            return

        entries = await _week_entries()
        if not entries:
            logger.info("coach weekly: nothing written in %s, staying quiet", week)
            return

        text = await coach_weekly.session(
            # Read the page first, like every other answer. Once a week, so the
            # throttle never applies and this is always a fresh read.
            profile=(await _memory_sync().pull()).profile,
            entries=entries,
            model=settings.coach_model,
        )
        if not text:
            # Already logged, with the reason. Nothing reaches the chat.
            return

        chat_id = settings.allowed_user_id
        parts = _chunks(text)
        first = await _send_plain(context.bot, chat_id, parts[0])
        sent = [first.message_id]
        for part in parts[1:]:
            follow = await _send_plain(context.bot, chat_id, part, reply_to=first.message_id)
            sent.append(follow.message_id)
    except Exception:
        logger.exception("coach weekly: the session failed")
        return

    # Written down only now, because "sent" is what must not happen twice. A
    # failure before this point is a week the coach may still speak about.
    try:
        coach_weekly.remember(path, week)
    except Exception:
        logger.exception("coach weekly: could not record that %s was spoken about", week)

    # Every message of it becomes a door into the conversation, exactly as a coach
    # answer does, so that the owner can simply reply to what arrived.
    try:
        mode = coach_prompts.mode_for(COACH_WEEKLY_MODE) or coach_prompts.MODES[0]
        items, _ = coach_threads.start(
            _load_threads(),
            mode=mode.key,
            turns=[
                coach_threads.Turn(role="user", text=coach_weekly.digest(entries)),
                coach_threads.Turn(role="assistant", text=text),
            ],
            keys=[coach_threads.key(chat_id, message_id) for message_id in sent],
        )
        _save_threads(items)
    except Exception:
        # The message is already in the chat. Losing the thread costs a reply its
        # context, which is a log line rather than a second message.
        logger.exception("coach weekly: could not record the conversation")


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Last line of defence: log the traceback, tell the user something happened.

    Whatever went wrong stays in the log. The user gets one fixed sentence, so a
    third-party response body can never reach the chat by this route.
    """
    logger.error("Unhandled exception while processing an update", exc_info=context.error)

    if not isinstance(update, Update):
        return

    if update.callback_query is not None:
        # One failure, one notification. `effective_message` for a callback update is
        # the message the button sits on, so falling through to the reply below would
        # tell the user twice — once as a toast and once in the chat.
        try:
            await update.callback_query.answer(SOMETHING_BROKE)
        except Exception:
            logger.warning("Could not answer the callback query after an error", exc_info=True)
        return

    message = update.effective_message
    if message is not None:
        try:
            await message.reply_text(SOMETHING_BROKE)
        except Exception:
            logger.warning("Could not tell the user about the error", exc_info=True)


def state_file_path() -> str:
    """Where the draft pickle goes.

    systemd exports STATE_DIRECTORY for a unit that declares StateDirectory=, and under
    the deployment's ProtectSystem=strict that directory is the only one the bot may
    write to -- the working directory is read-only. A relative path would not stop the
    bot if it failed: a persistence error is reported with no update attached, so it is
    logged and swallowed, and the draft silently never gets written. The current
    directory is the fallback for running outside systemd.
    """
    return os.path.join(os.environ.get("STATE_DIRECTORY", "."), STATE_FILE_NAME)


def build_application() -> Application:
    app = (
        ApplicationBuilder()
        .token(settings.telegram_token)
        .persistence(PicklePersistence(filepath=state_file_path()))
        .build()
    )

    user_filter = filters.User(user_id=settings.allowed_user_id)

    command_handlers = [
        CommandHandler("start", handle_start),
        CommandHandler("help", handle_help),
        CommandHandler("weekly", handle_weekly),
        CommandHandler("keywords", handle_keywords),
        CommandHandler("rules", handle_rules),
        CommandHandler("memory", handle_memory),
    ]
    cancel_handler = CommandHandler("cancel", handle_cancel)

    coach_pattern = "^coach:(" + "|".join(mode.key for mode in coach_prompts.MODES) + ")$"

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.VOICE & user_filter, handle_voice)],
        states={
            PREVIEW: [
                CallbackQueryHandler(save_callback, pattern="^save$"),
                CallbackQueryHandler(toggle_highlight_callback, pattern="^toggle_highlight$"),
                CallbackQueryHandler(edit_title_callback, pattern="^edit_title$"),
                CallbackQueryHandler(edit_text_callback, pattern="^edit_text$"),
                CallbackQueryHandler(edit_tags_callback, pattern="^edit_tags$"),
                CallbackQueryHandler(cancel_callback, pattern="^cancel$"),
                CallbackQueryHandler(date_open_callback, pattern="^date_open$"),
                CallbackQueryHandler(date_back_callback, pattern="^date_back$"),
                CallbackQueryHandler(date_chosen_callback, pattern=r"^date:\d{4}-\d{2}-\d{2}$"),
                CallbackQueryHandler(coach_callback, pattern=coach_pattern),
                *command_handlers,
            ],
            EDIT_TITLE: [MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, receive_new_title), *command_handlers],
            EDIT_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, receive_new_text), *command_handlers],
            EDIT_TAGS: [MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, receive_new_tags), *command_handlers],
            ConversationHandler.TIMEOUT: [TypeHandler(Update, handle_preview_timeout)],
        },
        fallbacks=[cancel_handler],
        # With allow_reentry the entry points are checked in every state, not only when
        # no conversation is running, so a new voice message is picked up from PREVIEW
        # and from all three editing states. A VOICE handler added to PREVIEW alone
        # would still swallow a recording sent while the bot waits for a new title.
        allow_reentry=True,
        conversation_timeout=PREVIEW_TIMEOUT,
        name=CONVERSATION_NAME,
        persistent=True,
    )

    # Registered before the conversation, and that ordering is the feature. Only
    # one handler in a group takes an update, so a reply to something the coach
    # said never reaches the conversation's voice entry point — which is how a
    # voice reply inside a thread is stopped from becoming a diary entry. Doing it
    # the other way round, with a check inside handle_voice, means a draft that is
    # built and then withdrawn, and one path through that where it is not.
    app.add_handler(
        MessageHandler(
            (filters.VOICE | (filters.TEXT & ~filters.COMMAND)) & user_filter & CoachReplyFilter(),
            coach_reply,
        )
    )

    # Ahead of the conversation for the same reason, and as narrow: it matches
    # only a reply to a prompt this process is still waiting on an answer to, so
    # a new title typed into an open draft goes on reaching the conversation.
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND & user_filter & FactEditReplyFilter(),
            fact_edit_reply,
        )
    )

    # Ahead of the conversation, and it takes voice as well as text: the focus a
    # rebuild is given may be dictated, and a dictated answer that reached the
    # conversation's entry point would become a diary entry instead. Narrow for
    # the same reason as the two above — it matches a reply to one prompt this
    # process is still waiting on.
    app.add_handler(
        MessageHandler(
            (filters.VOICE | (filters.TEXT & ~filters.COMMAND)) & user_filter & MemoryFocusFilter(),
            memory_focus_reply,
        )
    )

    app.add_handler(conv_handler)
    for handler in [*command_handlers, cancel_handler]:
        app.add_handler(handler)

    # Reached only when the conversation did not take the callback — a preview left in
    # the chat by an older process, or one whose conversation has already ended. Only
    # one handler per group runs, and the conversation is registered first, so this
    # never steals a callback from a live draft.
    app.add_handler(CallbackQueryHandler(
        _draft_missing,
        pattern=r"^(save|toggle_highlight|edit_title|edit_text|edit_tags|cancel"
                r"|date_open|date_back|date:\d{4}-\d{2}-\d{2}|coach:[a-z]+)$",
    ))

    # Outside the conversation on purpose: a memory note is posted after the
    # entry is saved, which is after the conversation has ended, and the note
    # stays pressable in the chat long after the next draft has come and gone.
    app.add_handler(CallbackQueryHandler(fact_callback, pattern=r"^fact:(drop|fix):\S{1,32}$"))

    # Outside the conversation for the same reason: the confirmation is answered
    # minutes after the command, and a draft may well have come and gone since.
    app.add_handler(CallbackQueryHandler(memory_callback, pattern=r"^memory:(run|cancel)$"))

    app.add_error_handler(handle_error)

    tz = zoneinfo.ZoneInfo(settings.timezone)
    app.job_queue.run_daily(
        send_daily_summary,
        time=time(21, 0, tzinfo=tz),
        name=DAILY_SUMMARY_JOB,
    )
    app.job_queue.run_daily(
        send_weekly_report,
        time=time(21, 0, tzinfo=tz),
        days=(SUNDAY,),
        name=WEEKLY_REPORT_JOB,
    )
    if settings.coach_weekly_enabled:
        # Not registered at all when it is switched off, rather than registered
        # and returning early: a job that exists is a job somebody has to reason
        # about when the next scheduled thing misfires.
        app.job_queue.run_daily(
            send_coach_weekly,
            time=time(settings.coach_weekly_hour, settings.coach_weekly_minute, tzinfo=tz),
            days=(settings.coach_weekly_day,),
            name=COACH_WEEKLY_JOB,
        )

    return app


def main() -> None:
    app = build_application()
    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
