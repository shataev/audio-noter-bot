import logging
import os
import tempfile
import zoneinfo
from datetime import time

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from config import settings
from services.formatter import format_entry
from services.notion import save_entry
from services.summary import generate_daily_summary
from services.whisper import transcribe

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PREVIEW, EDIT_TITLE, EDIT_TEXT = range(3)


def _tags_line(tags: list[str]) -> str:
    all_tags = ["Daily"] + [t for t in tags if t != "Daily"]
    return " ".join(f"`{t}`" for t in all_tags)


def _preview_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✓ Сохранить", callback_data="save"),
        InlineKeyboardButton("✎ Заголовок", callback_data="edit_title"),
        InlineKeyboardButton("✎ Текст", callback_data="edit_text"),
    ]])


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    message = update.effective_message
    await message.reply_text("Listening...")

    voice_file = await context.bot.get_file(message.voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
    await voice_file.download_to_drive(tmp_path)

    try:
        await message.reply_text("Transcribing...")
        transcription = await transcribe(tmp_path)
        logger.info("Transcription: %s", transcription)

        await message.reply_text("Formatting...")
        title, text, tags = await format_entry(transcription)

        title_msg = await message.reply_text(f"*{title}*", parse_mode="Markdown")
        text_msg = await message.reply_text(text)
        tags_msg = await message.reply_text(_tags_line(tags), parse_mode="Markdown")
        buttons_msg = await message.reply_text("Действия:", reply_markup=_preview_keyboard())

        context.user_data["pending"] = {"title": title, "text": text, "tags": tags}
        context.user_data["title_msg_id"] = title_msg.message_id
        context.user_data["text_msg_id"] = text_msg.message_id
        context.user_data["tags_msg_id"] = tags_msg.message_id
        context.user_data["buttons_msg_id"] = buttons_msg.message_id

    except Exception as e:
        logger.exception("Error processing voice message")
        await message.reply_text(f"Error: {e}")
        return ConversationHandler.END
    finally:
        os.unlink(tmp_path)

    return PREVIEW


async def save_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    pending = context.user_data.get("pending", {})
    title = pending.get("title", "")
    text = pending.get("text", "")
    tags = pending.get("tags", [])

    try:
        updated = await save_entry(title, text, tags)
        status = "Запись добавлена в сегодняшнюю страницу" if updated else "Запись сохранена в Notion"
        await query.edit_message_text(f"✓ {status}")
    except Exception as e:
        logger.exception("Error saving to Notion")
        await query.edit_message_text(f"Ошибка: {e}")

    context.user_data.clear()
    return ConversationHandler.END


async def edit_title_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    current = context.user_data["pending"]["title"]
    await query.message.reply_text(
        f"Текущий заголовок (скопируй и отредактируй):\n\n{current}\n\nОтправь новый вариант:"
    )
    return EDIT_TITLE


async def edit_text_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    current = context.user_data["pending"]["text"]
    await query.message.reply_text(
        f"Текущий текст (скопируй и отредактируй):\n\n{current}\n\nОтправь новый вариант:"
    )
    return EDIT_TEXT


async def receive_new_title(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    new_title = update.effective_message.text.strip()
    context.user_data["pending"]["title"] = new_title

    await context.bot.edit_message_text(
        chat_id=update.effective_chat.id,
        message_id=context.user_data["title_msg_id"],
        text=f"*{new_title}*",
        parse_mode="Markdown",
    )
    await context.bot.edit_message_reply_markup(
        chat_id=update.effective_chat.id,
        message_id=context.user_data["buttons_msg_id"],
        reply_markup=_preview_keyboard(),
    )
    return PREVIEW


async def receive_new_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    new_text = update.effective_message.text.strip()
    context.user_data["pending"]["text"] = new_text

    await context.bot.edit_message_text(
        chat_id=update.effective_chat.id,
        message_id=context.user_data["text_msg_id"],
        text=new_text,
    )
    await context.bot.edit_message_reply_markup(
        chat_id=update.effective_chat.id,
        message_id=context.user_data["buttons_msg_id"],
        reply_markup=_preview_keyboard(),
    )
    return PREVIEW


async def send_daily_summary(context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Generating daily summary...")
    try:
        summary = await generate_daily_summary()
        if summary:
            await context.bot.send_message(
                chat_id=settings.allowed_user_id,
                text=f"*Daily summary*\n\n{summary}",
                parse_mode="Markdown",
            )
        else:
            await context.bot.send_message(
                chat_id=settings.allowed_user_id,
                text="Хей, как прошел твой день? Уверен, тебе есть чем гордиться!",
            )
    except Exception:
        logger.exception("Error generating daily summary")


def main() -> None:
    app = ApplicationBuilder().token(settings.telegram_token).build()

    user_filter = filters.User(user_id=settings.allowed_user_id)

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.VOICE & user_filter, handle_voice)],
        states={
            PREVIEW: [
                CallbackQueryHandler(save_callback, pattern="^save$"),
                CallbackQueryHandler(edit_title_callback, pattern="^edit_title$"),
                CallbackQueryHandler(edit_text_callback, pattern="^edit_text$"),
            ],
            EDIT_TITLE: [MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, receive_new_title)],
            EDIT_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, receive_new_text)],
        },
        fallbacks=[],
    )

    app.add_handler(conv_handler)

    tz = zoneinfo.ZoneInfo(settings.timezone)
    app.job_queue.run_daily(
        send_daily_summary,
        time=time(21, 00, tzinfo=tz),
    )

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
