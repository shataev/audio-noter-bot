import logging
import os
import tempfile

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from config import settings
from services.formatter import format_entry
from services.notion import save_entry
from services.whisper import transcribe

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    await message.reply_text("Listening...")

    # Download voice message
    voice_file = await context.bot.get_file(message.voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
    await voice_file.download_to_drive(tmp_path)

    try:
        # Transcribe
        await message.reply_text("Transcribing...")
        transcription = await transcribe(tmp_path)
        logger.info("Transcription: %s", transcription)

        # Format with GPT
        await message.reply_text("Formatting...")
        title, text, tags = await format_entry(transcription)

        # Save to Notion
        updated = await save_entry(title, text, tags)

        status = "Added to today's page" if updated else "Created today's page"
        all_tags = ["Daily"] + [t for t in tags if t != "Daily"]
        tags_line = " ".join(f"`{t}`" for t in all_tags)
        await message.reply_text(
            f"✓ {status}\n\n"
            f"*{title}*\n\n"
            f"{text}\n\n"
            f"{tags_line}",
            parse_mode="Markdown",
        )

    except Exception as e:
        logger.exception("Error processing voice message")
        await message.reply_text(f"Error: {e}")
    finally:
        os.unlink(tmp_path)


def main() -> None:
    app = ApplicationBuilder().token(settings.telegram_token).build()
    allowed = filters.VOICE & filters.User(user_id=settings.allowed_user_id)
    app.add_handler(MessageHandler(allowed, handle_voice))
    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
