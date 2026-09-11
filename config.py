import os
from dotenv import load_dotenv

load_dotenv()


class _Settings:
    telegram_token: str = os.environ["TELEGRAM_TOKEN"]
    openai_api_key: str = os.environ["OPENAI_API_KEY"]
    notion_token: str = os.environ["NOTION_TOKEN"]
    notion_database_id: str = os.environ["NOTION_DATABASE_ID"]
    allowed_user_id: int = int(os.environ["ALLOWED_USER_ID"])
    timezone: str = os.getenv("TIMEZONE", "Europe/Moscow")

    # OpenAI models. Defaults keep the models the bot has always used.
    transcription_model: str = os.getenv("TRANSCRIPTION_MODEL", "whisper-1")
    formatter_model: str = os.getenv("FORMATTER_MODEL", "gpt-4o-mini")
    summary_model: str = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")

    # Transcription. An empty value (or "auto") lets the model detect the language.
    transcription_language: str = os.getenv("TRANSCRIPTION_LANGUAGE", "ru")
    # The transcription endpoint rejects uploads above 25 MB.
    max_audio_mb: float = float(os.getenv("MAX_AUDIO_MB", "25"))

    # Notion HTTP behaviour.
    notion_timeout: float = float(os.getenv("NOTION_TIMEOUT_SECONDS", "30"))
    notion_max_retries: int = int(os.getenv("NOTION_MAX_RETRIES", "3"))
    notion_retry_base_delay: float = float(os.getenv("NOTION_RETRY_BASE_DELAY", "1"))

    @property
    def max_audio_bytes(self) -> int:
        # Decimal MB, because that is what the OpenAI limit is quoted in. Reading
        # it as MiB would let a 25.5 MB file past a guard whose whole purpose is
        # to stop it reaching the API, and whose message says "at most 25 MB".
        return int(self.max_audio_mb * 1_000_000)


settings = _Settings()
