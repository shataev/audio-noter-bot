import logging
import os

from dotenv import load_dotenv


def _load_env_file() -> None:
    """Load .env if there is one, and survive one that cannot be read.

    On a hardened install the credentials come from systemd's EnvironmentFile,
    read as root before the service drops to its own user, and there is no .env
    at all. But python-dotenv looks for one anyway, and if it finds a file it
    cannot open it raises PermissionError out of the import — which took the bot
    down on the server the moment it stopped running as root and a leftover
    root-owned .env was still sitting in the working directory.

    A .env that cannot be read is not an error worth dying for: either the real
    values are already in the environment, in which case nothing is missing, or
    they are not, and config below fails with a KeyError naming the variable,
    which is a far better message than a traceback out of a library.
    """
    try:
        load_dotenv()
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "ignoring an unreadable .env (%s); "
            "settings must come from the environment instead",
            exc,
        )


_load_env_file()


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
