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


# Which API the chat models are called through: "openai" (the default) or
# "anthropic". Transcription is deliberately not part of this choice — there is
# no Anthropic equivalent of the audio endpoint, so it stays on OpenAI whatever
# this is set to, and OPENAI_API_KEY is required in every configuration.
OPENAI = "openai"
ANTHROPIC = "anthropic"

_provider = os.getenv("AI_PROVIDER", OPENAI).strip().lower()
if _provider not in (OPENAI, ANTHROPIC):
    raise ValueError(f"AI_PROVIDER must be {OPENAI!r} or {ANTHROPIC!r}, got {_provider!r}")


# One model per role rather than one model for everything: summarising is a
# cheap, mechanical job that gpt-4o-mini does well, and paying a reasoning model
# to do it buys nothing. The roles that need to think are expensive on purpose.
#
# Formatting used to be in the cheap group and is not any more. It is mechanical
# in shape — punctuate this, do not rewrite it — but the constraint is the whole
# job, and gpt-4o-mini demonstrably does not hold it on a dictated entry: it
# compresses, and the author loses words he cannot get back. That is a
# correctness cost, not a quality one, so the role pays for a model that holds
# the line. The two-path split in services/formatter.py and the shortfall guard
# beside it still apply — a better model rewrites less often, not never.
_MODEL_DEFAULTS = {
    OPENAI: {
        # Reasoning-grade, and one of the models that accepts a reasoning
        # effort — see the table in services/ai.py, which will not send the
        # parameter to a model that would reject it.
        "coach": "gpt-5",
        "formatter": "gpt-6-astra",
        "summary": "gpt-4o-mini",
    },
    ANTHROPIC: {
        # Anthropic ids are complete as written: no date suffix, ever.
        "coach": "claude-opus-5",
        "formatter": "claude-opus-5",
        "summary": "claude-opus-5",
    },
}


def _model_for(role: str, variable: str) -> str:
    """The model for one role: an explicit setting first, the provider's default after."""
    return os.getenv(variable) or _MODEL_DEFAULTS[_provider][role]


def _provider_key(variable: str) -> str:
    """A credential that only one provider needs, with a message that says which.

    The bare KeyError the required settings raise is a fine message for a
    variable everyone needs. It is a confusing one for a key that is suddenly
    required because AI_PROVIDER changed, on a box that has never spoken to
    Anthropic in its life.
    """
    try:
        return os.environ[variable]
    except KeyError:
        raise RuntimeError(f"{variable} is required when AI_PROVIDER={_provider}") from None


# The days of the week, numbered from Sunday. That is not the ISO numbering and
# it is not an accident: python-telegram-bot's ``JobQueue.run_daily`` numbers its
# ``days`` 0-6 as Sunday-Saturday, and the one time this repository assumed
# otherwise the weekly report arrived on Saturday for a month. Naming the day in
# the environment rather than numbering it there means nobody has to know that.
WEEKDAYS = (
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
)


def _weekday(variable: str, default: str) -> int:
    """A weekday name from the environment, as an index into ``WEEKDAYS``."""
    name = (os.getenv(variable) or default).strip().lower()
    try:
        return WEEKDAYS.index(name)
    except ValueError:
        raise ValueError(
            f"{variable} must be one of {', '.join(WEEKDAYS)}, got {name!r}"
        ) from None


def _flag(variable: str, default: bool) -> bool:
    """An on/off setting, read the way a person would write one."""
    raw = os.getenv(variable)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _bounded_int(variable: str, default: int, low: int, high: int) -> int:
    value = int(os.getenv(variable) or default)
    if not low <= value <= high:
        raise ValueError(f"{variable} must be {low}-{high}, got {value}")
    return value


class _Settings:
    telegram_token: str = os.environ["TELEGRAM_TOKEN"]
    openai_api_key: str = os.environ["OPENAI_API_KEY"]
    notion_token: str = os.environ["NOTION_TOKEN"]
    notion_database_id: str = os.environ["NOTION_DATABASE_ID"]
    allowed_user_id: int = int(os.environ["ALLOWED_USER_ID"])
    timezone: str = os.getenv("TIMEZONE", "Europe/Moscow")

    # When a diary day ends. A note dictated at half past midnight belongs to the
    # day that has just been lived through, not to the one that started thirty
    # minutes ago — so anything before this hour is filed under the previous
    # date. 0 restores plain calendar days.
    diary_day_start_hour: int = int(os.getenv("DIARY_DAY_START_HOUR", "4"))

    # The chat provider, and the key it needs. OpenAI's key is required above
    # whatever the provider is, because transcription never leaves OpenAI.
    ai_provider: str = _provider
    anthropic_api_key: str = (
        _provider_key("ANTHROPIC_API_KEY")
        if _provider == ANTHROPIC
        else os.getenv("ANTHROPIC_API_KEY", "")
    )

    # Models, one per role.
    # Transcription is always an OpenAI model: gpt-transcribe rather than
    # whisper-1, which has not been updated since 2022 and is measurably worse
    # outside English at the same price per minute.
    transcription_model: str = os.getenv("TRANSCRIPTION_MODEL", "gpt-transcribe")
    formatter_model: str = _model_for("formatter", "FORMATTER_MODEL")
    summary_model: str = _model_for("summary", "SUMMARY_MODEL")
    # Read by nobody yet. The coach feature lands over the next few branches and
    # should not have to come back here to be configured.
    coach_model: str = _model_for("coach", "COACH_MODEL")
    profile_model: str = os.getenv("PROFILE_MODEL") or summary_model

    # Transcription. An empty value (or "auto") lets the model detect the
    # language, which is the default: dictation is not reliably monolingual, and
    # forcing "ru" mangles the English and Thai words that turn up in it.
    transcription_language: str = os.getenv("TRANSCRIPTION_LANGUAGE", "auto")

    # The longest entry the formatter is asked to hand back. Past it the model is
    # asked for a title and tags only and the transcription is used untouched,
    # because the longer the entry the more of the reply is a copy of the input
    # and the more likely the model is to summarise it to fit.
    #
    # 6000 characters is roughly six or seven minutes of speaking, and it is the
    # value the upstream fork has run on. Here it lands just under the point where
    # an echoed reply stops being cheap: at about two characters per token it asks
    # for some three thousand output tokens, more than any other call this bot
    # makes. A setting rather than a constant so it can be moved without a deploy,
    # in either direction, if the model in use turns out to hold on for longer.
    formatter_full_text_limit: int = int(os.getenv("FORMATTER_FULL_TEXT_LIMIT", "6000"))

    # Literal terms the transcriber should lean towards — names of people and
    # places, jargon, anything it gets wrong the same way every time. Comma
    # separated. Hints, not instructions: a keyword appears in the transcript
    # only if it is actually in the audio.
    transcription_keywords: str = os.getenv("TRANSCRIPTION_KEYWORDS", "")
    # The transcription endpoint rejects uploads above 25 MB.
    max_audio_mb: float = float(os.getenv("MAX_AUDIO_MB", "25"))

    # The coach's weekly session: once a week it reads the week and the profile
    # and writes first, unprompted. Off is a supported configuration — it is one
    # model call and one message a week that nobody asked for in the moment, and
    # the owner must be able to stop it from the environment rather than from a
    # deploy. The default day and hour keep it well clear of the 21:00 recaps, so
    # that the week's report and the week's question are not read as one message.
    coach_weekly_enabled: bool = _flag("COACH_WEEKLY_ENABLED", True)
    coach_weekly_day: int = _weekday("COACH_WEEKLY_DAY", "sunday")
    coach_weekly_hour: int = _bounded_int("COACH_WEEKLY_HOUR", 12, 0, 23)
    coach_weekly_minute: int = _bounded_int("COACH_WEEKLY_MINUTE", 0, 0, 59)

    # Notion HTTP behaviour.
    notion_timeout: float = float(os.getenv("NOTION_TIMEOUT_SECONDS", "30"))
    notion_max_retries: int = int(os.getenv("NOTION_MAX_RETRIES", "3"))
    notion_retry_base_delay: float = float(os.getenv("NOTION_RETRY_BASE_DELAY", "1"))

    # Where the coach's two memory pages live. Empty is the ordinary case: they
    # are put beside the diary database, on whatever page that database sits on.
    # A database at the very top of a workspace has no such page — and the API
    # cannot create one there either — so that setup names a page here instead.
    notion_memory_parent_page_id: str = os.getenv("NOTION_MEMORY_PARENT_PAGE_ID", "").strip()

    @property
    def day_start_hour(self) -> int:
        hour = self.diary_day_start_hour
        if not 0 <= hour <= 23:
            raise ValueError(f"DIARY_DAY_START_HOUR must be 0-23, got {hour}")
        return hour

    @property
    def keywords(self) -> list[str]:
        return [k.strip() for k in self.transcription_keywords.split(",") if k.strip()]

    @property
    def max_audio_bytes(self) -> int:
        # Decimal MB, because that is what the OpenAI limit is quoted in. Reading
        # it as MiB would let a 25.5 MB file past a guard whose whole purpose is
        # to stop it reaching the API, and whose message says "at most 25 MB".
        return int(self.max_audio_mb * 1_000_000)


settings = _Settings()
