import asyncio
import os
import openai
from config import settings

client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

# Values that mean "let the model work the language out for itself".
_AUTO_DETECT = {"", "auto", "detect"}


class AudioTooLargeError(ValueError):
    """Raised for audio the transcription endpoint would reject on size."""


def _read_file(path: str) -> bytes:
    with open(path, "rb") as audio_file:
        return audio_file.read()


def _model_options() -> dict:
    """Language and vocabulary options, spelled the way the model expects.

    The endpoint takes both spellings of each idea and the *model* decides which
    it accepts, so sending the wrong one is a 400 rather than something ignored:

    - whisper-1 takes ``language`` (one code) and biases its vocabulary through
      ``prompt``, a free-text string it treats as preceding context.
    - gpt-transcribe and its relatives take ``languages`` (a list, because
      dictation is not reliably monolingual) and ``keywords`` (a list of literal
      terms), which is the same idea with an interface that cannot be mistaken
      for an instruction to the model.

    Omitting the language entirely is what asks the endpoint to detect it, and
    that is the default.
    """
    legacy = settings.transcription_model == "whisper-1"
    options: dict = {}

    language = settings.transcription_language.strip()
    if language.lower() not in _AUTO_DETECT:
        options["language" if legacy else "languages"] = language if legacy else [language]

    keywords = settings.keywords
    if keywords:
        # whisper-1's prompt is prose, not a list: the documented way to bias it
        # towards a vocabulary is to write the words out as if they had just
        # been said. 224 tokens is its limit, so this is a hint, not a glossary.
        options["prompt" if legacy else "keywords"] = ", ".join(keywords) if legacy else keywords

    return options


async def transcribe(audio_path: str) -> str:
    size = os.path.getsize(audio_path)
    if size > settings.max_audio_bytes:
        raise AudioTooLargeError(
            f"This recording is {size / 1024 / 1024:.1f} MB and the transcription API "
            f"accepts at most {settings.max_audio_mb:g} MB. Please send a shorter one."
        )

    # Reading the file is blocking, and the handler is on the bot's event loop.
    audio_bytes = await asyncio.to_thread(_read_file, audio_path)

    options = _model_options()

    response = await client.audio.transcriptions.create(
        model=settings.transcription_model,
        file=(os.path.basename(audio_path), audio_bytes),
        **options,
    )
    return response.text
