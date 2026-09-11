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


async def transcribe(audio_path: str) -> str:
    size = os.path.getsize(audio_path)
    if size > settings.max_audio_bytes:
        raise AudioTooLargeError(
            f"This recording is {size / 1024 / 1024:.1f} MB and the transcription API "
            f"accepts at most {settings.max_audio_mb:g} MB. Please send a shorter one."
        )

    # Reading the file is blocking, and the handler is on the bot's event loop.
    audio_bytes = await asyncio.to_thread(_read_file, audio_path)

    options = {}
    language = settings.transcription_language.strip()
    # Omitting the parameter is what asks the endpoint to detect the language.
    if language.lower() not in _AUTO_DETECT:
        options["language"] = language

    response = await client.audio.transcriptions.create(
        model=settings.transcription_model,
        file=(os.path.basename(audio_path), audio_bytes),
        **options,
    )
    return response.text
