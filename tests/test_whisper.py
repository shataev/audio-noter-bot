"""Transcription: the size guard and where the language comes from."""

# config.py reads os.environ at import time and raises KeyError on a missing
# value, so the stubs have to be in place before anything from the project is
# imported. setdefault throughout, so a real environment always wins.
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import types

import pytest

from config import settings
from services import whisper


class _RecordingTranscriptions:
    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(text="расшифровка")


@pytest.fixture
def api(monkeypatch):
    recorder = _RecordingTranscriptions()
    monkeypatch.setattr(
        whisper,
        "client",
        types.SimpleNamespace(audio=types.SimpleNamespace(transcriptions=recorder)),
    )
    return recorder


def _audio_file(tmp_path, megabytes: float, decimal: bool = False):
    path = tmp_path / "voice.ogg"
    unit = 1_000_000 if decimal else 1024 * 1024
    with open(path, "wb") as handle:
        handle.truncate(int(megabytes * unit))
    return str(path)


@pytest.mark.asyncio
async def test_a_26mb_recording_is_rejected_before_anything_is_uploaded(api, tmp_path):
    path = _audio_file(tmp_path, 26)

    with pytest.raises(whisper.AudioTooLargeError) as raised:
        await whisper.transcribe(path)

    assert api.calls == [], "nothing was sent to the API"
    assert "26.0 MB" in str(raised.value)
    assert "25 MB" in str(raised.value)


@pytest.mark.asyncio
async def test_a_255mb_recording_is_rejected_too(api, tmp_path):
    """OpenAI quotes 25 MB decimal, so the guard has to mean decimal.

    Read as MiB the ceiling is 26,214,400 bytes, and everything in the band
    between that and 25,000,000 sails past a guard whose own message says it
    accepts at most 25 MB. The 26 MB test above passes under either reading.
    """
    path = _audio_file(tmp_path, 25.5, decimal=True)

    with pytest.raises(whisper.AudioTooLargeError):
        await whisper.transcribe(path)

    assert api.calls == []


@pytest.mark.asyncio
async def test_a_recording_within_the_limit_is_transcribed(api, tmp_path):
    path = _audio_file(tmp_path, 0.5)

    assert await whisper.transcribe(path) == "расшифровка"
    assert len(api.calls) == 1


@pytest.mark.asyncio
async def test_the_language_comes_from_configuration(api, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "transcription_language", "en")
    path = _audio_file(tmp_path, 0.1)

    await whisper.transcribe(path)

    assert api.calls[0]["languages"] == ["en"]


@pytest.mark.asyncio
async def test_whisper_still_gets_the_singular_spelling(api, tmp_path, monkeypatch):
    """The endpoint takes both, and the model decides which is a 400."""
    monkeypatch.setattr(settings, "transcription_model", "whisper-1")
    monkeypatch.setattr(settings, "transcription_language", "en")
    path = _audio_file(tmp_path, 0.1)

    await whisper.transcribe(path)

    assert api.calls[0]["language"] == "en"
    assert "languages" not in api.calls[0]


@pytest.mark.asyncio
async def test_an_empty_language_asks_the_endpoint_to_detect_it(api, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "transcription_language", "")
    path = _audio_file(tmp_path, 0.1)

    await whisper.transcribe(path)

    assert "language" not in api.calls[0], "omitting the parameter is what turns detection on"
    assert "languages" not in api.calls[0]


@pytest.mark.asyncio
async def test_the_default_is_to_detect_the_language(api, tmp_path):
    """Dictation is not reliably monolingual, so nothing is forced."""
    path = _audio_file(tmp_path, 0.1)

    await whisper.transcribe(path)

    assert "language" not in api.calls[0]
    assert "languages" not in api.calls[0]


@pytest.mark.asyncio
async def test_keywords_are_sent_as_a_list_to_a_current_model(api, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "transcription_keywords", "Паттайя, Storyblok , ,Лёха")
    path = _audio_file(tmp_path, 0.1)

    await whisper.transcribe(path)

    assert api.calls[0]["keywords"] == ["Паттайя", "Storyblok", "Лёха"]
    assert "prompt" not in api.calls[0]


@pytest.mark.asyncio
async def test_keywords_reach_whisper_through_its_prompt(api, tmp_path, monkeypatch):
    """whisper-1 has no keywords parameter; prose is how it is biased."""
    monkeypatch.setattr(settings, "transcription_model", "whisper-1")
    monkeypatch.setattr(settings, "transcription_keywords", "Паттайя, Лёха")
    path = _audio_file(tmp_path, 0.1)

    await whisper.transcribe(path)

    assert api.calls[0]["prompt"] == "Паттайя, Лёха"
    assert "keywords" not in api.calls[0]


@pytest.mark.asyncio
async def test_no_keywords_means_no_parameter(api, tmp_path):
    path = _audio_file(tmp_path, 0.1)

    await whisper.transcribe(path)

    assert "keywords" not in api.calls[0]
    assert "prompt" not in api.calls[0]


@pytest.mark.asyncio
async def test_the_model_comes_from_configuration(api, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "transcription_model", "gpt-transcribe")
    path = _audio_file(tmp_path, 0.1)

    await whisper.transcribe(path)

    assert api.calls[0]["model"] == "gpt-transcribe"


@pytest.mark.asyncio
async def test_the_size_ceiling_is_configurable(api, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_audio_mb", 1)
    path = _audio_file(tmp_path, 2)

    with pytest.raises(whisper.AudioTooLargeError):
        await whisper.transcribe(path)

    assert api.calls == []
