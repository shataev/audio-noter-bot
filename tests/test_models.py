"""The chat models are configuration, not source."""
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

import json
import types

import pytest

from config import settings
from services import formatter


def _recorder(payload: str):
    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=payload))]
        )

    return calls, create


@pytest.mark.asyncio
async def test_the_formatter_model_comes_from_configuration(monkeypatch):
    calls, create = _recorder(json.dumps({"title": "Заголовок", "text": "Текст", "tags": ["Работа"]}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)
    monkeypatch.setattr(settings, "formatter_model", "gpt-4o")

    title, text, tags = await formatter.format_entry("сырая расшифровка")

    assert (title, text, tags) == ("Заголовок", "Текст", ["Работа"])
    assert calls[0]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_the_formatter_model_defaults_to_the_one_that_was_hardcoded(monkeypatch):
    calls, create = _recorder(json.dumps({"title": "З", "text": "Т"}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    await formatter.format_entry("сырая расшифровка")

    assert calls[0]["model"] == "gpt-4o-mini"
