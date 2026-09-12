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
from services import formatter, summary


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
    payload = {"title": "Заголовок", "text": "Текст", "tags": ["Работа"]}
    calls, create = _recorder(json.dumps(payload))
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


def _stub_day(monkeypatch, blocks):
    """Replaces the Notion reads summary.py imported, so no transport is needed."""

    async def today_page():
        return {
            "id": "page-1",
            "properties": {"title": {"title": [{"plain_text": "9 мая | Запись"}]}},
        }

    async def week_pages():
        return [await today_page()]

    async def page_blocks(page_id):
        return blocks

    monkeypatch.setattr(summary, "get_today_page", today_page)
    monkeypatch.setattr(summary, "get_week_pages", week_pages)
    monkeypatch.setattr(summary, "get_page_blocks", page_blocks)


DAY_BLOCKS = [{"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Текст записи"}]}}]


@pytest.mark.asyncio
async def test_the_daily_summary_model_comes_from_configuration(monkeypatch):
    _stub_day(monkeypatch, DAY_BLOCKS)
    calls, create = _recorder("Итог дня.")
    monkeypatch.setattr(summary.openai_client.chat.completions, "create", create, raising=False)
    monkeypatch.setattr(settings, "summary_model", "gpt-4o")

    assert await summary.generate_daily_summary() == "Итог дня."
    assert calls[0]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_the_weekly_report_model_comes_from_configuration(monkeypatch):
    _stub_day(monkeypatch, DAY_BLOCKS)
    calls, create = _recorder("Итог недели.")
    monkeypatch.setattr(summary.openai_client.chat.completions, "create", create, raising=False)
    monkeypatch.setattr(settings, "summary_model", "gpt-4o")

    assert await summary.generate_weekly_report() == "Итог недели."
    assert calls[0]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_the_summary_model_defaults_to_the_one_that_was_hardcoded(monkeypatch):
    _stub_day(monkeypatch, DAY_BLOCKS)
    calls, create = _recorder("Итог дня.")
    monkeypatch.setattr(summary.openai_client.chat.completions, "create", create, raising=False)

    await summary.generate_daily_summary()

    assert calls[0]["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_the_formatter_budget_follows_the_length_of_the_entry(monkeypatch):
    """A constant ceiling truncates a long entry mid-JSON and loses the save.

    The reply has to carry the whole entry back near enough word for word, so
    the budget cannot be a constant that was picked for a short one.
    """
    calls, create = _recorder(json.dumps({"title": "З", "text": "Т"}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    short = "Короткая запись."
    long = "Длинная запись. " * 2000  # 32000 characters

    await formatter.format_entry(short)
    await formatter.format_entry(long)

    small, large = (call["max_completion_tokens"] for call in calls)
    assert small == 1024, "the floor, so short entries are predictable"
    assert large > small
    assert large >= len(long) // 2, "enough room to echo the entry back"
    assert large <= 16384, "and never past the model's own output limit"


@pytest.mark.asyncio
async def test_the_formatter_is_told_not_to_rewrite(monkeypatch):
    """The entry is the author's own words; only the title is invented.

    A product decision, not a prompt-engineering detail: a diary read back in a
    year is worth what it is because of how it was said at the time.
    """
    calls, create = _recorder(json.dumps({"title": "З", "text": "Т"}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    await formatter.format_entry("ну вот эээ сегодня я это самое сходил на пробежку")

    prompt = calls[0]["messages"][0]["content"]
    assert "НЕ переписывая" in prompt
    assert "слова-паразиты" in prompt, "filler is named as something to keep, not strip"
    assert "Запрещено" in prompt
