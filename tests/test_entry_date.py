"""Which day an entry belongs to, and how it is chosen.

Two ideas, one subject. A diary day does not end at midnight — a note dictated
at half past one belongs to the day just lived through — and an entry dictated
today may be about an earlier one.
"""

import os
import pathlib
import sys
import zoneinfo
from datetime import date, datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import pytest  # noqa: E402

from config import settings  # noqa: E402
from services import notion  # noqa: E402


def _at(hour, minute=30, day=12):
    return datetime(2026, 9, day, hour, minute, tzinfo=zoneinfo.ZoneInfo(settings.timezone))


# --- the boundary -----------------------------------------------------------


@pytest.mark.parametrize(
    "hour, expected_day",
    [(0, 11), (1, 11), (3, 11), (4, 12), (5, 12), (13, 12), (23, 12)],
)
def test_a_night_note_belongs_to_the_day_just_lived_through(monkeypatch, hour, expected_day):
    monkeypatch.setattr(settings, "diary_day_start_hour", 4)

    assert notion.diary_today(_at(hour)) == date(2026, 9, expected_day)


def test_zero_gives_plain_calendar_days(monkeypatch):
    monkeypatch.setattr(settings, "diary_day_start_hour", 0)

    assert notion.diary_today(_at(0, 1)) == date(2026, 9, 12)
    assert notion.diary_today(_at(23)) == date(2026, 9, 12)


def test_an_hour_outside_the_clock_is_refused(monkeypatch):
    monkeypatch.setattr(settings, "diary_day_start_hour", 24)

    with pytest.raises(ValueError):
        notion.diary_today(_at(12))


def test_the_label_is_the_page_title_prefix():
    assert notion.day_label(date(2026, 5, 9)) == "9 мая"


# --- saving under a chosen day ----------------------------------------------


def _capture(monkeypatch):
    """Records what save_entry asks Notion for, without a transport."""
    seen = {}

    async def fake_get_today_page(day=None):
        seen["queried"] = day
        return None

    async def fake_create_page(title, text, tags, day=None):
        seen["created"] = day
        return None

    monkeypatch.setattr(notion, "get_today_page", fake_get_today_page)
    monkeypatch.setattr(notion, "create_page", fake_create_page)
    return seen


@pytest.mark.asyncio
async def test_save_entry_uses_the_day_it_is_given(monkeypatch):
    seen = _capture(monkeypatch)
    chosen = date(2026, 9, 8)

    await notion.save_entry("Заголовок", "тело", [], chosen)

    assert seen["queried"] == chosen, "the page is looked up for that day"
    assert seen["created"] == chosen, "and created with that date"


@pytest.mark.asyncio
async def test_save_entry_without_a_day_uses_the_diary_today(monkeypatch):
    seen = _capture(monkeypatch)

    await notion.save_entry("Заголовок", "тело", [])

    assert seen["queried"] == notion.diary_today()
