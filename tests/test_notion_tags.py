"""Tag names have to satisfy Notion before they are sent, not after.

The multi-select property object says it plainly: "Commas are not valid. Names
must be unique (case-insensitive)." Either rule broken is a 400 that fails the
entire save — and by then the transcription and the formatting have been paid
for and the entry is lost. gpt-4o-mini does produce "работа, дом" as a single
tag, so this is not a theoretical input.
"""

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

from services import notion  # noqa: E402


def names(existing, new):
    return [option["name"] for option in notion._combine_tags(existing, new)]


def page_with(tags):
    return {"properties": {"Tags": {"multi_select": [{"name": t} for t in tags]}}}


def test_a_tag_with_a_comma_becomes_two_tags():
    assert names(None, ["работа, дом"]) == ["Daily", "работа", "дом"]


def test_no_option_name_ever_contains_a_comma():
    combined = names(page_with(["спорт"]), ["a,b", " c , d ,", ",,"])

    assert all("," not in name for name in combined)
    assert combined == ["Daily", "спорт", "a", "b", "c", "d"]


def test_blank_and_whitespace_tags_are_dropped():
    assert names(None, ["", "   ", "\t", "работа"]) == ["Daily", "работа"]


def test_tags_are_deduplicated_the_way_notion_compares_them():
    """Case-insensitively, and the spelling already on the page wins."""
    combined = names(page_with(["Работа"]), ["работа", "РАБОТА", "дом"])

    assert combined == ["Daily", "Работа", "дом"]


def test_daily_is_first_and_never_repeated_in_another_case():
    assert names(page_with(["daily"]), ["DAILY", "спорт"]) == ["Daily", "спорт"]


def test_the_option_list_is_capped_at_what_a_multi_select_holds():
    combined = names(page_with([f"tag{i}" for i in range(150)]), ["новый"])

    assert len(combined) == notion.MAX_TAGS
    assert combined[0] == "Daily", "the one tag every entry is supposed to carry survives"


def test_an_entry_with_no_tags_still_gets_daily():
    assert names(None, []) == ["Daily"]
