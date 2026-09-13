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
import logging
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
async def test_the_formatter_model_defaults_to_the_reasoning_grade_one(monkeypatch):
    """The role moved off gpt-4o-mini, which did not hold the no-rewriting rule.

    The deployed environment sets no FORMATTER_MODEL, so this default is what
    actually runs on the server.
    """
    calls, create = _recorder(json.dumps({"title": "З", "text": "Т"}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    await formatter.format_entry("сырая расшифровка")

    assert calls[0]["model"] == "gpt-6-astra"


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
async def test_the_summary_model_defaults_to_the_configured_one(monkeypatch):
    """SUMMARY_MODEL is not set in the deployed environment, so the default in
    config.py is what actually runs the recap the owner reads every evening."""
    _stub_day(monkeypatch, DAY_BLOCKS)
    calls, create = _recorder("Итог дня.")
    monkeypatch.setattr(summary.openai_client.chat.completions, "create", create, raising=False)

    await summary.generate_daily_summary()

    assert calls[0]["model"] == settings.summary_model == "gpt-6-astra"


@pytest.mark.asyncio
async def test_the_formatter_budget_follows_the_length_of_the_entry(monkeypatch):
    """A constant ceiling truncates a long entry mid-JSON and loses the save.

    The reply has to carry the whole entry back near enough word for word, so
    the budget cannot be a constant that was picked for a short one.
    """
    calls, create = _recorder(json.dumps({"title": "З", "text": "Т"}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    short = "Короткая запись."
    # Still on the formatting path: the budget only has to grow up to the point
    # where the entry stops being asked for back at all.
    long = "Длинная запись. " * 300  # 4800 characters
    assert len(long) <= settings.formatter_full_text_limit

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


# --------------------------------------------------------------------------- #
# The prompt above is not enough on its own: `gpt-4o-mini` compresses a long
# dictation however firmly it is told not to. `measure_kept` is what lets the
# caller notice, and everything about it turns on the two mistakes it must not
# make — calling a well-punctuated reply short, and calling a rewritten one whole.
# --------------------------------------------------------------------------- #

DICTATED = (
    "ну вот сегодня я это самое сходил на пробежку было тяжело первые "
    "два километра потом как-то разбежался и стало нормально"
)


def test_punctuation_and_paragraphs_are_not_a_shortfall():
    """Everything the formatter is allowed to add has to count for nothing."""
    formatted = (
        "Ну вот, сегодня я, это самое, сходил на пробежку.\n\n"
        "Было тяжело первые два километра — потом как-то разбежался, и стало нормально!"
    )

    kept = formatter.measure_kept(DICTATED, formatted)

    assert kept.ratio == 1.0, "the same words, so nothing was lost"
    assert not kept.too_little


def test_a_word_swapped_for_a_longer_one_is_not_a_shortfall():
    """Fixing a misheard word is allowed, and it moves the count either way."""
    kept = formatter.measure_kept("щас пойду", "Сейчас пойду.")

    assert kept.ratio > 1.0
    assert not kept.too_little


def test_an_entry_that_came_back_compressed_is_a_shortfall():
    """The reported defect: the model summarises instead of punctuating."""
    kept = formatter.measure_kept(DICTATED, "Сходил на пробежку. Первые два километра тяжело.")

    assert kept.too_little
    assert kept.ratio < 0.9


def test_the_threshold_is_a_tenth_of_the_characters():
    """The boundary itself, from both sides.

    A guard that fires in ordinary use is worse than no guard, so exactly a tenth
    lost is still allowed through; it is the next character that is not.
    """
    dictated = "а" * 1000

    assert not formatter.measure_kept(dictated, "а" * 900).too_little
    assert formatter.measure_kept(dictated, "а" * 899).too_little


def test_a_transcription_of_nothing_has_nothing_to_lose():
    """A voice message that transcribed to silence must not trip the guard."""
    kept = formatter.measure_kept("", "")

    assert kept.spoken == 0
    assert kept.ratio == 1.0
    assert not kept.too_little


def test_an_empty_reply_loses_everything():
    """The extreme of the same defect, and the one a naive check would miss."""
    kept = formatter.measure_kept(DICTATED, "")

    assert kept.kept == 0
    assert kept.ratio == 0.0
    assert kept.too_little


def test_punctuation_cannot_disguise_a_loss():
    """The reason the comparison strips what the formatter is allowed to add.

    A reply that dropped a sixth of the words and punctuated what was left comes
    out almost exactly as long as the dictation it was made from. Compared raw it
    looks untouched; compared on letters and digits it is what it is.
    """
    dictated = "слово " * 50
    formatted = "Слово, " * 42

    assert len(formatted) / len(dictated) > 0.9, "raw, this reply looks whole"
    assert formatter.measure_kept(dictated, formatted).too_little


# --------------------------------------------------------------------------- #
# Two paths through the formatter
#
# Above a length the model is not asked to hand the entry back at all: it is
# asked to name it, and the transcription goes through as it is. That removes
# the failure the guard above exists to catch, rather than detecting it — a model
# that was never asked to echo six thousand characters cannot shorten them.
#
# The boundary is tested from both sides, because it is the whole of the
# behaviour: which path an entry takes is decided by its length and nothing else.
# --------------------------------------------------------------------------- #

LIMIT = settings.formatter_full_text_limit
DICTATION = "и вот еще что я хотел сказать про это самое "


def _entry_of(length: int) -> str:
    """A plausible dictation of exactly that many characters."""
    return (DICTATION * (length // len(DICTATION) + 1))[:length]


# Something no fixture or log format could produce by accident, so finding it in
# a log means it came out of the entry.
DISTINCTIVE = "ну вот сегодня я сходил на пробежку "


def _entry_of_exactly(length: int) -> str:
    """That many characters, opening with something recognisable.

    Exactly, because which path the entry takes is decided by its length: an
    opening bolted onto a full-length body pushes a "short" case over the limit
    and quietly tests the long path twice.
    """
    assert length >= len(DISTINCTIVE)
    return (DISTINCTIVE + _entry_of(length))[:length]


@pytest.mark.asyncio
async def test_an_entry_at_the_limit_is_still_asked_to_be_formatted(monkeypatch):
    entry = _entry_of(LIMIT)
    calls, create = _recorder(json.dumps({"title": "Заголовок", "text": "Отформатировано."}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    title, text, _ = await formatter.format_entry(entry)

    assert calls[0]["messages"][0]["content"] == formatter.SYSTEM_PROMPT
    assert (title, text) == ("Заголовок", "Отформатировано.")


@pytest.mark.asyncio
async def test_one_character_past_the_limit_asks_for_metadata_only(monkeypatch):
    entry = _entry_of(LIMIT + 1)
    calls, create = _recorder(json.dumps({"title": "Заголовок", "tags": ["sport"]}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    title, text, tags = await formatter.format_entry(entry)

    assert calls[0]["messages"][0]["content"] == formatter.METADATA_PROMPT
    assert text == entry, "byte for byte what went in"
    assert (title, tags) == ("Заголовок", ["sport"])


@pytest.mark.asyncio
async def test_the_long_prompt_says_not_to_return_the_text(monkeypatch):
    """A model handed the whole entry will volunteer it back unless told not to."""
    prompt = formatter.METADATA_PROMPT

    assert '"text"' in prompt
    assert "НЕ нужно" in prompt or "не нужно" in prompt
    assert "Запрещено" in prompt
    assert "пересказывать" in prompt
    assert '"title"' in prompt and '"tags"' in prompt


@pytest.mark.asyncio
async def test_the_long_path_does_not_pay_for_a_copy_of_the_entry(monkeypatch):
    """The budget that grows with the input is the losing side of this argument."""
    entry = _entry_of(LIMIT * 3)
    calls, create = _recorder(json.dumps({"title": "Заголовок"}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    await formatter.format_entry(entry)

    assert calls[0]["max_completion_tokens"] == 512
    assert calls[0]["max_completion_tokens"] < len(entry) // 2


@pytest.mark.asyncio
async def test_a_reply_with_no_title_costs_a_heading_and_not_the_entry(monkeypatch):
    entry = _entry_of_exactly(LIMIT + 1)
    calls, create = _recorder(json.dumps({"tags": []}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    title, text, tags = await formatter.format_entry(entry)

    assert title == "ну вот сегодня я сходил", "the opening words, which is better than nothing"
    assert text == entry
    assert tags == []
    assert calls, "the model was still asked"


@pytest.mark.asyncio
async def test_a_reply_that_is_not_readable_json_still_keeps_the_entry(monkeypatch):
    """Nothing the model returns can cost the owner the note on this path: the
    text was safe before the call was made."""
    entry = _entry_of_exactly(LIMIT + 1)
    _, create = _recorder("не json вовсе")
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    title, text, tags = await formatter.format_entry(entry)

    assert text == entry
    assert title == "ну вот сегодня я сходил"
    assert tags == []


@pytest.mark.asyncio
async def test_a_blank_title_is_not_a_title(monkeypatch):
    entry = _entry_of_exactly(LIMIT + 1)
    _, create = _recorder(json.dumps({"title": "   "}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    title, _, _ = await formatter.format_entry(entry)

    assert title == "ну вот сегодня я сходил"


@pytest.mark.asyncio
async def test_tags_that_are_not_strings_are_dropped_rather_than_passed_on(monkeypatch):
    entry = _entry_of(LIMIT + 1)
    _, create = _recorder(json.dumps({"title": "З", "tags": ["sport", 7, None, "  ", "work"]}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    _, _, tags = await formatter.format_entry(entry)

    assert tags == ["sport", "work"]


@pytest.mark.asyncio
async def test_a_long_entry_cannot_trip_the_shortfall_guard(monkeypatch):
    """The two halves of the fix meet here: the guard is belt, this is braces."""
    entry = _entry_of(LIMIT * 2)
    _, create = _recorder(json.dumps({"title": "З"}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    _, text, _ = await formatter.format_entry(entry)

    assert not formatter.measure_kept(entry, text).too_little
    assert formatter.measure_kept(entry, text).ratio == 1.0


@pytest.mark.asyncio
async def test_the_threshold_can_be_moved_without_a_deploy(monkeypatch):
    """It is a setting because the right value depends on the model in use."""
    monkeypatch.setattr(settings, "formatter_full_text_limit", 10)
    calls, create = _recorder(json.dumps({"title": "З", "text": "Т"}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    _, text, _ = await formatter.format_entry("одиннадцать!")

    assert calls[0]["messages"][0]["content"] == formatter.METADATA_PROMPT
    assert text == "одиннадцать!"


@pytest.mark.asyncio
async def test_a_model_that_returns_the_text_anyway_is_ignored(monkeypatch):
    """The prompt forbids it, which is not the same as the model obeying.

    Whatever comes back in a "text" field on this path is a model's idea of the
    entry rather than the entry, and it has already been told not to send one.
    The transcription is what the draft gets, and that must not depend on the
    reply happening to leave the field out.
    """
    entry = _entry_of(LIMIT + 1)
    _, create = _recorder(json.dumps({"title": "З", "text": "Краткий пересказ записи."}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    _, text, _ = await formatter.format_entry(entry)

    assert text == entry


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [LIMIT, LIMIT + 1], ids=["short path", "long path"])
async def test_neither_path_ever_logs_the_entry(monkeypatch, caplog, length):
    """Both paths, because the formatter holds the whole entry on both of them.

    Same rule the coach package is held to, and the same shape of test: a
    distinctive sentence, every handler listening, and the assertion that it is
    nowhere in what was written. What is wanted in the log is the length, which
    is what says whether the whole thing arrived.
    """
    entry = _entry_of_exactly(length)
    _, create = _recorder(json.dumps({"title": "З", "text": "Текст."}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    with caplog.at_level(logging.DEBUG):
        await formatter.format_entry(entry)

    assert DISTINCTIVE not in caplog.text
    assert entry not in caplog.text
    assert str(len(entry)) in caplog.text, "the length is the part worth keeping"


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [LIMIT, LIMIT + 1], ids=["short path", "long path"])
async def test_the_parametrised_lengths_really_do_take_different_paths(monkeypatch, length):
    """Guards the test above, which says nothing if both cases go the same way."""
    calls, create = _recorder(json.dumps({"title": "З", "text": "Т"}))
    monkeypatch.setattr(formatter.client.chat.completions, "create", create, raising=False)

    await formatter.format_entry(_entry_of_exactly(length))

    expected = formatter.SYSTEM_PROMPT if length <= LIMIT else formatter.METADATA_PROMPT
    assert calls[0]["messages"][0]["content"] == expected
