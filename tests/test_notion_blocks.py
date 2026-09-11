"""Notion write paths: block chunking, day titles and the children limit."""
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

import httpx
import pytest

from services import notion


@pytest.fixture(autouse=True)
def _reset_client():
    notion._client = None
    yield
    notion._client = None


def _install(handler) -> None:
    notion._client = httpx.AsyncClient(
        base_url=notion.API,
        headers=notion.HEADERS,
        transport=httpx.MockTransport(handler),
    )


def _no_today_page(request: httpx.Request) -> httpx.Response:
    """Answers the "is there a page for today" query with nothing."""
    return httpx.Response(200, json={"results": [], "has_more": False, "next_cursor": None})


def _rich_text_of(block: dict) -> list[dict]:
    return block[block["type"]].get("rich_text", [])


def _plain_text(blocks: list[dict]) -> str:
    """Reassembles the text a list of paragraph blocks would render as."""
    paragraphs = [
        "".join(rt["text"]["content"] for rt in _rich_text_of(block))
        for block in blocks
        if block["type"] == "paragraph"
    ]
    return "\n\n".join(paragraphs)


LONG_ENTRY = "\n\n".join(
    " ".join(f"Предложение номер {n} этого абзаца." for n in range(1, 31))
    for _ in range(6)
)


@pytest.mark.asyncio
async def test_long_entry_is_split_into_valid_rich_text_objects():
    assert len(LONG_ENTRY) > 6000

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/query"):
            return _no_today_page(request)
        seen.append(request)
        return httpx.Response(200, json={"id": "page-1"})

    _install(handler)
    await notion.save_entry("Длинная запись", LONG_ENTRY, ["Работа"])

    assert len(seen) == 1, "one page creation, nothing left over to append"
    body = json.loads(seen[0].content)
    children = body["children"]

    assert len(children) <= notion.MAX_CHILDREN_PER_REQUEST
    for block in children:
        rich_text = _rich_text_of(block)
        assert len(rich_text) <= notion.MAX_RICH_TEXT_PER_BLOCK
        for item in rich_text:
            assert len(item["text"]["content"]) <= notion.MAX_TEXT_CHARS
            # A cut mid-word would leave a fragment; cuts land on whitespace.
            assert not item["text"]["content"].endswith("Предложе")

    assert _plain_text(children) == LONG_ENTRY, "no text lost or duplicated in the split"


@pytest.mark.asyncio
async def test_entry_longer_than_one_request_is_appended_in_batches():
    # 250 short paragraphs become 250 blocks, well past the 100 a request holds.
    text = "\n\n".join(f"Абзац номер {n}." for n in range(250))

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/query"):
            return _no_today_page(request)
        seen.append(request)
        return httpx.Response(200, json={"id": "page-1"})

    _install(handler)
    await notion.save_entry("Очень длинная запись", text, [])

    assert len(seen) > 1, "a 251-block entry cannot be one request"
    assert seen[0].url.path.endswith("/pages")
    assert all(r.url.path.endswith("/blocks/page-1/children") for r in seen[1:])

    blocks: list[dict] = []
    for request in seen:
        children = json.loads(request.content)["children"]
        assert len(children) <= notion.MAX_CHILDREN_PER_REQUEST
        blocks.extend(children)

    assert _plain_text(blocks) == text


@pytest.mark.asyncio
async def test_appended_entry_respects_the_children_limit():
    text = "\n\n".join(f"Абзац номер {n}." for n in range(250))
    page = {
        "id": "page-1",
        "properties": {
            "title": {"title": [{"plain_text": "9 сентября | Первая запись"}]},
            "Tags": {"multi_select": []},
        },
    }

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "page-1"})

    _install(handler)
    await notion.update_page(page, "Вторая запись", text, [])

    appends = [r for r in seen if r.url.path.endswith("/children")]
    assert len(appends) > 1
    for request in appends:
        assert len(json.loads(request.content)["children"]) <= notion.MAX_CHILDREN_PER_REQUEST


def test_day_title_survives_many_entries_and_keeps_its_prefix():
    prefix = "9 сентября"
    title = notion._compose_day_title(prefix, ["Первая запись"])

    # Feed the rendered title back in the way update_page does, all day long.
    for n in range(2, 300):
        recovered_prefix, entries = notion._split_day_title(title)
        assert recovered_prefix == prefix
        title = notion._compose_day_title(recovered_prefix, [*entries, f"Запись номер {n}"])
        assert len(title) <= notion.MAX_TITLE_CHARS
        assert title.startswith(f"{prefix} | ")
        assert title.endswith(f"Запись номер {n}"), "the newest entry is always kept"

    # A realistic day never has to drop anything.
    short_day = notion._compose_day_title(prefix, [f"Запись номер {n}" for n in range(1, 66)])
    assert len(short_day) <= notion.MAX_TITLE_CHARS
    assert notion.ELLIPSIS not in short_day


PASTED_PARAGRAPH = "Это вставленный абзац из другого приложения. " * 60


def _oldest_were_dropped(title: str, prefix: str) -> bool:
    """True only for the drop guard.

    Both guards mark themselves with an ellipsis — a capped name ends in one,
    and dropped names are replaced by one — so "… appears somewhere" cannot tell
    them apart. The drop marker is always the first item in the list.
    """
    return title.startswith(f"{prefix} | {notion.ELLIPSIS}, ")


def test_an_ordinary_day_is_the_accumulated_list_and_nothing_else():
    """The format is the owner's table of contents; neither guard may touch it."""
    prefix = "9 мая"
    entries = [f"Событие {n}" for n in range(1, 66)]

    title = notion._compose_day_title(prefix, entries)

    assert title == f"{prefix} | " + ", ".join(entries), "no reshaping, no shortening"
    assert notion.ELLIPSIS not in title
    # Well past the 65 entries a real day would hold, still untouched.
    for count in (80, 100):
        longer = notion._compose_day_title(prefix, [f"Событие {n}" for n in range(1, count + 1)])
        assert len(longer) <= notion.MAX_TITLE_CHARS
        assert notion.ELLIPSIS not in longer


def test_capping_one_name_does_not_drop_any_others():
    """The cheap guard fires alone; the drop guard is a genuine last resort."""
    prefix = "9 мая"
    entries = ["Событие 1", PASTED_PARAGRAPH, "Событие 3"]

    title = notion._compose_day_title(prefix, entries)
    names = notion._split_day_title(title)[1]

    assert len(names) == 3, "every entry of the day is still named"
    assert max(len(name) for name in names) == notion.MAX_ENTRY_TITLE_CHARS
    assert not _oldest_were_dropped(title, prefix)


def test_names_are_dropped_only_once_capping_is_not_enough():
    prefix = "9 мая"

    # Capping alone absorbs a handful of pasted paragraphs.
    few = notion._compose_day_title(prefix, [PASTED_PARAGRAPH] * 5)
    assert not _oldest_were_dropped(few, prefix)

    # It takes an implausible day of them before the oldest have to go.
    many = notion._compose_day_title(prefix, [PASTED_PARAGRAPH] * 40)
    assert _oldest_were_dropped(many, prefix)
    assert len(many) <= notion.MAX_TITLE_CHARS
    assert many.startswith(f"{prefix} | "), "the date prefix survives the drop"


def test_one_pathological_entry_title_cannot_blow_the_limit():
    title = notion._compose_day_title("9 сентября", ["Обычная запись", PASTED_PARAGRAPH])

    assert len(title) <= notion.MAX_TITLE_CHARS
    assert title.startswith("9 сентября | Обычная запись, ")
    names = notion._split_day_title(title)[1]
    assert all(len(name) <= notion.MAX_ENTRY_TITLE_CHARS for name in names)


@pytest.mark.asyncio
async def test_update_page_sends_a_title_within_the_limit():
    entries = ", ".join(f"Запись номер {n}" for n in range(1, 400))
    page = {
        "id": "page-1",
        "properties": {
            "title": {"title": [{"plain_text": f"9 сентября | {entries}"}]},
            "Tags": {"multi_select": []},
        },
    }

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "page-1"})

    _install(handler)
    await notion.update_page(page, "Свежая запись", "Короткий текст.", [])

    properties = json.loads(seen[0].content)["properties"]
    sent = "".join(item["text"]["content"] for item in properties["title"]["title"])
    assert len(sent) <= notion.MAX_TITLE_CHARS
    assert sent.startswith("9 сентября | ")
    assert sent.endswith("Свежая запись")
