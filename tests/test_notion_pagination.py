"""Notion read paths: cursors are followed, so nothing is silently truncated."""
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

import httpx
import pytest

from services import notion, summary


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


def _paragraph(text: str) -> dict:
    return {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": text}]}}


def _page(page_id: str, title: str) -> dict:
    return {
        "id": page_id,
        "properties": {"title": {"title": [{"plain_text": title}]}},
    }


def _pages_of(items: list, size: int) -> list[dict]:
    """Splits items into Notion-shaped paginated responses."""
    responses = []
    for start in range(0, len(items), size):
        chunk = items[start:start + size]
        has_more = start + size < len(items)
        responses.append({
            "results": chunk,
            "has_more": has_more,
            "next_cursor": f"cursor-{start + size}" if has_more else None,
        })
    return responses


@pytest.mark.asyncio
async def test_a_page_of_more_than_100_blocks_is_read_back_completely():
    blocks = [_paragraph(f"Строка номер {n}") for n in range(260)]
    responses = _pages_of(blocks, 100)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        cursor = request.url.params.get("start_cursor")
        index = 0 if cursor is None else int(cursor.split("-")[1]) // 100
        return httpx.Response(200, json=responses[index])

    _install(handler)
    fetched = await notion.get_page_blocks("page-1")

    assert len(fetched) == 260
    assert len(requests) == 3
    assert all(r.url.params["page_size"] == "100" for r in requests), "page size is explicit"
    assert requests[1].url.params["start_cursor"] == "cursor-100"


@pytest.mark.asyncio
async def test_the_daily_summary_sees_the_whole_day(monkeypatch):
    # 40 entries: a heading and a paragraph each, plus a divider after the first.
    blocks = []
    for n in range(1, 41):
        if n > 1:
            blocks.append({"type": "divider", "divider": {}})
        blocks.append(
            {"type": "heading_3", "heading_3": {"rich_text": [{"plain_text": f"Запись {n}"}]}}
        )
        blocks.append(_paragraph(f"Текст записи номер {n}"))
    assert len(blocks) > 100

    responses = _pages_of(blocks, 100)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/query"):
            return httpx.Response(
                200,
                json={
                    "results": [_page("page-1", "9 сентября | Запись 1")],
                    "has_more": False,
                    "next_cursor": None,
                },
            )
        cursor = request.url.params.get("start_cursor")
        index = 0 if cursor is None else int(cursor.split("-")[1]) // 100
        return httpx.Response(200, json=responses[index])

    _install(handler)

    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Итог дня."))]
        )

    monkeypatch.setattr(
        summary.openai_client.chat.completions, "create", fake_create, raising=False
    )

    result = await summary.generate_daily_summary()

    assert result == "Итог дня."
    assert calls[0]["model"] == "gpt-4o-mini", "the default is the model that was hardcoded"
    sent = calls[0]["messages"][1]["content"]
    assert "Текст записи номер 1" in sent
    assert "Текст записи номер 40" in sent, "the tail of the day reached the model"


@pytest.mark.asyncio
async def test_the_week_query_follows_its_cursor():
    pages = [_page(f"page-{n}", f"День {n}") for n in range(150)]
    responses = _pages_of(pages, 100)
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        cursor = body.get("start_cursor")
        index = 0 if cursor is None else int(cursor.split("-")[1]) // 100
        return httpx.Response(200, json=responses[index])

    _install(handler)
    fetched = await notion.get_week_pages()

    assert len(fetched) == 150
    assert len(bodies) == 2
    assert bodies[0]["page_size"] == 100
    assert "start_cursor" not in bodies[0]
    assert bodies[1]["start_cursor"] == "cursor-100"


@pytest.mark.asyncio
async def test_a_duplicate_page_for_today_is_reported_and_resolved_stably(caplog):
    duplicates = [_page("older", "9 сентября | Первая"), _page("newer", "9 сентября | Другая")]
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, json={"results": duplicates, "has_more": False, "next_cursor": None}
        )

    _install(handler)
    with caplog.at_level("WARNING"):
        page = await notion.get_today_page()

    assert page["id"] == "older", "the same page all day, not whichever was listed first"
    assert bodies[0]["sorts"] == [{"timestamp": "created_time", "direction": "ascending"}]
    assert bodies[0]["page_size"] == 2
    assert "More than one Notion page is dated" in caplog.text
