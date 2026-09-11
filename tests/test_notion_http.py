"""Notion transport: the shared client, its timeout, and what is retried."""
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

import asyncio
import time

import httpx
import pytest

from config import settings
from services import notion


@pytest.fixture(autouse=True)
def _reset_client():
    notion._client = None
    yield
    notion._client = None


@pytest.fixture
def no_waiting(monkeypatch):
    """Records what _request would have slept for, without sleeping."""
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return waits


def _install(handler) -> None:
    notion._client = httpx.AsyncClient(
        base_url=notion.API,
        headers=notion.HEADERS,
        transport=httpx.MockTransport(handler),
    )


def test_the_client_is_shared_and_carries_the_configured_timeout(monkeypatch):
    monkeypatch.setattr(settings, "notion_timeout", 12.5)

    client = notion.get_client()

    assert notion.get_client() is client, "one pooled client, not one per call"
    assert client.timeout.connect == 12.5
    assert client.timeout.read == 12.5


@pytest.mark.asyncio
async def test_rate_limiting_is_retried_and_then_succeeds(no_waiting, monkeypatch):
    monkeypatch.setattr(settings, "notion_max_retries", 3)
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(
                429, headers={"Retry-After": "2"}, json={"code": "rate_limited"}
            )
        return httpx.Response(200, json={"results": [], "has_more": False})

    _install(handler)
    response = await notion._request("POST", "/databases/test-db/query", json={})

    assert response.status_code == 200
    assert len(attempts) == 2
    assert no_waiting == [2.0], "Notion's Retry-After is honoured rather than guessed at"


@pytest.mark.asyncio
async def test_a_validation_error_is_not_retried(no_waiting, monkeypatch):
    monkeypatch.setattr(settings, "notion_max_retries", 3)
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(
            400,
            json={
                "object": "error",
                "code": "validation_error",
                "message": "body failed validation",
            },
        )

    _install(handler)
    with pytest.raises(notion.NotionError) as raised:
        await notion._request("POST", "/pages", json={})

    assert len(attempts) == 1, (
        "a 400 fails identically every time; retrying only wastes the user's wait"
    )
    assert no_waiting == []
    assert raised.value.status_code == 400
    assert "validation_error" in str(raised.value)


@pytest.mark.asyncio
async def test_a_transient_server_error_is_retried_for_a_repeatable_request(
    no_waiting, monkeypatch
):
    monkeypatch.setattr(settings, "notion_max_retries", 3)
    monkeypatch.setattr(settings, "notion_retry_base_delay", 1.0)
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) < 3:
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, json={"ok": True})

    _install(handler)
    response = await notion._request("PATCH", "/pages/page-1", json={})

    assert response.status_code == 200
    assert len(attempts) == 3
    assert no_waiting == [1.0, 2.0], "backoff grows between attempts"


@pytest.mark.asyncio
async def test_appending_blocks_is_not_retried_after_a_server_error(no_waiting, monkeypatch):
    """A 5xx may mean Notion applied the append anyway; a replay would double it."""
    monkeypatch.setattr(settings, "notion_max_retries", 3)
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(502, text="bad gateway")

    _install(handler)
    with pytest.raises(notion.NotionError):
        await notion._request(
            "PATCH", "/blocks/page-1/children", json={"children": []}, repeatable=False
        )

    assert len(attempts) == 1
    assert no_waiting == []


@pytest.mark.asyncio
async def test_appending_blocks_is_still_retried_after_a_rate_limit(no_waiting, monkeypatch):
    """Notion rejects a 429 without doing any work, so a replay cannot duplicate."""
    monkeypatch.setattr(settings, "notion_max_retries", 3)
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(
                429, headers={"Retry-After": "1"}, json={"code": "rate_limited"}
            )
        return httpx.Response(200, json={"ok": True})

    _install(handler)
    response = await notion._request(
        "PATCH", "/blocks/page-1/children", json={"children": []}, repeatable=False
    )

    assert response.status_code == 200
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_a_connection_failure_before_sending_is_retried_even_when_appending(
    no_waiting, monkeypatch
):
    monkeypatch.setattr(settings, "notion_max_retries", 2)
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"ok": True})

    _install(handler)
    response = await notion._request(
        "PATCH", "/blocks/page-1/children", json={"children": []}, repeatable=False
    )

    assert response.status_code == 200
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_a_read_timeout_is_not_retried_when_appending(no_waiting, monkeypatch):
    """The request did leave this process, so Notion may have applied it."""
    monkeypatch.setattr(settings, "notion_max_retries", 3)
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    _install(handler)
    with pytest.raises(notion.NotionError):
        await notion._request(
            "PATCH", "/blocks/page-1/children", json={"children": []}, repeatable=False
        )

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_a_stalled_connection_fails_within_the_configured_timeout(monkeypatch):
    """Against a server that accepts the connection and then says nothing."""

    hang_up = asyncio.Event()

    async def never_answer(reader, writer):
        try:
            await hang_up.wait()
        finally:
            writer.close()

    server = await asyncio.start_server(never_answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    monkeypatch.setattr(notion, "API", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(settings, "notion_timeout", 0.5)
    monkeypatch.setattr(settings, "notion_max_retries", 0)

    started = time.monotonic()
    try:
        with pytest.raises(notion.NotionError) as raised:
            await notion.get_page_blocks("page-1")
    finally:
        elapsed = time.monotonic() - started
        await notion.close_client()
        hang_up.set()
        server.close()
        await server.wait_closed()

    # Not "< 5": httpx's own default is 5.0s, so that assertion would still pass
    # if the explicit timeout were dropped again.
    assert elapsed < 2, f"the 0.5s timeout was not enforced; the call took {elapsed:.1f}s"
    assert isinstance(raised.value, RuntimeError), "callers catching RuntimeError still see it"
    assert "ReadTimeout" in str(raised.value)
