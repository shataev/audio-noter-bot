import asyncio
import logging
from datetime import datetime, timedelta
import zoneinfo
import httpx
from config import settings

logger = logging.getLogger(__name__)

API = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {settings.notion_token}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

# Notion answers 429 when the connection exceeds its rate limit (three requests
# per second on non-enterprise plans) and documents 529 as needing the same
# treatment as 429. The 5xx codes are transient server failures.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504, 529})
# Upper bound on a single wait, so a large Retry-After cannot park a handler.
_MAX_RETRY_WAIT = 60.0
# Failures raised before any byte of the request left this process. Replaying
# one of these cannot duplicate a write, whatever the request was.
_PRE_SEND_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ProxyError,
)

_client: httpx.AsyncClient | None = None


class NotionError(RuntimeError):
    """A Notion request that could not be completed.

    Subclasses RuntimeError so existing callers that only catch broad errors are
    unaffected. `status_code` is None when the request never got a response.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def get_client() -> httpx.AsyncClient:
    """Returns the shared Notion client, building it on first use.

    One client keeps one connection pool, so a saved entry no longer pays for a
    TCP and TLS handshake per request, and the timeout is set in one place
    instead of being left at whatever httpx defaults to.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=API,
            headers=HEADERS,
            timeout=httpx.Timeout(settings.notion_timeout),
        )
    return _client


async def close_client() -> None:
    """Closes the shared client. Safe to call when there is nothing to close."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def _retry_after(response: httpx.Response) -> float | None:
    """Reads Notion's Retry-After header, which it sends as whole seconds."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, min(float(raw), _MAX_RETRY_WAIT))
    except ValueError:
        return None


async def _request(
    method: str,
    path: str,
    *,
    json: dict | None = None,
    params: dict | None = None,
    repeatable: bool = True,
) -> httpx.Response:
    """Sends one Notion request, retrying only the failures worth retrying.

    Pass repeatable=False for a request that would duplicate data if Notion had
    already applied it — appending blocks to a page is the one that matters here.
    Such a request is retried only when the failure happened before it left this
    process, or when Notion rejected it with 429, which it does without doing any
    work. A 400 validation_error is never retried: it fails identically forever.
    """
    client = get_client()
    attempts = max(0, settings.notion_max_retries) + 1
    delay = max(0.0, settings.notion_retry_base_delay)

    for attempt in range(1, attempts + 1):
        final = attempt == attempts
        try:
            response = await client.request(method, path, json=json, params=params)
        except httpx.TransportError as exc:
            safe = repeatable or isinstance(exc, _PRE_SEND_ERRORS)
            if final or not safe:
                raise NotionError(f"Notion {method} {path} failed: {exc!r}") from exc
            wait = delay
            logger.warning(
                "Notion %s %s failed with %s, retrying in %.1fs",
                method, path, type(exc).__name__, wait,
            )
        else:
            if response.is_success:
                return response
            status = response.status_code
            retryable = status in _RETRY_STATUS and (repeatable or status == 429)
            if final or not retryable:
                raise NotionError(
                    f"Notion {method} {path} returned {status}: {response.text[:500]}",
                    status_code=status,
                )
            wait = _retry_after(response) or delay
            logger.warning(
                "Notion %s %s returned %s, retrying in %.1fs", method, path, status, wait
            )

        await asyncio.sleep(wait)
        delay = min(delay * 2, _MAX_RETRY_WAIT)

    # Not reachable: the final attempt either returns or raises above.
    raise NotionError(f"Notion {method} {path} exhausted its {attempts} attempts")


MONTHS_RU = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}


def _today_date() -> str:
    tz = zoneinfo.ZoneInfo(settings.timezone)
    return datetime.now(tz).date().isoformat()  # e.g. "2026-04-09"


def _today_label() -> str:
    tz = zoneinfo.ZoneInfo(settings.timezone)
    now = datetime.now(tz)
    return f"{now.day} {MONTHS_RU[now.month]}"


def _extract_title(page: dict) -> str:
    title_prop = page["properties"].get("title") or page["properties"].get("Name")
    parts = title_prop.get("title", [])
    return "".join(p["plain_text"] for p in parts)


def _combine_tags(existing_page: dict | None, new_tags: list[str]) -> list[dict]:
    """Merges page tags with new ones, always prepends Daily, no duplicates."""
    existing = []
    if existing_page:
        existing = [t["name"] for t in existing_page["properties"].get("Tags", {}).get("multi_select", [])]
    all_tags = ["Daily"] + [t for t in (existing + new_tags) if t != "Daily"]
    # deduplicate while preserving order
    seen = set()
    unique = []
    for t in all_tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return [{"name": t} for t in unique]


async def get_today_page() -> dict | None:
    resp = await _request(
        "POST",
        f"/databases/{settings.notion_database_id}/query",
        json={
            "filter": {
                "property": "Created",
                "date": {"equals": _today_date()},
            }
        },
    )
    results = resp.json().get("results", [])
    return results[0] if results else None


async def create_page(entry_title: str, entry_text: str, entry_tags: list[str]) -> None:
    title = f"{_today_label()} | {entry_title}"
    await _request(
        "POST",
        "/pages",
        json={
            "parent": {"database_id": settings.notion_database_id},
            "properties": {
                "title": {"title": [{"text": {"content": title}}]},
                "Created": {"date": {"start": _today_date()}},
                "Tags": {"multi_select": _combine_tags(None, entry_tags)},
            },
            "children": [
                {
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {"rich_text": [{"text": {"content": entry_title}}]},
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": entry_text}}]},
                },
            ],
        },
        repeatable=False,
    )


async def update_page(page: dict, entry_title: str, entry_text: str, entry_tags: list[str]) -> None:
    page_id = page["id"]
    new_title = f"{_extract_title(page)}, {entry_title}"
    # Setting properties is an absolute write, so replaying it is harmless.
    await _request(
        "PATCH",
        f"/pages/{page_id}",
        json={
            "properties": {
                "title": {"title": [{"text": {"content": new_title}}]},
                "Tags": {"multi_select": _combine_tags(page, entry_tags)},
            }
        },
    )
    await _request(
        "PATCH",
        f"/blocks/{page_id}/children",
        json={
            "children": [
                {"object": "block", "type": "divider", "divider": {}},
                {
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {"rich_text": [{"text": {"content": entry_title}}]},
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": entry_text}}]},
                },
            ]
        },
        repeatable=False,
    )


async def get_week_pages() -> list[dict]:
    """Returns all diary pages created in the last 7 days, oldest first."""
    tz = zoneinfo.ZoneInfo(settings.timezone)
    today = datetime.now(tz).date()
    week_ago = today - timedelta(days=6)
    resp = await _request(
        "POST",
        f"/databases/{settings.notion_database_id}/query",
        json={
            "filter": {
                "and": [
                    {"property": "Created", "date": {"on_or_after": week_ago.isoformat()}},
                    {"property": "Created", "date": {"on_or_before": today.isoformat()}},
                ]
            },
            "sorts": [{"property": "Created", "direction": "ascending"}],
        },
    )
    return resp.json().get("results", [])


async def save_entry(entry_title: str, entry_text: str, entry_tags: list[str]) -> bool:
    """Creates or updates today's diary page. Returns True if updated, False if created."""
    page = await get_today_page()
    if page:
        await update_page(page, entry_title, entry_text, entry_tags)
        return True
    else:
        await create_page(entry_title, entry_text, entry_tags)
        return False


async def get_page_blocks(page_id: str) -> list[dict]:
    """Returns the top-level blocks of a page."""
    resp = await _request("GET", f"/blocks/{page_id}/children")
    return resp.json().get("results", [])
