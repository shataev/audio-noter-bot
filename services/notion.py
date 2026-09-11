import asyncio
import json
import logging
import re
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

# Notion's documented request limits. A rich text object holds at most 2000
# characters, a block at most 100 rich text objects, a children array at most
# 100 blocks, and a request body at most 500KB.
MAX_TEXT_CHARS = 2000
MAX_RICH_TEXT_PER_BLOCK = 100
MAX_CHILDREN_PER_REQUEST = 100
MAX_PAYLOAD_BYTES = 450_000
# Paginated endpoints return at most 100 results and default to the same value;
# it is sent explicitly so the page size is a decision rather than a default.
MAX_PAGE_SIZE = 100
# A page title is rich text too, so 2000 characters is the hard ceiling for the
# whole accumulated day title.
MAX_TITLE_CHARS = MAX_TEXT_CHARS
# Ceiling for one entry's name within that title. Far above the 3-5 words the
# formatter is asked for; it exists only so that a single pathological name — a
# model that ignored the instruction on garbled input, or a paragraph pasted
# into the edit flow — cannot consume the whole budget on its own.
MAX_ENTRY_TITLE_CHARS = 100
ELLIPSIS = "…"

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
            retry_after = _retry_after(response)
            wait = delay if retry_after is None else retry_after
            logger.warning(
                "Notion %s %s returned %s, retrying in %.1fs", method, path, status, wait
            )

        await asyncio.sleep(wait)
        delay = min(delay * 2, _MAX_RETRY_WAIT)

    # Not reachable: the final attempt either returns or raises above.
    raise NotionError(f"Notion {method} {path} exhausted its {attempts} attempts")


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_WHITESPACE = re.compile(r"\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _cut_point(text: str, limit: int) -> int:
    """Finds where to cut `text` so that the left piece is at most `limit` long.

    Prefers the last sentence boundary, falls back to the last whitespace, and
    only cuts mid-word when the window contains neither. Boundary whitespace
    stays on the left piece, so concatenating the pieces reproduces the input.
    A boundary in the first half of the window is ignored: it would waste most
    of a rich text object for the sake of a tidier seam.
    """
    window = text[:limit]
    floor = limit // 2
    for pattern in (_SENTENCE_END, _WHITESPACE):
        cut = 0
        for match in pattern.finditer(window):
            if match.end() >= floor:
                cut = match.end()
        if cut:
            return cut
    return limit


def _split_text(text: str, limit: int = MAX_TEXT_CHARS) -> list[str]:
    """Splits text into pieces that each fit in one rich text object."""
    pieces = []
    remaining = text
    while len(remaining) > limit:
        cut = _cut_point(remaining, limit)
        pieces.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        pieces.append(remaining)
    return pieces or [""]


def _rich_text(text: str) -> list[dict]:
    return [{"text": {"content": piece}} for piece in _split_text(text)]


def _paragraph_block(pieces: list[str]) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"text": {"content": piece}} for piece in pieces]},
    }


def _text_blocks(text: str) -> list[dict]:
    """Turns entry text into paragraph blocks that respect Notion's limits.

    A paragraph too long for one rich text object is carried by several of them
    inside the same block, which Notion renders as one continuous paragraph, so
    the split leaves no visible seam. New blocks are only started at the blank
    lines that were already there, or when a paragraph exceeds the 100 rich text
    objects a single block can hold.
    """
    blocks = []
    for paragraph in _PARAGRAPH_BREAK.split(text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        pieces = _split_text(paragraph)
        for start in range(0, len(pieces), MAX_RICH_TEXT_PER_BLOCK):
            blocks.append(_paragraph_block(pieces[start:start + MAX_RICH_TEXT_PER_BLOCK]))
    return blocks or [_paragraph_block([""])]


def _entry_blocks(entry_title: str, entry_text: str, *, divider: bool) -> list[dict]:
    """Builds the blocks for one diary entry: heading, then the text."""
    blocks = []
    if divider:
        blocks.append({"object": "block", "type": "divider", "divider": {}})
    blocks.append({
        "object": "block",
        "type": "heading_3",
        "heading_3": {"rich_text": _rich_text(entry_title)},
    })
    blocks.extend(_text_blocks(entry_text))
    return blocks


def _batch_blocks(blocks: list[dict]) -> list[list[dict]]:
    """Groups blocks into batches small enough to be one request each."""
    batches: list[list[dict]] = []
    batch: list[dict] = []
    size = 0
    for block in blocks:
        block_size = len(json.dumps(block, ensure_ascii=False).encode("utf-8"))
        too_many = len(batch) >= MAX_CHILDREN_PER_REQUEST
        too_big = size + block_size > MAX_PAYLOAD_BYTES
        if batch and (too_many or too_big):
            batches.append(batch)
            batch, size = [], 0
        batch.append(block)
        size += block_size
    if batch:
        batches.append(batch)
    return batches


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= len(ELLIPSIS):
        return text[:limit]
    return text[: limit - len(ELLIPSIS)].rstrip() + ELLIPSIS


def _split_day_title(title: str) -> tuple[str, list[str]]:
    """Splits "9 May | Entry 1, Entry 2" into its date prefix and entry titles.

    A title without the separator was not written by this bot; it is kept whole
    as the prefix so that nothing already on the page is lost.
    """
    prefix, separator, rest = title.partition(" | ")
    if not separator:
        return title, []
    entries = [part.strip() for part in rest.split(", ")]
    return prefix, [entry for entry in entries if entry and entry != ELLIPSIS]


def _compose_day_title(prefix: str, entries: list[str], limit: int = MAX_TITLE_CHARS) -> str:
    """Renders "<date> | <entry, entry, ...>" within Notion's title limit.

    The growing list is deliberate — it is the day's table of contents in the
    database view — so the format is unchanged and the list is never shortened
    for readability. Two guards keep it from being rejected outright: one name
    is capped at MAX_ENTRY_TITLE_CHARS, and only if the assembled title would
    still be too long are the oldest names dropped for an ellipsis. Ordinary use
    reaches neither: roughly 30 characters per entry means about 65 entries in a
    single day, and the title starts again at midnight.
    """
    names = [_truncate(entry.strip(), MAX_ENTRY_TITLE_CHARS) for entry in entries if entry.strip()]
    if not names:
        return _truncate(prefix, limit)

    head = f"{prefix} | "
    room = limit - len(head)
    if room <= 0:
        return _truncate(prefix, limit)

    joined = ", ".join(names)
    if len(joined) <= room:
        return head + joined

    # Last resort: keep the date prefix and the newest names, drop the oldest.
    kept = [_truncate(names[-1], room)]
    for index in range(len(names) - 2, -1, -1):
        candidate = ", ".join([names[index], *kept])
        if len(candidate) + len(ELLIPSIS) + 2 > room:
            break
        kept.insert(0, names[index])
    kept.insert(0, ELLIPSIS)
    return head + ", ".join(kept)


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
    """Returns today's diary page, or None if it has not been created yet.

    Deliberately not paginated: there should be exactly one page per date. Two
    results are asked for so that a duplicate is noticed instead of silently
    ignored, and the sort makes the choice stable — entries keep going to the
    page they have been going to all day rather than to whichever page the API
    happened to list first.
    """
    resp = await _request(
        "POST",
        f"/databases/{settings.notion_database_id}/query",
        json={
            "filter": {
                "property": "Created",
                "date": {"equals": _today_date()},
            },
            "sorts": [{"timestamp": "created_time", "direction": "ascending"}],
            "page_size": 2,
        },
    )
    results = resp.json().get("results", [])
    if len(results) > 1:
        logger.warning(
            "More than one Notion page is dated %s; appending to the oldest (%s)",
            _today_date(), results[0].get("id"),
        )
    return results[0] if results else None


async def _append_blocks(page_id: str, blocks: list[dict]) -> None:
    """Appends blocks to a page, one request per batch that fits the limits."""
    for batch in _batch_blocks(blocks):
        await _request(
            "PATCH",
            f"/blocks/{page_id}/children",
            json={"children": batch},
            repeatable=False,
        )


async def create_page(entry_title: str, entry_text: str, entry_tags: list[str]) -> None:
    title = _compose_day_title(_today_label(), [entry_title])
    batches = _batch_blocks(_entry_blocks(entry_title, entry_text, divider=False))
    resp = await _request(
        "POST",
        "/pages",
        json={
            "parent": {"database_id": settings.notion_database_id},
            "properties": {
                "title": {"title": _rich_text(title)},
                "Created": {"date": {"start": _today_date()}},
                "Tags": {"multi_select": _combine_tags(None, entry_tags)},
            },
            "children": batches[0],
        },
        repeatable=False,
    )
    if len(batches) > 1:
        page_id = resp.json()["id"]
        for batch in batches[1:]:
            await _request(
                "PATCH",
                f"/blocks/{page_id}/children",
                json={"children": batch},
                repeatable=False,
            )


async def update_page(page: dict, entry_title: str, entry_text: str, entry_tags: list[str]) -> None:
    page_id = page["id"]
    prefix, entries = _split_day_title(_extract_title(page))
    new_title = _compose_day_title(prefix, [*entries, entry_title])
    # Setting properties is an absolute write, so replaying it is harmless.
    await _request(
        "PATCH",
        f"/pages/{page_id}",
        json={
            "properties": {
                "title": {"title": _rich_text(new_title)},
                "Tags": {"multi_select": _combine_tags(page, entry_tags)},
            }
        },
    )
    await _append_blocks(page_id, _entry_blocks(entry_title, entry_text, divider=True))


async def get_week_pages() -> list[dict]:
    """Returns all diary pages created in the last 7 days, oldest first."""
    tz = zoneinfo.ZoneInfo(settings.timezone)
    today = datetime.now(tz).date()
    week_ago = today - timedelta(days=6)
    body = {
        "filter": {
            "and": [
                {"property": "Created", "date": {"on_or_after": week_ago.isoformat()}},
                {"property": "Created", "date": {"on_or_before": today.isoformat()}},
            ]
        },
        "sorts": [{"property": "Created", "direction": "ascending"}],
        "page_size": MAX_PAGE_SIZE,
    }

    pages: list[dict] = []
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        payload = body if cursor is None else {**body, "start_cursor": cursor}
        resp = await _request(
            "POST", f"/databases/{settings.notion_database_id}/query", json=payload
        )
        data = resp.json()
        pages.extend(data.get("results", []))
        cursor = data.get("next_cursor")
        # The cursor check also stops a server that keeps handing back the same
        # one from spinning this loop forever.
        if not data.get("has_more") or not cursor or cursor in seen:
            return pages
        seen.add(cursor)


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
    """Returns every top-level block of a page, following Notion's cursor.

    One response carries at most 100 blocks. A saved entry costs three blocks
    after the first, so a single request stops seeing a busy day somewhere
    around its mid-thirties entry.
    """
    blocks: list[dict] = []
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        params = {"page_size": MAX_PAGE_SIZE}
        if cursor is not None:
            params["start_cursor"] = cursor
        resp = await _request("GET", f"/blocks/{page_id}/children", params=params)
        data = resp.json()
        blocks.extend(data.get("results", []))
        cursor = data.get("next_cursor")
        if not data.get("has_more") or not cursor or cursor in seen:
            return blocks
        seen.add(cursor)
