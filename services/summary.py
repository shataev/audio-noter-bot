"""The daily recap, the weekly report, and the one reader both of them share.

Three jobs read the diary back out of Notion: the 21:00 daily summary, the
Sunday weekly report, and — since the retrospective profile rebuild and the
weekly coach session — two passes that want the day broken back into the
entries it was written as. All of them go through the block walk in this
module, so that "how a page is turned back into text" is one piece of code
rather than one per caller that drifts from the others.
"""

import logging
from dataclasses import dataclass
from datetime import date

from config import settings
from services import notion
from services.ai import Message, create_chat_client
from services.notion import get_page_blocks, get_today_page, get_week_pages

logger = logging.getLogger(__name__)

chat = create_chat_client()
# The SDK object underneath it — the one that actually puts the request on the
# wire — under the name this module has always kept its transport by.
openai_client = chat.sdk

# Neither recap reasons. They restate a day, or a week, that is already written
# down, which is the same argument that keeps them on a cheap model; and on a
# provider where thinking is spent out of the same budget as the answer, leaving
# it on would take tokens from a summary that is sent to the user as it is.
NO_REASONING = None

SUMMARY_PROMPT = """You are helping the user reflect on their day.
Below are the diary entries they recorded throughout the day.
Write a concise, warm daily summary in Russian (2-4 sentences):
highlight the key events, mood, and any notable thoughts.
Do not use bullet points — write as a short paragraph."""


def _block_text(block: dict) -> str:
    """The plain text of one top-level block, empty for a block that carries none."""
    block_type = block.get("type")
    rich_text = block.get(block_type, {}).get("rich_text", [])
    return "".join(item.get("plain_text", "") for item in rich_text)


async def _fetch_page_text(page_id: str) -> str:
    """Fetches all text blocks from a Notion page and returns them as plain text."""
    blocks = await get_page_blocks(page_id)

    lines = []
    for block in blocks:
        text = _block_text(block)
        if text:
            lines.append(text)

    return "\n\n".join(lines)


WEEKLY_PROMPT = """You are helping the user reflect on their week.
Below are all diary entries from the past 7 days.
Entries marked with ⭐ were highlighted by the user as personally significant.

Write a weekly highlight report in Russian:
1. A warm narrative paragraph (3-5 sentences) capturing the spirit of the week.
2. A list of 5-7 highlights — include all ⭐-marked entries first, then add any others you find significant. Format each as a short bullet point.

Do not use headers. Write naturally and warmly, as if summarizing a meaningful week to a friend."""


async def generate_weekly_report() -> str | None:
    """Generates a GPT weekly highlight report. Returns None if no pages found."""
    pages = await get_week_pages()
    if not pages:
        return None

    sections = []
    for page in pages:
        title_prop = page["properties"].get("title") or page["properties"].get("Name")
        page_title = "".join(p["plain_text"] for p in title_prop.get("title", []))
        page_text = await _fetch_page_text(page["id"])
        if page_text.strip():
            sections.append(f"### {page_title}\n{page_text}")

    if not sections:
        return None

    full_text = "\n\n".join(sections)
    completion = await chat.complete(
        model=settings.summary_model,
        system=WEEKLY_PROMPT,
        messages=[Message(role="user", content=full_text)],
        max_output_tokens=1024,
        effort=NO_REASONING,
    )
    return completion.text


async def generate_daily_summary() -> str | None:
    """Generates a GPT summary of today's diary page. Returns None if no page exists."""
    page = await get_today_page()
    if not page:
        return None

    page_text = await _fetch_page_text(page["id"])
    if not page_text.strip():
        return None

    completion = await chat.complete(
        model=settings.summary_model,
        system=SUMMARY_PROMPT,
        messages=[Message(role="user", content=page_text)],
        max_output_tokens=512,
        effort=NO_REASONING,
    )
    return completion.text


# --------------------------------------------------------------------------- #
# Reading the diary back as entries.
#
# The recaps above want a day as one lump of prose. The retrospective profile
# rebuild and the weekly coach session want the entries the day was written as,
# one at a time, because that is the unit the profile extraction was built for.
#
# One page per day with entries appended is this bot's data model, and this is
# the only place that knows how to undo it: `_entry_blocks` in services/notion.py
# writes a divider, a heading_3 with the entry's title, then the paragraphs, so
# a heading_3 is where one entry ends and the next begins.
# --------------------------------------------------------------------------- #

ENTRY_HEADING = "heading_3"


@dataclass(frozen=True)
class DiaryEntry:
    """One entry as it was written, and the day whose page carries it."""

    day: date | None
    title: str
    text: str

    @property
    def source(self) -> str:
        """Where the entry came from, in the form the profile records as a source."""
        stamp = self.day.isoformat() if self.day else "?"
        return f"{stamp} · {self.title}" if self.title else stamp


def split_entries(blocks: list[dict], day: date | None = None) -> list[DiaryEntry]:
    """Splits one day's blocks back into the entries that were appended to it.

    Text written above the first heading — a page started by hand, or one from
    before the bot titled its entries — is kept as an entry with no title rather
    than dropped: it is still the owner's diary, and the pass that reads it is
    the one that would otherwise never see it.
    """
    entries: list[DiaryEntry] = []
    title: str | None = None
    lines: list[str] = []

    def flush() -> None:
        text = "\n\n".join(lines).strip()
        if title or text:
            entries.append(DiaryEntry(day=day, title=(title or "").strip(), text=text))

    for block in blocks:
        if block.get("type") == ENTRY_HEADING:
            flush()
            title, lines = _block_text(block), []
            continue
        text = _block_text(block)
        if text:
            lines.append(text)

    flush()
    return entries


def page_day(page: dict) -> date | None:
    """The date a page is filed under, from the property the bot sets when it creates it.

    The title carries the day too, but only as "9 мая" — no year, and in Russian.
    The `Created` property is the one that is machine-readable, and a page whose
    property has been cleared by hand is dated `None` rather than guessed at.
    """
    created = (page.get("properties") or {}).get("Created") or {}
    start = (created.get("date") or {}).get("start")
    if not isinstance(start, str) or not start:
        return None
    try:
        return date.fromisoformat(start[:10])
    except ValueError:
        logger.warning("A diary page is dated %r, which is not a date", start[:10])
        return None


async def read_page_entries(page: dict) -> list[DiaryEntry]:
    """Every entry on one diary page, in the order they were written."""
    return split_entries(await get_page_blocks(page["id"]), page_day(page))


async def all_pages() -> list[dict]:
    """Every diary page there has ever been, oldest first.

    The same query `get_week_pages` runs with no date filter on it, and the same
    cursor loop with the same guard against a server that keeps handing back the
    one cursor. It lives here rather than in services/notion.py only because the
    retrospective rebuild is the sole caller and that file is being changed on
    another branch this week; if a second caller appears, this belongs beside
    the other queries.
    """
    body = {
        "sorts": [{"property": "Created", "direction": "ascending"}],
        "page_size": notion.MAX_PAGE_SIZE,
    }

    pages: list[dict] = []
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        payload = body if cursor is None else {**body, "start_cursor": cursor}
        resp = await notion._request(
            "POST", f"/databases/{settings.notion_database_id}/query", json=payload
        )
        data = resp.json()
        pages.extend(data.get("results", []))
        cursor = data.get("next_cursor")
        if not data.get("has_more") or not cursor or cursor in seen:
            return pages
        seen.add(cursor)
