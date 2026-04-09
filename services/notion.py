from datetime import datetime, timedelta
import zoneinfo
import httpx
from config import settings

API = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {settings.notion_token}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

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
    async with httpx.AsyncClient() as http:
        resp = await http.post(
            f"{API}/databases/{settings.notion_database_id}/query",
            headers=HEADERS,
            json={
                "filter": {
                    "property": "Created",
                    "date": {"equals": _today_date()},
                }
            },
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else None


async def create_page(entry_title: str, entry_text: str, entry_tags: list[str]) -> None:
    title = f"{_today_label()} | {entry_title}"
    async with httpx.AsyncClient() as http:
        resp = await http.post(
            f"{API}/pages",
            headers=HEADERS,
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
        )
        resp.raise_for_status()


async def update_page(page: dict, entry_title: str, entry_text: str, entry_tags: list[str]) -> None:
    page_id = page["id"]
    new_title = f"{_extract_title(page)}, {entry_title}"
    async with httpx.AsyncClient() as http:
        resp = await http.patch(
            f"{API}/pages/{page_id}",
            headers=HEADERS,
            json={
                "properties": {
                    "title": {"title": [{"text": {"content": new_title}}]},
                    "Tags": {"multi_select": _combine_tags(page, entry_tags)},
                }
            },
        )
        if not resp.is_success:
            raise RuntimeError(f"Notion PATCH pages error {resp.status_code}: {resp.text}")
        resp.raise_for_status()

        resp = await http.patch(
            f"{API}/blocks/{page_id}/children",
            headers=HEADERS,
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
        )
        resp.raise_for_status()


async def save_entry(entry_title: str, entry_text: str, entry_tags: list[str]) -> bool:
    """Creates or updates today's diary page. Returns True if updated, False if created."""
    page = await get_today_page()
    if page:
        await update_page(page, entry_title, entry_text, entry_tags)
        return True
    else:
        await create_page(entry_title, entry_text, entry_tags)
        return False
