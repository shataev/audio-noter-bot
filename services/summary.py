import json
import httpx
import openai
from config import settings
from services.notion import API, HEADERS, get_today_page

openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

SUMMARY_PROMPT = """You are helping the user reflect on their day.
Below are the diary entries they recorded throughout the day.
Write a concise, warm daily summary in Russian (2-4 sentences):
highlight the key events, mood, and any notable thoughts.
Do not use bullet points — write as a short paragraph."""


async def _fetch_page_text(page_id: str) -> str:
    """Fetches all text blocks from a Notion page and returns them as plain text."""
    async with httpx.AsyncClient() as http:
        resp = await http.get(
            f"{API}/blocks/{page_id}/children",
            headers=HEADERS,
        )
        resp.raise_for_status()
        blocks = resp.json().get("results", [])

    lines = []
    for block in blocks:
        block_type = block.get("type")
        rich_text = block.get(block_type, {}).get("rich_text", [])
        text = "".join(t["plain_text"] for t in rich_text)
        if text:
            lines.append(text)

    return "\n\n".join(lines)


async def generate_daily_summary() -> str | None:
    """Generates a GPT summary of today's diary page. Returns None if no page exists."""
    page = await get_today_page()
    if not page:
        return None

    page_text = await _fetch_page_text(page["id"])
    if not page_text.strip():
        return None

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=512,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": page_text},
        ],
    )
    return response.choices[0].message.content
