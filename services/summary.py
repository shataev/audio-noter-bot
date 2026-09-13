from config import settings
from services.ai import Message, create_chat_client
from services.notion import get_page_blocks, get_today_page, get_week_pages

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


async def _fetch_page_text(page_id: str) -> str:
    """Fetches all text blocks from a Notion page and returns them as plain text."""
    blocks = await get_page_blocks(page_id)

    lines = []
    for block in blocks:
        block_type = block.get("type")
        rich_text = block.get(block_type, {}).get("rich_text", [])
        text = "".join(t["plain_text"] for t in rich_text)
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
