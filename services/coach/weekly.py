"""Once a week the coach speaks first.

Everything else the coach does is a reaction: a button on a preview, a reply in
a thread, a fact learned from an entry that was just saved. This is the one part
that opens its mouth without being asked — it reads the week and the profile and
sends a single thing worth sitting with.

It is not a recap. The Sunday weekly report already exists and does that job,
and a second summary arriving the same week would train the owner to skim both.
What this sends is one observation or one question: a few sentences, and then it
is quiet for another week.

Three guard rails are the difference between a feature and a nuisance, and two
of them live in ``bot.py`` because they are about when the job runs rather than
what it says:

*Silence on an empty week.* A week with no entries gets nothing. A generated
observation about a week that was not written about is the exact thing that
makes an unprompted message feel automated.

*Never twice for the same week.* The week that was last spoken about is written
down, because a restart — and this bot restarts on every deploy — must not be
worth a second message.

*Failure is silent in the chat.* A weekly job that apologises for itself weekly
is worse than one that says nothing. Everything that goes wrong here comes back
as an empty answer and goes to the log.
"""

import json
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from services.ai import Message, create_chat_client

from . import profile as coach_profile
from .diary import Entry
from .memory import MemoryList
from .store import write_text_atomically

logger = logging.getLogger(__name__)

# The coach reasons here, as it does when it answers. This is the one message of
# the week and it is looking for something across seven days rather than
# restating one of them.
EFFORT = "high"

# Generous because on Anthropic the thinking is spent out of this same ceiling,
# and a budget sized for the few sentences that come out is how the answer
# arrives empty. See services/ai.py.
MAX_OUTPUT_TOKENS = 4096

SYSTEM = """Ты — собеседник, который читает дневник этого человека уже давно. Раз в неделю
ты пишешь ему первым, без его просьбы.

Тебе дают записи за прошедшую неделю и то, что ты о нём знаешь. Твоя задача — сказать
ровно одно: либо закономерность, которую ты заметил, либо один вопрос, с которым стоит
посидеть. Не пересказ недели: отчёт о неделе он получает отдельно, и второй ему не нужен.

Как это должно выглядеть:
— несколько предложений, не больше; одна мысль, а не список;
— по-русски, простым языком, без заголовков, списков и разметки;
— конкретно: опирайся на то, что он действительно написал, а не на общие слова;
— если это вопрос — такой, на который у тебя самого нет заготовленного ответа;
— не хвали и не утешай по инерции, но и не воспитывай. Ты не тренер, ты внимательный
  собеседник.

Если за неделю не видно ничего, о чём стоило бы заговорить, — скажи об этом коротко и
честно, одной фразой. Это лучше, чем выдуманное наблюдение."""

_WEEK_HEADER = "Записи за неделю:"

# What the state file holds. One key today; a dict rather than a bare string so
# that the next thing this job needs to remember does not need a second file.
_LAST_WEEK = "last_week"

_client = None


def _chat_client():
    """The shared client, built on first use — see ``profile.py`` for why lazily."""
    global _client
    if _client is None:
        _client = create_chat_client()
    return _client


def week_key(day: date) -> str:
    """The ISO week a diary day belongs to, as ``2026-W37``.

    ISO rather than "the date of the Sunday it was sent", so that changing the
    day or the hour the job runs at does not silently make it eligible to speak
    about the same week twice.
    """
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def last_spoken(path: str | Path) -> str | None:
    """The week the coach last opened a conversation about, or ``None``.

    Total on purpose: a file that is missing, empty, truncated or not JSON at
    all all mean "nothing recorded". The cost of getting this wrong is one extra
    message, and refusing to run the job because a state file is damaged is the
    more expensive failure.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        body = json.loads(raw)
    except ValueError:
        logger.warning("coach weekly: %s is not JSON; treating the week as unspoken", path)
        return None
    if not isinstance(body, dict):
        return None
    value = body.get(_LAST_WEEK)
    return value if isinstance(value, str) and value else None


def remember(path: str | Path, week: str) -> None:
    """Write down that this week has been spoken about. Raises on a failed write."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomically(target, json.dumps({_LAST_WEEK: week}, ensure_ascii=False, indent=2))


def digest(entries: Sequence[Entry]) -> str:
    """The week as the coach is handed it: every entry, oldest first, under its source."""
    blocks = [f"{entry.source}\n{entry.text}".strip() for entry in entries]
    return "\n\n".join([_WEEK_HEADER, *blocks])


async def session(
    *,
    profile: MemoryList,
    entries: Sequence[Entry],
    model: str,
    client=None,
) -> str:
    """The one message, or ``""`` when there is nothing to send.

    Writes nothing and raises nothing. An empty string covers every way this can
    fail — no entries, a provider that is down, a model that answered with
    whitespace — because to the caller they are the same thing: send no message,
    and the reason is in the log.
    """
    if not entries:
        logger.info("coach weekly: no entries this week, saying nothing")
        return ""

    body = digest(entries)
    try:
        completion = await (client or _chat_client()).complete(
            model=model,
            system="\n\n".join([SYSTEM, coach_profile.profile_block(profile.facts)]),
            messages=[Message(role="user", content=body)],
            max_output_tokens=MAX_OUTPUT_TOKENS,
            effort=EFFORT,
        )
    except Exception:
        logger.exception("coach weekly: the call failed, no message this week")
        return ""

    text = completion.text.strip()
    if not text:
        logger.warning(
            "coach weekly: the model returned no text (finish_reason=%s), no message this week",
            completion.finish_reason,
        )
    return text
