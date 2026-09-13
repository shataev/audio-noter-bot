"""One coach turn, start to finish: ask, read the answer, update the rules.

This is the whole pipeline, and there is only one of it. The first press of a
mode button and the fifth reply in a thread run exactly the same code with a
different list of turns, which is what the brief means by "this works in any
turn": the first rule the bot ever learns has to be able to arrive in the first
conversation, and it does, because there is no separate first-turn path for it to
be missing from.

Nothing here knows about Telegram. It takes plain turns and hands back plain
text plus the new rules list; ``bot.py`` maps messages onto that and decides what
to persist. That is the seam ``tests/test_coach_isolation.py`` enforces, and it
is the reason this module can be the entry point of a coach service later.

The effort is ``high`` deliberately. This is the one role in the project that is
paid to think — the formatter and the recaps run with reasoning off precisely so
that this one can afford it.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from services.ai import Message, create_chat_client

from . import memory, prompts
from .threads import Turn

logger = logging.getLogger(__name__)

EFFORT = "high"

# Generous, and generous for a reason rather than by guess. The answer itself is
# a paragraph, but on Anthropic the thinking tokens are spent out of this same
# ceiling, and a reasoning-grade model at high effort can spend thousands of them
# before it writes a word. A budget sized for the paragraph would come back
# empty. Nothing is billed for room that is not used.
MAX_OUTPUT_TOKENS = 8192

_client = None


def _chat_client():
    """The shared client, built on first use.

    Lazily rather than at import, unlike the formatter's: this module is imported
    by ``bot.py`` whatever the configuration, and the coach is the one feature
    that is switched off when the provider has no key. Building a client for it at
    import time would be work done for a feature that is not going to run.
    """
    global _client
    if _client is None:
        _client = create_chat_client()
    return _client


@dataclass(frozen=True)
class Answer:
    """What came back: the text to show, and what it did to the rules.

    ``changed`` is ``None`` when the reply carried no rules block at all, which is
    the ordinary case. It is what tells the caller there is nothing to write —
    distinct from a block that was present and turned out to change nothing, which
    is worth logging.
    """

    text: str
    rules: memory.MemoryList
    changed: memory.ApplyResult | None = None


async def answer(
    *,
    mode: prompts.Mode,
    rules: memory.MemoryList,
    turns: Sequence[Turn],
    model: str,
    client=None,
    now: datetime | None = None,
) -> Answer:
    """Ask one question in one mode and apply whatever the reply asks of the rules.

    ``rules`` goes in and a possibly-new ``rules`` comes back; this function
    writes nothing. Persisting is the caller's, because the caller is the one that
    knows whether the answer reached the owner.
    """
    completion = await (client or _chat_client()).complete(
        model=model,
        system=prompts.system_prompt(mode, rules.facts),
        messages=[Message(role=turn.role, content=turn.text) for turn in turns],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        effort=EFFORT,
    )

    visible, ops = prompts.split_rules_update(completion.text)
    logger.info(
        "coach: %s answered in %d character(s), finish_reason=%s",
        mode.key,
        len(visible),
        completion.finish_reason,
    )

    if ops is None:
        return Answer(text=visible, rules=rules)

    applied = memory.apply_ops(rules, ops, kinds=memory.RULE_KINDS, now=now)
    logger.info(
        "coach: rules %d created, %d modified, %d deleted, %d skipped",
        len(applied.created),
        len(applied.modified),
        len(applied.deleted),
        len(applied.skipped),
    )
    return Answer(text=visible, rules=applied.items, changed=applied)
