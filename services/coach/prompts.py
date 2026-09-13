"""What the coach is told, and how it edits its own instructions.

Three modes, one pipeline. They differ only in the paragraph describing who is
answering — everything else about the request is identical, which is what keeps
"add a fourth mode" a two-line change rather than a second code path.

Two things here are less obvious than the prompt text.

*The personas in this file are neutral defaults.* The owner's real ones are
personal and this repository is public, so the live text comes from the
environment — ``COACH_PROMPT_ROAST`` and the two beside it — and what is
committed here is written to be usable by a stranger who has just cloned this.
That is the privacy boundary for the whole feature, and it costs one
``os.getenv`` per mode.

*The rules go last, and they say so.* The owner's standing instructions are
appended after the persona under a header stating that they outrank it. That
ordering is the point of the feature: it is how the bot's behaviour is changed by
talking to it rather than by editing this file. A model that reads "be blunt"
first and "never swear" second resolves the conflict the way the owner wants; the
other order resolves it the way the prompt author wanted, which is worse.

The model edits that list from inside its own reply — see
:func:`split_rules_update` for the marker protocol and for why it is parsed as
forgivingly as it is.
"""

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .memory import Fact

logger = logging.getLogger(__name__)

# What separates the answer the owner reads from the operations the model wants
# applied to its own rules. Deliberately something no human writes by accident
# and no Telegram parse mode gives a meaning to.
RULES_MARKER = "<<<RULES>>>"

# A line that is nothing but a code fence. Models fence a JSON block whether or
# not they were asked to, and the opening fence lands in the visible half of the
# answer while the closing one lands in the ignored half.
_FENCE_LINE = re.compile(r"^`{3,}[A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class Mode:
    """One persona: what the button says, where its live text comes from.

    ``key`` is what the callback data carries and what a thread records, so it is
    written into persisted state — renaming one silently orphans every
    conversation that was opened in it.
    """

    key: str
    label: str
    variable: str
    default: str


_ROAST_DEFAULT = """Ты — прямой собеседник, который читает запись из чужого дневника.

Скажи то, вокруг чего запись ходит кругами и чего автор в ней не называет.
Если за этим виден повторяющийся сценарий — назови его. Не смягчай, не подбадривай
и не ищи хорошее там, где его нет. Резкость здесь нужна ради точности, а не сама
по себе: обидеть легко, попасть в цель трудно, и ценно только второе."""

_BREAKDOWN_DEFAULT = """Ты — спокойный собеседник, который читает запись из чужого дневника.

Разбери её по частям: что произошло, что этим двигало, что из этого уже повторялось.
Опирайся на то, что в записи действительно есть, а не на то, что можно было бы
предположить. Без оценок, без утешения и без советов, если о них не просили."""

_SUPPORT_DEFAULT = """Ты — тёплый собеседник, который читает запись из чужого дневника.

Сегодня автору не нужно, чтобы его разбирали на части. Назови своими словами то,
что ему тяжело, и признай, что это действительно тяжело. Не преуменьшай, не
переводи в урок и не обещай, что всё наладится. Достаточно того, что его услышали
и что он в этом не один."""


# Order matters twice over: it is the order of the buttons on the preview, and
# the first mode is the one the owner reaches for most.
MODES = (
    Mode(
        key="roast",
        label="🔥 Разъёб",
        variable="COACH_PROMPT_ROAST",
        default=_ROAST_DEFAULT,
    ),
    Mode(
        key="breakdown",
        label="🧭 Разбор",
        variable="COACH_PROMPT_BREAKDOWN",
        default=_BREAKDOWN_DEFAULT,
    ),
    Mode(
        key="support",
        label="🫂 Поддержка",
        variable="COACH_PROMPT_SUPPORT",
        default=_SUPPORT_DEFAULT,
    ),
)


# The shape of the answer, which is the same whoever is giving it. Every line of
# this is a habit models fall into unprompted: a summary of what was just said, a
# question to keep the conversation going, an announcement of what is about to be
# analysed, and bullet points nobody asked for.
_SHARED = """Формат ответа — одинаковый для любой роли:
— несколько предложений, один абзац, одна главная мысль;
— не пересказывай автору его же запись: он знает, что в ней;
— не заканчивай вопросом, чтобы вытянуть продолжение разговора;
— не объявляй, что собираешься сделать, — сразу говори по сути;
— никакого markdown: ни звёздочек, ни заголовков, ни списков."""


# Built by concatenation rather than as one literal so that the marker in the
# instructions is by construction the marker the parser looks for.
_RULES_PROTOCOL = (
    "Автор может прямо в разговоре попросить изменить твоё поведение — например\n"
    "«не начинай с приветствия» или «забудь правило 2». Такие просьбы ты\n"
    "записываешь сам. Если и только если такая просьба есть в последнем сообщении,\n"
    "добавь самой последней строкой ответа маркер и сразу за ним один JSON-объект:\n\n"
    + RULES_MARKER
    + '{"ops": [{"action": "create", "text": "не начинать ответ с приветствия"}]}\n\n'
    "Операции адресуются по id из списка ниже: «create» с полем «text», «modify» с\n"
    "«id» и «text», «delete» с «id». Маркер и всё, что идёт после него, автор не\n"
    "увидит — не упоминай их в самом ответе и не пиши ничего после JSON. Если такой\n"
    "просьбы не было, маркера в ответе быть не должно вовсе."
)


_RULES_HEADER = """Правила поведения, которые задал сам автор. Они важнее всего, что
написано выше: если правило противоречит описанию роли или формату ответа,
выполняется правило."""

_NO_RULES = "Автор пока не задавал правил поведения — список пуст."


def mode_for(key: str) -> Mode | None:
    """The mode a callback or a stored thread names, or ``None`` for one that is gone."""
    return next((mode for mode in MODES if mode.key == key), None)


def persona(mode: Mode) -> str:
    """The live text for one mode: the environment's, or the neutral default.

    A variable that is set but empty counts as unset. The alternative is a typo in
    a deploy silently handing the model an empty persona, which produces a reply
    that is confidently nothing like the mode that was pressed.
    """
    return os.getenv(mode.variable, "").strip() or mode.default


def rule_line(fact: Fact) -> str:
    """One rule as both the model and the owner see it: its id, then the rule.

    The same rendering in the prompt and in ``/rules`` on purpose — the number the
    owner reads out ("забудь правило 2") has to be the number the model can act on.
    """
    return f"[{fact.id}] {fact.text}"


def rules_block(rules: Sequence[Fact]) -> str:
    if not rules:
        return _NO_RULES
    return "\n".join([_RULES_HEADER, "", *(rule_line(fact) for fact in rules)])


def system_prompt(mode: Mode, rules: Sequence[Fact] = ()) -> str:
    """Persona, shape, the protocol for editing the rules, then the rules."""
    return "\n\n".join([persona(mode), _SHARED, _RULES_PROTOCOL, rules_block(rules)])


def entry_message(title: str, text: str) -> str:
    """The diary entry as the coach is handed it, on the first turn of a thread."""
    return f"Запись из дневника.\n\nЗаголовок: {title}\n\n{text}"


def _visible(text: str) -> str:
    """The half of the answer the owner sees, with a stray fence taken off.

    A model that fences the rules block opens the fence *before* the marker, so
    the opening backticks are the last thing in the visible half. They are not the
    owner's problem.
    """
    lines = text.strip().splitlines()
    while lines and _FENCE_LINE.match(lines[0].strip()):
        lines.pop(0)
    while lines and _FENCE_LINE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).strip()


def split_rules_update(answer: str) -> tuple[str, list | None]:
    """Split a reply into what the owner reads and what the rules list should do.

    Returns the visible text and either a list of operations for
    ``memory.apply_ops`` or ``None``, which means *change nothing*. No marker is
    ``None``, and that is the overwhelmingly common case: an ordinary reply costs
    nothing extra and cannot touch local state by accident.

    Three details are load-bearing.

    *The cut is at the first marker.* A model that emits two of them has produced
    one block of operations and some debris, and the debris must not reach the
    owner — so everything from the first marker onwards is gone from the visible
    half whether or not it parses.

    *The JSON is found rather than assumed to start immediately.* Fences,
    newlines and a stray word all turn up between the marker and the object.

    *It is read with ``raw_decode``, not ``json.loads``.* ``loads`` fails on
    trailing junk, and trailing junk — a closing fence, a sign-off line — is
    exactly what a fenced block leaves behind. ``raw_decode`` reads one value and
    ignores the rest, which is the behaviour wanted here.

    Anything unparseable is dropped and logged, and the visible text is still
    delivered: a malformed block costs the owner a rule he has to repeat, never
    the answer he was waiting for.
    """
    marker = answer.find(RULES_MARKER)
    if marker == -1:
        return _visible(answer), None

    visible = _visible(answer[:marker])
    tail = answer[marker + len(RULES_MARKER) :]

    starts = [at for at in (tail.find("{"), tail.find("[")) if at != -1]
    if not starts:
        logger.info("coach: a rules marker with nothing after it, ignored")
        return visible, None

    try:
        payload, _ = json.JSONDecoder().raw_decode(tail[min(starts) :])
    except ValueError:
        logger.info("coach: the rules block did not parse as JSON, ignored")
        return visible, None

    ops = payload.get("ops") if isinstance(payload, dict) else payload
    if not isinstance(ops, list):
        logger.info("coach: the rules block carried no list of operations, ignored")
        return visible, None

    logger.info("coach: the reply carries %d rule operation(s)", len(ops))
    return visible, ops
