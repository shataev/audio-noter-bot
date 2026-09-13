"""What the bot learns about its author, one saved entry at a time.

Every entry that reaches Notion is also handed to a model with the profile as it
stands, and the model answers with *operations* against the ids in it. Nothing
here is a command the owner has to remember: the whole point is that the profile
grows without being asked for, and that the bot then says out loud what it
learned, so a wrong fact can be corrected while it is still one line old.

Three decisions are worth reading before changing anything in here.

*The effort is ``low``, and that is not a saving.* This is merge-and-dedup work
against a list the model can see — there is nothing to reason about. On a
reasoning model an unpinned effort spends the output budget thinking and comes
back with JSON cut off mid-object, which this module then has to throw away in
full. Cheap and complete beats clever and truncated.

*The answer is asked for through ``json_schema``, not through a sentence at the
end of the prompt.* The provider enforces the shape, so the failure modes left
are an empty answer and a truncated one, both of which are handled below.

*Every failure is a no-op.* An empty reply, unparseable or truncated JSON, a
timeout, a provider that says no — the profile is handed back exactly as it came
in and the entry contributes nothing. A profile accumulated over a year is worth
more than any single entry's contribution to it, and the next entry will offer
the same facts again anyway.

Standard library and ``services/ai.py`` only, like everything else in this
package, and nothing here logs the text of a fact or of an entry.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from services.ai import Message, create_chat_client

from . import memory, prompts
from .memory import ApplyResult, Fact, MemoryList

logger = logging.getLogger(__name__)

# Mechanical work on a list the model is looking at. See the module docstring:
# this is correctness, not economy.
EFFORT = "low"

# Roomy for what is actually produced — a handful of one-sentence operations —
# because on a reasoning model the thinking is spent out of this same ceiling,
# and a budget sized for the JSON alone is how the answer arrives truncated.
MAX_OUTPUT_TOKENS = 4096

# The header of the note, and the two sections under it. The owner reads this
# after every entry that taught the bot something, so it is three words and a
# list rather than a report.
NOTE_HEADER = "🧠 Память обновлена"
ABOUT_LABEL = "About you:"
RULES_LABEL = "Rules:"
ADDED = "+"
REMOVED = "−"


# Every property is required and additional ones are refused, which is what
# OpenAI's strict structured outputs demand; the fields that do not apply to an
# action are sent as null. `kind` is described rather than enumerated here
# because `apply_ops` already holds the vocabulary and files an unknown kind
# under `other` — one list of kinds, in one place, rather than two that can drift.
OPS_SCHEMA = {
    "title": "profile_ops",
    "type": "object",
    "additionalProperties": False,
    "required": ["ops"],
    "properties": {
        "ops": {
            "type": "array",
            "description": "Operations against the profile. Empty when the entry taught nothing.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "id", "text", "kind", "reason"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "modify", "delete"],
                    },
                    "id": {
                        "type": ["string", "null"],
                        "description": "The id of the fact, for modify and delete. Null on create.",
                    },
                    "text": {
                        "type": ["string", "null"],
                        "description": "The fact, for create and modify. Null on delete.",
                    },
                    "kind": {
                        "type": ["string", "null"],
                        "description": "One of: " + ", ".join(memory.PROFILE_KINDS),
                    },
                    "reason": {
                        "type": ["string", "null"],
                        "description": "Why the fact is being deleted. Null otherwise.",
                    },
                },
            },
        }
    },
}


_SYSTEM = """Ты ведёшь досье на человека по его дневнику. Тебе дают одну новую запись
и список того, что о нём уже известно. Ты отвечаешь только операциями над этим списком.

Что стоит помнить: устойчивые черты характера, искажения и привычные реакции; ценности
и страхи; повторяющиеся сценарии в поведении и решениях; ключевые отношения и роль
каждого из этих людей; работа, деньги и большие цели; тело, здоровье и режим; навыки и
уровень в них; и этап жизни, в котором он сейчас находится.

Что помнить не нужно: что он сегодня ел, настроение одного дня, пересказ событий. Запись
— это повод, а не содержание. Но устойчивая закономерность за разовым случаем — это факт:
сам случай шум, закономерность сигнал.

Один факт — одно короткое предложение о человеке в третьем лице, понятное без записи, из
которой оно взялось. Если новая запись уточняет то, что уже есть в списке, — это modify
существующего факта, а не ещё один почти такой же create.

Удалять факт можно ровно по двум причинам: он перестал быть правдой, или он сливается с
дубликатом. Не потому что он кажется мелким, не потому что он не связан с сегодняшней
записью, не потому что он старый. Факт, о котором в новой записи ничего не сказано,
остаётся как есть.

У каждого факта есть вид — kind. Виды: {kinds}. Вид phase — это текущий этап жизни;
он единственный, который заменяется на новый по мере того, как этап меняется, а не
копится. Остальные накапливаются.

Если запись не добавляет ничего устойчивого — верни пустой список операций. Это обычный
случай, и он лучше, чем выдуманный факт."""

_PROFILE_HEADER = "Что уже известно. Операции modify и delete адресуются по этим id:"
_NO_PROFILE = "О человеке пока ничего не известно — список пуст."

_client = None


def _chat_client():
    """The shared client, built on first use.

    One client for the life of the process rather than one per entry, and built
    lazily for the same reason ``conversation.py`` does it: this module is
    imported whatever the configuration, and the coach is the one feature that is
    switched off when the provider has no key.
    """
    global _client
    if _client is None:
        _client = create_chat_client()
    return _client


@dataclass(frozen=True)
class Learned:
    """The profile after one pass, and what the pass did to it.

    ``changed`` is ``None`` whenever nothing was actually added, reworded or
    removed — a failed call, a reply with no operations in it, and a reply whose
    operations all turned out to be no-ops are the same thing to a caller: there
    is nothing to write and nothing to say. The distinctions between them are in
    the log, which is where they are useful.
    """

    profile: MemoryList
    changed: ApplyResult | None = None


def fact_line(fact: Fact) -> str:
    """One fact as the model is shown it: its id, its kind, then the fact.

    The id leads, here and on the note's ``+`` lines and on the buttons beside
    them, so the number the model addressed is the number the owner corrects.
    """
    return f"[{fact.id}] ({fact.kind}) {fact.text}"


def profile_block(facts: Sequence[Fact]) -> str:
    if not facts:
        return _NO_PROFILE
    return "\n".join([_PROFILE_HEADER, "", *(fact_line(fact) for fact in facts)])


def system_prompt(facts: Sequence[Fact] = ()) -> str:
    """The instructions, then the list they address."""
    return "\n\n".join([_SYSTEM.format(kinds=", ".join(memory.PROFILE_KINDS)), profile_block(facts)])


def _operations(raw: str) -> list | None:
    """The operations in a reply, or ``None`` if there are none to be had.

    Truncation is the failure this is really written for. A reasoning model that
    overruns its budget stops mid-object, and ``json.loads`` rejects that — which
    is the wanted answer: half a list of operations is not a list of operations,
    and applying the half that parsed would delete a fact whose replacement never
    arrived.
    """
    text = raw.strip()
    if not text:
        logger.info("coach profile: the model returned nothing, the profile is unchanged")
        return None

    try:
        payload = json.loads(text)
    except ValueError:
        logger.info(
            "coach profile: the answer did not parse as JSON (%d character(s)), "
            "the profile is unchanged",
            len(text),
        )
        return None

    ops = payload.get("ops") if isinstance(payload, dict) else payload
    if not isinstance(ops, list):
        logger.info("coach profile: the answer carried no list of operations, ignored")
        return None
    return ops


async def learn(
    *,
    profile: MemoryList,
    title: str,
    text: str,
    model: str,
    source: str | None = None,
    client=None,
    now: datetime | None = None,
) -> Learned:
    """Fold one saved entry into the profile. Writes nothing; raises nothing.

    ``profile`` goes in and a possibly-new ``profile`` comes back, exactly like
    ``conversation.answer`` and for the same reason: persisting it is the
    caller's, because the caller is the one holding the lock and the one that
    knows whether the profile has moved since.
    """
    try:
        completion = await (client or _chat_client()).complete(
            model=model,
            system=system_prompt(profile.facts),
            messages=[Message(role="user", content=prompts.entry_message(title, text))],
            max_output_tokens=MAX_OUTPUT_TOKENS,
            effort=EFFORT,
            json_schema=OPS_SCHEMA,
        )
    except Exception:
        # Deliberately everything: a timeout, a refusal, a provider outage and a
        # bug in the translation layer all cost this entry's contribution and
        # nothing else. The traceback is the log's problem, the profile is this
        # function's, and it is handed back untouched.
        logger.exception("coach profile: the extraction call failed, the profile is unchanged")
        return Learned(profile=profile)

    ops = _operations(completion.text)
    if ops is None:
        logger.info("coach profile: nothing applied, finish_reason=%s", completion.finish_reason)
        return Learned(profile=profile)

    applied = memory.apply_ops(
        profile,
        ops,
        source=source,
        now=now,
        kinds=memory.PROFILE_KINDS,
    )
    logger.info(
        "coach profile: %d created, %d modified, %d deleted, %d skipped (kinds: %s)",
        len(applied.created),
        len(applied.modified),
        len(applied.deleted),
        len(applied.skipped),
        ", ".join(sorted({fact.kind for fact in (*applied.created, *applied.modified)})) or "none",
    )

    if not (applied.created or applied.modified or applied.deleted):
        return Learned(profile=profile)
    return Learned(profile=applied.items, changed=applied)


def change_lines(before: Sequence[Fact], changed: ApplyResult) -> list[str]:
    """The ``+``/``−`` lines for one list's changes, as the note shows them.

    Built from the :class:`~services.coach.memory.ApplyResult` rather than by
    diffing two lists, because a diff cannot tell a reworded fact from a delete
    and a create — and those mean different things to whoever reads the note.

    A reword is both lines: the sentence that went and the sentence that came. A
    ``+`` line carries the id, because that is what the button beside it acts on;
    a ``−`` line does not, because nothing about a fact that is gone can be
    corrected.
    """
    was = {fact.id: fact.text for fact in before}

    lines = [f"{ADDED} [{fact.id}] {fact.text}" for fact in changed.created]
    for fact in changed.modified:
        old = was.get(fact.id)
        if old is not None and memory.normalise(old) != memory.normalise(fact.text):
            lines.append(f"{REMOVED} {old}")
        lines.append(f"{ADDED} [{fact.id}] {fact.text}")
    lines += [f"{REMOVED} {fact.text}" for fact in changed.deleted]
    return lines


def note(*, about: Sequence[str] = (), rules: Sequence[str] = ()) -> str | None:
    """The message posted under a saved entry, or ``None`` when there is none.

    ``None`` is the common case and the important one: most entries teach nothing
    new, and an entry that taught nothing must not produce a message. A bot that
    speaks after every recording is a bot whose notes stop being read.

    Both sections are optional and each is drawn only when it has lines. The
    profile pass never writes rules — the coach does, from inside a conversation
    — so ``rules`` is here because the note is the one place a change to either
    list is announced, and announcing them differently would be two formats to
    read instead of one.
    """
    if not about and not rules:
        return None

    lines = [NOTE_HEADER]
    if about:
        lines += [ABOUT_LABEL, *about]
    if rules:
        lines += [RULES_LABEL, *rules]
    return "\n".join(lines)
