"""One extraction pass, against a client that records instead of sending.

There are no credentials on this machine, so what is checked here is the request
that would go out, what is done with the reply, and — more than either — what is
done with a reply that is broken. The profile is the one piece of state in this
project that cannot be recreated: a diary entry that fails to teach it anything
costs a day, and a truncated answer that is applied anyway costs a year. Every
malformed reply below therefore has the same expectation, which is that the
profile comes back byte for byte as it went in.

The note is tested next to the extraction because they are the same promise seen
from two sides: what the model changed is what the owner is told, and what he is
told is what he can correct.
"""

import os

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import json
import logging

import pytest

from services.ai import Completion
from services.coach import memory, profile
from services.coach.memory import Fact, MemoryList

TITLE = "Опять не позвонил"
TEXT = "Третий раз за месяц откладываю этот разговор и каждый раз нахожу причину."

# What a fact looks like once it is in the profile. Text that is a pattern rather
# than an episode, because that is what the prompt asks for and what the note has
# to be able to show.
KNOWN = "Откладывает трудные разговоры, пока они не решаются сами."
LEARNED = "Находит рациональную причину, чтобы не делать того, чего боится."


class FakeChat:
    """Records the request and returns a fixed reply. Never reaches the network."""

    def __init__(self, text="", fail=False, finish_reason="stop"):
        self.text = text
        self.fail = fail
        self.finish_reason = finish_reason
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("the provider said no")
        return Completion(text=self.text, finish_reason=self.finish_reason)

    @property
    def call(self):
        assert len(self.calls) == 1, f"expected one call, got {len(self.calls)}"
        return self.calls[0]


def fact(fact_id, text, kind="pattern"):
    return Fact(
        id=fact_id,
        text=text,
        kind=kind,
        created_at="2026-09-01T10:00:00+00:00",
        updated_at="2026-09-01T10:00:00+00:00",
    )


def stored(*facts, next_id=None):
    facts = tuple(facts)
    return MemoryList(facts=facts, next_id=next_id or len(facts) + 1)


def reply(*ops):
    return json.dumps({"ops": list(ops)}, ensure_ascii=False)


async def run(chat, items=None, **kwargs):
    return await profile.learn(
        profile=items if items is not None else stored(fact("4", KNOWN)),
        title=TITLE,
        text=TEXT,
        model="test-model",
        client=chat,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# What goes out
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_model_is_shown_the_entry_and_the_ids_it_must_address():
    chat = FakeChat(reply())

    await run(chat, stored(fact("4", KNOWN), fact("9", "Бегает по утрам", kind="body")))

    system = chat.call["system"]
    assert "[4]" in system and KNOWN in system
    assert "[9]" in system and "(body)" in system
    said = chat.call["messages"][0].content
    assert TITLE in said and TEXT in said


@pytest.mark.asyncio
async def test_an_empty_profile_says_so_rather_than_showing_an_empty_list():
    chat = FakeChat(reply())

    await run(chat, MemoryList())

    assert "пуст" in chat.call["system"]


@pytest.mark.asyncio
async def test_the_call_is_pinned_to_low_effort_and_a_schema():
    """An unpinned effort spends the budget thinking and truncates the JSON."""
    chat = FakeChat(reply())

    await run(chat)

    assert chat.call["effort"] == "low"
    assert chat.call["json_schema"] is profile.OPS_SCHEMA
    assert chat.call["model"] == "test-model"


def test_the_schema_is_one_openai_accepts_in_strict_mode():
    """Strict structured outputs: every property required, no extra properties."""
    item = profile.OPS_SCHEMA["properties"]["ops"]["items"]

    assert profile.OPS_SCHEMA["additionalProperties"] is False
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])


def test_the_prompt_gives_the_two_reasons_a_fact_may_be_deleted():
    """A model told only "keep it tidy" deletes what it does not recognise."""
    system = profile.system_prompt()

    assert "перестал быть правдой" in system
    assert "дубликат" in system
    assert "остаётся как есть" in system


def test_the_prompt_names_the_kinds_the_store_actually_has():
    system = profile.system_prompt()

    for kind in memory.PROFILE_KINDS:
        assert kind in system
    assert "phase" in system and "заменяется" in system


# --------------------------------------------------------------------------- #
# What comes back
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_operations_are_applied_by_id():
    chat = FakeChat(
        reply(
            {"action": "create", "text": LEARNED, "kind": "bias"},
            {"action": "modify", "id": "4", "text": f"{KNOWN} Особенно с родителями."},
            {"action": "delete", "id": "9", "reason": "перестало быть правдой"},
        )
    )

    learned = await run(chat, stored(fact("4", KNOWN), fact("9", "Бегает по утрам", kind="body")))

    texts = {item.id: item.text for item in learned.profile.facts}
    assert texts["4"].endswith("Особенно с родителями.")
    assert LEARNED in texts.values()
    assert "9" not in texts
    assert learned.changed is not None
    assert (len(learned.changed.created), len(learned.changed.modified)) == (1, 1)
    assert [item.id for item in learned.changed.deleted] == ["9"]


@pytest.mark.asyncio
async def test_an_operation_against_an_unknown_id_changes_nothing():
    """The model was looking at a list that id was not in. Guessing is worse."""
    before = stored(fact("4", KNOWN))
    chat = FakeChat(reply({"action": "modify", "id": "77", "text": "чужой факт"}))

    learned = await run(chat, before)

    assert learned.profile == before
    assert learned.changed is None


@pytest.mark.asyncio
async def test_the_entry_reference_is_recorded_on_what_it_taught():
    chat = FakeChat(reply({"action": "create", "text": LEARNED, "kind": "bias"}))

    learned = await run(chat, MemoryList(), source="2026-09-13 · Опять не позвонил")

    assert learned.profile.facts[0].sources == ("2026-09-13 · Опять не позвонил",)


@pytest.mark.asyncio
async def test_a_kind_the_store_does_not_have_is_filed_rather_than_dropped():
    chat = FakeChat(reply({"action": "create", "text": LEARNED, "kind": "vibes"}))

    learned = await run(chat, MemoryList())

    assert learned.profile.facts[0].kind == "other"


# --------------------------------------------------------------------------- #
# Every failure is a no-op
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("   \n", id="blank"),
        pytest.param('{"ops": [{"action": "create", "text": "Он оп', id="truncated"),
        pytest.param("Вот что я понял про него: он откладывает.", id="prose"),
        pytest.param('{"ops": {"action": "create"}}', id="ops-not-a-list"),
        pytest.param("null", id="null"),
    ],
)
async def test_a_reply_that_is_not_a_list_of_operations_leaves_the_profile_alone(text):
    before = stored(fact("4", KNOWN))

    learned = await run(FakeChat(text), before)

    assert learned.profile == before
    assert learned.changed is None


@pytest.mark.asyncio
async def test_a_call_that_raises_leaves_the_profile_alone():
    """A timeout, an outage, a 400 — the same answer as no answer at all."""
    before = stored(fact("4", KNOWN))

    learned = await run(FakeChat(fail=True), before)

    assert learned.profile == before
    assert learned.changed is None


@pytest.mark.asyncio
async def test_a_truncated_reply_does_not_apply_the_half_that_parsed():
    """A delete whose replacement never arrived is the expensive version of this."""
    before = stored(fact("4", KNOWN))
    cut = '{"ops": [{"action": "delete", "id": "4"}, {"action": "create", "text": "Он'

    learned = await run(FakeChat(cut, finish_reason="length"), before)

    assert learned.profile.facts == before.facts


@pytest.mark.asyncio
async def test_an_empty_list_of_operations_is_not_a_change():
    before = stored(fact("4", KNOWN))

    learned = await run(FakeChat(reply()), before)

    assert learned.profile == before
    assert learned.changed is None


@pytest.mark.asyncio
async def test_operations_that_all_skip_are_not_a_change():
    """Skips are worth logging and are not worth a message to the owner."""
    before = stored(fact("4", KNOWN))
    chat = FakeChat(reply({"action": "create", "text": KNOWN}, {"action": "delete", "id": "77"}))

    learned = await run(chat, before)

    assert learned.changed is None


@pytest.mark.asyncio
async def test_nothing_in_the_log_carries_the_text_of_a_fact_or_an_entry(caplog):
    chat = FakeChat(
        reply(
            {"action": "create", "text": LEARNED, "kind": "bias"},
            {"action": "delete", "id": "77"},
        )
    )

    with caplog.at_level(logging.DEBUG, logger="services.coach"):
        await run(chat)

    logged = caplog.text
    assert LEARNED not in logged
    assert KNOWN not in logged
    assert TEXT not in logged
    assert TITLE not in logged
    assert "unknown-id" in logged, "the skip itself is worth logging"


# --------------------------------------------------------------------------- #
# The note
# --------------------------------------------------------------------------- #


def applied(before, *ops):
    return memory.apply_ops(before, list(ops), kinds=memory.PROFILE_KINDS)


def test_the_note_matches_the_counts_it_was_built_from():
    before = stored(fact("4", KNOWN), fact("9", "Бегает по утрам", kind="body"))
    change = applied(
        before,
        {"action": "create", "text": LEARNED, "kind": "bias"},
        {"action": "delete", "id": "9"},
    )

    lines = profile.change_lines(before.facts, change)
    body = profile.note(about=lines)

    assert body.splitlines()[0] == profile.NOTE_HEADER
    assert profile.ABOUT_LABEL in body
    assert sum(line.startswith("+") for line in lines) == len(change.created)
    assert sum(line.startswith("−") for line in lines) == len(change.deleted)
    assert f"+ [{change.created[0].id}] {LEARNED}" in lines
    assert "− Бегает по утрам" in lines


def test_a_reworded_fact_shows_as_both_a_plus_and_a_minus():
    """A diff cannot tell this from a delete and a create; the ApplyResult can."""
    before = stored(fact("4", KNOWN))
    change = applied(before, {"action": "modify", "id": "4", "text": LEARNED})

    lines = profile.change_lines(before.facts, change)

    assert lines == [f"− {KNOWN}", f"+ [4] {LEARNED}"]


def test_a_changed_fact_carries_the_id_its_buttons_address():
    before = stored(fact("4", KNOWN))
    change = applied(before, {"action": "modify", "id": "4", "text": LEARNED})

    assert profile.change_lines(before.facts, change)[1].startswith("+ [4] ")


def test_a_note_with_nothing_in_it_is_not_a_note():
    """Silence is the common case and it has to stay silent."""
    assert profile.note() is None
    assert profile.note(about=[], rules=[]) is None


def test_both_sections_are_drawn_when_both_lists_moved():
    body = profile.note(about=["+ [1] a"], rules=["− b"])

    assert body.splitlines() == [profile.NOTE_HEADER, "About you:", "+ [1] a", "Rules:", "− b"]


def test_a_section_with_no_lines_is_not_drawn():
    body = profile.note(about=["+ [1] a"])

    assert profile.RULES_LABEL not in body
