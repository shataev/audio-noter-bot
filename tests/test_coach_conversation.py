"""One coach turn end to end, against a client that records instead of sending.

There are no credentials on this machine, so what is checked here is the request
body and what is done with the reply: that the rules reach the model with the ids
it has to address them by, that a reply which asks for a rule gets one applied,
and above all that a reply which asks for nothing leaves the stored list exactly
as it was — which is almost every reply, and the one case where a bug would go
unnoticed for months.

The same function answers the first press and the fifth reply. The test for that
is deliberately about the *first* turn: a rule the model writes down in its very
first answer is the case a separate first-turn path would quietly not support.
"""

import os

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import pytest

from services.ai import Completion
from services.coach import conversation, prompts
from services.coach.memory import Fact, MemoryList
from services.coach.threads import Turn

ANSWER = "Ты третий раз за неделю пишешь одно и то же."


class FakeChat:
    """Records the request and returns a fixed reply. Never reaches the network."""

    def __init__(self, text=ANSWER):
        self.text = text
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return Completion(text=self.text, finish_reason="stop")

    @property
    def call(self):
        assert len(self.calls) == 1, f"expected one call, got {len(self.calls)}"
        return self.calls[0]


def rule(fact_id, text):
    return Fact(
        id=fact_id,
        text=text,
        kind="rule",
        created_at="2026-09-01T10:00:00+00:00",
        updated_at="2026-09-01T10:00:00+00:00",
    )


def rules(*facts):
    return MemoryList(facts=tuple(facts), next_id=len(facts) + 1)


async def ask(client, *, mode="roast", items=None, turns=None):
    return await conversation.answer(
        mode=prompts.mode_for(mode),
        rules=items if items is not None else MemoryList(),
        turns=turns if turns is not None else [Turn(role="user", text="сегодня опять ничего")],
        model="test-model",
        client=client,
    )


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_coach_is_the_role_that_is_paid_to_think():
    """The formatter and the recaps pass effort=None precisely so this one can be high."""
    client = FakeChat()

    await ask(client)

    assert client.call["effort"] == "high"


@pytest.mark.asyncio
async def test_the_model_is_the_one_it_was_given_rather_than_a_hardcoded_id():
    client = FakeChat()

    await ask(client)

    assert client.call["model"] == "test-model"


@pytest.mark.asyncio
async def test_the_budget_leaves_room_for_thinking_as_well_as_the_answer():
    """On Anthropic the thinking comes out of this ceiling, so a paragraph-sized one
    can come back empty."""
    client = FakeChat()

    await ask(client)

    assert client.call["max_output_tokens"] >= 4096


@pytest.mark.asyncio
async def test_no_json_format_is_asked_for():
    """The answer is prose the owner reads; the rules block rides inside it."""
    client = FakeChat()

    await ask(client)

    assert not client.call.get("json_schema")
    assert not client.call.get("require_json")


@pytest.mark.asyncio
async def test_the_mode_decides_the_persona(monkeypatch):
    monkeypatch.setenv("COACH_PROMPT_SUPPORT", "ПОДДЕРЖКА")
    client = FakeChat()

    await ask(client, mode="support")

    assert "ПОДДЕРЖКА" in client.call["system"]


@pytest.mark.asyncio
async def test_the_rules_reach_the_model_with_their_ids():
    client = FakeChat()

    await ask(client, items=rules(rule("2", "не задавать вопросов")))

    assert "[2] не задавать вопросов" in client.call["system"]


@pytest.mark.asyncio
async def test_the_rules_come_after_the_persona_in_the_request(monkeypatch):
    """Checked on the request itself, not just on the prompt builder."""
    monkeypatch.setenv("COACH_PROMPT_ROAST", "ПЕРСОНА")
    client = FakeChat()

    await ask(client, items=rules(rule("2", "не задавать вопросов")))

    system = client.call["system"]
    assert system.index("ПЕРСОНА") < system.index("не задавать вопросов")


@pytest.mark.asyncio
async def test_the_whole_conversation_is_sent_in_order():
    client = FakeChat()

    await ask(
        client,
        turns=[
            Turn(role="user", text="запись"),
            Turn(role="assistant", text="ответ"),
            Turn(role="user", text="а если нет"),
        ],
    )

    assert [(m.role, m.content) for m in client.call["messages"]] == [
        ("user", "запись"),
        ("assistant", "ответ"),
        ("user", "а если нет"),
    ]


# --------------------------------------------------------------------------- #
# The reply
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_ordinary_reply_changes_nothing():
    """No marker, no write. This is what keeps an ordinary answer free."""
    before = rules(rule("1", "не ругаться"))

    answer = await ask(FakeChat(), items=before)

    assert answer.text == ANSWER
    assert answer.changed is None
    assert answer.rules is before


@pytest.mark.asyncio
async def test_a_rule_can_be_written_down_in_the_very_first_answer():
    """The first rule the bot ever learns arrives in the first conversation."""
    client = FakeChat(
        ANSWER + '\n<<<RULES>>>{"ops": [{"action": "create", "text": "не начинать с приветствия"}]}'
    )

    answer = await ask(client)

    assert answer.text == ANSWER
    assert [fact.text for fact in answer.rules.facts] == ["не начинать с приветствия"]


@pytest.mark.asyncio
async def test_a_new_rule_is_filed_as_a_rule_rather_than_as_a_profile_fact():
    """The kind vocabulary is a kwarg, and the default is the profile's."""
    client = FakeChat(ANSWER + '\n<<<RULES>>>{"ops": [{"action": "create", "text": "короче"}]}')

    answer = await ask(client)

    assert [fact.kind for fact in answer.rules.facts] == ["rule"]


@pytest.mark.asyncio
async def test_a_rule_can_be_deleted_by_the_id_the_model_was_shown():
    client = FakeChat(ANSWER + '\n<<<RULES>>>{"ops": [{"action": "delete", "id": "2"}]}')

    answer = await ask(client, items=rules(rule("1", "не ругаться"), rule("2", "не спрашивать")))

    assert [fact.id for fact in answer.rules.facts] == ["1"]
    assert [fact.id for fact in answer.changed.deleted] == ["2"]


@pytest.mark.asyncio
async def test_a_rule_can_be_reworded_without_losing_its_id():
    client = FakeChat(
        ANSWER + '\n<<<RULES>>>{"ops": [{"action": "modify", "id": "1", "text": "не материться"}]}'
    )

    answer = await ask(client, items=rules(rule("1", "не ругаться")))

    assert [(f.id, f.text) for f in answer.rules.facts] == [("1", "не материться")]


@pytest.mark.asyncio
async def test_the_marker_never_reaches_the_visible_answer():
    client = FakeChat(f'{ANSWER}\n\n```json\n<<<RULES>>>{{"ops": []}}\n```')

    answer = await ask(client)

    assert answer.text == ANSWER


@pytest.mark.asyncio
async def test_an_unparseable_block_writes_nothing_and_still_delivers_the_answer():
    before = rules(rule("1", "не ругаться"))
    client = FakeChat(ANSWER + '\n<<<RULES>>>{"ops": [{"action": ')

    answer = await ask(client, items=before)

    assert answer.text == ANSWER
    assert answer.changed is None
    assert answer.rules is before


@pytest.mark.asyncio
async def test_an_operation_against_an_id_that_is_gone_is_skipped_not_guessed_at():
    before = rules(rule("1", "не ругаться"))
    client = FakeChat(ANSWER + '\n<<<RULES>>>{"ops": [{"action": "delete", "id": "9"}]}')

    answer = await ask(client, items=before)

    assert answer.rules.facts == before.facts
    assert [op.reason for op in answer.changed.skipped] == ["unknown-id"]


@pytest.mark.asyncio
async def test_a_reply_that_is_nothing_but_a_block_leaves_no_text_to_send():
    """The caller has to notice and say something rather than send an empty message."""
    client = FakeChat('<<<RULES>>>{"ops": [{"action": "create", "text": "короче"}]}')

    answer = await ask(client)

    assert answer.text == ""
    assert [fact.text for fact in answer.rules.facts] == ["короче"]


@pytest.mark.asyncio
async def test_a_model_that_returned_no_text_at_all_is_not_an_exception():
    """`complete` returns "" rather than None when a reply carried no text block."""
    answer = await ask(FakeChat(""))

    assert answer.text == ""
    assert answer.changed is None
