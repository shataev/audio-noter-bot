"""The chat client translates one request onto two different APIs.

The value of these tests is in the request bodies. Nothing here can be checked
against a live API — there are no credentials on this machine — so what they
assert is that the kwargs handed to each SDK are the ones that API documents,
and in particular that the parameters the other provider would have rejected are
not among them.
"""

# config.py reads os.environ at import time and raises KeyError on a missing
# value, so the stubs have to be in place before anything from the project is
# imported. setdefault throughout, so a real environment always wins.
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import types

import pytest

import config
from services import ai

ENTRY_SCHEMA = {
    "title": "diary_entry",
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def _openai_reply(text="Ответ"):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


def _anthropic_reply(text="Ответ"):
    """A reply that thought: a thinking block, then the answer."""
    return types.SimpleNamespace(
        content=[
            types.SimpleNamespace(type="thinking", thinking="рассуждение про себя"),
            types.SimpleNamespace(type="text", text=text),
        ],
        stop_reason="end_turn",
        usage=types.SimpleNamespace(input_tokens=11, output_tokens=7),
    )


def _recorder(reply):
    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        return reply

    return calls, create


def _openai_client(reply=None):
    """An OpenAI client whose SDK records the request instead of sending it."""
    client = ai.OpenAIChatClient(api_key="test-key")
    calls, create = _recorder(reply or _openai_reply())
    client.sdk = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    return client, calls


def _anthropic_client(reply=None):
    client = ai.AnthropicChatClient(api_key="test-anthropic")
    calls, create = _recorder(reply or _anthropic_reply())
    client.sdk = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    return client, calls


ASK = {
    "system": "Ты помогаешь вести дневник.",
    "messages": [ai.Message(role="user", content="сегодня был длинный день")],
    "max_output_tokens": 512,
}


@pytest.mark.asyncio
async def test_an_anthropic_request_carries_none_of_openais_parameters():
    """The single thing most likely to break in production, and silently.

    `reasoning_effort` is OpenAI's parameter and does not exist here.
    `budget_tokens` is how thinking used to be asked for and is removed from
    current models. `temperature` and `top_p` are removed too. All four are a
    400 from the API rather than something ignored, and none of them would be
    noticed on this machine, where nothing can call the API at all.
    """
    client, calls = _anthropic_client()

    await client.complete(model="claude-opus-5", effort="high", **ASK)

    sent = calls[0]
    assert "reasoning_effort" not in sent
    assert "temperature" not in sent
    assert "top_p" not in sent
    assert "budget_tokens" not in sent
    assert "budget_tokens" not in sent["thinking"]


@pytest.mark.asyncio
async def test_effort_reaches_anthropic_as_adaptive_thinking():
    client, calls = _anthropic_client()

    await client.complete(model="claude-opus-5", effort="high", **ASK)

    sent = calls[0]
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_no_effort_switches_anthropic_thinking_off():
    """Thinking is spent out of max_tokens, so leaving it on is not free.

    A budget sized for the reply, spent mostly on reasoning, comes back with no
    text block in it at all — not an error, an empty answer. A role that has no
    use for reasoning has to switch it off rather than leave it to the default.
    """
    client, calls = _anthropic_client()

    await client.complete(model="claude-opus-5", effort=None, **ASK)

    sent = calls[0]
    assert sent["thinking"] == {"type": "disabled"}
    assert "output_config" not in sent, "no effort and no format is nothing to configure"


@pytest.mark.asyncio
async def test_no_effort_sends_no_reasoning_effort_even_to_a_model_that_reasons():
    """There is no one value meaning "do not reason" across the OpenAI models,
    so the parameter is omitted rather than guessed at."""
    client, calls = _openai_client()

    await client.complete(model="gpt-5", effort=None, **ASK)

    assert "reasoning_effort" not in calls[0]


@pytest.mark.asyncio
async def test_a_schema_survives_thinking_being_switched_off():
    """The two travel in the same parameter on Anthropic; one must not lose the other."""
    client, calls = _anthropic_client()

    await client.complete(model="claude-opus-5", effort=None, json_schema=ENTRY_SCHEMA, **ASK)

    sent = calls[0]
    assert sent["thinking"] == {"type": "disabled"}
    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": ENTRY_SCHEMA}}


@pytest.mark.asyncio
async def test_the_anthropic_model_id_is_sent_exactly_as_configured():
    """Anthropic ids are complete as written and never carry a date suffix."""
    client, calls = _anthropic_client()

    await client.complete(model="claude-opus-5", **ASK)

    assert calls[0]["model"] == "claude-opus-5"


@pytest.mark.asyncio
async def test_the_system_prompt_is_not_a_message_on_anthropic():
    """It is its own parameter there, and a message on OpenAI. Same prompt."""
    client, calls = _anthropic_client()

    await client.complete(model="claude-opus-5", **ASK)

    sent = calls[0]
    assert sent["system"] == ASK["system"]
    assert sent["messages"] == [{"role": "user", "content": "сегодня был длинный день"}]
    assert all(message["role"] != "system" for message in sent["messages"])


@pytest.mark.asyncio
async def test_the_budget_goes_to_each_providers_own_parameter():
    """`max_tokens` on Anthropic, `max_completion_tokens` on OpenAI."""
    anthropic_client, anthropic_calls = _anthropic_client()
    openai_client, openai_calls = _openai_client()

    await anthropic_client.complete(model="claude-opus-5", **ASK)
    await openai_client.complete(model="gpt-4o-mini", **ASK)

    assert anthropic_calls[0]["max_tokens"] == 512
    assert openai_calls[0]["max_completion_tokens"] == 512


@pytest.mark.asyncio
async def test_a_model_that_does_not_reason_is_not_sent_an_effort():
    """OpenAI rejects `reasoning_effort` on a model that cannot use it.

    gpt-4o-mini formats every diary entry this bot saves, so sending it the
    parameter would break saving entries rather than degrade it.
    """
    client, calls = _openai_client()

    await client.complete(model="gpt-4o-mini", effort="high", **ASK)

    assert "reasoning_effort" not in calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5.1", "o3", "o4-mini"])
async def test_a_model_that_reasons_is_sent_the_effort(model):
    client, calls = _openai_client()

    await client.complete(model=model, effort="low", **ASK)

    assert calls[0]["reasoning_effort"] == "low"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-6", "gpt-6-astra", "gpt-6.1", "GPT-6-Astra"])
async def test_the_gpt_6_family_is_sent_the_effort(model):
    """The coach roles are being configured onto gpt-6, and both of them mean it.

    The coach asks for `high` because it is the one role paid to think. The
    profile extractor pins `low` on purpose: an unpinned effort spends the
    completion budget reasoning and truncates the JSON. A family missing from
    the table drops either of those without a word.
    """
    client, calls = _openai_client()

    await client.complete(model=model, effort="high", **ASK)

    assert calls[0]["reasoning_effort"] == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-60-something", "gpt-6x", "gpt-55", "o30"])
async def test_a_family_matches_a_whole_name_and_not_a_bare_prefix(model):
    """`gpt-60-something` is not a gpt-6 and never was.

    A plain `startswith` would call every one of these a reasoning model and
    turn each request into the 400 the table exists to avoid.
    """
    client, calls = _openai_client()

    await client.complete(model=model, effort="high", **ASK)

    assert "reasoning_effort" not in calls[0]


@pytest.mark.asyncio
async def test_the_system_prompt_is_the_first_message_on_openai():
    client, calls = _openai_client()

    await client.complete(model="gpt-4o-mini", **ASK)

    assert calls[0]["messages"] == [
        {"role": "system", "content": ASK["system"]},
        {"role": "user", "content": "сегодня был длинный день"},
    ]


@pytest.mark.asyncio
async def test_no_json_asked_for_means_no_format_parameter_at_all():
    """Prose is the default on both, and asking for prose asks for nothing."""
    openai_client, openai_calls = _openai_client()
    anthropic_client, anthropic_calls = _anthropic_client()

    await openai_client.complete(model="gpt-4o-mini", json_schema=None, **ASK)
    await anthropic_client.complete(model="claude-opus-5", json_schema=None, **ASK)

    assert "response_format" not in openai_calls[0]
    assert "format" not in anthropic_calls[0]["output_config"]
    assert anthropic_calls[0]["system"] == ASK["system"]


@pytest.mark.asyncio
async def test_a_schema_becomes_structured_output_on_both():
    """The parameter is the mechanism. The prompt is not asked to do this job."""
    openai_client, openai_calls = _openai_client()
    anthropic_client, anthropic_calls = _anthropic_client()

    await openai_client.complete(model="gpt-4o-mini", json_schema=ENTRY_SCHEMA, **ASK)
    await anthropic_client.complete(model="claude-opus-5", json_schema=ENTRY_SCHEMA, **ASK)

    assert openai_calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "diary_entry", "schema": ENTRY_SCHEMA, "strict": True},
    }
    assert anthropic_calls[0]["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": ENTRY_SCHEMA},
    }
    assert anthropic_calls[0]["system"] == ASK["system"], "no prompt suffix behind a parameter"


@pytest.mark.asyncio
async def test_json_without_a_schema_uses_each_providers_best_available():
    """OpenAI has a mode for it. Anthropic has no parameter, so the prompt says it."""
    openai_client, openai_calls = _openai_client()
    anthropic_client, anthropic_calls = _anthropic_client()

    await openai_client.complete(model="gpt-4o-mini", require_json=True, **ASK)
    await anthropic_client.complete(model="claude-opus-5", require_json=True, **ASK)

    assert openai_calls[0]["response_format"] == {"type": "json_object"}
    assert "format" not in anthropic_calls[0]["output_config"]
    assert anthropic_calls[0]["system"].startswith(ASK["system"])
    assert "JSON" in anthropic_calls[0]["system"]


@pytest.mark.asyncio
async def test_an_unknown_effort_is_refused_before_a_request_is_made():
    openai_client, openai_calls = _openai_client()
    anthropic_client, anthropic_calls = _anthropic_client()

    for client, model in ((openai_client, "gpt-5"), (anthropic_client, "claude-opus-5")):
        with pytest.raises(ValueError, match="effort"):
            await client.complete(model=model, effort="maximum", **ASK)

    assert openai_calls == []
    assert anthropic_calls == []


@pytest.mark.asyncio
async def test_the_answer_comes_back_the_same_shape_from_either_provider():
    openai_client, _ = _openai_client()
    anthropic_client, _ = _anthropic_client()

    from_openai = await openai_client.complete(model="gpt-4o-mini", **ASK)
    from_anthropic = await anthropic_client.complete(model="claude-opus-5", **ASK)

    for completion in (from_openai, from_anthropic):
        assert completion.text == "Ответ"
        assert completion.usage == ai.Usage(input_tokens=11, output_tokens=7)
    assert from_openai.finish_reason == "stop"
    assert from_anthropic.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_anthropic_thinking_never_reaches_the_text():
    """A thinking block is not part of the answer, and this is a diary."""
    client, _ = _anthropic_client()

    completion = await client.complete(model="claude-opus-5", effort="high", **ASK)

    assert completion.text == "Ответ"
    assert "рассуждение" not in completion.text


@pytest.mark.asyncio
async def test_a_reply_without_usage_is_still_a_completion():
    """Not every response carries one, and a missing count is not an error."""
    client, _ = _openai_client(
        types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Ответ"))]
        )
    )

    completion = await client.complete(model="gpt-4o-mini", **ASK)

    assert completion.text == "Ответ"
    assert completion.usage is None
    assert completion.finish_reason is None


def _stub_notion_day(monkeypatch, summary):
    """The Notion reads summary.py imported, so no transport is needed."""

    async def today_page():
        return {"id": "page-1", "properties": {"title": {"title": [{"plain_text": "9 мая"}]}}}

    async def week_pages():
        return [await today_page()]

    async def page_blocks(page_id):
        return [{"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Текст"}]}}]

    monkeypatch.setattr(summary, "get_today_page", today_page)
    monkeypatch.setattr(summary, "get_week_pages", week_pages)
    monkeypatch.setattr(summary, "get_page_blocks", page_blocks)


@pytest.mark.asyncio
async def test_the_formatter_still_asks_for_a_json_object(monkeypatch):
    """Moving it behind the client must not quietly drop the JSON mode.

    The formatter parses what comes back with json.loads, so a reply wrapped in
    a markdown fence is a lost entry rather than a worse one.
    """
    from services import formatter

    client, calls = _openai_client(_openai_reply('{"title": "З", "text": "Т"}'))
    monkeypatch.setattr(formatter, "chat", client)

    await formatter.format_entry("сырая расшифровка")

    assert calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_the_formatter_does_not_pay_for_thinking(monkeypatch):
    """Punctuating dictation is mechanical, and the budget is the entry's.

    The formatter's budget is computed to carry the whole entry back word for
    word; thinking taken out of it is text that never arrives, and the caller
    hands what does arrive to json.loads.
    """
    from services import formatter

    client, calls = _anthropic_client(_anthropic_reply('{"title": "З", "text": "Т"}'))
    monkeypatch.setattr(formatter, "chat", client)

    await formatter.format_entry("сырая расшифровка")

    assert calls[0]["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_the_recaps_do_not_pay_for_thinking(monkeypatch):
    """A recap that spent its budget thinking comes back empty, and an empty
    daily summary is sent to the user as the message it is."""
    from services import summary

    client, calls = _anthropic_client()
    monkeypatch.setattr(summary, "chat", client)
    _stub_notion_day(monkeypatch, summary)

    assert await summary.generate_daily_summary() == "Ответ"
    assert await summary.generate_weekly_report() == "Ответ"

    assert [call["thinking"] for call in calls] == [{"type": "disabled"}] * 2


@pytest.mark.asyncio
async def test_the_recaps_ask_for_no_format_at_all(monkeypatch):
    """They were prose before this and have to stay prose: the daily summary is
    sent to the user as the message it is."""
    from services import summary

    client, calls = _openai_client()
    monkeypatch.setattr(summary, "chat", client)
    _stub_notion_day(monkeypatch, summary)

    assert await summary.generate_daily_summary() == "Ответ"
    assert "response_format" not in calls[0]


def test_the_provider_setting_picks_the_implementation(monkeypatch):
    monkeypatch.setattr(config.settings, "ai_provider", config.OPENAI)
    assert isinstance(ai.create_chat_client(), ai.OpenAIChatClient)

    monkeypatch.setattr(config.settings, "ai_provider", config.ANTHROPIC)
    monkeypatch.setattr(config.settings, "anthropic_api_key", "test-anthropic")
    assert isinstance(ai.create_chat_client(), ai.AnthropicChatClient)
