"""One chat interface, two providers behind it.

Everything in this project that talks to a chat model goes through `complete()`.
The interface is this project's own, deliberately: it is not the shape of either
SDK, and neither SDK's vocabulary leaks through it.

The alternative — a thin object that exposes `chat.completions.create(...)` and
quietly redirects it at whichever SDK is configured — is tempting because no call
site has to change. It is also exactly wrong for the three things this module
exists to get right. Effort, JSON output and thinking are the places where the
two APIs diverge, and an OpenAI-shaped façade forces every one of those
divergences to be smuggled through an OpenAI-shaped parameter. That is how
`reasoning_effort` ends up in an Anthropic request body and comes back as a 400
that nobody sees until it is running in production. So the translation is
explicit, in one function per provider, whose whole job is that translation.

One asymmetry is worth knowing before reading the two translations. On
Anthropic, thinking tokens are output tokens: they are spent out of the same
`max_tokens` ceiling as the answer itself. A request that thinks on a budget
sized for the reply can return no text block at all — not an error, just an
empty answer — so a role that has no use for reasoning has to say so, and gets
thinking switched off rather than left on with its budget quietly halved.

Transcription is not here. There is no Anthropic equivalent of the audio
endpoint, so `services/whisper.py` keeps its own OpenAI client and the provider
setting cannot reach it.
"""

import logging
from dataclasses import dataclass

import openai

from config import ANTHROPIC, OPENAI, settings

logger = logging.getLogger(__name__)

# How hard a model may think, or None for a role that does not reason at all.
# Anthropic also accepts "xhigh" and "max"; they are left out until something
# asks for them — but see the thinking translation below before adding one.
EFFORTS = ("low", "medium", "high")

# The models that accept an effort. OpenAI rejects `reasoning_effort` outright on
# a model that does not reason, so the mapping has to know which is which rather
# than hope: gpt-4o-mini formats a diary entry perfectly well and would fail the
# request if it were sent one. Families rather than individual ids, because the
# families are stable and the ids are not.
#
# The two ways of being wrong here are not symmetrical, which is why this is a
# list that gets added to rather than a guess. A family wrongly present fails
# loudly: the first request to such a model is a 400 and nobody can miss it. A
# family missing fails silently: the request succeeds, the effort is dropped on
# the floor, and a role that pinned one — the profile extractor pins `low` to
# keep reasoning from eating the completion budget — quietly gets the model's
# default instead. So add a family once it reasons; do not try to be clever
# about the ones that might.
_OPENAI_REASONING_FAMILIES = ("o1", "o3", "o4", "gpt-5", "gpt-6")

# Belt-and-braces only. The format parameters below are the mechanism; this is a
# sentence appended to the prompt for the one case that has no parameter —
# Anthropic asked for JSON with no schema to describe it.
_JSON_INSTRUCTION = "Reply with valid JSON only: no prose, no markdown fences."


@dataclass(frozen=True)
class Message:
    """One turn of the conversation. Plain text; no provider block structure."""

    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class Usage:
    """What the call cost, as both providers report it under different names."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class Completion:
    """What came back: the text, why it stopped, and what it cost.

    `finish_reason` and `usage` are the provider's own values, normalised only in
    name. Either can be None: not every response carries them.
    """

    text: str
    finish_reason: str | None = None
    usage: Usage | None = None


def _check_effort(effort: str | None) -> None:
    if effort is not None and effort not in EFFORTS:
        raise ValueError(f"effort must be None or one of {EFFORTS}, got {effort!r}")


def _openai_takes_effort(model: str) -> bool:
    name = model.lower()
    return any(
        name == family or name.startswith(f"{family}-") or name.startswith(f"{family}.")
        for family in _OPENAI_REASONING_FAMILIES
    )


def _schema_name(json_schema: dict) -> str:
    """OpenAI requires a name for a structured output; Anthropic does not.

    A schema that titles itself gets to keep its name; anything else is just
    "response", which the model never sees anyway.
    """
    title = json_schema.get("title")
    return title if isinstance(title, str) and title else "response"


class ChatClient:
    """The interface. `sdk` is the provider client that puts a request on the wire."""

    sdk: object

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        max_output_tokens: int,
        effort: str | None = "medium",
        json_schema: dict | None = None,
        require_json: bool = False,
    ) -> Completion:
        """Ask a model for one answer.

        `effort` is this project's abstraction, not a passthrough: how hard the
        model should think, in terms both providers can be asked in. It is
        dropped rather than translated for a model that cannot reason, and None
        says the role does not reason at all — a mechanical job that would be
        paying for thinking it has no use for, out of a budget it needs for the
        answer.

        `json_schema` is the strong form of a machine-readable answer — the
        schema is enforced by the provider. `require_json` is the weak form, for
        a caller that wants JSON but has no schema to describe it. Neither given,
        no format parameter is sent at all and the answer is prose.
        """
        raise NotImplementedError


class OpenAIChatClient(ChatClient):
    """Translates a request onto OpenAI's chat completions API."""

    def __init__(self, api_key: str) -> None:
        self.sdk = openai.AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        max_output_tokens: int,
        effort: str | None = "medium",
        json_schema: dict | None = None,
        require_json: bool = False,
    ) -> Completion:
        _check_effort(effort)

        kwargs: dict = {
            "model": model,
            # The system prompt is a message here, and the first one.
            "messages": [
                {"role": "system", "content": system},
                *({"role": m.role, "content": m.content} for m in messages),
            ],
            "max_completion_tokens": max_output_tokens,
        }

        # Nothing to send for a role that does not reason, and nothing to send
        # to a model that cannot. There is no one value that means "do not
        # reason" across the OpenAI models — "none" and "minimal" are each
        # accepted by some and rejected by others — so this omits the parameter
        # rather than guess at a 400. The roles that ask for no effort are
        # configured onto models that do not reason in the first place.
        if effort is not None:
            if _openai_takes_effort(model):
                kwargs["reasoning_effort"] = effort
            else:
                # The silent half of the asymmetry above, said out loud. A role
                # that asked to think and is not going to is worth a line in the
                # journal: without it the only symptom is output that slowly
                # gets worse. Nothing is logged on the ordinary path, where no
                # effort was asked for in the first place.
                logger.info(
                    "ai: %s is in no known reasoning family, dropping effort %r",
                    model,
                    effort,
                )

        if json_schema is not None:
            # Structured outputs: strict, so the schema is a guarantee rather
            # than a suggestion. A schema given here has to be one OpenAI
            # accepts in strict mode — every property required, no additional
            # properties.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(json_schema),
                    "schema": json_schema,
                    "strict": True,
                },
            }
        elif require_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.sdk.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return Completion(
            text=choice.message.content or "",
            finish_reason=getattr(choice, "finish_reason", None),
            usage=_openai_usage(response),
        )


class AnthropicChatClient(ChatClient):
    """Translates a request onto Anthropic's messages API.

    Three things this must never send, all of them removed from current models
    and all of them a 400 rather than a warning: `reasoning_effort`, which is
    OpenAI's parameter and does not exist here; `budget_tokens`, the old way of
    asking for thinking; and `temperature` or `top_p`.
    """

    def __init__(self, api_key: str) -> None:
        # Imported here rather than at module scope on purpose. The default
        # provider is OpenAI, and a server whose install predates this change
        # has no `anthropic` package; an ImportError at import time would take
        # the whole bot down instead of the one feature that needs it.
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "AI_PROVIDER=anthropic needs the `anthropic` package, which is "
                "not installed: run `pip install -r requirements.txt`"
            ) from exc

        self.sdk = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        max_output_tokens: int,
        effort: str | None = "medium",
        json_schema: dict | None = None,
        require_json: bool = False,
    ) -> Completion:
        _check_effort(effort)

        # Adaptive thinking: the model decides when to think, and the effort
        # says how hard. This is the whole of the effort mapping here — there is
        # no token budget to set, and setting one is an error.
        #
        # No effort means no thinking, and that has to be said rather than left
        # to the default: thinking is spent out of max_tokens, so leaving it on
        # for a role that does not need it takes the budget away from the answer
        # and can return a reply with no text in it at all. Switching it off is
        # only accepted alongside an effort of "high" or below, which EFFORTS
        # guarantees — read this before widening that tuple to "xhigh" or "max".
        thinking: dict = {"type": "adaptive"} if effort is not None else {"type": "disabled"}

        output_config: dict = {}
        if effort is not None:
            output_config["effort"] = effort

        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        elif require_json:
            # No parameter exists for JSON without a schema, so this is the one
            # place the prompt does the work.
            system = f"{system}\n\n{_JSON_INSTRUCTION}"

        kwargs: dict = {
            "model": model,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_output_tokens,
            "thinking": thinking,
        }
        # Asked for neither an effort nor a format, there is nothing to configure.
        if output_config:
            kwargs["output_config"] = output_config

        response = await self.sdk.messages.create(**kwargs)

        return Completion(
            text=_anthropic_text(response),
            finish_reason=getattr(response, "stop_reason", None),
            usage=_anthropic_usage(response),
        )


def _openai_usage(response: object) -> Usage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return Usage(
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
    )


def _anthropic_usage(response: object) -> Usage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return Usage(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


def _anthropic_text(response: object) -> str:
    """The text blocks of the reply, and only those.

    A reply that thought comes back as a list of blocks with the thinking in it.
    Joining the lot would paste the model's reasoning into a diary entry.
    """
    blocks = getattr(response, "content", None) or []
    return "".join(block.text for block in blocks if getattr(block, "type", None) == "text")


def create_chat_client() -> ChatClient:
    """The chat client for the configured provider."""
    if settings.ai_provider == ANTHROPIC:
        return AnthropicChatClient(settings.anthropic_api_key)
    if settings.ai_provider == OPENAI:
        return OpenAIChatClient(settings.openai_api_key)
    raise ValueError(f"unknown AI_PROVIDER: {settings.ai_provider!r}")
