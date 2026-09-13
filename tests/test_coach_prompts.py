"""The system prompt, and the marker the model edits its own rules through.

Two things are being pinned here.

The prompt has an *order*: persona, then shape, then the protocol, then the
owner's rules under a header saying they win. Every one of those is checked by
position rather than by presence, because presence is not the property that
matters — a rules list that arrives before the persona is a rules list the
persona overrides, which is the opposite of what this feature is for.

The split is checked against messy input rather than the clean case, because the
clean case is not where it breaks. A model that is told to end with a marker and
a JSON object will sooner or later fence it, write something after it, emit two,
or emit the marker and then nothing — and every one of those has to leave the
owner with a readable answer and the rules list either correctly edited or
untouched.
"""

import os

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import pytest

from services.coach import prompts
from services.coach.memory import Fact

ANSWER = "Ты третий раз за неделю пишешь одно и то же и каждый раз ждёшь другого конца."


def rule(fact_id, text):
    return Fact(
        id=fact_id,
        text=text,
        kind="rule",
        created_at="2026-09-01T10:00:00+00:00",
        updated_at="2026-09-01T10:00:00+00:00",
    )


# --------------------------------------------------------------------------- #
# The three modes
# --------------------------------------------------------------------------- #


def test_there_are_three_modes_with_distinct_keys_and_labels():
    keys = [mode.key for mode in prompts.MODES]
    labels = [mode.label for mode in prompts.MODES]

    assert len(prompts.MODES) == 3
    assert len(set(keys)) == 3
    assert len(set(labels)) == 3


@pytest.mark.parametrize("mode", prompts.MODES, ids=lambda mode: mode.key)
def test_every_mode_can_be_found_by_the_key_a_callback_carries(mode):
    assert prompts.mode_for(mode.key) is mode


def test_an_unknown_mode_key_is_not_an_error():
    """A button from an older version must not raise out of a handler."""
    assert prompts.mode_for("therapist") is None


@pytest.mark.parametrize("mode", prompts.MODES, ids=lambda mode: mode.key)
def test_the_committed_persona_is_used_when_the_environment_says_nothing(mode, monkeypatch):
    monkeypatch.delenv(mode.variable, raising=False)

    assert prompts.persona(mode) == mode.default
    assert mode.default.strip()


@pytest.mark.parametrize("mode", prompts.MODES, ids=lambda mode: mode.key)
def test_the_environment_replaces_the_persona_entirely(mode, monkeypatch):
    """The privacy boundary: the owner's real persona never reaches the repository."""
    monkeypatch.setenv(mode.variable, "Отвечай одним словом.")

    assert prompts.persona(mode) == "Отвечай одним словом."
    assert mode.default not in prompts.system_prompt(mode)


@pytest.mark.parametrize("mode", prompts.MODES, ids=lambda mode: mode.key)
def test_a_variable_that_is_set_but_blank_falls_back_to_the_default(mode, monkeypatch):
    """A typo in a deploy must not hand the model an empty persona."""
    monkeypatch.setenv(mode.variable, "   \n ")

    assert prompts.persona(mode) == mode.default


def test_the_modes_differ_only_in_the_persona(monkeypatch):
    """Same pipeline, three personas — not three prompts that have drifted apart."""
    for mode in prompts.MODES:
        monkeypatch.delenv(mode.variable, raising=False)

    without_persona = {
        prompts.system_prompt(mode).replace(mode.default, "") for mode in prompts.MODES
    }

    assert len(without_persona) == 1


# --------------------------------------------------------------------------- #
# The rules go last, and they say they win
# --------------------------------------------------------------------------- #


def test_the_rules_come_after_the_persona(monkeypatch):
    monkeypatch.setenv("COACH_PROMPT_ROAST", "ПЕРСОНА")
    roast = prompts.mode_for("roast")

    prompt = prompts.system_prompt(roast, [rule("4", "не желать доброго утра")])

    assert prompt.index("ПЕРСОНА") < prompt.index("не желать доброго утра")


def test_the_rules_are_the_last_thing_in_the_prompt():
    """Last, so that nothing the prompt author wrote gets the final word."""
    roast = prompts.mode_for("roast")

    prompt = prompts.system_prompt(roast, [rule("4", "не желать доброго утра")])

    assert prompt.rstrip().endswith("[4] не желать доброго утра")


def test_each_rule_carries_the_id_the_model_has_to_address_it_by():
    prompt = prompts.system_prompt(
        prompts.mode_for("breakdown"),
        [rule("2", "не задавать вопросов"), rule("7", "обращаться на «ты»")],
    )

    assert "[2] не задавать вопросов" in prompt
    assert "[7] обращаться на «ты»" in prompt


def test_the_header_says_plainly_that_a_rule_outranks_the_persona():
    """The ordering is only half of it: the model is told which wins on conflict."""
    prompt = prompts.system_prompt(prompts.mode_for("roast"), [rule("1", "не ругаться")])

    header = prompt[prompt.index("[1] не ругаться") - 400 : prompt.index("[1] не ругаться")]

    assert "важнее" in header
    assert "правило" in header


def test_an_empty_list_says_so_rather_than_leaving_the_mechanism_unmentioned():
    prompt = prompts.system_prompt(prompts.mode_for("support"))

    assert "пуст" in prompt
    assert prompts.RULES_MARKER in prompt


def test_the_protocol_in_the_prompt_uses_the_marker_the_parser_looks_for():
    """A prompt that documents a different marker is a feature that silently never fires."""
    prompt = prompts.system_prompt(prompts.mode_for("roast"))

    assert prompt.count(prompts.RULES_MARKER) == 1


def test_the_rule_line_is_rendered_the_same_way_for_the_model_and_for_the_owner():
    """`/rules` prints what the model reads, so "забудь правило 2" addresses rule 2."""
    fact = rule("2", "не задавать вопросов")

    assert prompts.rule_line(fact) in prompts.system_prompt(prompts.mode_for("roast"), [fact])


# --------------------------------------------------------------------------- #
# Splitting the reply — the ordinary case
# --------------------------------------------------------------------------- #


def test_no_marker_means_no_change_at_all():
    """The overwhelmingly common reply: it costs nothing and touches nothing."""
    visible, ops = prompts.split_rules_update(ANSWER)

    assert visible == ANSWER
    assert ops is None


def test_a_clean_block_yields_its_operations():
    visible, ops = prompts.split_rules_update(
        ANSWER + '\n<<<RULES>>>{"ops": [{"action": "create", "text": "короче"}]}'
    )

    assert visible == ANSWER
    assert ops == [{"action": "create", "text": "короче"}]


def test_operations_can_address_an_existing_rule_by_id():
    _, ops = prompts.split_rules_update(
        ANSWER + '\n<<<RULES>>>{"ops": [{"action": "delete", "id": "3"}]}'
    )

    assert ops == [{"action": "delete", "id": "3"}]


# --------------------------------------------------------------------------- #
# Splitting the reply — the messy cases, which are the real ones
# --------------------------------------------------------------------------- #


def test_a_fenced_block_leaves_no_backticks_in_the_visible_answer():
    """The opening fence lands before the marker, so it is in the half the owner reads."""
    visible, ops = prompts.split_rules_update(
        f"{ANSWER}\n\n```json\n<<<RULES>>>"
        '{"ops": [{"action": "create", "text": "короче"}]}\n```'
    )

    assert visible == ANSWER
    assert "`" not in visible
    assert ops == [{"action": "create", "text": "короче"}]


def test_a_fence_between_the_marker_and_the_json_is_stepped_over():
    visible, ops = prompts.split_rules_update(
        f"{ANSWER}\n<<<RULES>>>\n```json\n"
        '{"ops": [{"action": "create", "text": "короче"}]}\n```'
    )

    assert visible == ANSWER
    assert ops == [{"action": "create", "text": "короче"}]


def test_text_after_the_json_is_ignored_rather_than_failing_the_parse():
    """`json.loads` raises on this; `raw_decode` reads one value and stops."""
    visible, ops = prompts.split_rules_update(
        ANSWER + '\n<<<RULES>>>{"ops": [{"action": "create", "text": "короче"}]}\nГотово, записал.'
    )

    assert visible == ANSWER
    assert ops == [{"action": "create", "text": "короче"}]
    assert "Готово" not in visible


def test_a_second_marker_never_reaches_the_owner():
    """Cut at the first. The rest is debris and debris must not be delivered."""
    visible, ops = prompts.split_rules_update(
        ANSWER + '\n<<<RULES>>>{"ops": [{"action": "create", "text": "первое"}]}'
        '\n<<<RULES>>>{"ops": [{"action": "create", "text": "второе"}]}'
    )

    assert visible == ANSWER
    assert prompts.RULES_MARKER not in visible
    assert ops == [{"action": "create", "text": "первое"}]


def test_a_marker_with_nothing_after_it_is_dropped_and_the_answer_still_arrives():
    visible, ops = prompts.split_rules_update(ANSWER + "\n<<<RULES>>>")

    assert visible == ANSWER
    assert ops is None


def test_a_marker_followed_by_junk_is_dropped_and_the_answer_still_arrives():
    visible, ops = prompts.split_rules_update(ANSWER + "\n<<<RULES>>> записал!")

    assert visible == ANSWER
    assert ops is None


def test_an_unparseable_json_object_is_dropped_and_the_answer_still_arrives():
    """Nothing is written, and the owner loses a rule he can repeat — not the answer."""
    visible, ops = prompts.split_rules_update(ANSWER + '\n<<<RULES>>>{"ops": [{"action": ')

    assert visible == ANSWER
    assert ops is None


def test_a_block_that_is_valid_json_but_carries_no_operations_is_dropped():
    visible, ops = prompts.split_rules_update(ANSWER + '\n<<<RULES>>>{"ops": "создай правило"}')

    assert visible == ANSWER
    assert ops is None


def test_a_bare_array_of_operations_is_accepted():
    """Not what the prompt asks for, and not worth losing a rule over."""
    _, ops = prompts.split_rules_update(
        ANSWER + '\n<<<RULES>>>[{"action": "create", "text": "короче"}]'
    )

    assert ops == [{"action": "create", "text": "короче"}]


def test_an_empty_operations_list_is_a_block_that_changes_nothing():
    """Distinct from no marker: the model answered the protocol, it just asked for nothing."""
    visible, ops = prompts.split_rules_update(ANSWER + '\n<<<RULES>>>{"ops": []}')

    assert visible == ANSWER
    assert ops == []


def test_a_reply_that_is_nothing_but_a_block_leaves_no_visible_text():
    """The caller has to notice this and say something; it must not send an empty message."""
    visible, ops = prompts.split_rules_update(
        '<<<RULES>>>{"ops": [{"action": "create", "text": "короче"}]}'
    )

    assert visible == ""
    assert ops == [{"action": "create", "text": "короче"}]


@pytest.mark.parametrize(
    "answer",
    [
        ANSWER + '\n<<<RULES>>>{"ops": [{"action": "create", "text": "короче"}]}',
        f'{ANSWER}\n```json\n<<<RULES>>>{{"ops": []}}\n```',
        ANSWER + "\n<<<RULES>>>",
        ANSWER + '\n<<<RULES>>>{"ops": [{"action": ',
        ANSWER + '\n<<<RULES>>>{"ops": []}\n<<<RULES>>>{"ops": []}',
    ],
)
def test_the_owner_never_sees_the_marker_whatever_shape_it_arrived_in(answer):
    """One assertion over every messy case above: this is the one that must not regress."""
    visible, _ = prompts.split_rules_update(answer)

    assert prompts.RULES_MARKER not in visible
    assert "```" not in visible
    assert visible == ANSWER


def test_a_reply_wrapped_entirely_in_a_fence_loses_both_backticks():
    """The trailing fence is the common case; a model can open one at the top too.

    The fenced-block test above only ever exercises the closing fence, because a
    fence opened before the marker leaves its backticks at the *end* of the
    visible half. This is the other end of the same strip.
    """
    visible, ops = prompts.split_rules_update(f"```\n{ANSWER}\n```")

    assert visible == ANSWER
    assert "`" not in visible
    assert ops is None


def test_a_language_tagged_opening_fence_goes_too():
    visible, _ = prompts.split_rules_update(f"```markdown\n{ANSWER}\n```")

    assert visible == ANSWER
