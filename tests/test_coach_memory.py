"""The failure modes of the id-addressed memory.

The happy path — create a fact, reword it, delete it — is three lines and proves
almost nothing. What this module has to be right about is everything around it: a
model that names an id which no longer exists, a payload that is not a list of
operations at all, and an id that gets handed out twice. The last one is the worst
of the three, because it is silent: a stale operation lands on a fact the model
was never looking at, and the corruption is in exactly the data the feature exists
to accumulate.
"""

import logging
from datetime import datetime, timezone

from services.coach.memory import (
    MAX_SOURCES,
    MAX_TOMBSTONES,
    PROFILE_KINDS,
    RULE_KINDS,
    ApplyResult,
    EditedItem,
    Fact,
    MemoryList,
    SkippedOp,
    Tombstone,
    adopt,
    apply_ops,
    restore,
)

MONDAY = datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)
TUESDAY = datetime(2026, 8, 4, 9, 0, tzinfo=timezone.utc)


def create(text, kind=None):
    op = {"action": "create", "text": text}
    if kind is not None:
        op["kind"] = kind
    return op


def seeded(*texts, source="entry-1", now=MONDAY):
    """A list holding one fact per text, created in order."""
    return apply_ops(
        MemoryList(),
        [create(text) for text in texts],
        source=source,
        now=now,
    ).items


def texts(items):
    return [fact.text for fact in items.facts]


def ids(items):
    return [fact.id for fact in items.facts]


# --- minting ---------------------------------------------------------------


def test_ids_are_minted_in_order_from_one():
    items = seeded("first", "second", "third")

    assert ids(items) == ["1", "2", "3"]
    assert items.next_id == 4


def test_an_id_is_never_reused_after_the_highest_fact_is_deleted():
    """The whole reason the counter is persisted rather than derived.

    Derive the next id from the facts present and this sequence hands "3" out
    twice: an operation written against the first fact 3 then silently edits the
    second one.
    """
    items = seeded("first", "second", "third")

    items = apply_ops(items, [{"action": "delete", "id": "3"}], now=TUESDAY).items
    items = apply_ops(items, [create("fourth")], now=TUESDAY).items

    assert ids(items) == ["1", "2", "4"]
    assert items.next_id == 5


def test_the_counter_survives_deleting_every_fact():
    items = seeded("first", "second")
    items = apply_ops(
        items,
        [{"action": "delete", "id": "1"}, {"action": "delete", "id": "2"}],
    ).items

    assert items.facts == ()
    items = apply_ops(items, [create("later")]).items
    assert ids(items) == ["3"]


# --- operations that must change nothing -----------------------------------


def test_an_operation_naming_an_unknown_id_changes_nothing_and_is_counted():
    items = seeded("first")

    result = apply_ops(
        items,
        [
            {"action": "modify", "id": "404", "text": "rewritten"},
            {"action": "delete", "id": "405"},
        ],
    )

    assert result.items.facts == items.facts
    assert [op.reason for op in result.skipped] == ["unknown-id", "unknown-id"]
    assert [op.id for op in result.skipped] == ["404", "405"]
    assert result.created == ()


def test_an_unknown_id_is_never_turned_into_a_create():
    """A create here would be a guess: the model was looking at a list this id
    was in, so the disagreement is about history, not about the text."""
    items = seeded("first")

    result = apply_ops(items, [{"action": "modify", "id": "9", "text": "invented"}])

    assert texts(result.items) == ["first"]
    assert result.items.next_id == items.next_id


def test_a_malformed_payload_leaves_the_list_untouched():
    items = seeded("first", "second")

    for payload in (None, "create", 7, {"action": "create", "text": "sneaky"}):
        result = apply_ops(items, payload)

        assert result.items is items
        assert result.skipped == (SkippedOp(action=None, id=None, reason="payload-not-a-list"),)


def test_a_string_payload_is_not_iterated_character_by_character():
    """A string is iterable, and iterating one would make five operations out of
    the word "create"."""
    result = apply_ops(MemoryList(), "create")

    assert result.items.facts == ()
    assert len(result.skipped) == 1


def test_junk_among_good_operations_is_skipped_without_stopping_the_rest():
    result = apply_ops(
        MemoryList(),
        [
            "not an object",
            None,
            {"nothing": "useful"},
            {"action": "fly", "text": "unknown action"},
            create("a real fact"),
            create("   "),
        ],
    )

    assert texts(result.items) == ["a real fact"]
    assert [op.reason for op in result.skipped] == [
        "op-not-an-object",
        "op-not-an-object",
        "unknown-action",
        "unknown-action",
        "blank-text",
    ]


def test_a_create_repeating_an_existing_fact_is_skipped():
    items = seeded("He cycles to work")

    result = apply_ops(items, [create("  he   CYCLES to work ")])

    assert texts(result.items) == ["He cycles to work"]
    assert [op.reason for op in result.skipped] == ["duplicate-text"]
    assert result.items.next_id == items.next_id


def test_two_identical_creates_in_one_payload_produce_one_fact():
    result = apply_ops(MemoryList(), [create("twice over"), create("Twice Over")])

    assert texts(result.items) == ["twice over"]
    assert len(result.skipped) == 1


# --- kinds -----------------------------------------------------------------


def test_an_unknown_kind_is_filed_as_other_rather_than_dropped(caplog):
    with caplog.at_level(logging.INFO):
        result = apply_ops(MemoryList(), [create("a fact", kind="astrology")])

    assert [fact.kind for fact in result.items.facts] == ["other"]
    assert result.skipped == ()
    assert "other" in caplog.text


def test_a_missing_kind_is_filed_as_other():
    result = apply_ops(MemoryList(), [{"action": "create", "text": "a fact"}])

    assert [fact.kind for fact in result.items.facts] == ["other"]


def test_every_kind_in_the_vocabulary_is_accepted():
    ops = [create(f"fact {kind}", kind=kind) for kind in PROFILE_KINDS]

    result = apply_ops(MemoryList(), ops)

    assert [fact.kind for fact in result.items.facts] == list(PROFILE_KINDS)


def test_the_rules_list_files_everything_as_rule():
    result = apply_ops(
        MemoryList(),
        [create("stop asking questions", kind="trait")],
        kinds=RULE_KINDS,
    )

    assert [fact.kind for fact in result.items.facts] == ["rule"]


def test_a_modify_with_an_unrecognised_kind_keeps_the_kind_it_had():
    """Unlike a create, a modify has something worth keeping to fall back on."""
    items = apply_ops(MemoryList(), [create("he lifts weights", kind="body")]).items

    result = apply_ops(items, [{"action": "modify", "id": "1", "kind": "astrology"}])

    assert [op.reason for op in result.skipped] == ["nothing-to-change"]
    assert [fact.kind for fact in result.items.facts] == ["body"]


# --- modify ----------------------------------------------------------------


def test_modify_preserves_created_at_and_bumps_updated_at():
    items = seeded("he sleeps badly", now=MONDAY)

    result = apply_ops(
        items,
        [{"action": "modify", "id": "1", "text": "he sleeps badly on Sundays"}],
        now=TUESDAY,
    )

    fact = result.items.facts[0]
    assert fact.id == "1"
    assert fact.text == "he sleeps badly on Sundays"
    assert fact.created_at == "2026-08-03T09:00:00+00:00"
    assert fact.updated_at == "2026-08-04T09:00:00+00:00"
    assert result.modified == (fact,)


def test_modify_appends_a_source_without_duplicating_it():
    items = seeded("he sleeps badly", source="entry-1")

    items = apply_ops(
        items,
        [{"action": "modify", "id": "1", "text": "one"}],
        source="entry-2",
    ).items
    items = apply_ops(
        items,
        [{"action": "modify", "id": "1", "text": "two"}],
        source="entry-1",
    ).items

    assert items.facts[0].sources == ("entry-1", "entry-2")


def test_sources_keep_only_the_most_recent_twenty():
    items = seeded("he sleeps badly", source="entry-0")

    for number in range(1, MAX_SOURCES + 5):
        items = apply_ops(
            items,
            [{"action": "modify", "id": "1", "text": f"revision {number}"}],
            source=f"entry-{number}",
        ).items

    sources = items.facts[0].sources
    assert len(sources) == MAX_SOURCES
    assert sources[-1] == f"entry-{MAX_SOURCES + 4}"
    assert "entry-0" not in sources


def test_a_modify_with_nothing_to_change_is_skipped():
    items = seeded("unchanged")

    result = apply_ops(items, [{"action": "modify", "id": "1"}], now=TUESDAY)

    assert result.items.facts == items.facts
    assert [op.reason for op in result.skipped] == ["nothing-to-change"]


def test_a_modify_that_blanks_the_text_is_skipped():
    items = seeded("still here")

    result = apply_ops(items, [{"action": "modify", "id": "1", "text": "   "}])

    assert texts(result.items) == ["still here"]
    assert [op.reason for op in result.skipped] == ["blank-text"]


def test_an_id_sent_as_a_number_still_addresses_its_fact():
    items = seeded("first")

    result = apply_ops(items, [{"action": "modify", "id": 1, "text": "reworded"}])

    assert texts(result.items) == ["reworded"]


def test_op_is_accepted_as_a_spelling_of_action():
    result = apply_ops(MemoryList(), [{"op": "create", "text": "a fact"}])

    assert texts(result.items) == ["a fact"]


# --- tombstones ------------------------------------------------------------


def test_a_deleted_fact_leaves_the_list_but_not_the_store():
    items = seeded("he hates his job", now=MONDAY)

    result = apply_ops(
        items,
        [{"action": "delete", "id": "1", "reason": "he quit"}],
        now=TUESDAY,
    )

    assert result.items.facts == ()
    assert result.deleted == items.facts
    (stone,) = result.items.tombstones
    assert stone.fact.text == "he hates his job"
    assert stone.fact.created_at == "2026-08-03T09:00:00+00:00"
    assert stone.deleted_at == "2026-08-04T09:00:00+00:00"
    assert stone.reason == "he quit"


def test_a_tombstoned_fact_is_restorable_under_its_own_id():
    items = seeded("first", "second")
    items = apply_ops(items, [{"action": "delete", "id": "1"}]).items

    result = restore(items.facts, items.tombstones, "1")

    assert [fact.text for fact in result.items] == ["second", "first"]
    assert result.restored is not None and result.restored.id == "1"
    assert result.tombstones == ()


def test_restoring_an_id_that_is_not_buried_changes_nothing():
    items = seeded("first")

    for wanted in ("1", "404"):
        result = restore(items.facts, items.tombstones, wanted)

        assert result.items == items.facts
        assert result.tombstones == items.tombstones
        assert result.restored is None


def test_tombstones_keep_the_two_hundred_most_recent():
    items = seeded(*[f"fact {number}" for number in range(MAX_TOMBSTONES + 10)])

    items = apply_ops(
        items,
        [{"action": "delete", "id": fact.id} for fact in items.facts],
    ).items

    assert len(items.tombstones) == MAX_TOMBSTONES
    assert items.tombstones[0].fact.text == "fact 10"
    assert items.tombstones[-1].fact.text == f"fact {MAX_TOMBSTONES + 9}"


# --- adopt -----------------------------------------------------------------


def stored():
    """Two facts with keys, as a rendered-and-read-back list would have them."""
    return MemoryList(
        facts=(
            Fact(
                id="1",
                text="He avoids conflict",
                kind="trait",
                created_at="2026-08-03T09:00:00+00:00",
                updated_at="2026-08-03T09:00:00+00:00",
                sources=("entry-1",),
                key="block-a",
            ),
            Fact(
                id="2",
                text="He is saving for a flat",
                kind="work",
                created_at="2026-08-03T09:00:00+00:00",
                updated_at="2026-08-03T09:00:00+00:00",
                sources=("entry-2",),
                key="block-b",
            ),
        ),
        next_id=3,
    )


def test_adopt_keeps_the_id_when_the_key_matches_and_the_text_changed():
    """The point of ``key``: a bullet reworded by hand is the same fact."""
    result = adopt(
        stored(),
        [
            EditedItem("block-a", "He walks away from conflict"),
            EditedItem("block-b", "He is saving for a flat"),
        ],
        now=TUESDAY,
    )

    first, second = result.items.facts
    assert first.id == "1"
    assert first.text == "He walks away from conflict"
    assert first.kind == "trait"
    assert first.created_at == "2026-08-03T09:00:00+00:00"
    assert first.updated_at == "2026-08-04T09:00:00+00:00"
    assert first.sources == ("entry-1",)
    assert result.modified == (first,)
    assert second.updated_at == "2026-08-03T09:00:00+00:00"
    assert result.deleted == ()


def test_adopt_keeps_the_id_when_the_text_matches_and_there_is_no_key():
    items = MemoryList(
        facts=(
            Fact(
                id="7",
                text="He reads before bed",
                kind="pattern",
                created_at="2026-08-03T09:00:00+00:00",
                updated_at="2026-08-03T09:00:00+00:00",
                sources=("entry-9",),
            ),
        ),
        next_id=8,
    )

    result = adopt(items, [EditedItem("block-new", "He reads before bed")], now=TUESDAY)

    (fact,) = result.items.facts
    assert fact.id == "7"
    assert fact.sources == ("entry-9",)
    assert fact.key == "block-new", "the key is recorded so the next edit matches on it"


def test_adopt_matches_on_the_key_before_the_text_when_the_two_disagree():
    """The one case that separates the two orders, and the reason ``key`` exists.

    The owner has moved a sentence from one bullet to another — recycled the
    wording of the second bullet into the first, and given the second something
    new. Every other case in this file agrees whichever order the two matches are
    tried in; this one does not.

    Matching on the key, the edit lands on the bullet he was editing. Matching on
    the text first, the first line lands on fact 2 instead, fact 1 is left
    unclaimed and tombstoned, and the sentence he moved arrives carrying the other
    fact's kind, its creation date and the entries that taught it. Two facts trade
    histories and nothing anywhere says so.
    """
    result = adopt(
        stored(),
        [
            EditedItem("block-a", "He is saving for a flat"),
            EditedItem("block-b", "He wants to learn to sail"),
        ],
        now=TUESDAY,
    )

    first, second = result.items.facts
    assert (first.id, first.key, first.text) == ("1", "block-a", "He is saving for a flat")
    assert (second.id, second.key, second.text) == ("2", "block-b", "He wants to learn to sail")
    assert (first.kind, first.sources) == ("trait", ("entry-1",))
    assert (second.kind, second.sources) == ("work", ("entry-2",))
    assert result.deleted == (), "nothing was removed, so nothing may be buried"
    assert result.created == (), "both lines matched, so no id may be minted"
    assert result.items.next_id == 3


def test_adopt_falls_back_to_the_text_when_the_key_names_nothing():
    """A key that has been reissued — a bullet deleted and rewritten in Notion.

    It matches no stored fact, so the text decides, and the new key is recorded
    in place of the one that is gone: next time round there is a key to match on
    again.
    """
    result = adopt(
        stored(),
        [
            EditedItem("block-reissued", "He avoids conflict"),
            EditedItem("block-b", "He is saving for a flat"),
        ],
        now=TUESDAY,
    )

    first, _ = result.items.facts
    assert first.id == "1"
    assert first.key == "block-reissued"
    assert first.sources == ("entry-1",)
    assert first.created_at == "2026-08-03T09:00:00+00:00"
    assert result.created == ()
    assert result.deleted == ()


def test_adopt_mints_an_id_for_a_line_that_matches_nothing():
    result = adopt(
        stored(),
        [
            EditedItem("block-a", "He avoids conflict"),
            EditedItem("block-b", "He is saving for a flat"),
            EditedItem("block-c", "He wants to learn to sail"),
        ],
        now=TUESDAY,
    )

    assert [fact.id for fact in result.items.facts] == ["1", "2", "3"]
    new = result.items.facts[-1]
    assert new.kind == "other"
    assert new.created_at == "2026-08-04T09:00:00+00:00"
    assert result.created == (new,)
    assert result.items.next_id == 4


def test_adopt_tombstones_a_fact_that_vanished_from_the_list():
    result = adopt(stored(), [EditedItem("block-a", "He avoids conflict")], now=TUESDAY)

    assert [fact.id for fact in result.items.facts] == ["1"]
    assert [fact.id for fact in result.deleted] == ["2"]
    (stone,) = result.items.tombstones
    assert stone.fact.id == "2"
    assert stone.reason == "removed by hand"
    assert stone.deleted_at == "2026-08-04T09:00:00+00:00"


def test_adopt_takes_the_order_the_owner_left():
    result = adopt(
        stored(),
        [
            EditedItem("block-b", "He is saving for a flat"),
            EditedItem("block-a", "He avoids conflict"),
        ],
    )

    assert [fact.id for fact in result.items.facts] == ["2", "1"]
    assert result.deleted == ()


def test_adopt_does_not_wipe_a_key_when_the_edited_line_has_none():
    result = adopt(stored(), [EditedItem(None, "He avoids conflict")])

    (fact,) = [f for f in result.items.facts if f.id == "1"]
    assert fact.key == "block-a"


def test_adopt_skips_a_blank_line_rather_than_storing_it():
    result = adopt(
        stored(),
        [EditedItem("block-a", "He avoids conflict"), EditedItem(None, "   ")],
    )

    assert [fact.id for fact in result.items.facts] == ["1"]
    assert [op.reason for op in result.skipped] == ["blank-text"]


def test_adopt_of_an_empty_list_buries_everything_it_had():
    """Nothing is lost even then — this is what the tombstones are for."""
    result = adopt(stored(), [])

    assert result.items.facts == ()
    assert [stone.fact.id for stone in result.items.tombstones] == ["1", "2"]


def test_adopt_returns_an_apply_result_like_apply_ops_does():
    assert isinstance(adopt(stored(), []), ApplyResult)


# --- the log ---------------------------------------------------------------


SECRET = "He is in therapy for the thing that happened in 2011"


def test_no_fact_text_ever_reaches_the_log(caplog):
    """A diary is not something to put in a journal the deploy account can read."""
    with caplog.at_level(logging.DEBUG):
        items = apply_ops(MemoryList(), [create(SECRET, kind="nonsense")]).items
        items = apply_ops(items, [{"action": "modify", "id": "1", "text": SECRET + "!"}]).items
        items = apply_ops(items, [{"action": "delete", "id": "1", "reason": SECRET}]).items
        apply_ops(items, [create(SECRET), "junk", {"action": "modify", "id": "404"}])
        adopt(MemoryList(), [EditedItem(None, SECRET)])
        restore(items.facts, items.tombstones, "1")

    assert SECRET not in caplog.text
    assert "2011" not in caplog.text


def test_a_skipped_operation_carries_its_id_but_not_its_text():
    result = apply_ops(MemoryList(), [{"action": "modify", "id": "404", "text": SECRET}])

    (skipped,) = result.skipped
    assert skipped.id == "404"
    assert SECRET not in repr(skipped)


# --- shape -----------------------------------------------------------------


def test_facts_and_tombstones_are_immutable():
    fact = Fact(id="1", text="t", kind="other", created_at="x", updated_at="x")
    stone = Tombstone(fact=fact, deleted_at="x")

    for frozen in (fact, stone, MemoryList(), apply_ops(MemoryList(), [])):
        try:
            frozen.nonsense = 1
        except Exception as error:  # noqa: BLE001 - the type is the assertion
            assert type(error).__name__ in ("FrozenInstanceError", "AttributeError")
        else:  # pragma: no cover - only reached if a dataclass stops being frozen
            raise AssertionError(f"{type(frozen).__name__} is not frozen")
