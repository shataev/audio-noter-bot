"""The file the coach's memory lives in, and the ways it could lose it.

This file is rewritten after every diary entry, on a small VPS, and it holds
everything the coach has accumulated about its owner. Three things are therefore
worth more than the rest of the module put together: a partly written file must
never be observable, a file written by some other version must still load, and
what this store writes must be the same bytes every time so a human can read the
diff.
"""

import json
import logging
import os
from datetime import datetime, timezone

import pytest

from services.coach import store as store_module
from services.coach.memory import (
    EditedItem,
    Fact,
    MemoryList,
    Tombstone,
    adopt,
    apply_ops,
)
from services.coach.store import (
    MAX_SNAPSHOTS,
    UNKNOWN_STAMP,
    MemoryStore,
    StoreData,
    StoreError,
)

MONDAY = datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "coach-memory.state.json")


def populated():
    profile = apply_ops(
        MemoryList(),
        [
            {"action": "create", "text": "Он спит плохо", "kind": "body"},
            {"action": "create", "text": "He is saving for a flat", "kind": "work"},
        ],
        source="page-1",
        now=MONDAY,
    ).items
    profile = apply_ops(
        profile,
        [{"action": "delete", "id": "1", "reason": "outdated"}],
        now=MONDAY,
    ).items
    rules = apply_ops(
        MemoryList(),
        [{"action": "create", "text": "Stop asking questions"}],
        kinds=("rule",),
        now=MONDAY,
    ).items
    return StoreData(profile=profile, rules=rules)


# --- loading ---------------------------------------------------------------


def test_a_missing_file_is_an_empty_store_not_an_error(store):
    data = store.load()

    assert data == StoreData()
    assert data.profile.next_id == 1
    assert not store.path.exists(), "reading must not create the file"


def test_an_empty_file_is_an_empty_store(store):
    store.path.write_text("", encoding="utf-8")

    assert store.load() == StoreData()


def test_a_file_that_is_not_json_is_refused_rather_than_treated_as_empty(store):
    """Returning an empty store here would make the next save delete everything."""
    store.path.write_text("{not json at all", encoding="utf-8")

    with pytest.raises(StoreError):
        store.load()

    assert store.path.read_text(encoding="utf-8") == "{not json at all"


def test_a_json_document_that_is_not_an_object_is_refused(store):
    store.path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(StoreError):
        store.load()


def test_a_legacy_fact_loads_with_its_missing_fields_filled_in(store):
    """A fact written before ``kind``, ``created_at`` and ``sources`` existed."""
    store.path.write_text(
        json.dumps({"profile": {"facts": [{"id": "4", "text": "He avoids conflict"}]}}),
        encoding="utf-8",
    )

    (fact,) = store.load().profile.facts

    assert fact.id == "4"
    assert fact.text == "He avoids conflict"
    assert fact.kind == "other"
    assert fact.created_at == UNKNOWN_STAMP
    assert fact.updated_at == UNKNOWN_STAMP
    assert fact.sources == ()
    assert fact.key is None


def test_a_legacy_rule_defaults_to_the_rules_vocabulary(store):
    store.path.write_text(
        json.dumps({"rules": {"facts": [{"id": "1", "text": "Be blunter about money"}]}}),
        encoding="utf-8",
    )

    (rule,) = store.load().rules.facts

    assert rule.kind == "rule"


def test_an_unknown_kind_survives_a_load(store):
    """This version's vocabulary is not the last word on what a kind may be."""
    store.path.write_text(
        json.dumps({"profile": {"facts": [{"id": "1", "text": "x", "kind": "astrology"}]}}),
        encoding="utf-8",
    )

    assert store.load().profile.facts[0].kind == "astrology"


def test_unknown_top_level_keys_survive_a_round_trip(store):
    store.path.write_text(
        json.dumps({"version": 2, "profile": {}, "rules": {}, "threads": {"last": 9}}),
        encoding="utf-8",
    )

    data = store.load()
    assert data.extra == {"threads": {"last": 9}}

    store.save(data)
    assert json.loads(store.path.read_text(encoding="utf-8"))["threads"] == {"last": 9}


def test_a_newer_version_is_read_rather_than_refused(store, caplog):
    store.path.write_text(
        json.dumps({"version": 99, "profile": {"facts": [{"id": "1", "text": "kept"}]}}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        data = store.load()

    assert [fact.text for fact in data.profile.facts] == ["kept"]
    assert "99" in caplog.text


def test_an_entry_with_no_text_is_dropped_and_the_rest_survive(store, caplog):
    store.path.write_text(
        json.dumps(
            {
                "profile": {
                    "facts": [
                        {"id": "1", "text": "  "},
                        "not an object",
                        {"id": "2", "text": "kept"},
                    ],
                    "tombstones": [{"deleted_at": "x"}],
                }
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        data = store.load()

    assert [fact.text for fact in data.profile.facts] == ["kept"]
    assert data.profile.tombstones == ()
    assert "dropped" in caplog.text


# --- the counter -----------------------------------------------------------


def test_the_counter_is_raised_past_every_id_in_the_file(store):
    """A hand-edited or truncated ``next_id`` must not hand out an id in use."""
    store.path.write_text(
        json.dumps(
            {
                "profile": {
                    "next_id": 2,
                    "facts": [{"id": "17", "text": "seventeen"}],
                }
            }
        ),
        encoding="utf-8",
    )

    assert store.load().profile.next_id == 18


def test_a_tombstoned_id_also_holds_the_counter_up(store):
    """Reusing a buried id would make a restore collide with a live fact."""
    store.path.write_text(
        json.dumps(
            {
                "profile": {
                    "next_id": 2,
                    "facts": [{"id": "1", "text": "one"}],
                    "tombstones": [{"fact": {"id": "40", "text": "buried"}, "deleted_at": "x"}],
                }
            }
        ),
        encoding="utf-8",
    )

    assert store.load().profile.next_id == 41


def test_the_counter_is_never_lowered_to_match_the_facts(store):
    """The counter on disk is the authority; the facts can only push it up."""
    store.path.write_text(
        json.dumps({"profile": {"next_id": 500, "facts": [{"id": "3", "text": "three"}]}}),
        encoding="utf-8",
    )

    assert store.load().profile.next_id == 500


def test_a_fact_with_no_id_is_given_one_rather_than_dropped(store):
    store.path.write_text(
        json.dumps({"profile": {"next_id": 9, "facts": [{"text": "no id at all"}]}}),
        encoding="utf-8",
    )

    data = store.load()

    assert [(fact.id, fact.text) for fact in data.profile.facts] == [("9", "no id at all")]
    assert data.profile.next_id == 10


def test_a_duplicated_id_is_reassigned_so_both_facts_stay_addressable(store, caplog):
    store.path.write_text(
        json.dumps(
            {
                "profile": {
                    "next_id": 2,
                    "facts": [{"id": "1", "text": "first"}, {"id": "1", "text": "second"}],
                }
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        data = store.load()

    assert [(fact.id, fact.text) for fact in data.profile.facts] == [
        ("1", "first"),
        ("2", "second"),
    ]
    assert "reassigned" in caplog.text


# --- serialisation ---------------------------------------------------------


def test_a_round_trip_leaves_the_file_byte_identical(store):
    store.save(populated())
    first = store.path.read_bytes()

    store.save(store.load())

    assert store.path.read_bytes() == first


def test_the_file_is_readable_russian_and_ends_with_a_newline(store):
    store.save(populated())

    text = store.path.read_text(encoding="utf-8")

    assert "Он спит плохо" in text, "ensure_ascii=False, so a human can read the diff"
    assert text.endswith("\n")
    assert "\\u" not in text


def test_the_key_order_is_fixed(store):
    store.save(populated())

    body = json.loads(store.path.read_text(encoding="utf-8"))

    assert list(body) == ["version", "profile", "rules"]
    assert list(body["profile"]) == ["next_id", "facts", "tombstones"]
    assert list(body["profile"]["facts"][0])[:6] == [
        "id",
        "text",
        "kind",
        "created_at",
        "updated_at",
        "sources",
    ]


def test_a_fact_without_a_key_does_not_carry_a_null_one(store):
    store.save(populated())

    body = json.loads(store.path.read_text(encoding="utf-8"))

    assert "key" not in body["profile"]["facts"][0]


def test_a_key_survives_a_round_trip(store):
    adopted = adopt(
        MemoryList(facts=(Fact("1", "a fact", "trait", "t", "t"),), next_id=2),
        [EditedItem("block-a", "a fact reworded")],
    ).items

    store.save(StoreData(profile=adopted))

    assert store.load().profile.facts[0].key == "block-a"


def test_a_tombstone_survives_a_round_trip(store):
    store.save(populated())

    (stone,) = store.load().profile.tombstones

    assert isinstance(stone, Tombstone)
    assert stone.fact.text == "Он спит плохо"
    assert stone.fact.sources == ("page-1",)
    assert stone.reason == "outdated"


def test_everything_a_fact_knows_survives_a_round_trip(store):
    store.save(populated())

    fact = store.load().profile.facts[0]

    assert fact.created_at == "2026-08-03T09:00:00+00:00"
    assert fact.sources == ("page-1",)
    assert fact.kind == "work"


def test_a_missing_directory_is_created(tmp_path):
    store = MemoryStore(tmp_path / "state" / "nested" / "coach.json")

    store.save(StoreData())

    assert store.load() == StoreData()


# --- atomic writes ---------------------------------------------------------


def test_the_real_path_is_never_opened_for_writing(store, monkeypatch):
    """The property, not the implementation: whatever a reader opens is whole."""
    store.save(populated())
    opened = []
    real_open = os.open

    def spy(path, flags, *args, **kwargs):
        opened.append((os.fspath(path), flags))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy)
    store.save(StoreData())

    written = [path for path, flags in opened if flags & (os.O_WRONLY | os.O_RDWR)]
    assert written, "if the store stops writing through os.open this test must be rewritten"
    assert str(store.path) not in written
    assert all(path.endswith(".tmp") for path in written)


def test_a_write_that_fails_at_the_last_moment_leaves_the_previous_file_intact(store, monkeypatch):
    store.save(populated())
    before = store.path.read_bytes()

    def boom(src, dst):
        raise OSError("no space left on device")

    monkeypatch.setattr(store_module.os, "replace", boom)
    with pytest.raises(OSError):
        store.save(StoreData())

    assert store.path.read_bytes() == before
    assert json.loads(store.path.read_text(encoding="utf-8"))


def test_a_failed_write_leaves_no_temporary_file_behind(store, monkeypatch):
    store.save(populated())

    monkeypatch.setattr(store_module.os, "replace", _raise)
    with pytest.raises(OSError):
        store.save(StoreData())

    assert [path.name for path in store.path.parent.iterdir()] == [store.path.name]


def test_a_serialisation_failure_never_touches_the_file(store, monkeypatch):
    """The document is built in full before anything is opened."""
    store.save(populated())
    before = store.path.read_bytes()

    monkeypatch.setattr(store_module, "_dumps", _raise)
    with pytest.raises(OSError):
        store.save(StoreData())

    assert store.path.read_bytes() == before
    assert [path.name for path in store.path.parent.iterdir()] == [store.path.name]


def _raise(*args, **kwargs):
    raise OSError("nope")


def test_the_file_is_not_world_readable(store):
    """Diary facts, on a box where the deploy account can read the directory."""
    store.save(populated())

    assert store.path.stat().st_mode & 0o077 == 0


# --- snapshots -------------------------------------------------------------


def test_a_snapshot_copies_the_file_as_it_stands(store):
    store.save(populated())

    target = store.snapshot("before-rebuild")

    assert target is not None
    assert target.parent == store.path.parent
    assert target.read_bytes() == store.path.read_bytes()


def test_a_snapshot_is_not_disturbed_by_later_saves(store):
    store.save(populated())
    target = store.snapshot("before-rebuild")
    kept = target.read_bytes()

    store.save(StoreData())

    assert target.read_bytes() == kept
    assert store.path.read_bytes() != kept


def test_snapshotting_a_store_that_has_never_been_written_does_nothing(store):
    assert store.snapshot("before-rebuild") is None
    assert list(store.path.parent.iterdir()) == []


def test_only_the_three_most_recent_snapshots_are_kept(store):
    store.save(populated())

    for number in range(1, 6):
        store.snapshot(f"run-{number}")

    remaining = sorted(path.name for path in store.snapshots())
    assert len(remaining) == MAX_SNAPSHOTS
    assert remaining == [
        "coach-memory.state.snapshot-run-3.json",
        "coach-memory.state.snapshot-run-4.json",
        "coach-memory.state.snapshot-run-5.json",
    ]


def test_a_snapshot_label_can_only_name_a_sibling(store):
    store.save(populated())

    target = store.snapshot("../../etc/passwd")

    assert target.parent == store.path.parent
    assert target.name == "coach-memory.state.snapshot-etc-passwd.json"


def test_a_label_that_is_all_punctuation_still_names_a_file(store):
    store.save(populated())

    target = store.snapshot("///")

    assert target.name == "coach-memory.state.snapshot-unlabelled.json"


def test_a_snapshot_can_be_loaded_as_a_store(store):
    store.save(populated())
    target = store.snapshot("before-rebuild")
    store.save(StoreData())

    assert MemoryStore(target).load() == populated()


# --- the log ---------------------------------------------------------------


SECRET = "Он в терапии из-за того, что случилось в 2011"


def test_no_fact_text_ever_reaches_the_log(store, caplog):
    data = StoreData(profile=apply_ops(MemoryList(), [{"action": "create", "text": SECRET}]).items)

    with caplog.at_level(logging.DEBUG):
        store.save(data)
        store.load()
        store.snapshot("before-rebuild")

    assert SECRET not in caplog.text
    assert "2011" not in caplog.text
