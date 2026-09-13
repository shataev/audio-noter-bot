"""Conversations that survive a deploy, and bounds that stop them growing forever.

The restart is the point of this module, so it is what most of these tests do:
write through one store, read through another built on the same path, and check
that what comes back is what a conversation needs to continue — the turns, in
order, reachable from every message the coach sent.

The bounds are tested at their edges rather than by filling the file up, and the
pruning tests care as much about the ``forgotten`` list as about what was
dropped. A thread that is pruned silently is a reply that vanishes; a thread that
is pruned into ``forgotten`` is a reply that gets an answer.
"""

import os

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from services.coach import threads

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return threads.ThreadStore(tmp_path / "coach_threads.state.json")


def turns(*texts):
    return [
        threads.Turn(role="user" if index % 2 == 0 else "assistant", text=text)
        for index, text in enumerate(texts)
    ]


def opened(items=None, *, message_id=101, mode="roast", now=NOW, said=("запись", "ответ")):
    return threads.start(
        items if items is not None else threads.Threads(),
        mode=mode,
        turns=turns(*said),
        keys=[threads.key(1, message_id)],
        now=now,
    )


# --------------------------------------------------------------------------- #
# Addressing
# --------------------------------------------------------------------------- #


def test_a_conversation_is_found_from_the_message_that_was_replied_to():
    items, thread = opened()

    assert threads.find(items, threads.key(1, 101)) == thread


def test_every_message_of_one_answer_leads_to_the_same_conversation():
    """A long answer is several messages, and replying to any of them means the same."""
    items, thread = threads.start(
        threads.Threads(),
        mode="roast",
        turns=turns("запись", "длинный ответ"),
        keys=[threads.key(1, 101), threads.key(1, 102), threads.key(1, 103)],
        now=NOW,
    )

    found = [threads.find(items, threads.key(1, at)) for at in (101, 102, 103)]

    assert found == [thread, thread, thread]


def test_a_message_from_another_chat_is_not_the_same_conversation():
    items, _ = opened()

    assert threads.find(items, threads.key(2, 101)) is None


def test_an_unrelated_message_belongs_to_no_conversation():
    """The preview's own messages must keep reaching the handlers they always did."""
    items, _ = opened()

    assert threads.find(items, threads.key(1, 55)) is None
    assert not threads.was_forgotten(items, threads.key(1, 55))


# --------------------------------------------------------------------------- #
# Continuing one
# --------------------------------------------------------------------------- #


def test_a_reply_is_appended_after_what_was_already_said():
    items, thread = opened()

    items, extended = threads.extend(
        items,
        thread.id,
        turns=turns("а если нет", "тогда так"),
        keys=[threads.key(1, 110)],
        now=NOW,
    )

    assert [turn.text for turn in extended.turns] == ["запись", "ответ", "а если нет", "тогда так"]


def test_the_new_messages_become_doors_into_the_same_conversation():
    items, thread = opened()

    items, _ = threads.extend(items, thread.id, keys=[threads.key(1, 110)], now=NOW)

    assert threads.find(items, threads.key(1, 110)) is not None
    assert threads.find(items, threads.key(1, 101)).id == thread.id


def test_extending_a_conversation_that_is_gone_changes_nothing():
    items, _ = opened()

    unchanged, missing = threads.extend(items, "999", turns=turns("эй"), now=NOW)

    assert missing is None
    assert unchanged == items


def test_a_key_is_not_recorded_twice():
    items, thread = opened()

    items, extended = threads.extend(items, thread.id, keys=[threads.key(1, 101)], now=NOW)

    assert extended.keys == (threads.key(1, 101),)


# --------------------------------------------------------------------------- #
# A restart
# --------------------------------------------------------------------------- #


def test_nothing_is_stored_until_something_is_said(store):
    assert store.load() == threads.Threads()
    assert not store.path.exists(), "reading must not create the file"


def test_a_conversation_survives_a_restart(store):
    """The whole reason this is a file. `make deploy` restarts the bot on every push."""
    items, thread = opened()
    store.save(items)

    reloaded = threads.ThreadStore(store.path).load(now=NOW)
    found = threads.find(reloaded, threads.key(1, 101))

    assert found is not None
    assert found.id == thread.id
    assert found.mode == "roast"
    assert [(turn.role, turn.text) for turn in found.turns] == [
        ("user", "запись"),
        ("assistant", "ответ"),
    ]


def test_every_door_into_a_conversation_survives_a_restart(store):
    items, _ = threads.start(
        threads.Threads(),
        mode="support",
        turns=turns("запись", "ответ"),
        keys=[threads.key(1, 101), threads.key(1, 102)],
        now=NOW,
    )
    store.save(items)

    reloaded = threads.ThreadStore(store.path).load(now=NOW)

    assert threads.find(reloaded, threads.key(1, 102)) is not None


def test_a_conversation_continued_after_a_restart_keeps_everything_said_before(store):
    items, thread = opened()
    store.save(items)

    reloaded = threads.ThreadStore(store.path).load(now=NOW)
    found = threads.find(reloaded, threads.key(1, 101))
    _, extended = threads.extend(reloaded, found.id, turns=turns("а если нет"), now=NOW)

    assert [turn.text for turn in extended.turns] == ["запись", "ответ", "а если нет"]


def test_the_id_counter_is_not_reused_after_a_restart(store):
    """Derived ids hand the same number out twice; the counter is persisted for that."""
    items, first = opened()
    store.save(items)

    reloaded = threads.ThreadStore(store.path).load(now=NOW)
    _, second = opened(reloaded, message_id=201)

    assert second.id != first.id


def test_a_hand_edited_counter_cannot_hand_out_an_id_already_in_use(store):
    items, _ = opened()
    store.save(items)
    body = json.loads(store.path.read_text(encoding="utf-8"))
    body["next_id"] = 1
    store.path.write_text(json.dumps(body), encoding="utf-8")

    reloaded = threads.ThreadStore(store.path).load(now=NOW)
    _, fresh = opened(reloaded, message_id=201)

    assert fresh.id not in {thread.id for thread in items.threads}


def test_the_file_is_written_for_a_human_to_read(store):
    items, _ = opened()

    store.save(items)
    raw = store.path.read_text(encoding="utf-8")

    assert "запись" in raw, "Cyrillic escaped into \\u would make the file unreadable"
    assert raw.endswith("\n")


def test_a_file_that_is_not_this_store_is_never_read_as_an_empty_one(store):
    """Loading it as empty would let the next save overwrite whatever it really was."""
    store.path.write_text("{not json at all", encoding="utf-8")

    with pytest.raises(threads.ThreadStoreError):
        store.load()

    assert store.path.read_text(encoding="utf-8") == "{not json at all"


def test_an_empty_file_is_an_empty_store(store):
    store.path.write_text("", encoding="utf-8")

    assert store.load() == threads.Threads()


def _stored(store, thread):
    store.path.write_text(
        json.dumps({"version": 1, "next_id": 2, "threads": [thread], "forgotten": []}),
        encoding="utf-8",
    )
    return store.load(now=NOW)


def test_a_conversation_nothing_can_reply_into_is_not_kept(store):
    """No keys means no door: it would be scanned on every message and never match.

    The turns are deliberately present. Dropping a thread that has neither turns
    nor keys proves only that one of the two rules fires, and the one this test is
    named for is the other.
    """
    loaded = _stored(
        store,
        {
            "id": "1",
            "mode": "roast",
            "keys": [],
            "turns": [{"role": "user", "text": "запись"}],
        },
    )

    assert loaded.threads == ()


def test_a_conversation_with_nothing_in_it_is_not_kept(store):
    """And the other way round: a door into a conversation that has nothing to say."""
    loaded = _stored(
        store, {"id": "1", "mode": "roast", "keys": [threads.key(1, 101)], "turns": []}
    )

    assert loaded.threads == ()


# --------------------------------------------------------------------------- #
# The bounds
# --------------------------------------------------------------------------- #


def test_a_conversation_older_than_the_age_bound_is_dropped():
    items, _ = opened(now=NOW - timedelta(days=threads.MAX_AGE_DAYS + 1))

    pruned = threads.prune(items, now=NOW)

    assert pruned.threads == ()


def test_a_conversation_inside_the_age_bound_is_kept():
    items, _ = opened(now=NOW - timedelta(days=threads.MAX_AGE_DAYS - 1))

    pruned = threads.prune(items, now=NOW)

    assert len(pruned.threads) == 1


def test_age_is_counted_from_the_last_message_not_the_first():
    """A conversation still being used is not old, however long ago it started."""
    items, thread = opened(now=NOW - timedelta(days=100))
    items, _ = threads.extend(items, thread.id, turns=turns("ещё"), now=NOW)

    assert len(threads.prune(items, now=NOW).threads) == 1


def test_the_age_bound_is_applied_on_load_so_it_holds_while_the_bot_is_off(store):
    """A fortnight with the bot stopped has to expire what a fortnight running would."""
    items, _ = opened(now=NOW - timedelta(days=threads.MAX_AGE_DAYS + 1))
    store.save(items)

    assert store.load(now=NOW).threads == ()


def test_only_the_newest_conversations_are_kept():
    items = threads.Threads()
    for index in range(threads.MAX_THREADS + 5):
        items, _ = opened(items, message_id=100 + index, now=NOW)

    assert len(items.threads) == threads.MAX_THREADS


def test_the_conversation_being_used_is_not_the_one_dropped():
    """Least recently used, not oldest: an active thread survives a flood of new ones."""
    items, active = opened(message_id=100, now=NOW - timedelta(days=1))
    for index in range(threads.MAX_THREADS):
        items, _ = opened(items, message_id=200 + index, now=NOW)
        items, _ = threads.extend(items, active.id, turns=turns("ещё"), now=NOW)

    assert threads.find(items, threads.key(1, 100)) is not None


def test_a_long_conversation_keeps_its_most_recent_messages():
    items, thread = opened(said=("первое", "ответ"))
    for index in range(threads.MAX_TURNS):
        items, _ = threads.extend(items, thread.id, turns=turns(f"сообщение {index}"), now=NOW)

    kept = threads.find(items, threads.key(1, 101))

    assert len(kept.turns) == threads.MAX_TURNS
    assert kept.turns[-1].text == f"сообщение {threads.MAX_TURNS - 1}"
    assert "первое" not in [turn.text for turn in kept.turns]


# --------------------------------------------------------------------------- #
# Forgetting, out loud
# --------------------------------------------------------------------------- #


def test_the_keys_of_a_dropped_conversation_are_remembered_as_forgotten():
    """So a reply can be answered honestly instead of starting a blank conversation."""
    items, _ = opened(now=NOW - timedelta(days=threads.MAX_AGE_DAYS + 1))

    pruned = threads.prune(items, now=NOW)

    assert threads.find(pruned, threads.key(1, 101)) is None
    assert threads.was_forgotten(pruned, threads.key(1, 101))


def test_every_door_into_a_dropped_conversation_is_remembered():
    items, _ = threads.start(
        threads.Threads(),
        mode="roast",
        turns=turns("запись", "ответ"),
        keys=[threads.key(1, 101), threads.key(1, 102)],
        now=NOW - timedelta(days=threads.MAX_AGE_DAYS + 1),
    )

    pruned = threads.prune(items, now=NOW)

    assert threads.was_forgotten(pruned, threads.key(1, 102))


def test_a_forgotten_key_survives_a_restart(store):
    items, _ = opened(now=NOW - timedelta(days=threads.MAX_AGE_DAYS + 1))
    store.save(threads.prune(items, now=NOW))

    reloaded = threads.ThreadStore(store.path).load(now=NOW)

    assert threads.was_forgotten(reloaded, threads.key(1, 101))


def test_the_forgotten_list_is_bounded_too():
    items = threads.Threads()
    for index in range(threads.MAX_FORGOTTEN + 20):
        items, _ = opened(
            items, message_id=1000 + index, now=NOW - timedelta(days=threads.MAX_AGE_DAYS + 1)
        )
        items = threads.prune(items, now=NOW)

    assert len(items.forgotten) == threads.MAX_FORGOTTEN


def test_an_unreadable_timestamp_keeps_the_conversation_rather_than_ageing_it_out():
    """Destroying data on the strength of a bug is the wrong way to resolve one."""
    items, thread = opened()
    broken = threads.Threads(
        threads=(replace(thread, updated_at="не дата"),),
        next_id=items.next_id,
    )

    assert len(threads.prune(broken, now=NOW).threads) == 1


def test_a_key_is_not_remembered_as_forgotten_twice():
    """Otherwise a repeated key could push the rest of the window out with its own copies."""
    items, thread = opened(now=NOW - timedelta(days=threads.MAX_AGE_DAYS + 1))
    already = threads.Threads(
        threads=items.threads,
        forgotten=(threads.key(1, 101),),
        next_id=items.next_id,
    )

    pruned = threads.prune(already, now=NOW)

    assert pruned.forgotten == (threads.key(1, 101),)
    assert threads.was_forgotten(pruned, threads.key(1, 101))


# --------------------------------------------------------------------------- #
# Ending one on purpose
#
# A draft can be discarded, and the conversation the coach opened about it goes
# with it. That is the same thing pruning does when a bound comes due, asked for
# rather than waited for — which is why it has to leave the same trace behind.
# --------------------------------------------------------------------------- #


def test_a_dropped_conversation_is_gone():
    items, thread = opened()

    left = threads.drop(items, thread.id)

    assert left.threads == ()
    assert threads.find(left, threads.key(1, 101)) is None


def test_every_door_into_a_dropped_conversation_is_remembered_as_forgotten():
    """The messages may still be on screen — a delete can fail — and a reply to one
    of them has to be answered honestly rather than start a blank conversation."""
    items, thread = threads.start(
        threads.Threads(),
        mode="roast",
        turns=turns("запись", "ответ"),
        keys=[threads.key(1, 101), threads.key(1, 102)],
        now=NOW,
    )

    left = threads.drop(items, thread.id)

    assert threads.was_forgotten(left, threads.key(1, 101))
    assert threads.was_forgotten(left, threads.key(1, 102))


def test_dropping_one_conversation_leaves_the_others_alone():
    items, first = opened(message_id=101)
    items, second = opened(items, message_id=201)

    left = threads.drop(items, first.id)

    assert [thread.id for thread in left.threads] == [second.id]
    assert threads.find(left, threads.key(1, 201)) is not None
    assert not threads.was_forgotten(left, threads.key(1, 201))


def test_dropping_a_conversation_that_is_not_there_changes_nothing():
    """The caller holds an id it recorded earlier; the file may have pruned it since."""
    items, _ = opened()

    assert threads.drop(items, "не тот") == items


def test_the_id_counter_is_not_lowered_by_a_drop():
    """An id handed out twice would address two conversations at once."""
    items, thread = opened()

    left = threads.drop(items, thread.id)

    assert left.next_id == items.next_id


def test_a_dropped_conversation_stays_dropped_across_a_restart(store):
    items, thread = opened()
    store.save(threads.drop(items, thread.id))

    reloaded = threads.ThreadStore(store.path).load(now=NOW)

    assert reloaded.threads == ()
    assert threads.was_forgotten(reloaded, threads.key(1, 101))


def test_a_conversation_is_found_by_the_id_its_opener_kept():
    items, thread = opened()

    assert threads.find_by_id(items, thread.id) == thread
    assert threads.find_by_id(items, "не тот") is None


def test_an_address_gives_back_the_message_it_points_at():
    assert threads.message_of(threads.key(1, 101), 1) == 101


def test_an_address_in_another_chat_is_not_a_message_in_this_one():
    """Deleting message 101 of this chat because thread 101 exists in another
    would delete something the coach never sent."""
    assert threads.message_of(threads.key(2, 101), 1) is None


def test_an_address_that_is_not_one_gives_back_nothing():
    assert threads.message_of("1:не число", 1) is None
    assert threads.message_of("мусор", 1) is None
