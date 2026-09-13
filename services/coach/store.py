"""The one file the coach's two lists live in.

A single JSON document holding the author profile and the behaviour rules, each
with its own id counter and its own tombstones::

    {"version": 1,
     "profile": {"next_id": 42, "facts": [...], "tombstones": [...]},
     "rules":   {"next_id": 7,  "facts": [...], "tombstones": [...]}}

Three properties matter more than anything else this module does.

*Writes are atomic.* The file is rewritten after every diary entry, and a
half-written profile is worse than no profile at all. Every write goes to a
temporary file in the same directory and arrives at its real name through
``os.replace``, which is atomic on POSIX; the real path is never opened for
writing, so a reader either sees the previous document or the new one.

*Loads are tolerant.* A missing file is an empty store. Unknown keys survive a
round trip. A fact written before ``kind``, ``created_at`` or ``sources`` existed
loads with those filled in. The store has to survive its own future, because the
alternative is a version bump that loses a year of accumulated facts.

*Serialisation is stable.* Fixed key order, ``ensure_ascii=False``, two-space
indent, trailing newline. The owner reads and edits this file by hand, and a diff
should show what changed and nothing else.

The path is a constructor argument. This module does not read ``config.py`` and
has no opinion about where the bot keeps its state.
"""

import json
import logging
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from .memory import (
    PROFILE_KINDS,
    RULE_KINDS,
    Fact,
    MemoryList,
    Tombstone,
    fallback_kind,
)

logger = logging.getLogger(__name__)

VERSION = 1
# How many snapshots are kept. Three is enough to undo a bad retrospective
# rebuild and small enough that nobody has to think about disk.
MAX_SNAPSHOTS = 3

# What a timestamp that was never recorded reads as. The epoch rather than the
# time of the load, because a fact written before this field existed is old, and
# stamping it "now" would tell every later reader the opposite of the truth.
UNKNOWN_STAMP = "1970-01-01T00:00:00+00:00"

_UNSAFE_LABEL = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_LABEL_CHARS = 40


class StoreError(RuntimeError):
    """The file exists and is not this store's document.

    Raised for a file that is not valid JSON, or whose top level is not an
    object. Damage *inside* a document of the right shape is repaired instead —
    see :meth:`MemoryStore.load` — but a file that is not the store's format at
    all is never treated as an empty store, because the next save would then
    overwrite whatever it really was.
    """


@dataclass(frozen=True)
class StoreData:
    """Everything the file holds.

    ``extra`` carries top-level keys this version does not know about, so that a
    document written by a later version survives being read and written by this
    one.
    """

    profile: MemoryList = MemoryList()
    rules: MemoryList = MemoryList()
    version: int = VERSION
    extra: Mapping[str, object] = field(default_factory=dict)


def _int(raw: object, default: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return default
    return raw


def _text(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    return raw.strip() or None


def _sources(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(item.strip() for item in raw if isinstance(item, str) and item.strip())


def _fact_from_json(raw: object, *, default_kind: str) -> Fact | None:
    """One fact, with anything it predates filled in. ``None`` if it has no text."""
    if not isinstance(raw, dict):
        return None
    text = _text(raw.get("text"))
    if text is None:
        return None

    created_at = _text(raw.get("created_at")) or UNKNOWN_STAMP
    # An unrecognised kind is kept rather than flattened: this version's
    # vocabulary is not the last word on what a kind may be.
    return Fact(
        id=_text(raw.get("id")) or "",
        text=text,
        kind=_text(raw.get("kind")) or default_kind,
        created_at=created_at,
        updated_at=_text(raw.get("updated_at")) or created_at,
        sources=_sources(raw.get("sources")),
        key=_text(raw.get("key")),
    )


def _tombstone_from_json(raw: object, *, default_kind: str) -> Tombstone | None:
    if not isinstance(raw, dict):
        return None
    fact = _fact_from_json(raw.get("fact"), default_kind=default_kind)
    if fact is None:
        return None
    return Tombstone(
        fact=fact,
        deleted_at=_text(raw.get("deleted_at")) or UNKNOWN_STAMP,
        reason=_text(raw.get("reason")),
    )


def _numeric(fact_id: str) -> int | None:
    return int(fact_id) if fact_id.isdigit() else None


def _list_from_json(raw: object, *, default_kind: str, name: str) -> MemoryList:
    """One list, repaired as far as it can be without inventing anything.

    Three repairs happen here, all of them towards keeping data:

    * a fact with no id, or with an id another fact already has, is given a fresh
      one rather than dropped — it is unaddressable as it stands, and an
      unaddressable fact in the prompt is still a fact the coach knows;
    * the counter is raised past every id in the file, live or tombstoned, so
      that a hand-edited or truncated ``next_id`` cannot hand out an id that is
      already in use. It is only ever raised, never lowered;
    * an entry with no text at all is dropped, because there is nothing to keep.
    """
    body = raw if isinstance(raw, dict) else {}
    raw_facts = body.get("facts") if isinstance(body.get("facts"), list) else []
    raw_tombs = body.get("tombstones") if isinstance(body.get("tombstones"), list) else []

    facts: list[Fact] = []
    tombstones: list[Tombstone] = []
    dropped = 0
    for item in raw_facts:
        fact = _fact_from_json(item, default_kind=default_kind)
        if fact is None:
            dropped += 1
        else:
            facts.append(fact)
    for item in raw_tombs:
        stone = _tombstone_from_json(item, default_kind=default_kind)
        if stone is None:
            dropped += 1
        else:
            tombstones.append(stone)

    highest = 0
    for fact_id in [fact.id for fact in facts] + [stone.fact.id for stone in tombstones]:
        number = _numeric(fact_id)
        if number is not None:
            highest = max(highest, number)
    next_id = max(_int(body.get("next_id"), 1), highest + 1)

    repaired = 0
    taken: set[str] = set()
    for index, fact in enumerate(facts):
        if fact.id and fact.id not in taken:
            taken.add(fact.id)
            continue
        facts[index] = replace(fact, id=str(next_id))
        taken.add(str(next_id))
        next_id += 1
        repaired += 1

    if repaired or dropped:
        logger.warning(
            "coach store: %s loaded with %d id(s) reassigned and %d unusable entr(ies) dropped",
            name,
            repaired,
            dropped,
        )

    return MemoryList(facts=tuple(facts), tombstones=tuple(tombstones), next_id=next_id)


def _fact_to_json(fact: Fact) -> dict[str, object]:
    """Fixed key order, and ``key`` omitted when there isn't one.

    Order is spelled out rather than sorted because this is the shape a human
    reads: the id first, the sentence second, the bookkeeping after it.
    """
    body: dict[str, object] = {
        "id": fact.id,
        "text": fact.text,
        "kind": fact.kind,
        "created_at": fact.created_at,
        "updated_at": fact.updated_at,
        "sources": list(fact.sources),
    }
    if fact.key is not None:
        body["key"] = fact.key
    return body


def _tombstone_to_json(stone: Tombstone) -> dict[str, object]:
    body: dict[str, object] = {
        "fact": _fact_to_json(stone.fact),
        "deleted_at": stone.deleted_at,
    }
    if stone.reason is not None:
        body["reason"] = stone.reason
    return body


def _list_to_json(items: MemoryList) -> dict[str, object]:
    return {
        "next_id": items.next_id,
        "facts": [_fact_to_json(fact) for fact in items.facts],
        "tombstones": [_tombstone_to_json(stone) for stone in items.tombstones],
    }


def _dumps(data: StoreData) -> str:
    body: dict[str, object] = {
        "version": data.version,
        "profile": _list_to_json(data.profile),
        "rules": _list_to_json(data.rules),
    }
    # Unknown keys go last and in a fixed order, so that adding one later moves
    # nothing that was already in the file.
    for name in sorted(data.extra):
        if name not in body:
            body[name] = data.extra[name]
    return json.dumps(body, ensure_ascii=False, indent=2) + "\n"


def _safe_label(label: str) -> str:
    """A snapshot label that can only ever name a sibling of the store file."""
    safe = _UNSAFE_LABEL.sub("-", label).strip("-")[:_MAX_LABEL_CHARS]
    return safe or "unlabelled"


class MemoryStore:
    """Reads and writes the coach's memory file. Holds no state of its own."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> StoreData:
        """The stored lists, or empty ones if there is nothing stored yet."""
        try:
            raw_text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return StoreData()

        if not raw_text.strip():
            logger.warning("coach store: %s is empty, starting from nothing", self._path)
            return StoreData()

        try:
            body = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise StoreError(f"{self._path} is not valid JSON: {error}") from error
        if not isinstance(body, dict):
            raise StoreError(f"{self._path} does not hold a JSON object")

        version = _int(body.get("version"), VERSION)
        if version > VERSION:
            # Loaded anyway. Refusing would strand the owner on a newer file with
            # no way back, and every field this version does not know about is
            # carried through ``extra``.
            logger.warning(
                "coach store: %s was written by version %d, reading it as version %d",
                self._path,
                version,
                VERSION,
            )

        data = StoreData(
            profile=_list_from_json(
                body.get("profile"),
                default_kind=fallback_kind(PROFILE_KINDS),
                name="profile",
            ),
            rules=_list_from_json(
                body.get("rules"),
                default_kind=fallback_kind(RULE_KINDS),
                name="rules",
            ),
            version=version,
            extra={
                name: value
                for name, value in body.items()
                if name not in ("version", "profile", "rules")
            },
        )
        logger.info(
            "coach store: loaded %d profile fact(s) and %d rule(s) from %s",
            len(data.profile.facts),
            len(data.rules.facts),
            self._path,
        )
        return data

    def save(self, data: StoreData) -> None:
        """Write the store, atomically.

        The document is serialised in full before anything is opened, so a
        failure in serialisation cannot leave a partial file behind either.
        """
        text = _dumps(data)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomically(self._path, text)
        logger.info(
            "coach store: wrote %d profile fact(s) and %d rule(s) to %s",
            len(data.profile.facts),
            len(data.rules.facts),
            self._path,
        )

    def snapshot(self, label: str) -> Path | None:
        """Copy the current file to a labelled sibling, keeping the newest three.

        Returns the snapshot's path, or ``None`` when there is no file to copy
        yet. A rebuild that goes badly is then one file move away from being
        undone, which is the only reason it is safe to let one run at all.
        """
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

        target = self._path.with_name(f"{self._path.stem}.snapshot-{_safe_label(label)}.json")
        _write_text_atomically(target, text)
        self._prune_snapshots()
        logger.info("coach store: snapshot written to %s", target)
        return target

    def snapshots(self) -> list[Path]:
        """Existing snapshots, newest first."""
        found = list(self._path.parent.glob(f"{self._path.stem}.snapshot-*.json"))
        return sorted(found, key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)

    def _prune_snapshots(self) -> None:
        for stale in self.snapshots()[MAX_SNAPSHOTS:]:
            try:
                stale.unlink()
            except OSError:  # pragma: no cover - a snapshot we cannot remove is not fatal
                logger.warning("coach store: could not remove the old snapshot %s", stale)


def _write_text_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` without ``path`` ever holding a partial document.

    The temporary file is a sibling so that ``os.replace`` stays within one
    filesystem, and is created with mode 0600: these are diary facts, and the
    deploy account can read the directory they live in.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
