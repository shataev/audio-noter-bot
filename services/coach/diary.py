"""What a diary entry looks like on this side of the seam.

Nothing under ``services/coach/`` may import ``services.notion``, so the package
cannot use the reader in ``services/summary.py`` and cannot know what a Notion
page is. What it gets instead is this: plain data, handed in by ``bot.py``, with
no dependency on where it was read from or on the shape it was stored in.

There are two things a walk over the diary can produce, and both of them matter
to a pass that has to finish: an entry, and a place where an entry should have
been and could not be read. ``Unreadable`` is the second one, so that a page the
reader could not fetch arrives as a countable event in the same sequence rather
than as an exception that ends the walk half-way through a year of diary.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    """One diary entry, as the coach is handed it.

    ``source`` is what the profile records against a fact it learned here — the
    day and the entry's own title. It is written into the memory file, so it is
    the one field here that ever reaches disk.
    """

    source: str
    title: str
    text: str


@dataclass(frozen=True)
class Unreadable:
    """A day, or an entry, the walk could not read.

    ``where`` is a date or a page id — never diary text, because this is the one
    field of the pair that is written to the log.
    """

    where: str
