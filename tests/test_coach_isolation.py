"""The seam around ``services/coach/``.

The coach is meant to be liftable into a service of its own later. That is only
true while nothing inside the package reaches for Telegram, for Notion, or for a
model SDK directly: it takes plain data from ``bot.py`` and talks to a model
through ``services/ai.py``, and those two are the whole of its contact with the
outside world.

A seam nobody enforces is a seam that closes, and it closes one convenient import
at a time — each of which looks harmless on the day it is written. So this reads
every module in the package with ``ast`` and fails on anything imported that is
not on the list. It is deliberately written against the directory rather than
against a list of module names, so that it covers the modules the next branches
add without anyone remembering to update it.

If a change genuinely needs the rule relaxed, that belongs in a pull request
description, not in an edit to this file.
"""

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "services" / "coach"

# The two the package is allowed to reach for. ``config`` because settings have
# to come from somewhere, ``services.ai`` because that is the one door to a
# model. Both are named in full: ``services`` as a whole is not allowed, which
# is what keeps ``services.notion`` out.
ALLOWED_PROJECT_IMPORTS = frozenset({"config", "services.ai"})

# Named individually as well as excluded by the allowlist, so that the failure
# says which rule was broken rather than "unexpected import".
FORBIDDEN = frozenset(
    {
        "telegram",
        "bot",
        "services.notion",
        "openai",
        "anthropic",
        "httpx",
    }
)

# The modules in this package that may only use the standard library. Everything
# here is pure, synchronous logic with no reason to need anything else, and
# keeping it that way is what makes it testable without a single mock.
STDLIB_ONLY = ("memory.py", "store.py")


def modules():
    return sorted(PACKAGE.rglob("*.py"))


def imported_names(path):
    """Every module named by an import anywhere in ``path``.

    ``ast.walk`` rather than the module body, so that an import tucked inside a
    function is caught too — which is exactly where an inconvenient one would be
    put. Relative imports are skipped: those are the package talking to itself.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def is_forbidden(name):
    return any(name == bad or name.startswith(f"{bad}.") for bad in FORBIDDEN)


def is_allowed(name):
    if name.split(".")[0] in sys.stdlib_module_names:
        return True
    return any(name == good or name.startswith(f"{good}.") for good in ALLOWED_PROJECT_IMPORTS)


def test_the_package_has_modules_to_scan():
    """Otherwise every test below passes by finding nothing."""
    found = [path.name for path in modules()]

    assert PACKAGE.is_dir()
    assert "memory.py" in found
    assert "store.py" in found


@pytest.mark.parametrize("path", modules(), ids=lambda path: path.name)
def test_no_module_imports_telegram_notion_or_a_model_sdk(path):
    offenders = [name for name in imported_names(path) if is_forbidden(name)]

    assert offenders == [], (
        f"{path.name} imports {offenders}. The coach reaches the outside world "
        f"through services/ai.py and through data handed to it by bot.py."
    )


@pytest.mark.parametrize("path", modules(), ids=lambda path: path.name)
def test_no_module_imports_anything_outside_the_allowlist(path):
    unexpected = [name for name in imported_names(path) if not is_allowed(name)]

    assert unexpected == [], (
        f"{path.name} imports {unexpected}, which is neither the standard library "
        f"nor one of {sorted(ALLOWED_PROJECT_IMPORTS)}."
    )


@pytest.mark.parametrize("name", STDLIB_ONLY)
def test_the_memory_and_its_store_use_the_standard_library_only(name):
    path = PACKAGE / name
    outside = [
        imported
        for imported in imported_names(path)
        if imported.split(".")[0] not in sys.stdlib_module_names
    ]

    assert outside == [], f"{name} is meant to need nothing but the standard library"


def test_the_scan_would_notice_a_forbidden_import(tmp_path):
    """The test that stops this file from being decoration.

    Put the import back and the scan has to fail — including the one hidden in a
    function body, which is how it would actually be written.
    """
    offender = tmp_path / "profile.py"
    offender.write_text(
        "import logging\n\n\ndef send():\n    import telegram\n\n    return telegram\n",
        encoding="utf-8",
    )

    names = imported_names(offender)

    assert "telegram" in names
    assert [name for name in names if is_forbidden(name)] == ["telegram"]
    assert [name for name in names if not is_allowed(name)] == ["telegram"]


def test_the_scan_accepts_what_the_package_is_allowed_to_use(tmp_path):
    allowed = tmp_path / "later.py"
    allowed.write_text(
        "import json\n"
        "from datetime import datetime\n"
        "from config import settings\n"
        "from services.ai import chat\n"
        "from .memory import Fact\n",
        encoding="utf-8",
    )

    assert [name for name in imported_names(allowed) if not is_allowed(name)] == []


def test_a_sibling_service_is_not_allowed_just_because_services_ai_is(tmp_path):
    """``services`` as a whole is not the allowlist — one module in it is."""
    offender = tmp_path / "profile.py"
    offender.write_text("from services.notion import append_entry\n", encoding="utf-8")

    assert [name for name in imported_names(offender) if not is_allowed(name)] == [
        "services.notion"
    ]
