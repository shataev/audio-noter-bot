"""Import every module in the project.

This is the cheapest test that fails on a real breakage: a syntax error, a
missing dependency, a renamed symbol in an SDK, or a module-level call that
raises. It is also what keeps the suite non-empty, so CI has something to run.
"""

import importlib

import pytest

MODULES = [
    "config",
    "services.formatter",
    "services.notion",
    "services.summary",
    "services.whisper",
    "bot",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    assert importlib.import_module(name) is not None
