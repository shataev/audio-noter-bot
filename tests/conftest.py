"""Test-wide setup.

``config.py`` reads its settings from ``os.environ`` at import time and raises
``KeyError`` when one is missing, so importing any project module without these
set fails during collection. Fill them with obvious dummy values before any test
module is imported.

``setdefault`` is used throughout: test modules that set the same variables
themselves — so they can be run standalone — are unaffected, and a real value
already in the environment always wins.
"""

import os

DUMMY_ENV = {
    "TELEGRAM_TOKEN": "test-token",
    "OPENAI_API_KEY": "test-key",
    "NOTION_TOKEN": "test-notion",
    "NOTION_DATABASE_ID": "test-db",
    "ALLOWED_USER_ID": "1",
    "TIMEZONE": "Europe/Moscow",
}

for _name, _value in DUMMY_ENV.items():
    os.environ.setdefault(_name, _value)
