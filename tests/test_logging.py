"""The bot token must not reach the log.

The Telegram API carries the token in the request path, and httpx logs every
request it makes at INFO. With the root logger at INFO — which ``bot.py`` sets,
because the bot's own messages are wanted — long-polling writes

    POST https://api.telegram.org/bot<TOKEN>/getUpdates "HTTP/1.1 200 OK"

into the journal every few seconds. Anyone who can read the unit's log then has
the token, which includes the deploy account: it holds a sudo rule for
``journalctl -u noter`` and is otherwise deliberately kept away from the
credentials.
"""

import logging
import os

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import bot  # noqa: E402  (the environment has to exist before config is imported)

SECRET = "8549823108:AAE-a-token-shaped-string"


def test_httpx_requests_are_not_logged_at_info():
    assert bot is not None
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING


def test_a_telegram_url_logged_by_httpx_never_reaches_a_handler(caplog):
    """The property that matters, not just the level that produces it."""
    with caplog.at_level(logging.INFO):
        logging.getLogger("httpx").info(
            'HTTP Request: POST https://api.telegram.org/bot%s/getUpdates "HTTP/1.1 200 OK"',
            SECRET,
        )

    assert SECRET not in caplog.text


def test_the_bot_can_still_log_its_own_messages(caplog):
    """The fix must not silence the log the deploy's liveness check reads."""
    with caplog.at_level(logging.INFO):
        logging.getLogger("bot").info("Bot started")

    assert "Bot started" in caplog.text
