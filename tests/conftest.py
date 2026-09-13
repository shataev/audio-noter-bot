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


# --------------------------------------------------------------------------- #
# Nothing in this suite talks to a real API, and it should not be able to start
# doing so by accident.
#
# That became possible when a coach answer started asking Notion for the memory
# pages before it answers: a test that has not mocked the transport reaches it.
# What happens then is not a failure — the sync falls back to the file on purpose
# — it is a pile of connection attempts and retry sleeps, and on a machine with a
# route out it is a request to Notion carrying whatever token the environment
# happens to hold.
#
# Appended rather than woven into the file above, and the two imports carry a
# noqa for the same reason: `feat-coach-batch` is working in this file too, and a
# conflict in added lines is cheap where a reordered file is not.
# --------------------------------------------------------------------------- #

import httpx  # noqa: E402
import pytest  # noqa: E402

# Loopback is still allowed: one test in tests/test_notion_http.py deliberately
# points the client at a local socket that never answers, to prove the read
# timeout is enforced. Nothing that leaves this machine is.
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})

_REAL_ASYNC_SEND = httpx.AsyncHTTPTransport.handle_async_request
_REAL_SYNC_SEND = httpx.HTTPTransport.handle_request


class NetworkUsedInATest(RuntimeError):
    """A test reached a real socket. Mock at the boundary instead."""


def _refuse(request):
    raise NetworkUsedInATest(
        f"a test tried to send a real request to {request.url}. Mock at the boundary — "
        f"httpx.MockTransport, or patch the function that makes the call."
    )


def _guarded_async_send(self, request):
    if request.url.host in LOOPBACK:
        return _REAL_ASYNC_SEND(self, request)
    return _refuse(request)


def _guarded_sync_send(self, request):
    if request.url.host in LOOPBACK:
        return _REAL_SYNC_SEND(self, request)
    return _refuse(request)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Replaces httpx's real transports. ``httpx.MockTransport`` does not use them,
    so every test that already mocks at the boundary is unaffected."""
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _guarded_async_send)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _guarded_sync_send)
