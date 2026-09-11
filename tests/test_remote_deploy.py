"""Regression tests for the deploy scripts.

These are the only part of the project that cannot be exercised for real from
here — they run on a server nobody has credentials for. So they are run against
a throwaway git checkout with stub ``systemctl``, ``journalctl`` and ``sudo`` on
``PATH``, which is enough to pin down the behaviour that matters: that a deploy
only reports success when the revision you asked for is the revision that
landed and the bot is demonstrably alive, and that every failure says what state
it left the checkout in.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "scripts" / "remote-deploy.sh"
ROLLBACK = REPO / "scripts" / "remote-rollback.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None,
    reason="needs bash and git",
)

STUBS = {
    # $1 is the subcommand: restart / is-active / show.
    "systemctl": """#!/bin/sh
case "$1" in
  restart)   echo restart >> "$FAKE_EVENTS"; exit 0 ;;
  is-active) [ "${FAKE_STATE:-active}" = active ] && exit 0 || exit 3 ;;
  show)      echo "${FAKE_RESTARTS-0}" ; exit 0 ;;
esac
exit 0
""",
    "journalctl": """#!/bin/sh
printf '%s\\n' "${FAKE_LOG:-Jan 01 00:00:00 host python[1]: Bot started}"
""",
    # `sudo -n` as the scripts call it. FAKE_SUDO_FAILS reproduces a missing
    # NOPASSWD rule, which is what a non-root deploy hits with no TTY.
    "sudo": """#!/bin/sh
if [ -n "$FAKE_SUDO_FAILS" ]; then
  echo "sudo: a terminal is required to authenticate" >&2
  exit 1
fi
[ "$1" = "-n" ] && shift
exec "$@"
""",
}


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def server(tmp_path):
    """A stand-in for the checkout on the VPS, plus stubs and an upstream."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in STUBS.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)

    origin = tmp_path / "origin.git"
    app = tmp_path / "app"
    _git(tmp_path, "init", "--quiet", "--bare", str(origin))
    _git(tmp_path, "clone", "--quiet", str(origin), str(app))
    _git(app, "config", "user.email", "test@example.com")
    _git(app, "config", "user.name", "Test")

    (app / "requirements.txt").write_text("# stub\n")
    venv_bin = app / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "pip").write_text('#!/bin/sh\necho "[stub] pip $*" >&2\n')
    (venv_bin / "pip").chmod(0o755)

    _git(app, "add", "-A")
    _git(app, "commit", "--quiet", "-m", "initial")
    _git(app, "push", "--quiet", "-u", "origin", "HEAD")

    return {"tmp": tmp_path, "app": app, "bin": bin_dir, "events": tmp_path / "events"}


def head(app):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=app, capture_output=True, text=True, check=True
    ).stdout.strip()


def run(script, server, *args, **fake_env):
    env = dict(os.environ)
    env["PATH"] = f"{server['bin']}:{env['PATH']}"
    env["SETTLE_SECONDS"] = "0"
    env["FAKE_EVENTS"] = str(server["events"])
    env.update(fake_env)
    return subprocess.run(
        ["bash", str(script), str(server["app"]), "noter", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def events(server):
    if not server["events"].exists():
        return []
    return server["events"].read_text().split()


def advance_origin(server):
    """Put a commit on the upstream that the checkout has yet to pull, so the
    deploy actually moves the code. Returns the new revision."""
    app = server["app"]
    (app / "change").write_text("new code\n")
    _git(app, "add", "-A")
    _git(app, "commit", "--quiet", "-m", "new code")
    _git(app, "push", "--quiet", "origin", "HEAD")
    new = head(app)
    _git(app, "reset", "--quiet", "--hard", "HEAD~1")
    return new


# --- deploy -----------------------------------------------------------------


def test_healthy_deploy_succeeds(server):
    result = run(DEPLOY, server, head(server["app"]))
    assert result.returncode == 0, result.stderr
    assert "is up and logged" in result.stdout
    assert events(server) == ["restart"]


def test_deploy_of_a_revision_that_did_not_land_fails(server):
    """The check that was missing: every liveness probe passes, and nothing
    the caller asked for was actually deployed."""
    result = run(DEPLOY, server, "0" * 40)
    assert result.returncode == 1
    assert "not the 0000000000000000000000000000000000000000 you deployed" in result.stderr
    # It must give up before touching the unit, not after.
    assert events(server) == []
    assert "checkout is unchanged" in result.stderr


def test_deploy_fails_when_the_unit_is_dead(server):
    result = run(DEPLOY, server, head(server["app"]), FAKE_STATE="inactive")
    assert result.returncode == 1
    assert "is not running" in result.stderr
    # Nothing new was pulled, so there is nothing to roll back to and saying
    # otherwise would send the operator down a dead end.
    assert "checkout is unchanged" in result.stderr
    assert "make rollback REV=" not in result.stderr


def test_failed_deploy_of_new_code_offers_the_rollback(server):
    previous = head(server["app"])
    result = run(DEPLOY, server, advance_origin(server), FAKE_STATE="inactive")
    assert result.returncode == 1
    assert "has MOVED" in result.stderr
    assert f"make rollback REV={previous}" in result.stderr


def test_deploy_fails_when_the_unit_is_crash_looping(server):
    result = run(DEPLOY, server, head(server["app"]), FAKE_RESTARTS="3")
    assert result.returncode == 1
    assert "crash-looping" in result.stderr


def test_deploy_fails_when_the_startup_line_never_appears(server):
    result = run(DEPLOY, server, head(server["app"]), FAKE_LOG="ModuleNotFoundError: telegram")
    assert result.returncode == 1
    assert "never logged" in result.stderr
    assert "ModuleNotFoundError" in result.stderr  # the journal tail is shown


def test_missing_nrestarts_property_does_not_fail_the_deploy(server):
    """An older systemd that does not expose NRestarts loses the check rather
    than failing every deploy."""
    result = run(DEPLOY, server, head(server["app"]), FAKE_RESTARTS="")
    assert result.returncode == 0, result.stderr


def test_sudo_failure_reports_instead_of_dying_silently(server):
    """No NOPASSWD rule, no TTY. This used to exit through `set -e` at the
    restart, so the checkout had already moved to the new code while the old
    process kept running — and nothing said so, or offered a way back."""
    previous = head(server["app"])
    result = run(DEPLOY, server, advance_origin(server), FAKE_SUDO_FAILS="1")
    assert result.returncode == 1
    assert "DEPLOY FAILED" in result.stderr
    assert "has MOVED" in result.stderr
    assert f"make rollback REV={previous}" in result.stderr


# --- rollback ---------------------------------------------------------------


def test_rollback_returns_to_the_requested_revision(server):
    app = server["app"]
    first = head(app)
    (app / "new").write_text("x\n")
    _git(app, "add", "-A")
    _git(app, "commit", "--quiet", "-m", "second")
    _git(app, "push", "--quiet", "origin", "HEAD")

    result = run(ROLLBACK, server, first)
    assert result.returncode == 0, result.stderr
    assert head(app) == first


def test_rollback_into_a_crash_loop_is_not_reported_as_success(server):
    """A rollback that reports success into a revision that also crash-loops is
    a false reassurance at the worst possible moment."""
    result = run(ROLLBACK, server, head(server["app"]), FAKE_RESTARTS="2")
    assert result.returncode == 1
    assert "crash-looping too" in result.stderr
