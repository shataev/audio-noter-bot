"""Regression tests for deploy/noter-deploy, the deploy key's forced command.

This is the one part of the project that cannot be exercised for real from
here — it runs on a server nobody has credentials for, reached by a key that
cannot open a shell. So it is run directly, with the request in
``SSH_ORIGINAL_COMMAND`` exactly as sshd would set it, against a throwaway git
checkout with stub ``systemctl``, ``journalctl`` and ``sudo`` on ``PATH``.

Two things are being pinned. That the request is parsed strictly — it is
attacker-controlled input from anyone holding the key. And that a deploy only
reports success when the revision that was asked for is the revision that
landed and the bot is demonstrably alive, saying what state it left behind
whenever it is not.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "deploy" / "noter-deploy"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None,
    reason="needs bash and git",
)

# `systemctl is-active` prints the state and exits non-zero unless it is
# active; this is the exact form the server's sudoers rule allows, without
# --quiet. `journalctl` replays FAKE_LOG. `sudo -n` fails when FAKE_SUDO_FAILS
# is set, which is what a missing NOPASSWD rule looks like with no terminal.
STUBS = {
    "systemctl": """#!/bin/sh
case "$1" in
  restart)   echo restart >> "$FAKE_EVENTS"; exit 0 ;;
  is-active) echo "${FAKE_STATE:-active}"
             [ "${FAKE_STATE:-active}" = active ] && exit 0 || exit 3 ;;
  show)      echo "NRestarts is not permitted by the server's sudo rules" >&2; exit 1 ;;
esac
exit 0
""",
    "journalctl": """#!/bin/sh
printf '%s\\n' "${FAKE_LOG:-Sep 11 00:00:00 h python[1]: 2026-09-11 [INFO] Bot started}"
""",
    "sudo": """#!/bin/sh
if [ -n "$FAKE_SUDO_FAILS" ]; then
  echo "sudo: a password is required" >&2
  exit 1
fi
[ "$1" = "-n" ] && shift
exec "$@"
""",
}

# A healthy window: systemd's own start line, then the bot's. The systemd line
# mentions the unit description, which is why startup lines are counted only
# from non-systemd entries.
HEALTHY_LOG = (
    "Sep 11 08:00:00 h systemd[1]: Started noter.service - Noter Telegram Bot.\n"
    "Sep 11 08:00:01 h python[42]: 2026-09-11 08:00:01,123 [INFO] Bot started"
)

# What a crash loop actually looks like in the journal, as produced by a
# transient Restart=always unit.
CRASH_LOOP_LOG = (
    "Sep 11 08:00:00 h systemd[1]: Started noter.service - Noter Telegram Bot.\n"
    "Sep 11 08:00:00 h python[42]: 2026-09-11 [INFO] Bot started\n"
    "Sep 11 08:00:01 h systemd[1]: noter.service: Main process exited, status=1/FAILURE\n"
    "Sep 11 08:00:03 h systemd[1]: noter.service: Scheduled restart job, "
    "restart counter is at 1.\n"
    "Sep 11 08:00:03 h systemd[1]: Started noter.service - Noter Telegram Bot.\n"
    "Sep 11 08:00:03 h python[51]: 2026-09-11 [INFO] Bot started"
)


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


def advance_origin(server):
    """Put a commit on the upstream that the checkout has yet to pull, so a
    deploy actually moves the code. Returns the new revision."""
    app = server["app"]
    (app / "change").write_text("new code\n")
    _git(app, "add", "-A")
    _git(app, "commit", "--quiet", "-m", "new code")
    _git(app, "push", "--quiet", "origin", "HEAD")
    new = head(app)
    _git(app, "reset", "--quiet", "--hard", "HEAD~1")
    return new


def request(server, ssh_original_command, **fake_env):
    """Invoke the wrapper the way sshd does: no arguments, the client's request
    in SSH_ORIGINAL_COMMAND."""
    env = dict(os.environ)
    env["PATH"] = f"{server['bin']}:{env['PATH']}"
    env["SSH_ORIGINAL_COMMAND"] = ssh_original_command
    env["NOTER_APP_DIR"] = str(server["app"])
    env["NOTER_UNIT"] = "noter"
    env["NOTER_SETTLE_SECONDS"] = "0"
    env["NOTER_SYSTEMCTL"] = "systemctl"  # the stubs, found on PATH
    env["NOTER_JOURNALCTL"] = "journalctl"
    env["FAKE_EVENTS"] = str(server["events"])
    env.setdefault("FAKE_LOG", HEALTHY_LOG)
    env.update(fake_env)
    return subprocess.run(["bash", str(WRAPPER)], capture_output=True, text=True, env=env)


def events(server):
    if not server["events"].exists():
        return []
    return server["events"].read_text().split()


# --- the request is untrusted input -----------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",  # no request at all: must not deploy an implicit HEAD
        "deploy",
        "deploy HEAD",
        "deploy main",
        "deploy " + "0" * 39,  # too short
        "deploy " + "0" * 41,  # too long
        "deploy " + "0" * 39 + "Z",  # not hex
        "deploy " + "0" * 40 + " extra",
        "deploy " + "0" * 40 + "; rm -rf /",
        "deploy $(whoami)",
        "rollback",
        "rollback ../../etc/passwd",
        "systemctl stop noter",
        "bash",
        "/bin/sh",
    ],
)
def test_a_request_that_is_not_a_deploy_or_rollback_is_refused(server, bad):
    result = request(server, bad)
    assert result.returncode == 2, result.stdout
    assert "noter-deploy:" in result.stderr
    assert events(server) == []


def test_the_refusal_does_not_echo_the_request_back(server):
    """It is attacker-controlled and goes straight to the client's terminal."""
    result = request(server, "deploy $(id) \x1b[31mred")
    assert "\x1b[31m" not in result.stderr
    assert "$(id)" not in result.stderr


# --- deploy -----------------------------------------------------------------


def test_healthy_deploy_succeeds(server):
    result = request(server, f"deploy {head(server['app'])}")
    assert result.returncode == 0, result.stderr
    assert "is up and logged" in result.stdout
    assert events(server) == ["restart"]


def test_deploy_of_a_revision_that_did_not_land_fails(server):
    """Every liveness probe would pass, and nothing that was asked for is
    running. This is the check the client's revision exists to make possible."""
    result = request(server, f"deploy {'0' * 40}")
    assert result.returncode == 1
    assert f"not the {'0' * 40} that was requested" in result.stderr
    assert events(server) == []  # gives up before touching the unit
    assert "checkout is unchanged" in result.stderr


def test_deploy_fails_when_the_unit_is_dead(server):
    result = request(server, f"deploy {head(server['app'])}", FAKE_STATE="failed")
    assert result.returncode == 1
    assert "is 'failed', not active" in result.stderr
    # Nothing new was pulled, so there is nothing to roll back to and saying
    # otherwise would send the operator down a dead end.
    assert "checkout is unchanged" in result.stderr
    assert "make rollback REV=" not in result.stderr


def test_failed_deploy_of_new_code_offers_the_rollback(server):
    previous = head(server["app"])
    result = request(server, f"deploy {advance_origin(server)}", FAKE_STATE="failed")
    assert result.returncode == 1
    assert "has MOVED" in result.stderr
    assert f"make rollback REV={previous}" in result.stderr


def test_deploy_fails_on_a_crash_loop_without_asking_systemd_for_nrestarts(server):
    """The server's sudo rules do not permit `systemctl show -p NRestarts`, so
    the crash loop has to be visible in the journal — and it is."""
    result = request(server, f"deploy {head(server['app'])}", FAKE_LOG=CRASH_LOOP_LOG)
    assert result.returncode == 1
    assert "crash-looping" in result.stderr
    assert "NRestarts" not in result.stderr


def test_deploy_fails_when_systemd_gives_up_restarting(server):
    log = HEALTHY_LOG + (
        "\nSep 11 08:00:09 h systemd[1]: noter.service: Start request repeated too quickly."
    )
    result = request(server, f"deploy {head(server['app'])}", FAKE_LOG=log)
    assert result.returncode == 1
    assert "crash-looping" in result.stderr


def test_deploy_fails_when_the_startup_line_never_appears(server):
    log = (
        "Sep 11 08:00:00 h systemd[1]: Started noter.service - Noter Telegram Bot.\n"
        "Sep 11 08:00:01 h python[42]: ModuleNotFoundError: No module named 'telegram'"
    )
    result = request(server, f"deploy {head(server['app'])}", FAKE_LOG=log)
    assert result.returncode == 1
    assert "never logged" in result.stderr
    assert "ModuleNotFoundError" in result.stderr  # the journal tail is shown


def test_systemd_start_line_alone_does_not_count_as_a_startup(server):
    """systemd's "Started noter.service - Noter Telegram Bot." mentions the unit
    description; only the bot's own lines count."""
    log = "Sep 11 08:00:00 h systemd[1]: Started noter.service - Noter Telegram Bot."
    result = request(server, f"deploy {head(server['app'])}", FAKE_LOG=log)
    assert result.returncode == 1
    assert "never logged" in result.stderr


def test_sudo_failure_reports_instead_of_dying_silently(server):
    """No NOPASSWD rule, no terminal. This must not exit through `set -e` at the
    restart, leaving the checkout on new code the running process is not."""
    previous = head(server["app"])
    result = request(server, f"deploy {advance_origin(server)}", FAKE_SUDO_FAILS="1")
    assert result.returncode == 1
    assert "DEPLOY FAILED" in result.stderr
    assert "has MOVED" in result.stderr
    assert f"make rollback REV={previous}" in result.stderr


# --- rollback ---------------------------------------------------------------


def test_rollback_returns_to_the_requested_revision(server):
    app = server["app"]
    first = head(app)
    advance_origin(server)
    _git(app, "pull", "--quiet", "--ff-only")

    result = request(server, f"rollback {first}")
    assert result.returncode == 0, result.stderr
    assert head(app) == first


def test_rollback_into_a_crash_loop_is_not_reported_as_success(server):
    """A rollback that reports success into a revision that also crash-loops is
    a false reassurance at the moment it costs most."""
    result = request(server, f"rollback {head(server['app'])}", FAKE_LOG=CRASH_LOOP_LOG)
    assert result.returncode == 1
    assert "ROLLBACK FAILED" in result.stderr
