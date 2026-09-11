#!/usr/bin/env bash
#
# Runs on the deploy target. `make deploy` pipes it in over ssh
# (`ssh $HOST bash -s -- ...`), so the version that runs is always the one in
# the checkout you deployed from — it is never installed on the server.
#
#   usage: remote-deploy.sh APP_DIR UNIT EXPECTED_REVISION
#
# Exits non-zero, loudly, if the revision that landed is not the one you meant
# to deploy, or if the bot is not demonstrably alive afterwards. `systemctl
# status` on its own is not proof: a process that is about to die on a missing
# import is reported active for the second or two it takes to get there.

# -E so that the ERR trap is inherited: every command below routes its failure
# through fail(), which says what state the checkout was left in.
set -Eeuo pipefail

APP_DIR=${1:?usage: remote-deploy.sh APP_DIR UNIT EXPECTED_REVISION}
UNIT=${2:?usage: remote-deploy.sh APP_DIR UNIT EXPECTED_REVISION}
EXPECTED=${3:?usage: remote-deploy.sh APP_DIR UNIT EXPECTED_REVISION}

# How long to let the bot get through its imports and reach run_polling before
# asking whether it is still alive.
SETTLE_SECONDS=${SETTLE_SECONDS:-10}

# The line bot.py logs once the handlers are registered and polling starts.
START_LINE='Bot started'

# Deploying as root works with no further setup; so does deploying as the user
# that owns APP_DIR, given a NOPASSWD sudo rule for this unit's systemctl and
# journalctl. See "Configuring the target" in the README.
if [ "$(id -u)" -eq 0 ]; then
    as_root() { "$@"; }
else
    as_root() { sudo -n "$@"; }
fi

cd "$APP_DIR"

previous=$(git rev-parse HEAD)
restart_attempted=no

echo "-- currently deployed: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo "-- deploying:          ${EXPECTED:0:7}"

fail() {
    trap - ERR                      # no recursion out of the reporting path
    echo >&2
    echo "DEPLOY FAILED: $1" >&2
    echo >&2

    if [ "$restart_attempted" = yes ]; then
        as_root journalctl -u "$UNIT" -n 40 --no-pager >&2 || true
        echo >&2
    fi

    # The difference that matters when something fails halfway: whether the
    # code on disk still matches the process that is running.
    if [ "$(git rev-parse HEAD)" = "$previous" ]; then
        echo "The checkout is unchanged, still at $previous." >&2
        echo "$UNIT is running whatever it was running before." >&2
    else
        echo "The checkout has MOVED to $(git rev-parse HEAD) and no longer" >&2
        echo "matches the running process. Put it back with:" >&2
        echo >&2
        echo "    make rollback REV=$previous" >&2
    fi
    exit 1
}

# Anything that exits non-zero from here on reports through fail() rather than
# dying silently under set -e.
trap 'fail "\"$BASH_COMMAND\" failed (line $LINENO)"' ERR

echo "-- fetching"
git pull --ff-only

# `make deploy` pushes the branch you are on; this checkout pulls the branch it
# is on. They are not necessarily the same branch, and when they are not, every
# later check passes while nothing you wrote has been deployed.
landed=$(git rev-parse HEAD)
[ "$landed" = "$EXPECTED" ] || fail "$(
    printf '%s\n' \
        "the checkout is at $landed, not the $EXPECTED you deployed." \
        "" \
        "        This checkout is on branch '$(git rev-parse --abbrev-ref HEAD)'." \
        "        Either you deployed from a different branch, or the push has" \
        "        not reached this server's branch. Nothing has been restarted."
)"

echo "-- installing dependencies"
.venv/bin/pip install --quiet --disable-pip-version-check -r requirements.txt

echo "-- restarting $UNIT"
# Epoch form: journalctl parses "@<seconds>" as absolute UTC, so the window is
# not affected by sudo resetting TZ between this line and the read below.
started_at=$(date +%s)
restart_attempted=yes
as_root systemctl restart "$UNIT"

echo "-- waiting ${SETTLE_SECONDS}s for it to settle"
sleep "$SETTLE_SECONDS"

as_root systemctl is-active --quiet "$UNIT" ||
    fail "$UNIT is not running $SETTLE_SECONDS seconds after the restart"

# Restart=always hides a crash loop behind an "active" unit: the unit keeps
# coming back, and NRestarts counts how often. An explicit restart resets the
# counter, so anything above zero here happened since this deploy. Defaults to
# 0 rather than empty on a systemd that does not expose the property, so an
# older server loses the check instead of failing every deploy on it.
restarts=$(as_root systemctl show -p NRestarts --value "$UNIT" || true)
[ "${restarts:-0}" = "0" ] ||
    fail "$UNIT has restarted $restarts time(s) since the deploy — it is crash-looping"

# Collected first rather than piped into grep: grep -q closes the pipe as soon
# as it matches, which under `set -o pipefail` would look like a failure.
log=$(as_root journalctl -u "$UNIT" --since "@$started_at" --no-pager)
grep -qF "$START_LINE" <<<"$log" ||
    fail "$UNIT is running but never logged \"$START_LINE\" — it did not finish starting"

trap - ERR

echo
echo "-- deployed: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo "-- $UNIT is up and logged \"$START_LINE\""
echo "-- previous revision was $previous (make rollback REV=$previous)"
