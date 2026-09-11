#!/usr/bin/env bash
#
# Runs on the deploy target. `make deploy` pipes it in over ssh
# (`ssh $HOST bash -s -- ...`), so the version that runs is always the one in
# the checkout you deployed from — it is never installed on the server.
#
#   usage: remote-deploy.sh APP_DIR UNIT
#
# Exits non-zero, loudly, if the bot is not demonstrably alive afterwards.
# `systemctl status` on its own is not proof: a process that is about to die on
# a missing import is reported active for the second or two it takes to get
# there.

set -euo pipefail

APP_DIR=${1:?usage: remote-deploy.sh APP_DIR UNIT}
UNIT=${2:?usage: remote-deploy.sh APP_DIR UNIT}

# How long to let the bot get through its imports and reach run_polling before
# asking whether it is still alive.
SETTLE_SECONDS=${SETTLE_SECONDS:-10}

# The line bot.py logs once the handlers are registered and polling starts.
START_LINE='Bot started'

# Deploying as root still works; so does deploying as an unprivileged user who
# is allowed to run systemctl and journalctl through sudo.
if [ "$(id -u)" -eq 0 ]; then
    as_root() { "$@"; }
else
    as_root() { sudo "$@"; }
fi

cd "$APP_DIR"

previous=$(git rev-parse HEAD)
echo "-- currently deployed: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"

fail() {
    echo >&2
    echo "DEPLOY FAILED: $1" >&2
    echo >&2
    as_root journalctl -u "$UNIT" -n 40 --no-pager >&2 || true
    echo >&2
    echo "The previous revision was $previous — roll back with:" >&2
    echo "    make rollback REV=$previous" >&2
    exit 1
}

echo "-- fetching"
git pull --ff-only

echo "-- installing dependencies"
.venv/bin/pip install --quiet --disable-pip-version-check -r requirements.txt

echo "-- restarting $UNIT"
started_at=$(date '+%Y-%m-%d %H:%M:%S')
as_root systemctl restart "$UNIT"

echo "-- waiting ${SETTLE_SECONDS}s for it to settle"
sleep "$SETTLE_SECONDS"

as_root systemctl is-active --quiet "$UNIT" ||
    fail "$UNIT is not running $SETTLE_SECONDS seconds after the restart"

# Restart=always hides a crash loop behind an "active" unit: the unit keeps
# coming back, and NRestarts counts how often. An explicit restart resets the
# counter, so anything above zero here happened since this deploy.
restarts=$(as_root systemctl show -p NRestarts --value "$UNIT")
[ "$restarts" = "0" ] ||
    fail "$UNIT has restarted $restarts time(s) since the deploy — it is crash-looping"

# Collected first rather than piped into grep: grep -q closes the pipe as soon
# as it matches, which under `set -o pipefail` would look like a failure.
log=$(as_root journalctl -u "$UNIT" --since "$started_at" --no-pager)
grep -qF "$START_LINE" <<<"$log" ||
    fail "$UNIT is running but never logged \"$START_LINE\" — it did not finish starting"

echo
echo "-- deployed: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo "-- $UNIT is up and logged \"$START_LINE\""
echo "-- previous revision was $previous (make rollback REV=$previous)"
