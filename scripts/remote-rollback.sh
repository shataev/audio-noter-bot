#!/usr/bin/env bash
#
# Runs on the deploy target. `make rollback REV=<sha>` pipes it in over ssh.
#
#   usage: remote-rollback.sh APP_DIR UNIT REVISION
#
# Moves the checkout back to REVISION and restarts, with the same liveness
# check the deploy does — a rollback that reports success into a revision that
# also crash-loops is worse than no rollback at all. `git reset --hard` rather
# than a detached checkout, so that the next `make deploy` fast-forwards back
# onto the branch normally.

set -Eeuo pipefail

APP_DIR=${1:?usage: remote-rollback.sh APP_DIR UNIT REVISION}
UNIT=${2:?usage: remote-rollback.sh APP_DIR UNIT REVISION}
REVISION=${3:?usage: remote-rollback.sh APP_DIR UNIT REVISION}

SETTLE_SECONDS=${SETTLE_SECONDS:-10}
START_LINE='Bot started'

if [ "$(id -u)" -eq 0 ]; then
    as_root() { "$@"; }
else
    as_root() { sudo -n "$@"; }
fi

cd "$APP_DIR"

fail() {
    trap - ERR
    echo >&2
    echo "ROLLBACK FAILED: $1" >&2
    echo >&2
    as_root journalctl -u "$UNIT" -n 40 --no-pager >&2 || true
    echo >&2
    echo "The checkout is at $(git rev-parse HEAD). $UNIT is not verifiably up;" >&2
    echo "this needs a look by hand." >&2
    exit 1
}

trap 'fail "\"$BASH_COMMAND\" failed (line $LINENO)"' ERR

echo "-- rolling back from $(git rev-parse --short HEAD) to $REVISION"
git fetch --quiet origin
git reset --hard "$REVISION"

echo "-- installing dependencies"
.venv/bin/pip install --quiet --disable-pip-version-check -r requirements.txt

echo "-- restarting $UNIT"
started_at=$(date +%s)
as_root systemctl restart "$UNIT"
sleep "$SETTLE_SECONDS"

as_root systemctl is-active --quiet "$UNIT" ||
    fail "$UNIT is not running $SETTLE_SECONDS seconds after the restart"

# Same reasoning as the deploy: Restart=always means an active unit can be a
# crash loop, and START_LINE is logged on every turn of it.
restarts=$(as_root systemctl show -p NRestarts --value "$UNIT" || true)
[ "${restarts:-0}" = "0" ] ||
    fail "$UNIT has restarted $restarts time(s) since the rollback — $REVISION is crash-looping too"

log=$(as_root journalctl -u "$UNIT" --since "@$started_at" --no-pager)
grep -qF "$START_LINE" <<<"$log" ||
    fail "$UNIT is running but never logged \"$START_LINE\" — it did not finish starting"

trap - ERR

echo "-- rolled back to $(git rev-parse --short HEAD), $UNIT is up"
