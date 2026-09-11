#!/usr/bin/env bash
#
# Runs on the deploy target. `make rollback REV=<sha>` pipes it in over ssh.
#
#   usage: remote-rollback.sh APP_DIR UNIT REVISION
#
# Moves the checkout back to REVISION and restarts, with the same liveness
# check the deploy does. `git reset --hard` rather than a detached checkout, so
# that the next `make deploy` fast-forwards back onto the branch normally.

set -euo pipefail

APP_DIR=${1:?usage: remote-rollback.sh APP_DIR UNIT REVISION}
UNIT=${2:?usage: remote-rollback.sh APP_DIR UNIT REVISION}
REVISION=${3:?usage: remote-rollback.sh APP_DIR UNIT REVISION}

SETTLE_SECONDS=${SETTLE_SECONDS:-10}
START_LINE='Bot started'

if [ "$(id -u)" -eq 0 ]; then
    as_root() { "$@"; }
else
    as_root() { sudo "$@"; }
fi

cd "$APP_DIR"

echo "-- rolling back from $(git rev-parse --short HEAD) to $REVISION"
git fetch --quiet origin
git reset --hard "$REVISION"

echo "-- installing dependencies"
.venv/bin/pip install --quiet --disable-pip-version-check -r requirements.txt

echo "-- restarting $UNIT"
started_at=$(date '+%Y-%m-%d %H:%M:%S')
as_root systemctl restart "$UNIT"
sleep "$SETTLE_SECONDS"

log=$(as_root journalctl -u "$UNIT" --since "$started_at" --no-pager || true)
if ! as_root systemctl is-active --quiet "$UNIT" || ! grep -qF "$START_LINE" <<<"$log"; then
    echo >&2
    echo "ROLLBACK FAILED: $UNIT did not come back up at $REVISION" >&2
    as_root journalctl -u "$UNIT" -n 40 --no-pager >&2 || true
    exit 1
fi

echo "-- rolled back to $(git rev-parse --short HEAD), $UNIT is up"
