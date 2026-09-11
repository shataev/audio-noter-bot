# Where to deploy is not in this file, and not in this repository: it is public.
# Put HOST (and APP_DIR or UNIT, if they are not the defaults) in deploy.mk,
# which is gitignored. See deploy.mk.example.
#
# A value on the command line — `make deploy HOST=...` — overrides deploy.mk.
-include deploy.mk

APP_DIR ?= /opt/noter
UNIT ?= noter
PY ?= .venv/bin/python

.PHONY: run check deploy rollback dev stop-dev require-host require-clean require-dev-tools

# Run the bot on this machine against a development bot of your own, so that
# local work never touches the one on the server. .env.dev holds whatever
# differs from .env — usually just TELEGRAM_TOKEN — and wins, because
# python-dotenv does not overwrite variables that are already in the
# environment. See "Development" in the README.
run:
	@test -f .env.dev || { \
	  echo 'make: .env.dev is missing.'                                        >&2; \
	  echo 'make: It holds the token of a second bot from @BotFather, so that'  >&2; \
	  echo 'make: running the bot here does not fight the one on the server.'   >&2; \
	  echo 'make: See "Development" in the README.'                             >&2; \
	  exit 1; \
	}
	@set -a; . ./.env.dev; set +a; exec $(PY) bot.py

# The same four things CI runs, so a failure is found here rather than in a
# pull request — or, before `make deploy` depended on it, on the server.
check: require-dev-tools
	$(PY) -m compileall -q bot.py config.py services tests
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m pytest -q

# Nothing reaches the network until the tree is clean and the checks pass.
# The revision is passed through to the server, which refuses to call a deploy
# successful unless that is the revision it ends up on.
deploy: require-host require-clean check
	git push
	@echo "-- deploying $$(git rev-parse --short HEAD) to $(HOST):$(APP_DIR)"
	@ssh $(HOST) bash -s -- '$(APP_DIR)' '$(UNIT)' "$$(git rev-parse HEAD)" < scripts/remote-deploy.sh

# Put the server back on an earlier revision. `make deploy` prints the one it
# replaced, and the remote script repeats it when a deploy fails.
rollback: require-host
	@if [ -z "$(REV)" ]; then \
	  echo 'make: usage: make rollback REV=<git-revision>' >&2; \
	  exit 1; \
	fi
	@ssh $(HOST) bash -s -- '$(APP_DIR)' '$(UNIT)' '$(REV)' < scripts/remote-rollback.sh

# Deprecated: running the bot locally against the production token means
# stopping production first, and leaving it stopped if this command is
# interrupted. Set up a second bot instead — see "Development" in the README.
dev: require-host
	@echo '!! make dev STOPS THE BOT ON $(HOST). It stays stopped until you'   >&2
	@echo '!! run `make stop-dev`, including if this command is interrupted or' >&2
	@echo '!! this machine sleeps. Nobody is watching the journal for you.'     >&2
	@echo '!!'                                                                  >&2
	@echo '!! The fix is a second bot token and a local .env of your own; the'  >&2
	@echo '!! README section "Development" has the five minutes it takes.'      >&2
	@echo '!!'                                                                  >&2
	@echo '!! Ctrl-C now, or wait 5 seconds.'                                   >&2
	@sleep 5
	ssh $(HOST) "systemctl stop $(UNIT)"
	$(PY) bot.py

stop-dev: require-host
	ssh $(HOST) "systemctl start $(UNIT)"

# Fail before anything touches the network, with an explanation rather than an
# ssh error about a host called "".
require-host:
	@if [ -n "$(HOST)" ]; then exit 0; fi; \
	{ \
	  echo 'make: HOST is not set, so there is nothing to deploy to.'; \
	  echo; \
	  echo 'The deploy target is deliberately not in this repository — it is public.'; \
	  echo 'Create deploy.mk, which is gitignored:'; \
	  echo; \
	  echo '    cp deploy.mk.example deploy.mk'; \
	  echo '    $$EDITOR deploy.mk        # set HOST'; \
	  echo; \
	  echo 'HOST is an ssh destination: an ssh_config alias (preferred) or user@host.'; \
	  echo 'For a one-off:  make deploy HOST=<ssh-destination>'; \
	} >&2; \
	exit 1

# Deploying pushes this branch and the server pulls its own; the files on your
# disk are never copied anywhere. Uncommitted work is therefore not deployed,
# and the server would be running something other than what you are reading.
require-clean:
	@if [ -z "$$(git status --porcelain)" ]; then exit 0; fi; \
	{ \
	  echo 'make: the working tree has uncommitted changes.'; \
	  echo; \
	  echo 'Deploying pushes commits and the server pulls them; nothing is copied'; \
	  echo 'from your disk. Anything uncommitted would simply not be deployed, and'; \
	  echo 'the server would run something other than what you are looking at.'; \
	  echo 'Commit or discard these first:'; \
	  echo; \
	  git status --short; \
	} >&2; \
	exit 1

require-dev-tools:
	@$(PY) -c 'import ruff, pytest' 2>/dev/null || { \
	  echo 'make: ruff and pytest are not installed in $(PY).' >&2; \
	  echo 'make:     pip install -r requirements-dev.txt'     >&2; \
	  exit 1; \
	}
