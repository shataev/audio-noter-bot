# The server this deploys to is not named in this repository, and does not need
# to be: HOST is the name of an entry in your own ~/.ssh/config, which holds the
# address, the login and the key. Put it in deploy.mk, which is gitignored —
# see deploy.mk.example — or pass it for one command: make deploy HOST=...
-include deploy.mk

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
#
# The deploy key on the server is a forced-command key: it cannot reach a shell,
# and the request below arrives at /usr/local/bin/noter-deploy as
# SSH_ORIGINAL_COMMAND. Sending the revision explicitly is what lets the server
# refuse to call a deploy successful unless that is the revision it ends up on.
deploy: require-host require-clean check
	git push
	@echo "-- deploying $$(git rev-parse --short HEAD) via $(HOST)"
	ssh $(HOST) deploy $$(git rev-parse HEAD)

# Put the server back on an earlier revision. `make deploy` prints the one it
# replaced, and the server repeats it when a deploy fails.
rollback: require-host
	@if [ -z "$(REV)" ]; then \
	  echo 'make: usage: make rollback REV=<git-revision>' >&2; \
	  exit 1; \
	fi
	ssh $(HOST) rollback $(REV)

# Kept as signposts. Both stopped the bot on the server so that a local run
# would not fight it for the same Telegram token, and both are now impossible:
# the deploy key accepts two requests and neither of them is "stop".
dev stop-dev:
	@{ \
	  echo 'make: `make $@` is gone.'; \
	  echo; \
	  echo 'It used to stop the bot on the server so that a local run would not'; \
	  echo 'fight it for the same Telegram token — and left it stopped whenever'; \
	  echo 'the command was interrupted. The deploy key can no longer stop'; \
	  echo 'anything; it can deploy and it can roll back.'; \
	  echo; \
	  echo 'Use a second bot instead. `make run` starts it, and production is'; \
	  echo 'never touched. See "Development" in the README.'; \
	} >&2; \
	exit 1

# Fail before anything touches the network, with an explanation rather than an
# ssh error about a host called "".
require-host:
	@if [ -n "$(HOST)" ]; then exit 0; fi; \
	{ \
	  echo 'make: HOST is not set, so there is nothing to deploy to.'; \
	  echo; \
	  echo 'HOST is the name of an entry in your ~/.ssh/config — that is where'; \
	  echo 'the address, the login and the key live, which is why none of them'; \
	  echo 'appear in this repository. Create deploy.mk, which is gitignored:'; \
	  echo; \
	  echo '    cp deploy.mk.example deploy.mk'; \
	  echo '    $$EDITOR deploy.mk        # set HOST to your ssh alias'; \
	  echo; \
	  echo 'For a one-off:  make deploy HOST=<your-ssh-alias>'; \
	  echo 'See "Deployment" in the README for the matching ~/.ssh/config entry.'; \
	} >&2; \
	exit 1

# Deploying pushes this branch and the server pulls it; the files on your disk
# are never copied anywhere. Uncommitted work is therefore not deployed, and the
# server would be running something other than what you are reading.
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
