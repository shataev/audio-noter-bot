# Where to deploy is not in this file, and not in this repository: it is public.
# Put HOST (and APP_DIR or UNIT, if they are not the defaults) in deploy.mk,
# which is gitignored. See deploy.mk.example.
#
# A value on the command line — `make deploy HOST=...` — overrides deploy.mk.
-include deploy.mk

APP_DIR ?= /opt/noter
UNIT ?= noter

.PHONY: deploy dev stop-dev require-host

deploy: require-host
	git push
	ssh $(HOST) "cd $(APP_DIR) && git pull && systemctl restart $(UNIT) && systemctl status $(UNIT) --no-pager"

dev: require-host
	ssh $(HOST) "systemctl stop $(UNIT)"
	.venv/bin/python bot.py

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
