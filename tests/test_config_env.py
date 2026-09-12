"""A .env that cannot be read must not take the bot down.

On the server the credentials come from systemd's ``EnvironmentFile``, which is
root-owned and mode 600 so that the deploy account cannot read it. The service
itself runs as an unprivileged user. If a leftover ``.env`` is still sitting in
the working directory — readable only by whoever owned it before — python-dotenv
finds it, tries to open it, and raises ``PermissionError`` out of ``config``'s
import. That is exactly how the bot went into a restart loop the first time it
stopped running as root.
"""

import logging
import os
import subprocess
import sys
import textwrap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_ENV = {
    "TELEGRAM_TOKEN": "test-token",
    "OPENAI_API_KEY": "test-key",
    "NOTION_TOKEN": "test-notion",
    "NOTION_DATABASE_ID": "test-db",
    "ALLOWED_USER_ID": "1",
    "TIMEZONE": "Europe/Moscow",
}


def _run_import_in(cwd, env):
    """Import config in a fresh interpreter, with cwd as the working directory.

    A subprocess rather than importlib.reload: python-dotenv searches upward
    from the working directory, and the settings are read once at import.
    """
    program = textwrap.dedent(
        """
        import config
        print(config.settings.telegram_token)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", program],
        cwd=cwd,
        env={**os.environ, **env, "PYTHONPATH": PROJECT_ROOT},
        capture_output=True,
        text=True,
    )


def test_an_unreadable_dotenv_does_not_stop_the_settings_loading(tmp_path):
    unreadable = tmp_path / ".env"
    unreadable.write_text("TELEGRAM_TOKEN=from-the-file\n")
    unreadable.chmod(0o000)

    result = _run_import_in(tmp_path, REAL_ENV)

    assert result.returncode == 0, result.stderr
    # The environment won, because the file could not be read at all.
    assert result.stdout.strip() == "test-token"
    assert "ignoring an unreadable .env" in result.stderr


def test_a_readable_dotenv_is_still_loaded(tmp_path):
    (tmp_path / ".env").write_text("EXTRA_FROM_FILE=yes\n")

    program = textwrap.dedent(
        """
        import config  # noqa: F401
        import os
        print(os.environ.get("EXTRA_FROM_FILE"))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={**os.environ, **REAL_ENV, "PYTHONPATH": PROJECT_ROOT},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "yes"


def test_the_warning_names_the_file_rather_than_its_contents(caplog):
    """The message goes to the journal, so it must not carry a credential."""
    import config

    with caplog.at_level(logging.WARNING):
        config._load_env_file()

    assert "test-token" not in caplog.text
