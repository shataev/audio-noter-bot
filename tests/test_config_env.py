"""A .env that cannot be read must not take the bot down.

On the server the credentials come from systemd's ``EnvironmentFile``, which is
root-owned and mode 600 so that the deploy account cannot read it. The service
itself runs as an unprivileged user. If a leftover ``.env`` is still sitting in
the working directory — readable only by whoever owned it before — python-dotenv
finds it, tries to open it, and raises ``PermissionError`` out of ``config``'s
import. That is exactly how the bot went into a restart loop the first time it
stopped running as root.

The provider settings are checked here for the same reason: they are read at
import as well, and a configuration that cannot work has to say so at startup
rather than at the first voice message.
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


def _import_and_print(expression, env, cwd):
    """Import config in a fresh interpreter and print one expression from it.

    A subprocess rather than importlib.reload, for the same reason as above: the
    settings are read once, at import, and the provider branch with them.
    """
    program = textwrap.dedent(
        f"""
        import config
        print({expression})
        """
    )
    return subprocess.run(
        [sys.executable, "-c", program],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def _env(**overrides):
    """The real environment plus overrides, with the provider variables cleared.

    The machine running the tests may well have an ANTHROPIC_API_KEY of its own
    exported; a test about a missing key has to be sure it is missing.
    """
    env = {
        **os.environ,
        **REAL_ENV,
        "PYTHONPATH": PROJECT_ROOT,
    }
    for name in ("AI_PROVIDER", "ANTHROPIC_API_KEY", "FORMATTER_MODEL", "SUMMARY_MODEL"):
        env.pop(name, None)
    for name, value in overrides.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env


def test_the_default_provider_is_openai(tmp_path):
    result = _import_and_print("config.settings.ai_provider", _env(), tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "openai"


def test_anthropic_without_its_key_fails_at_import_naming_both(tmp_path):
    """A key that is only required because a setting changed has to say so.

    "KeyError: ANTHROPIC_API_KEY" on a server that has never spoken to Anthropic
    is a puzzle. The message has to name the variable and the provider that
    asked for it.
    """
    result = _import_and_print(
        "config.settings.ai_provider", _env(AI_PROVIDER="anthropic"), tmp_path
    )

    assert result.returncode != 0
    assert "ANTHROPIC_API_KEY is required when AI_PROVIDER=anthropic" in result.stderr


def test_the_anthropic_key_is_not_required_when_the_provider_is_openai(tmp_path):
    """The VPS has one API key today, and this must not ask it for a second."""
    result = _import_and_print("config.settings.ai_provider", _env(AI_PROVIDER="openai"), tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "openai"


def test_an_unknown_provider_is_refused_by_name(tmp_path):
    result = _import_and_print("config.settings.ai_provider", _env(AI_PROVIDER="gemini"), tmp_path)

    assert result.returncode != 0
    assert "AI_PROVIDER must be" in result.stderr
    assert "'gemini'" in result.stderr


def test_transcription_stays_on_openai_under_the_anthropic_provider(tmp_path):
    """There is no Anthropic audio endpoint, so the setting cannot reach this.

    The model has to stay an OpenAI one and the key it is sent with has to stay
    the OpenAI key, or voice messages stop working the moment the provider is
    switched.
    """
    env = _env(AI_PROVIDER="anthropic", ANTHROPIC_API_KEY="test-anthropic")

    result = _import_and_print(
        "(config.settings.transcription_model, config.settings.openai_api_key)", env, tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "('gpt-transcribe', 'test-key')"


def test_the_chat_models_default_to_claude_under_the_anthropic_provider(tmp_path):
    """And to the ids exactly as written: Anthropic ids never carry a date."""
    env = _env(AI_PROVIDER="anthropic", ANTHROPIC_API_KEY="test-anthropic")

    result = _import_and_print(
        "(config.settings.formatter_model, config.settings.summary_model,"
        " config.settings.coach_model)",
        env,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "('claude-opus-5', 'claude-opus-5', 'claude-opus-5')"


def test_the_chat_models_default_to_the_cheap_openai_ones(tmp_path):
    """Formatting and summarising are mechanical; only the coach reasons."""
    result = _import_and_print(
        "(config.settings.formatter_model, config.settings.summary_model)", _env(), tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "('gpt-4o-mini', 'gpt-4o-mini')"


def test_an_explicit_model_wins_over_the_provider_default(tmp_path):
    env = _env(
        AI_PROVIDER="anthropic", ANTHROPIC_API_KEY="test-anthropic", FORMATTER_MODEL="gpt-4o"
    )

    result = _import_and_print("config.settings.formatter_model", env, tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "gpt-4o"


def test_the_profile_model_follows_the_summary_model(tmp_path):
    """Nobody reads it yet; the point is that the coach branches need not edit
    config.py to pick it up."""
    result = _import_and_print(
        "config.settings.profile_model", _env(SUMMARY_MODEL="gpt-4o"), tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "gpt-4o"
