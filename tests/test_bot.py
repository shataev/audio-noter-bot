"""Unit tests for the bot's handlers and job scheduling.

There are no credentials on this machine: every Telegram, OpenAI and Notion call is
replaced by a stub. The environment variables below are what ``config`` reads at import
time, so they have to be in place before any project module is imported.
"""

import os

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import pytest

import bot


def _trigger_fields(job):
    return {field.name: str(field) for field in job.job.trigger.fields}


def test_weekly_report_is_scheduled_for_sunday(tmp_path, monkeypatch):
    """`days=(6,)` means Saturday in python-telegram-bot >= 20; Sunday is 0."""
    monkeypatch.setattr(bot, "STATE_FILE", str(tmp_path / "state.pickle"), raising=False)
    app = bot.build_application()

    jobs = {job.name: job for job in app.job_queue.jobs()}
    assert bot.WEEKLY_REPORT_JOB in jobs

    fields = _trigger_fields(jobs[bot.WEEKLY_REPORT_JOB])
    assert fields["day_of_week"] == "sun"
    assert fields["hour"] == "21"
    assert fields["minute"] == "0"


def test_daily_summary_runs_every_day(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "STATE_FILE", str(tmp_path / "state.pickle"), raising=False)
    app = bot.build_application()

    jobs = {job.name: job for job in app.job_queue.jobs()}
    fields = _trigger_fields(jobs[bot.DAILY_SUMMARY_JOB])
    assert fields["day_of_week"] == "sun,mon,tue,wed,thu,fri,sat"
    assert fields["hour"] == "21"
