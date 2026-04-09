# Noter

A Telegram bot that turns voice messages into structured diary entries in Notion.

## How it works

1. You send a voice message to the bot
2. OpenAI Whisper transcribes the audio
3. GPT-4o-mini formats the transcription into a title, clean text, and extracts tags
4. The entry is saved to a Notion database — creating today's page if it doesn't exist, or appending to it if it does
5. Every day at 21:00 the bot sends a GPT-generated summary of all entries recorded that day

## Notion database setup

Your database must have the following properties:

| Property | Type        | Notes                                      |
|----------|-------------|--------------------------------------------|
| `Name`   | Title       | Page title. Format: `9 апреля \| Entry 1, Entry 2` |
| `Created`| Date        | Set manually per entry. Used to find today's page |
| `Tags`   | Multi-select| Auto-populated. `Daily` is always added    |

## Installation

```bash
git clone https://github.com/shataev/audio-noter-bot.git
cd audio-noter-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable             | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `TELEGRAM_TOKEN`     | Bot token from [@BotFather](https://t.me/BotFather)         |
| `OPENAI_API_KEY`     | OpenAI API key (used for Whisper + GPT-4o-mini)              |
| `NOTION_TOKEN`       | Internal integration secret from notion.so/profile/integrations |
| `NOTION_DATABASE_ID` | ID from the database URL: `notion.so/workspace/{ID}?v=...`  |
| `ALLOWED_USER_ID`    | Your Telegram user ID — get it from [@userinfobot](https://t.me/userinfobot) |
| `TIMEZONE`           | Your timezone, e.g. `Asia/Bangkok`, `Europe/Moscow`          |

### Connecting Notion integration to your database

1. Open your database in Notion
2. Click `...` → `Connections` → select your integration

## Development

Run locally (stops the bot on VPS to avoid conflicts):

```bash
make dev
```

When done, restore the bot on VPS:

```bash
make stop-dev
```

## Deployment

Deploy to VPS with one command:

```bash
make deploy
```

This pushes changes to GitHub, pulls them on the VPS, and restarts the bot.

### First-time VPS setup

```bash
git clone https://github.com/shataev/audio-noter-bot.git /opt/noter
cd /opt/noter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env  # fill in all values
```

Create `/etc/systemd/system/noter.service`:

```ini
[Unit]
Description=Noter Telegram Bot
After=network.target

[Service]
User=root
WorkingDirectory=/opt/noter
ExecStart=/opt/noter/.venv/bin/python bot.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable noter
systemctl start noter
```

Useful commands:

```bash
systemctl status noter        # check status
journalctl -u noter -f        # live logs
systemctl restart noter       # restart manually
```

## Using tags

Mention tags naturally in your voice message — GPT will extract them automatically:

> "Had a great workout today. Tags: sport, health"

The `Daily` tag is always added automatically. When appending to an existing page, tags are merged without duplicates.

## Daily summary

Every day at 21:00 (in your timezone) the bot sends a GPT-generated summary of all diary entries recorded that day. If no entries were recorded, it sends a friendly reminder instead.

## Project structure

```
noter/
├── bot.py                  # Telegram bot entry point
├── config.py               # Settings loaded from .env
├── services/
│   ├── whisper.py          # Audio transcription via OpenAI Whisper
│   ├── formatter.py        # Entry formatting via GPT-4o-mini
│   ├── notion.py           # Notion API: create/update diary pages
│   └── summary.py          # Daily summary generation
├── Makefile                # Dev and deploy commands
├── requirements.txt
└── .env.example
```

## Estimated costs

Both Whisper and GPT-4o-mini are very cheap for personal use:

| Service    | Price              | Cost per entry (~1 min voice) |
|------------|--------------------|-------------------------------|
| Whisper    | $0.006 / minute    | ~$0.006                       |
| GPT-4o-mini| $0.15 / 1M tokens  | ~$0.00005                     |

100 entries/month ≈ **$0.60**
