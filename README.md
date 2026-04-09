# Noter

A Telegram bot that turns voice messages into structured diary entries in Notion.

## How it works

1. You send a voice message to the bot
2. OpenAI Whisper transcribes the audio
3. GPT-4o-mini formats the transcription into a title, clean text, and extracts tags
4. The entry is saved to a Notion database — creating today's page if it doesn't exist, or appending to it if it does

## Notion database setup

Your database must have the following properties:

| Property | Type        | Notes                                      |
|----------|-------------|--------------------------------------------|
| `Name`   | Title       | Page title. Format: `9 апреля \| Entry 1, Entry 2` |
| `Created`| Date        | Set manually per entry. Used to find today's page |
| `Tags`   | Multi-select| Auto-populated. `Daily` is always added    |

## Installation

```bash
git clone <repo>
cd noter
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

## Running

```bash
source .venv/bin/activate
python3 bot.py
```

## Running on a VPS (systemd)

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
sudo systemctl daemon-reload
sudo systemctl enable noter
sudo systemctl start noter
```

Useful commands:

```bash
sudo systemctl status noter       # check status
sudo journalctl -u noter -f       # live logs
sudo systemctl restart noter      # restart after code update
```

## Using tags

Mention tags naturally in your voice message — GPT will extract them automatically:

> "Had a great workout today. Tags: sport, health"

The `Daily` tag is always added automatically. When appending to an existing page, tags are merged without duplicates.

## Project structure

```
noter/
├── bot.py                  # Telegram bot entry point
├── config.py               # Settings loaded from .env
├── services/
│   ├── whisper.py          # Audio transcription via OpenAI Whisper
│   ├── formatter.py        # Entry formatting via GPT-4o-mini
│   └── notion.py           # Notion API: create/update diary pages
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
