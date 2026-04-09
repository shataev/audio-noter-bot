import os
from dotenv import load_dotenv

load_dotenv()


class _Settings:
    telegram_token: str = os.environ["TELEGRAM_TOKEN"]
    openai_api_key: str = os.environ["OPENAI_API_KEY"]
    notion_token: str = os.environ["NOTION_TOKEN"]
    notion_database_id: str = os.environ["NOTION_DATABASE_ID"]
    allowed_user_id: int = int(os.environ["ALLOWED_USER_ID"])
    timezone: str = os.getenv("TIMEZONE", "Europe/Moscow")


settings = _Settings()
