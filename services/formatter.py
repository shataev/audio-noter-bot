import json

from config import settings
from services.ai import Message, create_chat_client

chat = create_chat_client()
# The SDK object underneath it — the one that actually puts the request on the
# wire — under the name this module has always kept its transport by.
client = chat.sdk

SYSTEM_PROMPT = """Ты помогаешь вести личный дневник.
Тебе дают сырую транскрипцию голосового сообщения.

Текст записи — это слова автора. Твоя задача сделать их читаемыми, НЕ переписывая.

Разрешено:
- исправить очевидные ошибки распознавания речи: перепутанные похожие слова,
  неверные окончания, склеенные или разорванные слова, неправильно распознанные
  имена и названия
- расставить знаки препинания и заглавные буквы
- разбить сплошной текст на абзацы по смыслу

Запрещено:
- удалять хоть что-нибудь, включая слова-паразиты, повторы, оговорки и
  незаконченные фразы — автор сказал именно так
- перефразировать, сокращать, пересказывать, менять порядок слов
- добавлять слова, которых в записи не было
- менять стиль, тон, лексику или регистр речи

Если сомневаешься, исправлять или оставить как есть — оставь как есть.

Верни JSON с тремя полями:
- "title": короткий заголовок записи (3-5 слов, на русском, без кавычек) —
  единственное поле, которое ты придумываешь сам
- "text": текст записи по правилам выше
- "tags": массив тегов, которые автор явно назвал в сообщении (пустой массив,
  если не называл)

Отвечай ТОЛЬКО валидным JSON, без markdown-блоков и пояснений."""

# The reply has to carry the whole entry back, near enough word for word, so the
# budget follows the input instead of sitting at a constant that silently
# truncates a long one. Russian runs about two characters per token; the slack
# covers the title, the tags and JSON escaping. The floor keeps short entries
# cheap to reason about, the ceiling is the model's own output limit.
_MIN_OUTPUT_TOKENS = 1024
_MAX_OUTPUT_TOKENS = 16384


def _output_budget(transcription: str) -> int:
    estimated = len(transcription) // 2 + 512
    return max(_MIN_OUTPUT_TOKENS, min(estimated, _MAX_OUTPUT_TOKENS))


async def format_entry(transcription: str) -> tuple[str, str, list[str]]:
    completion = await chat.complete(
        model=settings.formatter_model,
        system=SYSTEM_PROMPT,
        messages=[Message(role="user", content=transcription)],
        max_output_tokens=_output_budget(transcription),
        # The shape is three fixed fields, but the prompt describes them and a
        # schema here would be a second description to keep in step with it.
        require_json=True,
    )
    data = json.loads(completion.text)
    return data["title"], data["text"], data.get("tags", [])
