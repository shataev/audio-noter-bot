import json

import openai

from config import settings

client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

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
    response = await client.chat.completions.create(
        model=settings.formatter_model,
        max_completion_tokens=_output_budget(transcription),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcription},
        ],
    )
    data = json.loads(response.choices[0].message.content)
    return data["title"], data["text"], data.get("tags", [])
