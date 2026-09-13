import json
from dataclasses import dataclass

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

# --------------------------------------------------------------------------- #
# Did the model keep its side of the bargain?
#
# The prompt above forbids removing anything, and on a long dictation the cheap
# model does not reliably hold to that: it compresses. The caller needs to be
# able to tell, because an entry that reaches the diary shortened is the one
# failure the author cannot see — there is no copy of the transcription anywhere
# he can reach, so the words are simply gone.
# --------------------------------------------------------------------------- #

# Below this share of the dictated characters, the model rewrote rather than
# punctuated. A tenth is the width of the gap between the two: fixing misheard
# words moves the count by a percent or two and in both directions — "щас" for
# "сейчас" lengthens it — while a paragraph quietly dropped from a long entry
# takes far more than a tenth with it. Wide enough not to fire on ordinary work,
# which matters most: a guard that fires on a good reply is worse than no guard.
MIN_KEPT = 0.9


def _spoken(text: str) -> str:
    """Just the letters and digits: what the author said, with nothing else.

    The formatter is allowed to add punctuation, capitals and paragraph breaks,
    so comparing raw lengths would call a well-punctuated reply longer than the
    dictation it came from and a terse one shorter, both for reasons that are
    exactly what the model was asked to do. Stripping everything it may change
    leaves only what it may not touch.
    """
    return "".join(character for character in text if character.isalnum())


@dataclass(frozen=True)
class Kept:
    """How much of a dictation survived being formatted."""

    spoken: int
    kept: int

    @property
    def ratio(self) -> float:
        """1.0 when nothing was lost. Above it when the model spelled something out."""
        if self.spoken == 0:
            return 1.0
        return self.kept / self.spoken

    @property
    def too_little(self) -> bool:
        """True when the reply is short enough that it cannot be the same words."""
        return self.ratio < MIN_KEPT


def measure_kept(transcription: str, formatted: str) -> Kept:
    """Compare a formatted entry against the transcription it was made from.

    An empty transcription has nothing to lose and comes back whole, so a voice
    message that transcribed to nothing cannot trip the guard.
    """
    return Kept(spoken=len(_spoken(transcription)), kept=len(_spoken(formatted)))


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
        # No reasoning. Punctuating dictation is mechanical work — the same
        # argument that keeps this role on a cheap model — and on a provider
        # where thinking is spent out of the budget above, it would be taken
        # from the room the reply needs to carry the whole entry back.
        effort=None,
        # The shape is three fixed fields, but the prompt describes them and a
        # schema here would be a second description to keep in step with it.
        require_json=True,
    )
    data = json.loads(completion.text)
    return data["title"], data["text"], data.get("tags", [])
