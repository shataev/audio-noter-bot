import json
import logging
from dataclasses import dataclass

from config import settings
from services.ai import Message, create_chat_client

logger = logging.getLogger(__name__)

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
# Two paths, and why the long one exists
#
# The prompt above asks the model to hand the whole entry back, near enough word
# for word. That is a fair thing to ask of a paragraph and a bad thing to ask of
# ten minutes of dictation: the longer the entry, the more of the reply is a copy
# of the input, and the more likely the model is to summarise it to fit. The
# guard below catches that — but not creating the situation beats catching it,
# and above the threshold there is nothing to be gained by asking at all.
#
# So a long entry is never asked to be echoed. The model is asked for a title and
# tags, told in so many words not to return the text, and the transcription goes
# through untouched. Punctuation is what that costs, and punctuation is worth
# much less than the words.
# --------------------------------------------------------------------------- #

METADATA_PROMPT = """Ты помогаешь вести личный дневник.
Тебе дают сырую транскрипцию голосового сообщения.

Запись длинная. Её текст уже сохранён дословно и будет использован как есть —
возвращать его НЕ нужно. От тебя нужны только заголовок и теги.

Верни JSON ровно с двумя полями:
- "title": короткий заголовок записи (3-5 слов, на русском, без кавычек)
- "tags": массив тегов, которые автор явно назвал в сообщении (пустой массив,
  если не называл)

Запрещено:
- возвращать поле "text" или любое другое поле с текстом записи
- пересказывать, цитировать или сокращать запись

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


# On the formatting path the reply has to carry the whole entry back, near enough
# word for word, so the budget follows the input instead of sitting at a constant
# that silently truncates a long one. Russian runs about two characters per token;
# the slack covers the title, the tags and JSON escaping. The floor keeps short
# entries cheap to reason about, the ceiling is the model's own output limit —
# only reachable now if the threshold above is raised a long way past its default.
_MIN_OUTPUT_TOKENS = 1024
_MAX_OUTPUT_TOKENS = 16384

# On the metadata path the whole reply is a short title and a few tags, so the
# budget is a constant and a small one. Generous for what is asked: the point of
# the room is that a model which starts volunteering the text anyway runs out of
# it rather than being paid for a copy of the entry.
_METADATA_OUTPUT_TOKENS = 512


def _output_budget(transcription: str) -> int:
    estimated = len(transcription) // 2 + 512
    return max(_MIN_OUTPUT_TOKENS, min(estimated, _MAX_OUTPUT_TOKENS))


# A title is three to five words. This is the longest one that can be built out
# of the entry itself, and it exists so that a reply with no usable title costs
# the owner a good heading rather than the note.
_FALLBACK_TITLE_WORDS = 5
_FALLBACK_TITLE_CHARS = 60
# Only reachable from a transcription with no word characters in it at all.
_UNTITLED = "Без названия"


def _opening_words(transcription: str) -> str:
    """The first few words of what was said, as a heading.

    Not a summary and not pretending to be one: it is the beginning of the entry,
    which is what the owner would see scrolling past it anyway.
    """
    opening = " ".join(transcription.split()[:_FALLBACK_TITLE_WORDS])
    if not opening:
        return _UNTITLED
    return opening[:_FALLBACK_TITLE_CHARS].rstrip()


def _title_or_opening(data: dict, transcription: str) -> str:
    """The model's title, or one built from the entry when it did not give one."""
    title = data.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    logger.warning("formatter: the reply carried no title, using the opening words")
    return _opening_words(transcription)


def _tags(data: dict) -> list[str]:
    """The tags the author named, and nothing that is not a string."""
    raw = data.get("tags")
    if not isinstance(raw, list):
        return []
    return [tag for tag in raw if isinstance(tag, str) and tag.strip()]


async def format_entry(transcription: str) -> tuple[str, str, list[str]]:
    """Title, text and tags for one dictated entry.

    Which path is taken is decided by the length of the transcription and nothing
    else — not by what the model does with it — so the same entry always gets the
    same treatment and the boundary is something a person can reason about.
    """
    if len(transcription) > settings.formatter_full_text_limit:
        return await _metadata_only(transcription)
    return await _formatted(transcription)


async def _formatted(transcription: str) -> tuple[str, str, list[str]]:
    """Short enough to ask the model to punctuate it and hand it back.

    A reply with no title or no text is a broken reply here: the text of the entry
    is the thing being asked for, so there is nothing to salvage and the caller
    should say so and let the voice message be sent again.
    """
    logger.info(
        "formatter: asking for the whole entry back (%d characters)", len(transcription)
    )
    completion = await chat.complete(
        model=settings.formatter_model,
        system=SYSTEM_PROMPT,
        messages=[Message(role="user", content=transcription)],
        max_output_tokens=_output_budget(transcription),
        # No reasoning. Punctuating dictation is mechanical work, whatever the
        # model costs, and on a provider where thinking is spent out of the
        # budget above it would be taken from the room the reply needs to carry
        # the whole entry back.
        effort=None,
        # The shape is three fixed fields, but the prompt describes them and a
        # schema here would be a second description to keep in step with it.
        require_json=True,
    )
    data = json.loads(completion.text)
    return data["title"], data["text"], data.get("tags", [])


async def _metadata_only(transcription: str) -> tuple[str, str, list[str]]:
    """Too long to ask for back. The model names it; the words go through as they are.

    The text is safe before the call is made, so nothing the model returns can
    cost the owner the entry — which is why a reply that cannot be read is a
    heading built from the entry itself rather than a failure. What still fails
    is the call not happening at all: that is the caller's to report, and re-sending
    the voice message is the right answer to it.
    """
    logger.info(
        "formatter: entry of %d characters is past the %d-character limit, "
        "asking for a title and tags only",
        len(transcription),
        settings.formatter_full_text_limit,
    )
    completion = await chat.complete(
        model=settings.formatter_model,
        system=METADATA_PROMPT,
        messages=[Message(role="user", content=transcription)],
        max_output_tokens=_METADATA_OUTPUT_TOKENS,
        effort=None,
        require_json=True,
    )

    try:
        data = json.loads(completion.text)
    except ValueError:
        logger.warning("formatter: the reply was not readable JSON, keeping the entry as it is")
        data = {}
    if not isinstance(data, dict):
        logger.warning("formatter: the reply was not an object, keeping the entry as it is")
        data = {}

    # Byte for byte what came out of the transcriber. This is the whole point of
    # the path: a model that was never asked to echo the entry back cannot shorten
    # it, so there is nothing here for the guard above to catch.
    return _title_or_opening(data, transcription), transcription, _tags(data)
