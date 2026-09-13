"""The two Notion pages the coach's memory is mirrored onto.

Everything here is against ``httpx.MockTransport``. Nothing in this repository
can call the real API — there are no credentials on the machine that runs these
tests — so what is checked is the request this code sends and what it does with
the answer, and the shape of both rests on Notion's published documentation.

The one thing worth saying twice: a bullet's identity is its block id. A page
that is rewritten from scratch on every change would hand every fact a new id
every time, and the reworded bullet this whole feature invites the owner to write
would come back as a stranger. So these tests are mostly about which requests are
*not* sent.
"""

# config.py reads os.environ at import time and raises KeyError on a missing
# value, so the stubs have to be in place before anything from the project is
# imported. setdefault throughout, so a real environment always wins.
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NOTION_TOKEN", "test-notion")
os.environ.setdefault("NOTION_DATABASE_ID", "test-db")
os.environ.setdefault("ALLOWED_USER_ID", "1")
os.environ.setdefault("TIMEZONE", "Europe/Moscow")

import json

import httpx
import pytest

from config import settings
from services import notion

PARENT = "parent-page"


def bullet_block(block_id, text, *, pieces=None):
    """A bulleted list item as Notion returns one, ``plain_text`` and all."""
    contents = pieces if pieces is not None else [text]
    return {
        "object": "block",
        "id": block_id,
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": [{"plain_text": piece, "text": {"content": piece}} for piece in contents]
        },
    }


def child_page_block(block_id, title):
    return {"object": "block", "id": block_id, "type": "child_page", "child_page": {"title": title}}


class FakeNotion:
    """Just enough of the API to answer the five requests this module makes.

    Blocks live in a dict keyed by the page they are on, which is what makes the
    interesting assertions possible: that an unchanged bullet was never written
    to, and that a block kept the id it had.
    """

    def __init__(self, *, database_parent=None):
        self.database_parent = (
            database_parent
            if database_parent is not None
            else {"type": "page_id", "page_id": PARENT}
        )
        self.blocks: dict[str, list[dict]] = {PARENT: []}
        self.requests: list[tuple[str, str]] = []
        self.page_size = 100
        self._next = 0

    # -- building fixtures --------------------------------------------------- #

    def page(self, page_id, title, blocks=()):
        self.blocks[PARENT].append(child_page_block(page_id, title))
        self.blocks[page_id] = list(blocks)
        return page_id

    def texts(self, page_id):
        return [
            "".join(piece["plain_text"] for piece in block["bulleted_list_item"]["rich_text"])
            for block in self.blocks[page_id]
            if block["type"] == "bulleted_list_item"
        ]

    def ids(self, page_id):
        return [
            block["id"] for block in self.blocks[page_id] if block["type"] == "bulleted_list_item"
        ]

    def sent(self, method, contains=""):
        return [path for verb, path in self.requests if verb == method and contains in path]

    def mint(self, prefix):
        self._next += 1
        return f"{prefix}-{self._next}"

    # -- the transport ------------------------------------------------------- #

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        self.requests.append((request.method, path))
        body = json.loads(request.content) if request.content else {}

        if request.method == "GET" and path.startswith("/databases/"):
            return httpx.Response(200, json={"parent": self.database_parent})

        if request.method == "GET" and path.endswith("/children"):
            return self._children(path.split("/")[2], request)

        if request.method == "PATCH" and path.endswith("/children"):
            return self._append(path.split("/")[2], body)

        if request.method == "PATCH":
            return self._update(path.split("/")[2], body)

        if request.method == "DELETE":
            return self._delete(path.split("/")[2])

        if request.method == "POST" and path == "/pages":
            page_id = self.mint("page")
            title = body["properties"]["title"]["title"][0]["text"]["content"]
            self.blocks[body["parent"]["page_id"]].append(child_page_block(page_id, title))
            self.blocks[page_id] = []
            return httpx.Response(200, json={"id": page_id})

        raise AssertionError(f"unexpected request: {request.method} {path}")

    def _children(self, page_id, request):
        blocks = self.blocks[page_id]
        start = int(request.url.params.get("start_cursor", 0))
        window = blocks[start : start + self.page_size]
        has_more = start + self.page_size < len(blocks)
        return httpx.Response(
            200,
            json={
                "results": window,
                "has_more": has_more,
                "next_cursor": str(start + self.page_size) if has_more else None,
            },
        )

    def _append(self, page_id, body):
        created = []
        for child in body["children"]:
            pieces = [
                piece["text"]["content"] for piece in child["bulleted_list_item"]["rich_text"]
            ]
            created.append(bullet_block(self.mint("block"), None, pieces=pieces))
        self.blocks[page_id].extend(created)
        return httpx.Response(200, json={"object": "list", "results": created})

    def _update(self, block_id, body):
        for blocks in self.blocks.values():
            for index, block in enumerate(blocks):
                if block["id"] != block_id:
                    continue
                pieces = [
                    piece["text"]["content"] for piece in body["bulleted_list_item"]["rich_text"]
                ]
                blocks[index] = bullet_block(block_id, None, pieces=pieces)
                return httpx.Response(200, json=blocks[index])
        raise AssertionError(f"update of a block that is not there: {block_id}")

    def _delete(self, block_id):
        for blocks in self.blocks.values():
            for index, block in enumerate(blocks):
                if block["id"] == block_id:
                    del blocks[index]
                    return httpx.Response(200, json={"id": block_id, "archived": True})
        raise AssertionError(f"delete of a block that is not there: {block_id}")


@pytest.fixture
def api(monkeypatch):
    fake = FakeNotion()
    monkeypatch.setattr(settings, "notion_memory_parent_page_id", "")
    notion._client = httpx.AsyncClient(
        base_url=notion.API, headers=notion.HEADERS, transport=httpx.MockTransport(fake)
    )
    yield fake
    notion._client = None


# --------------------------------------------------------------------------- #
# Finding the two pages
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_pages_go_beside_the_diary_database(api):
    """Its parent page, which is where the owner already looks for the diary."""
    assert await notion.memory_parent_page_id() == PARENT
    assert api.sent("GET", "/databases/test-db") == ["/databases/test-db"]


@pytest.mark.asyncio
async def test_a_configured_parent_wins_and_costs_no_lookup(api, monkeypatch):
    """A database at the top of a workspace has no parent page to sit beside."""
    monkeypatch.setattr(settings, "notion_memory_parent_page_id", "chosen-page")

    assert await notion.memory_parent_page_id() == "chosen-page"
    assert api.requests == []


@pytest.mark.asyncio
async def test_a_database_with_no_parent_page_says_what_to_set(api):
    api.database_parent = {"type": "workspace", "workspace": True}

    with pytest.raises(notion.NotionError) as raised:
        await notion.memory_parent_page_id()

    assert "NOTION_MEMORY_PARENT_PAGE_ID" in str(raised.value)


@pytest.mark.asyncio
async def test_an_existing_page_is_found_by_title_and_not_made_again(api):
    api.page("rules-page", notion.MEMORY_RULES_TITLE)
    api.page("profile-page", notion.MEMORY_PROFILE_TITLE)

    found = await notion.find_child_page(PARENT, notion.MEMORY_PROFILE_TITLE)

    assert found == "profile-page"
    assert api.sent("POST", "/pages") == []


@pytest.mark.asyncio
async def test_a_page_that_is_not_there_is_not_invented_by_the_lookup(api):
    api.page("something-else", "Diary archive 2024")

    assert await notion.find_child_page(PARENT, notion.MEMORY_PROFILE_TITLE) is None


@pytest.mark.asyncio
async def test_creating_a_page_puts_it_under_the_parent_with_that_title(api):
    page_id = await notion.create_child_page(PARENT, notion.MEMORY_RULES_TITLE)

    assert await notion.find_child_page(PARENT, notion.MEMORY_RULES_TITLE) == page_id


# --------------------------------------------------------------------------- #
# Reading a page
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_bullet_is_read_with_the_block_id_that_identifies_it(api):
    api.page(
        "profile-page",
        notion.MEMORY_PROFILE_TITLE,
        [bullet_block("b1", "Откладывает трудные разговоры."), bullet_block("b2", "Не звонит.")],
    )

    read = await notion.read_bullets("profile-page")

    assert read == [
        notion.Bullet(id="b1", text="Откладывает трудные разговоры."),
        notion.Bullet(id="b2", text="Не звонит."),
    ]


@pytest.mark.asyncio
async def test_a_bullet_split_across_rich_text_objects_reads_as_one_line(api):
    api.page(
        "profile-page",
        notion.MEMORY_PROFILE_TITLE,
        [bullet_block("b1", None, pieces=["первая половина ", "и вторая"])],
    )

    assert (await notion.read_bullets("profile-page"))[0].text == "первая половина и вторая"


@pytest.mark.asyncio
async def test_whatever_else_the_owner_put_on_the_page_is_not_a_fact(api):
    """He may write on his own page. Only the bulleted list is the bot's."""
    api.page(
        "profile-page",
        notion.MEMORY_PROFILE_TITLE,
        [
            {"id": "h1", "type": "heading_2", "heading_2": {"rich_text": []}},
            bullet_block("b1", "Ценит тишину."),
            {"id": "p1", "type": "paragraph", "paragraph": {"rich_text": []}},
        ],
    )

    assert await notion.read_bullets("profile-page") == [
        notion.Bullet(id="b1", text="Ценит тишину.")
    ]


@pytest.mark.asyncio
async def test_a_list_longer_than_one_response_is_read_whole(api):
    """One response carries at most a hundred blocks, and a profile outgrows that."""
    api.page(
        "profile-page",
        notion.MEMORY_PROFILE_TITLE,
        [bullet_block(f"b{n}", f"факт {n}") for n in range(250)],
    )

    read = await notion.read_bullets("profile-page")

    assert len(read) == 250
    assert read[-1] == notion.Bullet(id="b249", text="факт 249")
    assert len(api.sent("GET", "/blocks/profile-page/children")) == 3


@pytest.mark.asyncio
async def test_a_read_that_failed_raises_rather_than_looking_empty(api, monkeypatch):
    """The distinction the whole feature turns on: no answer is not an empty page."""
    monkeypatch.setattr(settings, "notion_max_retries", 0)
    api.page("profile-page", notion.MEMORY_PROFILE_TITLE, [bullet_block("b1", "Ценит тишину.")])

    def refuse(request):
        return httpx.Response(503, json={"code": "service_unavailable"})

    notion._client = httpx.AsyncClient(
        base_url=notion.API, headers=notion.HEADERS, transport=httpx.MockTransport(refuse)
    )

    with pytest.raises(notion.NotionError):
        await notion.read_bullets("profile-page")


# --------------------------------------------------------------------------- #
# Writing a page
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_line_that_has_not_changed_costs_no_write(api):
    api.page(
        "rules-page",
        notion.MEMORY_RULES_TITLE,
        [bullet_block("b1", "не начинать с приветствия"), bullet_block("b2", "короче")],
    )

    keys = await notion.write_bullets("rules-page", ["не начинать с приветствия", "короче"])

    assert keys == ["b1", "b2"]
    assert api.sent("PATCH") == []
    assert api.sent("DELETE") == []


@pytest.mark.asyncio
async def test_a_changed_line_is_rewritten_in_place_and_keeps_its_id(api):
    """Not deleted and recreated: the id is the fact's identity across an edit."""
    api.page("rules-page", notion.MEMORY_RULES_TITLE, [bullet_block("b1", "короче")])

    keys = await notion.write_bullets("rules-page", ["отвечай короче"])

    assert keys == ["b1"]
    assert api.ids("rules-page") == ["b1"]
    assert api.texts("rules-page") == ["отвечай короче"]
    assert api.sent("PATCH", "/blocks/b1") == ["/blocks/b1"]


@pytest.mark.asyncio
async def test_a_new_line_is_appended_and_its_id_comes_back(api):
    api.page("rules-page", notion.MEMORY_RULES_TITLE, [bullet_block("b1", "короче")])

    keys = await notion.write_bullets("rules-page", ["короче", "без приветствия"])

    assert keys[0] == "b1"
    assert api.texts("rules-page") == ["короче", "без приветствия"]
    assert api.ids("rules-page") == keys


@pytest.mark.asyncio
async def test_a_line_that_is_gone_is_deleted_from_the_page(api):
    api.page(
        "rules-page",
        notion.MEMORY_RULES_TITLE,
        [bullet_block("b1", "короче"), bullet_block("b2", "без приветствия")],
    )

    keys = await notion.write_bullets("rules-page", ["короче"])

    assert keys == ["b1"]
    assert api.texts("rules-page") == ["короче"]
    assert api.sent("DELETE", "/blocks/b2") == ["/blocks/b2"]


@pytest.mark.asyncio
async def test_the_owners_own_blocks_are_never_touched(api):
    note = {"id": "p1", "type": "paragraph", "paragraph": {"rich_text": []}}
    api.page("rules-page", notion.MEMORY_RULES_TITLE, [note, bullet_block("b1", "короче")])

    await notion.write_bullets("rules-page", [])

    assert api.blocks["rules-page"] == [note]
    assert api.sent("DELETE", "/blocks/p1") == []


@pytest.mark.asyncio
async def test_a_line_too_long_for_one_rich_text_object_is_split_inside_one_bullet(api):
    """Notion caps a rich text object at 2000 characters; a bullet may hold many."""
    api.page("profile-page", notion.MEMORY_PROFILE_TITLE)
    long_line = "слово " * 700

    await notion.write_bullets("profile-page", [long_line])

    written = api.blocks["profile-page"][0]["bulleted_list_item"]["rich_text"]
    assert len(written) > 1
    assert all(len(piece["text"]["content"]) <= notion.MAX_TEXT_CHARS for piece in written)
    assert "".join(piece["text"]["content"] for piece in written) == long_line
    assert api.texts("profile-page") == [long_line]


@pytest.mark.asyncio
async def test_more_new_lines_than_one_request_holds_go_in_several(api):
    """A children array holds at most a hundred blocks."""
    api.page("profile-page", notion.MEMORY_PROFILE_TITLE)
    facts = [f"факт {n}" for n in range(250)]

    keys = await notion.write_bullets("profile-page", facts)

    assert api.texts("profile-page") == facts
    assert keys == api.ids("profile-page")
    assert len(keys) == 250
    assert len(api.sent("PATCH", "/blocks/profile-page/children")) == 3


@pytest.mark.asyncio
async def test_a_reordered_list_is_written_as_the_owner_left_it(api):
    api.page(
        "rules-page",
        notion.MEMORY_RULES_TITLE,
        [bullet_block("b1", "первое"), bullet_block("b2", "второе")],
    )

    keys = await notion.write_bullets("rules-page", ["второе", "первое"])

    assert api.texts("rules-page") == ["второе", "первое"]
    assert keys == ["b1", "b2"]
