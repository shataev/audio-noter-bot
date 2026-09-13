"""Two Notion memory pages, faked at ``services/notion.py``'s own boundary.

Shared by the tests for the sync itself and by the tests for the handlers that
call it, because both want the same three things: a page whose bullets can be
edited from the test, a count of how often it was asked, and a way to make it
fail. The requests the real functions send have their own tests in
``tests/test_notion_memory_pages.py``; nothing here asserts anything about HTTP.

Not a ``test_`` module, so pytest imports it rather than collecting it.
"""

from services import notion

PARENT = "parent-page"


class FakePages:
    """The two pages, and a record of what was asked of them.

    ``write`` reconciles positionally and mints an id for anything new, which is
    what the real ``write_bullets`` does. The ids it hands back are the whole
    reason push exists, so a fake that returned nothing would hide the bug it is
    here to catch.
    """

    def __init__(self):
        self.bullets = {"profile-page": [], "rules-page": []}
        self.reads = 0
        self.writes = 0
        self.lookups = 0
        self.created = []
        self.exists = {
            notion.MEMORY_PROFILE_TITLE: "profile-page",
            notion.MEMORY_RULES_TITLE: "rules-page",
        }
        self.fails = None
        self._next = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(notion, "memory_parent_page_id", self.parent)
        monkeypatch.setattr(notion, "find_child_page", self.find)
        monkeypatch.setattr(notion, "create_child_page", self.create)
        monkeypatch.setattr(notion, "read_bullets", self.read)
        monkeypatch.setattr(notion, "write_bullets", self.write)
        return self

    def put(self, page, *pairs):
        """Set a page's bullets to ``(block id, text)`` pairs — a hand edit."""
        self.bullets[page] = [notion.Bullet(id=key, text=text) for key, text in pairs]

    def texts(self, page):
        return [bullet.text for bullet in self.bullets[page]]

    def _raise(self):
        if self.fails is not None:
            raise self.fails

    async def parent(self):
        self._raise()
        self.lookups += 1
        return PARENT

    async def find(self, parent, title):
        self._raise()
        return self.exists.get(title)

    async def create(self, parent, title):
        self._raise()
        page_id = f"new-{title}"
        self.exists[title] = page_id
        self.bullets[page_id] = []
        self.created.append(title)
        return page_id

    async def read(self, page_id):
        self._raise()
        self.reads += 1
        return list(self.bullets[page_id])

    async def write(self, page_id, texts):
        self._raise()
        self.writes += 1
        existing = self.bullets[page_id]
        kept = []
        for index, text in enumerate(texts):
            if index < len(existing):
                kept.append(notion.Bullet(id=existing[index].id, text=text))
            else:
                self._next += 1
                kept.append(notion.Bullet(id=f"new-block-{self._next}", text=text))
        self.bullets[page_id] = kept
        return [bullet.id for bullet in kept]
