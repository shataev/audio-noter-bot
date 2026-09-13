"""The coach's own package.

Everything the coach knows and everything it decides lives under here. The
package reaches the outside world through ``services/ai.py`` and through plain
data handed to it by ``bot.py``; it never imports Telegram, Notion, an AI SDK or
``bot`` itself. ``tests/test_coach_isolation.py`` enforces that with an AST scan.

The rule exists so the coach can be lifted out into a service of its own later
without archaeology, and a seam nobody enforces is a seam that closes.
"""
