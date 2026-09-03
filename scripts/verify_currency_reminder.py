""" when a station-buy was unaffordable, the bot should remember
the Rune balance and surface a "needs N more ice for grave" reminder
in the board state. Without this, the model forgets between steps
that it has 0 ice Runes and re-asks buy_station in a tight loop.

Run: `.venv/bin/python scripts/verify_currency_reminder.py`
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {detail}")


def _build_planner(feeds, max_level, cost_cache, mana=1.0,
                 strategy_family=None):
    """Build a minimal VisionDrivenPlanner for testing _currency_line_text
    + strategy-aware affordance reminder."""
    from planner.vision_drive import VisionDrivenPlanner
    from vision.classifier import TemplateClassifier

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        chat_log = tmp / "c.jsonl"
        chat_log.touch()
        clf = TemplateClassifier(str(tmp / "templates"), seed=True)
        p = VisionDrivenPlanner(
            client=None, live=False, tool_enabled=True, hints=True,
            log=None, chat_log_path=chat_log,
            classifier=clf,
            learnings_path=tmp / "learnings.md",
            glossary_path=tmp / "glossary.md",
            max_retries=2)
        # Stub the values used by _currency_line_text
        p._step_count = 100
        p._currency = feeds
        p._last_currency_step = 100
        p._last_currency_result = {
            "family": "grave",
            "confirm": False,
            "bought": False,
            "error": "unaffordable: needs 20 ice; have 0 ice, 16 poison, 0 blood, 0 moon, 0 death",
            "panel_verified": True,
            "currency": feeds,
        }
        p._max_level_ids = lambda: set(max_level)
        p._chain_map = lambda: {}
        if strategy_family:
            p._strategy = {
                "feat": f"Own a lvl 3+ {strategy_family.capitalize()}.",
                "target": None,
                "step": 90,
            }
            p._strategy_family = strategy_family
            p._strategy_fresh = lambda: True
        else:
            p._strategy = None
            p._strategy_family = None
            p._strategy_fresh = lambda: False
        # Stub the shop so cost_cache is readable
        class FakeShop:
            def __init__(self, cc):
                self.cost_cache = cc
        p.shop = FakeShop(cost_cache)
        return p


def test_no_currency_yet_returns_empty():
    """When the model hasn't called buy_station yet, no balance line."""
    with tempfile.TemporaryDirectory() as td:
        from planner.vision_drive import VisionDrivenPlanner
        from vision.classifier import TemplateClassifier
        tmp = Path(td)
        chat_log = tmp / "c.jsonl"
        chat_log.touch()
        clf = TemplateClassifier(str(tmp / "templates"), seed=True)
        p = VisionDrivenPlanner(
            client=None, live=False, tool_enabled=True, hints=True,
            log=None, chat_log_path=chat_log,
            classifier=clf,
            learnings_path=tmp / "learnings.md",
            glossary_path=tmp / "glossary.md")
        p._currency = None
        p._last_currency_result = None
        check("A: no buy_station yet -> empty line",
              p._currency_line_text() == "",
              f"got {p._currency_line_text()!r}")


def test_rune_balance_surfaced():
    """The cached balance should be visible in the board state line."""
    p = _build_planner(feeds={"ice": 0, "poison": 16, "blood": 0,
                            "moon": 0, "death": 0},
                     max_level=set(),
                     cost_cache={})
    line = p._currency_line_text()
    check("B: balance shown",
          "0 ice" in line and "16 poison" in line,
          f"got {line!r}")


def test_unaffordable_reminder_with_strategy():
    """When the build strategy is grave and the last buy said 'unaffordable:
    needs 20 ice; have 0 ice', the line tells the model how to fix it."""
    p = _build_planner(feeds={"ice": 0, "poison": 16, "blood": 0,
                            "moon": 0, "death": 0},
                     max_level=set(),
                     cost_cache={"grave": {"ice": 20, "poison": 0, "blood": 0,
                                            "moon": 0, "death": 0}},
                     strategy_family="grave")
    line = p._currency_line_text()
    check("C: short 20 ice for grave",
          "20 ice short" in line and "grave" in line and "Ice Chests" in line,
          f"got {line!r}")


def test_unaffordable_reminder_no_strategy():
    """Without an active build strategy, just show the balance (no recipe)."""
    p = _build_planner(feeds={"ice": 0, "poison": 16, "blood": 0,
                            "moon": 0, "death": 0},
                     max_level=set(),
                     cost_cache={})
    line = p._currency_line_text()
    check("D: no strategy, just balance",
          "20 ice short" not in line and "0 ice" in line,
          f"got {line!r}")


def test_partial_balance_reminder():
    """Model has 5 ice, needs 20 -> short 15. The reminder is concrete."""
    p = _build_planner(feeds={"ice": 5, "poison": 16, "blood": 0,
                            "moon": 0, "death": 0},
                     max_level=set(),
                     cost_cache={"grave": {"ice": 20, "poison": 0, "blood": 0,
                                            "moon": 0, "death": 0}},
                     strategy_family="grave")
    line = p._currency_line_text()
    check("E: short 15 ice for grave",
          "15 ice short" in line and "have 5" in line,
          f"got {line!r}")


if __name__ == "__main__":
    test_no_currency_yet_returns_empty()
    test_rune_balance_surfaced()
    test_unaffordable_reminder_with_strategy()
    test_unaffordable_reminder_no_strategy()
    test_partial_balance_reminder()
    print(f"\n{PASS} pass, {FAIL} fail")
    sys.exit(0 if FAIL == 0 else 1)
