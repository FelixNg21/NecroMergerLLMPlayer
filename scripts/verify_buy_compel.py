"""Verify the Aug 30 buy_station compel.

The user reported the grave is never bought. The session log shows
`strategy_set("Own a lvl 3+ Grave.")` followed by `merge`/`spawn` moves
with NO `buy_station` tool call at all — the model never even starts the
two-phase buy. The strategy note told it "call buy_station(\"grave\",
confirm=false)" but the model went straight to merge/spawn.

The fix: when a station-build strategy is fresh AND the cached currency
shows the cost is affordable AND no pending buy exists, force a
`buy_station(confirm=false)` tool call so the dialog-read happens. The
post-loop compelled-confirm then forces the matching `confirm=true`.

These tests verify the data flow without running an actual LLM:
  A. `_strategy_affordable` returns True when cost <= currency
  B. `_strategy_affordable` returns False when currency is short
  C. `_strategy_affordable` returns False when no cost_cache entry exists
  D. `_strategy_affordable` returns False when no _currency exists
  E. The compel-gate condition matches the affordability check
"""
import sys
from unittest.mock import MagicMock

sys.path.insert(0, '.')

from planner.vision_drive import VisionDrivenPlanner


def _planner():
    return VisionDrivenPlanner(
        live=False,
        classifier=None,
        log=MagicMock(),
        tool_enabled=False,
    )


# ---------- Test A: cost <= currency -> affordable ----------

def test_a_affordable_when_currency_meets_cost():
    p = _planner()
    p._step_count = 5
    p._strategy = {"feat": "Own a lvl 3+ Grave.", "noun": "grave",
                    "step": 4, "kind": "station", "target": None}
    p.shop = MagicMock()
    p.shop.cost_cache = {"grave": {"ice": 20}}
    p._currency = {"ice": 57, "poison": 16}
    assert p._strategy_affordable() is True
    print("  PASS: A: 57 ice >= 20 ice cost -> affordable")
    return True


# ---------- Test B: currency short -> not affordable ----------

def test_b_unaffordable_when_currency_below_cost():
    p = _planner()
    p._strategy = {"feat": "Build a Mana Pool.", "noun": "manapool",
                    "step": 4, "kind": "station", "target": None}
    p.shop = MagicMock()
    p.shop.cost_cache = {"manapool": {"ice": 5, "poison": 10}}
    p._currency = {"ice": 3, "poison": 16}
    assert p._strategy_affordable() is False
    print("  PASS: B: 3 ice < 5 ice cost -> NOT affordable")
    return True


# ---------- Test C: no cost_cache entry -> not affordable ----------

def test_c_unknown_cost_returns_not_affordable():
    """When the bot hasn't read the Station panel yet, the cost_cache is
    empty for the target family. We MUST return False — calling buy
    without knowing the cost would open the panel and the LLM dialog
    read is the only thing that confirms affordability."""
    p = _planner()
    p._strategy = {"feat": "Build a Mana Pool.", "noun": "manapool",
                    "step": 4, "kind": "station", "target": None}
    p.shop = MagicMock()
    p.shop.cost_cache = {}    # never read the panel
    p._currency = {"ice": 100}
    assert p._strategy_affordable() is False
    print("  PASS: C: no cost_cache entry -> NOT affordable (don't guess)")
    return True


# ---------- Test D: no _currency -> not affordable ----------

def test_d_unknown_currency_returns_not_affordable():
    """Without a cached currency, we can't know if the user can afford
    the build. The cached currency comes from the most recent
    buy_station(confirm=false) call. Before the first buy, the cache
    is None."""
    p = _planner()
    p._strategy = {"feat": "Build a Mana Pool.", "noun": "manapool",
                    "step": 4, "kind": "station", "target": None}
    p.shop = MagicMock()
    p.shop.cost_cache = {"manapool": {"ice": 5}}
    p._currency = None    # never read the panel
    assert p._strategy_affordable() is False
    print("  PASS: D: no _currency cache -> NOT affordable (don't guess)")
    return True


# ---------- Test E: no shop -> not affordable ----------

def test_e_no_shop_returns_not_affordable():
    p = _planner()
    p._strategy = {"feat": "Build a Mana Pool.", "noun": "manapool",
                    "step": 4, "kind": "station", "target": None}
    p.shop = None
    p._currency = {"ice": 100}
    assert p._strategy_affordable() is False
    print("  PASS: E: no shop -> NOT affordable")
    return True


# ---------- Test F: multi-rune cost is fully covered ----------

def test_f_multi_rune_cost_all_covered():
    """A station with two cost components (manapool: 5 ice + 10 poison)
    is affordable only when BOTH are covered."""
    p = _planner()
    p._strategy = {"feat": "Build a Mana Pool.", "noun": "manapool",
                    "step": 4, "kind": "station", "target": None}
    p.shop = MagicMock()
    p.shop.cost_cache = {"manapool": {"ice": 5, "poison": 10}}
    # All covered
    p._currency = {"ice": 10, "poison": 20}
    assert p._strategy_affordable() is True
    # Poison short
    p._currency = {"ice": 10, "poison": 5}
    assert p._strategy_affordable() is False
    # Ice short
    p._currency = {"ice": 3, "poison": 20}
    assert p._strategy_affordable() is False
    print("  PASS: F: multi-rune cost is fully covered -> affordable")
    return True


# ---------- Test G: empty cost dict is conservatively NOT affordable ----------

def test_g_empty_cost_dict_conservative():
    """A cost dict of {} means the bot read the panel but didn't find
    the family (no card visible, or all-zero cost). Without a real
    cost we can't affirmatively say "affordable" — the conservative
    answer is False, so the compel waits for a real cost_cache entry."""
    p = _planner()
    p._strategy = {"feat": "Build a Mana Pool.", "noun": "manapool",
                    "step": 4, "kind": "station", "target": None}
    p.shop = MagicMock()
    p.shop.cost_cache = {"manapool": {}}
    p._currency = {"ice": 0, "poison": 0}
    assert p._strategy_affordable() is False
    print("  PASS: G: empty cost dict -> conservatively NOT affordable")
    return True


# ---------- Test H: strategy not a station -> not affordable ----------

def test_h_non_station_strategy_returns_not_affordable():
    """A creature or 'other' strategy must not be considered for the
    buy compel — the cost is unknown and the action is wrong anyway."""
    p = _planner()
    p._strategy = {"feat": "Own a lvl 3+ Skeleton.", "noun": "skeleton",
                    "step": 4, "kind": "creature", "target": None}
    p.shop = MagicMock()
    p.shop.cost_cache = {}
    p._currency = {}
    assert p._strategy_affordable() is False
    print("  PASS: H: non-station strategy -> NOT affordable (not eligible)")
    return True


# ---------- Main ----------

if __name__ == "__main__":
    tests = [
        test_a_affordable_when_currency_meets_cost,
        test_b_unaffordable_when_currency_below_cost,
        test_c_unknown_cost_returns_not_affordable,
        test_d_unknown_currency_returns_not_affordable,
        test_e_no_shop_returns_not_affordable,
        test_f_multi_rune_cost_all_covered,
        test_g_empty_cost_dict_conservative,
        test_h_non_station_strategy_returns_not_affordable,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            if t():
                passed += 1
        except AssertionError as exc:
            print(f"  FAIL: {t.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR: {t.__name__}: {exc}")
            failed += 1
    print(f"\n{passed}/{passed+failed} buy_station compel tests passed")
    sys.exit(0 if failed == 0 else 1)
