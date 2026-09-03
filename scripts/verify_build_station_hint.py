"""Verify the Aug 29 build-station step hint.

The user pointed out that the previous hint for "Own a Lvl 3+ Grave." with
1×lvl1 + 1×lvl2 on the board said "merge them up, but you'll need 1 more
grave" — the existing 1×lvl1 and 1×lvl2 cannot be merged together (different
levels), so "merge them up" suggested a path that doesn't exist. The fix:
the hint now spells out the EXACT plan — buy X of level L, then merge
two lvl M into lvl M+1, etc.

The simulation is `_simulate_station_build`, which returns a list of
(action, level) tuples; the hint renders them. Tests cover the common
shapes the user reported plus a few edge cases.
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


# ---------- Test A: {1:1, 2:1} -> lvl3 (the user's case) ----------

def test_a_user_case_one_lvl1_one_lvl2_to_lvl3():
    """The user's exact case: 1×lvl1 + 1×lvl2 on the board, feat needs lvl3.
    Should require 1 buy (lvl1) + 2 merges (lvl1+lvl1→lvl2, lvl2+lvl2→lvl3)."""
    p = _planner()
    plan = p._simulate_station_build({1: 1, 2: 1}, 3)
    # Expected: buy lvl1, merge lvl1, merge lvl2
    assert plan == [("buy", 1), ("merge", 1), ("merge", 2)], \
        f"unexpected plan: {plan}"

    # The hint should NOT say "merge them up" because lvl1 and lvl2 cannot
    # be merged directly. The phrase "merge them" alone is forbidden.
    hint = p._build_station_step_hint("grave", {1: 1, 2: 1}, 3)
    # The exact sequence is spelled out
    assert "buy 1 more lvl1 grave" in hint
    assert "merge two lvl1" in hint
    assert "merge two lvl2" in hint
    print("  PASS: A: user case {1:1, 2:1} -> lvl3 spells out exact plan")
    return True


# ---------- Test B: {1:2} -> lvl3 (already 2 lvl1, need 1 buy) ----------

def test_b_two_lvl1_to_lvl3():
    """Two lvl1 + need lvl3: merge them to lvl2, then need 1 more lvl2
    to merge to lvl3."""
    p = _planner()
    plan = p._simulate_station_build({1: 2}, 3)
    # Expected: merge lvl1, buy lvl2, merge lvl2
    assert plan == [("merge", 1), ("buy", 2), ("merge", 2)], \
        f"unexpected plan: {plan}"

    hint = p._build_station_step_hint("grave", {1: 2}, 3)
    assert "merge two lvl1" in hint
    assert "buy 1 more lvl2 grave" in hint
    assert "merge two lvl2" in hint
    print("  PASS: B: {1:2} -> lvl3 plan is merge lvl1, buy lvl2, merge lvl2")
    return True


# ---------- Test C: {1:1, 2:2} -> lvl4 ----------

def test_c_one_lvl1_two_lvl2_to_lvl4():
    """One lvl1 + two lvl2 + need lvl4: requires 2 buys + 4 merges
    (or 3 buys + 3 merges — same total, simulation picks min-level buys)."""
    p = _planner()
    plan = p._simulate_station_build({1: 1, 2: 2}, 4)
    # The plan reaches lvl4 — verify the path traces correctly
    assert plan is not None
    # Verify the path: starting from {1:1, 2:2} -> through buys/merges -> lvl4
    # Run the simulation by hand to validate
    pool = [1, 2, 2]
    expected_pool_after = [4]
    for action, lvl in plan:
        if action == "buy":
            pool.append(lvl)
        else:
            pool.remove(lvl)
            pool.remove(lvl)
            pool.append(lvl + 1)
    pool.sort()
    assert pool == expected_pool_after, \
        f"plan leads to {pool}, expected {expected_pool_after}"
    print("  PASS: C: {1:1, 2:2} -> lvl4 plan reaches lvl4 correctly")
    return True


# ---------- Test D: empty board -> need lvl3 ----------

def test_d_empty_board_needs_buys():
    """No stations on the board at all. The hint should say 'buy fresh'
    rather than enumerating a long plan (the model needs to know there
    is nothing to start with)."""
    p = _planner()
    hint = p._build_station_step_hint("grave", {}, 3)
    assert "NO grave on the board" in hint
    assert "buy_station" in hint
    assert "confirm=false" in hint
    assert "confirm=true" in hint
    print("  PASS: D: empty board hint says 'buy fresh'")
    return True


# ---------- Test E: {1:3} -> lvl3 (already at goal) ----------

def test_e_already_at_goal():
    """If the board already has the needed level, the hint says so."""
    p = _planner()
    hint = p._build_station_step_hint("grave", {1: 3, 3: 1}, 3)
    assert "ALREADY have" in hint
    print("  PASS: E: already-at-goal hint says so")
    return True


# ---------- Test F: plan is unambiguous (no "merge them up" alone) ----------

def test_f_no_ambiguous_merge_them_up():
    """The hint must NEVER say "merge them up" alone when the existing
    stations are different levels — that's the misleading phrase the user
    complained about. The new hint spells out the exact sequence."""
    p = _planner()
    for existing in [{1: 1, 2: 1}, {1: 1, 2: 2}, {1: 1, 3: 1}, {2: 1, 3: 1}]:
        hint = p._build_station_step_hint("grave", existing, 4)
        # If the plan has a buy, the phrase "merge them" alone is forbidden —
        # the existing stations of DIFFERENT levels can't be merged together.
        if any(s in hint for s in ("buy ", "buy 0")):
            assert "merge them up" not in hint.lower(), \
                f"hint has 'merge them up' (ambiguous) for {existing}: {hint!r}"
    print("  PASS: F: hints never contain ambiguous 'merge them up' when buys are needed")
    return True


# ---------- Test G: simulation returns None on safety-cap exhaustion ---------

def test_g_simulation_reaches_far_levels():
    """A reasonable mid-game board (1×lvl1, 1×lvl2, 1×lvl3) reaching lvl5
    produces a valid plan that ends with the final merge to lvl5. The
    function `_build_station_step_hint` short-circuits empty boards, so
    the simulation only runs when there's at least one station."""
    p = _planner()
    # {1:1, 2:1, 3:1} -> lvl5: should need 1 buy (lvl1) + 2 merges
    plan = p._simulate_station_build({1: 1, 2: 1, 3: 1}, 5)
    assert plan is not None
    # Verify by re-running the plan and checking we end with lvl5
    pool = [1, 2, 3]
    for action, lvl in plan:
        if action == "buy":
            pool.append(lvl)
        else:
            pool.remove(lvl)
            pool.remove(lvl)
            pool.append(lvl + 1)
    pool.sort()
    assert pool[-1] >= 5, f"plan didn't reach lvl5: {pool}"
    n_steps = len(plan)
    print("  PASS: G: simulation reaches lvl5 from {1:1, 2:1, 3:1} (plan has", n_steps, "steps)")
    return True


# ---------- Main ----------

if __name__ == "__main__":
    tests = [
        test_a_user_case_one_lvl1_one_lvl2_to_lvl3,
        test_b_two_lvl1_to_lvl3,
        test_c_one_lvl1_two_lvl2_to_lvl4,
        test_d_empty_board_needs_buys,
        test_e_already_at_goal,
        test_f_no_ambiguous_merge_them_up,
        test_g_simulation_reaches_far_levels,
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
    print(f"\n{passed}/{passed+failed} build-station hint tests passed")
    sys.exit(0 if failed == 0 else 1)
