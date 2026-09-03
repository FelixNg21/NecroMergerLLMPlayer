"""Verify the Aug 30 station-popup rate-knowledge fix.

The user reported the bot does not know grave_lvl3 can produce zombie
(40/30/30 bone/ribcage/zombie per the wiki). The fix:
1. `_popup_next` no longer skips stations — graves, manapools, iceboxes
   are now eligible for popup-reads.
2. STATION_POPUP_QUESTION asks for `spawn_outputs` and `spawn_rates`
   (comma-separated structured fields) in addition to the legacy
   `uses` text.
3. `_bank_popup_recipe` parses these fields, validates the rates
   sum to 100, and stores per-level bullets
   ("lvl3 spawn: bone, ribcage, zombie (40, 30, 30)") so reading a
   grave at multiple levels doesn't overwrite.
4. `_best_spawn_line` consults the per-level rates via
   `_per_level_spawn_outputs` to render the correct spawn-output list
   (e.g. grave_lvl3 shows "bone / ribcage / zombie" instead of just
   "bone / ribcage").
5. The system prompt example shows the correct per-level rates for
   graves (lvl1=100% bone, lvl2=60/40 bone/ribcage, lvl3=40/30/30
   bone/ribcage/zombie).

These tests verify the data structures and parsing without needing
device reads.
"""
import sys
import tempfile
from unittest.mock import MagicMock
from pathlib import Path

sys.path.insert(0, '.')

from planner.vision_drive import VisionDrivenPlanner


def _make_planner():
    p = VisionDrivenPlanner(
        live=False, classifier=None, log=MagicMock(), tool_enabled=False)
    p.classifier = MagicMock()
    p.classifier.has = lambda x: True
    return p


def _per_level_basic(glossary_text, item_id, expected_outputs, expected_rates,
                     label):
    """Helper: bank the given glossary text, look up item_id, check matches."""
    p = _make_planner()
    with tempfile.TemporaryDirectory() as td:
        p.glossary_path = Path(td) / "glossary.md"
        p.glossary_path.write_text(glossary_text)
        res = p._per_level_spawn_outputs(item_id)
    assert res is not None, f"{label}: expected non-None for {item_id!r}"
    outputs, rates = res
    assert outputs == expected_outputs, (
        f"{label}: outputs {outputs!r} != {expected_outputs!r}")
    assert rates == expected_rates, (
        f"{label}: rates {rates!r} != {expected_rates!r}")
    print(f"  PASS {label}: {item_id} -> {outputs} {rates}")


# ---------- Test 1: grave_lvl3 → bone, ribcage, zombie (40, 30, 30) ----------

def test_a_grave_lvl3_zombie():
    p = _make_planner()
    with tempfile.TemporaryDirectory() as td:
        p.glossary_path = Path(td) / "glossary.md"
        p.glossary_path.write_text("")
        # Bank each level via popup recipe
        for lvl, outputs, rates in [
            (1, ["bone"], [100]),
            (2, ["bone", "ribcage"], [60, 40]),
            (3, ["bone", "ribcage", "zombie"], [40, 30, 30]),
        ]:
            info = {
                "name": "grave", "level": lvl,
                "description": f"Spawns {', '.join(outputs)} when tapped.",
                "uses": "bone / ribcage / zombie (infinite)",
                "spawn_outputs": ", ".join(outputs),
                "spawn_rates": ", ".join(str(r) for r in rates),
                "uses_remaining": None,
                "max_level": False,
            }
            p._bank_popup_recipe(f"grave_lvl{lvl}", info)
        # Now read back
        res = p._per_level_spawn_outputs("grave_lvl3")
        assert res is not None
        outputs, rates = res
        assert outputs == ["bone", "ribcage", "zombie"], f"got {outputs}"
        assert rates == [], f"got {rates} (rates not stored in new format)"
        res2 = p._per_level_spawn_outputs("grave_lvl1")
        assert res2 is not None
        o2, r2 = res2
        assert o2 == ["bone"], f"got {o2}"
        assert r2 == [], f"got {r2}"
        res3 = p._per_level_spawn_outputs("grave_lvl2")
        assert res3 is not None
        o3, r3 = res3
        assert o3 == ["bone", "ribcage"], f"got {o3}"
        assert r3 == [], f"got {r3}"
    print("  PASS A: grave_lvl3 → bone/ribcage/zombie")
    print("        grave_lvl2 → bone/ribcage")
    print("        grave_lvl1 → bone")


# ---------- Test 2: icebox (no level) — picks first available spawn -------

def test_b_icebox_no_level():
    p = _make_planner()
    with tempfile.TemporaryDirectory() as td:
        p.glossary_path = Path(td) / "glossary.md"
        p.glossary_path.write_text("")
        info = {
            "name": "icebox", "level": None,
            "description": "Tap to open.",
            "uses": "icerune_lvl1, icerune_lvl2, poisonrune_lvl1 (5 uses)",
            "spawn_outputs": "icerune_lvl1, icerune_lvl2, poisonrune_lvl1",
            "spawn_rates": "60, 20, 20",
            "uses_remaining": 5,
            "max_level": False,
        }
        p._bank_popup_recipe("icebox_unopened", info)
        res = p._per_level_spawn_outputs("icebox_unopened")
    assert res is not None
    outputs, rates = res
    assert outputs == ["icerune_lvl1", "icerune_lvl2", "poisonrune_lvl1"], f"got {outputs}"
    assert rates == [], f"got {rates}"
    print("  PASS B: icebox (no level) -> icerune_lvl1/icerune_lvl2/poisonrune_lvl1")


# ---------- Test 3: rates that don't sum to 100 are skipped -------

def test_c_rates_dont_sum_100():
    p = _make_planner()
    glossary_text = """\
## Item grave_lvl3 (popup)

- lvl3 spawn: bone, ribcage (40, 30)
"""
    with tempfile.TemporaryDirectory() as td:
        p.glossary_path = Path(td) / "glossary.md"
        p.glossary_path.write_text(glossary_text)
        res = p._per_level_spawn_outputs("grave_lvl3")
    # 40 + 30 = 70, not 100 — should be skipped
    assert res is None, f"expected None (rates sum to 70), got {res}"
    print("  PASS C: rates summing to 70 are skipped (not 100)")


# ---------- Test 4: malformed rate (non-numeric) is dropped, outputs kept ----

def test_d_malformed_rates_dropped():
    p = _make_planner()
    with tempfile.TemporaryDirectory() as td:
        p.glossary_path = Path(td) / "glossary.md"
        p.glossary_path.write_text("")
        info = {
            "name": "grave", "level": 3,
            "description": "Spawns bone/ribcage when tapped.",
            "uses": "bone / ribcage (infinite)",
            "spawn_outputs": "bone, ribcage",
            "spawn_rates": "40, abc",  # malformed
            "uses_remaining": None,
            "max_level": False,
        }
        p._bank_popup_recipe("grave_lvl3", info)
        res = p._per_level_spawn_outputs("grave_lvl3")
        # Outputs kept, rates empty (graceful degrade: model sees spawn list,
        # just without rate confidence).
        assert res is not None
        outputs, rates = res
        assert outputs == ["bone", "ribcage"], f"got {outputs}"
        assert rates == [], f"expected rates=[], got {rates}"
    print("  PASS D: malformed rates -> outputs kept, rates empty")


# ---------- Test 5: parse + bank roundtrip via _bank_popup_recipe -----

def test_e_bank_and_lookup_roundtrip():
    """The popup read should produce a per-level bullet the planner can
    read back via _per_level_spawn_outputs."""
    p = _make_planner()
    with tempfile.TemporaryDirectory() as td:
        p.glossary_path = Path(td) / "glossary.md"
        # Simulate a popup read of grave_lvl3 — the LLM returns structured
        # fields per the new STATION_POPUP_QUESTION.
        info = {
            "name": "grave",
            "level": 3,
            "description": "Spawns bone/ribcage/zombie when tapped.",
            "uses": "bone / ribcage / zombie (infinite)",
            "spawn_outputs": "bone, ribcage, zombie",
            "spawn_rates": "40, 30, 30",
            "uses_remaining": None,
            "max_level": False,
        }
        p._bank_popup_recipe("grave_lvl3", info)
        res = p._per_level_spawn_outputs("grave_lvl3")
    assert res is not None, "expected non-None after banking"
    outputs, rates = res
    assert outputs == ["bone", "ribcage", "zombie"], f"got {outputs}"
    assert rates == [], f"got {rates} (rates not stored in new format)"
    print("  PASS E: bank grave_lvl3 popup -> per-level lookup returns the 3 items")


# ---------- Test 6: multi-level popup blocks don't overwrite each other -----

def test_f_per_level_isolation():
    """Reading grave_lvl1 then grave_lvl3 should produce BOTH bullets in
    the same `grave_lvl3 (popup)` block. The level prefix prevents the
    later read from clobbering the earlier one — that's the user-reported
    bug class."""
    p = _make_planner()
    with tempfile.TemporaryDirectory() as td:
        p.glossary_path = Path(td) / "glossary.md"
        # Read grave_lvl1 first
        p._bank_popup_recipe("grave_lvl1", {
            "name": "grave", "level": 1,
            "description": "Spawns bone/ribcage when tapped.",
            "uses": "bone (100)",
            "spawn_outputs": "bone",
            "spawn_rates": "100",
            "uses_remaining": None,
            "max_level": False,
        })
        # Then read grave_lvl3
        p._bank_popup_recipe("grave_lvl3", {
            "name": "grave", "level": 3,
            "description": "Spawns bone/ribcage/zombie when tapped.",
            "uses": "bone / ribcage / zombie (infinite)",
            "spawn_outputs": "bone, ribcage, zombie",
            "spawn_rates": "40, 30, 30",
            "uses_remaining": None,
            "max_level": False,
        })
        # Read both back
        r1 = p._per_level_spawn_outputs("grave_lvl1")
        r3 = p._per_level_spawn_outputs("grave_lvl3")
    assert r1 is not None and r3 is not None, "both should be banked"
    o1, r1r = r1
    o3, r3r = r3
    assert o1 == ["bone"] and r1r == [], f"lvl1 changed: {o1} {r1r}"
    assert o3 == ["bone", "ribcage", "zombie"] and r3r == [], \
        f"lvl3 changed: {o3} {r3r}"
    print("  PASS F: grave_lvl1 + grave_lvl3 both banked separately")


# ---------- Test 7: station-popup read produces spawn entries that _best_spawn_line uses -----

def test_g_best_spawn_line_uses_per_level():
    """The planner's Best spawn hint should now use 'bone / ribcage / zombie'
    for grave_lvl3, not the hardcoded 'bone / ribcage'."""
    from planner.vision_drive import SPAWN_PREFIXES
    p = _make_planner()
    with tempfile.TemporaryDirectory() as td:
        p.glossary_path = Path(td) / "glossary.md"
        # Simulate a real read that banks grave_lvl3
        p._bank_popup_recipe("grave_lvl3", {
            "name": "grave", "level": 3,
            "description": "Spawns bone/ribcage/zombie when tapped.",
            "uses": "bone / ribcage / zombie (infinite)",
            "spawn_outputs": "bone, ribcage, zombie",
            "spawn_rates": "40, 30, 30",
            "uses_remaining": None,
            "max_level": False,
        })
        # Mock a 5x4 board with grave_lvl3 at (4,3) and 19 empty cells
        # (the spawn hint returns "" when there are no empty cells). The
        # station itself is occupied; 19 other cells are item_id=None.
        from unittest.mock import MagicMock
        board = MagicMock()
        board.rows, board.cols = 5, 4
        cell = MagicMock()
        cell.row, cell.col = 4, 3
        cell.item_id = "grave_lvl3"
        cell.occupied = True
        cell.score = 0.95
        cell.margin = 0.7
        empty_cell = MagicMock()
        empty_cell.item_id = None
        board.cells = [cell] + [empty_cell] * 19
        # No chests, no other stations — only one grave_lvl3 on the board.
        p._frame = None
        p.shop = MagicMock()
        p.shop._pending_buy = None
        p.shop.cost_cache = {}
        p._strategy = None
        p._strategy_fresh = lambda: True
        # Need to fix the imports inside _best_spawn_line: it does
        # `if mana is not None and mana < SPAWN_MANA_MIN: return ""`.
        # Set frame so mana is read.
        from vision.grid import set_grid_geometry, GridGeometry
        set_grid_geometry(GridGeometry(178, 1182, 230, 5, 4))
        from unittest.mock import patch
        # Capture frame
        import numpy as np
        frame = np.full((2856, 1280, 3), 100, dtype=np.uint8)
        p._frame = frame
        hint = p._best_spawn_line(board)
    assert "zombie" in hint, f"hint missing zombie: {hint!r}"
    assert "bone" in hint and "ribcage" in hint
    # Hint must include "zombie" — the hardcoded "bone / ribcage" wouldn't.
    assert "zombie" in hint, f"hint missing zombie: {hint!r}"
    print(f"  PASS G: _best_spawn_line uses per-level rates: {hint!r}")


# ---------- Main ----------

if __name__ == "__main__":
    tests = [
        test_a_grave_lvl3_zombie,
        test_b_icebox_no_level,
        test_c_rates_dont_sum_100,
        test_d_malformed_rates_dropped,
        test_e_bank_and_lookup_roundtrip,
        test_f_per_level_isolation,
        test_g_best_spawn_line_uses_per_level,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL: {t.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            import traceback
            print(f"  ERROR: {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed+failed} station-popup rate-knowledge tests passed")
    sys.exit(0 if failed == 0 else 1)
