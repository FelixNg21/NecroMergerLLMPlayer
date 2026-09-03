"""Verify the Aug 30 strategy-prioritized Best hints.

The user reported the graves were never merged. Root cause: the
`_best_merge_line` returned the highest-value merge on the board
(skeleton_lvl2 + skeleton_lvl2), not a merge of the strategy's target
family. The `_best_spawn_line` was telling the model to spawn from
grave_lvl2 (the highest-level spawn station) — which would consume a
grave the strategy needs to merge up.

These tests verify the strategy-prioritized hint picks a strategy-family
merge first AND skips strategy-family spawn stations.
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, '.')

import cv2
from planner.vision_drive import VisionDrivenPlanner, Move
from vision.classifier import TemplateClassifier
from vision.pipeline import classify_board
from vision.grid import set_grid_geometry, GridGeometry


def _planner():
    import tempfile
    tmpdir = tempfile.mkdtemp()
    # Create a dummy glossary.md so the code can find the item_knowledge dir next to it
    glossary_path = Path(tmpdir) / "glossary.md"
    glossary_path.write_text("")
    return VisionDrivenPlanner(
        live=False,
        classifier=None,
        log=MagicMock(),
        tool_enabled=False,
        glossary_path=glossary_path,
    )


def _setup_grid():
    set_grid_geometry(GridGeometry(178, 1182, 230, 5, 4))


def _make_cell(row, col, item_id, score=0.95, margin=0.5, occupied=True,
               runner_up_id=None):
    cell = MagicMock()
    cell.row = row
    cell.col = col
    cell.item_id = item_id
    cell.score = score
    cell.margin = margin
    cell.occupied = occupied
    cell.cell_px = 230
    cell.cx = 100 + col * 230
    cell.cy = 100 + row * 230
    cell.runner_up_id = runner_up_id
    return cell


def _board(cells):
    b = MagicMock()
    b.cells = cells
    b.rows = 5
    b.cols = 4
    return b


# ---------- Test A: strategy-prioritized merge picks strategy family first ----

def test_a_strategy_merge_picks_grave_pair():
    """With strategy = "Own a lvl 3+ Grave" and the board holding
    2x grave_lvl1 + 1x grave_lvl2 + 2x skeleton_lvl2, the Best merge
    hint should pick the grave pair, not the skeleton pair (which has
    higher value on the board)."""
    p = _planner()
    p._step_count = 5
    p._strategy = {"feat": "Own a lvl 3+ Grave.", "noun": "grave",
                    "step": 4, "kind": "station", "target": None}
    p._strategy_fresh = lambda: True
    cells = [
        _make_cell(1, 1, "grave_lvl1"),
        _make_cell(4, 2, "grave_lvl1"),
        _make_cell(4, 3, "grave_lvl2"),
        _make_cell(2, 1, "skeleton_lvl2"),
        _make_cell(2, 2, "skeleton_lvl2"),
    ]
    board = _board(cells)
    line = p._best_merge_line(board)
    assert "grave" in line, f"expected grave in line: {line!r}"
    assert "skeleton" not in line, \
        f"expected NO skeleton in line (strategy should win): {line!r}"
    assert "Best merge (strategy)" in line
    print(f"  PASS: A: strategy merge picks grave, not skeleton: {line!r}")
    return True


# ---------- Test B: no strategy-family merge -> fall back to default ---------

def test_b_no_strategy_merge_falls_back_to_default():
    """When the strategy names a family with NO mergeable pair on the
    board, the hint falls back to the highest-value merge (default
    behavior)."""
    p = _planner()
    p._step_count = 5
    p._strategy = {"feat": "Own a lvl 3+ Grave.", "noun": "grave",
                    "step": 4, "kind": "station", "target": None}
    p._strategy_fresh = lambda: True
    # No graves on the board, only 2x skeleton_lvl2
    cells = [
        _make_cell(2, 1, "skeleton_lvl2"),
        _make_cell(2, 2, "skeleton_lvl2"),
    ]
    board = _board(cells)
    line = p._best_merge_line(board)
    assert "skeleton" in line, f"expected fallback to skeleton: {line!r}"
    assert "Best merge (strategy)" not in line
    print(f"  PASS: B: no strategy pair -> fallback to default: {line!r}")
    return True


# ---------- Test C: spawn hint skips the strategy-family spawn station -----

def test_c_spawn_skips_strategy_family_station():
    """With strategy = "Own a Lvl 3+ Grave" and the board holding
    2x grave_lvl1 + 1x grave_lvl2 + 1x icebox_unopened, the Best
    spawn hint should point to the icebox (different family — safe
    to spawn) and skip the graves (they're merge material)."""
    p = _planner()
    p._step_count = 5
    p._strategy = {"feat": "Own a lvl 3+ Grave.", "noun": "grave",
                    "step": 4, "kind": "station", "target": None}
    p._strategy_fresh = lambda: True
    p._frame = None  # don't trigger read_mana_fraction
    # 4 occupied + 16 empty (5x4 board minus occupied cells)
    cells = [
        _make_cell(1, 1, "grave_lvl1"),
        _make_cell(4, 2, "grave_lvl1"),
        _make_cell(4, 3, "grave_lvl2"),
        _make_cell(2, 0, "icebox_unopened"),
    ] + [_make_cell(r, c, None) for r in range(5) for c in range(4)
         if (r, c) not in [(1, 1), (4, 2), (4, 3), (2, 0)]]
    board = _board(cells)
    line = p._best_spawn_line(board)
    assert "icebox" in line, f"expected icebox in line: {line!r}"
    assert "grave" not in line, \
        f"expected NO grave in line (strategy should skip): {line!r}"
    print(f"  PASS: C: spawn hint picks icebox, skips graves: {line!r}")
    return True


# ---------- Test D: no other spawner -> empty spawn hint -------------------

def test_d_no_spawn_when_only_strategy_station_present():
    """When the only spawner on the board is the strategy's target
    family (a grave with strategy = "Own a Lvl 3+ Grave"), the
    spawn hint returns "" — the model is told NOT to spawn."""
    p = _planner()
    p._step_count = 5
    p._strategy = {"feat": "Own a lvl 3+ Grave.", "noun": "grave",
                    "step": 4, "kind": "station", "target": None}
    p._strategy_fresh = lambda: True
    p._frame = None
    cells = [
        _make_cell(1, 1, "grave_lvl1"),
        _make_cell(4, 2, "grave_lvl1"),
        _make_cell(4, 3, "grave_lvl2"),
    ] + [_make_cell(r, c, None) for r in range(5) for c in range(4)
         if (r, c) not in [(1, 1), (4, 2), (4, 3)]]
    board = _board(cells)
    line = p._best_spawn_line(board)
    assert line == "", f"expected empty spawn hint, got: {line!r}"
    print(f"  PASS: D: no other spawner -> empty spawn hint: {line!r}")
    return True


# ---------- Test E: non-station strategy doesn't change merge priority ----

def test_e_non_station_strategy_uses_default():
    """A 'creature' or 'other' strategy (e.g. 'Own a Lvl 3+ Skeleton.')
    doesn't change the merge priority — the default highest-value merge
    wins. Only 'station' kind strategies need the special handling."""
    p = _planner()
    p._step_count = 5
    p._strategy = {"feat": "Own a Lvl 3+ Skeleton.", "noun": "skeleton",
                    "step": 4, "kind": "creature", "target": "skeleton_lvl3"}
    p._strategy_fresh = lambda: True
    cells = [
        _make_cell(2, 1, "skeleton_lvl2"),
        _make_cell(2, 2, "skeleton_lvl2"),
    ]
    board = _board(cells)
    line = p._best_merge_line(board)
    assert "skeleton" in line
    assert "Best merge (strategy)" not in line
    print(f"  PASS: E: non-station strategy -> default behavior: {line!r}")
    return True


# ---------- Test F: no strategy at all -> default behavior -----------------

def test_f_no_strategy_uses_default():
    """When the model has no committed strategy, the hints fall back
    to the default highest-value behavior."""
    p = _planner()
    p._step_count = 5
    p._strategy = None
    cells = [
        _make_cell(2, 1, "skeleton_lvl2"),
        _make_cell(2, 2, "skeleton_lvl2"),
    ]
    board = _board(cells)
    line = p._best_merge_line(board)
    assert "skeleton" in line
    assert "Best merge (strategy)" not in line
    print(f"  PASS: F: no strategy -> default behavior: {line!r}")
    return True


# ---------- Main ----------

if __name__ == "__main__":
    tests = [
        test_a_strategy_merge_picks_grave_pair,
        test_b_no_strategy_merge_falls_back_to_default,
        test_c_spawn_skips_strategy_family_station,
        test_d_no_spawn_when_only_strategy_station_present,
        test_e_non_station_strategy_uses_default,
        test_f_no_strategy_uses_default,
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
    print(f"\n{passed}/{passed+failed} strategy hint tests passed")
    sys.exit(0 if failed == 0 else 1)
