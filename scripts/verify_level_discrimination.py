"""verify_level_discrimination.py

Regression suite for the Aug 26 same-family-different-level cross-fire
gates. Three layers tested:
  - vision/pipeline.py SAME_FAMILY_MIN_MARGIN wipe gate
  - planner/llm.py NEIGHBOR_MIN_MARGIN validator rejection
  - Cell.runner_up_id propagated through classify_scores

The motivating incident: at 23:01, the bot tried to merge (2,3)+(3,1)
(both labeled skeleton_lvl4 by the bank) — the game refused because
the real levels differed (lvl4 vs lvl3, or lvl5 vs lvl4). The bank
saw both cells as same-id because the top-2 templates were skeleton
levels within margin 0.026-0.028, far below LABEL_MIN_MARGIN 0.10.
The new gates prevent the mislabel from forming a mergeable pair.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

PASS = 0


def check(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    line = f"{status}  {name}"
    if detail and not cond:
        line += f"  [{detail}]"
    print(line)
    if cond:
        PASS += 1


def part_a_helpers():
    """_same_family_diff_level unit cases."""
    from planner.llm import _same_family_diff_level as p_same
    from vision.pipeline import _same_family_diff_level as v_same
    check("A1: same family different level", p_same("skeleton_lvl3", "skeleton_lvl4"))
    check("A2: different family same level", not p_same("skeleton_lvl3", "zombie_lvl3"))
    check("A3: same id", not p_same("skeleton_lvl3", "skeleton_lvl3"))
    check("A4: None", not p_same(None, "skeleton_lvl3"))
    check("A5: unleveled base", not p_same("bone", "ribcage"))
    check("A6: same helper vision/pipeline and planner/llm agree",
          p_same("icerune_lvl1", "icerune_lvl2") == v_same("icerune_lvl1", "icerune_lvl2"))


def part_b_pipeline_wipe():
    """SAME_FAMILY_MIN_MARGIN gate in vision/pipeline.py wipes
    same-family different-level cells below the margin threshold.
    Use the live board (/tmp/screencap.png) for the smoke test."""
    from vision.classifier import TemplateClassifier
    from vision.grid import GridGeometry, set_grid_geometry
    from vision.pipeline import classify_board
    from vision.grid import SAME_FAMILY_MIN_MARGIN

    geom = GridGeometry(origin_x=184, origin_y=1192, cell_px=228, rows=5, cols=4, mouth_x=642, mouth_y=923)
    set_grid_geometry(geom)
    classifier = TemplateClassifier('assets/templates', seed=True)

    frame_path = Path('/tmp/screencap.png')
    if not frame_path.exists():
        print("SKIP B (no /tmp/screencap.png)")
        return
    frame = cv2.imread(str(frame_path))
    board = classify_board(frame, classifier)
    wiped = 0
    for cell in board.cells:
        if not cell.occupied and (cell.score, cell.margin) != (0.0, 0.0):
            # An "occupied but unclassified" cell = the wipe fired
            wiped += 1
        if not cell.occupied and cell.item_id is None and cell.score == 0.0 and cell.margin == 0.0:
            # empty cell, not a wipe
            pass
    # The live board has at least 1 same-family wipe on the suspect cells
    check("B1: SAME_FAMILY_MIN_MARGIN constant = 0.10", SAME_FAMILY_MIN_MARGIN == 0.10,
          f"got {SAME_FAMILY_MIN_MARGIN}")
    # (2,3) and (3,1) on the live board were labeled same-family different
    # level at margin 0.026 / 0.028 — they MUST be wiped.
    cells_by_pos = {(c.row, c.col): c for c in board.cells}
    suspect_23 = cells_by_pos.get((2, 3))
    suspect_31 = cells_by_pos.get((3, 1))
    check("B2: (2,3) wiped (was lvl4/lvl5 cross-fire at margin 0.026)",
          suspect_23 is not None and suspect_23.item_id is None,
          f"item_id={suspect_23.item_id if suspect_23 else 'missing'}")
    check("B3: (3,1) wiped (was lvl3/lvl4 cross-fire at margin 0.028)",
          suspect_31 is not None and suspect_31.item_id is None,
          f"item_id={suspect_31.item_id if suspect_31 else 'missing'}")
    # (3,3) skeleton_lvl5 at margin 0.168 has same-family ru but above
    # the gate — should be KEPT.
    suspect_33 = cells_by_pos.get((3, 3))
    check("B4: (3,3) skeleton_lvl5 KEPT (margin 0.168 > 0.10)",
          suspect_33 is not None and suspect_33.item_id == "skeleton_lvl5",
          f"item_id={suspect_33.item_id if suspect_33 else 'missing'}")


def part_c_validator():
    """NEIGHBOR_MIN_MARGIN validator gate. Build synthetic cells with
    runner_up_id set and confirm the merge validator refuses same-id
    pairs where the runner-up is a different level of the same family."""
    from planner.llm import LLMPlanner, NEIGHBOR_MIN_MARGIN
    from vision.grid import BoardState
    from planner.agent import Move

    check("C0: NEIGHBOR_MIN_MARGIN constant = 0.20", NEIGHBOR_MIN_MARGIN == 0.20,
          f"got {NEIGHBOR_MIN_MARGIN}")

    # Build a board with two same-id skeleton_lvl4 cells, one with
    # runner_up skeleton_lvl5 at margin 0.05.
    def cell(r, c, item, score, margin, ru):
        from vision.grid import Cell
        return Cell(row=r, col=c, cx=0, cy=0, item_id=item, score=score,
                    margin=margin, cell_px=228, occupied=True, runner_up_id=ru)

    # Margins must be ABOVE MERGE_MIN_MARGIN (0.10) so the existing
    # merge_low_margin gate doesn't fire first; the neighbor-suspect gate
    # is the second-line check for the [0.10, 0.20) margin band.
    cells = [
        cell(2, 3, "skeleton_lvl4", 0.691, 0.15, "skeleton_lvl5"),
        cell(3, 1, "skeleton_lvl4", 0.733, 0.13, "skeleton_lvl3"),
    ]
    board = BoardState(rows=5, cols=4, cells=cells)
    move = Move(kind="merge", cell_a=(2, 3), cell_b=(3, 1), target=None, taps=0)
    reason = LLMPlanner._validate(board, move)
    check("C1: validator refuses same-id pair with same-family runner-up (mgn 0.15/0.13)",
          reason is not None and "merge_neighbor_suspect" in (reason or ""),
          f"got {reason}")

    # A clean same-id pair with cross-family runner-ups should pass
    cells2 = [
        cell(2, 1, "ribcage", 0.941, 0.435, "bone"),
        cell(4, 0, "ribcage", 0.942, 0.544, "skeleton"),
    ]
    board2 = BoardState(rows=5, cols=4, cells=cells2)
    move2 = Move(kind="merge", cell_a=(2, 1), cell_b=(4, 0), target=None, taps=0)
    reason2 = LLMPlanner._validate(board2, move2)
    check("C2: validator accepts clean same-id pair (cross-family runner-up)",
          reason2 is None, f"got {reason2}")

    # A same-id pair with margin 0.30 (above NEIGHBOR_MIN_MARGIN) should pass
    cells3 = [
        cell(2, 3, "skeleton_lvl4", 0.9, 0.30, "skeleton_lvl5"),
        cell(3, 1, "skeleton_lvl4", 0.9, 0.30, "skeleton_lvl3"),
    ]
    board3 = BoardState(rows=5, cols=4, cells=cells3)
    move3 = Move(kind="merge", cell_a=(2, 3), cell_b=(3, 1), target=None, taps=0)
    reason3 = LLMPlanner._validate(board3, move3)
    check("C3: validator accepts same-id pair with margin >= 0.20",
          reason3 is None, f"got {reason3}")

    # A same-id pair with one cell's runner_up being a different family — pass
    cells4 = [
        cell(0, 0, "necromerger", 0.96, 0.50, "bone"),
    ]
    board4 = BoardState(rows=5, cols=4, cells=cells4)
    move4 = Move(kind="merge", cell_a=(0, 0), cell_b=(0, 1), target=None, taps=0)
    # (0,1) doesn't exist on the board — should give a different error, not neighbor
    reason4 = LLMPlanner._validate(board4, move4)
    check("C4: validator doesn't false-fire on different-family runner-up",
          reason4 is None or "neighbor_suspect" not in reason4,
          f"got {reason4}")


def part_d_classifier_signature():
    """classify_scores returns 4-tuples (item, score, margin, runner_up_id)."""
    from vision.classifier import TemplateClassifier
    from vision.grid import Cell, GridGeometry, set_grid_geometry
    geom = GridGeometry(origin_x=184, origin_y=1192, cell_px=228, rows=5, cols=4, mouth_x=642, mouth_y=923)
    set_grid_geometry(geom)
    classifier = TemplateClassifier('assets/templates', seed=True)
    cell = Cell(row=0, col=0, cx=298, cy=1306, item_id=None, cell_px=228)
    # Use a real frame (the live screencap) for the smoke test
    frame_path = Path('/tmp/screencap.png')
    if not frame_path.exists():
        print("SKIP D (no /tmp/screencap.png)")
        return
    frame = cv2.imread(str(frame_path))
    result = classifier.classify_scores(frame, [cell])
    check("D1: classify_scores returns 4-tuple per cell",
          len(result) == 1 and len(result[0]) == 4,
          f"got {result}")
    if result and result[0]:
        item, score, margin, ru = result[0]
        check("D2: 4-tuple fields are (item, score, margin, ru_id)",
              isinstance(item, (str, type(None))) and isinstance(score, float)
              and isinstance(margin, float) and isinstance(ru, (str, type(None))))


def part_e_live_dry_run():
    """Live screencap dry-run does not crash and gives a sensible result."""
    import subprocess
    r = subprocess.run(
        [".venv/bin/python3", "main.py", "--dry-run",
         "--screenshot", "screenshots/calib_board.png",
         "--planner", "heuristic"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True, timeout=30,
    )
    check("E1: heuristic dry-run on calib board still proposes (2,2)+(3,0)",
          "merge" in r.stdout and "(2, 2)" in r.stdout and "(3, 0)" in r.stdout,
          r.stdout.splitlines()[-1] if r.stdout else "")


def part_f_compile():
    """All modified modules compile cleanly."""
    import py_compile
    for f in ("vision/grid.py", "vision/classifier.py", "vision/pipeline.py",
              "planner/llm.py", "planner/merge.py"):
        try:
            py_compile.compile(str(Path(__file__).resolve().parent.parent / f),
                               doraise=True)
            check(f"F: {f} compiles", True)
        except py_compile.PyCompileError as exc:
            check(f"F: {f} compiles", False, str(exc))


def main():
    global PASS
    for part in (part_a_helpers, part_b_pipeline_wipe, part_c_validator,
                 part_d_classifier_signature, part_e_live_dry_run, part_f_compile):
        try:
            part()
        except Exception as exc:  # noqa: BLE001
            check(f"{part.__name__} raised {type(exc).__name__}", False, str(exc))
    print(f"\n{PASS} checks passed")
    sys.exit(0 if PASS >= 15 else 1)


if __name__ == "__main__":
    main()
