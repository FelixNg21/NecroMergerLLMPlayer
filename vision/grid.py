"""Screen state and board detection. Implement the vision here."""

from dataclasses import dataclass

import cv2
import numpy as np

# Fallback board layout in pixels. The board grows with game state (Devourer
# level), so `detect_grid` returns the cached geometry read by
# vision/geometry.py at session start; these constants are the calibrated
# 5x3 default used when the LLM geometry read is skipped or fails.
ORIGIN_X, ORIGIN_Y = 305, 1207
CELL_PX = 226
TEMPLATE_SIZE = int(CELL_PX * 0.7)
ROWS, COLS = 5, 3  # number of rows and columns in the board

# Calibrated occupancy cutoff. 5x3 (226px cells): empties <= 0.01, items >=
# 0.069. 4x3 (262px cells): empties <= 0.0045, small items (bones) 0.035-0.042,
# stations 0.07+. 0.03 separates both geometries (margin ~7x above empties).
OCCUPANCY_THRESHOLD = 0.03

# A template label with margin at/above this is a real item regardless of the
# cell's edge-occupancy (e.g. bones on 4x3 read occ 0.035 < threshold but match
# bone at 0.85 with margin 0.43). Mirrors the planner's merge-margin gate.
LABEL_MIN_MARGIN = 0.10

# A label on a cell BELOW the occupancy threshold is only trusted when its
# absolute score is also strong. Background/noise on a genuinely EMPTY cell can
# match a sprite at 0.5-0.65 with a low runner-up (so the margin gate alone
# passes it) — e.g. an empty (3,0) labeled skeleton_lvl5 at 0.61 / margin 0.10,
# which made the heuristic propose an un-dragable merge that no-opped forever.
# Real items on both boards pass occupancy (>0.03), so this floor only gates
# phantom labels on empty cells.
PHANTOM_SCORE_FLOOR = 0.70

# Same-family-different-level cross-fire wipe (Aug 26). When a cell's top-2
# templates are different levels of the same family (e.g. skeleton_lvl3 vs
# skeleton_lvl4) and the margin is below this floor, the cell is in a bob
# phase the bank can't disambiguate — wipe the label to unidentified so the
# model can popup-read the true level. (Live incident: (2,3) at margin 0.026
# was labeled skeleton_lvl4 with runner-up skeleton_lvl5; (3,1) at margin
# 0.028 was labeled skeleton_lvl3 with runner-up skeleton_lvl4. Both cells
# got assigned a same-id label (lvl4) and the LLM proposed the merge, but
# the game refused because the real levels differed.) Set equal to
# LABEL_MIN_MARGIN so it only fires for cells that are already on the margin
# floor; confident cells (margin > 0.10) keep their label.
SAME_FAMILY_MIN_MARGIN = 0.10


@dataclass(frozen=True)
class GridGeometry:
    """Pixel geometry of the board grid (single source of truth).

    origin is the TOP-LEFT PIXEL of the grid's top-left cell; cell_px is the
    edge length of one square cell; mouth is the Devourer's feed target.
    """
    origin_x: int
    origin_y: int
    cell_px: int
    rows: int
    cols: int
    mouth_x: int = 642
    mouth_y: int = 923

    @property
    def template_size(self) -> int:
        return int(self.cell_px * 0.7)

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        return (
            self.origin_x + col * self.cell_px + self.cell_px // 2,
            self.origin_y + row * self.cell_px + self.cell_px // 2,
        )

    def contains(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def __str__(self) -> str:
        return (f"{self.rows}x{self.cols} @ ({self.origin_x},{self.origin_y}) "
                f"cell {self.cell_px}px mouth ({self.mouth_x},{self.mouth_y})")


FALLBACK_GEOMETRY = GridGeometry(ORIGIN_X, ORIGIN_Y, CELL_PX, ROWS, COLS)

# Geometry in effect for the current session (set by vision.geometry at start).
_CACHED: GridGeometry = FALLBACK_GEOMETRY


def set_grid_geometry(geom: GridGeometry) -> None:
    """Adopt a detected geometry for the rest of the session."""
    global _CACHED
    if geom is not None:
        _CACHED = geom


def session_geometry() -> GridGeometry:
    """The geometry in effect for this session (detected at startup, or the
    calibrated fallback). For code that needs board dims without a frame —
    e.g. the fixed NecroMerger cell `(0, cols-1)`, which moves when the board
    grows past 5x3."""
    return _geom(None)


def _geom(geometry: GridGeometry | None) -> GridGeometry:
    return geometry or _CACHED


def detect_grid(frame, *, geometry: GridGeometry | None = None) -> GridGeometry:
    """Return the board geometry for this screenshot.

    The session geometry is read once at startup (vision/geometry.py, LLM
    vision read + template-anchor refinement). Passing `geometry` overrides the
    cache; otherwise the cached geometry is returned.
    """
    if frame is None:
        raise ValueError("detect_grid: frame is None. Check the screenshot path or cv2.imread call.")
    return _geom(geometry)


def cell_center(row: int, col: int, geometry: GridGeometry | None = None) -> tuple[int, int]:
    return _geom(geometry).cell_center(row, col)


def occupancy_score(frame, row: int, col: int, geometry: GridGeometry | None = None) -> float:
    """Item sprites add strong edges vs. the flat board background (0..1)."""
    return float(cv2.Canny(cv2.cvtColor(crop_cell(frame, row, col, geometry), cv2.COLOR_BGR2GRAY), 80, 200).mean() / 255.0)


def crop_cell(frame, row: int, col: int, geometry: GridGeometry | None = None):
    """Crop the template_size square at the cell's pixel center."""
    geom = _geom(geometry)
    half = geom.template_size // 2
    cx, cy = geom.cell_center(row, col)
    return frame[cy - half : cy + half, cx - half : cx + half]


@dataclass
class Cell:
    row: int
    col: int
    cx: int  # pixel center
    cy: int
    item_id: str | None  # e.g. "bone_lvl1"; None == no trustworthy template label
    score: float = 1.0   # classifier confidence (best-match); 1.0 = unknown/absent
    margin: float = 0.0  # best-match minus runner-up match; 0.0 = unknown/absent
    cell_px: int = CELL_PX  # board cell pitch this cell belongs to
    occupied: bool = False  # occupancy source of truth: item_id None can mean
                            # "empty" OR "occupied but unclassified"
    runner_up_id: str | None = None  # second-best template id; for neighbor-suspect
                                     # merge gates (Aug 26 incident: a cell with
                                     # best=skeleton_lvl4 ru=skeleton_lvl5 at margin
                                     # 0.026 was misclassified and tried to merge with
                                     # another misclassified cell -> game refused)


@dataclass
class BoardState:
    rows: int
    cols: int
    cells: list[Cell]
    geometry: GridGeometry | None = None

    def cell_at(self, row: int, col: int) -> Cell | None:
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        return None


def diff_boards(prev: BoardState, curr: BoardState, move=None) -> dict:
    """Unexplained board changes between two consecutive classifies.

    Compares occupancy + item ids cell by cell, then filters out diffs the
    given move explains (merge transforms its two cells, feed empties its
    cell, spawn lands up to 2 new items anywhere, attack may clear its
    target). Same-cell id changes (bob-phase relabels) are vision noise
    unless on a move cell — but move cells are already filtered, so any
    remaining relabel is reported as a vanish+appear pair.

    Returns {"appeared": [(r, c, item_id)], "vanished": [...],
    "moved": [((r1, c1), (r2, c2), item_id)]} — all UNEXPLAINED, i.e.
    game-side changes (champion spawn, reward arrival, board growth) or
    misclassification worth surfacing to the summary. `move` is duck-typed
    (kind/cell_a/cell_b/target) to avoid a planner import cycle.
    """
    def snap(board):
        return {(c.row, c.col): (c.occupied, c.item_id) for c in board.cells}

    before, after = snap(prev), snap(curr)
    appeared, vanished = [], []
    for rc in set(before) | set(after):
        was_occ, was_id = before.get(rc, (False, None))
        is_occ, is_id = after.get(rc, (False, None))
        if is_occ and not was_occ:
            appeared.append((rc[0], rc[1], is_id))
        elif was_occ and not is_occ:
            vanished.append((rc[0], rc[1], was_id))
        elif was_occ and is_occ and was_id != is_id:
            vanished.append((rc[0], rc[1], was_id))
            appeared.append((rc[0], rc[1], is_id))

    kind = getattr(move, "kind", None)
    if kind == "merge":
        touched = set()
        for m in (move.cell_a, move.cell_b):
            if m is not None:
                touched.add((m[0], m[1]))
        appeared = [a for a in appeared if (a[0], a[1]) not in touched]
        vanished = [v for v in vanished if (v[0], v[1]) not in touched]
    elif kind == "feed" and move.cell_a is not None:
        vanished = [v for v in vanished
                    if (v[0], v[1]) != (move.cell_a[0], move.cell_a[1])]
    elif kind == "spawn":
        appeared = appeared[2:] if len(appeared) > 2 else []
    elif kind == "attack" and getattr(move, "target", None) is not None:
        t = (move.target[0], move.target[1])
        appeared = [a for a in appeared if (a[0], a[1]) != t]
        vanished = [v for v in vanished if (v[0], v[1]) != t]

    # Moved = same id vanished somewhere, appeared elsewhere (game-side
    # motion or paired mislabels). Pair greedily; None ids never pair.
    moved = []
    van_ids: dict = {}
    for v in vanished:
        if v[2] is not None:
            van_ids.setdefault(v[2], []).append(v)
    used_v, used_a = set(), set()
    for i, a in enumerate(appeared):
        if a[2] is None or i in used_a:
            continue
        cands = [v for v in van_ids.get(a[2], [])
                 if (v[0], v[1]) not in used_v
                 and (v[0], v[1]) != (a[0], a[1])]
        if cands:
            v = cands[0]
            used_v.add((v[0], v[1]))
            used_a.add(i)
            moved.append(((v[0], v[1]), (a[0], a[1]), a[2]))
    appeared = [a for i, a in enumerate(appeared) if i not in used_a]
    vanished = [v for v in vanished if (v[0], v[1]) not in used_v]
    return {"appeared": appeared, "vanished": vanished, "moved": moved}


def build_cells(geometry: GridGeometry | None = None) -> list[Cell]:
    geom = _geom(geometry)
    cells = []
    for r in range(geom.rows):
        for c in range(geom.cols):
            cx, cy = geom.cell_center(r, c)
            cells.append(Cell(r, c, cx, cy, item_id=None, cell_px=geom.cell_px))
    return cells


def draw_grid_overlay(frame, geometry: GridGeometry | None = None):
    geom = _geom(geometry)
    out = frame.copy()
    for r in range(geom.rows):
        for c in range(geom.cols):
            cx, cy = geom.cell_center(r, c)
            cv2.circle(out, (cx, cy), 8, (0, 255, 0), -1)
    return out
