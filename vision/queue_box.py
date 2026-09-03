"""Queue dock button: reward placement (Aug 24, generalized Aug 25).

The Queue dock button holds earned rewards (feat/level chests, rune piles,
any future reward type). With a reward queued the button shows the item;
empty it shows a bare skull (animated, banked in `assets/bottombar/queue__N.png`).
Tapping the button PLACES the queued item on the board — a board-mutating
placement, not a readable panel (why `tap_button` refuses it).

`has_reward` is a GENERAL detector: the empty state is a known bare-skull
icon bank (the same animated skull used by the bottom-bar reader), so a
queue holds a reward iff it does NOT match the empty state. This handles
every reward sprite (Ice Chest, poison rune, green gems, ...) without
needing a template per reward.

`QueueBox.collect` is now placement-only (Aug 25 spawn unification):
chests on the board are spawn stations — tap them via a normal `spawn`
move (one tap per use, mana-free, each rune occupies a cell). The queue
tool only places the queued dock reward.

Placement is bounded, dock-verified, and every state change is verified
before the next step; any surprise returns a structured error instead of
tapping blind.
"""

import time
from pathlib import Path

import cv2

from planner.agent import CONGESTION_THRESHOLD
from vision.bottombar import ICON_CROPS

QUEUE_ICON = ICON_CROPS["queue"]          # (cx, cy, w, h) — center crop (y is calibrated)
QUEUE_TAP_Y = QUEUE_ICON[1] - 4           # the button's y tap point
PLACE_POLLS = 6
PLACE_PAUSE = 1.0
EMPTY_BANK = "assets/bottombar"            # queue__0/1/2.png = bare-skull frames
EMPTY_MATCH_MIN = 0.60                     # empty-state match threshold
REWARD_TEX_MIN = 32.0                      # gray std above which a reward icon is present


def _load_empty_templates() -> list:
    """Animated bare-skull frames (the EMPTY queue state)."""
    out = []
    for p in sorted(Path(EMPTY_BANK).glob("queue__*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            out.append(img)
    return out


    # dock-reward icon ROI. When a reward is queued, the icon
# displayed in the Queue button is at the same x/y as the empty-state
# skull, but the icon itself is a different sprite (icebox_unopened,
# valuablechest, etc.). Re-using the ICON_CROPS geometry from
# vision/bottombar.py keeps the dock position consistent with the dock
# detection. Re-exported here so `vision/identify.read_dock_reward` can
# import it without taking a dependency on the BottomBarReader.
from vision.bottombar import ICON_CROPS
DOCK_BTN_ICON = ICON_CROPS["queue"]   # (cx, cy, w, h) = (384, 2750, 62, 51)


def _empty_match(crop, frames) -> float:
    if not frames or crop is None:
        return -1.0
    best = -1.0
    for t in frames:
        if t.shape[0] > crop.shape[0] or t.shape[1] > crop.shape[1]:
            continue
        s = float(cv2.matchTemplate(crop, t, cv2.TM_CCOEFF_NORMED).max())
        if s > best:
            best = s
    return best


def has_reward(frame) -> bool:
    """True when the Queue button holds a queued reward.

    A reward changes the button's icon away from the bare skull. The empty
    state is the animated bare skull (banked in the bottombar's queue bank);
    the inverted detector (does NOT match the empty skull) handles every
    reward sprite (chest, rune pile, gem, ...) without per-reward templates.
    Texture gate (REWARD_TEX_MIN) catches a near-empty icon (e.g. a black
    panel) that happens not to match the skull frames."""
    if frame is None:
        return False
    cx, cy, w, h = QUEUE_ICON
    crop = frame[cy - h // 2: cy - h // 2 + h, cx - w // 2: cx - w // 2 + w]
    if crop is None or crop.size == 0:
        return False
    frames = _load_empty_templates()
    empty_score = _empty_match(crop, frames)
    if empty_score >= EMPTY_MATCH_MIN:
        return False  # matches the bare skull — empty
    gray = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std())
    return gray >= REWARD_TEX_MIN


class QueueBox:
    """Place the queued reward. Chests on the board are now spawn stations
    (use a normal `spawn` move, one tap per use — see planner/agent.py)."""

    def __init__(self, device, classifier, bottombar):
        self.device = device
        self.classifier = classifier
        self.bottombar = bottombar

    def _cells(self, frame):
        """Full classified cell list, best-effort ([] on errors)."""
        if self.classifier is None or frame is None:
            return []
        try:
            from vision.pipeline import classify_board
            return classify_board(frame, self.classifier).cells
        except Exception:
            return []

    def _board_cells(self, frame):
        """Occupied (row, col) -> item_id map, best-effort ({} on errors)."""
        return {(c.row, c.col): c.item_id for c in self._cells(frame)
                if c.occupied}

    def collect(self, open_chest: bool = True) -> dict:  # noqa: ARG002
        """Place the queued reward from the Queue button — refused when the
        board is congested (each placed reward occupies a cell, and chest
        spawns keep producing rune stacks, so a full board is a real failure
        mode). Chests already on the board are NOT drained here — tap them
        via a `spawn` move (mana-free, one rune per tap).

        Returns a structured result; only BoardLost-style game-exit conditions
        raise (via the caller). `open_chest` is kept for compat (ignored)."""
        _ = open_chest
        res = {"placed": False, "opened": False, "drained": False,
               "error": None, "placed_item": None, "cell": None}
        self.device.screencap()
        lair = cv2.imread(str(self.device.screencap_path))
        if not self.bottombar.bar_visible(lair):
            res["error"] = "dock is not visible (a panel or non-lair screen is open)"
            return res
        cells = self._board_cells(lair)
        if not has_reward(lair):
            res["error"] = "queue is empty (no reward queued)"
            return res
        # Board-space gate: each placed reward occupies a cell, and chest
        # spawns keep producing rune stacks, so an unchecked queue can fill
        # the board.
        from vision.grid import detect_grid
        geom = detect_grid(lair)
        empty = geom.rows * geom.cols - len(cells)
        if empty <= CONGESTION_THRESHOLD:
            res["error"] = (f"board congested: only {empty} empty cells — "
                            "not placing a queued reward")
            return res
        before = cells
        # use the runtime-detected queue x position (BottomBarReader
        # falls back to the static ICON_CROPS when detection fails, e.g.
        # when the dock is covered by a panel).
        cx = self.bottombar.button_center("queue")
        cy = QUEUE_TAP_Y
        self.device.tap(cx, cy)
        # The queue empties when the reward is placed.
        placed = False
        for _ in range(PLACE_POLLS):
            time.sleep(PLACE_PAUSE)
            self.device.screencap()
            f = cv2.imread(str(self.device.screencap_path))
            if f is not None and not has_reward(f):
                placed = True
                break
        if not placed:
            res["error"] = "queue still shows the reward after tapping it"
            return res
        res["placed"] = True
        time.sleep(1.0)
        self.device.screencap()
        after = cv2.imread(str(self.device.screencap_path))
        after_cells = self._board_cells(after)
        new_cells = [(rc, iid) for rc, iid in after_cells.items()
                     if rc not in before or before[rc] != iid]
        if new_cells:
            (rc, iid) = new_cells[0]
            res["cell"] = list(rc)
            res["placed_item"] = iid or "unidentified"
        elif after_cells and not before:
            res["placed_item"] = "unknown"
        return res

    @staticmethod
    def _cell_center(frame, rc) -> tuple[int, int]:  # kept for compat
        """Pixel center of a board cell via the current grid geometry."""
        from vision.grid import cell_center
        r, c = rc
        x, y = cell_center(r, c)
        return int(x), int(y)
