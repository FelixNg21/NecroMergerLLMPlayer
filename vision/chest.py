"""Chest Uses tracking via OCR (Aug 25).

Chests on the board (icebox_unopened) show "Uses N" in the top info area
when selected. Tracking remaining uses lets the planner weight chest spawns
by need and lets the summary learn the spawn edge grounded in UI.

Mechanic: each chest has 5 Uses; each `spawn` tap on the chest consumes one
use and releases one random rune (ice lvl1 60%, ice lvl2 20%, poison lvl1 20%)
onto the board. The chest sprite may change after depletion, but the info
area's "Uses N" is the ground truth while it exists.

This module mirrors `vision/satiety.py` and `vision/queue_box.py` old Uses
reader but scoped to chests. It reuses the popup's red "Uses N" badge crop
(930,425,140,145) — the same badge that `QueueBox._read_uses` used — because
the chest info popup shares that badge geometry. When no popup is open, the
info area may still show Uses; we OCR the same crop.

The planner calls `read_chest_uses(frame)` after a chest is selected or after
a `spawn` on the chest to update `uses_remaining` for needs-based ranking.
"""

import re
import tempfile
from pathlib import Path

import cv2

from vision.satiety import apple_vision_text

# Badge crop from QueueBox — red "Uses N" badge in chest popup / info area
USES_BADGE = (930, 425, 140, 145)


def read_chest_uses(frame) -> int | None:
    """OCR the chest's "Uses N" badge -> N (None when unreadable/no chest)."""
    if frame is None:
        return None
    x, y, w, h = USES_BADGE
    # guard against frame size mismatch
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return None
    crop = frame[y:y + h, x:x + w]
    if crop is None or crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    big = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
        cv2.imwrite(t.name, big)
        txt = apple_vision_text(t.name)
        Path(t.name).unlink(missing_ok=True)
    m = re.search(r"(\d+)", txt or "")
    return int(m.group(1)) if m else None


def chest_uses_line(uses: int | None, max_uses: int = 5) -> str:
    """Human-readable line for board state, e.g. "Chest (4,0) Uses 3/5"."""
    if uses is None:
        return ""
    return f"Chest Uses {uses}/{max_uses}"


# Back-compat alias for old QueueBox._read_uses call sites
def _read_uses(frame) -> int | None:
    return read_chest_uses(frame)
