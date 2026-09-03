"""HUD readouts from the live emulator frame (top resource strip).

The board geometry can grow/shrink with game state, but the HUD is drawn at
fixed screen pixels, so these readouts are geometry-independent.

Mana bar (calibrated live on the Devourer L2 board, screen 1280x2856):
  - the mana bar is a cyan fill in the top resource strip at fixed pixels
  - fill band: rows y 244-258, cols x 139-461 (a bit wider than the bar)
  - fill color: cyan hue H in [70,95), S > 90, V > 70 (OpenCV HSV)
  - the bar's left edge is fixed at x=210; the right edge moves left as mana
    is spent. Full bar right edge x=395 -> extent 186px -> fraction ~1.0.
Non-lair frames (loading, home screen, full-screen panels) have no cyan in the
band and read as None (bar not visible).
"""

import cv2
import numpy as np

MANA_BAND_ROWS = slice(244, 258)   # y band of the mana bar strip
MANA_BAND_COLS = slice(139, 461)   # x window (bar + padding)
MANA_BAR_LEFT = 210                # fixed left edge of the cyan fill
MANA_BAR_FULL_EXTENT = 186         # px width when full (right edge x=395)

_MANA_H_LO, _MANA_H_HI = 70, 95
_MANA_S_MIN = 90
_MANA_V_MIN = 70

# Heuristic/LLM gate: with the mana bar below this fill fraction, spawning from
# the grave silently no-ops (spawn costs ~10 mana) — collect/feed instead.
SPAWN_MANA_MIN = 0.15

# Inverse gate for collect: with the bar at/above this fill fraction, tapping
# the NecroMerger for mana is wasted (over-cap mana can't be stored) — the
# planner should merge/spawn/feed instead of collecting.
COLLECT_MANA_MAX = 0.9


def read_mana_fraction(frame) -> float | None:
    """Fill fraction of the HUD mana bar in [0,1], or None if no bar is visible.

    Fraction is derived from the rightmost cyan column: a half-full bar has its
    right edge halfway between the fixed left edge (x=210) and the full edge
    (x=395). A genuinely empty bar also reads as None (no cyan) — callers that
    know they are on the lair should treat None-with-no-moves as 'mana low'.
    """
    if frame is None or frame.shape[0] < 258 or frame.shape[1] < 461:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[MANA_BAND_ROWS, MANA_BAND_COLS]
    cyan = ((hsv[..., 0] >= _MANA_H_LO) & (hsv[..., 0] < _MANA_H_HI)
            & (hsv[..., 1] > _MANA_S_MIN) & (hsv[..., 2] > _MANA_V_MIN))
    cols = np.where(cyan.any(axis=0))[0]
    if cols.size == 0:
        return None
    right = MANA_BAND_COLS.start + int(cols.max())
    fraction = (right - MANA_BAR_LEFT + 1) / MANA_BAR_FULL_EXTENT
    return min(1.0, max(0.0, fraction))