"""Bottom-bar reader: the 5-button dock (Feats / Station / Queue / Spellbook /
Shop) at the bottom of the lair screen.

Geometry (Aug 27): the dock band y range and per-button x positions are
detected at RUNTIME because the live save's dock layout has drifted from
the original Aug 7 calibration. Static constants (BUTTON_CENTERS,
ICON_CROPS) are used as the FALLBACK when runtime detection fails (e.g.
the dock is covered by a panel — no green tile visible).

Runtime detection (detect_positions):
  1. Find the dock's y range by scanning for the densest green-tile
     band (the button tile color is saturated green).
  2. Find button x positions by scanning for the 5 most-prominent
     connected green regions in that band.
  3. Match each detected position against the icon bank to identify
     the button (Feats / Station / Queue / Spellbook / Shop).
  4. Order left-to-right and assign names.

This makes the dock reader self-healing against layout drift. Static
calibration constants are kept for back-compat (read_bar() callers
expecting {x: ...} dicts work unchanged).
"""

from pathlib import Path

import cv2
import numpy as np

# Dock band (x, y, w, h) in native pixels. Panels are 256px wide.
DOCK_BAND = (0, 2600, 1280, 256)

# Fallback panel x centers (Aug 7 calibration). Used when runtime
# detection fails. Note: the live save has drifted; runtime detection
# supersedes these whenever it succeeds.
BUTTON_CENTERS = [128, 384, 640, 896, 1152]
BUTTON_NAMES = ["feats", "station", "queue", "spellbook", "shop"]

# Icon crop per button: (cx, cy, w, h) in native pixels. Fallback.
# queue's cy was originally 2786 (the Aug 7 calibration pointed
# at the NecroMerger character's body, not the queue icon). The live queue
# icon is at y=2750 — we update the fallback to match so any code that
# reads the static ICON_CROPS (when runtime detection is unavailable)
# at least hits the right area.
#
# queue x was 384 (Sep 2 fix): that was the STATION's x, not the queue's.
# The 5 dock buttons are left-to-right feats(161) station(381) queue(~640)
# spellbook(896) shop(1152); the queue is the MIDDLE button (~640), directly
# measured on the live save with a queued reward (mana_potion) centered at
# x~638-640. has_reward/collect_queue read the reward slot, so a wrong x=384
# pointed them at the empty-skull-looking station region and misread any
# queued reward as "queue is empty."
ICON_CROPS = {
    "feats": (161, 2753, 95, 104),
    "station": (381, 2766, 60, 60),
    "queue": (638, 2750, 62, 51),
    "spellbook": (896, 2760, 60, 60),
    "shop": (1152, 2760, 60, 60),
}

BANK_DIR = "assets/bottombar"
BANK_THRESHOLD = 0.6

# Locked icons are dim/low-texture. Unlocked icons are bright (std ~46-58);
# locked ones ~17-21 (measured on a Devourer-L1 fresh save where Spellbook and
# Shop are locked). Icons with std below this are "locked".
LOCKED_STD_MAX = 30.0
# Minimum number of buttons whose icon matches its bank for the bar to be
# considered present (a covered screen matches none).
BAR_PRESENT_MIN_MATCH = 2

# green-tile color (the live save's button tile background).
# The original calibration had buttons 256 px wide; the live save has
# smaller buttons (~140 px wide) with green backgrounds. Unlocked
# buttons are bright (S>=120, V>=100); LOCKED buttons (Spellbook/Shop
# at low Devourer levels) are darker (V as low as 60). We accept both.
GREEN_TILE_H_LO = 35
GREEN_TILE_H_HI = 100         # covers both shapes
GREEN_TILE_W_LO = 40          # the live tile is ~140 px wide; min 40 to
                              # accept narrower runs (close-X panel etc.)
GREEN_TILE_W_HI = 250
GREEN_S_THRESH = 100          # the live tile is moderately saturated
GREEN_V_THRESH = 50           # accept both bright (unlocked) and dim
                              # (locked) tiles
GREEN_H_LO = 50
GREEN_H_HI = 90
DOCK_Y_LO = 2680               # ignore non-dock green (e.g. tiles above)
DOCK_Y_HI = 2850
# How often to re-run runtime detection (steps). Cheap, but re-running
# every step is wasteful; the dock doesn't drift every step.
DOCK_DETECT_STEPS = 10
# Min column-sum (count of green pixels) for a column to count as part
# of a button tile. ~30 of 170 dock-y rows is enough.
DOCK_TILE_COL_THRESH = 30


class BottomBarReader:
    """Read the 5 bottom-bar buttons: identity via icon bank, lock state via
    icon texture, and presence (is the lair dock showing)."""

    def __init__(self, bank_dir: str = BANK_DIR, threshold: float = BANK_THRESHOLD):
        self.bank_dir = Path(bank_dir)
        self.threshold = threshold
        self.bank: dict[str, list] = {}
        self._load_bank()
        # live dock positions detected at runtime. Cached.
        self._detected_centers: list[tuple[str, int]] | None = None
        self._last_detect_step = -DOCK_DETECT_STEPS - 1
        self._step_count = 0
        # Fallback mapping (name -> x) when runtime detection is unavailable.
        self._fallback = {n: ICON_CROPS[n][0] for n in BUTTON_NAMES}

    def _load_bank(self) -> None:
        for path in sorted(self.bank_dir.glob("*.png")):
            name = path.stem.split("__")[0]
            if name in BUTTON_NAMES:
                self.bank.setdefault(name, []).append(cv2.imread(str(path)))

    def set_step(self, step: int) -> None:
        """Tell the reader the current step count (used to throttle
        runtime dock re-detection)."""
        self._step_count = step

    # ---- per-button helpers --------------------------------------------

    def icon_crop(self, frame, name: str):
        """Crop the icon for `name` at the LIVE-detected x (or fallback
        ICON_CROPS x) and the calibrated y. Uses the first template's
        shape as the crop size."""
        cx, cy, w_fb, h_fb = ICON_CROPS[name]
        frames = self.bank.get(name, [])
        if frames:
            t0 = next((t for t in frames if t is not None), None)
            if t0 is not None:
                crop_h, crop_w = t0.shape[:2]
            else:
                crop_h, crop_w = h_fb, w_fb
        else:
            crop_h, crop_w = h_fb, w_fb
        cx = self.button_center(name)
        x0 = max(cx - crop_w // 2, 0)
        y0 = max(cy - crop_h // 2, 0)
        x1 = min(x0 + crop_w, frame.shape[1])
        y1 = min(y0 + crop_h, frame.shape[0])
        return frame[y0:y1, x0:x1]

    def button_center(self, name: str) -> int:
        """x center of `name` in the live frame (uses detected pos when
        available, otherwise the static fallback)."""
        if self._detected_centers is not None:
            for n, x in self._detected_centers:
                if n == name:
                    return x
        return ICON_CROPS[name][0]

    def tap_xy(self, name: str) -> tuple[int, int]:
        """Where to tap for `name` (uses the live-detected x; y stays at
        the calibrated dock y)."""
        return (self.button_center(name), ICON_CROPS[name][1] - 4)

    @staticmethod
    def _std(crop) -> float:
        if crop is None:
            return 0.0
        return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std())

    def _match(self, crop, frames) -> float:
        best = -1.0
        for t in frames:
            if t is None:
                continue
            if t.shape[0] > crop.shape[0] or t.shape[1] > crop.shape[1]:
                continue
            s = float(cv2.matchTemplate(crop, t, cv2.TM_CCOEFF_NORMED).max())
            if s > best:
                best = s
        return best

    # ---- runtime detection (Aug 27) ------------------------------------

    def _detect_positions(self, frame) -> list[tuple[str, int]] | None:
        """Find the 5 dock button positions in the live frame and identify
        each by matching the icon bank. Returns [(name, x_center), ...] in
        left-to-right order, or None on detection failure.

        Method:
          1. For each button, scan a wide x range (the queue button has
             shifted hundreds of px from its Aug 7 calibration; scan the
             full dock) at the calibrated y. For each x, take a crop at
             the template size (NOT the ICON_CROPS size — templates vary
             in pixel dimensions and a 1-px mismatch skips the match).
          2. Template-match against the bank's templates; record the
             best (x, score) per button.
          3. Greedy assign: best score per name, one assignment each.
          4. Sort by x for canonical left-to-right order.
        """
        candidates: list[tuple[float, str, int]] = []   # (score, name, x)
        for name in BUTTON_NAMES:
            _, cy_n, _, _ = ICON_CROPS[name]
            frames = self.bank.get(name, [])
            if not frames:
                continue
            # Use the FIRST template's size as the crop size — this
            # avoids 1-px mismatches between ICON_CROPS w/h and the
            # actual template size (which can vary by 1 px).
            t0 = next((t for t in frames if t is not None), None)
            if t0 is None:
                continue
            crop_h, crop_w = t0.shape[:2]
            if name == "queue":
                # Queue has shifted hundreds of px; scan the whole dock.
                x_lo, x_hi = 0, 1280
            else:
                cx_cal = ICON_CROPS[name][0]
                x_lo = max(0, cx_cal - 80)
                x_hi = min(1280, cx_cal + 80)
            best = (self.threshold - 0.001, 0)
            for cx in range(x_lo, x_hi + 1, 4):
                x0 = cx - crop_w // 2
                y0 = cy_n - crop_h // 2
                x1 = x0 + crop_w
                y1 = y0 + crop_h
                if x0 < 0 or y0 < 0 or x1 > 1280 or y1 > frame.shape[0]:
                    continue
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                s = self._match(crop, frames)
                if s > best[0]:
                    best = (s, cx)
            if best[0] >= self.threshold:
                candidates.append((best[0], name, best[1]))
        # Greedy assign: best score first, one name each.
        candidates.sort(key=lambda t: -t[0])
        used: set[str] = set()
        result: list[tuple[str, int]] = []
        for score, name, cx in candidates:
            if name in used:
                continue
            result.append((name, cx))
            used.add(name)
        if len(result) < 5:
            return None
        result.sort(key=lambda p: p[1])
        return self._sanitize_positions(result)

    @staticmethod
    def _sanitize_positions(result: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Fix a dock layout where the queue's x collides with the station's.

        The queue button is identified by matching its EMPTY-skull bank, but
        when the queue holds a reward (e.g. a mana_potion) its icon does NOT
        match that bank, so greedy assignment can latch the queue onto the
        first skull-looking region — frequently the station's x — and leave
        the real (reward-holding) middle button undetected. Observed live:
        `queue` returned x=384 (≡ station 381) while the potion was centered
        at x~638.

        The dock buttons are ~evenly spaced left-to-right, so when `queue`
        is within a small tolerance of `station` (a duplicate), re-derive the
        queue's x as the midpoint of the station->spellbook gap — the middle
        button of a 5-slot dock. Both anchors are reliably detected.
        """
        pos = {name: x for name, x in result}
        qx = pos.get("queue")
        sx = pos.get("station")
        if qx is not None and sx is not None and abs(qx - sx) <= 12:
            a, b = sx, pos.get("spellbook", sx + 500)
            if b <= a:
                b = a + 500
            # midpoint gives ~638 between station(381) and spellbook(896).
            pos["queue"] = (a + b) // 2
            # re-sort by the corrected x to keep canonical left->right order
            return sorted(pos.items(), key=lambda kv: kv[1])
        return result

    def _maybe_detect(self, frame) -> None:
        """Re-run runtime dock detection when stale (DOCK_DETECT_STEPS
        steps since last detect, or first call)."""
        if (self._step_count - self._last_detect_step) < DOCK_DETECT_STEPS \
                and self._detected_centers is not None:
            return
        detected = self._detect_positions(frame)
        if detected is not None:
            self._detected_centers = detected
        self._last_detect_step = self._step_count

    # ---- bar state -----------------------------------------------------

    def bar_visible(self, frame) -> bool:
        """True if the lair dock is showing: at least MIN_MATCH of the 5
        buttons' icon crops match their own banked icon (a full-screen
        panel/menu replaces the dock band, so the icons no longer match)."""
        # Use the detected x positions when available so the bank matching
        # actually looks at the right pixel region.
        centers = (self._detected_centers or
                   [(n, ICON_CROPS[n][0]) for n in BUTTON_NAMES])
        matched = 0
        for name, cx in centers:
            cy, w_fb, h_fb = ICON_CROPS[name][1], ICON_CROPS[name][2], ICON_CROPS[name][3]
            frames = self.bank.get(name, [])
            if frames:
                t0 = next((t for t in frames if t is not None), None)
                if t0 is not None:
                    crop_h, crop_w = t0.shape[:2]
                else:
                    crop_h, crop_w = h_fb, w_fb
            else:
                crop_h, crop_w = h_fb, w_fb
            x0 = max(cx - crop_w // 2, 0)
            y0 = max(cy - crop_h // 2, 0)
            x1 = min(x0 + crop_w, frame.shape[1])
            y1 = min(y0 + crop_h, frame.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            crop = frame[y0:y1, x0:x1]
            if frames and self._match(crop, frames) >= self.threshold:
                matched += 1
        return matched >= BAR_PRESENT_MIN_MATCH

    def read_bar(self, frame) -> list[dict]:
        """Return one dict per button, left -> right:
        {"name", "x", "unlocked", "score"}.

        runs the runtime dock-position detection on first call
        (or every DOCK_DETECT_STEPS steps) so x reflects the LIVE dock
        layout, not the stale Aug 7 calibration. The template bank varies
        by 1-2 px from ICON_CROPS, so this uses the FIRST TEMPLATE'S
        SHAPE as the crop size to avoid 1-px template/size mismatches.
        """
        self._maybe_detect(frame)
        out = []
        centers = (self._detected_centers or
                   [(n, ICON_CROPS[n][0]) for n in BUTTON_NAMES])
        for name, cx in centers:
            cy = ICON_CROPS[name][1]
            # Use the first template's shape (not the ICON_CROPS w/h)
            frames = self.bank.get(name, [])
            if frames:
                t0 = next((t for t in frames if t is not None), None)
                if t0 is not None:
                    crop_h, crop_w = t0.shape[:2]
                else:
                    crop_h, crop_w = ICON_CROPS[name][2], ICON_CROPS[name][3]
            else:
                crop_h, crop_w = ICON_CROPS[name][2], ICON_CROPS[name][3]
            x0 = max(cx - crop_w // 2, 0)
            y0 = max(cy - crop_h // 2, 0)
            x1 = min(x0 + crop_w, frame.shape[1])
            y1 = min(y0 + crop_h, frame.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            crop = frame[y0:y1, x0:x1]
            std = self._std(crop)
            score = self._match(crop, frames) if frames else -1.0
            out.append({
                "name": name,
                "x": cx,
                "unlocked": std >= LOCKED_STD_MAX,
                "score": round(score, 2),
            })
        return out
