"""Devourer Level-Up screen detection + dismissal.

When the Devourer's satiety bar fills (a `feed` lands the last point needed),
NecroMerger shows a level-up screen: a large green panel with the Devourer in
the center and a green "Continue" button just below the screen's halfway mark.
Tapping Continue dismisses it and returns to the lair.

Detection is a template bank of the *button* (not the panel): the button
(crop + "Continue" label) is a strong fingerprint that stays valid across
future Devourer levels whose panel art may change. The 1280x2856 emulator is
pinned, and the button is a fixed-geometry overlay (like the HUD mana bar and
the bottom dock — independent of the dynamic board geometry).

Geometry (native 1280x2856, verified on assets/calib/Level Up Screen.png):
- BUTTON: solid green body at (471, 1737, 338, 151); center (640, 1812); OCR
  reads "Continue" cleanly.
- BAND: generous search window around the button so matchTemplate absorbs
  small shifts (region search, same idea as the checkbox in classify.py).
Offline margin check (Aug 13): button match = 1.000 on the level-up frame vs
0.053-0.091 across 14 lair/panel/popup control frames, so THRESHOLD 0.6 gives
a ~10x safety gap.
"""

from pathlib import Path

import cv2

# Continue button body (x, y, w, h) in native pixels.
LEVELUP_BUTTON = (471, 1737, 338, 151)
# Tap point — user-verified: returns to the lair (no BACK involved; a BACK on
# the bare board would exit the game, but this button is the safe dismiss).
BUTTON_CENTER = (640, 1812)
# Search window around the button for tolerant region matching.
BAND = (420, 1700, 460, 250)

BANK_DIR = "assets/levelup"
THRESHOLD = 0.6            # validated: 1.000 vs <=0.091 on controls
DISMISS_POLLS = 3          # re-screencaps before giving up on the transition
DISMISS_PAUSE = 0.8        # seconds between polls (level-up animates out)


class LevelUpScreen:
    """Detect the Devourer level-up screen and dismiss its Continue button.

    Detection needs only the template bank (no device/LLM); `dismiss` needs a
    `Device` for the tap + verification screencaps.
    """

    def __init__(self, bank_dir: str = BANK_DIR, threshold: float = THRESHOLD):
        self.bank_dir = Path(bank_dir)
        self.threshold = threshold
        self.bank: list = []
        self._load_bank()

    def _load_bank(self) -> None:
        for path in sorted(self.bank_dir.glob("*.png")):
            self.bank.append(cv2.imread(str(path)))

    # ---- crops -------------------------------------------------------------

    def button_crop(self, frame):
        x, y, w, h = LEVELUP_BUTTON
        return frame[y:y + h, x:x + w]

    def _band_crop(self, frame):
        x, y, w, h = BAND
        return frame[y:y + h, x:x + w]

    # ---- banking -----------------------------------------------------------

    def bank_button(self, frame, name: str = "continue") -> Path | None:
        """Persist the current button crop (multi-frame bank). Idempotent: the
        crop is skipped if it already matches the bank at threshold."""
        if frame is None:
            return None
        crop = self.button_crop(frame)
        if self._best(crop) >= self.threshold:
            return None
        path = self.bank_dir / f"{name}__{len(self.bank)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop)
        self.bank.append(crop)
        return path

    # ---- detection ---------------------------------------------------------

    @staticmethod
    def _match(crop, template) -> float:
        """Best template score of `template` inside `crop` (region search)."""
        return float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED).max())

    def _best(self, crop) -> float:
        best = -1.0
        for t in self.bank:
            s = self._match(crop, t)
            if s > best:
                best = s
        return best

    def is_level_up(self, frame) -> tuple[bool, float]:
        """(present, match_score): True when the Continue button is found in
        the detection band at threshold."""
        if frame is None or not self.bank:
            return False, -1.0
        score = self._best(self._band_crop(frame))
        return score >= self.threshold, score

    # ---- dismissal ----------------------------------------------------------

    def dismiss(self, device) -> bool:
        """Tap Continue, then verify the level-up screen cleared (polling a few
        screencaps). Returns True when the screen is gone. Never raises on a
        stubborn screen — the main loop re-checks next step. (Tapping Continue
        is the safe dismiss; no BACK is ever sent here.)"""
        if device is None:
            return False
        device.tap(*BUTTON_CENTER)
        for _ in range(DISMISS_POLLS):
            device.wait_for_idle(DISMISS_PAUSE)
            device.screencap()
            frame = cv2.imread(str(device.screencap_path))
            present, _score = self.is_level_up(frame)
            if not present:
                return True
        return False