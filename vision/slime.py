"""Slime Vat HUD + popup reader (Aug 30).

Two readouts:
- read_count(frame): OCR the slime-vat counter on the HUD (e.g. "8").
  Mirrors the SatietyReader pattern — CLAHE + Apple Vision + a Jaccard
  template bank for the slime digit. Bootstrap from a live "read on
  the screen" manual confirmation. The same approach generalizes to
  the future darkness bar (Q2).
- read_capacity(frame, vat_level=None): When the slimevat popup is
  open, OCR the "Holds up to N" line (or equivalent). Returns None
  if the popup isn't visible. Fullness fraction is slime_count /
  capacity.

Live calibration: the slime counter "8" sits at roughly (615, 200)
through (680, 230) in the 1280x2856 frame (slime-vat badge on the
HUD strip). Calibrate POPUP_VAT_CAPACITY_CROP against the live
slimevat popup at implementation time.
"""

from pathlib import Path

import cv2
import numpy as np
import re
import tempfile

from vision.satiety import apple_vision_text


# Slime-vat counter on the HUD: a single digit to the right of the
# slime pile icon. Live calibration: "8" sits at (615, 200) through
# (680, 230) in the 1280x2856 frame.
SLIME_NUM_CROP = (615, 200, 65, 30)        # (x, y, w, h)

# When the slimevat popup is open, the "Holds up to N" line is in
# the popup body. Coordinates TBD — calibrate at implementation.
POPUP_VAT_CAPACITY_CROP = (0, 0, 0, 0)     # placeholder; TODO calibrate

# Per Q4 = C: permanently empty. Only live popup-OCR populates
# capacity; the offline path always returns None so the line degrades
# to "Slime: 8" (no fake percent).
_DEFAULT_CAPACITY = {}


BANK_DIR = Path("assets/slime")
BANK_THRESHOLD = 0.75


class SlimeVatReader:
    """OCR the slime-vat HUD counter and (when open) its max-capacity line."""

    def __init__(self, bank_dir: Path = BANK_DIR, threshold: float = BANK_THRESHOLD):
        self.bank_dir = Path(bank_dir)
        self.threshold = threshold
        self.bank: dict[str, list] = {}
        self._load_bank()

    def _load_bank(self) -> None:
        for path in sorted(self.bank_dir.glob("*.png")):
            value = path.stem.split("__")[0]
            self.bank.setdefault(value, []).append(
                cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))

    @staticmethod
    def _overlap(a, b, W: int = 130, H: int = 36) -> float:
        """Right-aligned same-height Jaccard (IoU) of two tight-crop masks."""
        ca = np.zeros((H, W), dtype=bool)
        cb = np.zeros((H, W), dtype=bool)

        def paste(c, m):
            m = (m * 1.0).astype(np.float32)
            m = cv2.resize(m, (max(1, int(round(m.shape[1] * H / m.shape[0]))), H),
                           interpolation=cv2.INTER_AREA)
            hh, ww = m.shape
            c[H - hh:H, W - ww:W] = m > 64

        paste(ca, a)
        paste(cb, b)
        inter = (ca & cb).sum()
        union = (ca | cb).sum()
        return float(inter / (union + 1e-9))

    def _match(self, glyph) -> tuple[str | None, float]:
        best, best_score = None, -1.0
        for value, templates in self.bank.items():
            for t in templates:
                s = self._overlap(glyph, t)
                if s > best_score:
                    best, best_score = value, s
        return best, best_score

    def bank_token(self, value: str, glyph, overwrite: bool = False) -> Path | None:
        """Persist a confirmed token glyph under `value` (idempotent unless
        overwrite). Returns the saved path or None if the token is already
        banked with a matching glyph."""
        if not overwrite:
            for t in self.bank.get(value, []):
                if self._overlap(glyph, t) >= self.threshold:
                    return None
        n = len(self.bank.get(value, []))
        path = self.bank_dir / f"{value}__{n}.png"
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), glyph)
        self.bank.setdefault(value, []).append(glyph)
        return path

    def read_count(self, frame) -> int | None:
        """OCR the slime-vat counter; returns int or None when unreadable.

        Mirrors SatietyReader.read_satiety: CLAHE + Apple Vision over a
        normalized crop, then a digit regex. Bootstrap-able from a manual
        "read on the screen" via bank_token.
        """
        if frame is None:
            return None
        x, y, w, h = SLIME_NUM_CROP
        if w == 0 or h == 0:
            return None
        if frame.shape[0] < y + h or frame.shape[1] < x + w:
            return None
        crop = frame[y:y + h, x:x + w]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        big = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
            cv2.imwrite(t.name, big)
            txt = apple_vision_text(t.name)
            Path(t.name).unlink(missing_ok=True)
        m = re.search(r"(\d+)", txt or "")
        return int(m.group(1)) if m else None

    def read_capacity(self, frame, vat_level: int | None = None) -> int | None:
        """Read max-capacity from the open slimevat popup.

        Returns int (e.g. 40) when the popup's "Holds up to N" line is
        readable. Returns None when the popup isn't visible — callers
        fall back to a level-keyed default OR show the raw count
        without a percent (Q4 = C: never guess a capacity).
        """
        if frame is not None:
            x, y, w, h = POPUP_VAT_CAPACITY_CROP
            if w > 0 and h > 0:
                if frame.shape[0] >= y + h and frame.shape[1] >= x + w:
                    crop = frame[y:y + h, x:x + w]
                    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
                    big = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
                        cv2.imwrite(t.name, big)
                        txt = apple_vision_text(t.name)
                        Path(t.name).unlink(missing_ok=True)
                    m = re.search(r"(\d+)", txt or "")
                    if m:
                        return int(m.group(1))
        return None
