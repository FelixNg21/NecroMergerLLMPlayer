"""OCR helpers for the NecroMerger item-info popup title banner.

Tapping any board item opens a popup with a FIXED title banner at POPUP_BANNER
with two text lines:
    top:    "<Name> Lvl <N>"
    bottom: a tag line (varies; ignored here).
Tesseract reads the game's stylized font with systematic confusions (S->K/H,
Z->T, B->H, 5->6/&), so OCR is a LABEL HINT producing (name, level) — the
template-bank id is assembled from both. `identify.py` dedups OCR variance via
signature template-matching on the name region.
"""

import re

import cv2
import pytesseract

POPUP_BANNER = (239, 290, 808, 140)   # (x, y, w, h) full title banner on screen
NAME_REGION = (0, 0, 420, 75)         # (dx, dy, w, h) within banner: name only.
                                      # 420 fits the longest observed name
                                      # ("Valuable Chest" ends at x=344); 340
                                      # truncated it, corrupting the signature.
TOP_LINE = (0, 0, 712, 75)            # within banner: "<Name> Lvl <N>".
                                      # 712 excludes the right-edge ornament
                                      # (starts at x=710) which OCR'd as a
                                      # trailing "fi" on level-less items, while
                                      # keeping the "Lvl N" text (x=584-688).

# "Lvl" is read with many font confusions: "Lvl", "Lul", "Lol", "Le]", "Lv]", "Lu}".
# The digit can also be read as "&" (-> 5). The token is "L" + 0-2 non-space
# non-digit glyphs; the capture group is the digit itself.
_LEVEL_RE = re.compile(r"(?:l[^\s\d]{0,2}\s*[:.]?\s*|&)\s*[:.]?\s*(\d|&)", re.IGNORECASE)
_LETTERS_RE = re.compile(r"[^a-z]")


def crop_banner(frame):
    """Return the fixed popup title banner crop (RGB)."""
    x, y, w, h = POPUP_BANNER
    return frame[y:y + h, x:x + w]


def crop_name(frame):
    """Return the name-only fingerprint crop (RGB) used for signature matching."""
    banner = crop_banner(frame)
    dx, dy, w, h = NAME_REGION
    return banner[dy:dy + h, dx:dx + w]


# The "Lvl N" digit glyph sits at this position within the TOP_LINE crop.
DIGIT_REGION = (655, 28, 40, 34)      # (dx, dy, w, h) within banner


def glyph_crop(frame):
    """Return the level-digit glyph crop (grayscale) for template matching.

    Position is stable across popups (verified skeleton/zombie). More reliable
    than OCR for the digit (this font confuses 5<->6 in tesseract).
    """
    banner = crop_banner(frame)
    dx, dy, w, h = DIGIT_REGION
    return cv2.cvtColor(banner[dy:dy + h, dx:dx + w], cv2.COLOR_BGR2GRAY)


def ocr_text(img, psm=7) -> str:
    """OCR a crop with the calibrated recipe (3x upscale + Otsu)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(th, config=f"--psm {psm}").strip()


def _parse(text: str) -> tuple[str, int | None]:
    """Split an OCR'd top line into (name, level).

    Examples handled: ". Skeleton Lvl 6 fi" -> ("skeleton", 6),
    "| Tonbie Lvl 4 fi" -> ("tonbie", 4), "| Bone f" -> ("bone", None).
    """
    m = _LEVEL_RE.search(text)
    if m:
        level_digit = m.group(1)
        level = 5 if level_digit == "&" else int(level_digit)
        name = text[:m.start()]
    else:
        level = None
        name = text
    name = _LETTERS_RE.sub("", name.lower())
    return name, level


def extract_item_info(frame) -> tuple[str, int | None]:
    """Return (name, level) from a popup screenshot's title banner.

    name is the OCR'd, normalized name (lowercase letters only). level is the
    "Lvl N" digit or None when the item has no level (e.g. Bone).
    """
    banner = crop_banner(frame)
    dx, dy, w, h = TOP_LINE
    top = banner[dy:dy + h, dx:dx + w]
    text = ocr_text(top, psm=7)
    name, level = _parse(text)
    if level is None:
        # psm 7 sometimes merges the number away; retry with psm 6.
        text6 = ocr_text(top, psm=6)
        name6, level6 = _parse(text6)
        if level6 is not None:
            level = level6
            if not name:
                name = name6
    return name, level
