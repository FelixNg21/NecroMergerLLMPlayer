"""Ad-offer observer (detect-only; never taps).

NecroMerger offers ad rewards (locked-chest unlock is explicitly "watch an
advert" per its popup; free-chest/mana offers also exist). Watching an ad
means leaving the game for 15-30s of video — far too risky to automate
blind, so this module ONLY detects offers and logs them. Tapping/watching
stays explicitly unimplemented until a live offer is captured and the
full watch→return→claim cycle is verified.

Gating: the only grounded ad trigger is the locked chest on the board, so
detection runs only then (full-frame OCR is seconds-expensive; running it
every step would stall the loop for a UI that is absent 99% of the time).
"""

import re

# Matched against lowercased full-frame OCR text.
AD_KEYWORDS = (
    "watch ad",
    "watch video",
    "watch to",
    "free chest",
    "free reward",
    "free mana",
)


def _default_ocr(frame) -> str:
    """Full-frame OCR text via Apple Vision (slow; see gating note)."""
    import cv2
    import tempfile
    from pathlib import Path
    from vision.satiety import apple_vision_text
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
        cv2.imwrite(t.name, big)
        try:
            return apple_vision_text(t.name) or ""
        finally:
            Path(t.name).unlink(missing_ok=True)


def detect_ad_offer(frame, locked_chest_present: bool,
                    ocr=None) -> dict | None:
    """Return {'text': matched_line} when an ad offer is showing, else None.

    Never taps. `ocr` injects the text reader (unit tests); defaults to
    full-frame Apple Vision OCR. Returns fast (no OCR at all) when no
    locked chest is on the board.
    """
    if frame is None or not locked_chest_present:
        return None
    try:
        text = (ocr(frame) if ocr is not None else _default_ocr(frame)) or ""
    except Exception:
        return None
    low = text.lower()
    for line in low.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if any(k in stripped for k in AD_KEYWORDS):
            return {"text": stripped[:120]}
    return None
