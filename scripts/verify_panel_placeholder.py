"""Verify the Aug 29 panel-population check in `_identify_one`.

The Lair info panel is a persistent UI element (not a modal popup) that
updates when a board cell is tapped. If the tap doesn't land or the
update races the screencap, the panel still shows the placeholder
"Select something to learn more about it." — and the LLM obediently
reads that back as the description (the proactive_identify round was
returning this 100% of the time in early tests).

The fix: `_panel_placeholder_text` is a heuristic on the panel crop
that returns True when the panel still shows the placeholder. The
caller retries the tap if so.

These tests verify the heuristic on real board frames (the user pointed
out the placeholder and the populated panel renderings).
"""
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, '.')

import cv2
from vision.identify import Identifier


# ---------- Test A: placeholder panel is detected ----------

def test_a_placeholder_panel_detected():
    """A board frame where the Lair panel shows the placeholder text
    must return True from _panel_placeholder_text."""
    frame = cv2.imread('/tmp/lair_now.png')
    assert frame is not None, "missing /tmp/lair_now.png fixture"
    # Build a full board frame that contains the placeholder panel
    full = cv2.imread('/tmp/post_close.png')
    assert full is not None
    detected = Identifier._panel_placeholder_text(full)
    assert detected, "placeholder panel was not detected as placeholder"
    print("  PASS: A: placeholder panel detected")
    return True


# ---------- Test B: populated panel is NOT detected as placeholder ----------

def test_b_populated_panel_not_detected():
    """A board frame where the Lair panel shows a real item (e.g.
    Skeleton Lvl 5) must return False from _panel_placeholder_text."""
    full = cv2.imread('/tmp/after_tap.png')
    assert full is not None
    detected = Identifier._panel_placeholder_text(full)
    assert not detected, "populated panel was incorrectly flagged as placeholder"
    print("  PASS: B: populated panel NOT detected as placeholder")
    return True


# ---------- Test C: empty input ----------

def test_c_none_input_safe():
    """None input should be safe (return False — we can't say it's
    placeholder, but we don't want to crash)."""
    assert Identifier._panel_placeholder_text(None) is False
    print("  PASS: C: None input returns False (safe default)")
    return True


# ---------- Main ----------

if __name__ == "__main__":
    tests = [
        test_a_placeholder_panel_detected,
        test_b_populated_panel_not_detected,
        test_c_none_input_safe,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            if t():
                passed += 1
        except AssertionError as exc:
            print(f"  FAIL: {t.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR: {t.__name__}: {exc}")
            failed += 1
    print(f"\n{passed}/{passed+failed} panel-placeholder tests passed")
    sys.exit(0 if failed == 0 else 1)
