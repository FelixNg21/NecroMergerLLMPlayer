#!/usr/bin/env python3
"""
Verify SlimeVatReader OCR pipeline and wiring into vision drive.
Tests 8 cases covering count, capacity, defaults, and integration.
"""
import sys
sys.path.insert(0, '.')
from vision.slime import SlimeVatReader
from vision.grid import FALLBACK_GEOMETRY
import cv2
import numpy as np

print("=== verify_slime_vat.py ===")
print()

# Test 1: Class instantiation
print("Test 1: SlimeVatReader instantiation")
try:
    reader = SlimeVatReader()
    print("  PASS: SlimeVatReader() instantiates")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 2: read_count on empty frame (returns None)
print("Test 2: read_count returns None on blank frame")
blank = np.zeros((100, 100, 3), dtype=np.uint8)
result = reader.read_count(blank)
if result is None:
    print("  PASS: returns None for blank")
else:
    print(f"  FAIL: expected None, got {result}")
    sys.exit(1)

# Test 3: read_capacity on empty frame (returns None)
print("Test 3: read_capacity returns None on blank frame")
result = reader.read_capacity(blank)
if result is None:
    print("  PASS: returns None for blank")
else:
    print(f"  FAIL: expected None, got {result}")
    sys.exit(1)

# Test 4: _overlap identical grayscale images returns 1.0
print("Test 4: _overlap identical images returns 1.0")
glyph = np.ones((20, 20), dtype=np.uint8) * 128
overlap = reader._overlap(glyph, glyph)
if abs(overlap - 1.0) < 0.001:
    print(f"  PASS: overlap = {overlap:.3f}")
else:
    print(f"  FAIL: expected ~1.0, got {overlap:.3f}")
    sys.exit(1)

# Test 5: _overlap different images returns < 1.0
print("Test 5: _overlap different images returns < 1.0")
glyph1 = np.ones((20, 20), dtype=np.uint8) * 100
glyph2 = np.ones((20, 20), dtype=np.uint8) * 200
overlap = reader._overlap(glyph1, glyph2)
if 0 <= overlap < 1.0:
    print(f"  PASS: overlap = {overlap:.3f}")
else:
    print(f"  FAIL: expected < 1.0, got {overlap:.3f}")
    sys.exit(1)

# Test 6: _match returns (None, -1) when bank empty
print("Test 6: _match on empty bank returns (None, -1)")
best, score = reader._match(glyph)
if best is None and score == -1.0:
    print(f"  PASS: best={best}, score={score}")
else:
    print(f"  FAIL: expected (None, -1), got ({best}, {score})")
    sys.exit(1)

# Test 7: bank_token requires glyph image (not string)
print("Test 7: bank_token rejects non-glyph input")
try:
    reader.bank_token("test", "not_a_glyph")
    print("  FAIL: should have raised error")
    sys.exit(1)
except Exception:
    print("  PASS: raised error for non-glyph")
    # Expected

# Test 8: Integration with vision_drive (imports and wiring)
print("Test 8: vision_drive imports and _slime_line_text exists")
from planner.vision_drive import VisionDrivenPlanner
import inspect
if hasattr(VisionDrivenPlanner, '_slime_line_text'):
    sig = inspect.signature(VisionDrivenPlanner._slime_line_text)
    print(f"  PASS: _slime_line_text exists, sig={sig}")
else:
    print("  FAIL: _slime_line_text missing from VisionDrivenPlanner")
    sys.exit(1)

print()
print("=== All 8 tests passed ===")
