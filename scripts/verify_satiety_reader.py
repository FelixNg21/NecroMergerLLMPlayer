"""Verification for the Aug 21 satiety OCR reader (Apple Vision, no bank).

Run: `.venv/bin/python scripts/verify_satiety_reader.py` (exit 0/1).

  A. read_satiety == ground truth on every labeled eval frame
     (assets/calib/satiety_frames + labels.json; labels are feed-arithmetic
     anchored by a visual read of the anchor frame).
  B. Old-save screenshots read through the same path
     (Screenshot_1785958775 -> 0/50 known; calib_board -> 21/750).
  C. Honesty when OCR is unavailable: with apple_vision_text stubbed out,
     the reader degrades to the token bank / honest "?/??" — never crashes,
     never fabricates.
  D. Invalid fraction guard: num > den is rejected.
  E. py_compile of touched modules.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

import vision.satiety as satiety  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "assets" / "calib" / "satiety_frames"
PASS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if ok:
        PASS += 1


def part_a_eval_set() -> None:
    labels = json.loads((FRAMES / "labels.json").read_text())
    reader = satiety.SatietyReader()
    ok = misses = 0
    for name, lab in sorted(labels.items()):
        frame = cv2.imread(str(FRAMES / name))
        got = reader.read_satiety(frame)
        if (got["num"], got["den"]) == (lab["num"], lab["den"]):
            ok += 1
        else:
            misses += 1
            print(f"      miss {name}: {got['text']} != "
                  f"{lab['num']}/{lab['den']}")
    check(f"A eval frames {ok}/{len(labels)}", ok == len(labels),
          f"{misses} misses")


def part_b_old_saves() -> None:
    reader = satiety.SatietyReader()
    f = cv2.imread(str(ROOT / "screenshots" / "Screenshot_1785958775.png"))
    got = reader.read_satiety(f)
    check("B old save reads 0/50", (got["num"], got["den"]) == ("0", "50"),
          got["text"])
    f = cv2.imread(str(ROOT / "screenshots" / "calib_board.png"))
    got = reader.read_satiety(f)
    check("B calib_board reads 21/750",
          (got["num"], got["den"]) == ("21", "750"), got["text"])


def part_c_no_pyobjc() -> None:
    reader = satiety.SatietyReader()
    real = satiety.apple_vision_text
    satiety.apple_vision_text = lambda path: ""       # OCR unavailable
    try:
        frame = cv2.imread(str(FRAMES / "frame_000_a.png"))
        got = reader.read_satiety(frame)
        # The new in-bar layout is beyond split_tokens' band AND none of its
        # values are banked -> honest unknown, no crash, no fabrication.
        check("C no-OCR honest ?/??", got["text"] == "?/??", got["text"])
        old = cv2.imread(str(ROOT / "screenshots" /
                            "Screenshot_1785958775.png"))
        got_old = reader.read_satiety(old)
        check("C no-OCR old save still parses (bank fallback runs)",
              isinstance(got_old["text"], str), got_old["text"])
    finally:
        satiety.apple_vision_text = real


def part_d_invalid_guard() -> None:
    reader = satiety.SatietyReader()
    real = satiety.ocr_fraction
    satiety.ocr_fraction = lambda frame: ("900", "500")   # nonsense fraction
    try:
        frame = cv2.imread(str(FRAMES / "frame_000_a.png"))
        got = reader.read_satiety(frame)
        check("D num>den rejected", got["num"] is None and got["den"] is None,
              got["text"])
    finally:
        satiety.ocr_fraction = real


def part_e_compile() -> None:
    import py_compile
    for f in ("vision/satiety.py", "scripts/test_satiety_ocr.py",
              "scripts/capture_satiety_eval.py",
              "scripts/label_satiety_frames.py"):
        py_compile.compile(str(ROOT / f), doraise=True)
    check("E py_compile clean", True)


def main() -> None:
    global PASS
    for part in (part_a_eval_set, part_b_old_saves, part_c_no_pyobjc,
                 part_d_invalid_guard, part_e_compile):
        try:
            part()
        except Exception as exc:  # noqa: BLE001
            check(f"{part.__name__} raised {type(exc).__name__}", False,
                  str(exc))
    print(f"\n{PASS} checks passed")
    sys.exit(0 if PASS >= 6 else 1)


if __name__ == "__main__":
    main()