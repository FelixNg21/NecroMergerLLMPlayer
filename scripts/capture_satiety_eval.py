"""Capture labeled satiety-eval frames by walking the numerator with feeds.

Walks the Devourer satiety bar across distinct values: anchors the current
value from the template bank (or OCR consensus), performs cheap feeds one at a
time, and saves before/after full frames plus an expectation chain
(meta.json). Labels are resolved LATER by engine consensus
(scripts/test_satiety_ocr.py --autolabel) + human arbitration of the contact
sheet — this script never guesses a label itself.

Usage:
  .venv/bin/python scripts/capture_satiety_eval.py --feeds 8 [--serial emulator-5556]
  .venv/bin/python scripts/capture_satiety_eval.py --watch 120   # passive
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.actions import Layout  # noqa: E402
from env.adb import Device  # noqa: E402
from planner.glossary import read_feed_values  # noqa: E402
from vision import grid, satiety  # noqa: E402
from vision.classifier import TemplateClassifier  # noqa: E402
from vision.geometry import llm_grid_geometry  # noqa: E402
from vision.levelup import LevelUpScreen  # noqa: E402
from vision.pipeline import classify_board  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "calib" / "satiety_frames"


def grab(device: Device) -> object:
    device.screencap()
    return cv2.imread(str(device.screencap_path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeds", type=int, default=8)
    ap.add_argument("--watch", type=int, default=0,
                    help="passive capture seconds instead of feeding")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--serial", default="emulator-5556")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reader = satiety.SatietyReader()
    device = Device(serial=args.serial)
    classifier = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)

    print("[1/3] geometry read...")
    llm = None
    from planner.llm_client import LLMClient
    llm = LLMClient()
    first = grab(device)
    geometry = llm_grid_geometry(first, llm, classifier)
    grid.set_grid_geometry(geometry)
    layout = Layout(devourer_xy=(geometry.mouth_x, geometry.mouth_y),
                    geom=geometry)
    print(f"      {geometry.rows}x{geometry.cols} "
          f"@ ({geometry.origin_x},{geometry.origin_y}) cell={geometry.cell_px}")

    idx = 0
    meta: dict[str, dict] = {}

    def save(frame, tag: str, **info) -> str:
        nonlocal idx
        name = f"frame_{idx:03d}_{tag}.png"
        cv2.imwrite(str(OUT_DIR / name), frame)
        entry = {k: v for k, v in info.items() if v is not None}
        if entry:
            meta[name] = entry
        idx += 1
        return name

    if args.watch:
        print(f"[2/3] watching {args.watch}s every {args.interval}s — play!")
        t_end = time.time() + args.watch
        while time.time() < t_end:
            save(grab(device), "w")
            time.sleep(args.interval)
        (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[3/3] saved {idx} frames -> {OUT_DIR}")
        return 0

    from planner.agent import best_feed_cell

    print("[2/3] anchor + feed walk")
    anchor = grab(device)
    read0 = reader.read_satiety(anchor)
    num0, den0 = read0.get("num"), read0.get("den")
    save(anchor, "a", expected=num0, den=den0, source="bank-anchor")
    if num0 is None:
        print("      WARNING: template bank cannot anchor the start value; "
              "labels will rely purely on OCR consensus.")

    feed_values = read_feed_values()
    fed_total = 0
    prev_expected = int(num0) if num0 else None
    for i in range(args.feeds):
        board = classify_board(grab(device), classifier)
        target = best_feed_cell(board, feed_values=feed_values,
                                remaining_satiety=None)
        if target is None:
            print(f"      [{i}] nothing feedable on board; stopping")
            break
        before = grab(device)
        z = feed_values.get(target.item_id)
        x, y = layout.cell_center(target.row, target.col)
        print(f"      [{i}] feeding {target.item_id} "
              f"(z={z}) ({target.row},{target.col})")
        device.swipe(x, y, *layout.devourer_xy, duration_ms=1000)
        device.wait_for_idle(2.0)
        after = grab(device)
        lvl = LevelUpScreen()
        if lvl.is_level_up(after)[0]:
            print("      level-up screen; dismissing")
            lvl.dismiss(device)
            after = grab(device)
        nb = f"frame_{idx:03d}_b.png"; cv2.imwrite(str(OUT_DIR / nb), before); idx += 1
        exp_b = prev_expected if prev_expected is not None else None
        if exp_b is not None:
            meta[nb] = {"expected": str(exp_b), "den": den0, "z": z}
        na = save(after, "a",
                  expected=(str(prev_expected + (z or 0))
                            if (prev_expected is not None and z is not None)
                            else None),
                  den=den0, z=z, item=target.item_id)
        if prev_expected is not None and z is not None:
            prev_expected += z
            fed_total += z
        time.sleep(1.0)

    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[3/3] saved {idx} frames -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
