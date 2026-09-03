"""Feed-the-Devourer-once bootstrap for the satiety fraction digit bank.

The satiety fraction (top-left lair HUD, band y436-486) is ornate small text
that template/vision-LLM/tesseract all fail to read in general (see AGENTS.md
Aug 13 entry). SatietyReader uses a memory bank of confirmed tokens; this script
grows that bank. It is satiety-aware: it never feeds at a full bar or with an
oversized creature, and it auto-dismisses the full-screen level-up screen that
an overflowing feed triggers.

Usage:
  .venv/bin/python scripts/bank_satiety_digit.py --bank-only --label 1 --den-label 250
      bank the CURRENT on-screen numerator under 1 and denominator under 250,
      with NO feed (the usual way to re-bank after a reader fix or save switch)
  .venv/bin/python scripts/bank_satiety_digit.py --label N --den-label M
      read current satiety, feed ONE creature whose recorded feed value fits the
      remaining bar, then bank the new numerator under N and the denominator
      (constant per level) under M
    --label N      value to bank the (post-feed or current) numerator under
    --den-label M  value to bank the denominator under (the "/M" part)
    --bank-only    bank from the CURRENT frame, no feed (needs --label/--den-label)
    --dry-run      do everything except feed/bank

The gesture: swipe the best feedable creature (satiety-gated via best_feed_cell,
never a station/champion) to the Devourer mouth. Geometry is read from the live
frame (vision LLM) exactly like main.py. Frames land in /tmp/sat_before.png and
/tmp/sat_after.png for the operator to eyeball.

WARNING — never DERIVE the post-feed numerator as `before + feed value`: feeding
the CRAVED item grants a bonus on top, so the on-screen value can be much higher
(observed: 1 + 10 skeleton feed + 50 craving bonus = 61, not 11). Always confirm
the on-screen value. The script refuses to bank a numerator that the reader
already knows under a different label, and re-banking a wrong guess is a one-line
`rm assets/satiety/<value>__*.png`.
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.actions import Layout
from env.adb import Device
from vision import grid, satiety
from vision.classifier import TemplateClassifier
from vision.levelup import LevelUpScreen
from vision.pipeline import classify_board

STATION_PREFIXES = ("grave", "necromerger", "manapool", "manapot")
CHAMPION_PREFIXES = ("peasant", "knight", "cleric", "paladin", "rival", "protector")
NECROMERGER_CELL = (0, 2)


def _guard_label(reader, frame, label, what):
    """Refuse to bank a numerator under a label that contradicts what the reader
    already knows. A craving bonus (or a stale wrong label) makes derived
    arithmetic like `before + feed value` wrong, and re-labeling a banked value
    under a different name silently corrupts the bank. Screen truth wins."""
    known = reader.read_satiety(frame).get("num")
    if known is not None and known != label:
        print(f"      REFUSING: reader reads {what} {known!r} on this frame but "
              f"--label is {label!r}. If {known!r} is what the screen shows, just "
              f"re-run with --label {known} (already banked). If the screen shows "
              f"{label!r}, delete assets/satiety/{known}__*.png first.")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None,
                    help="value to bank the numerator under")
    ap.add_argument("--den-label", default=None,
                    help="value to bank the denominator under (the /M part)")
    ap.add_argument("--bank-only", action="store_true",
                    help="bank from the current frame; do NOT feed")
    ap.add_argument("--dry-run", action="store_true",
                    help="read satiety WITHOUT feeding or banking")
    args = ap.parse_args()

    from planner.glossary import read_feed_values
    from planner.llm_client import LLMClient
    from vision.geometry import llm_grid_geometry

    reader = satiety.SatietyReader()
    device = Device()
    classifier = TemplateClassifier("assets/templates", seed=True)

    print("[1/5] geometry read (vision LLM)...")
    device.screencap()
    before = cv2.imread(str(device.screencap_path))
    if before is None:
        print("screencap failed")
        return 1
    llm = LLMClient()
    geometry = llm_grid_geometry(before, llm, classifier)
    grid.set_grid_geometry(geometry)
    layout = Layout(devourer_xy=(geometry.mouth_x, geometry.mouth_y), geom=geometry)
    print(f"      geometry: {geometry.rows}x{geometry.cols} "
          f"origin=({geometry.origin_x},{geometry.origin_y}) cell={geometry.cell_px}")

    print("[2/5] current satiety:", reader.read_satiety(before))
    if args.dry_run:
        print("[3/5] dry-run: skipping feed and banking")
        return 0

    if args.bank_only:
        toks = reader.split_tokens(before)
        if toks is None:
            print("      strip not parseable; nothing to bank")
            return 2
        num_mask, den_mask = toks
        if args.label:
            if not _guard_label(reader, before, args.label, "numerator"):
                return 8
            p = reader.bank_token(args.label, num_mask)
            print(f"[bank-only] numerator banked assets/satiety/{p.name}" if p
                  else f"[bank-only] numerator {args.label!r} already banked")
        if args.den_label:
            p = reader.bank_token(args.den_label, den_mask)
            print(f"[bank-only] denominator banked assets/satiety/{p.name}" if p
                  else f"[bank-only] denominator {args.den_label!r} already banked")
        if not args.label and not args.den_label:
            print("      --bank-only needs --label and/or --den-label")
            return 2
        return 0

    read = reader.read_satiety(before)
    num, den = read.get("num"), read.get("den")
    if num is None or den is None:
        print("satiety tokens not banked yet; cannot gate the feed.")
        print("run --bank-only --label N --den-label M first to bank the tokens "
              "that are on screen now, then re-run to bank a feed increment.")
        return 3
    remaining = int(den) - int(num)
    if remaining <= 0:
        print(f"bar full ({num}/{den}) — feeding would waste food / level up.")
        print("level up the Devourer (or run later) and re-run when the bar is empty.")
        return 4

    board = classify_board(before, classifier)
    from planner.agent import best_feed_cell
    target = best_feed_cell(board, feed_values=read_feed_values(),
                            remaining_satiety=remaining)
    if target is None:
        print(f"no creature on board whose feed value fits {remaining} remaining; "
              "not feeding (no waste). Spawn/soak a cheap creature first.")
        return 5

    print(f"[3/5] feeding {target.item_id} from ({target.row},{target.col}) "
          f"-> mouth (satiety range {remaining} left)")
    x, y = layout.cell_center(target.row, target.col)
    device.swipe(x, y, *layout.devourer_xy, duration_ms=1000)
    device.wait_for_idle(2.0)

    print("[4/5] post-feed screencap...")
    device.screencap()
    after = cv2.imread(str(device.screencap_path))
    levelup = LevelUpScreen()
    if levelup.is_level_up(after)[0]:
        print("      level-up screen detected; tapping Continue")
        levelup.dismiss(device)
        device.screencap()
        after = cv2.imread(str(device.screencap_path))
    cv2.imwrite("/tmp/sat_before.png", before)
    cv2.imwrite("/tmp/sat_after.png", after)
    read_after = reader.read_satiety(after)
    print("      satiety now:", read_after)

    if args.den_label:
        den_toks = reader.split_tokens(before)
        if den_toks is None:
            print("      strip unreadable on the pre-feed frame; not banking den")
            return 6
        p = reader.bank_token(args.den_label, den_toks[1])
        print(f"[5/5] denominator banked assets/satiety/{p.name}" if p
              else f"[5/5] denominator {args.den_label!r} already banked")

    if args.label:
        toks = reader.split_tokens(after)
        if toks is None:
            print("      no numerator tokens on the post-feed frame; not banking")
            return 7
        pre_num = reader.split_tokens(before)[0]
        if reader.overlap(toks[0], pre_num) >= 0.95:
            print("      REFUSING: post-feed numerator matches the pre-feed one "
                  "-- the feed likely did not register; nothing new to bank.")
            return 9
        if not _guard_label(reader, after, args.label, "post-feed numerator"):
            return 8
        p = reader.bank_token(args.label, toks[0])
        print(f"[5/5] numerator banked assets/satiety/{p.name}" if p
              else f"[5/5] numerator {args.label!r} already banked")
        print("      confirm /tmp/sat_after.png matches the screen. A craving "
              "feed grants a BONUS, so the value is NOT before+feed_value. "
              "If the banked label is wrong: rm assets/satiety/*__0.png and re-run.")
    elif not args.den_label:
        print("[5/5] no --label given; not banking (saved /tmp/sat_after.png)")
        print("      confirm the new value on screen, then re-run with --label")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())