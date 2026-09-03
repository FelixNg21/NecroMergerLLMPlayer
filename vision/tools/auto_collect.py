"""Automated template collection.

Modes (can be combined):
  --cells  "r,c:id;r,c:id"   capture known cells (multi-frame for idle-bob items)
  --discover                 capture unclassified occupied cells to assets/review/

Options:
  --frames N --interval S    multi-frame cadence (static --image => 1 frame)
  --image PATH               static screenshot instead of live device
  --occ-threshold T          min edge-density for a cell to count as "occupied"
  --verify                   re-score the board and list cells still below 0.8
"""

import argparse
import time
from pathlib import Path

import cv2

from env.adb import Device
from vision.classifier import TemplateClassifier
from vision.grid import (
    CELL_PX,
    COLS,
    ORIGIN_X,
    ORIGIN_Y,
    ROWS,
    TEMPLATE_SIZE,
    build_cells,
    cell_center,
    crop_cell,
    detect_grid,
    occupancy_score,
)

TEMPLATES = Path("assets/templates")
REVIEW = Path("assets/review")


def save_crop(frame, row: int, col: int, path: Path, geom=None) -> None:
    cv2.imwrite(str(path), crop_cell(frame, row, col, geom))


def load_frames(device, image, frames: int, interval: float) -> list:
    if image is not None:
        frame = cv2.imread(str(image))
        return [frame] if frame is not None else []
    outs = []
    for i in range(frames):
        device.screencap()
        frame = cv2.imread(str(device.screencap_path))
        if frame is not None:
            outs.append(frame)
        if i < frames - 1:
            time.sleep(interval)
    return outs


def parse_cells(cells: str) -> list[tuple[int, int, str]]:
    spec = []
    for token in cells.split(";"):
        if not token.strip():
            continue
        pos, _, item_id = token.partition(":")
        row, col = pos.split(",")
        spec.append((int(row), int(col), item_id.strip()))
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="", help='r,c:id;r,c:id — capture known cells')
    ap.add_argument("--discover", action="store_true",
                    help="capture unclassified occupied cells to assets/review/")
    ap.add_argument("--frames", type=int, default=4, help="multi-frame captures per cell (default 4)")
    ap.add_argument("--interval", type=float, default=0.25, help="seconds between frames")
    ap.add_argument("--image", default=None, help="static screenshot instead of live device")
    ap.add_argument("--occ-threshold", type=float, default=0.05,
                    help="min edge-density for a cell to count as occupied")
    ap.add_argument("--verify", action="store_true", help="re-score board and list cells below 0.8")
    args = ap.parse_args()

    if not args.cells and not args.discover and not args.verify:
        ap.error("provide --cells, --discover, and/or --verify")

    TEMPLATES.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    device = None if args.image else Device()
    classifier = TemplateClassifier(str(TEMPLATES))

    for row, col, item_id in parse_cells(args.cells):
        frames = load_frames(device, args.image, args.frames, args.interval)
        geom = detect_grid(frames[0]) if frames else None
        for i, frame in enumerate(frames):
            save_crop(frame, row, col, TEMPLATES / f"{item_id}__{i}.png", geom)
        print(f"captured {item_id} at ({row},{col}): {len(frames)} frame(s)")

    if args.discover:
        frames = load_frames(device, args.image, args.frames, args.interval)
        if not frames:
            raise SystemExit("no frame available for discovery")
        base = frames[0]
        geom = detect_grid(base)
        cells = build_cells(geom)
        report = REVIEW / "report.txt"
        with report.open("w") as f:
            for cell in cells:
                best_id, best_score = classifier.score_cell(base, cell)
                occ = occupancy_score(base, cell.row, cell.col, geom)
                if best_score >= classifier.threshold:
                    status = "known"
                elif occ >= args.occ_threshold:
                    status = "review"
                    for i, frame in enumerate(frames):
                        save_crop(frame, cell.row, cell.col,
                                  REVIEW / f"cell_{cell.row}_{cell.col}__{i}.png", geom)
                else:
                    status = "empty"
                f.write(f"({cell.row},{cell.col}) best={best_id} score={best_score:.3f} "
                        f"occ={occ:.3f} {status}\n")
        print(f"discovery done — see {REVIEW} and {report}")
        print("  rename cell_* files to <item>__N.png and move them into assets/templates/")

    if args.verify:
        frame = load_frames(device, args.image, 1, 0)[0]
        geom = detect_grid(frame)
        cells = build_cells(geom)
        print("cells below 0.8:")
        found = False
        for cell in cells:
            best_id, best_score = classifier.score_cell(frame, cell)
            if best_score < classifier.threshold:
                found = True
                print(f"  ({cell.row},{cell.col}) best={best_id} score={best_score:.3f}")
        if not found:
            print("  (none)")


if __name__ == "__main__":
    main()
