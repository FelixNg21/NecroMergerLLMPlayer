import argparse
import os

import cv2

from vision import grid
from vision.classifier import TemplateClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "screenshot",
        nargs="?",
        default="screenshots/Screenshot_1785886911.png",
        help="path to the screenshot to evaluate on",
    )
    args = ap.parse_args()

    frame = cv2.imread(args.screenshot)
    geom = grid.detect_grid(frame)
    cells = grid.build_cells(geom)
    clf = TemplateClassifier()

    out = frame.copy()
    for cell, item in zip(cells, clf.classify(frame, cells)):
        _, score = clf.score_cell(frame, cell)
        label = item if item else "-"
        print(f"({cell.row},{cell.col}) {label:14s} {score:.2f}")
        color = (0, 255, 0) if item else (0, 0, 255)
        cv2.putText(out, f"{label} {score:.2f}", (cell.cx - 50, cell.cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    os.makedirs("assets/debug", exist_ok=True)
    cv2.imwrite("assets/debug/classify_check.png", out)
    print("wrote assets/debug/classify_check.png")


if __name__ == "__main__":
    main()
