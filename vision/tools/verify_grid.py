"""Overlay the detected grid on a screenshot and save to assets/debug/grid_check.png.

Usage: python -m vision.tools.verify_grid [screenshot]
"""

import sys

import cv2

from vision import grid


def main() -> None:
    import os

    os.makedirs("assets/debug", exist_ok=True)
    path = sys.argv[1] if len(sys.argv) > 1 else "screenshots/Screenshot_1785886911.png"
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"screenshot not found: {path} — run from repo root")
    geom = grid.detect_grid(frame)
    out = grid.draw_grid_overlay(frame, geom)
    cv2.imwrite("assets/debug/grid_check.png", out)
    print(f"grid check -> assets/debug/grid_check.png ({geom})")


if __name__ == "__main__":
    main()
