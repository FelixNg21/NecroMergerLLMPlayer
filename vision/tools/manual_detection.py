"""Click-coordinate printer for calibration.

Loads a screenshot and prints the pixel coordinates of every click. By
default it screencaps the live emulator first; pass --path to use a saved
image instead. Click-to-open coordinates are the ground truth for calibrating
UI regions (board origin, cravings bubble, etc.).
"""

import argparse
import sys
from pathlib import Path

import cv2


def load_frame(path: Path | None):
    if path is not None:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"cannot read image: {path}")
        return frame
    from env.adb import Device
    device = Device()
    device.screencap()
    frame = cv2.imread(str(device.screencap_path))
    if frame is None:
        raise SystemExit(f"cannot read screenshot: {device.screencap_path}")
    print(f"captured {device.screencap_path} ({frame.shape[1]}x{frame.shape[0]})")
    return frame


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="print click coordinates for calibration")
    parser.add_argument("--path", type=Path, default=None,
                        help="image to load (default: live emulator screencap)")
    args = parser.parse_args(argv)

    frame = load_frame(args.path)
    title = "frame (click to print coordinates, ESC to exit)"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"click: ({x}, {y})")

    cv2.imshow(title, frame)
    cv2.setMouseCallback(title, on_click)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
