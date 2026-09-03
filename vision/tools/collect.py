import argparse

import cv2

from env.adb import Device
from vision.grid import GridGeometry, detect_grid

TEMPLATES = "assets/templates"
HOVERED = (-1, -1)             # (col, row) of cell under mouse
CAPTURE = None                 # {"name", "frame_idx", "cx", "cy"} while capturing frames
frame = None
geom: GridGeometry | None = None


def cell_center(c, r):
    return geom.cell_center(r, c)


def refresh(device):
    global frame
    if device is not None:
        device.screencap()
        frame = cv2.imread(str(device.screencap_path))
    detect()
    print("frame refreshed")


def detect():
    global geom
    if frame is not None:
        geom = detect_grid(frame)


def save_crop(cx, cy, path):
    half = geom.template_size // 2
    crop = frame[cy - half : cy + half, cx - half : cx + half]   # [row, col]!
    cv2.imwrite(path, crop)


def on_mouse(event, x, y, flags, param):
    global HOVERED, CAPTURE
    c = (x - geom.origin_x) // geom.cell_px
    r = (y - geom.origin_y) // geom.cell_px
    HOVERED = (c, r)
    if CAPTURE is not None:                     # locked to a cell while capturing
        return
    if event == cv2.EVENT_LBUTTONDOWN and 0 <= c < geom.cols and 0 <= r < geom.rows:
        cx, cy = cell_center(c, r)
        name = input("template id (enter to skip): ").strip()
        if name:
            save_crop(cx, cy, f"{TEMPLATES}/{name}__0.png")
            CAPTURE = {"name": name, "frame_idx": 1, "cx": cx, "cy": cy}
            print(f"saved {name}__0.png — space=advance  c=save frame  n=finish  q=quit")


def render():
    out = frame.copy()
    if CAPTURE is not None:
        cx, cy = CAPTURE["cx"], CAPTURE["cy"]
    else:
        c, r = HOVERED
        if not (0 <= c < geom.cols and 0 <= r < geom.rows):
            return out
        cx, cy = cell_center(c, r)
    half = geom.template_size // 2
    cv2.rectangle(out, (cx - half, cy - half), (cx + half, cy + half), (0, 0, 255), 3)
    if CAPTURE is not None:
        cv2.putText(out, f"capturing {CAPTURE['name']}: space=advance  c=frame  n=finish",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return out


def main():
    global frame, CAPTURE
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="static screenshot (instead of live device)")
    args = ap.parse_args()

    device = None
    if args.image:
        frame = cv2.imread(args.image)
        detect()
    else:
        device = Device()
        refresh(device)

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", on_mouse)
    while True:
        cv2.imshow("frame", render())
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if CAPTURE is not None:
            if key == ord("c"):
                if device is not None:
                    refresh(device)
                save_crop(CAPTURE["cx"], CAPTURE["cy"],
                          f"{TEMPLATES}/{CAPTURE['name']}__{CAPTURE['frame_idx']}.png")
                print(f"saved {CAPTURE['name']}__{CAPTURE['frame_idx']}.png")
                CAPTURE["frame_idx"] += 1
            elif key == ord(" "):
                if device is not None:
                    refresh(device)
            elif key == ord("n"):
                print(f"finished {CAPTURE['name']} ({CAPTURE['frame_idx']} frames)")
                CAPTURE = None


if __name__ == "__main__":
    main()
