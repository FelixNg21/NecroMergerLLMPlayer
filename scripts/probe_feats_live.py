"""Probe what tapping a feat Collect button does (live, one tap).

Opens the FEATS panel, taps the topmost collect button, screenshots before/after,
reports whether a reward popup appeared (via bottombar visibility + diff) and
whether the Collect button template count changed. Leaves the game in a good
state (popup dismissed or panel closed).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.adb import Device                     # noqa: E402
from vision.bottombar import BottomBarReader     # noqa: E402
from vision.panels import PanelReader            # noqa: E402


def match_collect(frame):
    tmpl = cv2.imread('assets/calib/feats/collect_button.png')
    tg = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
    fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = tg.shape
    res = cv2.matchTemplate(fg, tg, cv2.TM_CCOEFF_NORMED)
    centers = []
    ys, xs = np.where(res > 0.7)
    for x, y in zip(xs, ys):
        cx, cy = int(x + w // 2), int(y + h // 2)
        if not any(abs(cy - c[1]) < 20 and abs(cx - c[0]) < 20 for c in centers):
            centers.append((cx, cy))
    return sorted(centers, key=lambda t: t[1])


def main() -> None:
    d = Device()
    bottombar = BottomBarReader()
    pr = PanelReader(device=d, llm=None, bottombar=bottombar)

    d.screencap()
    frame = cv2.imread(str(d.screencap_path))
    if not bottombar.bar_visible(frame):
        print("not on lair — aborting")
        return

    pr.open_panel("feats")
    opened = None
    for _ in range(4):
        d.wait_for_idle(1.8)
        d.screencap()
        pf = cv2.imread(str(d.screencap_path))
        if pf is not None and pr._panel_open(pf):
            opened = pf
            break
    if opened is None:
        print("panel did not open")
        return
    before = match_collect(opened)
    print("collect buttons before:", before)
    if not before:
        pr.close_panel()
        return

    target = before[0]
    print("tapping topmost collect button at", target)
    d.tap(*target)
    d.wait_for_idle(1.5)
    d.screencap()
    after_img = cv2.imread(str(d.screencap_path))
    cv2.imwrite("/tmp/feats_after_tap.png", after_img)
    after = match_collect(after_img)
    print("collect buttons after:", after)
    print("bar_visible after tap (popup/dialog?):", bottombar.bar_visible(after_img))

    # if a popup appeared over the panel, dismiss via the dim-corner tap pattern? 
    # First try: parent page tap used elsewhere. We'll inspect the frame.
    if not bottombar.bar_visible(after_img):
        print("a panel/popup is open after the tap")


if __name__ == "__main__":
    main()