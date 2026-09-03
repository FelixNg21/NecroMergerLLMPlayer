"""Repeatable E2E verification for the feats tier-reward collection
(vision/panels.py collect_tier_reward).

Uses persisted fixtures under assets/calib/feats/ (captured Aug 15):
- tier_claimable.png      -> feats panel open, tier reward CLAIMABLE (badge up)
- tier_popup_up.png       -> reward popup covering the screen after the tap
- tier_cleared_lair.png   -> cleared back to the lair (no badge, no popup)
- tier_not_claimable.png  -> feats panel open, NOT claimable (badge gone)

Checks:
A) badge detection: present on claimable, absent on not-claimable/lair.
B) popup detection: band mean ~127 when popup up, ~75/63 otherwise; threshold
   splits cleanly.
C) collect_tier_reward happy path: claimable -> tap -> popup up -> popup
   cleared; returns (frame, True, False), tapped (640,1190), NO back presses.
D) not claimable -> no tap, collected False.
E) popup never clears -> stuck True, caller must not BACK while popup up.
F) collect_feat_rewards integration on claimable: result["tier"] True, and
   because the popup auto-clears back to the lair, no BACK is pressed.
G) stuck popup -> error set, NO BACK pressed.
H) py_compile of the two changed modules.
Run: .venv/bin/python scripts/verify_tier_reward.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.panels import TIER_TAP, PanelReader

FRAMES = {
    "claimable": str(ROOT / "assets" / "calib" / "feats" / "tier_claimable.png"),
    "popup_up": str(ROOT / "assets" / "calib" / "feats" / "tier_popup_up.png"),
    "cleared": str(ROOT / "assets" / "calib" / "feats" / "tier_cleared_lair.png"),
    "not_claimable": str(ROOT / "assets" / "calib" / "feats" / "tier_not_claimable.png"),
}


def frame(name: str):
    img = cv2.imread(FRAMES[name])
    if img is None:
        raise SystemExit(f"MISSING fixture {FRAMES[name]!r}")
    return img


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))


class StubDevice:
    """Device stub: screencap replays a queue of saved-frame paths."""

    def __init__(self, queue, out_path: Path):
        self.queue = list(queue)
        self.screencap_path = out_path
        self.taps = []
        self.back_presses = 0

    def tap(self, x, y):
        self.taps.append((x, y))

    def wait_for_idle(self, seconds):
        pass

    def screencap(self):
        self.screencap_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.queue:
            raise SystemExit("stub device queue exhausted")
        name = self.queue.pop(0)
        src = FRAMES[name]
        shutil.copyfile(src, self.screencap_path)

    def back(self):
        self.back_presses += 1


class StubBottomBar:
    """bottombar stub: reports dock visibility by replay-name.

    Names are passed via `set_current(name)` on each read; the caller drives
    which screen is 'up' independent of the image contents.
    """

    def __init__(self, visible_names):
        self.visible_names = set(visible_names)
        self.current = None

    def set_current(self, name):
        self.current = name

    def bar_visible(self, frame) -> bool:
        return self.current in self.visible_names

    def read_bar(self, frame):
        return [{"name": "feats", "unlocked": True}]


class StubLLM:
    """chat_message stub: returns the feats-panel JSON for read_panel."""

    def chat_message(self, messages, max_tokens=256, temperature=0.0, **kw):
        return '{"feats": [], "tier": 3}', None, None


def named(name: str):
    return frame(name)


def part_a():
    p = PanelReader()
    ok = bool(p._tier_badge_present(named("claimable")))
    ok &= not p._tier_badge_present(named("not_claimable"))
    ok &= not p._tier_badge_present(named("cleared"))
    check("A: badge on claimable / absent elsewhere", ok)


def part_b():
    p = PanelReader()
    up = frame("popup_up")[2240:2560].mean()
    lo = frame("cleared")[2240:2560].mean()
    from vision.panels import TIER_POPUP_MEAN_MIN
    ok = p._reward_popup_present(named("popup_up"))
    ok &= not p._reward_popup_present(named("cleared"))
    ok &= not p._reward_popup_present(named("claimable"))
    ok &= up > TIER_POPUP_MEAN_MIN > lo
    check(f"B: popup band split (up={up:.0f} lo={lo:.0f})", ok)


def part_c():
    # After the tap, screencap #1 is the popup (badge gone -> collected), #2 is
    # the cleared lair (popup gone -> not stuck).
    dev = StubDevice(["claimable", "popup_up", "cleared"], ROOT / "stub_tmp.png")
    p = PanelReader(device=dev)
    final, collected, stuck = p.collect_tier_reward(named("claimable"))
    ok = collected is True and stuck is False
    ok &= dev.taps == [TIER_TAP]
    ok &= dev.back_presses == 0
    ok &= not p._reward_popup_present(final)
    check("C: happy path collects once, taps tier button, no BACK", ok)


def part_d():
    dev = StubDevice([], ROOT / "stub_tmp.png")
    p = PanelReader(device=dev)
    final, collected, stuck = p.collect_tier_reward(named("not_claimable"))
    ok = collected is False and stuck is False and dev.taps == []
    check("D: not claimable -> no tap, not collected", ok)


def part_e():
    dev = StubDevice(["claimable"] + ["popup_up"] * 20, ROOT / "stub_tmp.png")
    p = PanelReader(device=dev)
    final, collected, stuck = p.collect_tier_reward(named("claimable"))
    ok = collected is True and stuck is True
    ok &= p._reward_popup_present(final)
    check("E: popup never clears -> stuck True", ok)


def part_f():
    # Integration: claimable frame, no collect buttons, popup auto-clears to lair.
    dev = StubDevice(["claimable", "popup_up", "cleared"] + ["cleared"] * 8,
                     ROOT / "stub_tmp.png")
    bb = StubBottomBar({"cleared"})
    p = PanelReader(device=dev, llm=StubLLM(), bottombar=bb)
    p.open_panel = lambda name: None
    p._wait_panel_open = lambda: named("claimable")
    p.read_panel = lambda fr, name: {"feats": [], "tier": 3}
    bb.set_current("cleared")
    res = p.collect_feat_rewards(named("claimable"))
    ok = res.get("tier") is True
    ok &= res.get("error") is None
    ok &= dev.taps == [TIER_TAP]
    ok &= dev.back_presses == 0
    check("F: integration collects tier, no BACK (auto-clear to lair)", ok)


def part_g():
    dev = StubDevice(["claimable"] + ["popup_up"] * 20, ROOT / "stub_tmp.png")
    bb = StubBottomBar(set())
    p = PanelReader(device=dev, llm=StubLLM(), bottombar=bb)
    p._check = lambda fr, name: None if name in ("feats",) else "refused"
    p.open_panel = lambda name: None
    p._wait_panel_open = lambda: named("claimable")
    p.read_panel = lambda fr, name: {"feats": [], "tier": 3}
    bb.set_current("popup_up")
    res = p.collect_feat_rewards(named("claimable"))
    ok = res.get("tier") is True
    ok &= res.get("error") == "tier reward popup did not clear — left in place"
    ok &= dev.back_presses == 0
    check("G: stuck popup -> error, still NO BACK", ok)


def part_h():
    ok = True
    for f in ("vision/panels.py", "planner/vision_drive.py"):
        r = subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / f)],
                           capture_output=True)
        ok = ok and r.returncode == 0
    check("H: py_compile", ok)


def part_i_red_badge_detection():
    """ the live panel renders SQUARE reward buttons (~230x230 px)
    with a RED EXCLAMATION BADGE in the upper-right corner when the feat
    is done and claimable. Preview buttons (in-progress feats) have the
    same green border but no badge. Already-collected rows don't have a
    green button at all.

    The structural filter (w=150-280, h=35-80) was calibrated for the OLD
    rectangular layout and excluded every live button — only the legacy
    template path was finding anything, and the count-mismatch guard in
    _claimable_centers then refused to claim. The fix is twofold:
      1. accept square buttons (REWARD_BTN_W/H widened)
      2. detect the red badge as the authoritative claimable signal

    These tests build synthetic button crops and verify the badge
    detector accepts claimable, rejects preview, and rejects already-
    collected (no button at all).
    """
    # Claimable button: green border + icon + red badge in top-right.
    # Build a 233x233 green button with a red circle in the upper-right.
    claimable_btn = np.full((233, 233, 3), (50, 200, 50), dtype=np.uint8)
    # Draw a red circle (badge) in the upper-right corner
    cv2.circle(claimable_btn, (200, 30), 25, (0, 0, 220), -1)
    cv2.circle(claimable_btn, (200, 30), 18, (50, 50, 220), -1)  # hollow center
    # Add an icon in the middle
    cv2.rectangle(claimable_btn, (90, 90), (143, 143), (200, 100, 50), -1)

    # Preview button: same green border + icon, NO red badge.
    preview_btn = np.full((233, 233, 3), (50, 200, 50), dtype=np.uint8)
    cv2.rectangle(preview_btn, (90, 90), (143, 143), (200, 100, 50), -1)

    # Empty/collected: no button (background)
    empty_btn = np.full((233, 233, 3), (40, 40, 80), dtype=np.uint8)

    class FakeDev:
        def tap(self, x, y): pass
        def screencap(self): pass
        def wait_for_idle(self, s): pass
        def back(self): pass
    bb = StubBottomBar(set())
    p = PanelReader(device=FakeDev(), llm=StubLLM(), bottombar=bb)

    # I-1: claimable button has the red badge
    has_badge = p._has_claim_badge(claimable_btn)
    check("I-1: claimable button has the red exclamation badge",
          has_badge is True, f"got {has_badge}")

    # I-2: preview button does NOT have the badge
    has_badge_p = p._has_claim_badge(preview_btn)
    check("I-2: preview button has no red badge",
          has_badge_p is False, f"got {has_badge_p}")

    # I-3: empty crop has no badge
    has_badge_e = p._has_claim_badge(empty_btn)
    check("I-3: empty button has no red badge",
          has_badge_e is False, f"got {has_badge_e}")

    # I-4: structural filter accepts the square 233x233 button
    # Build a synthetic full-frame: the panel-header region + a 233x233
    # green button at y=1310.
    panel_frame = np.full((2856, 1280, 3), (40, 40, 80), dtype=np.uint8)
    # Add a green button with a "Reward" text
    button_y0 = 1310
    button_y1 = button_y0 + 233
    panel_frame[button_y0:button_y1, 831:831+233] = claimable_btn
    # The OCR check requires "reward" text; fake it with a small white rect
    # (the OCR's job is to filter non-Reward buttons; we want to test the
    # structural filter accepts this shape, not the OCR gate).
    # The OCR filter is incidental — verify the structural check only.
    btn_w, btn_h = p.REWARD_BTN_W, p.REWARD_BTN_H
    structural_match = (btn_w[0] <= 233 <= btn_w[1] and
                        btn_h[0] <= 233 <= btn_h[1])
    check("I-4: structural filter accepts 233x233 square button",
          structural_match is True,
          f"btn_w={btn_w}, btn_h={btn_h}, button=233x233")


def part_j_claimable_mismatch():
    """ when the panel read returns 4 feats but only 3 buttons
    render (one already-collected, no green button), the count-mismatch
    guard previously returned []. The new behavior: fall back to
    badge-only detection so the claim still happens."""
    # Build a synthetic frame with 1 claimable button (with red badge) and
    # 2 preview buttons (no badge). The LLM read says 4 feats (one already
    # collected, three showing buttons).
    panel_frame = np.full((2856, 1280, 3), (40, 40, 80), dtype=np.uint8)
    # Row 1: claimable Build Mana Pool (with red badge)
    claimable_btn = np.full((233, 233, 3), (50, 200, 50), dtype=np.uint8)
    cv2.circle(claimable_btn, (200, 30), 25, (0, 0, 220), -1)
    cv2.circle(claimable_btn, (200, 30), 18, (50, 50, 220), -1)
    cv2.rectangle(claimable_btn, (90, 90), (143, 143), (200, 100, 50), -1)
    panel_frame[1310:1543, 831:1064] = claimable_btn
    # Rows 2-3: preview buttons (no badge)
    preview_btn = np.full((233, 233, 3), (50, 200, 50), dtype=np.uint8)
    cv2.rectangle(preview_btn, (90, 90), (143, 143), (200, 100, 50), -1)
    panel_frame[1833:2066, 831:1064] = preview_btn
    panel_frame[2097:2330, 831:1064] = preview_btn

    class FakeDev:
        def tap(self, x, y): pass
        def screencap(self): pass
        def wait_for_idle(self, s): pass
        def back(self): pass
    bb = StubBottomBar(set())
    p = PanelReader(device=FakeDev(), llm=StubLLM(), bottombar=bb)

    # Use 3 manually-placed centers (skip _collect_centers' OCR/text gate
    # by passing centers directly).
    centers = [(947, 1426),   # claimable Build Mana Pool
               (949, 1949),   # preview Own Grave
               (949, 2213)]   # preview Own Zombie
    feats = [
        {"name": "Build a Mana Pool.", "done": True},   # claimable
        {"name": "Open a Chest.", "done": True},         # already collected
        {"name": "Own Grave.", "done": False},           # not done
        {"name": "Own Zombie.", "done": False},          # not done
    ]
    claimable = p._claimable_centers(panel_frame, centers, feats)
    # Should return ONLY the Build Mana Pool center (the only one with
    # the red badge), even though centers count (3) != feats count (4).
    check("J: count-mismatch path returns the badge-marked button only",
          claimable == [(947, 1426)],
          f"got {claimable}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    part_f()
    part_g()
    part_h()
    part_i_red_badge_detection()
    part_j_claimable_mismatch()