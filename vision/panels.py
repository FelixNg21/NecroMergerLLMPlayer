"""Bottom-bar dock panel reader (tap_button tool, Phase 3 MenuExplorer).

The lair dock (`vision/bottombar.py`) has 5 buttons. Two of them open real
full-screen panels the model can safely open → read → close:

- FEATS (missions): read-only tier + task list. Safe.
- STATION (buy stations like the Grave): read-only for now — the panel lists
  stations with their costs, but the tool never taps inside it (no purchase).

The other three are NOT panel-read targets:
- QUEUE: a direct placement action, not a menu — when it holds a reward the
  item is shown in the bar itself and tapping the bar places it on the board.
  NEVER offered as a panel read (tapping it mutates the board).
- SPELLBOOK / SHOP: locked until a higher Devourer level on this save.

Safety invariants (from AGENTS.md "Menu exploration Phase 1" + cravings):
- BACK only when a panel is CONFIRMED open — never on the bare board (BACK
  exits the game). Confirmation = bar_visible flips False after the tap (the
  full-screen panels cover the dock band).
- Locked / non-panel buttons are refused before any tap.
- After closing, verify the dock returned; raise BoardLostError otherwise.
- The read is read-only: no taps inside the panel, so nothing is ever bought.
"""

import base64
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from planner.llm import _extract_json
from planner.llm_client import LLMError
from planner.llm_vision import _encode_frame
from vision.bottombar import BottomBarReader, ICON_CROPS
from vision.cravings import BoardLostError, game_in_foreground
from vision.satiety import apple_vision_text

# Panel body ROI read by the LLM (x, y, w, h) in native pixels. The feats and
# station panels are full-screen: content sits below the top status/title band
# and above the bottom dock.
PANEL_BODY = (0, 300, 1280, 2200)

# Buttons that open read-only panels (vs placement / locked).
PANEL_BUTTONS = ("feats", "station")
# Buttons that must NEVER be treated as panels (explicit refusal reasons).
PLACEMENT_BUTTONS = {"queue": "tapping the Queue docks the reward item onto the board (a placement action, not a readable panel)"}
LOCKED_BUTTONS = {"spellbook", "shop"}

# Feat-reward collection. A finished feat shows a fixed-size blue 'Collect'
# button at the row's right edge (202x53 in natives, template in
# assets/calib/feats/collect_button.png, verified live Aug 15: tapping it
# collects instantly, button disappears, row re-renders — NO separate reward
# popup to dismiss). We template-match for these buttons and tap each one,
# bounded, verifying the button count drops after each tap.
COLLECT_BUTTON_TEMPLATE = str(Path(__file__).resolve().parents[1] / "assets" / "calib" / "feats" / "collect_button.png")
COLLECT_MATCH_MIN = 0.7
COLLECT_TAPS_MAX = 4                # never tap more collect buttons per visit
COLLECT_SETTLE = 1.2                 # let the row re-render after each tap
COLLECT_VERIFY_POLLS = 4             # re-screencap budgets before accepting a tap
COLLECT_VERIFY_PAUSE = 0.8

# Tier-completion reward collection. When all missions of a Tier are done, the
# feats panel shows a wide tier reward button (green band x174-1105, y1096-1284)
# with a small red circle-exclamation badge at its top-right (x995-1069,
# y1075-1143, ~74x68, center ~(1032,1109)) — the badge is present ONLY while the
# reward is claimable. Tapping the button center (640,1190) claims the reward.
# The badge is detected with an anchored template match (claimable 1.000 vs
# not-claimable 0.427 vs other states -0.010, verified Aug 15); template 80x85.
TIER_BADGE_TEMPLATE = str(Path(__file__).resolve().parents[1] / "assets" / "calib" / "feats" / "tier_reward_badge.png")
TIER_BADGE_ROI = (985, 1065, 1080, 1155)     # (x0, y0, x1, y1) anchored search box
TIER_BADGE_MIN = 0.8
TIER_TAP = (640, 1190)                       # tier reward button center (green band)
TIER_POPUP_BAND = (2240, 2560)               # y band: a reward popup raises mean gray
TIER_POPUP_MEAN_MIN = 90                     # ~123 popup up vs ~62 panel/lair down
TIER_POPUP_POLLS = 8                         # wait budget for the popup to clear
TIER_POPUP_PAUSE = 0.8

OPEN_PAUSE = 1.8            # let the panel animate in after the tap
OPEN_POLLS = 4              # re-check the panel for up to OPEN_PAUSE*OPEN_POLLS sec
CLOSE_PAUSE = 0.8           # let the board return after BACK
CLOSE_POLLS = 6             # re-check the dock for up to CLOSE_PAUSE*CLOSE_POLLS sec

FEATS_QUESTION = """This is the FEATS panel of NecroMerger (screen crop). Read the current feat tier
and every task visible: its name, progress, and whether it is marked Done.
Reply with ONLY JSON:
{"feats": [{"name": "<text>", "progress": "<text>", "done": <bool>}], "tier": <int or null>}
If the panel is unreadable, return {"feats": [], "tier": null}. No prose outside the JSON."""

STATION_QUESTION = """This is the STATION panel of NecroMerger (screen crop). List every station shown
and its cost. Reply with ONLY JSON:
{"stations": [{"name": "<text>", "cost": "<text or null>"}]}
Do NOT tap or suggest tapping any Buy/Confirm button. If unreadable, return
{"stations": []}. No prose outside the JSON."""

PANEL_QUESTIONS = {
    "feats": FEATS_QUESTION,
    "station": STATION_QUESTION,
}


class PanelReader:
    """Open a dock panel, LLM-read it, and close it, with cravings-style
    safety: BACK only after the panel is confirmed open, and dock-return
    verification after closing.

    Only `feats` / `station` are openable (real full-screen panels). Queue is
    a placement action and spellbook/shop are locked — all refused before any
    tap, so a tap only ever happens when a real panel will open.
    """

    def __init__(self, device=None, llm=None, bottombar: BottomBarReader | None = None):
        self.device = device
        self.llm = llm
        self.bottombar = bottombar or BottomBarReader()

    # ---- open / read / close -------------------------------------------

    # live reward buttons are SQUARE (~230x230 px) with a green border,
    # not the rectangular shape the original 2024 calibration used. The legacy
    # filter `35 <= h <= 80` excluded every live button; only the legacy
    # template-match path was finding anything. Update the structural filter
    # to accept both shapes (rectangular fallback + square layout).
    REWARD_BTN_W = (100, 320)
    REWARD_BTN_H = (30, 280)         # covers both rect (h=35-80) and square (h=200+)
    # Badge ROI: a small window in the very top-right corner. The badge is
    # a saturated-red circle ~30-40 px wide that overlaps the button's
    # top-right corner. A wider ROI catches red icon bleed (e.g. Own
    # Grave's red hourglass); a too-tight ROI misses small badges.
    REWARD_BADGE_R_PIXELS = 35
    REWARD_BADGE_ROI = (0.65, 0.0, 1.0, 0.30)   # (x0, y0, x1, y1) relative — widened for bottom-row clipping

    def _has_claim_badge(self, button_crop) -> bool:
        """True when the green reward button has the red exclamation badge
        in its upper-right corner. Claimable buttons have this badge;
        preview buttons (in-progress feats) do not; already-collected
        rows don't have a green button at all.

        The badge is a saturated-red circle with a white '!' in the
        top-right corner. A wider ROI (e.g. upper half) catches red
        icon bleed from preview buttons (e.g. the red hourglass on
        the 'Own a lvl 3+ Grave' button); the tight top-right corner
        strip is the discriminating region.
        """
        h, w = button_crop.shape[:2]
        x0 = int(w * self.REWARD_BADGE_ROI[0])
        y0 = int(h * self.REWARD_BADGE_ROI[1])
        x1 = int(w * self.REWARD_BADGE_ROI[2])
        y1 = int(h * self.REWARD_BADGE_ROI[3])
        roi = button_crop[y0:y1, x0:x1]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Red hue: 0-10 or 170-180. Tight saturation/value thresholds
        # filter out the dim red of partial icons bleeding into the
        # corner.
        red = ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) \
              & (hsv[:, :, 1] > 130) & (hsv[:, :, 2] > 130)
        return int(red.sum()) >= self.REWARD_BADGE_R_PIXELS

    def _collect_centers(self, frame) -> list[tuple[int, int]]:
        """Locate feat 'Reward' claim buttons -> sorted center list.

        Returns [] when none are present (all collected / none ready).

        The button DRAWS THE REWARD ITEM inside itself, so a full-button
        template only ever matches the exact feat it was cut from (Aug 24:
        zero claims across a 60-step soak — every completed feat's button
        differed from the template). Detection is now structural:
        saturated-green button-sized rectangles (or squares, in the current
        live layout) in the missions region, verified by OCRing 'Reward'
        in the crop. The legacy full-button template still accepts as a
        secondary signal (identical-art buttons).

        live buttons are square (~230x230 px) with a red exclamation
        badge in the upper-right when claimable. The original rectangular
        filter (w=150-280, h=35-80) excluded every live button, so the
        legacy template path was the only thing finding anything. Updated
        the filter to accept both shapes."""
        if frame is None:
            return []
        centers: list[tuple[int, int]] = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90)
                 & (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 60)).astype(np.uint8)
        # missions region: below the panel header, above the dock
        y0, y1 = 600, 2600
        search = green[y0:y1, :]
        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(search, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            # accept both rectangular (h=35-80) and square (h=200-280)
            # reward-button shapes. The 0.35 area fraction guard drops
            # hollow / sparse greens (e.g. a chest icon's outline).
            if not (self.REWARD_BTN_W[0] <= w <= self.REWARD_BTN_W[1]
                    and self.REWARD_BTN_H[0] <= h <= self.REWARD_BTN_H[1]):
                continue
            if area < 0.35 * w * h:
                continue
            cx, cy = x + w // 2, y0 + y + h // 2
            crop = frame[y0 + y:y0 + y + h, x:x + w]
            if self._ocr_contains_reward(crop):
                centers.append((cx, cy))
        # legacy template accept (secondary — identical-art buttons)
        tpl = cv2.imread(COLLECT_BUTTON_TEMPLATE, cv2.IMREAD_COLOR)
        if tpl is not None:
            res = cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res > COLLECT_MATCH_MIN)
            for y, x in zip(ys, xs):
                centers.append((x + tpl.shape[1] // 2,
                                y + tpl.shape[0] // 2))
        # dedup within 40px, reading order
        dedup: list[tuple[int, int]] = []
        for c in sorted(set(centers), key=lambda p: (p[1], p[0])):
            if all(abs(c[0] - d[0]) > 40 or abs(c[1] - d[1]) > 40
                   for d in dedup):
                dedup.append(c)
        return sorted(dedup, key=lambda p: (p[1], p[0]))

    @staticmethod
    def _ocr_contains_reward(crop) -> bool:
        """True when Apple Vision reads 'Reward' in the button crop —
        item-icon-independent verification for structurally-found green
        buttons."""
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        big = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
            cv2.imwrite(t.name, big)
            txt = apple_vision_text(t.name)
            Path(t.name).unlink(missing_ok=True)
        return "reward" in (txt or "").lower()

    # ---- tier-completion reward -----------------------------------------

    def _tier_badge_tpl(self):
        if getattr(self, "_tier_badge_tpl_cache", None) is None:
            self._tier_badge_tpl_cache = cv2.imread(TIER_BADGE_TEMPLATE, cv2.IMREAD_COLOR)
        return self._tier_badge_tpl_cache

    def _tier_badge_present(self, frame) -> bool:
        """True when the tier-reward red badge is claimable in `frame`.

        Anchored template match on the badge's screen box (the badge sits at the
        tier button's top-right corner, x995-1069 y1075-1143). Verified live:
        claimable 1.000, not-claimable 0.427, other states -0.010.
        """
        if frame is None:
            return False
        tpl = self._tier_badge_tpl()
        if tpl is None:
            return False
        x0, y0, x1, y1 = TIER_BADGE_ROI
        roi = frame[y0:y1, x0:x1]
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        return bool(np.any(res > TIER_BADGE_MIN))

    def _reward_popup_present(self, frame) -> bool:
        """True when a reward popup covers the screen.

        The popup raises the mean gray of the y2240-2560 band to ~123 vs ~62
        for a normal panel/lair (verified on post_tier_tap/settle/current vs
        feats_now/state_now frames). Used so we NEVER BACK while the popup is
        up (BACK on a popup or bare board exits the game).
        """
        if frame is None:
            return False
        y0, y1 = TIER_POPUP_BAND
        g = cv2.cvtColor(frame[y0:y1], cv2.COLOR_BGR2GRAY)
        return float(g.mean()) >= TIER_POPUP_MEAN_MIN

    def collect_tier_reward(self, opened):
        """Claim the tier-completion reward when the badge is up.

        Taps the tier button center (640,1190), verifies the badge disappeared
        (reward consumed), then waits (bounded) for the reward popup to clear —
        popups clear on their own (observed live) and we NEVER BACK on them.
        Returns (frame, collected: bool, stuck: bool) where `frame` is the
        latest screencap (for the caller's close decision) and `stuck` means
        the popup was still up when the wait budget ran out.
        """
        if not self._tier_badge_present(opened):
            return opened, False, False
        cx, cy = TIER_TAP
        self.device.tap(cx, cy)
        collected = False
        frame = opened
        for _ in range(COLLECT_VERIFY_POLLS):
            self.device.wait_for_idle(COLLECT_VERIFY_PAUSE)
            self.device.screencap()
            nxt = cv2.imread(str(self.device.screencap_path))
            if nxt is None:
                continue
            frame = nxt
            if not self._tier_badge_present(nxt):
                collected = True
                break
        if not collected:
            return frame, False, False
        stuck = False
        for _ in range(TIER_POPUP_POLLS):
            self.device.wait_for_idle(TIER_POPUP_PAUSE)
            self.device.screencap()
            nxt = cv2.imread(str(self.device.screencap_path))
            if nxt is None:
                continue
            frame = nxt
            if not self._reward_popup_present(nxt):
                stuck = False
                break
            stuck = True
        return frame, True, stuck

    def _claimable_centers(self, frame, centers, feats) -> list[tuple[int, int]]:
        """Filter detected Reward buttons down to CLAIMABLE ones (Aug 27).

        The live panel renders square reward buttons (~230x230 px) with a
        RED EXCLAMATION BADGE in the upper-right when the feat is done and
        claimable. Preview buttons (in-progress feats) have the same green
        border but no badge. Already-collected rows don't have a green
        button at all (they show a blue checkmark instead).

        The badge is the authoritative signal — it does not depend on the
        panel-read's done flag (which is unreliable on this save) and does
        not depend on button/feat count matching (the LLM may report 4
        feats while the panel shows 3 green buttons because the 4th is
        already-collected).

        ALL-DONE SHORT-CIRCUIT (Sep 4): the badge design is not stable
        across rows — the bottom-row claim button renders with NO corner
        badge (verified live: 4/4 feats Done, single green "Reward" button
        at (949,2285), zero red px in the badge ROI, only red label text
        inside the button). But preview buttons can only exist for
        IN-PROGRESS feats, so when every cached feat is done, every green
        button is a claim by elimination — no badge needed. This is airtight
        (it cannot misfire on a preview) and covers exactly the stuck case.
        """
        feats = feats or []
        if not centers:
            return []
        if feats and all(bool(f.get("done")) for f in feats):
            return list(centers)
        if not feats or len(centers) == len(feats):
            # Pairing logic preserved for backward compat: if the LLM
            # read is well-formed, prefer the done-flag mapping. The
            # badge check is the secondary signal in case the LLM got
            # done flags wrong.
            out = []
            for c, f in zip(centers, feats):
                if f.get("done") or self._center_has_claim_badge(frame, c):
                    out.append(c)
            return out
        # Count mismatch (live Aug 27 case: 4 feats, 3 buttons, 1 already
        # collected). Fall back to badge-only detection — the LLM read
        # can not tell us which center is which, but the badge can.
        return [c for c in centers if self._center_has_claim_badge(frame, c)]

    def _center_has_claim_badge(self, frame, center) -> bool:
        """True when the button centered at `center` has the red exclamation
        badge. Crops a 233-px window (the live button size) around the center
        — bigger than the button would include the surrounding brown
        panel which has its own red bleed; smaller would clip the button.
        The 233-px crop puts the button corners at the crop corners, so
        the ROI strip is exactly the button's top-right corner."""
        cx, cy = int(center[0]), int(center[1])
        s = 233                  # live button edge length
        x0 = max(cx - s // 2, 0)
        y0 = max(cy - s // 2, 0)
        x1 = min(cx + s // 2, frame.shape[1])
        y1 = min(cy + s // 2, frame.shape[0])
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return False
        return self._has_claim_badge(crop)

    def collect_feat_rewards(self, frame) -> dict:
        """Open the FEATS panel and collect every completed feat reward.

        Returns {"collected": n, "panel": <parsed read or None>, "error": ...}.

        The collect buttons are template-matched (safe: only fixed-size blue
        collect buttons are ever tapped, never inside-panel UI like Buy).
        Each tap is verified — the tapped center must disappear from the next
        screencap's match set — before the next tap; taps are capped at
        COLLECT_TAPS_MAX. Same board-return safety as tap_button (BACK only
        when the panel is confirmed open, dock-return verification). Caller
        (vision_drive) invokes this from _observe_feats so rewards are claimed
        as soon as feats are — historically the bot read status but never
        collected the reward.
        """
        result = {"collected": 0, "panel": None, "error": None}
        if self.device is None or self.llm is None:
            result["error"] = "feat collection requires live mode with an LLM"
            return result
        refuse = self._check(frame, "feats")
        if refuse:
            result["error"] = refuse
            return result
        self.open_panel("feats")
        opened = None
        try:
            opened = self._wait_panel_open()
            if opened is None:
                result["error"] = f"panel did not open after tapping feats"
                return result
            result["panel"] = self.read_panel(opened, "feats")
            # Collect each ready reward, bounded and verified. Only buttons
            # whose feat the panel READ reports done are claimable (every row
            # renders a Reward-shaped preview button).
            feats = (result["panel"] or {}).get("feats") or []
            for _ in range(COLLECT_TAPS_MAX):
                all_centers = self._collect_centers(opened)
                centers = self._claimable_centers(opened, all_centers, feats)
                if not centers:
                    if all_centers:
                        # Buttons exist but none map to a done feat — log the
                        # miss for calibration instead of tapping previews.
                        result["claim_miss"] = {
                            "buttons": [list(c) for c in all_centers],
                            "feats": [{"name": f.get("name"),
                                       "done": bool(f.get("done"))}
                                      for f in feats]}
                    break
                cx, cy = centers[0]
                self.device.tap(int(cx), int(cy))
                tapped_ok = False
                for _ in range(COLLECT_VERIFY_POLLS):
                    self.device.wait_for_idle(COLLECT_VERIFY_PAUSE)
                    self.device.screencap()
                    nxt = cv2.imread(str(self.device.screencap_path))
                    if nxt is None:
                        continue
                    if (cx, cy) not in self._collect_centers(nxt):
                        tapped_ok = True
                        opened = nxt
                        break
                if not tapped_ok:
                    # Button didn't disappear (tap missed?) — stop, don't loop.
                    break
                result["collected"] += 1
            # Tier-completion reward: when all missions of a Tier are done, the
            if self._tier_badge_present(opened):
                opened, tier_done, tier_stuck = self.collect_tier_reward(opened)
                if tier_done:
                    result["tier"] = True
                if tier_stuck:
                    # Popup didn't clear — do NOT BACK while it's up. Leaving
                    # it safe; the next feats visit retries (badge is gone).
                    result["error"] = "tier reward popup did not clear — left in place"
            result["claim_miss"] = None
            return result
        finally:
            # Only close when a panel was confirmed open AND no popup is up
            # (BACK on a popup or the bare board exits the game); the dock-return
            # check also runs.
            if opened is not None and not self.bottombar.bar_visible(opened) \
                    and not self._reward_popup_present(opened):
                self.close_panel()

    def tap_button(self, frame, name: str) -> dict:
        """Open + read + close a dock panel. Returns a JSON-ready dict:
        {"button": name, "panel": <parsed JSON>, "error": null|"..."}.

        Never called for locked/placement buttons (refused first, in
        `_check`). Raises BoardLostError if the game fails to return the board
        after closing, so the caller can halt instead of tapping a dead screen.
        """
        result = {"button": name, "panel": None, "error": None}
        if self.device is None or self.llm is None:
            result["error"] = "panels require live mode with an LLM"
            return result
        refuse = self._check(frame, name)
        if refuse:
            result["error"] = refuse
            return result
        # Bare-lair reference: dock visible + board up.
        lair = frame
        self.open_panel(name)
        try:
            frame = self._wait_panel_open()
            if frame is None:
                # Tap didn't open a panel — leaves the board bare. Crucially
                # NO BACK here (BACK on the bare board exits the game).
                result["error"] = f"panel did not open after tapping {name}"
                return result
            result["panel"] = self.read_panel(frame, name)
            return result
        finally:
            # Only close when the panel was confirmed open (BACK on the bare
            # board would exit the game).
            if frame is not None and not self.bottombar.bar_visible(frame):
                self.close_panel()

    def _check(self, frame, name: str) -> str | None:
        """Refuse taps that would not open a safe read-only panel."""
        if name in LOCKED_BUTTONS:
            return f"{name} is locked until a higher Devourer level — never tap it"
        if name in PLACEMENT_BUTTONS:
            return PLACEMENT_BUTTONS[name]
        if name not in PANEL_BUTTONS:
            return f"unknown dock button {name!r}"
        # If a panel for this button is ALREADY open, the dock is hidden
        # (the panel covers the dock band) so `bar_visible()` flips False.
        # In that case the dock-button check is moot — we're already inside
        # the right panel. This is the path the Aug 28 buy(confirm=true)
        # buy path follows when the confirm dialog has replaced the Station
        # panel content.
        if not self.bottombar.bar_visible(frame):
            return None
        for b in self.bottombar.read_bar(frame):
            if b["name"] == name and not b["unlocked"]:
                return f"{name} is locked until a higher Devourer level"
        return None

    def open_panel(self, name: str) -> None:
        """Tap the button's icon center (side-effect free: only opens the panel)."""
        cx, cy = ICON_CROPS[name][0], ICON_CROPS[name][1]
        self.device.tap(int(cx), int(cy))

    def _panel_open(self, frame) -> bool:
        """True if the full-screen panel is showing: the dock band is covered,
        so `bar_visible()` flips False. Idle animation can't change the dock.
        """
        if frame is None:
            return False
        return not self.bottombar.bar_visible(frame)

    def _wait_panel_open(self):
        """Screencap until the panel is confirmed open AND its body has
        rendered, or the poll budget runs out. Returns the open frame, or
        None if no panel.

        the dock gets covered as soon as the panel starts sliding
        in (mid-animation), so the original "dock covered" check returned
        the first mid-animation frame — the cards/currency bar weren't
        drawn yet, and downstream readers (currency, dialog tap) silently
        failed. The poll now additionally waits for the card row to
        render before returning; mid-animation frames fall through and
        the next poll re-checks.
        """
        for _ in range(OPEN_POLLS):
            self.device.wait_for_idle(OPEN_PAUSE)
            self.device.screencap()
            frame = cv2.imread(str(self.device.screencap_path))
            if frame is not None and self._panel_open(frame) \
                    and self._panel_body_rendered(frame):
                return frame
        return None

    def _panel_body_rendered(self, frame) -> bool:
        """Lightweight check that a panel's body has actually rendered (not a
        mid-animation frame). Looks for the card row at its expected y-band
        via a low-cost structural signal: non-empty std in the card row
        (catches the empty/transparent mid-animation state) — exact card
        identity is left to the StationShop verification step.

        Returns True for non-station panels too (the dock-covered check is
        the primary gate; this is just an anti-mid-animation guard so
        callers that read panel state don't trip on a half-drawn frame).
        """
        if frame is None or frame.shape[0] < 2200 or frame.shape[1] < 1280:
            return True        # can't check; assume rendered
        # Card row at y=2180-2621 across all card x-slots; mid-animation
        # shows the panel sliding in from the bottom, so the top of the
        # card row is rendered first. We sample y=2230-2350 (the top half
        # of a card) across all 3 card x-slots; rendered = high std in
        # at least one slot.
        from vision.panels import CARD_XS
        any_rendered = False
        for x0, x1 in CARD_XS:
            crop = frame[2230:2350, x0:x1]
            if crop.size == 0:
                continue
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # v2 SHOP view: card top std ~ 60-90; mid-animation: ~ 5-15
            if g.std() >= 25:
                any_rendered = True
                break
        return any_rendered

    def read_panel(self, frame, name: str):
        """LLM-read the open panel body -> parsed dict (or None on failure)."""
        if self.llm is None:
            return None
        # Assistant prefill: skips the reasoning model's thinking block (same
        # fix as the cravings menu + item popup reads). The starter differs per
        # panel so the model continues valid JSON.
        starter = '{"feats":' if name == "feats" else '{"stations":'
        messages = [
            {"role": "system", "content": "You read NecroMerger dock panels."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _encode_frame(self.panel_crop(frame))}},
                {"type": "text", "text": PANEL_QUESTIONS[name]},
            ]},
            {"role": "assistant", "content": starter},
        ]
        try:
            content, _, _full = self.llm.chat_message(messages, max_tokens=256, json_mode=False)
            data = _extract_json(content)
        except (LLMError, ValueError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def panel_crop(self, frame):
        x, y, w, h = PANEL_BODY
        return frame[y:y + h, x:x + w]

    def close_panel(self) -> None:
        """BACK-dismiss the panel (safe when `_panel_open` confirmed it open)
        and verify the dock returned; raise BoardLostError if the game exited.

        The full-screen panels cover the dock band, so `bar_visible` going
        back to True proves the panel closed AND the lair returned — no
        separate floor-blob check needed (and idle animation can't fake a
        dock back into existence). Polls several times (the panel animates
        out; a single check catches a mid-animation frame and false-alarms),
        retries BACK only when the panel is CONFIRMED still open (a blind
        re-BACK could hit the bare board and really exit), and raises ONLY
        when the game is no longer the foreground activity — a live game with
        a slow/stuck panel logs a warning and lets the next cycle retry.
        """
        if self.device is None:
            return
        for _attempt in range(2):
            self.device.back()
            for _ in range(CLOSE_POLLS):
                self.device.wait_for_idle(CLOSE_PAUSE)
                self.device.screencap()
                frame = cv2.imread(str(self.device.screencap_path))
                if frame is not None and self.bottombar.bar_visible(frame):
                    return
            if _attempt == 0:
                # Only re-BACK if the panel is confirmed still open (dock still
                # covered). A mid-animation board that the poll missed must NOT
                # be re-BACKed — that hits the bare board and exits the game.
                self.device.screencap()
                frame = cv2.imread(str(self.device.screencap_path))
                if frame is None or self.bottombar.bar_visible(frame):
                    break
        if game_in_foreground(self.device):
            print("[panels] dock did not return after BACK; the game is still "
                  "running (panel animating or stuck open) — continuing, "
                  "next cycle retries")
            return
        raise BoardLostError(
            "dock did not return after BACK — the game likely exited")

# ---- Station shop: currency HUD + station cards + safe buy flow (Aug 24) ----
# The Station panel is a bottom sheet (the board stays visible above it) with
# one card per purchasable station: icon + rune cost(s). The top HUD shows the
# current rune counts. Buying taps a card -> a confirm dialog appears -> the
# model verifies the dialog read and confirms in a SECOND call, so no purchase
# ever happens without an explicit model-in-the-loop confirmation.

# widened to (180, 260) after a calibration run on the user's
# live panel. The digits sit at y=195-245 but Vision's recognizer on
# Apple Silicon often drops one of the "57" digits when the crop
# starts at y=190 (it cuts off the top of the '5' and '7' shapes).
# y=(180, 260) is the safe range: it covers the full digit height plus
# the slot BG above and below, and OCR consistently reads "57 16 0 0 0"
# on the user's live panel. Cropping tighter (240 lower bound, 190 upper)
# returned "1F" or empty on the same digits.
CURRENCY_Y = (180, 260)
# In-game currency names (Aug 26 confirmation): ice, poison, blood, moon, death.
# The slot x-coordinates are unchanged from the original calibration; only the
# rune NAME keyed to each slot is corrected (the old "green/red/purple" labels
# silently read the wrong runes — Aug 26 user report: bot saw 19 "green" when
# the actual rune was poison).
CURRENCY_SLOTS = [  # (x0, x1, rune name) left-to-right; count digits right-aligned
    (167, 318, "ice"),
    (353, 532, "poison"),
    (557, 746, "blood"),
    (764, 959, "moon"),
    (991, 1173, "death"),
]
# removed back-compat color aliases (green/red/purple) — only the
# 5 real runes (ice, poison, blood, moon, death) appear in currency/cost
# dicts now. The old names leaked into session.jsonl as clutter ("green":
# 14 alongside "poison": 14) — user report. The 5-currency buy check
# already used the real names; the legacy keys were observation-only.
CARD_Y = (2180, 2621)          # station-card row (bottom sheet)
CARD_XS = [(19, 439), (480, 954), (892, 1280)]   # up to 3 visible cards
CARD_TAP_Y = 2400              # vertical center of a card
# Card-row paging: the sheet is scrollable left-right and only 3 cards are
# visible — later stations (Supply Cupboard, Fridge, ...) live off-screen.
# Swipe left across the card row to reveal later cards; swipe right to
# restore. Duration 500ms (merge-drag tuning: faster swipes no-op).
CARD_SWIPE_LEFT = (1000, 300)  # (x_from, x_to) drag to page left
CARD_SWIPE_Y = CARD_TAP_Y
CARD_SWIPE_MS = 500
CARD_SWIPE_SETTLE = 0.8
MAX_CARD_PAGES = 8             # pages scanned (covers 17 stations at ~3/page)
CARD_RESCAN_SECONDS = 300      # re-scan for a previously-missing card at most this often
# The REAL "Build?" dialog (Aug 24 live capture, assets/calib/feats/
# buy_dialog_full.png): centered box y~987-1638 with the station name + cost,
# and Cancel/Confirm buttons at y~1748-1899. The old Aug 7 model-sourced
# coordinates (584, 636) were WRONG — measured from the capture instead.
DIALOG_REGION = (320, 980, 640, 670)   # x, y, w, h — the dialog box
CONFIRM_XY = (736, 1823)       # green check button center
CANCEL_XY = (567, 1823)        # red X button center
CONFIRM_TEMPLATE = Path(__file__).resolve().parent.parent / \
    "assets/calib/feats/confirm_button.png"
CONFIRM_BAND = (1650, 2000)    # y-band where the Confirm button appears
CONFIRM_MIN = 0.7              # template-match threshold
DIALOG_POLLS = 6
DIALOG_PAUSE = 0.8

# Dialog cost-row rune icons (Aug 26) — used by the icon guard to override
# the LLM's misclassification of which cost_* field a value belongs to.
# Real dialog-sized templates for ice/poison (extracted from the live
# manapool dialog); HUD-derived scaled versions for blood/moon/death.
DIALOG_RUNE_BANK_DIR = Path(__file__).resolve().parent.parent / \
    "assets" / "calib" / "dialog_runes"
# Primary templates (real dialog-sized for ice/poison; scaled from HUD for
# blood/moon/death — close enough; only one rune is on the board at a time
# for those, so a single best-match score separates them cleanly).
DIALOG_RUNE_TEMPLATES = {
    "ice":    "ice_dialog.png",
    "poison": "poison_dialog.png",
    "blood":  "blood_dialog_scaled.png",
    "moon":   "moon_dialog_scaled.png",
    "death":  "death_dialog_scaled.png",
}
# Cost-row search region: the cost box is below the station icon/name
# and above the Cancel/Confirm buttons. Different stations render the
# dialog at slightly different y positions; the search range below
# covers both the grave (large box, y~1490-1670) and manapool (smaller,
# y~1530-1670) variants.
COST_ROW_SEARCH_Y = (1490, 1700)
COST_ROW_SEARCH_X = (380, 900)
# Threshold above which a rune match overrides the LLM's classification.
# 0.7 separates confident matches (real icons) from random patterns.
RUNE_MATCH_MIN = 0.7
# Minimum horizontal gap between two icon centers (avoids double-matching
# the same icon). 50px is less than the width of an icon (60-70px) so two
# icons at different x positions are kept; one icon matched twice at
# nearly the same x is collapsed.
RUNE_MIN_GAP = 50

BUY_QUESTION = ('This is a station-purchase card from the NecroMerger Station panel. '
                'Identify which station it sells by its icon: Grave = a gray cross '
                'tombstone on dirt; Manapool = a round blue pool/orb with a rim; '
                'Lectern = a wooden desk. Reply ONLY compact JSON: '
                '{"station": "grave|manapool|lectern|locked|empty|other", '
                '"cost_ice": N, "cost_poison": N, "cost_blood": N, "cost_moon": N, '
                '"cost_death": N, "affordable_guess": true|false} where N is 0 when '
                'that rune is not shown. A grayed card with lock/requirement text = '
                '"locked". The 5 currencies are ice, poison, blood, moon, death.')

# LLM card names -> canonical families (the vision model describes icons
# loosely: "Blue orb" for the Manapool, etc.). Covers ALL buyable stations:
# off-screen cards (Supply Cupboard, Fridge, ...) are only reachable via
# card-row swiping, and without an alias the family lookup misses ("supply
# cupboard" != "supplycupboard") so the card is never found.
FAMILY_ALIASES = {
    "grave": "grave", "cross": "grave", "tombstone": "grave",
    "manapool": "manapool", "pool": "manapool", "mana": "manapool",
    "orb": "manapool", "pond": "manapool",
    "manapot": "manapot", "flask": "manapot", "potion": "manapot",
    "lectern": "lectern", "desk": "lectern",
    "supplycupboard": "supplycupboard", "supply cupboard": "supplycupboard",
    "cupboard": "supplycupboard",
    "fridge": "fridge", "refrigerator": "fridge",
    "foulchicken": "foulchicken", "foul chicken": "foulchicken",
    "chicken": "foulchicken",
    "slimevat": "slimevat", "slime vat": "slimevat", "vat": "slimevat",
    "slime": "slimevat",
    "altar": "altar",
    "darkstores": "darkstores", "dark stores": "darkstores",
    "stores": "darkstores",
    "portal": "portal",
    "crashedsaucer": "crashedsaucer", "crashed saucer": "crashedsaucer",
    "saucer": "crashedsaucer",
    "telepad": "telepad",
    "soulgrinder": "soulgrinder", "soul grinder": "soulgrinder",
    "grinder": "soulgrinder",
    "prism": "prism",
    "meteor": "meteor",
    "throne": "throne",
    "unexpectedparcel": "unexpectedparcel", "unexpected parcel": "unexpectedparcel",
    "parcel": "unexpectedparcel",
}


def normalize_station_name(name: str) -> str:
    """Map an LLM icon description to a canonical station family."""
    n = (name or "").strip().lower()
    if n in FAMILY_ALIASES:
        return FAMILY_ALIASES[n]
    for key, fam in FAMILY_ALIASES.items():
        if key in n:
            return FAMILY_ALIASES[key]
    return n or "unknown"


def cvt_gray(crop):
    """BGR -> grayscale helper (shared by StationShop and other readers)."""
    if crop is None or crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

DIALOG_QUESTION = ('This is a purchase-confirmation dialog from the NecroMerger Station panel. '
                   'Reply ONLY compact JSON: {"is_purchase_dialog": true|false, '
                   '"station": "grave|manapool|lectern|other", "cost_ice": N, '
                   '"cost_poison": N, "cost_blood": N, "cost_moon": N, "cost_death": N} '
                   'with N = 0 when that rune is not shown. Each cost is shown as '
                   'a colored rune icon (ice = blue diamond, poison = green droplet, '
                   'blood = red droplet, moon = orange crescent, death = dark purple) '
                   'next to a "-N" number — associate the value with the rune icon, '
                   'NOT the text color. A code-side icon-template guard will validate '
                   'your assignment; if you misread an icon, the guard will correct it.')


class StationShop:
    """Buy stations from the Station panel with model-in-the-loop confirmation.

    `buy(family, confirm=False)` runs the full read path (open panel ->
    locate the family's card -> affordability check -> tap -> read the confirm
    dialog) but only taps Confirm when `confirm=True` AND the dialog read
    matches the requested family. Without confirmation it BACKs out of the
    dialog and returns what it saw, so the model can decide on the second
    call. Every gate failure returns a structured result instead of raising;
    only BoardLostError propagates (the game is gone)."""

    def __init__(self, device, llm, bottombar, classifier=None):
        self.device = device
        self.llm = llm
        self.bottombar = bottombar
        self.classifier = classifier
        self.cost_cache = {}           # family -> {ice, poison, blood, moon, death} from card reads
        self._panel = PanelReader(device=device, llm=llm, bottombar=bottombar)
        # track the staged two-phase buy. buy(confirm=false) sets
        # this to {family, dialog, step} so the planner's compelled
        # follow-through gate can force the next buy(..., confirm=true).
        # buy(..., confirm=true) with a matching family clears it.
        # Initialized to None so reads BEFORE the first buy don't AttributeError
        # when the planner proxies via self.shop._pending_buy.
        self._pending_buy = None
        # family -> card-row page index where the card was last seen (skips
        # re-scanning on repeat buys; the sheet is scrollable and only 3
        # cards are visible at once).
        self._card_pages = {}
        # unlock-requirement texts seen on grayed locked cards during the
        # latest scan (e.g. "Tier 5 Feats Requires"). Rebuilt per _find_card
        # scan; lets the planner distinguish "not on sheet" from "locked"
        # and steer toward the unlock instead of rescanning forever.
        self.locked_requirements = []
        # family -> epoch seconds of the last FULL scan that missed it. A
        # locked/unreleased station (e.g. Supply Cupboard before its
        # Devourer level) would otherwise make every buy compel re-scan the
        # whole sheet each step. The planner throttles re-scans against
        # CARD_RESCAN_SECONDS (unlocks only happen on level-ups, so a
        # re-check every few minutes is plenty).
        self._card_missing = {}

    # ---- currency HUD ------------------------------------------------------

    @staticmethod
    def _parse_count(txt: str) -> int:
        """First digit group of an OCR reading — the balance. The HUD renders
        a transient "+N" gain indicator next to the counter; concatenating all
        digits turned '16 +2' into 162 and made an unaffordable 20-ice grave
        look affordable (the inert Confirm mystery, Aug 24)."""
        m = re.search(r"\d+", txt or "")
        return int(m.group(0)) if m else 0

    def read_currency(self, frame) -> dict:
        """Rune counts from the top HUD via Apple Vision OCR.

        Aug 28 rewrite: per-slot crops (each ~50x100 px after a 2x scale-up)
        are too small for Apple Vision to reliably read digits — the per-slot
        OCR kept returning "0" for "57" because Vision failed the recognizer
        on the narrow strip. Reading the WHOLE bar in one shot (1300x100) and
        splitting the recognized text by horizontal position is reliable:
        OCR reads "57 16 0 0 0" or similar with both digits and zeros
        preserved. Aug 28 calibration on the user's live panel: y=(190, 240)
        — the digits are at y~195-240; the empty y=(240, 314) is the slot BG
        that previously misled Vision into reading "0" instead of "57".
        """
        out = {}
        if frame is None or frame.shape[0] < 240 or frame.shape[1] < 1280:
            return out
        crop = frame[CURRENCY_Y[0]:CURRENCY_Y[1] + 1, 0:1280]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        big = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
            cv2.imwrite(t.name, big)
            txt = apple_vision_text(t.name)
            Path(t.name).unlink(missing_ok=True)
        # Parse the whole-bar OCR. Expected pattern:
        # "57 16 0 0 0" or "57 16 0 0" (OCR may drop trailing zeros).
        nums = re.findall(r"\d+", txt or "")
        # If Vision found 5 numbers, map them 1:1. If fewer, default missing
        # to 0 (the last few are usually zeros that Vision skips).
        rune_order = ("ice", "poison", "blood", "moon", "death")
        for i, name in enumerate(rune_order):
            if i < len(nums):
                out[name] = int(nums[i])
            else:
                out[name] = 0
        return out

    # ---- cards -------------------------------------------------------------

    def _card_crop(self, frame, idx: int):
        x0, x1 = CARD_XS[idx]
        return frame[CARD_Y[0]:CARD_Y[1], x0:x1]

    @staticmethod
    def _card_signature(cards: list[dict]) -> tuple:
        """Order-sensitive page fingerprint for end-of-list detection."""
        return tuple(c.get("station") for c in cards)

    def _swipe_cards(self, left: bool = True):
        """Page the card row one step (left reveals later stations)."""
        x0, x1 = CARD_SWIPE_LEFT if left else (CARD_SWIPE_LEFT[1], CARD_SWIPE_LEFT[0])
        self.device.swipe(x0, CARD_SWIPE_Y, x1, CARD_SWIPE_Y,
                          duration_ms=CARD_SWIPE_MS)
        self.device.wait_for_idle(CARD_SWIPE_SETTLE)
        self.device.screencap()
        return cv2.imread(str(self.device.screencap_path))

    def _find_card(self, frame, family: str) -> tuple[dict | None, object, int, list]:
        """Locate `family`'s card, paging left as needed.

        Returns (card_or_None, latest_frame, pages_advanced, last_page_cards).
        Starts from the cached page when known (skips re-reads), otherwise
        scans from the current position. Stops at the first page whose
        signature repeats (end of list) or MAX_CARD_PAGES. Every scanned
        page runs read_cards, so cost_cache accumulates across pages as a
        side effect. Callers must restore the scroll (swipe right
        pages_advanced times) before closing so the next buy starts from a
        known position.
        """
        pages = 0
        last_cards: list = []
        self.locked_requirements = []
        jump = self._card_pages.get(family, 0)
        for _ in range(jump):
            frame = self._swipe_cards(left=True)
            if frame is None:
                return None, frame, pages, last_cards
            pages += 1
        prev_sig = None
        for _ in range(MAX_CARD_PAGES):
            cards = self.read_cards(frame)
            last_cards = cards
            for c in cards:
                if c.get("station") == "locked" and c.get("requires"):
                    if c["requires"] not in self.locked_requirements:
                        self.locked_requirements.append(c["requires"])
            sig = self._card_signature(cards)
            card = next((c for c in cards if c["station"] == family), None)
            if card is not None:
                self._card_pages[family] = pages
                self._card_missing.pop(family, None)
                return card, frame, pages, last_cards
            if sig == prev_sig:
                break  # end of list (swipe no-ops at the edge)
            prev_sig = sig
            frame = self._swipe_cards(left=True)
            if frame is None:
                return None, frame, pages, last_cards
            pages += 1
        self._card_missing[family] = time.time()
        return None, frame, pages, last_cards

    def _restore_scroll(self, pages: int) -> None:
        """Swipe back to the pre-search position before closing the panel."""
        for _ in range(max(0, pages)):
            self._swipe_cards(left=False)

    @staticmethod
    def _card_row_hash(frame) -> int | None:
        """Cheap card-row fingerprint for scroll-end detection (no LLM)."""
        try:
            row = frame[CARD_Y[0]:CARD_Y[1], :]
            small = cv2.resize(row, (160, 27), interpolation=cv2.INTER_AREA)
            return hash(small.tobytes())
        except Exception:
            return None

    def _rewind_to_start(self) -> None:
        """Swipe right until the sheet stops moving (true first page).

        One-shot recovery for a sheet left mid-list by a flow that bypassed
        buy() (all buy() exits restore scroll, so the steady state starts
        leftmost). NOT called per-buy: it costs swipes + screencaps that
        would shift scripted test frames and slow every purchase.
        """
        prev = None
        for _ in range(3):
            self.device.screencap()
            frame = cv2.imread(str(self.device.screencap_path))
            if frame is None:
                return
            sig = self._card_row_hash(frame)
            if sig is not None and sig == prev:
                return
            prev = sig
            self._swipe_cards(left=False)

    def _llm_card(self, frame, idx: int) -> dict | None:
        crop = self._card_crop(frame, idx)
        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            return None
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        messages = [
            {"role": "system", "content": "You read NecroMerger shop cards."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": BUY_QUESTION}]},
            {"role": "assistant", "content": '{"station":'},
        ]
        reply, _msg = self.llm.chat(messages, max_tokens=128, json_mode=False)
        data = _extract_json('{"station":' + reply)
        return data if isinstance(data, dict) else None

    def read_cards(self, frame) -> list[dict]:
        """Identify each visible station card: index, x-center, station family,
        costs. Family resolution: multi-scale icon template-match against
        banked station sprites first (deterministic — the LLM described the
        Manapool card as 'other'/'Blue orb', which broke exact-name matching),
        then the LLM name with family-alias normalization. COSTS always come
        from the LLM read (the affordability gate depends on them; the
        template path carries no cost info). Locked/gray cards report
        station "locked". Aug 26: only the 5 real currency keys are used
        (ice, poison, blood, moon, death) — no cost_green alias."""
        cards = []
        for idx in range(len(CARD_XS)):
            x0, x1 = CARD_XS[idx]
            crop = self._card_crop(frame, idx)
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            info = {"index": idx, "x": (x0 + x1) // 2, "station": "unknown",
                    "cost_ice": 0, "cost_poison": 0, "cost_blood": 0,
                    "cost_moon": 0, "cost_death": 0}
            if g.std() < 12:            # blank slot (fewer stations unlocked)
                info["station"] = "empty"
                cards.append(info)
                continue
            fam, score = self._match_card_icon(crop)
            data = {}
            if self.llm is not None:
                data = self._llm_card(frame, idx) or {}
            if fam is not None:
                info["station"] = fam
                info["icon_match"] = round(score, 3)
            else:
                info["station"] = normalize_station_name(
                    str(data.get("station") or "unknown"))
            for k in ("cost_ice", "cost_poison", "cost_blood",
                       "cost_moon", "cost_death"):
                info[k] = int(data.get(k) or 0)
            if info["station"] in ("locked", "unknown"):
                # Grayed locked card: read its unlock requirement ("Tier 5
                # Feats Requires...") so the planner knows WHY a family is
                # missing and what unlocks it, instead of rescanning
                # forever. Also repairs "unknown": a card whose icon the
                # bank/LLM both miss but whose text names a requirement is
                # a locked card, not an unidentified buyable one.
                info["requires"] = self._ocr_card_requirement(crop)
                if (info["station"] == "unknown" and info["requires"]
                        and re.search(r"requir|locked|tier \d|level \d",
                                      info["requires"], re.I)):
                    info["station"] = "locked"
            if info["station"] not in ("unknown", "empty", "locked"):
                self.cost_cache[info["station"]] = {
                    "ice": info["cost_ice"],
                    "poison": info["cost_poison"],
                    "blood": info["cost_blood"],
                    "moon": info["cost_moon"],
                    "death": info["cost_death"],
                }
            cards.append(info)
        return cards

    @staticmethod
    def _ocr_card_requirement(crop) -> str:
        """Unlock-requirement text off a grayed locked card (best effort)."""
        try:
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            big = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
                cv2.imwrite(t.name, big)
                txt = apple_vision_text(t.name)
                Path(t.name).unlink(missing_ok=True)
            return " ".join((txt or "").split())[:120]
        except Exception:
            return ""

    # card icon regions (x-offset within the card, y-band) + the station
    # sprite templates they are matched against (multi-scale)
    CARD_ICON_Y = (2230, 2520)
    ICON_MATCH_MIN = 0.65
    ICON_SCALES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    def _match_card_icon(self, card_crop):
        """Best station-family match for a card's icon region via multi-scale
        template matching against the banked station sprites. Returns
        (family_or_None, best_score)."""
        if card_crop is None or card_crop.size == 0:
            return None, -1.0
        g = cv2.cvtColor(card_crop, cv2.COLOR_BGR2GRAY)
        tpls = self._station_templates()
        best_fam, best_score = None, -1.0
        for fam, tpl in tpls.items():
            _, score = self._best_scale(g, tpl)
            if score > best_score:
                best_fam, best_score = fam, score
        if best_score >= self.ICON_MATCH_MIN:
            return best_fam, best_score
        return None, best_score

    _station_tpl_cache = None

    def _station_templates(self) -> dict:
        """{family: gray sprite} for card-icon matching, loaded once."""
        if self._station_tpl_cache is None:
            out = {}
            base = Path(__file__).resolve().parent.parent / "assets" / "templates"
            for fam, fname in (("grave", "grave_lvl1__0.png"),
                               ("manapool", "manapool_lvl1__0.png"),
                               ("manapot", "manapot_lvl1.png"),
                               ("supplycupboard", "supplycupboard_lvl1__0.png")):
                p = base / fname
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    out[fam] = img
            self._station_tpl_cache = out
        return self._station_tpl_cache

    @staticmethod
    def _best_scale(crop, tpl, scales=None):
        best_s, best_m = None, -1.0
        for s in (scales or StationShop.ICON_SCALES):
            t = cv2.resize(tpl, None, fx=s, fy=s) if s != 1.0 else tpl
            if crop.shape[0] < t.shape[0] or crop.shape[1] < t.shape[1]:
                continue
            res = cv2.matchTemplate(crop, t, cv2.TM_CCOEFF_NORMED)
            m = float(res.max())
            if m > best_m:
                best_s, best_m = s, m
        return best_s, best_m

    # ---- buy flow ----------------------------------------------------------

    def _family_count(self, frame, family: str) -> int:
        """How many stations of `family` the board currently holds (the board
        is visible inside the panel frame). Used as the purchase truth-check:
        a successful build places a new station. Best-effort — classifier or
        geometry problems count as 0 (the currency check remains the primary
        signal)."""
        if self.classifier is None:
            return 0
        try:
            from vision.pipeline import classify_board
            board = classify_board(frame, self.classifier)
            return sum(1 for c in board.cells
                       if c.occupied and c.item_id
                       and c.item_id.startswith(family))
        except Exception:
            return 0

    def _llm_dialog(self, frame) -> dict | None:
        x, y, w, h = DIALOG_REGION
        crop = frame[y:y + h, x:x + w]
        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            return None
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        messages = [
            {"role": "system", "content": "You read NecroMerger dialogs."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": DIALOG_QUESTION}]},
            {"role": "assistant", "content": '{"is_purchase_dialog":'},
        ]
        reply, _msg = self.llm.chat(messages, max_tokens=128, json_mode=False)
        data = _extract_json('{"is_purchase_dialog":' + reply)
        if isinstance(data, dict) and data.get("station"):
            data["station"] = normalize_station_name(str(data["station"]))
        return data

    # ---- dialog cost-icon guard (Aug 26) --------------------------------
    # The 4B vision LLM occasionally misclassifies a single-icon dialog's
    # rune (e.g. reads blue-diamond ice as dark-purple death), putting the
    # cost value in the wrong cost_* field. The guard: detect each rune
    # icon in the cost row via template matching against banked dialog-
    # sized icons, then re-route the LLM's value to the matched rune's
    # field. Multi-icon (manapool: poison + ice) is handled — each icon is
    # matched in the same scan.

    _rune_bank_cache: dict | None = None

    @classmethod
    def _load_rune_bank(cls) -> dict[str, np.ndarray]:
        """Load the 5 dialog-sized rune icon templates (cached)."""
        if cls._rune_bank_cache is None:
            out = {}
            for rune, fname in DIALOG_RUNE_TEMPLATES.items():
                p = DIALOG_RUNE_BANK_DIR / fname
                img = cv2.imread(str(p))
                if img is not None:
                    out[rune] = img
            cls._rune_bank_cache = out
        return cls._rune_bank_cache

    @staticmethod
    def _rune_icon_score(crop, tpl) -> float:
        """TM_CCOEFF_NORMED max — best score where the template fits in crop."""
        if (crop is None or tpl is None
                or crop.size == 0 or tpl.size == 0):
            return -1.0
        if crop.shape[0] < tpl.shape[0] or crop.shape[1] < tpl.shape[1]:
            return -1.0
        res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
        return float(res.max())

    def _rune_icon_locate(self, frame) -> list[dict]:
        """Detect every rune icon in the dialog cost row. Returns a list
        of {rune, x, y, score, sufficient} dicts, sorted by x left-to-right.
        `sufficient` is True if the adjacent "-N" text is white (resource
        sufficient) and False if it's orange/red (insufficient); None if
        text is unreadable. Empty list if no icon matches above
        RUNE_MATCH_MIN (the LLM's read is used unchanged in that case)."""
        if frame is None:
            return []
        # Adapt the search region to the frame size. Two cases:
        # - Full emulator frame (1280x2856): cost row at y=1490-1700,
        #   centered horizontally at x=380-900. Use the canonical range.
        # - Cropped dialog (variable size, w<=700): the dialog is mostly
        #   dialog content; the cost row is in the lower-middle. Use a
        #   percentage-based range that captures both grave (large
        #   vertical dialog) and manapool (small vertical dialog).
        h, w = frame.shape[:2]
        if w > 700:               # full emulator frame
            y0, y1 = 1490, 1700
            x0, x1 = 380, 900
        else:                     # cropped dialog
            # The crop is a DIALOG_REGION slice that may include the
            # station icon, name, cost box, and (sometimes) the buttons.
            # The cost row sits in the lower-middle of the dialog. Use
            # a wide y-band (35% - 95%) to cover all dialog variants.
            y0, y1 = int(h * 0.35), int(h * 0.95)
            x0, x1 = int(w * 0.10), int(w * 0.90)
        if y1 <= y0 or x1 <= x0:
            return []
        search = frame[y0:y1, x0:x1]
        bank = self._load_rune_bank()
        # Collect all candidate matches across all 5 runes
        candidates: list[dict] = []
        for rune, tpl in bank.items():
            score = self._rune_icon_score(search, tpl)
            if score < RUNE_MATCH_MIN:
                continue
            # Re-locate to get x,y
            res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
            _, _, _, maxloc = cv2.minMaxLoc(res)
            th, tw = tpl.shape[:2]
            cx = maxloc[0] + tw // 2
            cy = maxloc[1] + th // 2
            candidates.append({
                "rune": rune, "x": cx + x0, "y": cy + y0,
                "w": tw, "h": th, "score": score,
            })
        # Collapse duplicate matches at the same x (same icon matched by
        # different templates — keep the highest score per cluster).
        candidates.sort(key=lambda c: c["score"], reverse=True)
        kept: list[dict] = []
        for c in candidates:
            too_close = any(abs(c["x"] - k["x"]) < RUNE_MIN_GAP for k in kept)
            if not too_close:
                kept.append(c)
        kept.sort(key=lambda c: c["x"])
        # Determine text-color for each icon (sufficient vs insufficient)
        for c in kept:
            c["sufficient"] = self._cost_text_color(frame, c)
        return kept

    @staticmethod
    def _cost_text_color(frame, icon_info: dict) -> bool | None:
        """Sample the cost text color immediately to the right of the icon.
        Returns True if the text appears white (sufficient — user has
        enough of the resource), False if reddish/orange (insufficient),
        None if unreadable. The game uses a font where the "-" sign and
        the digit run together; we sample a band starting ~10px right of
        the icon and stretching ~50px."""
        ix, iy, iw, ih = icon_info["x"], icon_info["y"], icon_info["w"], icon_info["h"]
        x_start = ix + iw // 2 + 6
        x_end = min(x_start + 55, frame.shape[1])
        y_top = max(iy - ih // 2 + 4, 0)
        y_bot = min(iy + ih // 2 - 4, frame.shape[0])
        if x_end <= x_start or y_bot <= y_top:
            return None
        region = frame[y_top:y_bot, x_start:x_end]
        # Mask out the cost box background (~purple [54, 0, 41]) — keep
        # only bright text pixels.
        b, g, r = cv2.split(region)
        bright = (b.astype(int) + g.astype(int) + r.astype(int)) > 350
        if bright.sum() < 5:
            return None
        # Mean color of the bright pixels
        mask3 = np.stack([bright] * 3, axis=-1)
        masked = np.where(mask3, region, 0)
        n = bright.sum()
        avg_b = masked[:, :, 0].sum() / n
        avg_g = masked[:, :, 1].sum() / n
        avg_r = masked[:, :, 2].sum() / n
        # Sufficient: white-ish (R, G, B all close, all > 150)
        # Insufficient: orange/red (R > G + 30, R > B + 30, R > 150)
        if avg_r > 150 and avg_r > avg_g + 30 and avg_r > avg_b + 30:
            return False
        if avg_r > 150 and avg_g > 150 and avg_b > 150:
            return True
        return None

    def _apply_icon_guard(self, llm_read: dict, icons: list[dict],
                          frame) -> tuple[dict, bool]:
        """Re-assign the LLM's cost_* values to the runes matched in
        `icons` (in left-to-right order). Returns (corrected_dict, was_overridden).

        The override fires ONLY when the LLM's nonzero cost values are in
        cost_* keys that don't all match a detected icon. This catches the
        single-icon misclassification (e.g. grave's ice icon read as death)
        without breaking correct multi-icon reads (manapool: the LLM pairs
        each value with the correct rune, regardless of left-to-right
        position).

        When the LLM's set of runes matches the guard's set, the LLM is
        trusted — the model reads the "(rune_icon, -N)" pair atomically
        and gets both right together (verified live for 8/9 grave reads
        and the manapool case)."""
        if not icons or not isinstance(llm_read, dict):
            return llm_read, False
        cost_keys = ("cost_ice", "cost_poison", "cost_blood", "cost_moon", "cost_death")
        rune_to_key = {"ice": "cost_ice", "poison": "cost_poison",
                       "blood": "cost_blood", "moon": "cost_moon",
                       "death": "cost_death"}
        key_to_rune = {v: k for k, v in rune_to_key.items()}
        # LLM's nonzero costs as a set of runes
        llm_runes = {key_to_rune[k] for k in cost_keys
                     if int(llm_read.get(k) or 0) > 0}
        guard_runes = {icon["rune"] for icon in icons}
        if not llm_runes:
            return llm_read, False
        # CASE A: LLM's set of runes matches the guard's set — trust LLM.
        # The LLM paired each value with the correct rune by reading the
        # dialog visually; an override would just shuffle correct values.
        if llm_runes == guard_runes:
            return llm_read, False
        # CASE B: LLM has a rune the guard didn't detect (e.g. death when
        # the icon is ice). Override: use the OCR value (if readable) for
        # the detected runes, fall back to the LLM's value mapped to the
        # detected runes positionally.
        if len(llm_runes) != len(icons):
            return llm_read, False
        # All detected runes differ from the LLM's. OCR the values for
        # the detected icons and use those.
        ocr_values = []
        for icon in icons:
            v = self._read_icon_value(frame, icon)
            if v is None:
                break
            ocr_values.append((icon["rune"], v))
        if len(ocr_values) != len(icons):
            # OCR failed for some — fall back to using the LLM's values
            # mapped to the detected runes positionally (best-effort).
            llm_values = [int(llm_read.get(k) or 0) for k in cost_keys
                          if int(llm_read.get(k) or 0) > 0]
            if len(llm_values) != len(icons):
                return llm_read, False
            ocr_values = list(zip([i["rune"] for i in icons], llm_values))
        # Build the corrected dict
        corrected = dict(llm_read)
        for k in cost_keys:
            corrected[k] = 0
        for rune, val in ocr_values:
            corrected[rune_to_key[rune]] = val
        return corrected, True

    @staticmethod
    def _read_icon_value(frame, icon_info: dict) -> int | None:
        """OCR the "-N" cost text immediately to the right of a detected
        icon. Returns the integer value, or None on failure (no frame,
        out-of-bounds, OCR returns empty)."""
        if frame is None:
            return None
        ix, iy, iw, ih = icon_info["x"], icon_info["y"], icon_info["w"], icon_info["h"]
        x_start = ix + iw // 2 + 4
        x_end = min(x_start + 65, frame.shape[1])
        y_top = max(iy - ih // 2 - 2, 0)
        y_bot = min(iy + ih // 2 + 2, frame.shape[0])
        if x_end <= x_start or y_bot <= y_top:
            return None
        region = frame[y_top:y_bot, x_start:x_end]
        g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        big = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
            cv2.imwrite(t.name, big)
            txt = apple_vision_text(t.name)
            Path(t.name).unlink(missing_ok=True)
        m = re.search(r"\d+", txt or "")
        if m:
            return int(m.group(0))
        return None

    def read_dialog(self, dialog_frame) -> dict | None:
        """Read the purchase dialog with LLM + deterministic icon-template
        guard. Returns the LLM's dict with `cost_*` values corrected when
        the guard detects a misclassification, plus a `cost_icons` list
        describing the detected icons and a `cost_text_sufficient` bool
        (True if all cost entries appear sufficient). Returns None on
        LLM failure."""
        llm_read = self._llm_dialog(dialog_frame)
        if not llm_read:
            return None
        # Station-name OCR override: the LLM repeatedly misnames the dialog
        # station ("lectern"/"grave" for a Supply Cupboard dialog, observed
        # live), which trips the exact-match dialog guard and aborts real
        # buys. The dialog title OCR ("Supply Cupboard ... Lvl1 ... -20")
        # matched against the fixed station vocabulary is authoritative.
        ocr_fam = self._dialog_station_ocr(dialog_frame)
        if ocr_fam and normalize_station_name(str(llm_read.get("station") or "")) != ocr_fam:
            llm_read["station"] = ocr_fam
            llm_read["station_ocr_override"] = True
        icons = self._rune_icon_locate(dialog_frame)
        if icons:
            corrected, was_overridden = self._apply_icon_guard(
                llm_read, icons, dialog_frame)
            llm_read = corrected
            llm_read["cost_icons"] = icons
            llm_read["cost_icon_override"] = was_overridden
            # Aggregate sufficient flag
            sufficients = [c.get("sufficient") for c in icons
                           if c.get("sufficient") is not None]
            llm_read["cost_text_sufficient"] = (
                bool(sufficients) and all(sufficients)) if sufficients else None
        return llm_read

    def _dialog_station_ocr(self, dialog_frame) -> str | None:
        """Station family from the dialog title via OCR (vocabulary-matched).

        Returns the canonical family when the title text contains a known
        station name, else None. Never invents: unknown text yields None
        and the LLM read stands.
        """
        try:
            x, y, w, h = DIALOG_REGION
            box = dialog_frame[y:y + h, x:x + w]
            g = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
            big = cv2.resize(clahe, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
                cv2.imwrite(t.name, big)
                txt = apple_vision_text(t.name)
                Path(t.name).unlink(missing_ok=True)
            low = re.sub(r"[^a-z ]", "", (txt or "").lower())
            low = " " + re.sub(r"\s+", " ", low).strip() + " "
            for key, fam in sorted(FAMILY_ALIASES.items(), key=lambda kv: -len(kv[0])):
                if " " + key + " " in low:
                    return fam
        except Exception:
            pass
        return None

    def _find_confirm(self, frame):
        """Locate the green Confirm button via template match.
        Returns (x, y, score) in frame coords, or None below threshold."""
        tpl = cv2.imread(str(CONFIRM_TEMPLATE))
        if tpl is None or frame is None:
            return None
        y0, y1 = CONFIRM_BAND
        band = frame[y0:y1, :]
        if band.shape[0] < tpl.shape[0] or band.shape[1] < tpl.shape[1]:
            return None
        res = cv2.matchTemplate(band, tpl, cv2.TM_CCOEFF_NORMED)
        _minv, maxv, _minloc, maxloc = cv2.minMaxLoc(res)
        if maxv < CONFIRM_MIN:
            return None
        th, tw = tpl.shape[:2]
        return (maxloc[0] + tw // 2, y0 + maxloc[1] + th // 2, float(maxv))

    def _dialog_up(self, frame) -> bool:
        return self._find_confirm(frame) is not None

    def _back(self):
        self.device.back()

    def _tap_retry(self, x, y, attempts: int = 3, pause: float = 1.0):
        """Tap with bounded retries — adb injections drop intermittently when
        the LLM's CPU bursts starve the emulator (the Aug 19 failure mode), and
        a dropped CONFIRM tap silently aborts the purchase while the dialog
        stays up. Callers re-check state between attempts."""
        for i in range(attempts):
            self.device.tap(int(x), int(y))
            self.device.wait_for_idle(pause)
            if i < attempts - 1:      # caller re-checks; pause between retries
                self.device.wait_for_idle(0.5)

    # Card-icon match threshold for the "station panel verified" check
    # (template-match the Grave cross icon at its expected card region). Below
    # this we assume the open frame is NOT the Station panel and refuse the
    # buy rather than reading random content as currency (the Aug 26 19/0
    # swapped reading came from a feats panel).
    STATION_VERIFY_MIN = 0.55

    # the Station panel's currency bar at y=190-314 (5 rune slots
    # left-to-right) is a strong "is this actually the SHOP view?" signal.
    # A real SHOP view has each slot's rune icon sprite (rounded colored
    # shape) at the left edge of the slot. A mid-animation frame has
    # either an empty bar (slot crops are dark) or the bar is offscreen
    # (the panel is sliding in from the bottom). The check: the LEFT EDGE
    # of at least one slot has enough color saturation to look like an
    # icon (std >= ICON_PRESENT_STD), AND the RIGHT side of the bar has
    # any non-trivial bright content (the digits).
    CURRENCY_BAR_Y = CURRENCY_Y
    ICON_PRESENT_STD = 30

    def _check_currency_bar(self, frame) -> tuple[bool, str]:
        """Cheap structural check that the Station panel's 5-rune currency
        bar is present. Looks for rune-icon texture in the icon area
        (left edge of each slot) — a real bar has colored icons; a
        mid-animation or wrong-panel frame is mostly flat. Returns
        (ok, detail).
        """
        if frame is None or frame.shape[0] < CURRENCY_Y[1] or frame.shape[1] < 1280:
            return True, "frame too small to check"
        slots_with_icons = 0
        for x0, x1, _rune in CURRENCY_SLOTS:
            icon_end = x0 + int((x1 - x0) * 0.42)
            # icon area = the rounded rune shape at the slot's left
            icon_crop = frame[CURRENCY_Y[0]:CURRENCY_Y[1] + 1, x0:icon_end]
            if icon_crop.size == 0:
                continue
            # Saturation-based: real icons are saturated (red, green, blue,
            # orange, purple); flat backgrounds are gray. Use HSV.
            hsv = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)
            sat = hsv[..., 1]
            saturated_frac = float((sat > 80).mean())
            if saturated_frac >= 0.10:
                slots_with_icons += 1
        if slots_with_icons >= 2:
            return True, f"currency bar rendered ({slots_with_icons} slots with icons)"
        return False, f"currency bar empty ({slots_with_icons} slots with icons)"

    def _verify_station_panel(self, frame) -> tuple[bool, str]:
        """Confirm the open panel is the Station panel by template-matching at
        least one banked station sprite inside the first card region
        (CARD_ICON_Y) AND confirming the 5-rune currency bar is rendered.
        Returns (ok, detail). Catches the tutorial-overlay/feats-panel/
        transition-state cases that previously produced wrong currency
        readings (Aug 26 19/0 swapped reading) and the mid-animation
        cases that returned empty currency dicts (Aug 27 0/0/0/0/0)."""
        if frame is None:
            return False, "no frame"
        y0, y1 = self.CARD_ICON_Y
        x0, x1 = CARD_XS[0]
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return False, "card region empty"
        tpls = self._station_templates()
        best = -1.0
        best_fam = None
        for fam, tpl in tpls.items():
            _, score = self._best_scale(cvt_gray(crop), tpl)
            if score > best:
                best = score
                best_fam = fam
        card_ok = best >= self.STATION_VERIFY_MIN
        bar_ok, bar_detail = self._check_currency_bar(frame)
        if card_ok and bar_ok:
            return True, f"card icon matched {best_fam}@{best:.2f} + {bar_detail}"
        reasons = []
        if not card_ok:
            reasons.append(f"no station card icon (best {best_fam}@{best:.2f} < {self.STATION_VERIFY_MIN})")
        if not bar_ok:
            reasons.append(bar_detail)
        return False, " + ".join(reasons)

    def buy(self, family: str, confirm: bool = False) -> dict:
        """Attempt to buy `family` (e.g. "grave", "manapool"). See the class
        docstring for the confirmation model."""
        res = {"family": family, "confirm": confirm, "bought": False,
               "error": None, "dialog": None}
        card_pages = 0  # card-row pages advanced (restored before every close)
        try:
            self.device.screencap()
            lair = cv2.imread(str(self.device.screencap_path))
            # for confirm=true, the panel is already open with the
            # confirm dialog on top. The dock is covered (the panel hides it),
            # so `bar_visible()` flips False and `_check(lair, "station")`
            # would say "dock is not visible". That was a false-positive
            # rejection — the bot has the right button, just wrong gating.
            # Skip the dock check for confirm=true (the previous call staged
            # the dialog) and let `_verify_station_panel` confirm the panel
            # content before reading currency / tapping the green button.
            if not confirm:
                refuse = self._panel._check(lair, "station")
                if refuse:
                    res["error"] = refuse
                    return res
            self._panel.open_panel("station")
            frame = self._panel._wait_panel_open()
            if frame is None:
                res["error"] = "station panel did not open"
                return res
            # Verify the open panel is actually the Station panel before
            # reading currency/cards — the dock-flip detector accepts ANY
            # full-screen panel (tutorial, feats, transition), so without
            # this check the y=190-314 strip could be tutorial text or
            # feat numbers and produce wrong readings (Aug 26 19/0).
            ok, detail = self._verify_station_panel(frame)
            res["panel_verified"] = ok
            res["panel_verify_detail"] = detail
            if not ok:
                self._panel.close_panel()
                res["error"] = (f"station panel not detected (wrong panel open — "
                                f"{detail}). The buy aborted to avoid misreading "
                                f"non-station content as currency.")
                return res
            # Currency ONLY reads correctly on the open panel frame — the
            # fixed slots overlap the lair's MANA bar otherwise, which read
            # "16.8k"->168 ... "17.8k"->178 and made an unaffordable grave
            # look affordable (the inert-Confirm mystery, Aug 24).
            currency = self.read_currency(frame)
            # if read_currency returned all-zero, the panel is
            # still mid-animation (the body-rendered guard at OPEN_PAUSE
            # may have slipped through on a slow frame) or we captured a
            # transition state. Poll once more for a fully-rendered frame
            # and re-read. Logged as `currency_empty` so we can see the
            # case in telemetry.
            if all(v == 0 for v in currency.values()):
                better = self._panel._wait_panel_open()
                if better is not None:
                    frame = better
                    currency = self.read_currency(frame)
                    res["currency_retry"] = True
            res["currency"] = currency
            card, frame, card_pages, page_cards = self._find_card(frame, family)
            res["cards"] = page_cards
            res["card_pages"] = card_pages
            if card is None:
                self._restore_scroll(card_pages)
                self._panel.close_panel()
                res["error"] = (f"no {family} card in the station panel "
                                f"after scanning {card_pages + 1} pages")
                if self.locked_requirements:
                    res["locked"] = list(self.locked_requirements)
                return res
                # all 5 currencies checked (was ice + green-only)
            cost_keys = ("cost_ice", "cost_poison", "cost_blood",
                         "cost_moon", "cost_death")
            short = [(k.replace("cost_", ""),
                      card[k], currency.get(k.replace("cost_", ""), 0))
                     for k in cost_keys if card[k] > 0
                     and card[k] > currency.get(k.replace("cost_", ""), 0)]
            if short:
                self._restore_scroll(card_pages)
                self._panel.close_panel()
                need = ", ".join(f"{v} {r}" for r, v, _ in short)
                have = ", ".join(
                    f"{currency.get(k.replace('cost_', ''), 0)} {k.replace('cost_', '')}"
                    for k in cost_keys)
                res["error"] = (f"unaffordable: needs {need}; have {have}")
                # the previous hint told the model to use
                # collect_queue to tap chests — but collect_queue is the
                # DOCK Queue button (places a queued dock reward), NOT a
                # board tap. The right way to drain a board chest is
                # `spawn` (one tap per use, mana-free). Be explicit so
                # the model doesn't try the wrong tool.
                res["hint"] = ("gather the missing currencies first: tap "
                               "Ice Chests ON THE BOARD with the `spawn` "
                               "action (one tap per use, mana-free, each "
                               "tap drops a rune onto the board), then merge "
                               "the rune stacks up, then feed the max-level "
                               "stack to the Devourer for the Ice Rune "
                               "currency. Retry the purchase once the "
                               "balance is sufficient.")
                return res
            before_stations = self._family_count(frame, family)
            # Tap the card -> poll for the confirm dialog. Retried: a dropped
            # card tap would look identical to "unaffordable".
            self._tap_retry(int(card["x"]), CARD_TAP_Y)
            dialog_frame = None
            for _ in range(DIALOG_POLLS):
                self.device.wait_for_idle(DIALOG_PAUSE)
                self.device.screencap()
                f2 = cv2.imread(str(self.device.screencap_path))
                if f2 is not None and self._dialog_up(f2):
                    dialog_frame = f2
                    break
            if dialog_frame is None:
                # No dialog: unaffordable in game terms or a no-op tap. BACK is
                # NOT needed (no dialog); just close the sheet.
                self._restore_scroll(card_pages)
                self._panel.close_panel()
                res["error"] = "no confirm dialog appeared (unaffordable or not purchasable)"
                return res
            read = self.read_dialog(dialog_frame) or {}
            res["dialog"] = read
            # the dialog is the AUTHORITATIVE cost (the card read
            # can be wrong; the icon guard corrects the dialog read). Use it
            # for the post-read re-check.
            if read.get("is_purchase_dialog") and read.get("station"):
                for k in cost_keys:
                    if int(read.get(k) or 0) > 0:
                        card[k] = int(read[k])
            dst = str(read.get("station") or "").lower()
            # exact family match (was a loose 4-char prefix). The
            # panel is now slidable and the model's card-tap may land on the
            # wrong neighbor; refuse a mismatch instead of confirming.
            if dst != family and family not in dst and dst not in family:
                self._back()
                self._restore_scroll(card_pages)
                self._panel.close_panel()
                res["error"] = f"dialog mismatch: wanted {family}, dialog says {dst!r}"
                return res
            if not confirm:
                self._back()
                self._restore_scroll(card_pages)
                self._panel.close_panel()
                # stash the staged two-phase buy so the planner's
                # compelled follow-through gate can force the next
                # buy(family, confirm=True). The dict carries the family +
                # dialog + step so the planner can show the user a reminder
                # line and detect a family mismatch.
                self._pending_buy = {
                    "family": family,
                    "dialog": read,
                    "step": 0,  # filled in by the planner (StationShop has no _step_count)
                }
                res["note"] = ("dialog verified — call buy(family, confirm=True) "
                               "to complete the purchase")
                res["pending_buy"] = True
                return res
            # confirm=True path. The pending_buy is cleared below after a
            # successful bought=True (so a failed confirm doesn't wipe the
            # staged state — the planner can re-try with confirm=true).
            before_ice = currency.get("ice", 0)
            # Confirm with retries: tap where the template ACTUALLY matched
            # (fixed coordinates missed when the live dialog sits elsewhere),
            # re-locating each attempt; a dropped injection retries too.
            for attempt in range(3):
                self.device.screencap()
                f_now = cv2.imread(str(self.device.screencap_path))
                loc = self._find_confirm(f_now)
                if loc is None:
                    break                    # dialog gone (already confirmed?)
                self._tap_retry(*loc[:2], attempts=2)
                closed = False
                for _ in range(DIALOG_POLLS):
                    self.device.wait_for_idle(DIALOG_PAUSE)
                    self.device.screencap()
                    f3 = cv2.imread(str(self.device.screencap_path))
                    if f3 is None or not self._dialog_up(f3):
                        closed = True
                        break
                if closed:
                    after = self.read_currency(f3)
                    res["currency_after"] = after
                    # Purchase truth = a NEW station of the family on the
                    # board (the panel frame shows the board). The currency
                    # read alone is unreliable post-confirm: the sheet may
                    # have closed, putting the lair's mana bar under the
                    # fixed slots.
                    gained = self._family_count(f3, family) > before_stations
                    paid = after.get("ice", before_ice) < before_ice
                    res["bought"] = bool(gained or paid)
                    res["confirm_xy"] = [loc[0], loc[1]]
                    if res["bought"]:
                        # clear the pending buy on a successful
                        # confirm so the planner knows the buy completed.
                        self._pending_buy = None
                    break
                res["error"] = f"dialog still up after Confirm attempt {attempt + 1}"
            else:
                res["error"] = res.get("error") or "dialog did not close after Confirm"
            if self._panel._panel_open(cv2.imread(str(self.device.screencap_path))) \
                    or not self.bottombar.bar_visible(
                        cv2.imread(str(self.device.screencap_path))):
                self._restore_scroll(card_pages)
                self._panel.close_panel()
            return res
        except BoardLostError:
            raise
        except Exception as exc:  # noqa: BLE001
            res["error"] = f"{type(exc).__name__}: {exc}"
            try:
                if lair is not None and not self.bottombar.bar_visible(
                        cv2.imread(str(self.device.screencap_path))):
                    self._restore_scroll(card_pages)
                    self._panel.close_panel()
            except Exception:  # noqa: BLE001
                pass
            return res
