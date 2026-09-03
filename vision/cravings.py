"""Devourer Cravings reader (Phase 3 MenuExplorer, cravings-first).

Cravings are the Devourer's current demand (unlocked at Devourer L3): feed it
a specific creature to get a food bonus (e.g. "Skeleton Lvl 1, 0/7, +150").
The cravings bubble is always visible above the board on the lair screen and
is tappable to open the Cravings menu — both verified live on NecroMerger_PS.

Geometry (native 1280x2856, user-click calibrated):
- CRAVINGS_BUBBLE: bubble at (38,546)-(164,642), shows the craved monster's
  icon. Self-match across bare-lair captures ~0.75 (idle animation), so a
  multi-frame bubble icon bank is needed for identity (like board sprites).
- CRAVINGS_BUBBLE_CENTER: tap opens the Cravings menu.
- CRAVINGS_PANEL: generous menu panel ROI covering the title + cards
  (verified: LLM reads {item, level, count_done, count_required, reward} from
  this crop on live + saved frames).
- Tracker (X/Y count under the bubble, ~(60,680)-(280,770)) is NOT tappable
  and OCR-unreadable (stylized font) — the menu read is the authoritative
  count/reward source.

Safety invariants (from AGENTS.md "Menu exploration Phase 1"):
- BACK key closes an open panel safely; never BACK on the bare board.
- The menu is read-only (no purchase/build buttons); bubble tap is side-effect
  free (opens the menu only). Verify the board returned after close.
"""

import json
import subprocess
import time
from pathlib import Path

import cv2

from planner.llm import _extract_json
from planner.llm_client import LLMError
from planner.llm_vision import _encode_frame
from vision.classifier import TemplateClassifier
from vision.geometry import _blob_bbox

# Always-visible cravings bubble (x, y, w, h), native pixels.
CRAVINGS_BUBBLE = (38, 546, 126, 96)
CRAVINGS_BUBBLE_CENTER = (101, 594)      # tap opens the Cravings menu
# Menu panel ROI covering title + current/upcoming cards (x, y, w, h).
CRAVINGS_PANEL = (20, 700, 1240, 1460)
# Dedicated cravings icon bank (bubble crops keyed by item base name).
CRAVINGS_BANK_DIR = "assets/cravings"
CRAVINGS_BANK_THRESHOLD = 0.6

OPEN_PAUSE = 1.5            # let the menu animate in
CLOSE_PAUSE = 0.6           # let the board return after BACK
CLOSE_POLLS = 6             # re-check the board for up to CLOSE_PAUSE*CLOSE_POLLS sec
CLOSE_BACK_RETRIES = 1      # extra BACKs when the menu is confirmed still open

# Anti-pollution guards for the bubble icon bank. A near-constant (blank/dark)
# crop is an EMPTY bubble slot — never bank or match it (a template with std
# ~9 matched the empty slot at 1.0 and drove false "skeleton" cravings all
# run). Real craving icons have strong texture (std ~78).
BANK_MIN_STD = 20.0          # minimum gray std to bank a bubble crop
MATCH_MIN_STD = 20.0         # skip templates/crops at or below this std
BUBBLE_MARGIN_MIN = 0.10     # best bubble match must beat the runner-up by this
# Menu-open detection: the cravings menu covers the board, so the floor blob
# relocates (bare board top y~1150, menu open y~1900). Require a real shift
# instead of a fragile mean-diff threshold (idle animation trips ≥1.0 diff).
BLOB_SHIFT_MIN = 150         # min |blob_top_shift| (px) to treat as menu-open


class BoardLostError(RuntimeError):
    """The game is no longer showing the board (e.g. BACK exited it)."""


# The game's main (Unity) activity; the authoritative "game alive" signal is
# 'is NecroMerger the topResumedActivity?' (see game_in_foreground).
GAME_PACKAGE = "com.grumpyrhinogames.necromerger"


def game_in_foreground(device) -> bool:
    """True when the NecroMerger activity is the foreground (top-resumed)
    activity — i.e. the game is definitely still running.

    The Android launcher / another app has no game window, so this cleanly
    separates a genuine exit (BACK on a bare board) from a *still-running*
    game whose menu/panel just hasn't animated out yet. Returns True on any
    uncertainty (no device, dumpsys failure/window hidden) — the close paths
    must never halt a game that might be alive; a stuck menu is recoverable,
    a wrong STOPPING requires a human to relaunch."""
    if device is None:
        return True  # nothing to verify against -> assume alive
    try:
        pkg = device.top_resumed_activity()
    except Exception:
        return True
    if not pkg:
        return True  # cannot confirm an exit -> assume alive
    return pkg == GAME_PACKAGE


CRAVINGS_QUESTION = """This is the CRAVINGS menu of NecroMerger (screen crop). Read the ACTIVE craving
card: the item required, its level, the progress count (like 0/7), and the food
reward (like +150). Reply with ONLY JSON:
{"cravings": [{"item": "<name>", "level": <int or null>, "count_done": <int>,
"count_required": <int>, "reward": <int or null>}]}
Example: for a Skeleton Lvl 1 craving at 0/7 feeding +150 reply
{"cravings": [{"item": "Skeleton", "level": 1, "count_done": 0,
"count_required": 7, "reward": 150}]}.
If the text is unreadable or no craving is shown, return {"cravings": []}.
No prose outside the JSON."""


class CravingsReader:
    """Read the Devourer's cravings: cheap no-tap bubble identity via a
    dedicated icon bank, plus an authoritative open->read->BACK menu cycle.

    The bubble icon is template-matched against `assets/cravings/` (dedicated
    bank — board templates do NOT match craving icons, they render
    differently). When the menu read identifies the item, the bubble crop is
    banked under that base name so future no-tap reads recognize it.
    """

    def __init__(self, device=None, llm=None, bank_dir: str = CRAVINGS_BANK_DIR,
                 threshold: float = CRAVINGS_BANK_THRESHOLD):
        self.device = device
        self.llm = llm
        self.bank_dir = Path(bank_dir)
        self.threshold = threshold
        self.bank: dict[str, list] = {}
        self._load_bank()

    # ---- crops / geometry -------------------------------------------------

    def bubble_crop(self, frame):
        x, y, w, h = CRAVINGS_BUBBLE
        return frame[y:y + h, x:x + w]

    def panel_crop(self, frame):
        x, y, w, h = CRAVINGS_PANEL
        return frame[y:y + h, x:x + w]

    # ---- bubble icon bank -------------------------------------------------

    def _load_bank(self) -> None:
        for path in sorted(self.bank_dir.glob("*.png")):
            item_id = path.stem.split("__")[0]
            self.bank.setdefault(item_id, []).append(cv2.imread(str(path)))

    def bank_bubble(self, item_id: str, frame) -> bool:
        """Save the current bubble crop under `item_id` (multi-frame bank).

        Rejects near-constant crops: an empty bubble slot is dark/uniform
        (std ~9) and must never be banked — it would false-match at 1.0 and
        read as a craving on a board with no bubble (the Aug 9 exit bug).
        """
        if not item_id or frame is None:
            return False
        crop = self.bubble_crop(frame)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < BANK_MIN_STD:
            return False
        frames = self.bank.setdefault(item_id, [])
        path = self.bank_dir / f"{item_id}__{len(frames)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop)
        frames.append(crop)
        return True

    @staticmethod
    def _std(crop) -> float:
        """Gray std of a crop (texture proxy); 0 for None/blank."""
        if crop is None:
            return 0.0
        return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std())

    def match_bubble(self, frame) -> tuple[str | None, float]:
        """Best banked item for the current bubble crop, else (None, score).

        Hardened against empty-slot false positives: the crop must have real
        texture (a blank slot isn't a craving), near-constant templates are
        skipped, and the best match must beat the runner-up by
        BUBBLE_MARGIN_MIN (same margin pattern as board merges) so a dark
        slot can never read `skeleton 1.0`.
        """
        crop = self.bubble_crop(frame)
        if self._std(crop) < MATCH_MIN_STD:
            return None, -1.0
        best_id, best_score, runner_up = None, -1.0, -1.0
        for item_id, frames in self.bank.items():
            item_score = -1.0
            for t in frames:
                if self._std(t) < MATCH_MIN_STD:
                    continue
                s = TemplateClassifier._match_score(crop, t)
                if s > item_score:
                    item_score = s
            if item_score > best_score:
                runner_up = best_score
                best_score, best_id = item_score, item_id
            elif item_score > runner_up:
                runner_up = item_score
        if best_id is None or best_score < self.threshold:
            return None, best_score
        if best_score - runner_up < BUBBLE_MARGIN_MIN:
            return None, best_score
        return best_id, best_score

    # ---- menu open / read / close ----------------------------------------

    def open_menu(self) -> None:
        if self.device is None:
            return
        self.device.tap(*CRAVINGS_BUBBLE_CENTER)
        self.device.wait_for_idle(OPEN_PAUSE)

    def close_menu(self, lair=None) -> None:
        """BACK-dismiss the cravings menu (safe: menu confirmed open).

        Polls for the board to return: the menu animates out slowly under
        emulator load and a single screencap can catch it mid-transition
        (blob still displaced), which used to false-alarm BoardLostError on a
        perfectly healthy game. Retries BACK only when the menu is CONFIRMED
        still open (a blind re-BACK could hit the bare board and really exit).
        Raises BoardLostError ONLY when the game genuinely exited (it is no
        longer the foreground activity) — a still-running game with a slow/
        stuck menu logs a warning and lets the next cycle retry instead.
        """
        if self.device is None:
            return
        for _attempt in range(CLOSE_BACK_RETRIES + 1):
            self.device.back()
            if self._board_visible(lair):
                return
            if _attempt < CLOSE_BACK_RETRIES:
                # Only re-BACK if the menu is confirmed still open (displaced
                # blob / board fully covered). Otherwise the first BACK closed
                # it and the poll just missed the animation tail — re-BACKing
                # now would hit the bare board and exit the game.
                self.device.screencap()
                frame = cv2.imread(str(self.device.screencap_path))
                if frame is None or not self._menu_opened(lair, frame):
                    break
        if game_in_foreground(self.device):
            print("[cravings] board did not return after BACK; the game is still "
                  "running (menu/panel animating or stuck open) — continuing, "
                  "next cycle retries")
            return
        raise BoardLostError(
            "board did not return after BACK — the game likely exited")

    def _board_visible(self, lair=None, polls=CLOSE_POLLS,
                       pause=CLOSE_PAUSE) -> bool:
        """True once the board floor blob is back at its board position.

        Polls for up to `polls` frames: the menu frame itself contains a
        floor-colored blob (at y~1900) and the exit animation passes through
        partial coverage, so "any blob present" is NOT enough — the blob must
        be at the bare-board location (y~1150 in the lair reference) before
        the board counts as restored. When `lair` is given, its blob top is
        the reference; otherwise a blob whose top is in the bare-board band
        counts as the board. No blob found = still fully covered (or the game
        exited) — keep polling, the blob returns as the menu animates out.
        """
        for _ in range(polls):
            try:
                self.device.screencap()
                frame = cv2.imread(str(self.device.screencap_path))
                if frame is None:
                    time.sleep(pause)
                    continue
                fx, fy, fw, fh = _blob_bbox(frame)
            except (ValueError, subprocess.CalledProcessError):
                time.sleep(pause)
                continue
            if lair is not None:
                try:
                    _lx, ly, _lw, _lh = _blob_bbox(lair)
                except ValueError:
                    return False
                if abs(ly - fy) <= BLOB_SHIFT_MIN:
                    return True
            elif 1000 <= fy <= 1450:
                return True
            time.sleep(pause)
        return False

    def read_menu(self, frame) -> list[dict]:
        """LLM-read the Cravings menu from a full-screen frame. Returns the
        active craving list [{item, level, count_done, count_required, reward}]
        (empty on unreadable text / no craving / failure). Grounded in the
        game's own UI. `frame` must have the menu open; no tap happens here."""
        if self.llm is None:
            return []
        messages = [
            {"role": "system", "content": "You read NecroMerger Cravings menus from screenshot crops."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _encode_frame(self.panel_crop(frame))}},
                {"type": "text", "text": CRAVINGS_QUESTION},
            ]},
            # Assistant prefill: skips the reasoning model's thinking block
            # (which otherwise exhausts max_tokens and returns empty content).
            {"role": "assistant", "content": '{"cravings":'},
        ]
        try:
            content, _, _full = self.llm.chat_message(messages, max_tokens=256, json_mode=False)
            data = _extract_json(content)
        except (LLMError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        cravings = data.get("cravings")
        if not isinstance(cravings, list):
            return []
        out = []
        for c in cravings:
            if not isinstance(c, dict):
                continue
            item = str(c.get("item") or "").strip().lower()
            if not item:
                continue
            def _int(v):
                try:
                    return int(v) if v is not None else None
                except (TypeError, ValueError):
                    return None
            out.append({
                "item": item,
                "level": _int(c.get("level")),
                "count_done": _int(c.get("count_done")),
                "count_required": _int(c.get("count_required")),
                "reward": _int(c.get("reward")),
            })
        return out

    # ---- full cycle -------------------------------------------------------

    def _menu_opened(self, lair, frame) -> bool:
        """True if tapping the bubble actually opened the Cravings menu.

        The cravings menu covers the board, so the board's floor blob relocates
        (bare board top y~1150, menu open y~1900) or disappears entirely. A
        mean-abs-diff over CRAVINGS_PANEL alone is NOT reliable (idle
        animation trips a >= 1.0 diff without any menu). Requiring a real
        blob shift encodes the safety invariant: never BACK unless the board
        is gone. Prevents BACK on the bare board, which exits the game
        (pre-unlock there is no cravings menu and no blob shift).
        """
        if lair is None or frame is None:
            return False
        try:
            lx, ly, lw, lh = _blob_bbox(lair)
        except ValueError:
            return False  # no board visible in the lair frame
        try:
            fx, fy, fw, fh = _blob_bbox(frame)
        except ValueError:
            return True   # board fully covered by the menu
        return abs(ly - fy) > BLOB_SHIFT_MIN

    def get_cravings(self, screencap_path=None) -> dict:
        """Open the menu, read the craving, close it, bank the bubble icon.

        Returns {"cravings": [...], "bubble": {"item_id": ..., "score": ...}}.
        Safe only in live mode (requires device). Best-effort for ordinary
        failures (menu closed when actually open, partial data returned), but
        raises BoardLostError if the game is no longer showing the board after
        BACK — the caller must halt, not keep tapping a dead screen.

        The bare-lair frame (bubble visible) is captured BEFORE opening the
        menu: the always-visible bubble is the identity signal the cheap no-tap
        read uses, so it must be banked from the lair frame — NOT the
        menu-open frame (the menu overlays a different UI in that region).
        """
        result = {"cravings": [], "bubble": {"item_id": None, "score": -1.0},
                  "error": None}
        if self.device is None:
            result["error"] = "no device (live mode only)"
            return result
        menu_opened = False
        try:
            self.device.screencap()
            lair = cv2.imread(str(screencap_path or self.device.screencap_path))
            if lair is None:
                result["error"] = "screencap unreadable"
                return result
            self.open_menu()
            self.device.screencap()
            frame = cv2.imread(str(screencap_path or self.device.screencap_path))
            if frame is None:
                result["error"] = "menu screencap unreadable"
                return result
            menu_opened = self._menu_opened(lair, frame)
            if not menu_opened:
                # E.g. cravings not yet unlocked (Devourer < L3): the bubble tap
                # opened nothing, so no BACK — BACK on the bare board exits the game.
                result["error"] = "menu did not open"
                bubble_id, bubble_score = self.match_bubble(lair)
                result["bubble"] = {"item_id": bubble_id, "score": round(bubble_score, 3)}
                return result
            cravings = self.read_menu(frame)
            # Sanity: a real craving always has count_required >= 1. Filtering
            # here keeps confabulated reads (LLM inventing a craving from a
            # non-menu crop) from being banked or surfaced to the planner.
            cravings = [c for c in cravings
                        if (c.get("count_required") or 0) >= 1]
            result["cravings"] = cravings
            if cravings:
                # Bank the lair-frame bubble crop under the current craved
                # item's base name so the cheap no-tap bubble read can
                # identify it next step.
                self.bank_bubble(cravings[0]["item"], lair)
            bubble_id, bubble_score = self.match_bubble(lair)
            result["bubble"] = {"item_id": bubble_id, "score": round(bubble_score, 3)}
            return result
        except BoardLostError:
            raise  # game likely exited — let the caller halt the loop
        except Exception as exc:
            result["error"] = str(exc)
            return result
        finally:
            if menu_opened:
                self.close_menu(lair)
