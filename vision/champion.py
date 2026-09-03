"""Champion spawn tracker reader (lair HUD, top-right).

Champions (The Peasant, The Knight, ...) spawn into the lair when their
progress counter hits 100%. The counter fills up by MERGING creatures of
specific families tied to each Champion (e.g. The Peasant's progress
fills when you merge Skeletons, Zombies, or Mummies — NOT on a timer).
Once spawned, the Champion sits on the board and attacks the Devourer
until you feed it enough creatures. Their current spawn progress is
shown by a lair HUD element mirrored directly across from the
Devourer's Cravings bubble: the current champion's portrait plus a
progress number at the bottom of the element. Tapping the element
opens the Champion spawn screen that shows each champion's spawn
progress authoritatively.

Geometry (native 1280x2856, user-click calibrated):
- CRAVINGS_BUBBLE is at (38,546)-(164,642) top-left; the champion tracker is
  its mirror across the screen center x=640 -> (1116,546)-(1242,642), tap
  center (1179,594). Verified live: tapping it opens the champion screen.
- The tracker element is animated (idle bob), so the portrait identity needs a
  multi-frame icon bank (like cravings/bottombar).
- CHAMPION_PANEL: the champion screen's content ROI (title + champion cards),
  verified the vision LLM reads {name, active, progress, ready} from it.

Safety invariants (same as cravings.py):
- BACK only after the champion screen is confirmed open; verify the board
  returned after BACK; raise BoardLostError otherwise (BACK on the bare board
  exits the game — a false "menu open" must never cause a BACK).
- Menu-open detection: the champion screen covers the top strip, so the
  tracker element region reads dark (~11) while the bare lair reads bright
  (~60, textured portrait). The blob-shift trick from cravings does NOT work
  here (this menu leaves a blob at y~1185).
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
from vision.cravings import game_in_foreground
from vision.geometry import _blob_bbox

# Always-visible champion tracker element (x, y, w, h), native pixels.
CHAMPION_TRACKER = (1116, 546, 126, 96)   # mirror of the cravings bubble
CHAMPION_TRACKER_CENTER = (1179, 594)     # tap opens the champion screen
# Champion screen content ROI covering title + champion cards (x, y, w, h).
CHAMPION_PANEL = (90, 260, 1100, 2200)
# Dedicated champion portrait bank (tracker crops keyed by champion name).
CHAMPION_BANK_DIR = "assets/champions"
CHAMPION_BANK_THRESHOLD = 0.6

OPEN_PAUSE = 1.8            # let the champion screen animate in
CLOSE_PAUSE = 0.8           # let the board return after BACK

# Anti-pollution guards for the portrait bank (mirror of cravings.py).
BANK_MIN_STD = 20.0          # minimum gray std to bank a tracker crop
MATCH_MIN_STD = 20.0         # skip templates/crops at or below this std
TRACKER_MARGIN_MIN = 0.10    # best tracker match must beat the runner-up by this

# Menu-open detection: the champion screen darkens the whole panel region
# (CHAMPION_PANEL gray mean ~38.5) vs the bare lair (~65.6). Live-measured
# across lair (65.6) and both champion-screen layouts (38.5, 40.7). The tracker
# element mean is NOT reliable (one valid menu layout reads 32.9, above the
# dark-tracker threshold). Threshold 50 sits mid-gap with >10px margin.
PANEL_OPEN_MEAN_MAX = 50.0  # CHAMPION_PANEL gray mean <= this => menu is open


class BoardLostError(RuntimeError):
    """The game is no longer showing the board (e.g. BACK exited it)."""


CHAMPION_QUESTION = """This is the CHAMPION spawn screen of NecroMerger (screen crop). For EACH champion shown, read:
the champion name, whether it is currently active/attacking (spawning next), and its spawn progress
(as a fraction like 1/3 or a percentage like 33% or 'ready'/100%).
Reply with ONLY JSON:
{"champions": [{"name": str, "active": bool, "progress": str, "ready": bool}]}
Example: {"champions": [{"name": "The Peasant", "active": true, "progress": "100%", "ready": true}]}
If the text is unreadable, return {"champions": []}. No prose outside the JSON."""


class ChampionReader:
    """Read the champion spawn tracker: cheap no-tap portrait identity via a
    dedicated icon bank, plus an authoritative open->read->BACK menu cycle.

    The tracker element (top-right lair HUD) always shows the CURRENT champion
    whose spawn meter is filling. When its progress reaches 100% the champion
    spawns and attacks. `match_tracker` identifies which champion is queued
    without a tap; `get_champion_status` opens the champion screen for the
    authoritative name/progress/ready read.
    """

    def __init__(self, device=None, llm=None, bank_dir: str = CHAMPION_BANK_DIR,
                 threshold: float = CHAMPION_BANK_THRESHOLD):
        self.device = device
        self.llm = llm
        self.bank_dir = Path(bank_dir)
        self.threshold = threshold
        self.bank: dict[str, list] = {}
        self._load_bank()

    # ---- crops / geometry -------------------------------------------------

    def tracker_crop(self, frame):
        x, y, w, h = CHAMPION_TRACKER
        return frame[y:y + h, x:x + w]

    def panel_crop(self, frame):
        x, y, w, h = CHAMPION_PANEL
        return frame[y:y + h, x:x + w]

    # ---- portrait icon bank -----------------------------------------------

    def _load_bank(self) -> None:
        for path in sorted(self.bank_dir.glob("*.png")):
            champion_id = path.stem.split("__")[0]
            self.bank.setdefault(champion_id, []).append(cv2.imread(str(path)))

    def bank_tracker(self, champion_id: str, frame) -> bool:
        """Save the current tracker crop under `champion_id` (multi-frame bank).

        Rejects near-constant crops: a dark/blank tracker slot (e.g. no
        champion incoming) must never be banked — it would false-match at 1.0
        and read as a champion on a board with no tracker.
        """
        if not champion_id or frame is None:
            return False
        crop = self.tracker_crop(frame)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < BANK_MIN_STD:
            return False
        frames = self.bank.setdefault(champion_id, [])
        path = self.bank_dir / f"{champion_id}__{len(frames)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop)
        frames.append(crop)
        return True

    @staticmethod
    def _std(crop) -> float:
        if crop is None:
            return 0.0
        return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std())

    def match_tracker(self, frame) -> tuple[str | None, float]:
        """Best banked champion for the current tracker crop, else (None, score).

        Hardened against empty-slot false positives: the crop must have real
        texture, near-constant templates are skipped, and the best match must
        beat the runner-up by TRACKER_MARGIN_MIN.
        """
        crop = self.tracker_crop(frame)
        if self._std(crop) < MATCH_MIN_STD:
            return None, -1.0
        best_id, best_score, runner_up = None, -1.0, -1.0
        for champion_id, frames in self.bank.items():
            item_score = -1.0
            for t in frames:
                if self._std(t) < MATCH_MIN_STD:
                    continue
                s = TemplateClassifier._match_score(crop, t)
                if s > item_score:
                    item_score = s
            if item_score > best_score:
                runner_up = best_score
                best_score, best_id = item_score, champion_id
            elif item_score > runner_up:
                runner_up = item_score
        if best_id is None or best_score < self.threshold:
            return None, best_score
        if best_score - runner_up < TRACKER_MARGIN_MIN:
            return None, best_score
        return best_id, best_score

    # ---- menu open / read / close ----------------------------------------

    def open_menu(self) -> None:
        if self.device is None:
            return
        self.device.tap(*CHAMPION_TRACKER_CENTER)
        self.device.wait_for_idle(OPEN_PAUSE)

    def close_menu(self, lair=None) -> None:
        """BACK-dismiss the champion screen (safe: screen confirmed open).

        Polls for the board to return (the screen animates out slowly under
        emulator load — a single check catches a mid-animation frame and
        false-alarms). Retries BACK only when the champion screen is CONFIRMED
        still open (never a blind re-BACK onto a bare board — that really
        exits the game). Raises BoardLostError ONLY when the game genuinely
        exited (no longer the foreground activity); a still-running game with
        a slow/stuck screen logs a warning and lets the next cycle retry.
        """
        if self.device is None:
            return
        import time as _t
        for _attempt in range(2):
            self.device.back()
            if self._board_visible():
                return
            if _attempt == 0:
                # Only re-BACK if the champion screen is confirmed still open
                # (panel region still darkened). A mid-animation board that the
                # poll missed must NOT be re-BACKed — that hits the bare board.
                self.device.screencap()
                frame = cv2.imread(str(self.device.screencap_path))
                if frame is None or not self._menu_opened(lair, frame):
                    break
                _t.sleep(0.5)
        if game_in_foreground(self.device):
            print("[champion] board did not return after BACK; the game is still "
                  "running (screen animating or stuck open) — continuing, "
                  "next cycle retries")
            return
        raise BoardLostError(
            "board did not return after BACK — the game likely exited")

    def _board_visible(self, lair=None) -> bool:
        """Poll until the board is clean (or ~2.5s elapses).

        Board-visible == panel region NOT darkened by the champion screen
        (panel mean > PANEL_OPEN_MEAN_MAX). The floor-blob check does NOT work
        for this menu: the champion screen leaves a blob at y~1185, in the
        same band as the lair blob (y~1150), so blob geometry can't tell them
        apart. Poll a few times because the board animates back after BACK.
        """
        for _ in range(5):
            try:
                self.device.screencap()
                frame = cv2.imread(str(self.device.screencap_path))
                if frame is None:
                    return False
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                x, y, w, h = CHAMPION_PANEL
                mean = float(gray[y:y + h, x:x + w].mean())
                if mean > PANEL_OPEN_MEAN_MAX:
                    return True
            except (ValueError, subprocess.CalledProcessError):
                pass
            time.sleep(0.5)
        return False

    def read_menu(self, frame) -> list[dict]:
        """LLM-read the champion screen from a full-screen frame. Returns
        [{name, active, progress, ready}] (empty on failure/unreadable)."""
        if self.llm is None:
            return []
        messages = [
            {"role": "system", "content": "You read NecroMerger Champion spawn screens from screenshot crops."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _encode_frame(self.panel_crop(frame))}},
                {"type": "text", "text": CHAMPION_QUESTION},
            ]},
            {"role": "assistant", "content": '{"champions":'},
        ]
        try:
            content, _, _full = self.llm.chat_message(messages, max_tokens=256, json_mode=False)
            data = _extract_json(content)
        except (LLMError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        champions = data.get("champions")
        if not isinstance(champions, list):
            return []
        out = []
        for c in champions:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "").strip()
            if not name:
                continue
            out.append({
                "name": name,
                "active": bool(c.get("active")),
                "progress": str(c.get("progress") or ""),
                "ready": bool(c.get("ready")),
            })
        return out

    # ---- full cycle -------------------------------------------------------

    def _menu_opened(self, lair, frame) -> bool:
        """True if tapping the tracker opened the champion screen.

        The champion screen darkens the whole panel region (~38.5 mean vs
        ~65.6 on the bare lair). Checked on the MENU frame, independent of any
        blob (this menu leaves a blob at y~1185, so the cravings blob-shift
        trick does not apply).
        """
        if frame is None:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = CHAMPION_PANEL
        mean = float(gray[y:y + h, x:x + w].mean())
        return mean <= PANEL_OPEN_MEAN_MAX

    def get_champion_status(self, screencap_path=None) -> dict:
        """Open the champion screen, read it, close it, bank the tracker icon.

        Returns {"champions": [...], "tracker": {"champion_id": ..., "score": ...}}.
        Safe only in live mode (requires device). Raises BoardLostError if the
        game is no longer showing the board after BACK.

        The bare-lair frame (tracker visible) is captured BEFORE opening the
        screen: the tracker portrait is the cheap identity signal, so it must
        be banked from the lair frame, not the menu-open frame.
        """
        result = {"champions": [], "tracker": {"champion_id": None, "score": -1.0},
                  "error": None}
        if self.device is None:
            result["error"] = "no device (live mode only)"
            return result
        menu_opened = False
        try:
            # Self-heal: if a champion screen is already open (e.g. leftover
            # from a prior stuck cycle), BACK it and POLL until the board is
            # clean before proceeding — the screen animates out slowly and the
            # tap must target the tracker on the bare board.
            if not self._board_visible():
                self.device.back()
                if not self._board_visible():
                    result["error"] = "board did not return after self-heal BACK"
                    return result
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
                # The tap may have missed (observed intermittently). Retry ONCE
                # from the clean lair before giving up. Still no BACK — BACK on
                # the bare board exits the game.
                self.open_menu()
                self.device.screencap()
                frame = cv2.imread(str(screencap_path or self.device.screencap_path))
                if frame is not None:
                    menu_opened = self._menu_opened(lair, frame)
            if not menu_opened:
                result["error"] = "menu did not open"
                cid, cscore = self.match_tracker(lair)
                result["tracker"] = {"champion_id": cid, "score": round(cscore, 3)}
                return result
            champions = self.read_menu(frame)
            result["champions"] = champions
            # Bank the lair-frame tracker crop under the active champion's
            # normalized name so the cheap no-tap read can identify it next step.
            if champions:
                active = next((c for c in champions if c.get("active")), None)
                name = (active or champions[0]).get("name", "")
                cid = name.lower().replace("the ", "").replace(" ", "_")
                if cid:
                    self.bank_tracker(cid, lair)
            cid, cscore = self.match_tracker(lair)
            result["tracker"] = {"champion_id": cid, "score": round(cscore, 3)}
            return result
        except BoardLostError:
            raise  # game likely exited — let the caller halt the loop
        except Exception as exc:
            result["error"] = str(exc)
            return result
        finally:
            if menu_opened:
                self.close_menu(lair)
