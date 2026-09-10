"""Entry point: screenshot -> board -> move -> gesture, repeat."""

import argparse
import subprocess
import time
from pathlib import Path

import cv2

from controller.actions import GestureFailed, Layout, execute
from env.adb import Device
from metrics.logger import SessionLog
from planner.agent import HeuristicPlanner, Planner, necromerger_cell
from planner.glossary import read_chains, read_damage_values, read_feed_values
from planner.llm import LLMPlanner
from planner.llm_vision import VisionLLMPlanner
from planner.llm_client import LLMError
from planner.vision_drive import VisionDrivenPlanner
from vision import grid
from vision.classifier import TemplateClassifier
from vision.geometry import _blob_bbox, NoFloorBlobError
from vision.cravings import BoardLostError
from vision.grid import OCCUPANCY_THRESHOLD, crop_cell, occupancy_score
from vision.identify import Identifier
from vision.levelup import BUTTON_CENTER, LevelUpScreen
from vision.pipeline import classify_board
from vision.satiety import SatietyReader
from vision.slime import SlimeVatReader

DEVOURER_XY = (642, 923)   # calibrated mouth orifice centroid (was 985 — 62px too low, feeds dropped)
REVIEW_DIR = Path("assets/review")
IDENTIFY_PER_STEP = 2        # popup-tap+OCR ~4s each; don't stall the loop
KNOWN_DIP_MIN = 0.45         # best-match at/above this = known item in a bad
                             # bob phase, NOT a new item (fold into its bank)
STATION_PREFIXES = ("grave", "necromerger", "manapool", "manapot")
# NecroMerger cell is geometry-derived (top-right); never popup-tapped.

# A merge can silently no-op (500ms swipe was too fast to register as a
# drag-pickup — Aug 11, board fully static across merge attempts; Aug 12 the
# 1000ms swipe got flaky again under emulator CPU load). Verify the
# gesture took effect and retry once with a slower swipe before accepting.
MERGE_VERIFY_SWIPE_MS = 1750
# How long to wait after the merge gesture before the verify screencap. The
# merge animation takes a moment: an early frame shows the source cell emptied
# but the target still holding the pre-merge sprite (Aug 11 soak: icerune_lvl2
# merge at (1,1)+(3,0) was logged merge_noop while the board actually merged to
# icerune_lvl3). Without this settle, _verify_merge false-alarms on in-flight
# merges and wastes a retry swipe. 1600ms: under load 1200ms was still catching
# mid-animation frames.
MERGE_SETTLE_MS = 1600

# Repeated gesture failures mean the emulator's input system is wedged (CPU
# starvation — the LLM server and the emulator share the M1 Pro) or the game
# exited. After this many consecutive failures, attempt device recovery
# (relaunch + tap through the title screen) instead of retrying forever.
GESTURE_FAIL_LIMIT = 5
# Green continue block on the dark title screen a cold-started app lands on
# (measured live: block ~y1246-1741, centroid ~(654,1570)).
TITLE_CONTINUE_XY = (654, 1570)


def _rotate_logs(chat_path: Path, max_bytes: int = 8 * 1024 * 1024) -> None:
    """Rotate run logs at startup so they can't grow unbounded.

    session.jsonl and llm_chats.jsonl append across runs (observed 3.8MB+
    chat logs slowing the summary to a 240s timeout). When either exceeds
    max_bytes, move it to .bak (single backup, overwritten) and start
    fresh. In-memory history (SessionLog.events) is unaffected — only the
    on-disk files rotate, and rotation happens before any logging.
    """
    for p in (Path("session.jsonl"), Path(chat_path)):
        try:
            if p.exists() and p.stat().st_size > max_bytes:
                bak = p.with_suffix(p.suffix + ".bak")
                if bak.exists():
                    bak.unlink()
                p.rename(bak)
                print(f"  [rotated {p} ({max_bytes // 1048576}MB+) -> {bak}]")
        except OSError as exc:
            print(f"  [log rotation skipped for {p}: {exc}]")


def _recover_device(device: Device, log: SessionLog) -> bool:
    """Bring the game back to the lair after repeated gesture failures.

    Relaunch NecroMerger via am start (safe whether or not it's still running),
    then tap the title-screen continue block until the lair's floor blob
    returns. Returns True if the lair is showing again, False if recovery failed
    (caller halts)."""
    log.log("device_recovery", stage="relaunch")
    print("  [device recovery — relaunching NecroMerger]")
    device.relaunch()
    for _ in range(10):
        if device.top_resumed_activity() == Device.NECROMERGER_PACKAGE:
            break
        time.sleep(1)
    for _ in range(3):
        device.wait_for_idle(1.5)
        try:
            device.screencap()
            frame = cv2.imread(str(device.screencap_path))
            if frame is None:
                continue
            _blob_bbox(frame)  # floor blob present -> the lair is showing
            log.log("device_recovery", stage="lair_restored")
            return True
        except (NoFloorBlobError, subprocess.CalledProcessError):
            try:
                device.tap(*TITLE_CONTINUE_XY)  # title screen -> continue to lair
            except subprocess.CalledProcessError:
                pass
    log.log("device_recovery", stage="failed")
    return False


def _verify_spawn(device, move, board, log, classifier):
    """After a spawn tap, screencap + reclassify. A spawn produces a new
    item (new occupied cell, or a changed cell where it landed), so an
    identical board means the tap silently no-oped (drained station, empty
    mana the validator misread, or a missed tap). Logs `spawn_noop` with
    the station id for the summary to consolidate — detection only, no
    retry (retrying blind taps on a possibly-drained station wastes uses).
    Returns the (latest) frame. Never raises (all failures log).
    """
    from vision.pipeline import classify_board as _classify
    try:
        device.wait_for_idle(MERGE_SETTLE_MS / 1000.0)
        device.screencap()
        frame = cv2.imread(str(device.screencap_path))
        if frame is None:
            return frame
        before = {(c.row, c.col): c.item_id for c in board.cells if c.occupied}
        after_board = _classify(frame, classifier)
        after = {(c.row, c.col): c.item_id for c in after_board.cells if c.occupied}
        if after == before:
            station = board.cell_at(*move.cell_a)
            log.log("spawn_noop", cell_a=move.cell_a,
                    station=station.item_id if station else None)
            print(f"  [spawn no-op at {move.cell_a} — board unchanged]")
        return frame
    except Exception as exc:
        log.log("spawn_noop", cell_a=move.cell_a,
                error=f"{type(exc).__name__}: {exc}")
        try:
            device.screencap()
            return cv2.imread(str(device.screencap_path))
        except Exception:
            return None


def _verify_merge(device, layout, move, pre_ids, log, classifier, planner):
    """After a merge, screencap + reclassify. If both source cells still hold
    their original items, the merge no-oped: retry once with a slower swipe and
    re-verify. A cell that already emptied while the other still holds the
    original means the merge animation is still in flight — wait for it to
    settle and re-check rather than swiping again. Returns the (latest) frame.
    Every detected no-op is recorded on the planner so the pair stops being
    proposed (merge_noop backoff)."""
    for attempt in (1, 2):
        device.wait_for_idle(MERGE_SETTLE_MS / 1000.0)  # let the merge animation finish
        device.screencap()
        frame = cv2.imread(str(device.screencap_path))
        if frame is None:
            return frame
        board = classify_board(frame, classifier)
        a = board.cell_at(*move.cell_a)
        b = board.cell_at(*move.cell_b)
        a_id = a.item_id if a else None
        b_id = b.item_id if b else None
        if pre_ids[0] is not None and a_id == pre_ids[0] and b_id == pre_ids[0]:
            log.log("merge_noop", cell_a=move.cell_a, cell_b=move.cell_b,
                    attempt=attempt, still=(a_id, b_id))
            if planner is not None:
                planner.record_merge_noop(move.cell_a, move.cell_b)
            if attempt == 1:
                x1, y1 = layout.cell_center(*move.cell_a)
                x2, y2 = layout.cell_center(*move.cell_b)
                try:
                    device.swipe(x1, y1, x2, y2, duration_ms=MERGE_VERIFY_SWIPE_MS)
                except subprocess.CalledProcessError as exc:
                    log.log("merge_noop", cell_a=move.cell_a, cell_b=move.cell_b,
                            attempt=2, still=(a_id, b_id),
                            error=f"retry swipe failed (exit {exc.returncode})")
                    return frame
                continue
            print("  [merge still no-op after retry — identifying cells]")
            _identify_noop_cells(board, frame, move, pre_ids, log,
                                 classifier, planner)
        return frame
    return frame


def _identify_noop_cells(board, frame, move, pre_ids, log, classifier, planner) -> None:
    """Self-improvement on a confirmed merge no-op: popup-identify both cells.

    A no-op means at least one label was wrong (same-id pair the game
    refuses) or both are an unknown max level. Either way the bank is
    missing truth: popup-read both cells, bank the sprite templates under
    the true ids plus the popup recipes (feed/damage/max-level), and log
    `noop_identified` with before/after ids so the session summary folds
    it into learnings/glossary. Next time these sprites classify correctly
    (mislabeled pair never proposed) or gate correctly (newly-known max
    level refuses the merge up front). Bounded to 2 taps, only on confirmed
    no-ops (post-retry), vision-drive only (owns `discover`); all failures
    log and never stall the loop.
    """
    discover = getattr(planner, "discover", None) if planner is not None else None
    if discover is None or board is None or frame is None:
        return
    classifier = getattr(planner, "classifier", None)
    known_families = set()
    if classifier is not None:
        try:
            known_families = {t.split("_lvl")[0]
                              for t in classifier.templates}
        except Exception:
            known_families = set()
    for pos, pre in ((move.cell_a, pre_ids[0]), (move.cell_b, pre_ids[1])):
        try:
            target = board.cell_at(*pos)
            if target is None:
                continue
            item_id, info = discover(target, frame)
            if not item_id:
                continue
            info = info or {}
            if info.get("error"):
                log.log("noop_identify_failed", cell=list(pos),
                        error=info.get("error"))
                continue
            # Trust gate: a noop cell already had a confident label, so a
            # freshly-minted UNKNOWN base (OCR fragment like "be_lvl2",
            # observed Sep 5) is evidence of a bad read, not a new item —
            # banking it would poison the bank AND the glossary. Only bank
            # when the id is already known or its base matches a banked
            # family. Genuine new items never reach here (UNID cells can't
            # be proposed for merges).
            base = item_id.split("_lvl")[0]
            trusted = (classifier is not None
                       and (classifier.has(item_id) or base in known_families))
            if not trusted:
                log.log("noop_identify_rejected", cell=list(pos), was=pre,
                        now=item_id,
                        reason="unknown base; refusing to bank")
                print(f"  [noop read rejected: {pos} {pre} -> {item_id} (unknown base)]")
                continue
            if hasattr(planner, "_bank_popup_recipe"):
                try:
                    planner._bank_popup_recipe(item_id, info)
                except Exception:
                    pass
            bank_ok = False
            if classifier is not None:
                try:
                    planner._frame = frame
                    planner._bank_unid_sprite(target, item_id)
                    bank_ok = True
                except Exception:
                    pass
            # Max-level signal: popup says only "Feed to the Devourer."
            # with no merge line — the same ground truth the max-level
            # gate runs on. Banking the recipe above already records it;
            # surfaced here so the event is self-describing.
            desc = (info.get("description") or "").lower()
            merge_info = (info.get("merge_info") or "").lower()
            feed_only = "feed to the devourer" in desc and "merge" not in merge_info
            log.log("noop_identified", cell=list(pos), was=pre,
                    now=item_id, feed_only=feed_only,
                    sprite_banked=bank_ok, sig=info.get("sig"),
                     digit=info.get("digit"), ocr=info.get("ocr"),
                     level=info.get("level"),
                     overruled=info.get("overruled"))
            _note = getattr(planner, "note_identified", None)
            if _note is not None:
                try:
                    _note(tuple(pos), item_id)
                except Exception:
                    pass
            print(f"  [noop learn: {pos} was {pre}, popup says {item_id}]")
        except Exception as exc:
            log.log("noop_identify_failed", cell=list(pos),
                    error=f"{type(exc).__name__}: {exc}")
            continue


def make_planner(name: str, llm_url: str | None = None,
                 llm_log: str | None = None,
                 reasoning: bool = True,
                 vision_with_text: bool = False,
                 log: SessionLog | None = None,
                 classifier: TemplateClassifier | None = None,
                 live: bool = False,
                 vision_tool: bool = True,
                 vision_max_dim: int | None = 1536,
                 wiki_check: bool = True,
                 wiki_tool: bool = True,
                 cravings=None,
                 bottombar=None,
                 panels=None,
                 satiety_reader=None,
                 champions=None,
                 hints=True,
                 shop=None,
                 queue_box=None,
                 slime_vat=None) -> Planner:
    if name == "heuristic":
        return HeuristicPlanner()
    if name == "llm":
        return LLMPlanner(base_url=llm_url or "http://localhost:8080",
                          chat_log_path=Path(llm_log) if llm_log else Path("llm_chats.jsonl"),
                          reasoning=reasoning,
                          chain_map=read_chains(),
                          feed_values=read_feed_values(),
                          damage_values=read_damage_values(),
                          live=live)
    if name == "vision":
        return VisionLLMPlanner(base_url=llm_url or "http://localhost:8080",
                                chat_log_path=Path(llm_log) if llm_log else Path("llm_chats.jsonl"),
                                reasoning=reasoning,
                                with_text=vision_with_text,
                                chain_map=read_chains(),
                                feed_values=read_feed_values(),
                                live=live)
    if name == "vision-drive":
        return VisionDrivenPlanner(base_url=llm_url or "http://localhost:8080",
                                   chat_log_path=Path(llm_log) if llm_log else Path("llm_chats.jsonl"),
                                   reasoning=reasoning,
                                   log=log,
                                   classifier=classifier,
                                   live=live,
                                   tool_enabled=vision_tool,
                                   vision_max_dim=vision_max_dim,
                                    wiki_check=wiki_check,
                                    wiki_tool=wiki_tool,
                                     cravings=cravings,
                                     bottombar=bottombar,
                                     panels=panels,
                                      satiety_reader=satiety_reader,
                                      slime_vat=slime_vat,
                                      champions=champions,
                                      hints=hints,
                                       shop=shop,
                                       queue_box=queue_box)
    raise ValueError(f"unknown planner: {name}")


def _matches_seed(classifier, item_id: str, crop, floor: float = 0.20) -> bool:
    """True when `crop` resembles item_id's hand-verified seed template.

    Matches the seed (`<id>__0.png`, the calibration capture) inside the
    cell crop — the same direction the classifier scores. True bob-phase
    variants correlate moderately; wrong-level sprites score ~0.08
    (observed Sep 5). Missing seed/unreadable crop -> True (never block
    on missing data).
    """
    try:
        seed = cv2.imread(str(Path("assets/templates") / f"{item_id}__0.png"),
                          cv2.IMREAD_GRAYSCALE)
        if seed is None or crop is None or getattr(crop, "size", 0) == 0:
            return True
        g = crop if len(crop.shape) == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if seed.shape[0] > g.shape[0] or seed.shape[1] > g.shape[1]:
            return True
        return float(cv2.matchTemplate(g, seed, cv2.TM_CCOEFF_NORMED).max()) >= floor
    except Exception:
        return True


def capture_unknowns(frame, board, classifier, counts, log=None) -> tuple[list, list]:
    """Classify below-threshold occupied cells into two buckets.

    Returns (to_identify, accumulated):
    - `to_identify`: genuinely unknown cells (low best-match, not a known
      station) — candidates for popup-tap identification. Crop saved to
      assets/review/ so the bank can grow offline too.
    - `accumulated`: known items in a bad idle-bob phase. Rather than re-mint
      `__alt` ids (which fragments merges), fold the phase into the existing
      template's multi-frame bank — the same hardening as hand-collected
      8-frame templates.
    Stations (grave/necromerger/manapool/manapot) are never popup-tapped:
    they're known, and tapping has side effects. The necromerger is fixed at
    NECROMERGER_CELL regardless of what the classifier reports there.
    """
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    to_identify, accumulated = [], []
    for cell in board.cells:
        best_id, score = classifier.score_cell(frame, cell)
        if score >= classifier.threshold:
            continue
        if occupancy_score(frame, cell.row, cell.col) < OCCUPANCY_THRESHOLD:
            continue
        key = (cell.row, cell.col)
        counts[key] = counts.get(key, 0) + 1
        if (cell.row, cell.col) == necromerger_cell():
            continue  # fixed station, never identify or accumulate
        if best_id and best_id.startswith(STATION_PREFIXES):
            continue  # known station dipping; ignore, planner hardcodes it
        if best_id and score >= KNOWN_DIP_MIN:
            # Seed guard: only fold the dip phase into the bank when the crop
            # still resembles the id's hand-verified seed template (__0).
            # Without this, a mislabeled cell banks its (wrong-level) sprite
            # under the guessed id, and later dips match the pollutant and
            # bank more of them — self-reinforcing level confusion (observed
            # Sep 5: fourteen lvl6 sprites in the lvl5 bank, all ~0.08 vs the
            # lvl5 seed while matching lvl6 at 0.9+). Floor is permissive on
            # purpose (true bob phases correlate moderately); rejections log
            # for calibration.
            if _matches_seed(classifier, best_id,
                             crop_cell(frame, cell.row, cell.col)):
                classifier.add_template(best_id, crop_cell(frame, cell.row, cell.col))
                accumulated.append((cell.row, cell.col, best_id, round(score, 3), counts[key]))
            else:
                if log is not None:
                    log.log("accumulate_seed_rejected", cell=(cell.row, cell.col),
                            best=best_id, score=round(score, 3))
            continue
        path = REVIEW_DIR / f"cell_{cell.row}_{cell.col}__{counts[key]}.png"
        cv2.imwrite(str(path), crop_cell(frame, cell.row, cell.col))
        to_identify.append((cell.row, cell.col, best_id, round(score, 3), counts[key]))
    return to_identify, accumulated


def main():
    parser = argparse.ArgumentParser(description="NecroMerger autonomous agent")
    parser.add_argument("--dry-run", action="store_true", help="print moves, no gestures")
    parser.add_argument("--planner", choices=["heuristic", "llm", "vision", "vision-drive"],
                        default="vision-drive",
                        help="heuristic is DEPRECATED as a primary planner (it remains the "
                             "validator/fallback engine inside the LLM planners); vision-drive is "
                             "the intended brain")
    parser.add_argument("--vision-with-text", action="store_true",
                        help="--planner vision: append the text board grid to the image observation (A/B)")
    parser.add_argument("--llm-url", default="http://localhost:8080",
                        help="llama-server base URL for --planner llm")
    parser.add_argument("--llm-log", default=None,
                        help="log full LLM prompt+reply exchanges here (default: llm_chats.jsonl)")
    parser.add_argument("--no-llm-reasoning", action="store_true",
                        help="disable LLM reasoning (faster, uses JSON prefill)")
    parser.add_argument("--no-vision-tool", action="store_true",
                        help="vision-drive: skip the get_board_state tool (raw pixels only, A/B)")
    parser.add_argument("--vision-max-dim", type=int, default=1536,
                        help="vision-drive screenshot long-side cap (default 1536 = ~half res; "
                             "set 0/None for full 1280x2856)")
    parser.add_argument("--summarize-every", type=int, default=5,
                        help="vision-drive: run memory maintenance every N steps (deterministic folds; full LLM pattern-mining every 10th + session end; 0 = only at session end)")
    parser.add_argument("--no-wiki-factcheck", action="store_true",
                        help="vision-drive: skip wiki fact-checking of candidate learnings in summarize")
    parser.add_argument("--no-wiki-tool", action="store_true",
                        help="vision-drive: don't offer the lookup_wiki tool during play")
    parser.add_argument("--auto-drain-every", type=int, default=0,
                        help="(deprecated: chests are now spawn stations — tap them via a normal spawn move; "
                             "this flag is a no-op kept for compat; the model's collect_queue tool remains for queue placement)")
    parser.add_argument("--no-hints", action="store_true",
                        help="vision-drive: omit the computed Best merge/feed/spawn hint lines "
                             "from get_board_state — the model must reason from feats/state "
                             "instead of echoing the code's optimum (A/B experiment)")
    parser.add_argument("--steps", type=int, default=None,
                        help="max steps (default: run forever; 1 with --screenshot)")
    parser.add_argument("--screenshot", type=Path, default=None,
                        help="static image mode (no adb): replay this frame")
    parser.add_argument("--seed-templates", type=Path, default=None,
                        help="start from a pre-banked template dir (default: seedless "
                             "autoboot — items are popup-identified and auto-banked "
                             "at runtime into assets/templates)")
    args = parser.parse_args()

    if args.screenshot and args.steps is None:
        args.steps = 1

    device = Device()
    seed_dir = str(args.seed_templates) if args.seed_templates else None
    templates_dir = seed_dir or "assets/templates"
    classifier = TemplateClassifier(templates_dir, seed=True)
    levelup = LevelUpScreen()
    _rotate_logs(Path(args.llm_log) if args.llm_log else Path("llm_chats.jsonl"))
    log = SessionLog()
    live = not args.dry_run and args.screenshot is None
    # Planner brain: --screenshot is a frozen OFFLINE regression check (no
    # LLM, deterministic fallback decision); plain/--dry-run runs think with
    # the full vision-drive LLM (dry-run just never gestures or tap-reads).
    planner_online = args.screenshot is None
    llm_client = None
    cravings = None
    champions = None
    if live:
        from planner.llm_client import LLMClient
        llm_client = LLMClient(base_url=args.llm_url or "http://localhost:8080")
        from vision.cravings import CravingsReader
        cravings = CravingsReader(device=device, llm=llm_client)
        from vision.champion import ChampionReader
        champions = ChampionReader(device=device, llm=llm_client)
        from vision.bottombar import BottomBarReader
        bottombar = BottomBarReader()
        from vision.panels import PanelReader
        panels = PanelReader(device=device, llm=llm_client, bottombar=bottombar)
        satiety_reader = SatietyReader()
        slime_vat = SlimeVatReader()
        from vision.panels import StationShop
        shop = StationShop(device=device, llm=llm_client,
                           bottombar=bottombar, classifier=classifier)
        from vision.queue_box import QueueBox
        queue_box = QueueBox(device=device, classifier=classifier,
                             bottombar=bottombar)
    else:
        bottombar = None
        panels = None
        satiety_reader = None
        slime_vat = None
        shop = None
        queue_box = None
    try:
        planner = make_planner(args.planner, args.llm_url, args.llm_log,
                               reasoning=not args.no_llm_reasoning,
                               vision_with_text=args.vision_with_text,
                               log=log,
                               classifier=classifier,
                               live=planner_online,
                               vision_tool=not args.no_vision_tool,
                               vision_max_dim=args.vision_max_dim,
                               wiki_check=not args.no_wiki_factcheck,
                               wiki_tool=not args.no_wiki_tool,
                               hints=not args.no_hints,
                               cravings=cravings,
                               bottombar=bottombar,
                               panels=panels,
                               satiety_reader=satiety_reader,
                                slime_vat=slime_vat,
                               champions=champions,
                               shop=shop,
                               queue_box=queue_box)
        # Wiring visibility: a silently-missed call-site argument (queue_box
        # was once dropped exactly this way) disables a whole tool class with
        # no error — print the backends so every soak log shows them.
        print("  [tool backends] " + " ".join(
            f"{n}={'on' if v else 'OFF'}" for n, v in
            (("cravings", cravings), ("panels", panels),
             ("champions", champions), ("shop", shop),
             ("queue", queue_box))))
    except LLMError as exc:
        raise SystemExit(str(exc))
    unknown_counts: dict[tuple[int, int], int] = {}
    prev_board = None   # last step's classified board (board evolution memory)
    prev_move = None    # move executed into the current board (explains diffs)
    summarize_every = args.summarize_every if live else 0
    identifier = None

    # Board geometry: the layout changes with game state (board grows as the
    # Devourer levels), so the 5x3 constants are only a fallback. Read it once
    # from the live screen: pinned config dims first (no LLM round — the grid
    # only changes on board-expansion feat rewards), else the vision LLM
    # (validated + refined against template anchors); fall back to the
    # calibrated constants on any failure. A first successful detect
    # self-seeds the config for future runs.
    geometry = grid.FALLBACK_GEOMETRY
    if live:
        from vision.geometry import (NoFloorBlobError, llm_grid_geometry,
                                      read_board_dims, write_board_dims)
        pinned = read_board_dims()
        if pinned is not None:
            print(f"  [geometry: pinned dims {pinned[0]}x{pinned[1]} "
                  f"(no LLM round)]")
        max_attempts = 5
        attempts = 0
        while attempts < max_attempts:
            try:
                device.screencap()
                geom_frame = cv2.imread(str(device.screencap_path))
                if geom_frame is not None:
                    geometry = llm_grid_geometry(
                        geom_frame, None if pinned else llm_client,
                        classifier, prior_dims=pinned)
                    grid.set_grid_geometry(geometry)
                    print(f"  [geometry read successful on attempt {attempts + 1}]")
                    if (geometry.rows, geometry.cols) != pinned:
                        if write_board_dims(geometry.rows, geometry.cols):
                            print(f"  [geometry: config (re)seeded "
                                  f"{geometry.rows}x{geometry.cols}]")
                    break
                else:
                    raise RuntimeError("Unable to read screenshot for geometry read.")
            except (LLMError, ValueError, NoFloorBlobError, RuntimeError) as exc:
                attempts += 1
                print(f"  [geometry read failed ({exc.__class__.__name__}) on attempt {attempts}. "
                      f"Retrying geometry read...]")
                if attempts >= max_attempts:
                    print("  [max attempts reached for geometry read. Using calibrated constants.]")
                    break
        else:
            print("  [geometry read failed after all attempts. Using calibrated constants.]")

    layout = Layout(devourer_xy=(geometry.mouth_x, geometry.mouth_y), geom=geometry)
    if live:
        if isinstance(planner, VisionDrivenPlanner):
            # Vision-drive owns discovery: the model calls identify_item, which
            # names items via the LLM (OCR as fallback). Skip the OCR loop below.
            llm_ident = Identifier(device, classifier, llm=planner.client)
            planner.discover = lambda cell, frame: llm_ident.identify(frame, cell)
            planner.popup_reader = lambda frame, cell: llm_ident.read_recipe(frame, cell)
        else:
            identifier = Identifier(device, classifier)

    step = 0
    last_auto_drain = -1000
    auto_drain_every = max(0, args.auto_drain_every)
    gesture_fails = 0
    try:
        while args.steps is None or step < args.steps:
            # tick the dock reader so its runtime detection
            # cadence (DOCK_DETECT_STEPS) refreshes.
            if bottombar is not None:
                bottombar.set_step(step)
            if args.screenshot:
                frame = cv2.imread(str(args.screenshot))
                if frame is None:
                    raise RuntimeError(f"Unable to read screenshot: {args.screenshot}")
            else:
                try:
                    device.screencap()
                except subprocess.CalledProcessError:
                    raise SystemExit("adb screencap failed — is the emulator running? "
                                     "start it with: emulator -avd NecroMerger_PS")
                frame = cv2.imread(str(device.screencap_path))
                if frame is None:
                    raise RuntimeError(f"Unable to read screenshot: {device.screencap_path}")

            # Lair-frame guard: every plan below assumes `frame` is the bare
            # lair board. A menu/panel/popup frame has no floor blob and reads
            # as a phantom empty board (13/15 empty + no Bottom bar + geometry
            # silently fell back) — the class of bug that caused the empty-board
            # episode. Check the floor blob first; dismiss the only screen we
            # know how to dismiss safely (level-up), then halt rather than play
            # blind taps on a dead screen.
            if live:
                try:
                    _blob_bbox(frame)
                    is_lair = True
                except NoFloorBlobError:
                    is_lair = False
                except Exception as exc:
                    print(f"  [WARNING: floor-blob check failed "
                          f"({exc.__class__.__name__}); treating frame as non-lair]")
                    is_lair = False
                if not is_lair:
                    lvl_present, _lvl_score = levelup.is_level_up(frame)
                    if lvl_present and levelup.dismiss(device):
                        log.log("level_up_screen", recovered=True)
                        print("  [recovered from level-up screen to the lair]")
                        device.screencap()
                        frame = cv2.imread(str(device.screencap_path))
                    else:
                        log.log("lair_frame_guard", ok=False)
                        print("  [WARNING: frame is not the lair board and no "
                              "known recovery applied — halting]")
                        raise BoardLostError(
                            "frame was not the lair board (no floor blob); "
                            "a panel/popup is probably covering it")

            # The Devourer levels up when a feed fills its satiety bar: a
            # level-up screen pops with a green "Continue" button that must be
            # tapped to return to the lair. Detect + dismiss it before any
            # planning so the board state below is always the lair.
            if live:
                lvl_present, lvl_score = levelup.is_level_up(frame)
                if lvl_present:
                    log.log("level_up_screen", button=BUTTON_CENTER,
                            score=round(lvl_score, 3))
                    print("  [level-up screen — tapping Continue]")
                    if levelup.dismiss(device):
                        log.log("level_up_dismissed")
                        # A Devourer level-up almost always means the current
                        # craving just completed. Drop any cached craving read
                        # so the planner re-reads get_cravings next step and
                        # learns the NEW craving instead of believing the old
                        # one (e.g. "skeleton 1/2") for the cache lifetime.
                        planner.invalidate_cravings()
                        device.screencap()
                        frame = cv2.imread(str(device.screencap_path))
                    else:
                        print("  [level-up screen did not clear — retrying next step]")
                        continue

            board = classify_board(frame, classifier)
            # Popup-label persistence: a popup-resolved id (e.g.
            # eyemonster_lvl1) overrules a template flip-flop inside a
            # template-indistinguishable pair until the cell genuinely
            # changes. Applied BEFORE unknown-capture so persisted cells
            # don't re-trigger identify taps. Planner-agnostic (no-op for
            # planners without the memory); never stalls the loop.
            try:
                _apply_mem = getattr(planner, "apply_label_memory", None)
                if _apply_mem is not None:
                    _n_mem = _apply_mem(board, prev_move)
                    if _n_mem:
                        print(f"  [label memory: {int(_n_mem)} cell(s) restored]")
            except Exception as exc:
                log.log("label_memory", error=f"{type(exc).__name__}: {exc}")
            # Board evolution memory: diff against last step unexplained by
            # the move that produced this board. Game-side changes (champion
            # spawn, reward arrival, board growth) surface here for the
            # summary to learn from; our own moves are filtered by kind.
            if prev_board is not None and prev_move is not None:
                try:
                    from vision.grid import diff_boards
                    _bd = diff_boards(prev_board, board, prev_move)
                    if _bd["appeared"] or _bd["vanished"] or _bd["moved"]:
                        log.log("board_diff", appeared=_bd["appeared"][:6],
                                vanished=_bd["vanished"][:6], moved=_bd["moved"][:4],
                                after=prev_move.kind)
                except Exception as exc:
                    log.log("board_diff", error=f"{type(exc).__name__}: {exc}")
            unknown, accumulated = capture_unknowns(frame, board, classifier, unknown_counts, log)
            for row, col, best_id, score, n in accumulated:
                log.log("accumulate", cell=(row, col), best=best_id, score=score, frame=n)
            for row, col, best_id, score, n in unknown:
                log.log("discover_unknown", cell=(row, col), best=best_id, score=score, frame=n)

            # Chests are now spawn stations (tapped via a normal `spawn`
            # move, mana-free, one rune per tap) — no auto-drain. The queue
            # tool remains for dock-queue placement only.

            if identifier:
                identified = 0
                for row, col, _best, _score, _n in unknown:
                    if identified >= IDENTIFY_PER_STEP:
                        break
                    cell = board.cell_at(row, col)
                    if cell is None or cell.item_id is not None:
                        continue
                    try:
                        item_id, info = identifier.identify(frame, cell)
                    except Exception as exc:
                        log.log("identify", cell=(row, col), error=str(exc))
                        continue
                    cell.item_id = item_id
                    cell.score = 1.0
                    cell.margin = 1.0
                    _note = getattr(planner, "note_identified", None)
                    if _note is not None:
                        try:
                            _note((row, col), item_id)
                        except Exception:
                            pass
                    log.log("identify", cell=(row, col), item_id=item_id, **info)
                    print(f"  identified ({row},{col}) -> {item_id}  {info}")
                    identified += 1

            move = planner.next_move(board, frame)
            log.log(move.kind, cell_a=move.cell_a, cell_b=move.cell_b,
                    target=move.target)
            discover = ""
            if unknown:
                discover = "  [discover: " + "; ".join(
                    f"({r},{c})={best}:{score}" for r, c, best, score, _ in unknown) + "]"
            if args.dry_run:
                items = [(c.row, c.col, c.item_id) for c in board.cells if c.item_id]
                print(f"[{step}] {move}  items={items}{discover}")
            else:
                print(f"[{step}] {move}{discover}")
                ca = board.cell_at(*move.cell_a) if move.cell_a else None
                cb = board.cell_at(*move.cell_b) if move.cell_b else None
                pre_ids = (ca.item_id if ca else None,
                           cb.item_id if cb else None)
                try:
                    execute(device, layout, move)
                except GestureFailed as exc:
                    log.log("gesture_failed", kind=move.kind,
                            cell_a=move.cell_a, cell_b=move.cell_b,
                            code=exc.code, stderr=exc.stderr)
                    print(f"  [WARNING: {exc} — retrying next step]")
                    gesture_fails += 1
                    if gesture_fails >= GESTURE_FAIL_LIMIT:
                        if not _recover_device(device, log):
                            raise BoardLostError(
                                "device recovery failed after repeated gesture "
                                "errors — relaunch the emulator/game and resume")
                        gesture_fails = 0
                    continue
                gesture_fails = 0
                if move.kind == "merge":
                    frame = _verify_merge(device, layout, move, pre_ids, log, classifier, planner)
                elif move.kind == "spawn":
                    frame = _verify_spawn(device, move, board, log, classifier)
                device.wait_for_idle()
                prev_board, prev_move = board, move
            step += 1
            if (live and summarize_every and step % summarize_every == 0
                    and isinstance(planner, VisionDrivenPlanner)):
                try:
                    if planner.maintain_memory(board, frame):
                        print(f"  [learnings updated @ step {step}]")
                except Exception as exc:
                    print(f"  learnings update failed: {exc}")
    except BoardLostError:
        print("\nSTOPPING: the game is no longer showing the board "
              "(a BACK likely exited it). Relaunch the game and resume.")
    except KeyboardInterrupt:
        print("\nsession interrupted")
    finally:
        # Flush any trailing un-summarized events (skips when the last periodic
        # update already covered them).
        if live and isinstance(planner, VisionDrivenPlanner) and step > 0:
            try:
                learnings = planner.summarize_session(board, frame)
                if learnings:
                    print("\n--- learnings written to learnings.md ---")
                    print(learnings)
            except Exception as exc:
                print(f"session summarization failed: {exc}")


if __name__ == "__main__":
    main()
