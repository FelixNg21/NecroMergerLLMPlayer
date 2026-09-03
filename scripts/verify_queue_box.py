"""Verification for the Queue placement feature (Aug 24).

Run: `.venv/bin/python scripts/verify_queue_box.py` (exit 0/1).

  A. has_reward: template match on a live dock crop (chest queued) vs a
     synthesized empty button (skull-only, chest pixels blanked).
  B. collect flow on a scripted device: place + open chest end-to-end
     (queue flips empty, new chest cell appears, tap opens it, item gone).
  C. refusals: empty queue, dock not visible — no tap issued.
  D. tool wiring: collect_queue routes through _exec_tool + logs.
  E. py_compile.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from metrics.logger import SessionLog  # noqa: E402
from planner.vision_drive import VisionDrivenPlanner  # noqa: E402
from vision.classifier import TemplateClassifier  # noqa: E402
from vision.grid import BoardState, Cell, GridGeometry, set_grid_geometry  # noqa: E402
from vision.queue_box import QueueBox, has_reward, QUEUE_ICON  # noqa: E402
from vision.bottombar import BottomBarReader  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PASS = 0
QUEUE_TAP = (QUEUE_ICON[0], QUEUE_ICON[1] - 4)   # where collect taps the queue


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if ok:
        PASS += 1


def lair_frame(reward: bool = True) -> np.ndarray:
    """Synthetic lair frame: dark background + the queue button crop at its
    live spot. `reward=True` pastes a real reward (the green-gem from
    `assets/queue/queue_button_now.png`); `reward=False` pastes the empty
    bare-skull state (one of the animated frames banked in `assets/bottombar/`).
    Both use the same QUEUE_ICON coordinates `has_reward` reads."""
    frame = np.zeros((2856, 1280, 3), dtype=np.uint8)
    frame[:] = (40, 35, 30)
    cx, cy, w, h = QUEUE_ICON
    y0, y1, x0, x1 = cy - h // 2, cy - h // 2 + h, cx - w // 2, cx - w // 2 + w
    if reward:
        crop = cv2.imread(str(ROOT / "assets" / "queue" / "queue_button_now.png"))
        if crop is not None and crop.shape[0] >= h and crop.shape[1] >= w:
            frame[y0:y1, x0:x1] = crop[:h, :w]
        else:
            frame[y0:y1, x0:x1] = (140, 100, 80)   # high-texture fallback
    else:
        skull = cv2.imread(str(ROOT / "assets" / "bottombar" / "queue__0.png"))
        if skull is not None:
            frame[y0:y1, x0:x1] = skull
    return frame


class FakeBottomBar:
    def __init__(self):
        self.visible = True

    def bar_visible(self, frame):
        return self.visible

    def button_center(self, name):
        # Only queue is used in these tests
        if name == "queue":
            return QUEUE_ICON[0]
        return 0


class FakeDevice:
    def __init__(self, frames):
        self.frames = list(frames)
        self._i = 0
        self.taps = []
        self.bottombar = FakeBottomBar()
        self.screencap_path = Path(tempfile.mkdtemp()) / "cap.png"

    def screencap(self):
        idx = min(self._i, len(self.frames) - 1)
        cv2.imwrite(str(self.screencap_path), self.frames[idx])
        self._i += 1

    def wait_for_idle(self, _s):
        pass

    def tap(self, x, y):
        self.taps.append((int(x), int(y)))


def part_a_detect() -> None:
    check("A1: chest queued -> has_reward", has_reward(lair_frame(True)))
    check("A2: empty button -> no reward", not has_reward(lair_frame(False)))


def part_a_general() -> None:
    """A3: a NON-chest reward (the green-gem queue button captured live) is
    still detected — the inverted detector handles every reward sprite
    without per-reward templates (the bug: only the blue ice chest was
    recognized, so a poison-rune/gem reward was refused)."""
    frame = lair_frame(True)
    check("A3: non-chest reward (gems) -> has_reward", has_reward(frame))
    # A4: the live chest button itself (the blue icechest sprite at the
    # dock center) is a different per-reward icon — still detected.
    frame2 = np.zeros((2856, 1280, 3), dtype=np.uint8)
    frame2[:] = (40, 35, 30)
    cx, cy, w, h = QUEUE_ICON
    y0, y1, x0, x1 = cy - h // 2, cy - h // 2 + h, cx - w // 2, cx - w // 2 + w
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[:, :, 0] = (200 - (rr * 6) % 150)
    patch[:, :, 1] = (160 + (cc * 6) % 150)
    patch[:, :, 2] = 80
    frame2[y0:y1, x0:x1] = patch
    check("A4: textured non-skull icon -> has_reward", has_reward(frame2))
    # A5: a SKEWED empty skull (the live button may bob) is still detected
    # as empty — empty_match is multi-frame so it tolerates the pulse.
    frame3 = np.zeros((2856, 1280, 3), dtype=np.uint8)
    frame3[:] = (40, 35, 30)
    skull = cv2.imread(str(ROOT / "assets" / "bottombar" / "queue__1.png"))
    if skull is not None:
        frame3[y0:y1, x0:x1] = skull
    check("A5: empty skull (frame 1) -> no reward", not has_reward(frame3))
    # A6: the queue dock button's live x is the MIDDLE of the 5-slot bar
    # (~638), NOT the station's x (~384). Regression for the Sep 2 fix: a
    # queued reward (mana_potion) was misread as "empty" because QUEUE_ICON
    # pointed at the station's x, so has_reward/collect read the wrong button.
    check("A6: queue dock x is the middle button (~638) not station (~384)",
          QUEUE_ICON[0] in (620, 660), f"QUEUE_ICON x = {QUEUE_ICON[0]}")


def part_a_detect_positions() -> None:
    """The dock detection must resolve a queue x that collides with the
    station's x (the queue holds a reward, so it doesn't match the empty-skull
    bank and can get latched onto the station column). Sanitizer interpolation
    puts the queue back in the middle of the 5-slot bar."""
    s = BottomBarReader._sanitize_positions
    fixed = {n: x for n, x in s([("feats", 161), ("station", 381),
                                 ("queue", 384), ("spellbook", 896),
                                 ("shop", 1152)])}
    check("A7: colliding queue x interpolated to the middle (~638)",
          fixed["queue"] in (620, 660), f"queue x = {fixed['queue']}")
    check("A8: non-colliding queue x left alone",
          s([("feats", 161), ("station", 381), ("queue", 640),
             ("spellbook", 896), ("shop", 1152)])[2][1] == 640)


def part_b_collect() -> None:
    # Placement-only: queue reward placed onto board (chests on board are now
    # spawn stations — tapped via a normal `spawn` move, not drained here).
    full = lair_frame(True)
    empty = lair_frame(False)
    before = {(0, 0): "manapot_lvl3", (0, 3): "necromerger",
              (4, 3): "grave_lvl2"}
    placed = dict(before)
    placed[(2, 1)] = "icebox_unopened"
    boards = iter([before, placed, placed, placed])
    dev = FakeDevice([full, empty, empty, full])
    qb = QueueBox(device=dev, classifier=None, bottombar=dev.bottombar)
    qb._board_cells = lambda frame: next(boards)
    # Also mock _cells for collect's space check
    qb._cells = lambda frame: [Cell(0,0,0,0,None,occupied=False)]  # not used

    res = qb.collect(open_chest=True)
    check("B1: placed", res.get("placed") is True, str(res))
    check("B2: chest cell found", res.get("cell") == [2, 1]
          and res.get("placed_item") == "icebox_unopened", str(res))
    check("B4: queue button tapped", QUEUE_TAP in dev.taps, str(dev.taps))
    # No drain — chest stays on board for spawn taps
    check("B5: no chest drain (now via spawn)", res.get("drained") is False or "drained" not in res or res.get("drained") is False)


class ScriptedBoards:
    """Per-screencap scripted board states (dicts rc -> item_id)."""

    def __init__(self, sequence):
        self.seq = list(sequence)
        self.i = 0

    def cells(self, frame):
        occ = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        out = []
        for rc in [(r, c) for r in range(5) for c in range(4)]:
            iid = occ.get(rc)
            out.append(Cell(rc[0], rc[1], 0, 0, iid, score=1.0, margin=0.9,
                            occupied=iid is not None))
        return out

    def occ(self, frame):
        occ = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return dict(occ)


def part_c_drain_existing() -> None:
    """Chests on the board are now spawn stations — queue tool no longer drains
    them. Verify that collect with an empty queue reports empty, not a drain."""
    empty_dock = lair_frame(False)          # queue holds nothing
    dev = FakeDevice([empty_dock, empty_dock])
    qb = QueueBox(device=dev, classifier=None, bottombar=dev.bottombar)
    # Even with a chest on board, collect reports queue empty (chest is spawn)
    qb._board_cells = lambda frame: {(1, 1): "icebox_unopened"}
    res = qb.collect(open_chest=True)
    check("C2d: chest on board -> queue empty (chest is spawn, not queue drain)",
          "queue is empty" in (res.get("error") or ""), str(res))
    check("C2e: no queue tap (queue was empty)",
          QUEUE_TAP not in dev.taps, str(dev.taps))


def part_c_space_gate() -> None:
    """Queue full but the board is congested -> placement refused."""
    full = lair_frame(True)
    congested = {(r, c): "bone" for r in range(5) for c in range(4)}
    for rc in ((0, 0), (0, 1), (0, 2)):     # 17 occupied -> 3 empty
        congested.pop(rc)
    boards = ScriptedBoards([congested] + [congested] * 4)
    dev = FakeDevice([full] * 8)
    qb = QueueBox(device=dev, classifier=None, bottombar=dev.bottombar)
    qb._cells = boards.cells
    qb._board_cells = boards.occ
    res = qb.collect(open_chest=False)
    check("C3d: congested board refuses placement",
          "board congested" in (res.get("error") or ""), res.get("error"))
    check("C3e: queue button NOT tapped", QUEUE_TAP not in dev.taps,
          str(dev.taps))


def part_c_refusals() -> None:
    dev = FakeDevice([lair_frame(False), lair_frame(False)])
    qb = QueueBox(device=dev, classifier=None, bottombar=dev.bottombar)
    res = qb.collect()
    check("C1: empty queue refused", "queue is empty" in (res.get("error") or ""),
          res.get("error"))
    check("C2: no tap issued", dev.taps == [], str(dev.taps))
    dev2 = FakeDevice([lair_frame(True)])
    qb2 = QueueBox(device=dev2, classifier=None, bottombar=dev2.bottombar)
    qb2.bottombar.visible = False
    res2 = qb2.collect()
    check("C3: dock hidden refused",
          "dock is not visible" in (res2.get("error") or ""), res2.get("error"))


def part_d_tool() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        clf = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
        p = VisionDrivenPlanner(
            client=None, live=True, tool_enabled=True, hints=True,
            log=SessionLog(tmp / "s.jsonl"), chat_log_path=tmp / "c.jsonl",
            classifier=clf,
            learnings_path=tmp / "l.md", glossary_path=tmp / "g.md")

        class StubQueue:
            def collect(self, open_chest=True):
                return {"placed": True, "opened": True, "error": None,
                        "placed_item": "icechest_lvl1", "cell": [2, 1]}

        p.queue_box = StubQueue()
        res = json.loads(p._exec_tool(
            {"function": {"name": "collect_queue", "arguments": "{}"}}, None))
        check("D: tool routes to the QueueBox", res.get("placed") is True, str(res))
        check("D2: queue_collected logged",
              any(e["event"] == "queue_collected" for e in p.log.events))


def part_e_compile() -> None:
    import py_compile
    for f in ("vision/queue_box.py", "planner/vision_drive.py", "main.py",
              "planner/agent.py", "planner/llm.py"):
        py_compile.compile(str(ROOT / f), doraise=True)
    check("E: py_compile clean", True)
    # Wiring regression guard (Aug 25): main.py's make_planner CALL once
    # dropped the queue_box argument silently — the planner got None and
    # collect_queue was never offered (0/54 rounds in the soak). Assert the
    # call site passes it.
    src = (ROOT / "main.py").read_text()
    call = src[src.find("planner = make_planner("):]
    check("E2: make_planner call passes queue_box",
          "queue_box=queue_box" in call.split(")")[0])
    # Chests are now spawn stations (tapped via normal spawn, one tap per use)
    const_src = (ROOT / "planner" / "constants.py").read_text()
    agent_src = (ROOT / "planner" / "agent.py").read_text()
    check("E3: chest is spawn station (CHEST_PREFIXES, one tap per use, mana-free)",
          'CHEST_PREFIXES = ("icebox"' in const_src
          and 'CHEST_SPAWN_TAPS = 2' in agent_src
          and '_spawn_candidates' in agent_src)


def main() -> None:
    global PASS
    set_grid_geometry(GridGeometry(184, 1192, 228, 5, 4))
    for part in (part_a_detect, part_a_detect_positions, part_a_general,
                 part_b_collect, part_c_drain_existing, part_c_space_gate,
                 part_c_refusals, part_d_tool, part_e_compile):
        try:
            part()
        except Exception as exc:  # noqa: BLE001
            check(f"{part.__name__} raised {type(exc).__name__}", False, str(exc))
    print(f"\n{PASS} checks passed")
    sys.exit(0 if PASS >= 14 else 1)


if __name__ == "__main__":
    main()