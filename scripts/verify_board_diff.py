"""Repeatable verification for board evolution memory (diff_boards).

Run: `.venv/bin/python scripts/verify_board_diff.py` (exit 0/1).

diff_boards(prev, curr, move) reports only UNEXPLAINED changes: the move's
own cells are filtered by kind (merge/feed/spawn/attack), same-cell
relabels from bob-phase noise pair up as vanish+appear at one cell (kept
visible), and genuine motion pairs across cells into `moved`.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.grid import GridGeometry, BoardState, build_cells, diff_boards  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PASS = 0
GEOM = GridGeometry(184, 1202, 228, 5, 4)


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if ok:
        PASS += 1


def _board(occ):
    cells = build_cells(GEOM)
    for c in cells:
        c.occupied = False
        c.item_id = None
        c.score = 0.0
        c.margin = 0.0
    for r, c, iid in occ:
        t = next(x for x in cells if (x.row, x.col) == (r, c))
        t.occupied = True
        t.item_id = iid
        t.score = 0.9
        t.margin = 0.5
    return BoardState(5, 4, cells, geometry=GEOM)


def _mv(kind, a=None, b=None, t=None):
    return Mock(kind=kind, cell_a=a, cell_b=b, target=t)


def part_explained() -> None:
    base = _board([(0, 0, "bone"), (0, 1, "bone"), (2, 2, "grave_lvl1")])
    merged = _board([(0, 0, "ribcage"), (2, 2, "grave_lvl1")])
    d = diff_boards(base, merged, _mv("merge", (0, 0), (0, 1)))
    check("A: merge explains its cells", d == {"appeared": [], "vanished": [], "moved": []}, str(d))
    fed = _board([(2, 2, "grave_lvl1")])
    d = diff_boards(base, fed, _mv("feed", (0, 0)))
    check("B: feed explains its cell (other vanish unexplained)",
          d["vanished"] == [(0, 1, "bone")] and not d["appeared"], str(d))
    spawned = _board([(0, 0, "bone"), (0, 1, "bone"), (2, 2, "grave_lvl1"), (3, 3, "bone")])
    d = diff_boards(base, spawned, _mv("spawn", (2, 2)))
    check("C: spawn landing unexplained-filtered", d["appeared"] == [] and d["vanished"] == [], str(d))


def part_unexplained() -> None:
    base = _board([(0, 0, "ribcage"), (2, 2, "grave_lvl1")])
    champ = _board([(0, 0, "ribcage"), (2, 2, "grave_lvl1"), (1, 1, "peasant")])
    d = diff_boards(base, champ, _mv("idle", None, None))
    check("D: champion appearance reported",
          d["appeared"] == [(1, 1, "peasant")], str(d))
    moved = _board([(1, 1, "ribcage"), (2, 2, "grave_lvl1")])
    d = diff_boards(base, moved, _mv("idle", None, None))
    check("E: motion pairs into moved",
          d["moved"] == [((0, 0), (1, 1), "ribcage")] and not d["appeared"] and not d["vanished"], str(d))
    relabel = _board([(0, 0, "skeleton_lvl2"), (2, 2, "grave_lvl1")])
    d = diff_boards(base, relabel, _mv("idle", None, None))
    check("F: same-cell relabel stays visible (not silently dropped)",
          d["vanished"] == [(0, 0, "ribcage")] and d["appeared"] == [(0, 0, "skeleton_lvl2")], str(d))


def part_heartbeat() -> None:
    """Observe heartbeat: live call on heartbeat steps, direct inject otherwise."""
    from planner.vision_drive import VisionDrivenPlanner, OBSERVE_HEARTBEAT_STEPS
    check("G: heartbeat constant sane",
          OBSERVE_HEARTBEAT_STEPS >= 5, str(OBSERVE_HEARTBEAT_STEPS))
    p = VisionDrivenPlanner.__new__(VisionDrivenPlanner)
    p.tool_enabled = True
    p.client = Mock()
    p.log = Mock()
    p.log.log = Mock()
    p._log_chat = Mock()
    p._board_state_text = Mock(return_value="BOARD")
    p._exec_tool = Mock(return_value="BOARD")
    base = _board([(0, 0, "bone")])
    # non-heartbeat step: no LLM call, synthetic tool message injected
    p._step_count = 2
    msgs: list = []
    p._observe(msgs, base)
    check("H: skip step injects without LLM call",
          p.client.chat_message.call_count == 0
          and [m.get("role") for m in msgs] == ["assistant", "tool"]
          and msgs[1]["content"] == "BOARD",
          str([m.get("role") for m in msgs]))
    # heartbeat step: live call attempted (retry loop may call twice)
    p._step_count = 1
    p.client.chat_message = Mock(return_value=("", None, {}))
    msgs = []
    p._observe(msgs, base)
    check("I: heartbeat step calls the model",
          p.client.chat_message.call_count >= 1, "")


def main() -> None:
    for part in (part_explained, part_unexplained, part_heartbeat):
        try:
            part()
        except Exception as exc:  # noqa: BLE001
            check(f"{part.__name__} raised {type(exc).__name__}", False, str(exc))
    print(f"\n{PASS} checks passed")
    sys.exit(0 if PASS >= 8 else 1)


if __name__ == "__main__":
    main()
