"""Verification for the Aug 19 noop root-cause fixes.

Run: `.venv/bin/python scripts/verify_noop_backoff.py` (exit 0/1).

Covers:
  A. Phantom-label wipe (vision/pipeline.py): a cell BELOW the occupancy
     threshold keeps a label only when its absolute score is ALSO strong.
     On the noop-run fixture (screenshots/noop_run_final.png, 4x3 geometry)
     the empty (3,0) that was labeled skeleton_lvl5@0.61/0.10 is wiped to
     empty while the real items are kept — so the heuristic can no longer
     propose the un-dragable (3,0)+(4,1) merge that no-opped forever.
  B. Merge-noop backoff registry (planner/agent.py): a pair excluded after
     NOOP_BACKOFF_THRESHOLD no-ops, released after NOOP_BACKOFF_STEPS ticks.
  C. ranked_merge_groups exclude_pairs: an excluded pair is skipped (next
     available pair / group dropped when nothing else remains).
  D. HeuristicPlanner honors backoff: the excluded pair is never proposed.
  E. _ask_once tool-history strip (planner/vision_drive.py): the `{"action":`
     prefill no longer follows tool messages (the Aug 19 HTTP 400 root cause);
     the get_board_state text is folded into the user message instead.
  F. py_compile of the changed modules.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from planner.agent import (  # noqa: E402
    NOOP_BACKOFF_STEPS,
    HeuristicPlanner,
    _MergeNoopRegistry,
)
from planner.merge import ranked_merge_groups  # noqa: E402
from planner.vision_drive import VisionDrivenPlanner  # noqa: E402
from vision.classifier import TemplateClassifier  # noqa: E402
from vision.grid import (  # noqa: E402
    BoardState,
    Cell,
    GridGeometry,
    set_grid_geometry,
)
from vision.pipeline import classify_board  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = str(ROOT / "screenshots" / "noop_run_final.png")
GEO = GridGeometry(293, 1178, 230, 5, 3)   # blob-derived geometry of the fixture

PASS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if ok:
        PASS += 1


def board(cells: list[tuple[int, int, str, float]]) -> BoardState:
    """Cells as (row, col, item_id, margin); score defaults 1.0, occupied True."""
    cc = [Cell(r, c, 0, 0, iid, score=1.0, margin=m, occupied=True)
          for r, c, iid, m in cells]
    return BoardState(rows=5, cols=3, cells=cc)


def part_a_phantom_wipe() -> None:
    set_grid_geometry(GEO)
    frame = cv2.imread(FIXTURE)
    check("A frame loads", frame is not None)
    if frame is None:
        return
    classifier = TemplateClassifier("assets/templates", seed=True)
    b = classify_board(frame, classifier)
    cells = {(c.row, c.col): c for c in b.cells}
    # The noop-run fixture: (3,0) empty-but-phantom skeleton_lvl5, (4,1) a REAL
    # skeleton_lvl5. The wipe must drop the phantom and keep the real item.
    check("A (3,0) phantom wiped to empty",
          cells[(3, 0)].item_id is None and not cells[(3, 0)].occupied,
          f"id={cells[(3,0)].item_id} occ={cells[(3,0)].occupied}")
    check("A (4,1) real skeleton_lvl5 kept",
          cells[(4, 1)].item_id == "skeleton_lvl5" and cells[(4, 1)].occupied,
          f"id={cells[(4,1)].item_id}")
    # No skeleton_lvl5 pair remains -> the (3,0)+(4,1) merge is gone.
    groups = ranked_merge_groups(b)
    check("A no skeleton_lvl5 merge proposed",
          all(g[0] != "skeleton_lvl5" for g in groups),
          f"groups={[g[0] for g in groups]}")


def part_b_registry() -> None:
    reg = _MergeNoopRegistry()
    check("B one noop: not excluded",
          reg.excluded() == set(), f"{reg.excluded()}")
    reg.record((3, 0), (4, 1))
    reg.record((3, 0), (4, 1))
    check("B two noops: pair excluded",
          reg.excluded() == {((3, 0), (4, 1))}, f"{reg.excluded()}")
    # reverse-order keys normalize the same way
    reg2 = _MergeNoopRegistry()
    reg2.record((4, 1), (3, 0))
    reg2.record((3, 0), (4, 1))
    check("B key normalization (order-independent)",
          ((3, 0), (4, 1)) in reg2.excluded(), f"{reg2.excluded()}")
    for _ in range(NOOP_BACKOFF_STEPS - 1):
        reg.tick()
    check("B still excluded before expiry",
          ((3, 0), (4, 1)) in reg.excluded())
    reg.tick()
    check("B released after NOOP_BACKOFF_STEPS ticks",
          reg.excluded() == set(), f"{reg.excluded()}")


def part_c_exclude_pairs() -> None:
    b = board([(0, 0, "bone", 0.9), (0, 1, "bone", 0.9),
               (1, 0, "skeleton_lvl1", 0.9), (1, 1, "skeleton_lvl1", 0.9)])
    ranked = ranked_merge_groups(b, exclude_pairs={((0, 0), (0, 1))})
    check("C excluded pair skipped -> other merge proposed",
          len(ranked) == 1 and ranked[0][0] == "skeleton_lvl1",
          f"{[(g[0], [(c.row, c.col) for c in g[1]]) for g in ranked]}")
    b2 = board([(0, 0, "bone", 0.9), (0, 1, "bone", 0.9)])
    ranked2 = ranked_merge_groups(b2, exclude_pairs={((0, 0), (0, 1))})
    check("C only pair excluded -> group dropped",
          len(ranked2) == 0, f"{len(ranked2)}")
    # 3-cell group: excluded top-left adjacent pair falls to the next pair.
    b3 = board([(0, 0, "bone", 0.9), (0, 1, "bone", 0.9),
                (1, 1, "bone", 0.9)])
    ranked3 = ranked_merge_groups(b3, exclude_pairs={((0, 0), (0, 1))})
    check("C 3-cell group falls to next adjacent pair",
          len(ranked3) == 1
          and {(c.row, c.col) for c in ranked3[0][1]} == {(0, 1), (1, 1)},
          f"{[(c.row, c.col) for c in ranked3[0][1]]}")


def part_d_heuristic_backoff() -> None:
    hp = HeuristicPlanner()
    b = board([(0, 0, "bone", 0.9), (0, 1, "bone", 0.9)])
    mv = hp.next_move(b, frame=None)
    check("D without backoff: merge proposed", mv.kind == "merge", f"{mv.kind}")
    hp.record_merge_noop((0, 0), (0, 1))
    hp.record_merge_noop((0, 0), (0, 1))
    mv2 = hp.next_move(b, frame=None)
    check("D after 2 noops: no merge proposed", mv2.kind != "merge", f"{mv2.kind}")
    for _ in range(NOOP_BACKOFF_STEPS):
        hp.next_move(b, frame=None)   # ticks the shared registry
    mv3 = hp.next_move(b, frame=None)
    check("D after backoff expiry: merge proposed again", mv3.kind == "merge",
          f"{mv3.kind}")


class StubLLM:
    """Records the exact request messages; answers with a fixed idle JSON."""

    def __init__(self):
        self.last_messages = None

    def chat(self, messages, max_tokens=1024, temperature=0.0, json_mode=True):
        self.last_messages = messages
        return '{"action": "idle"}', {}


def part_e_ask_once_strip() -> None:
    planner = VisionDrivenPlanner(client=StubLLM(), tool_enabled=False,
                                  chat_log_path=Path(tempfile.mkdtemp()) / "chat.jsonl")
    base = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "[image]"}},
            {"type": "text", "text": "pick a move"}]},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "get_board_state", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "BOARD-STATE-TEXT"},
    ]
    reply, move = planner._ask_once([dict(m) for m in base])
    msgs = planner.client.last_messages
    check("E reply parsed to a move", move is not None and move.kind == "idle",
          f"{reply!r}")
    check("E no tool messages in request",
          all(m.get("role") != "tool" for m in msgs))
    check("E no assistant tool_calls in request",
          all(not (m.get("role") == "assistant" and m.get("tool_calls")) for m in msgs))
    check("E last message is the action prefill",
          msgs[-1] == {"role": "assistant", "content": '{"action":'})
    user_parts = [p for m in msgs if m["role"] == "user" for p in m["content"]]
    check("E board-state text folded into user message",
          any("BOARD-STATE-TEXT" in p.get("text", "") for p in user_parts),
          f"{[p.get('text', '')[:20] for p in user_parts]}")


def part_f_compile() -> None:
    import py_compile
    for f in ("planner/agent.py", "planner/merge.py", "planner/llm.py",
              "planner/vision_drive.py", "vision/grid.py", "vision/pipeline.py"):
        py_compile.compile(str(ROOT / f), doraise=True)
    check("F py_compile clean", True)


def main() -> None:
    global PASS
    parts = (part_a_phantom_wipe, part_b_registry, part_c_exclude_pairs,
             part_d_heuristic_backoff, part_e_ask_once_strip, part_f_compile)
    for part in parts:
        try:
            part()
        except Exception as exc:  # noqa: BLE001
            check(f"{part.__name__} raised {type(exc).__name__}", False, str(exc))
    print(f"\n{PASS} checks passed")
    sys.exit(0 if PASS >= 24 else 1)


if __name__ == "__main__":
    main()