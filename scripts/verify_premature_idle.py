"""`_drive`'s retry loop must not accept `idle` immediately after a
non-idle rejection. The model has up to `max_retries` attempts; emitting
`idle` on the first attempt after a rejection was a known failure mode
(the model gave up immediately, the board stalled for the rest of the
step, the validator had 50 more chances unused). The fix is a small
gate right after the existing `repeat_rejected` check that re-sends the
correction message when the model emits `idle` with `rejected` non-empty.

Run: `.venv/bin/python scripts/verify_premature_idle.py`
"""

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# A real frame is required by `_drive` (it base64-encodes the frame for
# the LLM image_url). Reuse the calibration screenshot.
FRAME_PATH = ROOT / "screenshots" / "calib_board.png"

import cv2

from vision.classifier import TemplateClassifier
from vision.grid import BoardState, Cell, GridGeometry, set_grid_geometry

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {detail}")


@contextmanager
def _make_planner(feeds, max_level, max_retries=3, mana=None,
                 satiety_remaining=188, satiety_capacity=1000):
    """Yield a minimal VisionDrivenPlanner. The tmp dir lives for the
    duration of the `with` block so the planner can write to chat_log_path
    during the retry loop."""
    from planner.vision_drive import VisionDrivenPlanner
    from planner.agent import Move

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        chat_log = tmp / "c.jsonl"
        chat_log.touch()
        clf = TemplateClassifier(str(tmp / "templates"), seed=True)
        p = VisionDrivenPlanner(
            client=None, live=False, tool_enabled=True, hints=True,
            log=None, chat_log_path=chat_log,
            classifier=clf,
            learnings_path=tmp / "learnings.md", glossary_path=tmp / "glossary.md",
            max_retries=max_retries)
        # Stub out the things _drive reads
        p._mana_fraction = mana
        p._max_level_ids = lambda: set(max_level)
        p._chain_map = lambda: {}
        p._last_craving = None

        class FakeFB:
            def __init__(self, fv):
                self.feed_values = fv
                self.damage_values = {}
                self.satiety_remaining = satiety_remaining
                self.satiety_capacity = satiety_capacity
                self.craved_item = None
                self.craved_level = None
                self.craving_bonus_est = 0
                self.feed_prefer_item = None
        p.fallback = FakeFB(feeds)
        yield p


def test_idle_after_rejection_is_rejected():
    """The headline bug: model emits `feed (0,0)` (rejected as
    feed_mana_overflow), then emits `idle`. The new gate must REJECT
    the `idle` and force the loop to keep going. With max_retries=3
    and the model alternating `idle`, the loop should exhaust
    (retry_exhausted) and the function should raise ValueError.

    Without the gate, the model would emit `idle` on attempt 1 after
    a rejection, the function would return `idle` as the answer, and
    the board would stall.
    """
    set_grid_geometry(GridGeometry(293, 1178, 230, 5, 3))
    frame = cv2.imread(str(FRAME_PATH))
    with _make_planner(feeds={"manapotion_lvl3": 22}, max_level=set(),
                       mana=1.0, max_retries=3) as p:
        # Build a 5x3 board with a manapotion at (0,0)
        cells = [Cell(r, c, 0, 0, None, score=0.0, margin=0.0, occupied=False)
                 for r in range(5) for c in range(3)]
        cells[0] = Cell(0, 0, 0, 0, 'manapotion_lvl3', score=1.0, margin=1.0, occupied=True)
        cells[1] = Cell(1, 0, 0, 0, 'bone', score=1.0, margin=0.5, occupied=True)
        cells[2] = Cell(1, 1, 0, 0, 'bone', score=1.0, margin=0.5, occupied=True)
        board = BoardState(rows=5, cols=3, cells=cells)

        from unittest.mock import MagicMock
        from planner.vision_drive import Move

        p.client = MagicMock()
        p.client.chat_message.return_value = ("", [], {"content": None})
        responses = iter([
            ('{"action":"feed","cell":[0,0]}', Move(kind="feed", cell_a=(0, 0))),
            ('{"action":"idle"}', Move(kind="idle")),
            ('{"action":"idle"}', Move(kind="idle")),
            ('{"action":"idle"}', Move(kind="idle")),
            ('{"action":"idle"}', Move(kind="idle")),
        ])

        def _ask(msgs):
            return next(responses)

        p._ask_once = _ask

        try:
            move = p._drive(board, frame=frame)
            # If the bug isn't fixed, the function would return Move(kind="idle")
            # after attempt 1 (feed rejected) + attempt 2 (idle accepted as
            # the answer). The function should NOT return early with idle
            # because the model still has retries.
            check("A: idle-after-rejection doesn't end the loop",
                  False, f"got {move}")
        except ValueError as e:
            check("A: idle-after-rejection exhausts retries (raises ValueError)",
                  "no valid move" in str(e), f"got: {e}")
        except Exception as e:
            check("A: idle-after-rejection exhausts retries", False,
                  f"unexpected exception: {type(e).__name__}: {e}")


def test_idle_first_action_is_accepted():
    """`idle` on the FIRST attempt (no prior rejection) is still
    accepted. The gate is gated on `rejected` being non-empty.
    Without that gate, the model couldn't ever emit `idle`."""
    set_grid_geometry(GridGeometry(293, 1178, 230, 5, 3))
    frame = cv2.imread(str(FRAME_PATH))
    with _make_planner(feeds={}, max_level=set(), max_retries=3) as p:
        cells = [Cell(r, c, 0, 0, None, score=0.0, margin=0.0, occupied=False)
                 for r in range(5) for c in range(3)]
        board = BoardState(rows=5, cols=3, cells=cells)

        from unittest.mock import MagicMock
        from planner.vision_drive import Move

        p.client = MagicMock()
        p.client.chat_message.return_value = ("", [], {"content": None})
        responses = iter([
            ('{"action":"idle"}', Move(kind="idle")),
        ])
        p._ask_once = lambda msgs: next(responses)

        move = p._drive(board, frame=frame)
        check("B: idle as the first action is accepted",
              move.kind == "idle", f"got {move.kind}")


def test_idle_after_multiple_rejections_still_uses_budget():
    """After 1 feed rejection + 2 idle rejections, the model emits a
    different action. That action should be accepted (the loop has
    already burned 3 attempts but a 4th attempt is allowed under
    max_retries=4).
    """
    set_grid_geometry(GridGeometry(293, 1178, 230, 5, 3))
    frame = cv2.imread(str(FRAME_PATH))
    with _make_planner(feeds={"manapotion_lvl3": 22}, max_level=set(),
                       mana=1.0, max_retries=4) as p:
        cells = [Cell(r, c, 0, 0, None, score=0.0, margin=0.0, occupied=False)
                 for r in range(5) for c in range(3)]
        cells[0] = Cell(0, 0, 0, 0, 'manapotion_lvl3', score=1.0, margin=1.0, occupied=True)
        cells[1] = Cell(1, 0, 0, 0, 'bone', score=1.0, margin=0.5, occupied=True)
        cells[2] = Cell(1, 1, 0, 0, 'bone', score=1.0, margin=0.5, occupied=True)
        board = BoardState(rows=5, cols=3, cells=cells)

        from unittest.mock import MagicMock
        from planner.vision_drive import Move

        p.client = MagicMock()
        p.client.chat_message.return_value = ("", [], {"content": None})
        responses = iter([
            ('{"action":"feed","cell":[0,0]}', Move(kind="feed", cell_a=(0, 0))),
            ('{"action":"idle"}', Move(kind="idle")),
            ('{"action":"idle"}', Move(kind="idle")),
            ('{"action":"merge","a":[1,0],"b":[1,1]}',
             Move(kind="merge", cell_a=(1, 0), cell_b=(1, 1))),
        ])
        p._ask_once = lambda msgs: next(responses)

        move = p._drive(board, frame=frame)
        check("C: non-idle after idles is accepted",
              move.kind == "merge" and move.cell_a == (1, 0),
              f"got {move}")


if __name__ == "__main__":
    test_idle_after_rejection_is_rejected()
    test_idle_first_action_is_accepted()
    test_idle_after_multiple_rejections_still_uses_budget()
    print(f"\n{PASS} pass, {FAIL} fail")
    sys.exit(0 if FAIL == 0 else 1)
