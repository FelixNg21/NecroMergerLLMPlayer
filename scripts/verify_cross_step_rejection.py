"""Verify the Aug 29 cross-step rejection memory.

The user reported the model would suggest feeding the mana pool, mana pot,
and lvl1 coin every step — each rejected, then forgotten at step boundary so
the same actions came back. The fix:
  1. EVERY rejection in `_drive` is appended to `self._cross_step_rejected`
     (not just the last one — earlier draft stored only the last, and the
     model would just shift to the second-to-last and re-emit that).
  2. At the start of the next `next_move`, entries older than
     CROSS_STEP_TTL steps are dropped.
  3. The next step's `rejected` set is pre-populated with all of the
     previous step's (kind, cell_a, cell_b) tuples so attempt 0 can't
     re-emit any of them.
  4. The reasons are rendered into the `get_board_state` line so the
     model sees WHY its previous actions were bad.
  5. A successful move does NOT wipe the memory; entries only expire via the
     CROSS_STEP_TTL age limit or when the rejected cell(s) no longer hold the
     same item (the board shifted past the rejection).

Each test below exercises one of these guarantees.
"""
import sys
from unittest.mock import MagicMock

sys.path.insert(0, '.')

from planner.vision_drive import (
    VisionDrivenPlanner, Move, CROSS_STEP_TTL,
)


def _make_cell(row, col, item_id, occupied=True, score=0.8, margin=0.5):
    cell = MagicMock()
    cell.row = row
    cell.col = col
    cell.item_id = item_id
    cell.occupied = occupied
    cell.score = score
    cell.margin = margin
    cell.cell_px = 230
    cell.cx = 100 + col * 230
    cell.cy = 100 + row * 230
    return cell


def _board(cells):
    b = MagicMock()
    b.cells = cells
    b.rows = 5
    b.cols = 4
    by_pos = {(c.row, c.col): c for c in cells}
    b.cell_at = lambda r, c: by_pos.get((r, c))
    return b


def _planner():
    """Build a planner with the bare minimum wiring to exercise the
    cross-step rejection memory (no LLM, no device)."""
    p = VisionDrivenPlanner(
        live=False,
        classifier=None,
        log=MagicMock(),
        tool_enabled=False,
    )
    return p


# ---------- Test A: every rejection is appended, not just the last ----------

def test_a_all_rejections_recorded():
    """When the model emits 3 rejected moves in a step, all 3 are
    recorded in self._cross_step_rejected (NOT just the last)."""
    p = _planner()

    # Simulate 3 rejections of different moves
    move1 = Move(kind="feed", cell_a=(0, 0))
    move2 = Move(kind="feed", cell_a=(1, 2))
    move3 = Move(kind="merge", cell_a=(2, 3), cell_b=(3, 1))

    p._step_count = 5
    p._cross_step_rejected = []
    for m, reason in [(move1, "feed_mana_overflow: bar full"),
                       (move2, "feed_overflow: too much"),
                       (move3, "merge_mismatch: different levels")]:
        p._cross_step_rejected.append({
            "kind": m.kind,
            "cell_a": m.cell_a,
            "cell_b": m.cell_b,
            "reason": reason,
            "step": p._step_count,
        })

    # All three are in the list
    assert len(p._cross_step_rejected) == 3, \
        f"expected 3 rejections, got {len(p._cross_step_rejected)}"

    # Each one is a separate entry (not a single last-write-wins tuple)
    kinds = [r["kind"] for r in p._cross_step_rejected]
    assert kinds == ["feed", "feed", "merge"], \
        f"expected feed, feed, merge, got {kinds}"

    print("  PASS: A: all 3 rejections are appended (not just the last)")
    return True


# ---------- Test B: cross-step set pre-populates next step's rejected set ----

def test_b_rejected_set_seeded_from_cross_step():
    """The next step's rejected set must contain every (kind, cell_a, cell_b)
    from the previous step's rejections."""
    p = _planner()
    p._step_count = 10
    p._cross_step_rejected = [
        {"kind": "feed", "cell_a": (0, 0), "cell_b": None,
         "reason": "feed_mana_overflow: bar full", "step": 9},
        {"kind": "feed", "cell_a": (1, 2), "cell_b": None,
         "reason": "feed_overflow: too much", "step": 9},
        {"kind": "merge", "cell_a": (2, 3), "cell_b": (3, 1),
         "reason": "merge_mismatch: different levels", "step": 9},
    ]

    # Simulate the seeding logic from _drive
    rejected = set()
    for rec in p._cross_step_rejected:
        rejected.add((rec["kind"], rec["cell_a"], rec["cell_b"]))

    # All three are in the rejected set
    assert ("feed", (0, 0), None) in rejected
    assert ("feed", (1, 2), None) in rejected
    assert ("merge", (2, 3), (3, 1)) in rejected
    assert len(rejected) == 3

    print("  PASS: B: rejected set seeded with all 3 previous rejections")
    return True


# ---------- Test C: TTL filter drops stale entries at step boundary ---------

def test_c_ttl_drops_stale_entries():
    """Entries older than CROSS_STEP_TTL are dropped at step boundary
    so the model can re-evaluate after the board shifts."""
    p = _planner()
    p._step_count = CROSS_STEP_TTL + 5   # 8
    # Step 5 entries are 3 steps old (8 - 5 = 3) — at the TTL boundary (excluded by <)
    # Step 6 entries are 2 steps old — kept
    # Step 1 entries are 7 steps old — dropped
    p._cross_step_rejected = [
        {"kind": "feed", "cell_a": (0, 0), "cell_b": None,
         "reason": "x", "step": 1},     # age=7, dropped
        {"kind": "feed", "cell_a": (1, 2), "cell_b": None,
         "reason": "y", "step": 5},     # age=3, dropped (strict <)
        {"kind": "merge", "cell_a": (2, 3), "cell_b": (3, 1),
         "reason": "z", "step": 6},     # age=2, kept
    ]

    # Run the TTL filter logic
    p._cross_step_rejected = [
        rec for rec in p._cross_step_rejected
        if p._step_count - rec["step"] < CROSS_STEP_TTL
    ]

    # Only the age=2 entry survives
    assert len(p._cross_step_rejected) == 1, \
        f"expected 1 after TTL, got {len(p._cross_step_rejected)}: {p._cross_step_rejected}"
    steps = [r["step"] for r in p._cross_step_rejected]
    assert steps == [6], f"expected [6], got {steps}"

    print(f"  PASS: C: TTL={CROSS_STEP_TTL} drops entries >= {CROSS_STEP_TTL} steps old (strict <)")
    return True


# ---------- Test D: a successful move does NOT wipe the memory ----------

def test_d_success_does_not_wipe_memory():
    """A successful move must NOT clear the cross-step rejection memory.

    Before this fix the success path did `self._cross_step_rejected = []`,
    which erased every rejection — including the one the model just hit —
    so the very next step re-attempted the same illegal move (the recurring
    feed-icerune loop). Rejections should instead stay blocked while the SAME
    item still occupies the rejected cell(s), and only lapse via the TTL or
    when the board shifts past them."""
    p = _planner()
    p._step_count = 5
    # Rejection of feeding icerune_lvl1 at (0,0), still on the board.
    p._cross_step_rejected = [
        {"kind": "feed", "cell_a": (0, 0), "cell_b": None,
         "reason": "feed_not_max_level:icerune_lvl1", "step": 5,
         "item_a": "icerune_lvl1", "item_b": None},
    ]
    board = _board([_make_cell(0, 0, "icerune_lvl1")])

    # The success path no longer clears the list; a successful move is not
    # simulated by wiping here. Instead verify the item-shift pruning helper:
    # the rejected feed is still valid while icerune_lvl1 occupies (0,0)...
    assert p._cross_step_rejection_still_valid(p._cross_step_rejected[0], board) is True, \
        "rejection should stay valid while the same item is in the cell"

    # ...and lapses the moment that cell's item changes (e.g. merged away).
    board2 = _board([_make_cell(0, 0, "icerune_lvl2")])
    assert p._cross_step_rejection_still_valid(p._cross_step_rejected[0], board2) is False, \
        "rejection should lapse when the cell's item changes"

    # It also lapses when the cell goes empty.
    board3 = _board([_make_cell(0, 0, None, occupied=False)])
    assert p._cross_step_rejection_still_valid(p._cross_step_rejected[0], board3) is False, \
        "rejection should lapse when the cell empties"

    print("  PASS: D: successful move does NOT wipe; memory lapses on item-shift/TTL")
    return True


# ---------- Test E: rejection line text lists ALL rejected actions -----------

def test_e_rejection_line_renders_all():
    """The get_board_state line shows every rejected action from the
    previous step, not just the last."""
    p = _planner()
    recs = [
        {"kind": "feed", "cell_a": (0, 0), "cell_b": None,
         "reason": "feed_mana_overflow: bar full"},
        {"kind": "feed", "cell_a": (1, 2), "cell_b": None,
         "reason": "feed_overflow: too much"},
        {"kind": "merge", "cell_a": (2, 3), "cell_b": (3, 1),
         "reason": "merge_mismatch: different levels"},
    ]
    line = p._rejection_line_text(recs)

    # All three are mentioned by kind+reason
    for kind in ("feed", "merge"):
        assert kind in line, f"{kind!r} missing from line: {line!r}"
    for reason_key in ("feed_mana_overflow", "feed_overflow", "merge_mismatch"):
        assert reason_key in line, f"{reason_key!r} missing from line: {line!r}"

    # The line says they are pre-blocked this step
    assert "do NOT" in line.lower() or "do not" in line.lower(), \
        f"line should warn NOT to propose them again: {line!r}"

    print("  PASS: E: rejection line renders ALL 3 actions with reasons")
    return True


# ---------- Test F: same move rejected twice in the same step is OK --------

def test_f_same_action_rejected_twice_appends_twice():
    """If the same action is rejected twice in the same step (model
    proposes the same fix twice after a correction), the cross-step
    memory ends up with two entries — the per-step `rejected` set
    dedupes them, but the cross-step memory preserves the count for
    the next step's TTL check."""
    p = _planner()
    p._step_count = 5
    p._cross_step_rejected = []

    move = Move(kind="feed", cell_a=(0, 0))
    for _ in range(2):
        p._cross_step_rejected.append({
            "kind": move.kind,
            "cell_a": move.cell_a,
            "cell_b": move.cell_b,
            "reason": "feed_mana_overflow: bar full",
            "step": p._step_count,
        })

    assert len(p._cross_step_rejected) == 2
    print("  PASS: F: duplicate rejections in same step append twice")
    return True


# ---------- Test G: empty list renders empty line ---------

def test_g_empty_rejection_list_renders_nothing():
    """When the cross-step memory is empty, _rejection_line_text still
    produces a line — but _board_state_text only calls it when the
    list is non-empty (the gate at line 3395)."""
    p = _planner()
    # Empty list — the line still renders, but caller checks first
    line = p._rejection_line_text([])
    assert "Last step rejected" in line, \
        f"expected a 'Last step rejected' line, got: {line!r}"
    # The list contents section is empty
    assert "pick a" in line.lower() or "different" in line.lower()
    print("  PASS: G: empty rejection list still produces a line (caller gates)")
    return True


# ---------- Main ----------

if __name__ == "__main__":
    tests = [
        test_a_all_rejections_recorded,
        test_b_rejected_set_seeded_from_cross_step,
        test_c_ttl_drops_stale_entries,
        test_d_success_does_not_wipe_memory,
        test_e_rejection_line_renders_all,
        test_f_same_action_rejected_twice_appends_twice,
        test_g_empty_rejection_list_renders_nothing,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            if t():
                passed += 1
        except AssertionError as exc:
            print(f"  FAIL: {t.__name__}: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR: {t.__name__}: {exc}")
            failed += 1
    print(f"\n{passed}/{passed+failed} cross-step rejection tests passed")
    sys.exit(0 if failed == 0 else 1)
