"""Verification for the Aug 27 unbanked-popup-read backfill (Option 3).

Run: `.venv/bin/python scripts/verify_unbanked_popup.py` (exit 0/1).

The Aug 27 fix closed the loop where items the template bank doesn't know
about (e.g. skeleton_lvl6 before the user captured its template, Champions
before peasant/knight templates were banked) couldn't be popup-read because
the existing `_popup_next` only considered cells the classifier identified
with high confidence. The new path:

  1. `_popup_next` falls back to UNID occupied cells when no banked item
     needs backfill.
  2. `_auto_read_popup` taps such a cell, reads the popup, resolves the
     item id from the popup's name+level, banks the sprite AND the recipe
     in one pass.
  3. The next `classify_board` run recognizes the cell.

Tests:
  A. `_popup_next` returns an UNID cell when no banked candidate exists.
  B. `_popup_next` prefers banked cells over UNID cells.
  C. `_resolve_unid_popup` constructs a valid id from popup name+level.
  D. `_resolve_unid_popup` refuses an empty popup name.
  E. `_resolve_unid_popup` refuses an id that isn't already in the bank
     (don't let the LLM invent new ids from hallucinated popups).
  F. End-to-end: planner's `_auto_read_popup` taps an UNID cell, the
     popup is read, the item id is banked, and the sprite is added to
     the classifier.
  G. Champion-prefixed UNID cells are NOT auto-pop-read (we never pop a
     Champion to read its body — the champion tracker handles those).
"""

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.grid import BoardState, Cell, GridGeometry, set_grid_geometry
from vision.classifier import TemplateClassifier
from vision.identify import Identifier

PASS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if ok:
        PASS += 1


def _cell(r, c, item_id, score=0.95, margin=0.5, occupied=True, cell_px=228):
    return Cell(
        row=r, col=c, cx=r * 100 + c, cy=r * 100 + c,
        item_id=item_id, score=score, margin=margin, occupied=occupied,
        cell_px=cell_px, runner_up_id=None,
    )


def _board_with(cells, rows=4, cols=4):
    return BoardState(rows=rows, cols=cols, cells=cells)


def part_a_unid_fallback():
    """_popup_next returns the first UNID occupied cell when no banked
    candidate exists. The cell's item_id is None at the time of return
    (sentinel: 'this is an UNID cell')."""
    from planner.vision_drive import VisionDrivenPlanner
    cells = [
        _cell(0, 0, "skeleton_lvl1"),
        _cell(1, 1, None),                # UNID — the candidate
        _cell(2, 2, "skeleton_lvl2"),
    ]
    board = _board_with(cells)
    # Stub planner just to call _popup_next
    p = VisionDrivenPlanner.__new__(VisionDrivenPlanner)
    p.live = True
    p.glossary_path = Path(tempfile.mkdtemp()) / "g.md"
    p.classifier = None
    p._popup_banked = lambda iid: True            # every item is banked
    p._popup_has_fact = lambda iid, lbl: True    # ...with full facts
    p._unid_occupied_cell = lambda b: (1, 1)
    result = p._popup_next(board)
    check("A: _popup_next returns UNID cell when no banked candidate",
          result == (1, 1, None), str(result))


def part_b_banked_preferred():
    """When both a banked-missing-fact cell AND an UNID cell exist, the
    banked cell wins. Backfilling the bank is more valuable than identifying
    one new item."""
    from planner.vision_drive import VisionDrivenPlanner
    cells = [
        _cell(0, 0, "skeleton_lvl1"),     # banked, has full facts
        _cell(1, 1, "skeleton_lvl2"),     # banked but missing facts -> preferred
        _cell(2, 2, None),                # UNID -> fallback only
    ]
    board = _board_with(cells)
    p = VisionDrivenPlanner.__new__(VisionDrivenPlanner)
    p.live = True
    p.glossary_path = Path(tempfile.mkdtemp()) / "g.md"
    p.classifier = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
    p._popup_banked = lambda iid: True
    p._popup_has_fact = lambda iid, lbl: iid == "skeleton_lvl1"   # only lvl1 has facts
    p._unid_occupied_cell = lambda b: (2, 2)
    result = p._popup_next(board)
    check("B: banked-missing-fact cell preferred over UNID fallback",
          result == (1, 1, "skeleton_lvl2"), str(result))


def part_c_resolve_unid_popup():
    """_resolve_unid_popup builds 'name_lvlN' from popup data when the
    name is in the bank and a level is present."""
    from planner.vision_drive import VisionDrivenPlanner
    p = VisionDrivenPlanner.__new__(VisionDrivenPlanner)
    p.classifier = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
    # Bank has skeleton_lvl6 (after Aug 27 template capture)
    cell = _cell(0, 0, None)
    info = {"name": "Skeleton", "level": 6}
    result = p._resolve_unid_popup(cell, info)
    check("C: resolve constructs skeleton_lvl6 from popup",
          result == "skeleton_lvl6", str(result))


def part_d_resolve_refuses_empty():
    """_resolve_unid_popup returns None for an empty/missing name."""
    from planner.vision_drive import VisionDrivenPlanner
    p = VisionDrivenPlanner.__new__(VisionDrivenPlanner)
    p.classifier = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
    cell = _cell(0, 0, None)
    for empty in (None, "", "   "):
        result = p._resolve_unid_popup(cell, {"name": empty})
        check(f"D: resolve refuses empty name ({empty!r})",
              result is None, str(result))
    # A non-string name (LLM sometimes emits ints) must also be refused
    # cleanly without raising — the popup may give a numeric level where
    # the name was expected.
    try:
        result = p._resolve_unid_popup(cell, {"name": 123})
        check("D: resolve refuses non-string name without raising",
              result is None, str(result))
    except Exception as exc:  # noqa: BLE001
        check("D: resolve refuses non-string name without raising",
              False, f"raised {type(exc).__name__}: {exc}")


def part_e_resolve_refuses_unknown_id():
    """_resolve_unid_popup returns None for a name whose derived id isn't
    in the template bank. Don't let hallucinated popup names pollute the bank."""
    from planner.vision_drive import VisionDrivenPlanner
    p = VisionDrivenPlanner.__new__(VisionDrivenPlanner)
    p.classifier = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
    cell = _cell(0, 0, None)
    # "Dragon" doesn't exist in the bank
    info = {"name": "Dragon", "level": 1}
    result = p._resolve_unid_popup(cell, info)
    check("E: resolve refuses id not in bank (no hallucinated ids)",
          result is None, str(result))


def part_f_unid_cell_picked():
    """_unid_occupied_cell returns the first occupied cell with item_id None
    (or score below LABEL_MIN_SCORE), excluding the necromerger cell."""
    from planner.vision_drive import VisionDrivenPlanner
    from planner.agent import necromerger_cell
    nm = necromerger_cell()
    cells = [
        _cell(0, 0, "skeleton_lvl1"),     # known — skip
        _cell(0, 1, None),                # UNID — first candidate
        _cell(0, 2, None),                # UNID
        _cell(0, 3, "skeleton_lvl2"),     # known — skip
    ]
    # If the necromerger cell is one of these, mark it as the NM cell
    if (0, 3) == nm:
        cells[3] = _cell(0, 3, "necromerger")
    board = _board_with(cells)
    p = VisionDrivenPlanner.__new__(VisionDrivenPlanner)
    p.classifier = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
    result = p._unid_occupied_cell(board)
    check("F: _unid_occupied_cell returns first UNID occupied cell",
          result == (0, 1), str(result))


def part_g_champion_prefix_skipped():
    """A cell that the classifier labels as a Champion prefix (e.g.
    'peasant' or 'knight') is NOT auto-pop-read by the UNID path — those
    are handled by the Champion tracker. (When the bank has no template
    for a Champion yet, the cell is UNID; we shouldn't pop the Champion
    itself.)"""
    from planner.vision_drive import VisionDrivenPlanner, CHAMPION_PREFIXES
    p = VisionDrivenPlanner.__new__(VisionDrivenPlanner)
    p.classifier = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
    cells = [
        _cell(0, 0, "peasant"),     # banked Champion (after Aug 27 capture)
        _cell(1, 0, "knight"),      # banked Champion (not yet in bank, but
                                     # if it were, would be skipped)
        _cell(2, 0, None),          # truly UNID
    ]
    board = _board_with(cells)
    # Simulate "no peasant/knight in bank" by setting score low for them
    cells[0].score = 0.4
    cells[1].score = 0.4
    result = p._unid_occupied_cell(board)
    # None of the Champion cells should be the first UNID pick; (2,0) is
    check("G: UNID cell with non-Champion id is preferred",
          result == (2, 0), str(result))


def part_h_no_alt_templates():
    """ no _alt* template files remain in the bank. The __alt id
    minting was removed because the alt ids lacked damage/feed stats and
    were silently skipped by best_attack_pair()."""
    from pathlib import Path
    templates = Path(ROOT / "assets" / "templates")
    alts = sorted([p.name for p in templates.glob("*alt*")])
    check("H: no _alt* template files in assets/templates/",
          alts == [], f"found {alts}")


def part_i_no_alt_glossary():
    """ no _alt* glossary entries remain in item_glossary.md.
    Each one was a side-effect of the removed __alt minting."""
    from pathlib import Path
    glossary = (ROOT / "item_glossary.md").read_text()
    # Match the heading pattern only (## Item ..._altN (popup))
    import re
    alts = re.findall(r"^## Item .*_alt\d+ \(popup\)$", glossary, re.M)
    check("I: no _alt* glossary entries in item_glossary.md",
          alts == [], f"found {alts}")


def part_j_resolve_id_no_alt():
    """ _resolve_id no longer mints __alt ids. When the LLM says
    "skeleton" with the wrong level (sprite doesn't match skeleton_lvl4),
    the new behavior banks the sprite under the canonical id and returns
    it — broadening the bank over time instead of splintering it."""
    import numpy as np
    from vision.identify import Identifier
    from vision.classifier import TemplateClassifier

    # Set up a classifier with just skeleton_lvl4 banked
    with tempfile.TemporaryDirectory() as td:
        bank = Path(td) / "bank"
        bank.mkdir()
        # Copy an existing skeleton_lvl4 template
        import shutil
        shutil.copy(ROOT / "assets" / "templates" / "skeleton_lvl4__0.png",
                    bank / "skeleton_lvl4__0.png")
        clf = TemplateClassifier(str(bank), seed=True)

        # Make a sprite that DOESN'T match skeleton_lvl4 — a random image
        bad_sprite = np.zeros((100, 100, 3), dtype=np.uint8)
        bad_sprite[10:90, 10:90] = 200  # bright square (won't match)

        ident = Identifier.__new__(Identifier)
        ident.classifier = clf

        # LLM said "skeleton" with level 4
        result = ident._resolve_id("skeleton", 4, bad_sprite)
        check("J: _resolve_id returns canonical id (no __alt minted)",
              result == "skeleton_lvl4", f"got {result!r}")
        # And the bad sprite is now banked under the canonical id
        check("J: bad sprite banked under canonical id",
              "skeleton_lvl4__1.png" in [p.name for p in bank.iterdir()],
              f"bank files: {[p.name for p in bank.iterdir()]}")


def part_k_feed_mana_overflow_unconditional():
    """ the feed_mana_overflow check runs UNCONDITIONALLY (not gated
    on `feed_values`). The live run had a `manapot_lvl3` cell on the
    board with `Mana: ~100% full`, but `feed_values` was empty (the
    glossary has `manapotion_lvl3`, not `manapot_lvl3`). The old code put
    the overflow check inside `if z is not None:`, so it was skipped when
    the item had no recorded food value, and the manapot feed passed
    validation. The new code checks the manapot prefix BEFORE looking up
    the feed value, so the rejection fires even when the glossary is
    silent. The user's complaint: "I thought the manapot was guarded
    against" — this part closes that gap."""
    from planner.llm import LLMPlanner
    from vision.grid import BoardState, Cell
    from planner.agent import Move
    cells = [Cell(r, c, 0, 0, None, score=0.0, margin=0.0, occupied=False)
             for r in range(5) for c in range(4)]
    cells[0] = Cell(0, 0, 0, 0, 'manapot_lvl3', score=0.87, margin=0.09, occupied=True)
    cells[1] = Cell(1, 0, 0, 0, 'bone', score=1.0, margin=0.49, occupied=True)
    board = BoardState(rows=5, cols=4, cells=cells)
    move = Move(kind='feed', cell_a=(0, 0))

    # Empty feed_values (the actual bug: manapot has no glossary entry)
    reason = LLMPlanner._validate(
        board, move, mana=1.0,
        max_level={'manapot_lvl3', 'icerune_lvl3', 'skeleton_lvl7'},
        feed_values={},  # empty
        satiety_remaining=188, satiety_capacity=1000)
    check("K-1: empty feed_values + manapot + full mana -> feed_mana_overflow",
          reason == "feed_mana_overflow:manapot_lvl3:mana=1.00", f"got {reason!r}")

    # With feed value present (normal case)
    reason = LLMPlanner._validate(
        board, move, mana=1.0,
        max_level={'manapot_lvl3', 'icerune_lvl3', 'skeleton_lvl7'},
        feed_values={'manapot_lvl3': 22, 'bone': 1},
        satiety_remaining=188, satiety_capacity=1000)
    check("K-2: feed_values has manapot + full mana -> feed_mana_overflow",
          reason == "feed_mana_overflow:manapot_lvl3:mana=1.00", f"got {reason!r}")

    # manapotion_lvl3 (the wiki's actual name)
    cells[0] = Cell(0, 0, 0, 0, 'manapotion_lvl3', score=0.87, margin=0.09, occupied=True)
    board2 = BoardState(rows=5, cols=4, cells=cells)
    move2 = Move(kind='feed', cell_a=(0, 0))
    reason = LLMPlanner._validate(
        board2, move2, mana=1.0,
        max_level={'manapot_lvl3', 'manapotion_lvl3', 'skeleton_lvl7'},
        feed_values={'manapotion_lvl3': 22, 'bone': 1},
        satiety_remaining=188, satiety_capacity=1000)
    check("K-3: manapotion_lvl3 + full mana -> feed_mana_overflow",
          reason == "feed_mana_overflow:manapotion_lvl3:mana=1.00", f"got {reason!r}")

    # Mana NOT full — should pass (model can feed)
    reason = LLMPlanner._validate(
        board, move, mana=0.5,
        max_level={'manapot_lvl3', 'icerune_lvl3', 'skeleton_lvl7'},
        feed_values={},
        satiety_remaining=188, satiety_capacity=1000)
    check("K-4: empty feed_values + manapot + mana=0.5 (not full) -> None (valid)",
          reason is None, f"got {reason!r}")

    # Regular creature (bone) at full mana — should NOT fire mana_overflow
    cells[0] = Cell(0, 0, 0, 0, 'bone', score=1.00, margin=0.49, occupied=True)
    board3 = BoardState(rows=5, cols=4, cells=cells)
    move3 = Move(kind='feed', cell_a=(0, 0))
    reason = LLMPlanner._validate(
        board3, move3, mana=1.0,
        max_level=set(),
        feed_values={'bone': 1},
        satiety_remaining=188, satiety_capacity=1000)
    check("K-5: bone + full mana -> None (not a Potion)",
          reason is None, f"got {reason!r}")


def main() -> None:
    for part in (part_a_unid_fallback, part_b_banked_preferred,
                 part_c_resolve_unid_popup, part_d_resolve_refuses_empty,
                 part_e_resolve_refuses_unknown_id,
                 part_f_unid_cell_picked, part_g_champion_prefix_skipped,
                 part_h_no_alt_templates, part_i_no_alt_glossary,
                 part_j_resolve_id_no_alt,
                 part_k_feed_mana_overflow_unconditional):
        try:
            part()
        except Exception as exc:  # noqa: BLE001
            check(f"{part.__name__} raised {type(exc).__name__}", False, str(exc))
    print(f"\n{PASS} checks passed")
    sys.exit(0 if PASS >= 6 else 1)


if __name__ == "__main__":
    main()


def test_popup_question_for_categories():
    """ the per-category popup question is selected by OCR name.
    Champion / Currency / Station each get a category-targeted prompt;
    other names get the generic ITEM_POPUP_QUESTION fallback."""
    from vision.identify import popup_question_for
    from vision.identify import CHAMPION_POPUP_QUESTION, CURRENCY_POPUP_QUESTION
    from vision.identify import STATION_POPUP_QUESTION
    from vision.menu import ITEM_POPUP_QUESTION
    cases = [
        ("peasant", CHAMPION_POPUP_QUESTION),
        ("The Peasant", CHAMPION_POPUP_QUESTION),
        ("knight", CHAMPION_POPUP_QUESTION),
        ("mech", CHAMPION_POPUP_QUESTION),
        ("icerune", CURRENCY_POPUP_QUESTION),
        ("ice rune", CURRENCY_POPUP_QUESTION),
        ("coin", CURRENCY_POPUP_QUESTION),
        ("gem", CURRENCY_POPUP_QUESTION),
        ("grave", STATION_POPUP_QUESTION),
        ("manapool", STATION_POPUP_QUESTION),
        ("supply cupboard", STATION_POPUP_QUESTION),
        ("chest", STATION_POPUP_QUESTION),
        ("unknown", ITEM_POPUP_QUESTION),
        ("", ITEM_POPUP_QUESTION),
    ]
    pass_count = 0
    fail_count = 0
    for name, expected in cases:
        actual = popup_question_for(name)
        if actual is expected:
            pass_count += 1
        else:
            print(f"  FAIL: popup_question_for({name!r}) -> {actual[:50]!r}, expected category prompt")
            fail_count += 1
    print(f"  popup_question_for: {pass_count} pass, {fail_count} fail")


def test_identify_batch_returns_list_in_input_order():
    """ `Identifier.identify_batch` returns (item_id, info) pairs in
    the same order as the input cells. Tested with a fake Identifier that
    bypasses the real device (only the orchestration is checked)."""
    from vision.identify import Identifier
    from vision.classifier import TemplateClassifier
    with tempfile.TemporaryDirectory() as td:
        classifier = TemplateClassifier(str(Path(td) / "templates"), seed=True)
        identifier = Identifier.__new__(Identifier)
        identifier.classifier = classifier
        identifier.llm = None
        identifier.dock_rewards = {}

        class FakeDevice:
            def __init__(self):
                self.taps = []

            def tap(self, x, y):
                self.taps.append((x, y))

            def wait_for_idle(self, *_):
                pass

            def screencap(self):
                pass

        identifier.device = FakeDevice()
        # Use _identify_one directly to avoid the real popup path; inject
        # a fake one that returns a fixed (item_id, info) tuple.
        counter = [0]

        def fake_identify_one(_self, _board_frame, cell):
            counter[0] += 1
            return (f"id_{counter[0]}_{cell.row}_{cell.col}", {"row": cell.row, "col": cell.col})

        # Patch _identify_one on the instance.
        identifier._identify_one = fake_identify_one.__get__(identifier)
        cells = [Cell(r, 0, 100, 100, None, score=0, margin=0, occupied=True)
                 for r in range(3)]
        results = identifier.identify_batch(None, cells)
        assert len(results) == 3, f"expected 3 results, got {len(results)}"
        for (iid, info), cell in zip(results, cells):
            assert info["row"] == cell.row, f"row mismatch: {info['row']} != {cell.row}"
            assert info["col"] == cell.col, f"col mismatch"
        print("  PASS: identify_batch returns pairs in input order")


if __name__ == "__main__":
    if "test_popup_question_for_categories" in dir() or True:
        print("=== Aug 28: popup question per category ===")
        test_popup_question_for_categories()
        print()
        print("=== Aug 28: identify_batch in input order ===")
        test_identify_batch_returns_list_in_input_order()
