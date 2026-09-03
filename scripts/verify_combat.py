"""Verify the attack (champion combat) move end-to-end.

added the `attack` move kind so the bot can damage Champions
(peasant/knight/cleric/paladin/rival/protector) by dragging one of OUR
creatures onto the champion cell. Champions NEVER attack us back — the
only way to make progress is to spend a creature dealing its `damage`
stat. This file proves the whole pipeline:

  - best_attack_pair() picks the highest-damage creature
  - heuristic + LLM validators accept the move
  - _validate rejects the documented edge cases
  - controller.execute() issues the right swipe
  - parse_action reads the new JSON shape
  - board state rendering tags champion cells
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.grid import BoardState, Cell as _CellT


def _cell(r, c, item_id, score=0.95, margin=0.5):
    """A simple Cell-like object for the validator tests (no template bank)."""
    return _CellT(r, c, cx=0, cy=0, item_id=item_id, score=score, margin=margin,
                  occupied=bool(item_id))


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    line = f"{mark}  {name}"
    if detail:
        line += f"  [{detail}]"
    print(line)
    if not ok:
        global PASS
        PASS = False


PASS = True


def part_a_best_pair() -> None:
    """best_attack_pair returns the highest-damage creature when a champion
    is on the board, and None when no champion is present."""
    from planner.agent import best_attack_pair

    # A1: no champion -> None
    board = BoardState(rows=5, cols=4, cells=[
        _cell(r, c, None) for r in range(5) for c in range(4)])
    check("A1: no champion -> best_attack_pair returns None",
          best_attack_pair(board, damage_values={}) is None)

    # A2: champion on board, no attackers with damage -> None
    board = BoardState(rows=2, cols=2, cells=[
        _cell(0, 0, "peasant"),
        _cell(0, 1, "bone"),
        _cell(1, 0, None),
        _cell(1, 1, None),
    ])
    check("A2: champion + creature without damage -> None",
          best_attack_pair(board, damage_values={}) is None)

    # A3: champion + damage-bearing creature -> the highest-damage one wins
    board = BoardState(rows=2, cols=3, cells=[
        _cell(0, 0, "peasant"),
        _cell(0, 1, "bone"),                # dmg 1, low priority
        _cell(0, 2, "skeleton_lvl1"),       # dmg 5
        _cell(1, 0, "skeleton_lvl4"),       # dmg 50, the winner
        _cell(1, 1, None),
        _cell(1, 2, None),
    ])
    pair = best_attack_pair(board, damage_values={
        "bone": 1, "skeleton_lvl1": 5, "skeleton_lvl4": 50,
    })
    check("A3: highest-damage creature wins",
          pair is not None and pair[0].item_id == "skeleton_lvl4"
          and pair[2] == 50 and pair[1].item_id == "peasant",
          f"pair={pair}")

    # A4: stations/champions excluded as attackers
    board = BoardState(rows=2, cols=2, cells=[
        _cell(0, 0, "knight"),
        _cell(0, 1, "grave_lvl2"),  # station, dmg 0 (or unspecified)
        _cell(1, 0, "skeleton_lvl1"),
        _cell(1, 1, None),
    ])
    pair = best_attack_pair(board, damage_values={
        "grave_lvl2": 9999, "skeleton_lvl1": 5,
    })
    check("A4: station excluded as attacker even with huge dmg",
          pair is not None and pair[0].item_id == "skeleton_lvl1",
          f"pair={pair}")


def part_b_validate() -> None:
    """LLMPlanner._validate accepts a valid attack and rejects each invalid
    variant with a specific reason string."""
    from vision.grid import BoardState
    from planner.agent import Move
    from planner.llm import LLMPlanner

    def mk_board():
        return BoardState(rows=3, cols=3, cells=[
            _cell(0, 0, "peasant"),        # champion
            _cell(0, 1, "skeleton_lvl1"), # attacker
            _cell(0, 2, None),
            _cell(1, 0, None),
            _cell(1, 1, "skeleton_lvl2"),
            _cell(1, 2, None),
            _cell(2, 0, None),
            _cell(2, 1, None),
            _cell(2, 2, None),
        ])

    dmg = {"skeleton_lvl1": 5, "skeleton_lvl2": 25}

    # B1: valid attack
    board = mk_board()
    move = Move(kind="attack", cell_a=(0, 1), target=(0, 0))
    reason = LLMPlanner._validate(board, move, damage_values=dmg)
    check("B1: valid attack accepted", reason is None, str(reason))

    # B2: attacker is a station -> attack_is_station
    board = mk_board()
    move = Move(kind="attack", cell_a=(0, 1), target=(0, 0))
    board.cells[1].item_id = "grave_lvl2"
    reason = LLMPlanner._validate(board, move, damage_values=dmg)
    check("B2: station attacker refused",
          reason and reason.startswith("attack_is_station:"), str(reason))

    # B3: attacker is a champion -> attack_is_champion
    board = mk_board()
    move = Move(kind="attack", cell_a=(0, 1), target=(0, 0))
    board.cells[1].item_id = "peasant"
    reason = LLMPlanner._validate(board, move, damage_values=dmg)
    check("B3: champion attacker refused",
          reason and reason.startswith("attack_is_champion:"), str(reason))

    # B4: no target at all -> attack_no_target
    move = Move(kind="attack", cell_a=(0, 1), target=None)
    reason = LLMPlanner._validate(mk_board(), move, damage_values=dmg)
    check("B4: missing target refused", reason == "attack_no_target", str(reason))

    # B5: attacker == target -> attack_self_target (use a non-champion
    # so the validator reaches the self-target check, not the champion
    # attacker check)
    move = Move(kind="attack", cell_a=(1, 1), target=(1, 1))
    reason = LLMPlanner._validate(mk_board(), move, damage_values=dmg)
    check("B5: self-target refused", reason == "attack_self_target", str(reason))

    # B6: target is not a champion -> attack_target_not_champion
    move = Move(kind="attack", cell_a=(0, 1), target=(1, 1))
    reason = LLMPlanner._validate(mk_board(), move, damage_values=dmg)
    check("B6: non-champion target refused",
          reason and reason.startswith("attack_target_not_champion:"), str(reason))

    # B7: target cell empty -> target_empty
    move = Move(kind="attack", cell_a=(0, 1), target=(0, 2))
    reason = LLMPlanner._validate(mk_board(), move, damage_values=dmg)
    check("B7: empty target refused", reason == "target_empty", str(reason))

    # B8: attacker has no known damage value -> attack_no_damage
    board = mk_board()
    move = Move(kind="attack", cell_a=(0, 1), target=(0, 0))
    reason = LLMPlanner._validate(board, move, damage_values={})
    check("B8: attacker with no damage refused",
          reason and reason.startswith("attack_no_damage:"), str(reason))

    # B9: feeding merge-material (ribcage -> skeleton) while a non-mergeable
    # feedable exists is refused, so the model can't burn merge pairs.
    cells = [
        _cell(0, 0, "ribcage"),      # merge material (ribcage -> skeleton_lvl2)
        _cell(0, 1, "skeleton_lvl7"),  # max level -> NOT merge material (plain)
        _cell(1, 0, None),
        _cell(1, 1, None),
        _cell(2, 0, None),
        _cell(2, 1, None),
        _cell(2, 2, None),
        _cell(0, 2, None),
        _cell(1, 2, None),
    ]
    board = BoardState(rows=3, cols=3, cells=cells)
    move = Move(kind="feed", cell_a=(0, 0))
    cmap = {"skeleton": ["bone", "ribcage", "skeleton_lvl2"]}
    reason = LLMPlanner._validate(
        board, move, mana=0.1, max_level={"skeleton_lvl7"},
        feed_values={"ribcage": 2, "skeleton_lvl7": 1500},
        satiety_remaining=10, satiety_capacity=10, chain_map=cmap)
    check("B9: feeding merge-material w/ plain alt refused",
          reason and reason.startswith("feed_merge_material:"), str(reason))

    # B10: feeding merge-material on a board that has FREE cells is refused,
    # even when no plain (non-merge-material) feedable exists. With room to
    # spawn/merge food, burning a merge-material component is never necessary
    # (a "not congested" board with empty space must NOT feed merge material).
    cells = [
        _cell(0, 0, "ribcage"),      # material
        _cell(0, 1, "bone"),         # material (bone -> ribcage)
        _cell(1, 0, None),           # free cell -> must refuse
        _cell(1, 1, "ribcage"),
        _cell(2, 0, "bone"),
        _cell(2, 1, None),
        _cell(2, 2, None),
        _cell(0, 2, None),
        _cell(1, 2, None),
    ]
    board = BoardState(rows=3, cols=3, cells=cells)
    move = Move(kind="feed", cell_a=(0, 0))
    reason = LLMPlanner._validate(
        board, move, mana=0.1, max_level={"skeleton_lvl7"},
        feed_values={"ribcage": 2, "bone": 1},
        satiety_remaining=10, satiety_capacity=10, chain_map=cmap)
    check("B10: merge-material refused while board has free cells",
          reason and reason.startswith("feed_merge_material:"), str(reason))

    # B10b: on a genuinely FULL board with no plain feedable, the desperation
    # allowance still applies (feed merge material rather than starve).
    cells = [
        _cell(0, 0, "ribcage"),      # material
        _cell(0, 1, "bone"),         # material
        _cell(0, 2, "bone"),
        _cell(1, 0, "ribcage"),
        _cell(1, 1, "bone"),
        _cell(1, 2, "ribcage"),
        _cell(2, 0, "bone"),
        _cell(2, 1, "bone"),
        _cell(2, 2, "bone"),
    ]
    board = BoardState(rows=3, cols=3, cells=cells)
    reason = LLMPlanner._validate(
        board, move, mana=0.1, max_level={"skeleton_lvl7"},
        feed_values={"ribcage": 2, "bone": 1},
        satiety_remaining=10, satiety_capacity=10, chain_map=cmap)
    check("B10b: merge-material desperation allowed on a FULL board",
          reason is None, str(reason))

    # B11: on a FULL board, strategy-critical merge material is protected —
    # feeding a zombie on the "Own a lvl 3+ Zombie" strategy path is refused
    # (`feed_strategy_material`) as long as any OTHER feedable exists.
    cells = [
        _cell(0, 0, "peasant"),        # champion (not feedable)
        _cell(0, 1, "zombie_lvl2"),    # strategy-critical merge material
        _cell(0, 2, "grave_lvl3"),     # station (not feedable)
        _cell(1, 0, "zombie_lvl1"),    # other material feedable
        _cell(1, 1, "bone"),           # other material feedable
        _cell(1, 2, "manapool_lvl1"),
        _cell(2, 0, "manapool_lvl1"),  # full board (no free cells)
        _cell(2, 1, "skeleton_lvl3"),
        _cell(2, 2, "necromerger"),
    ]
    board = BoardState(rows=3, cols=3, cells=cells)
    move = Move(kind="feed", cell_a=(0, 1))
    cmap = {"skeleton": ["bone", "ribcage", "skeleton_lvl1"],
            "zombie": ["rottenflesh", "severedhand", "zombie_lvl1"]}
    reason = LLMPlanner._validate(
        board, move, mana=0.1, max_level={"skeleton_lvl7"},
        feed_values={"zombie_lvl2": 60, "zombie_lvl1": 30, "bone": 1},
        satiety_remaining=1000, satiety_capacity=1000, chain_map=cmap,
        protect_family="zombie")
    check("B11: strategy-critical merge material refused when other feedable exists",
          reason and reason.startswith("feed_strategy_material:"), str(reason))

    # B12: same FULL board WITHOUT strategy protection falls back to the plain
    # desperation allowance (protect_family=None preserves old behavior).
    reason = LLMPlanner._validate(
        board, move, mana=0.1, max_level={"skeleton_lvl7"},
        feed_values={"zombie_lvl2": 60, "zombie_lvl1": 30, "bone": 1},
        satiety_remaining=1000, satiety_capacity=1000, chain_map=cmap)
    check("B12: no strategy protection -> desperation allowance preserved",
          reason is None, str(reason))

    # B13: on a FULL board with the protected material as the ONLY feedable,
    # feeding it is allowed (last resort — can't starve the Devourer).
    only = BoardState(rows=2, cols=2, cells=[
        _cell(0, 0, "peasant"),
        _cell(0, 1, "zombie_lvl2"),
        _cell(1, 0, "grave_lvl3"),
        _cell(1, 1, "manapool_lvl1"),
    ])
    reason = LLMPlanner._validate(
        only, Move(kind="feed", cell_a=(0, 1)), mana=0.1,
        max_level={"skeleton_lvl7"}, feed_values={"zombie_lvl2": 60},
        satiety_remaining=1000, satiety_capacity=1000, chain_map=cmap,
        protect_family="zombie")
    check("B13: strategy material allowed when it is the ONLY feedable",
          reason is None, str(reason))


def part_c_execute() -> None:
    """controller.execute performs a swipe from attacker to target on attack."""
    import controller.actions as actions
    from planner.agent import Move
    from vision.grid import GridGeometry
    # Build a layout with an explicit geometry (100px cells, origin 0,0)
    # so the test's cell_center math is predictable regardless of any
    # earlier test that may have set the module-level FALLBACK_GEOMETRY.
    geom = GridGeometry(origin_x=0, origin_y=0, cell_px=100,
                        rows=3, cols=3, mouth_x=150, mouth_y=150)

    class _Device:
        def __init__(self):
            self.swipes = []
            self.taps = []
        def swipe(self, x1, y1, x2, y2, duration_ms=0):
            self.swipes.append(((x1, y1), (x2, y2), duration_ms))
        def tap(self, x, y):
            self.taps.append((x, y))
        def wait_for_idle(self, _s):
            pass

    layout = actions.Layout(devourer_xy=(0, 0), geom=geom)
    dev = _Device()
    move = Move(kind="attack", cell_a=(1, 1), target=(0, 0))
    actions.execute(dev, layout, move)
    # Origin (0,0) + cell_px=100: cell_center(1,1) = (150, 150),
    # cell_center(0,0) = (50, 50).
    check("C1: attack issues a swipe from attacker to target",
          len(dev.swipes) == 1
          and dev.swipes[0][0] == (150, 150)
          and dev.swipes[0][1] == (50, 50),
          str(dev.swipes))

    # C2: missing target raises NotImplementedError
    try:
        actions.execute(dev, layout, Move(kind="attack", cell_a=(1, 1)))
    except NotImplementedError as exc:
        check("C2: attack without target raises NotImplementedError",
              "without target" in str(exc), str(exc))
    else:
        check("C2: attack without target raises NotImplementedError", False,
              "no exception raised")


def part_d_parse_action() -> None:
    """_parse_action handles the new 'attack' action shape."""
    from planner.vision_drive import VisionDrivenPlanner
    # Valid
    m = VisionDrivenPlanner._parse_action(
        '{"action":"attack","cell":[1,2],"target":[3,0]}')
    check("D1: parse action attack (cell, target)",
          m is not None and m.kind == "attack"
          and m.cell_a == (1, 2) and m.target == (3, 0), str(m))
    # Missing target
    m = VisionDrivenPlanner._parse_action(
        '{"action":"attack","cell":[1,2]}')
    check("D2: attack without target returns None", m is None, str(m))
    # Garbled
    m = VisionDrivenPlanner._parse_action('not json at all')
    check("D3: garbled JSON returns None", m is None, str(m))


def part_e_heuristic_attack_branch() -> None:
    """HeuristicPlanner.next_move returns an attack when a champion is on the
    board and a damage-bearing creature is available, even with no active feats."""
    from vision.grid import BoardState
    from planner.agent import HeuristicPlanner
    board = BoardState(rows=3, cols=3, cells=[
        _cell(0, 0, "peasant"),
        _cell(0, 1, "skeleton_lvl1"),
        _cell(0, 2, "skeleton_lvl4"),  # highest dmg
        _cell(1, 0, None),
        _cell(1, 1, None),
        _cell(1, 2, None),
        _cell(2, 0, None),
        _cell(2, 1, None),
        _cell(2, 2, None),
    ])
    p = HeuristicPlanner()
    p.damage_values = {"skeleton_lvl1": 5, "skeleton_lvl4": 50}
    move = p.next_move(board, frame=None)
    check("E1: heuristic returns attack when champion + damage present",
          move.kind == "attack" and move.target == (0, 0)
          and move.cell_a == (0, 2),  # skeleton_lvl4 wins
          f"move={move}")

    # E2: no champion -> no attack (falls through to merge/idle)
    board2 = BoardState(rows=3, cols=3, cells=[
        _cell(0, 0, "skeleton_lvl1"),
        _cell(0, 1, "skeleton_lvl1"),
        _cell(0, 2, None),
        _cell(1, 0, None),
        _cell(1, 1, None),
        _cell(1, 2, None),
        _cell(2, 0, None),
        _cell(2, 1, None),
        _cell(2, 2, None),
    ])
    p2 = HeuristicPlanner()
    p2.damage_values = {"skeleton_lvl1": 5}
    move2 = p2.next_move(board2, frame=None)
    check("E2: no champion -> heuristic returns merge (not attack)",
          move2.kind == "merge", f"move={move2}")

    # E3: champion on board but no creature with damage -> no attack
    board3 = BoardState(rows=3, cols=3, cells=[
        _cell(0, 0, "peasant"),
        _cell(0, 1, "bone"),
        _cell(0, 2, None),
        _cell(1, 0, None),
        _cell(1, 1, None),
        _cell(1, 2, None),
        _cell(2, 0, None),
        _cell(2, 1, None),
        _cell(2, 2, None),
    ])
    p3 = HeuristicPlanner()
    p3.damage_values = {}  # no known damage
    move3 = p3.next_move(board3, frame=None)
    check("E3: champion but no known-damage creature -> no attack",
          move3.kind != "attack", f"move={move3}")


def part_f_board_state_tag() -> None:
    """VisionDrivenPlanner._board_state_text tags champion cells with
    (champion) so the model can identify them in its context."""
    from vision.grid import BoardState
    from planner.vision_drive import VisionDrivenPlanner
    with tempfile.TemporaryDirectory() as td:
        from metrics.logger import SessionLog
        from vision.grid import Cell
        from pathlib import Path
        tmp = Path(td)
        # Minimal planner: no LLM, no real readers; just the renderer.
        p = VisionDrivenPlanner(
            client=None, live=False, tool_enabled=False,
            log=SessionLog(tmp / "s.jsonl"),
            chat_log_path=tmp / "c.jsonl",
            classifier=None, glossary_path=tmp / "g.md",
            learnings_path=tmp / "l.md")
        board = BoardState(rows=2, cols=2, cells=[
            Cell(0, 0, 0, 0, "peasant", score=0.95, margin=0.5, occupied=True),
            Cell(0, 1, 0, 0, "skeleton_lvl1", score=0.95, margin=0.5, occupied=True),
            Cell(1, 0, 0, 0, None, score=0.0, margin=0.0, occupied=False),
            Cell(1, 1, 0, 0, "bone", score=0.95, margin=0.5, occupied=True),
        ])
        text = p._board_state_text(board)
        check("F1: champion cell labeled (champion)",
              "(champion)" in text, text[:200])
        check("F2: non-champion cell NOT labeled (champion)",
              text.count("(champion)") == 1, f"count={text.count('(champion)')}")


def part_g_menu() -> None:
    """LLMPlanner._menu includes the attack when a champion + damage creature
    is on the board, and does NOT include it otherwise."""
    from vision.grid import BoardState
    from planner.agent import Move
    from planner.llm import LLMPlanner
    with tempfile.TemporaryDirectory() as td:
        from metrics.logger import SessionLog
        from pathlib import Path
        tmp = Path(td)
        p = LLMPlanner(client=None, live=False,
                       chat_log_path=tmp / "c.jsonl",
                       log=SessionLog(tmp / "s.jsonl"))
        p.fallback.damage_values = {"skeleton_lvl1": 5}
        # G1: with a champion, the menu contains an attack entry
        board = BoardState(rows=2, cols=2, cells=[
            _cell(0, 0, "peasant"),
            _cell(0, 1, "skeleton_lvl1"),
            _cell(1, 0, None),
            _cell(1, 1, None),
        ])
        menu = p._menu(board, mana=None)
        check("G1: menu includes attack when champion + damage present",
              any(m.kind == "attack" for m in menu), str(menu))
        # G2: no champion -> no attack
        board2 = BoardState(rows=2, cols=2, cells=[
            _cell(0, 0, "skeleton_lvl1"),
            _cell(0, 1, "skeleton_lvl1"),
            _cell(1, 0, None),
            _cell(1, 1, None),
        ])
        menu2 = p._menu(board2, mana=None)
        check("G2: menu does NOT include attack without a champion",
              not any(m.kind == "attack" for m in menu2), str(menu2))


def part_h_dry_run() -> None:
    """Live end-to-end: classifier on the live board sees a peasant and a
    skeleton_lvl1, the heuristic attack branch fires, no crash."""
    from env.adb import Device
    from vision.pipeline import classify_board
    from vision.classifier import TemplateClassifier
    from vision.grid import GridGeometry, set_grid_geometry
    from planner.agent import HeuristicPlanner
    import cv2

    geom = GridGeometry(origin_x=184, origin_y=1192, cell_px=228,
                        rows=5, cols=4, mouth_x=642, mouth_y=923)
    set_grid_geometry(geom)

    dev = Device()
    dev.screencap()
    frame = cv2.imread(str(dev.screencap_path))
    if frame is None:
        check("H1: screencap returned a frame", False, "no frame")
        return
    classifier = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
    board = classify_board(frame, classifier)
    p = HeuristicPlanner()
    p.damage_values = {}  # the live bank likely has no known damage values
    move = p.next_move(board, frame=None)
    # We don't assert move.kind == "attack" because the live board may
    # not have a champion right now — but the call must not crash.
    check("H1: dry-run attack path does not crash on the live board",
          True, f"move={move}")


def main() -> None:
    for part in (part_a_best_pair, part_b_validate, part_c_execute,
                 part_d_parse_action, part_e_heuristic_attack_branch,
                 part_f_board_state_tag, part_g_menu, part_h_dry_run):
        try:
            part()
        except Exception as exc:
            check(f"{part.__name__} raised {type(exc).__name__}",
                  False, str(exc))
    print()
    print("PASS" if PASS else "FAIL")


if __name__ == "__main__":
    main()
