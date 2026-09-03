"""Verification for smarter merge selection (Aug 13 build).

Run: `.venv/bin/python scripts/verify_merge_rank.py` (exit 0/1).

Covers:
  A. ranked_merge_groups ordering (pure, no device/LLM):
     A1. max-level-result merge beats a lower-level merge
     A2. higher-level pair beats a lower-level pair
     A3. craved pair deprioritized unless it's the only merge
     A4. deterministic tiebreak (dict insertion order irrelevant)
     A5. exclusions still enforced (champion / max-level / low-margin)
  B. HeuristicPlanner uses the best merge (board with two mergeable pairs)
  C. LLMPlanner._menu orders merges best-first (menu #1 = best merge)
  D. vision-drive get_board_state renders the Best merge hint (+ empty when none)
  E. glossary read_chains parses (chain) blocks; py_compile regression
  G. Feed-to-progress (Aug 13 build): best_feed_cell ranking (craved >
     max-level > lowest feed value > unknown) + the heuristic's feed gate
     (congestion / craved / max-level always / known-cheap with a spare)
     + the vision-drive Best feed hint rendering
H. Satiety-aware feeding (Aug 13 build): best_feed_cell capacity gate
      (largest fit / nothing at 0 / over-cap excluded / unknown held back)
      + damage-value bank -> glossary -> (dmg N) render path
   J. Feat-driven priorities (Aug 14 build): parse_obj classification,
      feat_weights ratio boost, craving folded in as a feed objective,
      kind_order driving the heuristic branch order AND the vision-drive
      Best-feed/merge hint (feed outranks merge under a craving)
   K. Overflow-aware craved feeding (Aug 14 user rule): the craved feed is
      refused when Z + bonus overflows past 10% of max satiety, the fallback
      prefers spawning to rebuild the craved item, smallest-fit top-ups, the
      starvation guard, bonus learned live from satiety deltas, and the
      Best-spawn/feed hint mirror
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.agent import HeuristicPlanner, Move, best_feed_cell
from planner.constants import (is_craved_precursor, item_name_matches,
                                item_level, normalize_item_name)
from planner.merge import ranked_merge_groups, score_merge
from planner.priorities import (BASE_KIND_WEIGHT, CRAVING_BONUS_WEIGHT,
                                craving_objective, feat_weights, kind_order,
                                parse_obj)
from vision.grid import BoardState, Cell

PASS = []
FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def board(cells: list[tuple[int, int, str, float]]) -> BoardState:
    """Cells as (row, col, item_id, margin). score defaults 1.0."""
    cc = [Cell(r, c, 0, 0, iid, score=1.0, margin=m, occupied=True)
          for r, c, iid, m in cells]
    return BoardState(rows=4, cols=3, cells=cc)


def full_board(rows=4, cols=3,
               occ: dict[tuple[int, int], str] | None = None) -> BoardState:
    """A realistic 4x3 grid: every cell present, most empty — so empty_count
    and the congestion trigger behave like classify_board output."""
    occ = occ or {}
    cc = []
    for r in range(rows):
        for c in range(cols):
            iid = occ.get((r, c))
            cc.append(Cell(r, c, 0, 0, iid, score=1.0 if iid else 0.0,
                           margin=0.9 if iid else 0.0,
                           occupied=bool(iid)))
    return BoardState(rows=rows, cols=cols, cells=cc)


def merge_a(ranked):
    a, b = ranked[0][1][0], ranked[0][1][1]
    return (a.row, a.col), (b.row, b.col)


def part_a():
    # A1: skeleton_lvl4 -> skeleton_lvl5 where lvl5 is max-level beats bone merge.
    b = board([(0, 0, "skeleton_lvl4", 0.9), (0, 1, "skeleton_lvl4", 0.9),
               (1, 0, "bone", 0.9), (1, 1, "bone", 0.9)])
    ranked = ranked_merge_groups(b, max_level_ids={"skeleton_lvl5"})
    check("A1: max-result merge ranked first", ranked[0][0] == "skeleton_lvl4",
          "top=" + ranked[0][0])

    # A2: skeleton_lvl2 pair beats skeleton_lvl1 pair (no max knowledge).
    b = board([(0, 0, "skeleton_lvl1", 0.9), (0, 1, "skeleton_lvl1", 0.9),
               (1, 0, "skeleton_lvl2", 0.9), (1, 1, "skeleton_lvl2", 0.9)])
    ranked = ranked_merge_groups(b)
    check("A2: higher level first", ranked[0][0] == "skeleton_lvl2",
          "top=" + ranked[0][0])

    # A3a: craved pair (skeleton) deprioritized behind a non-craved merge.
    b = board([(0, 0, "skeleton_lvl1", 0.9), (0, 1, "skeleton_lvl1", 0.9),
               (1, 0, "bone", 0.9), (1, 1, "bone", 0.9)])
    ranked = ranked_merge_groups(b, craved_item="skeleton")
    check("A3a: craved merge deprioritized", ranked[0][0] == "bone",
          "top=" + ranked[0][0])
    # A3b: when the craved pair is the ONLY merge it still ranks.
    b = board([(0, 0, "skeleton_lvl1", 0.9), (0, 1, "skeleton_lvl1", 0.9)])
    ranked = ranked_merge_groups(b, craved_item="skeleton")
    check("A3b: sole craved merge still available", len(ranked) == 1
          and ranked[0][0] == "skeleton_lvl1")

    # A4: two equal-value pairs -> deterministic by (row, col), not insertion.
    b = board([(2, 0, "bone", 0.9), (2, 1, "bone", 0.9),
               (0, 0, "bone", 0.9), (0, 1, "bone", 0.9)])
    r1 = ranked_merge_groups(b)
    b2 = board([(0, 0, "bone", 0.9), (0, 1, "bone", 0.9),
                (2, 0, "bone", 0.9), (2, 1, "bone", 0.9)])
    r2 = ranked_merge_groups(b2)
    check("A4: deterministic tiebreak",
          merge_a(r1) == merge_a(r2) and merge_a(r1) == ((0, 0), (0, 1)),
          str(merge_a(r1)))

    # A5a: champion never merges.
    b = board([(0, 0, "peasant", 0.9), (0, 1, "peasant", 0.9),
               (1, 0, "bone", 0.9), (1, 1, "bone", 0.9)])
    ranked = ranked_merge_groups(b)
    check("A5a: champion excluded", all(g[0] != "peasant" for g in ranked))
    # A5b: max-level ids never merge.
    b = board([(0, 0, "skeleton_lvl5", 0.9), (0, 1, "skeleton_lvl5", 0.9),
               (1, 0, "bone", 0.9), (1, 1, "bone", 0.9)])
    ranked = ranked_merge_groups(b, max_level_ids={"skeleton_lvl5"})
    check("A5b: max-level excluded", all(g[0] != "skeleton_lvl5" for g in ranked))
    # A5c: low-margin cells excluded from merge groups.
    b = board([(0, 0, "bone", 0.05), (0, 1, "bone", 0.05),
               (1, 0, "skeleton_lvl1", 0.9), (1, 1, "skeleton_lvl1", 0.9)])
    ranked = ranked_merge_groups(b)
    check("A5c: low-margin excluded", len(ranked) == 1 and ranked[0][0] == "skeleton_lvl1")


def part_b():
    hp = HeuristicPlanner()
    b = board([(0, 0, "skeleton_lvl1", 0.9), (0, 1, "skeleton_lvl1", 0.9),
               (1, 0, "skeleton_lvl2", 0.9), (1, 1, "skeleton_lvl2", 0.9)])
    mv = hp.next_move(b)
    cells = {mv.cell_a, mv.cell_b}
    check("B: heuristic picks best merge", mv.kind == "merge"
          and {(1, 0), (1, 1)} <= cells, f"{mv.kind} {mv.cell_a}+{mv.cell_b}")


def part_c():
    from planner.llm import LLMPlanner
    lp = LLMPlanner(client=None, live=False)
    b = board([(0, 0, "skeleton_lvl1", 0.9), (0, 1, "skeleton_lvl1", 0.9),
               (1, 0, "skeleton_lvl2", 0.9), (1, 1, "skeleton_lvl2", 0.9)])
    menu = [m for m in lp._menu(b) if m.kind == "merge"]
    check("C: LLM menu lists best merge first",
          menu and (menu[0].cell_a in ((1, 0), (1, 1))
                    and menu[0].cell_b in ((1, 0), (1, 1))),
          str([(m.cell_a, m.cell_b) for m in menu]))


def part_d():
    import tempfile
    from planner.vision_drive import VisionDrivenPlanner
    tmp = Path(tempfile.mkdtemp())
    planner = VisionDrivenPlanner(
        client=None, log=None, live=False, tool_enabled=False,
        learnings_path=tmp / "learnings.md", glossary_path=tmp / "glossary.md")
    b = board([(0, 0, "skeleton_lvl4", 0.9), (0, 1, "skeleton_lvl4", 0.9),
               (1, 0, "bone", 0.9), (1, 1, "bone", 0.9)])
    line = planner._best_merge_line(b)
    check("D: best-merge hint rendered",
          line.startswith("Best merge: skeleton_lvl4"),
          line)
    empty = planner._best_merge_line(board([(0, 0, "bone", 0.9), (1, 0, "ribcage", 0.9)]))
    check("D2: no hint when no merge", empty == "", repr(empty))


def part_e():
    import tempfile
    from planner.glossary import read_chains
    tmp = Path(tempfile.mkdtemp())
    gl = tmp / "glossary.md"
    gl.write_text(
        "## Item skeleton (chain)\n- merge chain: bone -> ribcage -> skeleton_lvl1\n"
        "## Item skeleton (popup)\n- max_level: true\n")
    chains = read_chains(gl)
    check("E: read_chains parses (chain) blocks",
          chains == {"skeleton": ["bone", "ribcage", "skeleton_lvl1"]}, str(chains))
    check("E2: empty without chain blocks", read_chains(tmp / "nope.md") == {})


def part_f():
    import py_compile
    for f in ("planner/merge.py", "planner/constants.py", "planner/agent.py",
              "planner/llm.py", "planner/glossary.py", "planner/vision_drive.py"):
        py_compile.compile(f, doraise=True)
    check("F: py_compile all touched", True)


def part_g():
    """Feed-to-progress (Aug 13 build): best_feed_cell ranking + heuristic gate."""
    from planner.agent import best_feed_cell

    # G1a: craved item ranked first (food bonus), picked over a cheap non-craved.
    b = board([(0, 0, "bone", 0.9), (1, 0, "skeleton_lvl1", 0.9)])
    t = best_feed_cell(b, feed_values={"bone": 1, "skeleton_lvl1": 10},
                       craved_item="skeleton")
    check("G1a: craved item first", t is not None and t.item_id == "skeleton_lvl1",
          t.item_id if t else "None")
    # G1b: GENERATOR PROTECTION (Aug 23) — a max-level skeleton is a mana
    # generator; the CHEAPEST feedable wins (max-level has no priority tier).
    b = board([(0, 0, "bone", 0.9), (1, 0, "skeleton_lvl5", 0.9)])
    t = best_feed_cell(b, feed_values={"bone": 1, "skeleton_lvl5": 300},
                       max_level_ids={"skeleton_lvl5"})
    check("G1b: cheapest beats max-level (generator protection)",
          t is not None and t.item_id == "bone")
    # G1c: lowest feed value when no craving / no max-level.
    b = board([(0, 0, "skeleton_lvl2", 0.9), (1, 0, "bone", 0.9),
               (2, 0, "ribcage", 0.9)])
    t = best_feed_cell(b, feed_values={"bone": 1, "ribcage": 2, "skeleton_lvl2": 25})
    check("G1c: lowest feed value first", t is not None and t.item_id == "bone")
    # G1d: unknown feed values sort last (known-cheap preferred over unknown).
    b = board([(0, 0, "zombie_lvl2", 0.9), (1, 0, "bone", 0.9)])
    t = best_feed_cell(b, feed_values={"bone": 1})
    check("G1d: known value preferred over unknown", t is not None and t.item_id == "bone")
    # G1e: stations + champions never picked.
    b = board([(0, 0, "grave_lvl1", 0.9), (0, 1, "peasant", 0.9),
               (1, 0, "bone", 0.9), (2, 0, "necromerger", 0.9)])
    t = best_feed_cell(b, feed_values={"bone": 1})
    check("G1e: stations/champions excluded", t is not None and t.item_id == "bone")
    check("G1f: nothing feedable -> None", best_feed_cell(
        board([(0, 0, "grave_lvl1", 0.9), (1, 0, "necromerger", 0.9)])) is None)

    # G2a: A (spawn-first) — on a sparse board with a grave that can spawn, the
    # heuristic SPAWNS instead of feed-to-progress (the observed bug: 10 empty,
    # full mana bar, bot fed skeleton_lvl1). The known-cheap trigger is
    # suspended while spawning to rebuild a cheap chain starter is possible.
    hp = HeuristicPlanner()
    hp.feed_values = {"bone": 1, "skeleton_lvl1": 10}
    g = full_board(occ={(0, 0): "bone", (0, 1): "ribcage",
                        (1, 0): "skeleton_lvl1", (2, 0): "grave_lvl1"})
    mv = hp.next_move(g)
    check("G2a: sparse board + grave -> spawn (feed-to-progress suspended)",
          mv.kind == "spawn" and mv.cell_a == (2, 0), f"{mv.kind} {mv.cell_a}")
    # G2a2: same sparse board WITHOUT a grave -> feed-to-progress still fires.
    mv2 = hp.next_move(full_board(occ={(0, 0): "bone", (0, 1): "ribcage",
                                       (1, 0): "skeleton_lvl1"}))
    check("G2a2: no grave -> feed-to-progress fires", mv2.kind == "feed"
          and mv2.cell_a == (0, 0), f"{mv2.kind} {mv2.cell_a}")
    # G2a3: B — merge-material protection still applies WITHIN the pool, but
    # generator protection (Aug 23) demoted max-level from "fed before
    # material" to last resort: the starvation guard now sacrifices the
    # CHEAPEST material (bone) rather than the 300-feed zombie generator.
    hp3 = HeuristicPlanner()
    hp3.feed_values = {"bone": 1, "skeleton_lvl1": 10, "zombie_lvl5": 300}
    hp3.max_level_ids = {"zombie_lvl5"}
    g3 = full_board(occ={(0, 0): "bone", (0, 1): "skeleton_lvl1",
                         (1, 0): "zombie_lvl5"})
    mv3 = hp3.next_move(g3)
    check("G2a3: starvation feeds cheapest material, not the generator",
          mv3.kind == "feed" and mv3.cell_a == (0, 0), f"{mv3.kind} {mv3.cell_a}")
    # G2a4: B — when EVERYTHING feedable is merge material (no grave, no
    # non-material target), material is sacrificed (last resort), cheapest first.
    hp4 = HeuristicPlanner()
    hp4.feed_values = {"bone": 1, "skeleton_lvl1": 10}
    g4 = full_board(occ={(0, 0): "bone", (0, 1): "skeleton_lvl1",
                         (1, 0): "ribcage"})
    mv4 = hp4.next_move(g4)
    check("G2a4: all-material board still feeds (last resort), cheapest first",
          mv4.kind == "feed" and mv4.cell_a == (0, 0), f"{mv4.kind} {mv4.cell_a}")
    # G2b: lone high-value single is NOT fed (kept for merge pipeline).
    hp2 = HeuristicPlanner()
    hp2.feed_values = {"skeleton_lvl3": 60}
    g = full_board(occ={(0, 0): "skeleton_lvl3", (1, 0): "grave_lvl1"})
    mv = hp2.next_move(g)
    check("G2b: lone high-value kept (spawn instead)", mv.kind == "spawn",
          f"{mv.kind} {mv.cell_a}")
    # G2c: GENERATOR PROTECTION — a lone max-level skeleton is NOT fed as
    # default progress food any more: with a grave + mana the heuristic
    # SPAWNS cheap food instead of sacrificing the mana generator.
    hp3 = HeuristicPlanner()
    hp3.feed_values = {"skeleton_lvl5": 300}
    hp3.max_level_ids = {"skeleton_lvl5"}
    g = full_board(occ={(0, 0): "skeleton_lvl5", (1, 0): "grave_lvl1"})
    mv = hp3.next_move(g)
    check("G2c: lone max-level kept; spawn cheap food instead",
          mv.kind == "spawn" and mv.cell_a == (1, 0),
          f"{mv.kind} {mv.cell_a}")
    # G2c2: same lone generator with NO grave and NO merge -> starvation guard
    # still feeds it (never softlock).
    mv = hp3.next_move(full_board(occ={(0, 0): "skeleton_lvl5"}))
    check("G2c2: no merge no spawn -> starvation feeds the generator",
          mv.kind == "feed" and mv.cell_a == (0, 0), f"{mv.kind} {mv.cell_a}")
    # G2d: back-compat — no feed_values, congested board still feeds (item_id key).
    hp4 = HeuristicPlanner()
    g = full_board(occ={(0, 0): "bone", (0, 1): "ribcage", (0, 2): "skeleton_lvl1",
                        (1, 0): "skeleton_lvl2", (1, 1): "zombie_lvl2",
                        (1, 2): "zombie_lvl1", (2, 0): "grave_lvl1",
                        (2, 1): "skeleton_lvl3", (2, 2): "skeleton_lvl4"})
    mv = hp4.next_move(g)
    check("G2d: congestion feed still works", mv.kind == "feed"
          and mv.cell_a == (0, 0), f"{mv.kind} {mv.cell_a}")

    # G3: vision-drive Best feed hint renders / stays empty.
    import tempfile
    from planner.vision_drive import VisionDrivenPlanner
    tmp = Path(tempfile.mkdtemp())
    pl = VisionDrivenPlanner(client=None, log=None, live=False, tool_enabled=False,
                             learnings_path=tmp / "learnings.md",
                             glossary_path=tmp / "glossary.md")
    gl = tmp / "glossary.md"
    gl.write_text(
        "## Item bone (popup)\n- feed value: 1\n"
        "## Item skeleton_lvl1 (popup)\n- feed value: 10\n")
    b = board([(0, 0, "bone", 0.9), (1, 0, "skeleton_lvl1", 0.9)])
    line = pl._best_feed_line(b)
    check("G3a: Best feed hint rendered", line == "Best feed: bone (0,0) (feed 1)",
          repr(line))
    # G3b: a merge available -> the hint follows the kind ordering (default
    #      weights: merge beats feed), so _best_move_hint shows the merge.
    b2 = board([(0, 0, "bone", 0.9), (0, 1, "bone", 0.9),
                (1, 0, "skeleton_lvl1", 0.9)])
    check("G3b: merge beats feed under default weights",
          pl._best_move_hint(b2).startswith("Best merge:"),
          repr(pl._best_move_hint(b2)))
    # G3c: lone high-value single -> no hint (mirrors heuristic gate). Uses the
    #      realistic full_board so congestion (empty_count<=3) doesn't trip.
    b3 = full_board(occ={(0, 0): "skeleton_lvl3"})
    check("G3c: lone high-value -> no hint", pl._best_feed_line(b3) == "",
          repr(pl._best_feed_line(b3)))
    # G3d: capacity known -> hint shows the fitted target + remaining.
    b4 = board([(0, 0, "bone", 0.9), (1, 0, "skeleton_lvl3", 0.9)])
    fl = pl._best_feed_line
    orig = pl._satiety_remaining
    pl._satiety_remaining = lambda frame=None: 10
    fl4 = pl._best_feed_line(b4)
    pl._satiety_remaining = orig
    check("G3d: capacity-limited hint", "skeleton_lvl3" not in fl4
          and "bone" in fl4 and "(fits 10 remaining)" in fl4, repr(fl4))
    # G3e: capacity 0 -> no feed hint at all (bar full).
    pl._satiety_remaining = lambda frame=None: 0
    fl5 = pl._best_feed_line(b4)
    pl._satiety_remaining = orig
    check("G3e: full bar -> no feed hint", fl5 == "", repr(fl5))


def part_h():
    """Satiety-aware feeding + damage-value capture (Aug 13 cont.)."""
    from planner.agent import best_feed_cell
    from planner.glossary import read_damage_values

    fv = {"bone": 1, "skeleton_lvl1": 10, "skeleton_lvl2": 25}
    # H1a: within capacity the SMALLEST fitting value is fed — topping up with
    # the cheapest sacrifice preserves high-value / generative monsters (the
    # user's "don't always feed the highest monster" rule).
    b = board([(0, 0, "bone", 0.9), (1, 0, "skeleton_lvl1", 0.9)])
    t = best_feed_cell(b, feed_values=fv, remaining_satiety=10)
    check("H1a: smallest fit picked", t is not None and t.item_id == "bone",
          t.item_id if t else "None")
    # H1b: nothing feeds when the bar is full (remaining <= 0).
    check("H1b: full bar -> None",
          best_feed_cell(b, feed_values=fv, remaining_satiety=0) is None
          and best_feed_cell(b, feed_values=fv, remaining_satiety=-5) is None)
    # H1c: an over-cap item (skeleton_lvl2=25 > remaining 10) is excluded,
    #      the fitting bone is chosen instead.
    b = board([(0, 0, "bone", 0.9), (1, 0, "skeleton_lvl2", 0.9)])
    t = best_feed_cell(b, feed_values=fv, remaining_satiety=10)
    check("H1c: over-cap excluded", t is not None and t.item_id == "bone",
          t.item_id if t else "None")
    # H1d: craved item still preferred when it fits.
    b = board([(0, 0, "bone", 0.9), (1, 0, "skeleton_lvl1", 0.9)])
    t = best_feed_cell(b, feed_values=fv, remaining_satiety=10, craved_item="skeleton")
    check("H1d: craved preferred within capacity",
          t is not None and t.item_id == "skeleton_lvl1")
    # H1e: unknown-value items are held back when capacity is known.
    b = board([(0, 0, "bone", 0.9), (1, 0, "zombie_lvl2", 0.9)])
    t = best_feed_cell(b, feed_values=fv, remaining_satiety=5)
    check("H1e: unknown-value held back when capacity known",
          t is not None and t.item_id == "bone")
    # H1f: unknown capacity -> old (lowest-value) ordering unchanged.
    b = board([(0, 0, "bone", 0.9), (1, 0, "skeleton_lvl1", 0.9)])
    t = best_feed_cell(b, feed_values=fv)
    check("H1f: unknown capacity unchanged", t is not None and t.item_id == "bone")
    # H1g: nothing that fits -> None (not a wasteful partial fill).
    b = board([(0, 0, "skeleton_lvl2", 0.9)])
    check("H1g: nothing fits -> None",
          best_feed_cell(b, feed_values=fv, remaining_satiety=10) is None)

    import tempfile
    from planner.vision_drive import VisionDrivenPlanner
    tmp = Path(tempfile.mkdtemp())
    pl = VisionDrivenPlanner(client=None, log=None, live=False, tool_enabled=False,
                             learnings_path=tmp / "learnings.md",
                             glossary_path=tmp / "glossary.md")
    gl = tmp / "glossary.md"
    gl.write_text(
        "## Item skeleton_lvl1 (popup)\n- feed value: 10\n- damage value: 4\n"
        "## Item bone (popup)\n- feed value: 1\n")
    check("H2a: read_damage_values parses popup blocks",
          read_damage_values(gl) == {"skeleton_lvl1": 4}, str(read_damage_values(gl)))
    check("H2b: _damage_values reads the glossary",
          pl._damage_values() == {"skeleton_lvl1": 4}, str(pl._damage_values()))
    b = board([(0, 0, "skeleton_lvl1", 0.9)])
    text = pl._board_state_text(b)
    check("H2c: board text renders (dmg N)",
          "(0,0) skeleton_lvl1 (feed 10) (dmg 4) (score" in text, repr(text[:160]))
    # H3: HeuristicPlanner passes satiety_remaining through to best_feed_cell.
    hp = HeuristicPlanner()
    hp.feed_values = {"bone": 1, "skeleton_lvl2": 25}
    hp.satiety_remaining = 10
    # No grave on this board, so spawn isn't possible and the capacity-limited
    # feed actually runs: only bone (feed 1) fits the 10 remaining.
    g = full_board(occ={(0, 0): "bone", (1, 0): "skeleton_lvl2"})
    mv = hp.next_move(g)
    check("H3: heuristic capacity-limited feed", mv.kind == "feed"
          and mv.cell_a == (0, 0), f"{mv.kind} {mv.cell_a}")
    # H3c: A — sparse board + grave + full-ish bar: spawn (no feed, no fit).
    hp2 = HeuristicPlanner()
    hp2.feed_values = {"bone": 1}
    hp2.satiety_remaining = 0
    g2 = full_board(occ={(0, 0): "bone", (2, 0): "grave_lvl1"})
    mv2b = hp2.next_move(g2)
    check("H3c: sparse board + grave -> spawn over a no-fit feed",
          mv2b.kind == "spawn", f"{mv2b.kind}")
    hp.satiety_remaining = 0
    mv2 = hp.next_move(full_board(occ={(0, 0): "bone", (2, 0): "grave_lvl1"}))
    check("H3b: full bar -> heuristic does not feed", mv2.kind != "feed",
          f"{mv2.kind}")


def part_j():
    """Feat-driven priorities (new): parse / weights / kind ordering / the
    heuristic AND the fallback actually respecting them / the vision-drive
    hint following the same ordering."""
    # J1: parse_obj classifies feat text (merge/feed/spawn/collect/neutral).
    m = parse_obj({"name": "Merge things 50 times.", "progress": "49/50"})
    f = parse_obj({"name": "Feed the Devourer a bone", "progress": "2/5"})
    s = parse_obj({"name": "Spawn 10 creatures", "progress": "3/10"})
    c = parse_obj({"name": "Collect mana 5 times", "progress": "1/5"})
    n = parse_obj({"name": "Spend 500 coins", "progress": "0/500"})
    check("J1: parse kinds", m.kind == "merge" and f.kind == "feed"
          and s.kind == "spawn" and c.kind == "collect" and n.kind == "neutral",
          f"{m.kind}/{f.kind}/{s.kind}/{c.kind}/{n.kind}")
    check("J1b: ratios parsed", abs(m.ratio - 0.98) < 1e-6
          and abs(f.ratio - 0.4) < 1e-6, f"{m.ratio:.3f} {f.ratio:.3f}")

    # J2: done feats -> neutral, no weight effect.
    done = parse_obj({"name": "Merge things 50 times.", "progress": "49/50",
                      "done": True})
    check("J2: done -> neutral", done.kind == "neutral", done.kind)

    # J3: ratio boost — nearly-done feat (49/50) outranks a 0/N one across kinds.
    w = feat_weights([parse_obj({"name": "Merge things 50 times.", "progress": "49/50"})])
    check("J3: merge 49/50 boosted", w["merge"] > BASE_KIND_WEIGHT["merge"],
          f"{w['merge']:.2f}")

    # J4: craving folds in as a feed objective carrying CRAVING_BONUS_WEIGHT
    #     (Unlock the Mana Well = feed objective too).
    w_crave = feat_weights([], craving_objective("rib cage", 0, 2))
    check("J4: craving boosts feed", w_crave["merge"] < w_crave["feed"],
          f"merge {w_crave['merge']:.2f} feed {w_crave['feed']:.2f}")
    w_mana = feat_weights([parse_obj({"name": "Unlock the Mana Well",
                                     "progress": "1/4 Feats Completed"})])
    check("J4b: Unlock ... = feed objective", w_mana["feed"] > BASE_KIND_WEIGHT["feed"],
          f"{w_mana['feed']:.2f}")

    # J5: kind_order — no weights -> base order (attack, merge, feed,
    #     spawn, collect) — attack outranks the rest so a champion is
    #     always handled ASAP when no feats are active. Weights override,
    #     but the base 3.0 attack weight is a strong floor: only feats
    #     whose ratio boost crosses 3.0 (or adds on top) overtake it.
    check("J5: base order includes attack first",
          kind_order(None) == ["attack", "merge", "feed", "spawn", "collect"],
          str(kind_order(None)))
    # A 0/2 craving (no progress) only adds 0.5 to feed (CRAVING_BONUS_WEIGHT
    # is 0.6) -> feed ends at 2.1, still under attack's 3.0.
    check("J5b: weak craving boost < attack weight -> attack still first",
          kind_order(w_crave) == ["attack", "feed", "merge", "spawn", "collect"],
          str(kind_order(w_crave)))
    # A 49/50 merge feat adds 49/50 = 0.98 to merge -> 2.98, still under 3.0.
    check("J5c: strong-but-not-max merge feat < attack weight -> attack first",
          kind_order(w) == ["attack", "merge", "feed", "spawn", "collect"],
          str(kind_order(w)))

    # J6: HeuristicPlanner branch order follows kind_order. Fresh requires the
    #     goal to outrank merging on the same board.
    # S1: the craving is folded into feat_weights -> feed outranks merge on a
    #     mergeable board.
    b = board([(3, 1, "ribcage", 0.9), (3, 2, "ribcage", 0.9),
               (2, 1, "skeleton_lvl1", 0.9), (4, 1, "grave_lvl1", 0.9)])
    pl2 = HeuristicPlanner()
    pl2.craved_item = "rib cage"
    pl2.feed_values = {"bone": 1, "ribcage": 3, "skeleton_lvl1": 10}
    pl2.feat_weights = feat_weights([], craving_objective("rib cage", 0, 2))
    mv2 = pl2.next_move(b)
    check("J6: craving outranks merge -> feed", mv2.kind == "feed"
          and mv2.cell_a in ((3, 1), (3, 2)), f"{mv2.kind} {mv2.cell_a}")

    # S3: a nearly-done merge feat outranks the craving feed.
    pl3 = HeuristicPlanner()
    pl3.craved_item = "rib cage"
    pl3.feed_values = {"bone": 1, "ribcage": 3, "skeleton_lvl1": 10}
    pl3.feat_weights = feat_weights(
        [parse_obj({"name": "Merge things 50 times.", "progress": "49/50"})],
        craving_objective("rib cage", 0, 2))
    mv3 = pl3.next_move(b)
    check("J6b: near-done merge feat outranks craving", mv3.kind == "merge",
          f"{mv3.kind}")

    # J6c: no feats/craving -> base order: merge still wins on a mergeable board.
    pl5 = HeuristicPlanner()
    pl5.feed_values = {"bone": 1, "ribcage": 3, "skeleton_lvl1": 10}
    mv5 = pl5.next_move(b)
    check("J6c: base order merge wins", mv5.kind == "merge", f"{mv5.kind}")

    # J7: an active feed objective -> heuristic feeds even without a merge.
    pl4 = HeuristicPlanner()
    pl4.feed_objective_active = True
    pl4.feed_values = {"bone": 1, "ribcage": 3, "skeleton_lvl1": 10}
    b4 = board([(0, 0, "bone", 0.9), (4, 1, "grave_lvl1", 0.9)])
    mv4 = pl4.next_move(b4)
    check("J7: active feed objective -> feed without merge", mv4.kind == "feed"
          and mv4.cell_a == (0, 0), f"{mv4.kind} {mv4.cell_a}")

    # J8: name normalization connects the craving display name to board ids.
    check("J8: normalize rib cage -> ribcage",
          normalize_item_name("rib cage") == "ribcage"
          and item_name_matches("ribcage", "rib cage")
          and item_name_matches("skeleton_lvl1", "skeleton")
          and not item_name_matches("ribcage", "skeleton"),
          f"{normalize_item_name('rib cage')}")

    # J9: the vision-drive hint follows the SAME kind ordering as the
    #     heuristic (feed wins under a craving even with a merge available).
    import tempfile
    from planner.vision_drive import VisionDrivenPlanner
    tmp = Path(tempfile.mkdtemp())
    gl = tmp / "g.md"
    gl.write_text("## Item ribcage (popup)\n- feed value: 3\n"
                  "## Item bone (popup)\n- feed value: 1\n")
    pl = VisionDrivenPlanner(client=None, log=None, live=False,
                             tool_enabled=False,
                             learnings_path=tmp / "l.md", glossary_path=gl)
    pl.fallback.feed_values = pl._feed_values()
    pl.fallback.craved_item = None
    b = board([(3, 1, "ribcage", 0.9), (3, 2, "ribcage", 0.9),
               (2, 1, "skeleton_lvl1", 0.9), (4, 1, "grave_lvl1", 0.9)])
    check("J9a: no craving -> merge hint", pl._best_move_hint(b).startswith("Best merge"),
          repr(pl._best_move_hint(b)))
    pl.fallback.craved_item = "rib cage"
    check("J9b: craving -> feed hint beats merge", pl._best_move_hint(b).startswith("Best feed"),
          repr(pl._best_move_hint(b)))
    pl.fallback.craved_item = None
    pl._feats_cache = {"tier": 3, "feats": [
        {"name": "Merge things 50 times.", "progress": "49/50", "done": False}],
        "step": 0}
    check("J9c: near-done merge feat outranks base -> merge hint",
          pl._best_move_hint(b).startswith("Best merge"),
          repr(pl._best_move_hint(b)))


def part_k():
    """Overflow-aware craved feeding + spawn-priority (user rule, Aug 14):
    a craved feed is refused when Z + craving_bonus > R + 10%*Y; the fallback
    then PREFERS spawning to rebuild the (grave-chain) craved item; feeding a
    near-full bar top-up uses the SMALLEST fitting sacrifice (never the
    highest monster); a starvation guard unblocks a softlocked near-full bar;
    the craving bonus is learned live from satiety deltas."""
    from planner.agent import best_feed_cell, HeuristicPlanner
    from planner.vision_drive import VisionDrivenPlanner
    fv = {"bone": 1, "ribcage": 3, "skeleton_lvl1": 10, "skeleton_lvl3": 60}
    Y = 250
    tol = int(0.10 * Y)  # 25

    # K1: craved feed accepted when it fits within the tolerance (bonus known).
    b = board([(0, 0, "bone", 0.9), (1, 0, "skeleton_lvl1", 0.9)])
    t = best_feed_cell(b, feed_values=fv, craved_item="skeleton",
                       remaining_satiety=100, satiety_capacity=Y,
                       craving_bonus=50)
    check("K1: craved within tolerance fed", t is not None
          and t.item_id == "skeleton_lvl1", t.item_id if t else "None")
    # K2: craved feed refused when Z + bonus overflows past the tolerance.
    b = board([(1, 0, "skeleton_lvl3", 0.9)])
    t = best_feed_cell(b, feed_values=fv, craved_item="skeleton",
                       remaining_satiety=20, satiety_capacity=Y,
                       craving_bonus=50)
    check("K2: wasteful craved refused", t is None, str(t))
    # K2b: refused even without a recorded bonus (Z alone overflows).
    check("K2b: wasteful craved refused without bonus",
          best_feed_cell(b, feed_values=fv, craved_item="skeleton",
                         remaining_satiety=20, satiety_capacity=Y) is None)
    # K2c: a cheap craved item is still fed into a near-full bar (fits within
    #      tolerance, so feeding it wastes nothing beyond the 10% envelope).
    b = board([(0, 0, "bone", 0.9)])
    t = best_feed_cell(b, feed_values=fv, craved_item="bone",
                       remaining_satiety=30, satiety_capacity=Y,
                       craving_bonus=50)
    check("K2c: cheap craved fed near-full", t is not None
          and t.item_id == "bone", t.item_id if t else "None")
    # K3: heuristic PREFERS spawning when the only matching craved item is
    #     wasteful + a grave + empty cells exist (spawn beats a cheap top-up).
    #     Level-aware: only the EXACT-level craved monster counts, so the board
    #     holds skeleton_lvl3 and the craving is skeleton L3 (that's the cell
    #     that has to survive to rebuild — a level-ALL/unknown match would
    #     wrongly apply the rule to the wrong monster).
    hp = HeuristicPlanner()
    hp.craved_item = "skeleton"
    hp.craved_level = 3
    hp.feed_values = fv
    hp.satiety_remaining = 20
    hp.satiety_capacity = Y
    g = full_board(occ={(1, 0): "skeleton_lvl3", (0, 0): "bone",
                        (2, 0): "grave_lvl1"})
    mv = hp.next_move(g)
    check("K3: wasteful craved -> spawn instead of feed", mv.kind == "spawn",
          f"{mv.kind} {mv.cell_a}")
    # K3b: wasteful craved + NO spawn -> starvation feed of the cheapest (a
    #      softlocked near-full bar levels up instead of idling forever).
    hp2 = HeuristicPlanner()
    hp2.craved_item = "skeleton"
    hp2.craved_level = 3
    hp2.feed_values = fv
    hp2.satiety_remaining = 20
    hp2.satiety_capacity = Y
    mv2 = hp2.next_move(full_board(occ={(1, 0): "skeleton_lvl3"}))
    check("K3b: starvation guard feeds the only feedable", mv2.kind == "feed"
          and mv2.cell_a == (1, 0), f"{mv2.kind} {mv2.cell_a}")
    # K3c: wasteful craved + no grave + a fitting bone -> cheap top-up feed.
    hp3 = HeuristicPlanner()
    hp3.craved_item = "skeleton"
    hp3.craved_level = 3
    hp3.feed_values = fv
    hp3.satiety_remaining = 20
    hp3.satiety_capacity = Y
    mv3 = hp3.next_move(full_board(occ={(1, 0): "skeleton_lvl3",
                                        (0, 0): "bone"}))
    check("K3c: no spawn -> cheapest top-up feed", mv3.kind == "feed"
          and mv3.cell_a == (0, 0), f"{mv3.kind} {mv3.cell_a}")

    # K4: craving bonus B is LEARNED from the satiety delta across a craved
    #     feed (1/250 -> 61/250, recorded skeleton feed value 10 -> B = 50).
    class _FakeReader:
        def __init__(self, num, den):
            self.num, self.den = num, den

        def read_satiety(self, frame):
            return {"num": self.num, "den": self.den, "text": f"{self.num}/{self.den}"}

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    pl = VisionDrivenPlanner(client=None, log=None, live=False,
                             tool_enabled=False,
                             learnings_path=tmp / "l.md",
                             glossary_path=tmp / "g.md",
                             satiety_reader=_FakeReader("61", "250"))
    pl._last_satiety = (1, 250)
    pl._prev_feed_was_craved = True
    pl._prev_feed_value = 10
    pl._satiety_context(frame=object())
    check("K4: bonus learned from satiety delta", pl.fallback.craving_bonus_est == 50,
          f"bonus={pl.fallback.craving_bonus_est}")
    check("K4b: remaining/capacity cached", pl.fallback.satiety_remaining == 189
          and pl.fallback.satiety_capacity == 250,
          f"{pl.fallback.satiety_remaining}/{pl.fallback.satiety_capacity}")
    check("K4c: satiety tracked + feed context cleared",
          pl._last_satiety == (61, 250)
          and pl._prev_feed_was_craved is False and pl._prev_feed_value is None)
    # K4d: a level-up between reads (den changes) never corrupts the bonus.
    pl2 = VisionDrivenPlanner(client=None, log=None, live=False,
                              tool_enabled=False,
                              learnings_path=tmp / "l2.md",
                              glossary_path=tmp / "g2.md",
                              satiety_reader=_FakeReader("5", "300"))
    pl2._last_satiety = (61, 250)
    pl2._prev_feed_was_craved = True
    pl2._prev_feed_value = 10
    pl2._satiety_context(frame=object())
    check("K4d: den change -> no bonus learned", pl2.fallback.craving_bonus_est == 0,
          f"bonus={pl2.fallback.craving_bonus_est}")
    # K4e: _record_feed_context flags a craved feed for next-step learning.
    pl3 = VisionDrivenPlanner(client=None, log=None, live=False,
                              tool_enabled=False,
                              learnings_path=tmp / "l3.md",
                              glossary_path=tmp / "g3.md")
    pl3.fallback.craved_item = "skeleton"
    pl3.fallback.feed_values = fv
    b = board([(0, 0, "skeleton_lvl1", 0.9), (2, 0, "grave_lvl1", 0.9)])
    pl3._record_feed_context(Move(kind="feed", cell_a=(0, 0)), b)
    ok = pl3._prev_feed_was_craved and pl3._prev_feed_value == 10
    pl3._record_feed_context(Move(kind="spawn", cell_a=(2, 0)), b)
    ok = ok and not pl3._prev_feed_was_craved and pl3._prev_feed_value is None
    check("K4e: feed context recorded only for craved feeds", ok,
          f"craved={pl3._prev_feed_was_craved} value={pl3._prev_feed_value}")

    # K5: hint layer mirrors the fallback — wasteful craved -> Best spawn;
    #     craved fits -> Best feed.
    gl = tmp / "gloss.md"
    gl.write_text(
        "## Item skeleton_lvl3 (popup)\n- feed value: 60\n"
        "## Item bone (popup)\n- feed value: 1\n"
        "## Item skeleton_lvl1 (popup)\n- feed value: 10\n")
    pl5 = VisionDrivenPlanner(client=None, log=None, live=False,
                              tool_enabled=False,
                              learnings_path=tmp / "l5.md", glossary_path=gl)
    pl5.fallback.feed_values = pl5._feed_values()
    g5 = full_board(occ={(1, 0): "skeleton_lvl3", (2, 0): "grave_lvl1"})
    pl5.fallback.craved_item = "skeleton"
    pl5.fallback.satiety_remaining = 20
    pl5.fallback.satiety_capacity = Y
    check("K5: hint = Best spawn when craved wasteful",
          pl5._best_move_hint(g5).startswith("Best spawn:"),
          repr(pl5._best_move_hint(g5)))
    pl5.fallback.satiety_remaining = 100
    g5b = full_board(occ={(0, 0): "skeleton_lvl1", (1, 0): "bone",
                          (2, 0): "grave_lvl1"})
    check("K5b: hint = Best feed when craved fits", pl5.fallback.feed_values
          and pl5._best_move_hint(g5b).startswith("Best feed:"),
          repr(pl5._best_move_hint(g5b)))


def part_l():
    """Level-aware craving matching + craving-cache invalidation (Aug 14 user
    rules): a skeleton_lvl1 craving must NEVER steer a skeleton_lvl5 feed
    (`skeleton_lvl5` is simply not the craved monster), only the exact-level
    feed counts 1:1 toward completion, and a completing feed (or a level-up)
    invalidates the cached craving so get_cravings re-reads next step — the
    bot can't keep believing a completed craving for the cache lifetime."""
    from planner.agent import best_feed_cell
    from planner.vision_drive import VisionDrivenPlanner
    fv = {"bone": 1, "ribcage": 3, "skeleton_lvl1": 10, "skeleton_lvl5": 700}
    Y = 250

    # L1: craved_matches is level-specific — a L1 craving does NOT match L5.
    from planner.constants import craved_matches
    check("L1a: L5 not the craved cell for L1 craving", not
          craved_matches("skeleton_lvl5", "skeleton", craving_level=1),
          "skeleton_lvl5 matched L1 craving")
    check("L1b: L1 is the craved cell", craved_matches(
        "skeleton_lvl1", "skeleton", craving_level=1),
        "skeleton_lvl1 not matched")
    check("L1c: unleveled never matches a level-known craving",
          not craved_matches("bone", "skeleton", craving_level=1))
    check("L1d: unknown craving level -> low-tier only (L5 excluded, L2 ok)",
          not craved_matches("skeleton_lvl5", "skeleton")
          and craved_matches("skeleton_lvl2", "skeleton"),
          "unknown-level gate broken")

    # L2: best_feed_cell never picks the L5 as the craved feed for a L1 craving.
    b = board([(0, 0, "skeleton_lvl5", 0.9), (1, 0, "skeleton_lvl1", 0.9)])
    t = best_feed_cell(b, feed_values=fv, craved_item="skeleton",
                       craved_level=1)
    check("L2: L1 craving feeds the L1, not the L5", t is not None
          and t.item_id == "skeleton_lvl1", t.item_id if t else "None")

    # L3: heuristic feed branch uses the level-aware match (L5 alone is NOT a
    #     craved feed). With no merge and no spawn possible the STARVATION
    #     guard feeds it (anti-softlock; generator protection is a priority,
    #     not a ban) — but it must never be fed AS the craving.
    hp = HeuristicPlanner()
    hp.craved_item = "skeleton"
    hp.craved_level = 1
    hp.feed_values = fv
    g = full_board(occ={(0, 0): "skeleton_lvl5"})
    mv = hp.next_move(g)
    check("L3: lone L5 fed only via starvation guard (not as craving)",
          mv.kind == "feed" and mv.cell_a == (0, 0),
          f"{mv.kind} {mv.cell_a if mv.cell_a else ''}")

    # L4: _record_feed_context flags the craved feed only for the EXACT level
    #     (and completion-invalidates the cache when this feed completes it).
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    pl = VisionDrivenPlanner(client=None, log=None, live=False,
                             tool_enabled=False,
                             learnings_path=tmp / "l.md",
                             glossary_path=tmp / "g.md")
    pl.fallback.craved_item = "skeleton"
    pl.fallback.craved_level = 1
    pl.fallback.feed_values = fv
    pl._craving_cache = {"item": "skeleton", "level": 1, "count_done": 1,
                         "count_required": 2, "reward": 100, "step": 3}
    b = board([(0, 0, "skeleton_lvl5", 0.9), (1, 0, "skeleton_lvl1", 0.9)])
    pl._record_feed_context(Move(kind="feed", cell_a=(0, 0)), b)
    check("L4a: L5 feed not flagged as craved", not pl._prev_feed_was_craved
          and pl._prev_feed_value is None and pl._craving_cache is not None,
          f"craved={pl._prev_feed_was_craved} cache={pl._craving_cache}")
    pl._record_feed_context(Move(kind="feed", cell_a=(1, 0)), b)
    check("L4b: L1 feed flagged + completing feed invalidates cache",
          pl._prev_feed_was_craved and pl._prev_feed_value == 10
          and pl._craving_cache is None,
          f"craved={pl._prev_feed_was_craved} cache={pl._craving_cache}")

    # L4c: a non-completing craved feed keeps the cache (count_done 1/7).
    pl2 = VisionDrivenPlanner(client=None, log=None, live=False,
                              tool_enabled=False,
                              learnings_path=tmp / "l2.md",
                              glossary_path=tmp / "g2.md")
    pl2.fallback.craved_item = "skeleton"
    pl2.fallback.craved_level = 1
    pl2.fallback.feed_values = fv
    pl2._craving_cache = {"item": "skeleton", "level": 1, "count_done": 1,
                          "count_required": 7, "reward": 150, "step": 3}
    b2 = board([(1, 0, "skeleton_lvl1", 0.9)])
    pl2._record_feed_context(Move(kind="feed", cell_a=(1, 0)), b2)
    check("L4c: non-completing craved feed keeps cache",
          pl2._prev_feed_was_craved and pl2._craving_cache is not None,
          f"cache={pl2._craving_cache}")

    # L4d: a get_cravings READ that observes the craving completed also drops
    #      the cache (the game is about to pick a new craving) — the completed
    #      state must not be believed for the cache lifetime.
    pl4 = VisionDrivenPlanner(client=None, log=None, live=False,
                              tool_enabled=False,
                              learnings_path=tmp / "l4.md",
                              glossary_path=tmp / "g4.md")
    pl4._craving_cache = {"item": "skeleton", "level": 1, "count_done": 1,
                          "count_required": 2, "reward": 100, "step": 1}
    pl4._track_cravings({"cravings": [
        {"item": "skeleton", "level": 1, "count_done": 2,
         "count_required": 2, "reward": 100}]})
    check("L4d: completed craving read drops the cache",
          pl4._craving_cache is None, f"cache={pl4._craving_cache}")

    # L5: invalidate_cravings drops the cached read (level-up hook) and makes
    #     the freshness gate stale.
    pl3 = VisionDrivenPlanner(client=None, log=None, live=False,
                              tool_enabled=False,
                              learnings_path=tmp / "l3.md",
                              glossary_path=tmp / "g3.md")
    pl3._last_craving = "skeleton"   # bubble still matches the cached item
    pl3._last_craving_level = 1      # level key matches the cached read
    pl3._craving_cache = {"item": "skeleton", "level": 1, "count_done": 1,
                          "count_required": 2, "reward": 100, "step": 3}
    check("L5a: cache fresh before invalidation", pl3._craving_cache_fresh())
    pl3.invalidate_cravings()
    check("L5b: cache dropped by invalidate_cravings",
          pl3._craving_cache is None and not pl3._craving_cache_fresh(),
          f"cache={pl3._craving_cache}")
    # Base planner invalidate_cravings is a no-op (exists, returns None).
    hp2 = HeuristicPlanner()
    check("L5c: base invalidate_cravings no-op", hp2.invalidate_cravings() is None)

    # L6: _best_feed_line never tags the L5 as (craved) for a L1 craving, and
    #     tags the L1 when it fits.
    gl = tmp / "gloss.md"
    gl.write_text(
        "## Item skeleton_lvl1 (popup)\n- feed value: 10\n"
        "## Item skeleton_lvl5 (popup)\n- feed value: 700\n")
    pl6 = VisionDrivenPlanner(client=None, log=None, live=False,
                              tool_enabled=False,
                              learnings_path=tmp / "l6.md", glossary_path=gl)
    pl6.fallback.feed_values = pl6._feed_values()
    pl6.fallback.craved_item = "skeleton"
    pl6.fallback.craved_level = 1
    g5 = full_board(occ={(0, 0): "skeleton_lvl5", (1, 0): "skeleton_lvl1"})
    line = pl6._best_feed_line(g5)
    check("L6a: L1 craved hint tags the L1", "skeleton_lvl1 (1,0)" in line
          and "(craved)" in line, repr(line))
    check("L6b: L5 never (craved)", "skeleton_lvl5 (0,0)" not in line,
          repr(line))


def part_m():
    """M. Craved-chain protection (Aug 17 fix): while an exact-level craving is
    active, merge-chain PRECURSORS of the craved monster (same family, level
    below the craving's level, incl. the bone/ribcage bases) are never fed —
    they are the recipe to BUILD the craving. Exact-craved (level == craving)
    still feeds; above-craving-level (can't merge down) still feeds; level-
    unknown (`craved_level=None`) skips protection. Plus: the level-aware
    craving-cache freshness key (same-family level switch forces a re-read)."""
    fv = {"bone": 1, "ribcage": 3, "skeleton_lvl1": 10,
          "skeleton_lvl2": 25, "skeleton_lvl3": 60, "skeleton_lvl5": 700}
    chain = {"skeleton": ["bone", "ribcage", "skeleton_lvl1",
                          "skeleton_lvl2", "skeleton_lvl3"]}

    # M1: is_craved_precursor classification.
    check("M1a: skeleton_lvl1 precursor of skeleton L3",
          is_craved_precursor("skeleton_lvl1", "skeleton", 3, chain)
          and is_craved_precursor("skeleton_lvl2", "skeleton", 3, chain),
          "leveled precursors")
    check("M1b: bone/ribcage bases are precursors via chain_map",
          is_craved_precursor("bone", "skeleton", 3, chain)
          and is_craved_precursor("ribcage", "skeleton", 3, chain),
          "unleveled bases")
    check("M1c: exact-craved level is NOT a precursor",
          not is_craved_precursor("skeleton_lvl3", "skeleton", 3, chain),
          "exact level")
    check("M1d: above-craving level is NOT a precursor",
          not is_craved_precursor("skeleton_lvl5", "skeleton", 3, chain),
          "above level")
    check("M1e: unknown craving level -> nothing protected",
          not is_craved_precursor("skeleton_lvl1", "skeleton", None, chain)
          and not is_craved_precursor("bone", "skeleton", None, chain),
          "level None")
    b = board([(0, 0, "skeleton_lvl3", 0.9)])
    check("M1f: family match via level suffix alone (no chain_map) protects lvl1",
          is_craved_precursor("skeleton_lvl1", "skeleton", 2, None)
          and not is_craved_precursor("bone", "skeleton", 2, None),
          "suffix-only misses bone")

    # M2: best_feed_cell never selects a precursor under an active craving.
    g = full_board(occ={(0, 0): "bone", (0, 1): "ribcage",
                        (1, 0): "skeleton_lvl1", (1, 1): "skeleton_lvl3"})
    t = best_feed_cell(g, feed_values=fv, craved_item="skeleton",
                       craved_level=3, chain_map=chain)
    check("M2a: exact-craved skel_lvl3 picked, precursors excluded",
          t is not None and t.item_id == "skeleton_lvl3",
          t.item_id if t else "None")
    # With NO exact-craved cell present, the precursors are still protected:
    # the only remaining feedable is above the crave level (lvl5) -> fed.
    g = full_board(occ={(0, 0): "bone", (0, 1): "ribcage",
                        (1, 0): "skeleton_lvl1", (2, 0): "skeleton_lvl5"})
    t = best_feed_cell(g, feed_values=fv, craved_item="skeleton",
                       craved_level=3, chain_map=chain)
    check("M2b: precursors protected, lvl5 still fed (above crave)",
          t is not None and t.item_id == "skeleton_lvl5",
          t.item_id if t else "None")
    # All-precursor board + no above-crave cell -> nothing to feed (None).
    g = full_board(occ={(0, 0): "bone", (1, 0): "skeleton_lvl1",
                        (1, 1): "skeleton_lvl2"})
    check("M2c: all-precursor board -> no feed",
          best_feed_cell(g, feed_values=fv, craved_item="skeleton",
                         craved_level=3, chain_map=chain) is None)
    # Level-unknown craving keeps the container behavior (no protection):
    # the unknown-level gate still treats lvl1/lvl2 as possibly-craved, and
    # the craved branch wins over the cheapest (skeleton_lvl1 feed 10 vs bone 1).
    g = full_board(occ={(0, 0): "bone", (1, 0): "skeleton_lvl1"})
    t = best_feed_cell(g, feed_values=fv, craved_item="skeleton",
                       craved_level=None, chain_map=chain)
    check("M2d: level-unknown craving -> old behavior (possibly-craved lvl1)",
          t is not None and t.item_id == "skeleton_lvl1",
          t.item_id if t else "None")
    # Without chain_map, the suffix match protects leveled precursors only;
    # unleveled bases (bone/ribcage) can't be connected to 'skeleton' -> among
    # the remaining feedable the cheapest feeds.
    g = full_board(occ={(0, 0): "bone", (0, 1): "ribcage",
                        (1, 0): "skeleton_lvl1"})
    t = best_feed_cell(g, feed_values=fv, craved_item="skeleton",
                       craved_level=3, chain_map=None)
    check("M2e: without chain_map, leveled precursor excluded, unleveled bases remain",
          t is not None and t.item_id == "bone",
          t.item_id if t else "None")

    # M3: heuristic feed branch under an active craving never feeds a precursor
    #     when a rebuild path exists — with a grave + empty cells it SPAWNS
    #     instead of feeding bone/ribcage/skeleton_lvl1 into the Devourer.
    hp = HeuristicPlanner()
    hp.craved_item = "skeleton"
    hp.craved_level = 3
    hp.feed_values = fv
    hp.chain_map = chain
    hp.satiety_remaining = 100  # plenty of room so capacity isn't the blocker
    g = full_board(occ={(0, 0): "bone", (0, 1): "ribcage",
                        (1, 0): "skeleton_lvl1", (2, 1): "grave_lvl2"})
    mv = hp.next_move(g)
    check("M3: heuristic spawns (not feeds a precursor) when a grave is open",
          mv.kind == "spawn" and mv.cell_a == (2, 1),
          f"{mv.kind} {mv.cell_a}")
    # Congestion relief must not override the protection either.
    g = full_board(occ={(0, 0): "bone", (0, 1): "ribcage",
                        (1, 0): "skeleton_lvl1"})
    hp2 = HeuristicPlanner()
    hp2.craved_item = "skeleton"
    hp2.craved_level = 3
    hp2.feed_values = fv
    hp2.chain_map = chain
    hp2.satiety_remaining = 100
    hp2.satiety_capacity = 250
    hp2.feat_weights = None
    # excerpt the feed branch decision (congestion triggers _should_feed, but the
    # only feedable cells are precursors -> best_feed_cell returns None above).
    g2 = full_board(occ={(0, 0): "bone", (0, 1): "ribcage",
                         (1, 0): "skeleton_lvl1"})
    t = best_feed_cell(g2, feed_values=fv, craved_item="skeleton",
                       craved_level=3, chain_map=chain,
                       remaining_satiety=100, satiety_capacity=250)
    check("M3b: congestion cannot resurrect a precursor feed",
          t is None, t.item_id if t else "None")

    # M4: starvation guard still feeds when EVERYTHING is a precursor and no
    #     merge/spawn exists (a softlock is worse than one recipe step).
    hp3 = HeuristicPlanner()
    hp3.craved_item = "skeleton"
    hp3.craved_level = 3
    hp3.feed_values = fv
    hp3.chain_map = chain
    hp3.satiety_remaining = 100
    g = full_board(occ={(0, 0): "bone"})
    mv = hp3.next_move(g)
    check("M4: all-precursor board still reaches the starvation feed",
          mv.kind == "feed" and mv.cell_a == (0, 0),
          f"{mv.kind} {mv.cell_a}")

    # M5: LLMPlanner._ranked_feed_cells excludes precursors from the menu.
    from planner.llm import LLMPlanner
    llmpl = LLMPlanner(client=None, live=False)
    llmpl.fallback.craved_item = "skeleton"
    llmpl.fallback.craved_level = 3
    llmpl.fallback.chain_map = chain
    g = full_board(occ={(0, 0): "bone", (0, 1): "ribcage",
                        (1, 0): "skeleton_lvl1", (1, 1): "skeleton_lvl3"})
    cells = llmpl._ranked_feed_cells(g, set())
    check("M5: LLM feed menu contains the exact-craved, never precursors",
          len(cells) == 1 and cells[0].item_id == "skeleton_lvl3",
          ", ".join(c.item_id for c in cells) or "empty")

    # M6: vision-drive level-aware craving-cache freshness key.
    import tempfile
    from planner.vision_drive import VisionDrivenPlanner
    tmp = Path(tempfile.mkdtemp())
    pl = VisionDrivenPlanner(client=None, log=None, live=False,
                             tool_enabled=False,
                             learnings_path=tmp / "m.md",
                             glossary_path=tmp / "gm.md")
    pl._last_craving = "skeleton"
    pl._last_craving_level = 1
    pl._craving_cache = {"item": "skeleton", "level": 1, "count_done": 1,
                         "count_required": 2, "reward": 100, "step": 3}
    check("M6a: cache fresh when item AND level match",
          pl._craving_cache_fresh())
    pl._last_craving_level = 2   # same-family craving switched to level 2
    check("M6b: same-family level change makes the cache stale (re-read)",
          not pl._craving_cache_fresh(), f"level key roll={pl._last_craving_level}")
    pl._last_craving_level = 1
    check("M6c: level key restored -> fresh again",
          pl._craving_cache_fresh())


def main() -> int:
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    part_f()
    part_g()
    part_h()
    part_j()
    part_k()
    part_l()
    part_m()
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
