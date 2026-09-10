"""Repeatable verification for Supply Cupboard spawning (eye economy).

Run: `.venv/bin/python scripts/verify_cupboard_spawn.py` (exit 0/1).

The cupboard was invisible to the whole spawn path (hints, validator,
whitelist, fallback ranked grave/chest only), so an eyemonster craving
with a cupboard on the board never produced a spawn. Covered here:

  A. validator accepts cupboard spawn (slime ok/unknown), refuses on
     slime==0 and on a full board; grave/chest behavior unchanged.
  B. Best-spawn hint shows the cupboard only when eye components are
     wanted (eyemonster craving), with slime cost noted; hidden otherwise
     and on slime==0.
  C. fallback _spawn_move ranks cupboard between chest and grave when the
     craving wants eyes, and ignores it otherwise.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.agent import HeuristicPlanner  # noqa: E402
from planner.constants import SLIME_SPAWN_PREFIXES  # noqa: E402
from planner.llm import LLMPlanner  # noqa: E402
from planner.agent import Move  # noqa: E402
from vision.grid import GridGeometry, BoardState, build_cells, set_grid_geometry  # noqa: E402

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
        t.score = 0.95
        t.margin = 0.5
    return BoardState(5, 4, cells, geometry=GEOM)


def part_a_validator() -> None:
    set_grid_geometry(GEOM)
    b = _board([(1, 3, "supplycupboard_lvl1"), (0, 0, "bone")])
    mv = Move(kind="spawn", cell_a=(1, 3))
    r = LLMPlanner._validate(b, mv, 0.9, set(), slime_count=1500)
    check("A1: cupboard spawn accepted (slime ok)", r is None, str(r))
    r = LLMPlanner._validate(b, mv, 0.9, set(), slime_count=None)
    check("A2: cupboard spawn accepted (slime unknown)", r is None, str(r))
    r = LLMPlanner._validate(b, mv, 0.9, set(), slime_count=0)
    check("A3: cupboard refused on slime==0",
          r is not None and r.startswith("spawn_no_slime"), str(r))
    full = _board([(r_, c_, "bone") for r_ in range(5) for c_ in range(4)])
    full.cell_at(1, 3).item_id = "supplycupboard_lvl1"
    r = LLMPlanner._validate(full, mv, 0.9, set(), slime_count=1500)
    check("A4: cupboard refused on full board", r == "spawn_no_room", str(r))
    g = _board([(4, 3, "grave_lvl3"), (0, 0, "bone")])
    r = LLMPlanner._validate(g, Move(kind="spawn", cell_a=(4, 3)), 0.9, set())
    check("A5: grave unchanged", r is None, str(r))
    r = LLMPlanner._validate(b, Move(kind="spawn", cell_a=(0, 0)), 0.9, set())
    check("A6: bone still spawn_not_grave",
          (r or "").startswith("spawn_not_grave"), str(r))
    f = _board([(1, 3, "fridge_lvl1"), (0, 0, "bone")])
    r = LLMPlanner._validate(f, Move(kind="spawn", cell_a=(1, 3)), 0.9, set(),
                             slime_count=1500)
    check("A7: fridge spawn accepted (slime cost, like cupboard)",
          r is None, str(r))


def _drive_planner(tmp: Path):
    from metrics.logger import SessionLog  # noqa
    from planner.vision_drive import VisionDrivenPlanner  # noqa
    from vision.classifier import TemplateClassifier  # noqa
    p = VisionDrivenPlanner.__new__(VisionDrivenPlanner)
    p.live = True
    p.tool_enabled = True
    p.hints = True
    p.log = Mock()
    p.log.log = Mock()
    p.classifier = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
    p._cross_step_rejected = []
    p._feats_cache = None
    p._strategy = None
    p._strategy_family = None
    p._strategy_fresh = lambda: False
    p._last_craving = None
    p._last_craving_level = None
    p._craving_cache = None
    p.fallback = Mock()
    p.fallback.craved_item = None
    p.fallback.craved_level = None
    p.fallback.slime_count = None
    p._noop = Mock()
    p._noop.excluded.return_value = set()
    p._max_level_ids = lambda: set()
    p._chain_map = lambda: {}
    p._feed_values = lambda: {}
    p._damage_values = lambda: {}
    p.glossary_path = tmp / "glossary.md"
    p.knowledge_dir = tmp / "item_knowledge"
    p._frame = None
    p._mana_fraction = 1.0
    p.bottombar = None
    p.panels = None
    p.cravings = None
    p.champions = None
    p.slime_vat = None
    p.satiety_reader = None
    p.queue_box = None
    p.shop = None
    p._currency = {}
    p._last_currency_result = None
    p._step_count = 1
    p._last_feat_collect_step = -1000
    p._last_feat_collect_found = True
    p._champion_cache = None
    p._last_champion = None
    p._bubble_craving_item = lambda: None
    return p


def part_b_hint() -> None:
    import tempfile
    set_grid_geometry(GEOM)
    with tempfile.TemporaryDirectory() as td:
        p = _drive_planner(Path(td))
        b = _board([(1, 3, "supplycupboard_lvl1"), (4, 3, "grave_lvl3"),
                    (0, 0, "bone")])
        # craving wants eyes -> cupboard hinted with slime note
        p._craving_cache = {"item": "eyemonster", "level": 1,
                            "count_done": 0, "count_required": 1,
                            "reward": 50, "step": 0}
        p._step_count = 1
        hint = p._best_spawn_line(b)
        check("B1: cupboard hinted on eye craving",
              "supplycupboard_lvl1" in hint and "Slime" in hint, hint[:120])
        # no eye want -> grave hint instead
        p._craving_cache = None
        p._last_craving = None
        p.fallback.craved_item = None
        hint = p._best_spawn_line(b)
        check("B2: grave hint without eye want",
              hint.startswith("Best spawn: grave_lvl3"), hint[:120])
        # slime known-empty -> no cupboard hint even when wanted
        p._craving_cache = {"item": "eyemonster", "level": 1,
                            "count_done": 0, "count_required": 1,
                            "reward": 50, "step": 0}
        p.fallback.slime_count = 0
        hint = p._best_spawn_line(b)
        check("B3: no cupboard hint on slime==0",
              "supplycupboard" not in hint, hint[:120])
        p.fallback.slime_count = None


def part_c_fallback() -> None:
    set_grid_geometry(GEOM)
    hp = HeuristicPlanner()
    hp.craved_item = "eyemonster"
    b = _board([(1, 3, "supplycupboard_lvl1"), (4, 3, "grave_lvl3"),
                (0, 0, "bone")])
    mv = hp._spawn_move(b, frame=None)
    check("C1: fallback spawns cupboard when craved",
          mv is not None and mv.cell_a == (1, 3),
          f"{mv.kind} {mv.cell_a}" if mv else "None")
    hp.craved_item = None
    mv = hp._spawn_move(b, frame=None)
    check("C2: fallback ignores cupboard without eye want",
          mv is not None and mv.cell_a == (4, 3),
          f"{mv.kind} {mv.cell_a}" if mv else "None")


def part_d_const() -> None:
    check("D1: slime prefixes registered",
          SLIME_SPAWN_PREFIXES == ("supplycupboard", "fridge"),
          str(SLIME_SPAWN_PREFIXES))
    import py_compile
    for f in ("planner/agent.py", "planner/llm.py", "planner/constants.py",
              "planner/vision_drive.py"):
        py_compile.compile(str(ROOT / f), doraise=True)
    check("D2: py_compile clean", True)


def part_e_board_integration() -> None:
    """Whitelist, per-cell tag, and full board-state line on an eye-craving board."""
    import re
    import tempfile
    set_grid_geometry(GEOM)
    with tempfile.TemporaryDirectory() as td:
        p = _drive_planner(Path(td))
        p._mana_fraction = 1.0
        p.fallback.slime_count = 1500
        p.fallback.slime_capacity = None
        p.fallback.satiety_remaining = 1500
        p.fallback.satiety_capacity = 2000
        p.fallback.feed_prefer_item = None
        p.fallback.craving_bonus_est = 0
        p._feat_weights = lambda board: {}
        p._craving_objective = lambda: None
        p._craving_cache = {"item": "eyemonster", "level": 1,
                            "count_done": 0, "count_required": 1,
                            "reward": 50, "step": 0}
        b = _board([(1, 3, "supplycupboard_lvl1"), (4, 3, "grave_lvl3"),
                    (0, 0, "bone")])
        wl = p._whitelist_line_text(b, rejected=set())
        check("E1: whitelist lists cupboard spawn",
              "spawn (1,3) supplycupboard_lvl1" in (wl or ""), (wl or "")[:160])
        text = p._board_state_text(b)
        m = re.search(r"\(1,3\)[^\n]*", text)
        check("E2: cupboard cell tagged [SPAWNABLE]",
              m is not None and "[SPAWNABLE]" in m.group(0),
              m.group(0)[:160] if m else "no (1,3) line")
        h = re.search(r"Best spawn:[^\n]*", text)
        check("E3: board-state hint names cupboard",
              h is not None and "supplycupboard_lvl1" in h.group(0),
              h.group(0)[:160] if h else "no hint")


def main() -> None:
    for part in (part_a_validator, part_b_hint, part_c_fallback, part_d_const,
                 part_e_board_integration):
        try:
            part()
        except Exception as exc:  # noqa: BLE001
            check(f"{part.__name__} raised {type(exc).__name__}", False, str(exc)[:200])
    print(f"\n{PASS} checks passed")
    sys.exit(0 if PASS >= 15 else 1)


if __name__ == "__main__":
    main()
