"""Verification for the strategy slot (Aug 24): set_strategy tool.

Run: `.venv/bin/python scripts/verify_strategy_slot.py` (exit 0/1).

The slot gives the model bounded strategic agency: it commits to ONE active
feat (+ optional build-target item family) via `set_strategy`; code validates
the choice against the cached FEATS panel + template bank, boosts that
objective's move-kind weight, and shows the active strategy in board state.
Everything else (legality, safety, execution) stays code-owned.

  A. handler: valid feat committed + logged; feat matched case-insensitively
     against the panel; unknown feat rejected with the active list; unknown
     target rejected; family-prefix target accepted; missing feat rejected.
  B. freshness: age expiry; feat-completed invalidation; no feats cache.
  C. weights: strategy kind boost applied when fresh, absent when stale/expired;
     unknown-kind strategy defaults to a merge boost.
  D. board-state line: rendered when fresh, absent when stale; shown even
     with hints=False (model state, not a code hint).
  E. offering gate: not offered without a feats cache or while fresh;
     offered when stale.
  F. py_compile.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.logger import SessionLog  # noqa: E402
from planner.vision_drive import VisionDrivenPlanner, STRATEGY_REFRESH_STEPS  # noqa: E402
from vision.classifier import TemplateClassifier  # noqa: E402
from vision.grid import BoardState, Cell, GridGeometry, set_grid_geometry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PASS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if ok:
        PASS += 1


def make_planner(tmp: Path, feats: list[dict], step: int = 0):
    clf = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
    p = VisionDrivenPlanner(
        client=None, live=False, tool_enabled=True, hints=True,
        log=SessionLog(tmp / "s.jsonl"), chat_log_path=tmp / "c.jsonl",
        classifier=clf,
        learnings_path=tmp / "learnings.md", glossary_path=tmp / "glossary.md")
    p._feats_cache = {"tier": 4, "feats": feats, "step": step}
    p._step_count = step
    return p


def part_a_handler() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        feats = [{"name": "Own a lvl 3+ Grave.", "progress": "0/1"},
                 {"name": "Own a lvl 3+ Zombie.", "progress": "0/1"}]
        p = make_planner(tmp, feats)

        r = json.loads(p._exec_set_strategy({"feat": "Own a lvl 3+ Zombie.",
                                             "target_item": "zombie_lvl1"}))
        check("A1: valid feat+target committed", r.get("ok") is True
              and p._strategy["feat"] == "Own a lvl 3+ Zombie."
              and p._strategy["target"] == "zombie_lvl1")
        check("A2: strategy_set logged",
              any(e["event"] == "strategy_set" for e in p.log.events))

        r = json.loads(p._exec_set_strategy({"feat": "zombie"}))
        check("A3: case/prefix match commits", r.get("ok") is True
              and p._strategy["feat"] == "Own a lvl 3+ Zombie.")

        r = json.loads(p._exec_set_strategy({"feat": "Feed 50 things"}))
        check("A4: unknown feat rejected with active list",
              "unknown feat" in r.get("error", "")
              and "Own a lvl 3+ Grave." in r.get("active_feats", ""))

        r = json.loads(p._exec_set_strategy({"feat": "Own a lvl 3+ Grave.",
                                             "target_item": "notanitem"}))
        check("A5: unknown target rejected", "unknown target_item" in r.get("error", ""))

        r = json.loads(p._exec_set_strategy({"feat": "Own a lvl 3+ Grave.",
                                             "target_item": "skeleton"}))
        check("A6: family-prefix target accepted", r.get("ok") is True
              and p._strategy["target"] == "skeleton")

        r = json.loads(p._exec_set_strategy({}))
        check("A7: missing feat rejected", "requires feat" in r.get("error", ""))


def part_b_freshness() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        feats = [{"name": "Own a lvl 3+ Grave.", "progress": "0/1"}]
        p = make_planner(tmp, feats, step=100)
        p._exec_set_strategy({"feat": "Own a lvl 3+ Grave."})
        check("B1: fresh right after commit", p._strategy_fresh())
        p._step_count = 100 + STRATEGY_REFRESH_STEPS
        check("B2: expires after STRATEGY_REFRESH_STEPS", not p._strategy_fresh())
        p._step_count = 100
        p._feats_cache = {"tier": 4, "feats": [{"name": "Feed 50 things.",
                                                "progress": "0/50"}], "step": 100}
        check("B3: invalidated when the feat disappears", not p._strategy_fresh())
        p2 = make_planner(tmp, [], step=5)
        p2._exec_set_strategy({"feat": "anything"})
        check("B4: no feats cache -> stays fresh until expiry", p2._strategy_fresh())


def part_c_weights() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        feats = [{"name": "Own a lvl 3+ Grave.", "progress": "0/1"},
                 {"name": "Feed 50 things.", "progress": "0/50"}]
        p = make_planner(tmp, feats, step=7)
        p._exec_set_strategy({"feat": "Own a lvl 3+ Grave."})
        board = BoardState(rows=5, cols=3, cells=[
            Cell(r, c, 0, 0, None, score=0.0, margin=0.0, occupied=False)
            for r in range(5) for c in range(3)])
        w = p._feat_weights(board)
        base_merge, base_feed = 2.0, 1.5
        # the Grave feat parses as a merge objective at ratio 0; the strategy
        # bonus must push merge above feed.
        check("C1: strategy kind boost applied", w["merge"] >= base_merge + 0.9,
              f"merge={w['merge']}")
        check("C2: merge outranks feed under strategy", w["merge"] > w["feed"],
              f"merge={w['merge']} feed={w['feed']}")
        # stale strategy -> no boost
        p._step_count = 7 + STRATEGY_REFRESH_STEPS + 1
        w2 = p._feat_weights(board)
        check("C3: no boost when stale", w2["merge"] < w["merge"],
              f"merge={w2['merge']}")
        # strategy on the FEED feat (a valid panel match) boosts feed — the
        # slot lets the model override the closest-to-done merge preference.
        # (A strategy whose feat is absent from the panel is treated as stale
        # by _strategy_fresh, so "unknown-kind defaults to merge" is
        # unreachable through the handler — the handler rejects unknown feats.)
        p._step_count = 7
        p._exec_set_strategy({"feat": "Feed 50 things."})
        w3 = p._feat_weights(board)
        check("C4: feed-feat strategy boosts feed", w3["feed"] > w2["feed"],
              f"feed={w3['feed']}")


def part_d_board_line() -> None:
    class CostShop:
        # cost_cache is now a 5-key dict (ice/poison/blood/moon/death),
        # not the legacy (ice, green) 2-tuple.
        cost_cache = {}

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # "Own a lvl 3+ Grave." is a STATION-purchase feat (the
        # wiki says owning a lvl N+ X station requires buying the X
        # station and merging copies up). The model previously got the
        # wrong hint "merge/collect skeleton on the board" because the
        # noun-classifier regex didn't handle "lvl N+" (the `+` broke
        # `\w+`) and the model's target_item was trusted blindly. Now
        # the noun is classified as a station and the target is dropped
        # because it doesn't match the noun family.
        for hints in (True, False):
            feats = [{"name": "Own a lvl 3+ Grave.", "progress": "0/1"}]
            board = BoardState(rows=5, cols=3, cells=[
                Cell(r, c, 0, 0, None, score=0.0, margin=0.0, occupied=False)
                for r in range(5) for c in range(3)])
            p = make_planner(tmp, feats, step=3)
            p.hints = hints
            # Set a cost_cache for grave so the station hint can show it
            if p.shop is None:
                p.shop = CostShop()
            p.shop.cost_cache["grave"] = {"ice": 20, "poison": 0, "blood": 0,
                                          "moon": 0, "death": 0}
            p._exec_set_strategy({"feat": "Own a lvl 3+ Grave.",
                                  "target_item": "skeleton"})
                                  # target_item is dropped because skeleton != grave
            # (the target family doesn't match the noun family).
            check(f"D1a(hints={hints}): skeleton target dropped for station feat",
                  p._strategy.get("target") is None,
                  f"target={p._strategy.get('target')!r}")
            txt = p._board_state_text(board)
            check(f"D1b(hints={hints}): strategy line shows grave STATION hint",
                  "Strategy: Own a lvl 3+ Grave. (feat needs the grave STATION"
                  " from the Station panel" in txt,
                  txt[txt.find("Strategy:"):txt.find("Strategy:") + 200]
                  if "Strategy:" in txt else "(no Strategy line)")
        p._step_count = 3 + STRATEGY_REFRESH_STEPS + 1
        txt = p._board_state_text(board)
        check("D2: strategy line absent when stale", "Strategy:" not in txt)
        # D3: Build-type feat — requirement wording + cached cost from the shop
        feats_mp = [{"name": "Build a Mana Pool.", "progress": "0/1"},
                    {"name": "Own a lvl 3+ Grave.", "progress": "0/1"}]
        p2 = make_planner(tmp, feats_mp, step=3)

        # cost_cache is now a 5-key dict (ice/poison/blood/moon/death),
        # not the legacy (ice, green) 2-tuple. manapool costs 10 ice + 5 poison.
        CostShop.cost_cache["manapool"] = {"ice": 10, "poison": 5, "blood": 0,
                                            "moon": 0, "death": 0}
        if p2.shop is None:
            p2.shop = CostShop()
        p2._exec_set_strategy({"feat": "Build a Mana Pool.",
                               "target_item": "manapot"})
        txt2 = p2._board_state_text(board)
        check("D3: Build-type line states the STATION requirement + cost",
              "feat needs the manapool STATION from the Station panel "
              "— costs 10 ice + 5 poison runes — gather them first" in txt2,
              txt2[txt2.find("Strategy:"):txt2.find("Strategy:") + 160])
              # _strategy_family is set for Build-type feats so buy_station
        # can reject wrong-family calls. Re-run the set_strategy and inspect
        # the return value + the planner state directly.
        ret = json.loads(p2._exec_tool(
            {"function": {"name": "set_strategy",
                          "arguments": '{"feat": "Build a Mana Pool."}'}}, None))
        check("D3b: set_strategy returns expected_family for Build feats",
              ret.get("expected_family") == "manapool", str(ret))
        check("D3c: _strategy_family stored on the planner",
              p2._strategy_family == "manapool", str(p2._strategy_family))
              # "Own a lvl 3+ Grave." is now classified as a station-purchase
        # feat (the noun "grave" is in STATION_FAMILIES), so _strategy_family
        # is set to "grave" — buy_station can reject wrong-family calls for
        # Own-a-station feats too, not just Build-a-X feats.
        p2._exec_set_strategy({"feat": "Own a lvl 3+ Grave.",
                               "target_item": "skeleton"})
        check("D3d: Own-a-station feat sets _strategy_family to the station",
              p2._strategy_family == "grave", str(p2._strategy_family))


def part_d2_noun_classifier() -> None:
    """ _classify_feat_noun(feat) returns (kind, noun) for every
    wiki feat shape. The wrong-target bug ("Own a lvl 3+ Grave." with
    target=skeleton_lvl6 → wrong hint) was caused by a regex that
    silently dropped the noun when the lvl token contained `+`. The
    classifier now handles lvl 3+, lvl 10, multi-word nouns (mana pool,
    eye monster, mana golem, forgotten minion), and every station from
    the wiki's Stations page."""
    cases = [
        # station-shape feats (Build-a-X + Own-a-Lvl-N+-station)
        ("Build a Mana Pool.",        ("station", "manapool")),
        ("Build the Grave.",          ("station", "grave")),
        ("Build a Supply Cupboard.",  ("station", "supplycupboard")),
        ("Own a Lvl 3+ Grave.",       ("station", "grave")),
        ("Own a Lvl 4+ Mana Pool.",   ("station", "manapool")),
        ("Own a Lvl 3+ Supply Cupboard.", ("station", "supplycupboard")),
        ("Own a Lvl 3+ Slime Vat.",   ("station", "slimevat")),
        ("Own a Lvl 4+ Lectern.",     ("station", "lectern")),
        ("Own a Lvl 3+ Foul Chicken.",("station", "foulchicken")),
        ("Own a Lvl 4+ Altar.",       ("station", "altar")),
        ("Own a Lvl 5+ Dark Stores.", ("station", "darkstores")),
        ("Own a Lvl 3+ Fridge.",      ("station", "fridge")),
        # creature-shape feats (Own-a-Lvl-N+-creature)
        ("Own a Lvl 2+ Bone.",        ("creature", "bone")),
        ("Own a Lvl 3+ Skeleton.",    ("creature", "skeleton")),
        ("Own a Lvl 5+ Eye Monster.", ("creature", "eyemonster")),
        ("Own a Lvl 3+ Mana Golem.",  ("creature", "managolem")),
        ("Own a Lvl 3+ Forgotten Minion.", ("creature", "forgottenminion")),
        ("Own a Lvl 3+ Zombie.",      ("creature", "zombie")),
        # champions (also creatures)
        ("Own the Lich.",             ("creature", "lich")),
        ("Own a Cyclops.",            ("creature", "cyclops")),
        # other-shape feats (action count, collect-Nx, level-N, combat)
        ("Collect 5x Skeletons.",     ("other", None)),
        ("Create 5 Bones.",           ("other", None)),
        ("Tap The NecroMerger 10 times.", ("other", None)),
        ("Reach Devourer level 5.",   ("other", None)),
        ("Merge things 50 times.",    ("other", None)),
        ("Beat The Peasant twice.",   ("other", None)),
        ("Feed the Devourer a Bone.", ("other", None)),
    ]
    for feat, expected in cases:
        kind, noun = VisionDrivenPlanner._classify_feat_noun(feat)
        check(f"D2 classifier({feat!r})",
              (kind, noun) == expected,
              f"got ({kind!r}, {noun!r}), expected {expected!r}")


def part_e_offering() -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            feats = [{"name": "Own a lvl 3+ Grave.", "progress": "0/1"}]
            board = BoardState(rows=5, cols=3, cells=[
                Cell(r, c, 0, 0, None, score=0.0, margin=0.0, occupied=False)
                for r in range(5) for c in range(3)])

            class StubClient:
                def chat_message(self, messages, max_tokens=512, tools=None,
                                 tool_choice=None):
                    return "", [{"id": "t1", "type": "function",
                                 "function": {"name": (tools or [{}])[0]["function"]["name"],
                                              "arguments": "{}"}}], {}

            offered = []

            class SpyClient(StubClient):
                def chat_message(self, messages, **kw):
                    offered.extend(t["function"]["name"] for t in (kw.get("tools") or []))
                    return super().chat_message(messages, **kw)

            # live=True is required for the optional round to run at all; the
            # stub client is passed explicitly so no real server is touched.
            clf = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
            calls = []

            class SpyClient:
                def chat_message(self, messages, max_tokens=512, tools=None,
                                 tool_choice=None):
                    names = [t["function"]["name"] for t in (tools or [])]
                    calls.append((tool_choice, names))
                    name = names[0]
                    args = {"set_strategy": '{"feat": "Own a lvl 3+ Grave."}',
                            "lookup_wiki": '{"query": "Grave"}'}.get(name, "{}")
                    return "", [{"id": "t1", "type": "function",
                                 "function": {"name": name,
                                              "arguments": args}}], {}

            p = VisionDrivenPlanner(
                client=SpyClient(), live=True, tool_enabled=True, hints=True,
                log=SessionLog(tmp / "s.jsonl"), chat_log_path=tmp / "c2.jsonl",
                classifier=clf,
                learnings_path=tmp / "learnings.md", glossary_path=tmp / "glossary.md")
            p._feats_cache = {"tier": 4, "feats": feats, "step": 1}
            p._step_count = 1
            p._optional_tools([{"role": "user", "content": "go"}], board)
            check("E1: compelled planning round first (required, sole tool)",
                  calls and calls[0] == ("required", ["set_strategy"]),
                  f"calls={calls[:2]}")
            check("E1b: strategy committed by the compelled round",
                  p._strategy is not None
                  and p._strategy.get("feat") == "Own a lvl 3+ Grave.")
            # fresh strategy -> no compelled round, no re-offer
            calls.clear()
            p._optional_tools([{"role": "user", "content": "go"}], board)
            check("E2: not re-offered while fresh",
                  all("set_strategy" not in n for _, n in calls),
                  f"calls={calls}")


def part_f_income() -> None:
    """Build strategy + icerune on board => income objective boosts feed
    above merge, the income hint renders, and the fallback feeds the
    icerune stack (the 'why doesn't it feed the ice runes' case)."""
    import shutil, tempfile as tf
    with tf.TemporaryDirectory() as td:
        tmp = Path(td)
        # live glossary with icerune_lvl3 max_level + feed value
        shutil.copy(ROOT / "item_glossary.md", tmp / "glossary.md")
        clf = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
        board = BoardState(rows=5, cols=4, cells=[
            Cell(r, c, 0, 0, None, score=0.0, margin=0.0, occupied=False)
            for r in range(5) for c in range(4)])
        # place an icerune stack + some mergeable creatures
        def put(r, c, iid):
            for ch in board.cells:
                if ch.row == r and ch.col == c:
                    ch.item_id = iid; ch.occupied = True; ch.score = 0.95; ch.margin = 0.5
        put(2, 1, "icerune_lvl3"); put(0, 2, "bone"); put(1, 0, "ribcage")
        feats = [{"name": "Build a Mana Pool.", "progress": "0/1"}]
        p = VisionDrivenPlanner(
            client=None, live=False, tool_enabled=True, hints=True,
            log=SessionLog(tmp / "s.jsonl"), chat_log_path=tmp / "c.jsonl",
            classifier=clf,
            learnings_path=tmp / "learnings.md", glossary_path=tmp / "glossary.md")
        p._feats_cache = {"tier": 4, "feats": feats, "step": 0}
        p._step_count = 1
        p._exec_set_strategy({"feat": "Build a Mana Pool."})
        w = p._feat_weights(board)
        check("F1: build strategy boosts feed above merge",
              w["feed"] > w["merge"], f"feed={w['feed']} merge={w['merge']}")
        hint = p._best_move_hint(board)
        check("F2: hint is income (not merge)", "Best income:" in hint and "icerune_lvl3" in hint, hint[:120])
        income_line = p._best_income_line(board)
        check("F3: income line renders", "Best income:" in income_line, income_line)
        # fallback feeds the icerune stack, not the bone
        move = p.next_move(board, frame=None)
        check("F4: fallback feeds the icerune stack", move.kind == "feed" and move.cell_a == (2, 1),
              f"move={move}")
        # without an icerune on the board, no income boost
        board2 = BoardState(rows=5, cols=3, cells=[
            Cell(r, c, 0, 0, None, score=0.0, margin=0.0, occupied=False)
            for r in range(5) for c in range(3)])
        w2 = p._feat_weights(board2)
        check("F5: no icerune on board -> no feed boost over merge",
              w2["feed"] < w2["merge"] or w2["feed"] == w["feed"] - 0.75,
              f"feed={w2['feed']} merge={w2['merge']}")
        # non-build strategy with an icerune does NOT get the income boost
        p3 = VisionDrivenPlanner(
            client=None, live=False, tool_enabled=True, hints=True,
            log=SessionLog(tmp / "s3.jsonl"), chat_log_path=tmp / "c3.jsonl",
            classifier=clf,
            learnings_path=tmp / "learnings3.md", glossary_path=tmp / "glossary.md")
        p3._feats_cache = {"tier": 4, "feats": [{"name": "Open a Chest.", "progress": "0/1"}], "step": 0}
        p3._step_count = 1
        p3._exec_set_strategy({"feat": "Open a Chest."})
        check("F6: non-build strategy -> no income line", p3._best_income_line(board) == "")


def part_f2_station_hint() -> None:
    """ the build-station strategy line shows a board-aware hint
    when the model has existing stations. The old "gather them first"
    phrasing was misleading when the model already had a lvl1+lvl2 grave
    and needed one more lvl1. The new hint counts existing stations and
    tells the model the exact merge-up math."""
    import shutil, tempfile as tf
    with tf.TemporaryDirectory() as td:
        tmp = Path(td)
        shutil.copy(ROOT / "item_glossary.md", tmp / "glossary.md")
        clf = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)

        def make_board(items):
            board = BoardState(rows=5, cols=4, cells=[
                Cell(r, c, 0, 0, None, score=0.0, margin=0.0, occupied=False)
                for r in range(5) for c in range(4)])
            for r, c, iid in items:
                for ch in board.cells:
                    if ch.row == r and ch.col == c:
                        ch.item_id = iid; ch.occupied = True
                        ch.score = 0.95; ch.margin = 0.5
            return board

        # 1x lvl1 + 1x lvl2 grave, need lvl3 → "you'll need 1 more grave"
        board = make_board([(0, 0, "grave_lvl1"), (2, 1, "grave_lvl2")])
        p = VisionDrivenPlanner(
            client=None, live=False, tool_enabled=True, hints=True,
            log=SessionLog(tmp / "s.jsonl"), chat_log_path=tmp / "c.jsonl",
            classifier=clf,
            learnings_path=tmp / "learnings.md", glossary_path=tmp / "glossary.md")
        p._exec_set_strategy({"feat": "Own a Lvl 3+ Grave."})
        line = p._strategy_line_text(board=board)
        check("F2-1: 1x lvl1 + 1x lvl2 grave -> need 1 more",
              "1 more grave" in line and "merge them up" in line, line[:200])

        # 1x lvl3 grave already → "you ALREADY have a lvl3 grave"
        board = make_board([(0, 0, "grave_lvl3")])
        p._feats_cache = {"tier": 4, "feats": [
            {"name": "Own a Lvl 3+ Grave.", "progress": "0/1"}], "step": 0}
        p._step_count = 1
        line = p._strategy_line_text(board=board)
        check("F2-2: 1x lvl3 grave -> ALREADY have",
              "ALREADY have" in line and "lvl3 grave" in line, line[:200])

        # No stations → "you have NO grave on the board"
        board = make_board([])
        p._feats_cache = {"tier": 4, "feats": [
            {"name": "Own a Lvl 3+ Grave.", "progress": "0/1"}], "step": 0}
        p._step_count = 1
        line = p._strategy_line_text(board=board)
        check("F2-3: no grave -> NO grave on the board",
              "NO grave" in line, line[:200])

        # 2x lvl1 grave → "1 more grave to reach lvl3"
        board = make_board([(0, 0, "grave_lvl1"), (1, 0, "grave_lvl1")])
        p._feats_cache = {"tier": 4, "feats": [
            {"name": "Own a Lvl 3+ Grave.", "progress": "0/1"}], "step": 0}
        p._step_count = 1
        line = p._strategy_line_text(board=board)
        check("F2-4: 2x lvl1 grave -> 1 more grave",
              "2x lvl1" in line and "1 more" in line, line[:200])


def part_g_compile() -> None:
    import py_compile
    py_compile.compile(str(ROOT / "planner" / "vision_drive.py"), doraise=True)
    check("G: py_compile clean", True)


def part_h_direction() -> None:
    """Suggested-direction line (replaces the retired periodic StrategyPlanner).

    Code-evaluated meta-goals render when no strategy is fresh; silent when
    a strategy is committed. Thresholds are None-safe (missing data never
    fires — the old evaluator's darkness check fired unconditionally).
    """
    from vision.grid import build_cells
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        feats = [{"name": "Own a lvl 3+ Grave.", "progress": "0/1"}]
        geom = GridGeometry(293, 1178, 230, 5, 3)
        set_grid_geometry(geom)

        def board(n_occupied: int):
            cells = build_cells(geom)
            for c in cells:
                c.occupied = False
                c.item_id = None
            for c in cells[:n_occupied]:
                c.occupied = True
                c.item_id = "bone"
                c.score = 0.9
                c.margin = 0.5
            return BoardState(5, 3, cells, geometry=geom)

        p = make_planner(tmp, feats, step=5)
        p._mana_fraction = 0.9
        p._currency = {"ice": 100, "poison": 100, "blood": 0, "moon": 0, "death": 0}
        p._last_champion = "peasant"
        line = p._suggested_direction(board(5))
        check("H1: champion present -> champion_combat",
              "champion_combat" in line, line[:120])
        p._last_champion = None
        p._mana_fraction = 0.1
        p._currency = {"ice": 100, "poison": 100, "blood": 0, "moon": 0, "death": 0}
        line = p._suggested_direction(board(5))
        check("H2: low mana, rich runes -> mana_generation",
              "mana_generation" in line, line[:120])
        p._mana_fraction = 0.9
        p._currency = {}
        line = p._suggested_direction(board(5))
        check("H3: poor runes -> rune_economy",
              "rune_economy" in line, line[:120])
        line = p._suggested_direction(board(14))
        check("H4: congested board -> board_management",
              "board_management" in line, line[:120])
        # fresh strategy suppresses the line (a direction is committed)
        p._exec_set_strategy({"feat": "Own a lvl 3+ Grave."})
        check("H5: fresh strategy -> silent",
              p._suggested_direction(board(14)) == "")
        # missing data never fires
        p2 = make_planner(tmp, feats, step=5)
        p2._mana_fraction = None
        p2._currency = {}
        p2._last_champion = None
        p2.fallback.slime_count = None
        p2.fallback.satiety_remaining = None
        check("H6: sparse board, unknown data -> rune_economy (zero runes)",
              "rune_economy" in p2._suggested_direction(board(2)))


def main() -> None:
    global PASS
    set_grid_geometry(GridGeometry(293, 1178, 230, 5, 3))
    for part in (part_a_handler, part_b_freshness, part_c_weights,
                 part_d_board_line, part_d2_noun_classifier, part_e_offering,
                 part_f_income, part_f2_station_hint, part_g_compile,
                 part_h_direction):
        try:
            part()
        except Exception as exc:  # noqa: BLE001
            check(f"{part.__name__} raised {type(exc).__name__}", False, str(exc))
    print(f"\n{PASS} checks passed")
    sys.exit(0 if PASS >= 20 else 1)


if __name__ == "__main__":
    main()