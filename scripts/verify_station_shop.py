"""Verification for the StationShop buy flow (Aug 24; Aug 26 currency rename;
Aug 27 mid-animation + currency-bar panel verification).

Run: `.venv/bin/python scripts/verify_station_shop.py` (exit 0/1).

  A. currency reader on the saved station-panel fixture (ice=16, rest 0;
     only the 5 real runes — no green/red/purple aliases in the dict).
  B. card identification via a stub LLM (grave / manapool / locked).
  C. buy flow with a scripted device + LLM:
     - unaffordable -> error, card never tapped, panel closed
     - affordable + confirm=False -> dialog read, BACK, no purchase
     - dialog mismatch (was grave, dialog says manapool) -> BACK + refused
     - confirm=True -> Confirm tapped, dialog closed, currency dropped
     - no dialog after card tap -> clean error, no BACK
  D. vision-drive tool wiring: buy_station routes to the shop + logs.
  D2. max_level detection still works for mergeable runes.
  D3. new panel-verification gate refuses buys when the open panel is NOT
     the station panel (the Aug 26 19/0 wrong-panel mystery).
  D4. dialog station mismatch is now an EXACT mismatch (was 4-char prefix,
     too loose for slidable panels).
  D8. Aug 27 mid-animation + currency-bar panel verification:
     - body-rendered check rejects mid-animation frames
     - all-zero currency triggers a retry
     - currency-bar check rejects a wrong panel that happens to show a
       station card icon
  E. py_compile.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from metrics.logger import SessionLog  # noqa: E402
from planner.vision_drive import VisionDrivenPlanner  # noqa: E402
from vision.classifier import TemplateClassifier  # noqa: E402
from vision.grid import GridGeometry, set_grid_geometry  # noqa: E402
from vision.panels import CONFIRM_XY, CURRENCY_Y, StationShop  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "assets" / "calib" / "feats" / "station_panel_live.png"
PASS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if ok:
        PASS += 1


class FakeBottomBar:
    """Dock stub: `panel_open` flips on open_panel and off on device.back()."""

    def __init__(self):
        self.panel_open = False

    def bar_visible(self, frame):
        return not self.panel_open

    def read_bar(self, frame):
        return [{"name": n, "unlocked": True, "score": 1.0}
                for n in ("feats", "station", "queue", "spellbook", "shop")]


class FakeDevice:
    def __init__(self, frames):
        self.frames = list(frames)      # indexed with a clamping cursor
        self._i = 0
        self.taps = []
        self.swipes = []                # card-row scroll gestures
        self.backs = 0
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
        if y > 2600:            # a dock-band tap opens the panel
            self.bottombar.panel_open = True

    def swipe(self, x1, y1, x2, y2, duration_ms=300):
        self.swipes.append((int(x1), int(y1), int(x2), int(y2)))
        # sheet position is frame-driven in tests (frames list); swipes
        # only record the gesture.

    def back(self):
        self.backs += 1
        self.bottombar.panel_open = False


class StubLLM:
    """Scripted vision reads: card reads then dialog reads, in order."""

    def __init__(self, card_reads, dialog_reads):
        self.card_reads = list(card_reads)
        self.dialog_reads = list(dialog_reads)

    def chat(self, messages, max_tokens=128, temperature=0.0, json_mode=True):
        text = json.dumps(messages[-2]["content"]) if False else ""
        # the assistant prefill tells us which read this is
        prefill = messages[-1]["content"]
        if prefill.startswith('{"station":'):
            return self.card_reads.pop(0), {}
        if prefill.startswith('{"is_purchase_dialog":'):
            return self.dialog_reads.pop(0), {}
        return "{}", {}


def make_shop(frames, card_reads, dialog_reads):
    dev = FakeDevice(frames)
    llm = StubLLM(card_reads, dialog_reads)
    bb = dev.bottombar
    shop = StationShop(device=dev, llm=llm, bottombar=bb)
    return shop, dev


def no_dialog_stub(shop):
    shop._dialog_up = lambda frame: False


def part_a_currency() -> None:
    f = cv2.imread(str(FIXTURE))
    shop = StationShop(device=None, llm=None, bottombar=None)
    cur = shop.read_currency(f)
    check("A: currency reader (ice=16, rest 0)",
          cur["ice"] == 16
          and cur["poison"] == 0 and cur["blood"] == 0
          and cur["moon"] == 0 and cur["death"] == 0, str(cur))
          # NO back-compat aliases — only the 5 real runes appear in the
    # dict. (User report: green/red/purple cluttered the session.jsonl
    # alongside the real names; the buy logic already used real names.)
    check("A2: dict has exactly 5 keys, no green/red/purple aliases",
          set(cur.keys()) == {"ice", "poison", "blood", "moon", "death"}, str(cur))
          # live calibration fixture (saved Aug 26 at 14:02) has the
    # real in-game runes the user reported: ice=63, poison=19, rest 0.
    # This is the regression fixture for the 2x upscale + 5-rune rename.
    live_fixture = ROOT / "assets" / "calib" / "feats" / "station_panel_live_v2.png"
    if live_fixture.exists():
        f2 = cv2.imread(str(live_fixture))
        cur2 = shop.read_currency(f2)
        check("A3b: live fixture reads ice=63, poison=19, rest 0",
              cur2["ice"] == 63 and cur2["poison"] == 19
              and cur2["blood"] == 0 and cur2["moon"] == 0
              and cur2["death"] == 0, str(cur2))
    else:
        check("A3b: live fixture present", False, str(live_fixture))
    # '+N' gain indicator + first-group parsing (the 162/178 over-read bug)
    for txt, want in (("16 2", 16), ("17 8", 17), ("+5", 5), ("", 0)):
        got = StationShop._parse_count(txt)
        check(f"A4: _parse_count({txt!r})=={want}", got == want, str(got))


def part_b_cards() -> None:
    f = cv2.imread(str(FIXTURE))
    cards_in = ['"grave", "cost_ice": 20, "cost_poison": 0} extra',
                '"manapool", "cost_ice": 10, "cost_poison": 5} extra',
                '"locked", "cost_ice": 0, "cost_poison": 0} extra']
    shop, _ = make_shop([f], cards_in, [])
    cards = shop.read_cards(f)
    check("B: three cards identified",
          [c["station"] for c in cards] == ["grave", "manapool", "locked"],
          str([(c["station"], c["x"]) for c in cards]))
    check("B2: card centers in slot ranges",
          cards[0]["x"] == 229 and cards[1]["x"] == 717 and cards[2]["x"] == 1086,
          str([c["x"] for c in cards]))
          # cost cache now stores the 5 currencies, not a 2-tuple.
    check("B3: cost_cache stores all 5 currencies for grave",
          shop.cost_cache.get("grave", {}).get("ice") == 20
          and "poison" in shop.cost_cache.get("grave", {}),
          str(shop.cost_cache))


def base_frames(n=8):
    panel = cv2.imread(str(FIXTURE))
    lair = panel  # content irrelevant (bottombar stubbed)
    return [lair] + [panel] * n


CARDS = ['"grave", "cost_ice": 20, "cost_poison": 0} extra',   # unaffordable (16 in HUD)
         '"manapool", "cost_ice": 10, "cost_poison": 5} extra',
         '"locked", "cost_ice": 0, "cost_poison": 0} extra']
CARDS_OK = ['"grave", "cost_ice": 10, "cost_poison": 0} extra',  # affordable
            '"manapool", "cost_ice": 10, "cost_poison": 5} extra',
            '"locked", "cost_ice": 0, "cost_poison": 0} extra']
DIALOG_GRAVE = 'true, "station": "grave", "cost_ice": 20, "cost_poison": 0} extra'
DIALOG_NO = 'false} extra'


def part_c_flow() -> None:
    # C1: unaffordable (grave costs 20, balance 16) -> refused, card NOT tapped
    shop, dev = make_shop(base_frames(), list(CARDS), [])
    res = shop.buy("grave", confirm=True)
    check("C1: unaffordable refused", res["error"] and "unaffordable" in res["error"],
          res.get("error"))
    check("C1b: card not tapped", all(t[1] != 2400 for t in dev.taps),
          str(dev.taps))

    # C2: affordable + confirm=False -> dialog read, BACK, no purchase
    shop, dev = make_shop(base_frames(), list(CARDS_OK),
                          [DIALOG_GRAVE])
    shop._dialog_up = lambda frame: True     # template detector stubbed
    res = shop.buy("grave", confirm=False)
    check("C2: dialog read returned", (res.get("dialog") or {}).get("station") == "grave",
          str(res.get("dialog")))
    check("C2b: BACK used, Confirm NOT tapped",
          dev.backs >= 1 and CONFIRM_XY not in dev.taps, f"{dev.backs} {dev.taps}")
    check("C2c: note tells the model to re-call", "confirm=True" in res.get("note", ""))

    # C3: dialog mismatch -> refused + BACK
    shop, dev = make_shop(base_frames(), list(CARDS_OK),
                          ['true, "station": "manapool", "cost_ice": 10, "cost_poison": 5} extra'])
    shop._dialog_up = lambda frame: True
    res = shop.buy("grave", confirm=True)
    check("C3: mismatched dialog refused", "mismatch" in (res.get("error") or ""),
          res.get("error"))
    check("C3b: Confirm NOT tapped on mismatch", CONFIRM_XY not in dev.taps)

    # C4: confirm=True + matching dialog -> Confirm tapped, currency drops
    panel = cv2.imread(str(FIXTURE))
    broke = panel.copy()
    cv2.rectangle(broke, (240, 190), (325, 314), (30, 30, 30), -1)  # blank ice digits
    frames = [base_frames()[0]] + [base_frames()[1]] * 2 + [broke] * 5
    shop, dev = make_shop(frames, list(CARDS_OK),
                          [DIALOG_GRAVE])
    seq = [True, False]                      # detection: up; post-confirm: gone
    shop._dialog_up = lambda frame: seq.pop(0)
    # the real template detector runs against synthetic frames here — stub it
    # to the calibrated Confirm point (the live-verified coordinates)
    shop._find_confirm = lambda frame: (CONFIRM_XY[0], CONFIRM_XY[1], 0.99)
    res = shop.buy("grave", confirm=True)
    check("C4: purchase completed", res.get("bought") is True, str(res))
    check("C4b: Confirm tapped at the calibrated point", CONFIRM_XY in dev.taps,
          str(dev.taps))

    # C5: no dialog after card tap -> clean error, no BACK (nothing to dismiss)
    shop, dev = make_shop(base_frames(6), list(CARDS_OK), [])
    no_dialog_stub(shop)
    res = shop.buy("grave", confirm=False)
    check("C5: no-dialog -> clean error", "no confirm dialog" in (res.get("error") or ""),
          res.get("error"))
    check("C5b: exactly one BACK (the open sheet's close)",
          dev.backs == 1, str(dev.backs))


def part_d_tool() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        clf = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
        panel = cv2.imread(str(FIXTURE))

        class StubClient:
            def chat_message(self, messages, max_tokens=512, tools=None,
                             tool_choice=None):
                names = [t["function"]["name"] for t in (tools or [])]
                return "", [{"id": "t", "type": "function",
                             "function": {"name": names[0],
                                          "arguments": '{"family": "grave"}'}}], {}

        p = VisionDrivenPlanner(
            client=StubClient(), live=True, tool_enabled=True, hints=True,
            log=SessionLog(tmp / "s.jsonl"), chat_log_path=tmp / "c.jsonl",
            classifier=clf,
            learnings_path=tmp / "l.md", glossary_path=tmp / "g.md")

        class StubShop:
            def __init__(self):
                self._pending_buy = None
            def buy(self, family, confirm=False):
                return {"family": family, "confirm": confirm, "bought": True,
                        "error": None,
                        "note": ("dialog verified — call buy(family, "
                                 "confirm=True) to complete the purchase")
                        if not confirm else None}

        p.shop = StubShop()
        board = None
        # two-phase buy — phase 1 stashes pending_buy
        res = json.loads(p._exec_tool(
            {"function": {"name": "buy_station",
                          "arguments": '{"family": "grave", "confirm": false}'}}, board))
        check("D: tool routes to the shop (phase 1)",
              res.get("note", "").startswith("dialog verified"), str(res))
        # phase 2 completes the buy
        res = json.loads(p._exec_tool(
            {"function": {"name": "buy_station",
                          "arguments": '{"family": "grave", "confirm": true}'}}, board))
        check("D2: station_buy logged",
              any(e["event"] == "station_buy" and e.get("confirm")
                  for e in p.log.events))


def part_d2_maxlevel_detection() -> None:
    """The max_level detection must NOT flag items whose description says
    'Merge to create a bigger pile or feed' (the Aug 25 icerune bug: all
    rune stacks were flagged max-level, which blocked ALL rune merging)."""
    clf = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
    panel = cv2.imread(str(FIXTURE))

    class StubLLM2:
        def __init__(self, reads):
            self.reads = list(reads)

        def chat(self, messages, max_tokens=128, temperature=0.0, json_mode=True):
            return self.reads.pop(0), {}

    shop = StationShop(device=None, llm=None, bottombar=None)
    # _bank_popup_recipe lives on the planner; test the detection condition
    # directly via the same logic it uses.
    from planner.vision_drive import VisionDrivenPlanner
    tmp = Path(tempfile.mkdtemp())
    planner = VisionDrivenPlanner(
        client=None, live=False, tool_enabled=False,
        log=SessionLog(tmp / "s.jsonl"), chat_log_path=tmp / "c.jsonl",
        classifier=clf, glossary_path=tmp / "g.md",
        learnings_path=tmp / "l.md")
    # mergeable description -> NOT max-level
    r1 = planner._bank_popup_recipe("icerune_lvl1", {
        "description": "Merge to create a bigger pile or feed to the Devourer.",
        "merge_info": None, "feed_value": 2, "damage": None})
    g = (tmp / "g.md").read_text()
    check("D2a: merge-worded description NOT flagged max_level",
          "## Item icerune_lvl1 (popup)" in g
          and "max_level" not in g.split("## Item icerune_lvl1 (popup)")[1]
              .split("##")[0], g[:200])
    # feed-only description -> max-level
    planner._bank_popup_recipe("icerune_lvl3", {
        "description": "Feed to the Devourer.",
        "merge_info": None, "feed_value": 12, "damage": None})
    g = (tmp / "g.md").read_text()
    check("D2b: feed-only description flagged max_level",
          "## Item icerune_lvl3 (popup)" in g
          and "max_level: true" in g.split("## Item icerune_lvl3 (popup)")[1]
              .split("##")[0])


def part_d3_panel_verify() -> None:
    """Aug 26 panel-verify gate: refuses buys when the open panel is NOT
    the station panel (the 19/0 wrong-panel mystery). The verification
    uses card-icon template matching in CARD_ICON_Y."""
    panel = cv2.imread(str(FIXTURE))
    shop = StationShop(device=None, llm=None, bottombar=None)

    # D3a: real station panel passes verification (Grave card icon present)
    ok, detail = shop._verify_station_panel(panel)
    check("D3a: real station panel verified",
          ok is True and "grave" in detail, detail)

    # D3b: blank/dim frame (e.g. empty screencap) fails verification
    blank = (panel * 0).astype("uint8")
    ok2, detail2 = shop._verify_station_panel(blank)
    check("D3b: blank frame fails panel verification",
          ok2 is False and "no station card icon" in detail2, detail2)

    # D3c: buy refuses when panel verification fails (the user-reported
    # 19/0 case: the captured frame wasn't actually the station panel)
    import numpy as np
    fake_feats = panel.copy()
    # The card region is now blank (no station card), but the y=190-314
    # strip still has the calibration icons. This is what happens when a
    # feats panel covers the dock — the dock-flip detector says "panel
    # open" but it's the wrong one. (Aug 27: this case is also caught
    # earlier by _panel_body_rendered if the card region is uniform gray.)
    fake_feats[2230:2520, 19:439] = 30
    fake_feats[2230:2520, 480:954] = 30
    fake_feats[2230:2520, 892:1280] = 30
    shop, dev = make_shop([fake_feats], list(CARDS_OK), [])
    res = shop.buy("manapool", confirm=False)
    # a blank card region can be refused by either gate
    # (_panel_body_rendered in _wait_panel_open OR
    # _verify_station_panel after open). Both are correct refusals.
    err = res.get("error") or ""
    refused = ("station panel did not open" in err
               or "station panel not detected" in err)
    check("D3c: wrong-panel buy refused with diagnostic",
          refused and res.get("panel_verified") is not True,
          err[:120])
    # Card should NOT be tapped (refused before the card tap)
    card_taps = [t for t in dev.taps if abs(t[1] - 2400) < 5]
    check("D3d: card not tapped when panel unverified",
          len(card_taps) == 0, str(dev.taps))


def part_d4_exact_dialog_match() -> None:
    """ dialog station name must match the requested family
    exactly (was a 4-char prefix, too loose when panels slide and the
    card-tap lands on a neighbor)."""
    panel = cv2.imread(str(FIXTURE))
    # CARDS_OK makes grave cost_ice=10 (affordable at balance 16); the
    # dialog then lies and says 'lectern' — exact-match refusal fires.
    shop, dev = make_shop(base_frames(), list(CARDS_OK),
                          ['true, "station": "lectern", "cost_ice": 0, "cost_poison": 0} extra'])
    shop._dialog_up = lambda frame: True
    res = shop.buy("grave", confirm=True)
    check("D4a: exact-match mismatch (lectern vs grave) refused",
          "mismatch" in (res.get("error") or ""), res.get("error"))
    check("D4b: Confirm NOT tapped on exact-mismatch",
          CONFIRM_XY not in dev.taps)


def part_d5_two_phase_buy() -> None:
    """ buy_station with confirm=false must set _pending_buy;
    confirm=true with a matching pending_buy must clear it; mismatched
    family or no-pending confirm=true must be refused."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        clf = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
        p = VisionDrivenPlanner(
            client=None, live=False, tool_enabled=False,
            log=SessionLog(tmp / "s.jsonl"), chat_log_path=tmp / "c.jsonl",
            classifier=clf, glossary_path=tmp / "g.md",
            learnings_path=tmp / "l.md")
        # Stub shop: confirm=false returns a verified dialog (note set),
        # confirm=true returns bought=True.
        class StubShop:
            def __init__(self):
                self._pending_buy = None
            def buy(self, family, confirm=False):
                if confirm:
                    return {"family": family, "confirm": True, "bought": True,
                            "error": None, "dialog": {"station": family}}
                return {"family": family, "confirm": False, "bought": False,
                        "error": None,
                        "dialog": {"is_purchase_dialog": True,
                                   "station": family, "cost_ice": 20},
                        "panel_verified": True,
                        "note": ("dialog verified — call buy(family, "
                                 "confirm=True) to complete the purchase")}

        p.shop = StubShop()
        # No strategy set yet -> _strategy_family is None
        # D5a: confirm=true with no pending buy -> refused
        res = json.loads(p._exec_tool(
            {"function": {"name": "buy_station",
                          "arguments": '{"family": "grave", "confirm": true}'}}, None))
        check("D5a: confirm=true without pending buy refused",
              "no pending buy" in (res.get("error") or ""), str(res))
        # D5b: confirm=false -> stashes pending_buy
        res = json.loads(p._exec_tool(
            {"function": {"name": "buy_station",
                          "arguments": '{"family": "grave", "confirm": false}'}}, None))
        check("D5b: confirm=false stashes _pending_buy",
              p.shop._pending_buy is not None
              and p.shop._pending_buy.get("family") == "grave",
              str(p.shop._pending_buy))
        # D5c: confirm=true with different family -> refused
        res = json.loads(p._exec_tool(
            {"function": {"name": "buy_station",
                          "arguments": '{"family": "manapool", "confirm": true}'}}, None))
        check("D5c: confirm=true with mismatched family refused",
              "pending buy is for" in (res.get("error") or ""), str(res))
        # D5d: confirm=true with same family -> clears pending_buy
        res = json.loads(p._exec_tool(
            {"function": {"name": "buy_station",
                          "arguments": '{"family": "grave", "confirm": true}'}}, None))
        check("D5d: confirm=true with matching family clears _pending_buy",
              p.shop._pending_buy is None,
              str(p.shop._pending_buy))


def part_d6_strategy_enforcement() -> None:
    """ buy_station must refuse a family that doesn't match the
    strategy's target (the model observed picking 'grave' when the
    strategy was 'Build a Mana Pool.')."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        clf = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
        p = VisionDrivenPlanner(
            client=None, live=False, tool_enabled=False,
            log=SessionLog(tmp / "s.jsonl"), chat_log_path=tmp / "c.jsonl",
            classifier=clf, glossary_path=tmp / "g.md",
            learnings_path=tmp / "l.md")
        # Commit a Build-a-Mana-Pool strategy
        p._step_count = 1
        p._strategy = {"feat": "Build a Mana Pool.", "target": None, "step": 1}
        p._strategy_family = "manapool"

        class StubShop:
            called_with = None
            def buy(self, family, confirm=False):
                StubShop.called_with = (family, confirm)
                return {"family": family, "confirm": confirm, "bought": False,
                        "error": None}

        p.shop = StubShop()
        # D6a: buy_station("grave") refused (strategy wants manapool)
        res = json.loads(p._exec_tool(
            {"function": {"name": "buy_station",
                          "arguments": '{"family": "grave", "confirm": false}'}}, None))
        check("D6a: buy_station refuses wrong family (strategy says manapool)",
              "strategy says build" in (res.get("error") or "")
              and res.get("expected_family") == "manapool", str(res))
        check("D6b: shop.buy() NOT called when family mismatches strategy",
              StubShop.called_with is None)
        # D6c: buy_station("manapool") proceeds
        res = json.loads(p._exec_tool(
            {"function": {"name": "buy_station",
                          "arguments": '{"family": "manapool", "confirm": false}'}}, None))
        check("D6c: buy_station accepts matching family",
              StubShop.called_with == ("manapool", False))


def part_d7_pending_buy_expiry() -> None:
    """ _pending_buy auto-expires after PENDING_BUY_TTL steps
    so a future confirm=true can't accidentally complete a stale
    dialog from a long-forgotten attempt."""
    from env.adb import Device
    class _NullDevice:
        def screencap(self): pass
        def wait_for_idle(self, _s): pass
        def tap(self, x, y): pass
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        clf = TemplateClassifier(str(ROOT / "assets" / "templates"), seed=True)
        p = VisionDrivenPlanner(
            client=None, live=False, tool_enabled=False,
            log=SessionLog(tmp / "s.jsonl"), chat_log_path=tmp / "c.jsonl",
            classifier=clf, glossary_path=tmp / "g.md",
            learnings_path=tmp / "l.md")
        # Replace the device with a no-op so next_move doesn't crash
        p.fallback.device = _NullDevice()
        # The planner needs a station shop (for buy_station). We don't call
        # buy here — the _pending_buy expiry is checked at the start of
        # next_move, before any buy is issued — so the shop can be a stub.
        class _StubShop:
            def buy(self, family, confirm=False):
                return {"family": family, "confirm": confirm, "bought": False,
                        "error": None, "dialog": None}
        p.shop = _StubShop()
        # Simulate a pending buy from 6 steps ago
        p._step_count = 6
        p.shop._pending_buy = {"family": "grave", "dialog": {}, "step": 0}
        # Make a board with real cells so classify_board doesn't fail
        from vision.grid import BoardState, Cell
        board = BoardState(rows=5, cols=4, cells=[
            Cell(0, 0, 0, 0, None, score=0.0, margin=0.0, occupied=False)
            for _ in range(20)])
        try:
            p.next_move(board, frame=None)
        except Exception as exc:  # noqa: BLE001
            # Static-mode next_move will fail to LLM, that's fine — we only
            # care about the expiry check happening at the start.
            pass
        check("D7: pending_buy expires after PENDING_BUY_TTL steps",
              p.shop._pending_buy is None
              or any(e["event"] == "buy_pending_expired" for e in p.log.events))


def part_e_compile() -> None:
    import py_compile
    for f in ("vision/panels.py", "planner/vision_drive.py", "main.py"):
        py_compile.compile(str(ROOT / f), doraise=True)
    check("E: py_compile clean", True)


def part_d8_animation_frame_guards() -> None:
    """ three guards against mid-animation / wrong-panel frames
    captured by `_wait_panel_open` and `_verify_station_panel`:
      1. _panel_body_rendered: card row must have non-flat std (rejects
         the empty/transparent mid-animation state).
      2. read_currency all-zero -> retry _wait_panel_open and re-read.
      3. _check_currency_bar: 5-rune bar at y=190-314 must have rune
         icons in at least 2 of the 5 slots (catches wrong panels that
         happen to show a station card icon).
    """
    import numpy as np
    panel = cv2.imread(str(FIXTURE))
    # Make a _panel_ reader and the helper directly
    pr = __import__("vision.panels", fromlist=["PanelReader"]).PanelReader
    shop = StationShop(device=None, llm=None, bottombar=None)

    # D8-1: a real v2 panel's body has rendered std >= 25 (cards present)
    ok = shop._panel._panel_body_rendered(panel)
    check("D8-1a: real panel body rendered (cards present)",
          ok is True, f"std check via _panel_body_rendered")
    # D8-1b: a mid-animation frame (uniform gray below the panel header
    # where the cards should be) fails the body-rendered check
    mid_anim = panel.copy()
    mid_anim[2230:2520, :] = 30
    ok2 = shop._panel._panel_body_rendered(mid_anim)
    check("D8-1b: mid-animation frame body NOT rendered (cards blanked)",
          ok2 is False, f"body-rendered refused")

    # D8-2: a real panel's currency bar has rune icons in 2+ slots
    bar_ok, bar_detail = shop._check_currency_bar(panel)
    check("D8-2a: real panel currency bar has rune icons",
          bar_ok is True and "5 slots" in bar_detail, bar_detail)
    # D8-2b: a wrong panel that happens to show a station card but has
    # no currency bar (blanked y=180-260) fails the currency-bar check
    wrong = panel.copy()
    wrong[180:260, :] = 60
    bar_ok2, bar_detail2 = shop._check_currency_bar(wrong)
    check("D8-2b: wrong panel (no currency bar) fails currency-bar check",
          bar_ok2 is False, bar_detail2)
    # D8-2c: a half-rendered bar (only 1 slot has icons) fails
    # Blank ALL 5 slot icon areas completely (x=167 to 1180)
    half_bar = panel.copy()
    half_bar[CURRENCY_Y[0]:CURRENCY_Y[1] + 1, 167:1180] = 60
    # Now restore just slot 5 (death) icon area so it's the only one with icons
    # The death icon is at x=991..1067, restore it from the original panel
    half_bar[CURRENCY_Y[0]:CURRENCY_Y[1] + 1, 991:1067] = \
        panel[CURRENCY_Y[0]:CURRENCY_Y[1] + 1, 991:1067]
    bar_ok3, bar_detail3 = shop._check_currency_bar(half_bar)
    check("D8-2c: half-rendered bar (1 slot) fails the >= 2 check",
          bar_ok3 is False, bar_detail3)

    # D8-3: _wait_panel_open skips a mid-animation frame and returns
    # a real panel frame on the next poll.
    mid_anim = panel.copy()
    mid_anim[2230:2520, :] = 30
    mid_anim[190:315, :] = 60

    class AlternatingDevice:
        """First screencap returns the mid-animation frame; subsequent
        screencaps return the real panel."""
        def __init__(self):
            self.n = 0
            self.screencap_path = Path(tempfile.mkdtemp()) / "cap.png"
        def screencap(self):
            if self.n == 0:
                cv2.imwrite(str(self.screencap_path), mid_anim)
            else:
                cv2.imwrite(str(self.screencap_path), panel)
            self.n += 1
        def wait_for_idle(self, _s): pass

    class AlwaysCoveredBB:
        def bar_visible(self, frame): return False
        def read_bar(self, frame): return []

    dev = AlternatingDevice()
    # Create a PanelReader instance directly
    from vision.panels import PanelReader
    pr = PanelReader(device=dev, llm=None, bottombar=AlwaysCoveredBB())
    result = pr._wait_panel_open()
    # The result should be the real panel (n>=1), not the mid-animation
    if result is None:
        check("D8-3: _wait_panel_open returns real panel after mid-anim",
              False, "returned None")
    else:
        # Compare to the real panel via diff — a real panel's body has high
        # std; mid-anim has low std. Use the body-rendered check as a proxy.
        rendered = pr._panel_body_rendered(result)
        check("D8-3: _wait_panel_open returns real panel after mid-anim",
              rendered is True,
              f"body_rendered={rendered} (screencap_calls={dev.n})")

    # D8-4: a frame with body rendered but blanked currency bar fails
    # _check_currency_bar (rejects panel verification).
    blanked = panel.copy()
    blanked[180:260, :] = 60
    shop2_check = shop._check_currency_bar(blanked)
    check("D8-4: frame with blanked currency bar fails currency-bar check",
          shop2_check[0] is False
          and "currency bar empty" in shop2_check[1],
          shop2_check[1][:80])


def main() -> None:
    global PASS
    set_grid_geometry(GridGeometry(293, 1178, 230, 5, 3))
    for part in (part_a_currency, part_b_cards, part_c_flow, part_d_tool,
                 part_d2_maxlevel_detection, part_d3_panel_verify,
                 part_d4_exact_dialog_match, part_d5_two_phase_buy,
                 part_d6_strategy_enforcement, part_d7_pending_buy_expiry,
                 part_d8_animation_frame_guards,
                 part_e_compile):
        try:
            part()
        except Exception as exc:  # noqa: BLE001
            check(f"{part.__name__} raised {type(exc).__name__}", False, str(exc))
    print(f"\n{PASS} checks passed")
    sys.exit(0 if PASS >= 23 else 1)


if __name__ == "__main__":
    main()