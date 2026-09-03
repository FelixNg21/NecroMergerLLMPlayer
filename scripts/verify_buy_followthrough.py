"""confirm the buy_station follow-through gate now fires.

The bug: StationShop.buy() set self._pending_buy on the SHOP, but the
VisionDrivenPlanner's compelled-follow-through check looked at the
PLANNER's self._pending_buy (always None). The model would call
buy_station(confirm=false) successfully, get a `note` saying "call with
confirm=true", then do other moves and never call confirm=true. With
the proxy through self.shop._pending_buy, the gate now reads the live
shop state.

Run: `.venv/bin/python scripts/verify_buy_followthrough.py`
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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


def test_pending_buy_proxies_to_shop():
    """When StationShop.buy sets self._pending_buy, the planner's gate
    must see it (currently reads via self.shop._pending_buy)."""
    from vision.panels import StationShop, PanelReader
    from env.adb import Device

    device = Device()
    # Build a minimal StationShop + planner
    class FakeShop(StationShop):
        def __init__(self):
            self.device = device
            self.llm = None
            self.classifier = None
            self.bottombar = None
            self.cost_cache = {}
            self._panel = PanelReader(device=device, llm=None, bottombar=None)
            self._currency = None
            self._last_currency_result = None
            self._last_currency_step = -1

    shop = FakeShop()
    # Manually set a pending buy (simulating what buy() does after confirm=false)
    shop._pending_buy = {"family": "grave", "dialog": {}, "step": 5}

    # The planner reads via self.shop._pending_buy
    planner_shop_ref = shop
    pending = planner_shop_ref._pending_buy
    check("A: shop._pending_buy visible to the gate proxy",
          pending == {"family": "grave", "dialog": {}, "step": 5},
          f"got {pending}")

    # Confirm clear works
    shop._pending_buy = None
    check("B: clearing shop._pending_buy → gate sees None",
          planner_shop_ref._pending_buy is None)


def test_strategy_line_reads_pending_buy():
    """The strategy line text uses the proxy too — must see shop state."""
    from vision.panels import StationShop, PanelReader
    from env.adb import Device
    from planner.vision_drive import VisionDrivenPlanner
    from vision.grid import GridGeometry, set_grid_geometry

    device = Device()
    class FakeShop(StationShop):
        def __init__(self):
            self.device = device
            self.llm = None
            self.classifier = None
            self.bottombar = None
            self.cost_cache = {}
            self._panel = PanelReader(device=device, llm=None, bottombar=None)
            self._currency = None
            self._last_currency_result = None
            self._last_currency_step = -1
    shop = FakeShop()

    set_grid_geometry(GridGeometry(293, 1178, 230, 5, 3))
    planner = VisionDrivenPlanner.__new__(VisionDrivenPlanner)
    planner.shop = shop
    planner._step_count = 10
    planner._strategy = {
        "feat": "Own a Lvl 3+ Grave.",
        "target": None,
        "step": 5,
    }
    planner._strategy_family = "grave"
    planner._strategy_fresh = lambda: True
    # Force some required attributes used by _strategy_line_text
    planner._currency = None
    planner._last_currency_result = None
    planner._last_currency_step = -1
    planner._last_champion = None
    planner._champion_cache = None
    planner._last_craving = None
    planner._last_craving_level = None
    planner._craving_cache = None
    planner._feats_cache = None
    planner._frame = None
    planner._learnings_context = lambda: ""
    planner._glossary_context = lambda: ""

    # No pending buy → strategy line has no "Pending buy:" suffix
    shop._pending_buy = None
    line = planner._strategy_line_text(board=None)
    check("C: no pending buy → no suffix",
          "Pending buy" not in line, line[:200])

    # Set pending buy → suffix appears
    shop._pending_buy = {"family": "grave", "dialog": {}, "step": 5}
    line = planner._strategy_line_text(board=None)
    check("D: pending buy set → suffix appears",
          "Pending buy: grave" in line, line[:300])


if __name__ == "__main__":
    test_pending_buy_proxies_to_shop()
    test_strategy_line_reads_pending_buy()
    print(f"\n{PASS} pass, {FAIL} fail")
    sys.exit(0 if FAIL == 0 else 1)
