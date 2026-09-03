"""Verify the Aug 30 StationShop._pending_buy state machine.

The user reported the bot never bought the grave. Root cause was that
`StationShop._pending_buy` was NEVER set inside `buy()` — it was only
initialized to None, so reading `self.shop._pending_buy` always returned
None. The two-phase buy state machine in the planner couldn't track
pending buys, so the post-loop compelled-confirm never fired (it had
nothing to compel).

These tests verify the pending_buy state machine directly, without
running the full device-driven buy() flow. We patch the read methods
to return canned data and inspect self._pending_buy after each call.
"""
import sys
from unittest.mock import MagicMock
from pathlib import Path

sys.path.insert(0, '.')

import numpy as np
import cv2

from vision.panels import StationShop, BottomBarReader


def _make_shop():
    """Build a StationShop with a stubbed device that always returns a
    blank panel image. The full buy() flow is patched via direct method
    overrides so we can test the pending_buy state machine in isolation."""
    device = MagicMock()
    device.screencap_path = Path("/tmp/buy_pending_test.png")
    cv2.imwrite(str(device.screencap_path),
                np.full((2856, 1280, 3), 100, dtype=np.uint8))

    bottombar = BottomBarReader()
    shop = StationShop(device=device, llm=None, bottombar=bottombar,
                        classifier=None)
    return shop


# ---------- Test A: initial state is None ----------

def test_a_initial_pending_buy_is_none():
    shop = _make_shop()
    assert shop._pending_buy is None
    print("  PASS: A: initial _pending_buy is None")
    return True


# ---------- Test B: directly setting _pending_buy works (state machine) ----

def test_b_pending_buy_state_set():
    shop = _make_shop()
    # Simulate phase 1 completion by directly setting the state.
    shop._pending_buy = {"family": "grave", "step": 0, "dialog": {}}
    assert shop._pending_buy is not None
    assert shop._pending_buy["family"] == "grave"
    # Simulate phase 2 completion by clearing.
    shop._pending_buy = None
    assert shop._pending_buy is None
    print("  PASS: B: _pending_buy set/clear round-trips")
    return True


# ---------- Test C: confirm=false in buy() sets _pending_buy ----------

def test_c_confirm_false_sets_pending_buy():
    """Patch buy() to only run the relevant parts (skip device taps)."""
    shop = _make_shop()

    # Patch the expensive parts
    shop._verify_station_panel = lambda frame: (True, "ok")
    shop.read_currency = lambda frame: {"ice": 100, "poison": 100, "blood": 0,
                                        "moon": 0, "death": 0}
    shop.read_cards = lambda frame: [
        {"index": 0, "x": 400, "station": "grave", "cost_ice": 20,
         "cost_poison": 0, "cost_blood": 0, "cost_moon": 0, "cost_death": 0}
    ]
    # First _dialog_up call: True (dialog up). All subsequent: False
    # (after the confirm tap, dialog closes).
    counter = {"n": 0}
    def _dialog_up(frame):
        # First 2 calls: True (initial dialog read + poll).
        # Calls 3+: False (after confirm tap closes the dialog).
        counter["n"] += 1
        return counter["n"] <= 2
    shop._dialog_up = _dialog_up
    shop.read_dialog = lambda frame: {
        "is_purchase_dialog": True, "station": "grave", "cost_ice": 20,
        "cost_poison": 0, "cost_blood": 0, "cost_moon": 0, "cost_death": 0,
        "cost_icons": [], "cost_icon_override": False,
        "cost_text_sufficient": True,
    }
    shop._panel.open_panel = lambda *a, **k: None
    shop._panel._wait_panel_open = lambda: np.full((2856, 1280, 3), 100,
                                                     dtype=np.uint8)
    shop._panel.close_panel = lambda *a, **k: None
    shop._back = lambda: None
    shop._tap_retry = lambda *a, **k: None
    shop._find_confirm = lambda frame: (400, 400, 0.9)
    shop._family_count = MagicMock(return_value=1)
    shop.device.wait_for_idle = lambda s: None

    # confirm=false: should set _pending_buy
    res = shop.buy("grave", confirm=False)
    assert shop._pending_buy is not None, \
        f"pending_buy not set after confirm=false: {res}"
    assert shop._pending_buy["family"] == "grave"
    assert res.get("pending_buy") is True
    assert "dialog verified" in (res.get("note") or "")
    print("  PASS: C: confirm=false stashes _pending_buy (real buy() path)")
    return True


# ---------- Test D: confirm=true clears _pending_buy after success ---------

def test_d_confirm_true_clears_pending_buy():
    shop = _make_shop()
    # Same patches as test C
    shop._verify_station_panel = lambda frame: (True, "ok")
    shop.read_currency = lambda frame: {"ice": 100, "poison": 100, "blood": 0,
                                        "moon": 0, "death": 0}
    shop.read_cards = lambda frame: [
        {"index": 0, "x": 400, "station": "grave", "cost_ice": 20,
         "cost_poison": 0, "cost_blood": 0, "cost_moon": 0, "cost_death": 0}
    ]
    counter = {"n": 0}
    def _dialog_up(frame):
        # First 2 calls: True (initial dialog read + poll).
        # Calls 3+: False (after confirm tap closes the dialog).
        counter["n"] += 1
        return counter["n"] <= 2
    shop._dialog_up = _dialog_up
    shop.read_dialog = lambda frame: {
        "is_purchase_dialog": True, "station": "grave", "cost_ice": 20,
        "cost_poison": 0, "cost_blood": 0, "cost_moon": 0, "cost_death": 0,
        "cost_icons": [], "cost_icon_override": False,
        "cost_text_sufficient": True,
    }
    shop._panel.open_panel = lambda *a, **k: None
    shop._panel._wait_panel_open = lambda: np.full((2856, 1280, 3), 100,
                                                     dtype=np.uint8)
    shop._panel.close_panel = lambda *a, **k: None
    shop._back = lambda: None
    shop._tap_retry = lambda *a, **k: None
    shop._find_confirm = lambda frame: (400, 400, 0.9)
    # _family_count is called 3 times: phase 1's before_stations, then
    # phase 2's before_stations (still 0 stations pre-confirm), then
    # phase 2's gained (1 station post-confirm). 0, 0, 1.
    _fc_state = {"count": 0}
    def _family_count(frame, family):
        _fc_state["count"] += 1
        if _fc_state["count"] <= 2:
            return 0
        return 1
    shop._family_count = _family_count
    shop.device.wait_for_idle = lambda s: None

    # Phase 1
    shop.buy("grave", confirm=False)
    assert shop._pending_buy is not None
    # Phase 2
    res = shop.buy("grave", confirm=True)
    assert res.get("bought") is True, f"bought should be True: {res}"
    assert shop._pending_buy is None, \
        f"pending_buy not cleared after success: {res}, " \
        f"_pending_buy={shop._pending_buy}"
    print("  PASS: D: confirm=true clears _pending_buy (real buy() path)")
    return True


# ---------- Test E: failed confirm does NOT clear _pending_buy ---------

def test_e_failed_confirm_keeps_pending_buy():
    """When the confirm tap fails (dialog still up after retries), the
    pending_buy must stay so the next step can retry. The previous code
    cleared it OUTSIDE the success check."""
    shop = _make_shop()
    shop._verify_station_panel = lambda frame: (True, "ok")
    shop.read_currency = lambda frame: {"ice": 100, "poison": 100, "blood": 0,
                                        "moon": 0, "death": 0}
    shop.read_cards = lambda frame: [
        {"index": 0, "x": 400, "station": "grave", "cost_ice": 20,
         "cost_poison": 0, "cost_blood": 0, "cost_moon": 0, "cost_death": 0}
    ]
    # _dialog_up is ALWAYS True — the confirm tap never closes the dialog
    shop._dialog_up = lambda frame: True
    shop.read_dialog = lambda frame: {
        "is_purchase_dialog": True, "station": "grave", "cost_ice": 20,
        "cost_poison": 0, "cost_blood": 0, "cost_moon": 0, "cost_death": 0,
        "cost_icons": [], "cost_icon_override": False,
        "cost_text_sufficient": True,
    }
    shop._panel.open_panel = lambda *a, **k: None
    shop._panel._wait_panel_open = lambda: np.full((2856, 1280, 3), 100,
                                                     dtype=np.uint8)
    shop._panel.close_panel = lambda *a, **k: None
    shop._back = lambda: None
    shop._tap_retry = lambda *a, **k: None
    shop._find_confirm = lambda frame: (400, 400, 0.9)
    shop._family_count = MagicMock(return_value=0)
    shop.device.wait_for_idle = lambda s: None

    # Phase 1
    shop.buy("grave", confirm=False)
    assert shop._pending_buy is not None
    # Phase 2 — fails (dialog doesn't close)
    res = shop.buy("grave", confirm=True)
    assert res.get("bought") is False
    # CRITICAL: _pending_buy should still be set so the planner can retry
    assert shop._pending_buy is not None, \
        "_pending_buy was cleared even though confirm failed"
    print("  PASS: E: failed confirm keeps _pending_buy (retry-able)")
    return True


# ---------- Main ----------

if __name__ == "__main__":
    tests = [
        test_a_initial_pending_buy_is_none,
        test_b_pending_buy_state_set,
        test_c_confirm_false_sets_pending_buy,
        test_d_confirm_true_clears_pending_buy,
        test_e_failed_confirm_keeps_pending_buy,
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
            import traceback
            print(f"  ERROR: {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed+failed} StationShop pending_buy tests passed")
    sys.exit(0 if failed == 0 else 1)
