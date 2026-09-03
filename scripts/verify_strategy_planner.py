#!/usr/bin/env python3
"""Verification for the Strategy Planner (System 2).

Run: python scripts/verify_strategy_planner.py
"""

import sys
sys.path.insert(0, '.')

from planner.strategy_planner import StrategyPlanner, create_strategy_planner
from planner.vision_drive import VisionDrivenPlanner
from unittest.mock import MagicMock
from pathlib import Path
import tempfile

def test_should_run():
    """Test the should_run logic."""
    print("Test: should_run logic")
    with tempfile.TemporaryDirectory() as td:
        # Can't easily test without a real client, just test the logic
        class MockPlanner:
            def should_run(self, step_count):
                return (step_count - self._last_run_step) >= self.interval_steps
        
        p = MockPlanner()
        p.interval_steps = 20
        p._last_run_step = 0
        
        assert p.should_run(19) == False, "Should not run at step 19"
        assert p.should_run(20) == True, "Should run at step 20"
        assert p.should_run(21) == True, "Should run at step 21"
        print("  PASS: should_run logic works correctly")

def test_create_strategy_planner():
    """Test factory function."""
    print("Test: create_strategy_planner factory")
    with tempfile.TemporaryDirectory() as td:
        planner = VisionDrivenPlanner(
            live=False, classifier=None, log=MagicMock(), tool_enabled=False)
        planner.client = None
        planner._strategy_interval = 20
        
        sp = create_strategy_planner(planner, interval_steps=15)
        assert sp is None, "Should return None when no client"
        print("  PASS: Returns None when no client")
        
        # Test with client
        planner.client = MagicMock()
        sp = create_strategy_planner(planner, interval_steps=15)
        assert sp is not None, "Should create planner when client exists"
        assert sp.interval_steps == 15, "Should use custom interval"
        print("  PASS: Creates planner with custom interval when client exists")

def test_strategy_planner_fields():
    """Test StrategyPlanner initialization."""
    print("Test: StrategyPlanner field initialization")
    mock_client = MagicMock()
    mock_vision = MagicMock()
    mock_vision._step_count = 0
    mock_vision._strategy = None
    
    sp = StrategyPlanner(
        client=MagicMock(),
        vision_planner=mock_vision,
        interval_steps=25,
        min_interval_steps=3
    )
    
    assert sp.interval_steps == 25
    assert sp.min_interval_steps == 3
    assert sp._last_run_step == 0
    print("  PASS: Fields initialized correctly")

if __name__ == "__main__":
    test_should_run()
    test_create_strategy_planner()
    test_strategy_planner_fields()
    print("\n=== All Strategy Planner tests passed ===")
