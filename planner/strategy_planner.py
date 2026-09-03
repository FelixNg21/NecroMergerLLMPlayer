"""Strategy Planner (System 2) - Deliberative strategy selection.

Runs periodically to evaluate the board state, rune economy, and feats,
then commits a strategic objective via set_strategy tool.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from planner.llm_client import LLMClient

if False:
    from planner.vision_drive import VisionDrivenPlanner


class StrategyPlanner:
    """System 2: Deliberative strategy selection. Runs periodically 
    to evaluate board state and commit a strategic objective."""
    
    # Meta-goal definitions with their default target families and threshold checks
    META_GOALS = {
        "slime_generation": {
            "target_family": "slimevat",
            "threshold_check": lambda state: state.get("slime", {}).get("count", 0) < 20,
            "description": "slime is low → prioritize slimevat builds, slime-producing minions",
        },
        "mana_generation": {
            "target_family": "manapool",
            "threshold_check": lambda state: state.get("mana_pct", 100) < 30,
            "description": "mana is low → prioritize manapool builds, mana-producing minions",
        },
        "rune_economy": {
            "target_family": "grave",  # graves spawn runes via chests
            "threshold_check": lambda state: sum(state.get("runes", {}).values()) < 50,
            "description": "rune balances low → prioritize chest spawns, rune merges, rune feeds",
        },
        "champion_combat": {
            "target_family": None,  # depends on available high-damage minions
            "threshold_check": lambda state: state.get("champion") is not None,
            "description": "champion present → prioritize high-damage minions, attack",
        },
        "board_management": {
            "target_family": None,
            "threshold_check": lambda state: state.get("board_congestion", 0) > 0.8,
            "description": "board congested → merge aggressively, feed non-critical",
        },
        "feeding_optimization": {
            "target_family": None,
            "threshold_check": lambda state: (
                state.get("satiety_remaining", 100) < 20 and state.get("satiety_capacity", 100) > 0
            ),
            "description": "satiety near cap → feed efficiently, avoid overflow",
        },
        "darkness_generation": {
            "target_family": "darkstores",
            "threshold_check": lambda state: state.get("darkness", 0) < 10,
            "description": "darkness low → prioritize darkstores builds",
        },
    }
    
    def __init__(self, 
                 client: "LLMClient",
                 vision_planner: "VisionDrivenPlanner",
                 interval_steps: int = 20,
                 min_interval_steps: int = 5):
        self.client = client
        self.vision_planner = vision_planner
        self.interval_steps = interval_steps
        self.min_interval_steps = min_interval_steps
        self._last_run_step = 0
        
    def should_run(self, step_count: int) -> bool:
        """Check if it's time to run strategy planning."""
        return (step_count - self._last_run_step) >= self.interval_steps
    
    def plan(self, board, frame) -> Optional[str]:
        """Run strategy deliberation. Returns set_strategy feat text or None.
        
        This is the main entry point. Implementation:
        1. Gather comprehensive state
        2. Build the strategy prompt
        3. Call the LLM with the strategy tool
        4. Execute set_strategy tool call
        5. Return the committed feat text
        """
        if self.vision_planner.client is None:
            return None
            
        # 1. Gather state
        state = self._gather_state(board, frame)
        
        # Check if any meta-goal threshold is met
        meta_goal = self._evaluate_meta_goals(state)
        
        # 2. Build prompt
        prompt = self._build_prompt(state)
        
        # 3. Call LLM with strategy tool
        try:
            content, tool_calls, full_msg = self.client.chat(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                tools=[self._get_strategy_tool()],
                tool_choice="required",
                max_tokens=512,
            )
        except Exception as exc:
            self.vision_planner.log.log("strategy_planner_error", error=str(exc))
            return None
            
        # 4. Parse tool call
        if not tool_calls:
            return None
            
        for tc in tool_calls:
            if tc.get("type") == "function" and tc["function"]["name"] == "set_strategy":
                args = json.loads(tc["function"]["arguments"])
                feat = args.get("feat")
                meta_goal = args.get("meta_goal")
                target_family = args.get("target_family")
                
                # Build the strategy object
                strategy = {
                    "step": self.vision_planner._step_count,
                }
                if feat:
                    strategy["feat"] = feat
                if meta_goal:
                    strategy["meta_goal"] = meta_goal
                if target_family:
                    strategy["target_family"] = target_family
                
                # Execute the tool call via vision planner's tool executor
                self.vision_planner._exec_tool({"name": "set_strategy", "arguments": json.dumps(args)})
                
                # Return the committed strategy text
                if feat:
                    return feat
                elif meta_goal:
                    family_str = f" (target: {target_family})" if target_family else ""
                    return f"META {meta_goal}{' (target: ' + target_family + ')' if target_family else ''}"
        
        return None
    
    def _gather_state(self, board, frame) -> Dict[str, Any]:
        """Gather all relevant state for strategy decision."""
        vp = self.vision_planner
        state = {}
        
        # Board summary
        occupied = []
        for cell in board.cells:
            if cell.occupied and cell.item_id:
                state.setdefault("items", []).append({
                    "row": cell.row,
                    "col": cell.col,
                    "item_id": cell.item_id,
                    "score": cell.score,
                    "margin": cell.margin,
                })
        
        # Resources
        state["runes"] = vp._currency or {}
        state["mana_pct"] = vp._mana_fraction * 100 if vp._mana_fraction else None
        
        # Slime
        slime_count = vp.fallback.slime_count if vp.fallback else None
        slime_capacity = vp.fallback.slime_capacity if vp.fallback else None
        state["slime"] = {"count": slime_count, "capacity": slime_capacity}
        
        # Mana
        state["mana_pct"] = vp._mana_fraction * 100 if vp._mana_fraction else None
        
        # Feats
        state["feats"] = (vp._feats_cache or {}).get("feats", [])
        
        # Craving
        state["craving"] = vp._craving_cache
        
        # Resources
        state["mana_pct"] = vp._mana_fraction * 100 if vp._mana_fraction else None
        state["slime"] = {"count": slime_count, "capacity": slime_capacity}
        
        # Satiety
        satiety_remaining = vp.fallback.satiety_remaining if vp.fallback else None
        satiety_capacity = vp.fallback.satiety_capacity if vp.fallback else None
        state["satiety_remaining"] = satiety_remaining
        state["satiety_capacity"] = satiety_capacity
        
        # Champion
        state["champion"] = vp._last_champion
        
        # Current strategy
        state["current_strategy"] = vp._strategy
        state["strategy_fresh"] = vp._strategy_fresh()
        state["strategy_affordable"] = vp._strategy_affordable()
        
        # Station costs
        state["station_costs"] = vp.shop.cost_cache if vp.shop else {}
        
        # Stations on board
        station_families = ["grave", "manapool", "slimevat", "darkstores", "icebox"]
        stations = {}
        for fam in station_families:
            counts = vp._count_stations_on_board(board, fam)
            if counts:
                stations[fam] = counts
        state["stations_on_board"] = stations
        
        # Board congestion
        occupied_cells = sum(1 for c in board.cells if c.occupied)
        total_cells = len(board.cells)
        state["board_congestion"] = occupied_cells / total_cells if total_cells > 0 else 0
        
        # Best hints
        state["best_merge"] = vp._best_merge_line(board)
        state["best_spawn"] = vp._best_spawn_line(board)
        state["best_feed"] = vp._best_feed_line(board)
        
        # Champion
        state["champion"] = vp._last_champion
        
        # Station costs
        state["station_costs"] = vp.shop.cost_cache if vp.shop else {}
        
        return state
    
    def _evaluate_meta_goals(self, state: Dict[str, Any]) -> Optional[str]:
        """Check if any meta-goal threshold is met and return the highest priority one."""
        # Priority order
        priority = [
            "champion_combat",
            "feeding_optimization",
            "board_management",
            "slime_generation",
            "mana_generation",
            "darkness_generation",
            "rune_economy",
        ]
        
        for goal in priority:
            config = self.META_GOALS[goal]
            try:
                if config["threshold_check"](state):
                    return goal
            except Exception:
                continue
        return None
    
    def _get_system_prompt(self) -> str:
        """Build the system prompt for the strategy LLM."""
        # This should match the updated system prompt in vision_drive.py
        return """You are the STRATEGIST for a NecroMerger bot. Your job: evaluate the current game state and commit to ONE strategic objective using the `set_strategy` tool.

AVAILABLE FEATS (pick exactly ONE from this list):
- "Build a Mana Pool."
- "Own a lvl 3+ Grave."
- "Own a lvl 5+ Skeleton."
- "Reach Devourer level N."
- "Open a chest."
- "Defeat a Champion."
- "Collect Nx Mana."
- "Level N."

RULES:
- Choose EXACTLY ONE: either a feat from the list above, OR a meta-goal.
- Explicit feats ALWAYS take priority over meta-goals.
- META-GOALS (use when no specific feat applies):
  - slime_generation: slime is low → prioritize slimevat builds, slime-producing minions
  - mana_generation: mana is low → prioritize manapool builds, mana-producing minions
  - rune_economy: rune balances low → prioritize chest spawns, rune merges, rune feeds
  - champion_combat: champion present → prioritize high-damage minions, attack
  - board_management: board congested → merge aggressively, feed non-critical
  - feeding_optimization: satiety near cap → feed efficiently, avoid overflow
  - darkness_generation: darkness low → prioritize darkstores builds

OUTPUT: Call `set_strategy` with the chosen feat OR meta_goal + target_family."""

    def _get_strategy_tool(self) -> Dict[str, Any]:
        """Return the set_strategy tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "set_strategy",
                "description": (
                    "Commit to a strategy: pick ONE active feat to pursue (from the "
                    "`Feats (tier N):` line of get_board_state) OR a meta-goal, "
                    "and optionally the item family to build toward. For the next ~15 steps "
                    "the priority layer boosts YOUR chosen objective instead of the "
                    "closest-to-done default, and the board state shows your active "
                    "strategy. Call again any time to change course. Only feats "
                    "shown in the Feats line are valid for feat strategies."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "feat": {"type": "string",
                                 "description": "feat name exactly as shown in the Feats line (explicit feat takes priority over meta-goal)"},
                        "meta_goal": {"type": "string",
                                      "enum": ["slime_generation", "mana_generation", "rune_economy", 
                                               "champion_combat", "board_management", "feeding_optimization",
                                               "darkness_generation"],
                                        "description": "meta-goal when no explicit feat applies (e.g. slime_generation, mana_generation, rune_economy)"},
                        "target_family": {"type": "string",
                                          "description": "optional family to build toward (e.g. 'grave', 'manapool', 'slimevat', 'darkstores')"},
                    },
                    "required": [],
                },
            }
        }
    
    def _build_prompt(self, state: Dict[str, Any]) -> str:
        """Build the strategy prompt for the LLM."""
        parts = []
        
        # Board summary
        items = state.get("items", [])
        if items:
            parts.append("BOARD:")
            for item in sorted(items, key=lambda x: (x["row"], x["col"])):
                parts.append(f"  ({item['row']},{item['col']}): {item['item_id']}")
        
        # Resources
        runes = state.get("runes", {})
        if runes:
            parts.append(f"RUNES: {', '.join(f'{k}={v}' for k,v in runes.items())}")
        
        # Mana/Slime
        mana = state.get("mana_pct")
        if mana is not None:
            parts.append(f"MANA: {mana:.0f}%")
        
        slime = state.get("slime", {})
        if slime.get("count") is not None:
            cap = slime.get("capacity")
            cap_str = f"/{cap}" if cap else ""
            parts.append(f"SLIME: {slime['count']}{cap_str}")
        
        # Feats
        feats = state.get("feats", [])
        if feats:
            parts.append("FEATS:")
            for f in feats:
                name = f.get("name", "?")
                done = f.get("count_done", 0)
                req = f.get("count_required", 0)
                reward = f.get("reward", 0)
                parts.append(f"  {name}: {done}/{req} (reward +{reward})")
        
        # Craving
        craving = state.get("craving")
        if craving:
            c = craving
            parts.append(f"CRAVING: {c['item']} (lvl {c['level']}) {c['count_done']}/{c['count_required']} (reward +{c['reward']})")
        
        # Current strategy
        if state.get("current_strategy"):
            s = state["current_strategy"]
            parts.append(f"CURRENT STRATEGY: {s.get('feat') or s.get('meta_goal')}")
        
        # Station costs
        costs = state.get("station_costs", {})
        if costs:
            parts.append("STATION COSTS:")
            for fam, cost in costs.items():
                parts.append(f"  {fam}: {cost}")
        
        if state.get("stations_on_board"):
            parts.append("STATIONS ON BOARD:")
            for fam, counts in state["stations_on_board"].items():
                parts.append(f"  {fam}: " + ", ".join(f"lvl{lvl}={cnt}" for lvl, cnt in counts.items()))
        
        # Best hints
        if state.get("best_merge"):
            parts.append(f"BEST MERGE: {state['best_merge']}")
        if state.get("best_spawn"):
            parts.append(f"BEST SPAWN: {state['best_spawn']}")
        if state.get("best_feed"):
            parts.append(f"BEST FEED: {state['best_feed']}")
        
        # Champion
        if state.get("champion"):
            parts.append(f"CHAMPION: {state['champion']}")
        
        state_text = "\n".join(parts)
        
        return f"""CURRENT GAME STATE:
{state_text}

Reply with ONLY a `set_strategy` tool call choosing ONE feat from the list above OR a meta-goal with target_family."""

    def _evaluate_meta_goals(self, state: Dict[str, Any]) -> Optional[str]:
        """Check if any meta-goal threshold is met and return the highest priority one."""
        priority = [
            "champion_combat",
            "feeding_optimization",
            "board_management",
            "slime_generation",
            "mana_generation",
            "darkness_generation",
            "rune_economy",
        ]
        
        for goal in priority:
            config = self.META_GOALS[goal]
            try:
                if config["threshold_check"](state):
                    return goal
            except Exception:
                continue
        return None


def create_strategy_planner(vision_planner: "VisionDrivenPlanner", 
                            interval_steps: int = 20) -> Optional["StrategyPlanner"]:
    """Factory function to create StrategyPlanner with shared client."""
    if vision_planner.client is None:
        return None
    return StrategyPlanner(
        client=vision_planner.client,
        vision_planner=vision_planner,
        interval_steps=interval_steps
    )
