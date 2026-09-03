"""LLMPlanner: ask an LLM (llama.cpp server) to pick the next move.

Text-only observation for now: the board is rendered as a 5x3 grid of item
ids. The LLM's reply is parsed as JSON, validated against the same rules as
HeuristicPlanner, and any failure falls back to the heuristic. Raw replies are
logged to session.jsonl as `llm_plan` events for debugging; every full
prompt+reply exchange is logged to llm_chats.jsonl.
"""

import json
import re
import time
from pathlib import Path

from metrics.logger import SessionLog
from planner.agent import (
    COLLECT_TAPS,
    CONGESTION_THRESHOLD,
    FEED_PROGRESS_MAX_VALUE,
    HeuristicPlanner,
    Move,
    Planner,
    SPAWN_PREFIXES,
    SPAWN_TAPS,
    STATION_PREFIXES,
    _feedable_pool,
    best_feed_cell,
    necromerger_cell,
)
from vision.hud import COLLECT_MANA_MAX, SPAWN_MANA_MIN, read_mana_fraction
from planner.agent import best_attack_pair
from planner.constants import (
    CHAMPION_PREFIXES,
    CHEST_PREFIXES,
    MERGE_MIN_MARGIN,
    SATIETY_TOL_FRACTION,
    craved_matches,
    is_craved_precursor,
)
from planner.priorities import INCOME_FAMILIES
from planner.llm_client import LLMClient, LLMError
from planner.merge import ranked_merge_groups, is_merge_material
from vision.grid import BoardState, Cell


# Same-family-different-level merge gate (Aug 26). When a cell's top-2
# templates are different levels of the same family AND the margin is below
# this, the merge is refused (the cell is in a bob phase the bank can't
# disambiguate). SAME_FAMILY_MIN_MARGIN in vision/pipeline.py is the
# first-line wipe; this is the second-line validator.
NEIGHBOR_MIN_MARGIN = 0.20


def _same_family_diff_level(a: str | None, b: str | None) -> bool:
    """True when a, b are same creature family at different levels (e.g.
    skeleton_lvl3 vs skeleton_lvl4). Skips None and equal ids.
    """
    if not a or not b or a == b:
        return False
    base_a, _, _ = a.rpartition("_lvl")
    base_b, _, _ = b.rpartition("_lvl")
    return bool(base_a) and base_a == base_b


SYSTEM_PROMPT = """You are the brain of a bot playing NecroMerger on a 5x3 board.
Rules:
- Two identical items merge into one of the next level (e.g. bone_lvl1 + bone_lvl1 -> bone_lvl2).
- A grave spawns new items when tapped. Spawning costs mana (~10) and silently
  does nothing when the mana bar is empty; when no spawn move is listed, the
  mana bar is too low to spawn — feed or collect instead.
- Tapping the necromancer collects mana. Do not collect when the mana bar is full.
- Feed a creature to the Devourer to gain food.
- NEVER feed stations (grave, necromancer, manapool, manapot) to the Devourer.
- Prefer merges, especially of higher-level items; keep the board from filling up.

You will be given a numbered list of VALID moves. Decide what the best move is then reply with ONLY a JSON object
selecting exactly one: {"choice": N}  (N is a number from the list)."""


def _render_board(board: BoardState) -> str:
    lines = []
    for r in range(board.rows):
        row = []
        for c in range(board.cols):
            cell = board.cell_at(r, c)
            row.append(f"({r},{c}) {cell.item_id}" if cell and cell.item_id else f"({r},{c}) _")
        lines.append("  ".join(row))
    return "\n".join(lines)


def _match_object(text: str, start: int) -> tuple[str, int]:
    """Return (json-slice, end_index) of the object opened at `start`.

    Brace depth is counted string-aware so braces inside quoted strings don't
    confuse the match. Returns ("", -1) when no matching close exists.
    """
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1], i
    return "", -1


def _extract_json(content: str) -> dict:
    """Pull the last complete JSON object out of the model's reply (thinking
    bleed, fences, prose...). Handles nested objects and braces in strings."""
    content = content.strip()
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1].strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if fence:
        content = fence.group(1)
    content = content.strip()
    starts = [m.start() for m in re.finditer(r"\{", content)]
    if not starts:
        raise ValueError(f"no JSON object in reply: {content[:200]!r}")
    # Prefer the object whose matching close leaves only whitespace after it
    # (clean tail). Otherwise fall back to the last object that parses, so
    # trailing prose after a flat object is still tolerated.
    for start in reversed(starts):
        obj, end = _match_object(content, start)
        if end != -1 and not content[end + 1:].strip():
            try:
                return json.loads(obj)
            except json.JSONDecodeError:
                continue
    for start in reversed(starts):
        obj, end = _match_object(content, start)
        if end == -1:
            continue
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON object in reply: {content[:200]!r}")


class LLMPlanner(Planner):
    def __init__(self, client: LLMClient | None = None,
                 base_url: str = "http://localhost:8080",
                 log: SessionLog | None = None,
                 chat_log_path: Path = Path("llm_chats.jsonl"),
                 reasoning: bool = True,
                 chain_map: dict[str, list[str]] | None = None,
                 feed_values: dict[str, int] | None = None,
                 damage_values: dict[str, int] | None = None,
                 live: bool = True):
        super().__init__()
        # live=False (static --screenshot regression runs) must be genuinely
        # offline: no default client construction, so a running llama server
        # can never influence a frozen regression check.
        self.live = live
        self.client = client or (LLMClient(base_url=base_url) if live else None)
        self.fallback = HeuristicPlanner(noop=self._noop)  # shared noop backoff
        self.log = log or SessionLog()
        self.chat_log_path = chat_log_path
        self.reasoning = reasoning
        self.chain_map = chain_map
        if feed_values is not None:
            self.fallback.feed_values = feed_values
        if damage_values is not None:
            self.fallback.damage_values = damage_values
        self._step_count = 0
        self._turn = 0                                # API round counter within this planner

    def _menu(self, board: BoardState, mana: float | None = None,
               max_level: set | None = None) -> list[Move]:
        """Valid moves only (merges -> spawns -> feeds -> collect -> idle).

        `mana` gates grave spawns (silently no-op when empty, so withheld below
        threshold); chest spawns are mana-free. `max_level` withholds merges of
        top-level items (the game refuses them; they can only be fed). Merge
        moves are ordered best-first by ranked_merge_groups (max-level result >
        higher level), so menu item #1 is the most valuable merge. Pairs under
        merge_noop backoff are withheld too (the game keeps ignoring them).
        """
        max_level = set(max_level or ())
        mana_low = mana is not None and mana < SPAWN_MANA_MIN
        mana_full = mana is not None and mana >= COLLECT_MANA_MAX
        moves: list[Move] = []
        for item_id, cells in ranked_merge_groups(board, max_level_ids=max_level,
                                                   chain_map=self.chain_map,
                                                   exclude_pairs=self._noop.excluded()):
            a, b = cells[0], cells[1]
            moves.append(Move(kind="merge", cell_a=(a.row, a.col), cell_b=(b.row, b.col)))
        for cell in board.cells:
            if not cell.item_id:
                continue
            is_grave = any(cell.item_id.startswith(p) for p in SPAWN_PREFIXES)
            is_chest = any(cell.item_id.startswith(p) for p in CHEST_PREFIXES)
            if is_grave:
                if mana_low:
                    continue
                moves.append(Move(kind="spawn", cell_a=(cell.row, cell.col), taps=SPAWN_TAPS))
            elif is_chest:
                # chest spawn is mana-free — always offer when on board
                from planner.agent import CHEST_SPAWN_TAPS
                moves.append(Move(kind="spawn", cell_a=(cell.row, cell.col), taps=CHEST_SPAWN_TAPS))
        for target in self._ranked_feed_cells(board, max_level):
            moves.append(Move(kind="feed", cell_a=(target.row, target.col)))
        # Champion combat: offer the best (highest-damage) attacker as the
        # first attack option. Only listed when a champion is actually on
        # the board — best_attack_pair() returns None otherwise, so the
        # empty-board case stays clean.
        pair = best_attack_pair(board, damage_values=self.fallback.damage_values)
        if pair is not None:
            attacker, champion, _dmg = pair
            moves.append(Move(kind="attack",
                              cell_a=(attacker.row, attacker.col),
                              target=(champion.row, champion.col)))
        nec = board.cell_at(*necromerger_cell())
        if nec and not mana_full:
            moves.append(Move(kind="collect", cell_a=(nec.row, nec.col), taps=COLLECT_TAPS))
        if not moves:
            moves.append(Move(kind="idle"))
        return moves

    def _ranked_feed_cells(self, board: BoardState,
                           max_level: set[str]) -> list[Cell]:
        """Feed cells best-first: the single best feed target (craved >
        max-level > lowest feed value > unknown; capacity-limited when the
        Devourer satiety remaining is known) first, then the remaining feedable
        cells in scan order. Mirrors best_feed_cell so the LLM sees the
        recommended feed as the first feed menu option, and the heuristic
        fallback shares the same ranking.
        """
        best = best_feed_cell(board, feed_values=self.fallback.feed_values,
                              max_level_ids=max_level,
                              craved_item=self.fallback.craved_item,
                              craved_level=self.fallback.craved_level,
                              remaining_satiety=self.fallback.satiety_remaining,
                              craving_bonus=self.fallback.craving_bonus_est,
                              satiety_capacity=self.fallback.satiety_capacity,
                              chain_map=self.fallback.chain_map,
                              prefer_max=getattr(self.fallback, "feed_prefer_max", False))
        def _not_precursor(c) -> bool:
            return not is_craved_precursor(
                c.item_id, self.fallback.craved_item,
                self.fallback.craved_level, self.fallback.chain_map)
        feedable = [c for c in board.cells if c.item_id
                    and not any(c.item_id.startswith(p) for p in STATION_PREFIXES)
                    and not c.item_id.startswith(CHAMPION_PREFIXES)
                    and _not_precursor(c)]
        if not feedable:
            return []
        # MERGE-MATERIAL PROTECTION (mirrors best_feed_cell): the offered feed
        # menu prefers non-material / craved-eligible targets and only lists
        # merge material (bone/ribcage/known-mid-chain ids) when nothing else
        # is feedable — feeding material burns a mergeable pair.
        plain, material = _feedable_pool(
            feedable, feed_values=self.fallback.feed_values,
            chain_map=self.fallback.chain_map, max_level_ids=max_level,
            craved_item=self.fallback.craved_item,
            craved_level=self.fallback.craved_level)
        rest = plain or material
        if best is not None:
            return [best] + [c for c in rest
                             if (c.row, c.col) != (best.row, best.col)]
        # The only matching craved item is capacity-wasteful: drop the craved
        # cells from the offered feeds so the LLM can't pick the wasteful feed
        # (spawn entries already lead the menu via `_menu`).
        if self.fallback._craved_feed_wasteful(board):
            craved = self.fallback.craved_item
            rest = [c for c in rest
                    if not (craved and craved_matches(
                        c.item_id, craved,
                        craving_level=self.fallback.craved_level))]
        return rest

    @staticmethod
    def _render_menu(moves: list[Move]) -> str:
        lines = []
        for i, m in enumerate(moves, start=1):
            if m.kind == "merge":
                label = f"merge {m.cell_a} + {m.cell_b}"
            elif m.kind == "spawn":
                label = f"spawn at {m.cell_a}"
            elif m.kind == "feed":
                label = f"feed {m.cell_a}"
            elif m.kind == "attack":
                label = f"attack champion at {m.target} (drag {m.cell_a} onto it)"
            elif m.kind == "collect":
                label = f"collect mana at {m.cell_a}"
            else:
                label = "idle (do nothing)"
            lines.append(f"{i}. {label}")
        return "\n".join(lines)
    
    def next_move(self, board: BoardState, frame=None) -> Move:
        self._step_count += 1
        self._noop.tick()
        reply = None
        try:
            reply, move = self._ask_llm(board, frame)
            mana = read_mana_fraction(frame) if frame is not None else None
            reason = self._validate(board, move, mana,
                                     feed_values=self.fallback.feed_values,
                                     damage_values=self.fallback.damage_values,
                                     chain_map=self.chain_map)
            if reason is None:
                return move
            self.log.log("llm_plan", ok=False, reason=reason, reply=reply,
                         board=_render_board(board))
        except LLMError as exc:
            self.log.log("llm_plan", ok=False, reason="llm_error", detail=str(exc))
        except (ValueError, KeyError, TypeError) as exc:
            self.log.log("llm_plan", ok=False, reason=str(exc), reply=reply)
        return self.fallback.next_move(board, frame)

    def _log_chat(self, messages: list[dict], reply: str) -> None:
        # Accept an optional full assistant message in `reply` position when
        # callers pass a tuple (reply, full_msg). Backwards-compatible callers
        # may still pass only the reply string.
        full_msg = None
        if isinstance(reply, tuple) and len(reply) == 2:
            reply, full_msg = reply
        self._turn += 1
        reasoning = None
        if full_msg:
            reasoning = full_msg.get("reasoning_content") or full_msg.get("reasoning")
        with self.chat_log_path.open("a") as f:
            f.write(json.dumps({"t": time.time(), "step": self._step_count,
                                "turn": self._turn, "messages": messages,
                                "reply": reply, "reasoning": reasoning}) + "\n")

    def _ask_llm(self, board: BoardState, frame=None) -> tuple[str, Move]:
        if self.client is None:
            raise LLMError("llm planner: offline (static mode)")
        mana = read_mana_fraction(frame) if frame is not None else None
        moves = self._menu(board, mana)
        messages = self._build_messages(board, moves, frame)
        return self._respond(messages, moves)

    def _build_messages(self, board: BoardState, moves: list[Move], frame=None) -> list[dict]:
        user = (f"Board:\n{_render_board(board)}\n\n"
                f"Valid moves:\n{self._render_menu(moves)}\n\n"
                f"Pick the best single move. Reply with only JSON.")
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        
    def _respond(self, messages: list[dict], moves: list[Move]) -> tuple[str, Move]:
        """Ask the LLM to pick a move, with reasoning on/off and prefill fallback"""
        if self.reasoning:
            reply, full_msg = self.client.chat(messages, max_tokens=1024)
            self._log_chat(messages, (reply, full_msg))
            try:
                return reply, self._move_from_json(reply, moves)
            except (ValueError, KeyError, TypeError):
                # Thinking bled into `content` and no JSON answer was emitted
                # (llama-server duplicates reasoning into content with
                # --reasoning on). Retry with the prefill trick, which skips
                # thinking and forces a clean {"choice": N} answer.
                self.log.log("llm_plan", ok=False, reason="retry_prefill",
                             reply=reply)
        # Prefill: with reasoning off (or a bleeded retry) this "thinking"
        # model rambles in reasoning_content and exhausts max_tokens before
        # answering. Seeding the assistant turn with the JSON opening skips
        # thinking.
        messages.append({"role": "assistant", "content": '{"choice":'})
        reply, full_msg = self.client.chat(messages, max_tokens=64)
        self._log_chat(messages, (reply, full_msg))
        return reply, self._move_from_json(reply, moves)

    @staticmethod
    def _move_from_json(reply: str, moves: list[Move]) -> Move:
        data = _extract_json(reply)
        choice = data["choice"]
        if not isinstance(choice, int) or not 1 <= choice <= len(moves):
            raise ValueError(f"choice out of range: {choice!r} (menu has {len(moves)})")
        return moves[choice - 1]

    @staticmethod
    def _validate(board: BoardState, move: Move, mana: float | None = None,
                  max_level: set | None = None,
                  feed_values: dict | None = None,
                  damage_values: dict | None = None,
                  satiety_remaining: int | None = None,
                  satiety_capacity: int | None = None,
                  craved_item: str | None = None,
                  craved_level: int | None = None,
                  craving_bonus: int = 0,
                  prefer_item: str | None = None,
                  chain_map: dict[str, list[str]] | None = None,
                  protect_family: str | None = None) -> str | None:
        """Return a reason string if the move is invalid, else None.

        The reason includes the offending item ids / margins so "why did this
        fail" is answerable from the session log (e.g.
        "merge_mismatch:skeleton_lvl3!=skeleton_lvl1" or
        "merge_low_margin:0.09/0.42").

        `item_id is None` has two meanings and the reason must distinguish them
        (an occupied cell with no template label is NOT empty):
          - empty:               cell.occupied is False  -> "<a|b>_empty"
          - unidentified:        cell.occupied is True   -> "<a|b>_unidentified"
        `mana` (HUD bar fill fraction [0,1], None when unknown) gates actions
        that depend on it: spawning below the mana threshold no-ops, and
        collecting above the mana cap wastes a step.
        `max_level` (optional set of template ids) marks items whose popup says
        only "Feed to the Devourer." — merging two of a top-level item is
        always refused by the game, so it's invalid here too.

        Feed-context kwargs (all optional; absent = rule not checkable):
        - `feed_values` + `satiety_remaining/capacity` enforce the game's
          overflow rule on the LLM path (the heuristic gets it via
          best_feed_cell, but the drive model proposes feeds directly):
          a known feed value (+ craving bonus when craved) larger than
          remaining + 10% of capacity wastes food -> `feed_overflow`.
        - Generator protection (the Aug 23 skeleton_lvl4 incident: the model
          fed two 140-feed skeletons into a 78 bar while cheaper food and
          spawns existed): a known feed value above FEED_PROGRESS_MAX_VALUE
          is refused while a cheaper-known feedable fits, unless the target
          is craved / the feat target / the board is congested / no cheaper
          feedable exists -> `feed_generous`. High-level creatures are mana
          generators; feeding them is a last resort.
        - `damage_values` gates the `attack` move: only creatures with a
          known damage value can attack, and the target cell MUST be a
          Champion. Champions NEVER attack us back — drag our creature
          onto the champion to deal its `damage` HP. Unknown-damage
          attackers are refused (`attack_no_damage:<id>`) to keep the
          validator honest: the model can't claim an arbitrary creature is
          useful in combat.
        """
        def cell_reason(label: str, cell) -> str:
            # Never call an occupied cell "empty": the model listed it because
            # a sprite is there, and telling it the cell is empty is a lie that
            # makes it distrust the board state. Distinguish instead.
            if cell is None or not cell.occupied:
                return f"{label}_empty"
            return f"{label}_unidentified"

        def in_bounds(c) -> bool:
            return bool(c) and 0 <= c[0] < board.rows and 0 <= c[1] < board.cols

        max_level = set(max_level or ())
        feed_values = feed_values or {}

        if move.kind == "idle":
            return None
        if not in_bounds(move.cell_a):
            return "out_of_bounds"
        a = board.cell_at(*move.cell_a)
        if move.kind == "merge":
            if move.cell_a == move.cell_b:
                return "same_cell"
            if not in_bounds(move.cell_b):
                return "out_of_bounds"
            b = board.cell_at(*move.cell_b)
            if a is None or a.item_id is None:
                return cell_reason("cell_a", a)
            if b is None or b.item_id is None:
                return cell_reason("cell_b", b)
            if b.item_id != a.item_id:
                return f"merge_mismatch:{a.item_id}!={b.item_id}"
            if a.item_id.startswith(CHAMPION_PREFIXES) \
                    or b.item_id.startswith(CHAMPION_PREFIXES):
                return f"merge_is_champion:{a.item_id}"
            if a.item_id in max_level or b.item_id in max_level:
                return f"merge_max_level:{a.item_id}"
            if a.margin < MERGE_MIN_MARGIN or b.margin < MERGE_MIN_MARGIN:
                return f"merge_low_margin:{a.margin:.2f}/{b.margin:.2f}"
            # Neighbor-suspect (Aug 26). When a cell's top-2 templates are
            # different levels of the same family (e.g. skeleton_lvl4 with
            # runner-up skeleton_lvl5 at margin < NEIGHBOR_MIN_MARGIN) the
            # cell is in a bob phase the bank can't disambiguate. Pairing two
            # such cells is unsafe — the true levels may differ and the game
            # will refuse. Same-family wipe (SAME_FAMILY_MIN_MARGIN) is the
            # first line of defense; this is the second line for cells that
            # were identified by popup-read (no margin) or where the bob
            # phase shifted between classify and validate.
            for cell in (a, b):
                if (cell.runner_up_id
                        and cell.margin < NEIGHBOR_MIN_MARGIN
                        and _same_family_diff_level(cell.item_id, cell.runner_up_id)):
                    return (f"merge_neighbor_suspect:{cell.item_id}/"
                            f"{cell.runner_up_id}@m{cell.margin:.2f}")
            return None
        if move.kind == "spawn":
            if a is None or a.item_id is None:
                return cell_reason("cell_a", a)
            is_grave = any(a.item_id.startswith(p) for p in SPAWN_PREFIXES)
            is_chest = any(a.item_id.startswith(p) for p in CHEST_PREFIXES)
            if not (is_grave or is_chest):
                return f"spawn_not_grave:{a.item_id}"
            if is_grave and mana is not None and mana < SPAWN_MANA_MIN:
                return f"spawn_low_mana:{mana:.2f}"
            # chest is mana-free — no low-mana gate
            # A spawn on a full board silently no-ops (the Aug 25 spawn-spam:
            # 17 spawns onto a 20/20-full board while merge hints showed).
            if not any(not c.occupied for c in board.cells):
                return "spawn_no_room"
            return None
        if move.kind == "feed":
            if a is None or a.item_id is None:
                return cell_reason("cell_a", a)
            if any(a.item_id.startswith(p) for p in STATION_PREFIXES):
                return f"feed_is_station:{a.item_id}"
            if a.item_id.startswith(CHAMPION_PREFIXES):
                return f"feed_is_champion:{a.item_id}"
                # refuse `feed` on a sub-max income stack (Rune or Coin).
            # Both Rune and Coin chains follow the same max-level-or-bust
            # rule: bigger stacks grant more currency per the wiki (icerune
            # lvl3 grants 12 vs lvl1's 2; coin lvl4 grants 30 vs lvl1's 2).
            # The merge ranking already prefers merging these stacks up
            # before feeding; the LLM path bypasses that and can otherwise
            # feed a lvl2 Rune (or coin) for 5 currency when merging it to
            # lvl3 first would have given 12 — a 2x+ loss per feed.
            # Allow when: (a) it's the craved item, (b) it's the prefer
            # target (income boost), (c) the stack is already at max level,
            # or (d) no other feedable exists (desperation fallback).
            is_craved = craved_matches(a.item_id, craved_item,
                                       craving_level=craved_level)
            # MERGE-MATERIAL PROTECTION (the heuristic's `_feedable_pool`
            # already refuses to sacrifice cheap merge chain material —
            # bone/ribcage/skeletonN — when a non-material feedable exists;
            # the LLM path bypassed that split. Feed a merge material item
            # only when (a) it's craved, or (b) nothing plain is feedable.
            material = (is_merge_material(a.item_id, chain_map=chain_map,
                                          max_level_ids=max_level)
                        and not is_craved and a.item_id != prefer_item)
            if material:
                # any cell that COULD be fed (non-station, non-champion),
                # excluding the candidate cell itself.
                def _other_feedable(c):
                    return (c.item_id and c.occupied
                            and (c.row, c.col) != (a.row, a.col)
                            and not c.item_id.startswith(CHAMPION_PREFIXES)
                            and not any(c.item_id.startswith(p)
                                        for p in STATION_PREFIXES))
                any_plain_feedable = any(
                    _other_feedable(c)
                    and not is_merge_material(c.item_id, chain_map=chain_map,
                                              max_level_ids=max_level)
                    for c in board.cells)
                if any_plain_feedable:
                    return f"feed_merge_material:{a.item_id}"
                # DESPERATION fallback: no plain feedable exists, so we may
                # feed merge material rather than starve the Devourer. BUT the
                # fallback is only justified when the board is genuinely FULL.
                # If any cell is free there is room to spawn/merge food instead,
                # so burning a merge-material component is never necessary —
                # refuse regardless of whether a plain feedable happens to
                # exist right now (a "not congested" board with free space must
                # NOT feed merge material; empty cells aren't feedable, so the
                # plain-feedable check above would otherwise let it through).
                has_free_cell = any(not c.occupied for c in board.cells)
                if has_free_cell:
                    return f"feed_merge_material:{a.item_id}"
                # Board is full: if this material is on the ACTIVE STRATEGY'S
                # critical path (e.g. feeding a zombie_lvl2 that is one merge
                # from the required "Own a lvl 3+ Zombie" lvl3), still refuse
                # while ANY other feedable exists — burning the merge that
                # completes the goal is worse than feeding a non-strategy
                # component.
                if (protect_family and a.item_id.startswith(protect_family)
                        and any(_other_feedable(c) for c in board.cells)):
                    return f"feed_strategy_material:{a.item_id}"
            is_income_stack = any(
                a.item_id.startswith(fam) or fam in a.item_id
                for fam in INCOME_FAMILIES)
            if (is_income_stack and a.item_id not in max_level
                    and not is_craved and a.item_id != prefer_item):
                # Income stack below max level — refuse unless the model
                # has no other option. A "valid alternative" includes any
                # non-station, non-champion cell that isn't itself a
                # sub-max income stack — specifically: a max-level income
                # stack (the model should feed that one for max currency),
                # a merge material pair (bone/ribcage), or any other
                # creature. If the only feedable cells are sub-max income
                # stacks, fall through to the existing gates below.
                # Exclude the current target cell from this check — we
                # need OTHER cells that could be fed instead, not itself.
                any_other_feedable = any(
                    c.item_id
                    and c.occupied
                    and (c.row, c.col) != (a.row, a.col)
                    and not c.item_id.startswith(CHAMPION_PREFIXES)
                    and not any(c.item_id.startswith(p)
                                for p in STATION_PREFIXES)
                    and (c.item_id in max_level
                         or not any(c.item_id.startswith(fam) or fam in c.item_id
                                    for fam in INCOME_FAMILIES)
                         or feed_values.get(c.item_id, 0) <= FEED_PROGRESS_MAX_VALUE)
                    for c in board.cells)
                if any_other_feedable:
                    return f"feed_not_max_level:{a.item_id}"
                    # manapot / manapotion Mana-overflow check. Feeding
            # a manapot gives Mana (the wiki's max-level manapotion
            # grants 100% of Mana cap). If the Mana bar is already at
            # 100% the feed is wasted. The Satiety-bar `feed_overflow`
            # check above only checks the food bar; this catches the
            # Mana case for manapot/manapotion items specifically.
            # NOTE: this check runs UNCONDITIONALLY (not gated on
            # feed_values), because the manapot template may not have a
            # recorded food value in the glossary but the item is still
            # a Potion (the only effect is Mana) and a full Mana bar
            # still makes the feed wasted. Running it outside the
            # `if z is not None:` block ensures the check fires even when
            # `manapot_lvl3` is missing from the feed_values dict.
            if (mana is not None and mana >= COLLECT_MANA_MAX
                    and (a.item_id.startswith("manapot")
                         or a.item_id.startswith("manapotion"))):
                return (f"feed_mana_overflow:{a.item_id}:mana={mana:.2f}")
            z = feed_values.get(a.item_id)
            if z is not None:
                is_craved = craved_matches(a.item_id, craved_item,
                                           craving_level=craved_level)
                eff = z + (craving_bonus if is_craved else 0)
                if satiety_remaining is not None:
                    tol = (SATIETY_TOL_FRACTION * satiety_capacity
                           if satiety_capacity else 0)
                    if eff > satiety_remaining + tol:
                        return (f"feed_overflow:{eff}>{satiety_remaining}"
                                f"+{tol:.0f}")
                # Generator protection: expensive creatures are mana
                # generators — feed them only as a last resort.
                if (z > FEED_PROGRESS_MAX_VALUE and not is_craved
                        and a.item_id != prefer_item
                        and satiety_remaining is not None):
                    empty_count = sum(1 for c in board.cells
                                      if not c.occupied)
                    congested = empty_count <= CONGESTION_THRESHOLD
                    cheaper_fits = any(
                        c.item_id in feed_values
                        and c.item_id != a.item_id
                        and feed_values[c.item_id] < z
                        and feed_values[c.item_id] <= satiety_remaining
                        and not any(c.item_id.startswith(p)
                                    for p in STATION_PREFIXES)
                        and not c.item_id.startswith(CHAMPION_PREFIXES)
                        for c in board.cells)
                    if not congested and cheaper_fits:
                        return f"feed_generous:{a.item_id}:{z}"
            return None
        if move.kind == "collect":
            if mana is not None and mana >= COLLECT_MANA_MAX:
                return f"collect_mana_full:{mana:.2f}"
            return None
        if move.kind == "attack":
            # Champion combat: drag our creature onto a champion. Champions
            # are enemies — they only attack if you leave them alive (the
            # game steals food/mana over time), so the rule is: hit them
            # with your best-damage creature as fast as possible.
            damage_values = damage_values or {}
            # The attacker must hold a real creature, not a station or
            # another champion (dragging a champion at a champion is a
            # no-op the game doesn't recognize).
            if a is None or a.item_id is None:
                return cell_reason("cell_a", a)
            if any(a.item_id.startswith(p) for p in STATION_PREFIXES):
                return f"attack_is_station:{a.item_id}"
            if a.item_id.startswith(CHAMPION_PREFIXES):
                return f"attack_is_champion:{a.item_id}"
            if move.target is None or not in_bounds(move.target):
                return "attack_no_target"
            if move.cell_a == move.target:
                return "attack_self_target"
            t = board.cell_at(*move.target)
            if t is None or t.item_id is None:
                return cell_reason("target", t)
            if not t.item_id.startswith(CHAMPION_PREFIXES):
                return f"attack_target_not_champion:{t.item_id}"
            # The attacker must have a known damage value. Without a popup
            # read of this creature's "Takes Damage" column we can't trust
            # the model to know its combat value — refuse rather than guess.
            dv = damage_values.get(a.item_id)
            if dv is None or dv <= 0:
                return f"attack_no_damage:{a.item_id}"
            return None
        return f"bad_kind:{move.kind}"
