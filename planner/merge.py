"""Merge candidate ranking shared by all planners.

When several merges are possible the game rewards picking a valuable one:
merging to a max-level item is the terminal goal (its popup says only "Feed
to the Devourer." and it yields the best food), and higher-level merges beat
lower-level ones. Merging the Devourer's current craving conflicts with
feeding it (feeding the craved creature grants a bonus), so craved merges
are deprioritized unless they're the only merge available.

Every planner uses this same ordering so behavior is consistent:
- HeuristicPlanner returns the best merge (was: the first dict group).
- LLMPlanner._menu lists merges best-first so menu item #1 is the best merge.
- VisionDrivenPlanner's get_board_state shows the best merge as a hint.

Scoring is a deterministic sort key (higher = better):
  1. result_is_max — merging produces a known max-level item (+bonus)
  2. merged level  — higher-level pairs preferred (matches the existing
     "prefer merges of higher-level items" rule)
  3. craved        — merging the craved item deprioritized (feed it first)
  4. tiebreak      — (row, col) so the order never depends on dict insertion
"""

import re
from collections import defaultdict

from planner.constants import CHAMPION_PREFIXES, MERGE_MIN_MARGIN, craved_matches, item_name_matches

_LEVEL_RE = re.compile(r"_lvl(\d+)$")


def item_level(item_id: str) -> int:
    """Numeric level from an id's `_lvlN` suffix (0 when there is none)."""
    m = _LEVEL_RE.search(item_id or "")
    return int(m.group(1)) if m else 0


def next_level_id(item_id: str) -> str | None:
    """id with level+1 (e.g. skeleton_lvl4 -> skeleton_lvl5), or None."""
    m = _LEVEL_RE.search(item_id or "")
    if not m:
        return None
    return item_id[: m.start()] + f"_lvl{int(m.group(1)) + 1}"


def _chain_family(item_id: str, chain_map: dict[str, list[str]]) -> str | None:
    if not chain_map:
        return None
    for family, ids in chain_map.items():
        if item_id in ids:
            return family
    return None


def merge_result_id(item_id: str, chain_map: dict[str, list[str]] | None = None) -> str | None:
    """Best-known id produced by merging two `item_id`: the next level when the
    id has a numeric level, else the next hop in a banked family chain."""
    nxt = next_level_id(item_id)
    if nxt is not None:
        return nxt
    family = _chain_family(item_id, chain_map or {})
    if family is not None:
        ids = chain_map[family]
        try:
            i = ids.index(item_id)
        except ValueError:
            return None
        if i + 1 < len(ids):
            return ids[i + 1]
    return None


def result_is_max(item_id: str, *, max_level_ids=None, chain_map=None) -> bool:
    """True when merging two `item_id` yields a known max-level item."""
    max_level = set(max_level_ids or ())
    nxt = next_level_id(item_id)
    if nxt is not None and nxt in max_level:
        return True
    family = _chain_family(item_id, chain_map or {})
    if family is not None:
        ids = chain_map[family]
        try:
            i = ids.index(item_id)
        except ValueError:
            return False
        if i + 1 < len(ids) and ids[i + 1] in max_level:
            return True
    return False


def is_merge_material(item_id: str | None, *, chain_map=None,
                      max_level_ids=None) -> bool:
    """True when the item can still merge up a KNOWN next level — it is
    mid-chain MERGE MATERIAL, not feed-only dead weight.

    The feed ranking uses this to never sacrifice cheap merge-chain material
    (bone -> ribcage -> skeletonN, or any id whose next hop is known) as
    generic feed when a non-material target exists: material exists to BUILD
    merges, feeding it destroys pairs. Max-level ids are not material (their
    merge line is gone — feeding is their only action), and items whose next
    hop isn't known (no `_lvlN` suffix AND no banked family chain) are not
    treated as material either — they fall to the normal feed ranking."""
    if not item_id:
        return False
    if item_id in (max_level_ids or ()):
        return False
    return merge_result_id(item_id, chain_map) is not None


def chain_reaches_max(item_id: str, *, max_level_ids=None,
                      chain_map=None, hops: int = 8) -> bool:
    """True when merging `item_id` leads to a known max-level item within
    `hops` merge generations (walking merge_result_id each step).

    This is the chain-lookahead that ranks income families correctly: a lone
    icerune_lvl1 pair is a level-1 merge with no max-level RESULT, but its
    chain reaches max-level icerune_lvl3 in 2 hops — and max-level stacks are
    the Ice income (collected for runes / fed at max value). Without the
    lookahead the ranking buries such merges behind unrelated higher-level
    pairs and income stacks never get merged up.

    Default `hops=8` covers the full skeleton chain (bone -> ribcage ->
    skeleton_lvl1 -> ... -> skeleton_lvl7, 8 hops from the earliest item to
    the max-level entry). The score_merge precedence only uses this as a
    binary bonus (reaches or not), so a higher default just means a longer
    chain still gets the boost — it doesn't change the ordering within a
    single chain."""
    max_level = set(max_level_ids or ())
    if not max_level:
        return False
    cur = merge_result_id(item_id, chain_map=chain_map)
    for _ in range(max(1, hops)):
        if cur is None:
            return False
        if cur in max_level:
            return True
        cur = merge_result_id(cur, chain_map=chain_map)
    return False


def score_merge(item_id: str, cells, *, max_level_ids=None, chain_map=None,
                craved_item: str | None = None) -> tuple:
    """Sort key for a merge group of identical items (higher = better).

    Order of precedence: max-level result > chain reaches max within 2 hops
    (income families — e.g. icerune stacks merged up so they can be collected
    at max value) > avoid the craved item > higher level > top-left position.
    The craved-penalty outranks level because feeding the craved creature
    grants a bonus, so merging it should only happen when no better merge
    exists.
    """
    max_bonus = 1 if result_is_max(item_id, max_level_ids=max_level_ids,
                                   chain_map=chain_map) else 0
    lookahead = 1 if not max_bonus and chain_reaches_max(
        item_id, max_level_ids=max_level_ids, chain_map=chain_map) else 0
    lvl = item_level(item_id)
    craved = 1 if craved_item and item_name_matches(item_id, craved_item) else 0
    r, c = min((cell.row, cell.col) for cell in cells)
    return (max_bonus, lookahead, -craved, lvl, -r, -c)


def destroys_craving(item_id: str, n_cells: int, *, craved_item: str | None,
                      craved_level: int | None,
                      craved_need: int | None) -> bool:
    """True when merging a pair of `item_id` would destroy needed craving material.

    Fires only on full knowledge: the merged id is the EXACT craved id at the
    craved level (`craved_level` from the menu read — never the degraded
    bubble-only match) and the surviving at-level count (`n_cells - 2`) falls
    below the remaining need. Merges that BUILD the craving (precursor ids)
    and merges from a surplus (`n_cells - 2 >= need`) are unaffected.
    """
    if not item_id or not craved_item or craved_level is None:
        return False
    if craved_need is None or craved_need <= 0:
        return False
    if not craved_matches(item_id, craved_item, craving_level=craved_level):
        return False
    return (n_cells - 2) < craved_need


def ranked_merge_groups(board, *, max_level_ids=None, chain_map=None,
                        craved_item: str | None = None,
                        craved_level: int | None = None,
                        craved_need: int | None = None,
                        exclude_pairs: set | None = None):
    """All mergeable identical-item groups on the board, best-first.

    Same exclusions as the merge gate elsewhere (confident margin, no
    champions, no max-level ids). Each entry is (item_id, [best pair of cells]).
    `exclude_pairs` = set of ((r,c),(r,c)) pairs to never propose (merge_noop
    backoff): the best non-excluded adjacent pair is chosen per group, and a
    group with no non-excluded pair is dropped entirely.
    Groups that would destroy needed craving material (see
    `destroys_craving`) are dropped too — merging the last two craved
    monsters into the next level starves the craving.
    """
    max_level = set(max_level_ids or ())
    exclude = {(tuple(a), tuple(b)) for a, b in (exclude_pairs or ())}
    groups: dict[str, list] = defaultdict(list)
    for cell in board.cells:
        if cell.item_id \
                and not cell.item_id.startswith(CHAMPION_PREFIXES) \
                and cell.item_id not in max_level:
            groups[cell.item_id].append(cell)
    ranked = []
    for item_id, cells in groups.items():
        confident = [c for c in cells if c.margin >= MERGE_MIN_MARGIN]
        confident.sort(key=lambda c: (c.row, c.col))
        # Count ALL cells of this id (not just confident ones): an
        # unconfident cell can't merge but is still a feedable craving
        # candidate, so it counts toward the surviving material.
        if destroys_craving(item_id, len(cells), craved_item=craved_item,
                             craved_level=craved_level, craved_need=craved_need):
            continue
        pair = None
        for i in range(len(confident) - 1):
            cand = (confident[i], confident[i + 1])
            if ((cand[0].row, cand[0].col), (cand[1].row, cand[1].col)) not in exclude:
                pair = cand
                break
        if pair is not None:
            ranked.append((item_id, list(pair)))
    ranked.sort(key=lambda g: score_merge(g[0], g[1], max_level_ids=max_level,
                                          chain_map=chain_map, craved_item=craved_item),
                reverse=True)
    return ranked
