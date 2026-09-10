# -*- coding: utf-8 -*-
"""VisionDrivenPlanner: the LLM reads the screenshot and proposes the move itself.

No valid-move menu (unlike LLMPlanner / VisionLLMPlanner): the model observes
the full emulator frame, decides what to do (merge/spawn/feed/collect/idle) and
returns raw JSON coordinates. Every proposal passes through the same
LLMPlanner._validate gate; invalid moves are fed back to the model with a reason
and it retries (up to max_retries) before the heuristic fallback takes over.
This is where vision is load-bearing: the choice depends on what the model
sees, not on a code-built menu.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from metrics.logger import SessionLog
from planner.agent import (
    CHAMPION_PREFIXES,
    COLLECT_TAPS,
    FEED_PROGRESS_MAX_VALUE,
    HeuristicPlanner,
    Move,
    NECROMERGER_CELL,
    Planner,
    SPAWN_TAPS,
    SPAWN_PREFIXES,
    STATION_PREFIXES,
    best_attack_pair,
    best_feed_cell,
    necromerger_cell,
    _should_feed,
)
from planner.glossary import DEFAULT_PATH as GLOSSARY_DEFAULT
from planner.glossary import (
    read_chains, read_glossary,
    read_merge_chains_full, read_spawn_rates, read_item_stats, read_visual_markers,
    write_merge_chain, write_spawn_rate, write_item_stat, write_visual_marker,
    prune_glossary_by_ids,
    _get_knowledge_dir,
)
from planner.learnings import DEFAULT_PATH as LEARNINGS_DEFAULT
from planner.learnings import (
    append_learning,
    confirm_learning,
    derate_learning,
    prune_learning,
    read_learnings,
    log_learning_outcome,
    demote_stale_learnings,
    Learning,
)
from planner.llm import LLMPlanner, _extract_json
from planner.llm_client import LLMClient, LLMError
from planner.llm_vision import _encode_frame
from planner.merge import is_merge_material, merge_result_id, ranked_merge_groups
from planner.constants import (
    CHEST_PREFIXES,
    CHEST_SPAWN_PREFIXES,
    GRAVE_SPAWN_PREFIXES,
    SATIETY_TOL_FRACTION,
    SLIME_SPAWN_PREFIXES,
    canonical_family,
    craved_matches,
    display_family,
    item_name_matches,
    normalize_item_name,
)
from planner.priorities import (Objective, craving_objective, feat_weights,
                                chest_spawn_objective, income_objective, INCOME_FAMILIES,
                                is_build_strategy, kind_order, parse_obj)
from planner.wiki import fetch_topic_texts, lookup as wiki_lookup, topics_for
from vision.classifier import TemplateClassifier
from vision.cravings import BoardLostError, CravingsReader
from vision.bottombar import BottomBarReader
from vision.champion import ChampionReader
from vision.panels import PanelReader, normalize_station_name
from vision.grid import FALLBACK_GEOMETRY, GridGeometry
from vision.hud import COLLECT_MANA_MAX, SPAWN_MANA_MIN, read_mana_fraction

SCREEN = (1280, 2856)
MAX_DISCOVERY = 4          # identify_item calls per step (covers a proactive batch + optional follow-ups)
MAX_WIKI = 2               # lookup_wiki fact-check rounds the model may make per step
MAX_CRAVINGS = 1           # get_cravings menu cycles allowed per step (intrusive)
MAX_PANELS = 1             # tap_button open/read/close cycles allowed per step (intrusive)
MAX_CHAMPIONS = 1          # get_champions menu cycles allowed per step (intrusive)
MAX_POPUP_READS = 1        # auto popup-body reads of known-but-unbanked items per step
LABEL_MEMORY_TTL = 10      # popup-label persistence lifetime (steps) — safety valve on stale reads
LABEL_MEMORY_PAIR = frozenset({"eyeball", "eyemonster_lvl1"})  # template-indistinguishable pair (cross-bank ~1.0)
CRAVINGS_REFRESH_STEPS = 10  # cache a full menu read for this many steps before re-reading
CHAMPIONS_REFRESH_STEPS = 10  # cache the champion-screen read for this many steps before re-reading
FEATS_REFRESH_STEPS = 15     # cache a feats-panel read for this many steps before re-reading
FEAT_COLLECT_STEPS = 5       # opportunistic feat-collect retry when a done feat sits uncollected
MAX_FEAT_COLLECT = 1         # collect_feat_rewards tool calls per step
# cross-step rejection TTL. A previous step's rejected action stays
# pre-blocked for up to this many steps before the TTL filter expires it
# (the board may have shifted past the rejection's premise by then — a
# feed that overflowed because the bar was full becomes valid again once
# the bar drains). Tuned to 3: a single retry usually surfaces the same
# failure mode, but more than 3 steps in a row is likely a stale
# rejection that the model should re-evaluate.
CROSS_STEP_TTL = 3
LABEL_MIN_SCORE = 0.6      # only show template labels at/above this confidence
KNOWN_DIP_MIN = 0.45       # best-match below this (occupied, unlabeled) = genuinely new item

# Strategy-feat noun classification. The wiki's "Build a X" and
# "Own a Lvl N+ X" feats split into two operational shapes:
#   STATION   - "X" is a station you buy from the Station panel. Owning a
#               lvl N+ X means buying the station and merging copies up to
#               level N. Examples: Grave, Mana Pool, Supply Cupboard, Lectern.
#   CREATURE  - "X" is a creature you merge on the board. Owning a lvl N+ X
#               means merging components to that level. Examples: Skeleton,
#               Zombie, Bone, Ribcage, Spider, Mummy, Bat, Shade, etc.
# The set_strategy note and the per-step strategy line use this to pick
# the right hint (buy_station vs merge/collect on the board). A target_item
# the model supplied that doesn't match the noun category is dropped.
#   OTHER     - The feat is not station-purchase nor target-collection
#               ("Create 5 Bones.", "Beat the Peasant twice.", "Tap the
#               NecroMerger 10 times.", "Reach Devourer level 5."). The hint
#               shows a free-form description and the progress counter; the
#               priority layer does not boost a station/creature kind for it.
STATION_FAMILIES = frozenset({
    "grave", "manapool", "supplycupboard", "foulchicken", "slimevat",
    "altar", "darkstores", "lectern", "fridge", "portal", "crashedsaucer",
    "telepad", "soulgrinder", "prism", "meteor", "throne", "unexpectedparcel",
})

# Meta-goal definitions (moved from the retired StrategyPlanner): threshold
# checks over cached state plus the family to build toward. Evaluated
# code-side into a one-line "Suggested direction" — no LLM call. Every
# check MUST be None-safe (missing data means "unknown", never a goal):
# the old evaluator's darkness check defaulted a missing key to 0 and
# fired unconditionally.
META_GOALS = {
    "champion_combat": {
        "target_family": None,
        "threshold_check": lambda state: state.get("champion") is not None,
        "description": "champion present → prioritize high-damage minions, attack",
    },
    "feeding_optimization": {
        "target_family": None,
        "threshold_check": lambda state: (
            state.get("satiety_remaining") is not None
            and state.get("satiety_capacity") is not None
            and state.get("satiety_capacity") > 0
            and state.get("satiety_remaining") < 20),
        "description": "satiety near cap → feed efficiently, avoid overflow",
    },
    "board_management": {
        "target_family": None,
        "threshold_check": lambda state: (state.get("board_congestion") or 0) > 0.8,
        "description": "board congested → merge aggressively, feed non-critical",
    },
    "slime_generation": {
        "target_family": "slimevat",
        "threshold_check": lambda state: (
            state.get("slime_count") is not None
            and state.get("slime_count") < 20),
        "description": "slime is low → prioritize slimevat builds, slime-producing minions",
    },
    "mana_generation": {
        "target_family": "manapool",
        "threshold_check": lambda state: (
            state.get("mana_pct") is not None and state.get("mana_pct") < 30),
        "description": "mana is low → prioritize manapool builds, mana-producing minions",
    },
    "rune_economy": {
        "target_family": "grave",
        "threshold_check": lambda state: (
            sum((state.get("runes") or {}).values()) < 50),
        "description": "rune balances low → prioritize chest spawns, rune merges, rune feeds",
    },
}
META_GOAL_PRIORITY = (
    "champion_combat",
    "feeding_optimization",
    "board_management",
    "slime_generation",
    "mana_generation",
    "rune_economy",
)

# Craving item family -> (producer station family, feedable/board component ids).
# Closes the "craving for something the board cannot produce" gap: the craving
# objective only boosts FEEDING the item when present, and no strategy, hint,
# or spawn line otherwise connects a craving to its source station (observed:
# eyemonster craving with no Supply Cupboard/Fridge on the board — the bot
# worked feats while the craving sat). Families with unknown producers are
# omitted (mapping returns None -> no line, no behavior change). Component
# ids use the normalize_item_name convention ('Eye in a Jar' -> 'eyeinjar').
CRAVING_PRODUCERS = {
    "eyemonster": ("supplycupboard", ("eyeball", "eyeinjar")),
    "eyeball": ("supplycupboard", ("eyeball",)),
    "eyeinjar": ("supplycupboard", ("eyeball", "eyeinjar")),
    "skeleton": ("grave", ("bone", "ribcage")),
    "zombie": ("grave", ("bone", "ribcage", "rottenflesh", "severedhand")),
    "bone": ("grave", ("bone",)),
    "ribcage": ("grave", ("bone", "ribcage")),
    "rottenflesh": ("grave", ("rottenflesh",)),
    "severedhand": ("grave", ("rottenflesh", "severedhand")),
}

# Resource economy (wiki-grounded: Mana, Slime, Grave, Supply_Cupboard
# pages). Generalizes CRAVING_PRODUCERS (demand -> station) to the resource
# layer: every tap-cost resource maps to the stations spending it, the
# monster families generating it over time, the station raising its cap,
# and other sources. Powers the rejection teaching ("X needs Y — Y comes
# from Z, build/keep Z") and the resource-hint lines. made_by lists
# board-relevant families first, late-game legendaries last.
RESOURCE_ECONOMY = {
    "slime": {
        "used_by": ("Supply Cupboard", "Fridge"),
        "made_by": ("Zombies", "Spiders", "Ghouls", "Slime Golems",
                    "Serv-O", "Gorgon", "Cyclops", "The Colossus",
                    "Shield Bot"),
        "cap_by": "Slime Vats",
        "also": "feeding Potions",
    },
    "mana": {
        "used_by": ("Grave", "Lectern"),
        "made_by": ("Skeletons", "Eye Monsters", "Banshees", "Mana Golems",
                    "Serv-O", "Lich", "Reaper", "The Cursed", "Shield Bot"),
        "cap_by": "Mana Pools",
        "also": "feeding Potions and tapping the NecroMerger (collect)",
    },
    # Darkness is the late-game tap-cost resource (Altar unlocks at Tier 7
    # feats; Portal later). No live plumbing reads it yet — no HUD reader,
    # no validator gate, no hint trigger — so this entry is knowledge-only
    # until the save reaches it (then: darkness reader + threshold +
    # DARKNESS_SPAWN_PREFIXES gate mirroring the cupboard path). Banked now
    # so the teaching and rates are ready, with zero behavior change.
    "darkness": {
        "used_by": ("Altar", "Portal"),
        "made_by": ("Mummies", "Bats", "Imps", "Darkness Golems",
                    "Serv-O", "Harpy", "Archdemon", "The Infernal",
                    "Shield Bot"),
        "cap_by": "Darkness Stores",
        "also": "feeding Shades or Potions",
    },
}
# Validator rejection reason -> tap-cost resource it ran out of. Lets the
# correction and the folded learnings name the producers (via
# _resource_teaching) instead of just saying "wait until it regenerates".
RESOURCE_REJECTION = {
    "spawn_no_slime": "slime",
    "spawn_low_mana": "mana",
}


def _resource_teaching(resource: str, short: bool = False) -> str:
    """Producer teaching sentence for a tap-cost resource (single source).

    Full form goes to the folded learnings rule (permanent knowledge);
    short form to the in-step correction and board-hint lines. Both are
    built from RESOURCE_ECONOMY so the facts can't drift apart.
    """
    cfg = RESOURCE_ECONOMY.get(resource or "", {})
    if not cfg:
        return ""
    makers = ", ".join(cfg["made_by"])
    if short:
        return (f"{resource.capitalize()} comes from {makers} on the "
                f"board — keep them; {cfg['cap_by']} raise the cap.")
    return (f"{resource.capitalize()} is generated over time by {makers} "
            f"on the board — keep and merge them, never feed your "
            f"generators when {resource} is low. {cfg['cap_by']} raise "
            f"the cap. Also: {cfg['also']}.")
# (vision/panels.py) only handles 4 stations because the Station panel dialog
# never needs to disambiguate the rest. The strategy classifier needs the
# full list because the wiki has feats for every station. The canonical
# family is the concatenated no-space form (matches STATION_FAMILIES).
_STATION_NAME_ALIASES = {
    "grave": "grave",
    "manapool": "manapool", "mana pool": "manapool", "pool": "manapool",
    "supplycupboard": "supplycupboard", "supply cupboard": "supplycupboard",
    "cupboard": "supplycupboard",
    "foulchicken": "foulchicken", "foul chicken": "foulchicken",
    "chicken": "foulchicken",
    "slimevat": "slimevat", "slime vat": "slimevat", "vat": "slimevat",
    "altar": "altar",
    "darkstores": "darkstores", "dark stores": "darkstores", "stores": "darkstores",
    "lectern": "lectern",
    "fridge": "fridge",
    "portal": "portal",
    "crashedsaucer": "crashedsaucer", "crashed saucer": "crashedsaucer",
    "saucer": "crashedsaucer",
    "telepad": "telepad",
    "soulgrinder": "soulgrinder", "soul grinder": "soulgrinder",
    "grinder": "soulgrinder",
    "prism": "prism",
    "meteor": "meteor",
    "throne": "throne",
    "unexpectedparcel": "unexpectedparcel", "unexpected parcel": "unexpectedparcel",
    "parcel": "unexpectedparcel",
}
CREATURE_FAMILIES = frozenset({
    "skeleton", "zombie", "bone", "ribcage", "eyemonster", "spider",
    "mummy", "bat", "shade", "abomination", "managolem", "slimegolem",
    "darknessgolem", "werewolf", "snake", "imp", "unicorn", "demon",
    "forgottenminion", "cadet", "cherub", "peacefulsoul", "vengefulspirit",
    "guzzler", "runeling", "lich", "harpy", "thief", "cleric", "paladin",
    "rival", "cyclops", "gorgon", "archdemon", "shieldbot", "robochicken",
    "roboprotector", "cursed", "colossus", "infernal", "soulstalker",
    "souldrinker", "soulgrinder", "soulstalker", "managolem",
    "darknessgolem", "slimegolem",
})

# Strategy slot (Aug 24): the model may commit to ONE active feat (+ optional
# item family to build toward) via set_strategy; the priority layer then
# boosts that objective's kind instead of the closest-to-done default. The
# blast radius is bounded: a wrong strategy costs a few suboptimal steps and
# expires; legality/safety stay fully code-owned (validator + fallback).
MAX_STRATEGY = 1            # set_strategy calls the model may make per step
STRATEGY_REFRESH_STEPS = 15 # a committed strategy expires (re-offer) after this many steps
STRATEGY_KIND_BONUS = 1.0   # weight added to the strategy objective's move kind
MAX_BUYS = 1                # buy_station attempts allowed per step (spends runes)
MAX_QUEUE = 1               # collect_queue placements allowed per step (board mutation)
OBSERVE_HEARTBEAT_STEPS = 10  # live get_board_state call every N steps; others inject the deterministic render
PENDING_BUY_TTL = 5         # a confirm=false buy auto-expires after this many steps

HARD_RULES_TEXT = """HARD RULES (every violation is rejected — do not test them):
1. Merge ONLY two cells with the SAME id; NEVER merge [NEVER MERGE] / (max level) cells.
2. Spawn ONLY on [SPAWNABLE] cells (grave needs Mana + room; chest needs room; cupboard needs Slime + room and an eye objective) — never on runes, creatures, or full boards.
3. Feed ONLY [FEEDABLE] cells; never feed stations, champions, or the NecroMerger.
4. Attack ONLY as your creature (with dmg N) -> champion cell; a champion is NEVER the attacker.
5. get_board_state / identify_item / buy_station / set_strategy are TOOLS, not actions — never output them.

Example legal move: {"action": "feed", "cell": [1,0]} where (1,0) is tagged [FEEDABLE].
Example illegal move (always rejected, do not repeat this shape): {"action": "merge", "a": [1,0], "b": [3,2]} where both cells are tagged [NEVER MERGE] — the validator refuses merge_max_level and the game no-ops."""


def _geom_header(geom: GridGeometry) -> str:
    x0, y0 = geom.origin_x, geom.origin_y
    x1, y1 = x0 + geom.cols * geom.cell_px, y0 + geom.rows * geom.cell_px
    return (f"""You are the brain of a bot playing NecroMerger on a {geom.rows}x{geom.cols} board.
The screenshot is {SCREEN[0]}x{SCREEN[1]} pixels, full emulator frame.
Board geometry: the board occupies pixels x={x0}-{x1}, y={y0}-{y1};
{geom.rows} rows x {geom.cols} columns, each cell {geom.cell_px}px square. Top-left cell is (0,0), top-right (0,{geom.cols-1}),
bottom-left ({geom.rows-1},0), bottom-right ({geom.rows-1},{geom.cols-1}). The Devourer's mouth is above the board at ~({geom.mouth_x},{geom.mouth_y}).""")


def _action_schema(geom: GridGeometry) -> str:
    return f"""Then look at the screenshot, decide the single best move, and reply with ONLY a
JSON object using EXACTLY one of these actions:
{{"action":"merge","a":[r,c],"b":[r,c]}}
{{"action":"spawn","cell":[r,c]}}
{{"action":"feed","cell":[r,c]}}
{{"action":"attack","cell":[r,c],"target":[r,c]}}
{{"action":"collect"}}
{{"action":"idle"}}
r is the row (0-{geom.rows-1}), c is the column (0-{geom.cols-1}). merge: pick two
cells holding the SAME item (identical sprites, same id, same level). spawn: on
a spawn station (grave — costs Mana; or chest — mana-free, one Rune per
tap; the chest on the board is a `spawn` cell, not a `collect` cell).
attack: drag one of YOUR creatures (a non-champion, non-station cell with a
known `(dmg N)` value) onto a Champion cell (a cell labeled `(champion)` in
get_board_state — e.g. The Peasant) to deal that creature's `(dmg N)` HP to
the Champion. The `cell` (attacker) is ALWAYS one of YOUR creatures — a
Champion can NEVER be the attacker (the validator refuses `attack_is_champion`),
only the `target` is the Champion. Per observation, Champions do not
auto-attack; only your attack drags deal damage. If no creature on the board
has a known `(dmg N)` value, attack is not available; merge/feed instead.
collect: tap the NecroMerger at (0,{geom.cols-1}) to gain Mana. It does NOT
remove the NecroMerger, does NOT clear or free a board cell, and you cannot
"collect" any other item — chests and stations (Grave, Mana Pool, etc.)
and creatures are never "collected" (use `spawn` for a chest, `feed`
for a creature, `buy_station` for a station).
feed: drag a creature into the Devourer's mouth. Never feed any station
(the station list above) and never the NecroMerger.
Do not explain, just output the JSON object."""


def _answer_prompt(geom: GridGeometry) -> str:
    """Minimal system prompt for the answer round (_ask_once).

    The full _drive_prompt (~25k chars) is needed for the tool rounds, but
    the final JSON decision only needs geometry + hard rules + schema.
    Per-cell legality tags, the whitelist, the Best hints, and the decision
    checklist all arrive via the Tool-results text, so nothing is lost —
    and the 4B model decides with ~2k chars of instruction instead of ~25k.
    """
    return _geom_header(geom) + "\nRules:\n" + HARD_RULES_TEXT + "\n" + _action_schema(geom)


def _drive_prompt(geom: GridGeometry) -> str:
    """System prompt for vision-drive; board geometry rendered from detection."""
    return _geom_header(geom) + """
Rules:
- Two identical items merge into one of the next level (e.g. zombie_lvl1 + zombie_lvl1 -> zombie_lvl2).
- The LEVEL is part of an item's identity: creatures of the SAME type but DIFFERENT
  levels are DIFFERENT items and NEVER merge (skeleton_lvl1 + skeleton_lvl2 is INVALID;
  only two cells with the SAME id, like skeleton_lvl1 + skeleton_lvl1, merge). The
  `get_board_state` labels (e.g. `skeleton_lvl1`) carry the exact level — trust the
  label, not what the sprites look like.
- MAX-LEVEL items: every item type has a TOP level. A max-level item's
  info popup says only "Feed to the Devourer." and has NO "merge" line —
  it can never merge again. When `get_board_state` marks a cell
  `(max level)`, do NOT try to merge two of them — the game refuses.
   Feeding one yields good food, but feeding is NOT its only use: max-level
   creatures GENERATE RESOURCES (Mana, Slime, Darkness) while on the board
   (higher levels generate more). Decide per situation: feed for Food/space
   when needed; keep high-level generators when resource income matters.
- SPAWNING — GRAVE: a Grave spawns bone/ribcage when tapped — it costs Mana
  (the amount grows with the station's level) and silently does nothing when
  the bar is empty, so do NOT spawn the grave when `get_board_state` shows
  `Mana: (bar not visible)` or a low percent — collect instead. Spawning the
  grave ALSO needs an EMPTY board cell: when `get_board_state` shows 0 empty
  cells, spawns silently no-op — free a cell first (merge two of the same to
  combine, or feed a non-critical creature to clear a slot). A FULL Mana bar
  is exactly when you SHOULD spawn the grave: spawning spends the bar down.
  (Other stations spawn from different Resources: a Slime Vat spawns with
  Slime, a Dark Stores with Darkness. The Mana Pool, Slime Vat, and Dark
  Stores are NOT spawn stations — they are Resource-CAPACITY stations that
  boost the max of their Resource; bought with Runes like any other station.)
- SPAWNING — ICE CHEST (and other chests): an Ice Chest (the `icebox` you see
  on the board) spawns one random rune per tap — see your item glossary for
  the exact `spawn` chain and current rates. It costs NO Resource (Mana,
  Slime, or Darkness — the bar at the top of the lair), but each spawned
  rune occupies a board cell, and the chest is FINITE USES (e.g. the
  standard icebox has 5 uses). The `Open a chest.` feat counts each tap as
  one open. Merge the rune stacks up to max level (icerune lvl1 -> lvl2 -> lvl3
  max; poison has its own chain), then FEED the max-level stack — that grants
  the corresponding Ice / Poison / Blood / Moon / Death Rune currency (NOT
  food). That's the primary way to gather the 5 station-buying currencies
  (the Shop also sells Runes for Gold/Gems, but the Shop isn't available
  on the current save). NEVER feed a Rune (or Coin) stack below max level —
  the validator refuses `feed` on sub-max income stacks when any other
  feedable exists; merge the stack up first so the per-feed payout is
  the largest (icerune lvl3 grants 12 vs lvl1's 2; coin lvl4 grants 30
  vs lvl1's 2).
   WHEN to tap a chest: only when the rune feeds a stack that reaches max
   level and gets fed within a few steps — a new stack, a mergeable stack,
   or one or two merges from max. Do NOT tap just because uses remain:
   runes accumulate faster than they merge, and each tap costs a cell.
   Never tap when the board is full of `(max level)` items.
- RESOURCE BAR: `get_board_state` shows the current Resource fill as
  `Mana: ~N% full` (the only Resource bar visible on the lair for now —
  Slime and Darkness bars appear later when Slime Vat / Dark Stores are
  built, and follow the same read convention). Do NOT `collect` when the
  bar is `~100% full` — the tap is wasted. (Spawn/merge/feed instead —
  a full bar is a signal to spend the surplus.)
- Feed a creature to the Devourer to gain food. When `get_board_state` labels a
  cell as `(feed N)`, that N is the food value that creature yields when fed
  (read from the in-game popup; see your item glossary for recorded values).
  When choosing what to feed, prefer the lowest-value creature you don't
  need (keep high-value/better items for merging or a future craving).
- SATIETY: the Devourer's satiety bar shows how much more food is needed before
  it levels up (`get_board_state` shows `Satiety: X/M (R remaining)` when it can
  be read). Feeding an item whose `(feed N)` is LARGER than R wastes food — the
  excess is lost and the game warns. Tolerance is up to ~10% of M (the safety
  margin); beyond that it counts as waste. So feed only items whose feed
  value fits R. Among fitting items, pick the SMALLEST one — feed value
  scales non-linearly (a ribcage is worth more than the two bones that
  create it, and a Skeleton L3 is worth more than the two L2s that merge
  into it), so the smallest fitting item minimizes value destruction
  while still making progress toward the level-up. Do NOT sacrifice a
  high-level / resource-generating monster just to fill a small gap. If
  nothing fits even with the tolerance, don't feed at all — spawn instead
  (or `attack` if a champion is on the board).
- CRAVING OVERFLOW: each feed of a craving's required item counts toward
  its quota (e.g. a `Rib Cage (lvl 1) 0/2` craving needs 2 ribcage feeds);
  the Food bonus is granted ONCE when the craving is FULFILLED (the final
  feed pays out the bonus on top of the item's normal food). Feeding the
  craved item into a nearly-full bar can therefore waste that big bonus:
  if the bar's R is smaller than the item's `(feed N)` (or so close that
  the final-completion bonus would push the total over the tolerance),
  do NOT feed it now — the craving persists, so PREFER SPAWNING from a
  spawn station to rebuild the craved item (the grave is the most common
  source; some cravings come from chests, Lectern, or other stations) and
  feed it after the bar resets on a level-up. A `Best spawn:` hint is shown
  in that case. Never waste the completion bonus on an overflowing bar.
- FEED-TO-PROGRESS: FEEDING the Devourer is how you advance — its level rises
  with food, and every level unlocks something (Tier-1 uncrates the Devourer;
  Lvl 3 unlocks cravings; Lvl 4+ drop ice chests; Lvl 5+ starts the Peasant;
  Lvl 7+ grows the lair; the per-tier feat `Reach Devourer level N` is your
  barometer of how close the next tier is). When NO merge is possible (no
  `Best merge:` line) and there is no craving to feed, feed a NON-CRITICAL
  creature to keep progressing instead of only spawning: prefer the CHEAPEST
  fitting creature by `(feed N)` (per the glossary, bone=1, ribcage=5,
  skeleton_lvl1=10 — these are the skeleton-chain bases, not needed for a
  pending merge). `(max level)` items and HIGH-VALUE creatures are RESOURCE
  GENERATORS (they passively make their Resource — Mana/Slime/Darkness —
  higher levels make more) — feeding them is a LAST RESORT: only when the
  board is congested or nothing cheaper fits. Spawning cheap food from the
  grave is usually better than sacrificing a generator. `get_board_state`
  may show a `Best feed:` hint — the code's recommended feed target; follow
  it when you agree. Keep at least one creature on the board for the merge
  pipeline (don't strip it bare), and never feed stations or the
  NecroMerger.
  NEVER feed a merge-chain PRECURSOR of an ACTIVE craving: a precursor
  is a same-family item whose level is BELOW the craving's level
  (e.g. for a `Skeleton (lvl 3)` craving, `skeleton_lvl1`, `skeleton_lvl2`,
  `ribcage`, and `bone` are all precursors — they merge up into the
  craved monster, so feeding them destroys the path to fulfilling the
  craving). Cells at the craving's exact level still feed normally
  (they count toward the quota), and cells ABOVE the craving's level
  still feed (they can't merge down). For a `Skeleton (lvl 1) 0/1`
  craving, the precursors are `bone` and `ribcage` (which merge up
  into the craved lvl 1) — protect those and feed the lvl 1 directly.
  Merge precursors toward the craving instead of feeding them to the
  Devourer.
- SPAWN-FIRST ON SPARSE BOARDS: when the board has an empty cell AND a
  spawn station (e.g. grave or chest) can spawn, generic feed-to-progress
  is SUSPENDED — spawn a fresh cheap-chain item for free instead of
  sacrificing one of only a few creatures that could merge. The `Best
  spawn:` hint fires in this case. Real reasons to feed still apply:
  congestion (few empty cells), an active craving, a `(max level)`
  target, or a near-done feed feat.
- MERGE-MATERIAL: `bone`, `ribcage`, and any mid-chain levelled creature
  whose next level is known are MERGE MATERIAL — feeding one of these
  removes a member of a future merge pair, so the `Best feed:` hint
  avoids them when a non-material or craved target exists. (See
  FEED-TO-PROGRESS above for the full feed-selection policy.)
- The Devourer may have an active CRAVING (bubble above the board; Food bonus
  granted ONCE when the quota is fulfilled). If a creature matching the
  craving at the EXACT level is on the board, FEED it — unless a
  nearly-done feat of another kind outranks it (see MOVE PRIORITY), or the
  bar is so full the feed would overflow past tolerance (see CRAVING
  OVERFLOW — spawn to rebuild it instead). NEVER feed a same-family
  LOWER-level precursor (it merges up into the craved monster); cells AT
  or ABOVE the craving's level feed normally. `get_board_state` shows
  `Cravings: <item> (lvl <N>) <done>/<required>, reward +<food>`; when no
  craving is shown, call `get_cravings` to read it. The craved item is
  never a station and never the NecroMerger.
- NEVER feed any station to the Devourer. The wiki's stations are:
  Grave, Mana Pool, Supply Cupboard, Foul Chicken, Slime Vat, Altar,
  Dark Stores, Lectern, Fridge, Portal, Crashed Saucer, Telepad, Soul
  Grinder, Prism, Unexpected Parcel, Meteor, Throne (plus post-prestige
  additions) — this is the canonical list, referenced as "the station
  list" elsewhere. The game itself permits feeding most stations behind a
  confirmation dialog, but we never risk it. The NecroMerger is never fed,
  spawned, or collected.
  MERGING stations is normal and valid: two identical manapools, Graves,
  etc. merge into the next level (e.g. manapool_lvl1 + manapool_lvl1 ->
  manapool_lvl2) exactly like any other item pair. So the Mana Pool is
  NOT a "don't touch" station: it merges like any item, and the grave's
  popup even says "Merge to level up". Only the fixed NecroMerger is
  never merged.
- A Champion (e.g. The Peasant, The Knight, The Cleric, The Paladin,
  The Rival, The Protector, The Mech, The King) may invade the board and
  occupy a cell. It is an ENEMY, not a rune or a regular item: it can never
  be merged, spawned, or fed as food. Champions are also NOT a target
  themselves for `feed` or `merge` (the validators refuse). To damage a
  Champion, drag a creature ONTO the Champion cell — that is the `attack`
  action, a special move distinct from merge/spawn/feed. The Peasant is a
  small human sprite — do NOT mistake it for a poison rune or any other
  item. When `get_board_state` labels an item `(dmg N)`, that N is the
  damage it deals if dropped on a Champion — so decide between feeding the
  creature (its `(feed N)` food, capped by remaining satiety) and using it
  to attack the Champion (its `(dmg N)`). Per observation, Champions do
  not auto-attack you; only your attack drags deal damage. If no creature
  on the board has a known `(dmg N)` value, attack is not available;
  merge/feed instead.
- NEVER name an unidentified cell from the screenshot. If `get_board_state`
  lists a cell as unidentified, call `identify_item` on it to learn what it
  is — never guess a template id (like poisonrune_lvl1) from the pixels.
  (Skip the call if a higher-priority move is available, e.g. a champion
  is on the board or a feed-it-or-lose-it craving is nearly complete —
  unidentified cells can wait one step.)
- MOVE PRIORITY (driven by ACTIVE FEATS): the `Feats (tier N):` line of
  `get_board_state` lists every active task with its parsed action kind
  (`[merge]` / `[feed]` / `[spawn]` / `[collect]`, or just the feat name
  when the kind couldn't be parsed) and progress, flagging `NEAR DONE`
  at >= 80% completion. Done feats are listed under `| done:` and no
  longer steer. Spend each move advancing the objective closest to
  completion: a task needing merges -> merge; feeds -> feed; spawns ->
  spawn; collects -> collect. The current CRAVING is a feed objective
  too (feeding the craved creature counts toward the quota; the Food
  bonus is granted once the craving is fulfilled), and it usually
  outweighs a merge unless a competing feat is closer to done. Combat
  feats (e.g. `Beat The Peasant twice.`) and station-build feats (e.g.
  `Build a Mana Pool.`) follow the same target kind: combat -> use
  `attack` on a Champion; build -> use `buy_station`. The `Best merge:`
  / `Best feed:` / `Best spawn:` / `Best attack:` / `Best income:`
  hints on the board state follow the same weighting — they're HINTS,
  not commands: follow them unless you see a clearly better move. When
  NO feat or craving steers the move, fall back to: merge two identical
  items > feed (feed-to-progress; but when the board is sparse and a
  grave can spawn, spawn instead — see SPAWN-FIRST) > spawn > collect
  > idle. Skip `spawn` when the mana bar is low (it no-ops); skip
  `collect` when the mana bar is full. `get_board_state` may also show
  a `Champion:` line (next champion to spawn + progress); when a
  champion is on the board, the attack branch becomes live.
- The bottom of the lair screen has a dock of 5 buttons: Feats, Station,
  Queue (placement action — tapping DROPS the queued reward onto the
  board, so leave it alone unless you want the item placed), Spellbook
  and Shop (both locked). `get_board_state` shows the live dock state in
  a `Bottom bar:` line — trust that, not this static description.
   A missing `Bottom bar:` line means a panel or non-lair screen is open,
   so the dock is hidden.
- POTIONS (manapot / manapotion): max-level Potions (e.g. `manapotion_lvl3`,
  `manapot_lvl3`) are FEEDABLE consumables, NOT stations — they grant the
  matching Resource (Mana) when fed, plus a small food value. The
  in-game popup says "Feed to the Devourer." with no merge line (they
  are max-level, single-use). DO NOT confuse them with `manapool` (a
  Resource-CAPACITY station that boosts Mana cap). The wiki's manapotion
  lvl 3 grants 100% of Mana cap when fed — if the Mana bar is already
  at 100% the feed is wasted (the Mana can't be stored). The validator
  refuses `feed` on a manapot/manapotion when the Mana bar is at 100%
  (`feed_mana_overflow`); feed potions only when you have Mana headroom,
  or build a Mana Pool to raise the cap first.

Each step you receive an automatic `get_board_state` result (the bot calls it
on your behalf every step) with the live board: occupied cells with their
template-bank ids, levels, and confidence scores; unidentified occupied cells
(candidates for `identify_item`); the bottom bar state; the mana bar fill;
the current craving; the active strategy; and the feat list with action
kinds + progress. The bot ALSO runs a PROACTIVE batch `identify_item` round
up-front when there are UNID cells (capped at `MAX_DISCOVERY` per step) so
you can plan moves with full knowledge — check the tool-result message just
above the board state for what was identified. You can still call
`identify_item` yourself in the optional-tools round for any cells the
proactive round missed (capped at `MAX_DISCOVERY` cells per batch). The board state also shows one or more of these HINTS —
the single highest-priority move under your current feats + craving:
  - `Best merge: <id> (<r>,<c>)+(<r>,<c>) -> <result>` — top merge pair
  - `Best feed: <id> (<r>,<c>) (feed N)` — top feed target
  - `Best spawn: <station_id> (r,c) -> <item> (<X> empty)` — top spawn target. The hint points at the HIGHEST-LEVEL spawn station on the board. Per the wiki, grave_lvl1 spawns bone (100%), grave_lvl2 spawns bone (60%)/ribcage (40%), grave_lvl3 spawns bone (40%)/ribcage (30%)/zombie (30%). Chests outrank graves since chests are finite-use. The cell id includes the level (e.g. `grave_lvl3`) so you can see it at a glance, and the per-level rates are read from the glossary's popup block when available (per-level `lvl3 spawn: ...` line banked by the auto popup-reader).
  - `Best attack: <creature> (r,c) (dmg N) -> champion (r,c)` — when a champion is on the board
  - `Best income: <stack> (r,c) — feed it to the Devourer for <ICE/POISON/BLOOD/MOON/DEATH> RUNES / GOLD / GEMS (the income target of the active economy)` — fires when a max-level Rune, Coin, or Gem stack is on the board. Rune stacks fire under a build strategy (Runes are station-buying currency); Coin and Gem stacks fire unconditionally (more Gold/Gems is always useful for the Shop / skins / Wobulan trades). Sub-max income stacks are not valid `feed` targets — the validator refuses them, and the merge ranking merges them up first.
These are HINTS, not commands: follow them unless you see a clearly better
move, and prefer them over feed/spawn/collect/attack when one is listed.
The board state also shows `Feats (tier N): ...` listing every task with
its `[kind]` and progress (`NEAR DONE` >= 80%), plus `Satiety: X/M
(R remaining)` when the satiety fraction can be read — feed items only
while R > 0 and only items whose `(feed N)` fits R.
You may also call `identify_item` on an unidentified cell to learn what it
is — it opens the item's info popup, reads its description + merge-chain
text, and banks the sprite in the template bank (recorded in your item
glossary). New items that appear from merges/spawns are popup-read
automatically, so `get_board_state` already flags max-level cells and
locks in the right id. Only call `identify_item` when a cell is
UNIDENTIFIED (no template label) — never for cells that already show a
label you don't recognize (the label is the truth; the sprite is the
lie).
And `get_cravings` reads the current craving — `get_board_state` already
shows it from a recent read, so only call when none is shown or progress
is stale. `tap_button` opens a dock panel (FEATS or STATION), read-only;
the feats are ALSO in `get_board_state`, so use it only for the full
panel or fresh progress.
And `set_strategy` is YOUR strategic voice: commit to ONE active feat OR a meta-goal,
and the priority layer boosts YOUR choice for ~15 steps instead of the
closest-to-done default. When feats compete or you see a better long-
term play than the default ordering, CALL IT (consider it at least once
after each Feats refresh — the Feats line on the board state is the
signal that a refresh happened). Your active strategy shows in
`get_board_state`.

STRATEGY TYPES:
  - EXPLICIT FEAT (takes priority over meta-goals):
    - STATION feat (`Build a X.` or `Own a Lvl N+ X.` where X is a station
      like Grave, Mana Pool, Supply Cupboard, etc.): the feat needs the X
      STATION bought from the Station panel. DO NOT set `target_item` —
      the station isn't on the board to merge toward; `buy_station` is the
      action. The X station costs Runes (Mana Pool = ice + poison, Slime
      Vat = poison + blood, Dark Stores = blood + moon, etc. — check the
      `cost_cache` or call `buy_station(family, confirm=false)` to see the
      current prices; feed max-level Rune stacks for the income boost).
    - CREATURE feat (`Own a Lvl N+ Y.` where Y is a creature like Skeleton,
      Zombie, Eye Monster, Spider, Mummy, Bat, Shade, Abomination, etc.,
      or a champion like Peasant, Knight, Gorgon, Cyclops, Harpy): the
      feat needs the creature merged/collected on the board. Set
      `target_item` to the creature's family + level (e.g. `skeleton_lvl3`
      for an `Own a Lvl 3+ Skeleton.` feat). A `target_item` whose family
      doesn't match the feat's noun is dropped silently — the bot only
      trusts a target that's the right shape.
    - OTHER feat (combat, action count, collect-Nx, level-N, etc., e.g.
      `Beat The Peasant twice.`, `Merge things 50 times.`, `Reach Devourer
      level 5.`): no station or creature direction. `target_item` is
      ignored. The priority layer does not boost a station/creature kind
      for these — the action is whatever the feat text describes.

  - META-GOAL (used when no explicit feat applies, or to override):
    Set `meta_goal` to ONE of:
      - slime_generation: slime is low → prioritize slimevat builds, slime-producing minions
      - mana_generation: mana is low → prioritize manapool builds, mana-producing minions
      - rune_economy: rune balances low → prioritize chest spawns, rune merges, rune feeds
      - champion_combat: champion present → prioritize high-damage minions, attack
      - board_management: board congested → merge aggressively, feed non-critical
      - feeding_optimization: satiety near cap → feed efficiently, avoid overflow
      - darkness_generation: darkness low → prioritize darkstores builds
    When using a meta-goal, set `target_family` to the family to build toward
    (e.g. "manapool" for mana_generation, "slimevat" for slime_generation,
    "grave" for rune_economy, "darkstores" for darkness_generation).
    Explicit feats ALWAYS take priority over meta-goals.

The `target_family` field replaces `target_item` for station/meta-goal strategies.
For creature feats, you may still use `target_item` (e.g. `skeleton_lvl3`).
And `buy_station` spends RUNES (ice, poison, blood, moon, death) to place
a NEW station. Two-phase (ENFORCED): confirm=false reads the dialog and
stashes a pending buy; the next call MUST be confirm=true (same family)
or it auto-expires. Only confirm purchases serving your strategy
(`strategy_mismatch` refuses other families). GATHER Runes by feeding
max-level Rune stacks (see ICE CHEST above; the `Best income:` hint fires
under a build strategy). Full semantics are in the tool description —
this is the summary.
And `collect_queue` places the dock Queue button's reward onto the board
(bare skull = empty); REFUSED when congested. Board chests are different:
they are spawn stations — tap with `spawn` (see ICE CHEST). Full
semantics are in the tool description.

""" + HARD_RULES_TEXT + "\n\n" + _action_schema(geom)

BOARD_TOOL = {
    "type": "function",
    "function": {
        "name": "get_board_state",
        "description": "Return the live board state text.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

IDENTIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "identify_item",
        "description": (
            "Investigate one or more unidentified occupied cells. For each "
            "cell, the bot taps it, reads the info popup, and banks the "
            "sprite as a new template item. The popup's description + "
            "merge-chain text is recorded in the item glossary. The bot uses "
            "category-specific popup questions (Champion / Currency / Station "
            "prompts) so the LLM extract is more accurate than a generic "
            "prompt. Pass a single cell as `cell:[r,c]` OR a list as "
            "`cells:[[r,c], [r,c], ...]` for batch identification (one tool "
            "call covers multiple unidentified cells). Returns a list of "
            "{item_id, cell, description, merge_info} for each cell identified. "
            "Only available in live mode."),
        "parameters": {
            "type": "object",
            "properties": {
                "cell": {"type": "array", "items": {"type": "integer"},
                         "description": "single cell [r,c] — back-compat with older tool templates"},
                "cells": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}},
                          "description": "list of cells [[r,c], [r,c], ...] for batch identification"},
            },
        },
    },
}

WIKI_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_wiki",
        "description": (
            "Fetch a NecroMerger wiki page (necromerger.wiki.gg) for a topic "
            "and return its text. Use this to fact-check item behavior, merge "
            "chains, or mechanics you are unsure about. Query with the ITEM "
            "NAME (e.g. 'Skeleton' or 'Zombie') — phrasal queries like 'zombie "
            "merge chain' are auto-resolved to the item's own page, and the "
            "result includes a `merge_info` field pulling the merge-relevant "
            "lines (components, what it summons, what it's used to make) "
            "straight from that article. So always query the ITEM — do NOT "
            "search 'X merge chain'; the item page already carries it. "
            "Returns the FULL matching article. Read-only. Only available in "
            "live mode."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "topic to look up, e.g. 'Skeleton'"},
            },
            "required": ["query"],
        },
    },
}

CRAVINGS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_cravings",
        "description": (
            "Open the Devourer's Cravings menu (a bubble above the board is "
            "tappable) and return the current craving: the craved item, its "
            "level, progress (count_done/count_required, e.g. 0/7), and the "
            "food reward (e.g. +150). The current craving is usually already "
            "shown in get_board_state from a recent read — only call this "
            "when no craving is shown there or you need an updated progress "
            "count. Only available in live mode."),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

CHAMPION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_champions",
        "description": (
            "Open the Champion spawn screen (the champion tracker at the "
            "top-right of the lair is tappable) and return each champion's "
            "spawn progress: name, active/attacking, progress (e.g. 1/3 or "
            "100%), ready. The current champion's status is usually already "
            "shown in get_board_state from a recent read — only call this "
            "when no Champion line is shown there or you need an updated "
            "progress count. Only available in live mode."),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

PANEL_TOOL = {
    "type": "function",
    "function": {
        "name": "tap_button",
        "description": (
            "Open a bottom-bar dock panel (FEATS: the current feat tier and "
            "its missions; STATION: stations available to buy and their "
            "costs), read its contents, and close it. Read-only: nothing is "
            "ever bought or changed — it just reports what the panel shows. "
            "Never call this for the Queue (tapping the Queue docks its "
            "reward item onto the board — a placement action, not a readable "
            "panel), nor for locked Spellbook/Shop. Only available in live "
            "mode."),
        "parameters": {
            "type": "object",
            "properties": {
                "button": {"type": "string", "enum": ["feats", "station"],
                           "description": "which dock panel to open"},
            },
            "required": ["button"],
        },
    },
}

STRATEGY_TOOL = {
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
            "shown in the Feats line are valid for feat strategies."),
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
    },
}

BUY_TOOL = {
    "type": "function",
    "function": {
        "name": "buy_station",
        "description": (
            "Buy a station (family: 'grave', 'manapool', 'supplycupboard', "
            "'foulchicken', 'slimevat', 'altar', 'darkstores', 'lectern', "
            "'fridge', 'portal', 'crashedsaucer', 'telepad', 'soulgrinder', "
            "'prism', 'meteor', 'throne', 'unexpectedparcel') from the "
            "Station panel using runes. The 5 in-game currencies are ice, "
            "poison, blood, moon, death (NOT green/red/purple). Two-phase "
            "confirmation: call with confirm=false first — it opens the "
            "panel, verifies it is actually the station panel (refuses if a "
            "tutorial/feats panel is open), checks affordability against "
            "your rune balance, taps the card and reads the confirmation "
            "dialog back to you WITHOUT buying; then call again with "
            "confirm=true to complete the purchase. Only confirm when the "
            "dialog's station and cost match what you want (and the panel "
            "was verified — the result includes 'panel_verified'). If your "
            "active strategy from set_strategy names a specific station, "
            "this tool refuses to buy a different family with "
            "'strategy_mismatch' (call the strategy's family instead). "
            "Bounded to one attempt per step. Only available in live mode."),
        "parameters": {
            "type": "object",
            "properties": {
                "family": {"type": "string",
                           "description": "station family to buy, e.g. 'grave' or 'manapool'"},
                "confirm": {"type": "boolean",
                            "description": "false = read the dialog only; true = complete the purchase"},
            },
            "required": ["family"],
        },
    },
}

# Short compel-time spec for _compel_buy (same parameters, minimal
# description). Probed live: this thinking model burns its whole budget
# debating the full two-phase/affordability/mismatch semantics and emits
# nothing, but emits a perfect call for a single-action positive framing.
# The full BUY_TOOL stays on the voluntary optional-tools round, where the
# semantics guide deliberate calls.
BUY_TOOL_SHORT = {
    "type": "function",
    "function": {
        "name": "buy_station",
        "description": "Read a station card from the Station panel and report its cost.",
        "parameters": {
            "type": "object",
            "properties": {
                "family": {"type": "string",
                           "description": "station family, e.g. 'grave'"},
                "confirm": {"type": "boolean",
                            "description": "false reads the dialog, true completes"},
            },
            "required": ["family"],
        },
    },
}

QUEUE_TOOL = {
    "type": "function",
    "function": {
        "name": "collect_queue",
        "description": (
            "Place the reward queued in the bottom-bar Queue button onto the "
            "board (the button shows the queued item; a bare skull means "
            "empty). Placement is REFUSED when the board is congested — each "
            "placed reward occupies a cell. Chests already on the board are "
            "spawn stations: use a `spawn` action on the chest cell to release "
            "one random rune per tap (mana-free, one tap per use). The runes "
            "merge up like any item (icerune lvl1 -> lvl2 -> lvl3 max; feed "
            "only the max-level stack). Feeding an icerune stack to the "
            "Devourer grants the Ice Rune currency (no food) — your main "
            "rune income: spawn chests, merge the stacks up, feed the max-"
            "level stack, repeat. Only available in live mode."),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

FEAT_COLLECT_TOOL = {
    "type": "function",
    "function": {
        "name": "collect_feat_rewards",
        "description": (
            "Open the Feats panel and collect any completed feat rewards "
            "(blue Collect buttons with a red badge, plus the green tier "
            "reward when its badge is up). Each tap is verified and bounded. "
            "Use when get_board_state shows a feat as done but its reward "
            "is still unclaimed, or when you have just completed a mission. "
            "The periodic background collect runs every 5 steps when a done "
            "feat is cached, but this tool lets you force an immediate "
            "collect without waiting. Only available in live mode."),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

SUMMARY_SYSTEM_PROMPT = """You are the brain of a bot playing NecroMerger. A play window just ended.
You will be given a transcript of the window (board states and the moves that were made and their outcomes), plus your current stored learnings.
Write lasting learnings that will make you play better in future sessions, and maintain your stored knowledge. Reply with ONLY a JSON object:
{"learnings": [{"text": "lesson text", "type": "fact|pattern|visual|anti-pattern", "confidence": 0.0-1.0}, ...],
 "items": ["one entry per item type visible in the attached final-board screenshot, with visual markers you can see in it", "..."],
 "remove": ["exact text of STORED learnings you now believe are wrong, contradicted, or were never validated; copy them verbatim from the stored list", "..."],
 "confirmed": ["exact text of STORED learnings this window's transcript shows were validated or applied successfully; copy verbatim", "..."],
 "remove_items": ["stored glossary entries (item ids) you now know are wrong", "..."]}
- learnings: max 10 short plain-text bullets (no markdown syntax), grounded ONLY in this window's transcript. Each learning MUST have:
    * "text": the lesson
    * "type": one of "fact" (verifiable game mechanic), "pattern" (board-state heuristic), "visual" (sprite identification tip), "anti-pattern" (mistake to avoid)
    * "confidence": 0.0-1.0, your self-assessed certainty. MINIMUMS: fact=0.9, pattern=0.7, visual=0.8, anti-pattern=0.7
  Learnings describe GAME facts only: item identity/levels, which sprites look alike, merge/feed/spawn behavior, board-congestion patterns, mistakes to avoid.
- items: ONLY item types you can see in the attached screenshot, max 10. Popup-read merge recipes (e.g. "merge: two Skeleton -> Skeleton of the next level") are grounded game-UI text and belong here too, even without a visible sprite. NEVER name an item from the screenshot that the final board's template-label list reports as "Unidentified" — a sprite you cannot positively identify must not be written as an item (it may be a Champion like The Peasant, not a rune).
- CONSOLIDATED CHAINS: when the stored glossary or this window shows TWO OR MORE popup-read merge links that chain a family together (e.g. "edge: bone -> ribcage" and "edge: ribcage -> skeleton"), emit ONE line per family of the exact form `merge chain <family>: <id> -> <id> -> ...` (e.g. `merge chain skeleton: bone -> ribcage -> skeleton_lvl1 -> skeleton_lvl2`), using only real template-bank ids. These become a dedicated `(chain)` glossary section. Do not emit chains with fewer than 2 links, and do not invent hops that no popup read or wiki text grounded. A `(chain)` line replaces any existing chain for that family.
- SPAWN EDGES: stations spawn items when tapped (like Grave and Ice Chest). When this window shows a station's spawn grounded in UI — a chest popup "Tap to open." plus observed rune spawn (Uses 5 badge, or 60/20/20 icerune/poisonrune spawns) or a Grave popup "Merge to level up" plus bone/ribcage spawns (or bone/ribcage/zombie for grave_lvl2+ per the wiki) — emit ONE line per station LEVEL of the exact form `lvl<N> spawn chain <family>: <station_id> -> <target1> (XX%), <target2> (YY%)` (e.g. `lvl1 spawn chain grave: grave_lvl1 -> bone (100%)`, `lvl2 spawn chain grave: grave_lvl2 -> bone (60%), ribcage (40%)`, `lvl3 spawn chain grave: grave_lvl3 -> bone (40%), ribcage (30%), zombie (30%)`, and `spawn chain icebox: icebox_unopened -> icerune_lvl1 (60%), icerune_lvl2 (20%), poisonrune_lvl1 (20%)`). Add `uses: 5` for chests (finite) or `uses: infinite` for graves. These become a dedicated `(spawn)` glossary section, like `(chain)`, and replace any existing spawn for that family. Do not invent spawn targets not grounded in observed spawns or popup text.
- remove/confirmed/remove_items: copy stored text verbatim; leave empty when none apply.
- If the window had validation failures / heuristic fallbacks, prefer removing the stored learning(s) that likely caused them.
- NEVER write learnings about the bot's own behavior or internals: vision_drive failures, heuristic fallback, validation rejections, tool calls, spawn/discovery/identify logic, or "board state" validity. Those are NOT game mechanics and never belong in learnings.
- NEVER write learnings about game mechanics absent from the transcript, and never write learnings that contradict the rules: the NecroMerger can be tapped for mana but is never collected, removed, or fed; stations (grave, manapool, chest) are never fed or collected (manapot/manapotion are NOT stations — they are feedable Potions, see the POTIONS section above); a move the heuristic fallback played is NOT evidence that the mechanic is valid.
- confirmed: only list stored learnings this window's transcript demonstrated as correct GAME behavior. "The bot did X" is not validation; never confirm a learning merely because the agent acted.
No markdown fences, no prose outside the JSON."""


def _synthesize_board_call(content: str) -> dict | None:
    """Turn a text-JSON `{"action":"get_board_state"}` reply into a real tool call.

    Some models served through llama.cpp (e.g. Gemma GGUFs) respond to
    `tool_choice="required"` with the tool as plain JSON text rather than a
    native `tool_calls` message. `_observe` needs a native call dict to inject
    the board state into the conversation, so this parses the text back into
    the same shape llama-server produces. Returns None when the reply isn't
    such a get_board_state call (falls through to the nudge retry / fallback).
    """
    if not content or not content.strip():
        return None
    try:
        data = _extract_json(content)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("action") != "get_board_state":
        return None
    return {"type": "function",
            "id": f"synth_{int(time.time() * 1000)}",
            "function": {"name": "get_board_state", "arguments": "{}"}}


def _synthesize_buy_call(content: str, family: str) -> dict | None:
    """Turn a text-JSON `buy_station` reply into a real tool call. Aug 28:
    the compelled buy follow-through uses tool_choice="required" with the
    BUY_TOOL spec, and small models (Gemma GGUFs) sometimes reply with the
    call as plain JSON text rather than a native `tool_calls` message.
    Synthesize the call so the confirm still lands. Family defaults to the
    pending buy's family so an empty JSON still resolves correctly.
    """
    if not content or not content.strip():
        return None
    try:
        data = _extract_json(content)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("action") != "buy_station":
        return None
    args = {"family": canonical_family(data.get("family")) or family,
            "confirm": bool(data.get("confirm", True))}
    return {"type": "function",
            "id": f"synth_buy_{int(time.time() * 1000)}",
            "function": {"name": "buy_station",
                         "arguments": json.dumps(args)}}


def _strip_images(messages: list[dict]) -> list[dict]:
    """Text-only copy of `messages` (image_url content parts dropped).

    Some servers (llama.cpp + this Qwen GGUF) hang when a tool-call request
    also carries an image — the vision tokens push the reasoning budget past
    any sane latency (observed: >120s timeout vs ~46s text-only). The tool
    rounds don't need the image: `get_board_state` returns the classified
    board as text, and the model's board knowledge comes from that tool
    result, not the screenshot. The image stays in the full message list for
    the prefill'd answer round (`_ask_once`), which is fast (~22s).
    """
    stripped = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [p for p in content if p.get("type") != "image_url"]
            if not content:
                content = ""
            m = {**m, "content": content}
        stripped.append(m)
    return stripped


class VisionDrivenPlanner(Planner):
    def __init__(self, client: LLMClient | None = None,
                 base_url: str = "http://localhost:8080",
                 log: SessionLog | None = None,
                 chat_log_path: Path = Path("llm_chats.jsonl"),
                 reasoning: bool = True,
                 max_retries: int = 50,
                 learnings_path: Path = LEARNINGS_DEFAULT,
                 glossary_path: Path = GLOSSARY_DEFAULT,
                 classifier: TemplateClassifier | None = None,
                 live: bool = True,
                 discover: Callable | None = None,
                 tool_enabled: bool = True,
                 vision_max_dim: int | None = 1536,
                 wiki_check: bool = True,
                 wiki_tool: bool = True,
                 cravings: CravingsReader | None = None,
                 bottombar: "BottomBarReader | None" = None,
                 panels: "PanelReader | None" = None,
                 popup_reader: Callable | None = None,
                 satiety_reader: "SatietyReader | None" = None,
                 slime_vat: "SlimeVatReader | None" = None,
                 champions: ChampionReader | None = None,
                 hints: bool = True,
                 shop=None,
                 queue_box=None):
        super().__init__()
        # live=False (static --screenshot regression runs) must be genuinely
        # offline: no default client construction, so a running llama server
        # can never influence a frozen regression check (the old
        # `client or LLMClient(...)` silently queried the live model).
        self.live = live
        self.client = client or (LLMClient(base_url=base_url) if live else None)
        self.fallback = HeuristicPlanner(noop=self._noop)  # shared noop backoff
        self.log = log or SessionLog()
        self.chat_log_path = chat_log_path
        self.reasoning = reasoning
        self.max_retries = max_retries
        self.learnings_path = learnings_path
        self.glossary_path = glossary_path
        self.knowledge_dir = _get_knowledge_dir(glossary_path) if glossary_path else None
        # Strategy planner (System 2) - runs periodically to set strategy
        self.classifier = classifier
        self.discover = discover
        # Text-only server support: the first failed image-carrying request
        # flips this off and _ask_once retries without the screenshot (the
        # board reaches the model via the get_board_state tool result text).
        self._vision_ok = True
        self.popup_reader = popup_reader
        self.tool_enabled = tool_enabled
        self.vision_max_dim = vision_max_dim
        self.wiki_check = wiki_check
        self.wiki_tool = wiki_tool
        self.cravings = cravings
        self.bottombar = bottombar
        self.panels = panels
        self.satiety_reader = satiety_reader
        self.slime_vat = slime_vat
        self.champions = champions
        self.hints = hints
        self.shop = shop               # StationShop: buy_station tool + Runes line
        self.queue_box = queue_box     # QueueBox: collect_queue placement tool
        self._currency = None          # per-step rune balance cache (set during buy_station)
        self._last_currency_result = None  # most recent buy_station(confirm=false) result; surfaced in board state when the panel closes
        self._last_currency_step = -1   # step the last buy_station was run
        self._last_champion = None      # last champion id seen on the tracker
        self._champion_cache = None     # last full champion screen read: {champions, step}
        self._last_craving = None      # last craved item base name seen
        self._last_craving_level = None  # level last believed for that item (menu read) — part of the cache-freshness key
        self._craving_cache = None     # last full menu read: {item, level, count_done, count_required, reward, step}
        self.fallback.slime_count = None      # per-step slime vat count (HUD digit)
        self.fallback.slime_capacity = None    # per-step slime vat max (popup-OCR or default)
        self._feats_cache = None       # last FEATS panel read: {tier, feats, step}
        self._last_feat_collect_step = -1000  # step of last feat collect attempt (for opportunistic retry)
        self._last_feat_collect_found = True  # did it collect anything (skip stale re-tries)
        self._defer_panel_reads = False  # set per-step: a buy will open the panel, skip scheduled feats reads
        self._answer_system = None     # short decision prompt for _ask_once (set per-step in _drive)
        self._compel_misfires = {}     # family -> consecutive compelled buys with no effect (backoff)
        self._unbanked_votes = {}      # (row,col) -> name votes for consensus banking
        # Popup-label persistence (Sep 7): eyeball vs eyemonster_lvl1 are
        # template-indistinguishable (cross-bank matches ~1.0 both ways), so
        # the classifier coin-flips every step and no bank growth can fix
        # it. A popup read is ground truth for that cell until the cell
        # genuinely changes — persist {(row,col): (item_id, step)} and
        # re-apply over a template label inside the ambiguous pair (see
        # apply_label_memory). Scoped to the pair: any other template
        # label, an empty cell, or a cell touched by our move drops the
        # entry; entries also expire after LABEL_MEMORY_TTL steps.
        self._label_memory: dict[tuple, tuple] = {}
        self._strategy = None          # model-committed objective: {feat, target, step}
        self._strategy_family = None   # family the strategy says to build (for buy_station validation)
        # The two-phase buy state lives on self.shop._pending_buy (set inside
        # StationShop.buy when confirm=false stages the dialog). The planner
        # reads it via that attribute; no local copy needed.
        self._frame = None
        self._session_start = time.time()
        self._last_summary_t = self._session_start   # events at/after this are un-summarized
        self._step_count = 0
        self._turn = 0                                # API round counter within this planner
        self._step_llm_rounds = 0                   # LLM rounds this step (llm_budget telemetry)
        # cross-step rejection memory. The previous per-step `rejected`
        # set was cleared at the start of each `_drive` call, so the model saw
        # the same rejected feed/merge every step until something else on the
        # board shifted (e.g. the user reported "always suggest feeding the
        # mana pool, the mana pot, and the lvl1 coin" — the model re-emits the
        # same feed over and over because the rejection correction only lives
        # in the in-step `history` list). The list accumulates this step's
        # rejected (kind, cell_a, cell_b, reason) tuples, ALL of them are
        # carried into the next step's `rejected` set so attempt 0 of the
        # next step pre-blocks every rejected action from the previous step
        # (NOT just the last one — the user pointed out that storing only the
        # last one lets the model re-emit the earlier ones), and the reasons
        # are summarized into the `get_board_state` line so the model knows
        # WHY. Entries expire after CROSS_STEP_TTL steps to keep the memory
        # bounded when the board shifts and the actions become valid again.
        self._step_rejections: list[tuple] = []
        self._cross_step_rejected: list[tuple] = []   # [(kind, cell_a, cell_b, reason), ...]
        # one-time dedup of an existing noisy learnings.md. The
        # append_learning dedup fires for NEW entries (paraphrase-aware
        # via the new token-set Jaccard match in _matches) but doesn't
        # touch STORED entries — so a history of repeated paraphrases
        # accumulates. Run a single dedup pass at startup to collapse
        # near-duplicates (token Jaccard >= 0.7) and keep the most
        # recently-confirmed entry. Cheap (~160 entries takes <1s) and
        # self-correcting: if the model is the same later, it'll be
        # re-added on the next summarize call.
        self._dedup_existing_learnings()
        # Satiety-based craving-bonus (B) learning: observe X before/after a
        # craved feed to back out the bonus (delta - recorded feed value), so
        # the craved-overflow gate tightens as the game's bonus scales.
        self._last_satiety = None        # (num, den) from the last step, or None
        self._prev_feed_was_craved = False  # last returned move fed the craved item
        self._prev_feed_value = None         # its recorded feed value Z

    def _learnings_context(self) -> str:
        """Past learnings to inject into the prompt (semantic memory)."""
        text = read_learnings(self.learnings_path)
        return "" if not text else f"\n\nPast learnings (apply them if relevant):\n{text}"

    def _glossary_context(self) -> str:
        """Past item knowledge to inject into the prompt (visual memory)."""
        text = read_glossary(knowledge_dir=self.knowledge_dir)
        return "" if not text else f"\n\nItems you have learned to recognize (apply if relevant):\n{text}"

    def _chain_map(self) -> dict[str, list[str]]:
        """Family -> merge-chain ids from the glossary `(chain)` blocks."""
        return read_chains(self.knowledge_dir)

    def invalidate_cravings(self) -> None:
        """Force `get_cravings` to re-read next step. main.py calls this after
        dismissing a Devourer level-up screen: a level-up almost always means
        the current craving just completed, and keeping the stale cached read
        would make the model believe the OLD craving (e.g. skeleton 1/2) for
        minutes. Dropping the cache makes `_craving_cache_fresh` return False
        so the tool is re-offered and the new craving is learned in one step."""
        stale = self._craving_cache
        if stale is not None:
            self.log.log("craving_cache_invalidated",
                         item=stale.get("item"),
                         count_done=stale.get("count_done"),
                         count_required=stale.get("count_required"))
        self._craving_cache = None
        self._last_craving_level = None

    def next_move(self, board, frame=None) -> Move:
        self._step_count += 1
        # tick the dock reader so it can throttle runtime
        # position detection.
        if self.bottombar is not None and hasattr(self.bottombar, "set_step"):
            self.bottombar.set_step(self._step_count)
            # per-step rejection accumulator reset. The previous step's
        # rejected (kind, cell_a, cell_b, reason) tuples live in
        # `self._cross_step_rejected` — they get folded into the new step's
        # `rejected` set in `_drive` so attempt 0 pre-blocks every one of
        # them. We clear the local list here; the cross-step list is
        # refreshed as this step's rejections come in.
        self._step_rejections = []
        # drop entries from `self._cross_step_rejected` that are
        # older than CROSS_STEP_TTL steps, OR whose rejected cell(s) no
        # longer hold the same items. The TTL bounds the memory so a
        # rejection that was valid N steps ago doesn't block a re-attempt
        # after the board shifted (e.g. a feed that overflowed because the
        # mana bar was full — once the bar drains, the same feed is valid
        # again and should be re-considered). The item check lapses a
        # rejection the moment the board moves on from the exact cell+item
        # (e.g. the rune got merged away) even before the TTL — otherwise a
        # rejection would stick to a cell that now holds something else.
        self._cross_step_rejected = [
            rec for rec in self._cross_step_rejected
            if self._step_count - rec.get("step", self._step_count) < CROSS_STEP_TTL
            and self._cross_step_rejection_still_valid(rec, board)
        ] if self._cross_step_rejected else []
        self._cross_step_rejected = [
            rec for rec in self._cross_step_rejected
            if self._step_count - rec.get("step", self._step_count) < CROSS_STEP_TTL
            and self._cross_step_rejection_still_valid(rec, board)
        ] if self._cross_step_rejected else []
        self._noop.tick()
        # pending-buy expiry. A confirm=false call starts the
        # two-phase buy; if the model never calls confirm=true, the
        # pending buy auto-expires after PENDING_BUY_TTL steps so a
        # future confirm=true can't accidentally complete a stale
        # dialog from a long-forgotten attempt.
        # Read from self.shop — StationShop sets _pending_buy on its own
        # attribute, so the planner's local _pending_buy is always None.
        # Proxy via the shop to keep this check in sync.
        pending = self.shop._pending_buy if self.shop is not None else None
        if (pending
                and self._step_count - pending.get("step", 0)
                > PENDING_BUY_TTL):
            self.log.log("buy_pending_expired",
                         family=pending.get("family"),
                         age=self._step_count - pending.get("step", 0))
            if self.shop is not None:
                self.shop._pending_buy = None
        self.fallback.max_level_ids = self._max_level_ids()
        self.fallback.chain_map = self._chain_map()
        self.fallback.feed_values = self._feed_values()
        self.fallback.damage_values = self._damage_values()
        # Level-aware craving: the authoritative level comes from the last full
        # menu read (`get_cravings`). The bubble cue alone has no level, so
        # this stays None until a menu read lands — craved feeds then fall back
        # to the conservative low-tier gate (CRAVED_UNKNOWN_LEVEL_MAX).
        self.fallback.craved_level = (self._craving_cache or {}).get("level")
        self.fallback.craved_need = self._craving_need_remaining()
        self._last_craving_level = self.fallback.craved_level
        self._satiety_context(frame)
        # Feat-driven priorities (from the cached FEATS read + the craving
        # + the income objective when a build strategy is active and an
        # icerune stack is on the board).
        weights = self._feat_weights(board)
        self.fallback.feat_weights = weights
        max_level_ids = self._max_level_ids()
        income = income_objective(self._strategy, board, max_level_ids)
        self.fallback.feed_objective_active = self._feed_objective_active() or income is not None
        self.fallback.feed_prefer_item = (
            income.target if income is not None else weights.get("feat_target"))
        # Income feeds target the MAX-LEVEL stack (largest Ice grant) — a lvl1
        # icerune feed wastes the economy. Feat-target prefers stay cheapest.
        self.fallback.feed_prefer_max = income is not None
        # Panel-visit economy: at most one scheduled panel cycle per step.
        # A buy (pending confirm, affordable/cold-start strategy or craving
        # need) will open the Station panel in the optional round — when it
        # will, skip this step's scheduled FEATS read/collect so the step
        # costs one panel cycle instead of two. Pure cache reads, no taps.
        self._defer_panel_reads = self._want_buy_this_step(board)
        if frame is None:
            self.log.log("vision_drive", ok=False, reason="no_frame")
            move = self.fallback.next_move(board, frame)
            self._record_feed_context(move, board)
            self._log_move_outcome(move, success=True, board=board)
            return move
        # Queue auto-collect (code-driven, not model-driven): the model
        # never emits native tool calls on this stack, so collect_queue can
        # never fire voluntarily (observed: zero queue_collected in 3329
        # lines). When a reward is queued AND the board has room, place it
        # BEFORE planning so _drive sees the post-placement board. The
        # placed cell is patched occupied (with the item id when known) to
        # keep this step's planning consistent; next step re-classifies.
        # Congested boards skip (collect would refuse anyway); errors log
        # as queue_collect_miss instead of stalling the step.
        self._auto_collect_queue(board, frame)
        # Ad-offer observer (detect-only, never taps): a locked chest on
        # the board is the only grounded ad trigger ("watch an advert to
        # unlock"). Log sightings so the watch→claim cycle can be built
        # once real offer UI is captured. See vision/ads.py.
        try:
            if any((c.item_id or "").startswith("lockedchest")
                   for c in board.cells if c.occupied):
                from vision.ads import detect_ad_offer
                offer = detect_ad_offer(frame, True)
                if offer:
                    self.log.log("ad_offer_seen", text=offer.get("text"))
        except Exception:
            pass
            # don't fall back to the heuristic on retry exhaustion. The
        # previous path caught ValueError (raised when `_drive` exhausts
        # max_retries) and ran the heuristic, which silently substituted a
        # move the model never agreed to. The model couldn't see why its
        # output kept getting overridden. We now let the retry-exhaustion
        # propagate and convert it to an `idle` move here, so the next
        # step gets a fresh attempt with the rejected-set cleared.
        # LLMError (network / API down) still falls back — the heuristic is
        # the only option when there's no LLM response at all.
        try:
            move = self._drive(board, frame)
        except LLMError as exc:
            self.log.log("vision_drive", ok=False, reason="llm_error", detail=str(exc))
            move = self.fallback.next_move(board, frame)
        except (ValueError, KeyError, TypeError) as exc:
            import traceback
            tb = traceback.format_exc()
            if isinstance(exc, ValueError) and "no valid move" in str(exc):
                # Retry exhaustion: the model produced only invalid moves.
                # Log + idle, don't substitute a heuristic move. The next
                # step gets a fresh attempt (the rejected set is per-step).
                self.log.log("vision_drive", ok=False, reason="retry_exhausted",
                             detail=str(exc))
                move = Move(kind="idle")
            else:
                # Unexpected error: log + idle (also no heuristic fallback).
                self.log.log("vision_drive", ok=False, reason=str(exc), traceback=tb)
                move = Move(kind="idle")
        self._record_feed_context(move, board)
        self._log_move_outcome(move, success=True, board=board)
        # Per-step LLM round telemetry (log-only, no behavior change):
        # every client round flows through _log_chat, so the count bounds
        # future budget enforcement and diagnoses slow steps today.
        try:
            rounds = self._step_llm_rounds
            self.log.log("llm_budget", rounds=rounds, step=self._step_count)
        except Exception:
            pass
        self._step_llm_rounds = 0
        return move

    def _satiety_context(self, frame) -> None:
        """Read the Devourer satiety into fallback.satiety_remaining/capacity
        AND learn the craving bonus (B) by comparing the satiety delta since the
        last step when the move returned then was a craved feed.

        B = X_after - X_before - Z (the craving grants Z+B food). Only measured
        when both numerators are read and the denominator is unchanged (a level
        -up between the two reads would reset the bar and make X_after invalid).
        Refresh per observance so it tracks the game's scaling bonus."""
        f = frame if frame is not None else getattr(self, "_frame", None)
        num = den = None
        if self.satiety_reader is not None and f is not None:
            try:
                read = self.satiety_reader.read_satiety(f)
                num, den = read.get("num"), read.get("den")
            except Exception:
                num = den = None
        self.fallback.satiety_remaining = None
        self.fallback.satiety_capacity = None
        if num is not None and den is not None:
            try:
                self.fallback.satiety_remaining = int(den) - int(num)
                self.fallback.satiety_capacity = int(den)
            except (TypeError, ValueError):
                pass
        if (self._last_satiety is not None and num is not None
                and den is not None
                and self._last_satiety[1] == int(den)
                and self._prev_feed_was_craved and self._prev_feed_value is not None):
            delta = int(num) - self._last_satiety[0]
            bonus = delta - self._prev_feed_value
            if delta >= self._prev_feed_value and bonus >= 0:
                self.fallback.craving_bonus_est = bonus
                self.log.log("craving_bonus", learned=int(bonus),
                             feed_value=self._prev_feed_value,
                             satiety_before=self._last_satiety[0],
                             satiety_after=int(num))
        self._prev_feed_was_craved = False
        self._prev_feed_value = None
        if num is not None and den is not None:
            self._last_satiety = (int(num), int(den))
        else:
            self._last_satiety = None
            # also cache the slime-vat count + capacity (separate HUD
        # readout, mirrors SatietyReader pattern; same bank + Apple Vision
        # pipeline. Q4 = C: capacity is read from the popup only, so the
        # cache is None whenever the popup isn't open and the line degrades
        # to "Slime: N" with no fake percent.
        self.fallback.slime_count = self._slime_count(frame)
        self.fallback.slime_capacity = self._slime_capacity(frame)

    def _record_feed_context(self, move: Move, board) -> None:
        """Record the feed context for a move, so the NEXT step can back out the craving bonus
        from the satiety delta.

        Also detects a craving COMPLETED by this feed (level-aware: feeding the
        exact craved monster increments the count 1:1) and invalidates the
        cache so the next step re-offers `get_cravings` and the bot learns the
        new craving in one step instead of believing the old one for minutes."""
        self._prev_feed_was_craved = False
        self._prev_feed_value = None
        if move.kind != "feed" or not move.cell_a or not self.fallback.craved_item:
            return
        cell = board.cell_at(*move.cell_a)
        if (cell and cell.item_id and craved_matches(
                cell.item_id, self.fallback.craved_item,
                craving_level=self.fallback.craved_level)):
            z = self.fallback.feed_values.get(cell.item_id)
            if z is not None:
                self._prev_feed_was_craved = True
                self._prev_feed_value = z
            cache = self._craving_cache
            if cache and cache.get("count_done") is not None \
                    and cache.get("count_required"):
                if int(cache["count_done"]) + 1 >= int(cache["count_required"]):
                    self.log.log("craving_complete_inferred",
                                 item=cache.get("item"),
                                 count_done=cache["count_done"] + 1,
                                 count_required=cache["count_required"])
                    self._craving_cache = None

    def _track_learning_advice(self, move: Move, board) -> str | None:
        """Identify which learning's advice was followed by this move.
        Returns the learning text if matched, else None."""
        if not move or not board:
            return None
        
        # Get committed learnings
        from planner.learnings import read as read_learnings
        committed = [l for l in read_learnings(self.learnings_path) if l.committed]
        
        move_desc = f"{move.kind} {move.cell_a} {move.cell_b or ''}".strip()
        
        # Simple keyword matching - check if move aligns with learning
        for l in committed:
            text = l.text.lower()
            move_lower = move_desc.lower()
            # Check for keyword overlap
            learning_words = set(text.split())
            move_words = set(move_lower.split())
            overlap = learning_words & move_words
            if len(overlap) >= 2:  # At least 2 word overlap
                return l.text
        return None

    def _log_move_outcome(self, move: Move, success: bool, board) -> None:
        """Log outcome of a move for learning efficacy tracking."""
        learning_text = self._track_learning_advice(move, board)
        if learning_text:
            outcome = "followed_success" if success else "followed_failure"
            log_learning_outcome(self.learnings_path, learning_text, f"step_{self._step_count}", outcome)

    def summarize_session(self, board, frame=None) -> str:
        """Distill un-summarized events since the last call into learnings.

        Called periodically during play (every few steps) and once at session
        end. Only events at/after the last-summary watermark are included, and
        the watermark advances on success, so a window is never summarized
        twice. Skips (returns "") when there is nothing new.

        The model sees the transcript plus its currently stored learnings and
        replies with JSON carrying:
          learnings      -> appended as NEW candidates (trust gating)
          items          -> appended to the item glossary
          remove         -> stored learnings to prune (self-correction)
          confirmed      -> stored learnings to promote (candidate -> committed)
          remove_items   -> glossary entries to prune
        Windows that ended in heuristic-fallback moves add negative evidence to
        the learnings they produced, so failure-prone lessons get dropped
        instead of persisting.
        """
        window = [e for e in self.log.events
                  if e.get("t", 0) >= self._last_summary_t
                  and e.get("event") != "learning"]
        if not window:
            return ""
        # Cap the window: a 50-retry step emits hundreds of events, and an
        # unbounded transcript + image blew the 240s request budget (observed
        # llm_error timeout). The tail carries the live signal; older events
        # in the window were already partially summarized by prior runs.
        # (Also see main._rotate_logs for on-disk rotation.)
        MAX_SUMMARY_EVENTS = 400
        truncated = 0
        if len(window) > MAX_SUMMARY_EVENTS:
            truncated = len(window) - MAX_SUMMARY_EVENTS
            window = window[-MAX_SUMMARY_EVENTS:]
        fallbacks = sum(1 for e in window
                        if e.get("event") == "vision_drive" and e.get("ok") is False)
        try:
            transcript = self._session_transcript(window)
            if truncated:
                transcript += (f"\n(earlier {truncated} events in this window "
                               "omitted for size; tail above is complete)")
            user_content = transcript
            if frame is not None:
                text = transcript
                if board is not None:
                    text += "\n\nFinal board (template labels):\n" + self._board_state_text(board)
                user_content = [
                    {"type": "image_url", "image_url": {"url": _encode_frame(frame, max_dim=self.vision_max_dim)}},
                    {"type": "text", "text": text},
                ]
            system = SUMMARY_SYSTEM_PROMPT
            knowledge = self._stored_knowledge_context()
            if knowledge:
                system += "\n\n" + knowledge
            if fallbacks:
                system += (f"\n\nNote: this window had {fallbacks} move(s) rejected by validation "
                           "and played by the heuristic fallback instead. If any STORED learning "
                           "likely caused them, list it under 'remove'.")
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": '{"learnings":'},
            ]
            # assistant prefill forces JSON emission instead of rambling
            # reasoning; without it the reasoning model exhausts max_tokens
            # mid-JSON and _split_summary stores a truncated raw blob.
            reply, full_msg = self.client.chat(messages, max_tokens=2048, json_mode=False)
        except LLMError as exc:
            self.log.log("learning", ok=False, reason="llm_error", detail=str(exc))
            return ""
        data = self._split_summary(reply)
        stamp = datetime.now().strftime("%b %d, %Y %H:%M")
        title = f"{stamp} (step {self._step_count})"
        # Mass-delete guard: a single window once listed 9 learnings for
        # removal (and 40+ glossary blocks) while simultaneously confirming
        # the same texts — self-contradictory output that gutted
        # learnings.md. Contradicted removals are dropped, and each list
        # is capped; legitimate pruning is occasional (1-2 per window).
        data["remove"], data["confirmed"], data["remove_items"] = \
            self._sanitize_prunes(data)
        pruned = (prune_learning(self.learnings_path, data["remove"])
                  if data["remove"] else 0)
        confirmed = (confirm_learning(self.learnings_path, data["confirmed"])
                     if data["confirmed"] else 0)
        pruned_items = (prune_glossary_by_ids(data["remove_items"], protected=True)
                        if data["remove_items"] else 0)
        data["learnings"] = self._valid_learning_texts(data["learnings"])
        # PHASE 2: Verification pass - check consistency against committed learnings + glossary
        verified_learnings = self._verify_learnings(data["learnings"])

        added = append_learning(self.learnings_path, verified_learnings, title=title)
        # Fold this window's validator rejection reasons into the memory
        # store as grounded game-rule learnings. These bypass the summary
        # model (whose "rejected"/"validate" learnings get stripped by
        # `_valid_learning_texts`), because a rejection reason like
        # `feed_not_max_level:icerune_lvl1` is a real game constraint, not
        # meta-noise. Persisting them stops the model from wasting LLM calls
        # re-attempting the same rejected feed across steps.
        rejection_added = self._rejection_reasons_to_learnings()
        if rejection_added:
            self.log.log("learning", ok=True, step=self._step_count,
                         rejection_added=rejection_added)
        reward_added = self._reward_learnings()
        if reward_added:
            self.log.log("learning", ok=True, step=self._step_count,
                         reward_added=reward_added)
        if data["items"]:
            items = self._valid_glossary_items(data["items"])
            chain_lines = [i for i in items if self._parse_chain_line(i) is not None]
            spawn_lines = [i for i in items if self._parse_spawn_line(i) is not None]
            # chain and spawn lines are handled separately; spawn lines are not
            # treated as chains even if they contain "->"
            chain_lines = [l for l in chain_lines if self._parse_spawn_line(l) is None]
            if chain_lines:
                chains_written = self._write_chain_blocks(chain_lines)
            else:
                chains_written = 0
            if spawn_lines:
                spawns_written = self._write_spawn_blocks(spawn_lines)
            else:
                spawns_written = 0
            plain = [i for i in items if i not in chain_lines and i not in spawn_lines]
            if plain:
                # Write plain items to new JSON format, but only keep valid glossary entries
                # (chain/spawn/popup format with facts). Bare IDs like "bone" or "bone: spawns"
                # are board snapshots that should be dropped (per Aug 23 prune behavior).
                from planner.glossary import write_visual_marker, write_item_stat, _get_knowledge_dir
                kd = _get_knowledge_dir(self.glossary_path) if hasattr(self, 'glossary_path') else None
                for item in plain:
                    item = item.strip()
                    if item.startswith("- "):
                        item = item[2:].strip()
                    # Only write items that look like proper glossary entries
                    if ": " in item:
                        name, rest = item.split(": ", 1)
                        name = name.strip()
                        # Check if it's a known fact format
                        if rest.startswith("feed value:") or rest.startswith("damage value:") or rest.startswith("max_level:") or rest.startswith("description:"):
                            if rest.startswith("feed value:"):
                                write_item_stat(name.strip(), feed=int(rest.split(":")[1].strip()), source="popup", knowledge_dir=kd)
                            elif rest.startswith("damage value:"):
                                write_item_stat(name.strip(), damage=int(rest.split(":")[1].strip()), source="popup", knowledge_dir=kd)
                            elif rest.startswith("max_level:"):
                                write_item_stat(name.strip(), max_level=True, source="popup", knowledge_dir=kd)
                            elif rest.startswith("description:"):
                                write_visual_marker(name.strip(), description=rest.split(":", 1)[1].strip(), source="popup", knowledge_dir=kd)
                        # Otherwise it's a bare ID or unrecognized format - drop it



        derated = 0
        if fallbacks and added:
            derated = derate_learning(self.learnings_path, verified_learnings)
        wiki_confirmed = wiki_refuted = 0
        if data["learnings"]:
            supported, refuted = self._wiki_check_learnings([l["text"] for l in verified_learnings if isinstance(l, dict)])
            if supported:
                wiki_confirmed = confirm_learning(self.learnings_path, supported)
            if refuted:
                wiki_refuted = prune_learning(self.learnings_path, refuted)
        self.log.log("learning", ok=True, step=self._step_count,
                     added=added, confirmed=confirmed, pruned=pruned,
                     pruned_items=pruned_items, dropped=derated, fallbacks=fallbacks,
                     wiki_confirmed=wiki_confirmed, wiki_refuted=wiki_refuted)
        # Demote stale committed learnings that haven't been reinforced
        demoted = demote_stale_learnings(self.learnings_path)
        if demoted:
            self.log.log("learning_demoted", count=demoted)
        # Drop stale inferred glossary entries (unconfirmed guesses age
        # out; grounded re-reads upgrade their source and exempt them).
        try:
            from planner.glossary import prune_inferred_entries, _get_knowledge_dir
            kd = _get_knowledge_dir(self.glossary_path) if hasattr(self, 'glossary_path') else None
            n_pruned = prune_inferred_entries(knowledge_dir=kd)
            if n_pruned:
                self.log.log("learning_pruned_inferred", count=n_pruned)
        except Exception:
            pass
        self._advance_memory_watermark()
        return reply

    def _advance_memory_watermark(self) -> None:
        """Advance the summarize watermark and drop summarized RAM events.

        The on-disk session.jsonl keeps everything; transcript + cross-step
        filters already scope by watermark, so nothing reads pruned events
        again. Shared by the full summary and the deterministic maintenance.
        """
        self._last_summary_t = time.time()
        # Bound in-RAM event growth: drop summarized events (the on-disk
        # session.jsonl keeps everything; transcript + cross-step filters
        # already scope by watermark, so nothing reads these again).
        try:
            before = len(self.log.events)
            self.log.events = [e for e in self.log.events
                               if e.get("t", 0) >= self._last_summary_t]
            if before > len(self.log.events):
                self.log.log("learning_pruned_events",
                             dropped=before - len(self.log.events))
        except Exception:
            pass

    # Full LLM summaries run every Nth maintenance (plus session end): the
    # per-window pattern-mining call is high-variance (it once gutted
    # learnings.md) and can blow the 240s budget, while the deterministic
    # folds below carry the durable signal every window.
    MAINTAIN_PER_SUMMARY = 10

    def maintain_memory(self, board=None, frame=None) -> bool:
        """Deterministic per-window memory maintenance (no LLM call).

        Folds validator rejections + rewards into learnings, demotes stale
        entries, prunes inferred glossary guesses, advances the watermark.
        Every MAINTAIN_PER_SUMMARY-th call (and only then) runs the full
        LLM pattern-mining summary instead. Returns True when anything was
        written (learnings updated) so the caller can announce it.
        """
        self._maintain_count = getattr(self, "_maintain_count", 0) + 1
        if self._maintain_count % self.MAINTAIN_PER_SUMMARY == 0:
            return bool(self.summarize_session(board, frame))
        window = [e for e in self.log.events
                  if e.get("t", 0) >= self._last_summary_t
                  and e.get("event") != "learning"]
        if not window:
            return False
        wrote = False
        try:
            if self._rejection_reasons_to_learnings():
                self.log.log("learning", ok=True, step=self._step_count,
                             rejection_added=True)
                wrote = True
        except Exception:
            pass
        try:
            if self._reward_learnings():
                self.log.log("learning", ok=True, step=self._step_count,
                             reward_added=True)
                wrote = True
        except Exception:
            pass
        try:
            if demote_stale_learnings(self.learnings_path):
                wrote = True
        except Exception:
            pass
        try:
            from planner.glossary import prune_inferred_entries, _get_knowledge_dir
            kd = _get_knowledge_dir(self.glossary_path) if hasattr(self, 'glossary_path') else None
            if prune_inferred_entries(knowledge_dir=kd):
                wrote = True
        except Exception:
            pass
        self._advance_memory_watermark()
        return wrote

    def _stored_knowledge_context(self) -> str:
        """Current stored learnings + glossary, so the summary call can prune
        or confirm them (references copied verbatim)."""
        parts = []
        stored = read_learnings(self.learnings_path)
        if stored:
            parts.append("Your current stored learnings (copy the exact text of any you "
                         "want to remove or confirm into the JSON fields):\n" + stored)
        gl = read_glossary(knowledge_dir=self.knowledge_dir)
        if gl:
            parts.append("Your current stored item glossary (copy any wrong item ids into "
                         "'remove_items'):\n" + gl)
        return "\n\n".join(parts)

    @staticmethod
    def _split_summary(reply: str) -> dict:
        """Parse the model's JSON summary into a dict of lists. On a parse
        failure the raw reply is treated as one new learning (graceful fallback
        so a bad reply never drops progress)."""

        def as_list(vals) -> list:
            if not vals:
                return []
            if isinstance(vals, str):
                vals = [vals]
            return [str(v).strip() for v in vals if str(v).strip()]

        def as_learnings(vals) -> list[dict]:
            """Convert learnings to new format: list of {text, type, confidence}."""
            if not vals:
                return []
            if isinstance(vals, str):
                vals = [vals]
            result = []
            for v in vals:
                if isinstance(v, dict):
                    # New format: {text, type, confidence}
                    text = str(v.get("text", "")).strip()
                    ltype = v.get("type", "pattern")
                    confidence = float(v.get("confidence", 0.7))
                    if text:
                        result.append({"text": text, "type": ltype, "confidence": confidence})
                else:
                    # Old format: plain string
                    text = str(v).strip()
                    if text:
                        result.append({"text": text, "type": "pattern", "confidence": 0.7})
            return result

        try:
            data = _extract_json(reply)
            if not isinstance(data, dict):
                raise ValueError("summary JSON is not an object")
        except (ValueError, json.JSONDecodeError):
            return {"learnings": [{"text": reply.strip(), "type": "pattern", "confidence": 0.7}] if reply.strip() else [],
                    "items": [], "remove": [], "confirmed": [], "remove_items": []}
        return {"learnings": as_learnings(data.get("learnings")),
                "items": as_list(data.get("items")),
                "remove": as_list(data.get("remove")),
                "confirmed": as_list(data.get("confirmed")),
                "remove_items": as_list(data.get("remove_items"))}

    # Per-window prune caps: legitimate pruning removes 1-2 entries; a
    # model dumping 9 learnings + 40 glossary blocks in one window while
    # confirming the same texts is misbehaving, not maintaining.
    MAX_REMOVE_PER_WINDOW = 3
    MAX_REMOVE_ITEMS_PER_WINDOW = 5

    @staticmethod
    def _sanitize_prunes(data: dict) -> tuple[list, list, list]:
        """Drop self-contradictory removals and cap prune lists.

        A text appearing in both remove and confirmed (or in the window's
        own new learnings) is a contradiction — keep it. Returns
        (remove, confirmed, remove_items), logging nothing (caller logs
        counts; dropped entries are visible as count deltas).
        """
        def norm(t: str) -> str:
            return re.sub(r"\s+", " ", str(t or "").strip().lower())

        learning_texts = {norm(l.get("text", "") if isinstance(l, dict) else l)
                          for l in (data.get("learnings") or [])}
        confirmed = [c for c in (data.get("confirmed") or [])]
        confirmed_norm = {norm(c) for c in confirmed}
        remove = [r for r in (data.get("remove") or [])
                  if norm(r) not in confirmed_norm
                  and norm(r) not in learning_texts]
        remove_items = list(data.get("remove_items") or [])
        return (remove[:VisionDrivenPlanner.MAX_REMOVE_PER_WINDOW],
                confirmed,
                remove_items[:VisionDrivenPlanner.MAX_REMOVE_ITEMS_PER_WINDOW])


    def _verify_learnings(self, new_learnings: list[dict]) -> list[dict]:
        """Verification pass (Pass 2): Check new candidate learnings for consistency
        against committed learnings, glossary, and confidence thresholds.
        Returns filtered list of learnings that pass verification (score >= 0.6)."""
        if not new_learnings:
            return []
        
        # Get committed learnings for contradiction detection
        from planner.learnings import read as read_learnings
        committed = [l for l in read_learnings(self.learnings_path) if l.committed]
        
        # Get glossary stats for fact/visual cross-check
        from planner.glossary import read_item_stats, _get_knowledge_dir
        kd = _get_knowledge_dir(self.glossary_path) if hasattr(self, 'glossary_path') else None
        glossary_stats = read_item_stats(kd)
        
        # Local verification (fast, no LLM)
        from planner.learnings import verify_learning_consistency
        results = verify_learning_consistency(new_learnings, committed, glossary_stats)
        
        # Filter: keep only those with score >= 0.6. `verify_learning_consistency`
        # returns the learning under "learning" as a Learning OBJECT (it calls
        # `_to_learning` on each candidate), so we normalize back to a dict here
        # (the contract of this method + all downstream consumers — append_learning,
        # derate_learning, _wiki_check_learnings — expect dicts, not Learning
        # objects). Passing a Learning object downstream crashes `_matches`/`_norm`
        # with "'Learning' object has no attribute 'strip'" (e.g. derate_learning
        # at the learning-summary step).
        verified = []
        for r in results:
            if r["consistent"]:
                l = r["learning"]
                if isinstance(l, Learning):
                    l = {"text": l.text, "type": l.type,
                         "confidence": l.confidence, "status": l.status,
                         "confirmed": l.confirmed, "negative": l.negative,
                         "title": l.title, "outcome_log": list(l.outcome_log)}
                elif isinstance(l, dict):
                    l = dict(l)  # copy
                # Add verification metadata
                l["_verified"] = True
                l["_verification_score"] = r["score"]
                if r["issues"]:
                    l["_verification_issues"] = r["issues"]
                verified.append(l)
            else:
                self.log.log("learning_verification", ok=False,
                             reason="inconsistent", detail={"issues": r["issues"], "score": r["score"]})
        
        # For fact/visual types with high confidence, also run LLM verification if needed
        # (Optional: could add a second LLM call here for deeper checking)
        
        return verified

    def _wiki_check_learnings(self, learnings: list[str]) -> tuple[list[str], list[str]]:
        """Fact-check candidate learnings against the wiki (option 2).

        Picks item/mechanics topics out of the new candidates, fetches the
        wiki text for them, and asks the model one batched verdict question.
        Returns (supported, refuted) learning texts:
          supported -> counted as one confirmation (wiki backing)
          refuted   -> dropped (the wiki contradicts a model-derived claim)
        Anything the wiki is silent on is left as-is. No-op (and cheap) when
        the candidates mention no known topics. Disabled via `--no-wiki-factcheck`.
        """
        if not self.wiki_check or not learnings:
            return [], []
        extra_names = []
        gl = read_glossary(knowledge_dir=self.knowledge_dir)
        if gl:
            extra_names = re.findall(r"## Item (\S+)", gl)
        topics = topics_for(learnings, extra_names)
        if not topics:
            return [], []
        wiki_texts = fetch_topic_texts(topics)
        if not wiki_texts:
            return [], []
        blob = "\n\n".join(f"## {page}\n{text}" for page, text in wiki_texts.items())
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(learnings, 1))
        prompt = (
            "Fact-check each numbered learning a bot wrote about NecroMerger "
            "against ONLY the wiki text below. For each, reply \"support\" if "
            "the wiki confirms/agrees, \"contradict\" if it disagrees, "
            "\"unknown\" if the wiki says nothing relevant.\n\n"
            f"Learnings:\n{numbered}\n\n"
            f"Wiki text (necromerger.wiki.gg):\n{blob}\n\n"
            "Reply with ONLY JSON: "
            '{"verdicts": [{"n": 1, "verdict": "support"}, {"n": 2, "verdict": "contradict"}, ...]} '
            "with one entry per learning (n from the list above).")
        messages = [
            {"role": "system", "content": "You fact-check NecroMerger learnings against wiki text."},
            {"role": "user", "content": prompt},
        ]
        try:
            content, _, _full = self.client.chat_message(messages, max_tokens=512, json_mode=True)
            data = _extract_json(content)
        except (LLMError, ValueError, json.JSONDecodeError):
            return [], []
        verdicts = data.get("verdicts") if isinstance(data, dict) else None
        if not isinstance(verdicts, list):
            return [], []
        supported, refuted = [], []
        for v in verdicts:
            if not isinstance(v, dict):
                continue
            try:
                idx = int(v.get("n")) - 1
                text = learnings[idx] if 0 <= idx < len(learnings) else None
            except (TypeError, ValueError):
                continue
            if not text:
                continue
            verdict = str(v.get("verdict") or "").lower()
            if verdict == "support":
                supported.append(text)
            elif verdict == "contradict":
                refuted.append(text)
        return supported, refuted

    def _session_transcript(self, events: list[dict] | None = None) -> str:
        """Render un-summarized session events into a compact transcript.

        Pass a pre-capped window (see summarize_session) to bound the
        prompt; defaults to all un-summarized events.
        """
        if events is None:
            events = [e for e in self.log.events
                      if e.get("t", 0) >= self._last_summary_t
                      and e.get("event") != "learning"]
        lines = []
        for e in events:
            name = e.get("event")
            fields = {k: v for k, v in e.items() if k not in ("t", "event")}
            lines.append(f"- {name}: {json.dumps(fields, default=str)}")
        if not lines:
            lines.append("- (no events recorded)")
        return "Session transcript:\n" + "\n".join(lines)

    def _drive(self, board, frame) -> Move:
        self._frame = frame
        self._currency = None   # re-read runes per step (they change with buys)
        self._mana_fraction = read_mana_fraction(frame) if frame is not None else None
        self._observe_craving_bubble()
        self._observe_champion_tracker()
        if self.client is None:
            # Static/offline mode: deterministic fallback decision (the mana
            # context above is still computed so the fallback gates on it).
            raise LLMError("vision_drive: offline (static mode)")
        image_url = _encode_frame(frame, max_dim=self.vision_max_dim)
        user = {"type": "text", "text": (
            "This is the current board. The board state is already provided "
            "below (see Tool results) — decide the single best move "
            "and reply with ONLY a valid JSON action "
            "(merge, spawn, feed, attack, collect, or idle). "
            "There are no tools in this round — never emit a tool call "
            "or a tool name as the action.")}
        text = _drive_prompt(board.geometry or FALLBACK_GEOMETRY)
        if self.wiki_tool:
            text += ("\nAnd `lookup_wiki` fetches a NecroMerger wiki page to "
                     "fact-check item behavior or merge chains you are unsure "
                     "about — prefer what the board and popups tell you, and "
                     "only use it when genuinely uncertain.")
        system = text + self._learnings_context() + self._glossary_context()
        # Short decision prompt for the answer round: the full system prompt
        # stays on the tool rounds, but _ask_once decides from geometry +
        # hard rules + schema only (~2k chars). Tags/whitelist/hints/checklist
        # arrive via Tool results, so nothing is lost.
        self._answer_system = _answer_prompt(board.geometry or FALLBACK_GEOMETRY)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}}, user]},
        ]
        self._observe_feats()   # refresh cached feats before get_board_state
        self._auto_read_popup(board)   # learn description/max-level of new items
        self._observe(messages, board)
        # PROACTIVE identification round. When the board has
        # unidentified cells (UNID), the model often gets stuck in
        # cell_a/cell_b_unidentified rejection loops because the bank
        # is too sparse (the 18 `discover_unknown` events in the Aug 28
        # session log were the dominant failure mode). Instead of waiting
        # for the model to call identify_item after a rejection, we run
        # a single batch identify_item call up-front when there are
        # unidentified cells, up to MAX_DISCOVERY cells. The model's
        # own identify_item calls later in the optional-tools round
        # are still allowed — this round is the safety net.
        self._proactive_identify(messages, board)
        optional = self._optional_tools(messages, board)   # (content, move) | None
        history: list[tuple[str, str]] = []
        rejected: set[tuple] = set()   # (kind, cell_a, cell_b) already rejected this step
        # cross-step rejection carry-over. EVERY previous step's
        # rejected (kind, cell_a, cell_b) is pre-loaded into `rejected` so
        # attempt 0 of THIS step can't re-emit any of them. The reasons
        # are summarized into the get_board_state line (handled in
        # _board_state_text via _cross_step_rejected) so the model sees
        # WHY its previous actions were bad. Without this, the model kept
        # suggesting the same feed/merge after every rejection — the
        # correction only lived in the in-step history and was forgotten
        # at step boundary. Storing only the LAST rejection (an earlier
        # draft) was wrong: the model would just shift to the second-to-
        # last one and re-emit it. We now keep all of them.
        for rec in self._cross_step_rejected:
            rejected.add((rec["kind"], rec["cell_a"], rec["cell_b"]))
        if self._cross_step_rejected:
            self.log.log("cross_step_rejection_carried",
                         count=len(self._cross_step_rejected),
                         kinds=[r["kind"] for r in self._cross_step_rejected])
        for attempt in range(1 + self.max_retries):
            msgs = list(messages)
            # Bound retry context: each round appends ~2-4k chars, so an
            # unbounded history overflows the context window (and buries the
            # board state under stale corrections). The last 6 pairs carry
            # all live signal — older rejections persist in `rejected` and
            # `_cross_step_rejected`, which the corrections already summarize.
            for reply, correction in history[-6:]:
                msgs.append({"role": "assistant", "content": reply})
                msgs.append({"role": "user", "content": correction})
            if attempt == 0 and optional is not None:
                # The optional-tools round already produced a valid action
                # JSON — reuse it as attempt 0 instead of burning a fresh
                # `_ask_once` round on the same decision.
                reply, move = optional
                optional = None
            else:
                reply, move = self._ask_once(msgs)
            if move is None:
                # --- fixation detection ---
                # Same reply repeated 3+ times = model is stuck (e.g.
                # `{"action": "get_board_state"}` — the `{"action":` prefill
                # primes the tool name the model just saw).  Emit a targeted
                # correction that names the wrong pattern and lists valid
                # actions, and truncate history so the wall of identical
                # wrong replies doesn't reinforce the fixation.
                recent_replies = [r for r, _ in history[-4:]]
                fixation = sum(1 for r in recent_replies if r == reply)
                # Prioritize the get_board_state hallucination: the model
                # sees get_board_state as a tool and as a string in the
                # prompt/board state, so it often completes '{"action":'
                # with '"get_board_state"}' even when not fixated yet.
                # Give a targeted correction immediately on the first
                # occurrence rather than waiting for fixation >=2.
                if "get_board_state" in (reply or ""):
                    action_list = ("merge, spawn, feed, attack, collect, "
                                   "or idle")
                    correction = (
                        "get_board_state is a TOOL you already called — "
                        "it is NOT a valid action. Reply with ONLY one "
                        "of: " + action_list + ". Example: "
                        '{"action": "merge", "a": [2,1], "b": [2,2]}. '
                        + self._whitelist_snippet(board, rejected))
                    if len(history) > 4:
                        history[:] = history[-2:]
                elif fixation >= 2:
                    action_list = ("merge, spawn, feed, attack, collect, "
                                   "or idle")
                    correction = (
                        f"Your reply {reply[:80]!r} is not a valid "
                        "action. Reply with ONLY one of: " + action_list
                        + ". Example: "
                        '{"action": "merge", "a": [2,1], "b": [2,2]}. '
                        + self._whitelist_snippet(board, rejected))
                    # Break pattern reinforcement: keep only the last 2
                    # history entries so the model doesn't see 10 copies of
                    # the same wrong answer amplifying the fixation.
                    if len(history) > 4:
                        history[:] = history[-2:]
                else:
                    correction = ("Your reply did not contain a valid JSON "
                                  "action. Look at the board again and reply "
                                  "with ONLY a valid JSON action from the "
                                  "schema: merge, spawn, feed, attack, "
                                  "collect, or idle. "
                                  + self._whitelist_snippet(board, rejected))
                history.append((reply, correction))
                self.log.log("vision_drive", ok=False,
                             reason="action_parse_fail",
                             reply=(reply or "")[:120], attempt=attempt + 1,
                             fixation=fixation >= 2)
                # 3-strikes heuristic fallback for parse fixation: the model
                # keeps emitting get_board_state as an action (the prefill
                # primes it) even after targeted corrections. Instead of
                # burning all 50 retries on the same mistake, fall back to
                # the deterministic heuristic after 6 parse failures with
                # fixation. This saves 40+ LLM calls per stuck step.
                if len(history) >= 6 and fixation >= 2 and attempt >= 5:
                    self.log.log("vision_drive", ok=False,
                                 reason="heuristic_fallback_fixation",
                                 attempts=attempt + 1)
                    return self.fallback.next_move(board, self._frame)
                continue
            key = (move.kind, move.cell_a, move.cell_b)
            if key in rejected:
                # Deterministic repeat blocker: never let the model re-propose a
                # move it already tried (it repeats the exact JSON after a
                # correction — a known failure mode). Reject immediately with an
                # explicit hint and a fresh LLM round, instead of re-validating
                # the same invalid move.
                self.log.log("vision_drive", ok=False, reason="repeat_rejected",
                             action=move.kind, cell_a=move.cell_a, cell_b=move.cell_b)
                correction = ("You already proposed this exact move and it was "
                              "rejected. Do NOT propose it again — pick a "
                              "DIFFERENT pair of cells or a different action. "
                              "Valid options this step: " + self._whitelist_snippet(board, rejected))
                history.append((reply, correction))
                # 3-strikes for exact repeats: the model keeps re-sampling
                # the same (kind, cells) even after being told not to. After
                # 6 history entries, fall back to heuristic rather than loop.
                if len(history) >= 6 and attempt >= 5:
                    self.log.log("vision_drive", ok=False,
                                 reason="heuristic_fallback_repeat",
                                 attempts=attempt + 1, rejected=len(rejected))
                    return self.fallback.next_move(board, self._frame)
                continue
                # don't accept `idle` immediately after a rejection. The
            # correction message already tells the model what was wrong; the
            # model has up to `max_retries` chances. If its next attempt is
            # `idle` (giving up), it's a wasted attempt — push the correction
            # back and let the loop continue. Without this, the model emits
            # `idle` on attempt 1 after a `feed_mana_overflow` rejection and
            # sits for the rest of the step — the board stalls even though
            # the validator has `max_retries - 1` more chances. A non-idle
            # action on attempt 2 is the desired outcome. The user's view:
            # "the reject move already tells the model to not suggest that
            # move again so option A is sufficient" — we just need to make
            # sure `idle` doesn't become a way to bypass the rejection
            # before the loop exhausts.
            if move.kind == "idle" and rejected:
                self.log.log("vision_drive", ok=False, reason="premature_idle",
                             attempts=attempt + 1,
                             rejected=[k[0] for k in rejected])
                correction = ("Your previous move was rejected. Do NOT respond "
                              "with `idle` — try a DIFFERENT action instead "
                              "(merge two identical items, spawn from a "
                              "chest or grave, feed a different cell, or "
                              "collect from the NecroMerger). Look at the "
                              "screenshot again and pick a move that doesn't "
                              "trigger the same rejection. The validator has "
                              "more retries available — `idle` is only the "
                              "right answer when the board has no valid moves "
                              "left at all.")
                history.append((reply, correction))
                continue
            fb = self.fallback
            reason = LLMPlanner._validate(
                board, move, self._mana_fraction, self._max_level_ids(),
                feed_values=fb.feed_values,
                damage_values=fb.damage_values,
                satiety_remaining=fb.satiety_remaining,
                satiety_capacity=fb.satiety_capacity,
                craved_item=fb.craved_item, craved_level=fb.craved_level,
                craving_bonus=fb.craving_bonus_est,
                craving_need=self._craving_need_remaining(),
                prefer_item=fb.feed_prefer_item,
                chain_map=self._chain_map(),
                protect_family=self._strategy_family,
                slime_count=getattr(fb, "slime_count", None))
            if reason is None:
                # A successful move DOES NOT wipe the cross-step rejection
                # memory. Previously we cleared the whole list here ("board
                # shifted, a stale rejection shouldn't block forever"), but
                # that erased EVERY rejection — including the one the model
                # just hit this step — so the very next step re-attempted the
                # same illegal move (the recurring feed-icerune loop). The
                # TTL filter and the item-shift prune at the top of the next
                # step handle staleness correctly: a rejection stays blocked
                # only while the SAME item still occupies the rejected
                # cell(s), and lapses if the board moves on.
                self.log.log("vision_drive", ok=True, attempt=attempt, action=move.kind,
                             cell_a=move.cell_a, cell_b=move.cell_b)
                return move
            if (self._auto_identify_unidentified(board, move)
                    and LLMPlanner._validate(
                        board, move, self._mana_fraction,
                        self._max_level_ids(),
                        feed_values=fb.feed_values,
                        damage_values=fb.damage_values,
                        satiety_remaining=fb.satiety_remaining,
                        satiety_capacity=fb.satiety_capacity,
                        craved_item=fb.craved_item,
                        craved_level=fb.craved_level,
                        craving_bonus=fb.craving_bonus_est,
                        craving_need=self._craving_need_remaining(),
                        prefer_item=fb.feed_prefer_item,
                        chain_map=self._chain_map(),
                        protect_family=self._strategy_family,
                        slime_count=getattr(fb, "slime_count", None)) is None):
                        # successful identify-then-validate path — leave the
                # cross-step rejection memory intact (see the note on the
                # plain-validate success above; the item-shift + TTL prune
                # clears stale entries).
                self.log.log("vision_drive", ok=True, attempt=attempt, action=move.kind,
                             cell_a=move.cell_a, cell_b=move.cell_b, resolved="identify")
                return move
            rejected.add(key)
            nudge = self._loop_nudge(board, rejected) if len(rejected) >= 5 else ""
            correction = self._correction(reason, rejected, nudge=nudge)
            history.append((reply, correction))
            # record the rejection for cross-step carry-over. Every
            # rejected (kind, cell_a, cell_b, reason) of the current step is
            # stashed in `self._cross_step_rejected` so the next step's
            # attempt 0 can't re-emit ANY of them (storing only the last
            # left a hole: the model would just shift to the second-to-last
            # and re-emit that). The reason and step number are stored so
            # the get_board_state line can summarize them and the TTL
            # filter can expire stale entries. Same action can be
            # duplicated — `dict` keeps insertion order and we use a list,
            # so a repeated rejection just extends the list (the dedup
            # happens via the `rejected` set inside the same step).
            # Store the item at the acted-on cell(s) too, so the next
            # step can tell whether the board has shifted there (a
            # rejection should only stay blocked while the SAME item is
            # still in that cell; if the cell empties or changes, the
            # rejection lapses and the move may be re-considered).
            item_a = (board.cell_at(*move.cell_a).item_id
                      if board is not None and board.cell_at(*move.cell_a) else None)
            item_b = None
            if move.cell_b is not None and board is not None:
                cb = board.cell_at(*move.cell_b)
                item_b = cb.item_id if cb else None
            self._cross_step_rejected.append({
                "kind": move.kind,
                "cell_a": move.cell_a,
                "cell_b": move.cell_b,
                "reason": reason,
                "step": self._step_count,
                "item_a": item_a,
                "item_b": item_b,
            })
            # Log the rejection as an event so (a) `_session_transcript`
            # surfaces the reason to the summary model and (b)
            # `_rejection_reasons_to_learnings` can deterministically fold
            # the ground-truth reason into learnings.md on the next
            # summarize (the meta-noise filter would otherwise strip any
            # summary-model learning containing "rejected"/"validate").
            self.log.log("rejection", reason=reason, action=move.kind,
                         cell_a=move.cell_a, cell_b=move.cell_b)
            # 3-strikes for general validation loops: if the model has
            # burned many retries without finding a valid move and the
            # nudge has no better hint (or the rejected set is huge),
            # fall back to heuristic rather than burning the remaining
            # 40+ retries. The model demonstrably ignores generic
            # corrections after ~5 failures (feed ×47, get_board_state ×50).
            if len(history) >= 8 or len(rejected) >= 8:
                if not nudge or len(rejected) >= 10:
                    self.log.log("vision_drive", ok=False,
                                 reason="heuristic_fallback_validation",
                                 attempts=attempt + 1, rejected=len(rejected))
                    return self.fallback.next_move(board, self._frame)
        raise ValueError(f"vision_drive: no valid move after {self.max_retries + 1} attempts")

    def _observe(self, messages: list[dict], board) -> None:
        """Mandatory get_board_state round; appends tool call + result to messages.

        Uses MINIMAL prompt messages (not the full conversation): probed
        live, this model emits a perfect native tool call for a short
        direct command, but burns its whole budget thinking (empty reply)
        when the same required call follows the ~25k-char system prompt +
        history. get_board_state takes no arguments, so no context is
        needed at all — the result is appended to the real `messages`.

        Heartbeat skip: the tool takes no arguments and the result text is
        fully deterministic, so most steps inject the render directly with
        no LLM round (~20-50s saved per step). A live call runs on heartbeat
        steps to keep the native channel warm and catch renderer drift.
        """
        if not self.tool_enabled:
            return
        if self._step_count % OBSERVE_HEARTBEAT_STEPS != 1:
            try:
                text = self._board_state_text(board)
            except Exception as exc:
                self.log.log("vision_drive", ok=False,
                             reason="board_text_failed", error=str(exc))
                return
            tc = {"type": "function",
                  "id": f"direct_{self._step_count}",
                  "function": {"name": "get_board_state", "arguments": "{}"}}
            messages.append(self._tool_call_msg(tc))
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": text})
            self.log.log("vision_drive", ok=True, reason="observe_heartbeat_skip")
            return
        # Retry once: the reasoning model can exhaust a small generation budget
        # on thinking and then emit JSON (or nothing) instead of the tool call.
        # NOTE: the retry nudge is scoped to a TEMPORARY copy of messages.
        # Appending "You must call get_board_state ..." to the shared list
        # leaks it into every later answer round (_ask_once strips only
        # tool-role messages), where it directly contradicts the answer
        # instruction ("reply with ONLY a JSON action, NOT a tool call") and
        # primes the {"action":"get_board_state"} fixation.
        observe_msgs = [
            {"role": "user", "content": (
                "Call get_board_state now. Emit ONLY the tool call.")},
        ]
        for attempt in range(2):
            attempt_msgs = (observe_msgs if attempt == 0
                            else observe_msgs + [{"role": "user", "content": (
                                "You must call get_board_state now — reply with "
                                "a tool call, not JSON text.")}])
            content, tool_calls, full_msg = self.client.chat_message(
                attempt_msgs, max_tokens=256, tools=[BOARD_TOOL],
                tool_choice="required")
            self._log_chat(attempt_msgs, content, tool_calls, full_msg)
            if tool_calls:
                break
            # Some models (e.g. Gemma GGUFs) reply to tool_choice="required"
            # with the tool as plain JSON text -- {"action":"get_board_state"}
            # -- instead of a native tool_calls message. Synthesize the call
            # from the text so the board state still reaches the model.
            if (synth := _synthesize_board_call(content)) is not None:
                self.log.log("vision_drive", ok=True, reason="synth_tool_call")
                tool_calls = [synth]
                break
            self.log.log("vision_drive", ok=False, reason="no_tool_call", reply=content)
        else:
            # Both tool rounds failed (model emitted JSON text or nothing
            # instead of a tool call). Fall back to the DETERMINISTIC
            # classifier render injected as a synthetic tool result — the
            # board text, whitelist, and Best hints are code-owned and need
            # no LLM. Without this the answer round decides from the
            # screenshot alone: no whitelist arrives, and the model proposes
            # blind moves (observed: spawn ×5 on non-graves, then a
            # nudge-quoted max-level merge the game no-ops). _ask_once folds
            # role=="tool" contents into "Tool results", so this flows
            # through the exact same path as a real tool result.
            try:
                text = self._board_state_text(board)
            except Exception as exc:
                self.log.log("vision_drive", ok=False,
                             reason="board_text_failed", error=str(exc))
                return
            self.log.log("vision_drive", ok=False,
                         reason="no_tool_call_fallback_board_text")
            messages.append({"role": "tool", "tool_call_id": "synth_board_state",
                             "content": text})
            return
        for tc in tool_calls:
            messages.append(self._tool_call_msg(tc))
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": self._exec_tool(tc, board)})

    def _compel_buy(self, messages: list[dict], board, family: str,
                    reason: str, affordable: bool = False,
                    confirm: bool = False) -> None:
        """Force a `buy_station(family, confirm=…)` dialog round.

        Uses MINIMAL prompt messages naming the family explicitly: probed
        live, this model emits a perfect native tool call for a short
        direct command, but burns its budget thinking (empty reply) when
        the same required call follows the full conversation. Appends the
        tool call + result to `messages` like any other tool round.
        `confirm=True` completes a staged pending buy; False only reads.
        """
        compel_msgs = [
            {"role": "user", "content": (
                f'Call buy_station with family "{display_family(family)}" '
                f"and confirm={'true' if confirm else 'false'}. "
                "Emit ONLY the tool call.")},
        ]
        content, tool_calls, full_msg = self.client.chat_message(
            compel_msgs, max_tokens=1024, tools=[BUY_TOOL_SHORT],
            tool_choice="required")
        self._log_chat(compel_msgs, content, tool_calls, full_msg)
        self.log.log("vision_drive", ok=True, reason=reason,
                     family=family, affordable=affordable)
        fired = False
        if tool_calls:
            fired = True
            for tc in tool_calls:
                messages.append(self._tool_call_msg(tc))
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": self._exec_tool(tc, board)})
        else:
            synth = _synthesize_buy_call(content, family=family)
            if synth is not None:
                # Pin confirm to the compelled value: the text default is
                # True, which would COMPLETE a purchase the compel only
                # meant to read. Parse first, then override when the text
                # didn't explicitly say confirm=true.
                try:
                    sargs = json.loads(synth["function"]["arguments"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    sargs = {}
                if confirm or "confirm" not in (sargs or {}):
                    sargs["confirm"] = confirm
                    synth["function"]["arguments"] = json.dumps(sargs)
                fired = True
                self.log.log("vision_drive", ok=True, reason="synth_buy_compel")
                messages.append(self._tool_call_msg(synth))
                messages.append({"role": "tool", "tool_call_id": synth["id"],
                                 "content": self._exec_tool(synth, board)})
        # No-effect backoff: this thinking model sometimes emits empty text
        # for required calls (probed: compounds like "supplycupboard" derail
        # it while "grave" lands). A compel that fires every step with no
        # resulting buy burns a full LLM round each time — after 3
        # consecutive no-effect compels, stand down until something changes
        # (a successful buy resets the count).
        if fired:
            self._compel_misfires.pop(family, None)
        else:
            self._compel_misfires[family] = self._compel_misfires.get(family, 0) + 1
            if self._compel_misfires[family] >= 3:
                self.log.log("vision_drive", ok=False,
                             reason="compel_backoff", family=family)

    def _compel_cravings_read(self, messages: list[dict], board,
                                bubble: str) -> None:
        """Force a `get_cravings` menu-read round.

        Same minimal-prompt required-call pattern as `_compel_buy`: the
        model reliably emits the tool call for a short direct command.
        Unlike buy there is no text synth fallback — a text reply can't
        replace a menu cycle — so a no-call round only logs a misfire
        (3 strikes stand down, same key family as the buy backoff).
        Firing appends the call + result to `messages`, so the menu's
        level/count land in context and `_track_cravings` caches them via
        the normal `_exec_tool` path.
        """
        compel_msgs = [
            {"role": "user", "content": (
                "Call get_cravings to read the current Devourer craving "
                "(item, level, progress, reward). Emit ONLY the tool call.")},
        ]
        content, tool_calls, full_msg = self.client.chat_message(
            compel_msgs, max_tokens=1024, tools=[CRAVINGS_TOOL],
            tool_choice="required")
        self._log_chat(compel_msgs, content, tool_calls, full_msg)
        self.log.log("vision_drive", ok=True,
                     reason="compelled_cravings_read", bubble=bubble)
        if tool_calls:
            for tc in tool_calls:
                messages.append(self._tool_call_msg(tc))
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": self._exec_tool(tc, board)})
            self._compel_misfires.pop("cravings", None)
        else:
            self._compel_misfires["cravings"] = \
                self._compel_misfires.get("cravings", 0) + 1
            if self._compel_misfires["cravings"] >= 3:
                self.log.log("vision_drive", ok=False,
                             reason="compel_backoff", family="cravings")

    def _optional_tools(self, messages: list[dict], board) -> tuple[str, Move] | None:
        """ONE consolidated optional-tool round for the whole step.

        Replaces the previous four separate rounds (`_cravings`, `_panels`,
        `_discover`, `_factcheck`) that each made their own LLM call every
        step — a step before the answer used up to 5 API calls. Here the tool
        list is built from what's stale/relevant RIGHT NOW (craving cache
        aged out, feats cache aged out, unidentified cells present, wiki
        enabled) and the model can chain any of them in ONE bounded round.
        Per-tool budgets are preserved (a hungry model can't run the
        intrusive get_cravings/tap_button device cycles more than before).

        Returns `(content, move)` when the model answered with a valid text
        action JSON instead of calling a tool (observed ~50% of the time in
        the Aug 19 soak — those answers were previously DISCARDED and the
        whole decision was made again in `_ask_once`). The caller consumes
        it as attempt 0 through the same validation gate, saving an LLM
        round when it's valid. Returns None when the model called a tool
        (or skipped the round entirely).
        """
        if not self.live or not self.tool_enabled:
            return None
        # Planning phase: when the strategy slot is stale and feats are known,
        # compel the choice once per window (required tool call, model fills
        # the args) — an auto-choice 4B model never calls optional planning
        # tools on its own (observed 16/16 offered, 0 called). Agency lives in
        # WHICH feat/target it picks, not in whether it plans.
        if (self._feats_cache or {}).get("feats") and not self._strategy_fresh():
            content, tool_calls, full_msg = self.client.chat_message(
                _strip_images(messages), max_tokens=512, tools=[STRATEGY_TOOL],
                tool_choice="required")
            self._log_chat(messages, content, tool_calls, full_msg)
            if tool_calls:
                for tc in tool_calls:
                    messages.append(self._tool_call_msg(tc))
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": self._exec_tool(tc, board)})
                                     # compelled buy follow-through. After the model called
        # `buy_station(family, confirm=false)` in a prior step, the next
        # step MUST resolve the pending buy — either `confirm=true` to
        # complete, or call it again with `confirm=true` for a DIFFERENT
        # family to explicitly cancel. Without this, the model was
        # observed opening the Station dialog and then forgetting to
        # confirm (the action queue went to merge/feed instead), and the
        # pending_buy expired after 5 steps with no purchase — a
        # 1-step 'open then close' pattern that wasted a turn's worth of
        # rune-balancing. The board state still carries the reminder
        # text, but the LLM doesn't reliably act on it; the forced tool
        # call closes the loop deterministically.
        # proxy through self.shop — StationShop is the single owner
        # of _pending_buy (set inside StationShop.buy when confirm=false
        # stages the dialog). The planner's _pending_buy is always None
        # without this proxy, so the compelled gate was never firing.
        if (self.shop is not None and self.shop._pending_buy
                and self.live and self.tool_enabled):
            self._compel_buy(messages, board,
                             self.shop._pending_buy.get("family"),
                             reason="compelled_buy_followthrough",
                             confirm=True)
                                     # buy_station compel. The model has a station-build strategy
        # committed but never calls `buy_station(confirm=false)` on its own
        # — observed in 4 consecutive live sessions: the strategy note
        # told the model "call buy_station(\"grave\", confirm=false) to
        # check affordability" but the model went straight to merge/spawn.
        # Without a starting buy, the dialog never opens, the pending
        # follow-through has nothing to compel, and the bot loops forever
        # doing prerequisite work (merging ribs, feeding runes) without
        # actually buying. The fix: when a station-build strategy is fresh
        # and no pending buy exists, force a `buy_station(confirm=false)`
        # tool call so the dialog-read happens. The post-loop
        # compelled-confirm then forces the matching `confirm=true` once
        # the loop ends.
        #
        # Two cases:
        # 1. cost_cache has the strategy's family + currency cache has a
        #    balance AND the balance covers the cost: compel a buy. The
        #    dialog stages the pending_buy; the post-loop compel confirms.
        # 2. cost_cache MISSING the family (never read the Station panel):
        #    ALSO compel a buy — the buy itself populates the cost_cache
        #    and _currency. This is the cold-start path: the model has
        #    never opened the panel, so it has no idea what the cost is.
        #    The first buy reads the cost, and the next step's compel
        #    either fires (affordable) or skips (still short).
        if (self.live and self.tool_enabled
                and self.shop is not None
                and self.shop._pending_buy is None
                and self._strategy_fresh()
                and self._strategy
                and self._strategy.get("kind") == "station"
                and self._compel_misfires.get(self._strategy.get("noun"), 0) < 3
                and (self._strategy_affordable()
                     or not (self.shop.cost_cache or {}).get(
                         self._strategy.get("noun")))):
            self._compel_buy(messages, board, self._strategy.get("noun"),
                             reason="compelled_buy_stage",
                             affordable=self._strategy_affordable())
        # Craving-producer buy compel. When the craving cannot be satisfied
        # from the board (no item, no producer station — see
        # `_craving_station_need`), the model will never buy the station on
        # its own: nothing connects the craving to the Station panel. Force
        # the same `buy_station(confirm=false)` dialog-read the strategy
        # compel forces, so cost/currency populate and the post-loop
        # compelled-confirm can complete the purchase. Skipped when the
        # strategy compel already staged a buy (single pending slot), when a
        # conflicting station strategy is active (the buy guard would refuse
        # with strategy_mismatch), and under the same affordable-or-unknown
        # gate so an unaffordable known cost doesn't reopen the panel.
        craving_fam = self._craving_station_need(board)
        # Throttle: a previous full scan that missed the family (locked /
        # unreleased station) suppresses re-scans for CARD_RESCAN_SECONDS —
        # otherwise every step reopens the panel and re-swipes the sheet.
        # Stronger gate first: a no-card miss recorded at the CURRENT feats
        # tier while locked slots exist means the station is unlock-gated
        # (observed: supply cupboard behind "Tier 5 Feats Requires") —
        # rescanning before the tier moves cannot succeed, so skip outright.
        craving_scan_due = True
        if craving_fam and self.shop is not None:
            try:
                from vision.panels import CARD_RESCAN_SECONDS
                last_miss = (self.shop._card_missing or {}).get(craving_fam, 0)
                craving_scan_due = (time.time() - last_miss) >= CARD_RESCAN_SECONDS
                last = self._last_currency_result or {}
                # Locked stand-down ONLY on fresh absence evidence: the
                # family must still be missing from the latest scan
                # (_card_missing set by a full scan, cleared on any find).
                # A stale miss record alone must never suppress a buy —
                # the sheet may have changed since (observed: cupboard
                # present but an old miss + unchanged tier stood the
                # compel down).
                if (craving_fam in (self.shop._card_missing or {})
                        and isinstance(last, dict)
                        and last.get("family") == craving_fam
                        and (last.get("error") or "").startswith("no ")
                        and last.get("locked")
                        and last.get("_tier_at_miss") is not None
                        and last.get("_tier_at_miss") == (self._feats_cache or {}).get("tier")):
                    craving_scan_due = False
                    self.log.log("vision_drive", ok=False,
                                 reason="craving_buy_locked",
                                 family=craving_fam,
                                 requires=last.get("locked"))
            except Exception:
                craving_scan_due = True
        if (craving_fam and craving_scan_due and self.live and self.tool_enabled
                and self.shop is not None
                and self.shop._pending_buy is None
                and self._compel_misfires.get(craving_fam, 0) < 3
                and not (self._strategy_family and self._strategy_family != craving_fam)
                and (self._family_affordable(craving_fam)
                     or not (self.shop.cost_cache or {}).get(craving_fam))):
            self._compel_buy(messages, board, craving_fam,
                             reason="compelled_craving_buy_stage",
                             affordable=self._family_affordable(craving_fam))
        # Compelled cravings read (Sep 7). The merge guard below can only
        # protect the craved level once the menu's level/count are known,
        # but the model never takes the optional get_cravings tool on its
        # own (observed: zero calls in a full run that then merged two
        # eyemonster_lvl1 into a lvl2 for a lvl1 craving). Force one
        # minimal required read when the bubble shows a craving the cache
        # doesn't know (never read, or switched since the last read) —
        # same forced-call pattern as the buy compels, same 3-strike
        # backoff. A successful read populates the cache via _exec_tool,
        # so this fires at most once per craving.
        try:
            _bubble = self._bubble_craving_item()
            _cache_item = (self._craving_cache or {}).get("item")
            if (_bubble and self.live and self.tool_enabled
                    and self.cravings is not None
                    and _cache_item != _bubble
                    and self._compel_misfires.get("cravings", 0) < 3):
                self._compel_cravings_read(messages, board, _bubble)
        except Exception:
            pass
        budgets = {"get_cravings": MAX_CRAVINGS,
                   "tap_button": MAX_PANELS,
                   "get_champions": MAX_CHAMPIONS,
                   "identify_item": MAX_DISCOVERY,
                   "lookup_wiki": MAX_WIKI,
                   "set_strategy": MAX_STRATEGY,
                   "buy_station": MAX_BUYS,
                   "collect_queue": MAX_QUEUE,
                   "collect_feat_rewards": MAX_FEAT_COLLECT}
        tool_specs = {
            "get_cravings": CRAVINGS_TOOL,
            "tap_button": PANEL_TOOL,
            "get_champions": CHAMPION_TOOL,
            "identify_item": IDENTIFY_TOOL,
            "lookup_wiki": WIKI_TOOL,
            "set_strategy": STRATEGY_TOOL,
            "buy_station": BUY_TOOL,
            "collect_queue": QUEUE_TOOL,
            "collect_feat_rewards": FEAT_COLLECT_TOOL,
        }
        # track the pending buy as we entered the optional-tools
        # loop, so the post-loop compelled-confirm knows whether a NEW buy
        # was staged THIS step (the prior `compelled buy follow-through`
        # only checked for a PREVIOUS-step pending, missing the case where
        # the model called buy_station(confirm=false) inside the loop
        # itself and then moved on to the answer round).
        pending_before = (
            self.shop._pending_buy if self.shop is not None else None)
        for _ in range(sum(budgets.values()) or 1):
            tools = []
            for name in budgets:
                if budgets[name] <= 0:
                    continue
                if name == "get_cravings":
                    if self.cravings is None or self._craving_cache_fresh():
                        continue
                elif name == "tap_button":
                    if self.panels is None or self._feats_cache_fresh():
                        continue
                elif name == "get_champions":
                    if self.champions is None or self._champion_cache_fresh():
                        continue
                elif name == "identify_item":
                    if self.discover is None or not self._unidentified_cells(board):
                        continue
                elif name == "lookup_wiki":
                    if not self.wiki_tool:
                        continue
                elif name == "set_strategy":
                    # Only meaningful once feats are known and the committed
                    # strategy (if any) has expired or its feat disappeared.
                    if not (self._feats_cache or {}).get("feats") \
                            or self._strategy_fresh():
                        continue
                elif name == "buy_station":
                    if self.shop is None:
                        continue
                elif name == "collect_queue":
                    if self.queue_box is None:
                        continue
                elif name == "collect_feat_rewards":
                    if self.panels is None:
                        continue
                    # Offer even when the read cache is fresh — the point is
                    # to let the model force an immediate collect when it
                    # believes a feat just completed, without waiting for the
                    # 15-step periodic read. Gate only on live/tool_enabled.
                    if not self.live or not self.tool_enabled:
                        continue
                tools.append(tool_specs[name])
            if not tools:
                return None
            self.log.log("optional_tools", names=[t["function"]["name"] for t in tools])
            # Batching nudge (scoped copy — never leaks into answer rounds):
            # one LLM round per tool is the worst case today; the loop below
            # already executes every returned call, so asking for all
            # independent calls up front collapses N rounds into one.
            round_msgs = messages
            if len(tools) > 1:
                round_msgs = messages + [
                    {"role": "user", "content": (
                        "If you need several of these, call them ALL in this "
                        "round (multiple tool calls at once) instead of one "
                        "per round.")}]
            content, tool_calls, full_msg = self.client.chat_message(
                _strip_images(round_msgs), max_tokens=512, tools=tools,
                tool_choice="auto")
            self._log_chat(round_msgs, content, tool_calls, full_msg)
            if not tool_calls:
                # No tool called — either the model skipped optional tools or
                # it answered with a text action JSON. Return the parsed move
                # (if any) so the caller can reuse it as attempt 0 instead of
                # discarding a full LLM round.
                move = self._parse_action(content)
                if move is not None:
                    return content, move
                # Text-JSON buy attempt: this model often answers required/
                # voluntary tool prompts with plain JSON text instead of a
                # native call (probed: compounds derail native generation).
                # Honor an explicit buy_station text call within the buy
                # budget rather than dropping the round.
                if self.shop is not None and budgets.get("buy_station", 0) > 0:
                    synth = _synthesize_buy_call(content, family="")
                    if synth is not None:
                        try:
                            fam = json.loads(synth["function"]["arguments"]).get("family")
                        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                            fam = None
                        if fam:
                            self.log.log("vision_drive", ok=True,
                                         reason="synth_buy_optional")
                            budgets["buy_station"] -= 1
                            messages.append(self._tool_call_msg(synth))
                            messages.append({"role": "tool",
                                             "tool_call_id": synth["id"],
                                             "content": self._exec_tool(synth, board)})
                            continue
                return None
            for tc in tool_calls:
                nm = tc.get("function", {}).get("name")
                if nm in budgets:
                    budgets[nm] -= 1
                messages.append(self._tool_call_msg(tc))
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": self._exec_tool(tc, board)})
                                 # post-loop compelled-confirm. If a buy was staged THIS step
        # (i.e. by the model calling buy_station(confirm=false) inside the
        # loop), the model is now about to enter the answer round and pick
        # a different action. Without this gate, the dialog stays open and
        # the pending_buy expires after PENDING_BUY_TTL — observed in the
        # 3 strategy_set + 3 station_buy(confirm=false) events
        # with zero confirm=true follow-ups, the model always went to
        # `spawn` or `merge` next. The fix: when the loop ends AND there's
        # a pending buy that was staged inside the loop, force a
        # confirm=true tool call so the purchase lands BEFORE the answer
        # round. The pre-loop compelled-confirm (above) still handles the
        # cross-step case from a prior run.
        pending_after = (
            self.shop._pending_buy if self.shop is not None else None)
        if (pending_after is not None
                and pending_before is None
                and self.live and self.tool_enabled):
            self._compel_buy(messages, board, pending_after.get("family"),
                             reason="compelled_buy_post_loop",
                             confirm=True)
        return None

    def _craving_cache_fresh(self) -> bool:
        """True when the last full cravings menu read is fresh enough that the
        model doesn't need the (intrusive) get_cravings tool this step.

        Freshness key covers item AND level: the game can switch a craving
        within the same family to a DIFFERENT level (a finished skeleton_lvl1
        craving is replaced by skeleton_lvl2), and the bubble cue carries only
        the base name — so an item-only key would keep believing the stale
        level. A level mismatch (`cache.level != self._last_craving_level`)
        forces a re-read even inside the refresh window."""
        cache = self._craving_cache
        return bool(cache is not None and cache.get("item") == self._last_craving
                    and cache.get("level") == self._last_craving_level
                    and self._step_count - cache.get("step", 0) < CRAVINGS_REFRESH_STEPS)

    def _craving_need_remaining(self) -> int | None:
        """Remaining craving count (required - done) from the last menu read.

        None when no menu read has landed (level/count unknown — the merge
        guard must not fire on bubble-only knowledge) or the counts are
        unparseable. Zero means complete-but-stale (guard allows merges)."""
        cache = self._craving_cache
        if not cache:
            return None
        try:
            req = int(cache.get("count_required"))
            done = int(cache.get("count_done") or 0)
        except (TypeError, ValueError):
            return None
        if req <= 0:
            return None
        return max(0, req - done)

    def note_identified(self, pos, item_id: str | None) -> None:
        """Record a popup-resolved cell label for persistence.

        Called for every successful popup identify (proactive UNID reads,
        noop re-reads). The label is ground truth until the cell genuinely
        changes — see `apply_label_memory`. Stamped with the planner's own
        step counter; bounded to 64 entries (oldest evicted).
        """
        if not item_id or not pos:
            return
        try:
            key = (int(pos[0]), int(pos[1]))
        except (TypeError, ValueError, IndexError):
            return
        self._label_memory[key] = (item_id, self._step_count)
        if len(self._label_memory) > 64:
            oldest = min(self._label_memory,
                         key=lambda k: self._label_memory[k][1])
            del self._label_memory[oldest]

    def apply_label_memory(self, board, prev_move=None) -> int:
        """Re-apply popup-resolved labels over template flip-flop. Returns count applied.

        Motivation (Sep 7): eyeball vs eyemonster_lvl1 are
        template-indistinguishable (cross-bank matches ~1.0 both ways), so a
        popup-resolved lvl1 flips back to eyeball next classify and the
        craving guard/feed never see it — and no bank growth can fix
        identical sprites. The popup is ground truth: re-apply the recorded
        label when the template disagrees *inside the ambiguous pair* (or
        reads UNID). Score/margin stay the template's (honest match
        quality); only the id is overridden.
        Safety: entries expire after LABEL_MEMORY_TTL steps; cells touched
        by our last move, emptied cells, and cells whose template label
        left the pair (genuine change — e.g. a champion arriving) drop the
        entry instead of overriding. Never raises (all failures return 0).
        """
        try:
            touched = set()
            if prev_move is not None:
                for c in (prev_move.cell_a, prev_move.cell_b,
                          getattr(prev_move, "target", None)):
                    if c:
                        try:
                            touched.add((int(c[0]), int(c[1])))
                        except (TypeError, ValueError, IndexError):
                            pass
            applied = 0
            for cell in board.cells:
                key = (cell.row, cell.col)
                mem = self._label_memory.get(key)
                if mem is None:
                    continue
                item_id, at_step = mem
                if self._step_count - at_step > LABEL_MEMORY_TTL:
                    del self._label_memory[key]
                    continue
                if key in touched or not cell.occupied:
                    if key in self._label_memory:
                        del self._label_memory[key]
                    continue
                if cell.item_id == item_id:
                    continue
                if (cell.item_id is not None
                        and (cell.item_id not in LABEL_MEMORY_PAIR
                             or item_id not in LABEL_MEMORY_PAIR)):
                    del self._label_memory[key]
                    continue
                was = cell.item_id
                cell.item_id = item_id
                applied += 1
                try:
                    self.log.log("label_memory_applied",
                                 cell=[cell.row, cell.col],
                                 was=was, now=item_id)
                except Exception:
                    pass
            return applied
        except Exception:
            return 0

    def _feats_cache_fresh(self) -> bool:
        """True when the cached feats-panel read is fresh enough that the model
        doesn't need the (intrusive) tap_button tool this step — the feats
        content is already rendered into `get_board_state`."""
        cache = self._feats_cache
        return bool(cache is not None
                    and self._step_count - cache.get("step", 0) < FEATS_REFRESH_STEPS)

    def _champion_cache_fresh(self) -> bool:
        """True when the last full champion-screen read is fresh enough that the
        model doesn't need the (intrusive) get_champions tool this step."""
        cache = self._champion_cache
        return bool(cache is not None
                    and self._step_count - cache.get("step", 0) < CHAMPIONS_REFRESH_STEPS)

    def _strategy_fresh(self) -> bool:
        """True when the model's committed strategy is still in force: within
        its refresh window AND its feat is still active in the cached FEATS
        read (a completed/removed feat invalidates the strategy so the tool is
        re-offered and the model picks the next objective)."""
        s = self._strategy
        if s is None:
            return False
        if self._step_count - s.get("step", 0) >= STRATEGY_REFRESH_STEPS:
            return False
        feats = (self._feats_cache or {}).get("feats") or []
        if feats and not any(
                s["feat"].lower() in (f.get("name") or "").lower()
                or (f.get("name") or "").lower() in s["feat"].lower()
                for f in feats):
            return False
        return True

    def _family_affordable(self, family: str) -> bool:
        """True when `family`'s station is affordable under the cached Rune
        balance. Cost from `shop.cost_cache`, balance from `_currency` (both
        populated by the most recent Station-panel read). Missing cost or
        balance -> False (the call would be a guess)."""
        if self.shop is None or not family:
            return False
        cost = (self.shop.cost_cache or {}).get(family)
        if not cost:
            return False
        currency = self._currency or {}
        if not currency:
            return False
        for rune, need in cost.items():
            if need and need > 0:
                if currency.get(rune, 0) < need:
                    return False
        return True

    def _strategy_affordable(self) -> bool:
        """True when the active station-build strategy is affordable
        under the cached Rune balance.

        The buy_station compel calls `buy_station(confirm=false)` ONLY when
        the cached currency can cover the cost — otherwise we'd be opening
        the Station panel every step just to read "unaffordable: needs
        20 ice; have 5 ice" and BACK out. See `_family_affordable`."""
        if self.shop is None:
            return False
        s = self._strategy or {}
        noun = s.get("noun")
        if not noun:
            return False
        return self._family_affordable(noun)

    def _want_buy_this_step(self, board) -> bool:
        """True when a buy_station panel visit is likely this step (pure
        cache reads, no device). Mirrors the three buy gates (pending
        follow-through, station-strategy compel, craving compel) minus
        their misfire/scan throttles — used to defer scheduled FEATS
        reads so a buy step costs one panel cycle, not two. Slight
        over-prediction (gate throttles later) only costs a deferred
        feats read, which TTLs tolerate.
        """
        try:
            if self.shop is None:
                return False
            if self.shop._pending_buy is not None:
                return True
            if (self._strategy_fresh() and self._strategy
                    and self._strategy.get("kind") == "station"
                    and self._strategy.get("noun")):
                return True
            if self._craving_station_need(board):
                return True
        except Exception:
            pass
        return False

    def _popup_next(self, board) -> tuple[int, int, str] | None:
        """Find a known item whose popup-body recipe isn't banked yet.

        Merge/spawn-produced items (e.g. icerune_lvl3 after merging two
        icerune_lvl2) appear with a template label but no glossary recipe, so
        the model knows the id but not whether it maxes out (feed it) or keeps
        merging. Pick the first such cell (occupied, labelled at/above
        LABEL_MIN_SCORE, real bank id, not a champion/necromerger) and
        popup-read its body. Returns (row, col, item_id) or None.

        stations (grave, manapool, icebox, etc.) ARE eligible for
        popup-reads. The previous skip of STATION_PREFIXES meant the bot
        never banked per-level grave/manapool info (spawn rates, build
        costs, max-level flag), so the `Best spawn` hint always used the
        hardcoded `GRAVE_SPAWN_PREFIXES = ("bone", "ribcage")` for any
        grave level — it never knew about grave_lvl3's zombie output. We
        now let the popup-reader through; the necromerger (the Lair NPC)
        stays skipped because tapping it does not open a useful info popup
        in the same way.

        If every known item already has a popup block, fall back to an item
        whose block LACKS a recorded feed value or damage value (an item banked
        before those stats existed) so the stat gets backfilled one per
        `_auto_read` pass.

        if no candidate is found by the above rules, fall
        back to an UNID occupied cell (classifier failed but cell is full).
        Such a cell may hold a brand-new item the template bank doesn't have
        yet (e.g. skeleton_lvl6 before the user captured its template, a
        Champion before peasant/knight templates were banked). The
        popup-reader can still identify the item from the popup title and
        bank the missing data — `_auto_read_popup_unid` handles the
        sprite+recipe banking after a successful popup read.
        """
        unbanked_known = None
        for cell in board.cells:
            if (cell.row, cell.col) == necromerger_cell():
                continue
            if not cell.item_id or not cell.occupied or cell.score < LABEL_MIN_SCORE:
                continue
            if self.classifier is not None and not self.classifier.has(cell.item_id):
                continue
                # skip the necromerger (it's the Lair NPC, not a
            # buildable station) and champions. ALL other stations
            # (grave, manapool, icebox) are eligible — popup-reading them
            # banks spawn rates / build costs / max-level flags.
            if cell.item_id.startswith(CHAMPION_PREFIXES):
                continue
            if cell.item_id == "necromerger" or cell.item_id.startswith("necromerger_"):
                continue
            if self._popup_banked(cell.item_id):
                if unbanked_known is None and (
                        not self._popup_has_fact(cell.item_id, "feed value")
                        or not self._popup_has_fact(cell.item_id, "damage value")):
                    unbanked_known = (cell.row, cell.col, cell.item_id)
                continue
            return cell.row, cell.col, cell.item_id
        if unbanked_known is not None:
            return unbanked_known
            # UNID-cell fallback. A non-empty cell with item_id=None is a
        # brand-new or animation-phase sprite the bank can't match. Pop it
        # and learn from the popup (title bar OCRs the name + level). Only
        # fire when no banked item needs backfill — backfilling the bank is
        # more valuable than identifying one new item, but this is the only
        # way to recover when the bank has no candidate at all.
        unid_cell = self._unid_occupied_cell(board)
        if unid_cell is not None:
            r, c = unid_cell
            return r, c, None        # sentinel: UNID cell, no item_id yet
        return None

    def _unid_occupied_cell(self, board):
        """First occupied cell the classifier could not identify (item_id is None
        or score below LABEL_MIN_SCORE). Excludes the necromerger and champions
        (those are skipped on purpose — we never pop a Champion to read its
        body, and the necromenger is the Lair NPC). Stations (grave,
        manapool, icebox) ARE eligible — if a station ever falls out of the
        classifier bank, the UNID path is our chance to re-identify it.
        """
        for cell in board.cells:
            if (cell.row, cell.col) == necromerger_cell():
                continue
            if not cell.occupied:
                continue
            if cell.item_id and cell.score >= LABEL_MIN_SCORE:
                continue        # already a known item, handled above
            iid = cell.item_id or ""
            if iid.startswith(CHAMPION_PREFIXES):
                continue
            if iid == "necromerger" or iid.startswith("necromerger_"):
                continue
            return cell.row, cell.col
        return None

    def _auto_read_popup(self, board) -> None:
        """Popup-read ONE known-but-unbanked item per step (bounded).

        Live only, requires the popup_reader (Identifier.read_recipe). Banks
        the description + max-level status into the glossary, so the board
        renderer can flag `(max level)` cells and the validator can refuse a
        bogus merge of two top-level items. Logs a `popup_read` event.
        """
        if not self.live or self.popup_reader is None or not self.tool_enabled:
            return
        if self._frame is None:
            return
        try:
            found = self._popup_next(board)
        except Exception:
            return
        if found is None:
            return
        for _ in range(MAX_POPUP_READS):
            r, c, item_id = found
            target = board.cell_at(r, c)
            if target is None:
                return
            try:
                info = self.popup_reader(self._frame, target) or {}
            except Exception as exc:
                self.log.log("popup_read", item=item_id or "UNID", cell=(r, c),
                             ok=False, error=str(exc))
                return
            if not info:
                continue
                # UNID-cell path — item_id is None, so the popup just
            # gave us a fresh name. Resolve it (sanitize, add _lvlN) and
            # bank the sprite BEFORE banking the recipe, so subsequent
            # classify_board runs recognize the cell.
            resolved_id = item_id
            if resolved_id is None:
                resolved_id = self._resolve_unid_popup(target, info)
                if resolved_id is None:
                    # Popup didn't reveal a usable name — try the next cell.
                    return
                self._bank_unid_sprite(target, resolved_id)
            self._bank_popup_recipe(resolved_id, info)
            self.log.log("popup_read", item=resolved_id, cell=(r, c),
                         merge_info=bool(info.get("merge_info")),
                         description=bool(info.get("description")),
                         feed_value=info.get("feed_value"),
                         damage=info.get("damage"),
                         unid=item_id is None)
            return

    def _resolve_unid_popup(self, cell, info: dict) -> str | None:
        """Turn a popup's name+level into a template-bank id for an UNID cell.

        Mirrors Identifier._resolve_id: lowercase the name, sanitize to
        [a-z0-9_], append `_lvlN` if the popup gave a level. Refuses empty
        names, non-string names (LLM occasionally emits ints), and
        LLM-invented base names that aren't already in the bank (we don't
        want to invent new ids from hallucinated popups).
        """
        name = (info.get("name") or "")
        if not isinstance(name, str):
            return None
        name = name.strip().lower()
        name = re.sub(r"[^a-z0-9_]", "", name)
        if not name:
            return None
        level = info.get("level")
        try:
            level = int(level) if level is not None else None
        except (TypeError, ValueError):
            level = None
        item_id = f"{name}_lvl{level}" if level is not None else name
        if self.classifier is not None and not self.classifier.has(item_id):
            return None
        return item_id

    def _bank_unid_sprite(self, cell, item_id: str) -> None:
        """Extract a board-sprite template from a freshly-identified UNID cell
        and add it to the runtime bank. This is the second half of the Aug 27
        unbanked-popup-read backfill: the popup gave us the name, the cell
        sprite gives us the visual signature for future classify_board calls."""
        if self._frame is None or self.classifier is None:
            return
        half = cell.cell_px // 2
        region = self._frame[cell.cy - half: cell.cy + half,
                             cell.cx - half: cell.cx + half]
        if region is None or region.size == 0:
            return
        try:
            self.classifier.add_template(item_id, region)
        except Exception:
            return

    # Independent popup reads agreeing on one id for one cell, after which
    # the id banks despite OCR disagreement (see identify_unbanked).
    UNBANKED_CONSENSUS = 3
    UNBANKED_VOTE_CELLS = 30

    def _unbanked_consensus(self, cell, item_id: str) -> bool:
        """Record one vote for (cell, item_id); True once consensus reached.

        Votes are keyed per cell and reset for other ids on the same cell
        (a changing popup story is not consensus). Bounded to
        UNBANKED_VOTE_CELLS cells (oldest evicted) so dead cells can't
        accumulate forever.
        """
        try:
            votes = getattr(self, "_unbanked_votes", None)
            if votes is None:
                votes = {}
                self._unbanked_votes = votes
            key = (cell.row, cell.col)
            cell_votes = votes.get(key)
            if cell_votes is None or cell_votes.get("__id__") != item_id:
                cell_votes = {"__id__": item_id, item_id: 0}
                votes[key] = cell_votes
            cell_votes[item_id] = cell_votes.get(item_id, 0) + 1
            while len(votes) > self.UNBANKED_VOTE_CELLS:
                votes.pop(next(iter(votes)))
            if cell_votes[item_id] >= self.UNBANKED_CONSENSUS:
                votes.pop(key, None)
                self.log.log("identify_consensus_banked",
                             cell=[cell.row, cell.col], item=item_id,
                             votes=self.UNBANKED_CONSENSUS)
                return True
            return False
        except Exception:
            return False

    def _auto_collect_queue(self, board, frame) -> None:
        """Place a queued dock reward before planning (code-driven).

        The model never emits native tool calls on this stack, so the
        collect_queue tool can never fire voluntarily. When has_reward is
        true AND the board has room (same congestion gate collect() itself
        enforces), place immediately — queued rewards are chests/rune piles
        the bot wants on the board anyway. The placed cell is patched so
        this step plans on the post-placement board; next step's classify
        re-reads it properly. All outcomes are logged; failures never
        stall the step.
        """
        if (not self.live or not self.tool_enabled
                or self.queue_box is None or frame is None):
            return
        try:
            from vision.queue_box import has_reward
            from planner.agent import CONGESTION_THRESHOLD
            if not has_reward(frame):
                return
            empty = sum(1 for c in board.cells if not c.occupied)
            if empty <= CONGESTION_THRESHOLD:
                # Visible skip (not silent): a queued reward rotting on a
                # full board otherwise looks identical to a broken collect
                # path in the logs. Throttled: only on transition into the
                # blocked state (tracked per reward-present episode).
                if not getattr(self, "_queue_blocked_logged", False):
                    self.log.log("queue_collect_skipped", via="auto",
                                 reason="congested", empty=empty)
                    self._queue_blocked_logged = True
                return
            self._queue_blocked_logged = False
            res = self.queue_box.collect()
        except Exception as exc:
            self.log.log("queue_collect_miss", via="auto", error=str(exc))
            return
        if not res.get("placed"):
            self.log.log("queue_collect_miss", via="auto",
                         error=res.get("error"))
            return
        self.log.log("queue_collected", placed=True, via="auto",
                     opened=bool(res.get("opened")),
                     item=res.get("placed_item"), cell=res.get("cell"),
                     error=res.get("error"))
        try:
            cell_rc = res.get("cell")
            if cell_rc:
                target = board.cell_at(int(cell_rc[0]), int(cell_rc[1]))
                if target is not None:
                    target.occupied = True
                    placed = res.get("placed_item")
                    if (placed and placed not in ("unidentified", "unknown")
                            and self.classifier is not None
                            and self.classifier.has(placed)):
                        target.item_id = placed
                        target.score = 1.0
                        target.margin = 1.0
                    else:
                        target.item_id = None
                        target.score = 0.0
                        target.margin = 0.0
        except Exception:
            pass

    def _observe_feats(self) -> None:
        """Periodically open the FEATS panel, collect any ready rewards, read
        it, and cache the result.

        Live only; requires a PanelReader. The open->(collect)->read->BACK
        cycle is as intrusive as get_cravings so the full read is gated by
        FEATS_REFRESH_STEPS and MAX_PANELS per refresh; the cached tier+feats
        are shown in `get_board_state`, so the model sees its current missions
        and can steer moves toward one that's close to completion.

        Collecting is decoupled from reading: even when the read cache is
        fresh, an opportunistic collect is attempted every FEAT_COLLECT_STEPS
        if the cached feats indicate a done reward sitting unclaimed (the
        periodic 15-step read alone left rewards sitting for up to 14 steps
        after completion — observed in the 100-step soak with 0 collects).
        The `collect_feat_rewards` tool also lets the model force an immediate
        collect without waiting for either timer. Uses
        `panels.collect_feat_rewards` (template-matched, verified, bounded);
        `feat_reward_collected` is logged with the count.
        """
        if not self.live or self.panels is None or not self.tool_enabled \
                or self._frame is None:
            return
        if self._defer_panel_reads:
            # A buy will open the Station panel this step — skip the
            # scheduled FEATS cycle so the step costs one panel visit, not
            # two. The buy's own visit still collects nothing feat-side;
            # the normal schedule resumes next step.
            self.log.log("feats_panel_deferred", reason="buy_this_step")
            return
        cache = self._feats_cache
        if cache is not None and self._step_count - cache.get("step", 0) < FEATS_REFRESH_STEPS:
            # Read cache is fresh — still opportunistically collect if a
            # cached feat is done and we haven't tried recently. This keeps
            # the read at 15 steps but collects at 5. Skip when the last
            # attempt found nothing AND the cache hasn't refreshed since
            # (reopening the panel every 5 steps for a permanently
            # unclaimable button wastes ~10-20s a time).
            has_done = any(bool(f.get("done")) for f in (cache.get("feats") or []))
            cache_newer = cache.get("step", 0) > self._last_feat_collect_step
            if (has_done
                    and self._step_count - self._last_feat_collect_step >= FEAT_COLLECT_STEPS
                    and (self._last_feat_collect_found or cache_newer)):
                try:
                    result = self.panels.collect_feat_rewards(self._frame)
                except Exception as exc:
                    self.log.log("feats_read", ok=False, error=str(exc))
                    return
                self._last_feat_collect_step = self._step_count
                panel = result.get("panel") or {}
                feats = panel.get("feats") or []
                collected = int(result.get("collected") or 0)
                self._last_feat_collect_found = collected > 0 or bool(result.get("tier"))
                if result.get("error"):
                    self.log.log("feat_collect_miss", via="opportunistic",
                                 error=result.get("error"),
                                 claim_miss=result.get("claim_miss"))
                    return
                if feats:
                    self._feats_cache = {"tier": panel.get("tier"), "feats": feats,
                                         "step": self._step_count}
                    self.log.log("feats_read", tier=panel.get("tier"), feats=len(feats))
                if collected:
                    self.log.log("feat_reward_collected", count=collected, via="opportunistic")
                elif result.get("claim_miss"):
                    # Buttons exist but none mapped to a done feat — visible
                    # for calibration instead of a silent no-op.
                    self.log.log("feat_collect_miss", via="opportunistic",
                                 claim_miss=result.get("claim_miss"))
                if result.get("tier"):
                    self.log.log("tier_reward_collected", tier=panel.get("tier"), via="opportunistic")
            return
        try:
            result = self.panels.collect_feat_rewards(self._frame)
        except Exception as exc:
            self.log.log("feats_read", ok=False, error=str(exc))
            return
        self._last_feat_collect_step = self._step_count
        panel = result.get("panel") or {}
        feats = panel.get("feats") or []
        collected = int(result.get("collected") or 0)
        self._last_feat_collect_found = collected > 0 or bool(result.get("tier"))
        if result.get("error"):
            self.log.log("feat_collect_miss", via="periodic",
                         error=result.get("error"),
                         claim_miss=result.get("claim_miss"))
            return
        if not feats:
            return
        self._feats_cache = {"tier": panel.get("tier"), "feats": feats,
                             "step": self._step_count}
        self.log.log("feats_read", tier=panel.get("tier"), feats=len(feats))
        if collected:
            self.log.log("feat_reward_collected", count=collected)
        elif result.get("claim_miss"):
            self.log.log("feat_collect_miss", via="periodic",
                         claim_miss=result.get("claim_miss"))
        if result.get("tier"):
            self.log.log("tier_reward_collected", tier=panel.get("tier"))

    @staticmethod
    def _tool_call_msg(tc: dict) -> dict:
        return {"role": "assistant", "content": None, "tool_calls": [tc]}

    def _exec_set_strategy(self, args: dict) -> str:
        """Validate + commit the model's strategy slot choice.

        `feat` must match an active feat in the cached FEATS read (the model
        may only pick from what the Feats line showed); `target_item` must be
        a real template-bank id or family prefix. A committed strategy boosts
        its objective's move-kind weight (STRATEGY_KIND_BONUS) and shows in
        get_board_state until it expires or its feat completes."""
        feat = str(args.get("feat") or "").strip()
        target = str(args.get("target_item") or "").strip()
        if not feat:
            return '{"error": "set_strategy requires feat"}'
        feats = (self._feats_cache or {}).get("feats") or []
        if feats:
            match = next((f for f in feats
                          if feat.lower() in (f.get("name") or "").lower()
                          or (f.get("name") or "").lower() in feat.lower()), None)
            if match is None:
                known = "; ".join(sorted({(f.get("name") or "")
                                          for f in feats if f.get("name")}))
                return json.dumps({"error": f"unknown feat {feat!r}",
                                   "active_feats": known})
            feat = match.get("name") or feat
        if target and self.classifier is not None:
            if not (self.classifier.has(target)
                    or any(iid.startswith(target)
                           for iid in self.classifier.templates)):
                return json.dumps({"error": f"unknown target_item {target!r}",
                                   "hint": "use a template id like skeleton_lvl2 or a family prefix like zombie"})
                                   # classify the feat noun so we can pick the right hint +
        # drop the target when it doesn't match the noun category (e.g.
        # feat "Own a Lvl 3+ Grave." with target "skeleton_lvl6" — the
        # grave feat needs a station, not a creature target).
        kind, noun = self._classify_feat_noun(feat)
        if kind == "station":
            target = ""   # drop creature targets; station feats buy + merge
            self._strategy_family = noun
        elif kind == "creature":
            # keep the target only if its family matches the noun
            if target:
                tnorm = (target.split("_")[0] or "").lower()
                if tnorm != noun:
                    target = ""
            self._strategy_family = None
        else:
            self._strategy_family = None
        self._strategy = {"feat": feat, "target": target or None,
                          "step": self._step_count, "kind": kind,
                          "noun": noun}
        self.log.log("strategy_set", feat=feat, target=target or None,
                     step=self._step_count)
        # Progress-path nudge: the model picks feat NAMES without knowing
        # what they require (observed: "Build a Mana Pool." + a random
        # target, then no prerequisite work). Teach the THREE known shapes:
        # station-purchase / creature-collect / other (action-count etc).
        note = "committed — the priority layer now serves this objective"
        if kind == "station" and noun:
            disp = display_family(noun)
            note += (f" — this feat needs the {disp} STATION "
                     f"from the Station panel: call buy_station(\""
                     f"{disp}\", confirm=false) to check "
                     f"affordability, then buy_station(\"{disp}"
                     f"\", confirm=true) to complete. Gather the runes first.")
        elif kind == "creature" and noun:
            note += (f" — progress by merging/collecting {noun} on the board")
        elif kind == "other":
            note += (f" — the priority layer cannot boost a particular kind "
                     f"for this feat; follow the action in the feat text")
        return json.dumps({"ok": True, "strategy": self._strategy,
                            "expected_family": self._strategy_family,
                            "kind": kind, "noun": noun,
                            "note": note})

    def _currency_line_text(self, need_family=None) -> str:
        """Rune balances and a station-build affordance reminder.

        The Rune bar is only visible when the Station menu is open (the
        fixed slots overlap the lair's mana bar otherwise, reading "16.8k"
        as 168 — see the Aug 24 inert-Confirm mystery). We instead surface
        the last CACHED Rune balance from the most recent buy_station
        (confirm=false) call: that call opens the Station panel, the
        read_currency OCR runs, and the result is stashed in
        `self._last_currency_result`. The board state line below uses the
        cache so the model retains the Rune balance across steps even
        after the panel closes — otherwise it forgets how many ice it has
        and re-asks buy_station in a tight loop. The line also renders a
        "needs N more ice for grave" reminder when a station-build strategy
        is active and the cached balance says the user is short.

        need_family lets the board state pass a non-strategy need (the
        craving's producer station): same shortfall math applies. The
        shortfall also names the feed-equivalent (≈N max-stack feeds) —
        the model can't divide, so code does it.
        """
        if not self._last_currency_result:
            return ""
        currency = self._currency or {}
        if not currency:
            return ""
        rune_order = ("ice", "poison", "blood", "moon", "death")
        parts = [f"{n} {r}" for r in rune_order
                 for n in (currency[r],) if n is not None]
        if not parts:
            return ""
        line = "Last buy_station Rune balance: " + " + ".join(parts) + "."
        # affordance reminder for the build strategy. If the
        # last buy_station for the strategy's family said "unaffordable",
        # tell the model how many Runes it's short and what to do
        # (spawn / merge / feed). The user reported the model "always
        # opens up the menu but never buys the grave" — the model saw
        # the unaffordable result but the next step's board state had
        # no Rune balance, so the model forgot and re-asked. This line
        # closes that loop.
        if self._strategy_family or need_family:
            family = self._strategy_family or need_family
            err = self._last_currency_result.get("error") or ""
            if "unaffordable" in err:
                # Parse the cost from the error message
                import re
                m = re.search(r"needs\s+(\d+)\s+(\w+)", err)
                if m:
                    need_n, need_rune = int(m.group(1)), m.group(2)
                    have = currency.get(need_rune, 0)
                    if have < need_n:
                        short = need_n - have
                        # Determine the gather recipe
                        recipe = ""
                        if need_rune == "ice":
                            recipe = ("Tap Ice Chests ON THE BOARD with `spawn` "
                                      "(mana-free, one Rune per use), merge the "
                                      "icerune stacks up, FEED the max-level "
                                      "icerune_lvl3 to the Devourer for Ice Runes.")
                        elif need_rune == "poison":
                            recipe = ("Same recipe for Poison Runes: tap chests, "
                                      "merge poisonrune stacks, feed max-level "
                                      "poisonrune_lvl3 to the Devourer.")
                        else:
                            recipe = (f"Open chests (mana-free), merge {need_rune} "
                                      f"stacks, feed max-level for {need_rune} Runes.")
                        line += (f" You're {short} {need_rune} short for "
                                  f"{family} (need {need_n}, "
                                  f"have {have}){self._feeds_equiv(need_rune, short)}. {recipe}")
        return line

    def _feeds_equiv(self, rune: str, short: int) -> str:
        """' (≈N max <stack> feeds)' for a rune shortfall, '' when unknown.

        The model cannot divide a deficit into feeds; code does it from the
        max-level stack's feed value (e.g. short 18 poison at 12/feed ≈
        2 max poisonrune_lvl3 feeds).
        """
        try:
            stack = f"{rune}rune_lvl3"
            fv = (self._feed_values() or {}).get(stack)
            if fv and fv > 0 and short > 0:
                import math
                return f" (≈{math.ceil(short / fv)} max {stack} feeds)"
        except Exception:
            pass
        return ""

    

    def _feats_line_text(self) -> str:
        """Feats (missions) from the cached panel read, with each active task's
        parsed action kind and progress, e.g.
        'Feats (tier 3): [feed] Unlock Mana Well (1/4 Feats Completed); [merge]
        Merge things 50 times. (done) | done: Own a lvl 2+ Grave.'.

        ALL feats are shown (previously the first 3 only hid the actionable
        ones). `[kind]` tells the model which action each task needs and
        `NEAR DONE` flags a task at >= 80% so it can steer toward completion.
        Empty when no feats read yet this session."""
        cache = self._feats_cache
        if not cache or not cache.get("feats"):
            return ""
        parts = []
        done_parts = []
        for f in cache["feats"]:
            name = f.get("name")
            if not name:
                continue
            o = parse_obj(f)
            if f.get("done"):
                done_parts.append(f"{name} (done)")
            elif o.kind == "neutral":
                parts.append(f"[{o.kind}] {name} ({f.get('count_done', 0)}/{f.get('count_required', 0)} Feats Completed)")
            elif o.kind == "station":
                parts.append(f"[{o.kind}] {name} ({f.get('count_done', 0)}/{f.get('count_required', 0)}) [build]")
            elif o.kind == "creature":
                parts.append(f"[{o.kind}] {name} ({f.get('count_done', 0)}/{f.get('count_required', 0)}) [collect]")
            elif o.kind == "combat":
                parts.append(f"[{o.kind}] {name} ({f.get('count_done', 0)}/{f.get('count_required', 0)}) [combat]")
            elif o.kind == "action":
                parts.append(f"[{o.kind}] {name} ({f.get('count_done', 0)}/{f.get('count_required', 0)}) [action]")
            elif o.kind == "level":
                parts.append(f"[{o.kind}] {name} ({f.get('count_done', 0)}/{f.get('count_required', 0)}) [level]")
            else:
                parts.append(f"[{o.kind}] {name} ({f.get('count_done', 0)}/{f.get('count_required', 0)}) [other]")
        if parts or done_parts:
            return "Feats (tier " + str(cache.get("tier", 0)) + "): " + "; ".join(parts + done_parts) + "."
        return "Feats (tier " + str(cache.get("tier", 0)) + "): (none)"

    def _strategy_line_text(self, board=None) -> str:
            """Render the model's active strategy for get_board_state (its own
            committed choice — shown even with --no-hints, since it is model
            state, not a code-computed hint).

            Wording matters: the old '(build toward manapot)' phrasing read as
            'spawn stuff to create it' to a 4B model, which spawn-spammed onto a
            full board while ignoring the merge hints (Aug 25). The line now
            states the REQUIREMENT (the station to buy, or the item to
            merge/collect) plus its cached rune cost when known.

            when `board` is provided, the build-station hint counts
            existing stations on the board and tells the model the exact
            merge-up math (e.g. "you have 1×lvl1 + 1×lvl2 — merge them
            up, but you'll need 1 more grave to reach lvl3"). Without the
            board read, the model only knew "gather them first" — the
            user reported the hint was insufficient.
            """
            s = self._strategy
            if not s or not self._strategy_fresh():
                return ""
            age = self._step_count - s.get("step", 0)
        
            # Handle meta-goals
            meta_goal = s.get("meta_goal")
            if meta_goal:
                family = s.get("target_family")
                req = self._strategy_requirement_text(s, board=board)
                line = f"Strategy: META {meta_goal}"
                if family:
                    line += f" (target: {family})"
                line += req
                line += f" (you set this {age} steps ago — change it with set_strategy)"
            else:
                # Explicit feat
                req = self._strategy_requirement_text(s, board=board)
                age = self._step_count - s.get("step", 0)
                line = f"Strategy: {s['feat']}{req} (you set this {age} steps ago — change it with set_strategy)"
        
            pending = self.shop._pending_buy if self.shop is not None else None
            if pending:
                pb_fam = pending.get("family")
                pb_age = self._step_count - pending.get("step", 0)
                line += (f"\nPending buy: {pb_fam} awaits confirm=true "
                          f"({pb_age} step{'s' if pb_age != 1 else ''} old) "
                          f"— call buy_station('{pb_fam}', confirm=true) to complete")
            return line

    def _suggested_direction(self, board) -> str:
        """One-line meta-goal hint when no strategy is committed.

        Replaces the retired StrategyPlanner's periodic LLM call: the same
        META_GOALS thresholds evaluate code-side over cached state (no taps,
        no LLM), and the winning goal renders here so the compelled
        set_strategy round (and the model generally) chooses with direction.
        Empty when a strategy is fresh (a direction is already committed) or
        when no threshold fires.
        """
        if self._strategy_fresh():
            return ""
        try:
            occupied = sum(1 for c in board.cells if c.occupied)
            total = len(board.cells) or 1
            fb = self.fallback
            state = {
                "champion": self._last_champion,
                "satiety_remaining": getattr(fb, "satiety_remaining", None),
                "satiety_capacity": getattr(fb, "satiety_capacity", None),
                "board_congestion": occupied / total,
                "slime_count": getattr(fb, "slime_count", None),
                "mana_pct": (self._mana_fraction * 100
                             if self._mana_fraction is not None else None),
                "runes": self._currency or {},
            }
            for goal in META_GOAL_PRIORITY:
                cfg = META_GOALS[goal]
                try:
                    if cfg["threshold_check"](state):
                        fam = cfg.get("target_family")
                        fam_txt = (f" — consider set_strategy with target_family "
                                   f"'{fam}'") if fam else ""
                        return (f"Suggested direction: {goal} "
                                f"({cfg['description']}{fam_txt}).")
                except Exception:
                    continue
        except Exception:
            pass
        return ""

    def _count_stations_on_board(self, board, family: str) -> dict[int, int]:
        """count stations of `family` on the board, grouped by level.

    Returns `{level: count}` -- e.g. `{1: 1, 2: 1}` for one lvl-1 and
    one lvl-2 grave. The strategy-line text uses this to tell the
    model exactly what merge-up step is needed.
    """
        counts = {}
        for cell in board.cells:
            if not cell.item_id:
                continue
            item_lc = cell.item_id.lower()
            if not item_lc.startswith(family):
                continue
            m = re.search(r"_lvl(\d+)$", item_lc)
            level = int(m.group(1)) if m else 0
            counts[level] = counts.get(level, 0) + 1
        return counts


    def _creature_chain_hint(self, noun: str, board) -> str:
        """Encourage discovering an UNKNOWN creature merge chain.

        When a creature-strategy target family has no entry in the stored
        merge-chain map (`merge_chains.json`), the model can't act on the
        "spawn/merge {noun} to reach lvl N" hint because it doesn't yet know
        how the family is produced. Instead of hand-seeding the answer, we
        drive the existing discovery loop (popup-edge -> chain consolidation):

        - Reverse-look up `spawn_rates.json` for any station the bot has
          already LEARNED spawns the target family (e.g. haunted-built
          `grave_lvl3 -> ... -> zombie`). If one exists, tell the model to
          spawn from that station and merge copies up, reading each item's
          info popup to bank the merge link — the summary model then
          consolidates the links into a stored `merge chain <family>`.
        - Otherwise fall back to generic: spawn the family's components and
          confirm the merge path via popup reads / lookup_wiki.

        Returns a suffix string ('' when the chain is already known)."""
        chain = self._chain_map()
        if noun in chain:
            return ""   # chain known; the plain spawn/merge hint suffices
        # Find a station (from grounded spawn_rates) that can produce the
        # family, so we can point the model at a concrete spawn source.
        source = self._spawn_source_for_family(noun)
        if source:
            return (f" The merge chain for {noun} is not in your memory yet "
                    f"— discover it by spawning {noun} components from "
                    f"{source} and merging copies up, reading each item's "
                    f"info popup to learn + bank the merge path.")
        return (f" The merge chain for {noun} is not in your memory yet — "
                f"discover it: spawn its components, merge copies up, read "
                f"the item popups to learn the merge path, and if still "
                f"unsure use lookup_wiki to confirm.")

    def _spawn_source_for_family(self, noun: str) -> str | None:
        """Reverse-lookup which station the bot has learned spawns `noun`.

        Scans `spawn_rates.json` (grounded popup reads) for the first station
        whose spawn-target base name matches the family prefix, e.g.
        grave_lvl3 -> [bone, ribcage, zombie] matches noun=zombie. Returns a
        human id like "grave_lvl3" or None."""
        try:
            from planner.glossary import read_spawn_rates, _get_knowledge_dir
        except Exception:
            return None
        try:
            kd = (None if not hasattr(self, 'knowledge_dir') else self.knowledge_dir)
            rates = read_spawn_rates(kd)
        except Exception:
            return None
        # Prefer the highest-level station so the spawn is likelier to drop
        # the family's higher-level components.
        def level(sid: str) -> int:
            m = re.search(r"_lvl(\d+)$", sid)
            return int(m.group(1)) if m else 0
        for sid in sorted(rates, key=level, reverse=True):
            for tgt in (rates[sid].get("targets") or []):
                if (tgt or "").split("_")[0].lower() == noun.lower():
                    return sid
        return None


    def _build_station_step_hint(self, family: str,
                                existing: dict[int, int],
                                need: int) -> str:
        """render a step-by-step station-build hint from the board state.

        The model knows it has, say, lvl1+lvl2 graves and needs lvl3. The
        previous generic "gather them first" phrase didn't say that the
        existing stations could be MERGED. The new hint shows the exact
        merge-up math: "you have 1×lvl1 + 1×lvl2 — merge a lvl1 into the
        lvl2 to make lvl3, then buy + merge another lvl1 to make lvl4."

        the previous simulation reached lvl3 with `buys_needed=1`
        for `{1:1, 2:1}` (buy 1 lvl1, then merge 2×lvl1→lvl2 + 2×lvl2→lvl3).
        The resulting hint "merge them up, but you'll need 1 more grave" was
        MISLEADING — the user pointed out that the existing 1×lvl1 and 1×lvl2
        cannot be merged together (different levels), so "merge them up"
        suggested a path that doesn't exist. The new hint spells out the
        EXACT merge sequence: how many of each level to buy, and which
        pairs to merge. The simulation now also returns the merge plan
        itself (a list of merge steps), not just the buys count.
        """
        # simulate the minimum number of buys required to reach
        # `need` from the current board state. The merge path is
        # deterministic: you can always merge two of the same level to
        # get the next level. The bottleneck is "how many copies of
        # each level do I have to combine" — a 1x1x2x1 chain (lvl1+lvl1+lvl2)
        # can produce a lvl3 without buying; a 1x1x1 (just lvl1+lvl1+lvl1) needs
        # one more lvl1 to make a lvl3.
        if not existing:
            return (f" — you have NO {family} on the board; buy a {family} "
                    f"from the Station panel with buy_station(\"{family}\", "
                    "confirm=false) to read the dialog, then confirm=true to "
                    "complete. You may need to buy multiple copies to merge up.")
        # Highest current level
        have = sorted(existing.items())  # [(lvl, count), ...] sorted by lvl asc
        max_have = max(lvl for lvl, _ in have)
        if max_have >= need:
            return (f" — you ALREADY have a lvl{max_have} {family} on the "
                    f"board (need lvl{need}); you should be done. The feat "
                    "may not be tracking the right cell — check the Feats panel.")
                    # simulate the merge plan explicitly. The old code only
        # counted `buys_needed`; the new code returns the FULL plan
        # ("buy 1 more lvl1, then merge 2×lvl1→lvl2, then merge 2×lvl2→lvl3")
        # so the model can follow the exact sequence. The plan is a list of
        # (action, level) tuples: ("buy", L) means "buy a L", ("merge", L)
        # means "merge two of L to get L+1".
        plan = self._simulate_station_build(existing, need)
        if plan is None:
            # Couldn't reach — fall back to a soft message
            return (f" — current {family} stock: " +
                    ", ".join(f"{cnt}x lvl{lvl}"
                                for lvl, cnt in sorted(existing.items())) +
                    f". Merge + buy copies until you have a lvl{need} {family}.")
        level_summary = ", ".join(f"{cnt}x lvl{lvl}"
                                    for lvl, cnt in sorted(existing.items()))
        buys = sum(1 for action, _ in plan if action == "buy")
        merges = sum(1 for action, _ in plan if action == "merge")
        if buys == 0:
            return (f" — you have {level_summary} {family} on the board; "
                    f"merge them up to lvl{need} (no buy needed). The merge "
                    "sequence: pair equal levels and merge them up.")
        # Spell out the plan. The model knows NecroMerger merges are
        # lvl N + lvl N -> lvl N+1; saying "merge two lvl1 to make a lvl2"
        # is the unambiguous form.
        plan_lines = []
        buys_so_far = 0
        merges_so_far = 0
        for action, lvl in plan:
            if action == "buy":
                buys_so_far += 1
                if buys_so_far == 1:
                    plan_lines.append(f"buy {buys} more lvl{lvl} {family}")
                else:
                    plan_lines.append(f"buy {buys - buys_so_far + 1} more lvl{lvl} {family}")
            else:
                merges_so_far += 1
                plan_lines.append(f"merge two lvl{lvl} → lvl{lvl + 1}")
        return (f" — you have {level_summary} {family} on the board. To reach "
                f"lvl{need}: " + ", then ".join(plan_lines) + ".")

    def _simulate_station_build(self, existing: dict[int, int],
                                need: int) -> list[tuple[str, int]] | None:
        """compute the explicit station-build plan.

        Returns a list of (action, level) tuples describing the steps to take:
            - ("buy", L): buy one more lvl-L station
            - ("merge", L): merge two lvl-L into one lvl-(L+1)
        The plan is the SHORTEST sequence that reaches at least one lvl>=need
        on the board, simulating the chain merge-up. Returns None if the
        plan can't reach the goal within 50 iterations (a safety cap).

        Why explicit? The previous code only returned `buys_needed=1` for
        `{1:1, 2:1}` and emitted "merge them up, but you'll need 1 more
        grave" — misleading because the existing 1×lvl1 and 1×lvl2 cannot
        be merged together (different levels). The user reported this was
        confusing. The new plan spells out the exact sequence: buy 1 more
        lvl1, merge 2×lvl1→lvl2, then merge 2×lvl2→lvl3.
        """
        # Working pool: list of levels, sorted asc
        pool = []
        for lvl, cnt in existing.items():
            pool.extend([lvl] * cnt)
        plan: list[tuple[str, int]] = []
        for _ in range(50):  # safety cap
            pool.sort()
            if pool and pool[-1] >= need:
                return plan
            levels = [l for l in pool if l > 0]
            if not levels:
                return None
            min_lvl = min(levels)
            if levels.count(min_lvl) >= 2:
                # Merge two of min_lvl into min_lvl+1
                pool.remove(min_lvl)
                pool.remove(min_lvl)
                pool.append(min_lvl + 1)
                plan.append(("merge", min_lvl))
            else:
                # Buy a fresh min_lvl (the cheapest one to add to the pool)
                pool.append(min_lvl)
                plan.append(("buy", min_lvl))
        return None

    def _strategy_requirement_text(self, s: dict, board=None) -> str:
        """Requirement clause for the strategy line. Three shapes:
            - STATION   feat (Build a X / Own a Lvl N+ X where X is a station)
                        -> buy the X station from the Station panel; merge copies
                            up to the required level; show cached rune cost.
            - CREATURE  feat (Own a Lvl N+ Y where Y is a creature; Collect Nx Y)
                        -> merge components on the board to level N+ Y; do NOT
                            buy any station.
            - OTHER     feat (e.g. "Beat the Peasant twice.", "Tap the
                            NecroMerger 10 times.", "Reach Devourer level 5.")
                        -> no station/creature direction; the priority layer
                            does not boost a particular kind. The hint states the
                            action the feat describes.

                            prior code only handled Build-a-X and treated the model's
        target_item as authoritative for non-build feats. That produced the
        "(merge/collect skeleton_lvl6 on the board)" hint when the feat was
        "Own a lvl 3+ Grave." — wrong kind, wrong target. The noun of the
        feat is now classified first; only creatures get the target-style
        hint, and only when the target's family matches the noun.
        """
        feat = s.get("feat") or ""
        target = s.get("target") or ""
        kind, noun = self._classify_feat_noun(feat)
        # if the model supplied a target that doesn't match the noun
        # kind (e.g. target=skeleton_lvl6 for "Own a lvl 3+ Grave."), the
        # target is the wrong shape — drop it. We keep it only when the
        # target's base family matches the noun family of a creature feat
        # (e.g. feat "Own a lvl 3+ Skeleton." + target "skeleton_lvl3").
        if kind == "station":
            target = ""  # station-purchase feats never use a creature target
        elif kind == "creature":
            if target:
                tnorm = (target.split("_")[0] or "").lower()
                if tnorm != noun:
                    target = ""
        if kind == "station":
            txt = f" (feat needs the {noun} STATION from the Station panel"
            cost = (self.shop.cost_cache.get(noun) if self.shop else None)
            if cost:
                parts = []
                rune_order = ("ice", "poison", "blood", "moon", "death")
                for rune in rune_order:
                    n = cost.get(rune, 0)
                    if n:
                        parts.append(f"{n} {rune}")
                if parts:
                    txt += " — costs " + " + ".join(parts) + " runes"
                    # the previous "gather them first" phrasing was misleading
            # when the board already had stations. The model knew it had a
            # lvl1+lvl2 grave and needed one more lvl1 to merge into a lvl3,
            # but the hint didn't say that. The new hint counts the existing
            # stations on the board and tells the model exactly what to do.
            if board is not None:
                existing = self._count_stations_on_board(board, noun)
                need = self._max_station_level_for_feat(feat)
                if need and not existing:
                    # No stations of this family on the board; tell the
                    # model to buy fresh, not the generic "merge + buy" copy.
                    txt += (" — you have NO {0} on the board; buy a {0} "
                            "from the Station panel with buy_station"
                            "(\"{0}\", confirm=false) to read the dialog, "
                            "then confirm=true to complete. You may need to "
                            "buy multiple copies to merge up.").format(noun)
                elif existing and need:
                    txt += self._build_station_step_hint(noun, existing, need)
                else:
                    # Board has stations OR feat has no level target.
                    # Generic guidance: merge + buy.
                    txt += (" — merge any {0} you have on the board up to "
                            "lvl {1}, then buy another {0} from the Station panel "
                            "and merge it in. Buy with buy_station(\"{0}\", "
                            "confirm=false) to read the dialog, then confirm=true "
                            "to complete.").format(noun, need or "?")
            else:
                # No board read — fall back to the generic guidance.
                need = self._max_station_level_for_feat(feat)
                txt += (" — merge any {0} you have on the board up to lvl {1}, "
                        "then buy another {0} from the Station panel and "
                        "merge it in. Buy with buy_station(\"{0}\", confirm=false) "
                        "to read the dialog, then confirm=true to complete.").format(
                            noun, need or "?")
            return txt
        if kind == "creature":
            # The required level ALWAYS comes from the feat text itself
            # ("Own a lvl 3+ Zombie."), never from the model's optional
            # target_item id. `_max_station_level_for_feat` reuses the
            # "own a lvl N+ X" regex; it's family-agnostic so it works here.
            lvl = self._max_station_level_for_feat(feat)
            if target:
                return (f" (spawn/merge {target} and its components to "
                        f"reach lvl {lvl or '?'} on the board)")
            return (f" (spawn/merge {noun} and its components to "
                    f"reach lvl {lvl or '?'} on the board)" +
                    self._creature_chain_hint(noun, board))
        if kind == "other":
            return (" — this feat isn't station-purchase or creature-collect; "
                    "follow the action in the feat text")
        return ""

    @staticmethod
    def _classify_feat_noun(feat: str) -> tuple[str | None, str | None]:
        """Return (kind, noun) for a feat string, where kind is one of
        'station' / 'creature' / 'other' / None.

        previously this only detected station-shape feats and
        returned the family directly. The wiki has THREE distinct feat
        shapes that all need different hints; the noun's classification
        (station vs creature) is what determines the right hint.
        """
        if not feat:
            return None, None
        low = feat.lower()
        # "Build a <X>." / "Build the <X>." — always a station-purchase feat.
        if "build" in low:
            stripped = (low.replace("build a", "").replace("build the", "")
                            .replace("build", "").strip(" ."))
            fam = _STATION_NAME_ALIASES.get(stripped) or normalize_station_name(stripped)
            if fam and fam != "unknown":
                if fam in STATION_FAMILIES:
                    return "station", fam
                # "Build a Mana Golem" — a creature? NecroMerger doesn't
                # have a station called that, so fall through to "other".
                return "other", fam
            return "other", None
        # "Own a Lvl N+ <X>." / "Own a <X>." — station OR creature.
        if "own" in low:
            noun = VisionDrivenPlanner._extract_own_noun(low)
            if noun:
                if noun in STATION_FAMILIES:
                    return "station", noun
                if noun in CREATURE_FAMILIES:
                    return "creature", noun
                return "other", noun
        return "other", None

    @staticmethod
    def _extract_own_noun(low: str) -> str | None:
        """Extract the final noun from an "Own ..." feat name.

        previous regex used `\\w+` which doesn't match `+` (in
        "lvl 3+") so the noun was silently dropped for every
        "Own a Lvl N+ <station>." feat. The fix: use `[\\w+-]+` for the
        lvl token so the regex consumes "lvl", digits, and `+` together
        and the final noun is captured correctly. Returns the
        canonical no-space family (matches STATION_FAMILIES /
        CREATURE_FAMILIES), e.g. "manapool" not "mana pool".
        """
        import re
        # "own a lvl 3+ grave."   -> group(1) = "grave"
        # "own a grave."          -> group(1) = "grave"
        # "own a lvl 10 altar."   -> group(1) = "altar"
        m = re.search(r"own\s+(?:a|an|the)?\s*(?:[\w+-]+\s+)*?(\w+)\s*\.?\s*$", low)
        if not m:
            return None
        raw = m.group(1)
        # The final noun may be multi-word ("mana pool", "supply cupboard",
        # "eye monster", "mana golem", "forgotten minion") that the regex
        # captures only the first OR last word of. Re-scan the trailing
        # text to recover the full noun.
        full_noun = VisionDrivenPlanner._scan_trailing_noun(low)
        if full_noun and full_noun in _STATION_NAME_ALIASES:
            return _STATION_NAME_ALIASES[full_noun]
        if full_noun and full_noun.replace(" ", "") in CREATURE_FAMILIES:
            return full_noun.replace(" ", "")
        return _STATION_NAME_ALIASES.get(raw) or normalize_station_name(raw) or None

    @staticmethod
    def _scan_trailing_noun(low: str) -> str | None:
        """Find the longest trailing noun (1-3 words) before the final
        period in an "Own a Lvl N+ <X>." feat. The captured noun may be
        multi-word ("mana pool", "supply cupboard", "eye monster", "mana
        golem", "forgotten minion"); the simple regex only sees the last
        word. We try all 1/2/3-word trailing combinations and return
        the longest match that resolves to a known family.
        """
        # Strip the final period and split on whitespace.
        body = low.rstrip(" .")
        words = body.split()
        for n_words in (3, 2, 1):
            if len(words) < n_words:
                continue
            cand = " ".join(words[-n_words:])
            if cand in _STATION_NAME_ALIASES:
                return cand
            if cand.replace(" ", "") in CREATURE_FAMILIES:
                return cand
        return None

    @staticmethod
    def _max_station_level_for_feat(feat: str) -> int | None:
        """parse the level-N target from an "Own a Lvl N+ X" feat.

        Returns the integer N (e.g. 3 for "Own a Lvl 3+ Grave.") or None
        if the feat isn't a level-target feat. The build-station hint uses
        this to tell the model the exact level to reach.
        """
        m = re.search(r"own a lvl\s*(\d+)\+", feat.lower())
        if m:
            return int(m.group(1))
        return None

    def _exec_tool(self, tc: dict, board) -> str:
        """Execute a tool call and return its JSON string result."""
        try:
            name = tc["function"]["name"]
            args = json.loads(tc["function"].get("arguments") or "{}")
        except (json.JSONDecodeError, KeyError, TypeError):
            return '{"error": "unparseable tool call"}'
        if name == "get_board_state":
            return self._board_state_text(board)
        if name == "set_strategy":
            return self._exec_set_strategy(args)
        if name == "buy_station":
            if self.shop is None:
                return '{"error": "buying unavailable in this mode"}'
            # All family intake canonicalizes (model emits spaced display
            # names — "supply cupboard" — per display_family; the board,
            # caches, and guards all use compact canonical ids).
            family = canonical_family(args.get("family"))
            if not family:
                return '{"error": "buy_station requires family"}'
                # strategy-family enforcement (B1) — if the committed
            # strategy names a specific station to build, reject a buy for
            # a different family. The model observed picking the cheaper
            # 'grave' when the strategy was 'Build a Mana Pool.'.
            if self._strategy_family and family != self._strategy_family:
                self.log.log("station_buy_refused", family=family,
                                strategy_family=self._strategy_family,
                                reason="strategy_mismatch")
                return json.dumps({
                    "error": (f"strategy says build {self._strategy_family!r}, "
                                f"not {family!r}"),
                    "expected_family": self._strategy_family,
                    "hint": (f"call buy_station({self._strategy_family!r}, "
                                f"confirm=false) instead"),
                })
            confirm = bool(args.get("confirm"))
            # two-phase follow-through (A1+A4) — track the pending
            # buy across steps. The model observed calling confirm=false
            # then choosing spawn instead of confirm=true; this forces the
            # second call to match the first (or be explicit about skipping).
            if confirm:
                if (self.shop is None or self.shop._pending_buy is None):
                    self.log.log("station_buy_refused", family=family,
                                    reason="no_pending_buy")
                    return json.dumps({
                        "error": (f"no pending buy for {family!r} — call "
                                    f"buy_station({family!r}, confirm=false) "
                                    f"first to read the dialog"),
                    })
                if self.shop._pending_buy.get("family") != family:
                    pending_fam = self.shop._pending_buy.get("family")
                    self.log.log("station_buy_refused", family=family,
                                    pending_family=pending_fam,
                                    reason="pending_family_mismatch")
                    return json.dumps({
                        "error": (f"pending buy is for {pending_fam!r}, not "
                                    f"{family!r} — either complete it or wait "
                                    f"for it to expire"),
                        "pending_family": pending_fam,
                    })
            res = self.shop.buy(family, confirm=confirm)
            dialog = res.get("dialog") or {}
            # cache the last buy_station result so the board state
            # can surface the Rune balance + affordability status even when
            # the panel is closed. Without this, the model forgets between
            # steps that it has 0 ice Runes and re-asks buy_station in a
            # tight loop. Stash every call (both confirm=false and
            # confirm=true) — the board-state line consumes the cached
            # balance to render the "needs N more ice for grave" reminder.
            self._last_currency_result = res
            self._last_currency_step = self._step_count
            self._currency = res.get("currency")
            # Stamp the feats tier on no-card misses: a family absent from
            # the sheet while locked slots exist is unlock-gated, not
            # rune-gated — the compel gate compares this tier to skip
            # rescanning until progression happens.
            if (res.get("error") or "").startswith("no ") and res.get("locked"):
                res["_tier_at_miss"] = (self._feats_cache or {}).get("tier")
                self._last_currency_result = res
            # two-phase state machine.
            # - confirm=false + dialog verified + affordable + bought not
            #   yet: stash the pending buy for the model to confirm.
            # - confirm=true + bought: clear the pending buy.
            if not confirm and res.get("note"):
                # Phase 1 succeeded — dialog verified, awaiting confirm.
                # pending lives on self.shop; StationShop.buy()
                # already wrote it there.
                pass
            elif confirm and res.get("bought"):
                # pending lives on self.shop; StationShop.buy()
                # already cleared it.
                pass
                # include the icon-guard fields so the log shows when
            # the LLM misread a rune icon and the guard corrected it.
            self.log.log("station_buy", ok=bool(res.get("bought")),
                            family=family, confirm=confirm,
                            error=res.get("error"),
                            dialog=dialog,
                            currency=res.get("currency"),
                            panel_verified=res.get("panel_verified"),
                            cost_icon_override=dialog.get("cost_icon_override"),
                            cost_icons=dialog.get("cost_icons"),
                            cost_text_sufficient=dialog.get("cost_text_sufficient"),
                            pending_buy=bool(self.shop._pending_buy)
                                if self.shop else False)
            return json.dumps(res)
        if name == "collect_queue":
            if self.queue_box is None:
                return '{"error": "queue placement unavailable in this mode"}'
            res = self.queue_box.collect(open_chest=True)
            self.log.log("queue_collected", placed=bool(res.get("placed")),
                            opened=bool(res.get("opened")),
                            item=res.get("placed_item"),
                            cell=res.get("cell"),
                            error=res.get("error"))
            return json.dumps(res)
        if name == "collect_feat_rewards":
            if self.panels is None:
                return '{"error": "feat collection unavailable in this mode"}'
            if self._frame is None:
                return '{"error": "no frame for feat collection"}'
            try:
                res = self.panels.collect_feat_rewards(self._frame)
            except Exception as exc:
                return json.dumps({"error": f"feat collection failed: {exc}"})
            self._last_feat_collect_step = self._step_count
            panel = res.get("panel") or {}
            feats = panel.get("feats") or []
            if feats:
                self._feats_cache = {"tier": panel.get("tier"), "feats": feats,
                                     "step": self._step_count}
            collected = int(res.get("collected") or 0)
            self._last_feat_collect_found = collected > 0 or bool(res.get("tier"))
            if collected:
                self.log.log("feat_reward_collected", count=collected, via="tool")
            if res.get("tier"):
                self.log.log("tier_reward_collected", tier=panel.get("tier"), via="tool")
            return json.dumps(res)
        if name == "identify_item":
            # support BATCH identification. The `cells` parameter
            # accepts a list of [r,c] pairs; the bot popup-reads each in
            # turn and returns a list of (item_id, info) dicts. Falls back
            # to single-cell `cell:[r,c]` for back-compat with older
            # tool-call templates the model may have memorized.
            cells_arg = args.get("cells")
            cell_arg = args.get("cell")
            targets: list = []
            if isinstance(cells_arg, list) and cells_arg:
                for c in cells_arg:
                    if (isinstance(c, list) and len(c) == 2
                            and all(isinstance(v, (int, float)) for v in c)):
                        targets.append((int(c[0]), int(c[1])))
            elif isinstance(cell_arg, list) and len(cell_arg) == 2:
                targets.append((int(cell_arg[0]), int(cell_arg[1])))
            if not targets:
                return json.dumps({"error": "identify_item requires cell:[r,c] or cells:[[r,c],...]"})
            # De-dup and bounds-check; the NecroMerger station is excluded
            # (it's a fixed NPC, never popup-tapped).
            nec = necromerger_cell()
            seen: set = set()
            ordered: list = []
            for (r, c) in targets:
                if (r, c) == nec:
                    continue
                if (r, c) in seen:
                    continue
                seen.add((r, c))
                target = board.cell_at(r, c)
                if target is None:
                    continue
                if (self.classifier is not None and target.item_id
                        and self.classifier.has(target.item_id)
                        and target.score >= LABEL_MIN_SCORE):
                    # Already identified — skip (the caller is using the
                    # batch round to find UNIDs; known cells are wasted taps).
                    continue
                ordered.append(target)
            if not ordered:
                return json.dumps({"results": [], "note": "all cells already identified"})
            if not self.live or self.discover is None:
                return json.dumps({"error": "identification unavailable in this mode"})
            try:
                pairs = self.discover.identify_batch(self._frame, ordered)
            except AttributeError:
                # Back-compat: an older Identifier without `identify_batch`
                # (e.g. an old test stub) falls back to per-cell `identify`.
                pairs = []
                for target in ordered:
                    try:
                        pairs.append(self.discover(target, self._frame))
                    except Exception as exc:
                        pairs.append((f"unknown_{len(pairs)}", {"error": str(exc)}))
            except Exception as exc:
                return json.dumps({"error": str(exc)})
            results = []
            for target, (item_id, info) in zip(ordered, pairs):
                target.item_id = item_id
                target.score = 1.0
                target.margin = 1.0
                info = info or {}
                # Genuine-new-item path with two-source agreement: an id
                # whose base is NOT banked (first sighting, e.g. eyeball)
                # gets its sprite + recipe banked only when the popup OCR
                # title independently agrees with the resolved name
                # ("Eyeball" vs "eyeball"). A hallucinated LLM name against
                # a disagreeing OCR title banks nothing anywhere — the
                # board label above is step-local and evaporates on next
                # classify. Known ids keep the existing unconditional path.
                bank_ok = True
                if (self.classifier is not None
                        and not self.classifier.has(item_id)):
                    ocr_name = normalize_item_name(info.get("ocr") or "")
                    base = (item_id.split("_lvl")[0]
                            if "_lvl" in (item_id or "") else (item_id or ""))
                    bank_ok = bool(ocr_name) and (
                        ocr_name == base or base in ocr_name
                        or ocr_name in base)
                    if not bank_ok:
                        # Consensus fallback: independent popup reads that
                        # keep naming the same id (e.g. LLM says "eyeball"
                        # 3+ times while OCR mangles it as "fuehball") are
                        # themselves a second source. Without this, a new
                        # minion whose OCR is degraded can NEVER be banked
                        # (observed: 12 consecutive eyeball refusals) and
                        # the bot re-taps the popup forever.
                        bank_ok = self._unbanked_consensus(
                            target, item_id)
                        if not bank_ok:
                            self.log.log("identify_unbanked",
                                         cell=[target.row, target.col],
                                         item=item_id, ocr=info.get("ocr"))
                if bank_ok:
                    self._bank_popup_recipe(item_id, info)
                    try:
                        self._bank_unid_sprite(target, item_id)
                    except Exception:
                        pass
                results.append({
                    "item_id": item_id,
                    "cell": [target.row, target.col],
                    "merge_info": info.get("merge_info") or "",
                    "description": info.get("description") or "",
                })
            return json.dumps({"results": results})
        if name == "lookup_wiki":
            query = args.get("query")
            if not query or not str(query).strip():
                return '{"error": "lookup_wiki requires a query string"}'
            try:
                return wiki_lookup(str(query))
            except Exception as exc:
                return json.dumps({"error": f"wiki lookup failed: {exc}"})
        if name == "get_cravings":
            if not self.live or self.cravings is None:
                return '{"error": "cravings unavailable in this mode"}'
            try:
                result = self.cravings.get_cravings()
                self._track_cravings(result)
                return json.dumps(result)
            except BoardLostError:
                raise  # game likely exited — halt the loop, don't fall back
            except Exception as exc:
                return json.dumps({"error": f"cravings read failed: {exc}"})
        if name == "get_champions":
            if not self.live or self.champions is None:
                return '{"error": "champion read unavailable in this mode"}'
            try:
                result = self.champions.get_champion_status()
                self._track_champions(result)
                return json.dumps(result)
            except BoardLostError:
                raise  # game likely exited — halt the loop, don't fall back
            except Exception as exc:
                return json.dumps({"error": f"champion read failed: {exc}"})
        if name == "tap_button":
            if not self.live or self.panels is None:
                return '{"error": "panels unavailable in this mode"}'
            button = args.get("button")
            if not button or button not in ("feats", "station"):
                return json.dumps({"error": "tap_button requires button in [feats, station] "
                                            "(queue is a placement action; spellbook/shop are locked)"})
            try:
                result = self.panels.tap_button(self._frame, button)
                return json.dumps(result)
            except BoardLostError:
                raise  # game likely exited — halt the loop, don't fall back
            except Exception as exc:
                return json.dumps({"error": f"panel read failed: {exc}"})
        return json.dumps({"error": f"unknown tool {name}"})

    def _auto_identify_unidentified(self, board, move) -> bool:
        """Escape-hatch rescue for rejected moves: popup-identify the move's
        cells (live only) so a genuinely-valid merge on unidentified cells can
        pass validation instead of being rejected forever.

        The drive LLM proposes a move straight from the screenshot; when the
        template bank has no label for a cell, validation must reject it
        (cell_a/cell_b_unidentified) — but the sprite may genuinely be an
        identical pair (e.g. two ribs). Popping open the item popup gives us
        ground-truth ids (banked as templates), and the move then validates.
        Returns True if the move now passes the validator after identification.
        """
        if not self.live or self.discover is None:
            return False
        for cell in (move.cell_a, move.cell_b):
            if not cell:
                continue
            r, c = cell
            if (r, c) == necromerger_cell():
                continue
            target = board.cell_at(r, c)
            if target is None or target.item_id is not None:
                continue
            if not target.occupied:
                continue
            try:
                item_id, info = self.discover(target, self._frame)
            except Exception:
                return False
            if not item_id:
                return False
            target.item_id = item_id
            target.score = 1.0
            target.margin = 1.0
            self._bank_popup_recipe(item_id, info)
            self.log.log("vision_drive", ok=False, reason="auto_identify",
                            cell=[r, c], item=item_id)
        return True

    def _proactive_identify(self, messages: list[dict], board) -> None:
        """identify up to MAX_DISCOVERY UNID cells per step, before
        the optional-tools round runs.

        The 18 `discover_unknown` events in the Aug 28 session log were the
        dominant failure mode — the model tried to merge/feed unidentified
        cells and got `cell_a/cell_b_unidentified` rejections. By running
        a single batch identify_item call up-front, we bank every UNID
        cell's template before the model even sees the action prompt. The
        model's own identify_item calls later in the optional-tools round
        are still allowed (this is a safety net, not a cap).

        Live only; no-op in dry-run mode. If `identify_batch` isn't
        available on the Identifier (e.g. an old test stub), falls back
        to per-cell `identify`.
        """
        if not self.live or self.discover is None:
            return
        unids = self._unidentified_cells(board)
        if not unids:
            return
        # Prioritize UNIDs that can act THIS step: cells sharing a family
        # hint (labeled id or runner-up guess) with another UNID are likely
        # a mergeable pair once identified — identifying a lone UNID banks
        # a template but rarely unblocks a move. Board order breaks ties.
        # (Cap still MAX_DISCOVERY; the optional round can do more.)
        from collections import Counter
        hints = Counter()
        for (r, c) in unids:
            t = board.cell_at(r, c)
            h = (t.item_id or t.runner_up_id) if t is not None else None
            if h:
                hints[h] += 1

        def _prio(rc) -> tuple:
            t = board.cell_at(rc[0], rc[1])
            h = (t.item_id or t.runner_up_id) if t is not None else None
            return (0 if h and hints[h] >= 2 else 1, rc[0], rc[1])

        # Cap at MAX_DISCOVERY cells per step (the model can call more
        # in the optional-tools round if it wants).
        targets = []
        for (r, c) in sorted(unids, key=_prio):
            if (r, c) == necromerger_cell():
                continue
            target = board.cell_at(r, c)
            if target is None:
                continue
            if target.item_id is not None:
                continue
            targets.append(target)
            if len(targets) >= MAX_DISCOVERY:
                break
        if not targets:
            return
        # Build a synthetic tool call for the batch identify.
        synth = {
            "type": "function",
            "id": f"proactive_identify_{int(time.time() * 1000)}",
            "function": {
                "name": "identify_item",
                "arguments": json.dumps(
                    {"cells": [[t.row, t.col] for t in targets]}),
            },
        }
        try:
            result = self._exec_tool(synth, board)
        except Exception as exc:
            self.log.log("vision_drive", ok=False, reason="proactive_identify_error",
                            detail=str(exc))
            return
        messages.append(self._tool_call_msg(synth))
        messages.append({"role": "tool", "tool_call_id": synth["id"],
                            "content": result})
        # Log the result for telemetry (one event per batch, with the count).
        try:
            data = json.loads(result)
            n = len(data.get("results") or [])
        except Exception:
            n = 0
        self.log.log("proactive_identify", count=n,
                        cells=[[t.row, t.col] for t in targets])

    def _track_cravings(self, result: dict) -> None:
        """Log craving progress/completion events from a get_cravings result.

        Every observed craving state logs a `craving` event (item, level,
        count_done/count_required, reward); a completed craving (count_done >=
        count_required) additionally logs `craving_complete` with its reward,
        so reward-food gains are fully tracked in session.jsonl.
        """
        if not isinstance(result, dict):
            return
        for c in result.get("cravings") or []:
            if not isinstance(c, dict):
                continue
            item = c.get("item")
            done = c.get("count_done")
            required = c.get("count_required")
            if done is None or required is None:
                continue
            if item:
                item = normalize_item_name(item)   # 'rib cage' -> 'ribcage' (matches board ids)
                self.fallback.craved_item = item   # heuristic feeds the craving too
                self._craving_cache = {
                    "item": item,
                    "level": c.get("level"),
                    "count_done": done,
                    "count_required": required,
                    "reward": c.get("reward"),
                    "step": self._step_count,
                }
                # Level key for cache freshness: a menu read that lands a
                # different level for the same family (e.g. skeleton_lvl1's
                # craving replaced by skeleton_lvl2) must invalidate the old
                # belief immediately, not just via the refresh age.
                if (self._last_craving_level is not None
                        and c.get("level") != self._last_craving_level):
                    self.log.log("craving_level_changed",
                                    previous=self._last_craving_level,
                                    level=c.get("level"), item=item)
                self._last_craving_level = c.get("level")
            self.log.log("craving", item=item, level=c.get("level"),
                            count_done=done, count_required=required,
                            reward=c.get("reward"))
            if done >= required:
                self.log.log("craving_complete", item=item,
                                reward=c.get("reward"), count_done=done,
                                count_required=required)
                # A completed craving is stale the moment it's read — the game
                # has (or is about to pick) a NEW craving. Drop the cache so
                # the next step re-offers get_cravings and learns it instead of
                # believing this finished one for the cache lifetime.
                if self._craving_cache is not None \
                        and self._craving_cache.get("item") == item:
                    self._craving_cache = None

    def _track_champions(self, result: dict) -> None:
        """Cache + log champion spawn state from a get_champions result.

        Caches the read for CHAMPIONS_REFRESH_STEPS so `get_board_state` can
        render it without re-opening the (intrusive) champion screen. Logs a
        `champion` event per champion (name, active, progress, ready) and a
        `champion_spawned` event when a champion flips to ready/spawned."""
        if not isinstance(result, dict):
            return
        champions = result.get("champions") or []
        self._champion_cache = {"champions": champions,
                                "step": self._step_count}
        for c in champions:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            if not name:
                continue
            self.log.log("champion", name=name, active=c.get("active"),
                            progress=c.get("progress"), ready=c.get("ready"))
            if c.get("ready"):
                self.log.log("champion_spawned", name=name,
                                progress=c.get("progress"))

    def _valid_learning_texts(self, texts: list[dict]) -> list[dict]:
        """Drop candidate learnings that reference the bot's own behavior or
        internals (vision_drive failures, heuristic fallback, validation
        rejections, tool calls, spawn/discovery logic). These are NOT game
        mechanics and pollute the memory store. Mechanical backstop for the
        summary-prompt rules; the wiki fact-check cannot be trusted to catch
        meta-noise (it backs plausible-sounding claims)."""
        bot_internal = (
            "vision_drive", "heuristic", "fallback", "validate", "rejected",
            "tool call", "identify_item", "get_board_state", "lookup_wiki",
            "accumulate", "planner", "board state", "retry", "soak",
            "no_tool_call", "the bot", "the agent", "the tool",
            # A merge_noop is a bot/emulator gesture failure, not a game rule:
            # merging two identical items ALWAYS works in NecroMerger. Blocking
            # the term stops the summary model from "learning" bogus rules like
            # "merging bone+bone is a no-op" (seen in learnings.md).
            "merge_noop", "no-op", "no op", "does nothing",
        )
        kept = []
        for t in texts or []:
            if isinstance(t, dict):
                text = t.get("text", "")
            else:
                text = str(t)
            low = text.lower()
            if any(k in low for k in bot_internal):
                continue
            # Ensure it's a dict with required fields
            if isinstance(t, dict):
                kept.append(t)
            else:
                kept.append({"text": text, "type": "pattern", "confidence": 0.7})
        return kept

    # validator reason-key prefixes -> human-readable game-rule learnings.
    # These are grounded constraints (not bot behavior), so they bypass the
    # `_valid_learning_texts` meta-noise filter and persist into the prompt.
    _REJECTION_RULE_TEXTS = {
        "feed_not_max_level": (
            "Don't feed a Rune or Coin stack below max level to the Devourer. "
            "Merge it to max level first to collect the full currency grant."),
        "feed_merge_material": (
            "Don't feed a mid-chain merge component (bone, ribcage, or an "
            "item that can still merge up) while a non-mergeable feedable "
            "exists — merge-material exists to build merges, feeding it "
            "destroys pairs."),
        "feed_strategy_material": (
            "Even when the board is full of merge-material and a plain "
            "feedable is scarce, don't feed a component on the ACTIVE "
            "STRATEGY's path (e.g. a zombie near the needed level for 'Own a "
            "lvl 3+ Zombie'). Feed a non-goal component instead — destroying "
            "the merge that completes the strategy wastes more than it feeds."),
        "feed_is_station": (
            "Never feed a Station to the Devourer — stations are board "
            "fixtures, not food."),
        "feed_is_champion": (
            "Never feed a Champion to the Devourer. A Champion is defeated by "
            "dragging a creature onto it, not by feeding."),
        "feed_mana_overflow": (
            "Feeding a Mana Potion while the Mana bar is full is wasted — "
            "collect or spend Mana first."),
        "feed_overflow": (
            "Feeding food larger than the Devourer's remaining Satiety "
            "(plus tolerance) wastes the overflow — use a smaller feed."),
        "feed_generous": (
            "Don't feed expensive high-level creatures unless they're craved "
            "or nothing cheaper fits — they're Mana generators."),
        "merge_mismatch": (
            "Only merge two IDENTICAL items (same family and same level). "
            "Different levels or families never merge."),
        "merge_max_level": (
            "An item at its max level cannot be merged — feed it or use it."),
        "merge_low_margin": (
            "Only merge cells whose sprite match is confident; a low-margin "
            "pair is likely mislabeled and the merge would be refused."),
        "merge_is_champion": (
            "Never merge a Champion with anything — Champions are unique "
            "enemies, not merge pieces."),
        "spawn_not_grave": (
            "Only a Grave, a Chest, or a Supply Cupboard can spawn — "
            "spawning from any other cell type no-ops."),
        "spawn_no_room": (
            "Cannot spawn onto a full board — merge or feed first to free a "
            "cell."),
        "spawn_low_mana": (
            "A Grave spawn needs the Mana bar above the minimum threshold — "
            "wait or collect Mana before spawning. "
            + _resource_teaching("mana")),
        "spawn_no_slime": (
            "A Supply Cupboard spawn needs Slime in the vat — spawning on "
            "empty Slime silently no-ops. Merge/feed instead until Slime "
            "regenerates. " + _resource_teaching("slime")),
        "feed_mana": (
            "Never feed a Mana item for Satiety — it grants Mana instead."),
        "attack_no_damage": (
            "Only creatures with a known damage value can attack a Champion."),
    }

    def _rejection_reasons_to_learnings(self) -> int:
        """Fold this window's validator rejection reasons into learnings.md
        as grounded game-rule facts.

        Reads `rejection` events logged since the last summarize, maps each
        reason-key prefix to a readable rule via `_REJECTION_RULE_TEXTS`, and
        `append_learning`s the unique ones as committed facts. Appending
        directly (rather than via the summary model) sidesteps the
        `_valid_learning_texts` meta-noise filter, which deliberately strips
        "rejected"/"validate" learnings. Returns how many were added. Repeats
        are deduped by `append_learning`'s `_matches`, so a reason only lands
        once."""
        window = [e for e in self.log.events
                  if e.get("t", 0) >= self._last_summary_t
                  and e.get("event") == "rejection"]
        if not window:
            return 0
        added = 0
        seen: set[str] = set()
        for e in window:
            reason = (e.get("reason") or "").split(":", 1)[0]
            rule = self._REJECTION_RULE_TEXTS.get(reason)
            if not rule or rule in seen:
                continue
            seen.add(rule)
            added += append_learning(
                self.learnings_path,
                [{"text": rule, "type": "fact", "confidence": 1.0}],
                title=f"{datetime.now().strftime('%b %d, %Y %H:%M')} (rejection rule)")
        return added

    def _reward_learnings(self) -> int:
        """Fold this window's observed GOAL COMPLETIONS into learnings.md as
        high-confidence success patterns — the RL positive-reward analog.

        Mirrors `_rejection_reasons_to_learnings` (which persists the *negative*
        signal of a rejection) but for the *positive* side: when a feat reward
        is collected, a craving completes, or a champion is defeated/spawned,
        we append a `type="pattern", confidence=0.9` learning crediting the
        action family that led to it. Persisting these directly (bypassing the
        summary model) keeps positive reinforcement deterministic and immune to
        `_valid_learning_texts` meta-noise stripping. Returns how many were
        added; repeats are deduped by `append_learning`'s `_matches`."""
        if not hasattr(self, "log") or self.log is None:
            return 0
        installed = (
            ("feat_reward_collected",
             "Completing and COLLECTING a feat's reward advances the board goal. "
             "Work toward the actively-listed feat (spawn/merge its target, buy "
             "its station) so the reward button enables — collecting it banks "
             "the progress."),
            ("craving_complete",
             "A Devourer craving was fulfilled to count_done >= count_required. "
             "Keep spawning/merging/collecting while feeding the craved creature "
             "so the craving completes and the Devourer levels up."),
            ("champion_spawned",
             "A champion spawned on the board. Because they cap fed progress, "
             "prioritize dragging your highest-damage creature onto the "
             "champion when it appears to remove it."),
        )
        window = [e for e in self.log.events
                  if e.get("t", 0) >= self._last_summary_t
                  and e.get("event") in ("feat_reward_collected",
                                         "craving_complete",
                                         "champion_spawned")]
        if not window:
            return 0
        added = 0
        seen: set[str] = set()
        for e in window:
            event = e.get("event")
            reward = next((r for k, r in installed if k == event), None)
            if not reward or reward in seen:
                continue
            seen.add(reward)
            added += append_learning(
                self.learnings_path,
                [{"text": reward, "type": "pattern", "confidence": 0.9}],
                title=f"{datetime.now().strftime('%b %d, %Y %H:%M')} (goal reward)")
        return added

    def _dedup_existing_learnings(self) -> None:
        """one-time startup dedup of an existing noisy learnings.md.

        The append_learning dedup (substring + Jaccard in `_matches`) only
        fires for NEW entries, so STORED entries accumulate paraphrases of
        the same fact (e.g. 14+ "Rib Cages can be merged to summon a
        Skeleton" entries in different wordings). At startup we collapse
        near-duplicates (token-set Jaccard >= 0.7) and keep the
        most-recently-confirmed entry. Cheap (~160 entries takes <1s) and
        self-correcting: if the model writes the same fact again later, it
        gets re-added through the normal summarize path.
        """
        try:
            from planner.learnings import read, write as _write_learnings, Learning as _Learning
            from planner.learnings import _norm_tokens
        except ImportError:
            return
        if not self.learnings_path.exists():
            return
        try:
            entries = read(self.learnings_path)
        except Exception:
            return
        if len(entries) <= 1:
            return
        # Build a token set per entry
        token_sets = [_norm_tokens(e.text) for e in entries]
        # Walk through; for each entry, find all later entries with
        # token Jaccard >= 0.7. Keep the FIRST match (earliest stored);
        # the later one is the duplicate to drop.
        keep_idx = set(range(len(entries)))
        for i in range(len(entries)):
            if i not in keep_idx:
                continue
            for j in range(i + 1, len(entries)):
                if j not in keep_idx:
                    continue
                a, b = token_sets[i], token_sets[j]
                if not a or not b:
                    continue
                small, large = (a, b) if len(a) <= len(b) else (b, a)
                if len(small & large) / len(small) >= 0.7:
                    keep_idx.discard(j)
        kept = [entries[i] for i in sorted(keep_idx)]
        if len(kept) < len(entries):
            try:
                _write_learnings(self.learnings_path, kept)
            except Exception:
                pass

    def _valid_glossary_items(self, items: list[str]) -> list[str]:
        """Keep only summary `items` entries whose leading id/name is a real
        template-bank id or family (or a recipe keyword like `merge:`/`description:`).
        The summary model invents ids (necromerger_lvl1, valuablechest_lvl1, ...)
        and display names that don't map to the bank; those are dropped so wrong
        names never re-enter the glossary. Display names are normalized (spaces
        removed, lowercased, optional "the " stripped) and accepted when they
        match or prefix a bank id family (e.g. "Poison Rune" -> poisonrune_lvl1)."""
        if not items or self.classifier is None:
            return items
        recipe_keys = ("merge", "description")
        bank_ids = set(self.classifier.templates)
        kept = []
        for item in items:
            if self._parse_chain_line(item) is not None:
                if self._valid_chain_line(item):
                    kept.append(item)
                continue
            m = re.match(r"^\s*[-*]?\s*([a-z0-9_]+)\s*:", item)
            if m and m.group(1) not in recipe_keys \
                    and not self.classifier.has(m.group(1)):
                continue
            if m and m.group(1) in recipe_keys:
                kept.append(item)
                continue
            name = re.match(r"^\s*[-*]?\s*([A-Za-z][A-Za-z0-9 ]*?)\s*:", item)
            if name:
                norm = re.sub(r"\s+", "", name.group(1)).lower()
                norm = norm[4:] if norm.startswith("the") else norm
                if not any(norm == iid or iid.startswith(norm) for iid in bank_ids):
                    continue
            # Bare-id lines ("ribcage", "- bone") are board snapshots, not
            # facts: the Aug 23 prune removed 41 such timestamped blocks as
            # pure noise, so they no longer re-enter the glossary. Only lines
            # carrying a fact (`key: value`) or a chain line survive.
            if not (m or name) and not re.search(r":", item):
                continue
            kept.append(item)
        return kept

    @staticmethod
    def _base_name(item_id: str) -> str:
        """Strip `_lvl<N>` suffix -> the item's base name. The legacy `__alt<N>`
        suffix is no longer minted (Aug 27) but the regex still strips it
        defensively in case any old ids linger in the bank or glossary."""
        return re.sub(r"(?:_lvl\d+)?(?:_alt\d+)?$", "", item_id or "")

    @staticmethod
    def _parse_merge_target(merge_info: str) -> str | None:
        """Best-effort target item token from a popup merge_info line.

        Handles the two observed forms: "two Bone -> Ribcage" and "Merge to
        summon a higher lvl Skeleton". Returns a sanitized lowercase token
        (e.g. "ribcage", "skeleton") or None when no target is parseable.
        """
        if not merge_info:
            return None
        t = (merge_info or "").strip().lower()
        m = re.search(r"->\s*([a-z][a-z ]*?)\s*\.?\s*$", t)
        target = m.group(1) if m else None
        if not target:
            m = re.search(r"(?:summon|create|make)\s+(?:a|an\s+)?([a-z][a-z ]*?)\s*\.?\s*$", t)
            target = m.group(1) if m else None
        if not target:
            return None
        target = re.sub(r"[^a-z ]", "", target).strip()
        target = re.sub(r"\s+", " ", target).strip()
        if not target:
            return None
        # drop level-relative phrasing ("skeleton of the next level" -> "skeleton")
        target = re.sub(r"\s+(?:of|at|next|higher)\s+(?:the\s+)?(?:next|higher)?\s*level.*$", "", target).strip()
        return target or None

    def _bank_popup_recipe(self, item_id: str, info: dict) -> None:
        """Bank a merge-chain recipe read from the item's info popup into the
        glossary. This is grounded game-UI text (not a model guess), so it's a
        safe `items`-style entry. Title = item_id makes it prunable via the
        summary's `remove_items`.

        The raw popup text is kept verbatim (`merge: <text>`); a canonical
        `edge: <base> -> <target>` line is added when parseable so the summary
        can later consolidate links into an explicit per-family chain. The
        popup's Feed stat is banked as `feed value: N` and its Takes-Damage
        stat as `damage value: N` (the amount dealt if dropped on a Champion).
        For an item whose popup block already exists, only a NEWLY-observed
        feed/damage stat is added (via `ensure_glossary_bullet`) — existing
        merge/description facts are never overwritten.

        parse the popup's `spawn_outputs` and `spawn_rates` fields
        (comma-separated lists of item names and integer percentages in
        spawn order). These are station-only facts (graves, iceboxes,
        manapools) and are stored on a per-level block, NOT a single
        family-level block — grave_lvl1, grave_lvl2, and grave_lvl3 all
        have different spawn rates, and reading grave_lvl3 must not
        overwrite the grave_lvl1 entry. The block title is still
        `{item_id} (popup)` (per-level) so `_popup_banked` returns True
        for any grave level, but the per-level rates live INSIDE the
        same block title as separate lines.
        """
        info = info or {}
        item_id = item_id or ""
        merge = (info.get("merge_info") or "").strip()
        desc = (info.get("description") or "").strip()
        feed_value = self._stat(info.get("feed_value"))
        damage = self._stat(info.get("damage"))
        spawn_outputs = (info.get("spawn_outputs") or "").strip()
        spawn_rates = (info.get("spawn_rates") or "").strip()
        uses_remaining = self._stat(info.get("uses_remaining"))
        if not item_id:
            return
        if self.classifier is not None and not self.classifier.has(item_id):
            return  # not a real template-bank id (reject model-invented ids)

        # Knowledge directory for new glossary format
        from planner.glossary import _get_knowledge_dir
        kd = _get_knowledge_dir(self.glossary_path) if hasattr(self, 'glossary_path') else None

        # Validate spawn_outputs vs spawn_rates (count match, rates sum to 100).
        # The model occasionally emits a 0-element spawn list with rates; we
        # drop the rate line in that case.
        rates_list = []
        if spawn_rates:
            try:
                rates_list = [int(x.strip()) for x in spawn_rates.split(",") if x.strip()]
            except (TypeError, ValueError):
                rates_list = []
        outputs_list = [x.strip() for x in spawn_outputs.split(",") if x.strip()]             if spawn_outputs else []
        if outputs_list and rates_list and len(outputs_list) != len(rates_list):
            # Mismatched lengths: drop the rates, keep the outputs as a hint
            rates_list = []
        spawn_rate_total = sum(rates_list) if rates_list else 0
        if spawn_rate_total != 100 and rates_list:
            # Off by rounding/etc. (e.g. 60+40=100, but 33+33+33=99). Drop
            # the rates since the model wasn't confident enough to sum to 100.
            rates_list = []

        if self._popup_banked(item_id):
            # Already known — only a newly-read feed/damage stat can (and
            # should) be added to the existing block (blanket backfill for
            # items that were popup-read before the stats were captured).
            if feed_value is not None and feed_value > 0                     and not self._popup_has_fact(item_id, "feed value"):
                write_item_stat(item_id, feed=feed_value, source="popup", knowledge_dir=kd)
            if damage is not None and damage > 0                     and not self._popup_has_fact(item_id, "damage value"):
                write_item_stat(item_id, damage=damage, source="popup", knowledge_dir=kd)
                # also append spawn_outputs / spawn_rates if newly
            # observed for THIS level. Multiple reads at different levels
            # (grave_lvl1, grave_lvl2, grave_lvl3) all share the same block
            # title (item_id) but the spawn_rate line gets the level
            # prefix so they don't overwrite each other.
            if outputs_list:
                # item_id already contains the level (e.g., "grave_lvl3")
                station_id = item_id
                targets = outputs_list
                write_spawn_rate(station_id, targets, uses=5, source="popup")
            if uses_remaining is not None:
                write_visual_marker(item_id, merge_edges=[f"uses remaining: {uses_remaining}"], source="popup", knowledge_dir=kd)
            return

        # Write to new typed glossary
        if merge:
            target = self._parse_merge_target(merge)
            if target:
                write_visual_marker(item_id, merge_edges=[f"{self._base_name(item_id)} -> {target}"], source="popup", knowledge_dir=kd)
        if desc and desc != merge:
            write_visual_marker(item_id, description=desc, source="popup", knowledge_dir=kd)
        # A popup with NO merge line and a feed instruction means this is the
        # item's TOP level: it can never merge again, only be fed. Mark it so
        # the board renderer + validator refuse a bogus max-level merge.
        # The description itself may mention merging ("Merge to create a
        # bigger pile or feed to the Devourer.") — that wording means the
        # item IS mergeable; only a feed-ONLY description is max-level
        # (Aug 25: icerune_lvl1/2 + poisonrune_lvl1 were wrongly flagged from
        # exactly such descriptions, which blocked all rune merging).
        if not merge and any(w in desc.lower() for w in ("feed", "devourer"))                 and not any(w in desc.lower() for w in
                            ("merge", "bigger pile", "combine")):
            write_item_stat(item_id, max_level=True, source="popup", knowledge_dir=kd)
        if feed_value is not None and feed_value > 0:
            write_item_stat(item_id, feed=feed_value, source="popup", knowledge_dir=kd)
        if damage is not None and damage > 0:
            write_item_stat(item_id, damage=damage, source="popup", knowledge_dir=kd)
            # bank the spawn outputs and rates as a per-level entry.
        if outputs_list:
            # item_id already contains the level (e.g., "grave_lvl3")
            station_id = item_id
            targets = outputs_list
            write_spawn_rate(station_id, targets, uses=5, source="popup", knowledge_dir=kd)
        if uses_remaining is not None:
            write_visual_marker(item_id, merge_edges=[f"uses remaining: {uses_remaining}"], source="popup", knowledge_dir=kd)

    def _popup_has_bullet(self, item_id: str, bullet: str) -> bool:
        """True if the popup glossary block for `item_id` already contains
        the literal `bullet` text on any of its lines. Used to detect
        per-level spawn-rate bullets like `lvl3 spawn: bone, ribcage, zombie (40, 30, 30)`
        so a re-read of the same level doesn't add a duplicate entry."""
        if not self.glossary_path.exists():
            return False
        text = self.glossary_path.read_text()
        for m in re.finditer(rf"## Item {re.escape(item_id)} \(popup\)\n(.*?)(?=\n## |\Z)",
                                "\n" + text, re.S):
            if bullet in m.group(1):
                return True
        return False

    @staticmethod
    def _stat(value):
        """Int-coerce a popup stat (feed_value / damage) the 4B model may emit
        as a float or junk; negative/zero values are dropped by the caller."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _popup_has_fact(self, item_id: str, label: str) -> bool:
        """True if this item's `(popup)` glossary block already records a
        numeric stat line like `feed value: N` / `damage value: N`."""
        if not self.glossary_path.exists():
            return False
        text = self.glossary_path.read_text()
        for m in re.finditer(rf"## Item {re.escape(item_id)} \(popup\)\n(.*?)(?=\n## |\Z)",
                                "\n" + text, re.S):
            if re.search(rf"-\s*{re.escape(label)}\s*:\s*\d+", m.group(1)):
                return True
        return False

    def _popup_has_feed(self, item_id: str) -> bool:
        """Back-compat: is a numeric `feed value:` recorded for this item?"""
        return self._popup_has_fact(item_id, "feed value")

    def _feed_values(self) -> dict[str, int]:
        """Template id -> numeric feed value (how much Food the Devourer gains
        when that item is fed), from `feed value:` facts. Precedence: the
        planner's own knowledge dir (live runs bank here) > its own markdown
        glossary (tests + legacy saves own their file) > the default JSON
        dir, but ONLY when default-configured (otherwise temp-path tests
        would leak live data)."""
        try:
            from planner.glossary import read_feed_values, _get_knowledge_dir, DEFAULT_PATH
            kd = _get_knowledge_dir(self.glossary_path) if self.glossary_path else None
            if kd is not None and kd.exists():
                vals = read_feed_values(kd)
                if vals:
                    return vals
            if self.glossary_path is not None and Path(self.glossary_path).exists():
                text = Path(self.glossary_path).read_text()
                values: dict[str, int] = {}
                for m in re.finditer(r"## Item ([a-z0-9_]+) \(popup\)\n(.*?)(?=\n## |\Z)",
                                        "\n" + text, re.S):
                    fm = re.search(r"-\s*feed value\s*:\s*(\d+)", m.group(2))
                    if fm:
                        values[m.group(1)] = int(fm.group(1))
                return values
            if self.glossary_path is not None and Path(self.glossary_path) == DEFAULT_PATH:
                vals = read_feed_values()
                if vals:
                    return vals
        except Exception:
            pass
        return {}

    def _damage_values(self) -> dict[str, int]:
        """Template id -> numeric Damage value from `damage value:` facts
        (the "Takes Damage" column — damage dealt if dropped on a Champion).
        Same precedence as `_feed_values`: own knowledge dir > own markdown
        > default JSON only when default-configured. Items without a
        recorded damage stat are absent."""
        try:
            from planner.glossary import read_damage_values, _get_knowledge_dir, DEFAULT_PATH
            kd = _get_knowledge_dir(self.glossary_path) if self.glossary_path else None
            if kd is not None and kd.exists():
                vals = read_damage_values(kd)
                if vals:
                    return vals
            if self.glossary_path is not None and Path(self.glossary_path).exists():
                text = Path(self.glossary_path).read_text()
                values: dict[str, int] = {}
                for m in re.finditer(r"## Item ([a-z0-9_]+) \(popup\)\n(.*?)(?=\n## |\Z)",
                                        "\n" + text, re.S):
                    fm = re.search(r"-\s*damage value\s*:\s*(\d+)", m.group(2))
                    if fm:
                        values[m.group(1)] = int(fm.group(1))
                return values
            if self.glossary_path is not None and Path(self.glossary_path) == DEFAULT_PATH:
                vals = read_damage_values()
                if vals:
                    return vals
        except Exception:
            pass
        return {}

    def _per_level_spawn_outputs(self, item_id: str) -> tuple[list[str], list[int]] | None:
        """Look up the per-level spawn outputs and rates for `item_id` from
        the new JSON glossary (spawn_rates.json).

        Returns `(outputs, rates)` lists, or None if no spawn data exists.
        """
        from planner.glossary import read_spawn_rates, _get_knowledge_dir
        kd = _get_knowledge_dir(self.glossary_path) if hasattr(self, 'glossary_path') else None
        spawn_data = read_spawn_rates(kd).get(item_id)
        if not spawn_data:
            return None
        targets = spawn_data.get("targets", [])
        if not targets:
            return None
        # The JSON stores targets as a list; rates are not stored separately
        # in the new format, so we return empty rates for compatibility.
        # The _best_spawn_line uses the targets list directly.
        return targets, []

    def _popup_banked(self, item_id: str) -> bool:
        """True if a popup entry for this item id already exists in the
        new JSON glossary (item_stats.json or visual_markers.json)."""
        from planner.glossary import read_item_stats, read_visual_markers, _get_knowledge_dir
        kd = _get_knowledge_dir(self.glossary_path) if hasattr(self, 'glossary_path') else None
        stats = read_item_stats(kd)
        markers = read_visual_markers(kd)
        return item_id in stats or item_id in markers

    def _max_level_ids(self) -> set[str]:
        """Template ids at max level (popup says only "Feed to the Devourer.",
        no merge line). Same precedence as `_feed_values`: own knowledge dir
        > own markdown > default JSON only when default-configured. An empty
        set here disables EVERY max-level gate (merge_max_level, income
        whitelist, feed ranking), so the JSON fallback is load-bearing, not
        cosmetic."""
        try:
            from planner.glossary import read_item_stats, _get_knowledge_dir, DEFAULT_PATH
            kd = _get_knowledge_dir(self.glossary_path) if self.glossary_path else None
            if kd is not None and kd.exists():
                stats = read_item_stats(kd)
                ids = {k for k, v in stats.items() if v.get("max_level")}
                if ids:
                    return ids
            if self.glossary_path is not None and Path(self.glossary_path).exists():
                text = Path(self.glossary_path).read_text()
                ids = set()
                # Glossary blocks look like: "## Item <id> (popup)\n- max_level: true"
                for m in re.finditer(r"## Item ([a-z0-9_]+) \(popup\)\n(.*?)(?=\n## |\Z)",
                                        "\n" + text, re.S):
                    if "max_level: true" in m.group(2):
                        ids.add(m.group(1))
                return ids
            if self.glossary_path is not None and Path(self.glossary_path) == DEFAULT_PATH:
                stats = read_item_stats()
                ids = {k for k, v in stats.items() if v.get("max_level")}
                if ids:
                    return ids
        except Exception:
            pass
        return set()

    @staticmethod
    def _parse_chain_line(line: str):
        """Parse a consolidated chain line from the summary.

        Expected format (emitted by the summary model):
            `- merge chain skeleton: bone -> ribcage -> skeleton_lvl1 -> skeleton_lvl2`
        Returns (family, [id, ...]) or None if the line isn't a chain line.
        """
        m = re.match(r"^\s*[-*]?\s*merge chain\s+([a-z0-9_]+)\s*:\s*(.+)$", line or "")
        if not m:
            return None
        family = m.group(1).strip()
        ids = [i.strip() for i in re.split(r"\s*->\s*", m.group(2)) if i.strip()]
        if not ids or not family:
            return None
        return family, ids

    def _valid_chain_line(self, line: str) -> bool:
        """A chain line passes only when every hop id is a real template-bank
        id (or a banked family prefix), so invented chain links can't enter the
        glossary."""
        parsed = self._parse_chain_line(line)
        if parsed is None:
            return False
        _family, ids = parsed
        if self.classifier is None:
            return True
        bank = set(self.classifier.templates)
        for i in ids:
            if i in bank:
                continue
            # allow family-level entries that prefix a bank id (e.g. "skeleton"
            # when the bank has skeleton_lvl1..5) — the summary consolidates
            # component names (bone/ribcage) that are banked as their own ids.
            if any(b == i or b.startswith(i + "_") for b in bank):
                continue
            return False
        return True

    @staticmethod
    def _parse_spawn_line(line: str):
        """Parse a spawn edge line from the summary.

        Expected format:
            `- spawn chain icebox: icebox_unopened -> icerune_lvl1 (60%), icerune_lvl2 (20%), poisonrune_lvl1 (20%)`
            `- spawn chain grave: grave_lvl1 -> bone`
        Returns (family, raw) or None.
        """
        m = re.match(r"^\s*[-*]?\s*spawn chain\s+([a-z0-9_]+)\s*:\s*(.+)$", line or "")
        if not m:
            return None
        family = m.group(1).strip()
        raw = m.group(2).strip()
        if not family or not raw:
            return None
        return family, raw

    def _valid_spawn_line(self, line: str) -> bool:
        """A spawn line passes when the station source looks plausible."""
        parsed = self._parse_spawn_line(line)
        if parsed is None:
            return False
        family, raw = parsed
        # at least "source -> target" present
        if "->" not in raw:
            return False
        # source should mention family (icebox/grave) — loose check
        src = raw.split("->")[0].strip().lower()
        return family in src or src.startswith(family[:4])

    def _write_spawn_blocks(self, lines: list[str]) -> int:
        """Write per-station spawn blocks, replacing any existing `(spawn)` block
        for the same family. Returns how many blocks were written."""
        written = 0
        seen: set[str] = set()
        for line in lines:
            parsed = self._parse_spawn_line(line)
            if parsed is None or not self._valid_spawn_line(line):
                continue
            family, raw = parsed
            if family in seen:
                continue
            seen.add(family)
            prune_glossary_by_ids([f"{family} (spawn)"], protected=False)
            # store raw spawn chain plus default uses (5 for icebox, infinite for grave)
            uses_val = 5 if family == "icebox" else None
            # try to preserve explicit uses from line if present ("uses: N")
            um = re.search(r"uses\s*:\s*(\w+)", line, re.I)
            if um:
                u = um.group(1).lower()
                uses_val = int(u) if u.isdigit() else None
            # Parse targets from raw
            parts = re.split(r"\s*->\s*", raw, maxsplit=1)
            source = parts[0].strip() if parts else family
            targets = []
            if len(parts) > 1:
                for t in re.split(r",\s*", parts[1]):
                    tid = re.sub(r"\s*\(.*?\)", "", t).strip()
                    if tid:
                        targets.append(tid)
            from planner.glossary import _get_knowledge_dir
            kd = _get_knowledge_dir(self.glossary_path) if hasattr(self, 'glossary_path') else None
            write_spawn_rate(source, targets, uses=uses_val, source="popup", knowledge_dir=kd)
            written += 1
        return written

    def _write_chain_blocks(self, lines: list[str]) -> int:
        """Write consolidated per-family chain blocks, replacing any existing
        `(chain)` block for the same family (last one wins). Returns how
        many blocks were written."""
        written = 0
        for line in lines:
            parsed = self._parse_chain_line(line)
            if parsed is None or not self._valid_chain_line(line):
                continue
            family, ids = parsed
            # Replacement is code-controlled: bypass the remove_items
            # protection so the family's OLD chain block is actually removed.
            prune_glossary_by_ids([f"{family} (chain)"], protected=False)
            from planner.glossary import _get_knowledge_dir
            kd = _get_knowledge_dir(self.glossary_path) if hasattr(self, 'glossary_path') else None
            write_merge_chain(family, ids, source="popup", knowledge_dir=kd)
            written += 1
        return written

    def _board_state_text(self, board) -> str:
        """Render the classified board for the get_board_state tool.

        Only template-bank ids at/above LABEL_MIN_SCORE are shown as known;
        everything else is reported as unidentified so garbage ids (failed-OCR
        mints like `ro`, `unknown_1`) never reach the model.
        """
        known = []
        max_level = self._max_level_ids()
        feed_values = self._feed_values()
        damage_values = self._damage_values()
        chain_map = self._chain_map()
        for cell in board.cells:
            if cell.item_id is None:
                continue
            if (self.classifier is not None
                    and not self.classifier.has(cell.item_id)):
                continue
            if cell.score < LABEL_MIN_SCORE:
                continue
            # Champion cells get a dedicated tag so the model knows they
            # are enemies (never merge/feed/spawn material) and can be
            # attacked (drag a damage-dealing creature onto them). The
            # tag is appended FIRST so it survives the max_level/feed/dmg
            # tag additions (champions are never max-level in the game
            # sense; the order is purely cosmetic for readability).
            if any(cell.item_id.startswith(p) for p in CHAMPION_PREFIXES):
                tag = " (champion)"
            else:
                tag = " (max level)" if cell.item_id in max_level else ""
            fv = feed_values.get(cell.item_id)
            tag += f" (feed {fv})" if fv else ""
            dv = damage_values.get(cell.item_id)
            tag += f" (dmg {dv})" if dv else ""
            if (fv is None and feed_values and cell.item_id not in max_level
                    and not is_merge_material(
                        cell.item_id, chain_map=chain_map,
                        max_level_ids=max_level)):
                tag += " [UNKNOWN VALUE — do not feed]"
            known.append((cell.row, cell.col, cell.item_id, tag, cell.score, cell.margin))
        unidentified = self._unidentified_cells(board)
        empty = sum(1 for c in board.cells if not c.occupied)
        # Per-cell legality tags: the 4B model cannot reliably multi-hop
        # `(feed N)` + `Satiety R remaining` + mana% + empty-count across
        # distant lines, so render the verdict on the line itself. Tags use
        # the same gates as the validator (satiety tolerance, SPAWN_MANA_MIN)
        # so a tagged cell validates. Guards: skip tags when the context the
        # gate needs (satiety/mana) is unavailable.
        try:
            remaining = getattr(self.fallback, "satiety_remaining", None)
            capacity = getattr(self.fallback, "satiety_capacity", None)
            tol = (SATIETY_TOL_FRACTION * capacity
                   if remaining is not None and capacity else 0)
            mana = getattr(self, "_mana_fraction", None)
            legible = []
            for r, c, iid, tag, s, m in known:
                cell = board.cell_at(r, c)
                is_champ = any(iid.startswith(p) for p in CHAMPION_PREFIXES)
                is_station = any(iid.startswith(p) for p in STATION_PREFIXES)
                if not is_champ and not is_station:
                    fvv = feed_values.get(iid)
                    if fvv is not None and remaining is not None and remaining > 0:
                        tag += (" [FEEDABLE]" if fvv <= remaining + tol
                                else f" [feed exceeds {remaining:.0f} remaining]")
                if not is_champ:
                    if any(iid.startswith(p) for p in ("grave",) + CHEST_PREFIXES):
                        if empty <= 0:
                            tag += " [spawn blocked: board full]"
                        elif iid.startswith("grave") and mana is not None and mana < SPAWN_MANA_MIN:
                            tag += " [spawn blocked: mana low]"
                        else:
                            tag += " [SPAWNABLE]"
                    elif any(iid.startswith(p) for p in SLIME_SPAWN_PREFIXES):
                        slime = getattr(getattr(self, "fallback", None),
                                        "slime_count", None)
                        if empty <= 0:
                            tag += " [spawn blocked: board full]"
                        elif slime is not None and slime <= 0:
                            tag += " [spawn blocked: slime empty]"
                        elif self._wants_eye_components(board):
                            tag += " [SPAWNABLE]"
                    if iid in max_level:
                        tag += " [NEVER MERGE]"
                legible.append((r, c, iid + tag, s, m))
            known = legible
        except Exception:
            pass
        lines = ["Occupied cells (template bank labels):"]
        if known:
            for r, c, iid, s, m in sorted(known):
                lines.append(f"- ({r},{c}) {iid} (score {s:.2f}, margin {m:.2f})")
        else:
            lines.append("- (none)")
        lines.append(f"Unidentified occupied cells: {sorted(unidentified)}")
        lines.append(f"Empty cells ({empty} of {board.rows * board.cols}): "
                     + (", ".join(f"({c.row},{c.col})" for c in board.cells
                                  if not c.occupied)
                        if empty else "none — board is FULL"))
        if self.hints:
            # --no-hints: omit the computed Best-move line so the model must
            # reason from feats/craving/satiety/board state instead of echoing
            # the code's optimum (the "does the LLM actually decide" A/B).
            hint = self._best_move_hint(board)
            if hint:
                lines.append(hint)
        mana = self._mana_line_text()
        if mana:
            lines.append(mana)
        slime = self._slime_line_text()
        if slime:
            lines.append(slime)
        satiety = self._satiety_line_text()
        if satiety:
            lines.append(satiety)
        craving = self._craving_line_text()
        if craving:
            lines.append(craving)
        # Craving producer gap: the craved item cannot be made from anything
        # on the board (no item, no producer station). Name the station to
        # buy so the model can act instead of working around a craving it
        # can never satisfy. Absent when satisfiable or unmapped.
        need = self._craving_station_need(board)
        if need:
            craving_item = ((self._craving_cache or {}).get("item")
                            or self._last_craving or self.fallback.craved_item)
            disp = display_family(need)
            # Locked beats poor: if the last full scan found no card while
            # locked slots exist, runes are not the blocker — progression is.
            last = self._last_currency_result or {}
            locked_reqs = []
            if (isinstance(last, dict) and last.get("family") == need
                    and (last.get("error") or "").startswith("no ")):
                locked_reqs = last.get("locked") or []
            if locked_reqs:
                req = "; ".join(locked_reqs)
                lines.append(f"Craving {craving_item}: {disp} is LOCKED ({req}) — "
                             f"complete feats to unlock it; gathering runes won't help yet. "
                             f"Do not open the Station panel for it.")
            else:
                lines.append(f"Craving {craving_item} has no producer on the board — buy {disp} "
                             f"from the Station panel (call buy_station(\"{disp}\", confirm=false)) "
                             f"to start the chain. Do not wait for the item to appear.")
        champion = self._champion_line_text()
        if champion:
            lines.append(champion)
        bar = self._bottom_bar_line_text()
        if bar:
            lines.append(bar)
        runes = self._currency_line_text(
            need_family=self._strategy_family or self._craving_station_need(board))
        if runes:
            lines.append(runes)
        feats = self._feats_line_text()
        if feats:
            lines.append(feats)
        strategy = self._strategy_line_text(board=board)
        if strategy:
            lines.append(strategy)
        else:
            # No committed strategy — show the code-evaluated meta-goal
            # direction (replaces the retired periodic StrategyPlanner).
            direction = self._suggested_direction(board)
            if direction:
                lines.append(direction)
            # cross-step rejection reminder. The model is shown its
        # previous step's last rejected action with a one-line reason, so
        # the next decision starts informed (e.g. "Last step rejected:
        # feed (0,0) because feed_mana_overflow — pick a different
        # target or a different action"). The full in-step rejection
        # correction already explains during a step, but the per-step
        # `rejected` set is wiped at step boundary — this line carries
        # the WHY forward.
        if self._cross_step_rejected:
            lines.append(self._rejection_line_text(self._cross_step_rejected))
        # Whitelist of valid moves not yet rejected — constrains the model's
        # action space so it cannot keep re-proposing the same invalid pair.
        # Without this the model re-samples the same high-prior move even
        # though the validator has already rejected it and the correction
        # says "do NOT propose these again". The whitelist is computed from
        # the same ranked groups the validator will accept, filtered by the
        # in-step + cross-step rejected set so stale options are not shown.
        whitelist = self._whitelist_line_text(board)
        if whitelist:
            lines.append(whitelist)
        # Decision checklist, LAST line (recency): the model follows a short
        # numbered list better than distant prose. Mirrors the HARD RULES.
        lines.append("Decision checklist: 1. Pick ONLY a move from the Choose ONLY from line"
                     " (or a Best hint); prefer [FEEDABLE]/[SPAWNABLE] cells. 2. Never repeat a"
                     " rejected move; never merge [NEVER MERGE] cells; never output a tool name."
                     " 3. Reply with ONLY the JSON action.")
        return "\n".join(lines)

    def _whitelist_line_text(self, board, rejected: set | None = None) -> str | None:
        """Compact whitelist of valid moves the model may still choose.

        Enumerates the top valid merges (via ranked_merge_groups), spawn
        stations (grave/chest), and attack pairs — all filtered by the
        rejected set so already-rejected moves are never suggested again.
        Feed is not enumerated because its validity depends on satiety/mana
        which the validator checks; listing it would risk suggesting an
        overflow feed. The line is short (<=3 merges, <=2 spawns) so it
        does not dominate the board state. When no moves are enumerated
        (e.g. empty board, all merges rejected) returns None.
        """
        if rejected is None:
            # In _board_state_text we want cross-step + in-step blocks
            # reflected, but _board_state_text is called before _drive
            # builds the per-step rejected set. At that point only
            # cross-step is known; in-step filtering happens in the
            # correction loop via _correction. So here we filter only
            # cross-step; the loop's correction handles in-step.
            rejected = set((r["kind"], r["cell_a"], r["cell_b"])
                           for r in (self._cross_step_rejected or []))
        empty = sum(1 for c in board.cells if not c.occupied)
        congested = empty <= 3
        parts = []
        # Feeds (max-level income) first when congested — freeing a full
        # board by feeding a max stack outranks merging. When not congested
        # feeds are listed last so merges outrank them via kind_order.
        feed_parts = []
        try:
            max_level = self._max_level_ids() or set()
            for cell in board.cells:
                if not cell.item_id or not cell.occupied:
                    continue
                if cell.item_id not in max_level:
                    continue
                if not any(cell.item_id.startswith(fam) or fam in cell.item_id
                           for fam in ("icerune", "poisonrune", "bloodrune", "moonrune", "deathrune", "coins", "coin")):
                    continue
                if any(cell.item_id.startswith(p) for p in STATION_PREFIXES):
                    continue
                if any(cell.item_id.startswith(p) for p in CHAMPION_PREFIXES):
                    continue
                key = ("feed", (cell.row, cell.col), None)
                if key in rejected:
                    continue
                feed_parts.append(f"feed ({cell.row},{cell.col}) {cell.item_id} (max level)")
                if len(feed_parts) >= 2:
                    break
            if feed_parts and congested:
                parts.append("Valid feeds: " + "; ".join(feed_parts))
        except Exception:
            pass
        # Merges — top 3 by value, not rejected, not noop-excluded
        try:
            ranked = ranked_merge_groups(
                board, max_level_ids=self._max_level_ids(),
                chain_map=self._chain_map(),
                craved_item=self._last_craving or self.fallback.craved_item,
                craved_level=self.fallback.craved_level,
                craved_need=self._craving_need_remaining(),
                exclude_pairs=self._noop.excluded())
            merges = []
            for item_id, cells in ranked:
                if len(cells) < 2:
                    continue
                # ranked is grouped by item_id; within a group cells are
                # already sorted. Take the first pair per group as the
                # representative. Filter rejected.
                a, b = cells[0], cells[1]
                key = ("merge", (a.row, a.col), (b.row, b.col))
                # also check swapped order — model may propose either
                swapped = ("merge", (b.row, b.col), (a.row, a.col))
                if key in rejected or swapped in rejected:
                    # try next pair in group if available
                    found = False
                    for i in range(1, len(cells) - 1):
                        a2, b2 = cells[i], cells[i + 1]
                        k2 = ("merge", (a2.row, a2.col), (b2.row, b2.col))
                        s2 = ("merge", (b2.row, b2.col), (a2.row, a2.col))
                        if k2 not in rejected and s2 not in rejected:
                            merges.append(f"merge {item_id} ({a2.row},{a2.col})+({b2.row},{b2.col})")
                            found = True
                            break
                    if found:
                        continue
                    continue
                merges.append(f"merge {item_id} ({a.row},{a.col})+({b.row},{b.col})")
                if len(merges) >= 3:
                    break
            if merges:
                parts.append("Valid merges: " + "; ".join(merges))
        except Exception:
            pass
        # Spawns — any grave/chest/cupboard not rejected, but NOT when
        # board is full (spawn_no_room). Listing a spawn on a 0/20 board
        # tricks the model into a guaranteed `spawn_no_room` rejection.
        # Cupboards list only when wanted (slime cost + clog risk).
        if empty > 0:
            try:
                spawns = []
                want_eye = self._wants_eye_components(board)
                for cell in board.cells:
                    if cell.item_id is None:
                        continue
                    is_spawn = any(cell.item_id.startswith(p)
                                   for p in ("grave",) + CHEST_PREFIXES)
                    is_cup = (want_eye and any(
                        cell.item_id.startswith(p) for p in SLIME_SPAWN_PREFIXES))
                    if not (is_spawn or is_cup):
                        continue
                    key = ("spawn", (cell.row, cell.col), None)
                    # _drive's rejected set stores (kind, cell_a, cell_b) where
                    # cell_b is None for spawns. Check both None and tuple forms.
                    if key in rejected or ("spawn", (cell.row, cell.col), (cell.row, cell.col)) in rejected:
                        continue
                    spawns.append(f"spawn ({cell.row},{cell.col}) {cell.item_id}")
                    if len(spawns) >= 2:
                        break
                if spawns:
                    parts.append("Valid spawns: " + "; ".join(spawns))
            except Exception:
                pass
        # Feeds when not congested — listed after merges/spawns
        if feed_parts and not congested:
            parts.append("Valid feeds: " + "; ".join(feed_parts))
        # Attacks — best attacker with known damage onto champion
        try:
            dmg = self._damage_values() or getattr(self.fallback, "damage_values", {}) or {}
            pair = best_attack_pair(board, damage_values=dmg)
            if pair is not None:
                atk, champ, dv = pair
                key = ("attack", (atk.row, atk.col), (champ.row, champ.col))
                # also check if this specific pair was rejected
                if key not in rejected:
                    parts.append(f"Valid attack: attack {atk.item_id} ({atk.row},{atk.col}) -> {champ.item_id} ({champ.row},{champ.col}) dmg {dv}")
        except Exception:
            pass
        if not parts:
            return None
        return "Choose ONLY from: " + " | ".join(parts) + " — do NOT repeat any rejected move."

    def _whitelist_snippet(self, board, rejected: set | None = None) -> str:
        """Short whitelist for inline corrections (single line, no prefix)."""
        line = self._whitelist_line_text(board, rejected=rejected)
        if not line:
            return "no valid merges/spawns enumerated — try collect or feed a low-value creature."
        # Strip the leading "Choose ONLY from: " for inline use
        return line.replace("Choose ONLY from: ", "", 1)

    def _best_merge_line(self, board) -> str:
        """Best-valued merge as a hint, e.g.
        'Best merge: skeleton_lvl4 (1,1)+(3,0) -> skeleton_lvl5 (max level)'.

        filter out cells with low margin (below
        `NEIGHBOR_MIN_MARGIN`). The validator's `merge_neighbor_suspect`
        check refuses merges where a cell's top-2 templates are different
        levels of the same family (the bob-phase disambiguation case).
        If the hint suggests a low-margin cell, the model proposes a move
        the validator immediately rejects — wasting a retry round. The
        `ranked_merge_groups` filter (`MERGE_MIN_MARGIN = 0.10`) was too
        permissive vs the validator (`NEIGHBOR_MIN_MARGIN = 0.20`).
        Filtering at the higher threshold keeps the hint safe.

        when a station-build or station-collect strategy is active
        (the model has committed to "Build a Mana Pool." or "Own a Lvl 3+
        Grave."), prioritize merges of the strategy's target family over
        the highest-value merge on the board. Without this, the hint
        told the model to merge skeleton_lvl2+skeleton_lvl2 (the highest
        base-level score) while the model was supposed to be merging
        graves for the strategy. The user reported: "the graves are never
        merged together" — the Best hint was steering the model to
        skeletons. The fix: when a station-kind strategy is active AND a
        mergeable pair of the strategy's family exists on the board, hint
        THAT pair first. Otherwise fall back to the default
        highest-value merge (so non-strategy saves still get useful hints).

        Empty when no merge is possible.
        """
        from planner.llm import NEIGHBOR_MIN_MARGIN, _same_family_diff_level
        craved = self._last_craving or self.fallback.craved_item
        ranked = ranked_merge_groups(board, max_level_ids=self._max_level_ids(),
                                        chain_map=self._chain_map(), craved_item=craved,
                                        craved_level=self.fallback.craved_level,
                                        craved_need=self._craving_need_remaining(),
                                        exclude_pairs=self._noop.excluded())
                                        # strategy-prioritized merge. When the active strategy
        # names a station (Build-a-X or Own-a-Lvl-N+-X), find a mergeable
        # pair of the strategy's family on the board and hint THAT one
        # first. Without this, the highest-value merge on the board wins
        # (skeletons outrank graves because they have more on-board
        # copies) and the model never makes progress on the strategy.
        strategy = self._strategy
        if (strategy and self._strategy_fresh()
                and strategy.get("kind") == "station"
                and strategy.get("noun")):
            target_fam = strategy.get("noun")
            for item_id, cells in ranked:
                if not (item_id.startswith(target_fam + "_lvl")
                        or item_id == target_fam):
                    continue
                if len(cells) < 2:
                    continue
                a, b = cells[0], cells[1]
                result = merge_result_id(item_id, self._chain_map())
                result_txt = ""
                if result:
                    result_txt = f" -> {result}"
                    if result in self._max_level_ids():
                        result_txt += " (max level)"
                return (f"Best merge (strategy): {item_id} ({a.row},{a.col})"
                        f"+({b.row},{b.col}){result_txt}")
                        # drop any cell from the candidates where the margin is
        # below the validator's neighbor-suspect threshold. Without this
        # filter, the hint can suggest a low-margin pair that the
        # validator immediately rejects.
        if ranked:
            item_id, cells = ranked[0]
            for cell in cells:
                if (cell.runner_up_id
                        and cell.margin < NEIGHBOR_MIN_MARGIN
                        and _same_family_diff_level(cell.item_id, cell.runner_up_id)):
                    return ""
        if not ranked:
            return ""
        item_id, cells = ranked[0]
        a, b = cells[0], cells[1]
        result = merge_result_id(item_id, self._chain_map())
        result_txt = ""
        if result:
            result_txt = f" -> {result}"
            if result in self._max_level_ids():
                result_txt += " (max level)"
        return (f"Best merge: {item_id} ({a.row},{a.col})+({b.row},{b.col})"
                f"{result_txt}")

    def _best_feed_line(self, board) -> str:
        """Best feed target as a hint, e.g.
        'Best feed: ribcage (3,1) (craved)' or
        'Best feed: skeleton_lvl2 (2,2) (feed 25) (fits 189 remaining)'.

        Mirrors the HeuristicPlanner feed pick EXACTLY (same best_feed_cell
        ranking + same `_feed_trigger` gate, including the craving and an
        active feed objective), so the model's suggestion matches the
        fallback. Does NOT self-suppress when a merge exists — whether to show
        feed-vs-merge is decided by `_best_move_hint` from feat (+craving)
        weights.
        """
        feed_values = self._feed_values()
        craved = self._last_craving or self.fallback.craved_item
        craved_level = self.fallback.craved_level
        max_level = self._max_level_ids()
        # Single source of truth for satiety: the values `next_move` cached on
        # the fallback (so the hint mirrors the fallback EXACTLY); fall back to
        # a fresh reader read only outside a live step (dry-run/static).
        remaining = self.fallback.satiety_remaining
        if remaining is None:
            remaining = self._satiety_remaining()
        capacity = self.fallback.satiety_capacity
        if capacity is None:
            capacity = self._satiety_capacity()
        income = income_objective(self._strategy, board,
                                    self._max_level_ids())
        prefer = income.target if income is not None else self.fallback.feed_prefer_item
        target = best_feed_cell(board, feed_values=feed_values,
                                max_level_ids=max_level, craved_item=craved,
                                craved_level=craved_level,
                                remaining_satiety=remaining,
                                prefer_item=prefer,
                                craving_bonus=self.fallback.craving_bonus_est,
                                satiety_capacity=capacity,
                                chain_map=self._chain_map(),
                                prefer_max=(self.fallback.feed_prefer_max
                                            if income is not None else False))
        if target is None:
            return ""
        empty_count = sum(1 for c in board.cells if c.item_id is None)
        spawn_possible = self.fallback._spawn_possible(board, self._frame)
        feed_active = self._feed_objective_active() or income is not None
        merge_available = bool(ranked_merge_groups(
            board, max_level_ids=max_level, chain_map=self._chain_map(),
            craved_item=craved, craved_level=craved_level,
            craved_need=self._craving_need_remaining(),
            exclude_pairs=self._noop.excluded()))
        if not _should_feed(board, target, empty_count,
                            max_level_ids=max_level, feed_values=feed_values,
                            craved_item=craved, craved_level=craved_level,
                            feed_objective_active=feed_active,
                            spawn_possible=spawn_possible,
                            merge_available=merge_available,
                            prefer_item=prefer):
            return ""  # the heuristic wouldn't feed either
        fv = feed_values.get(target.item_id)
        tag = f" (feed {fv})" if fv else ""
        if remaining is not None:
            tag += f" (fits {remaining} remaining)"
        if craved and craved_matches(target.item_id, craved,
                                        craving_level=craved_level):
            tag += " (craved)"
        return f"Best feed: {target.item_id} ({target.row},{target.col}){tag}"

    def _best_move_hint(self, board) -> str:
        """The single highest-priority move hint (Best attack / Best merge /
        Best feed / Best spawn / Best income).

        Chosen by the feat (+craving +income) weights via kind_order, the SAME
        ordering the HeuristicPlanner uses: if feed outranks merge (an active
        feed feat, the craving, or the income objective), show the feed hint
        even when a merge exists; otherwise show the merge hint, falling back
        to feed when no merge is available.
        When the only matching craved item is capacity-wasteful, feeding is
        declined and a `Best spawn` hint is shown instead (the fallback
        spawns to rebuild the craved item). Empty when none apply.

        Champion combat: when a champion is on the board, the `attack` hint
        is the highest-priority line (the kind weight is 3.0 vs merge's 2.0
        and a Champion steals food/mana while alive). Champions DO NOT spawn
        on a timer; they spawn when you've merged enough of a specific
        family (e.g. The Peasant tracks Skeletons, Zombies, and Mummies).
        So a high-merge-pressure save is naturally punctuated by combat.
        The hint is omitted when no creature on the board has a known
        damage value (the attack would be validator-rejected)."""
        attack = self._best_attack_line(board)
        feed = self._best_feed_line(board)
        income = self._best_income_line(board)
        merge = self._best_merge_line(board)
        spawn = self._best_spawn_line(board)
        # Hard priority: when the board is full (0 empty) and a max-level
        # income stack sits on it, feeding that stack frees a cell and banks
        # currency — merging anything else (bone, ribcage) keeps the board
        # clogged. The kind_order (feat-driven) would still return `merge`
        # first when no build strategy is active, so the model merges the
        # two icerune_lvl3 (max) — which the validator then refuses as
        # `merge_max_level`. Force the income/feed hint ahead of merge when
        # congested. Observed: board 0/20 empty with 2 icerune_lvl3, model
        # tried `merge (1,0)+(3,2)` instead of `feed`.
        empty = sum(1 for c in board.cells if not c.occupied)
        if empty <= 3 and income:
            return income
        if empty <= 3 and feed and any(c.item_id in (self._max_level_ids() or set()) for c in board.cells if c.item_id):
            # fallback: even if income is None (no build strategy), a plain
            # max-level feed is still better than merging when congested
            max_feed = None
            for c in board.cells:
                if c.item_id in (self._max_level_ids() or set()) and c.occupied:
                    # prefer the max-level income family already filtered
                    if any(c.item_id.startswith(f) for f in ("icerune","poisonrune","bloodrune","moonrune","deathrune","coin")):
                        max_feed = c
                        break
            if max_feed is not None and feed:
                return feed
        for kind in kind_order(self._feat_weights(board)):
            if kind == "attack" and attack:
                return attack
            if kind == "merge" and merge:
                return merge
            if kind == "feed":
                # income (feeding an icerune stack for Ice) outranks ordinary
                # feeding when a build strategy is active and the stack is on
                # the board — it's the strategy's income action.
                if income:
                    return income
                if feed:
                    return feed
                # the fallback's feed branch is declined: either the craved
                # feed is capacity-wasteful, or feed-to-progress is suspended
                # because a spawn is possible on a sparse board. Either way the
                # fallback spawns (or falls through) — mirror it.
                if spawn:
                    return spawn
            if kind == "spawn" and spawn:
                return spawn
        return attack or income or feed or merge or spawn or ""

    def _best_attack_line(self, board) -> str:
        """Best attack hint: 'Best attack: <creature> (r,c) (dmg N) -> champion
        (r,c)'. Shown only when (a) a champion is on the board and (b) a
        creature with a known damage value is available to attack with. The
        ranking uses the same `best_attack_pair` helper as the heuristic, so
        the hint and the fallback pick the same target."""
        from planner.agent import best_attack_pair
        pair = best_attack_pair(board, damage_values=self.fallback.damage_values)
        if pair is None:
            return ""
        attacker, champion, dmg = pair
        return (f"Best attack: {attacker.item_id} ({attacker.row},{attacker.col}) "
                f"(dmg {dmg}) -> champion ({champion.row},{champion.col})")

    def _best_income_line(self, board) -> str:
        """Best income hint: feeding a max-level income stack for its currency.

        Shown when a max-level stack of a station-buying Rune (ice / poison /
        blood / moon / death), a max-level Coin stack, or a max-level Gem
        stack is on the board — the stack is income, and the hint tells
        the model to feed it (merges outrank feeds by default, so income
        stacks starve without this).
            - Rune stacks fire only under a build strategy (Runes are only
            useful when you're saving for a station).
            - Coin stacks fire unconditionally (more Gold is always useful
            for the Shop / Merchant / Trader).
            - Gem stacks fire unconditionally (more Gems is always useful
            for the premium Shop / skins / Wobulan upgrades).
        Sub-max income stacks are NOT a valid target — merge them up first
        so the per-feed payout is the largest (the validator refuses
        `feed` on sub-max income stacks; see `_validate_feed_income_gate`).

        only MAX-LEVEL income stacks are valid targets. When
        multiple max-level stacks exist, the hint points at the LARGEST
        by feed value (the most currency per feed) so the model doesn't
        pick a low-value sub-max stack by accident.
        """
        # "Always useful" income families don't require a build strategy —
        # they're spent in the Shop / Wobulan trades regardless of what the
        # bot is currently building. Rune families need a build strategy
        # because Runes are only useful when saving for a specific station —
        # EXCEPT when the board is congested (≤3 empty cells): a max-level
        # rune stack occupying a cell blocks spawns/merges, so feeding it
        # to free space and bank currency is useful even without an active
        # build goal (observed: board full 0/20 empty with 2 icerune_lvl3,
        # no strategy active, feed never hinted, board stalled).
        always_useful = {"coin", "gem"}
        empty = sum(1 for c in board.cells if not c.occupied)
        congested = empty <= 3
        candidates = []   # (feed_value, item_id, row, col, label, who)
        for c in board.cells:
            if not c.item_id:
                continue
            base = normalize_item_name(c.item_id)
            matched = next(
                (f for f in INCOME_FAMILIES
                    if base.startswith(f) or f in base), None)
            if matched is None:
                continue
            # Rune income without a build strategy is only useful when congested
            if matched not in always_useful and not congested:
                if not (self._strategy and self._strategy_fresh()
                        and self._strategy.get("kind") == "station"):
                    continue
                # only max-level income stacks qualify. A coin_lvl1
            # or ice_rune_lvl1 is not a valid feed target — feeding it
            # would waste the currency vs. merging it up first.
            if c.item_id not in self._max_level_ids():
                continue
            # Build strategy gate: Rune families need an active build
            # strategy; Coin + Gem fire unconditionally. EXCEPT when
            # congested — see comment above.
            if matched not in always_useful and not congested and not is_build_strategy(self._strategy):
                continue
            # Per-feed currency label per family.
            if matched.endswith("rune"):
                label = "RUNES"
                rname = matched.replace("rune", "").upper()
                who = f"the {rname} Runes your build strategy needs"
            elif matched == "coin":
                label = "GOLD"
                who = "Gold (the Shop / Merchant / Trader economy)"
            elif matched == "gem":
                label = "GEMS"
                who = "Gems (the premium Shop / skins / Wobulan upgrades)"
            else:
                label = matched.upper()
                who = f"the {matched} currency"
            # Feed value tells us the per-feed currency grant (lvl4 > lvl3
            # > lvl2 > lvl1 for both Coins and Runes). Pick the largest.
            z = self._feed_values().get(c.item_id) or 0
            candidates.append((z, c.item_id, c.row, c.col, label, who))
        if not candidates:
            return ""
        # Sort by feed value desc, then by row/col for stable ordering.
        candidates.sort(key=lambda x: (-x[0], x[2], x[3]))
        z, item_id, r, col, label, who = candidates[0]
        return (f"Best income: {item_id} ({r},{col}) — feed it "
                f"to the Devourer for {label} ({who})")

    def _best_spawn_line(self, board) -> str:
        """Best spawn as a hint when feeding is declined and a spawn station
        exists, e.g. 'Best spawn: grave_lvl3 (3,2) -> bone / ribcage (9 empty)'
        or 'Best spawn: icebox_unopened (4,0) -> icerune/poisonrune (9 empty)'.

        prefer the HIGHEST-LEVEL spawn station on the board. A
        lvl 3 Grave produces bone (40%) / ribcage (30%) / zombie (30%) per
        the wiki, lvl 2 produces bone (60%) / ribcage (40%), lvl 1 only
        produces bone (100%). Spawning from the highest-level station
        maximizes the chain-progress value per tap (ribcages and zombies
        merge up faster than bones). The hint also includes the level in
        the cell id (e.g. `grave_lvl3`) so the model can see the level at
        a glance.

        the spawn output list (e.g. "bone / ribcage / zombie")
        is read from the glossary's per-level popup block, NOT the
        hardcoded `GRAVE_SPAWN_PREFIXES = ("bone", "ribcage")` constant.
        That constant was missing zombie for grave_lvl3 — the bot would
        never know to expect a zombie spawn from a level-3 grave.

        Chest spawns are mana-free; grave spawns need mana. Both need an
        empty cell. Prefers chest over grave when both exist (chests are
        limited-use finite resources and produce Runes; the next-spawn
        station should be chest-first to drain the limited resource).

        when a station-build or station-collect strategy is
        active, EXCLUDE stations of the strategy's target family from
        the spawn candidates. The model has 2×grave_lvl1 + 1×grave_lvl2
        and the strategy is "Own a Lvl 3+ Grave" — the Best hint was
        telling it to spawn from grave_lvl2 (the highest-level station),
        which would consume a grave and prevent the merge-up. The fix:
        if a station-collect strategy is active and the target family
        is on the board, skip those cells (they're merge material, not
        spawn material). The model can still spawn from a DIFFERENT
        family's spawner (e.g. an icebox if the strategy is "Own a Lvl
        3+ Grave") or, when no other spawner exists, the hint returns
        "" (no spawn recommended).
        """
        f = getattr(self, "_frame", None)
        mana = read_mana_fraction(f) if f is not None else None
        empty = sum(1 for c in board.cells if c.item_id is None)
        if empty <= 0:
            return ""
            # collect ALL spawn candidates (chests + graves), then pick
        # the best one. Chests outrank graves (chest > grave preference
        # unchanged). Among chests, level desc (Big chest > Regular).
        # Among graves, level desc (lvl 5 > lvl 1).

        def _level(item_id: str) -> int:
            """Parse the level suffix `_lvl<N>` from an item id. 0 for
            level-less items (chests, single-level stations). Used to sort
            candidates by level desc."""
            m = re.search(r"_lvl(\d+)$", item_id or "")
            return int(m.group(1)) if m else 0

        def _cell_level(cell) -> int:
            return _level(cell.item_id or "")

            # stations that are the strategy's target family are
        # merge material, not spawn material — exclude them from the
        # spawn candidates. Without this, the hint tells the model to
        # spawn from a grave when the strategy needs to merge graves up.
        strategy = self._strategy
        excluded_families = set()
        if (strategy and self._strategy_fresh()
                and strategy.get("kind") == "station"
                and strategy.get("noun")):
            excluded_families.add(strategy.get("noun"))

        def _is_excluded(cell) -> bool:
            if not cell.item_id:
                return False
            base = cell.item_id.split("_lvl")[0]
            return base in excluded_families

        chest_candidates = [
            c for c in board.cells
            if c.item_id
            and any(c.item_id.startswith(p) for p in CHEST_PREFIXES)
            and not _is_excluded(c)]
        grave_candidates = [
            c for c in board.cells
            if c.item_id
            and any(c.item_id.startswith(p) for p in SPAWN_PREFIXES)
            and not _is_excluded(c)]
        # Sort each by level desc, then row/col for stable ordering.
        chest_candidates.sort(key=lambda c: (-_cell_level(c), c.row, c.col))
        grave_candidates.sort(key=lambda c: (-_cell_level(c), c.row, c.col))
        # Chest > grave preference (chests are limited, finite-use).
        # Chest > grave preference (chests are limited, finite-use).
        if chest_candidates:
            cell = chest_candidates[0]
            what = "/".join(CHEST_SPAWN_PREFIXES)
            return (f"Best spawn: {cell.item_id} ({cell.row},{cell.col}) "
                    f"-> {what} ({empty} empty)")
        # Supply cupboard (Slime cost) — hinted ONLY when something wants
        # eye components: an eyemonster/eyeball/eyeinjar craving, or a
        # strategy targeting the cupboard/eye family. Otherwise a cupboard
        # tap spends Slime and clogs the board for no objective.
        if self._wants_eye_components(board):
            cup_candidates = [
                c for c in board.cells
                if c.item_id
                and any(c.item_id.startswith(p) for p in SLIME_SPAWN_PREFIXES)
                and not _is_excluded(c)]
            cup_candidates.sort(key=lambda c: (-_cell_level(c), c.row, c.col))
            if cup_candidates:
                cell = cup_candidates[0]
                slime = getattr(getattr(self, "fallback", None),
                                "slime_count", None)
                if slime is not None and slime <= 0:
                    pass  # vat known-empty: cupboard would no-op, skip
                else:
                    per_level = self._per_level_spawn_outputs(cell.item_id)
                    what = "eyeball"
                    if per_level is not None and per_level[0]:
                        what = " / ".join(per_level[0])
                    return (f"Best spawn: {cell.item_id} ({cell.row},{cell.col}) "
                            f"-> {what} (costs Slime) ({empty} empty)")
        if grave_candidates:
            cell = grave_candidates[0]
            if mana is not None and mana < SPAWN_MANA_MIN:
                return ""  # grave would no-op — no hint
                # prefer per-level spawn rates from the glossary
            # (popped via the auto-read path) over the hardcoded
            # GRAVE_SPAWN_PREFIXES. The glossary stores bullets like
            # "lvl3 spawn: bone, ribcage, zombie (40, 30, 30)" per
            # station level. Falls back to the hardcoded prefix list
            # (bone/ribcage for graves) when no per-level entry exists.
            per_level = self._per_level_spawn_outputs(cell.item_id)
            if per_level is not None:
                outputs, rates = per_level
                if outputs:
                    parts = [o for o, r in zip(outputs, rates) if r > 0]
                    if parts:
                        # show rates for better decision-making (e.g., zombie 30% from lvl3)
                        rate_strs = [f"{o} ({r}%)" for o, r in zip(outputs, rates) if r > 0]
                        what = " / ".join(rate_strs)
                    else:
                        what = "/".join(outputs)
                else:
                    what = "/".join(GRAVE_SPAWN_PREFIXES)
            else:
                what = "/".join(GRAVE_SPAWN_PREFIXES)
                # add context about zombie spawns from lvl3+ graves
            context = ""
            if "zombie" in what.lower():
                # Extract grave level
                m = re.search(r"_lvl(\d+)$", cell.item_id or "")
                if m and int(m.group(1)) >= 3:
                    context = " (zombies spawn from lvl3+ graves)"
                else:
                    context = ""
            return (f"Best spawn: {cell.item_id} ({cell.row},{cell.col}) "
                    f"-> {what}{context} ({empty} empty)")
        return ""

    def _bottom_bar_line_text(self) -> str:
        """Bottom-bar dock state, e.g. 'Bottom bar: unlocked feats,station,queue(reward),
        locked spellbook,shop'. Empty when the dock is not visible (panel open /
        non-lair frame), so the LLM knows it cannot interact with buttons.

        The queue reward flag is the model's ONLY signal for when to call
        collect_queue (the tool is offered every step but nothing else says
        when a reward is queued — observed: zero queue_collected events in
        3329 session lines). has_reward is a free pixel check (no tap, no
        panel), so it runs inline here."""
        if self.bottombar is None or self._frame is None:
            return ""
        try:
            if not self.bottombar.bar_visible(self._frame):
                return ""
            unlocked = []
            locked = []
            for b in self.bottombar.read_bar(self._frame):
                (unlocked if b["unlocked"] else locked).append(b["name"])
        except Exception:
            return ""
        if not unlocked and not locked:
            return ""
        queue_flag = ""
        if "queue" in unlocked:
            try:
                from vision.queue_box import has_reward, queued_reward_id, REWARD_DISPLAY
                if has_reward(self._frame):
                    rid = queued_reward_id(self._frame)
                    queue_flag = f"({REWARD_DISPLAY.get(rid, 'reward')})" if rid else "(reward)"
                else:
                    queue_flag = "(empty)"
            except Exception:
                queue_flag = ""
        unlocked = [n + queue_flag if n == "queue" else n for n in unlocked]
        return "Bottom bar: " + (
            ("unlocked " + ",".join(unlocked) if unlocked else "") +
            ("  locked " + ",".join(locked) if locked else "")
        ).strip()

    def _rejection_line_text(self, recs: list[dict]) -> str:
        """Render the cross-step rejection memory for the get_board_state line,
        e.g. 'Last step rejected (do NOT propose these again this step):
        feed (0,0) because feed_mana_overflow; feed (1,2) because
        feed_overflow; merge (2,3)+(3,1) because merge_mismatch.'.

        the previous per-step `rejected` set was wiped at the step
        boundary, so the same action was re-emitted on the very first attempt
        of the next step (e.g. the user reported "always suggests feeding the
        mana pool / mana pot / lvl1 coin" — the rejection correction lived
        only in the in-step history, not in the persisted board state). The
        `recs` is a list of `{kind, cell_a, cell_b, reason, step}` dicts from
        the previous step's rejections; we render ALL of them (NOT just the
        last) because the model would otherwise just shift to the second-to-
        last action and re-emit that. The action targets are pre-blocked
        from this step's attempt 0 in `_drive`."""
        parts = []
        for rec in recs:
            kind = rec["kind"]
            a = rec["cell_a"]
            b = rec["cell_b"]
            if a is None:
                target = ""
            elif kind == "merge":
                target = f"({a[0]},{a[1]})+({b[0]},{b[1]})"
            else:
                target = f"({a[0]},{a[1]})"
            # Strip the leading "key:" from the reason if present (the validator
            # emits "key: short hint" — keep just the key for terseness).
            reason = rec["reason"] or "?"
            key = reason.split(":", 1)[0].strip() if reason else "?"
            parts.append(f"{kind} {target} because {key}")
        return (f"Last step rejected (do NOT propose any of these again "
                f"this step — pick a DIFFERENT target or a DIFFERENT action): "
                + "; ".join(parts) + ".")

    def _cross_step_rejection_still_valid(self, rec: dict, board) -> bool:
        """Whether a carried-over rejection still applies to the CURRENT board.

        A rejection (kind, cell_a, cell_b) is only meaningful while the SAME
        item still occupies the cell(s) it was rejected against. If the cell
        emptied, the item merged away, or a different item spawned there, the
        board has shifted and the rejection no longer binds — it could be re-
        attempted (the TTL bounds how far back we look regardless). Entries
        with no captured item id (old format) can't be verified, so we keep
        them for TTL-only expiry."""
        if board is None:
            return True
        if rec.get("cell_a") is None or rec.get("item_a") is None:
            return True   # unverifiable -> keep (TTL bounds it)
        ca = board.cell_at(*rec["cell_a"])
        if ca is None or ca.item_id != rec["item_a"]:
            return False
        if rec.get("kind") == "merge" and rec.get("cell_b") is not None:
            if rec.get("item_b") is None:
                return True
            cb = board.cell_at(*rec["cell_b"])
            if cb is None or cb.item_id != rec["item_b"]:
                return False
        return True

    def _mana_line_text(self) -> str:
        """Mana bar fill from the HUD, e.g. 'Mana: ~95% full'. Empty when the
        HUD bar is not visible (non-lair frame) so the LLM knows spawning may
        no-op on an empty bar (it costs ~10 mana) and collecting is fine."""
        frac = getattr(self, "_mana_fraction", None)
        if frac is None:
            return ""
        line = f"Mana: ~{round(frac * 100)}% full"
        # Below the spawn threshold the grave is dead (spawn_low_mana) —
        # name the producers like the slime line does. Mirrors the
        # validator condition (SPAWN_MANA_MIN) exactly.
        if frac < SPAWN_MANA_MIN:
            line += " (low — " + _resource_teaching("mana", short=True) + ")"
        return line

    def _satiety_remaining(self, frame=None) -> int | None:
        """Devourer satiety capacity - current fill, or None when unreadable.

        The satiety fraction reader uses a memory bank of confirmed tokens
        (assets/satiety); on a save whose denominator/numerator tokens aren't
        banked yet, `read_satiety` honestly returns unknowns and this returns
        None (capacity gate off, old feeding behavior)."""
        if self.satiety_reader is None:
            return None
        f = frame if frame is not None else getattr(self, "_frame", None)
        if f is None:
            return None
        try:
            read = self.satiety_reader.read_satiety(f)
        except Exception:
            return None
        num, den = read.get("num"), read.get("den")
        if num is None or den is None:
            return None
        try:
            return int(den) - int(num)
        except (TypeError, ValueError):
            return None

    def _satiety_capacity(self, frame=None) -> int | None:
        """Devourer satiety total capacity Y (denominator), or None when the
        fraction can't be read (needed for the overflow tolerance 10%*Y)."""
        if self.satiety_reader is None:
            return None
        f = frame if frame is not None else getattr(self, "_frame", None)
        if f is None:
            return None
        try:
            read = self.satiety_reader.read_satiety(f)
            return int(read.get("den"))
        except (TypeError, ValueError, KeyError, AttributeError):
            return None

    def _satiety_line_text(self, frame=None) -> str:
        """Satiety fraction for get_board_state, e.g. 'Satiety: 0/250 (250
        remaining)'. Empty when the reader isn't wired or the tokens aren't
        banked (honest unknown — no fabricated number)."""
        rem = self._satiety_remaining(frame)
        if rem is None:
            return ""
        num = den = None
        f = frame if frame is not None else getattr(self, "_frame", None)
        try:
            read = self.satiety_reader.read_satiety(f)
            num, den = read.get("num"), read.get("den")
        except Exception:
            return ""
        if num is None or den is None:
            return ""
        return f"Satiety: {num}/{den} ({rem} remaining)"

    def _slime_count(self, frame=None) -> int | None:
        """Read the slime-vat HUD counter (e.g. 8).

        Mirrors SatietyReader.read_satiety: read_count delegates to
        SlimeVatReader.read_count (Apple Vision OCR + template bank,
        parallel to the satiety pipeline). Returns int or None.
        """
        if self.slime_vat is None:
            return None
        f = frame if frame is not None else getattr(self, "_frame", None)
        if f is None:
            return None
        try:
            return self.slime_vat.read_count(f)
        except Exception:
            return None

    def _slime_capacity(self, frame=None) -> int | None:
        """Read the slime-vat max capacity from the open popup, or return
        None when the popup isn't visible. Used to compute fullness %.

        Q3 = iii: derive from popup text. Q4 = C: never guess a
        capacity — when the popup is closed, return None and the
        board-state line degrades to "Slime: N" with no fake percent.
        """
        if self.slime_vat is None:
            return None
        f = frame if frame is not None else getattr(self, "_frame", None)
        try:
            return self.slime_vat.read_capacity(f)
        except Exception:
            return None

    def _slime_line_text(self) -> str:
        """Format the slime-vat line for the board state.

        Q4 = C: when no real capacity is read from the popup, show only
        the raw count. The percent is rendered only when the popup
        has been read this step — no fabricated fullness.
        """
        n = self.fallback.slime_count
        if n is None:
            return ""
        cap = self.fallback.slime_capacity
        if cap and cap > 0:
            pct = round(100 * n / cap)
            line = f"Slime: {n} / {cap} (~{pct}% full)"
        else:
            line = f"Slime: {n}"
        # Empty vat hard-blocks cupboard/fridge spawns (spawn_no_slime) —
        # name the producers so the model builds toward them instead of
        # tapping a dead station. Mirrors the validator condition exactly.
        if n <= 0:
            line += " (empty — " + _resource_teaching("slime", short=True) + ")"
        return line

    def _craving_objective(self) -> Objective | None:
        """The current craving as a feed objective (None when unknown).

        Prefers the richer cached menu read (has progress counts); falls back
        to the bubble identity (no counts -> ratio 0)."""
        cache = self._craving_cache
        if cache and cache.get("item"):
            return craving_objective(cache["item"],
                                        cache.get("count_done"), cache.get("count_required"))
        item = self._last_craving or self.fallback.craved_item
        return craving_objective(item)

    def _feat_objectives(self) -> list:
        """Parsed objectives from the cached FEATS read (empty when unread)."""
        cache = self._feats_cache
        if not cache or not cache.get("feats"):
            return []
        return [parse_obj(f) for f in cache["feats"] if f.get("name")]

    def _feat_weights(self, board=None) -> dict:
        """Kind weights from active feats + the craving (drives ordering).

        The craving folds in as a feed objective ONLY when a craved-matching
        cell is actually on the board (`board` given): feeding a non-craved
        creature does nothing for the craving, so without a target the boost
        must not distort the merge-vs-feed order — the Aug 23 soak showed it
        suppressing `Best merge` for 4 ribcage pairs while the bot fed merge
        material for a rottenflesh craving that wasn't on the board."""
        craving = self._craving_objective()
        if craving is not None and board is not None:
            on_board = any(
                craved_matches(c.item_id, self.fallback.craved_item,
                                craving_level=self.fallback.craved_level)
                for c in board.cells if c.item_id)
            if not on_board:
                craving = None
        chest = chest_spawn_objective(self._strategy, board)
        w = feat_weights(self._feat_objectives(), craving,
                            income=income_objective(self._strategy, board,
                                                    self._max_level_ids()),
                            chest_spawn=chest)
        # Strategy slot: boost the model-committed objective's move kind.
        # Build strategies (e.g. "Build a Mana Pool.") need Ice, and the
        # income (feeding an icerune stack) IS the work — boost feed, not
        # merge, so the income feed outranks merges when the stack is on the
        # board (otherwise the strategy's generic "merge" boost buries it).
        s = self._strategy
        if s is not None and self._strategy_fresh():
            if is_build_strategy(s):
                w["feed"] = w.get("feed", 0.0) + STRATEGY_KIND_BONUS
            else:
                obj = next((o for o in self._feat_objectives()
                            if s["feat"].lower() in (o.text or "").lower()
                            or (o.text or "").lower() in s["feat"].lower()), None)
                kind = obj.kind if obj is not None else "merge"
                if kind in w:
                    w[kind] = w.get(kind, 0.0) + STRATEGY_KIND_BONUS
        return w

    def _feed_objective_active(self) -> bool:
        """True when an active feed/advance feat is present (feed drives)."""
        return any(o.kind == "feed" for o in self._feat_objectives())

    def _wants_eye_components(self, board=None) -> bool:
        """True when an eye-component spawner is wanted right now.

        Either the craving maps to the Supply Cupboard (CRAVING_PRODUCERS)
        or a fresh strategy targets the cupboard/eye family. Guards the
        cupboard spawn hint, whitelist entries, and fallback spawn ranking
        so cupboard taps (Slime cost, board-clogging eyeballs) only happen
        with an objective behind them.
        """
        try:
            craving = ((self._craving_cache or {}).get("item")
                       or self._last_craving
                       or getattr(self.fallback, "craved_item", None))
            if craving:
                cf = normalize_item_name(craving)
                prod = CRAVING_PRODUCERS.get(cf)
                if prod is None:
                    for key, val in CRAVING_PRODUCERS.items():
                        if cf.startswith(key) or key.startswith(cf):
                            prod = val
                            break
                if prod is not None and prod[0] in SLIME_SPAWN_PREFIXES:
                    return True
            s = self._strategy
            if s and self._strategy_fresh():
                if s.get("kind") == "station" and (s.get("noun") or "") in SLIME_SPAWN_PREFIXES:
                    return True
                noun = (s.get("noun") or "").lower()
                if s.get("kind") == "creature" and noun in (
                        "eyemonster", "eyeball", "eyeinjar"):
                    return True
        except Exception:
            pass
        return False

    def _craving_station_need(self, board) -> str | None:
        """Producer-station family the craving needs but the board lacks.

        Returns the family (e.g. 'supplycupboard') when ALL hold: a craving
        item is known (menu cache, bubble, or fallback), CRAVING_PRODUCERS
        maps it to a station, NO board cell holds the craved item or its
        components, and NO station of that family is on the board. Else None.
        The caller renders the buy line / compels the panel read; a None
        means the craving is satisfiable from the board (or unmapped) and
        nothing changes.
        """
        craving = ((self._craving_cache or {}).get("item")
                   or self._last_craving or self.fallback.craved_item)
        if not craving:
            return None
        cf = normalize_item_name(craving)
        entry = CRAVING_PRODUCERS.get(cf)
        if entry is None:
            for key, val in CRAVING_PRODUCERS.items():
                if cf.startswith(key) or key.startswith(cf):
                    entry = val
                    break
        if entry is None:
            return None
        family, components = entry
        for cell in board.cells:
            if not cell.item_id or not cell.occupied:
                continue
            nid = normalize_item_name(cell.item_id)
            if (nid == cf or nid.startswith(cf) or cf.startswith(nid)
                    or nid in components
                    or any(nid.startswith(c) for c in components)):
                return None
            if nid.startswith(family):
                return None
        return family

    def _craving_line_text(self) -> str:
        """Single merged cravings line for get_board_state.

        Priority:
        1. Cached menu read exists -> show its full detail (item, level,
            count, reward). The bubble identity is strictly less informative
            (no level/count/reward), so it's dropped when they agree.
        2. Cached menu read exists BUT the current bubble icon shows a
            DIFFERENT item -> the craving just switched since the last menu
            read; surface both so the model knows to call get_cravings.
        3. No cache yet -> bubble identity only, as a cue to call
            get_cravings once to learn the details.
        """
        cache = self._craving_cache
        bubble = self._bubble_craving_item()
        if cache is None:
            # identity-only cue; empty when no bubble match. The level/count
            # live only in the menu — name the call explicitly, otherwise
            # the model never takes the optional tool and merges blind
            # past the craved level (observed Sep 7: two eyemonster_lvl1
            # merged to lvl2 for a lvl1 craving).
            return (f"Cravings (bubble): {bubble} (level/count unknown — "
                    f"call get_cravings)") if bubble else ""
        step = cache.get("step", 0)
        age = max(0, self._step_count - step)
        menu = (f"{cache.get('item')} (lvl {cache.get('level')}) "
                f"{cache.get('count_done')}/{cache.get('count_required')}, "
                f"reward +{cache.get('reward')} (read {age} steps ago)")
        if bubble and bubble != cache.get("item"):
            return (f"Cravings: bubble shows {bubble}; cached menu read {menu} "
                    f"(bubble changed since the last menu read — call get_cravings)")
        return f"Cravings: {menu}"

    def _bubble_craving_item(self) -> str:
        """Current bubble icon item identity (no tap), else empty string."""
        if self.cravings is None or self._frame is None:
            return ""
        try:
            item_id, _score = self.cravings.match_bubble(self._frame)
        except Exception:
            return ""
        return normalize_item_name(item_id) if item_id else ""

    def _observe_craving_bubble(self) -> None:
        """Cheap no-tap craving tracking per step (live, reader wired).

        Logs `craving_seen` when the banked bubble icon indicates the craved
        item; logs `craving_changed` when it switches to a different item. The
        bubble identity alone can't give count/reward (those need the menu via
        `get_cravings`), so this only tracks identity transitions — full
        progress/completion events come from `_track_cravings` on menu reads.
        """
        if self.cravings is None or self._frame is None:
            return
        try:
            item_id, score = self.cravings.match_bubble(self._frame)
        except Exception:
            return
        if item_id is None:
            return
        item_id = normalize_item_name(item_id)   # 'rib cage' -> 'ribcage' (matches board ids)
        self.fallback.craved_item = item_id  # heuristic feeds the craving even if menu read is empty
        if item_id != self._last_craving:
            self.log.log("craving_seen", item=item_id, score=round(score, 3),
                            previous=self._last_craving)
            # Family switch: the level is unknown until a fresh menu read, so
            # clear both the believed level and the bubble cache's level cue —
            # a stale skeleton-lvl3 belief must not survive a switch to bone.
            self._last_craving = item_id
            self._last_craving_level = None
            self.fallback.craved_level = None
        elif score >= self.cravings.threshold:
            self.log.log("craving_seen", item=item_id, score=round(score, 3),
                            unchanged=True)

    def _champion_line_text(self) -> str:
        """Champion spawn status line for get_board_state.

        Rendered from the cached champion-screen read (no tap). Empty when no
        read has happened yet or no champion detail is known.
        """
        cache = self._champion_cache
        if cache is None:
            return ""
        champions = cache.get("champions") or []
        if not champions:
            return ""
        step = cache.get("step", 0)
        age = max(0, self._step_count - step)
        parts = []
        for c in champions:
            name = c.get("name")
            if not name:
                continue
            parts.append(f"{name} ({c.get('progress') or '?'}"
                            f"{', ready' if c.get('ready') else ''})")
        return f"Champions: {'; '.join(parts)}"
        # (read {age} steps ago) appended by caller if needed

    def _observe_champion_tracker(self) -> None:
        """Cheap no-tap champion tracking per step (live, reader wired).

        Identifies which champion is queued on the lair's tracker element via
        the portrait bank; logs `champion_seen` on identity. Full progress/
        ready state needs the champion screen via `get_champions` (logged by
        `_track_champions`), so this only tracks identity transitions.
        """
        if self.champions is None or self._frame is None:
            return
        try:
            cid, score = self.champions.match_tracker(self._frame)
        except Exception:
            return
        if cid is None:
            return
        if cid != self._last_champion:
            self.log.log("champion_seen", champion=cid, score=round(score, 3),
                            previous=self._last_champion)
            self._last_champion = cid
        else:
            self.log.log("champion_seen", champion=cid, score=round(score, 3),
                            unchanged=True)

    def _unidentified_cells(self, board) -> list[tuple[int, int]]:
        """Occupied cells with no trustworthy template label (candidates for discovery).

        Two cases qualify:
        - item_id is None but the cell is occupied with a low best-match
            (< KNOWN_DIP_MIN): a genuinely new item with no template in the bank
            (e.g. a fresh-save Grave Lvl 1). Without this, such a cell reads as
            "empty" and the identify_item escape hatch can never fire on it.
            (A KNOWN item in a deep idle-bob dip also yields item_id=None but
            still matches its own bank >= KNOWN_DIP_MIN — it is NOT listed, so
            we don't re-mint __alt ids by popup-tapping known items.)
        - a template id that is not in the bank or scores below LABEL_MIN_SCORE.
        The fixed NecroMerger cell is excluded: it is a known station and is
        never popup-tapped (side effects, garbage-id minting).
        """
        cells = []
        for cell in board.cells:
            if (cell.row, cell.col) == necromerger_cell():
                continue
            if cell.item_id is None:
                if cell.occupied and cell.score < KNOWN_DIP_MIN:
                    cells.append((cell.row, cell.col))
                continue
            if (self.classifier is not None
                    and not self.classifier.has(cell.item_id)):
                cells.append((cell.row, cell.col))
                continue
            if cell.score < LABEL_MIN_SCORE:
                cells.append((cell.row, cell.col))
        return cells

    def _ask_once(self, messages: list[dict]) -> tuple[str, Move | None]:
        """One LLM round for the final action JSON, via the `{"action":` prefill.

        The action is a constrained single-JSON choice, so we skip the
        reasoning model's thinking block entirely (assistant prefill). The old
        reasoning answer call wasted its whole generation budget thinking and
        often failed to emit JSON (retry_prefill on ~45% of steps in the
        Aug 12 soak), doubling latency for no gain on a 4B model — reasoning
        stays on for the observation/tool rounds where it buys correctness.

        The tool-call/tool-result history is STRIPPED before the prefill:
        llama.cpp rejects a trailing assistant prefill that follows tool
        messages ("Cannot have 2 or more assistant messages at the end of the
        list" — the Aug 19 400 that made every decision round fail with
        `llm_error`, so the heuristic played every move). The get_board_state
        result text is folded back into the user message so the model still
        sees the board state the tool round produced.
        """
        msgs, tool_texts = self._strip_tool_history(messages)
        # Answer round uses the SHORT decision prompt (geometry + hard rules
        # + schema) instead of the full ~25k-char system prompt: the tool
        # rounds already ran under the full prompt, and everything the
        # decision needs (tags, whitelist, hints, checklist) is in the Tool
        # results below. Falls back to the incoming system message when no
        # short prompt was staged (tests, offline callers).
        if getattr(self, "_answer_system", None):
            for i, m in enumerate(msgs):
                if m.get("role") == "system":
                    msgs[i] = {**m, "content": self._answer_system}
                    break
        if tool_texts:
            joined = "\n\n".join(tool_texts)
            for i, m in enumerate(msgs):
                if m.get("role") == "user" and isinstance(m.get("content"), list):
                    user_msg = dict(m)
                    user_msg["content"] = list(m["content"]) + [
                        {"type": "text", "text": f"Tool results:\n{joined}"}]
                    msgs[i] = user_msg
                    break
            else:
                msgs.append({"role": "user", "content": joined})
        msgs.append({"role": "assistant", "content": '{"action":'})
        try:
            reply, full_msg = self.client.chat(msgs, max_tokens=96)
        except LLMError:
            # Text-only server (e.g. Qwen3-4B-Instruct-2507): the image part
            # makes llama-server return 500. Retry once without it and stay
            # text-only for the rest of the session — decisions still get the
            # full board via the get_board_state tool-result text.
            if not self._vision_ok:
                raise
            self._vision_ok = False
            self.log.log("vision_unavailable",
                            reason="image request failed; continuing text-only")
            msgs = _strip_images(msgs)
            reply, full_msg = self.client.chat(msgs, max_tokens=96)
        # Log the ASSEMBLED assistant message (prefill seed + reply) so
        # llm_chats.jsonl shows the complete answer instead of just the
        # `{"action":` seed. The server saw the truncated seed; the log is
        # for our own replay/debugging, so reconstruct the full text.
        # --jinja servers echo the seed back inside `content` — in that case
        # the reply is ALREADY the full object; don't double the prefix.
        seed = '{"action":'
        assistant_content = (reply if reply.lstrip().startswith(seed)
                                else seed + reply)
        self._log_chat(msgs, reply, None, full_msg,
                        assistant_content=assistant_content)
        return reply, self._parse_action(reply)

    @staticmethod
    def _strip_tool_history(messages: list[dict]) -> tuple[list[dict], list[str]]:
        """Drop tool-call/tool-result messages from a chat history, returning
        the cleaned conversation plus the tool-result texts.

        The `{"action":` answer prefill must not follow tool messages (the
        server rejects consecutive trailing assistant turns), so the answer
        round is rebuilt from the base prompt plus the board-state texts that
        the tool rounds produced (injected into the user turn instead)."""
        cleaned: list[dict] = []
        tool_texts: list[str] = []
        for m in messages:
            if m.get("role") == "tool":
                c = m.get("content")
                if c:
                    tool_texts.append(c)
                continue
            if m.get("role") == "assistant" and m.get("tool_calls"):
                continue
            cleaned.append(m)
        return cleaned, tool_texts

    @staticmethod
    def _parse_action(reply: str) -> Move | None:
        # Strict-then-loose: the thinking model leaks tool-call syntax into
        # the action answer (observed `{"action":"merge",...}</tool_call>
        # {"name":...}` fragments). Strip tool-call markup first so a valid
        # action buried in protocol debris still parses; fall back to the
        # raw reply (old behavior) when stripping yields nothing.
        cleaned = re.sub(r"</?tool_call>", "", reply or "")
        cleaned = re.sub(r"\{\s*\"name\"\s*:.*", "", cleaned).strip()
        for text in ([cleaned] if cleaned != (reply or "") else []):
            move = VisionDrivenPlanner._parse_action_json(text)
            if move is not None:
                return move
        return VisionDrivenPlanner._parse_action_json(reply)

    @staticmethod
    def _parse_action_json(text: str) -> Move | None:
        try:
            data = _extract_json(text)
        except (ValueError, json.JSONDecodeError):
            return None
        action = data.get("action")
        if action == "merge":
            a, b = data["a"], data["b"]
            return Move(kind="merge", cell_a=(int(a[0]), int(a[1])), cell_b=(int(b[0]), int(b[1])))
        if action == "spawn":
            cell = data["cell"]
            return Move(kind="spawn", cell_a=(int(cell[0]), int(cell[1])), taps=SPAWN_TAPS)
        if action == "feed":
            cell = data["cell"]
            return Move(kind="feed", cell_a=(int(cell[0]), int(cell[1])))
        if action == "attack":
            cell = data["cell"]
            target = data.get("target")
            if not target or len(target) != 2:
                return None  # attack without a target = malformed
            return Move(kind="attack",
                        cell_a=(int(cell[0]), int(cell[1])),
                        target=(int(target[0]), int(target[1])))
        if action == "collect":
            return Move(kind="collect", cell_a=necromerger_cell(), taps=COLLECT_TAPS)
        if action == "idle":
            return Move(kind="idle")
        return None

    # `_loop_nudge` is a normal instance method (uses self._best_*_line);
    # `_correction` below is a @staticmethod.

    def _loop_nudge(self, board, rejected: set) -> str:
        """When the model is stuck re-proposing one action kind that keeps
        getting rejected (e.g. only `feed` moves in a row), produce a concrete
        nudge naming a VALID alternative move from the board hints — the
        `Best spawn` / `Best merge` / `Best attack` lines. The model
        demonstrably ignores the generic "do NOT propose these again"
        correction, but responds to a targeted, named move (observed in the
        17:41:07 retry-exhaustion, where the model never once proposed `spawn`
        despite a sparse board + an explicit `Best spawn: grave_lvl3 (4,3)` hint).

        Returns "" when there's no useful hint or the model isn't stuck.
        """
        if not board or not rejected:
            return ""
        kinds = [k for k, _, _ in rejected]
        if not kinds:
            return ""
        # How many consecutive/all rejections share the dominant kind.
        from collections import Counter
        counts = Counter(kinds)
        dominant, n = counts.most_common(1)[0]
        # Not stuck if the model is already exploring multiple kinds.
        if n < 3 or len(counts) < 1:
            return ""
        # The model is fixated on `dominant`. Offer the single strongest
        # valid hint whose kind is NOT the one it's stuck on.
        #
        # Special case first: if the ONLY viable spawner is a grave but the
        # Mana bar is below SPAWN_MANA_MIN, the grave spawn is mana-blocked
        # (`_best_spawn_line` will suppress it, and a spawn nudge would just
        # get re-rejected). The DIRECT prerequisite for that blocked spawn is
        # `collect` — tapping the NecroMerger refills Mana so the grave can be
        # spawned next step. The correct nudge is `collect`, not another
        # rejected feed/merge/spawn (the 18:53 retry_exhausted burned 49 feeds
        # and never emitted `collect`). Only steer to collect when there's no
        # mana-free chest to spawn instead.
        mana = getattr(self, "_mana_fraction", None)
        if mana is None:
            try:
                mana = read_mana_fraction(self._frame)
            except Exception:
                mana = None
        has_grave = self._graves_on_board(board)
        has_chest = any(
            c.item_id and (any(c.item_id.startswith(p) for p in CHEST_PREFIXES)
                           or any(c.item_id.startswith(p) for p in SLIME_SPAWN_PREFIXES))
            for c in board.cells)
        if (dominant != "collect" and has_grave and not has_chest
                and mana is not None and mana < SPAWN_MANA_MIN):
            return (f"You keep proposing `{dominant}` moves and they're all being "
                    f"rejected. The board needs Mana before it can spawn from the "
                    f"grave (the bar is ~{mana:.0%}). Use `collect` to tap the "
                    f"NecroMerger and refill Mana first, THEN spawn from the grave "
                    f"next step.")
        hints = []
        other_spawn = self._best_spawn_line(board)
        other_merge = self._best_merge_line(board)
        other_attack = self._best_attack_line(board)
        candidates = []
        if dominant != "spawn" and other_spawn:
            candidates.append(other_spawn)
        if dominant != "merge" and other_merge:
            candidates.append(other_merge)
        if dominant != "attack" and other_attack:
            candidates.append(other_attack)
        # Prefer spawn when the board has room, matching SPAWN-FIRST.
        if not candidates:
            return ""
        best = candidates[0]
        prefix = best.split(":")[0]  # "Best spawn" / "Best merge" / "Best attack"
        return (f"You keep proposing `{dominant}` moves and they're all being "
                f"rejected. Try a DIFFERENT action kind instead — the board "
                f"suggests: {best}. Follow that {prefix} hint exactly.")

    def _graves_on_board(self, board) -> bool:
        return any(
            c.item_id and any(c.item_id.startswith(p) for p in SPAWN_PREFIXES)
            for c in board.cells)

    @staticmethod
    def _correction(reason: str, rejected: set | None = None,
                    nudge: str = "") -> str:
        hints = {
            "same_cell": "pick two DIFFERENT cells.",
            "cell_a_empty": "that cell is truly empty (nothing on it).",
            "cell_b_empty": "that cell is truly empty (nothing on it).",
            "cell_a_unidentified": "that cell is OCCUPIED but unidentified — a sprite is there but the template bank has no label for it. Call identify_item on it to learn what it is before merging/feeding.",
            "cell_b_unidentified": "that cell is OCCUPIED but unidentified — a sprite is there but the template bank has no label for it. Call identify_item on it to learn what it is before merging/feeding.",
            "merge_mismatch": "those cells hold different items (different levels of the same creature are different items) — only cells with the SAME id merge.",
            "merge_is_champion": "that's a Champion (e.g. The Peasant) — champions are enemies and can never be merged.",
            "merge_max_level": "those items are MAX level (their popup says only 'Feed to the Devourer.') — they can never merge again, so feed one to the Devourer instead.",
            "merge_low_margin": "one of those cells doesn't clearly match its label — pick a clearer pair.",
            "spawn_not_grave": "only a grave, chest, or supply cupboard can spawn items.",
            "spawn_low_mana": "the mana bar is too low to spawn — the grave spawn silently no-ops on an empty bar. Collect or feed instead. " + _resource_teaching("mana", short=True),
            "spawn_no_slime": "the slime vat is empty — the cupboard spawn silently no-ops with no Slime. Merge or feed instead. " + _resource_teaching("slime", short=True),
            "spawn_no_room": "the board is FULL — spawns silently no-op with no empty cell. Merge two identical items instead (the board state lists the available pairs) to free space.",
            "collect_mana_full": "the mana bar is already full — collecting is wasted, and a full bar means spawning is EASY right now. Spawn from the grave or merge instead.",
            "feed_is_station": "that cell is a station (grave/necromerger/manapool) — NEVER feed a station to the Devourer. (manapot / manapotion are Potions, NOT stations — they are feedable, but a full Mana bar wastes the feed: see `feed_mana_overflow`.)",
            "feed_is_champion": "that's a Champion (e.g. The Peasant) — never feed a champion; drag a damage-dealing creature onto it instead (action 'attack').",
            "feed_overflow": "that creature's feed value EXCEEDS the remaining satiety — the excess food is wasted and the game warns. Feed a creature whose (feed N) fits the remaining bar, or spawn/merge instead.",
            "feed_mana_overflow": "feeding a manapot/manapotion when the Mana bar is full WASTES the Mana (it can't be stored). The Mana cap is a hard limit — feed potions only when you have headroom, or build a Mana Pool to raise the cap. Tap chests on the board with `spawn` and feed max-level Rune stacks to gather the resources instead.",
            "feed_generous": "that creature is a high-value MANA GENERATOR — feeding it is a last resort. A cheaper feedable creature exists that fits the bar: feed that instead, or spawn from the grave to build cheap food.",
            "attack_is_station": "the attacker cell is a station (grave/necromerger/manapot) — stations don't deal damage. Pick a creature cell as the attacker instead.",
            "attack_is_champion": "the attacker cell is a Champion — only YOUR creatures attack; pick a non-champion cell with a known (dmg N) value as the attacker.",
            "attack_no_target": "attack requires a 'target' cell — the champion you want to hit. Reply with {\"action\":\"attack\",\"cell\":[r,c],\"target\":[r,c]}.",
            "attack_self_target": "you cannot attack a champion with itself — pick a DIFFERENT cell for the attacker and target.",
            "attack_target_not_champion": "the target cell is not a Champion — attacks only work on Champion cells (those labeled (champion) in get_board_state). Pick a Champion cell as the target.",
            "attack_no_damage": "that creature has no recorded damage value — the validator can't trust a 0-damage attack. Either (1) call identify_item on it to read its popup, (2) use a different creature that already shows (dmg N), or (3) pick a different action — the merge/feed menus don't need this stat.",
            "out_of_bounds": "cell coordinates are out of range (rows 0-4, cols 0-2).",
        }
        for key, hint in hints.items():
            if reason.startswith(key):
                msg = (f"Your proposed move was rejected: {reason}. {hint} "
                        "Look at the screenshot again and reply with ONLY a valid JSON action.")
                break
        else:
            msg = (f"Your proposed move was rejected: {reason}. "
                    "Look at the screenshot again and reply with ONLY a valid JSON action.")
        if rejected:
            def _mv(k):
                kind, a, b = k
                if kind == "merge":
                    return f"merge({a}+{b})"
                if kind in ("spawn", "feed"):
                    return f"{kind}({a})"
                return f"{kind}"
            blocked = ", ".join(sorted(_mv(k) for k in rejected))
            msg += (f"\nMoves already rejected this step — do NOT propose any of these again: "
                    f"{blocked}")
        if nudge:
            msg += f"\n{nudge}"
        return msg

    def _log_chat(self, messages: list[dict], reply: str, tool_calls: list | None = None,
                    full_msg: dict | None = None, assistant_content: str | None = None) -> None:
        """Log a chat round. If `full_msg` is provided include any reasoning
        content supplied by the assistant message dict (e.g. `reasoning_content`).
        If `assistant_content` is given, replace the LAST assistant message's
        content with it (used by `_ask_once` to log the full assembled answer
        instead of just the `{"action":` prefill seed).

        Also bumps the per-step LLM round counter (telemetry for the
        llm_budget event; getattr-guarded for bare test instances).
        """
        try:
            self._step_llm_rounds = getattr(self, "_step_llm_rounds", 0) + 1
        except Exception:
            pass
        sanitized = []
        for i, m in enumerate(messages):
            content = m["content"]
            if isinstance(content, list):
                content = [p if p.get("type") != "image_url"
                            else {"type": "image_url", "image_url": {"url": "[image]"}}
                            for p in content]
            if (assistant_content is not None
                    and i == len(messages) - 1
                    and m.get("role") == "assistant"):
                content = assistant_content
            sanitized.append({**m, "content": content})
        reasoning = None
        if full_msg:
            # common servers place internal chain-of-thought in `reasoning_content`
            reasoning = full_msg.get("reasoning_content") or full_msg.get("reasoning")
        self._turn += 1
        with self.chat_log_path.open("a") as f:
            f.write(json.dumps({"t": time.time(), "step": self._step_count,
                                "turn": self._turn, "messages": sanitized,
                                "reply": reply, "tool_calls": tool_calls,
                                "reasoning": reasoning}) + "\n")
