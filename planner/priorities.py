"""Feat-driven move priorities.

The bot's action ordering (feed vs merge vs spawn vs collect) is NOT a
hardcoded ladder: it is derived from the player's ACTIVE FEATS (missions,
read from the FEATS panel) plus the current craving, which is itself treated
as a feed objective. A nearly-done feat outranks everything else; a craving
feed usually wins unless a near-done feat of another kind outranks it.

Parsing is deliberately tolerant: feat text comes from LLM OCR of the FEATS
panel (ornate font, spelling drift), so unknown/garbled names classify as
`neutral` and can never corrupt the priority order.
"""

from dataclasses import dataclass
import re

from planner.constants import normalize_item_name

# Base weight per move kind = the effective ordering when no feats are active
# (merge > feed > spawn > collect, the old hardcoded ladder). Feat completion
# ratios add on top, so a nearly-done feat of any kind overtakes the base.
BASE_KIND_WEIGHT = {
    "merge": 2.0,
    "feed": 1.5,
    "spawn": 1.0,
    "collect": 0.5,
    # attack: outranks every other kind by default. A Champion on the board
    # is an UNDEAD threat — it occupies a board cell and steals food/mana
    # while alive, so we always want to hit it as fast as our best damage
    # creature allows. Champions DO NOT spawn on a timer; they spawn when
    # you've merged enough of a specific family (e.g. The Peasant tracks
    # Skeletons, Zombies, and Mummies — merging any of those pushes its
    # progress bar toward 100%, at which point the Champion materializes
    # on the board). So a high-merge-pressure save is naturally punctuated
    # by combat — you can't avoid it by just merging faster. The attack
    # branch in HeuristicPlanner only fires when a champion is actually on
    # the board, so the high weight is safe (no false-positive attacks
    # when the board is clear).
    "attack": 3.0,
    "neutral": 0.0,
}

# Extra weight a matching craving feed carries (its food bonus + craving
# progress), so an unresolved craving usually beats a merge unless a
# competing feat is close to completion. Tunable — single place.
CRAVING_BONUS_WEIGHT = 0.6

# Feat names naming a STATION build ("Build a Mana Pool." / "Build the
# Grave"). A build feat is completed by buying the station, which costs Ice
# Runes — and Ice comes from feeding icerune stacks to the Devourer (Aug 25).
# So while a build strategy is active and an icerune stack sits on the board,
# feeding that stack IS the income action toward the strategy.
STRATEGY_BUILD_MARKERS = ("build a", "build the", "build an", "construct")

# An income objective carries this boost — high enough to outrank a merge
# (2.0) when the strategy needs Ice and the stack is on the board, low enough
# to stay below a near-done feat of any kind. Tunable — single place.
INCOME_BONUS_WEIGHT = 0.75

# Item families that grant currency when fed (the income stacks). All
# follow the same max-level-or-bust rule: bigger stacks grant more
# currency per the wiki (icerune lvl3 grants 12 vs lvl1's 2; coin lvl4
# grants 30 vs lvl1's 2; gem lvl3 grants 12 vs lvl1's 2).
#   - The 5 Rune chains (ice / poison / blood / moon / death) feed the
#     station-buying economy (every build strategy needs a station).
#   - Coins (Gold) feed the Shop / Merchant / Trader economy.
#   - Gems feed the premium Shop / skins / Wobulan upgrades economy.
#   - Astro Coins, Energy Cubes, Wobular follow the same pattern
#     (max-level yields more) but aren't banked yet — add to this tuple
#     when the bot learns to recognize them.
INCOME_FAMILIES = (
    "icerune", "poisonrune", "bloodrune", "moonrune", "deathrune",
    "coin", "gem",
)

# Chest spawn bonus — when the goal is to build a station (needs ice + poison
# runes) and an unopened chest sits on the board, opening it (spawn on the
# chest, mana-free) is the income. Rank by need: Build a Mana Pool needs
# currency, so chest > merge rune > feed max. Boost spawn above merge when
# building, but hard guard in best_feed_cell still ensures a mergeable base
# isn't fed. Large boost previously starved merges; 1.1 keeps spawn 2.1 > merge
# 2.0 when building, and feed-income is suppressed while chest present.
CHEST_SPAWN_BONUS_WEIGHT = 1.1
CHEST_FAMILIES = ("icebox",)


def is_build_strategy(strategy: dict | None) -> bool:
    """True when the committed strategy is a station build (needs a station
    bought from the Station panel — costs Ice Runes)."""
    if not strategy:
        return False
    feat = (strategy.get("feat") or "").lower()
    return any(m in feat for m in STRATEGY_BUILD_MARKERS)


# Keyword families -> move kind. Feed markers checked first: "Feed N ..." and
# Devourer-advance goals ("Unlock the Mana Well", "Reach Lvl N Devourer") all
# mean "feed the Devourer", which is what our `feed` action does.
_FEED_MARKERS = ("feed", "unlock the", "unlock a", "unlock an",
                 "reach a lvl", "reach lvl")
_MERGE_MARKERS = ("merge", "create a lvl", "create an lvl", "own a lvl",
                  "own an lvl", "lvl 2+", "lvl 3+", "lvl 4+", "lvl 5+")
_SPAWN_MARKERS = ("spawn",)
_COLLECT_MARKERS = ("collect", "tap the", "tap a")

_RATIO_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

# Common item families that a feat may name, for the (best-effort) target.
# Display forms included ('rib cage', 'ice rune'). OCR drift is tolerated by
# normalize_item_name (space/punctuation/case-insensitive), though a mangled
# OCR token may still miss — `target` is a refinement, not a gate.
_ITEM_LEXICON = (
    "bone", "ribcage", "rib cage", "skeleton", "zombie", "grave",
    "manapot", "manapool", "ice rune", "icerune", "poison rune",
    "poisonrune", "blood rune", "bloodrune", "valuable chest",
    "valuablechest", "chest",
)


@dataclass
class Objective:
    """A parsed objective driving move selection."""
    kind: str            # "merge" | "feed" | "spawn" | "collect" | "neutral"
    target: str | None   # item family (compact) the objective names, or None
    ratio: float         # completion 0..1 (0 when unknown)
    text: str = ""       # original feat/craving text, for hints/logs


def _ratio(text: str) -> float:
    """Completion ratio (0..1) from text like '16/50' or '1/4 Feats Completed'.
    Missing/unparseable -> 0.0 (unknown progress never boosts an objective)."""
    m = _RATIO_RE.search(str(text or ""))
    if not m:
        return 0.0
    num, den = int(m.group(1)), int(m.group(2))
    if den <= 0:
        return 0.0
    return max(0.0, min(1.0, num / den))


def _target(name: str) -> str | None:
    """Best-effort item family named by a feat, compact form, or None."""
    low = name.lower()
    for tok in _ITEM_LEXICON:
        if tok in low:
            return normalize_item_name(tok)
    return None


def parse_obj(feat: dict) -> Objective:
    """Classify one FEATS-panel entry into an Objective.

    A completed feat is `neutral` (it no longer steers). An unparseable /
    garbled feat is `neutral` too, so bad OCR never distorts the order.
    """
    name = (feat.get("name") or "").strip()
    if not name:
        return Objective("neutral", None, 0.0, "")
    done = bool(feat.get("done"))
    ratio = _ratio(feat.get("progress")) or _ratio(name)
    if done:
        return Objective("neutral", None, ratio, name)
    low = name.lower()
    if any(k in low for k in _FEED_MARKERS):
        kind = "feed"
    elif any(k in low for k in _MERGE_MARKERS):
        kind = "merge"
    elif any(k in low for k in _SPAWN_MARKERS):
        kind = "spawn"
    elif any(k in low for k in _COLLECT_MARKERS):
        kind = "collect"
    else:
        kind = "neutral"
    return Objective(kind, _target(name), ratio, name)


def craving_objective(item: str | None, count_done=None,
                      count_required=None) -> Objective | None:
    """Fold the current craving into the same objective scheme as a feed goal.

    Feeding the craved creature grants a food bonus on top of its feed value,
    so the craving is a feed objective carrying CRAVING_BONUS_WEIGHT — it
    competes on the same scale as feats instead of sitting in a hardcoded top
    slot. Returns None when no craving is known.
    """
    if not item:
        return None
    ratio = 0.0
    if isinstance(count_required, int) and count_required > 0 \
            and isinstance(count_done, int):
        ratio = max(0.0, min(1.0, count_done / count_required))
    return Objective("feed", normalize_item_name(item), ratio, f"craving {item}")


def income_objective(strategy: dict | None, board,
                     max_level_ids=None) -> Objective | None:
    """An income objective: when a build strategy is active and a MAX-LEVEL
    icerune stack is on the board, feeding that stack grants the Ice Runes
    the strategy needs. Returns None otherwise (no-op).

    MAX-LEVEL GATE (Aug 26): the boost fires only when an icerune stack is
    DONE growing (max level — nothing left to merge on it). While only lvl1/
    lvl2 stacks exist, MERGING them up IS the income work (chain lookahead
    already ranks those merges first), so a feed boost here would starve the
    merge pipeline — 7 ribcages sat unmerged for a whole soak because a lvl1
    stack alone kept feed weight above merge every step.

    The objective is a feed objective carrying INCOME_BONUS_WEIGHT — it
    competes on the same scale as feats/craving, so the kind order puts the
    income feed ahead of merges once the stack is ready."""
    if not is_build_strategy(strategy):
        return None
    if board is None:
        return None
    max_level = set(max_level_ids or ())
    for c in board.cells:
        if not c.item_id:
            continue
        base = normalize_item_name(c.item_id)
        matched = next(
            (f for f in INCOME_FAMILIES
             if base.startswith(f) or f in base), None)
        if matched is None:
            continue
        if c.item_id not in max_level:
            continue  # still mergeable — merging up is the income step
        return Objective("feed", matched, 0.5,
                         f"{matched} income for {strategy.get('feat')}")
    return None


def chest_spawn_objective(strategy: dict | None, board) -> Objective | None:
    """Spawn objective for chests: when building a station needs ice+poison
    runes and a chest sits on the board, opening it (spawn on icebox, mana-free)
    is the income. Returns a spawn objective carrying CHEST_SPAWN_BONUS_WEIGHT so
    spawn outranks ribcage feeds."""
    if not is_build_strategy(strategy):
        return None
    if board is None:
        return None
    for c in board.cells:
        if not c.item_id:
            continue
        base = normalize_item_name(c.item_id)
        if any(base.startswith(f) or f in base for f in CHEST_FAMILIES):
            return Objective("spawn", "icebox", 0.5,
                             f"chest spawn for {strategy.get('feat')}")
    return None


def feat_weights(objectives: list[Objective],
                   craving: Objective | None = None,
                   income: Objective | None = None,
                   chest_spawn: Objective | None = None) -> dict[str, float]:
    """Per-kind weights: base + each active objective's completion ratio.

    Also exposes a `feat_target` entry (compact item family, or None): the
    target of the highest-ratio non-neutral objective, which the feed branch
    uses to prefer feeding exactly what a near-done feat asks for.
    """
    w = dict(BASE_KIND_WEIGHT)
    best_ratio = -1.0
    target = None
    for o in objectives:
        if o.kind == "neutral":
            continue
        w[o.kind] = w.get(o.kind, 0.0) + o.ratio
        if o.ratio > best_ratio and o.kind == "feed" and o.target:
            best_ratio, target = o.ratio, o.target
    if craving is not None:
        w["feed"] = w.get("feed", 0.0) + craving.ratio + CRAVING_BONUS_WEIGHT
    # When a chest is available while building, the chest is the better ice+
    # poison income than feeding an existing icerune stack — opening it should
    # outrank the income feed. Suppress the income boost while the chest is
    # present so spawn (chest) outranks feed; after the chest is gone income
    # resumes. This keeps merge > spawn > feed, so merges aren't starved.
    if chest_spawn is not None and income is not None:
        # chest present: prefer opening it before consuming stacks
        income = None
    if income is not None:
        w["feed"] = w.get("feed", 0.0) + INCOME_BONUS_WEIGHT
    if chest_spawn is not None:
        w["spawn"] = w.get("spawn", 0.0) + CHEST_SPAWN_BONUS_WEIGHT
    w["feat_target"] = target
    return w


def kind_order(weights: dict[str, float] | None) -> list[str]:
    """Move-kind check order from weights (highest weight first).

    Empty/None weights -> the base order (attack, merge, feed, spawn,
    collect) — the previous hardcoded behavior plus the new attack kind
    (Aug 26: champion combat). Feat completion ratios add on top via
    `feat_weights`, so a nearly-done feat of any kind overtakes the base."""
    base = weights or BASE_KIND_WEIGHT
    return sorted(("attack", "merge", "feed", "spawn", "collect"),
                  key=lambda k: -base.get(k, BASE_KIND_WEIGHT[k]))
