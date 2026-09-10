"""Shared planner constants, free of imports from other planner modules.

Kept import-free of other planner modules so both `planner/agent.py` and
`planner/merge.py` can use them without a circular import.
"""

import re

# Merge only when the best match clearly beats the runner-up, so a mislabeled
# item (small margin) can't trigger a merge the game refuses (retry-loop).
MERGE_MIN_MARGIN = 0.10

# Champions (The Peasant, etc.) invade the board. They are enemies, not items:
# never merged, never spawned, never fed as food (they are defeated only by
# feeding OTHER creatures to the Devourer).
CHAMPION_PREFIXES = ("peasant", "knight", "cleric", "paladin", "rival", "protector")

# Feeding the Devourer wastes food ONLY by overflow: the satiety bar caps at the
# max satiety Y, so a feed worth Z when satiety is X loses X+Z-Y food. A craved
# feed is worth Z + craving-bonus B. We tolerate overflow up to SATIETY_TOL_FRACTION
# of the current max satiety (a tiny overfill is noise); above that a craved feed
# is refused so the item's value isn't wasted (cravings persist, so the craved
# item can be fed after the bar resets on level-up).
SATIETY_TOL_FRACTION = 0.10

# What the Grave DIRECTLY spawns at the current Devourer level (bone/ribcage;
# skeletons only come from merging). Used by the `Best spawn` hint so it never
# claims the grave can conjure a skeleton.
GRAVE_SPAWN_PREFIXES = ("bone", "ribcage")

# What the Ice Chest spawns when tapped: each use releases one random rune onto
# the board (ice lvl1 60%, ice lvl2 20%, poison lvl1 20%) — a board-space cost.
# Mirrors GRAVE_SPAWN_PREFIXES for the chest's `Best spawn` hint.
CHEST_SPAWN_PREFIXES = ("icerune", "poisonrune")

# Chest tap is mana-free (unlike grave) — same spawn mechanic, no HUD cost.
# The game has several chest types: the Ice Box plus level reward chests
# (lockedchest / valuablechest). All are mana-free spawn stations. The
# `icebox` prefix catches icebox_unopened; the full ids catch the reward
# chests. (Aug 30 fix: lockedchest/valuablechest were MISSING, so spawning
# from them was wrongly rejected as `spawn_not_grave`.)
CHEST_PREFIXES = ("icebox", "lockedchest", "valuablechest")

# Slime-cost spawn stations: the Supply Cupboard spawns eye components for
# Slime (wiki: lvl1 ~1000/tap; higher levels cost more), and the Fridge
# likewise costs Slime to use (wiki: lvl1 500/tap-scale costs; Tier-13
# unlock). Mana-free like chests, but gated on Slime instead. Every
# consumer (validator, hints, whitelist, tags, fallback) reads this tuple,
# so grounding a new station's cost only requires adding its prefix here.
# Stations with ungrounded costs stay out (no invention).
SLIME_SPAWN_PREFIXES = ("supplycupboard", "fridge")

# The full grave-spawn family (bone -> ribcage -> skeletonN): members are cheaply
# rebuilt via the grave's chain, so sacrificing one to a craving when it would
# overflow the bar is unnecessary — spawning to rebuild it is preferred.
GRAVE_CHAIN_PREFIXES = ("bone", "ribcage", "skeleton")

# Craving exact-level matching: a craving is for ONE specific monster ("2x
# Skeleton Lvl 1"), and level is part of item identity — skeleton_lvl1 is
# distinct from skeleton_lvl2, so feeding a different level never counts. When
# the craving's level is unknown (only the bubble icon cue is visible, no menu
# read yet), only low-tier candidates (unleveled / _lvl1 / _lvl2) may be the
# craved monster — a high-level cell can never be CONFIRMED as the craving, and
# feeding the biggest monster is never what a craving asks for.
CRAVED_UNKNOWN_LEVEL_MAX = 2


def is_craved_precursor(item_id: str | None, craving_item: str | None,
                        craving_level: int | None,
                        chain_map: dict[str, list[str]] | None = None) -> bool:
    """True when `item_id` is a merge-chain PRECURSOR of the active craving.

    The craving is one exact monster (e.g. 'skeleton_lvl3'), and every member of
    its merge chain below that level (skeleton_lvl1, skeleton_lvl2, and the
    unleveled bases bone/ribcage that merge up into skeletons) is the RECIPE for
    the craved item — feeding one to the Devourer destroys the path to the
    craving. So while an exact level is known, those precursors must never be
    fed (they exist to merge toward the craving). Items at/above the craving
    level are NOT precursors (a skeleton_lvl5 can't merge down into lvl3, so
    feeding it loses nothing toward the craving).

    Requires `craving_level` to be known: with only the bubble family cue
    (level None) we cannot tell which level is craved, so the match degrades to
    the conservative CRAVED_UNKNOWN_LEVEL_MAX gate elsewhere and this returns
    False (nothing protected). Family-suffix matching (`_lvl<N>` ids) catches
    leveled members; chain_map (glossary `(chain)` blocks, family -> ids)
    additionally catches unleveled bases like bone/ribcage that `startswith`
    can't connect to the 'skeleton' family.""",
    if not item_id or not craving_item or craving_level is None:
        return False
    have = item_level(item_id)
    if item_name_matches(item_id, craving_item):
        # A leveled family member below the craving level is a precursor; an
        # unleveled family base (e.g. the 'skeleton' base template) is the
        # lowest rung, always a precursor. Equal/higher levels are not.
        return have is None or have < craving_level
    if have is not None:
        return have < craving_level
    chain = (chain_map or {}).get(normalize_item_name(craving_item))
    return bool(chain and item_id in chain)


def normalize_item_name(name: str) -> str:
    """Canonical form for matching item references across sources: lowercase,
    no non-alphanumerics.

    The game UI and our cravings bubble bank key items by their DISPLAY name
    ('Rib Cage' -> 'ribcage', 'Ice Rune' -> 'icerune'), while board template
    ids are compact ('ribcage') and may carry a level suffix
    ('skeleton_lvl1'). Raw `startswith` comparisons silently fail to connect a
    craving/menu name to its board cells, so every craving match normalizes
    both sides through here (see item_name_matches)."""
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


# Canonical station family -> spaced display name for model-facing text.
# Probed live: this thinking model derails mid-reasoning on compound
# words ("supplycupboard" burns the whole budget; "grave" lands), so every
# prompt, example call, and compel message uses the SPACED form, and every
# intake canonicalizes via canonical_family(). Single-word families
# display as-is.
STATION_DISPLAY_NAMES = {
    "supplycupboard": "supply cupboard",
    "manapool": "mana pool",
    "foulchicken": "foul chicken",
    "slimevat": "slime vat",
    "darkstores": "dark stores",
    "crashedsaucer": "crashed saucer",
    "soulgrinder": "soul grinder",
    "unexpectedparcel": "unexpected parcel",
}


def display_family(family: str) -> str:
    """Model-facing form of a canonical station family ('supply cupboard')."""
    canon = normalize_item_name(family)
    return STATION_DISPLAY_NAMES.get(canon, canon)


def canonical_family(name: str) -> str:
    """Canonical form of any station reference ('supply cupboard' or
    'SupplyCupboard' -> 'supplycupboard'). All tool-arg intake funnels
    through here so spaced and compact forms resolve identically."""
    return normalize_item_name(name)


def item_name_matches(item_id: str, craving_or_name: str) -> bool:
    """True when `item_id` names the same item family as `craving_or_name`.

    `item_id` is a board template id (maybe level-suffixed); `craving_or_name`
    may be a craving base name ('skeleton'), a display name ('rib cage'), or
    already compact ('ribcage'). Normalizing both sides means 'rib cage' ->
    'ribcage' matches 'ribcage', and 'skeleton' matches 'skeleton_lvl1'.

    FAMILY-level match only: a level-1 skeleton craving also matches
    skeleton_lvl5 here. For the exact-level rule feed consumers need, use
    `craved_matches`."""
    if not item_id or not craving_or_name:
        return False
    return normalize_item_name(item_id).startswith(normalize_item_name(craving_or_name))


def item_level(item_id: str) -> int | None:
    """The merge level a board template id encodes in its `_lvl<N>` suffix
    ('skeleton_lvl3' -> 3, 'manapool_lvl2' -> 2), or None for unleveled ids
    (bone, ribcage, manapot, ...)."""
    if not item_id:
        return None
    m = re.search(r"_lvl(\d+)", item_id)
    return int(m.group(1)) if m else None


def craved_matches(item_id: str, craving_item: str | None,
                   craving_level: int | None = None) -> bool:
    """True when `item_id` is the EXACT monster the craving wants.

    A craving is level-specific — the game UI shows the craved monster with its
    level ("2x Skeleton Lvl 1") and feeding a different level of the same
    family does not count toward it. So `craved_matches` requires the same
    family AND the same level; `skeleton_lvl5` is never the craved cell for a
    `skeleton_lvl1` craving.

    `craving_level` comes from the cravings-menu read (the authoritative
    source). When it's None (only the bubble icon cue is known, or a caller
    passes the base name only), the craving item's own `_lvl<N>` suffix is used
    if present; with neither, the match degrades conservatively to low-tier
    candidates only (unleveled / _lvl1 / _lvl2, see CRAVED_UNKNOWN_LEVEL_MAX)
    — an unconfirmed high-level cell is never treated as the craving.

    Feed consumers use this (best_feed_cell, _craved_feed_wasteful, the feed
    trigger, hints, bonus learning, the LLM feed menu) so a level-1 skeleton
    craving can never steer a high-level skeleton into the Devourer."""
    if not item_id or not craving_item:
        return False
    if not item_name_matches(item_id, craving_item):
        return False
    if craving_level is None:
        craving_level = item_level(craving_item)
    have = item_level(item_id)
    if craving_level is not None:
        return have == craving_level
    return have is None or have <= CRAVED_UNKNOWN_LEVEL_MAX
