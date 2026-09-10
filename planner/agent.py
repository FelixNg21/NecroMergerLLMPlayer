"""Agent policy: given the board, decide the next action. Implement the AI here."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from planner.constants import (
    CHAMPION_PREFIXES,
    CHEST_PREFIXES,
    GRAVE_CHAIN_PREFIXES,
    SATIETY_TOL_FRACTION,
    SLIME_SPAWN_PREFIXES,
    SLIME_SPAWN_PREFIXES,
    craved_matches,
    is_craved_precursor,
    item_name_matches,
)
from planner.merge import is_merge_material, ranked_merge_groups
from planner.priorities import kind_order
from vision.grid import BoardState, Cell, session_geometry
from vision.hud import COLLECT_MANA_MAX, SPAWN_MANA_MIN, read_mana_fraction

CONGESTION_THRESHOLD = 3          # empty cells left before we feed
FEED_PROGRESS_MAX_VALUE = 25      # feed-to-progress only feeds creatures this cheap (or cheaper) — bone/ribcage/skeleton_lvl1-2, never merge-chain material
SPAWN_TAPS = 2
CHEST_SPAWN_TAPS = 2              # chests are mana-free, double-tap per use (select+consume, was double-tap in QueueBox; single tap does nothing)
COLLECT_TAPS = 3
NOOP_BACKOFF_THRESHOLD = 2        # consecutive merge no-ops before a pair is excluded
NOOP_BACKOFF_STEPS = 10           # steps a no-op pair is excluded from merge candidates
# STATION_PREFIXES no longer includes "manapot" — the bot's template
# bank has both `manapot_lvl*` (3 templates) and `manapotion_lvl3` (1 template);
# the wiki treats the latter as a Potions item (feedable, gives Mana when fed)
# and the former as an obsolete bank entry. Removing the prefix match lets the
# model feed any manapot/manapotion; the existing feed_overflow check
# (eff > satiety_remaining + tol) handles the "bar is full" case correctly.
# (Aug 28: the user reported a "mana_pot" feed being rejected as a station;
# the actual reject reason should be feed_overflow, not feed_is_station.)
STATION_PREFIXES = ("grave", "necromerger", "manapool")  # never feed these
SPAWN_PREFIXES = ("grave",)                   # graves spawn via mana
ALL_SPAWN_PREFIXES = SPAWN_PREFIXES + CHEST_PREFIXES + SLIME_SPAWN_PREFIXES  # grave + chest + cupboard
# Our own character: geometrically fixed in the TOP-RIGHT cell and never
# merged. Its idle animation is too variable for reliable template matching, so
# treat the cell as the necromerger station regardless of what the classifier
# reports there. The CELL moves as the board grows ((0,2) on 5x3, (0,3) on the
# live 5x4) — always resolve through necromerger_cell(), never this constant.
NECROMERGER_CELL = (0, 2)   # deprecated: 5x3 fallback, kept for back-compat


def necromerger_cell() -> tuple[int, int]:
    """The NecroMerger's fixed cell for the CURRENT session geometry."""
    return (0, session_geometry().cols - 1)


class _MergeNoopRegistry:
    """Shared merge-noop backoff state.

    main.py calls `record` when a merge verifies as a no-op (the game ignored
    the swipe). A pair that no-ops NOOP_BACKOFF_THRESHOLD times is excluded
    from merge candidates for NOOP_BACKOFF_STEPS steps — re-proposing a pair
    the game keeps refusing (phantom labels, stale geometry, a max-level the
    bank doesn't know) wastes a swipe every step. Shared between a planner and
    its heuristic fallback so BOTH stop proposing the same dead pair.
    """

    def __init__(self):
        self._counts: dict[tuple, int] = {}
        self._backoff: dict[tuple, int] = {}

    @staticmethod
    def _key(a, b):
        a, b = tuple(a), tuple(b)
        return (a, b) if a <= b else (b, a)

    def record(self, cell_a, cell_b) -> None:
        key = self._key(cell_a, cell_b)
        self._counts[key] = self._counts.get(key, 0) + 1
        if self._counts[key] >= NOOP_BACKOFF_THRESHOLD:
            self._backoff[key] = NOOP_BACKOFF_STEPS

    def tick(self) -> None:
        """Decrement remaining backoff steps (call once per next_move)."""
        for key in list(self._backoff):
            self._backoff[key] -= 1
            if self._backoff[key] <= 0:
                del self._backoff[key]
                self._counts.pop(key, None)

    def excluded(self) -> set:
        return set(self._backoff)


def _is_champion(item_id: str) -> bool:
    """True if the item id names a Champion (invader enemy), not a mergeable item."""
    return bool(item_id) and item_id.startswith(CHAMPION_PREFIXES)


def _champion_cells(board) -> list[Cell]:
    """All Champion cells on the board. Champions NEVER attack — the bot drags
    one of OUR creatures onto the champion cell to deal the creature's
    `damage` value. Used by the attack branch and the validator."""
    return [c for c in board.cells if c.item_id and _is_champion(c.item_id)]


def best_attack_pair(board, *, damage_values: dict[str, int] | None = None
                     ) -> tuple[Cell, Cell, int] | None:
    """Best (attacker, champion, damage) for the attack move, or None when
    no attack is possible right now.

    Preconditions enforced:
    - At least one champion is on the board (an enemy to attack).
    - At least one of OUR creatures with a known damage value exists
      (a champion attack needs a real damage stat — unknown-damage cells
      are never offered as attackers, to keep the validator honest).

    Ranking: HIGHEST damage first (the goal is to deplete the champion's HP
    as fast as possible). Tiebreak: top-left creature cell (deterministic)."""
    if damage_values is None:
        damage_values = {}
    champions = _champion_cells(board)
    if not champions:
        return None
    attackers: list[tuple[Cell, int]] = []
    for c in board.cells:
        if not c.item_id:
            continue
        if _is_champion(c.item_id):
            continue  # can't attack with a champion
        if any(c.item_id.startswith(p) for p in STATION_PREFIXES):
            continue  # stations can't attack either
        dv = damage_values.get(c.item_id)
        if dv is None or dv <= 0:
            continue  # need a real damage value
        attackers.append((c, dv))
    if not attackers:
        return None
    # Highest damage first; tiebreak by top-left cell
    attackers.sort(key=lambda t: (-t[1], t[0].row, t[0].col))
    attacker, dv = attackers[0]
    champion = champions[0]  # any champion works; the board rarely has 2
    return (attacker, champion, dv)


def _feed_key(cell, feed_values: dict[str, int]):
    """Sort key for feed candidates: lowest known feed value first. Unknown
    values sort last, so we don't sacrifice an item we know nothing about when
    a known-cheap one is available."""
    return (feed_values.get(cell.item_id, float("inf")), cell.item_id)


def _should_feed(board, target, empty_count: int, *, max_level_ids,
                 feed_values, craved_item, feed_objective_active=False,
                 craved_level=None, spawn_possible: bool = False,
                 merge_available: bool = False,
                 prefer_item: str | None = None) -> bool:
    """Should the (already-picked) feed target actually be fed?

    Shared by the heuristic's feed branch and the vision-drive `Best feed`
    hint so both always agree (they already share best_feed_cell). Triggers:
    congestion relief, craving bonus, an active feed objective (feat-driven),
    or a known-cheap single with at least one other creature to keep the merge
    pipeline alive.

    Max-level items are NOT a standalone trigger any more (Aug 23 generator
    protection): they passively generate mana, so feeding them is a last
    resort — they still feed via congestion, craving, an active feed feat,
    or the heuristic's starvation guard (no merge AND no spawn possible).

    When `spawn_possible` is true (an empty cell AND a grave that can spawn),
    the known-cheap feed-to-progress trigger is SUSPENDED: spawning builds a
    new cheap-chain starter for free instead of sacrificing one of only a few
    creatures on a sparse board (the observed bug: 10/15 empty, full mana bar,
    bot fed skeleton_lvl1 instead of spawning). The other triggers (congestion,
    craving, active feed objective) still fire — those are genuine reasons to
    feed even when a spawn is available.

    MERGE-PRIORITY GATE (Aug 26): when a confident merge EXISTS on the board,
    a GENERIC feed (neither craved, nor the feat/income prefer target, nor a
    max-level feed-only item) DECLINES — an income-boosted feed weight
    otherwise outranks merge every step and the merge pipeline starves (the
    observed bug: 6 ribcages piled up for a whole soak while the bot fed coins
    each step under `Build a Mana Pool`). Congestion does NOT bypass this gate:
    merging is itself congestion relief — two cells become one, freeing space
    WITHOUT destroying value (feeding frees space only by eating the item), so
    when pairs exist the merge goes first and feeds resume once pairs run out."""
    is_craved = bool(craved_item
                     and craved_matches(target.item_id, craved_item,
                                        craving_level=craved_level))
    is_prefer = bool(prefer_item
                     and item_name_matches(target.item_id, prefer_item))
    is_max = target.item_id in set(max_level_ids or ())
    congested = empty_count <= CONGESTION_THRESHOLD
    if merge_available and not (is_craved or is_prefer or is_max):
        return False  # merging IS congestion relief — let it go first
    other_feedable = sum(
        1 for c in board.cells
        if c.item_id and (c.row, c.col) != (target.row, target.col)
        and not any(c.item_id.startswith(p) for p in STATION_PREFIXES)
        and not _is_champion(c.item_id))
    known_cheap = (target.item_id in feed_values
                   and feed_values[target.item_id] <= FEED_PROGRESS_MAX_VALUE)
    return (congested                              # congestion relief
            or is_craved                              # craving bonus
            or feed_objective_active                  # active feed feat
            or (known_cheap and other_feedable >= 1 and not spawn_possible))


def _feedable(board):
    """Feedable cells: occupied, never stations or champions (never food)."""
    return [c for c in board.cells
            if c.item_id
            and not any(c.item_id.startswith(p) for p in STATION_PREFIXES)
            and not _is_champion(c.item_id)]


def _feedable_pool(feedable, *, feed_values, chain_map, max_level_ids,
                   craved_item, craved_level):
    """Split feed candidates into `(plain, material)` pools.

    MERGE-MATERIAL PROTECTION: an item whose next merge level is KNOWN
    (banked chain hop / `_lvlN` suffix) and that is not max-level is merge
    material — feeding it burns a mergeable pair of the cheapest kind while
    other, non-mergeable targets exist. Exact-or-possibly-craved cells are
    ALWAYS plain (feeding the craving is the point — the bonus + progress),
    and max-level ids are plain (their only action IS feeding). Material is
    sacrificed only when nothing plain is feedable.

    UNKNOWN-VALUE PROTECTION: an item with no recorded feed value, no known
    chain, and not max-level is in NEITHER pool — feeding it blind destroys
    possibly-precious specimens for unknowable gain (observed: a
    step-locally labeled eyeball fed while its sprite banked toward
    consensus). Unknown-only boards yield no feed candidate (callers fall
    back to merge/spawn); once banked, normal rules apply. Guarded on a
    POPULATED book (non-empty feed_values): a knowledge-less planner
    can't judge value, so everything stays feedable as before."""
    plain, material = [], []
    book = feed_values or {}
    for c in feedable:
        if (craved_item
                and craved_matches(c.item_id, craved_item,
                                   craving_level=craved_level)):
            plain.append(c)
        elif c.item_id in (max_level_ids or ()):
            plain.append(c)
        elif is_merge_material(c.item_id, chain_map=chain_map,
                               max_level_ids=max_level_ids):
            material.append(c)
        elif book and book.get(c.item_id) is None:
            continue  # unknown food value and no chain: don't eat blind
        else:
            plain.append(c)
    return plain, material


def best_feed_cell(board, *, feed_values=None, max_level_ids=None, craved_item=None,
                   remaining_satiety=None, prefer_item=None, craving_bonus=0,
                   satiety_capacity=None, tol_fraction=SATIETY_TOL_FRACTION,
                   craved_level=None, chain_map=None, prefer_max=False):
    """The best cell to feed right now (feed-to-progress), or None.

    Order of preference: the craved item (food bonus) > a near-done feat's
    target item (`prefer_item`, feeding exactly what the mission asks for) >
    the cheapest feedable creature by known feed value (unknown feed values
    sort last). Max-level / high-value creatures carry NO priority tier
    they passively generate mana, so feeding
    them is a last resort that only happens when nothing cheaper exists.
    Never stations or champions. This is the shared feed-ranking used by the
    heuristic fallback and the vision-drive `Best feed` hint, so both agree.

    The craving is level-specific: `craved_level` (from the cravings-menu read)
    makes `craved_matches` only ever select cells of that exact level, so a
    `skeleton_lvl1` craving can never mark `skeleton_lvl5` as the craved feed.

    MERGE-MATERIAL PROTECTION: any feed candidate whose next merge level is KNOWN
    (banked family chain hop, or a `_lvlN` suffix) and is not max-level counts as
    merge material — feeding it burns a mergeable pair of the cheapest kind while
    another non-material target exists. The feed pool prefers non-material/craved
    targets; material pairs are sacrificed only as a last resort. This protects
    mid-chain merges even without an active craving (the feed-to-progress case).

    CRAVED-CHAIN PROTECTION: while an exact-level craving is active, any cell
    that is a merge-chain PRECURSOR of the craved item (same family, level
    below the craving's level — e.g. skeleton_lvl1/lvl2 or the bone/ribcage
    bases for a skeleton_lvl3 craving) is filtered out of EVERY candidate list.
    They are the recipe to build the craved monster, not food; feeding them
    destroys the path to the craving (the observed bug: the bot fed ribcage and
    skeleton_lvl1 while a skeleton_lvl3 craving was active). Exact-craved cells
    (level == craving) still feed; cells ABOVE the craving level still feed
    (they can't merge down). Unknown-level cravings (bubble-only) skip the
    protection (is_craved_precursor returns False without a known level).

    `remaining_satiety` (Y - X, or None when unknown) makes the pick
    saturation-aware: only items whose feed value fits the remaining bar are
    eligible, nothing is fed when the bar is full (remaining <= 0), and against
    a near-full bar the SMALLEST fitting item is fed (top up with the cheapest
    sacrifice). `satiety_capacity` (Y) allows a small overflow tolerance
    (SATIETY_TOL_FRACTION of Y): a feed within the tolerance is not "wasted".

    CRAVED overflow gate: a craved item's effective feed is `Z + craving_bonus`
    (the craving grants a bonus on top of the food). When capacity is known and
    `Z + craving_bonus` exceeds `remaining + tolerance`, feeding it would waste
    food — the craved item is REFUSED (returns None -> the caller should prefer
    spawning to rebuild it; cravings persist, so it can be fed after the bar
    resets on level-up). Unknown-value craved items are still fed (their bonus
    alone is treated as within tolerance). None/unknown capacity keeps the
    previous always-feed behavior for unknown-value or tiny-value cravings.
    """
    feed_values = feed_values or {}
    max_level = set(max_level_ids or ())
    feedable = _feedable(board)
    if not feedable:
        return None
    # Craved-chain protection: never feed the recipe for an active exact-level
    # craving (same family at/below the craving level). Filtered ONCE here so
    # every candidate list below (craved / prefer / max-level / cheapest) is
    # protected consistently.
    if craved_item and craved_level is not None:
        protected = set()
        for c in feedable:
            if is_craved_precursor(c.item_id, craved_item, craved_level,
                                   chain_map):
                protected.add((c.row, c.col))
        if protected:
            feedable = [c for c in feedable
                        if (c.row, c.col) not in protected]
    if not feedable:
        return None
    # MERGE-MATERIAL PROTECTION: after the precursor filter, feed candidates
    # can still include mergeable mid-chain items (bone/ribcage/skeletonN whose
    # next hop is known). Feeding them burns a merge pair of the cheapest kind.
    # Prefer the craved-eligible / non-material / max-level targets; ONLY when
    # nothing non-material exists fall back to sacrificing a material pair.
    plain, material = _feedable_pool(
        feedable, feed_values=feed_values, chain_map=chain_map,
        max_level_ids=max_level_ids, craved_item=craved_item,
        craved_level=craved_level)
    feedable = plain or material
    if not feedable:
        return None
    # HARD GUARD: never feed a mergeable base (ribcage/bone) while a confident
    # merge pair for that same item exists on the board. This prevents the
    # observed "ribcages instead of merging them" — merging frees a cell and
    # advances the skeleton chain, feeding burns it. Craved items are exempt.
    if not craved_item:
        try:
            mergeable_ids = {mid for mid, _ in ranked_merge_groups(
                board, max_level_ids=max_level, chain_map=chain_map,
                craved_item=craved_item)}
        except Exception:
            mergeable_ids = set()
        # if ribcage/bone pair exists, don't feed that same base
        if mergeable_ids & {"ribcage", "bone"}:
            # keep non-bone/ribcage candidates if any; otherwise we would idle
            non_base = [c for c in feedable if c.item_id not in ("ribcage", "bone")]
            if non_base:
                # prefer feeding something else; ribcage/bone will merge
                feedable = non_base
            else:
                # only ribcage/bone on board — let merge handle it; don't feed
                # (return None forces caller to try merge/spawn before feed)
                # unless every feedable is that base (then falling through to
                # merge is correct, so signal "nothing to feed")
                if any(c.item_id in mergeable_ids for c in feedable):
                    return None
    tol = 0
    if satiety_capacity is not None and satiety_capacity > 0:
        tol = max(1, int(tol_fraction * satiety_capacity))
    if remaining_satiety is not None and remaining_satiety <= 0:
        return None  # bar full: any feed wastes everything
    if remaining_satiety is not None:
        def _craved_eff(c):
            # effective value of feeding this item AS the craving (includes the
            # bonus); unknown recorded value -> just the bonus.
            return (feed_values.get(c.item_id) or 0) + craving_bonus

        def _craved_within(c):
            return _craved_eff(c) <= remaining_satiety + tol

        def _value_within(c):
            v = feed_values.get(c.item_id)
            # unknown-value items are held back when capacity is known — they
            # may be bigger than the remaining room (+ tolerance).
            return v is not None and v <= remaining_satiety + tol
        if craved_item:
            craved = [c for c in feedable
                      if craved_matches(c.item_id, craved_item,
                                        craving_level=craved_level)
                      and _craved_within(c)]
            if craved:
                # the largest fitting craved item (most food, still within the
                # bar + tolerance); the wasteful ones were filtered out above.
                return max(craved, key=lambda c: (_craved_eff(c), c.row, c.col))
        if prefer_item:
            pre = [c for c in feedable
                   if item_name_matches(c.item_id, prefer_item)
                   and _value_within(c)]
            if pre:
                # prefer_max (income stacks): the LARGEST stack grants the most
                # currency — feeding a lvl1 icerune instead of the lvl3 stack
                # wastes income (the Ice economy's "feed only the max-level
                # stack" rule). Default stays smallest (cheapest top-up).
                if prefer_max:
                    return max(pre, key=lambda c: (feed_values.get(c.item_id, 0),
                                                   c.row, c.col))
                return min(pre, key=lambda c: (feed_values[c.item_id], c.row, c.col))
        eligible = [c for c in feedable if _value_within(c)]
        if eligible:
            # Smallest fitting value: top up a near-full bar with the cheapest
            # sacrifice, preserving high-value / generative monsters.
            return min(eligible, key=lambda c: (feed_values[c.item_id], c.row, c.col))
        return None  # nothing fits even with the tolerance — don't waste
    # Capacity unknown: craved creature, then a feat-target, then the CHEAPEST
    # feedable by known feed value (unknown values sort last). Max-level /
    # high-value creatures carry NO priority tier any more (Aug 23 generator
    # protection): they are mana generators, so `_feed_key` naturally feeds
    # them only when nothing cheaper exists.
    if craved_item:
        craved = [c for c in feedable
                  if craved_matches(c.item_id, craved_item,
                                    craving_level=craved_level)]
        if craved:
            return min(craved, key=lambda c: _feed_key(c, feed_values))
    if prefer_item:
        pre = [c for c in feedable if item_name_matches(c.item_id, prefer_item)]
        if pre:
            if prefer_max:
                return max(pre, key=lambda c: (feed_values.get(c.item_id, 0),
                                               c.row, c.col))
            return min(pre, key=lambda c: _feed_key(c, feed_values))
    return min(feedable, key=lambda c: _feed_key(c, feed_values))


@dataclass
class Move:
    kind: str  # "merge" | "spawn" | "feed" | "collect" | "attack" | "dismiss_popup" | "idle"
    cell_a: tuple[int, int] | None = None
    cell_b: tuple[int, int] | None = None
    target: tuple[int, int] | None = None  # attack: champion cell; feed: Devourer mouth
    taps: int = 0


class Planner(ABC):
    def __init__(self):
        # Merge-noop backoff registry. main.py calls `record_merge_noop` when a
        # merge verifies as a no-op; the registry excludes that pair from merge
        # candidates for a few steps so the heuristic/LLM stop re-proposing a
        # pair the game keeps ignoring.
        self._noop = _MergeNoopRegistry()

    @abstractmethod
    def next_move(self, board: BoardState, frame=None) -> Move:
        """Decide the next move. `frame` is optional (unused by heuristics)."""

    def invalidate_cravings(self) -> None:
        """Called by main.py after a Devourer level-up (a craving is very
        likely complete then). Base = no-op; planners that cache a craving
        menu read override this to force a fresh read next step."""
        return None

    def record_merge_noop(self, cell_a, cell_b) -> None:
        """Record a merge that the game ignored (verified no-op). The pair is
        excluded from merge candidates after NOOP_BACKOFF_THRESHOLD no-ops."""
        self._noop.record(cell_a, cell_b)


class HeuristicPlanner(Planner):
    def __init__(self, noop: _MergeNoopRegistry | None = None):
        super().__init__()
        if noop is not None:
            self._noop = noop  # share backoff state with a wrapping planner
        self.craved_item: str | None = None   # base name of current craving, if known
        self.craved_level: int | None = None  # exact level of the craving (None if only the bubble cue is known)
        self.max_level_ids: set[str] = set()  # ids that are top-level (feed-only, never merge)
        self.chain_map: dict[str, list[str]] | None = None  # family -> merge chain ids
        self.feed_values: dict[str, int] = {}  # template id -> numeric Food gained when fed
        # Creature damage values: how much HP a creature removes from a
        # Champion if dropped ON the champion instead of fed. Read from
        # popup bodies ("Takes Damage" column) and threaded in by
        # vision-drive (live) or main.py (static). Default empty dict
        # so dry-run heuristics without popups still propose attacks when
        # the board has a champion (the validator refuses unknown-damage
        # attackers, and the attack menu is gated on best_attack_pair).
        self.damage_values: dict[str, int] = {}
        # Devourer satiety: remaining capacity (Y - X) and total capacity (Y),
        # or None when the HUD satiety fraction can't be read (unbanked token).
        # When known, feed picks are capacity-limited (never waste food on an
        # oversized feed) and a craved feed that would overflow past the
        # tolerance is refused (spawn to rebuild it instead).
        self.satiety_remaining: int | None = None
        self.satiety_capacity: int | None = None
        # Learned craving food bonus B (the craving grants Z+B food). Scales
        # with the game, so it's measured live (satiety delta on a craved feed)
        # and refreshed per observance; 0 until first measured.
        self.craving_bonus_est: int = 0
        # Feat-driven priority state (set per step by vision-drive; None/False
        # in standalone/dry-run use => the base order merge > feed > spawn >
        # collect, i.e. the previous hardcoded behavior).
        self.feat_weights: dict[str, float] | None = None  # kind weights from feats + craving
        self.feed_objective_active: bool = False  # an active feed feat exists
        self.feed_prefer_item: str | None = None  # compact family a near-done feed feat targets
        self.feed_prefer_max: bool = False  # prefer target = LARGEST stack (income)

    def _feed_trigger(self, board, target, empty_count: int,
                      spawn_possible: bool = False) -> bool:
        """Delegate: the shared feed gate with this planner's configured state.
        `merge_available` feeds the merge-priority gate so a generic feed
        declines while a confident merge waits (see _should_feed)."""
        merge_available = bool(ranked_merge_groups(
            board, max_level_ids=self.max_level_ids,
            chain_map=self.chain_map, craved_item=self.craved_item,
            craved_level=getattr(self, "craved_level", None),
            craved_need=getattr(self, "craved_need", None),
            exclude_pairs=self._noop.excluded()))
        return _should_feed(board, target, empty_count,
                            max_level_ids=self.max_level_ids,
                            feed_values=self.feed_values,
                            craved_item=self.craved_item,
                            feed_objective_active=self.feed_objective_active,
                            craved_level=self.craved_level,
                            spawn_possible=spawn_possible,
                            merge_available=merge_available,
                            prefer_item=self.feed_prefer_item)

    def _spawn_candidates(self, board, frame=None) -> list[Move]:
        """All spawn-capable stations on the board, mana/space gated.

        candidates are sorted by level desc (highest-level station
        first) so the spawn picks the most productive cell. A lvl 3 grave
        produces bone (40%) / ribcage (30%) / zombie (30%) per the wiki,
        while a lvl 1 only produces bone (100%). The chest outranks the
        grave (chest is mana-free, finite uses; grave costs mana). Both
        categories use the same level-desc sort.

        Grave costs mana (SPAWN_MANA_MIN); chest is mana-free (one tap per use).
        Both need an empty board cell or the spawn silently no-ops.
        Used by _spawn_move's ranking and by _spawn_possible for the feed gate."""
        import re
        if not any(cell.item_id is None for cell in board.cells):
            return []  # no empty cell: any spawn would silently no-op
        def _level(item_id: str) -> int:
            m = re.search(r"_lvl(\d+)$", item_id or "")
            return int(m.group(1)) if m else 0
            # collect graves and chests separately, sort each by level
        # desc, then concatenate (chests first, both sorted). The output
        # is the ranking used by _spawn_move.
        grave_cands: list[Move] = []
        chest_cands: list[Move] = []
        cup_cands: list[Move] = []
        # Cupboard taps cost Slime and spawn eye components — only rank them
        # when something wants eyes (eyemonster/eyeball/eyeinjar craving).
        # Otherwise they clog the board for no objective. Slime level is
        # unknown here; the validator's spawn_no_slime gate handles empty.
        craved = (self.craved_item or "").lower()
        want_eye = any(k in craved for k in ("eyemonster", "eyeball", "eyeinjar"))
        for cell in board.cells:
            if not cell.item_id:
                continue
            is_grave = any(cell.item_id.startswith(p) for p in SPAWN_PREFIXES)
            is_chest = any(cell.item_id.startswith(p) for p in CHEST_PREFIXES)
            is_cup = any(cell.item_id.startswith(p) for p in SLIME_SPAWN_PREFIXES)
            # A chest whose uses ran out may persist as a spent sprite (or be
            # removed by the game entirely). Never spawn-tap a spent state —
            # the Aug 26 live probe: double-taps on a used-up chest did
            # nothing while the planner kept ranking it first.
            #
            # word-boundary check (the previous substring match filtered
            # out `icebox_unopened` because the literal "opened" is a substring
            # of "unopened" — false positive on the most common unspent chest).
            if is_chest and any(re.search(rf"\b{w}\b", cell.item_id)
                                for w in ("depleted", "empty",
                                          "spent", "opened")):
                continue
            if not (is_grave or is_chest or (is_cup and want_eye)):
                continue
            if is_grave:
                mana = read_mana_fraction(frame) if frame is not None else None
                if mana is not None and mana < SPAWN_MANA_MIN:
                    continue  # not enough mana to spawn grave
                grave_cands.append((_level(cell.item_id), cell.row, cell.col,
                                    Move(kind="spawn", cell_a=(cell.row, cell.col),
                                         taps=SPAWN_TAPS)))
            elif is_cup:
                # cupboard spawn: slime cost, one tap per use
                cup_cands.append((_level(cell.item_id), cell.row, cell.col,
                                  Move(kind="spawn", cell_a=(cell.row, cell.col),
                                       taps=CHEST_SPAWN_TAPS)))
            else:
                # chest spawn: mana-free, one tap per use
                chest_cands.append((_level(cell.item_id), cell.row, cell.col,
                                    Move(kind="spawn", cell_a=(cell.row, cell.col),
                                         taps=CHEST_SPAWN_TAPS)))
        # Sort each by level desc, then row/col for stable ordering.
        grave_cands.sort(key=lambda x: (-x[0], x[1], x[2]))
        chest_cands.sort(key=lambda x: (-x[0], x[1], x[2]))
        cup_cands.sort(key=lambda x: (-x[0], x[1], x[2]))
        # Chests outrank cupboards outrank graves (finite-use first, then
        # slime-cost, then mana-cost).
        return ([m for _, _, _, m in chest_cands]
                + [m for _, _, _, m in cup_cands]
                + [m for _, _, _, m in grave_cands])

    def _spawn_move(self, board, frame=None) -> Move | None:
        """Best spawn on the board (chest vs grave), board-space + mana gated.

        `_spawn_candidates` already sorts by level desc (chest
        first, then grave). The first candidate wins. When no chest
        exists, the highest-level grave wins. The "Open a Chest." feat
        and the Ice-income loop both need chest uses (5); the chest
        outranks the grave so the limited-use resource is consumed first.
        Shared by the spawn kind AND the wasteful-craved spawn priority."""
        cands = self._spawn_candidates(board, frame)
        if not cands:
            return None
        return cands[0]

    def _spawn_possible(self, board, frame=None) -> bool:
        """True when a spawn is actually possible right now: a grave exists
        that can spawn (mana permitting) AND at least one board cell is empty.
        The empty-cell requirement keeps spawn-priority hints off a full board
        (spawn would silently no-op). Shared by the feed gate (`_feed_trigger`
        suspends feed-to-progress when spawning is possible) and the
        vision-drive `Best feed` hint, so they always agree."""
        if not any(cell.item_id is None for cell in board.cells):
            return False
        return self._spawn_move(board, frame) is not None

    def _craved_feed_wasteful(self, board) -> bool:
        """True when EVERY matching craved cell on the board is capacity-
        wasteful: feeding any of them would overflow the bar beyond the
        tolerance (Z + craving bonus > R + 10%*Y).

        Signals the "only matching craved item is expensive" case: the right
        move is to spawn from the grave to rebuild the (cheap chain) craved
        item instead of sacrificing it (or another creature) to a near-full
        bar. Mirrors best_feed_cell's craved gate so the fallback and the
        `Best spawn` hint always agree. Unknown capacity or a craved cell that
        fits (unknown value -> bonus alone, or value within tolerance) returns
        False (best_feed_cell feeds it, and the spawn priority must not fire).
        """
        if (not self.craved_item or self.satiety_remaining is None
                or self.satiety_capacity is None):
            return False
        tol = max(1, int(SATIETY_TOL_FRACTION * self.satiety_capacity))
        fvs = self.feed_values
        craved_cells = [c for c in _feedable(board)
                        if craved_matches(c.item_id, self.craved_item,
                                          craving_level=self.craved_level)]
        if not craved_cells:
            return False

        def _eff(c):
            return (fvs.get(c.item_id) or 0) + self.craving_bonus_est

        return all(_eff(c) > self.satiety_remaining + tol for c in craved_cells)

    def next_move(self, board: BoardState, frame=None) -> Move:
        # Branch ORDER is driven by active feats (+ the craving, folded in as a
        # feed objective): the kind that advances the closest-to-done objective
        # is checked first. With no feats known (standalone/dry-run) this is
        # the base order merge -> feed -> spawn -> collect, the previous
        # hardcoded ladder. Within a kind the existing gates still apply.
        self._noop.tick()
        excluded = self._noop.excluded()
        for kind in kind_order(self.feat_weights):
            if kind == "attack":
                # Champion combat: drag our best-damage creature onto the
                # champion. Champions NEVER attack (the game only auto-attacks
                # if you refuse to deal with them), so the only way to make
                # progress is to spend a creature dealing its `damage` value.
                # This branch only fires when best_attack_pair() returns a
                # valid (attacker, champion, dmg) triple; otherwise the loop
                # falls through to merge/feed/spawn. With a champion on the
                # board the attack weight (3.0) outranks every other kind, so
                # this branch runs first.
                pair = best_attack_pair(board, damage_values=self.damage_values)
                if pair is not None:
                    attacker, champion, _dmg = pair
                    return Move(kind="attack",
                                cell_a=(attacker.row, attacker.col),
                                target=(champion.row, champion.col))
            elif kind == "merge":
                # Mergeable identical-item pairs, best-valued first. The margin
                # gate (best match beats the runner-up) keeps a mislabeled item
                # from triggering a merge the game refuses (retry-loop);
                # ranked_merge_groups picks the most valuable pair (max-level
                # result > higher level > not the craved item).
                ranked = ranked_merge_groups(board, max_level_ids=self.max_level_ids,
                                             chain_map=self.chain_map,
                                             craved_item=self.craved_item,
                                             craved_level=getattr(self, "craved_level", None),
                                             craved_need=getattr(self, "craved_need", None),
                                             exclude_pairs=excluded)
                if ranked:
                    item_id, cells = ranked[0]
                    a, b = cells[0], cells[1]
                    return Move(kind="merge",
                                cell_a=(a.row, a.col), cell_b=(b.row, b.col))
            elif kind == "feed":
                # The only matching craved item is expensive (feeding it would
                # overflow the bar): PREFER spawning to rebuild the (cheap
                # grave-chain) craved item over sacrificing any creature to
                # top up — the user's spawn-priority rule. Only when a spawn
                # is actually possible (else fall through to a cheap top-up).
                if self._craved_feed_wasteful(board):
                    sp = self._spawn_move(board, frame)
                    if sp is not None:
                        return sp
                # Feeding: merges exhausted (or a feed objective outranks
                # merging). Pick the best feed target — the craved creature
                # (food bonus) > a near-done feat's target > the CHEAPEST
                # feedable creature (max-level / high-value creatures are
                # mana generators and rank last via _feed_key).
                empty_count = sum(1 for cell in board.cells if cell.item_id is None)
                spawn_possible = self._spawn_possible(board, frame)
                target = best_feed_cell(board, feed_values=self.feed_values,
                                        max_level_ids=self.max_level_ids,
                                        craved_item=self.craved_item,
                                        craved_level=self.craved_level,
                                        remaining_satiety=self.satiety_remaining,
                                        prefer_item=self.feed_prefer_item,
                                        craving_bonus=self.craving_bonus_est,
                                        satiety_capacity=self.satiety_capacity,
                                        chain_map=self.chain_map,
                                        prefer_max=self.feed_prefer_max)
                if target is not None and self._feed_trigger(
                        board, target, empty_count, spawn_possible=spawn_possible):
                    return Move(kind="feed", cell_a=(target.row, target.col))
                # Starvation guard: if we can neither merge nor spawn right
                # now, feeding (any cheap feedable, accepting slight overflow)
                # is the only real unlockable — let a near-full bar level up
                # instead of softlocking the bot. Reached either when
                # best_feed_cell found nothing that fits (target is None) OR
                # when the trigger refused the picked target (Aug 23 generator
                # protection: a lone high-value generator with no merge and no
                # spawn still feeds rather than idling forever).
                # Chain-protection applies: never feed a craving precursor
                # while a non-precursor feedable exists (fall back to any
                # creature only when everything is a precursor — a softlock is
                # worse than losing one recipe step).
                if (not ranked_merge_groups(board, max_level_ids=self.max_level_ids,
                                            chain_map=self.chain_map,
                                            craved_item=self.craved_item,
                                            craved_level=getattr(self, "craved_level", None),
                                            craved_need=getattr(self, "craved_need", None),
                                            exclude_pairs=excluded)
                        and self._spawn_move(board, frame) is None):
                    candidates = _feedable(board)
                    if self.craved_item and self.craved_level is not None:
                        non = [c for c in candidates
                               if not is_craved_precursor(
                                   c.item_id, self.craved_item,
                                   self.craved_level, self.chain_map)]
                        if non:
                            candidates = non
                    cheapest = min(candidates, key=lambda c: _feed_key(c, self.feed_values),
                                   default=None)
                    if cheapest is not None:
                        return Move(kind="feed", cell_a=(cheapest.row, cheapest.col))
            elif kind == "spawn":
                # If there are empty cells, spawn a new item using a spawn
                # station. A grave spawn costs mana (~10) and silently no-ops
                # when the mana bar is exhausted, so gate on the HUD bar
                # fraction (below SPAWN_MANA_MIN fall through to collect so the
                # NecroMerger can regenerate mana).
                sp = self._spawn_move(board, frame)
                if sp is not None:
                    return sp
            elif kind == "collect":
                # Filler action when resources are low. The NecroMerger is
                # geometrically fixed in the TOP-RIGHT cell (0, cols-1) — the
                # old hardcoded (0,2) was the 5x3 board only and tapped a board
                # item instead once the grid grew to 5x4. Collecting when the
                # mana bar is already full wastes a step (tap yields nothing
                # useful), so it falls through to idle.
                mana = read_mana_fraction(frame) if frame is not None else None
                if mana is not None and mana >= COLLECT_MANA_MAX:
                    continue
                col = session_geometry().cols - 1
                for cell in board.cells:
                    if (cell.row, cell.col) == (0, col):
                        return Move(kind="collect", cell_a=(cell.row, cell.col),
                                    taps=COLLECT_TAPS)

        return Move(kind="idle")  # No valid moves found, idle for now


def next_move(board: BoardState, frame=None) -> Move:
    """Back-compat wrapper: delegates to the heuristic planner."""
    return HeuristicPlanner().next_move(board, frame)
