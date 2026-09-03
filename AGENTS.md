# AGENTS.md — Merge Bot (NecroMerger autonomous agent)

This file is the **current state**: goal, environment, run commands, architecture, key design decisions, and pending verification. (The prior full chronological log was removed from the repo on Sep 3, 2026 during a GitHub-prep cleanup; a fresh one will be built up from here.)

## Goal

- An agent that autonomously plays NecroMerger on an Android emulator via screenshot + adb input.
- Extended goal (Claude-Plays-Pokemon style): an LLM planner that reads the board state and decides the next move.

## Environment

- **Host:** macOS (Apple Silicon M1 Pro); Android Studio AVD `NecroMerger_PS` (Pixel 5, Android 13, arm64, Play Store image). Emulator res: **1280x2856**.
- **adb:** auto-located by `env/adb.py` (no PATH dependency).
- **Python:** 3.11; venv at `.venv`; deps in `requirements.txt`.

## Run / commands

- **Start emulator:** `emulator -avd NecroMerger_PS &` then `adb wait-for-device`.
- **Live loop:** `python main.py` (live loop) | `--dry-run` (print moves, no gestures) | `--planner heuristic|llm|vision-drive` | `--no-llm-reasoning` (skip LLM thinking; faster).
- **Live soak (self-correcting learnings):** `python main.py --planner vision-drive --summarize-every 5 [--steps N]` — updates `learnings.md` / `item_glossary.md` every 5 steps (prune/confirm/drop).
- **Stop:** Ctrl-C, then check `session.jsonl` for the last move.

## Architecture

### Top-level
- `main.py` — entry point. Wires the device, the planner, the metrics logger; runs the live loop.
- `planner/` — the LLM-driven decision layer.
  - `vision_drive.py` — **the active planner**: tool-calling vision model observes the board, calls `get_board_state` / `identify_item` / `set_strategy` / `tap_button` / `buy_station` / `collect_queue` / `get_cravings` / `lookup_wiki`, and emits a single JSON action.
  - `agent.py`, `llm.py`, `llm_client.py`, `llm_vision.py` — older text-only / LLM-vision paths (kept for regression).
  - `merge.py`, `priorities.py` — the merge-ranking algorithm (used by both heuristic and the board-state hint).
  - `learnings.py`, `glossary.py` — persistent memory files (Markdown) the prompt injects as `Past learnings` / `Items you have learned to recognize`.
  - `wiki.py` — `lookup_wiki` tool backend.
- `vision/` — board detection + UI readers.
  - `classifier.py` — multi-scale template matching against the sprite bank (`assets/templates/`).
  - `pipeline.py`, `grid.py` — board geometry detection + per-cell classification.
  - `hud.py`, `satiety.py`, `cravings.py`, `champion.py`, `chest.py`, `queue_box.py`, `bottombar.py`, `levelup.py`, `menu.py` — UI-element readers (one per widget).
  - `panels.py` — Station shop / dialogs / `_wait_panel_open` (panel-open detection).
  - `geometry.py` — one-time screen calibration.
  - `tools/` — vision-tool backends.
- `scripts/` — verification scripts (each is `python scripts/verify_X.py`).

### System prompt assembly (vision-drive)

`_drive_prompt(geom)` at `planner/vision_drive.py:94-318` is a single ~225-line f-string with the board geometry interpolated in. The full system prompt is:

```
_drive_prompt(geom)
+ (wiki sentence if self.wiki_tool is True)
+ _learnings_context()    # "\n\nPast learnings (apply them if relevant):\n" + read_learnings()
+ _glossary_context()     # "\n\nItems you have learned to recognize (apply if relevant):\n" + read_glossary()
```

- Static rules: merge, level, max-level, spawning (grave + chest), feeding, satiety overflow, craving overflow, feed-to-progress, champion combat, station buying (two-phase), station economy (5 currencies), bottom-bar dock, `set_strategy`, `collect_queue` placement gate.
- Dynamic parts: board geometry (rows/cols/cell_px/origin/mouth), `learnings.md`, `item_glossary.md`, wiki-on/off flag.
- Active strategy is rendered into the `get_board_state` tool RESULT (per-step), not into the system prompt.

### Action schema (vision-drive reply)

```json
{"action": "merge",  "a": [r,c], "b": [r,c]}
{"action": "spawn",  "cell": [r,c]}
{"action": "feed",   "cell": [r,c]}
{"action": "attack", "cell": [r,c], "target": [r,c]}    // drag creature onto Champion
{"action": "collect"}                                     // tap NecroMerger for mana
{"action": "idle"}
```

## Key design decisions

- **Margin-based merge gate** (replaces absolute score gate) — same family, different level still rejected even with high absolute score. The wipe gate (margin >= 0.10) and validator (margin >= 0.20) prevent the `(2,3)+(3,1)` skeleton_lvl4 noop class.
- **Two-phase `buy_station`** — `confirm=false` reads the dialog and stashes a pending buy; `confirm=true` completes it. Mismatched family → refused. Stale pending_buy expires after PENDING_BUY_TTL steps.
- **`set_strategy` strategy-target guard** — `buy_station` refuses any family that doesn't match the active strategy's target.
- **Strategy noun classification (Aug 28)** — `_classify_feat_noun(feat)` parses feat names into three shapes: `station` (Build-a-X, Own-a-Lvl-N+-X where X is a station from the wiki list), `creature` (Own-a-Lvl-N+-Y where Y is a minion/champion), and `other` (combat, action-count, collect-Nx, level-N feats). Station feats get the buy_station hint with cached rune cost; creature feats get the merge/collect hint (target only kept when its family matches the noun); other feats get a "follow the action in the feat text" hint. A `target_item` that doesn't match the noun category is dropped — fixes the Aug 28 "Own a lvl 3+ Grave. (merge/collect skeleton_lvl6 on the board)" wrong-target wrong-kind bug. `_STATION_NAME_ALIASES` covers all 17 wiki stations; `_extract_own_noun` regex uses `[\w+-]+` for the "lvl N+" token (the `+` broke `\w+` and silently dropped every Own-a-station feat). `_strategy_family` is set for any station-shape feat (Build OR Own), not just Build-a-X.
- **`collect_queue` placement gate** — refused when the board is congested; refuses to drop rewards into a full board.
- **Station panel verification** — three guards now (Aug 27):
  - `_wait_panel_open` requires the panel body to be rendered (anti mid-animation).
  - `read_currency` all-zero → re-poll for a fully-rendered frame.
  - `_check_currency_bar` requires >= 2 of 5 currency slots to have a saturated rune icon (catches wrong panels).
- **5 rune currencies** (Aug 26 rename from green/red/purple): `ice, poison, blood, moon, death`. Only these 5 appear in `currency` / `cost_*` dicts.
- **Panel-safe BACK** — never BACK on a bare board (would exit the game). The `close_panel` re-BACKs only when the dock is confirmed still covered.
- **`top_resumed_activity` check** — only raise `BoardLostError` when the game is no longer in the foreground; a slow panel just logs a warning.
- **`wiki_tool` default ON** — disabled by `--no-wiki-tool`. Used for fact-checking item behavior / merge chains.
- **Per-step learnings refresh** — `vision-drive` updates `learnings.md` / `item_glossary.md` every 5 steps; pruning happens on `summarize_every`.
- **Geometry-aware everywhere** — board cells are computed from detected origin + cell_px, never hard-coded. The Necromerger cell, board cell coordinates, all downstream code respects the current geometry.
- **Self-correcting learning memory** — candidates must be `wiki_confirmed` or `confirmed` to commit; failing candidates are pruned; drop items with high refute rate.
- **Cost-cache** — `StationShop.cost_cache` stores last-known costs per family so the planner's strategic reasoning has known input.
- **Unbanked-popup-read backfill (Aug 27, Option 3)** — `_popup_next` falls back to UNID occupied cells (item_id is None) when no banked item needs backfill. The popup-reader resolves the name+level, banks the sprite + recipe in one pass. Catches brand-new items the template bank doesn't have yet (e.g. `skeleton_lvl6` before its template was captured, Champions before `peasant`/`knight` templates were banked). Refuses to invent ids not already in the bank (no hallucinated popups). One popup read per step (`MAX_POPUP_READS = 1`); backfilling the bank is preferred over identifying new items.
- **No `__alt` ids (Aug 27)** — `_resolve_id` in `vision/identify.py` no longer mints `skeleton_lvl4__alt1`-style ids when the LLM popup reports a sprite that doesn't match the canonical bank. Instead, the new sprite is banked under the canonical id (broadening the bank over time). The `__alt` family lacked damage/feed stats, so the bot silently refused them as attackers (`best_attack_pair` requires a known damage value). 12 `_alt*` template files and 7 `_alt*` glossary blocks were removed.
- **Feat reward collection fixed (Aug 27)** — Three layered bugs prevented `_observe_feats` from ever logging `feat_reward_collected`:
  1. The structural filter in `_collect_centers` (`w=150-280, h=35-80`) was calibrated for the OLD rectangular reward-button shape; the live layout renders SQUARE buttons (~233x233 px). The legacy template path matched 3 of the 5 buttons (false positives at 40-px dedup tolerance), but the structural filter excluded every live button. Widened to `w=100-320, h=30-280` to accept both shapes.
  2. `_claimable_centers` refused to claim when `len(centers) != len(feats)` (safe-mapping guard). The live panel shows 3 green buttons + 1 already-collected row (checkmark, no button) → 3 centers vs 4 feats → guard fires → `[]`. Now falls back to badge-only detection when the counts mismatch.
  3. The LLM read's `done` flag is unreliable for distinguishing claimable from preview buttons (both are green with "Reward" labels). The authoritative signal is the **red exclamation badge** in the upper-right corner of claimable buttons. Added `_has_claim_badge` (saturated-red pixel count in a tight top-right ROI) and `_center_has_claim_badge` (crops a 233-px window around the center, calls the pixel check). Filters out preview buttons (e.g. red hourglass on Own Grave) by constraining the ROI to the top-right corner and tightening S/V thresholds.

## Status (what works as of Aug 27, 2026)

- 13/13 test suites green: 47 station_shop, 33 dialog_icons, 13 chains_digits, 111 merge_rank, 21 noop_backoff, 7 satiety_reader, 12 tier_reward (NEW: 4 badge-detection checks), 31 strategy_slot, 9 gesture_retry, 20 queue_box, 21 combat, 22 level_discrimination, 14 unbanked_popup.
- Offline dry-runs stable: heuristic + vision-drive both return canonical `merge (2,2)+(3,0)` on the calib board.
- Vision-drive live loop runs end-to-end on the live save: spawn / merge / feed / attack / collect / idle actions emit, the board state refreshes, and the learning cycle updates `learnings.md` + `item_glossary.md`.
- Two-phase station buy enforces the affordability + dialog check; pending_buy auto-expires.
- Champion combat works on The Peasant; damage values are banked for known creatures.
- Same-family cross-fire mislabels are gated (wipe + validator).
- Station panel mid-animation captures are rejected; currency reads are reliable on the open panel.

## Pending verification (live-soak asks, grouped)

The "Next:" asks from each Aug 26/27 entry, deduplicated and grouped. Each item is a runtime observation, not new feature work — the next live soak should confirm these.

### Currency reader + station buy (Aug 26, Aug 27)
- Live soak with 53 ice / 14 poison on the current save — confirm `station_buy` events log the correct balance (was reading `ice: 0` before the Aug 27 fix). `currency_retry: True` may fire on transient mid-animation catches.
- Next manapool buy should complete end-to-end: `buy_station("manapool", confirm=true)` → `-5 poison, -10 ice` → station on the board.
- `cost_icon_override: true` events in `session.jsonl` — anytime the LLM misread a rune icon, the guard overrode it. Watch the rate.

### Cross-fire / noop gates (Aug 26)
- Live soak: confirm the `(2,3)/(3,1)` class of noops is impossible. The LLM may still see UNID cells in some positions and popup-identify them.

### Champion templates (Aug 27)
- Banked `peasant` (6 multi-scale templates) and `skeleton_lvl6` (6 templates) live from the current save. Watch the next soak: the Peasant on the board should now be recognized at high confidence (>= 0.55), and the attack branch should fire on it without falling through to merge/feed.
- `skeleton_lvl6` damage backfilled to 225 (matches skeleton_lvl5; the user confirmed from the in-game popup).

### Strategy flow (Aug 26)
- Restart soak, expect: `set_strategy` commits "Build a Mana Pool." → `expected_family=manapool`; `buy_station("manapool", confirm=false)` → (then) `confirm=true` → station placed. A longer soak to confirm the strategy line is stable across steps and currency auto-gather via the rune-stack feed works.

### Combat (Aug 26, Aug 27)
- Wait for a Peasant spawn on the live save, watch the bot fight it. Long-term: bank more damage values (zombie_lvl1, etc.) so attack targets the highest-damage creature.
- With peasant/skeleton_lvl6 templates now banked (Aug 27), the attack branch should fire on the Peasant without falling through to merge/feed. Next live soak: confirm `attack <creature> -> (peasant)` is the chosen move when the Peasant is on the board.
- With `__alt` ids removed (Aug 27), the `best_attack_pair` check should now find damage values for every cell the classifier labels. If a cell is still rejected as an attacker, it's a real missing-stat issue (not an alt-id one).

### Chest / Queue (Aug 27)
- Two separate concepts: (1) **chests on the board** are spawn stations (tapped with a `spawn` action — `_spawn_move` ranks chest > grave); (2) **the Queue dock button** holds earned rewards (level/feat chests, rune piles) — `collect_queue` taps the dock to drop the reward on the board. Don't confuse the two.
- Live observed: a chest appeared on the live save after the Peasant was killed (the user's transcription didn't have one). The bot did spawn on it (`cell_a: [4, 1]` in 0-indexed = the chest cell). The chest info popup is a docked panel; the chest cell on the board is the spawn target.

### Feat reward collection (Aug 27)
- Three layered bugs prevented `_observe_feats` from ever collecting any reward:
  1. Structural filter (w=150-280, h=35-80) excluded every live button (live = ~233x233).
  2. `_claimable_centers` count-mismatch guard returned [] when centers (3) != feats (4) — the already-collected row has no green button.
  3. LLM read's `done` flag couldn't distinguish claimable from preview (both have green "Reward" buttons).
- All three fixed. Next live soak should see `feat_reward_collected` events fire whenever a feat completes and the badge is on the button. Build a Mana Pool was the first to test (1/4 → 2/4 after collection).

### Ribcage merge pipeline (Aug 26)
- Expect the strategy line "gather runes then buy" to steer toward icerune merges. The previous "Why does it never merge the ribcages?" incident was a two-layer fix; the live run should now execute the ribcage → skeleton → undead chain naturally.

### Code-audit followup (Aug 26)
- The "too many values to unpack (expected 2)" bug was found by static reasoning; search for other stale shape assumptions in `priorities.py` and `dialog read` paths. (Not blocking; a refactor for a future change.)

## Work state (Sep 2, 2026)

### Merge-material + strategy protection (`planner/llm.py`)
- The `feed` validator now protects merge-chain material from being fed. `feed_merge_material:<item>` fires when the fed item is mid-chain material AND a plain (non-material) feedable exists — **or** when the board has any FREE cell (my `has_free_cell` addition: a sparse "not congested" board must never burn a merge component; empty cells aren't feedable, so the plain-feedable check alone was insufficient — this is what let the 17:41:07 run feed a strategy-critical zombie on a board with 9 empty cells).
- `feed_strategy_material:<item>` fires when the material is on the **active strategy's** critical path (`protect_family`, threaded from `vision_drive._strategy_family`) and any other feedable exists, even on a full board.
- Desperation allowance (feed material rather than starve) now requires a genuinely **full board** with no other feedable.
- `_validate` gained `chain_map` and `protect_family` kwargs threaded through both `_drive` call sites (~1846, ~1873).

### Zombie/skeleton merge chains corrected (`item_knowledge/merge_chains.json`, `planner/glossary.py`)
- `skeleton: bone -> ribcage -> skeleton_lvl1` and `zombie: rottenflesh -> severedhand -> zombie_lvl1` (both terminate at L1; numeric levels self-chain via `next_level_id`). Chains name *origins*; the `_lvlN` suffix derives ascension, so higher minions are intentionally NOT enumerated.

### wiki merge_info + phrasal deflection (`planner/wiki.py`)
- `_leading_item_noun` deflects `<item> merge chain` queries to the item page; `lookup` now returns `merge_info` (merge-relevant lines). Added "rotten flesh"/"severed hand" to fact-check keywords.

### Loop-nudge anti-fixation (`planner/vision_drive.py`)
- Root cause of the recurring `retry_exhausted` (3rd occurrence, 17:41:07): the model fixated on one action kind (`feed` × 47) and never pivoted to the valid `spawn` despite the sparse board + explicit `Best spawn: grave_lvl3 (4,3)` hint. The generic "do NOT propose these again" correction didn't break the loop.
- New `_loop_nudge(self, board, rejected)` (instance method, ~5247): when ≥5 rejections are dominated by a single action kind and a valid `Best spawn/merge/attack` hint exists for a DIFFERENT kind, appends a targeted "you keep proposing `feed`... the board suggests: `<hint>`" nudge to the correction. Wired at the rejection site (~1897). `_correction` gained a `nudge` param.
- Also strengthened the attack action-schema wording: a Champion is NEVER the attacker (`attack_is_champion` already enforced in `_validate`, but the prompt now states it explicitly to stop the model wasting an attempt on `attack (peasant -> ...)`).
- All suites green after changes (merge_rank 111, combat PASS incl. B9-B13, chains_digits 18, strategy_slot 51, etc.).

### `_loop_nudge` collect / chest-spawn branch (`planner/vision_drive.py`) + chest-prefix fix (`planner/constants.py`, Sep 2)
- **`_loop_nudge` gains a mana-aware collect/chest branch.** The 18:53 `retry_exhausted` (chat 327) showed the real stall: the board had 6 empty cells, `Best spawn: grave_lvl3 (4,3)`, Mana ~17%, but the only *valid* fallback was `collect` (its `spawn (4,3)` was rejected `spawn_low_mana` and the model then cycled `feed` ×49, never emitting `collect`). Root cause: Mana hovers at `SPAWN_MANA_MIN=0.15`, so `spawn_low_mana` flaps and `_best_spawn_line` (which reads `self._frame` live, not `self._mana_fraction`) can still emit a grave-spawn hint the validator re-rejects.
- New logic in `_loop_nudge`: compute mana once (from `self._mana_fraction` if set, else `read_mana_fraction(self._frame)`); if `dominant != "collect"`, a grave is on the board, **no chest** is on the board, and mana `< SPAWN_MANA_MIN`, return a `collect` nudge ("tap the NecroMerger, refill Mana, THEN spawn from the grave next step") — evaluated BEFORE the spawn/merge/attack candidates so it can't be preempted by a spawn hint that will be re-rejected.
- If a **chest** IS on the board at low mana, don't nudge collect — chest spawns are mana-free, so `_best_spawn_line` correctly offers the chest and the normal spawn hint fires instead.
- **New `_graves_on_board(self, board)` (vision_drive.py ~5313)** — `any(cell.item_id.startswith(p) for p in SPAWN_PREFIXES)`; used by the collect-branch guard.
- **`CHEST_PREFIXES` fixed (`planner/constants.py:37`).** Was `("icebox",)` only — that silently excluded `lockedchest` and `valuablechest` (both banked in `assets/templates/`, both in `item_knowledge/index.json`). Because every consumer uses `item_id.startswith(p)`, a `spawn` on `lockedchest`/`valuablechest` was **rejected `spawn_not_grave`** (llm.py:501-502) — the model literally couldn't spawn from them, and `_best_spawn_line` mis-ranked them. Now `("icebox", "lockedchest", "valuablechest")`. Verified: `spawn` on `lockedchest`/`valuablechest`/`icebox_unopened` all validate `None` (accepted).
- `_loop_nudge` tests: low-mana grave-only → collect nudge; adequate-mana grave → spawn hint; low-mana grave+`lockedchest` → chest spawn hint (mana-free). All correct.
- Pre-existing (NOT from these changes): `verify_noop_backoff` check A fails — fixture cell (4,1) now classifies `skeleton_lvl6` (templates banked Aug 27) but test still asserts `skeleton_lvl5`. Test-expectation drift only; the exit=1 is this single stale assertion (other 19 checks PASS). Board classification is untouched by today's edits.

### Queue-dock x 384→638: queued mana_potion misread as "queue is empty" (`vision/bottombar.py`, `vision/queue_box.py`, `scripts/verify_queue_box.py`, Sep 2)
- **User report:** `collect_queue` triggers a few times but fails "even though a mana_potion is available"; the potion is in the **middle of the bottom bar**.
- **Root cause (verified on the live save / `screenshots/Screenshot_1788408249.png`):** the queue dock button is the **middle of the 5-slot bar, x≈638** (bright reward item centered there; the real empty-skull is at x=384 = the **station's** x). But `ICON_CROPS["queue"] = (384, 2750, 62, 51)` pointed `has_reward`/`collect_queue`/`read_dock_reward`/`bar_visible` at x=384 — where the frame reads a clean empty skull (`empty_match` **1.0**) → `has_reward` False → "queue is empty," even though the mana_potion was sitting in the real queue button at x≈638 (where `empty_match` is only ~0.27 → would have been True). Evidence: at the corrected x≈620-700 `has_reward` returns **REWARD**; at the old x=384 it returns **EMPTY**.
- **Fix #1 (`vision/bottombar.py:ICON_CROPS["queue"]`):** x 384 → **638** (the live middle-button position). All static consumers now read the real queue slot.
- **Fix #2 (`vision/bottombar.py:_sanitize_positions`, new):** the runtime `_detect_positions` could still latch `queue` onto x=384 because it matches the empty-skull bank (and a reward-holding queue doesn't match that bank). Post-detection, if the detected `queue` x lands within 12px of `station`, re-derive it as the **midpoint of the station→spellbook gap** (~638). Verified: collision 384/380→638; already-correct/640 left alone.
- **Fix #3 (`scripts/verify_queue_box.py`):** fixtures hardcoded the old x=384 (`lair_frame`, `FakeBottomBar.button_center`, tap assertions) — updated to canonical `QUEUE_ICON` / `QUEUE_TAP`. Added regression checks **A6** (queue dock x is ~638 not 384), **A7/A8** (sanitizer interpolation). Suite now **22/22 pass**.
- **Verified end-to-end:** `has_reward(Screenshot_1788408249.png)` → **True** (was False). Live `detected centers` → `feats 161, station 381, queue 638, spellbook 896, shop 1152`; `tap_xy(queue)` → **(638, 2746)**.
- **Full suite green** (queue_box 22, merge_rank 111, strategy_slot 51, combat, chains_digits 18, etc.). `verify_noop_backoff` exit=1 remains the pre-existing skeleton_lvl6 drift.

## TODO (next feature work)

- **Handle watching ads.** NecroMerger offers ad-rewards (bonus mana/dark offering/chest refills, typically triggered by tapping an ad-voucher button next to a station or an ad offer). Currently the bot has no path to detect or click these, so it ignores the free-mana / chest-refill economy. Exploration needed:
  - Detect the ad-voucher button (a small blue "watch ad" / free-chest icon near the NecroMerger or station; `vision/` has no reader for it yet).
  - Interact with the ad dialog ("watch"/"x ad" interstitial), wait for the ad to finish, then claim the reward.
  - Gate it (e.g. once per N steps, or only when mana is low / when a free chest is offered) so it doesn't spam ads and stall the loop.
  - Not in the current action schema (`merge`/`spawn`/`feed`/`attack`/`collect`/`idle`) — would need a new action kind or a `buy_station`-style `watch_ad` tool.

## File map

- `main.py` — entry point.
- `planner/vision_drive.py` — active planner; ~3000 lines, the heart of the system.
- `planner/{agent,llm,llm_client,llm_vision}.py` — older LLM paths.
- `planner/{merge,priorities}.py` — merge ranking.
- `planner/{learnings,glossary,wiki}.py` — persistent memory + wiki tool.
- `vision/{classifier,pipeline,grid,geometry}.py` — board detection.
- `vision/{hud,satiety,cravings,champion,chest,queue_box,bottombar,levelup,menu,panels}.py` — UI readers.
- `scripts/verify_*.py` — 12 verification scripts (one per major feature area).
- `assets/templates/` — sprite bank (per-item + per-level PNGs).
- `assets/calib/feats/` + `assets/calib/satiety_frames/` + `assets/calib/title_banner.png` — verification fixtures.
- `item_knowledge/` — per-item feed/damage/max-level stats + merge chains (JSON; injected into the prompt via `learnings.md`/glossary).
- `learnings.md` — committed past learnings (injected into the prompt).
- `session.jsonl` / `llm_chats.jsonl` / `stub_tmp.png` / `assets/debug/` / `assets/review/` / `assets/calib/menu/` — runtime-generated; gitignored (not committed).
- `AGENTS.md` (this file) — current state.
