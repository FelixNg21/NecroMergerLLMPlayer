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
- **Noop-triggered identification (Sep 4)** — a confirmed merge no-op (both cells still holding after the retry swipe) now popup-identifies both cells via `main._identify_noop_cells`: banks the sprite templates under the true ids + the popup recipes (feed/damage/max-level), logs `noop_identified` (was/now/feed_only) for the summary to consolidate. Previously a noop only fed the backoff registry (temporary avoidance, zero learning, retried after expiry). Bounded to 2 taps, vision-drive only, failures never stall.
- **Identify trust gates (Sep 5)** — the first live noop-identify poisoned the bank two ways: an OCR fragment minted garbage `be_lvl2` (templates + glossary entries, since removed), and a weak digit read (lvl3→lvl2) broadened the WRONG level's bank (`coins_lvl2__2/__3`, since removed). Now: `identify._resolve_id` takes `level_trusted` (confident digit or LLM agreement) and only broadens known ids on trust; `identify.py:538` never mints bank entries for unknown ids; `main._identify_noop_cells` refuses to bank unknown bases (`noop_identify_rejected`) since noop cells had confident labels by definition. Genuine new items still bank via the proactive UNID path. Follow-up: a trusted digit now also OVERRIDES a conflicting bank match (a `digit=1.0 level=3` read was being discarded because the poisoned lvl2 bank "matched" — the merge kept no-oping); the canonical lvl3 id assembles and broadens correctly instead.
- **Blank signature cleanup (Sep 5)** — the `be` repeat-failure was a poisoned `assets/signatures/be.png`: a byte-blank title crop (std 0.0) banked Sep 4 that matched every empty title bar at ~1.0 and won all future reads deterministically (bypassing the LLM path entirely). Removed `be.png` + near-blank fragment `ug27.png` (std 14.6, unreferenced); `_save_signature` now refuses crops below `SIGNATURE_MIN_STD=15.0` (mirrors the template bank's texture gate).
- **Skeleton bank decontamination (Sep 5)** — the "only skeletons, all weak" symptom was real poisoning: 14 lvl6 sprites banked under `skeleton_lvl5` (`__48`, `__61-73`, all ~0.08 vs the lvl5 seed while matching the lvl6 bank at 0.9+) via the accumulate dip path, plus 1 unidentifiable capture (`__74`, deleted). Re-filed the 14 under `skeleton_lvl6`; new `main._matches_seed` seed-guard blocks future dip-banking unless the crop resembles the id's `__0` calibration seed (logs `accumulate_seed_rejected`). Pollution vector was self-reinforcing (each misfiled sprite attracted more).
- **Digit "2" banked (Sep 5)** — `assets/digits/` had only 1/3/4, so a true lvl2 glyph could never template-confirm (best cross-match ~0.673, below threshold) and every lvl2 read leaned on LLM/OCR text. Grounded a live `coins_lvl2` popup three ways (classifier label + banner OCR "Lvl 2" + noop elimination) and banked `2__0.png` via `bank_digit` (conflict check passed). Verified symmetric separation: true-2 reads 1.000 vs 0.673, true-3 reads 1.000 vs 0.673. Region left unchanged deliberately (templates + live crops share the box, so any clipping cancels out; jitter is covered by multi-variant banking over time).
- **Manual bank purge (Sep 6, user-led, stays deleted)** — the user removed ~64 bank files judged image-vs-name mismatches. Audited per-file against HEAD: the `skeleton_lvl6` series + several families scored own-wins (a review UI + applier were built: `scripts/build_review_ui.py`, `scripts/apply_review.py`, output in `/tmp/review/`), but per user decision nothing is restored. Consequence: `manapotion_lvl1`, `necromancer`, `icerunes` (and digit `5`) have zero templates and are unrecognizable until re-banked live via the identify/accumulate paths (which is the designed self-healing route). Do NOT bulk-restore.
- **Coins lvl4 grounded (Sep 5)** — the prompt/code knew "coin max → feed for gold" but `coins_lvl4` had no stats, no chain, no max flag, so every max-keyed gate (tags, whitelist, income hint, merge-max) silently skipped it. Wiki-grounded: `item_stats` (feed 30, max true) + `coins` merge chain (lvl1→lvl4). Matches the banked valuablechest rates (50/30/15/5).
- **Phantom `_drive` removed (Sep 5)** — `_run_strategy_planner` contained a byte-identical copy of `next_move`'s tail (pre-existing duplication, confirmed via AST on HEAD): every strategy-planner step ran a second full `_drive` with live side effects and a discarded result. Deleted; `strategy_slot`/`strategy_planner`/`buy_compel` green.
- **Retry context bound (Sep 5)** — the `_drive` loop now builds messages from `history[-6:]`; full history still feeds fixation counters and cross-step memory. Prevents context-window overflow on long rejection loops.
- **Per-cell legality tags (Sep 5)** — board lines now carry `[FEEDABLE]`/`[feed exceeds R]` (same satiety-tolerance gate as the validator), `[SPAWNABLE]`/`[spawn blocked: …]`, `[NEVER MERGE]` on max-level. The 4B model can't multi-hop feed+mana+empty across distant lines; verdicts are rendered, not computed.
- **HARD RULES + few-shot + checklist (Sep 5)** — shared `HARD_RULES_TEXT` (5 most-violated rules) sits before the action schema in both prompts, with a permanent legal/illegal JSON pair; board state ends with a 3-step decision checklist.
- **Answer-round prompt split (Sep 5)** — `_answer_prompt` (~2.6k chars: geometry + hard rules + schema, ~90% cut) swapped in by `_ask_once`; full ~25k prompt stays on tool rounds. Tags/whitelist/hints/checklist arrive via Tool results.
- **Queue auto-collect (Sep 5)** — `collect_queue` could never fire voluntarily (zero native tool calls on this stack), so `next_move` now places queued rewards code-driven when `has_reward` + room, patches the cell, logs `queue_collected(via=auto)`. Queue identity via `queued_reward_id` (template-only chest match) renders `queue(ice chest)` etc. in the Bottom bar line.
- **Rune-gap math (Sep 5)** — currency line appends shortfall + feed-equivalent (`short 18 poison (≈2 max poisonrune_lvl3 feeds)`) for strategy AND craving-need families.
- **Ad observer, detect-only (Sep 5)** — `vision/ads.py::detect_ad_offer` (locked-chest-gated full-frame OCR for watch-ad keywords, injectable OCR for tests) + `next_move` hook logging `ad_offer_seen`. Never taps. Watching flow still unbuilt pending a captured offer.
- **Forced-call unification (Sep 5)** — all required buy rounds (strategy/craving compels, pre-loop follow-through, post-loop confirm) funnel through `_compel_buy` (user-only minimal prompt + `BUY_TOOL_SHORT` + 1024 budget + synth fallback + confirm pinning). Probes showed full-context required calls die while minimal ones land; arg-bearing calls need ~300 reasoning tokens (256 truncates).
- **Compel backoff (Sep 5)** — `_compel_misfires` counts consecutive no-effect compels per family; 3 strikes stands down (prevents burning a round every step when the model can't emit).
- **Optional-round batching + buy text path (Sep 5)** — multi-tool rounds carry a batching nudge (scoped copy); explicit `buy_station` text JSON synthesizes within `MAX_BUYS` instead of being discarded.
- **Targeted identify + collect skip (Sep 5)** — proactive identify prioritizes UNIDs sharing a family hint (likely merge pair) over lone cells; opportunistic feat-collect skips when the last attempt found nothing and the cache hasn't refreshed (stops reopening the panel for unclaimable buttons).
- **Strategy unification (Sep 5)** — retired the periodic `StrategyPlanner` (own LLM call every 20 steps overlapping the compelled `set_strategy` path): the compelled required round is now the sole forcing mechanism, and `META_GOALS` thresholds evaluate code-side into a `Suggested direction:` board line when no strategy is fresh. None-safe by construction (the old darkness check fired unconditionally on missing data). Deleted `planner/strategy_planner.py` + its verify script; direction covered by new `verify_strategy_slot` part H.
- **Summary frugality (Sep 5)** — transcript capped at 400 events with omission note (a 50-retry step blew the 240s budget); summarized events pruned from RAM (disk file untouched); `session.jsonl`/`llm_chats.jsonl` rotate to `.bak` past 8MB at startup.
- **Answer hygiene (Sep 5)** — `_parse_action` strips tool-call markup before JSON extraction (the model leaks `</tool_call> {"name":…}` fragments into answers); strict-then-loose, old behavior as fallback.
- **Panel economy (Sep 5)** — `next_move` pre-computes `_want_buy_this_step` (pending/strategy/craving, pure cache reads); when a buy will open the Station panel, scheduled FEATS reads/collects defer (`feats_panel_deferred`) so the step costs one panel cycle, not two.
- **Summary split (Sep 6)** — periodic LLM summarization retired as a cadence: `maintain_memory` runs the deterministic folds every window (rejection/reward learnings, demote, prune inferred, watermark + RAM prune) with no model call; the full pattern-mining summary fires every 10th window + session end. The per-window LLM call's value had collapsed (immediate deterministic banking covers items; its last run gutted learnings.md) while its cost/risk dominated (240s timeout, mass-delete).
- **Consensus banking (Sep 6)** — repeated independent popup reads agreeing on one unbanked id (3x, per-cell) now bank sprite + recipe even when OCR mangles the title (observed: 12 consecutive "eyeball"-LLM vs "fuehball"-OCR refusals = infinite re-tap loop, labels evaporating every step so merges stayed validator-refused). Changing popup story resets votes; capped at 30 cells.
- **Unknown-value feed protection (Sep 6)** — the eyeball was being fed for ~1 food while banking toward consensus: step-local labels bypass every material/income gate (no stats, no chain). Now `feed_unknown_value` refuses statless+chainless items in the validator, heuristic pool, and board tags (`[UNKNOWN VALUE — do not feed]`), with the merge-material desperation shape (full board of only-unknowns still feeds rather than stalls). Guarded on a populated book so knowledge-less test planners behave as before.
- **Board evolution memory (Sep 6)** — `vision/grid.py::diff_boards` diffs consecutive classifies with move-aware explained-filtering (merge/feed/spawn/attack cells filtered by kind; same-cell relabels stay visible; cross-cell pairs become `moved`); `main.py` logs unexplained `board_diff` events into the transcript for the summary to learn from (champion spawns, reward arrivals). Covered by `scripts/verify_board_diff.py` (9 checks).
- **Observe heartbeat (Sep 6)** — `get_board_state` takes no arguments and its text is deterministic, so only every 10th step makes the live call (`OBSERVE_HEARTBEAT_STEPS`); other steps inject the identical render with zero LLM cost (~20-50s saved/step).
- **Locked-station handling (Sep 6)** — the craving buy looped 28 identical full-sheet scans for a Supply Cupboard that isn't purchasable: the sheet holds grave/manapool/lectern + a grayed 4th slot. Now `read_cards` OCRs locked cards' requirement text ("Tier 5 Feats Requires"), failed scans stash it (`locked_requirements`), the craving compel stands down while the feats tier hasn't moved (re-scans on tier-up, not timer), and the board line says LOCKED + what unlocks it instead of sending the bot rune-gathering for an unbuyable station. Wiki-grounded: cupboard = Tier-4-feats unlock, 20 poison, eyeball/eye-in-jar spawns.
- **Supply Cupboard purchase (Sep 6)** — the sheet's 3rd card IS the cupboard (wooden cabinet, 20 poison) but the LLM misnamed it "lectern"/"grave", so `_find_card` never matched and the dialog guard refused. Fixed: banked `supplycupboard_lvl1` templates + `_station_templates` entry; dialog station OCR-override (`station_ocr_override`); spaced-name canonicalization; `_rewind_to_start` for scroll persistence. Completed live (popup-confirmed "Supply Cupboard Lvl 1" at (1,3)). Note: `bought` detection only checks ice-spend/board-count — poison buys can report false; verify via popup.
- **Cupboard spawn path (Sep 6)** — the eye economy was unreachable: hints/validator/whitelist/fallback knew grave+chest only, so an eyemonster craving + cupboard board never produced a spawn. Now `SLIME_SPAWN_PREFIXES` flows everywhere (validator accepts with slime-known-zero refusal `spawn_no_slime`; hint shows cupboard with per-level outputs + Slime cost only when craving/strategy wants eyes; whitelist/tags gate the same way; fallback ranks chest > cupboard > grave with cupboard gated on eye craving). Covered by `scripts/verify_cupboard_spawn.py` (16 checks).
- **Fridge joins slime spawners (Sep 6)** — wiki-grounded (Slime cost, Tier-13 unlock, eye-component producer): added `fridge` to `SLIME_SPAWN_PREFIXES`, which every consumer already reads, so validator/hint/whitelist/tags/fallback cover it with zero further edits. Foul Chicken deliberately excluded (passive timer-layer, no tap mechanic); altar/dark-stores/lectern/portal stay out until their tap costs are grounded (no invention).
- **Knowledge write hygiene (Sep 5)** — genuine-new-item sprite/recipe banking now requires popup-OCR agreement (`identify_unbanked` logged when refused); `prune_inferred_entries` implemented for real (timestamp-aged, chains/spawns untouched) and wired into summarize; per-id template cap (`MAX_TEMPLATES_PER_ID=40`, seed-protected FIFO with max-index+1 naming so deletions can't collide) bounds bank growth.
- **Summary mass-delete guard (Sep 5)** — a summary window listed 9 learnings for removal + 40 glossary blocks while confirming the same texts, gutting `learnings.md` (484→99 lines; restored from git — JSON store survived via the `protected` flag). Now `_sanitize_prunes` drops removals contradicting the window's own confirmed/learnings and caps prunes (3 learnings, 5 items) per window.
- **Accumulate log crash (Sep 5)** — the seed-guard's `log.log` fired inside `capture_unknowns`, which never received `log` (`NameError` killed a live run at step 3). `log` is now an optional parameter, passed at the call site.
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

## Work state (Sep 6, 2026, eye-monster mislabel fix)

### Eye Monster Lvl 1 mislabeled as eyeball — root cause + fix (`vision/identify.py`, `item_knowledge/item_stats.json`)
- **Symptom (live):** 30+ `merge_noop` on (2,3)+(3,3) both labeled `eyeball`; `noop_identified` "confirmed" eyeball (sig 1.0) while the popup actually showed **Eye Monster Lvl 1** (digit-1 @1.0, OCR "fuemonster", live re-read "Eye Monster Lol1"). The craving (`eyemonster` bubble) never completed: the only monster candidate was mislabeled, and `get_cravings` was never called in 310 chats (optional tool, model never chose it) so level/count stayed unknown.
- **Root cause (two layers):** (1) `_identify_one`'s known-item branch trusts the title signature unconditionally — the banked `eyeball` signature matched the monster title at 1.0, skipping OCR/LLM entirely; (2) `_resolve_id`'s sprite-match returned `eyeball__N` (~0.94) with no conflict (regex only checks `_lvlN` ids), defeating any base-level remap.
- **Fix:** new `EYE_COMPONENTS = {"eyeball": "eyemonster", "eyeinajar": "eyemonster"}` (wiki Eye Monster page: components level-less, monster Lvl 1-6). Trusted level (digit ≥ threshold or LLM agreement) on an unlevelled base remaps in BOTH `_resolve_id` (conflict fall-through + pre-assembly remap) and `_identify_one` (info name). Scoped to eye family only (`bone`+trusted lvl still → `bone_lvl1`); untrusted levels stay conservative. Verified offline: eyeball+trusted-1 → `eyemonster_lvl1`, no-level → `eyeball`, untrusted → `eyeball`, eyeinajar+2 → `eyemonster_lvl2`, coins paths unchanged.
- **Self-heal (no offline banking needed):** `eyemonster` is already a known family (lvl2 templates), so the next live noop-identify resolves `eyemonster_lvl1` and the noop path banks the sprite via `_bank_unid_sprite` (trust gate passes on known family) — labels diverge, loop ends in ~1-2 steps. Restart bot to load.
- **Stat decontamination:** `eyeball` feed 15 was monster-popup poisoning (same run/mislabel; true component value wiki +1, consistent with popup-banked `eyeinajar` feed 2) → corrected to 1 (source wiki); added `eyemonster_lvl1` feed 15, `eyemonster_lvl2` feed 35 (wiki stats table).
- **Suites:** unbanked_popup 17, merge_rank 111/111, noop_backoff 17 — green.
- **Follow-up (done same day): cross-signal dispute vote.** The known-item branch read the LLM body but discarded its name, so one lying signature skipped every other signal. Now: `_sig_disputed` fires only on two-against-one (LLM body name AND OCR name both differ from the signature hit — OCR-only "tonbie"/"brave" mangling and LLM-only misreads keep the conservative label); on dispute, `_confirm_overrule` re-taps once and requires the second frame's LLM name to agree before adopting (bounded: +1 tap +1 LLM call, agreement path still costs 2 taps total). The body's level also feeds the digit fallback even when the signature stands. Overrules are traced via `info["overruled"]` (`sig->name`). Verified: committed `verify_unbanked_popup` part L (9 checks: predicate + `_resolve_id` on an empty temp bank, no fixture dependency) — suite now 26/26; full `_identify_one` plumbing with fake device/LLM on a real grave-popup frame (overrule adopts + 3 taps, agreement keeps + 2 taps). No bank writes from tests.
- **Follow-ups:** eye merge chain still ungrounded past "eyeball merges toward eye-in-jar" (no chain entry added — no invention); craving-menu call uptake (model never picks the optional `get_cravings`) unresolved.

## Work state (Sep 7, 2026, craving over-merge fix)

### Merged two eyemonster_lvl1 into lvl2 for a lvl1 craving — guard + knowledge fix
- **Symptom (live short run):** after the eye fix self-healed (`noop_identified` (3,3) eyeball → `eyemonster_lvl1`, sprite banked), the bot merged (1,0)+(2,3) eyeballs → eyeinajar, (2,3)+(3,1) eyeinajars → lvl1, then (3,1)+(3,3) lvl1+lvl1 → **eyemonster_lvl2**, and fed the lvl2 (`craving_bonus` feed 35, learned 0 — zero craving credit, craving still open).
- **Root causes (two):** (1) craving level/count were UNKNOWN — `Cravings (bubble): eyemonster` with no level, `get_cravings` never taken in 22 chats, so no guard could fire; `score_merge` only deprioritizes craved merges and proposes them when sole. (2) No guard existed against merging the last pair AT the craved level.
- **Guard:** `destroys_craving()` (`planner/merge.py`) + `merge_craving_material:<id>` (`planner/llm.py::_validate`) — fire only on full knowledge (menu-read level + remaining need>0) when survivors (`n - 2`) fall below need; precursor merges, surplus merges, and unknown-level/need pass. Wired through `ranked_merge_groups(craved_level, craved_need)` (whitelist, best-merge hint, availability), both `_validate` call sites, and heuristic `agent.py` sites via `getattr` (old behavior when unset).
- **Knowledge:** bubble-only craving line now names the call (`call get_cravings`); new `_compel_cravings_read` forces one minimal required menu read when the bubble shows a craving the cache doesn't know (never read or switched) — same pattern/backoff as buy compels, no synth fallback (text can't replace a menu cycle), fires at most once per craving. Verified live: `compelled_cravings_read` → `craving eyemonster lvl1 1/2 +200` on the next line.
- **Follow-up (done same day): label persistence for indistinguishable sprites.** The compel worked but the craving still wasn't fed: (3,3) re-classified `eyeball` @1.00 with no lvl1 anywhere on the board. Bank cross-match proved the sprites unseparable (every lvl1 crop matches the eyeball bank at 0.994–1.000 and 6/40 eyeball crops match the lvl1 bank ≥0.9 — ball+tail vs ball+wisps). A stateless ambiguity wipe was rejected (margins overlap 0.17–0.56 both ways; re-wipe would tap-loop forever since banking can't fix identical sprites). Instead: planner-owned `_label_memory` (`note_identified` on every popup resolve, `apply_label_memory` after classify in `main.py`) re-applies the popup id over a template label ONLY inside `LABEL_MEMORY_PAIR={eyeball,eyemonster_lvl1}` (either direction; template-UNID included), keeping template score/margin. Drops on: TTL 10 steps, cell touched by our move, empty cell, or template label outside the pair. `noop_identified` now logs `overruled`. Verified offline (9 behavior checks) + full suite batch green.
- **Suites:** merge_rank 121/121 (new part N, 10 checks), unbanked_popup 26, noop_backoff 17, combat, strategy_slot 52, queue_box 22, cupboard_spawn 17, cross_step 7/7 — green. Live dry-run not possible (no emulator; `Device()` init hangs without one — pre-existing).
- **To confirm live:** restart bot; expect `label_memory_applied (3,3) eyeball->eyemonster_lvl1`, then a craved lvl1 feed for 2/2 +200 (need is 1).

## Work state (Sep 7, 2026, generalized resource economy)

### Producer teaching: RESOURCE_ECONOMY + Makes rates (wiki-grounded)
- **Ask:** instead of hardening "resource low → build generator" as a guard, teach the causal chain so the model learns it: "can't spawn X (uses Y) → build/keep Z (makes Y)". And generalize producers beyond cravings.
- **Grounding (wiki Mana/Slime/Grave/Supply_Cupboard pages):** Grave/Lectern taps cost Mana (lvl1 −250); Cupboard/Fridge taps cost Slime (lvl1 −250). Mana is generated over time by Skeletons (1,2,3,5,7,10,14), Eye Monsters (2,4,6,9,12,15), Banshees, Mana Golems, Serv-O + legendaries; Slime by Zombies (1,2,4,6,9,12), Spiders, Ghouls, Slime Golems, Serv-O + legendaries; both also from feeding Potions; caps raised by Mana Pools / Slime Vats.
- **Generalization:** new `RESOURCE_ECONOMY` (planner/vision_drive.py) next to `CRAVING_PRODUCERS` — demand→station stays in CRAVING_PRODUCERS, resource→{used_by, made_by, cap_by, also} lives here; `RESOURCE_REJECTION` maps `spawn_no_slime`/`spawn_low_mana` to their resource; `_resource_teaching(res, short)` builds both prose forms from the map (single source, no drift).
- **Teaching wired three ways:** (1) in-step `_correction` hints name producers; (2) folded `_REJECTION_RULE_TEXTS` persist the full chain to learnings.md via the existing deterministic loop; (3) `_slime_line_text` (on empty, mirrors `slime<=0` validator gate) and `_mana_line_text` (below `SPAWN_MANA_MIN`) append the short producer clause.
- **Makes rates banked:** `item_stats.json` gains `makes: {mana|slime: rate}` for skeleton_lvl1-7, eyemonster_lvl1-6, zombie_lvl1-6 (wiki tables); `write_item_stat` preserves `makes` across popup writes (new param); schema doc updated. Enables future quantitative generator ranking.
- **Suites:** new `scripts/verify_resource_economy.py` 20/20 (map integrity, teaching prose, correction+fold coverage, rates, preservation); regression batch green (merge_rank 121, unbanked 26, noop 17, strategy 52, queue 22, cross_step 7/7).
- **Darkness (same day):** wiki Darkness + Altar pages ground the third economy — Altar/Portal taps cost Darkness (−250 lvl1), generated over time by Mummies (1,2,3,4,6), Bats (2,4,6,8), Imps (3,5,7,10), Darkness Golems, Serv-O + legendaries (+ feeding Shades/Potions), cap raised by Darkness Stores. Added to `RESOURCE_ECONOMY` + Makes rates banked (mummy/bat/imp levels); verify script now 28/28. Deliberately knowledge-only: the save is Tier-5 (Altar unlocks at Tier 7), and no live plumbing reads darkness yet (no HUD reader, no validator gate — altar/portal spawns still hit `spawn_not_grave` — no hint trigger). Live path (darkness reader + threshold + DARKNESS_SPAWN_PREFIXES gate mirroring cupboard) waits until the save reaches it.

## Work state (Sep 7, 2026, pinned board dims)

### Config-first geometry: LLM dims prior replaced by item_knowledge/board_dims.json
- **Ask:** the LLM dims read is per-session overhead with run-to-run variance (a 4x4-vs-5x4 flip moves `origin_y` by a whole cell height, shifting every banked template crop); grid size only changes on board-expansion feat rewards, so pin it in config.
- **Change (`vision/geometry.py`, `main.py`):** new `BOARD_DIMS_PATH` (`item_knowledge/board_dims.json`) + `read_board_dims`/`write_board_dims` (validated, never-raising); `llm_grid_geometry` gains `prior_dims` and accepts `llm=None` — prior order is now pinned config (no LLM round at all) → LLM read → full enumeration. Startup uses the pin when present; a first successful detect self-seeds the file; a stale pin self-corrects (blob-plausibility + template objective pick the true dims, then re-seed). Blob pixel derivation, candidate verification, and anchor refinement are untouched (deterministic CV, no LLM). Seeded `{"rows": 5, "cols": 4}` (live-verified). Prune-safe (`prune_inferred_entries` only touches item_stats inferred entries).
- **Mid-run growth:** out of scope (rare) — restart re-derives automatically; no feat hook (expansion-feat names not reliably identifiable).
- **Suites:** helpers verified offline (pin load, missing/garbage/invalid → None, round-trip, candidate ordering); regression batch green (resource 28, merge_rank 121, unbanked 26, noop 17, strategy 52, queue 22, cross_step 7/7).
- **To confirm live:** restart bot; expect `pinned dims 5x4 (no LLM round)` and no geometry LLM call at startup.

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
- `scripts/verify_*.py` — 13 verification scripts (one per major feature area).
- `assets/templates/` — sprite bank (per-item + per-level PNGs).
- `assets/calib/feats/` + `assets/calib/satiety_frames/` + `assets/calib/title_banner.png` — verification fixtures.
- `item_knowledge/` — per-item feed/damage/max-level stats + merge chains (JSON; injected into the prompt via `learnings.md`/glossary).
- `learnings.md` — committed past learnings (injected into the prompt).
- `session.jsonl` / `llm_chats.jsonl` / `stub_tmp.png` / `assets/debug/` / `assets/review/` / `assets/calib/menu/` — runtime-generated; gitignored (not committed).
- `AGENTS.md` (this file) — current state.
