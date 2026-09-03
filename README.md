# Merge Bot — NecroMerger autonomous agent

A Python framework + agent that plays **NecroMerger** on an Android emulator by
screenshot + adb input. A vision pipeline detects the board and UI widgets, and
a tool-calling LLM planner reads the board state and decides the next move.

## Run

```bash
# Start the emulator, then
source .venv/bin/activate
python main.py                 # live loop
python main.py --dry-run       # print moves, no gestures
python main.py --planner heuristic|llm|vision-drive
python main.py --planner vision-drive --no-llm-reasoning   # fast, no LLM thinking
```

`vision-drive` is the active planner: a tool-calling vision model observes the
board and emits one JSON action per step (`merge` / `spawn` / `feed` / `attack` /
`collect` / `idle`). A self-correcting memory (`learnings.md`,
`item_knowledge/`) is refreshed every few steps and injected into the prompt.

You need an AVD running NecroMerger (Pixel image, arm64) and adb on your PATH
(see `AGENTS.md` for the emulator setup).

## Layout

- `planner/vision_drive.py` — the active LLM planner (the heart of the system)
- `planner/{agent,llm,llm_client,llm_vision,merge,priorities,learnings,glossary,wiki}.py` — decision layer + persistent memory + wiki tool
- `vision/` — board detection + one UI-widget reader per file (hud, satiety, cravings, champion, chest, queue_box, bottombar, levelup, menu, panels) + `tools/` backends
- `controller/actions.py` — Move -> gesture; `Layout` with cell geometry
- `metrics/logger.py`, `env/adb.py`, `main.py` — device, logging, run loop
- `scripts/verify_*.py` — per-feature verification suites
- `assets/templates/` — sprite bank (per-item + per-level cropped PNGs)
- `learnings.md` / `item_knowledge/` — the agent's persistent memory + item stats

> Note: a few `scripts/verify_*.py` suites load live screenshot fixtures
> (`screenshots/*.png`) that are currently absent pending re-capture; the rest
> of the suite passes. See `AGENTS.md`.

## Verification

```bash
for f in scripts/verify_*.py; do .venv/bin/python "$f"; done
```
