"""Menu geometry + read recipes (Phase 1/2 of menu exploration).

Only the item-info popup is fully verified on the live emulator
(NecroMerger_PS, 1280x2856). The Devourer screen, station popups and
shop/bottom tabs are NOT reachable yet — the bottom strip is a uniform
decorative band and blind probes risk dangerous dialogs (see AGENTS.md
"Menu exploration Phase 1: safety findings").

Safety invariants for any code that opens/closes menus (MenuExplorer):
- `(120,2350)` is a LIVE bottom-UI button (opens a coin build dialog), NOT a
  universal dismiss. Only safe when an item popup's dim overlay is confirmed
  present (which `Identifier.identify` guarantees: it taps the popup open).
- BACK key dismisses an open dialog/panel safely; never press BACK on the
  bare board (it exits the game).
- Never tap build/confirm buttons.
"""

import json

from planner.llm import _extract_json
from planner.llm_client import LLMError
from planner.llm_vision import _encode_frame

# Full item-info popup panel ROI (x, y, w, h) in native pixels. Verified:
# frame[286:620, 235:1080] == assets/calib/menu/item_panel_crop.png (diff 0.0).
ITEM_PANEL = (235, 286, 845, 334)

ITEM_POPUP_QUESTION = """This is the body of a NecroMerger item info popup (title line, level, and the
description text). Read it and reply with ONLY JSON:
{"name": "<lowercase item name, e.g. skeleton>", "level": <int or null>,
 "description": "<the description text exactly as shown, or empty string>",
 "merge_info": "<the part saying what merging two of these produces, or empty string>",
 "feed_value": <int or null>,
 "damage": <int or null>}
Example: for a lvl 3 Skeleton whose text says "Merge to summon a higher lvl Skeleton."
reply {"name": "skeleton", "level": 3, "description": "Merge to summon a higher lvl Skeleton.",
"merge_info": "two Skeleton -> Skeleton of the next level", "feed_value": 60, "damage": 25}.
The body may show a row like "Takes Damage | Feed" with a second row of numbers
(e.g. "25 | 60") — `feed_value` is the Food value in the SECOND (Feed) column (how much
food the Devourer gains when this item is fed), and `damage` is the number in the FIRST
(Takes Damage) column (the damage dealt if this item is dropped on a Champion like the
Peasant). Both are null when no stats row is shown.
If the text is unreadable, return {"name": "", "level": null, "description": "",
"merge_info": "", "feed_value": null, "damage": null}. No prose outside the JSON."""


def crop_item_panel(frame):
    """Return the item-info popup panel crop (RGB), verified ROI."""
    x, y, w, h = ITEM_PANEL
    return frame[y:y + h, x:x + w]


def read_item_popup(frame, llm, question: str | None = None) -> dict:
    """LLM-read an item-info popup from a full-screen screenshot.

    `frame` must have the popup open; no tap happens here. Returns
    {"name", "level", "description", "merge_info", "feed_value", "damage"}
    (all optional/empty on failure or unreadable text). Grounded in the game's
    own UI, so the caller can bank the merge-chain + feed-value + damage facts
    into the item glossary.

    `question` overrides the default `ITEM_POPUP_QUESTION`. Caller passes a
    category-specific prompt (Champion / Currency / Station) so the LLM
    focuses on category-relevant fields — see `popup_question_for` in
    `vision/identify.py`.
    """
    messages = [
        {"role": "system", "content": "You read NecroMerger item info popups from screenshot crops."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _encode_frame(crop_item_panel(frame))}},
            {"type": "text", "text": question or ITEM_POPUP_QUESTION},
        ]},
        # Assistant prefill: skips the reasoning model's thinking block (which
        # otherwise exhausts max_tokens and returns empty content). The reply
        # is grammatically constrained to continue the JSON object.
        {"role": "assistant", "content": '{"name":'},
    ]
    try:
        content, _ , _full = llm.chat_message(messages, max_tokens=256, json_mode=False)
        data = _extract_json(content)
        return data if isinstance(data, dict) else {}
    except (LLMError, ValueError, json.JSONDecodeError):
        return {}
