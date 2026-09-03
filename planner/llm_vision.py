"""VisionLLMPlanner: screenshot-vision observation for the LLM planner.

Distinct from LLMPlanner (text-grid): the model sees the FULL emulator
screenshot as an image_url content part, plus the numbered VALID-move menu.
Inherits _respond/_validate/fallback from LLMPlanner — only the
user-message construction differs. (Plan A: menu still code-validated.)
"""

import base64
import json
import time
from pathlib import Path

import cv2

from planner.llm import LLMPlanner, _render_board
from planner.llm_client import LLMClient
from vision.grid import FALLBACK_GEOMETRY, GridGeometry

MAX_DIM = None          # full resolution (1280x2856) passed as-is; llama.cpp's
                        # vision tower enforces its own image_max_pixels budget.
                        # Set to a long-side cap (e.g. 1536) to downscale first.
JPEG_QUALITY = 85
SCREEN = (1280, 2856)   # for the geometry prompt (emulator res is pinned)

def _vision_prompt(geom: GridGeometry | None = None) -> str:
    geom = geom or FALLBACK_GEOMETRY
    x0, y0 = geom.origin_x, geom.origin_y
    x1, y1 = x0 + geom.cols * geom.cell_px, y0 + geom.rows * geom.cell_px
    return f"""You are the brain of a bot playing NecroMerger on a {geom.rows}x{geom.cols} board.
The screenshot is {SCREEN[0]}x{SCREEN[1]} pixels, full emulator frame.
Board geometry: the board occupies pixels x={x0}-{x1}, y={y0}-{y1};
{geom.rows} rows x {geom.cols} columns, each cell {geom.cell_px}px square. Top-left cell is (0,0), top-right (0,{geom.cols-1}),
bottom-left ({geom.rows-1},0), bottom-right ({geom.rows-1},{geom.cols-1}). The Devourer's mouth is above the board at ~({geom.mouth_x},{geom.mouth_y}).
Rules:
- Two identical items merge into one of the next level (e.g. bone_lvl1 + bone_lvl1 -> bone_lvl2).
- A grave spawns new items when tapped.
- Tapping the necromancer collects mana.
- Feed a creature to the Devourer to gain food.
- NEVER feed stations (grave, necromancer, manapool, manapot) to the Devourer.
- Prefer merges, especially of higher-level items; keep the board from filling up.

You will be given a numbered list of VALID moves (coordinates use the grid above).
Look at the screenshot, decide the best move, then reply with ONLY a JSON object
selecting exactly one: {{"choice": N}}  (N is a number from the list)."""


def _encode_frame(frame, max_dim: int | None = None) -> str:
    """Full screenshot -> JPEG base64 data URL.

    By default the frame is passed at its native resolution (the vision tower
    in llama-server enforces its own pixel budget, so downscaling here only
    throws information away). Set `max_dim` to a long-side cap to downscale
    first (mainly for probe A/B tests).
    """
    if max_dim is None:
        max_dim = MAX_DIM
    h, w = frame.shape[:2]
    if max_dim:
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class VisionLLMPlanner(LLMPlanner):
    def __init__(self, *, with_text: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.with_text = with_text

    def _build_messages(self, board, moves, frame=None) -> list[dict]:
        if frame is None:
            return super()._build_messages(board, moves, frame)  # degrade to text
        parts = [{"type": "image_url", "image_url": {"url": _encode_frame(frame)}}]
        text = f"Valid moves:\n{self._render_menu(moves)}\n\nPick the best single move. Reply with only JSON."
        if self.with_text:
            text = f"Board:\n{_render_board(board)}\n\n" + text
        parts.append({"type": "text", "text": text})
        return [{"role": "system", "content": _vision_prompt(board.geometry)},
                {"role": "user", "content": parts}]

    def _log_chat(self, messages: list[dict], reply: str) -> None:
        sanitized = []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                content = [p if p.get("type") != "image_url"
                           else {"type": "image_url", "image_url": {"url": "[image]"}}
                           for p in content]
            sanitized.append({**m, "content": content})
        with self.chat_log_path.open("a") as f:
            f.write(json.dumps({"t": time.time(), "messages": sanitized, "reply": reply}) + "\n")