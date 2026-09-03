"""probe_vision: verify the mmproj vision pipe before running the agent.

Sends one screenshot to the llama-server as an image_url content part (same
encoding VisionLLMPlanner uses) and prints the model's free-text reply. Use it
to check legibility + localization at the target downscale (MAX_DIM) — can the
model count items, name sprites, and place them in (r,c) cells?

Usage:
    python -m vision.tools.probe_vision <screenshot> [--question "..."] [--max-dim 1536]
"""

import argparse
from pathlib import Path

import cv2

from planner.llm_client import LLMClient, LLMError
from planner.llm_vision import _encode_frame




DEFAULT_Q = ("This is a screenshot of NecroMerger. The board is 5 rows x 3 cols; "
             "top-left cell is (0,0), bottom-right (4,2). List every item you can "
             "see on the board as (row,col): item-name. Then say which merges look "
             "possible and where the grave is.")


def main():
    parser = argparse.ArgumentParser(description="Probe the vision pipe")
    parser.add_argument("image", type=Path)
    parser.add_argument("--question", default=DEFAULT_Q)
    parser.add_argument("--max-dim", type=int, default=1536)
    parser.add_argument("--llm-url", default="http://localhost:8080")
    args = parser.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"cannot read image: {args.image}")

    # Keep probe's MAX_DIM in sync with the encode (override the module default).
    import planner.llm_vision as lv
    lv.MAX_DIM = args.max_dim

    url = _encode_frame(frame)
    print(f"image {frame.shape[1]}x{frame.shape[0]} -> base64 ({len(url)//1024} KB)")
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": args.question},
        ]},
    ]
    client = LLMClient(base_url=args.llm_url)
    reply, _full = client.chat(messages, max_tokens=1024, json_mode=False)
    print("\n--- model reply ---")
    print(reply)
    print("\n--- full JSON ---")
    print(_full)


if __name__ == "__main__":
    main()