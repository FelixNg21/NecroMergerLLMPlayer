"""Menu calibration tool (Phase 1 of menu exploration).

Measured UI geometry is baked into vision/menu.py; this tool discovers it.

Subcommands (live emulator unless --image is given):
  describe [--image PATH] [--question "..."][--save NAME]
      Send the screenshot to the multimodal LLM and print a JSON layout
      description (element bounding boxes in native 1280x2856 pixels).
  capture --xy X,Y --name NAME
      Tap (X,Y), screencap, save to assets/calib/menu/<name>.png, then describe.
  dismiss --xy X,Y [--name NAME]
      Tap (X,Y) then capture+describe a board screenshot to verify a menu closed.

All screenshots land in assets/calib/menu/ so regions can be re-measured offline.
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from env.adb import Device
from planner.llm import _extract_json
from planner.llm_client import LLMClient
from planner.llm_vision import _encode_frame

CALIB_DIR = Path("assets/calib/menu")
SCREEN = (1280, 2856)

LAYOUT_QUESTION = """Analyze this NecroMerger screenshot. Native resolution is 1280x2856; the
image you see may be downscaled, but REPORT COORDINATES IN NATIVE PIXELS by
scaling proportionally. Reply with ONLY JSON:
{"screen": "board|item_popup|devourer_popup|shop|tabs|other",
 "elements": [{"label": "<short name>", "box": [x1, y1, x2, y2]}, ...],
 "notes": "<one sentence on what this screen is>"}
List every visible UI element with a bounding box: popup/panel rectangles,
dismiss or X buttons, bottom tab bar buttons, the Devourer, purchase buttons,
text blocks. Keep the list to at most 14 elements (the important ones). x is
horizontal (0-1280), y is vertical (0-2856, y=0 at top).
If a modal or popup is covering the board, set screen accordingly and note it."""


def describe(client: LLMClient, frame, question: str = LAYOUT_QUESTION) -> dict:
    messages = [
        {"role": "system", "content": "You describe mobile game UI layouts precisely from screenshots."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _encode_frame(frame, max_dim=1536)}},
            {"type": "text", "text": question},
        ]},
    ]
    reply, _full = client.chat(messages, max_tokens=2048, json_mode=False)
    (CALIB_DIR / "last_reply.txt").write_text(reply)
    try:
        return _extract_json(reply)
    except (ValueError, json.JSONDecodeError):
        return {"_raw": reply}


def save_calib(name: str, frame) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    path = CALIB_DIR / f"{name}.png"
    cv2.imwrite(str(path), frame)
    return path


def run(argv=None) -> None:
    parser = argparse.ArgumentParser(description="NecroMerger menu geometry calibration")
    parser.add_argument("command", choices=["describe", "capture", "dismiss"])
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--question", default=LAYOUT_QUESTION)
    parser.add_argument("--save", default=None, help="screenshot name for describe")
    parser.add_argument("--xy", default=None, help="X,Y to tap for capture/dismiss")
    parser.add_argument("--name", default=None, help="calibration screenshot name")
    args = parser.parse_args(argv)

    client = LLMClient()
    device = Device()
    frame = None

    if args.command == "describe":
        if args.image is not None:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise SystemExit(f"cannot read image: {args.image}")
        else:
            device.screencap()
            frame = cv2.imread(str(device.screencap_path))
        name = args.save or (args.image.stem if args.image else "live")
        path = save_calib(name, frame)
        print(f"saved {path}")
        result = describe(client, frame, args.question)
        print(json.dumps(result, indent=1))

    elif args.command == "capture":
        if args.xy is None or args.name is None:
            raise SystemExit("capture needs --xy X,Y and --name NAME")
        x, y = (int(v) for v in args.xy.split(","))
        device.tap(x, y)
        device.wait_for_idle(1.3)
        device.screencap()
        frame = cv2.imread(str(device.screencap_path))
        path = save_calib(args.name, frame)
        print(f"tapped ({x},{y}) -> saved {path}")
        print(json.dumps(describe(client, frame, args.question), indent=1))

    elif args.command == "dismiss":
        if args.xy is None:
            raise SystemExit("dismiss needs --xy X,Y")
        x, y = (int(v) for v in args.xy.split(","))
        device.tap(x, y)
        device.wait_for_idle(0.8)
        device.screencap()
        frame = cv2.imread(str(device.screencap_path))
        name = args.name or "after_dismiss"
        path = save_calib(name, frame)
        print(f"tapped ({x},{y}) -> saved {path}")
        print(json.dumps(describe(client, frame, args.question), indent=1))


if __name__ == "__main__":
    run()
