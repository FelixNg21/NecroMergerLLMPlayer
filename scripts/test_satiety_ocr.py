"""Benchmark OCR engines on the satiety fraction strip (Aug 21 investigation).

Engines compared against labeled frames:
  tess<psm>   tesseract on the 6x-upscaled binary mask (psm 6/7/13)
  apple       Apple Vision framework (pyobjc), accurate mode
  llm         vision LLM (llama-server mmproj) with {"num": prefill JSON

Eval set: assets/calib/satiety_frames/*.png + labels.json mapping file name to
{"num": "61", "den": "250"} (values may be null to skip scoring).

Usage:
  .venv/bin/python scripts/test_satiety_ocr.py
  .venv/bin/python scripts/test_satiety_ocr.py --engines apple,tess13
  .venv/bin/python scripts/test_satiety_ocr.py --image /tmp/x.png --label 61/250
"""

import argparse
import base64
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.llm_client import LLMClient, LLMError  # noqa: E402
from vision.satiety import (  # noqa: E402
    SatietyReader,
    apple_vision_text,
    fraction_text_crop,
    parse_fraction,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "assets" / "calib" / "satiety_frames"
SCALE = 6
TESSERACT = "/opt/homebrew/bin/tesseract"


def extract_ink(frame, reader: SatietyReader) -> object | None:
    """Binary ink mask of the fraction text; dynamic crop first, fixed band
    as fallback (the Aug 13 small-strip save state)."""
    got = fraction_text_crop(frame)
    if got is not None:
        return got[0]
    g = cv2.cvtColor(frame[436:486, 80:176], cv2.COLOR_BGR2GRAY)
    return reader._ink(g).astype("uint8") * 255


def prep_mask_png(frame, reader: SatietyReader, tmpdir: Path, name: str) -> Path:
    """Binary satiety mask -> SCALE-x upscaled black-on-white PNG (padded)."""
    ink = extract_ink(frame, reader)
    big = cv2.resize(ink, None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_NEAREST)
    big = 255 - big                       # dark digits on white background
    big = cv2.copyMakeBorder(big, 40, 40, 40, 40, cv2.BORDER_CONSTANT,
                             value=(255,))
    path = tmpdir / f"{name}.png"
    cv2.imwrite(str(path), big)
    return path


def eng_tess(png: Path, psm: str) -> tuple[str | None, str | None]:
    out = subprocess.run(
        [TESSERACT, str(png), "stdout", "--psm", psm],
        capture_output=True).stdout.decode("utf-8", "replace").strip()
    return parse_fraction(out)


SYSTEM = ("You read numbers from a fantasy game HUD image. The image shows a "
          'fraction like "61/250". Reply with ONLY compact JSON '
          '{"num": "<digits>", "den": "<digits>"} using the digits you see.')


def _png_data_uri(png: Path) -> str:
    raw = png.read_bytes()
    return ("data:image/png;base64,"
            + base64.b64encode(raw).decode())


def eng_llm(reader: SatietyReader, frame, tmpdir: Path, name: str,
            client: LLMClient, fewshot: list[tuple[str, str]] | None = None
            ) -> tuple[str | None, str | None, float]:
    """Vision-LLM read of the mask crop. fewshot = [(value, png_path)] banked
    reference images shown before the query."""
    png = prep_mask_png(frame, reader, tmpdir, name)
    parts = [{"type": "image_url", "image_url": {"url": _png_data_uri(png)}}]
    text = "What fraction is shown? Reply ONLY the JSON."
    if fewshot:
        lines = []
        for value, ref in fewshot:
            parts.append({"type": "image_url",
                          "image_url": {"url": _png_data_uri(Path(ref))}})
            lines.append(f'Example: the image above reads "{value}".')
        text = ("\n".join(lines)
                + "\nNow the last image. " + text)
    parts.append({"type": "text", "text": text})
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": parts},
        {"role": "assistant", "content": '{"num": "'},   # prefill
    ]
    t0 = time.perf_counter()
    try:
        reply, _msg = client.chat(messages, max_tokens=64, json_mode=False)
    except LLMError:
        return None, None, time.perf_counter() - t0
    dt = time.perf_counter() - t0
    blob = '{"num": "' + reply
    m = re.search(r"\{.*\}", blob, re.S)
    if not m:
        return None, None, dt
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None, None, dt
    num = data.get("num"); den = data.get("den")
    if isinstance(num, int):
        num = str(num)
    if isinstance(den, int):
        den = str(den)
    if not isinstance(num, str) or not isinstance(den, str):
        return None, None, dt
    return num.strip(), den.strip(), dt


def load_samples(data_dir: Path) -> list[tuple[str, object, str, str]]:
    labels_path = data_dir / "labels.json"
    labels = (json.loads(labels_path.read_text())
              if labels_path.exists() else {})
    samples = []
    for png in sorted(data_dir.glob("*.png")):
        lab = labels.get(png.name) or {}
        num, den = lab.get("num"), lab.get("den")
        if not num or not den:
            continue
        frame = cv2.imread(str(png))
        if frame is None:
            continue
        samples.append((png.name, frame, str(num), str(den)))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--engines", default="tess13,tess7,apple,llm",
                    help="comma list: tess<psm>, apple, llm")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8080")
    ap.add_argument("--image", default=None,
                    help="ad-hoc single frame instead of the eval set")
    ap.add_argument("--label", default=None,
                    help="truth for --image, as num/den")
    args = ap.parse_args()

    reader = SatietyReader()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    client = None
    if any(e == "llm" for e in engines):
        client = LLMClient(base_url=args.llm_url)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"cannot read {args.image}")
            return 2
        num_lab, den_lab = (args.label.split("/") if args.label
                            else (None, None))
        samples = [(Path(args.image).name, frame, num_lab, den_lab)]
    else:
        samples = load_samples(Path(args.data))
    if not samples:
        print(f"no labeled samples under {args.data} (need labels.json)")
        return 2

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        # Pre-render masks once per sample for tess/apple.
        masks = {name: prep_mask_png(frame, reader, tmpdir, name)
                 for name, frame, _, _ in samples}
        # Few-shot references from the token bank (rendered at the same scale).
        fewshot = []
        for value in sorted(reader.bank, key=len):
            for i, tpl in enumerate(reader.bank[value][:1]):
                big = cv2.resize(tpl, None, fx=SCALE, fy=SCALE,
                                 interpolation=cv2.INTER_NEAREST)
                big = 255 - big
                p = tmpdir / f"ref_{value}_{i}.png"
                cv2.imwrite(str(p), cv2.copyMakeBorder(
                    big, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(255,)))
                fewshot.append((value, str(p)))

        results: dict[str, dict] = {}
        for eng in engines:
            rows = []
            for name, frame, num_lab, den_lab in samples:
                t0 = time.perf_counter()
                if eng.startswith("tess"):
                    got = eng_tess(masks[name], eng[4:] or "13")
                    dt = time.perf_counter() - t0
                elif eng == "apple":
                    got = parse_fraction(apple_vision_text(masks[name]))
                    dt = time.perf_counter() - t0
                elif eng == "llm":
                    got_num, got_den, dt = eng_llm(
                        reader, frame, tmpdir, name, client, fewshot)
                    got = (got_num, got_den)
                else:
                    print(f"unknown engine {eng}")
                    return 2
                rows.append((name, got, (num_lab, den_lab), dt))
            results[eng] = {
                "rows": rows,
                "num_ok": sum(1 for _, g, l, _ in rows
                              if l[0] and g[0] == l[0]),
                "den_ok": sum(1 for _, g, l, _ in rows
                              if l[1] and g[1] == l[1]),
                "n": sum(1 for _, g, l, _ in rows if l[0] and l[1]),
                "ms": sum(r[3] for r in rows) / max(1, len(rows)) * 1000,
            }

        print(f"\n{'engine':10} {'n':>3} {'num':>8} {'den':>8} {'both':>8} "
              f"{'avg ms':>9}")
        for eng, r in results.items():
            both = sum(1 for _, g, l, _ in r["rows"]
                       if l[0] and l[1] and g == l)
            n = r["n"] or 1
            print(f"{eng:10} {r['n']:>3} {r['num_ok']}/{n:>4} "
                  f"{r['den_ok']}/{n:>4} {both}/{n:>5} {r['ms']:>9.0f}")

        print("\nmismatches:")
        for eng, r in results.items():
            for name, got, lab, _dt in r["rows"]:
                if not (lab[0] and lab[1]):
                    continue
                mark = "OK " if got == lab else \
                    (f"{got[0] or '?'}/{got[1] or '?'}".ljust(12))
                if got != lab:
                    print(f"  [{eng}] {name}: truth {lab[0]}/{lab[1]} "
                          f"got {got[0]}/{got[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())