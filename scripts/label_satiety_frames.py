"""Resolve satiety-eval frame labels by engine consensus + render contact sheet.

For every frame in assets/calib/satiety_frames/ runs tesseract(psm13) and Apple
Vision on the binary strip mask; combines with meta.json's feed-arithmetic
expectation. A frame is LABELED when >=2 independent signals agree exactly;
otherwise it stays unlabeled ("?" in labels.json) for human arbitration via
the rendered contact_sheet.png (strips enlarged, proposed reads printed).

Usage:
  .venv/bin/python scripts/label_satiety_frames.py [--dir DIR]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.test_satiety_ocr import TESSERACT, SCALE, prep_mask_png  # noqa: E402
from vision.satiety import SatietyReader, apple_vision_text, parse_fraction  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "assets" / "calib" / "satiety_frames"))
    args = ap.parse_args()
    data = Path(args.dir)
    reader = SatietyReader()

    meta = {}
    meta_path = data / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    tiles = []
    labels: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for png in sorted(data.glob("frame_*.png")):
            frame = cv2.imread(str(png))
            if frame is None:
                continue
            mask = prep_mask_png(frame, reader, tmpdir, png.stem)
            tess = parse_fraction(subprocess.run(
                [TESSERACT, str(mask), "stdout", "--psm", "13"],
                capture_output=True).stdout.decode("utf-8", "replace").strip())
            appl = parse_fraction(apple_vision_text(str(mask)))
            exp = (meta.get(png.name, {}).get("expected"),
                   meta.get(png.name, {}).get("den"))
            signals = [s for s in (tess, appl, exp) if s[0] and s[1]]
            chosen = None
            for i in range(len(signals)):
                for j in range(i + 1, len(signals)):
                    if signals[i] == signals[j]:
                        chosen = signals[i]
                        break
                if chosen:
                    break
            num, den = chosen if chosen else (None, None)
            labels[png.name] = {"num": num, "den": den,
                                "_tess": f"{tess[0]}/{tess[1]}",
                                "_apple": f"{appl[0]}/{appl[1]}",
                                "_exp": f"{exp[0]}/{exp[1]}"}
            print(f"{png.name}: tess={tess[0]}/{tess[1]} "
                  f"apple={appl[0]}/{appl[1]} exp={exp[0]}/{exp[1]} "
                  f"-> {'LABELED ' + str(chosen) if chosen else 'UNRESOLVED'}")
            tiles.append((png.name, cv2.imread(str(mask)), chosen))

    out = {k: {"num": v["num"], "den": v["den"]} for k, v in labels.items()}
    (data / "labels.json").write_text(json.dumps(out, indent=1))

    # Contact sheet: one row per frame — enlarged strip + name + verdict.
    rows = []
    for name, mask, chosen in tiles:
        row = cv2.copyMakeBorder(mask, 12, 12, 12, 12,
                                 cv2.BORDER_CONSTANT, value=(200,))
        tag = f"{name[:22]} {chosen[0]+'/'+chosen[1] if chosen else '?'}"
        cv2.putText(row, tag, (8, row.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30,), 2, cv2.LINE_AA)
        w = max(760, row.shape[1])
        row = cv2.copyMakeBorder(row, 4, 4, 0, w - row.shape[1],
                                 cv2.BORDER_CONSTANT, value=(255,))
        rows.append(row)
    if rows:
        width = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1],
                                   cv2.BORDER_CONSTANT, value=(255,))
                for r in rows]
        sheet = cv2.vconcat(rows)
        cv2.imwrite(str(data / "contact_sheet.png"), sheet)
        print(f"contact sheet -> {data / 'contact_sheet.png'}")
    print(f"labels.json written ({sum(1 for v in out.values() if v['num'])}"
          f"/{len(out)} resolved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
