"""Apply review decisions from the bank-purge gallery.

Usage: `.venv/bin/python scripts/apply_review.py decisions.json [--dry-run]`

decisions.json maps repo-relative asset paths to actions:
  "restore"          -> git checkout -- <path> (recover HEAD content)
  "delete"           -> confirm absent (already absent; recorded only)
  "refile:<new_id>"  -> write HEAD content of <path> to
                        assets/templates/<new_id>__<next>.png
  "unsure"           -> skip (left for later)

Refile indices continue at max+1 (never reuse freed numbers). Prints every
action taken. Safe to re-run (idempotent for restore/delete).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args, binary=False):
    if binary:
        return subprocess.run(list(args), capture_output=True, cwd=ROOT)
    r = subprocess.run(list(args), capture_output=True, text=True, cwd=ROOT)
    return r


def next_index(item_id: str) -> int:
    nums = []
    for p in (ROOT / "assets" / "templates").glob(f"{item_id}__*.png"):
        m = re.search(r"__(\d+)\.png$", p.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 0


def main() -> int:
    dry = "--dry-run" in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not paths:
        print("usage: apply_review.py decisions.json [--dry-run]")
        return 2
    try:
        decisions = json.loads(Path(paths[0]).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read decisions: {exc}")
        return 2
    for path, action in sorted(decisions.items()):
        if action == "restore":
            print(("would restore " if dry else "restore ") + path)
            if not dry:
                r = run("git", "checkout", "--", path)
                if r.returncode != 0:
                    print("  FAILED:", r.stderr.strip()[:160])
        elif action == "delete":
            print("keep deleted " + path)
        elif action.startswith("refile:"):
            new_id = action.split(":", 1)[1]
            if not re.fullmatch(r"[a-z0-9_]+", new_id):
                print(f"SKIP {path}: bad target id {new_id!r}")
                continue
            dest = ROOT / "assets" / "templates" / f"{new_id}__{next_index(new_id)}.png"
            print(("would refile " if dry else "refile ") + f"{path} -> {dest.name}")
            if not dry:
                r = run("git", "show", f"HEAD:{path}", binary=True)
                if r.returncode != 0:
                    print("  FAILED: not in HEAD")
                    continue
                dest.write_bytes(r.stdout)
        elif action == "unsure":
            print("skip (unsure) " + path)
        else:
            print(f"SKIP {path}: unknown action {action!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
