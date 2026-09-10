"""Build a static review gallery for bank-file purge decisions.

For every deleted-under-HEAD asset (templates/digits/signatures), renders a
card with: the image (extracted from git), own-family match score, top-3
cross-family matches, and a precomputed recommendation (restore / refile /
delete / unsure). The reviewer picks per item (or uses bulk buttons) and
downloads decisions.json; `scripts/apply_review.py` executes it.

Run: `.venv/bin/python scripts/build_review_ui.py` -> /tmp/review/index.html
"""

import base64
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path("/tmp/review")
EXCLUDE_IDS = ("supplycupboard_lvl1",)  # new card-art template distorts matching


def git_blob(path: str):
    r = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, cwd=ROOT)
    if r.returncode != 0:
        return None
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


def bank(exclude=()):
    out = {}
    for p in sorted((ROOT / "assets" / "templates").glob("*.png")):
        iid = p.stem.split("__")[0]
        if iid in exclude:
            continue
        t = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if t is None:
            continue
        out.setdefault(iid, []).append(t)
    return out


def head_bank(exclude=()):
    """Template bank as committed in HEAD (not the depleted worktree).

    Scoring deletions against the current worktree is circular: purged
    files make their own family match worse. Export HEAD once via
    git archive and match against that. Maps id -> [(repo_path, image)].
    """
    import tarfile
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="review_bank_"))
    with subprocess.Popen(["git", "archive", "HEAD", "assets/templates"],
                           stdout=subprocess.PIPE, cwd=ROOT) as proc:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            tar.extractall(path=tmp)
    out = {}
    for p in sorted((tmp / "assets" / "templates").glob("*.png")):
        iid = p.stem.split("__")[0]
        if iid in exclude:
            continue
        t = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if t is None:
            continue
        rel = "assets/templates/" + p.name
        out.setdefault(iid, []).append((rel, t))
    return out


def bank(exclude=()):
    out = {}
    for p in sorted((ROOT / "assets" / "templates").glob("*.png")):
        iid = p.stem.split("__")[0]
        if iid in exclude:
            continue
        t = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if t is None:
            continue
        out.setdefault(iid, []).append(("assets/templates/" + p.name, t))
    return out


def top_matches(img_gray, bank, k=3, exclude_self=None):
    """Best ids for img, EXCLUDING the query file itself (else every file
    trivially self-matches at 1.0 and all verdicts say restore)."""
    res = {}
    for iid, frames in bank.items():
        best = -1.0
        for rel, t in frames:
            if rel == exclude_self:
                continue
            a, b = (img_gray, t) if img_gray.size >= t.size else (t, img_gray)
            try:
                s = float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).max())
            except Exception:
                continue
            best = max(best, s)
        res[iid] = best
    return sorted(res.items(), key=lambda kv: -kv[1])[:k]


def recommend(fam, own, top):
    """(verdict, reason) — restore / refile:<id> / delete / unsure."""
    runner = top[0] if top else (None, -1)
    if own is not None and own >= 0.4 and (len(top) < 2 or own >= top[1][1] + 0.1):
        return "restore", f"own-family wins at {own:.2f}"
    if runner[0] is not None and runner[0] != fam and runner[1] >= 0.7:
        return f"refile:{runner[0]}", f"matches {runner[0]} at {runner[1]:.2f}"
    if (own or -1) < 0.2 and (runner[1] if runner[0] else -1) < 0.35:
        return "delete", "matches nothing (unidentifiable capture)"
    return "unsure", "ambiguous — needs human eyes"


def main() -> None:
    deleted = subprocess.run(
        ["git", "status", "--short", "assets/"], capture_output=True,
        text=True, cwd=ROOT).stdout
    files = [l[3:] for l in deleted.splitlines() if l.startswith(" D ")]
    bank_map = head_bank(exclude=EXCLUDE_IDS)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for f in sorted(files):
        img = git_blob(f)
        if img is None:
            continue
        name = Path(f).name
        stem = Path(f).stem
        fam = stem.split("__")[0] if "__" in stem else stem.rsplit(".", 1)[0]
        safe = re.sub(r"[^a-z0-9_.-]", "_", f.replace("/", "__"))
        cv2.imwrite(str(OUT / "img" / safe), img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if f.startswith("assets/templates/"):
            top = top_matches(gray, bank_map, exclude_self=f)
            own = dict(top).get(fam)
            verdict, reason = recommend(fam, own, top)
        else:
            top, own = [], None
            verdict, reason = "unsure", "non-template bank — eyeball it"
        label = None
        if verdict.startswith("refile:"):
            label = verdict.split(":", 1)[1]
        items.append({
            "path": f, "file": name, "family": fam, "img": "img/" + safe,
            "own": round(own, 3) if own is not None else None,
            "top": [[k, round(v, 3)] for k, v in top],
            "verdict": verdict, "reason": reason, "refile_to": label,
        })
    counts = {}
    for it in items:
        counts[it["verdict"].split(":")[0]] = counts.get(it["verdict"].split(":")[0], 0) + 1
    print(f"items: {len(items)} {counts}")
    with open(OUT / "items.json", "w") as fh:
        json.dump(items, fh)
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Bank purge review</title>
<style>
body{font-family:system-ui,sans-serif;background:#1a1a1a;color:#eee;margin:0;padding:16px}
h1{font-size:20px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{background:#2a2a2a;border-radius:8px;padding:10px;border:2px solid #444}
.card[data-v=restore]{border-color:#3a3}.card[data-v^=refile]{border-color:#39c}
.card[data-v=delete]{border-color:#a33}.card[data-v=unsure]{border-color:#aa3}
img{width:100%;image-rendering:pixelated;background:#000;border-radius:4px}
.meta{font-size:12px;color:#bbb;margin:6px 0}.verdict{font-weight:bold}
.row{display:flex;gap:8px;margin-top:6px;align-items:center;font-size:13px}
button{margin:10px 8px 10px 0;padding:8px 14px;font-size:14px;cursor:pointer}
#bar{position:sticky;top:0;background:#111;padding:10px;border-radius:8px}
</style></head><body>
<h1>Bank purge review (<span id="n"></span> files)</h1>
<div id="bar">
<button onclick="bulk('restore')">All restore</button>
<button onclick="bulk('delete')">All delete</button>
<button onclick="bulk('verdict')">Reset to recommendations</button>
<button onclick="dl()">Download decisions.json</button>
<span id="counts"></span>
</div><div class="grid" id="g"></div>
<script>const ITEMS_JSON = ITEMS_DATA;</script>
<script>
let items = ITEMS_JSON;
function render(){
  document.getElementById('n').textContent = items.length;
  const g = document.getElementById('g'); g.innerHTML = '';
  const c = {};
  items.forEach((it, i) => {
    const v = it.decision || it.verdict;
    const base = v.split(':')[0];
    c[base] = (c[base]||0)+1;
    const d = document.createElement('div');
    d.className = 'card'; d.dataset.v = base;
    d.innerHTML = `<img loading="lazy" src="${it.img}"><div class="meta">
<b>${it.file}</b><br>family: ${it.family} | own: ${it.own}
<br>top: ${it.top.map(t=>t[0]+' '+t[1]).join(', ')}
<br><span class="verdict">suggested: ${it.verdict}</span> — ${it.reason}</div>
<div class="row">
<label><input type="radio" name="d${i}" value="restore" ${v.startsWith('restore')?'checked':''}> restore</label>
<label><input type="radio" name="d${i}" value="delete" ${v==='delete'?'checked':''}> delete</label>
${it.refile_to?`<label><input type="radio" name="d${i}" value="refile:${it.refile_to}" ${v.startsWith('refile')?'checked':''}> refile→${it.refile_to}</label>`:`<label><input type="radio" name="d${i}" value="unsure" ${v==='unsure'?'checked':''}> unsure</label>`}
</div>`;
    d.querySelectorAll('input').forEach(inp => inp.onchange = () => { it.decision = inp.value; render(); });
    g.appendChild(d);
  });
  document.getElementById('counts').textContent = JSON.stringify(c);
}
function bulk(mode){
  items.forEach(it => { it.decision = mode==='verdict' ? it.verdict : mode; });
  render();
}
function dl(){
  const dec = {};
  items.forEach(it => { dec[it.path] = it.decision || it.verdict; });
  const blob = new Blob([JSON.stringify(dec, null, 1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'decisions.json'; a.click();
}
render();
</script>
</body></html>"""
    with open(OUT / "items.json") as fh:
        data = fh.read()
    html = html.replace("ITEMS_DATA", data)
    with open(OUT / "index.html", "w") as fh:
        fh.write(html)
    print("wrote", OUT / "index.html")


if __name__ == "__main__":
    sys.exit(main())
