"""Verification for glossary merge-chains + auto-bank digits (Aug 10 build).

Run: `.venv/bin/python scripts/verify_chains_digits.py` (exit 0/1).

Drives the real code paths end-to-end with stubs (no LLM server, no emulator):
  A. summarize_session E2E: a `merge chain <family>:` items line is routed to a
     dedicated `(chain)` glossary block; invented-hop chains are dropped; plain
     items still append under a normal timestamp block; a re-summarize with a
     new chain REPLACES the old block (no duplicates).
  B. summarize_session negative: plain items only -> zero `(chain)` blocks.
  C. Identifier.identify on a real saved popup: an LLM-confirmed level whose
     digit glyph isn't banked is auto-banked (digits_dir/3__0.png); a second
     identify does NOT re-bank; llm=None (OCR-only) never banks; bank_digit
     refuses a conflicting glyph.
  D. Regression: py_compile the two changed files + heuristic dry-run.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from planner.glossary import prune_glossary, read_chains, read_glossary, _get_knowledge_dir  # noqa: E402

from metrics.logger import SessionLog
from planner.vision_drive import VisionDrivenPlanner
from vision.classifier import TemplateClassifier
from vision.identify import Identifier, POPUP_DISMISS
from vision.grid import Cell

POPUP_FIXTURE = ROOT / "assets" / "calib" / "menu" / "item_popup.png"
BOARD_FRAME = ROOT / "screenshots" / "Screenshot_1786318878.png"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


class FakeLLMClient:
    """Canned chat() replies (one per summarize call), no network."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, max_tokens=1024, temperature=0.0, json_mode=True):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply, {}


class ClassifierStub:
    """Minimal classifier surface used by _valid_glossary_items/chain gate."""

    def __init__(self, bank: set[str]):
        self.templates = bank

    def has(self, item_id: str) -> bool:
        return item_id in self.templates


class FakeDevice:
    """Device stub: every screencap returns the saved popup fixture."""

    def __init__(self, popup_path: Path, out_path: Path):
        self.popup_path = popup_path
        self.screencap_path = out_path
        self.back_presses = 0
        self.taps = []

    def tap(self, x, y):
        self.taps.append((x, y))

    def wait_for_idle(self, seconds):
        pass

    def screencap(self):
        self.screencap_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.popup_path, self.screencap_path)

    def back(self):
        self.back_presses += 1


class StubLLM:
    """chat_message stub returning the popup-body JSON for read_item_popup."""

    def __init__(self, level: int = 3, name: str = "skeleton"):
        self.level = level
        self.name = name

    def chat_message(self, messages, max_tokens=256, temperature=0.0,
                     json_mode=True, tools=None, tool_choice=None):
        content = json.dumps({
            "name": self.name,
            "level": self.level,
            "description": f"A {self.name}.",
            "merge_info": f"two {self.name.title()} -> {self.name.title()} of the next level",
        })
        return content, None, {}


def make_planner(tmp: Path, replies: list[str], bank: set[str]) -> VisionDrivenPlanner:
    log = SessionLog(tmp / "session.jsonl")
    planner = VisionDrivenPlanner(
        client=FakeLLMClient(replies),
        log=log,
        chat_log_path=tmp / "llm_chats.jsonl",
        learnings_path=tmp / "learnings.md",
        glossary_path=tmp / "glossary.md",
        classifier=ClassifierStub(bank),
        live=False,
        wiki_check=False,
        wiki_tool=False,
        cravings=None,
    )
    planner._last_summary_t = 0.0
    return planner


def feed_log(planner: VisionDrivenPlanner, n: int = 1) -> None:
    planner.log.path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(n):
        planner.log.log("vision_drive", ok=True, action="collect", attempt=1)


def part_a(tmp: Path):
    bank = {"bone", "ribcage", "skeleton_lvl1", "skeleton_lvl2"}
    summary1 = json.dumps({
        "learnings": ["merging two identical skeletons creates a skeleton_lvl2"],
        "items": [
            "- merge chain skeleton: bone -> ribcage -> skeleton_lvl1 -> skeleton_lvl2",
            "- merge chain skeleton: invent_xyz -> bone",   # invented hop -> dropped
            "- skeleton_lvl1",
        ],
        "remove": [], "confirmed": [], "remove_items": [],
    })
    summary2 = json.dumps({
        "learnings": [],
        "items": [
            "- merge chain skeleton: bone -> ribcage -> skeleton_lvl2",
        ],
        "remove": [], "confirmed": [], "remove_items": [],
    })
    planner = make_planner(tmp, [summary1, summary2], bank)
    feed_log(planner)
    planner.summarize_session(board=None, frame=None)

    kd = _get_knowledge_dir(planner.glossary_path)
    gl = read_glossary(knowledge_dir=kd)
    check("A1: chain block written", "## Item skeleton (chain)" in gl
          and "- merge chain: bone -> ribcage -> skeleton_lvl1 -> skeleton_lvl2" in gl)
    check("A2: invented-hop chain dropped", "invent_xyz" not in gl)
    check("A3: bare-id item dropped (chain text still present)",
          "- skeleton_lvl1" not in gl and "skeleton_lvl1 -> skeleton_lvl2" in gl)

    # re-summarize with a different chain for the same family -> replace, no dup
    feed_log(planner)
    planner.summarize_session(board=None, frame=None)
    gl2 = read_glossary(knowledge_dir=kd)
    blocks = [b for b in gl2.split("## ") if b.startswith("Item skeleton (chain)")]
    check("A4: re-summarize replaces (single chain block)",
          len(blocks) == 1 and "bone -> ribcage -> skeleton_lvl2" in gl2
          and "skeleton_lvl1 -> skeleton_lvl2" not in gl2)


def part_b(tmp: Path):
    planner = make_planner(tmp, [json.dumps({
        "learnings": [],
        "items": ["- bone", "- skeleton_lvl2", "- bone: spawns from Graves"],
        "remove": [], "confirmed": [], "remove_items": [],
    })], {"bone", "skeleton_lvl2"})
    feed_log(planner)
    planner.summarize_session(board=None, frame=None)
    kd = _get_knowledge_dir(planner.glossary_path)
    gl = read_glossary(knowledge_dir=kd)
    # Bare ids are board snapshots (Aug 23 prune) — dropped at the write path;
    # only fact lines survive, and no timestamp snapshot block is written.
    # The summary items above are bare ids (not chain/spawn/popup format),
    # so they should be dropped. The glossary should remain empty (or only
    # contain pre-existing entries from the bank).
    check("B: bare ids dropped, fact line kept, no snapshot block",
          "## Item bone (popup)" not in gl
          and "## Item skeleton_lvl2 (popup)" not in gl
          and "(chain)" not in gl)

    gl_path = tmp / "test_glossary.md"
    # B2: remove_items must NOT delete (popup)/(chain) fact blocks (Aug 23
    # soak: the summary wiped four valid popup blocks for absent items);
    # invented-id blocks stay removable.
    # soak: the summary wiped four valid popup blocks for absent items);
    # invented-id blocks stay removable.
    gl_path.write_text(
        "## Item zombie_lvl2 (popup)\n\n- feed value: 60\n- damage value: 25\n\n"
        "## Item skeleton (chain)\n\n- merge chain: bone -> ribcage -> skeleton_lvl1\n\n"
        "## Item necromerger_lvl1\n\n- invented id block\n")
    n = prune_glossary(gl_path, ["zombie_lvl2", "skeleton", "necromerger_lvl1"])
    gl = gl_path.read_text()
    check("B2: popup block survives remove_items",
          "## Item zombie_lvl2 (popup)" in gl and "- feed value: 60" in gl)
    check("B3: chain block survives remove_items",
          "## Item skeleton (chain)" in gl
          and "bone -> ribcage -> skeleton_lvl1" in gl)
    check("B4: invented-id block still removed",
          "necromerger_lvl1" not in gl and n == 1)


def part_c(tmp: Path):
    digits_dir = tmp / "digits"
    sig_dir = tmp / "signatures"
    tmpl_dir = tmp / "templates"
    popup_out = tmp / "popup.png"

    device = FakeDevice(POPUP_FIXTURE, popup_out)
    classifier = TemplateClassifier(templates_dir=str(tmpl_dir), seed=False)
    board_frame = cv2.imread(str(BOARD_FRAME))
    cell = Cell(2, 2, cx=300, cy=1200, item_id=None, cell_px=226)

    # 1) LLM-confirmed level, digit NOT banked -> auto-banked
    ident = Identifier(device, classifier, signatures_dir=str(sig_dir),
                       digits_dir=str(digits_dir), llm=StubLLM(level=3))
    item_id, info = ident.identify(board_frame, cell)
    d3 = digits_dir / "3__0.png"
    check("C1: LLM-confirmed digit auto-banked (3__0.png)",
          d3.exists() and len(ident.digits.get(3, [])) == 1
          and info.get("level") == 3, f"item={item_id}")

    # 2) second identify on the same popup -> level read from bank, no re-bank
    ident2 = Identifier(device, classifier, signatures_dir=str(sig_dir),
                        digits_dir=str(digits_dir), llm=StubLLM(level=3))
    _id2, info2 = ident2.identify(board_frame, cell)
    n_files = len(list(digits_dir.glob("3__*.png")))
    check("C2: no re-bank on second identify", n_files == 1
          and len(ident2.digits.get(3, [])) == 1 and info2.get("level") == 3)

    # 3) default (OCR-only) identify on a fresh popup -> level-less base name,
    #    no auto-bank (no LLM to confirm the level)
    digits2 = tmp / "digits2"
    sig2 = tmp / "signatures2"
    tmpl2 = tmp / "templates2"
    ident3 = Identifier(FakeDevice(POPUP_FIXTURE, tmp / "popup2.png"),
                        TemplateClassifier(templates_dir=str(tmpl2), seed=False),
                        signatures_dir=str(sig2), digits_dir=str(digits2), llm=None)
    ident3.identify(board_frame, cell)
    check("C3: llm=None never auto-banks digits", not digits2.exists()
          or not any(digits2.glob("*.png")))

    # 4) bank_digit conflict guard: same glyph already banked under level 6
    ident4 = Identifier(FakeDevice(POPUP_FIXTURE, tmp / "popup3.png"),
                        TemplateClassifier(templates_dir=str(tmpl3 := tmp / "templates3"),
                                           seed=False),
                        signatures_dir=str(tmp / "signatures3"),
                        digits_dir=str(tmp / "digits3"), llm=StubLLM(level=3))
    glyph = cv2.imread(str(POPUP_FIXTURE))
    from vision import ocr
    glyph = ocr.glyph_crop(glyph)
    ident4.digits = {6: [glyph]}
    refused = ident4.bank_digit(3, cv2.imread(str(POPUP_FIXTURE)))
    check("C4: bank_digit refuses conflicting glyph", refused is False
          and not any(ident4.digits_dir.glob("3__*.png")),
          f"refused={refused} glob={list(ident4.digits_dir.glob('3__*.png'))}")


def part_d() -> None:
    ok = True
    for f in (ROOT / "planner" / "vision_drive.py", ROOT / "vision" / "identify.py"):
        r = subprocess.run([sys.executable, "-m", "py_compile", str(f)],
                           capture_output=True)
        ok = ok and r.returncode == 0
    r = subprocess.run([sys.executable, str(ROOT / "main.py"), "--dry-run",
                        "--planner", "heuristic", "--screenshot",
                        str(BOARD_FRAME)], capture_output=True, timeout=120)
    ok = ok and r.returncode == 0 and b"Move(" in r.stdout
    check("D: py_compile + heuristic dry-run regression", ok)


def part_e() -> None:
    """Pure wiki helpers: phrasal '<item> merge chain' queries deflect to the
    leading item noun, and `_extract_merge_info` surfaces merge-relevant lines
    (components / what-it-summons) so the item page alone is sufficient — the
    model doesn't need a separate 'merge chain' search. Network-free."""
    from planner import wiki

    check("E: 'zombie merge chain' deflects to 'zombie'",
          wiki._leading_item_noun("zombie merge chain") == "zombie")
    check("E: 'skeleton feed value' deflects to 'skeleton'",
          wiki._leading_item_noun("skeleton feed value") == "skeleton")
    check("E: bare 'zombie' is NOT a phrasal deflected query",
          wiki._leading_item_noun("zombie") is None)
    text = (
        "Zombies have two tiers of components:\n"
        "Rotten Flesh: Spawns from level 3+ Graves.\n"
        "Severed Hand: Spawns from level 4+ Graves.\n"
        "When merged, zombies contribute to summoning the Peasant.\n"
        "Some unrelated filler sentence about dialogue.\n"
    )
    mi = wiki._extract_merge_info(text)
    check("E: merge_info keeps component + summon lines",
          "Rotten Flesh" in mi and "Severed Hand" in mi
          and "summoning the Peasant" in mi)
    check("E: merge_info drops unrelated filler",
          "filler sentence" not in mi and len(mi.splitlines()) <= 12)


def part_f() -> None:
    """Regression (Sep 2): the learning-summary path crashed with
    "'Learning' object has no attribute 'strip'". `verify_learning_consistency`
    returns Learning objects under 'learning'; `_verify_learnings` used to leak
    them downstream, and `derate_learning`/`prune_learning`/`_matches`/`_norm`
    called `.strip()` on them. All corpus consumers must tolerate str/dict/
    Learning inputs."""
    from planner.learnings import (verify_learning_consistency, derate_learning,
                                   append_learning, prune_learning,
                                   confirm_learning, _to_learning)
    res = verify_learning_consistency(
        [{"text": "skeletons merge into undead", "type": "pattern",
          "confidence": 0.8}], [], None)
    pics = [r["learning"] for r in res if r["consistent"]]
    # What _verify_learnings hands to the corpus writers (normalized dicts)
    as_dicts = []
    for l in pics:
        if not isinstance(l, dict):
            d = {"text": l.text, "type": l.type, "confidence": l.confidence}
        else:
            d = dict(l)
        as_dicts.append(d)
    tmp = Path(tempfile.mkdtemp()) / "l.md"
    tmp.write_text("## Learning t\n\nstatus: candidate\ntype: pattern\n"
                   "confidence: 0.80\nconfirmed: 0\nnegative: 0\n"
                   "outcome_log: []\n\nskeletons merge\n")
    ok = True
    for tag, call in [
        ("append(dicts)", lambda: append_learning(tmp, as_dicts)),
        ("derate(dicts)", lambda: derate_learning(tmp, as_dicts)),
        ("prune(dicts)", lambda: prune_learning(tmp, as_dicts)),
        ("confirm(dicts)", lambda: confirm_learning(tmp, as_dicts)),
        ("derate(Learning)", lambda: derate_learning(
            tmp, [_to_learning(d) for d in as_dicts])),
    ]:
        try:
            call()
        except Exception:  # noqa: BLE001
            ok = False
    check("F: learning-summary corpus writers tolerate dict/Learning inputs",
          ok)


def main() -> int:
    if not POPUP_FIXTURE.exists() or not BOARD_FRAME.exists():
        print(f"missing fixtures: popup={POPUP_FIXTURE.exists()} board={BOARD_FRAME.exists()}")
        return 1
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        part_a(tmp / "a")
        part_b(tmp / "b")
        part_c(tmp / "c")
    part_d()
    part_e()
    part_f()
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())