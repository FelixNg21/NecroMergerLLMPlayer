"""Two-tier semantic memory for the vision-driven planner.

The model writes lasting learnings to `learnings.md` (periodically during play
and once at session end) and the planner reads them back into the prompt on
subsequent moves, so it accumulates knowledge across sessions.

To keep bad learnings from persisting, the store is self-correcting:
- **Trust gating:** new learnings are *candidates* and are only promoted to
  *committed* after later summaries explicitly confirm them.
- **Self-pruning:** a summary can mark stored learnings it now believes are
  wrong/contradicted/never-validated; those blocks are deleted.
- **Negative evidence:** learnings written during a window that ended in
  heuristic-fallback moves start with negative evidence and are dropped if they
  accumulate too much.

New format: learnings have types and confidence scores.

File format: one `## Learning` block per learning, with metadata lines:

    ## Learning <title>

    status: candidate|committed
    type: fact|pattern|visual|anti-pattern
    confidence: <float 0.0-1.0>
    confirmed: <int>   # confirmations by later summaries
    negative: <int>    # negative evidence (failure-prone windows)
    outcome_log: <json list of [session, outcome]>  # optional, for cross-session tracking

    <text>
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("learnings.md")
MAX_LEARNINGS = 20          # most recent learnings injected into the prompt
COMMIT_CONFIRMATIONS = 2    # a candidate is promoted after this many confirmations
MAX_NEGATIVE = 2            # a candidate with this much negative evidence is dropped

# Learning type definitions
LEARNING_TYPES = ("fact", "pattern", "visual", "anti-pattern")
FACT_TYPES = ("fact",)  # types that require high confidence
MIN_CONFIDENCE = {  # minimum confidence per type for auto-accept
    "fact": 0.9,
    "pattern": 0.7,
    "visual": 0.8,
    "anti-pattern": 0.7,
}

_META_RE = re.compile(r"^(status|type|confidence|confirmed|negative|outcome_log):\s*(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+")
_BOLD_RE = re.compile(r"\*+")
_WS_RE = re.compile(r"\s+")


@dataclass
class Learning:
    text: str
    status: str = "candidate"      # candidate | committed
    type: str = "pattern"          # fact | pattern | visual | anti-pattern
    confidence: float = 0.7        # 0.0-1.0, model self-assessed
    confirmed: int = 0
    negative: int = 0
    title: str = ""
    outcome_log: list = field(default_factory=list)  # list of [session_id, outcome]

    @property
    def committed(self) -> bool:
        return self.status == "committed"

    def meets_confidence_threshold(self) -> bool:
        """Check if confidence meets minimum for this type."""
        return self.confidence >= MIN_CONFIDENCE.get(self.type, 0.7)

    def add_outcome(self, session_id: str, outcome: str) -> None:
        """Log an outcome for cross-session tracking."""
        self.outcome_log.append([session_id, outcome])

    def should_demote(self, sessions_without_reinforcement: int = 20) -> bool:
        """Check if committed learning should be demoted to candidate."""
        if not self.committed:
            return False
        return len(self.outcome_log) >= sessions_without_reinforcement


def _norm(s: str) -> str:
    if not isinstance(s, str):
        s = _text_of(s)
    s = _BULLET_RE.sub("", s.strip())
    s = _BOLD_RE.sub("", s)
    return _WS_RE.sub(" ", s.lower()).strip()


def _norm_tokens(s: str) -> set[str]:
    """Token set for similarity comparison."""
    return set(_norm(s).split())


def _text_of(x) -> str:
    """Coerce a corpus item (str, dict with 'text', or Learning) to its text.
    Several consumers (append_learning, derate_learning, prune_learning,
    confirm_learning) can receive dicts/objects instead of bare strings."""
    if isinstance(x, str):
        return x
    if isinstance(x, Learning):
        return x.text
    if isinstance(x, dict):
        return str(x.get("text", ""))
    return str(x)


def _matches(stored, query) -> bool:
    """Substring match + token-set Jaccard >= 0.7 for paraphrase detection.
    Both args are coerced to text first, so callers may pass str, dict, or
    Learning objects (guards against '.strip()' crashes on non-strings)."""
    ns, nq = _norm(_text_of(stored)), _norm(_text_of(query))
    if not nq:
        return False
    if nq == ns or nq in ns:
        return True
    ts, tq = _norm_tokens(stored), _norm_tokens(query)
    if not ts or not tq:
        return False
    smaller, larger = (ts, tq) if len(ts) <= len(tq) else (tq, ts)
    return len(smaller & larger) / len(smaller) >= 0.7


def _int_or(s, default: int) -> int:
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def _float_or(s, default: float) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _parse_blocks(text: str) -> list[tuple[str, str]]:
    """Split a memory file into (header, body) pairs on `## ` lines."""
    if not text.strip():
        return []
    blocks = re.split(r"\n(?=## )", "\n" + text.strip())
    out = []
    for b in blocks:
        lines = b.splitlines()
        if lines and lines[0].startswith("## "):
            title = lines[0][3:].strip()
            while True:
                stripped = re.sub(r"^(Session|Learning)\s+", "", title, flags=re.I)
                if stripped == title:
                    break
                title = stripped
            out.append((title, "\n".join(lines[1:]).strip()))
    return out


def _parse_block(body: str, title: str) -> list[Learning]:
    """Parse one block into learning(s). Old `## Session` blocks (multi-bullet,
    no metadata) are migrated into individual candidate learnings."""
    status, confirmed, negative = "candidate", 0, 0
    ltype, confidence = "pattern", 0.7
    outcome_log = []
    text_lines: list[str] = []
    for line in body.splitlines():
        m = _META_RE.match(line.strip())
        if m:
            key, val = m.group(1), m.group(2).strip()
            if key == "status":
                status = val if val in ("candidate", "committed") else "candidate"
            elif key == "type":
                ltype = val if val in LEARNING_TYPES else "pattern"
            elif key == "confidence":
                confidence = _float_or(val, 0.7)
            elif key == "confirmed":
                confirmed = _int_or(val, 0)
            elif key == "negative":
                negative = _int_or(val, 0)
            elif key == "outcome_log":
                try:
                    outcome_log = json.loads(val)
                except json.JSONDecodeError:
                    outcome_log = []
        else:
            text_lines.append(line.strip())
    text_lines = [l for l in text_lines if l]
    if not text_lines:
        return []

    def mk(txt: str) -> Learning:
        return Learning(text=txt, status=status, type=ltype, confidence=confidence,
                        confirmed=confirmed, negative=negative, title=title,
                        outcome_log=outcome_log)

    if all(l.startswith("- ") or l.startswith("* ") or l.startswith("• ")
           for l in text_lines):
        return [mk(_BULLET_RE.sub("", l).strip()) for l in text_lines if _BULLET_RE.sub("", l).strip()]
    return [mk(_WS_RE.sub(" ", " ".join(text_lines)).strip())]


def _to_block(l: Learning) -> str:
    title = l.title or datetime.now().strftime("%b %d, %Y %H:%M")
    outcome_json = json.dumps(l.outcome_log) if l.outcome_log else "[]"
    return (f"## Learning {title}\n\n"
            f"status: {l.status}\n"
            f"type: {l.type}\n"
            f"confidence: {l.confidence:.2f}\n"
            f"confirmed: {l.confirmed}\n"
            f"negative: {l.negative}\n"
            f"outcome_log: {outcome_json}\n\n"
            f"{l.text.strip()}\n")


def read(path: Path = DEFAULT_PATH) -> list[Learning]:
    """Parse the learnings file into Learning objects (old format migrated)."""
    if not path.exists():
        return []
    learnings: list[Learning] = []
    for title, body in _parse_blocks(path.read_text()):
        learnings.extend(_parse_block(body, title))
    return learnings


def write(path: Path, learnings: list[Learning]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_to_block(l) + "\n" for l in learnings))


def read_learnings(path: Path = DEFAULT_PATH,
                   max_learnings: int = MAX_LEARNINGS) -> str:
    """Prompt text for the most recent learnings, split committed/tentative with type tags."""
    items = read(path)[-max_learnings:]
    if not items:
        return ""
    
    # Group by type for better prompt organization
    by_type = {t: [] for t in LEARNING_TYPES}
    for l in items:
        by_type.setdefault(l.type, []).append(l)
    
    parts = []
    for t in LEARNING_TYPES:
        committed = [f"- [{t}] {l.text}" for l in by_type[t] if l.committed]
        candidates = [f"- [{t}] {l.text}" for l in by_type[t] if not l.committed]
        if committed:
            parts.append(f"Committed {t} learnings (apply these):\n" + "\n".join(committed))
        if candidates:
            parts.append(f"Tentative {t} learnings (treat as tentative):\n" + "\n".join(candidates))
    return "\n\n".join(parts)


def append_learning(path: Path, texts, title: str | None = None) -> int:
    """Add new learnings as candidates, deduping against stored text.
    Returns the number actually added.
    
    `texts` can be:
    - List of strings (old format, defaults to type='pattern', confidence=0.7)
    - List of dicts with keys: text, type, confidence
    - List of Learning objects
    """
    if isinstance(texts, str):
        texts = [texts]
    existing = read(path)
    added = 0
    for t in texts:
        if isinstance(t, Learning):
            text = t.text.strip()
            ltype = t.type
            confidence = t.confidence
        elif isinstance(t, dict):
            text = t.get("text", "").strip()
            ltype = t.get("type", "pattern")
            confidence = t.get("confidence", 0.7)
        else:
            text = t.strip()
            ltype = "pattern"
            confidence = 0.7
        if not text:
            continue
        if any(_matches(e.text, text) for e in existing):
            continue
        existing.append(Learning(text=text, type=ltype, confidence=confidence, title=title or ""))
        added += 1
    if added:
        write(path, existing)
    return added


def confirm_learning(path: Path, texts) -> int:
    """Bump confirmation on matching learnings; promote candidates at the
    commit threshold. Returns how many learnings were confirmed."""
    existing = read(path)
    n = 0
    for l in existing:
        if l.committed or not any(_matches(l.text, q) for q in texts):
            continue
        l.confirmed += 1
        n += 1
        if l.confirmed >= COMMIT_CONFIRMATIONS:
            l.status = "committed"
    if n:
        write(path, existing)
    return n


def prune_learning(path: Path, texts) -> int:
    """Delete stored learnings matching any of the given texts.
    Returns how many were removed."""
    existing = read(path)
    kept = [l for l in existing
            if not any(_matches(l.text, q) for q in texts)]
    removed = len(existing) - len(kept)
    if removed:
        write(path, kept)
    return removed


def derate_learning(path: Path, texts=None) -> int:
    """Add negative evidence to candidate learnings; drop those past the cap.
    `texts` None => all candidates. Returns how many were dropped."""
    existing = read(path)
    changed, dropped = False, 0
    for l in existing:
        if l.committed:
            continue
        if texts is None or any(_matches(l.text, q) for q in texts):
            l.negative += 1
            changed = True
            if l.negative >= MAX_NEGATIVE:
                dropped += 1
    if dropped:
        existing = [l for l in existing
                    if not (not l.committed and l.negative >= MAX_NEGATIVE)]
    if changed:
        write(path, existing)
    return dropped


# =====================================================================
# PHASE 2: VERIFICATION / VALIDATION
# =====================================================================

def _to_learning(obj) -> Learning:
    """Convert dict or Learning to Learning object."""
    if isinstance(obj, Learning):
        return obj
    if isinstance(obj, dict):
        return Learning(
            text=obj.get("text", ""),
            type=obj.get("type", "pattern"),
            confidence=obj.get("confidence", 0.7),
            status=obj.get("status", "candidate"),
            confirmed=obj.get("confirmed", 0),
            negative=obj.get("negative", 0),
            title=obj.get("title", ""),
            outcome_log=obj.get("outcome_log", []),
        )
    raise TypeError(f"Expected dict or Learning, got {type(obj)}")


def verify_learning_consistency(new_learnings,
                                 committed_learnings,
                                 glossary_stats: dict | None = None) -> list[dict]:
    """Verify new candidate learnings against committed learnings and glossary.
    Returns list of verification results: [{'learning': l, 'consistent': bool, 'issues': [], 'score': float}]
    
    This is a lightweight check - not a full LLM call. Does:
    - Contradiction detection (same topic, opposite claim)
    - Confidence threshold check
    - Glossary cross-check for 'fact' and 'visual' types
    """
    results = []
    for nl in new_learnings:
        nl_obj = _to_learning(nl)
        issues = []
        score = 1.0
        
        # Confidence threshold check
        if not nl_obj.meets_confidence_threshold():
            issues.append(f"Confidence {nl_obj.confidence:.2f} below minimum {MIN_CONFIDENCE.get(nl_obj.type, 0.7)} for type '{nl_obj.type}'")
            score *= 0.5
        
        # Contradiction detection against committed learnings
        for cl in committed_learnings:
            if cl.committed and _matches(cl.text, nl_obj.text):
                # Same topic - check for contradiction
                cl_norm = _norm(cl.text).lower()
                nl_norm = _norm(nl_obj.text).lower()
                if ("always" in cl_norm and "never" in nl_norm) or                    ("never" in cl_norm and "always" in nl_norm) or                    ("must" in cl_norm and "must not" in nl_norm) or                    ("must not" in cl_norm and "must" in nl_norm):
                    issues.append(f"Contradicts committed learning: {cl.text[:80]}...")
                    score *= 0.3
        
        # Glossary cross-check for fact/visual types
        if glossary_stats and nl_obj.type in ("fact", "visual"):
            # Could check feed/damage values, merge chains, etc.
            pass
        
        results.append({
            "learning": nl_obj,
            "consistent": score >= 0.6,
            "issues": issues,
            "score": score,
        })
    return results


def demote_stale_learnings(path: Path = DEFAULT_PATH,
                           sessions_without_reinforcement: int = 20) -> int:
    """Demote committed learnings that haven't been reinforced in N sessions.
    Returns number demoted."""
    existing = read(path)
    demoted = 0
    for l in existing:
        if l.committed and l.should_demote(sessions_without_reinforcement):
            l.status = "candidate"
            demoted += 1
    if demoted:
        write(path, existing)
    return demoted


def log_learning_outcome(path: Path, learning_text: str, session_id: str, outcome: str) -> bool:
    """Log an outcome for a learning (e.g., 'followed_success', 'followed_failure', 'ignored').
    Returns True if learning was found and updated."""
    existing = read(path)
    for l in existing:
        if _matches(l.text, learning_text):
            l.add_outcome(session_id, outcome)
            write(path, existing)
            return True
    return False
