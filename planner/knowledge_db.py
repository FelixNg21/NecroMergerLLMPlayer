"""SQLite knowledge backend (migration step 2).

Implements the backend contract (see scripts/verify_knowledge_contract.py)
against a single `knowledge.db`: one table per former JSON file, plus
`learnings` (former learnings.md) and `kv_store` (singletons like
board_dims, which finally get a typed home). The manual `index.json` is
deliberately NOT migrated — queries replace the hand-maintained index.

Behavioral parity rules (enforced by the contract test, not by sharing
code — the file backend stays untouched until cutover):
- ordering: file order == insertion order == ORDER BY id;
- dedup/matching reuses `_matches` from planner.learnings;
- prompt text reuses `format_learnings` (prompt rendering is behavior);
- timestamps use the same ISO `_now()` shape as the file backend;
- every mutator commits in one transaction (crash-safe by construction).

Single-process use only (the bot loop is single-threaded): one short
connection per call, WAL mode, parameterized queries throughout.
"""
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from planner.learnings import (
    COMMIT_CONFIRMATIONS,
    MAX_LEARNINGS,
    MAX_NEGATIVE,
    Learning,
    _matches,
    format_learnings,
)
from planner.glossary import _norm as _glossary_norm

DB_FILENAME = "knowledge.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS learnings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate',
    type TEXT NOT NULL DEFAULT 'pattern',
    confidence REAL NOT NULL DEFAULT 0.7,
    confirmed INTEGER NOT NULL DEFAULT 0,
    negative INTEGER NOT NULL DEFAULT 0,
    title TEXT NOT NULL DEFAULT '',
    outcome_log TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS item_stats (
    template_id TEXT PRIMARY KEY,
    feed INTEGER,
    damage INTEGER,
    max_level INTEGER NOT NULL DEFAULT 0,
    makes TEXT,
    source TEXT NOT NULL DEFAULT 'popup',
    updated TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS merge_chains (
    family TEXT PRIMARY KEY,
    chain TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT 'popup',
    updated TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS spawn_rates (
    station_id TEXT PRIMARY KEY,
    targets TEXT NOT NULL DEFAULT '[]',
    uses INTEGER,
    source TEXT NOT NULL DEFAULT 'popup',
    updated TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS visual_markers (
    template_id TEXT PRIMARY KEY,
    description TEXT,
    merge_edges TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT 'popup',
    updated TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS glossary_blocks (
    title TEXT PRIMARY KEY,
    body TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS kv_store (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT '{}',
    updated TEXT NOT NULL DEFAULT ''
);
"""


def _now() -> str:
    return datetime.now().isoformat()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    """Create schema (idempotent). Never leaves a torn file: SQLite DDL is
    transactional, and the file backend stays live until cutover anyway."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _connect(p) as conn:
        conn.executescript(SCHEMA)


def _row_to_learning(row: sqlite3.Row) -> Learning:
    try:
        outcomes = json.loads(row["outcome_log"] or "[]")
    except (json.JSONDecodeError, TypeError):
        outcomes = []
    return Learning(
        text=row["text"],
        status=row["status"],
        type=row["type"],
        confidence=float(row["confidence"]),
        confirmed=int(row["confirmed"]),
        negative=int(row["negative"]),
        title=row["title"] or "",
        outcome_log=outcomes if isinstance(outcomes, list) else [],
    )


def _db_read_learnings(conn: sqlite3.Connection) -> list[Learning]:
    return [_row_to_learning(r)
            for r in conn.execute("SELECT * FROM learnings ORDER BY id")]


def _db_write_learnings(conn: sqlite3.Connection, learnings: list[Learning]) -> None:
    conn.execute("DELETE FROM learnings")
    conn.executemany(
        "INSERT INTO learnings (text, status, type, confidence, confirmed,"
        " negative, title, outcome_log) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(l.text, l.status, l.type, l.confidence, l.confirmed,
          l.negative, l.title or "", json.dumps(l.outcome_log or []))
         for l in learnings],
    )


# ------------------------------------------------------------------
# learnings (mirrors planner/learnings.py semantics exactly)
# ------------------------------------------------------------------

def db_read(db_path: str | Path) -> list[Learning]:
    with _connect(db_path) as conn:
        return _db_read_learnings(conn)


def db_write(db_path: str | Path, learnings: list[Learning]) -> None:
    with _connect(db_path) as conn:
        with conn:
            _db_write_learnings(conn, learnings)


def db_read_learnings(db_path: str | Path,
                      max_learnings: int = MAX_LEARNINGS) -> str:
    with _connect(db_path) as conn:
        return format_learnings(_db_read_learnings(conn), max_learnings)


def _coerce_text(t) -> tuple[str, str, float]:
    if isinstance(t, Learning):
        return t.text.strip(), t.type, t.confidence
    if isinstance(t, dict):
        return (str(t.get("text", "")).strip(), t.get("type", "pattern"),
                float(t.get("confidence", 0.7)))
    return str(t).strip(), "pattern", 0.7


def db_append_learning(db_path: str | Path, texts, title: str | None = None) -> int:
    if isinstance(texts, str):
        texts = [texts]
    with _connect(db_path) as conn:
        with conn:
            existing = _db_read_learnings(conn)
            added = 0
            for t in texts:
                text, ltype, confidence = _coerce_text(t)
                if not text:
                    continue
                if any(_matches(e.text, text) for e in existing):
                    continue
                existing.append(Learning(text=text, type=ltype,
                                         confidence=confidence,
                                         title=title or ""))
                added += 1
            if added:
                _db_write_learnings(conn, existing)
            return added


def db_confirm_learning(db_path: str | Path, texts) -> int:
    with _connect(db_path) as conn:
        with conn:
            existing = _db_read_learnings(conn)
            n = 0
            for l in existing:
                if l.committed or not any(_matches(l.text, q) for q in texts):
                    continue
                l.confirmed += 1
                n += 1
                if l.confirmed >= COMMIT_CONFIRMATIONS:
                    l.status = "committed"
            if n:
                _db_write_learnings(conn, existing)
            return n


def db_prune_learning(db_path: str | Path, texts) -> int:
    with _connect(db_path) as conn:
        with conn:
            existing = _db_read_learnings(conn)
            kept = [l for l in existing
                    if not any(_matches(l.text, q) for q in texts)]
            removed = len(existing) - len(kept)
            if removed:
                _db_write_learnings(conn, kept)
            return removed


def db_derate_learning(db_path: str | Path, texts=None) -> int:
    with _connect(db_path) as conn:
        with conn:
            existing = _db_read_learnings(conn)
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
                            if not (not l.committed
                                    and l.negative >= MAX_NEGATIVE)]
            if changed:
                _db_write_learnings(conn, existing)
            return dropped


def db_demote_stale_learnings(db_path: str | Path,
                              sessions_without_reinforcement: int = 20) -> int:
    with _connect(db_path) as conn:
        with conn:
            existing = _db_read_learnings(conn)
            demoted = 0
            for l in existing:
                if l.committed and l.should_demote(sessions_without_reinforcement):
                    l.status = "candidate"
                    demoted += 1
            if demoted:
                _db_write_learnings(conn, existing)
            return demoted


def db_log_learning_outcome(db_path: str | Path, learning_text: str,
                            session_id: str, outcome: str) -> bool:
    with _connect(db_path) as conn:
        with conn:
            existing = _db_read_learnings(conn)
            for l in existing:
                if _matches(l.text, learning_text):
                    l.add_outcome(session_id, outcome)
                    _db_write_learnings(conn, existing)
                    return True
            return False


# ------------------------------------------------------------------
# item_stats
# ------------------------------------------------------------------

def _stat_row(row: sqlite3.Row) -> dict:
    try:
        makes = json.loads(row["makes"]) if row["makes"] else None
    except (json.JSONDecodeError, TypeError):
        makes = None
    d = {"feed": row["feed"], "damage": row["damage"],
         "max_level": bool(row["max_level"]),
         "source": row["source"], "updated": row["updated"]}
    if makes is not None:
        d["makes"] = makes
    return d


def db_read_item_stats(db_path: str | Path) -> dict[str, dict]:
    with _connect(db_path) as conn:
        return {r["template_id"]: _stat_row(r)
                for r in conn.execute("SELECT * FROM item_stats")}


def db_write_item_stat(db_path: str | Path, template_id: str,
                       feed: int | None = None, damage: int | None = None,
                       max_level: bool = False, makes: dict | None = None,
                       source: str = "popup") -> None:
    """Same merge semantics as the file backend: only provided fields
    overwrite; feed/damage/makes are never clobbered by a partial write."""
    with _connect(db_path) as conn:
        with conn:
            cur = conn.execute("SELECT * FROM item_stats WHERE template_id = ?",
                               (template_id,))
            row = cur.fetchone()
            existing = _stat_row(row) if row else {}
            if feed is None:
                feed = existing.get("feed")
            if damage is None:
                damage = existing.get("damage")
            if makes is None:
                makes = existing.get("makes")
            conn.execute(
                "INSERT INTO item_stats (template_id, feed, damage,"
                " max_level, makes, source, updated)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(template_id) DO UPDATE SET feed=excluded.feed,"
                " damage=excluded.damage, max_level=excluded.max_level,"
                " makes=excluded.makes, source=excluded.source,"
                " updated=excluded.updated",
                (template_id, feed, damage,
                 int(bool(max_level or existing.get("max_level", False))),
                 json.dumps(makes) if makes is not None else None,
                 source if feed is not None or damage is not None
                 else existing.get("source", source),
                 _now()))


def db_read_feed_values(db_path: str | Path) -> dict[str, int]:
    with _connect(db_path) as conn:
        return {r["template_id"]: r["feed"] for r in conn.execute(
            "SELECT template_id, feed FROM item_stats WHERE feed IS NOT NULL")}


def db_read_damage_values(db_path: str | Path) -> dict[str, int]:
    with _connect(db_path) as conn:
        return {r["template_id"]: r["damage"] for r in conn.execute(
            "SELECT template_id, damage FROM item_stats WHERE damage IS NOT NULL")}


# ------------------------------------------------------------------
# merge_chains / spawn_rates / visual_markers
# ------------------------------------------------------------------

def db_read_merge_chains(db_path: str | Path) -> dict[str, list[str]]:
    with _connect(db_path) as conn:
        return {r["family"]: (json.loads(r["chain"]) if r["chain"] else [])
                for r in conn.execute("SELECT family, chain FROM merge_chains")}


def db_read_merge_chains_full(db_path: str | Path) -> dict[str, dict]:
    with _connect(db_path) as conn:
        return {r["family"]: {"chain": json.loads(r["chain"]) if r["chain"] else [],
                              "source": r["source"], "updated": r["updated"]}
                for r in conn.execute("SELECT * FROM merge_chains")}


def db_write_merge_chain(db_path: str | Path, family: str, chain: list[str],
                         source: str = "popup") -> None:
    with _connect(db_path) as conn:
        with conn:
            conn.execute(
                "INSERT INTO merge_chains (family, chain, source, updated)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(family) DO UPDATE SET chain=excluded.chain,"
                " source=excluded.source, updated=excluded.updated",
                (family, json.dumps(chain), source, _now()))


def db_read_spawn_rates(db_path: str | Path) -> dict[str, dict]:
    with _connect(db_path) as conn:
        out = {}
        for r in conn.execute("SELECT * FROM spawn_rates"):
            out[r["station_id"]] = {
                "targets": json.loads(r["targets"]) if r["targets"] else [],
                "uses": r["uses"], "source": r["source"],
                "updated": r["updated"]}
        return out


def db_write_spawn_rate(db_path: str | Path, station_id: str,
                        targets: list[str], uses: int | None,
                        source: str = "popup") -> None:
    with _connect(db_path) as conn:
        with conn:
            conn.execute(
                "INSERT INTO spawn_rates (station_id, targets, uses, source,"
                " updated) VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(station_id) DO UPDATE SET targets=excluded.targets,"
                " uses=excluded.uses, source=excluded.source,"
                " updated=excluded.updated",
                (station_id, json.dumps(targets), uses, source, _now()))


def db_read_visual_markers(db_path: str | Path) -> dict[str, dict]:
    with _connect(db_path) as conn:
        out = {}
        for r in conn.execute("SELECT * FROM visual_markers"):
            try:
                edges = json.loads(r["merge_edges"]) if r["merge_edges"] else []
            except (json.JSONDecodeError, TypeError):
                edges = []
            out[r["template_id"]] = {"description": r["description"],
                                     "merge_edges": edges,
                                     "source": r["source"],
                                     "updated": r["updated"]}
        return out


def db_write_visual_marker(db_path: str | Path, template_id: str,
                           description: str | None = None,
                           merge_edges: list[str] | None = None,
                           source: str = "popup") -> None:
    with _connect(db_path) as conn:
        with conn:
            cur = conn.execute("SELECT description, merge_edges FROM"
                               " visual_markers WHERE template_id = ?",
                               (template_id,))
            row = cur.fetchone()
            if description is None and row:
                description = row["description"]
            if merge_edges is None:
                try:
                    merge_edges = json.loads(row["merge_edges"]) if row and row["merge_edges"] else []
                except (json.JSONDecodeError, TypeError):
                    merge_edges = []
            conn.execute(
                "INSERT INTO visual_markers (template_id, description,"
                " merge_edges, source, updated) VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(template_id) DO UPDATE SET"
                " description=excluded.description,"
                " merge_edges=excluded.merge_edges, source=excluded.source,"
                " updated=excluded.updated",
                (template_id, description, json.dumps(merge_edges or []),
                 source, _now()))


# ------------------------------------------------------------------
# pruning (mirrors file-backend scope: stats + markers only; chains and
# spawns have their own lifecycle and are never touched here; the manual
# index.json is not migrated — queries replace it)
# ------------------------------------------------------------------

def _stale(source: str | None, updated: str | None,
           cutoff: str, max_age_days: int) -> bool:
    if source != "inferred":
        return False
    if not updated:
        return True
    try:
        age = (datetime.now() - datetime.fromisoformat(updated)).days
        return age >= max_age_days
    except (ValueError, TypeError):
        return True


def db_prune_inferred_entries(db_path: str | Path,
                              max_age_days: int = 14) -> int:
    # File backend uses a datetime cutoff; comparing ISO strings in SQLite
    # would work but Python-side keeps the semantics (and the no-timestamp
    # rule) byte-identical.
    with _connect(db_path) as conn:
        with conn:
            pruned = 0
            for table, key in (("item_stats", "template_id"),
                               ("visual_markers", "template_id")):
                rows = conn.execute(f"SELECT {key}, source, updated"
                                    f" FROM {table}").fetchall()
                for r in rows:
                    if _stale(r["source"], r["updated"], "", max_age_days):
                        conn.execute(f"DELETE FROM {table} WHERE {key} = ?",
                                     (r[key],))
                        pruned += 1
            return pruned


# ------------------------------------------------------------------
# glossary markdown blocks (former item_glossary.md)
# ------------------------------------------------------------------

def db_seed_glossary_text(db_path: str | Path, text: str) -> None:
    """Parse raw markdown into title/body rows (same split as the file prune)."""
    with _connect(db_path) as conn:
        with conn:
            blocks = re.split(r"\n(?=## )", "\n" + (text or "").strip())
            for b in blocks:
                b = b.strip("\n")
                if not b.strip():
                    continue
                title = b.splitlines()[0].strip().lstrip("#").strip()
                if not title:
                    continue
                conn.execute(
                    "INSERT INTO glossary_blocks (title, body) VALUES (?, ?)"
                    " ON CONFLICT(title) DO UPDATE SET body=excluded.body",
                    (title, b))


def db_read_glossary_text(db_path: str | Path) -> str:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT body FROM glossary_blocks"
                            " ORDER BY rowid").fetchall()
        return "\n".join(r["body"] for r in rows)


def _is_protected(title: str, body: str) -> bool:
    header = (body.splitlines()[0].strip() if body.strip() else "")
    if not header:
        header = title
    return header.endswith("(popup)") or header.endswith("(chain)") \
        or header.endswith("(spawn)")


def db_prune_glossary(db_path: str | Path, names, protected: bool = True) -> int:
    """Same rule as the file prune: drop blocks matching names unless the
    block is protected and protection is on."""
    queries = [_glossary_norm(n) for n in (names or []) if n and str(n).strip()]
    if not queries:
        return 0
    with _connect(db_path) as conn:
        with conn:
            rows = conn.execute("SELECT title, body FROM glossary_blocks").fetchall()
            removed = 0
            for r in rows:
                if protected and _is_protected(r["title"], r["body"] or ""):
                    continue
                if any(q in _glossary_norm(r["body"] or "") for q in queries):
                    conn.execute("DELETE FROM glossary_blocks WHERE title = ?",
                                 (r["title"],))
                    removed += 1
            return removed


# ------------------------------------------------------------------
# kv_store (singletons: board_dims today, more later)
# ------------------------------------------------------------------

def db_kv_get(db_path: str | Path, key: str, default=None):
    with _connect(db_path) as conn:
        cur = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return default


def db_kv_set(db_path: str | Path, key: str, value) -> None:
    with _connect(db_path) as conn:
        with conn:
            conn.execute(
                "INSERT INTO kv_store (key, value, updated) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
                " updated=excluded.updated",
                (key, json.dumps(value), _now()))
