"""Migrate the live knowledge store (JSON + markdown) into knowledge.db.

Reads the real store — item_knowledge/*.json, learnings.md,
item_glossary.md, board_dims.json — and inserts every entry through the
SQLite backend. The file backend stays live until cutover (step 3), so
this is safe to run any time; re-running replaces rows idempotently.

Usage: .venv/bin/python scripts/migrate_knowledge_to_sqlite.py [--check]
  --check: migrate into a temp DB and compare counts only (writes nothing).
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def migrate(db_path: Path, knowledge_dir: Path, learnings_path: Path,
            glossary_path: Path, dims_path: Path) -> dict[str, int]:
    from planner import knowledge_db as K
    from planner import learnings as L

    counts: dict[str, int] = {}

    def _load(p: Path, default=None):
        if not p.exists():
            return default if default is not None else {}
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return default if default is not None else {}

    stats = _load(knowledge_dir / "item_stats.json")
    for tid, e in stats.items():
        K.db_write_item_stat(
            db_path, tid, feed=e.get("feed"), damage=e.get("damage"),
            max_level=bool(e.get("max_level", False)),
            makes=e.get("makes"), source=e.get("source", "popup"))
        # Preserve the original timestamp (write path stamps now()).
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("UPDATE item_stats SET updated = ? WHERE template_id = ?",
                         (e.get("updated", ""), tid))
            conn.commit()
        finally:
            conn.close()
    counts["item_stats"] = len(stats)

    chains = _load(knowledge_dir / "merge_chains.json")
    for fam, e in chains.items():
        chain = e.get("chain", []) if isinstance(e, dict) else e
        src = e.get("source", "popup") if isinstance(e, dict) else "popup"
        K.db_write_merge_chain(db_path, fam, chain, source=src)
    counts["merge_chains"] = len(chains)

    spawns = _load(knowledge_dir / "spawn_rates.json")
    for sid, e in spawns.items():
        e = e if isinstance(e, dict) else {}
        K.db_write_spawn_rate(db_path, sid, e.get("targets", []),
                              e.get("uses"), source=e.get("source", "popup"))
    counts["spawn_rates"] = len(spawns)

    markers = _load(knowledge_dir / "visual_markers.json")
    for tid, e in markers.items():
        e = e if isinstance(e, dict) else {}
        K.db_write_visual_marker(db_path, tid, e.get("description"),
                                 e.get("merge_edges"), source=e.get("source", "popup"))
    counts["visual_markers"] = len(markers)

    learned = L.read(learnings_path)
    K.db_write(db_path, learned)
    counts["learnings"] = len(learned)

    if glossary_path.exists():
        K.db_seed_glossary_text(db_path, glossary_path.read_text())
    import sqlite3 as _s
    conn = _s.connect(str(db_path))
    try:
        n = conn.execute("SELECT COUNT(*) FROM glossary_blocks").fetchone()[0]
    finally:
        conn.close()
    counts["glossary_blocks"] = n

    if dims_path.exists():
        try:
            dims = json.loads(dims_path.read_text())
            K.db_kv_set(db_path, "board_dims",
                        {"rows": dims.get("rows"), "cols": dims.get("cols")})
            counts["board_dims"] = 1
        except json.JSONDecodeError:
            counts["board_dims"] = 0
    else:
        counts["board_dims"] = 0
    return counts


def main() -> int:
    from planner import knowledge_db as K

    check_only = "--check" in sys.argv
    knowledge_dir = ROOT / "item_knowledge"
    if check_only:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "knowledge.db"
            K.init_db(db)
            counts = migrate(db, knowledge_dir, ROOT / "learnings.md",
                             ROOT / "item_glossary.md",
                             knowledge_dir / "board_dims.json")
        print("check counts:", counts)
        return 0
    db = knowledge_dir / K.DB_FILENAME
    K.init_db(db)
    counts = migrate(db, knowledge_dir, ROOT / "learnings.md",
                     ROOT / "item_glossary.md",
                     knowledge_dir / "board_dims.json")
    print(f"migrated -> {db}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
