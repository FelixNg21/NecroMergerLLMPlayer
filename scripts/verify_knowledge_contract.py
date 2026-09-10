"""Knowledge-backend contract (SQLite migration, step 1).

Defines the exact behavior both backends must satisfy: a fixed workload
run against a backend namespace. Today only the file backend exists; the
SQLite change adds `sqlite_backend` and runs the IDENTICAL workload
against both, diffing results. Backend-specific paths are bound once in
the constructor so the workload is fully backend-agnostic (learnings
takes `path`, glossary takes `knowledge_dir` — the namespace hides that).

The workload pins the semantics that encode past incidents, not just
I/O: append dedup (`_matches`), write_item_stat merge-preservation
(feed/damage/makes never clobber each other), inferred-only pruning,
and (popup)/(chain)/(spawn) glossary protection.
"""
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = []
FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def file_backend(tmpdir: str) -> SimpleNamespace:
    """Backend namespace bound to temp paths (never the real store)."""
    from planner import glossary as G
    from planner import learnings as L

    root = Path(tmpdir)
    lpath = root / "learnings.md"
    kdir = root / "knowledge"
    kdir.mkdir(parents=True, exist_ok=True)
    gpath = root / "item_glossary.md"

    def seed(table: str, key: str, entry: dict) -> None:
        """Raw-seed one store entry (test scaffolding, not production API)."""
        if table == "item_stats":
            data = G._load_json(G._get_paths(kdir)["item_stats"])
            data[key] = entry
            G._save_json(G._get_paths(kdir)["item_stats"], data)
        else:
            raise ValueError(f"no raw seed for {table}")

    return SimpleNamespace(
        # learnings
        append_learning=lambda texts, title=None: L.append_learning(lpath, texts, title),
        read=lambda: L.read(lpath),
        read_learnings=lambda: L.read_learnings(lpath),
        write=lambda ls: L.write(lpath, ls),
        confirm_learning=lambda texts: L.confirm_learning(lpath, texts),
        prune_learning=lambda texts: L.prune_learning(lpath, texts),
        derate_learning=lambda texts=None: L.derate_learning(lpath, texts),
        demote_stale=lambda n=20: L.demote_stale_learnings(lpath, n),
        log_outcome=lambda text, sid, out: L.log_learning_outcome(lpath, text, sid, out),
        # glossary
        read_item_stats=lambda: G.read_item_stats(kdir),
        write_item_stat=lambda tid, **kw: G.write_item_stat(tid, knowledge_dir=kdir, **kw),
        read_merge_chains=lambda: G.read_merge_chains(kdir),
        write_merge_chain=lambda fam, chain: G.write_merge_chain(fam, chain, knowledge_dir=kdir),
        read_spawn_rates=lambda: G.read_spawn_rates(kdir),
        write_spawn_rate=lambda sid, targets, uses: G.write_spawn_rate(
            sid, targets, uses, knowledge_dir=kdir),
        read_feed_values=lambda: G.read_feed_values(kdir),
        read_damage_values=lambda: G.read_damage_values(kdir),
        read_visual_markers=lambda: G.read_visual_markers(kdir),
        write_visual_marker=lambda tid, desc: G.write_visual_marker(tid, desc, knowledge_dir=kdir),
        prune_inferred=lambda days=14: G.prune_inferred_entries(days, kdir),
        prune_glossary=lambda names, protected=True: G.prune_glossary(gpath, names, protected),
        seed_glossary_text=lambda text: gpath.write_text(text),
        read_glossary_text=lambda: gpath.read_text() if gpath.exists() else "",
        seed=seed,
    )


def sqlite_backend(tmpdir: str) -> SimpleNamespace:
    """Backend namespace over a temp knowledge.db (same contract as files)."""
    from planner import knowledge_db as K

    root = Path(tmpdir)
    db = root / "knowledge.db"
    K.init_db(db)

    def seed(table: str, key: str, entry: dict) -> None:
        if table == "item_stats":
            import sqlite3
            conn = sqlite3.connect(str(db))
            try:
                conn.execute(
                    "INSERT INTO item_stats (template_id, feed, damage,"
                    " max_level, makes, source, updated)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(template_id) DO UPDATE SET feed=excluded.feed,"
                    " damage=excluded.damage, max_level=excluded.max_level,"
                    " makes=excluded.makes, source=excluded.source,"
                    " updated=excluded.updated",
                    (key, entry.get("feed"),
                     entry.get("damage"),
                     int(bool(entry.get("max_level", False))),
                     json.dumps(entry["makes"]) if entry.get("makes") is not None else None,
                     entry.get("source", "popup"),
                     entry.get("updated", "")))
                conn.commit()
            finally:
                conn.close()
        else:
            raise ValueError(f"no raw seed for {table}")

    return SimpleNamespace(
        append_learning=lambda texts, title=None: K.db_append_learning(db, texts, title),
        read=lambda: K.db_read(db),
        read_learnings=lambda: K.db_read_learnings(db),
        write=lambda ls: K.db_write(db, ls),
        confirm_learning=lambda texts: K.db_confirm_learning(db, texts),
        prune_learning=lambda texts: K.db_prune_learning(db, texts),
        derate_learning=lambda texts=None: K.db_derate_learning(db, texts),
        demote_stale=lambda n=20: K.db_demote_stale_learnings(db, n),
        log_outcome=lambda text, sid, out: K.db_log_learning_outcome(db, text, sid, out),
        read_item_stats=lambda: K.db_read_item_stats(db),
        write_item_stat=lambda tid, **kw: K.db_write_item_stat(db, tid, **kw),
        read_merge_chains=lambda: K.db_read_merge_chains(db),
        write_merge_chain=lambda fam, chain: K.db_write_merge_chain(db, fam, chain),
        read_spawn_rates=lambda: K.db_read_spawn_rates(db),
        write_spawn_rate=lambda sid, targets, uses: K.db_write_spawn_rate(
            db, sid, targets, uses),
        read_feed_values=lambda: K.db_read_feed_values(db),
        read_damage_values=lambda: K.db_read_damage_values(db),
        read_visual_markers=lambda: K.db_read_visual_markers(db),
        write_visual_marker=lambda tid, desc: K.db_write_visual_marker(db, tid, desc),
        prune_inferred=lambda days=14: K.db_prune_inferred_entries(db, days),
        prune_glossary=lambda names, protected=True: K.db_prune_glossary(db, names, protected),
        seed_glossary_text=lambda text: K.db_seed_glossary_text(db, text),
        read_glossary_text=lambda: K.db_read_glossary_text(db),
        seed=seed,
    )


def _strip_updated(stats: dict) -> dict:
    return {k: {kk: vv for kk, vv in v.items() if kk != "updated"}
            for k, v in stats.items()}


def workload(be: SimpleNamespace, tag: str) -> None:
    p = f"{tag}: "
    # L1: append new + exact-duplicate dedup.
    check(p + "append new returns 1",
          be.append_learning([{"text": "cupboards cost slime per tap",
                               "type": "fact", "confidence": 1.0}]) == 1)
    check(p + "exact duplicate returns 0 (dedup)",
          be.append_learning(["cupboards cost slime per tap"]) == 0)
    check(p + "distinct text returns 1",
          be.append_learning(["graves cost mana per tap"]) == 1)
    # L2: confirm promotes at threshold 2.
    check(p + "confirm once bumps but stays candidate",
          be.confirm_learning(["cupboards cost slime"]) == 1
          and not be.read()[0].committed)
    be.confirm_learning(["cupboards cost slime"])
    check(p + "second confirm promotes to committed",
          be.read()[0].committed)
    # L3: read_learnings sections.
    txt = be.read_learnings()
    check(p + "read_learnings has committed + tentative sections",
          "Committed" in txt and "Tentative" in txt, txt[:60])
    # L4: outcome log found/missing.
    check(p + "log outcome True on match",
          be.log_outcome("cupboards cost slime per tap", "s1", "followed_success") is True)
    check(p + "log outcome False on miss",
          be.log_outcome("no such learning anywhere", "s1", "ignored") is False)
    # L5: prune removes.
    check(p + "prune removes match",
          be.prune_learning(["graves cost mana"]) == 1
          and len(be.read()) == 1)
    # L6: write/read round-trip preserves fields.
    from planner.learnings import Learning
    be.write([Learning(text="round trip", type="pattern", confidence=0.8,
                       status="committed", title="t", confirmed=3)])
    back = be.read()
    check(p + "write/read round-trip",
          len(back) == 1 and back[0].text == "round trip"
          and back[0].committed and back[0].confirmed == 3
          and abs(back[0].confidence - 0.8) < 1e-9)
    # L7: derate keeps store parseable.
    be.append_learning(["shaky claim about drops"])
    be.derate_learning(["shaky claim"])
    check(p + "derate leaves parseable store",
          isinstance(be.read(), list))
    # G1: write_item_stat merge-preservation (feed/damage/makes never clobber).
    be.write_item_stat("eye_lvl1", feed=15)
    be.write_item_stat("eye_lvl1", damage=25)
    be.write_item_stat("eye_lvl1", makes={"mana": 2})
    st = _strip_updated(be.read_item_stats())["eye_lvl1"]
    check(p + "stat writes merge (feed+damage+makes)",
          st.get("feed") == 15 and st.get("damage") == 25
          and st.get("makes") == {"mana": 2}, str(st))
    # G2: chains + rates round-trip.
    be.write_merge_chain("eyefam", ["eyeball", "eyeinajar"])
    check(p + "chain round-trip",
          be.read_merge_chains().get("eyefam") == ["eyeball", "eyeinajar"])
    be.write_spawn_rate("cup1", ["eyeball"], 5)
    check(p + "spawn rate round-trip",
          be.read_spawn_rates().get("cup1", {}).get("targets") == ["eyeball"])
    # G3: feed/damage projections.
    check(p + "feed values project written feed",
          be.read_feed_values().get("eye_lvl1") == 15)
    check(p + "damage values project written damage",
          be.read_damage_values().get("eye_lvl1") == 25)
    # G4: inferred-only pruning (seeded raw).
    be.seed("item_stats", "old_guess", {"feed": 1, "source": "inferred",
                                        "updated": "2000-01-01T00:00:00"})
    be.seed("item_stats", "fresh_guess", {"feed": 1, "source": "inferred",
                                          "updated": "2999-01-01T00:00:00"})
    be.seed("item_stats", "old_popup", {"feed": 2, "source": "popup",
                                        "updated": "2000-01-01T00:00:00"})
    check(p + "prune removes only stale inferred",
          be.prune_inferred(14) == 1
          and "old_guess" not in be.read_item_stats()
          and "fresh_guess" in be.read_item_stats()
          and "old_popup" in be.read_item_stats())
    # G5: glossary protection ((popup) survives, plain goes).
    be.seed_glossary_text("## eyeball (popup)\nfeed value: 15\n\n## junkitem\nsome text\n")
    check(p + "prune_glossary keeps protected, drops plain",
          be.prune_glossary(["eyeball", "junkitem"]) == 1
          and "eyeball" in be.read_glossary_text()
          and "junkitem" not in be.read_glossary_text())
    # G6: visual markers round-trip.
    be.write_visual_marker("eye_lvl1", "red pupil")
    check(p + "visual marker round-trip",
          "eye_lvl1" in be.read_visual_markers())


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        workload(file_backend(td), "file")
    with tempfile.TemporaryDirectory() as td:
        workload(sqlite_backend(td), "sqlite")
    # Backend-agnostic pure check: consistency helper takes Learning objects.
    from planner.learnings import Learning, verify_learning_consistency
    committed = [Learning(text="cupboards cost slime", type="fact",
                          status="committed")]
    out = verify_learning_consistency(
        [Learning(text="cupboards cost slime", type="fact")],
        committed, {"eye_lvl1": {"feed": 15}})
    check("verify_learning_consistency returns list", isinstance(out, list))
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
