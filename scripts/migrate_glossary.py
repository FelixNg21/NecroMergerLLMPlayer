#!/usr/bin/env python3
"""Migrate item_glossary.md -> item_knowledge/ JSON files.

Run: python scripts/migrate_glossary.py [--dry-run]
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLOSSARY_PATH = ROOT / "item_glossary.md"
OUT_DIR = ROOT / "item_knowledge"

MERGE_CHAINS_PATH = OUT_DIR / "merge_chains.json"
SPAWN_RATES_PATH = OUT_DIR / "spawn_rates.json"
ITEM_STATS_PATH = OUT_DIR / "item_stats.json"
VISUAL_MARKERS_PATH = OUT_DIR / "visual_markers.json"
INDEX_PATH = OUT_DIR / "index.json"


def _now() -> str:
    return datetime.now().isoformat()


def parse_glossary(text: str) -> list[dict]:
    """Parse glossary into list of {title, body, type} blocks."""
    if not text.strip():
        return []
    # Split on ## Item headers
    blocks = re.split(r"\n(?=## Item )", "\n" + text.strip())
    out = []
    for b in blocks:
        lines = b.lstrip("\n").splitlines()
        if not lines or not lines[0].startswith("## Item "):
            continue
        header = lines[0][8:].strip()  # remove "## Item "
        body = "\n".join(lines[1:]).strip()
        
        # Classify block type
        block_type = "bare"
        if header.endswith("(chain)"):
            block_type = "chain"
            family = header[:-7].strip()
        elif header.endswith("(spawn)"):
            block_type = "spawn"
            family = header[:-7].strip()
        elif header.endswith("(popup)"):
            block_type = "popup"
            template_id = header[:-7].strip()
        else:
            # Could be bare item name or something else
            template_id = header
            
        out.append({
            "header": header,
            "type": block_type,
            "family": family if block_type in ("chain", "spawn") else None,
            "template_id": template_id if block_type == "popup" else None,
            "body": body,
        })
    return out


def parse_chain_block(body: str) -> list[str] | None:
    """Extract merge chain from (chain) block body."""
    m = re.search(r"-\s*merge chain:\s*(.+)$", body, re.M)
    if not m:
        return None
    ids = [i.strip() for i in re.split(r"\s*->\s*", m.group(1)) if i.strip()]
    return ids if ids else None


def parse_spawn_block(body: str) -> dict | None:
    """Extract spawn info from (spawn) block body."""
    sm = re.search(r"-\s*spawn:\s*(.+)$", body, re.M)
    um = re.search(r"-\s*uses:\s*(\w+)", body, re.M)
    if not sm:
        return None
    raw = sm.group(1).strip()
    parts = re.split(r"\s*->\s*", raw, maxsplit=1)
    source = parts[0].strip() if parts else ""
    targets = []
    if len(parts) > 1:
        for t in re.split(r",\s*", parts[1]):
            tid = re.sub(r"\s*\(.*?\)", "", t).strip()
            if tid:
                targets.append(tid)
    uses = None
    if um:
        u = um.group(1).lower()
        if u.isdigit():
            uses = int(u)
        elif u in ("infinite", "inf", "-"):
            uses = None
    return {"source": source, "targets": targets, "uses": uses, "raw": raw}


def parse_popup_block(body: str) -> dict:
    """Extract facts from (popup) block body."""
    facts = {
        "feed_value": None,
        "damage_value": None,
        "max_level": False,
        "description": None,
        "merge_edges": [],
    }
    
    # feed value
    fm = re.search(r"-\s*feed value:\s*(\d+)", body)
    if fm:
        facts["feed_value"] = int(fm.group(1))
    
    # damage value
    dm = re.search(r"-\s*damage value:\s*(\d+)", body)
    if dm:
        facts["damage_value"] = int(dm.group(1))
    
    # max_level
    if "max_level: true" in body or "max_level: True" in body:
        facts["max_level"] = True
    
    # description
    dm = re.search(r"-\s*description:\s*(.+)$", body, re.M)
    if dm:
        facts["description"] = dm.group(1).strip()
    
    # merge edges
    for m in re.finditer(r"-\s*(?:merge|edge):\s*(.+)$", body, re.M):
        facts["merge_edges"].append(m.group(1).strip())
    
    return facts


def main():
    dry_run = "--dry-run" in sys.argv
    
    if not GLOSSARY_PATH.exists():
        print(f"Glossary not found: {GLOSSARY_PATH}")
        return 1
    
    text = GLOSSARY_PATH.read_text()
    blocks = parse_glossary(text)
    
    print(f"Parsed {len(blocks)} glossary blocks")
    
    # Initialize output structures
    merge_chains = {}
    spawn_rates = {}
    item_stats = {}
    visual_markers = {}
    index = {}
    
    for block in blocks:
        btype = block["type"]
        body = block["body"]
        header = block["header"]
        
        if btype == "chain":
            family = block["family"]
            ids = parse_chain_block(body)
            if ids:
                merge_chains[family] = {
                    "chain": ids,
                    "source": "popup",
                    "updated": _now(),
                }
                # Update index for each id
                for tid in ids:
                    index.setdefault(tid, {"sections": [], "last_seen": _now()})
                    if "merge_chain" not in index[tid]["sections"]:
                        index[tid]["sections"].append("merge_chain")
        
        elif btype == "spawn":
            family = block["family"]
            spawn_data = parse_spawn_block(body)
            if spawn_data:
                # Extract level from family if present (e.g., "grave_lvl1")
                # The family here is the station family, but spawn blocks are per-level
                # We'll store by station_id
                station_id = spawn_data["source"]
                spawn_rates[station_id] = {
                    "targets": spawn_data["targets"],
                    "uses": spawn_data["uses"],
                    "source": "popup",
                    "updated": _now(),
                }
                # Index
                index.setdefault(station_id, {"sections": [], "last_seen": _now()})
                if "spawn_rates" not in index[station_id]["sections"]:
                    index[station_id]["sections"].append("spawn_rates")
        
        elif btype == "popup":
            template_id = block["template_id"]
            facts = parse_popup_block(body)
            
            # Item stats
            if facts["feed_value"] is not None or facts["damage_value"] is not None or facts["max_level"]:
                item_stats[template_id] = {
                    "feed": facts["feed_value"],
                    "damage": facts["damage_value"],
                    "max_level": facts["max_level"],
                    "source": "popup",
                    "updated": _now(),
                }
            
            # Visual markers
            if facts["description"] or facts["merge_edges"]:
                visual_markers[template_id] = {
                    "description": facts["description"],
                    "merge_edges": facts["merge_edges"],
                    "source": "popup",
                    "updated": _now(),
                }
            
            # Index
            index.setdefault(template_id, {"sections": [], "last_seen": _now()})
            if facts["feed_value"] is not None or facts["damage_value"] is not None or facts["max_level"]:
                if "item_stats" not in index[template_id]["sections"]:
                    index[template_id]["sections"].append("item_stats")
            if facts["description"] or facts["merge_edges"]:
                if "visual_markers" not in index[template_id]["sections"]:
                    index[template_id]["sections"].append("visual_markers")
        
        else:
            # Bare block - could be anything, try to parse as popup-style
            facts = parse_popup_block(body)
            template_id = block["template_id"] or block["header"]
            if template_id and (facts["feed_value"] or facts["damage_value"] or facts["max_level"] or facts["description"]):
                item_stats[template_id] = {
                    "feed": facts["feed_value"],
                    "damage": facts["damage_value"],
                    "max_level": facts["max_level"],
                    "source": "inferred",
                    "updated": _now(),
                }
                if facts["description"] or facts["merge_edges"]:
                    visual_markers[template_id] = {
                        "description": facts["description"],
                        "merge_edges": facts["merge_edges"],
                        "source": "inferred",
                        "updated": _now(),
                    }
                index.setdefault(template_id, {"sections": [], "last_seen": _now()})
                if "item_stats" not in index[template_id]["sections"]:
                    index[template_id]["sections"].append("item_stats")
                if "visual_markers" not in index[template_id]["sections"]:
                    index[template_id]["sections"].append("visual_markers")
    
    print(f"  merge_chains: {len(merge_chains)} families")
    print(f"  spawn_rates: {len(spawn_rates)} stations")
    print(f"  item_stats: {len(item_stats)} items")
    print(f"  visual_markers: {len(visual_markers)} items")
    print(f"  index entries: {len(index)}")
    
    if dry_run:
        print("\n--- DRY RUN: would write ---")
        print(f"  {MERGE_CHAINS_PATH}")
        print(f"  {SPAWN_RATES_PATH}")
        print(f"  {ITEM_STATS_PATH}")
        print(f"  {VISUAL_MARKERS_PATH}")
        print(f"  {INDEX_PATH}")
        return 0
    
    # Write files
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for path, data in [
        (MERGE_CHAINS_PATH, merge_chains),
        (SPAWN_RATES_PATH, spawn_rates),
        (ITEM_STATS_PATH, item_stats),
        (VISUAL_MARKERS_PATH, visual_markers),
        (INDEX_PATH, index),
    ]:
        path.write_text(json.dumps(data, indent=2))
        print(f"  Wrote {path}")
    
    print("\nMigration complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
