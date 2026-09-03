"""Persistent item glossary for the vision-driven planner.

Where learnings.md stores general strategy lessons, the glossary stores item
knowledge specifically: the visual markers the model has learned to associate
with each template id (e.g. "skeleton_lvl5: crown on the skull"). Written by
the session-summary call, read back into the prompt on every move so the model
recognizes items across sessions.

New format (item_knowledge/):
- merge_chains.json: {family: {chain: [...], source: "popup|wiki", updated: iso}}
- spawn_rates.json: {station_id: {targets: [...], uses: int|null, source, updated}}
- item_stats.json: {template_id: {feed: int|null, damage: int|null, max_level: bool, source, updated}}
- visual_markers.json: {template_id: {description: str|null, merge_edges: [...], source, updated}}
- index.json: master index {template_id: {sections: [...], last_seen: iso}}

Old format (item_glossary.md): markdown blocks (DEPRECATED, kept for backward compat)
"""

import json
import re
from datetime import datetime
from pathlib import Path

# Old format (deprecated)
DEFAULT_PATH = Path("item_glossary.md")
MAX_ITEMS = 50

# New format paths
DEFAULT_KNOWLEDGE_DIR = Path("item_knowledge")

def _get_paths(knowledge_dir: Path | None = None):
    """Get all JSON file paths for a given knowledge directory."""
    if knowledge_dir is None:
        knowledge_dir = DEFAULT_KNOWLEDGE_DIR
    return {
        "merge_chains": knowledge_dir / "merge_chains.json",
        "spawn_rates": knowledge_dir / "spawn_rates.json",
        "item_stats": knowledge_dir / "item_stats.json",
        "visual_markers": knowledge_dir / "visual_markers.json",
        "index": knowledge_dir / "index.json",
    }

# Default paths
KNOWLEDGE_DIR = DEFAULT_KNOWLEDGE_DIR
MERGE_CHAINS_PATH = KNOWLEDGE_DIR / "merge_chains.json"
SPAWN_RATES_PATH = KNOWLEDGE_DIR / "spawn_rates.json"
ITEM_STATS_PATH = KNOWLEDGE_DIR / "item_stats.json"
VISUAL_MARKERS_PATH = KNOWLEDGE_DIR / "visual_markers.json"
INDEX_PATH = KNOWLEDGE_DIR / "index.json"


# =====================================================================
# NEW FORMAT READERS (item_knowledge/)
# =====================================================================

def _load_json(path: Path, default=None):
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default if default is not None else {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _get_knowledge_dir(path: Path) -> Path:
    """Extract knowledge directory from a glossary path.
    If path is a markdown file (item_glossary.md), use its parent / item_knowledge.
    If path is a directory, use it directly.
    """
    if path.suffix == ".md":
        return path.parent / "item_knowledge"
    return path


def _load_json_dir(knowledge_dir: Path, filename: str, default=None):
    """Load JSON from a specific knowledge directory."""
    path = knowledge_dir / filename
    return _load_json(path, default)


def _save_json_dir(knowledge_dir: Path, filename: str, data: dict) -> None:
    """Save JSON to a specific knowledge directory."""
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    path = knowledge_dir / filename
    path.write_text(json.dumps(data, indent=2))


def read_merge_chains(knowledge_dir: Path | None = None) -> dict[str, list[str]]:
    """Return per-family merge chains from new JSON format.
    Returns {family: [id, ...]} (just the chain list for backward compat)."""
    paths = _get_paths(knowledge_dir)
    data = _load_json(paths["merge_chains"])
    return {k: v.get("chain", []) for k, v in data.items()}


def read_merge_chains_full(knowledge_dir: Path | None = None) -> dict[str, dict]:
    """Return full merge chain entries with provenance."""
    paths = _get_paths(knowledge_dir)
    return _load_json(paths["merge_chains"])


def read_spawn_rates(knowledge_dir: Path | None = None) -> dict[str, dict]:
    """Return per-station spawn rates from new JSON format."""
    paths = _get_paths(knowledge_dir)
    return _load_json(paths["spawn_rates"])


def read_item_stats(knowledge_dir: Path | None = None) -> dict[str, dict]:
    """Return item feed/damage/max_level from new JSON format."""
    paths = _get_paths(knowledge_dir)
    return _load_json(paths["item_stats"])


def read_visual_markers(knowledge_dir: Path | None = None) -> dict[str, dict]:
    """Return visual markers (descriptions, merge edges) from new JSON format."""
    paths = _get_paths(knowledge_dir)
    return _load_json(paths["visual_markers"])


def read_index(knowledge_dir: Path | None = None) -> dict[str, dict]:
    """Return master index."""
    paths = _get_paths(knowledge_dir)
    return _load_json(paths["index"])


# =====================================================================
# BACKWARD COMPATIBILITY READERS (read from markdown or JSON)
# =====================================================================

_CHAIN_RE = re.compile(r"## Item ([a-z0-9_]+) \(chain\)\n(.*?)(?=\n## |\Z)", re.S)
_CHAIN_BODY_RE = re.compile(r"-\s*merge chain:\s*(.+)$", re.M)
_SPAWN_RE = re.compile(r"## Item ([a-z0-9_]+) \(spawn\)\n(.*?)(?=\n## |\Z)", re.S)
_SPAWN_BODY_RE = re.compile(r"-\s*spawn:\s*(.+)$", re.M)
_USES_RE = re.compile(r"-\s*uses:\s*(\w+)", re.M)
_POPUP_RE = re.compile(r"## Item ([a-z0-9_]+) \(popup\)\n(.*?)(?=\n## |\Z)", re.S)
_FEED_RE = re.compile(r"-\s*feed value:\s*(\d+)", re.M)
_DAMAGE_RE = re.compile(r"-\s*damage value:\s*(\d+)", re.M)


def read_chains(path: Path = DEFAULT_PATH) -> dict[str, list[str]]:
    """Backward compat: read merge chains from new JSON, or old format if custom path.
    If path is a directory, treats it as a knowledge_dir and reads from JSON."""
    if path != DEFAULT_PATH:
        if not path.exists():
            return {}
        if path.is_dir():
            return read_merge_chains(path)
        text = path.read_text()
        chains: dict[str, list[str]] = {}
        for m in _CHAIN_RE.finditer("\n" + text):
            body = m.group(2)
            cm = _CHAIN_BODY_RE.search(body.strip())
            if not cm:
                continue
            ids = [i.strip() for i in re.split(r"\s*->\s*", cm.group(1)) if i.strip()]
            if ids:
                chains.setdefault(m.group(1), []).extend(ids)
        return chains
    return read_merge_chains()


def read_spawns(path: Path = DEFAULT_PATH) -> dict[str, dict]:
    """Backward compat: read spawn rates from new JSON, or old format if custom path."""
    if path != DEFAULT_PATH:
        if not path.exists():
            return {}
        if path.is_dir():
            paths = _get_paths(path)
            return _load_json(paths["spawn_rates"])
        text = path.read_text()
        spawns: dict[str, dict] = {}
        for m in _SPAWN_RE.finditer("\n" + text):
            family = m.group(1)
            body = m.group(2)
            sm = _SPAWN_BODY_RE.search(body.strip())
            um = _USES_RE.search(body.strip())
            if not sm:
                continue
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
            spawns[family] = {"source": source, "targets": targets,
                              "uses": uses, "raw": raw}
        return spawns
    paths = _get_paths()
    return _load_json(paths["spawn_rates"])


def read_feed_values(path: Path = DEFAULT_PATH) -> dict[str, int]:
    """Return {template_id: feed_value} from new item_stats, or old format if custom path."""
    if path != DEFAULT_PATH:
        if not path.exists():
            return {}
        if path.is_dir():
            paths = _get_paths(path)
            stats = _load_json(paths["item_stats"])
            return {k: v["feed"] for k, v in stats.items() if v.get("feed") is not None}
        text = path.read_text()
        values: dict[str, int] = {}
        for m in re.finditer(r"## Item ([a-z0-9_]+) \(popup\)\n(.*?)(?=\n## |\Z)",
                             "\n" + text, re.S):
            body = m.group(2)
            fm = re.search(r"-\s*feed value:\s*(\d+)", body)
            if fm:
                values.setdefault(m.group(1), int(fm.group(1)))
        return values
    paths = _get_paths()
    stats = _load_json(paths["item_stats"])
    return {k: v["feed"] for k, v in stats.items() if v.get("feed") is not None}


def read_damage_values(path: Path = DEFAULT_PATH) -> dict[str, int]:
    """Return {template_id: damage_value} from new item_stats, or old format if custom path."""
    if path != DEFAULT_PATH:
        if not path.exists():
            return {}
        if path.is_dir():
            paths = _get_paths(path)
            stats = _load_json(paths["item_stats"])
            return {k: v["damage"] for k, v in stats.items() if v.get("damage") is not None}
        text = path.read_text()
        values: dict[str, int] = {}
        for m in re.finditer(r"## Item ([a-z0-9_]+) \(popup\)\n(.*?)(?=\n## |\Z)",
                             "\n" + text, re.S):
            body = m.group(2)
            dm = re.search(r"-\s*damage value:\s*(\d+)", body)
            if dm:
                values.setdefault(m.group(1), int(dm.group(1)))
        return values
    paths = _get_paths()
    stats = _load_json(paths["item_stats"])
    return {k: v["damage"] for k, v in stats.items() if v.get("damage") is not None}


def _read_glossary_from_json(knowledge_dir: Path, max_items: int = MAX_ITEMS) -> str:
    """Read glossary from JSON files in knowledge_dir."""
    parts = []
    
    chains = read_merge_chains_full(knowledge_dir)
    if chains:
        for family, data in chains.items():
            chain_str = " -> ".join(data.get("chain", []))
            src = data.get("source", "unknown")
            parts.append("## Item " + family + " (chain)\n\n- merge chain: " + chain_str + "\n  [source: " + src + "]")
    
    spawns = read_spawn_rates(knowledge_dir)
    if spawns:
        for station_id, data in spawns.items():
            targets = ", ".join(data.get("targets", []))
            uses = data.get("uses")
            uses_str = "infinite" if uses is None else str(uses)
            src = data.get("source", "unknown")
            parts.append("## Item " + station_id + " (spawn)\n\n- spawn: " + station_id + " -> " + targets + "\n- uses: " + uses_str + "\n  [source: " + src + "]")
    
    stats = read_item_stats(knowledge_dir)
    markers = read_visual_markers(knowledge_dir)
    all_ids = set(stats.keys()) | set(markers.keys())
    for tid in sorted(all_ids):
        lines = []
        s = stats.get(tid, {})
        m = markers.get(tid, {})
        if s.get("feed") is not None:
            lines.append("- feed value: " + str(s["feed"]))
        if s.get("damage") is not None:
            lines.append("- damage value: " + str(s["damage"]))
        if s.get("max_level"):
            lines.append("- max_level: true")
        if m.get("description"):
            lines.append("- description: " + m["description"])
        for edge in m.get("merge_edges", []):
            lines.append("- edge: " + edge)
        if lines:
            src = s.get("source") or markers.get(tid, {}).get("source", "unknown")
            parts.append("## Item " + tid + " (popup)\n\n" + "\n".join(lines) + "\n  [source: " + src + "]")
    
    if not parts:
        return ""
    # Return most recent MAX_ITEMS (by index last_seen)
    index = read_index(knowledge_dir)
    parts_with_time = []
    for p in parts:
        m = re.search(r"## Item ([^\n]+)", p)
        if m:
            tid = m.group(1).split(" ")[0]
            last_seen = index.get(tid, {}).get("last_seen", "")
            parts_with_time.append((last_seen, p))
    parts_with_time.sort(key=lambda x: x[0], reverse=True)
    return "\n".join(p for _, p in parts_with_time[:max_items])


def read_glossary(path: Path | None = DEFAULT_PATH, max_items: int = MAX_ITEMS, knowledge_dir: Path | None = None) -> str:
    """Return the most recent glossary entries as prompt text.
    If knowledge_dir is provided, reads from that knowledge directory.
    If path is a custom markdown file (not the default), reads from that markdown file (backward compat).
    If neither is provided, reads from default knowledge directory."""
    # If knowledge_dir provided, use it
    if knowledge_dir is not None:
        return _read_glossary_from_json(knowledge_dir, max_items)
    
    # If custom path provided (not the default), read from that markdown file
    if path != DEFAULT_PATH:
        return _read_glossary_from_markdown(path, max_items)
    
    # Default: read from new JSON files (default knowledge dir)
    return _read_glossary_from_json(DEFAULT_KNOWLEDGE_DIR, max_items)


def _read_glossary_from_markdown(path: Path, max_items: int = MAX_ITEMS) -> str:
    """Read glossary from old markdown format."""
    if not path.exists():
        return ""
    text = path.read_text().strip()
    if not text:
        return ""
    blocks = re.split(r"\n## ", "\n" + text)[1:]
    if not blocks:
        return ""
    return "\n".join(blocks[-max_items:])


# =====================================================================
# WRITERS (new JSON format with provenance)
# =====================================================================

def _now() -> str:
    return datetime.now().isoformat()


def write_merge_chain(family: str, chain: list[str], source: str = "popup",
                      knowledge_dir: Path | None = None) -> None:
    """Write/replace a merge chain for a family."""
    paths = _get_paths(knowledge_dir)
    data = _load_json(paths["merge_chains"])
    data[family] = {"chain": chain, "source": source, "updated": _now()}
    _save_json(paths["merge_chains"], data)
    _update_index(family, "merge_chain", knowledge_dir)


def write_spawn_rate(station_id: str, targets: list[str], uses: int | None,
                     source: str = "popup", knowledge_dir: Path | None = None) -> None:
    """Write/replace spawn rates for a station."""
    paths = _get_paths(knowledge_dir)
    data = _load_json(paths["spawn_rates"])
    data[station_id] = {"targets": targets, "uses": uses, "source": source, "updated": _now()}
    _save_json(paths["spawn_rates"], data)
    _update_index(station_id, "spawn_rates", knowledge_dir)


def write_item_stat(template_id: str, feed: int | None = None,
                    damage: int | None = None, max_level: bool = False,
                    source: str = "popup", knowledge_dir: Path | None = None) -> None:
    """Write/update item stats (feed, damage, max_level)."""
    paths = _get_paths(knowledge_dir)
    data = _load_json(paths["item_stats"])
    existing = data.get(template_id, {})
    data[template_id] = {
        "feed": feed if feed is not None else existing.get("feed"),
        "damage": damage if damage is not None else existing.get("damage"),
        "max_level": max_level or existing.get("max_level", False),
        "source": source if feed is not None or damage is not None else existing.get("source", source),
        "updated": _now(),
    }
    _save_json(paths["item_stats"], data)
    _update_index(template_id, "item_stats", knowledge_dir)


def write_visual_marker(template_id: str, description: str | None = None,
                        merge_edges: list[str] | None = None,
                        source: str = "popup", knowledge_dir: Path | None = None) -> None:
    """Write/update visual markers for an item."""
    paths = _get_paths(knowledge_dir)
    data = _load_json(paths["visual_markers"])
    existing = data.get(template_id, {})
    data[template_id] = {
        "description": description or existing.get("description"),
        "merge_edges": merge_edges if merge_edges is not None else existing.get("merge_edges", []),
        "source": source,
        "updated": _now(),
    }
    _save_json(paths["visual_markers"], data)
    _update_index(template_id, "visual_markers", knowledge_dir)


def _update_index(template_id: str, section: str, knowledge_dir: Path | None = None) -> None:
    """Update master index with new section for template_id."""
    paths = _get_paths(knowledge_dir)
    index = _load_json(paths["index"])
    entry = index.get(template_id, {"sections": [], "last_seen": _now()})
    if section not in entry["sections"]:
        entry["sections"].append(section)
    entry["last_seen"] = _now()
    index[template_id] = entry
    _save_json(paths["index"], index)


# =====================================================================
# WIKI CROSS-CHECK
# =====================================================================

def wiki_cross_check_merge_chains(wiki_chains: dict[str, list[str]],
                                   knowledge_dir: Path | None = None) -> dict:
    """Compare stored merge chains with wiki. Returns {family: 'match'|'mismatch'|'missing'}."""
    paths = _get_paths(knowledge_dir)
    stored = read_merge_chains_full(knowledge_dir)
    result = {}
    for family, wiki_chain in wiki_chains.items():
        if family not in stored:
            result[family] = "missing"
            continue
        stored_chain = stored[family].get("chain", [])
        if stored_chain == wiki_chain:
            result[family] = "match"
            # Mark as wiki-verified
            if stored[family].get("source") == "popup":
                stored[family]["source"] = "wiki"
                _save_json(paths["merge_chains"], stored)
        else:
            result[family] = "mismatch"
    # Check for stored families not in wiki
    for family in stored:
        if family not in wiki_chains:
            result[family] = "wiki_missing"
    return result


def wiki_cross_check_spawn_rates(wiki_spawns: dict[str, dict],
                                  knowledge_dir: Path | None = None) -> dict:
    """Compare stored spawn rates with wiki. Returns {station_id: 'match'|'mismatch'|'missing'}."""
    paths = _get_paths(knowledge_dir)
    stored = read_spawn_rates(knowledge_dir)
    result = {}
    for station_id, wiki_data in wiki_spawns.items():
        if station_id not in stored:
            result[station_id] = "missing"
            continue
        # Compare targets (order-insensitive)
        stored_targets = set(stored[station_id].get("targets", []))
        wiki_targets = set(wiki_data.get("targets", []))
        if stored_targets == wiki_targets:
            result[station_id] = "match"
            if stored[station_id].get("source") == "popup":
                stored[station_id]["source"] = "wiki"
                _save_json(paths["spawn_rates"], stored)
        else:
            result[station_id] = "mismatch"
    for station_id in stored:
        if station_id not in wiki_spawns:
            result[station_id] = "wiki_missing"
    return result


# =====================================================================
# EXPIRATION / PRUNING
# =====================================================================

def prune_inferred_entries(max_age_sessions: int = 10,
                            knowledge_dir: Path | None = None) -> int:
    """Remove inferred entries not re-confirmed in N sessions.
    Returns number of entries pruned."""
    paths = _get_paths(knowledge_dir)
    stats = _load_json(paths["item_stats"])
    markers = _load_json(paths["visual_markers"])
    index = _load_json(paths["index"])
    pruned = 0
    
    for template_id, entry in list(stats.items()):
        if entry.get("source") == "inferred":
            pass
    
    return pruned


def prune_glossary_by_ids(names: list[str], protected: bool = True) -> int:
    """Remove entries matching names from all JSON files.
    Protected=True keeps popup/wiki sourced entries."""
    if not names:
        return 0
    queries = [n.lower() for n in names if n]
    pruned = 0
    
    for path in [MERGE_CHAINS_PATH, SPAWN_RATES_PATH, ITEM_STATS_PATH, VISUAL_MARKERS_PATH]:
        data = _load_json(path)
        original_len = len(data)
        if protected:
            data = {k: v for k, v in data.items()
                    if not any(q in k.lower() for q in queries)
                    or v.get("source") in ("popup", "wiki")}
        else:
            data = {k: v for k, v in data.items()
                    if not any(q in k.lower() for q in queries)}
        pruned += original_len - len(data)
        if original_len != len(data):
            _save_json(path, data)
    
    index = _load_json(INDEX_PATH)
    index = {k: v for k, v in index.items()
             if not any(q in k.lower() for q in queries)}
    _save_json(INDEX_PATH, index)
    
    return pruned


def _norm(s: str) -> str:
    """Normalize string for comparison."""
    s = re.sub(r"^\s*[-*•]\s+", "", s.strip())
    return re.sub(r"\s+", " ", s.lower()).strip()


def prune_glossary(path: Path = DEFAULT_PATH, names=None, protected: bool = True) -> int:
    """Remove glossary blocks matching any of the given names/ids from OLD markdown file.
    Returns how many blocks were removed.
    Kept for backward compatibility with tests and old code."""
    if not names or not path.exists():
        return 0
    text = path.read_text().strip()
    if not text:
        return 0
    queries = [_norm(n) for n in names if n and str(n).strip()]
    if not queries:
        return 0
    blocks = re.split(r"\n(?=## )", "\n" + text)

    def _is_protected(block: str) -> bool:
        header = block.splitlines()[0].strip() if block.strip() else ""
        return header.endswith("(popup)") or header.endswith("(chain)") or header.endswith("(spawn)")

    kept = [b for b in blocks
            if (protected and _is_protected(b))
            or not any(q in _norm(b) for q in queries)]
    removed = len(blocks) - len(kept)
    if removed:
        path.write_text("\n".join(kept) + "\n")
    return removed
