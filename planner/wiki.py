"""NecroMerger wiki (wiki.gg) lookup for the LLM planner.

Read-only fetch of the NecroMerger wiki's MediaWiki API (stdlib only). The LLM
calls `lookup_wiki` to fact-check item behavior / merge chains it is unsure
about. The game's own UI (popup reads) remains the source of truth — the wiki
is a cross-check, never an override of a verified popup read.

The Fandom mirror (necromerger.fandom.com) returns 403 to scripted fetches, so
this targets https://necromerger.wiki.gg/ instead.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request

WIKI_BASE = "https://necromerger.wiki.gg/api.php"
USER_AGENT = "NecroMergerBot/1.0 (autonomous agent research)"
TIMEOUT = 15.0
MAX_SEARCH_RESULTS = 3
DEFAULT_MAX_CHARS = 8000   # return the whole article (pages are small, ~0.5-3.5k)
MAX_FACTCHECK_TOPICS = 6      # wiki pages fetched per learning fact-check

# Words with no page of their own that models append to item queries
# ("zombie merge chain", "how does the grave work"). Stripped when the phrasal
# query fails to resolve, so the item noun is tried on its own.
_QUERY_NOISE = frozenset("""
    a an the and or of to for with in on at how what which why when do does
    did is are was were be can could should would i you we my me mine your
    need want know about from work works working chain chains merge merges
    merging combine combines combining feed feeding level levels item items
    mechanic mechanics progress unlock feature action actions
""".split())

# Item/station nouns with real wiki pages. Used to pick topics out of a
# learning's text for fact-checking (generic verbs like "merge"/"feed" have no
# useful page and are deliberately excluded).
FACTCHECK_KEYWORDS = (
    "grave", "skeleton", "zombie", "bone", "ribcage", "rotten flesh",
    "severed hand", "manapool", "manapot", "necromerger", "devourer",
    "chest", "rune", "mana", "food", "station", "champion", "peasant",
    "eye monster", "floating skull", "ancient tablet", "lectern", "relic",
    "lair", "spells",
)

# Tail words that indicate a phrasal NON-NOUN query ("zombie merge chain",
# "how does the grave work", "skeleton feed value"). These are never page
# titles on their own; when the query is `<noun> <tail...>` we resolve the
# leading noun directly instead of probing the nonexistent phrasal title.
_PHRASAL_TAIL = frozenset("""
    merge chain merge-chain chain merges merging combine combines combining
    upgrade upgrades level levels how what which why when does do did work
    works feed feeding value values stat stats info
""".split())

_CACHE: dict[str, dict] = {}


def _api(params: dict) -> dict:
    params = dict(params)
    params["format"] = "json"
    params["formatversion"] = "2"
    url = WIKI_BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise WikiError(f"wiki request failed: {exc.reason}") from exc


class WikiError(RuntimeError):
    pass


def _strip_templates(text: str) -> str:
    """Remove `{{...}}` blocks (tolerant of wiki's over-closed `}}}}`) and
    `[[...]]` links. Templates are dropped; links become their display text
    (`[[Grave|Graves]]` -> `Graves`, `[[Devourer]]` -> `Devourer`).
    Files/images are removed.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("{{", i):
            depth = 1
            j = i + 2
            while j < n and depth:
                if text.startswith("{{", j):
                    depth += 1
                    j += 2
                elif text.startswith("}}", j):
                    depth = max(0, depth - 1)
                    j += 2
                else:
                    j += 1
            i = j
        elif text.startswith("[[", i):
            j = text.find("]]", i + 2)
            if j == -1:
                i += 2
                continue
            inner = text[i + 2 : j]
            lower = inner.lower()
            if lower.startswith("file:") or lower.startswith("image:"):
                i = j + 2
                continue
            out.append(inner.split("|")[-1])
            i = j + 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _clean_wikitext(wikitext: str) -> str:
    text = wikitext
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = _strip_templates(text)
    text = re.sub(r"\{\{|\}\}", " ", text)           # stray braces from over-closed templates
    text = re.sub(r"'''''|'''|''", "", text)
    text = re.sub(r"<[^>]+>", " ", text)          # stray tags
    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text)
    text = text.replace("= =", "==")              # handle malformed headers
    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\={2,}\s*(.*?)\s*\={2,}$", r"\1:", line)  # header -> label
        line = re.sub(r"^[*#:;]+", "", line)      # list markers
        if not line:
            continue
        if line in ("{|", "|}") or line.startswith("|-"):
            continue                             # table structure markers
        line = line.strip("|")
        line = re.sub(r"^!+\s*", "", line)       # table header marker
        line = line.strip()
        if not line:
            continue
        # drop pure-syntax table cells (digits/seps, no words)
        if len(line) < 40 and not any(ch.isalpha() for ch in line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def search(query: str, limit: int = MAX_SEARCH_RESULTS) -> list[str]:
    """Return up to `limit` page titles matching `query`."""
    data = _api({"action": "query", "list": "search",
                 "srsearch": query, "srlimit": limit})
    return [s.get("title", "") for s in data.get("query", {}).get("search", [])
            if s.get("title")]


def fetch_page(title: str) -> str:
    """Fetch a page's wikitext and return it cleaned to plain text."""
    data = _api({"action": "parse", "page": title, "prop": "wikitext",
                 "redirects": "1"})
    parse = data.get("parse", {})
    wikitext = parse.get("wikitext", "")
    if not wikitext:
        raise WikiError(f"no wikitext for page '{title}'")
    text = _clean_wikitext(wikitext)
    # A "#REDIRECT [[X]]" that survived (e.g. double redirects) still needs
    # chasing; fetch the target once.
    m = re.match(r"^REDIRECT\s+(\S.*)", text, flags=re.I)
    if m:
        target = m.group(1).strip()
        return fetch_page(target)
    return text


def _page_title(query: str) -> str | None:
    """Return the canonical page title if a page with this exact title exists,
    else None (MediaWiki resolves redirects)."""
    data = _api({"action": "query", "titles": query, "prop": "info"})
    for page in data.get("query", {}).get("pages", []):
        if not page.get("missing"):
            return page.get("title") or query
    return None


def topics_for(texts, extra_names=()) -> list[str]:
    """Pick fact-check topics (item/station nouns) out of learning texts.

    A topic is any known keyword (or glossary-derived item base name) that
    appears in the text. Returns a bounded, de-duplicated list.
    """
    names = set(FACTCHECK_KEYWORDS)
    for n in extra_names:
        base = re.sub(r"_lvl\d+.*$", "", str(n or "").strip().lower())
        if base:
            names.add(base)
    found: list[str] = []
    for text in texts:
        t = (text or "").lower()
        for kw in names:
            if kw in t and kw not in found:
                found.append(kw)
                if len(found) >= MAX_FACTCHECK_TOPICS:
                    return found
    return found


def fetch_topic_texts(topics, max_chars: int = 800) -> dict[str, str]:
    """Fetch cleaned wiki text for each topic (cached); skips missing pages.
    Returns {page_title: text}."""
    out: dict[str, str] = {}
    for topic in topics:
        try:
            result = json.loads(lookup(topic, max_chars=max_chars))
        except Exception:
            continue
        if result.get("found") and result.get("text"):
            out[result["page"]] = result["text"]
    return out


def _lookup_variants(query: str) -> list[str]:
    """Progressive query simplifications so phrasal queries ('zombie merge
    chain', 'how does the grave work') still resolve to the item page.

    Returns candidates from most-specific to least-specific: the original
    query, the phrase minus noise words, then each remaining meaningful token
    (so 'zombie merge chain' -> 'zombie merge chain', 'zombie', 'zombie').
    """
    variants = [query]
    tokens = [t for t in re.findall(r"[A-Za-z]+", query)
              if t.lower() not in _QUERY_NOISE]
    phrase = " ".join(tokens)
    if phrase and phrase != query:
        variants.append(phrase)
    for t in tokens:
        if t not in variants:
            variants.append(t)
    return variants


def _resolve_title(query: str) -> str | None:
    """Resolve a query to a canonical page title.

    A `<noun>`-style title match is preferred (item names are the common
    case). Phrasal queries like "zombie merge chain" are deflected to the
    leading item noun first — the "merge chain" tail is behavior, not a page,
    so we don't waste a probe on a nonexistent "zombie merge chain" title.
    Falls back to noise-stripped variants, individual tokens, then a keyword
    search. Returns None only when nothing plausible matches.
    """
    # Fast-path: deflect "<noun> <tail...>" phrasal queries to the leading
    # noun (e.g. "zombie merge chain" -> "zombie") before any exact-title
    # probe. This is the common LLM pattern and avoids a guaranteed-miss
    # wikitext fetch on the phrasal string.
    noun = _leading_item_noun(query)
    if noun is not None:
        title = _page_title(noun)
        if title:
            return title
    for variant in _lookup_variants(query):
        title = _page_title(variant)
        if title:
            return title
    for variant in _lookup_variants(query):
        hits = search(variant)
        if hits:
            return hits[0]
    return None


def _leading_item_noun(query: str) -> str | None:
    """If `query` looks like '<item> merge chain' / '<item> ... <tail>' return
    the leading searchable item noun (real keyword or glossary-derived base),
    else None. Falls back to simply stripping the phrasal tail off a query so
    'zombie merge chain' -> 'zombie' without assuming keyword membership."""
    q = (query or "").strip().lower()
    if not q:
        return None
    # Recognize a leading known keyword (longest match first).
    for kw in sorted(FACTCHECK_KEYWORDS, key=len, reverse=True):
        if not q.startswith(kw):
            continue
        tail = q[len(kw):].strip()
        # Require a phrasal tail (not a bare noun query like "zombie").
        if tail and any(t in _PHRASAL_TAIL for t in re.findall(r"[a-z+]+", tail)):
            return kw
    # Fallback: strip a trailing "merge chain" / "chain" tail from any query.
    m = re.match(r"^([a-z][a-z0-9 +-]*?)\s+((?:merge|chain|merge\s+chain|upgrade|info|feed|stat)(?:\s+\w+){0,2})$", q)
    if m:
        lead = m.group(1).strip()
        tokens = lead.split()
        if len(tokens) <= 2:
            return lead
    return None


def lookup(query: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Search the wiki for `query` and return the top page as a JSON string.

    An exact page-title match is preferred (item names are the common case);
    phrasal queries are simplified ('zombie merge chain' -> 'Zombie') until
    something resolves, then the top search hit is used. The result is bounded
    to `max_chars` of cleaned text (default: the whole article). Cached
    per-query for the session.
    """
    query = (query or "").strip()
    if not query:
        return json.dumps({"found": False, "error": "empty query"})
    cache_key = query.lower()
    if cache_key in _CACHE:
        return json.dumps(_CACHE[cache_key])
    try:
        title = _resolve_title(query)
        if title is None:
            result = {"found": False, "error": "no wiki results", "query": query}
        else:
            text = fetch_page(title)
            result = {"found": True, "page": title,
                      "other_pages": [t for t in search(query) if t != title][:2],
                      "text": text[:max_chars],
                      "merge_info": _extract_merge_info(text)}
    except WikiError as exc:
        result = {"found": False, "error": str(exc), "query": query}
    _CACHE[cache_key] = result
    return json.dumps(result)


def _extract_merge_info(text: str) -> str:
    """Surface the item page's own merge-relevant lines as a compact block.

    The wiki article for an item carries its merge behavior (component spawns,
    "merge to summon X", "max-level used in Y") in scattered free text. Pulling
    those lines out explicitly makes the returned page sufficient on its own —
    the model does not need a separate "<noun> merge chain" query because this
    field already carries the merge facts present on the article.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        low = line.lower()
        if not any(k in low for k in ("merge", "summon", "spawn", "component",
                                      "produce", "use", "level", "make",
                                      "combined", "tier")):
            continue
        s = " ".join(line.split())
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= 12:
            break
    return "\n".join(out)
