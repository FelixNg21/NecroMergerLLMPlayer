"""LLM-assisted board grid geometry.

The board layout changes with game state (it grows as the Devourer levels up),
so the fixed 5x3 calibration constants are no longer reliable. Geometry is read
once at session start by combining what each source is actually good at:

- the vision LLM counts the board's ROWS/COLS reliably from the screenshot
  (verified: 4x3 and 5x3 boards both read correctly), but its PIXEL
  coordinates are unreliable (it reported the board top at y~300 for both);
- the floor-mask blob gives the pixel numbers: the board's width is
  `cols * cell_px` (no horizontal margin) and its bottom edge sits on the
  lair floor, so `cell_px = blob_width/cols` and `origin_y = blob_bottom -
  rows*cell_px` come out within ~5px of the true values;
- the template-anchor local search below then nudges origin/cell_px to the
  match-quality peak (cells are 220-280px and sprites fill most of a cell, so
  taps and template matching tolerate tens of pixels of error).

The mouth is a lair-fixed element above the board; it is NOT trusted from the
LLM (y-coords unreliable) — the calibrated constant stays unless re-measured.
"""

import base64
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from planner.atomic import atomic_write_text
from planner.llm_client import LLMError
from vision.classifier import TemplateClassifier
from vision.grid import (
    FALLBACK_GEOMETRY,
    OCCUPANCY_THRESHOLD,
    Cell,
    GridGeometry,
    build_cells,
    occupancy_score,
)

class NoFloorBlobError(Exception):
    """The frame has no lair floor blob — it is not a bare lair board."""


JPEG_QUALITY = 85
MAX_DIM = 1536          # long-side cap for the geometry screenshot (~half res is plenty)
MAX_ANCHORS = 8         # occupied cells used as template-match anchors
ANCHOR_MIN_SCORE = 0.4  # an anchor must have a confident-enough match to trust its id
# Refinement is coordinate descent: pitch first (its error propagates into the
# blob-derived origin_y as rows*error), then origin. Windows are tuned to the
# blob's known accuracy: pitch ~2%, origin ~25px. The improvement guard below
# rejects flat-plateau noise (sparse boards score identically across the window).
REFINE_PITCH_WIN = 6    # +/- px around blob pitch (2px steps)
REFINE_ORIGIN_WIN = 25  # +/- px around blob origin (5px steps)
REFINE_ROUNDS = 2       # descent rounds (each axis re-searched)
REFINE_MIN_GAIN = 0.05  # absolute improvement required to adopt
REFINE_REL_GAIN = 0.005 # ...or 0.5% of the start score, whichever is larger

# Lair area where the board sits (fixed for the pinned 1280x2856 emulator).
LAIR_BAND = (150, 1150, 1130, 2380)   # x0, y0, x1, y1
FLOOR_LOW = (90, 40, 50)              # board/floor tiles' HSV range
FLOOR_HIGH = (160, 110, 200)
CLOSE_KERNEL = 61

# Floor-blob validation: the board floor is a WIDE, SHORT strip sitting on
# the BOTTOM edge of the board region. A menu column, station or dialog that
# aliases the floor HSV range is typically NARROW or TALL, so it must not be
# chosen as the board's floor even when it beats the true floor on area. These
# thresholds are LOSSY on purpose: they only reject components that cannot be
# the board floor, never the real one. `blob_bottom` feeds origin_y and
# `blob_width` feeds cell_px, so a wrong largest-component poisons the read.
BLOB_MIN_BAND_FRAC = 0.30  # width must be >= this fraction of the band width
BLOB_MAX_ASPECT = 1.6      # height must be < this multiple of width (wide, short)
# The board floor's bottom edge is the floor line, which sits in the LOWER
# portion of the lair band (a dialog covering the lower band is not the floor).
BLOB_MIN_BOTTOM_FRAC = 0.35

# Vertical-extent row discriminator. The floor blob's width feeds cell_px and
# `origin_y = blob_bottom - rows*cell_px`, so `rows` is NOT determined by the
# blob at all (4x4 and 5x4 derive identical cell_px and identical template
# scores — a tie the LLM's unreliable count decides). The physical ground truth
# is the board's HEIGHT: the tile-grid top edge and the lair floor line. Solving
# `rows = (blob_bottom - board_top) / cell_px` disambiguates square-ish boards
# (5x4 spans 5 cells; 4x4 only 4). Measured on a live 5x4: top edge ~1201,
# floor line ~2342, cell_px 228 -> (2342-1201)/228 = 5.01. Candidates whose
# implied rows fit the measured span are preferred over the template-objective
# tie. These tune the top-edge search only.
TOP_EDGE_EDGE_THR = 40     # vertical-gradient magnitude for a "line" pixel
TOP_EDGE_MIN_SPAN = 200    # a candidate top edge must span >= this many px
TOP_EDGE_SEARCH_FRAC = 0.75  # search the top 75% of the lair band for the edge
VERT_FIT_TOL = 0.45        # rows-fit residual (cells); candidates within this win a bonus

# The mouth is lair-fixed above the board. Only override the calibrated value
# when the LLM's reading lands in a plausible box around it (y-coords from the
# vision model are otherwise unreliable).
MOUTH_PLAUSIBLE = (500, 700, 780, 1100)   # x0, y0, x1, y1

GEOMETRY_SYSTEM_PROMPT = """You are a calibration tool for a NecroMerger game bot. The screenshot is a full 1280x2856 pixel emulator frame. The BOARD is a regular grid of square cells in the middle of the screen.

Output ONLY a JSON object, no other text:
{"rows": R, "cols": C, "cell_px": P, "origin_x": X, "origin_y": Y, "mouth_x": MX, "mouth_y": MY}

- rows/cols: the board's grid dimensions. A board with 5 rows x 3 columns has "rows": 5, "cols": 3. Count the actual tile rows/columns you can see.
- cell_px: the edge length of one board cell in PIXELS. Cells are typically 220-270px. The screen is 1280 pixels wide, so a 3-column board of ~226px cells spans roughly 700px of it — use the screen width as a ruler.
- origin_x/origin_y: your best estimate of the PIXEL coordinate of the TOP-LEFT CORNER of the grid's top-left cell.
- mouth_x/mouth_y: the PIXEL coordinate of the center of the Devourer's open mouth (a wide open maw ABOVE the board).

The rows/cols count is what matters most. Estimate the pixel numbers as best you can; they will be cross-checked."""


def _encode_frame(frame, max_dim: int | None = MAX_DIM) -> str:
    h, w = frame.shape[:2]
    if max_dim:
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _parse_json(text: str) -> dict:
    raw = text.strip()
    if not raw.startswith("{"):
        raw = '{"rows":' + raw
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in geometry reply: {text!r}")
    return json.loads(raw[start : end + 1])


# Pinned board dims (Sep 7): the lair grid only changes on board-expansion
# feat rewards, so its dims are deployment state, not a per-session vision
# read. The first run without the file detects (LLM prior or full
# enumeration) and self-seeds it; later runs load it and skip the startup
# LLM round entirely. A stale pin (board grew since) is caught by the
# blob-plausibility check, which falls back to full enumeration and
# re-seeds. Pinning also kills the biggest template-crop variance source:
# a 4x4-vs-5x4 flip moves origin_y by a whole cell height, shifting every
# banked crop. Deleting the file forces a fresh detect on next start.
BOARD_DIMS_PATH = Path(__file__).resolve().parent.parent / "item_knowledge" / "board_dims.json"


def read_board_dims(path: str | Path = BOARD_DIMS_PATH) -> tuple[int, int] | None:
    """Pinned (rows, cols), or None when missing/invalid (caller detects)."""
    try:
        data = json.loads(Path(path).read_text())
        rows, cols = int(data.get("rows", 0)), int(data.get("cols", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    except json.JSONDecodeError:
        return None
    return (rows, cols) if _valid_dims(rows, cols) else None


def write_board_dims(rows: int, cols: int, source: str = "detected",
                     path: str | Path = BOARD_DIMS_PATH) -> bool:
    """Persist detected dims for future runs. Never raises."""
    try:
        if not _valid_dims(int(rows), int(cols)):
            return False
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(p, json.dumps({"rows": int(rows), "cols": int(cols),
                                          "source": source,
                                          "updated": datetime.now().isoformat()}) + "\n")
        return True
    except (OSError, ValueError, TypeError):
        return False


def _valid_dims(rows: int, cols: int) -> bool:
    return 3 <= rows <= 6 and 2 <= cols <= 4


def _floor_components(frame) -> list[tuple[int, int, int, int, int]]:
    """(left, top, width, height, area) of each floor-colored component in the
    lair band, after morphological CLOSE (which bridges gaps between sprites
    so the board floor is usually one solid component)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FLOOR_LOW, FLOOR_HIGH)
    x0, y0, x1, y1 = LAIR_BAND
    sub = np.zeros_like(mask)
    sub[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE,
                              np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8))
    n, _lab, stats, _cents = cv2.connectedComponentsWithStats(closed)
    out = []
    for i in range(1, n):
        out.append((int(stats[i, cv2.CC_STAT_LEFT]),
                    int(stats[i, cv2.CC_STAT_TOP]),
                    int(stats[i, cv2.CC_STAT_WIDTH]),
                    int(stats[i, cv2.CC_STAT_HEIGHT]),
                    int(stats[i, cv2.CC_STAT_AREA])))
    return out


def _is_boardlike(comp, band_w, band_h) -> bool:
    """True if a floor component could be the board floor (wide, short, and
    its bottom edge sits low in the lair band)."""
    _l, _t, w, h, _a = comp
    if w < BLOB_MIN_BAND_FRAC * band_w:
        return False
    if h >= BLOB_MAX_ASPECT * w:
        return False
    _l, _t, _w, h, _a = comp
    if (_t + h) < BLOB_MIN_BOTTOM_FRAC * band_h:
        return False
    return True


def _blob_bbox(frame) -> tuple[int, int, int, int]:
    """(left, top, width, height) of the board floor region.

    The board sits on floor-colored tiles that form the target component in
    the lair band. Its width is exactly `cols*cell_px` and its bottom edge is
    the board's bottom (the lair floor continues below it), so cell_px and
    origin_y derive from it; the top is NOT reliable (the floor extends upward
    around the Devourer).

    Robustness: of the floor-colored components, the board floor is the WIDEST
    board-shaped (wide + short + low-bottom) one — a menu/dialog/station that
    aliases floor color is narrower or taller, so preferring width among
    board-like components beats blindly picking the largest-area component. If
    no component looks board-like, fall back to the largest area (the old
    behaviour) so a legitimate read is never lost.
    """
    comps = _floor_components(frame)
    if not comps:
        raise NoFloorBlobError("no floor blob found in lair band")
    bx, by, bw, bh = LAIR_BAND
    band_w, band_h = bw - bx, bh - by
    boardlike = [c for c in comps if _is_boardlike(c, band_w, band_h)]
    if boardlike:
        # Board floor is the widest board-shaped floor strip; ties to the
        # one with the lowest (largest-y) bottom, which is the floor line.
        best = max(boardlike,
                   key=lambda c: (c[2], c[3]))  # (width, height) descending
    else:
        best = max(comps, key=lambda c: c[4])   # largest area (legacy)
    return (best[0], best[1], best[2], best[3])


def _detect_board_top_y(frame) -> int | None:
    """y of the tile-grid's top edge, or None if no confident horizontal edge.

    The board's top boundary is a strong, horizontally-extensive vertical
    gradient band in the UPPER portion of the lair (below it the tiles begin,
    above it the rock/Devourer area). Vertical sprite-internal edges are
    suppressed by blur; a real board top is a long contiguous line. Measured
    live: 5x4 board top ~1201. Returns the row with the longest such edge.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(gy)
    x0, y0, x1, y1 = LAIR_BAND
    bx_lo, bx_hi = x0, x1
    search_bot = y0 + int((y1 - y0) * TOP_EDGE_SEARCH_FRAC)
    best_y, best_span = None, TOP_EDGE_MIN_SPAN
    for yy in range(y0, search_bot):
        cnt = int(np.count_nonzero(mag[yy:yy + 1, bx_lo:bx_hi] > TOP_EDGE_EDGE_THR))
        if cnt > best_span:
            best_y, best_span = yy, cnt
    return best_y


def _rows_fit(rows: int, cell_px: int, top_y: int, blob_bottom: int) -> float:
    """Abs residual (in cells) between a candidate's implied height and the
    measured board height. 0 is a perfect fit; a wrong row count is ~1 cell."""
    implied = rows * cell_px
    measured = blob_bottom - top_y
    return abs(implied - measured) / cell_px


def _derive_grid(frame, rows: int, cols: int, llm_cell_px: int) -> GridGeometry:
    """Pixel geometry from the floor blob (origin/mouth from calibrated fallback)."""
    bx, by, bw, bh = _blob_bbox(frame)
    cell_px = round(bw / cols)
    if not (150 <= cell_px <= 350):
        raise ValueError(f"blob-derived cell_px={cell_px} implausible (blob {bw}px wide, {cols} cols)")
    origin_x = bx + round((bw - cols * cell_px) / 2)
    origin_y = by + bh - rows * cell_px
    if not (100 <= origin_x <= 500 and 1050 <= origin_y <= 1600):
        raise ValueError(f"blob-derived origin=({origin_x},{origin_y}) implausible "
                         f"(blob ({bx},{by},{bw},{bh}), {rows}x{cols})")
    return GridGeometry(origin_x, origin_y, cell_px, rows, cols,
                        FALLBACK_GEOMETRY.mouth_x, FALLBACK_GEOMETRY.mouth_y)


def _plausible_mouth(mx: int, my: int) -> bool:
    x0, y0, x1, y1 = MOUTH_PLAUSIBLE
    return x0 <= mx <= x1 and y0 <= my <= y1


def _anchor_cells(frame, geom: GridGeometry, classifier) -> list[tuple[Cell, str, float]]:
    """Occupied cells with a confident template id under `geom` (strongest first).

    Each anchor is locked to its guess-time best template id: at a misaligned
    geometry that id no longer matches well, so only the correctly aligned grid
    scores high (spurious cross-template peaks are avoided).
    """
    scored = []
    for cell in build_cells(geom):
        if occupancy_score(frame, cell.row, cell.col, geom) < OCCUPANCY_THRESHOLD:
            continue
        best_id, best, _, _ = classifier._score_top2(frame, cell)
        if best_id is not None and best >= ANCHOR_MIN_SCORE:
            scored.append((cell, best_id, best))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:MAX_ANCHORS]


def _objective(frame, geom: GridGeometry, anchors, classifier) -> float:
    """Sum of per-anchor template-match scores (each locked to its guess id)."""
    cells = [(r, c, geom.cell_center(r, c)) for r in range(geom.rows) for c in range(geom.cols)]
    half = geom.cell_px // 2
    total = 0.0
    for anchor_cell, anchor_id, _score in anchors:
        _r, _c, (cx, cy) = min(
            cells, key=lambda t: (t[2][0] - anchor_cell.cx) ** 2 + (t[2][1] - anchor_cell.cy) ** 2)
        region = frame[cy - half : cy + half, cx - half : cx + half]
        frames = classifier.templates.get(anchor_id, [])
        if frames:
            total += max(classifier._match_score(region, t) for t in frames)
    return total


def _refine_geometry(frame, geom: GridGeometry, classifier: TemplateClassifier | None,
                     log=print) -> GridGeometry:
    """Coordinate descent around the blob-derived geometry on template-match quality.

    Maximises the anchored objective (each anchor locked to its guess-time id).
    Pitch is optimised first because its error propagates into the blob-derived
    origin_y (rows * error); origin follows. Ties resolve toward the blob start
    so a flat objective ridge cannot drag the grid away from the derived guess.
    """
    if classifier is None:
        return geom
    anchors = _anchor_cells(frame, geom, classifier)
    if not anchors:
        log("  geometry: no occupied anchors to refine against; trusting blob/LLM geometry")
        return geom
    start = _objective(frame, geom, anchors, classifier)
    cur = geom
    px0, ox0, oy0 = geom.cell_px, geom.origin_x, geom.origin_y

    def score(g):
        return _objective(frame, g, anchors, classifier)

    def best_along(cands, key_fn):
        cands = sorted(cands, key=lambda g: (-score(g), key_fn(g)))
        return cands[0]

    for _ in range(REFINE_ROUNDS):
        cur = best_along(
            [GridGeometry(cur.origin_x, cur.origin_y, px, cur.rows, cur.cols,
                          cur.mouth_x, cur.mouth_y)
             for px in range(cur.cell_px - REFINE_PITCH_WIN,
                             cur.cell_px + REFINE_PITCH_WIN + 1, 2)],
            lambda g: abs(g.cell_px - px0))
        cur = best_along(
            [GridGeometry(ox, cur.origin_y, cur.cell_px, cur.rows, cur.cols,
                          cur.mouth_x, cur.mouth_y)
             for ox in range(cur.origin_x - REFINE_ORIGIN_WIN,
                             cur.origin_x + REFINE_ORIGIN_WIN + 1, 5)],
            lambda g: abs(g.origin_x - ox0))
        cur = best_along(
            [GridGeometry(cur.origin_x, oy, cur.cell_px, cur.rows, cur.cols,
                          cur.mouth_x, cur.mouth_y)
             for oy in range(cur.origin_y - REFINE_ORIGIN_WIN,
                             cur.origin_y + REFINE_ORIGIN_WIN + 1, 5)],
            lambda g: abs(g.origin_y - oy0))

    best = score(cur)
    guard = max(REFINE_MIN_GAIN, REFINE_REL_GAIN * start)
    if best >= start + guard:
        if cur != geom:
            log(f"  geometry refined: {geom} -> {cur} (anchor score "
                f"{start:.2f} -> {best:.2f})")
        return cur
    return geom


def _candidate_dims(rows0: int, cols0: int):
    """(rows, cols) candidates ordered by distance from the LLM prior.

    The LLM's dims are a strong prior but not a verdict (it miscounted 4x3 as
    5x3 on the live board). Nearby dims are enumerated first so a blob-rejected
    prior falls through to its neighbors instead of killing the whole read.
    """
    dims = []
    for r in range(3, 7):
        for c in range(2, 5):
            if not _valid_dims(r, c):
                continue
            dist = abs(r - rows0) + abs(c - cols0)
            dims.append((dist, r, c))
    dims.sort()
    return [(r, c) for _d, r, c in dims]


def _score_geometry(frame, geom: GridGeometry, classifier) -> float:
    """Template-match quality of `geom` measured by its own anchors."""
    anchors = _anchor_cells(frame, geom, classifier)
    if not anchors:
        return 0.0
    return _objective(frame, geom, anchors, classifier)


def llm_grid_geometry(frame, llm=None, classifier: TemplateClassifier | None = None,
                      max_dim: int | None = MAX_DIM, log=print,
                      prior_dims: tuple[int, int] | None = None) -> GridGeometry:
    """Read board geometry off `frame`: dims prior + blob pixels + refine.

    The dims prior is, in order: explicit `prior_dims` (the pinned config —
    no LLM round at all), the LLM read, or nothing (enumerate ALL plausible
    dims and let the anchored-template objective pick the winner). The floor
    blob derives exact pixel geometry from the dims; candidates near the
    prior row/col count are each blob-derived, then the one whose grid best
    aligns its occupied cells to the template bank wins. Raises if no
    candidate is blob-plausible, so the caller can fall back to calibrated
    constants.
    """
    # The dims are a PRIOR, not a requirement: the pinned config skips the
    # LLM outright; text-only servers (e.g. Qwen3-4B-Instruct-2507) reject
    # image requests outright, and --jinja servers echo prefills into
    # garbled JSON. In those cases enumerate ALL plausible dims and let the
    # anchored-template objective pick the winner.
    rows0 = cols0 = 0
    mx = my = 0
    if prior_dims is not None:
        try:
            rows0, cols0 = int(prior_dims[0]), int(prior_dims[1])
        except (TypeError, ValueError, IndexError):
            rows0 = cols0 = 0
        if _valid_dims(rows0, cols0):
            log(f"  geometry: pinned dims {rows0}x{cols0} (no LLM round)")
        else:
            log(f"  geometry: pinned dims {rows0}x{cols0} invalid; enumerating all dims")
            rows0 = cols0 = 0
    elif llm is None:
        log("  geometry: no LLM and no pinned dims; enumerating all dims")
    else:
        image_url = _encode_frame(frame, max_dim=max_dim)
        messages = [
            {"role": "system", "content": GEOMETRY_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Output the board geometry JSON now."}]},
            {"role": "assistant", "content": '{"rows":'},  # prefill forces clean JSON (vision reads)
        ]
        try:
            reply, _full_msg = llm.chat(messages, max_tokens=128, json_mode=False)
            data = _parse_json(reply)
            rows0 = int(data.get("rows", 0))
            cols0 = int(data.get("cols", 0))
            mx, my = int(data.get("mouth_x", 0)), int(data.get("mouth_y", 0))
            if not _valid_dims(rows0, cols0):
                raise ValueError(f"implausible grid dims rows={rows0} cols={cols0}")
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            log(f"  geometry: LLM prior unavailable ({exc}); enumerating all dims")
            rows0 = cols0 = 0

    candidates = []
    for rows, cols in _candidate_dims(rows0, cols0):
        try:
            geom = _derive_grid(frame, rows, cols, 0)
        except ValueError:
            continue
        # Prune degenerate candidates cheaply: sprites (158px templates) nearly
        # fill a cell, so a cell_px far outside ~[1.2, 2.1]x template size means
        # the sprites wouldn't fit and this (rows, cols) is a miscount.
        if not (190 <= geom.cell_px <= 340):
            continue
        candidates.append((rows, cols, geom))
    if not candidates:
        raise ValueError(f"no blob-plausible geometry near LLM dims {rows0}x{cols0}")
    if _plausible_mouth(mx, my):
        candidates = [(*c[:2], GridGeometry(c[2].origin_x, c[2].origin_y, c[2].cell_px,
                                            c[2].rows, c[2].cols, mx, my))
                      for c in candidates]

    if classifier is None:
        _r, _c, geom = candidates[0]
        log(f"  geometry: LLM prior {rows0}x{cols0} -> blob {geom}")
        return geom

    # Row-count ambiguity: the blob's width fixes cell_px/cols precisely, but
    # rows only shifts origin_y, and square-ish boards (4x4 vs 5x4) can give
    # IDENTICAL template scores — a tie currently decided by the LLM's
    # unreliable count. The board's measured HEIGHT (top edge -> floor line)
    # resolves it: a candidate whose rows*cell_px matches that span is the true
    # one. Prefer the best vertical fit; among candidates that fit about as
    # well, fall back to the template objective to keep its anchor value.
    top_y = _detect_board_top_y(frame)
    _bx, _by, _bw, _bh = _blob_bbox(frame)
    blob_bottom = _by + _bh
    if top_y is None:
        scored = [(*c, _score_geometry(frame, c[2], classifier)) for c in candidates]
        best = max(scored, key=lambda c: c[3])
    else:
        scored = [(*c, _rows_fit(c[0], c[2].cell_px, top_y, blob_bottom),
                   _score_geometry(frame, c[2], classifier))
                  for c in candidates]
        best_fit = min(scored, key=lambda c: c[3])
        # Within the fit tolerance, the highest template score wins; otherwise
        # a candidate that clearly fits the measured height wins outright.
        in_tol = [c for c in scored if c[3] <= best_fit[3] + VERT_FIT_TOL]
        best = max(in_tol, key=lambda c: c[4])
    _br, _bc, geom = best[0], best[1], best[2]
    prior = f"{rows0}x{cols0}" if rows0 else "none"
    if (_br, _bc) != (rows0, cols0):
        log(f"  geometry: LLM prior {prior} overruled -> {_br}x{_bc} ({geom})")
    else:
        log(f"  geometry: LLM dims {prior}; blob -> {geom}")
    geom = _refine_geometry(frame, geom, classifier, log=log)
    return geom
