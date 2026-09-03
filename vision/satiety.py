"""Satiety fraction reader (top-left lair HUD).

The Devourer's satiety fraction (e.g. "1/250") is small text at the fixed
top-left of the lair: label at x0-49, numerator digit(s) ~x88-120, slash
~x105-114, and a "/NN" denominator ~x116-164, all on band y444-478 (native
1280x2856, geometry-independent like the mana bar).

It is an ornate font: glyphs partially connect and the vision LLM / tesseract
both fail on it (see AGENTS.md Aug 13 entry), so this reader is a *memory bank*:
observed numerator/denominator runs are banked as mask templates under their
confirmed value (assets/satiety/<value>__<n>.png, grown by
scripts/bank_satiety_digit.py). Reading matches the observed run against the
bank; unseen values read as None (unknown), which is honest rather than wrong.

Two structural lessons from the Aug 13 live calibration that shaped this reader:

1. The strip is LIGHT text on a DARK panel on this save (band gray mean ~52) and
   the old `abs(gray - median_bg) > 3` extraction flooded into near-solid
   rectangles, so EVERY bank token cross-matched at 0.84-0.99 and the reader
   returned aspect-ratio roulette ("50/50", then "250/250"), not a real read.
   Extraction is now polarity-agnostic with a real threshold (bright-on-dark or
   dark-on-light).

2. The numerator/denominator are NOT framed by fixed pixel windows: the digits
   render as separate runs and NUM_REGION (40px) swallowed "1" + "/" + the "2"
   of "/250". Runs are now auto-split at the slash, found by its diagonality
   (the '/' ink shifts vertically across its run; digits stay level). The
   numerator tokens = runs left of the slash, denominator = runs right of it.

Matching uses a right-aligned Jaccard overlap on same-height masks, which kills
the extra-digit false match (a banked "50" tails a "/250" at only ~0.57 IoU;
"1" vs "250" ~0.10; same values ~0.95-1.00), with BANK_THRESHOLD = 0.75.
"""

from pathlib import Path

import cv2
import numpy as np


def apple_vision_text(png_path: str) -> str:
    """Read text from an image file via Apple's Vision framework (pyobjc).

    Accurate-mode VNRecognizeTextRequest, language correction off (it would
    bend digits toward words). Returns the recognized strings sorted
    left-to-right and space-joined; "" when nothing is recognized or the
    binding is unavailable."""
    try:
        import Vision
        from Foundation import NSURL
    except ImportError:
        return ""
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        NSURL.fileURLWithPath_(str(png_path)), None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    req.setRecognitionLanguages_(["en-US"])
    ok, _err = handler.performRequests_error_([req], None)
    if not ok:
        return ""
    cands = []
    for obs in (req.results() or []):
        top = obs.topCandidates_(1)
        if top:
            cands.append((obs.boundingBox().origin.x, top[0].string()))
    return " ".join(s for _, s in sorted(cands))


def parse_fraction(text: str) -> tuple[str | None, str | None]:
    """Normalize an OCR reading like '0:50' / '21,750' / '0/50' to
    (num, den) digit groups split on non-digit runs.

    A reading WITHOUT a separator ("21750") parses as one group -> (None,
    None): the separator is structural evidence that two numbers were seen,
    and its absence means the reading is not trustworthy enough to use."""
    import re
    groups = re.findall(r"\d+", text or "")
    if len(groups) < 2:
        return None, None
    return groups[0], groups[1]


# ---- current-save HUD: a big pixel-font fraction rendered INSIDE the bar ----
# The Aug 13 small ornate strip (SAT_BAND) was one save state; on the current
# (higher-level Devourer) save the fraction is large chunky white digits with
# dark outlines at ~x63-205, y443-478. Layout evidently varies with game
# state, so this path finds the text dynamically instead of trusting fixed
# pixel windows.

TEXT_WINDOW = (50, 420, 240, 95)   # generous search window (x, y, w, h)


def _white_text_mask(frame) -> "np.ndarray":
    """Binary mask of near-white glyph pixels inside TEXT_WINDOW.

    The satiety bar fill (saturated yellow/cyan) and its thin border lines are
    excluded by saturation + component size filtering downstream."""
    import cv2
    x, y, w, h = TEXT_WINDOW
    if frame.shape[0] < y + h or frame.shape[1] < x + w:
        return None
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 2] > 185) & (hsv[:, :, 1] < 90)).astype(np.uint8)


def fraction_text_crop(frame) -> tuple["np.ndarray", tuple[int, int]] | None:
    """Locate the fraction text and return (binary_mask, top_left_in_frame).

    White-ish components of digit-like size are kept (the bar's long thin
    border lines and small specks are dropped), sorted left-to-right, and
    unioned into one tight crop. The mask is 0/255 with ink=255; callers pass
    it straight to OCR engines."""
    m = _white_text_mask(frame)
    if m is None:
        return None
    import cv2
    # A horizontal band of blended pixels (the bar fill/highlight line
    # crossing behind the text) splits glyphs into top/bottom halves;
    # bridge it vertically before component analysis or leading digits
    # fall below the height filter.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    n, lab, stats, _cent = cv2.connectedComponentsWithStats(m)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 15 or h > 55 or w > 60 or area < 120:
            continue
        comps.append((x, y, w, h))
    if len(comps) < 3:            # at least num digit(s), slash, den digit(s)
        return None
    x0 = min(c[0] for c in comps)
    x1 = max(c[0] + c[2] for c in comps)
    y0 = min(c[1] for c in comps)
    y1 = max(c[1] + c[3] for c in comps)
    keep = np.zeros_like(m)
    for x, y, w, h in comps:
        keep[y:y + h, x:x + w] |= (lab[y:y + h, x:x + w] > 0)
    crop = keep[y0:y1 + 1, x0:x1 + 1]
    wx, wy, _, _ = TEXT_WINDOW
    return ((crop * 255).astype(np.uint8), (wx + x0, wy + y0))

# Satiety strip search band (fixed HUD pixels, fraction zone). Starts at x=80
# to exclude the "Satiety" label (its glyphs have HIGHER diagonality than the
# slash and would hijack the split).
SAT_BAND = (80, 436, 96, 50)       # (x, y, w, h) — the whole fraction
BANK_DIR = "assets/satiety"
BANK_THRESHOLD = 0.75              # Jaccard (IoU) needed to accept a token match
_POLARITY = 150                    # ink = gray > this (light-on-dark) or < this
_RUN_MIN_INK = 2                   # columns with this many ink rows count as a run
_RUN_MIN_PX = 3                    # minimum run width
SLASH_MIN_DIAG = 5.0               # vertical row shift that marks a '/' run

# ---- OCR reading (Aug 21): Apple Vision over CLAHE-normalized crops --------
# The Aug 13 small ornate strip (SAT_BAND) was one save state; on the current
# (higher-level Devourer) save the fraction is big chunky digits rendered
# INSIDE the bar at ~x63-205, y443-478. Layout varies with game state, so the
# reader tries fixed windows newest-first and falls back to the token bank.

# Fixed OCR crops (the HUD is geometry-independent like the mana bar):
OCR_WINDOW = (52, 430, 173, 62)    # x, y, w, h — covers "NNN/MNN" fully
OCR_FALLBACKS = (SAT_BAND,)
OCR_UPSCALE = 4


def ocr_fraction(frame) -> tuple[str | None, str | None]:
    """Read the fraction via Apple Vision OCR; (None, None) when nothing
    parseable is found or the pyobjc binding is unavailable.

    Each candidate window is CLAHE-normalized (the text sits on a mixed
    yellow/dark background that defeats plain thresholds), upscaled, and read;
    a reading needs BOTH groups plus a separator to count (parse_fraction's
    structural guard)."""
    import tempfile
    if frame is None or getattr(frame, "ndim", 0) != 3:
        return None, None
    for rect in (OCR_WINDOW,) + tuple(OCR_FALLBACKS):
        x, y, w, h = rect
        if frame.shape[0] < y + h or frame.shape[1] < x + w:
            continue
        gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        big = cv2.resize(clahe, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE,
                         interpolation=cv2.INTER_CUBIC)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            cv2.imwrite(tmp.name, big)
            num, den = parse_fraction(apple_vision_text(tmp.name))
        finally:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)
        if num and den:
            return num, den
    return None, None


class SatietyReader:
    """Read the lair-top satiety fraction from a template bank of confirmed
    numerator/denominator tokens."""

    def __init__(self, bank_dir: str = BANK_DIR, threshold: float = BANK_THRESHOLD):
        self.bank_dir = Path(bank_dir)
        self.threshold = threshold
        self.bank: dict[str, list] = {}   # value -> list of glyph masks
        self._load_bank()

    def _load_bank(self) -> None:
        for path in sorted(self.bank_dir.glob("*.png")):
            value = path.stem.split("__")[0]
            self.bank.setdefault(value, []).append(
                cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))

    # ---- run extraction ----------------------------------------------------

    @staticmethod
    def _ink(gray) -> np.ndarray:
        """Binary ink mask, polarity-agnostic (light-on-dark or dark-on-light)."""
        if gray.mean() < 140:      # dark panel, light digits
            return gray > _POLARITY
        return gray < _POLARITY

    @staticmethod
    def _runs(ink) -> list[tuple[int, int]]:
        """Connected ink runs by column: list of (x0, x1) inclusive."""
        col = ink.sum(axis=0)
        runs, start = [], None
        for i, v in enumerate(col):
            if v > _RUN_MIN_INK and start is None:
                start = i
            if v <= _RUN_MIN_INK and start is not None:
                if i - start >= _RUN_MIN_PX:
                    runs.append((start, i - 1))
                start = None
        if start is not None and len(col) - 1 - start >= _RUN_MIN_PX:
            runs.append((start, len(col) - 1))
        return runs

    @staticmethod
    def _diag(ink, run: tuple[int, int]) -> float:
        """How much the ink shifts vertically across a run (the '/' is large;
        digits stay level)."""
        x0, x1 = run
        sub = ink[:, x0:x1 + 1].astype(np.float32)
        half = int((x1 - x0 + 1) / 2)
        lr = sub[:, :half].astype(bool)
        rr = sub[:, half:].astype(bool)
        ly = np.nonzero(lr)[0]
        ry = np.nonzero(rr)[0]
        if ly.size == 0 or ry.size == 0:
            return 0.0
        return float(abs(ly.mean() - ry.mean()))

    @staticmethod
    def _tight(ink, x0: int, x1: int) -> np.ndarray | None:
        """Tight-cropped 0/255 mask of the ink in x0..x1 across the whole band."""
        m = ink[:, x0:x1 + 1]
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            return None
        crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return (crop * 255).astype(np.uint8)

    def split_tokens(self, frame) -> tuple[np.ndarray, np.ndarray] | None:
        """Split the strip into (num_mask, den_mask) by locating the slash via
        its diagonality. Masks are 0/255 tight-crops (num = runs left of the
        slash, den = runs right of it). None when the strip can't be parsed."""
        x, y, w, h = SAT_BAND
        if frame.shape[0] < y + h or frame.shape[1] < x + w:
            return None
        gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        ink = self._ink(gray)
        runs = self._runs(ink)
        if len(runs) < 3:
            return None
        ranked = sorted(runs, key=lambda r: self._diag(ink, r), reverse=True)
        sl = ranked[0]
        if self._diag(ink, sl) < SLASH_MIN_DIAG:
            return None
        i = runs.index(sl)
        num_runs = [self._tight(ink, a, b) for a, b in runs[:i]]
        den_runs = [self._tight(ink, a, b) for a, b in runs[i + 1:]]
        num_runs = [m for m in num_runs if m is not None]
        den_runs = [m for m in den_runs if m is not None]
        if not num_runs or not den_runs:
            return None
        num = (np.concatenate(num_runs, axis=1) if len(num_runs) > 1
               else num_runs[0])
        den = (np.concatenate(den_runs, axis=1) if len(den_runs) > 1
               else den_runs[0])
        return num, den

    @staticmethod
    def glyph(frame, region: tuple[int, int, int, int]) -> np.ndarray | None:
        """Binarized tight-cropped mask for a whole region (single-run, kept for
        back-compat callers). Polarity-agnostic with a real threshold."""
        x, y, w, h = region
        g = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        ink = SatietyReader._ink(g)
        return SatietyReader._tight(ink, 0, w - 1)

    # ---- matching ----------------------------------------------------------

    @staticmethod
    def overlap(a, b, W: int = 130, H: int = 36) -> float:
        """Right-aligned same-height Jaccard (IoU) of two tight-crop masks.

        Aligning on the trailing edge and NOT re-normalizing width means an
        extra leading digit adds to the union without intersecting — the honest
        guard that a resize-normed similarity score hides."""
        ca = np.zeros((H, W), dtype=bool)
        cb = np.zeros((H, W), dtype=bool)

        def paste(c, m):
            m = (m * 1.0).astype(np.float32)
            m = cv2.resize(m, (max(1, int(round(m.shape[1] * H / m.shape[0]))), H),
                           interpolation=cv2.INTER_AREA)
            hh, ww = m.shape
            c[H - hh:H, W - ww:W] = m > 64

        paste(ca, a)
        paste(cb, b)
        inter = (ca & cb).sum()
        union = (ca | cb).sum()
        return float(inter / (union + 1e-9))

    def _match(self, glyph) -> tuple[str | None, float]:
        best, best_score = None, -1.0
        for value, templates in self.bank.items():
            for t in templates:
                s = self.overlap(glyph, t)
                if s > best_score:
                    best, best_score = value, s
        return best, best_score

    # ---- API -------------------------------------------------------------

    def bank_token(self, value: str, glyph, overwrite: bool = False) -> Path | None:
        """Persist a confirmed token glyph under `value` (idempotent unless
        overwrite). Returns the saved path or None if the token is already
        banked with a matching glyph."""
        if not overwrite:
            for t in self.bank.get(value, []):
                if self.overlap(glyph, t) >= self.threshold:
                    return None
        n = len(self.bank.get(value, []))
        path = self.bank_dir / f"{value}__{n}.png"
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), glyph)
        self.bank.setdefault(value, []).append(glyph)
        return path

    def read_satiety(self, frame) -> dict:
        """Return {"num","den","text","scores"} for the satiety fraction.

        Primary path: OCR (Apple Vision over a CLAHE-normalized crop of the
        in-bar fraction; falls back to the legacy small strip band). The
        template bank is the last-resort fallback (offline / no pyobjc /
        unparseable HUD), reading banked tokens only. num/den are str|None;
        `text` renders whatever was read, "?/??". A parsed fraction with
        num > den is rejected as invalid."""
        ocr = ocr_fraction(frame)
        scores = {"num": -1.0, "den": -1.0}
        num_v, den_v = ocr
        if num_v is None or den_v is None:
            toks = self.split_tokens(frame)
            if toks is not None:
                num_mask, den_mask = toks
                bnum, bnum_s = self._match(num_mask)
                bden, bden_s = self._match(den_mask)
                if num_v is None:
                    num_v = bnum if bnum_s >= self.threshold else None
                    scores["num"] = round(bnum_s, 3)
                if den_v is None:
                    den_v = bden if bden_s >= self.threshold else None
                    scores["den"] = round(bden_s, 3)
        if num_v is not None and den_v is not None:
            try:
                if int(num_v) > int(den_v):
                    num_v = den_v = None
            except ValueError:
                num_v = den_v = None
        return {
            "num": num_v,
            "den": den_v,
            "text": f"{num_v or '?'}/{den_v or '??'}",
            "scores": scores,
        }