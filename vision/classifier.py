import cv2
import numpy as np
import re
from pathlib import Path

from vision.grid import Cell

# A runtime-banked template below this gray std is background, not a sprite —
# refused by add_template (see its docstring for the incident this prevents).
MIN_TEMPLATE_STD = 15.0


class ItemClassifier:
    """Base interface: map a frame + board cells to item ids."""

    def classify(self, frame, cells: list[Cell]) -> list[str | None]:
        raise NotImplementedError

    def classify_scores(self, frame, cells: list[Cell]) -> list[tuple[str | None, float, float, str | None]]:
        """(item_id, confidence, margin, runner_up_id) per cell.
        margin = best minus runner-up match, a robust merge gate (absolute scores
        dip with idle-bob phase). runner_up_id = second-best template id; used by
        the neighbor-suspect merge gate to detect same-family different-level
        cross-fire (Aug 26 incident).
        Default: treat classify() hits as confident, no margin."""
        return [(item, 1.0 if item else 0.0, 1.0 if item else 0.0, None)
                for item in self.classify(frame, cells)]


class TemplateClassifier(ItemClassifier):
    def __init__(self, templates_dir="assets/templates", threshold=0.6, seed=True):
        self.templates_dir = templates_dir
        self.templates: dict[str, list[np.ndarray]] = {}
        if seed:
            for path in sorted(Path(templates_dir).glob("*.png")):
                item_id = path.stem.split("__")[0]      # "skeleton_lvl1__0" -> "skeleton_lvl1"
                self.templates.setdefault(item_id, []).append(cv2.imread(str(path)))
        self.threshold = threshold

    def _score_top2(self, frame, cell: Cell) -> tuple[str | None, float, float, str | None]:
        """(best_id, best_score, runner_up_score, runner_up_id) over all template ids.

        runner_up_id is the second-best template (different from best). Used by the
        neighbor-suspect merge gate (Aug 26) to detect "best=lvl4 but could be lvl5"
        ambiguity: when the top-2 are same-family different levels within a tight
        margin, the cell is in a bob phase where the bank can't disambiguate, and
        the merge may pair two mislabeled cells.
        """
        half = cell.cell_px // 2
        region = frame[cell.cy - half : cell.cy + half,
                       cell.cx - half : cell.cx + half]
        best_id, best_score = None, -1.0
        runner_up_id, runner_up_score = None, -1.0
        for item_id, frames in self.templates.items():
            item_score = max(self._match_score(region, t) for t in frames)
            if item_score > best_score:
                runner_up_id, runner_up_score = best_id, best_score
                best_score, best_id = item_score, item_id
            elif item_score > runner_up_score:
                runner_up_id, runner_up_score = item_id, item_score
        return best_id, best_score, runner_up_score, runner_up_id

    def score_cell(self, frame, cell: Cell) -> tuple[str | None, float]:
        best_id, best_score, _, _ = self._score_top2(frame, cell)
        return best_id, best_score

    def classify(self, frame, cells):
        results = []
        for cell in cells:
            item_id, score, _, _ = self._score_top2(frame, cell)
            results.append(item_id if score >= self.threshold else None)
        return results

    def classify_scores(self, frame, cells):
        results = []
        for cell in cells:
            item_id, score, runner_up_score, runner_up_id = self._score_top2(frame, cell)
            margin = max(0.0, score - runner_up_score) if runner_up_score > 0 else score
            results.append((item_id if score >= self.threshold else None,
                            score, margin, runner_up_id))
        return results

    def add_template(self, item_id: str, crop) -> bool:
        """Add a board-sprite template at runtime and persist it to disk.

        The bank grows as the agent identifies new items: `crop` is a
        TEMPLATE_SIZE square captured from the board. Persisted as
        `<id>__<N>.png` where N keeps accumulating bob phases per id.
        Ids are sanitized to `[a-z0-9_]` so LLM/OCR-mangled names
        ("Ice Runes") can never pollute the bank.

        Texture gate: a near-flat crop (gray std < MIN_TEMPLATE_STD) is
        REFUSED — banking empty-cell background as an item template makes
        that background self-match at 1.0 forever (the Aug 26 incident: a
        vanished chest's empty cell was banked as `icebox_unopened`, then
        matched itself at 1.00 and masked the cell being empty). Returns
        True when the template was stored.
        """
        if crop is None:
            return False
        item_id = re.sub(r"[^a-z0-9_]", "", str(item_id).lower())
        if not item_id:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < MIN_TEMPLATE_STD:
            return False
        frames = self.templates.setdefault(item_id, [])
        path = Path(self.templates_dir) / f"{item_id}__{len(frames)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop)
        frames.append(crop)
        return True

    def has(self, item_id: str) -> bool:
        return item_id in self.templates

    def sprite_matches(self, item_id: str, sprite) -> float:
        """Best match score of `sprite` against item_id's templates (or -1)."""
        frames = self.templates.get(item_id)
        if not frames or sprite is None:
            return -1.0
        return max(self._match_score(sprite, t) for t in frames)

    def match_existing(self, sprite) -> tuple[str | None, float]:
        """Best-matching id already in the bank for `sprite`, else (None, score)."""
        if sprite is None:
            return None, -1.0
        best_id, best_score = None, -1.0
        for item_id, frames in self.templates.items():
            s = max(self._match_score(sprite, t) for t in frames)
            if s > best_score:
                best_score, best_id = s, item_id
        if best_id is not None and best_score < self.threshold:
            return None, best_score
        return best_id, best_score

    @staticmethod
    def _match_score(region, template) -> float:
        if template.shape[0] > region.shape[0] or template.shape[1] > region.shape[1]:
            return -1.0   # template larger than search region
        res = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        return float(res.max())   # best offset within the cell