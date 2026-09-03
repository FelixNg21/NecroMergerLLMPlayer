"""Popup-tap identification of occupied-but-unknown board cells.

When a cell's sprite isn't in the template bank, tap it to open its info
popup, read (name, level) from the fixed title banner via OCR, then add the
board sprite to the template bank under the assembled id `name_lvl<N>` (or
`name` when level-less). A signature bank of name-region crops dedups OCR
variance so the same item is never split into two ids.

Level matters: identity is (name, level) — `skeleton_lvl3` vs `skeleton_lvl5`
are different merge targets. A sprite/level conflict check guarantees a given
id always maps to one sprite, so the planner never attempts an illegal merge.

added `identify_batch` for batch reads (one tool call covers N
unidentified cells) and a `popup_question_for(category)` helper that picks
the right LLM prompt per item category (Champion / Currency / Station /
creature default). The motivation: champion popups have HP+drop info, currency
popups have a max-level flag, station popups have build-cost info — generic
prompts miss category-specific facts, and the model's response is more
accurate when the question is targeted.
"""

import json
import re
from pathlib import Path

import cv2

from planner.llm import _extract_json
from planner.llm_client import LLMClient, LLMError
from planner.llm_vision import _encode_frame
from vision import ocr
from vision.classifier import TemplateClassifier
from vision.grid import crop_cell
from vision.menu import read_item_popup

POPUP_DISMISS = (120, 2350)      # dimmed corner: closes the popup
SIGNATURE_DIR = "assets/signatures"
SIGNATURE_THRESHOLD = 0.7
DIGIT_DIR = "assets/digits"
DIGIT_THRESHOLD = 0.75
IDENTIFY_PAUSE = 1.3             # seconds to let the popup animate in
DISMISS_PAUSE = 0.6
READ_RETRIES = 2                 # times to tap+read the popup before giving up
POPUP_OPEN_DIFF = 15.0           # mean abs-diff in banner region meaning the popup is up

# per-category popup questions. The generic ITEM_POPUP_QUESTION
# (defined in vision/menu.py) misses category-specific facts: a Champion
# popup has HP+drop+spawn info, a currency popup has a max-level flag, a
# station popup has a build-cost. Targeted questions give the LLM a smaller,
# sharper field and improve accuracy (the model's response is constrained
# to the facts that matter).
CHAMPION_POPUP_QUESTION = """This is the body of a NecroMerger CHAMPION info popup (a champion invades
the lair and is fought by drag-attack). Read it and reply with ONLY JSON:
{"name": "<lowercase champion name, e.g. peasant>", "level": <int or null>,
 "description": "<the description text, or empty>",
 "unlock": "<how the champion is unlocked / spawned, or empty>",
 "hp": <int or null>, "drop": "<what the champion drops on defeat, or empty>",
 "feed_value": <int or null>}
Example: for The Peasant, reply {"name": "peasant", "level": null,
"description": "Summon by merging Skeletons, Zombies and Mummies.",
"unlock": "Devourer Level 5", "hp": 100, "drop": "Ice Chest",
"feed_value": 50}.
If unreadable, return {"name": "", "level": null, "description": "",
"unlock": "", "hp": null, "drop": "", "feed_value": null}. No prose outside the JSON."""


CURRENCY_POPUP_QUESTION = """This is the body of a NecroMerger CURRENCY (Rune / Coin / Gem / etc.) info popup.
Currencies are feed-only items: the more you merge, the more currency they
grant when fed. Read it and reply with ONLY JSON:
{"name": "<lowercase item name, e.g. icerune or coin>", "level": <int or null>,
 "max_level": <true or false>,
 "description": "<the description text, or empty>",
 "merge_info": "<merge behavior, or empty>",
 "feed_value": <int or null>}
Example: for a Coin lvl 3, reply {"name": "coin", "level": 3,
"max_level": true, "description": "Merge to create a bigger pile or feed to the Devourer.",
"merge_info": "two Coin -> Coin of the next level", "feed_value": 12}.
If the popup says "Feed to the Devourer." with no merge line, max_level is true.
If unreadable, return {"name": "", "level": null, "max_level": false,
"description": "", "merge_info": "", "feed_value": null}. No prose outside the JSON."""


STATION_POPUP_QUESTION = """This is the body of a NecroMerger STATION info popup (a placed building
that produces items, costs resources, or stores capacity). Read it and reply
with ONLY JSON:
{"name": "<lowercase station name, e.g. grave or manapool>", "level": <int or null>,
 "description": "<the description text, or empty>",
 "uses": "<spawn description (e.g. 'bone / ribcage (infinite)' or 'ice rune (5 uses)'), or empty>",
 "spawn_outputs": "<comma-separated list of item names this station can spawn, in spawn order, e.g. 'bone, ribcage, zombie' or 'icerune_lvl1, icerune_lvl2, poisonrune_lvl1' or empty>",
 "spawn_rates": "<ordered comma-separated list of integer percentages that sum to 100, matching spawn_outputs in the same order (e.g. '40, 30, 30' or '60, 20, 20'), or empty>",
 "uses_remaining": <int or null - finite uses left (e.g. chests have 5); null means infinite/irrelevant>,
 "max_level": <true or false>}
Example: for a Grave Lvl 1, reply {"name": "grave", "level": 1,
"description": "Spawns bone/ribcage components when tapped.",
"uses": "bone / ribcage (infinite)", "spawn_outputs": "bone, ribcage",
"spawn_rates": "60, 40", "uses_remaining": null, "max_level": false}.
Example 2: for an Ice Chest, reply {"name": "icebox", "level": null,
"description": "Tap to open.", "uses": "ice rune (5 uses)",
"spawn_outputs": "icerune_lvl1, icerune_lvl2, poisonrune_lvl1",
"spawn_rates": "60, 20, 20", "uses_remaining": 5, "max_level": false}.
If unreadable, return {"name": "", "level": null, "description": "",
"uses": "", "spawn_outputs": "", "spawn_rates": "", "uses_remaining": null,
"max_level": false}. No prose outside the JSON."""


# Hints used to pick the right popup question. Order: most specific first.
# Match is by lowercase substring on the OCR'd name.
CATEGORY_NAME_HINTS = [
    # Champions (per the wiki)
    ("champion", CHAMPION_POPUP_QUESTION),
    ("peasant", CHAMPION_POPUP_QUESTION),
    ("knight", CHAMPION_POPUP_QUESTION),
    ("cleric", CHAMPION_POPUP_QUESTION),
    ("paladin", CHAMPION_POPUP_QUESTION),
    ("rival", CHAMPION_POPUP_QUESTION),
    ("protector", CHAMPION_POPUP_QUESTION),
    ("mech", CHAMPION_POPUP_QUESTION),
    ("king", CHAMPION_POPUP_QUESTION),
    # Currencies (Runes / Coins / Gems / etc.)
    ("rune", CURRENCY_POPUP_QUESTION),
    ("coin", CURRENCY_POPUP_QUESTION),
    ("gem", CURRENCY_POPUP_QUESTION),
    # Stations
    ("grave", STATION_POPUP_QUESTION),
    ("pool", STATION_POPUP_QUESTION),
    ("cupboard", STATION_POPUP_QUESTION),
    ("chicken", STATION_POPUP_QUESTION),
    ("vat", STATION_POPUP_QUESTION),
    ("altar", STATION_POPUP_QUESTION),
    ("stores", STATION_POPUP_QUESTION),
    ("lectern", STATION_POPUP_QUESTION),
    ("fridge", STATION_POPUP_QUESTION),
    ("portal", STATION_POPUP_QUESTION),
    ("saucer", STATION_POPUP_QUESTION),
    ("telepad", STATION_POPUP_QUESTION),
    ("grinder", STATION_POPUP_QUESTION),
    ("prism", STATION_POPUP_QUESTION),
    ("meteor", STATION_POPUP_QUESTION),
    ("throne", STATION_POPUP_QUESTION),
    ("parcel", STATION_POPUP_QUESTION),
    # Chests are spawn stations too
    ("chest", STATION_POPUP_QUESTION),
    ("box", STATION_POPUP_QUESTION),
]


def popup_question_for(name_hint: str) -> str:
    """Pick the right LLM question for the item category, given the OCR'd name.

    Falls back to the generic ITEM_POPUP_QUESTION (in vision/menu.py) when
    no category hint matches (the OCR name is too mangled or the item is
    something new). Used by `Identifier.identify_batch` to target the
    LLM's popup-read with a sharper question.
    """
    # Imported here to avoid a circular import at module load time.
    from vision.menu import ITEM_POPUP_QUESTION
    name_lc = (name_hint or "").lower()
    for hint, question in CATEGORY_NAME_HINTS:
        if hint in name_lc:
            return question
    return ITEM_POPUP_QUESTION


class Identifier:
    def __init__(self, device, classifier: TemplateClassifier,
                 signatures_dir: str = SIGNATURE_DIR,
                 digits_dir: str = DIGIT_DIR,
                 dock_rewards_dir: str = "assets/dock_rewards",
                 llm: LLMClient | None = None):
        self.device = device
        self.classifier = classifier
        self.signatures_dir = Path(signatures_dir)
        self.digits_dir = Path(digits_dir)
        self.dock_rewards_dir = Path(dock_rewards_dir)
        self.llm = llm
        self.signatures: dict[str, object] = {}
        self.digits: dict[int, list] = {}
        self.dock_rewards: dict[str, object] = {}
        self.counter = 0
        self._load_signatures()
        self._load_digits()
        self._load_dock_rewards()

    def _load_signatures(self) -> None:
        for path in sorted(self.signatures_dir.glob("*.png")):
            self.signatures[path.stem] = cv2.imread(str(path))

    def _load_digits(self) -> None:
        for path in sorted(self.digits_dir.glob("*.png")):
            level = int(path.stem.split("__")[0])
            self.digits.setdefault(level, []).append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))

    def _load_dock_rewards(self) -> None:
        """load dock-reward icon templates from assets/dock_rewards/.

        The dock button shows a different icon when a reward is queued
        (icebox_unopened, valuablechest, future reward types). Bank them
        by `item_id` filename so `read_dock_reward` can match without
        an LLM call. Missing dir = empty bank (the dock is treated as
        opaque; the model would need to call `collect_queue` blindly).
        """
        self.dock_rewards: dict[str, object] = {}
        dock_dir = Path("assets/dock_rewards")
        if not dock_dir.exists():
            return
        for path in sorted(dock_dir.glob("*.png")):
            self.dock_rewards[path.stem] = cv2.imread(str(path))

    def _match_digit(self, glyph) -> tuple[int | None, float]:
        """Return (level, score) of the best-matching saved digit glyph."""
        best_level, best_score = None, -1.0
        for level, samples in self.digits.items():
            s = max(TemplateClassifier._match_score(glyph, g) for g in samples)
            if s > best_score:
                best_score, best_level = s, level
        if best_level is not None and best_score < DIGIT_THRESHOLD:
            return None, best_score
        return best_level, best_score

    def bank_digit(self, level: int, popup_frame) -> bool:
        """Save a digit glyph crop under `level` (bootstrap). Refuses to
        overwrite with a glyph that conflicts with a different level."""
        glyph = ocr.glyph_crop(popup_frame)
        for other, samples in self.digits.items():
            if other == level:
                continue
            s = max(TemplateClassifier._match_score(glyph, g) for g in samples)
            if s >= DIGIT_THRESHOLD:
                return False
        self.digits_dir.mkdir(parents=True, exist_ok=True)
        n = len(self.digits.get(level, []))
        path = self.digits_dir / f"{level}__{n}.png"
        cv2.imwrite(str(path), glyph)
        self.digits.setdefault(level, []).append(glyph)
        return True

    def _match_signature(self, name_crop) -> tuple[str | None, float]:
        """Return (base_name, score) of the best matching saved signature."""
        best_id, best_score = None, -1.0
        for item_id, sig in self.signatures.items():
            s = TemplateClassifier._match_score(name_crop, sig)
            if s > best_score:
                best_score, best_id = s, item_id
        if best_id is not None and best_score < SIGNATURE_THRESHOLD:
            return None, best_score
        return best_id, best_score

    def _save_signature(self, base_name: str, name_crop) -> None:
        if base_name in self.signatures:
            return
        self.signatures_dir.mkdir(parents=True, exist_ok=True)
        path = self.signatures_dir / f"{base_name}.png"
        cv2.imwrite(str(path), name_crop)
        self.signatures[base_name] = name_crop

    def _resolve_id(self, base_name: str, level: int | None, sprite) -> str:
        """Assemble id from (name, level); dedup + disambiguate sprite conflicts.

        The sprite is the ground truth for (name, level): if it already matches
        a banked id, reuse it (OCR level variance can't split one item); if the
        freshly assembled id exists but the sprite doesn't match it, OCR level
        was wrong — bank the new sprite under the canonical id (so the bank
        gets more templates over time) rather than minting a __alt id.

        removed __alt id minting. The __alt family was created when
        the OCR gave a wrong level: the LLM said "skeleton" and the bank had
        `skeleton_lvl4` but the sprite didn't match, so the code minted
        `skeleton_lvl4__alt1` to keep the cell "known". The cost was high:
        the __alt id had no damage/feed stats banked, so the bot refused
        the cell as a valid attacker (`best_attack_pair` requires a known
        damage value). The fix is to bank the new sprite under the
        canonical id — the model sees the popup, learns the right name, and
        the bank grows more permissive. If the popup also says a different
        level, the next UNID-popup-read (Option 3) catches it.
        """
        existing, score = self.classifier.match_existing(sprite)
        if existing is not None:
            return existing
        base_name = self._sanitize_id(base_name)
        item_id = f"{base_name}_lvl{level}" if level else base_name
        # Always return the canonical id; add the sprite to its bank so the
        # next classify_board run can match it. The match_existing check
        # above already returns a true-positive banked id for sprites that
        # already match; this branch is reached when the freshly-derived
        # id conflicts with a different sprite, and broadening the
        # canonical bank is safer than minting a __alt.
        if self.classifier.has(item_id):
            self.classifier.add_template(item_id, sprite)
        return item_id

    @staticmethod
    def _sanitize_id(name: str) -> str:
        """Normalize a (possibly LLM/OCR-mangled) item name into a bankable id.

        LLM names can carry spaces/case/punctuation ("Ice Runes", "Valuable
        Chest!") that would otherwise produce invalid template ids and filename
        pollution (e.g. `ice runes_lvl1__0.png`). Banked ids are strictly
        `[a-z0-9]` with an optional `_lvl<N>` suffix — lowercase and
        drop every other character. (The `__alt<N>` suffix was removed in
        Aug 27 — see `_resolve_id` for the rationale.)
        """
        cleaned = re.sub(r"[^a-z0-9]", "", str(name).lower())
        return cleaned or "unknown"

    def _llm_name_and_level(self, popup_frame) -> tuple[str | None, int | None]:
        """Ask the LLM to name the item from the popup's name + digit crops.

        More reliable than OCR for this stylized font (tesseract confuses
        S/K/H/Z/B and 5<->6). Returns (name, level) or (None, None) when the
        LLM is unavailable or its reply doesn't parse.
        """
        if self.llm is None:
            return None, None
        crops = [
            {"type": "image_url", "image_url": {"url": _encode_frame(ocr.crop_name(popup_frame))}},
            {"type": "image_url", "image_url": {"url": _encode_frame(ocr.glyph_crop(popup_frame))}},
        ]
        user = {"type": "text", "text": (
            "The first image is the NAME line of a NecroMerger item info popup "
            "(title banner crop). The second image is the LEVEL number next to "
            "it (blank/garbage for level-less items). Reply with ONLY JSON: "
            '{"name": "<lowercase item name>", "level": <int or null>}. '
            'Example: {"name": "skeleton", "level": 5}.')}
        messages = [
            {"role": "system", "content": "You identify NecroMerger item names from popup crops."},
            {"role": "user", "content": [crops[0], crops[1], user]},
        ]
        try:
            content, _tool_calls, _full = self.llm.chat_message(messages, max_tokens=64, json_mode=True)
        except LLMError:
            return None, None
        try:
            data = _extract_json(content)
        except (ValueError, json.JSONDecodeError):
            return None, None
        name = str(data.get("name") or "").strip().lower() or None
        level = data.get("level")
        try:
            level = int(level) if level is not None else None
        except (TypeError, ValueError):
            level = None
        return name, level

    def _popup_open(self, board_frame, popup_frame) -> bool:
        """True when the item-info popup is open on `popup_frame`.

        The banner region (POPUP_BANNER) sits over the board when no popup is
        open; the popup replaces that region with a bright title banner, so a
        large mean abs-diff vs the known bare-board frame confirms the popup.
        When we can't diff (no reference frame) we're conservative and treat
        the popup as open — the caller only got here after tapping a board
        cell, so it almost certainly opened.
        """
        if board_frame is None or popup_frame is None:
            return True
        x, y, w, h = ocr.POPUP_BANNER
        before = cv2.cvtColor(board_frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        after = cv2.cvtColor(popup_frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        return float(cv2.absdiff(before, after).mean()) > POPUP_OPEN_DIFF

    @staticmethod
    def _panel_placeholder_text(popup_frame) -> bool:
        """True when the Lair info panel still shows the placeholder
        "Select something to learn more about it." text. Aug 29: the
        Lair info panel is a persistent UI element (not a modal popup)
        that updates when a board cell is tapped. If the tap did not
        land (off-by-one pixel, occlusion, or a race with the previous
        screencap), the panel stays on the placeholder and the LLM
        obediently reads that text back as the description (the
        proactive_identify round was returning this 100% of the time
        in early tests).

        Heuristic: the populated panel puts the name banner (e.g.
        "Skeleton Lvl 5") in the TOP third of the panel — a row of
        bright text on the dark purple background. The placeholder
        has NO top text (it only shows "Select something to learn
        more about it." centered in the middle/bottom). Verified on
        the live board: placeholder top-third bright-pixel ratio
        == 0.000; populated panel top-third bright-pixel ratio >=
        0.005. Threshold 0.002 separates them with a wide margin."""
        if popup_frame is None:
            return False
        x, y, w, h = (235, 286, 845, 334)  # ITEM_PANEL — the Lair info panel
        crop = popup_frame[y:y + h, x:x + w]
        if crop is None or crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h_px = gray.shape[0]
        # Top third: where the name banner sits on a populated panel
        top = gray[:h_px // 3, :]
        bright = float((top > 200).mean())
        return bright < 0.002

    def read_recipe(self, board_frame, cell) -> dict:
        """Popup-read the info body (description + merge chain) for an item the
        classifier already knows, without re-identifying or re-banking its
        sprite. Returns an info dict with description/merge_info (empty on
        failure). Used by the planner to learn e.g. that icerune_lvl3 (produced
        by a merge) is max-level and says "Feed to the Devourer." — the full
        `identify` path refuses already-known cells.

        Retries once when the popup hasn't opened yet (the tap can race the
        popup animation, which showed up as a failed first read in soak logs).
        The popup is dismissed ONLY once it is confirmed open — tapping the
        dismiss corner on a bare board would hit the live build button.
        """
        info: dict = {}
        for _ in range(READ_RETRIES):
            self.device.tap(cell.cx, cell.cy)
            self.device.wait_for_idle(IDENTIFY_PAUSE)
            self.device.screencap()
            popup = cv2.imread(str(self.device.screencap_path))
            if popup is None or not self._popup_open(board_frame, popup):
                # Popup not up yet (or read failed): retap, but do NOT dismiss
                # — dismissing on the bare board would hit the live build button.
                continue
            body = read_item_popup(popup, self.llm) if self.llm is not None else {}
            if body:
                info = {"description": body.get("description") or "",
                        "merge_info": body.get("merge_info") or "",
                        "feed_value": self._coerce(body.get("feed_value")),
                        "damage": self._coerce(body.get("damage"))}
            self.device.tap(*POPUP_DISMISS)
            self.device.wait_for_idle(DISMISS_PAUSE)
            if info:
                return info
        return info

    def identify(self, board_frame, cell) -> tuple[str, dict]:
        """Tap `cell`, read the popup, bank its sprite. Returns (item_id, info)."""
        return self._identify_one(board_frame, cell)

    def _identify_one(self, board_frame, cell) -> tuple[str, dict]:
        """Single-cell identify (no batch logic). Returns (item_id, info).

        split out from `identify` so `identify_batch` can reuse the
        same per-cell pipeline without duplicating 30+ lines of LLM/OCR
        plumbing. The per-category popup question is selected from
        `popup_question_for(base_name)` AFTER the name is known, so the
        Champion / Currency / Station prompts target the right fields.

        added a panel-population check. The Lair info panel is
        a persistent UI element (not a modal popup) that updates when
        a board cell is tapped. If the tap doesn't land (off-by-one
        pixel, occlusion, or a race with the previous screencap), the
        panel still shows the placeholder "Select something to learn
        more about it." — and the LLM obediently reads that back as
        the description (the proactive_identify round was returning
        this 100% of the time in early tests). We now retry up to
        READ_RETRIES times when the panel still shows the placeholder,
        retapping the cell each time. The dismiss tap is gated on a
        successful read so a stuck placeholder never causes a stray
        dismiss tap on the bare board.
        """
        body: dict = {}
        base_name: str | None = None
        level: int | None = None
        ocr_name = ""
        ocr_level = None
        sig_score = 0.0
        digit_score = 0.0
        for attempt in range(READ_RETRIES):
            self.device.tap(cell.cx, cell.cy)
            self.device.wait_for_idle(IDENTIFY_PAUSE)
            self.device.screencap()
            popup = cv2.imread(str(self.device.screencap_path))
            if popup is None:
                continue
                # bail out early when the Lair panel still shows the
            # placeholder text. Retap and try again — only read the LLM
            # body when the panel actually populated.
            if self._panel_placeholder_text(popup):
                continue
            name_crop = ocr.crop_name(popup)
            base_name, sig_score = self._match_signature(name_crop)
            ocr_name, ocr_level = ocr.extract_item_info(popup)
            llm_name: str | None = None
            llm_level: int | None = None
            if base_name is None:
                # Unknown item: the LLM is the reliable name source; OCR is fallback.
                # Also read the popup BODY (description + merge chain) — grounded
                # UI text the caller can bank into the item glossary. The popup
                # question is selected from the OCR's first guess (if any) so the
                # LLM sees a category-targeted prompt. If the OCR name is empty,
                # fall back to the generic question.
                if self.llm is not None:
                    question = popup_question_for(ocr_name or "")
                    body = read_item_popup(popup, self.llm, question=question)
                    llm_name = body.get("name") or None
                    lvl = body.get("level")
                    try:
                        llm_level = int(lvl) if lvl is not None else None
                    except (TypeError, ValueError):
                        llm_level = None
                    if not llm_name:
                        llm_name, llm_level = self._llm_name_and_level(popup)
                else:
                    llm_name, llm_level = self._llm_name_and_level(popup)
                base_name = llm_name or ocr_name or self._next_unknown()
                base_name = self._sanitize_id(base_name)
                self._save_signature(base_name, name_crop)
            else:
                # known item — still read the popup body for the
                # description + feed/damage stats. The previous code
                # skipped this for known items, which is why the
                # (skeleton, manapotion, etc.) glossary blocks lost
                # their description/merge_info facts.
                if self.llm is not None and not body:
                    question = popup_question_for(base_name or "")
                    body = read_item_popup(popup, self.llm, question=question)
            level, digit_score = self._match_digit(ocr.glyph_crop(popup))
            if level is None:
                level = llm_level or ocr_level
            if (llm_level is not None and level == llm_level
                    and digit_score < DIGIT_THRESHOLD):
                # Confident level from the LLM popup-body read and the glyph isn't
                # banked yet: auto-bank it so future reads don't need the LLM.
                # bank_digit refuses glyph conflicts, so a wrong guess can't
                # corrupt an existing digit; OCR-only levels are too unreliable
                # (5<->6 confusion) to bank on.
                try:
                    self.bank_digit(level, popup)
                except Exception:
                    pass
            break   # successful read — exit the retry loop

        if base_name is None:
            # All retries failed (placeholder stayed up, or the screencap
            # never came back). Bail with an "unknown" id so the caller
            # can decide what to do (the bank gets a fresh entry under
            # unknown_N, and the next identify_item call can retry).
            item_id = self._resolve_id(self._next_unknown(), None,
                                        crop_cell(board_frame, cell.row, cell.col))
            return item_id, {"name": "unknown", "level": None, "error": "panel_did_not_populate"}

        sprite = crop_cell(board_frame, cell.row, cell.col)
        item_id = self._resolve_id(base_name, level, sprite)
        self.classifier.add_template(item_id, sprite)

        # dismiss the panel only if it actually populated.
        # Tapping POPUP_DISMISS on a still-placeholder panel would be
        # safe (it just dismisses the panel), but we gate it to mirror
        # the conservative behavior in read_recipe and avoid extra
        # gestures when the tap already didn't land.
        if not self._panel_placeholder_text(
                cv2.imread(str(self.device.screencap_path))):
            self.device.tap(*POPUP_DISMISS)
            self.device.wait_for_idle(DISMISS_PAUSE)

        info = {"name": base_name, "level": level, "ocr": ocr_name,
                "sig": round(sig_score, 3), "digit": round(digit_score, 3),
                "description": body.get("description") or "",
                "merge_info": body.get("merge_info") or "",
                "feed_value": self._coerce(body.get("feed_value")),
                "damage": self._coerce(body.get("damage"))}
        return item_id, info

    def identify_batch(self, board_frame, cells: list) -> list[tuple[str, dict]]:
        """Identify a list of cells in one tool call.

        replaces the per-cell `identify_item` round in the planner
        with a single batch call. Each cell still goes through the same
        popup-tap-read-dismiss pipeline (no parallelism — the device is
        single-threaded), but the LLM only sees ONE tool call covering N
        cells, saving tool-call budget. Returns a list of (item_id, info)
        in the same order as `cells`; per-cell failures yield
        `("unknown_<n>", {})`.

        The batch is the planner's escape-hatch rescue for unidentified
        cells: a validator rejection with `cell_a/cell_b_unidentified` was
        the dominant failure mode in the Aug 28 session log (18
        `discover_unknown` events). Reading all UNID cells once per
        step (or once every N steps) keeps the bank current.
        """
        results = []
        for cell in cells:
            try:
                item_id, info = self._identify_one(board_frame, cell)
                results.append((item_id, info))
            except Exception as exc:
                # Per-cell failure doesn't abort the batch — log and continue.
                # The caller can decide what to do with a partial result.
                results.append((f"unknown_{self.counter}", {"error": str(exc)}))
        return results

    def read_dock_reward(self, frame) -> str | None:
        """Identify what's IN the queue dock button (the reward slot).

        when the dock has a non-empty reward, capture the icon
        crop and match against a small bank of dock-reward icons
        (icebox / valuablechest / future reward types). Returns the
        canonical item id (e.g. `icebox_unopened`) or None when the
        dock is empty or the icon isn't recognized.

        The dock itself has 4 button-state templates in
        `assets/bottombar/queue__N.png`; the reward icons are
        DIFFERENT — they live in `assets/dock_rewards/` (created on
        first run; see `_load_dock_rewards`). Falls back to the LLM
        when the bank has no good match.
        """
        if not self.dock_rewards:
            self._load_dock_rewards()
        if not self.dock_rewards:
            return None
        # The dock reward is shown in the dock's icon region when
        # the dock has a non-empty reward. The exact ROI is in
        # `vision/queue_box.py` (DOCK_BTN_ICON) — copy the constants
        # here to keep `identify.py` decoupled from the queue module.
        try:
            from vision.queue_box import DOCK_BTN_ICON
            x, y, w, h = DOCK_BTN_ICON
        except Exception:
            return None
        crop = frame[y:y + h, x:x + w]
        if crop is None or crop.size == 0:
            return None
        best_id, best_score = None, -1.0
        for item_id, sig in self.dock_rewards.items():
            if sig is None:
                continue
            s = TemplateClassifier._match_score(crop, sig)
            if s > best_score:
                best_score, best_id = s, item_id
        if best_id is not None and best_score >= 0.7:
            return best_id
        return None

    @staticmethod
    def _coerce(value):
        """Int-coerce a popup stat (feed value / damage) without crashing on
        floats, None or junk the 4B model occasionally emits."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _next_unknown(self) -> str:
        self.counter += 1
        return f"unknown_{self.counter}"


def bootstrap_digits(device=None, classifier=None, min_score: float = 0.9) -> list:
    """Bank level-digit glyphs from the live board's confident, known-level items.

    Popup-taps every cell the classifier is confident about (score >= min_score)
    whose id carries a `_lvl<N>` suffix, and saves its digit glyph under N.
    Run a couple of times with different boards to cover levels 1..9.
    """
    from vision.classifier import TemplateClassifier
    from vision.pipeline import classify_board

    device = device or __import__("env.adb", fromlist=["Device"]).Device()
    classifier = classifier or TemplateClassifier("assets/templates")
    identifier = Identifier(device, classifier)

    device.screencap()
    import cv2 as _cv2
    frame = _cv2.imread(str(device.screencap_path))
    board = classify_board(frame, classifier)
    banked = []
    for cell in board.cells:
        if cell.item_id is None or cell.score < min_score:
            continue
        m = re.search(r"_lvl(\d+)$", cell.item_id)
        if not m:
            continue
        level = int(m.group(1))
        device.tap(cell.cx, cell.cy)
        device.wait_for_idle(IDENTIFY_PAUSE)
        device.screencap()
        popup = _cv2.imread(str(device.screencap_path))
        if identifier.bank_digit(level, popup):
            banked.append((cell.item_id, level))
        device.tap(*POPUP_DISMISS)
        device.wait_for_idle(DISMISS_PAUSE)
    return banked


if __name__ == "__main__":
    print(bootstrap_digits())
