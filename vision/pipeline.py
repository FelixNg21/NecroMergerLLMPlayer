from vision import grid
from vision.classifier import ItemClassifier


def _same_family_diff_level(a: str | None, b: str | None) -> bool:
    """Two template ids are same creature family at different levels (e.g.
    skeleton_lvl3 vs skeleton_lvl4). Used by the neighbor-suspect gate (Aug 26
    incident: (2,3) and (3,1) were both mislabeled as skeleton_lvl4 at margin
    0.026-0.028, but the true labels were lvl5 and lvl3 — the bank couldn't
    tell the bob-dipped sprites apart).
    """
    if not a or not b or a == b:
        return False
    base_a, _, _ = a.rpartition("_lvl")
    base_b, _, _ = b.rpartition("_lvl")
    return bool(base_a) and base_a == base_b


def classify_board(frame, classifier: ItemClassifier) -> grid.BoardState:
    geom = grid.detect_grid(frame)
    cells = grid.build_cells(geom)
    for cell, (item, score, margin, ru_id) in zip(cells, classifier.classify_scores(frame, cells)):
        cell.item_id = item
        cell.score = score
        cell.margin = margin
        cell.runner_up_id = ru_id
        occ = grid.occupancy_score(frame, cell.row, cell.col, geom)
        # Occupancy is calibrated on big 5x3 sprites; small items on the 4x3
        # board (e.g. bones) read below OCCUPANCY_THRESHOLD yet are confidently
        # matched by the template bank (high margin). A strong-margin label is
        # trusted over the occupancy signal; occupancy only gates weak matches.
        # The absolute-score floor is the phantom guard: background on a truly
        # EMPTY cell can match a sprite at 0.5-0.65 with a plausible margin, so
        # margin alone can't distinguish it from a real item — require both a
        # strong margin AND a strong absolute score to override the emptiness.
        strong = (item is not None and margin >= grid.LABEL_MIN_MARGIN
                  and score >= grid.PHANTOM_SCORE_FLOOR)
        cell.occupied = occ >= grid.OCCUPANCY_THRESHOLD or strong
        # Same-family-different-level cross-fire (Aug 26). When the top-2 templates
        # are different levels of the same family AND the margin is below
        # SAME_FAMILY_MIN_MARGIN, the cell is in a bob phase the bank can't
        # disambiguate. Wipe the label to unidentified so the model can popup-read
        # the true level rather than mislabel a lvl3 as a lvl4 and try to merge it
        # with another mislabeled cell. The merge gate (NEIGHBOR_MIN_MARGIN) is
        # the second line of defense if the wipe missed.
        if (item is not None and ru_id is not None
                and cell.occupied
                and margin < grid.SAME_FAMILY_MIN_MARGIN
                and _same_family_diff_level(item, ru_id)):
            cell.item_id = None
            cell.score = 0.0
            cell.margin = 0.0
            cell.runner_up_id = None
        if item is not None and not cell.occupied:
            # Template score alone can exceed the label threshold on an EMPTY
            # cell (background matched e.g. zombie_lvl2 at 0.62-0.65) and such
            # matches are low-margin. Occupancy + weak margin = empty, whatever
            # the classifier reports.
            cell.item_id = None
            cell.score = 0.0
            cell.margin = 0.0
            cell.runner_up_id = None
    return grid.BoardState(geom.rows, geom.cols, cells, geometry=geom)