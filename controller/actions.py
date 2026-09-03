"""Turn planner Moves into gestures on the device."""

import subprocess
import time
from dataclasses import dataclass

from env.adb import Device
from planner.agent import Move
from vision.grid import FALLBACK_GEOMETRY, GridGeometry


@dataclass
class Layout:
    devourer_xy: tuple[int, int]
    geom: GridGeometry | None = None

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        geom = self.geom or FALLBACK_GEOMETRY
        return geom.cell_center(row, col)


# Merge drag duration. 500ms was too fast: the swipe didn't register as a
# drag-pickup on this game/emulator (Aug 11: merge swipes silently no-oped,
# board fully static). 1000ms (matching feed) verified live to merge. 1250ms:
# under emulator CPU load (LLM inference on the same M1 Pro) the 1000ms swipe
# increasingly failed to register as a drag-pickup (Aug 12: frequent "merge
# no-op"), so the primary swipe is a bit slower and only the verify retry
# stays slower still.
MERGE_SWIPE_MS = 1250

# A gesture can fail to inject when the emulator is CPU-starved (the LLM server
# and the emulator share the M1 Pro): `adb shell input` intermittently returns a
# non-zero exit (Aug 19 soak: merge swipe exit 20, popup tap exit 224). Retry
# the same gesture a couple of times before surfacing a GestureFailed so a single
# flaky injection doesn't crash the whole run.
GESTURE_RETRIES = 2
GESTURE_RETRY_WAIT_S = 1.0


class GestureFailed(Exception):
    """Raised by execute() when a gesture (tap/swipe) failed to inject after
    GESTURE_RETRIES + 1 attempts. Carries the underlying adb error for logging."""

    def __init__(self, kind: str, coords, code: int | None, stderr: str):
        self.kind = kind
        self.coords = coords
        self.code = code
        self.stderr = stderr
        super().__init__(
            f"gesture {kind} at {coords} failed to inject "
            f"(exit {code}): {stderr.strip()}")


def _with_retry(kind: str, coords, fn) -> None:
    for attempt in range(GESTURE_RETRIES + 1):
        try:
            fn()
            return
        except subprocess.CalledProcessError as exc:
            if attempt >= GESTURE_RETRIES:
                raise GestureFailed(
                    kind, coords, exc.returncode,
                    (exc.stderr or b"").decode("utf-8", "replace")) from exc
            time.sleep(GESTURE_RETRY_WAIT_S)


def execute(device: Device, layout: Layout, move: Move) -> None:
    if move.kind == "merge":
        x1, y1 = layout.cell_center(*move.cell_a)
        x2, y2 = layout.cell_center(*move.cell_b)
        _with_retry("swipe", ((x1, y1), (x2, y2)),
                    lambda: device.swipe(x1, y1, x2, y2, duration_ms=MERGE_SWIPE_MS))
        device.wait_for_idle(0.2)  # wait a bit after merge to avoid missing taps
    elif move.kind == "feed":
        x, y = layout.cell_center(*move.cell_a)
        _with_retry("swipe", ((x, y), layout.devourer_xy),
                    lambda: device.swipe(x, y, *layout.devourer_xy, duration_ms=1000))
    elif move.kind == "attack":
        # Champion combat: drag our creature onto the champion cell. The
        # gesture is the SAME shape as a merge swipe (drag-pickup from
        # cell_a, drop on target), just pointed at the champion instead
        # of an identical neighbor. Same MERGE_SWIPE_MS (verified live for
        # merge pickup) — the game reads both as a "drop on cell".
        if move.target is None:
            raise NotImplementedError("attack move without target")
        x1, y1 = layout.cell_center(*move.cell_a)
        x2, y2 = layout.cell_center(*move.target)
        _with_retry("swipe", ((x1, y1), (x2, y2)),
                    lambda: device.swipe(x1, y1, x2, y2, duration_ms=MERGE_SWIPE_MS))
        device.wait_for_idle(0.2)
    elif move.kind in ("spawn", "collect"):
        x, y = layout.cell_center(*move.cell_a)
        for _ in range(max(1, move.taps)):
            _with_retry("tap", (x, y), lambda: device.tap(x, y))
            device.wait_for_idle(0.2)  # wait a bit between taps to avoid missing taps
    elif move.kind == "idle":
        pass
    else:
        raise NotImplementedError(f"execute: {move.kind}")
