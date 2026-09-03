"""Repeatable E2E verification for the gesture-retry + device-recovery robustness
build (the Aug 19 crash: an uncaught `adb shell input` CalledProcessError exit-20
killed the whole run).

Checks:
A) execute() retries a flaky swipe (CalledProcessError x2, then success) and
   returns normally; the swipe was attempted GESTURE_RETRIES+1 times.
B) execute() retries a flaky tap the same way.
C) an always-failing gesture surfaces a GestureFailed carrying kind/coords/exit
   code/stderr after exactly GESTURE_RETRIES+1 attempts.
D) _verify_merge's retry swipe failure is caught -> logs merge_noop attempt=2
   with the exit code and returns the frame (does not raise).
E) _recover_device: game relaunched + already on the lair -> returns True, no
   title-screen tap.
F) _recover_device: relaunch lands on the title screen -> taps TITLE_CONTINUE_XY
   once, then the lair returns -> True.
G) _recover_device: lair never returns -> False after 3 title taps (caller halts).
H) py_compile of the three changed modules.
Run: .venv/bin/python scripts/verify_gesture_retry.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from controller.actions import (  # noqa: E402
    GESTURE_RETRIES,
    GestureFailed,
    Layout,
    _with_retry,
    execute,
)
from env.adb import Device  # noqa: E402
from metrics.logger import SessionLog  # noqa: E402
from planner.agent import HeuristicPlanner, Move  # noqa: E402
from vision.classifier import TemplateClassifier  # noqa: E402
from vision.geometry import FALLBACK_GEOMETRY  # noqa: E402

LAIR_FRAME = str(ROOT / "screenshots" / "calib_board.png")                 # floor blob present
TITLE_FRAME = str(ROOT / "assets" / "calib" / "title_banner.png")          # no floor blob

PASS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if ok:
        PASS += 1


class StubDevice:
    """Device stub. Gestures record calls; tap/swipe can be programmed to fail
    (raise CalledProcessError) for a set number of attempts. screencap replays a
    queue of fixture paths."""

    def __init__(self, screencap_path: Path):
        self.screencap_path = screencap_path
        self.taps = []
        self.swipes = []
        self.relaunches = 0
        self._screencap_queue = []
        self._top_pkg = Device.NECROMERGER_PACKAGE
        self._fail_tap = 0
        self._fail_swipe = 0
        self._swipe_code = 20
        self._tap_code = 224
        self._tap_attempts = 0
        self._swipe_attempts = 0

    def _maybe_raise(self, kind: str) -> None:
        fail = self._fail_tap if kind == "tap" else self._fail_swipe
        if fail > 0:
            if kind == "tap":
                self._fail_tap -= 1
            else:
                self._fail_swipe -= 1
            code = self._tap_code if kind == "tap" else self._swipe_code
            raise subprocess.CalledProcessError(
                code, ["adb", "shell", "input", kind],
                stderr=f"{kind} stderr".encode())

    def screencap(self) -> bytes:
        p = self._screencap_queue.pop(0)
        data = Path(p).read_bytes()
        self.screencap_path.write_bytes(data)
        return data

    def tap(self, x: int, y: int) -> None:
        self._tap_attempts += 1
        self._maybe_raise("tap")
        self.taps.append((x, y))

    def swipe(self, x1, y1, x2, y2, duration_ms=300) -> None:
        self._swipe_attempts += 1
        self._maybe_raise("swipe")
        self.swipes.append((x1, y1, x2, y2, duration_ms))

    def wait_for_idle(self, seconds: float = 0.5) -> None:
        pass

    def back(self) -> None:
        pass

    def relaunch(self) -> None:
        self.relaunches += 1

    def top_resumed_activity(self) -> str | None:
        return self._top_pkg


def make_device() -> tuple[StubDevice, Layout, Move]:
    dev = StubDevice(Path(tempfile.mkdtemp()) / "screencap.png")
    layout = Layout(devourer_xy=(642, 923), geom=FALLBACK_GEOMETRY)
    move = Move(kind="merge", cell_a=(2, 2), cell_b=(3, 0))
    return dev, layout, move


def test_retry_success(dev, layout, move) -> None:
    dev._fail_swipe = 2  # fail twice, succeed on the 3rd attempt
    execute(dev, layout, move)
    check("A retry: flaky merge swipe succeeds after 3 attempts",
          dev._swipe_attempts == GESTURE_RETRIES + 1 and dev.swipes[0][4] == 1250,
          f"attempts={dev._swipe_attempts}")


def test_tap_retry_success(dev, layout, _move=None) -> None:
    dev._fail_tap = 1  # fail once, succeed on the 2nd attempt
    execute(dev, layout, Move(kind="spawn", cell_a=(4, 2)))
    check("B retry: flaky spawn tap succeeds after 2 attempts",
          dev._tap_attempts == 2 and len(dev.taps) == 1,
          f"attempts={dev._tap_attempts}")


def test_always_fail(dev, layout, move) -> None:
    dev._fail_swipe = 999
    try:
        execute(dev, layout, move)
        check("C always-fail: GestureFailed raised", False, "no exception")
    except GestureFailed as exc:
        check("C always-fail: GestureFailed raised with code+stderr",
              exc.code == 20 and "swipe stderr" in exc.stderr and exc.kind == "swipe",
              f"code={exc.code} stderr={exc.stderr!r}")
        check("C2 always-fail: exactly GESTURE_RETRIES+1 attempts",
              dev._swipe_attempts == GESTURE_RETRIES + 1 and not dev.swipes,
              f"attempts={dev._swipe_attempts}")


def test_verify_merge_retry_fail() -> None:
    from main import _verify_merge
    dev, layout, move = make_device()
    dev._fail_swipe = 999
    dev._screencap_queue = [LAIR_FRAME]
    log = SessionLog(Path(tempfile.mkdtemp()) / "session.jsonl")
    classifier = TemplateClassifier(seed=True)
    frame = cv2.imread(LAIR_FRAME)
    pre_ids = ("zombie_lvl2", "zombie_lvl2")  # calib (2,2)+(3,0)
    out = _verify_merge(dev, layout, move, pre_ids, log, classifier, HeuristicPlanner())
    noops = [e for e in log.events if e["event"] == "merge_noop"]
    check("D verify-merge retry-swipe failure caught (merge_noop attempt=2)",
          len(noops) == 2 and noops[1]["attempt"] == 2
          and noops[1]["error"] == "retry swipe failed (exit 20)",
          f"events={[e['event'] for e in log.events]}")
    check("D2 returns the latest frame", out is not None, "")


def test_recover_lair() -> None:
    from main import _recover_device
    dev, _layout, _move = make_device()
    dev._screencap_queue = [LAIR_FRAME]
    log = SessionLog(Path(tempfile.mkdtemp()) / "session.jsonl")
    ok = _recover_device(dev, log)
    check("E recover: lair already showing -> True, relaunched, no title tap",
          ok and dev.relaunches == 1 and dev.taps == [],
          f"ok={ok} relaunches={dev.relaunches} taps={dev.taps}")


def test_recover_title_screen() -> None:
    from main import _recover_device, TITLE_CONTINUE_XY
    dev, _layout, _move = make_device()
    dev._screencap_queue = [TITLE_FRAME, LAIR_FRAME]
    log = SessionLog(Path(tempfile.mkdtemp()) / "session.jsonl")
    ok = _recover_device(dev, log)
    check("F recover: title screen -> tap continue once, lair returns -> True",
          ok and dev.taps == [tuple(TITLE_CONTINUE_XY)],
          f"ok={ok} taps={dev.taps}")


def test_recover_fails() -> None:
    from main import _recover_device, TITLE_CONTINUE_XY
    dev, _layout, _move = make_device()
    dev._screencap_queue = [TITLE_FRAME] * 5
    log = SessionLog(Path(tempfile.mkdtemp()) / "session.jsonl")
    ok = _recover_device(dev, log)
    check("G recover: lair never returns -> False after 3 title taps",
          not ok and dev.taps == [tuple(TITLE_CONTINUE_XY)] * 3,
          f"ok={ok} taps={dev.taps}")


def main() -> None:
    test_retry_success(*make_device())
    test_tap_retry_success(*make_device())
    test_always_fail(*make_device())
    test_verify_merge_retry_fail()
    test_recover_lair()
    test_recover_title_screen()
    test_recover_fails()
    import py_compile
    for m in ("main.py", "controller/actions.py", "env/adb.py"):
        py_compile.compile(str(ROOT / m), doraise=True)
    print("H py_compile clean")
    print(f"\n{PASS}/7 checks passed")


if __name__ == "__main__":
    main()