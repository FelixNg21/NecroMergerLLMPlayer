"""ADB interface to the emulator. Implement the raw actions here."""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path


def _find_adb() -> str:
    found = shutil.which("adb")
    if found:
        return found
    candidates = [
        Path.home() / "Library/Android/sdk/platform-tools/adb",
        Path(os.environ.get("ANDROID_HOME", "")) / "platform-tools/adb",
        Path(os.environ.get("ANDROID_SDK_ROOT", "")) / "platform-tools/adb",
        Path("/opt/homebrew/bin/adb"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("adb not found on PATH or in common SDK locations")


class Device:
    # NecroMerger's Unity activity. Relaunching with am start is safe whether
    # the app is running (brings the task to front) or has exited (cold start).
    NECROMERGER_PACKAGE = "com.grumpyrhinogames.necromerger"
    NECROMERGER_ACTIVITY = "com.google.firebase.MessagingUnityPlayerActivity"

    def __init__(self, serial: str = "", screencap_path: Path | None = None, adb_bin: str | None = None):
        self.serial = serial
        self.screencap_path = screencap_path or Path("/tmp/screencap.png")
        self._adb_bin = adb_bin or _find_adb()

    def _adb(self, *args: str) -> bytes:
        cmd = [self._adb_bin]
        if self.serial:
            cmd += ["-s", self.serial]
        cmd += list(args)
        return subprocess.run(cmd, capture_output=True, check=True).stdout

    def screencap(self) -> bytes:
        """Return a raw PNG of the screen."""
        png = self._adb("exec-out", "screencap", "-p")
        self.screencap_path.write_bytes(png)
        return png

    def tap(self, x: int, y: int) -> None:
        self._adb("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self._adb(
            "shell", "input", "swipe",
            str(x1), str(y1), str(x2), str(y2), str(duration_ms),
        )

    def back(self) -> None:
        """Send the BACK key. Safe ONLY when a dialog/panel is confirmed open
        (it dismisses it); on the bare board it exits the game (verified).
        Callers must verify the board returned afterwards."""
        self._adb("shell", "input", "keyevent", "4")

    def relaunch(self) -> None:
        """(Re)start NecroMerger via am start. Safe whether or not the app is
        running: an already-running task is foregrounded, a dead one cold-starts.
        After a cold start the app lands on a dark title screen (a green
        continue block, centroid ~(654,1570)) — callers must tap that to reach
        the lair and verify the floor blob returns."""
        self._adb("shell", "am", "start", "-n",
                  f"{self.NECROMERGER_PACKAGE}/{self.NECROMERGER_ACTIVITY}")
        time.sleep(3)

    def top_resumed_activity(self) -> str | None:
        """Package name of the current foreground (top-resumed) activity.

        This is the AUTHORITATIVE 'is the game alive' signal: when NecroMerger
        is showing ANY of its own screens (lair, menu, champion screen, feats
        panel, popup) it is the topResumedActivity; when the game exits, the
        Android launcher (or another app) takes that slot. Returns None when
        the info can't be read (emulator offline, dumpsys failure) — callers
        treat that as "can't confirm", never as "game exited"."""
        try:
            txt = self._adb("shell", "dumpsys", "activity", "activities").decode("utf-8", "replace")
        except subprocess.CalledProcessError:
            return None
        m = re.search(r"topResumedActivity=ActivityRecord\{\S+ \S+ (?P<pkg>[\w.]+)/", txt)
        return m.group("pkg") if m else None

    def wait_for_idle(self, seconds: float = 0.5) -> None:
        """TODO: capture two screenshots and return when they match."""
        time.sleep(seconds)
