"""Atomic file writes for knowledge stores."""
import os
from pathlib import Path


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write text atomically: serialize-first is the caller's job; here,
    tmp-in-same-dir + fsync + os.replace is the commit. Never leaves a
    torn live file; stale tmp self-overwrites next call."""
    if not isinstance(text, str):
        raise ValueError(f"Text to save must be a string, got {type(text)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
