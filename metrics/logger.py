"""Session metrics."""

import json
import time
from pathlib import Path


class SessionLog:
    def __init__(self, path: Path = Path("session.jsonl")):
        self.path = path
        self.events: list[dict] = []

    def log(self, event: str, **fields) -> None:
        entry = {"t": time.time(), "event": event, **fields}
        self.events.append(entry)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def count(self, event: str) -> int:
        return sum(1 for e in self.events if e["event"] == event)
