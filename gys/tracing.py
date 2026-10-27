"""JSONL trace sinks for TAP, judge, reflector, and GEPA events.

Every model interaction is appended to a per-stream JSONL file under
``runs/<run_id>/`` so a run is fully reconstructable after the fact. Run output
contains real target-model responses, so ``runs/`` is gitignored.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

STREAMS = ("tap", "judge", "reflector", "gepa")


class TraceSink(Protocol):
    """Minimal sink interface so the TAP loop can run without any I/O."""

    def emit(self, stream: str, record: dict[str, Any]) -> None: ...


class NullSink:
    """Discards every record. Used by tests and by library callers."""

    def emit(self, stream: str, record: dict[str, Any]) -> None:
        return None


class JSONLSink:
    """Appends records to ``<run_dir>/<stream>.jsonl``."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, Any] = {}

    def emit(self, stream: str, record: dict[str, Any]) -> None:
        handle = self._handles.get(stream)
        if handle is None:
            handle = open(self.run_dir / f"{stream}.jsonl", "a", encoding="utf-8")
            self._handles[stream] = handle
        payload = {"ts": datetime.now(UTC).isoformat(), **record}
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()

    def write_json(self, name: str, obj: Any) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
