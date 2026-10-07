"""Opt-in, behavior-neutral action execution tracing for diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class ActionErrorEventRecorder:
    def __init__(self, path: str | Path, *, site: str, method: str, seed: int, canonical_path: str, canonical_error_count: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        self.site = site
        self.method = method
        self.seed = int(seed)
        self.canonical_path = canonical_path
        self.canonical_error_count = int(canonical_error_count)
        self.reset_count = 0

    def record_reset(self, *, episode: int, reason: str, trace_id: str | None = None) -> None:
        self.reset_count += 1
        self._write({
            "event": "reset",
            "reset_id": f"{self.site}:{self.method}:{self.seed}:reset{self.reset_count}",
            "site": self.site,
            "method": self.method,
            "seed": self.seed,
            "episode": episode,
            "reset_reason": reason,
            "trigger_action_trace_id": trace_id,
        })

    def record_step(self, record: Mapping[str, Any]) -> None:
        self._write({"event": "step", **dict(record)})

    def _write(self, value: Mapping[str, Any]) -> None:
        self.handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        self.handle.flush()

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def __enter__(self) -> "ActionErrorEventRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
