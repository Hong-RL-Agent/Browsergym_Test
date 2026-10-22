"""Deterministic coverage-aware episode termination policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CoverageTerminationConfig:
    max_steps: int = 60
    max_runtime_seconds: float = 180.0
    stagnation_window: int = 8


class CoverageTerminationTracker:
    def __init__(self, config: CoverageTerminationConfig | None = None) -> None:
        self.config = config or CoverageTerminationConfig()
        self.steps = 0
        self.stagnant_steps = 0
        self.history: list[dict[str, Any]] = []

    def observe(self, progress: Mapping[str, Any], *, pending_candidate: bool = False, active_reproduction: bool = False, pending_network: bool = False) -> dict[str, Any]:
        self.steps += 1
        meaningful = any(bool(progress.get(key)) for key in ("new_state", "new_flow", "new_target", "new_evidence"))
        self.stagnant_steps = 0 if meaningful else self.stagnant_steps + 1
        protected = bool(pending_candidate or active_reproduction or pending_network)
        reason = ""
        if self.steps >= self.config.max_steps:
            reason = "MAX_STEPS"
        elif self.stagnant_steps >= self.config.stagnation_window and not protected:
            reason = "COVERAGE_STAGNATION"
        event = {"step": self.steps, "meaningful_progress": meaningful, "stagnant_steps": self.stagnant_steps, "protected": protected, "termination_reason": reason}
        self.history.append(event)
        return event

    def reset(self) -> None:
        self.steps = 0
        self.stagnant_steps = 0
        self.history.clear()
