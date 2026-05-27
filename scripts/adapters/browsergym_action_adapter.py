"""J.A.W.S action to BrowserGym action adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class BrowserGymAction:
    action: Any
    executable: bool
    action_type: str
    fallback_action: Any = "noop()"


class BrowserGymActionAdapter:
    def __init__(self, noop_action: Any = "noop()") -> None:
        self.noop_action = noop_action

    def adapt(
        self,
        decoded_action: Mapping[str, Any],
        raw_observation: Optional[Mapping[str, Any]] = None,
    ) -> BrowserGymAction:
        action_type = str(decoded_action.get("action_type", "noop"))
        candidate_index = int(decoded_action.get("candidate_index", 0) or 0)

        if action_type in {"inspect_dom", "inspect_layout", "finish_episode"}:
            return BrowserGymAction(action=None, executable=False, action_type=action_type)

        if action_type == "click_element":
            candidate = self._candidate_at(raw_observation, candidate_index)
            if not candidate:
                return BrowserGymAction(
                    action=self.noop_action,
                    executable=True,
                    action_type="noop",
                    fallback_action=self.noop_action,
                )
            return BrowserGymAction(
                action=self.to_browsergym_action("click_element", candidate),
                executable=True,
                action_type=action_type,
                fallback_action=self.noop_action,
            )

        if action_type == "scroll_down":
            return BrowserGymAction("scroll(0, 500)", True, action_type, self.noop_action)

        if action_type == "scroll_up":
            return BrowserGymAction("scroll(0, -500)", True, action_type, self.noop_action)

        return BrowserGymAction(self.noop_action, True, "noop", self.noop_action)

    def to_browsergym_action(self, action_type: str, candidate: Mapping[str, Any]) -> Any:
        if action_type == "click_element":
            bid = _escape_action_string(str(candidate.get("bid", "")))
            if not bid:
                return self.noop_action
            return f"click('{bid}')"
        return self.noop_action

    @staticmethod
    def _candidate_at(raw_observation: Optional[Mapping[str, Any]], index: int) -> Optional[Mapping[str, Any]]:
        if not raw_observation:
            return None
        candidates = raw_observation.get("candidate_elements", [])
        if not isinstance(candidates, list) or not (0 <= index < len(candidates)):
            return None
        candidate = candidates[index]
        return candidate if isinstance(candidate, Mapping) else None


def _escape_action_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")
