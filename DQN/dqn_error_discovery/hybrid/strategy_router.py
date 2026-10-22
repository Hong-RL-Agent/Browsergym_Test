"""Explicit detector-strategy-to-execution ownership contract.

This module is intentionally deterministic.  It does not inspect ground truth,
choose a browser action, or mutate a learning buffer.  It only decides which
runtime subsystem owns the next verification step for an already-produced
candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

ACTION_REPLAY = "ACTION_REPLAY"
SEQUENCE_REPLAY = "SEQUENCE_REPLAY"
STATE_REVISIT = "STATE_REVISIT"
OBSERVATION_RECHECK = "OBSERVATION_RECHECK"
TEMPORAL_RECHECK = "TEMPORAL_RECHECK"

RULE_REPLAY = "RULE_REPLAY"
RULE_BASED_REPLAY = "RULE_BASED_REPLAY"
RULE_SEQUENCE_REPLAY = "RULE_SEQUENCE_REPLAY"
STATE_RESTORATION_RULE = "STATE_RESTORATION_RULE"
UNRESOLVED = "UNRESOLVED"
DQN_LOCAL = "DQN_LOCAL"
OBSERVATION_OWNER = "OBSERVATION_RECHECK"
TEMPORAL_OWNER = "TEMPORAL_RECHECK"


@dataclass(frozen=True)
class StrategyRoute:
    strategy: str
    owner: str
    dqn_eligible: bool
    requires_action: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "owner": self.owner,
            "dqn_eligible": self.dqn_eligible,
            "requires_action": self.requires_action,
            "reason": self.reason,
        }


def _actionable(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("requires_action_reproduction")
        and candidate.get("reproduction_action_type")
        and candidate.get("reproduction_target_identity")
        and candidate.get("target_actionable", candidate.get("target_resolvable", False))
    )


def _exact_replayable(candidate: Mapping[str, Any]) -> bool:
    """Exact replay requires provenance, not merely an actionable target."""
    if not _actionable(candidate):
        return False
    if candidate.get("exact_action_known", True) is False:
        return False
    if str(candidate.get("reproduction_action_type") or "") == "select_option":
        option = candidate.get("reproduction_option_identity") or candidate.get("option_identity")
        if not isinstance(option, Mapping) or not any(option.get(key) not in (None, "") for key in ("value", "label", "index")):
            return False
    return True


def route_candidate(candidate: Mapping[str, Any]) -> StrategyRoute:
    """Return the sole subsystem allowed to own candidate verification."""
    strategy = str(candidate.get("reproduction_strategy") or candidate.get("strategy") or "").upper()
    if strategy == ACTION_REPLAY:
        if _exact_replayable(candidate):
            return StrategyRoute(strategy, RULE_BASED_REPLAY, False, True, "exact-action-replay")
        if _actionable(candidate):
            return StrategyRoute(strategy, DQN_LOCAL, True, True, "ambiguous-action-replay-fallback")
        return StrategyRoute(strategy, RULE_REPLAY, False, True, "action-replay-not-actionable")
    if strategy == SEQUENCE_REPLAY:
        if candidate.get("exact_sequence_known"):
            return StrategyRoute(strategy, RULE_SEQUENCE_REPLAY, False, False, "exact-sequence-replay")
        if candidate.get("sequence_action_required", True) and candidate.get("sequence_recoverable", True):
            return StrategyRoute(strategy, DQN_LOCAL, True, True, "ambiguous-sequence-local-decision")
        return StrategyRoute(strategy, UNRESOLVED, False, False, "sequence-unrecoverable")
    if strategy == STATE_REVISIT:
        if candidate.get("deterministic_restoration_known"):
            return StrategyRoute(strategy, STATE_RESTORATION_RULE, False, False, "deterministic-state-restoration")
        if candidate.get("restoration_decision_required", True) and candidate.get("restoration_recoverable", True):
            return StrategyRoute(strategy, DQN_LOCAL, True, True, "ambiguous-state-restoration")
        return StrategyRoute(strategy, UNRESOLVED, False, False, "state-restoration-failed")
    if strategy == OBSERVATION_RECHECK:
        return StrategyRoute(strategy, OBSERVATION_OWNER, False, False, "observation-owner")
    if strategy == TEMPORAL_RECHECK:
        return StrategyRoute(strategy, TEMPORAL_OWNER, False, False, "temporal-owner")
    return StrategyRoute(strategy or "UNKNOWN", RULE_REPLAY, False, False, "unknown-strategy-safe-default")


def routing_matrix() -> list[dict[str, Any]]:
    cases = [
        {"reproduction_strategy": ACTION_REPLAY, "requires_action_reproduction": True, "reproduction_action_type": "click_element", "reproduction_target_identity": "bid:1", "target_actionable": True, "exact_action_known": True},
        {"reproduction_strategy": ACTION_REPLAY, "requires_action_reproduction": True, "reproduction_action_type": "click_element", "reproduction_target_identity": "bid:1", "target_actionable": True, "exact_action_known": False},
        {"reproduction_strategy": ACTION_REPLAY, "requires_action_reproduction": True, "reproduction_action_type": "click_element", "reproduction_target_identity": "bid:1", "target_actionable": False},
        {"reproduction_strategy": SEQUENCE_REPLAY},
        {"reproduction_strategy": STATE_REVISIT},
        {"reproduction_strategy": OBSERVATION_RECHECK},
        {"reproduction_strategy": TEMPORAL_RECHECK},
    ]
    return [{"input": case, "route": route_candidate(case).as_dict()} for case in cases]
