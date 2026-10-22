"""Deterministic replay for candidates with complete runtime provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from dqn_error_discovery.hybrid.strategy_router import RULE_BASED_REPLAY, route_candidate


@dataclass(frozen=True)
class ReplayPlan:
    candidate_key: str
    chain_id: str
    action_type: str
    target_identity: str
    option_identity: Mapping[str, Any] | None
    source: str = RULE_BASED_REPLAY

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "candidate_key": self.candidate_key,
            "chain_id": self.chain_id,
            "action_type": self.action_type,
            "target_identity": self.target_identity,
            "option_identity": dict(self.option_identity or {}) or None,
        }


class RuleBasedReplayer:
    """Build and optionally execute an exact action supplied by runtime evidence."""

    def plan(self, candidate: Mapping[str, Any]) -> ReplayPlan | None:
        route = route_candidate(candidate)
        if route.owner != RULE_BASED_REPLAY:
            return None
        return ReplayPlan(
            candidate_key=str(candidate.get("candidate_logical_key") or candidate.get("logical_key") or ""),
            chain_id=str(candidate.get("chain_id") or ""),
            action_type=str(candidate.get("reproduction_action_type") or ""),
            target_identity=str(candidate.get("reproduction_target_identity") or ""),
            option_identity=candidate.get("reproduction_option_identity") or candidate.get("option_identity"),
        )

    def execute(self, candidate: Mapping[str, Any], executor: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> dict[str, Any]:
        plan = self.plan(candidate)
        if plan is None:
            return {"executed": False, "reason": "not-exact-replayable", "source": RULE_BASED_REPLAY}
        replay_action = {
            "action_type": plan.action_type,
            "target_identity": plan.target_identity,
            "option_identity": dict(plan.option_identity or {}) or None,
            "candidate_logical_key": plan.candidate_key,
            "chain_id": plan.chain_id,
            "source": RULE_BASED_REPLAY,
        }
        result = dict(executor(replay_action) or {})
        result.update({"executed": True, "source": RULE_BASED_REPLAY, "plan": plan.as_dict()})
        return result
