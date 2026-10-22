"""Common evaluation interface for legacy, history-MLP, and GRU PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PPOAblationModelSpec:
    name: str
    observation_schema: str
    recurrent: bool


ABLATION_MODELS = (
    PPOAblationModelSpec("legacy_mlp_ppo", "legacy", False),
    PPOAblationModelSpec("observation_v2_history_mlp_ppo", "rawd_observation_v2_contract_v1", False),
    PPOAblationModelSpec("observation_v2_gru_ppo", "rawd_observation_v2_contract_v1", True),
)


def coverage_metrics(history: Mapping[str, Any]) -> dict[str, int]:
    """Return a model-independent coverage metric snapshot."""
    return {
        "unique_functional_states": len(history.get("episode_seen_states", set()) or set()),
        "unique_flows": len(history.get("episode_seen_flows", set()) or set()),
        "unique_targets": len(history.get("episode_seen_targets", set()) or set()),
        "action_type_coverage": len(history.get("episode_seen_action_types", set()) or set()),
        "state_action_coverage": len(history.get("episode_seen_action_targets", set()) or set()),
        "state_action_outcome_coverage": len(history.get("episode_seen_state_action_outcomes", set()) or set()),
    }


def make_evaluation_record(model_name: str, history: Mapping[str, Any], *, opportunities: int = 0, selections: int = 0, runtime_errors: int = 0) -> dict[str, Any]:
    result = {"model": model_name, **coverage_metrics(history)}
    result.update({"opportunities": int(opportunities), "selections": int(selections), "runtime_errors": int(runtime_errors)})
    result["selection_rate"] = float(selections / opportunities) if opportunities else 0.0
    return result
