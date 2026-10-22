"""Audit what the fresh PPO training telemetry can and cannot prove."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "DQN" / "artifacts" / "training_v3" / "step7_fresh_training_final"
PREFIXES = [
    "fresh4_mlp42_rewardfix", "fresh4_mlp43_rewardfix", "fresh4_mlp44_rewardfix",
    "fresh4_gru42_rewardfix", "fresh4_gru43_rewardfix", "fresh4_gru44_rewardfix",
]


def numeric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def run() -> int:
    result = {"schema_version": "rawd_ppo_training_interaction_reward_audit_v1", "runs": []}
    for prefix in PREFIXES:
        path = OUT / f"{prefix}_training.jsonl"
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("event") == "PPO_REWARD_CONTEXT":
                events.append(item)
        breakdowns = [item.get("reward_breakdown", {}) for item in events]
        history_present = all(isinstance(item.get("history_sizes"), dict) for item in events)
        action_type_present = sum("action_type" in item for item in events)
        selected_action_present = sum("selected_action" in item or "selected_action_id" in item for item in events)
        logits_present = sum(any(key in item for key in ("logits", "raw_logits", "probabilities", "action_probabilities")) for item in events)
        result["runs"].append({
            "run_id": prefix,
            "reward_context_events": len(events),
            "anomaly_input_nonzero": sum(numeric(item.get("anomaly_input_count")) > 0 for item in events),
            "gt_reward_nonzero": sum(numeric(item.get("gt_reward_used")) != 0 for item in events),
            "history_sizes_present": history_present,
            "new_state_positive_events": sum(numeric(item.get("reward_breakdown", {}).get("new_state")) > 0 for item in events),
            "new_functional_state_positive_events": sum(numeric(item.get("reward_breakdown", {}).get("new_functional_state")) > 0 for item in events),
            "new_action_target_positive_events": sum(numeric(item.get("reward_breakdown", {}).get("new_action_target")) > 0 for item in events),
            "candidate_anomaly_positive_events": sum(numeric(item.get("reward_breakdown", {}).get("candidate_anomaly")) > 0 for item in events),
            "flow_progress_positive_events": sum(numeric(item.get("reward_breakdown", {}).get("flow_progress")) > 0 for item in events),
            "repeat_penalty_events": sum(numeric(item.get("reward_breakdown", {}).get("repeat_penalty")) < 0 for item in events),
            "action_type_fields_present": action_type_present,
            "selected_action_fields_present": selected_action_present,
            "logit_probability_fields_present": logits_present,
            "interaction_selection_measurable": action_type_present > 0 or selected_action_present > 0,
            "interaction_logit_measurable": logits_present > 0,
        })
    result["conclusion"] = {
        "reward_context_active": all(x["reward_context_events"] > 0 for x in result["runs"]),
        "fresh_anomaly_context_active": all(x["anomaly_input_nonzero"] > 0 for x in result["runs"]),
        "gt_reward_leakage": sum(x["gt_reward_nonzero"] for x in result["runs"]),
        "history_telemetry_active": all(x["history_sizes_present"] for x in result["runs"]),
        "interaction_selection_directly_measurable": any(x["interaction_selection_measurable"] for x in result["runs"]),
        "interaction_logits_directly_measurable": any(x["interaction_logit_measurable"] for x in result["runs"]),
        "required_next_action": "Add diagnostic-only selected action/mask/logit/functional-transition telemetry, then run a short fresh smoke before another full PPO retraining.",
    }
    (OUT / "ppo_training_interaction_reward_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
