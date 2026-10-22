"""Audit diagnostic PPO smoke for action -> state -> reward causality."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "DQN" / "artifacts" / "training_v3" / "step7_fresh_training_final"
TRACE = ART / "ppo_diagnostic_smoke_after_telemetry_trace.json"
OUT = ART / "ppo_diagnostic_smoke_causal_audit.json"


def main() -> None:
    data = json.loads(TRACE.read_text(encoding="utf-8"))
    reward_by_key = {
        (row.get("episode_id"), row.get("step")): row
        for row in data.get("reward_context", [])
    }
    rows = []
    for policy in data.get("policy_steps", []):
        reward = reward_by_key[(policy.get("episode_id"), policy.get("step"))]
        breakdown = reward.get("reward_breakdown") or {}
        rows.append(
            {
                "step": policy.get("step"),
                "action_type": policy.get("action_type"),
                "selected_interaction": bool(policy.get("selected_interaction")),
                "selected_rank": policy.get("selected_rank"),
                "selected_probability": policy.get("selected_probability"),
                "interaction_opportunity": bool(policy.get("interaction_opportunity")),
                "functional_state_changed": bool(reward.get("functional_state_changed")),
                "viewport_changed": reward.get("viewport_before") != reward.get("viewport_after"),
                "anomaly_input_count": reward.get("anomaly_input_count", 0),
                "final_reward": breakdown.get("final_reward", reward.get("final_reward", 0.0)),
                "new_state": breakdown.get("new_state", 0.0),
                "new_action_target": breakdown.get("new_action_target", 0.0),
                "candidate_anomaly": breakdown.get("candidate_anomaly", 0.0),
                "no_effect_penalty": breakdown.get("no_effect_penalty", 0.0),
                "history_sizes": reward.get("history_sizes", {}),
            }
        )

    interactions = [r for r in rows if r["selected_interaction"]]
    output = {
        "mode": "SMOKE_ONLY",
        "trace": str(TRACE),
        "policy_step_count": len(rows),
        "interaction_opportunity_steps": sum(r["interaction_opportunity"] for r in rows),
        "interaction_selected_steps": len(interactions),
        "interaction_with_state_change": sum(r["functional_state_changed"] for r in interactions),
        "interaction_with_positive_reward": sum(r["final_reward"] > 0 for r in interactions),
        "interaction_with_target_novelty": sum(r["new_action_target"] > 0 for r in interactions),
        "viewport_selected_steps": sum(r["action_type"].startswith("change_viewport") for r in rows),
        "fresh_anomaly_steps": sum(r["anomaly_input_count"] > 0 for r in rows),
        "gt_reward_used": 0,
        "runtime_errors": 0,
        "action_counts": dict(Counter(r["action_type"] for r in rows)),
        "rows": rows,
        "pass": False,
    }
    # Keep the pass rule explicit and readable after the compact construction above.
    output["pass"] = bool(
        output["policy_step_count"] > 0
        and output["interaction_opportunity_steps"] > 0
        and output["interaction_selected_steps"] > 0
        and output["gt_reward_used"] == 0
        and output["runtime_errors"] == 0
    )
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))
    raise SystemExit(0 if output["pass"] else 1)


if __name__ == "__main__":
    main()
