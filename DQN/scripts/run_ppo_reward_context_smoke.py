import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.append(str(ROOT / "DQN/scripts"))
from models.observation_v2 import ObservationV2Encoder
from models.recurrent_ppo import RecurrentPPOActorCritic
from run_ppo_full_training_v1 import episode, seed_all


def main():
    cfg = json.loads((ROOT / "configs/ppo_full_training_v1.json").read_text(encoding="utf-8"))
    sites = json.loads((ROOT / "configs/rl_errorsite_full_training_sites.json").read_text(encoding="utf-8"))["sites"]
    site = dict(next(x for x in sites if x["site_id"] == "site001"), cluster="SMOKE")
    seed_all(43)
    model = RecurrentPPOActorCritic(135, 640, 128)
    telemetry = []
    rows, summary = episode(model, True, site, cfg, ObservationV2Encoder(32), None, training=False, telemetry=telemetry)
    contexts = [e for e in telemetry if e.get("event") == "PPO_REWARD_CONTEXT"]
    policy_steps = [e for e in telemetry if e.get("event") == "PPO_POLICY_STEP"]
    interaction_opportunities = sum(bool(e.get("interaction_opportunity")) for e in policy_steps)
    interaction_selected = sum(bool(e.get("selected_interaction")) for e in policy_steps)
    functional_transitions = sum(bool(e.get("functional_state_changed")) for e in policy_steps)
    viewport_selected = sum(e.get("action_type") in {"change_viewport_mobile", "change_viewport_desktop"} for e in policy_steps)
    result = {
        "mode": "SMOKE_ONLY",
        "production": False,
        "optimizer_steps": 0,
        "rows": len(rows),
        "reward_context_events": len(contexts),
        "policy_step_events": len(policy_steps),
        "interaction_opportunity_steps": interaction_opportunities,
        "interaction_selected": interaction_selected,
        "functional_state_transitions": functional_transitions,
        "viewport_selected": viewport_selected,
        "selected_action_counts": {action: sum(e.get("action_type") == action for e in policy_steps) for action in sorted({e.get("action_type", "") for e in policy_steps})},
        "selected_rank_mean": (sum(float(e.get("selected_rank", 0)) for e in policy_steps) / len(policy_steps)) if policy_steps else 0.0,
        "interaction_selected_rank_mean": (sum(float(e.get("selected_rank", 0)) for e in policy_steps if e.get("selected_interaction")) / interaction_selected) if interaction_selected else None,
        "anomaly_input_counts": [e.get("anomaly_input_count", 0) for e in contexts],
        "history_sizes_present": all(bool(e.get("history_sizes")) for e in contexts),
        "gt_reward_used": sum(float(e.get("gt_reward_used", 0.0)) for e in contexts),
        "runtime_errors": summary.get("runtime_errors", 0),
        "policy_trace_complete": len(policy_steps) == len(rows),
        "pass": bool(rows) and len(contexts) == len(rows) and len(policy_steps) == len(rows) and all(bool(e.get("history_sizes")) for e in contexts) and summary.get("runtime_errors", 0) == 0,
    }
    out = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final/ppo_reward_context_smoke.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    trace = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final/ppo_diagnostic_smoke_after_telemetry_trace.json"
    trace.write_text(json.dumps({"policy_steps": policy_steps, "reward_context": contexts}, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
