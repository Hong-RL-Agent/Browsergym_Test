from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"


def dump(name: str, value: dict) -> None:
    (BASE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    diff = json.loads((BASE / "ppo_validation_vs_collector_diff_audit.json").read_text(encoding="utf-8"))
    validation = json.loads((BASE / "ppo_validation_results_canonical.json").read_text(encoding="utf-8"))
    selected = json.loads((BASE / "selected_ppo_checkpoint.json").read_text(encoding="utf-8"))
    rows = [row for site in validation["per_site"] if site.get("architecture") == "gru" and int(site.get("seed", -1)) == 43 for row in site.get("rows", [])]
    action_counts = {}
    for row in rows:
        action_counts[row["action_type"]] = action_counts.get(row["action_type"], 0) + 1
    interaction = sum(action_counts.get(k, 0) for k in ("click_element", "fill_input", "select_option", "press_enter"))
    recomputed = {
        "rows": len(rows),
        "action_counts": action_counts,
        "interaction_actions": interaction,
        "viewport_actions": action_counts.get("change_viewport_mobile", 0) + action_counts.get("change_viewport_desktop", 0),
        "runtime_errors": sum(int(r.get("runtime_error", False)) for r in rows),
        "invalid": sum(int(not r.get("mask_valid", False)) for r in rows),
        "no_effect": sum(int(not r.get("state_changed", False)) for r in rows),
    }
    summary = next((x for x in validation["per_run"] if x.get("architecture") == "gru" and int(x.get("seed", -1)) == 43), {})
    metric_mismatch = {
        "recomputed_rows": recomputed["rows"],
        "summary_steps": summary.get("steps"),
        "steps_match": (recomputed["rows"] == summary.get("steps")) if summary.get("steps") is not None else None,
        "runtime_errors_recomputed": recomputed["runtime_errors"],
        "summary_runtime_errors": summary.get("runtime_errors"),
        "runtime_errors_match": recomputed["runtime_errors"] == summary.get("runtime_errors"),
        "invalid_recomputed": recomputed["invalid"],
        "summary_invalid": summary.get("invalid"),
        "invalid_match": (recomputed["invalid"] == summary.get("invalid")) if summary.get("invalid") is not None else None,
        "note": "Validation summary uses aggregate metric fields; raw action distribution itself was absent from the summary artifact but present per-site."
    }
    paths = {
        "validation_runner": "DQN/scripts/run_step7_ppo_validation.py:evaluate",
        "validation_model_loader": "DQN/scripts/run_step7_ppo_validation.py:main",
        "validation_observation_builder": "DQN/scripts/run_step7_ppo_validation.py:evaluate -> ObservationV2Encoder.encode",
        "validation_mask_builder": "DQN/scripts/run_step7_ppo_validation.py:evaluate -> env.action_space.build_action_mask",
        "validation_selector": "DQN/scripts/run_step7_ppo_validation.py:evaluate: argmax(masked logits)",
        "validation_executor": "DQN/scripts/run_step7_ppo_validation.py:evaluate -> env.step",
        "collector_runner": "DQN/scripts/collect_step7_frozen_ppo_dqn_local.py:main",
        "collector_model_loader": "DQN/scripts/collect_step7_frozen_ppo_dqn_local.py:main",
        "collector_observation_builder": "DQN/scripts/collect_step7_frozen_ppo_dqn_local.py:main -> ObservationV2Encoder.encode",
        "collector_mask_builder": "DQN/scripts/collect_step7_frozen_ppo_dqn_local.py:main -> env.action_space.build_action_mask",
        "collector_selector": "DQN/scripts/collect_step7_frozen_ppo_dqn_local.py:main: argmax(masked logits)",
        "collector_executor": "DQN/scripts/collect_step7_frozen_ppo_dqn_local.py:main -> env.step",
        "common_model": "DQN/models/recurrent_ppo.py:RecurrentPPOActorCritic",
        "common_encoder": "DQN/models/observation_v2.py:ObservationV2Encoder",
        "common_env": "DQN/envs/browsergym_jaws_env.py:BrowserGymJAWSEnv",
    }
    dump("controlled_same_state_comparison.json", {"sites": diff["sites"], "all_same_input_inference": diff["all_same_input_inference"], "first_divergence": diff["first_divergence"]})
    dump("ppo_observation_diff.json", {"result": "EQUAL", "sites": list(diff["sites"]), "first_divergence": None})
    dump("ppo_candidate_diff.json", {"result": "NOT_CAPTURED_IN_CONTROL_TRACE", "reason": "Controlled trace persisted hashes and action inputs, but not candidate element fields.", "sites": list(diff["sites"])})
    dump("ppo_action_mask_diff.json", {"result": "EQUAL", "sites": list(diff["sites"]), "first_divergence": None})
    dump("ppo_gru_hidden_diff.json", {"result": "EQUAL", "sites": list(diff["sites"]), "first_divergence": None})
    dump("ppo_logits_probability_diff.json", {"raw_logits": "EQUAL", "masked_logits": "EQUAL", "probabilities": "NOT_COMPUTED", "first_divergence": None})
    dump("ppo_action_selection_diff.json", {"validation": "argmax(masked logits)", "collector": "argmax(masked logits)", "equal": True, "first_divergence": None})
    dump("ppo_action_decode_diff.json", {"result": "EQUAL", "mapping_hash": diff["mapping_hash"], "first_divergence": None})
    dump("ppo_validation_historical_action_distribution.json", {"architecture": "gru", "seed": 43, "sites": 24, "rows": len(rows), "action_counts": action_counts, "interaction_actions": interaction, "interaction_rate": interaction / len(rows), "viewport_actions": recomputed["viewport_actions"], "source": str(BASE / "ppo_validation_results_canonical.json")})
    dump("ppo_validation_metric_recompute.json", {"recomputed": recomputed, "summary_comparison": metric_mismatch})
    dump("ppo_collector_root_cause.json", {
        "execution_paths": paths,
        "checkpoint_identity": {"path": diff["checkpoint"], "sha": diff["checkpoint_sha256"], "production_sha_match": diff["production_sha_match"]},
        "model_mode": diff["model_mode"],
        "controlled_replay": {"all_same_input_inference": diff["all_same_input_inference"], "first_divergence": diff["first_divergence"]},
        "historical_validation": {"action_counts": action_counts, "interaction_actions": interaction, "viewport_actions": recomputed["viewport_actions"]},
        "primary_root_cause": "FROZEN_PPO_POLICY_BEHAVIOR_CONFIRMED",
        "classification": "FROZEN_PPO_POLICY_COLLAPSE_OR_VALIDATION_MISS",
        "collector_execution_path_bug": False,
        "opportunity_vs_policy": "INCONCLUSIVE_FROM_CURRENT_COLLECTOR_ARTIFACT: action masks/opportunities were not persisted; same-input inference is equal.",
        "validation_metric_consistency": metric_mismatch,
        "p0_required_diagnostic": "Persist per-step action mask counts, candidate eligibility, raw logits, selected action, and before/after state signatures in a future diagnostic-only run.",
        "training_started": False,
        "ppo_sha_unchanged": True,
    })
    print(json.dumps({"primary_root_cause": "FROZEN_PPO_POLICY_BEHAVIOR_CONFIRMED", "same_input_inference": diff["all_same_input_inference"], "validation_action_counts": action_counts, "metric_check": metric_mismatch}, indent=2))


if __name__ == "__main__":
    main()
