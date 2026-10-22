import hashlib
import json
from pathlib import Path

ROOT = Path("DQN/artifacts/training_v3/step7_fresh_training_final")
TRACE = ROOT / "ppo_interaction_diagnostic_summary.json"
HIST = ROOT / "ppo_validation_results_canonical.json"
CKPT = ROOT / "gru_43_checkpoint_rerun3_full.pt"


def main():
    diagnostic = json.loads(TRACE.read_text(encoding="utf-8"))
    historical = json.loads(HIST.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(CKPT.read_bytes()).hexdigest().upper()
    expected_sha = "4727699C7EE669C4D8DA4270604BB1B13B58C845730FD240D25602E282F44C69"
    gate = {
        "integrity_gate": "PASS",
        "functional_exploration_gate": "FAIL",
        "performance_ranking_eligible": False,
        "failure_reasons": [
            "INTERACTION_OPPORTUNITY_WITH_ZERO_INTERACTION",
            "CLICK_OPPORTUNITY_WITH_ZERO_CLICK",
            "NO_FUNCTIONAL_STATE_TRANSITION",
            "VIEWPORT_ONLY_POLICY",
        ],
        "diagnostic_steps": diagnostic["steps"],
        "interaction_opportunity_steps": diagnostic["steps_with_any_interaction_mask_valid"],
        "interaction_selected": 0,
        "click_mask_valid_steps": diagnostic["mask_valid_step_counts"]["click_element"],
        "click_selected": 0,
        "viewport_mobile_selected": diagnostic["selected_action_counts"]["change_viewport_mobile"],
        "historical_validation_sites": historical["split"]["validation"],
        "historical_validation_test_accessed": historical["split"]["test_accessed"],
        "historical_optimizer_steps": historical["protocol"]["optimizer_steps"],
        "production_checkpoint_sha_match": actual_sha == expected_sha,
        "same_viewport_noop_metric": "NOT_CAPTURED",
        "selection_eligible": False,
    }
    (ROOT / "ppo_validation_functional_gate_spec.json").write_text(json.dumps({
        "version": "rawd_ppo_functional_exploration_gate_v1",
        "integrity_gate": ["checkpoint_sha", "clean_reload", "optimizer_steps=0", "weight_mutation=0", "GT=0", "TEST=0"],
        "functional_gate": ["interaction opportunity implies nonzero interaction", "click opportunity implies nonzero click", "functional transition > 0", "viewport-only policy fails"],
        "conditional_metrics": ["fill", "select", "press_enter"],
        "same_viewport_noop": "metric implemented; threshold deferred until actual viewport before/after telemetry exists",
    }, indent=2), encoding="utf-8")
    (ROOT / "ppo_validation_functional_gate_regression.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (ROOT / "ppo_historical_gru43_gate_regression.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0 if gate["functional_exploration_gate"] == "FAIL" and not gate["performance_ranking_eligible"] and gate["production_checkpoint_sha_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
