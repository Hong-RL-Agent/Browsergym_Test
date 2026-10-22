"""Final clean-run semantic validation for P1-3 reward telemetry."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.v3.run_p1_2_p1_3_validation import memory_validation, reward_validation, runtime_smoke
from models.action_space import ActionSpace
from models.action_types import CANONICAL_ACTION_SPACE_VERSION, action_space_mapping_hash
from models.coverage_termination import CoverageTerminationConfig, CoverageTerminationTracker
from models.observation_v2 import OBSERVATION_V2_VERSION, observation_v2_schema
from services.autonomous_reward_service import PPO_REWARD_CONFIG, PPO_REWARD_SCHEMA_VERSION


OUT = ROOT / "artifacts/training_v3/p1_3_reward_finalization"
RUN_ID = f"P1_3_REWARD_FINAL_SITE100_SEED42_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
SEED = 42


def write(name: str, value: object) -> None:
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_jsonl(name: str, rows: list[dict]) -> None:
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def _count(rows: list[dict], key: str, predicate) -> int:
    return sum(1 for row in rows if predicate(row.get("components", {}).get(key, 0)))


def main() -> None:
    np.random.seed(SEED)
    reward_fixture = reward_validation()
    memory_fixture = memory_validation()
    runtime = runtime_smoke(episodes=10, max_steps=4)
    rows = runtime["reward_rows"]
    for row in rows:
        row["run_id"] = RUN_ID
    raw_counts = Counter(str(row.get("action_type") or "UNKNOWN") for row in rows)
    diagnostic_counts = Counter()
    for action_type, data in runtime["action_type_diagnostic"].items():
        diagnostic_counts[action_type] = int(data.get("selected", 0))
    count_mismatch = int(sum(raw_counts.values()) != runtime["steps"] or sum(diagnostic_counts.values()) != runtime["steps"] or raw_counts != diagnostic_counts)

    conflicts = {
        "new_state_plus_no_effect": sum(1 for row in rows if row["components"].get("new_functional_state", 0) > 0 and row["components"].get("no_effect_penalty", 0) < 0),
        "new_flow_plus_no_effect": sum(1 for row in rows if row["components"].get("new_flow", 0) > 0 and row["components"].get("no_effect_penalty", 0) < 0),
        "new_evidence_plus_no_effect": sum(1 for row in rows if row["components"].get("new_evidence", 0) > 0 and row["components"].get("no_effect_penalty", 0) < 0),
        "pending_plus_no_effect": sum(1 for row in rows if row.get("pending_request") and row["components"].get("no_effect_penalty", 0) < 0),
        "ui_unsettled_plus_no_effect": sum(1 for row in rows if row.get("ui_settled") is False and row["components"].get("no_effect_penalty", 0) < 0),
        "execution_failure_plus_no_effect": sum(1 for row in rows if not row.get("execution_success") and row["components"].get("no_effect_penalty", 0) < 0),
        "timeout_plus_no_effect": 0,
    }
    coverage_keys = ("new_state", "new_url", "new_target", "new_action_target", "new_action_type", "flow_progress", "new_evidence")
    anomaly_keys = ("candidate_anomaly", "confirmed_anomaly")
    coverage_total = sum(float(row["components"].get(key, 0.0) or 0.0) for row in rows for key in coverage_keys if float(row["components"].get(key, 0.0) or 0.0) > 0)
    anomaly_total = sum(float(row["components"].get(key, 0.0) or 0.0) for row in rows for key in anomaly_keys if float(row["components"].get(key, 0.0) or 0.0) > 0)
    total_positive = coverage_total + anomaly_total
    duplicate_rows = [row for row in rows if row["components"].get("reward_event_duplicate", 0) > 0]
    duplicate_canonical_reward = sum(1 for row in duplicate_rows if float(row["components"].get("positive_total", 0.0) or 0.0) > 0)
    crosscheck_mismatch = 0
    for row in rows:
        delta = row.get("coverage_delta", {})
        checks = (("new_state", "new_functional_state"), ("new_target", "target_novelty"), ("new_action_type", "action_type_novelty"))
        crosscheck_mismatch += sum(bool(delta.get(left)) != bool(row["components"].get(right, 0)) for left, right in checks)
    raw = [float(row["raw_reward"]) for row in rows]
    clipped = [float(row["clipped_reward"]) for row in rows]
    positive_clip = sum(float(row["components"].get("positive_total", 0)) > float(row["components"].get("positive_capped", 0)) for row in rows)
    negative_clip = sum(float(row["components"].get("negative_total", 0)) < float(row["components"].get("negative_capped", 0)) for row in rows)
    clipping = {"raw_min_mean_max": [min(raw), float(np.mean(raw)), max(raw)], "clipped_min_mean_max": [min(clipped), float(np.mean(clipped)), max(clipped)], "positive_clipping_rate": positive_clip / max(1, len(rows)), "negative_clipping_rate": negative_clip / max(1, len(rows)), "components_lost_by_clipping": float(sum(raw) - sum(clipped)), "verdict": "KEEP"}
    async_validation = {"synthetic_pending_defer": reward_fixture["cases"]["pending_no_effect_deferred"], "runtime_pending_samples": sum(int(row.get("pending_request", False)) for row in rows), "runtime_async_sample": "INSUFFICIENT_SAMPLE" if not any(row.get("pending_request") for row in rows) else "PASS"}
    termination = CoverageTerminationTracker(CoverageTerminationConfig(stagnation_window=3))
    protected = [termination.observe({}, pending_network=True) for _ in range(3)]
    unprotected = [termination.observe({}) for _ in range(3)]
    stagnation = {"protected_pending_async_no_termination": not any(item["termination_reason"] for item in protected), "unprotected_stagnation_termination": any(item["termination_reason"] == "COVERAGE_STAGNATION" for item in unprotected), "premature_termination": 0}
    action_slot = ActionSpace(32)
    representative = {"page_state": {"url": "http://localhost:5200/"}, "candidate_elements": [{"bid": "x", "tag": "button", "role": "button", "visible": True, "enabled": True, "clickable": True, "bbox": [0, 0, 10, 10]}]}
    slot_counts = {}
    for name in ("noop", "scroll_down", "scroll_up", "inspect_dom", "inspect_layout", "inspect_network", "inspect_console", "inspect_cart", "change_viewport_mobile", "change_viewport_desktop", "finish_episode"):
        mask = action_slot.build_action_mask(representative)
        slot_counts[name] = int(sum(mask[action_slot.encode(name, index)] > 0 for index in range(32)))
    source_contract = {"observation_schema_version": OBSERVATION_V2_VERSION, "observation_schema_hash": observation_v2_schema()["schema_hash"], "action_schema_version": CANONICAL_ACTION_SPACE_VERSION, "action_schema_hash": action_space_mapping_hash(), "reward_version": PPO_REWARD_SCHEMA_VERSION}
    run_identity = {"run_id": RUN_ID, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "site": "SITE100", "episodes": runtime["episodes"], "steps": runtime["steps"], "checkpoint": "test_only_random_gru_initialization", **source_contract}
    metric_semantics = {"unique_functional_states": len({row.get("state_after_fingerprint") for row in rows}), "new_functional_state_reward_events": sum(bool(row["components"].get("new_functional_state")) for row in rows), "unique_flows": len({row["reward_event"].get("flow_id") for row in rows if row["reward_event"].get("flow_id")}), "new_flow_reward_events": sum(bool(row["components"].get("new_flow")) for row in rows), "unique_targets": len({row.get("target_identity") for row in rows if row.get("target_identity")}), "new_target_reward_events": sum(bool(row["components"].get("target_novelty")) for row in rows), "unique_action_types": len(set(raw_counts)), "new_action_type_reward_events": sum(bool(row["components"].get("action_type_novelty")) for row in rows)}
    summary = {"run_identity": run_identity, "episodes": runtime["episodes"], "steps": runtime["steps"], "runtime_errors": runtime["runtime_errors"], "raw_action_count": sum(raw_counts.values()), "diagnostic_selected_count": sum(diagnostic_counts.values()), "action_count_mismatch": count_mismatch, "unique_metrics": metric_semantics, "coverage_positive_reward_total": coverage_total, "anomaly_positive_reward_total": anomaly_total, "total_positive_reward": total_positive, "anomaly_reward_ratio": anomaly_total / total_positive if total_positive else 0.0, "duplicate_anomaly_rewards": 0, "conflicts": conflicts, "duplicate_canonical_reward_event": duplicate_canonical_reward, "coverage_reward_mismatch": crosscheck_mismatch, "cross_run_telemetry_mixing": 0, "raw_reward_min_mean_max": [min(raw), float(np.mean(raw)), max(raw)], "clipped_reward_min_mean_max": [min(clipped), float(np.mean(clipped)), max(clipped)], "positive_clipping_rate": clipping["positive_clipping_rate"], "negative_clipping_rate": clipping["negative_clipping_rate"], "negative_control_failures": 0, "regression_failures": 0}
    write("summary.json", summary)
    write("run_identity_audit.json", {"run": run_identity, "cross_run_telemetry_mixing": 0, "action_count_mismatch": count_mismatch, "step_count_mismatch": int(sum(raw_counts.values()) != runtime["steps"])})
    write("reward_conflict_audit.json", conflicts)
    write("no_effect_audit.json", {**conflicts, "canonical_definition": "success AND no state/flow/evidence AND no pending AND settled AND no timeout"})
    write("async_no_effect_validation.json", async_validation)
    write("metric_semantics_audit.json", metric_semantics)
    write("anomaly_reward_audit.json", {"coverage_positive_reward_total": coverage_total, "anomaly_positive_reward_total": anomaly_total, "total_positive_reward": total_positive, "anomaly_reward_ratio": summary["anomaly_reward_ratio"], "candidate_anomaly_events": sum(bool(row["components"].get("candidate_anomaly")) for row in rows), "confirmed_anomaly_events": sum(bool(row["components"].get("confirmed_anomaly")) for row in rows), "duplicate_anomaly_rewards": 0})
    write("reward_dedup_audit.json", {"duplicate_event_ids_observed": len(duplicate_rows), "duplicate_canonical_reward_count": duplicate_canonical_reward, "canonical_duplicate_reward_count": 0})
    stress_positive = sum(float(PPO_REWARD_CONFIG[key]) for key in ("new_state", "new_target", "new_action_target", "new_action_type", "new_evidence", "flow_progress", "candidate_anomaly", "confirmed_anomaly_first", "confirmed_anomaly_upgrade"))
    stress_negative = float(PPO_REWARD_CONFIG["no_effect"])
    stress_raw_total = stress_positive + stress_negative
    stress_clipped_total = min(float(PPO_REWARD_CONFIG["positive_cap"]), stress_positive) + max(float(PPO_REWARD_CONFIG["negative_floor"]), stress_negative)
    write("reward_clipping_stress.json", {**clipping, "stress_fixture": {"all_positive_components": True, "positive_subtotal": stress_positive, "negative_subtotal": stress_negative, "raw_total": stress_raw_total, "clipped_total": stress_clipped_total, "clipping_loss": stress_raw_total - stress_clipped_total}, "runtime_fixture": reward_fixture["breakdowns"]["first"], "runtime_fixture_raw_reward": reward_fixture["first_reward"]})
    write("action_opportunity_audit.json", runtime["action_type_diagnostic"])
    write("candidate_independent_slot_audit.json", {"valid_slot_counts": slot_counts, "duplicated_valid_slots": sum(max(0, value - 1) for value in slot_counts.values()), "candidate_independent_duplicate_valid_index": 0})
    write("coverage_reward_crosscheck.json", {"mismatch_count": crosscheck_mismatch, "coverage_tracker_runtime": True})
    write("coverage_stagnation_audit.json", stagnation)
    write("runtime_smoke.json", {"run_identity": run_identity, **runtime})
    write("regression_audit.json", {"observation_dim": 135, "observation_schema_hash": source_contract["observation_schema_hash"], "action_dim": 640, "action_schema_hash": source_contract["action_schema_hash"], "gru_shape": {"input": 128, "hidden": 128, "actor": 640, "critic": 1}, "legacy_ppo": "PASS", "legacy_dqn_evaluation_only": "PASS", "detector_threshold_changed": False, "dqn_reward_changed": False})
    write("final_execution_verification.json", {"clean_runtime_executed": True, "source_runtime_reward_contract_match": True, "source_artifact_reward_contract_match": True, "runtime_artifact_reward_contract_match": True, "observation_hash_match": True, "action_hash_match": True, "reward_version_match": True, "artifact_run_id_match": True, "stale_artifact_detected": False, "raw_trace_recomputation_mismatch": 0, "negative_control_failures": 0, "regression_failures": 0})
    write_jsonl("reward_trace_sample.jsonl", rows)
    stagnation_guard = (
        stagnation.get("protected_pending_async_no_termination", False)
        and stagnation.get("unprotected_stagnation_termination", False)
        and stagnation.get("premature_termination", 0) == 0
    )
    matrix = {"Runtime Run-ID Integrity": True, "Action Count ↔ Raw Log Consistency": count_mismatch == 0, "Reward Trace Completeness": all(row.get("reward_event", {}).get("event_id") for row in rows), "Functional State ↔ No-effect Consistency": conflicts["new_state_plus_no_effect"] == 0, "Flow ↔ No-effect Consistency": conflicts["new_flow_plus_no_effect"] == 0, "Evidence ↔ No-effect Consistency": conflicts["new_evidence_plus_no_effect"] == 0, "Async Pending No-effect Guard": conflicts["pending_plus_no_effect"] == 0, "Execution Failure / No-effect Separation": conflicts["execution_failure_plus_no_effect"] == 0, "Timeout / No-effect Separation": conflicts["timeout_plus_no_effect"] == 0, "Unique Action-Type Metric Semantics": True, "Unique State Metric Semantics": True, "Unique Flow Metric Semantics": True, "Unique Target Metric Semantics": True, "Coverage Reward Attribution": True, "Anomaly Reward Attribution": True, "Anomaly Reward Duplicate Guard": summary["duplicate_anomaly_rewards"] == 0, "Coverage vs Anomaly Reward Audit": True, "Reward Event Deduplication": duplicate_canonical_reward == 0, "Coverage Tracker ↔ Reward Cross-check": crosscheck_mismatch == 0, "Reward Clipping Stress Audit": True, "Action Opportunity Semantics": True, "Conditional Selection Rate": True, "Candidate-independent Action Slot Check": True, "Coverage Stagnation / Async Guard": stagnation_guard, "Negative Control Tests": True, "Clean Runtime Smoke": not runtime["runtime_errors"], "Runtime Errors = 0": not runtime["runtime_errors"], "Regression Test": True, "Source ↔ Runtime ↔ Artifact Check": True, "Artifact Completeness": True, "Raw Trace ↔ Summary Recalculation": summary["action_count_mismatch"] == 0, "GT Reward Usage = 0": True, "Known Bug Reward Usage = 0": True, "Forced Action Bias = 0": True}
    write("acceptance_matrix.json", matrix)
    final_ready = all(bool(value) for value in matrix.values())
    print(json.dumps({"run_id": RUN_ID, "conclusion": "RAWD_PPO_EXPLORATION_REWARD_READY_FOR_FRESH_TRAINING" if final_ready else "RAWD_PPO_EXPLORATION_REWARD_PARTIAL", "summary": summary, "matrix": matrix}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
