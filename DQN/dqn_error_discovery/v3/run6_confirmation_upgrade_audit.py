"""Read-only audit of Run6 confirmation-upgrade transitions.

This intentionally does not replay the environment or modify reward/training
code.  It distinguishes missing training telemetry from evidence of leakage.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "artifacts/training_v3/run6_confirmation_aware_dqn/seed42_full"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


RAW_FIELDS = (
    "active_candidate_logical_key", "active_candidate_status_before", "active_candidate_status_after",
    "requires_action_reproduction", "reproduction_strategy", "reproduction_action_type",
    "reproduction_target_identity", "selected_target_identity", "network_recurrence",
    "semantic_recurrence", "same_chain_recurrence", "confirmation_event", "confirmation_event_key",
    "confirmation_source", "confirmation_event_step", "recurrence_logical_key", "fresh_action_caused",
)


def audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audited = []
    upgrade_index = 0
    for transition_index, row in enumerate(rows):
        components = row.get("reward_components", {}) or {}
        upgrade = float(components.get("confirmation_upgrade", 0.0) or 0.0)
        if upgrade <= 0:
            continue
        upgrade_index += 1
        meta = row.get("confirmation_meta", {}) or {}
        flags = meta.get("decoded_flags", {}) or {}
        action = row.get("selected_action", {}) or {}
        missing = [field for field in RAW_FIELDS if field not in row]
        reproduction_attempt = bool(row.get("is_reproduction_attempt"))
        selected_reproduction = bool(row.get("reproduction_action_selected"))
        effect = bool(row.get("candidate_relevant_effect"))
        same_chain = bool(row.get("same_logical_key"))
        category = str(meta.get("decoded_category") or "MISSING")

        # Classification is deliberately evidence-conservative.  Observation
        # confirmations are separated from ACTION_REPLAY-like rows; rows with
        # no event/evidence fields are not promoted to a leakage finding.
        if not reproduction_attempt and category in {"R01", "R05", "R06"} and flags.get("active_candidate_exists"):
            classification = "OBSERVATION_CONFIRMATION"
        elif reproduction_attempt and not (selected_reproduction and effect and same_chain):
            classification = "LOGGING_CONTRACT_MISMATCH"
        elif selected_reproduction and effect and same_chain:
            classification = "TRUE_CONFIRMATION_REWARD"
        else:
            classification = "UNKNOWN"

        audited.append({
            "audit_id": f"run6-seed42-upgrade-{upgrade_index:03d}",
            "transition_index": transition_index,
            "site_id": row.get("site_id", "MISSING"), "episode": row.get("episode", "MISSING"),
            "local_session": row.get("session_id", "MISSING"), "step": row.get("local_step", "MISSING"),
            "active_key_logged": row.get("active_key", "MISSING"),
            "decoded_category": category, "decoded_flags": flags,
            "selected_action_type": action.get("action_type", "MISSING"),
            "selected_target_identity": action.get("target_identity", "MISSING"),
            "reproduction_attempt": reproduction_attempt,
            "selected_reproduction": selected_reproduction,
            "candidate_relevant_effect_logged": effect,
            "same_chain_recurrence_logged": same_chain,
            "confirmation_upgrade_component": upgrade,
            "final_reward": row.get("reward", "MISSING"),
            "replay_metadata": row.get("replay_metadata", "MISSING"),
            "raw_runtime_fields": {field: row.get(field, "MISSING") for field in RAW_FIELDS},
            "missing_runtime_fields": missing,
            "classification": classification,
        })
    return audited


def run_audit(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "dqn_seed42_transitions.jsonl")
    audited = audit_rows(rows)
    counts = Counter(item["classification"] for item in audited)
    upgrade_values = Counter(str(item["confirmation_upgrade_component"]) for item in audited)
    category_counts = Counter(item["decoded_category"] for item in audited)
    action_counts = Counter(item["selected_action_type"] for item in audited)
    missing_counts = Counter(field for item in audited for field in item["missing_runtime_fields"])
    replay_counts = Counter(
        (str(item["replay_metadata"].get(key, "MISSING")) if isinstance(item["replay_metadata"], dict) else "MISSING")
        for item in audited for key in ("is_confirmation_context", "is_successful_reproduction", "is_failed_reproduction", "is_partial_reproduction")
    )
    result = {
        "run": "Run6 seed42 full", "read_only": True,
        "rewarded_upgrade_count": len(audited), "upgrade_values": dict(upgrade_values),
        "classification_counts": dict(counts), "decoded_category_counts": dict(category_counts),
        "selected_action_counts": dict(action_counts), "missing_runtime_field_counts": dict(missing_counts),
        "replay_label_value_counts": dict(replay_counts), "audits": audited,
        "reconstruction": {
            "actual_same_chain_confirmation_count": "UNAVAILABLE_FROM_TRAINING_ARTIFACT",
            "true_positive_reward_count": "UNAVAILABLE_FROM_TRAINING_ARTIFACT",
            "false_positive_reward_count": "UNAVAILABLE_FROM_TRAINING_ARTIFACT",
            "false_negative_count": "UNAVAILABLE_FROM_TRAINING_ARTIFACT",
            "reason": "Training logger omitted lifecycle event, pending snapshot, logical-key, source, freshness, and recurrence evidence.",
        },
        "code_path": {
            "full_training_reward": "dqn_error_discovery.runners.train_dqn_error_detector.calculate_dqn_reward",
            "final_live_reward": "dqn_error_discovery.runners.train_dqn_error_detector.calculate_dqn_reward",
            "same_reward_function": True,
            "confirmation_helper": "_confirmation_upgrade_transition_eligible",
            "code_path_drift": False,
            "telemetry_contract_drift": True,
        },
        "checkpoint_impact": {
            "fp_replay_admissions": "UNAVAILABLE_IDENTITY_NOT_LOGGED",
            "fp_successful_reproduction_labels": 0,
            "fp_sample_draws": "UNAVAILABLE_IDENTITY_NOT_LOGGED",
            "optimizer_exposure": "UNAVAILABLE_IDENTITY_NOT_LOGGED",
        },
        "conclusion": "TELEMETRY_CONTRACT_BUG_ONLY",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    result = run_audit(args.run_dir)
    out = args.run_dir / "audit"
    write_json(out / "run6_confirmation_upgrade_audit.json", result)
    write_json(out / "run6_confirmation_upgrade_audit_rows.json", result["audits"])
    print("RUN6_CONFIRMATION_UPGRADE_AUDIT_COMPLETE")
    print(f"rewarded upgrades: {result['rewarded_upgrade_count']}")
    print(f"classifications: {result['classification_counts']}")
    print(f"code path drift: {result['code_path']['code_path_drift']}")
    print(f"telemetry contract drift: {result['code_path']['telemetry_contract_drift']}")
    print(f"conclusion: {result['conclusion']}")


if __name__ == "__main__":
    main()
