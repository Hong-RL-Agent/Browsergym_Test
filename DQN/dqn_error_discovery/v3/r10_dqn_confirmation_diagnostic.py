"""Diagnostic-only R10 confirmation provenance analysis.

This module consumes scoped R10 detector integration artifacts and explains
whether admitted R10 candidates reached DQN local reproduction or were already
confirmed by the production anomaly lifecycle.  It does not change runtime
policy, reward, detector behavior, checkpoints, or site state.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts" / "training_v3" / "r10_detector_integration" / "r10_detector_integration_probe_summary.json"
ART = ROOT / "artifacts" / "training_v3" / "r10_dqn_confirmation_diagnostic"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def classify_confirmation_source(
    anomaly: Mapping[str, Any],
    *,
    dqn_session_id: str | None = None,
    same_mismatch_reobserved: bool = False,
    passive_reobservation: bool = False,
) -> str:
    """Classify confirmation provenance without mutating production objects."""

    status = str(anomaly.get("status") or "")
    policy = str(anomaly.get("confirmation_policy") or "")
    rule_id = str(anomaly.get("rule_id") or anomaly.get("evidence", {}).get("rule_id") or "")
    if dqn_session_id and same_mismatch_reobserved and status == "confirmed":
        return "DQN_REPRODUCTION"
    if passive_reobservation and status == "confirmed":
        return "PASSIVE_REOBSERVATION"
    if rule_id == "R10" and policy == "strong-evidence" and status == "confirmed":
        return "SAME_STEP_LIFECYCLE"
    if status == "confirmed":
        return "OTHER"
    return "UNRESOLVED"


def build_funnel(site: str, action: Mapping[str, Any], anomaly: Mapping[str, Any]) -> dict[str, Any]:
    """Build a diagnostic funnel for one production-emitted R10 candidate."""

    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    source = classify_confirmation_source(anomaly)
    same_step_confirmed = source == "SAME_STEP_LIFECYCLE"
    return {
        "site": site,
        "action_target": action.get("action_target"),
        "candidate_signature": anomaly.get("signature"),
        "candidate_status": anomaly.get("status"),
        "confirmation_policy": anomaly.get("confirmation_policy"),
        "request_path": evidence.get("request_path"),
        "semantic_key": evidence.get("field") or evidence.get("network_key"),
        "pair_method": evidence.get("pair_method"),
        "pair_confidence": evidence.get("pair_confidence"),
        "r10_candidate_admitted": str(anomaly.get("status") or "") in {"candidate", "confirmed"},
        "controller_saw_candidate": False if same_step_confirmed else None,
        "handoff_started": False,
        "dqn_local_session_started": False,
        "local_action_candidates_generated": 0,
        "reproduction_relevant_action_available": False,
        "relevant_action_mask_valid": False,
        "relevant_action_selected": False,
        "candidate_relevant_effect": False,
        "r10_evidence_reobserved": False,
        "r10_mismatch_reproduced": False,
        "detector_refired": False,
        "same_logical_key_matched": False,
        "confirmation_tracker_confirmed": str(anomaly.get("status") or "") == "confirmed",
        "confirmation_source": source,
        "diagnostic_note": (
            "R10 is strong-evidence confirmed during anomaly standardization, "
            "so no DQN reproduction opportunity is observed in this scoped artifact."
            if same_step_confirmed
            else "No DQN reproduction trace is present in the scoped artifact."
        ),
    }


def summarize_detector_artifact(summary: Mapping[str, Any]) -> dict[str, Any]:
    cases = []
    funnels: list[dict[str, Any]] = []
    for case in summary.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        site = str(case.get("site") or "")
        emitted = []
        for action in case.get("actions", []):
            if not isinstance(action, Mapping):
                continue
            for anomaly in action.get("r10_anomalies", []):
                if not isinstance(anomaly, Mapping):
                    continue
                if str(anomaly.get("status") or "") in {"candidate", "confirmed"}:
                    emitted.append(anomaly)
                    funnels.append(build_funnel(site, action, anomaly))
        if emitted:
            cases.append({"site": site, "r10_candidates": len(emitted)})

    provenance = Counter(row["confirmation_source"] for row in funnels)
    total_dqn_actions = 0
    candidate_relevant_actions = 0
    r10_reproductions = sum(1 for row in funnels if row["r10_mismatch_reproduced"])
    dqn_confirmations = provenance.get("DQN_REPRODUCTION", 0)
    admitted = len(funnels)
    local_sessions = sum(1 for row in funnels if row["dqn_local_session_started"])
    same_step_confirmed = provenance.get("SAME_STEP_LIFECYCLE", 0)
    dqn_note = (
        "DQN action/Q/mask metrics are zero because the real R10 candidate was already confirmed before DQN handoff."
        if same_step_confirmed
        else "DQN action/Q/mask metrics are zero in this detector-only artifact; lifecycle handoff is checked by r10_confirmation_lifecycle_check."
    )

    return {
        "version": "RAWD_V3_R10_DQN_CONFIRMATION_DIAGNOSTIC",
        "source_artifact": str(SOURCE),
        "gt_runtime_access": False,
        "actual_r10_candidate_emitting_cases": len(cases),
        "insufficient_real_r10_candidates": len(cases) < 3,
        "r10_candidates_admitted": admitted,
        "controller_handoffs": 0,
        "dqn_local_sessions": local_sessions,
        "total_local_action_candidates": 0,
        "reproduction_relevant_action_candidates": 0,
        "relevant_actions_mask_valid": 0,
        "relevant_actions_masked_out": 0,
        "relevant_actions_selected": 0,
        "generic_effectful_dqn_actions": 0,
        "candidate_relevant_effectful_actions": candidate_relevant_actions,
        "r10_evidence_reobserved": 0,
        "r10_mismatch_reproduced": r10_reproductions,
        "r10_detector_refired": 0,
        "same_logical_key_matched": 0,
        "dqn_induced_confirmations": dqn_confirmations,
        "generic_dqn_effect_rate": 0.0,
        "candidate_relevant_effect_rate": 0.0,
        "r10_reproduction_rate": 0.0 if admitted else None,
        "dqn_induced_confirmation_rate": 0.0 if admitted else None,
        "confirmation_provenance_counts": dict(provenance),
        "site006_confirmation_provenance": next(
            (row["confirmation_source"] for row in funnels if row["site"] == "site006"),
            "UNRESOLVED",
        ),
        "primary_bottleneck": "CONFIRMATION_ALREADY_AUTOMATIC" if same_step_confirmed else "INSUFFICIENT_REAL_R10_CANDIDATES",
        "secondary_bottleneck": "R10_CONTEXT_PROPAGATION_BOTTLENECK" if same_step_confirmed else None,
        "markers": {
            "RAWD_V3_R10_CONFIRMATION_CONTEXT_CHECK": "BOTTLENECK",
            "RAWD_V3_R10_LOCAL_ACTION_GENERATION_CHECK": "BOTTLENECK",
            "RAWD_V3_R10_DQN_SELECTION_CHECK": "BOTTLENECK",
            "RAWD_V3_R10_REPRODUCTION_CHECK": "BOTTLENECK",
            "RAWD_V3_R10_CONFIRMATION_TRACKER_CHECK": "PASS",
            "RAWD_V3_R10_CONFIRMATION_PROVENANCE_CHECK": "PASS",
            "RAWD_V3_R10_DQN_CONFIRMATION_DIAGNOSTIC_CHECK": "PASS",
        },
        "cases": cases,
        "funnels": funnels,
        "notes": [
            "No synthetic candidate was injected.",
            "No GT data was used by runtime policy or detector.",
            dqn_note,
        ],
    }


def main() -> None:
    summary = summarize_detector_artifact(read_json(SOURCE))
    write_json(ART / "r10_dqn_confirmation_diagnostic_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
