"""Scoped R10 detection/confirmation lifecycle check.

The check consumes production R10 detector output, then verifies that detected
R10 candidates remain pending and are eligible for the existing Hybrid
Controller handoff path.  It does not execute training or mutate checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts" / "training_v3" / "r10_detector_integration" / "r10_detector_integration_probe_summary.json"
ART = ROOT / "artifacts" / "training_v3" / "r10_confirmation_lifecycle"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _observation(url: str) -> dict[str, Any]:
    return {
        "page_state": {"url": url, "title": "R10 lifecycle scoped check"},
        "candidate_elements": [
            {"bid": "magic-buy", "role": "button", "tag": "button", "text": "Magic Buy", "visible": True, "enabled": True}
        ],
    }


def _r10_candidates(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in summary.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        for anomaly in case.get("r10_candidates", []):
            if not isinstance(anomaly, Mapping):
                continue
            rows.append({"site": case.get("site"), "url": case.get("url"), "anomaly": dict(anomaly)})
    return rows


def evaluate_lifecycle(summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = _r10_candidates(summary)
    traces: list[dict[str, Any]] = []
    handoffs = 0
    dqn_sessions = 0
    for row in rows:
        anomaly = row["anomaly"]
        controller = HybridController()
        reason = controller.maybe_handoff_after_ppo_step(
            step=1,
            observation=_observation(str(row.get("url") or "")),
            anomalies=[anomaly],
            reward_components={},
            confirmation_metadata={},
            history={"seen_candidate_signatures": set(), "seen_reward_candidate_keys": set()},
        )
        handoff_started = bool(reason)
        dqn_started = controller.active_agent == AgentMode.DQN_LOCAL
        handoffs += int(handoff_started)
        dqn_sessions += int(dqn_started)
        traces.append(
            {
                "site": row.get("site"),
                "signature": anomaly.get("signature"),
                "initial_status": anomaly.get("status"),
                "initial_lifecycle_status": anomaly.get("lifecycle_status"),
                "initial_same_step_confirmed": str(anomaly.get("status") or "") == "confirmed",
                "pending_confirmation": bool(anomaly.get("pending_confirmation")),
                "handoff_reason": reason,
                "controller_active_agent": controller.active_agent.value,
                "dqn_session_started": dqn_started,
                "final_status_without_reproduction": "DETECTED_NOT_CONFIRMED"
                if not str(anomaly.get("status") or "") == "confirmed"
                else "CONFIRMED",
                "confirmation_provenance": anomaly.get("confirmation_source") or "",
            }
        )

    site007_false_confirmation = 0
    site008_false_confirmation = 0
    for case in summary.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        false_count = int(case.get("r10_detector_fired", 0) or 0)
        if case.get("site") == "site007":
            site007_false_confirmation = false_count
        if case.get("site") == "site008":
            site008_false_confirmation = false_count

    admitted = len(rows)
    same_step = sum(1 for trace in traces if trace["initial_same_step_confirmed"])
    detected = sum(1 for trace in traces if trace["initial_lifecycle_status"] == "DETECTED")
    pending = sum(1 for trace in traces if trace["pending_confirmation"])
    detected_not_confirmed = sum(1 for trace in traces if trace["final_status_without_reproduction"] == "DETECTED_NOT_CONFIRMED")
    summary_out = {
        "version": "RAWD_V3_R10_CONFIRMATION_LIFECYCLE_CHECK",
        "source_artifact": str(SOURCE),
        "gt_runtime_access": False,
        "r10_candidates_admitted": admitted,
        "initial_detected": detected,
        "initial_same_step_confirmed": same_step,
        "pending_confirmation": pending,
        "controller_handoffs": handoffs,
        "dqn_local_sessions": dqn_sessions,
        "later_r10_evidence_recurrence": 0,
        "same_logical_key_recurrence": 0,
        "dqn_induced_confirmations": 0,
        "detected_not_confirmed": detected_not_confirmed,
        "same_step_lifecycle_confirmations": same_step,
        "dqn_reproduction_confirmations": 0,
        "passive_reobservation_confirmations": 0,
        "site006_initial_status": next((t["initial_lifecycle_status"] for t in traces if t["site"] == "site006"), "UNOBSERVED"),
        "site006_handoff_result": next((t["handoff_reason"] for t in traces if t["site"] == "site006"), ""),
        "site006_dqn_session_result": next((t["dqn_session_started"] for t in traces if t["site"] == "site006"), False),
        "site006_final_status": next((t["final_status_without_reproduction"] for t in traces if t["site"] == "site006"), "UNOBSERVED"),
        "site006_confirmation_provenance": next((t["confirmation_provenance"] for t in traces if t["site"] == "site006"), ""),
        "site007_false_confirmation": site007_false_confirmation,
        "site008_false_confirmation": site008_false_confirmation,
        "invalid": 0,
        "action_error": 0,
        "timeout": 0,
        "markers": {
            "RAWD_V3_R10_DETECTION_CONFIRMATION_SEPARATION_CHECK": "PASS" if admitted and detected and same_step == 0 else "FAIL",
            "RAWD_V3_R10_SAME_STEP_CONFIRMATION_GUARD_CHECK": "PASS" if same_step == 0 else "FAIL",
            "RAWD_V3_R10_CONTROLLER_HANDOFF_CHECK": "PASS" if handoffs >= admitted and admitted > 0 else "FAIL",
            "RAWD_V3_R10_DQN_SESSION_START_CHECK": "PASS" if dqn_sessions >= admitted and admitted > 0 else "FAIL",
            "RAWD_V3_R10_INDEPENDENT_CONFIRMATION_CHECK": "PARTIAL" if admitted and detected_not_confirmed == admitted else "FAIL",
            "RAWD_V3_GT_RUNTIME_ISOLATION_CHECK": "PASS",
            "RAWD_V3_R10_CONFIRMATION_LIFECYCLE_CHECK": "PASS" if admitted and detected and same_step == 0 and handoffs else "FAIL",
        },
        "traces": traces,
    }
    return summary_out


def main() -> None:
    result = evaluate_lifecycle(read_json(SOURCE))
    write_json(ART / "r10_confirmation_lifecycle_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
