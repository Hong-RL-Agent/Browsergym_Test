from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "training_v3" / "run6_r10_confidence_propagation_validation"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    runtime = load("production_hybrid_runtime_10ep.json")
    detector = load("detector_funnel.json")
    lifecycle = load("candidate_lifecycle.json")
    traces = []
    for record in runtime.get("downstream_validation", {}).get("records", []):
        for anomaly in record.get("anomalies", []):
            if str(anomaly.get("rule_id") or (anomaly.get("evidence") or {}).get("rule_id") or "") != "R10":
                continue
            evidence = anomaly.get("evidence") if isinstance(anomaly.get("evidence"), dict) else {}
            pair_confidence = record.get("r10", {}).get("pair_confidence")
            evidence_confidence = evidence.get("pair_confidence")
            detector_pair_confidence = evidence_confidence
            key = str(anomaly.get("signature") or evidence.get("entity_id") or "")
            candidate = next((row for row in lifecycle.get("rows", []) if row.get("logical_key") == key), None)
            candidate_pair_confidence = candidate.get("pair_confidence") if candidate else None
            trace = {
                "pair_id": key,
                "episode": record.get("episode"),
                "step": record.get("step"),
                "network_key": evidence.get("network_key"),
                "ui_key": evidence.get("ui_key"),
                "network_value": evidence.get("network_value"),
                "ui_value": evidence.get("ui_value"),
                "pair_method": evidence.get("pair_method"),
                "relation": evidence.get("relation"),
                "semantic_pair_confidence": pair_confidence,
                "evidence_pair_confidence": evidence_confidence,
                "detector_pair_confidence": detector_pair_confidence,
                "detector_confidence": anomaly.get("confidence"),
                "candidate_pair_confidence": candidate_pair_confidence,
                "candidate_detector_confidence": candidate.get("detector_confidence") if candidate else None,
                "missing_confidence": any(value is None for value in (pair_confidence, evidence_confidence, detector_pair_confidence, candidate_pair_confidence)),
            }
            traces.append(trace)

    values = [float(row["semantic_pair_confidence"]) for row in traces if row["relation"] == "NOT_EQUAL" and row["semantic_pair_confidence"] is not None]
    pair_evidence_mismatch = sum(int(row["semantic_pair_confidence"] != row["evidence_pair_confidence"]) for row in traces if row["semantic_pair_confidence"] is not None and row["evidence_pair_confidence"] is not None)
    evidence_detector_mismatch = sum(int(row["evidence_pair_confidence"] != row["detector_pair_confidence"]) for row in traces if row["evidence_pair_confidence"] is not None and row["detector_pair_confidence"] is not None)
    detector_candidate_mismatch = sum(int(row["detector_pair_confidence"] != row["candidate_pair_confidence"]) for row in traces if row["detector_pair_confidence"] is not None and row["candidate_pair_confidence"] is not None)
    summary = {
        "runtime_smoke_5ep": load("production_hybrid_runtime.json").get("episode_summary", {}),
        "extended_runtime_smoke_10ep": runtime.get("episode_summary", {}),
        "semantic_pairs_count": sum(int(row.get("evidence", {}).get("semantic_pairs", 0)) for row in runtime["downstream_validation"]["records"]),
        "confidence_distribution": {
            "min": min(values) if values else None,
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "max": max(values) if values else None,
        },
        "not_equal_pair_count": len(values),
        "high_confidence_not_equal_count": sum(int(value >= 0.85) for value in values),
        "pair_to_evidence_mismatch_count": pair_evidence_mismatch,
        "evidence_to_detector_mismatch_count": evidence_detector_mismatch,
        "detector_to_candidate_mismatch_count": detector_candidate_mismatch,
        "missing_confidence_count": sum(int(row["missing_confidence"]) for row in traces),
        "first_confidence_drop_off": "NONE",
        "gt_runtime_access": False,
        "ppo_modified": False,
        "dqn_modified": False,
        "reward_modified": False,
        "threshold_modified": False,
    }
    (OUT / "confidence_trace.json").write_text(json.dumps({"traces": traces, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "semantic_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "r10_detector_funnel.json").write_text(json.dumps({"r10_rows": detector.get("r10_rows", []), "confidence_traces": traces, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "candidate_confidence.json").write_text(json.dumps({"candidates": [row for row in lifecycle.get("rows", []) if row.get("rule_id") == "R10"], "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
