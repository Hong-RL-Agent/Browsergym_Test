"""Build live interactive-layout evidence summaries from matched run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .interactive_layout_evidence_evaluator import evaluate_runtime_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument("--react-runtime", default=None, type=Path)
    parser.add_argument("--next-runtime", default=None, type=Path)
    args = parser.parse_args()
    artifact_dir = args.root / "artifacts" / "framework_dev"
    react_path = args.react_runtime or artifact_dir / "react_r06_live_runtime.json"
    next_path = args.next_runtime or artifact_dir / "next_r06_live_runtime.json"
    reports = {}
    if react_path.exists():
        reports["react"] = evaluate_runtime_artifact(react_path, "React", 5, 3)
    else:
        reports["react"] = {"framework": "React", "runtime_artifact": str(react_path), "summary": {"evidence_collection_status": "UNAVAILABLE", "reason": "RUNTIME_ARTIFACT_MISSING"}, "evidence_rows": [], "r06_candidate_keys": [], "confirmed_r06_keys": [], "runtime_isolation": {"ground_truth_used_in_runtime": False}}
    if next_path.exists():
        reports["next"] = evaluate_runtime_artifact(next_path, "Next", 4, 1)
    else:
        reports["next"] = {"framework": "Next", "runtime_artifact": str(next_path), "summary": {"evidence_collection_status": "UNAVAILABLE", "reason": "RUNTIME_ARTIFACT_MISSING"}, "evidence_rows": [], "r06_candidate_keys": [], "confirmed_r06_keys": [], "runtime_isolation": {"ground_truth_used_in_runtime": False}}
    for name, report in reports.items():
        payload = json.dumps(report, indent=2)
        (artifact_dir / f"{name}_interactive_layout_evidence.json").write_text(payload, encoding="utf-8")
        (artifact_dir / f"{name}_interactive_layout_evidence_live.json").write_text(payload, encoding="utf-8")
        live = {
            "schema_version": "r06_live_validation_v1",
            "evaluation_only": True,
            "framework": report["framework"],
            "source": report["runtime_artifact"],
            "summary": report["summary"],
            "r06_candidates": report["r06_candidate_keys"],
            "confirmed_r06_candidates": report["confirmed_r06_keys"],
            "representative_normal_fp": next((row for row in report["evidence_rows"] if row.get("functional_impact") in {"NONE", "LOW"}), None),
            "runtime_isolation": report["runtime_isolation"],
        }
        (artifact_dir / f"{name}_r06_live_validation.json").write_text(json.dumps(live, indent=2), encoding="utf-8")
    summary = {"schema_version": "interactive_layout_evidence_summary_v1", "evaluation_only": True, "frameworks": reports,
               "classification": "INTERACTIVE_EVIDENCE_PASS" if reports and all(item["summary"]["evidence_completeness_rate"] is not None for item in reports.values()) else "EVIDENCE_COLLECTION_INCOMPLETE"}
    payload = json.dumps(summary, indent=2)
    (artifact_dir / "interactive_layout_evidence_summary.json").write_text(payload, encoding="utf-8")
    (artifact_dir / "interactive_layout_evidence_live_summary.json").write_text(payload, encoding="utf-8")
    (artifact_dir / "r06_live_validation_summary.json").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
