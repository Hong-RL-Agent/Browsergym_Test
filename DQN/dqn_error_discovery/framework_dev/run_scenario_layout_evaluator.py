"""CLI for the development-only scenario layout evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .scenario_evaluator import compare_evaluations, evaluate_layout_artifacts, reclassify_root_cause


def _evaluate(root: Path, framework: str) -> dict:
    prefix = "react" if framework == "react" else "next"
    return evaluate_layout_artifacts(
        catalog_path=root / "development_sites" / ("react_async" if prefix == "react" else "next_async") / "SCENARIOS.json",
        diagnostic_path=root / "artifacts" / "framework_dev" / f"{prefix}_temporal_stability_diagnostic.json",
        temporal_path=root / "artifacts" / "framework_dev" / f"{prefix}_temporal_v1.json",
        baseline_path=root / "artifacts" / "framework_dev" / f"{prefix}_v1_baseline.json",
        scenario_evidence_path=root / "artifacts" / "framework_dev" / f"{prefix}_scenario_results.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.root / "artifacts" / "framework_dev"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {framework: _evaluate(args.root, framework) for framework in ("react", "next")}
    for framework, result in results.items():
        path = output_dir / f"{framework}_scenario_layout_evaluation.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {"schema_version": "scenario_layout_evaluation_summary_v1", "evaluation_only": True, "frameworks": {}}
    for framework, result in results.items():
        summary["frameworks"][framework] = {"summary": result["summary"], "primary_summary": result["primary_summary"], "candidate_table": result["candidates"], "scenario_table": result["scenarios"], "representative": result["representative"], "root_cause_reclassification": reclassify_root_cause(result)}
    for framework, result in results.items():
        before = {"candidates": result["baseline"]["candidates"]}
        summary["frameworks"][framework]["before_after"] = compare_evaluations(before, result)
    summary["overall"] = {
        "layout_candidates": sum(item["summary"]["layout_candidates"] for item in summary["frameworks"].values()),
        "mapped": sum(item["summary"]["mapped"] for item in summary["frameworks"].values()),
        "unmapped": sum(item["summary"]["unmapped"] for item in summary["frameworks"].values()),
        "root_cause_reclassification": "MIXED_TRUE_DEFECT_AND_FP" if any(item["root_cause_reclassification"] == "MIXED_TRUE_DEFECT_AND_FP" for item in summary["frameworks"].values()) else ("PERSISTENT_LAYOUT_FALSE_POSITIVE" if any(item["root_cause_reclassification"] == "PERSISTENT_LAYOUT_FALSE_POSITIVE" for item in summary["frameworks"].values()) else "EVALUATOR_MAPPING_INSUFFICIENT"),
        "candidate_mapping_rate": (sum(item["summary"]["mapped"] for item in summary["frameworks"].values()) / sum(item["summary"]["layout_candidates"] for item in summary["frameworks"].values())) if sum(item["summary"]["layout_candidates"] for item in summary["frameworks"].values()) else None,
    }
    (output_dir / "scenario_layout_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
