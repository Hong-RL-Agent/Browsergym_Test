"""Generate post-hoc layout semantic diagnostics for DEV_REACT/DEV_NEXT."""

from __future__ import annotations

import json
from pathlib import Path

from services.layout_semantics_diagnostic_service import analyze_layout_semantics, build_summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    artifact_root = root / "artifacts" / "framework_dev"
    outputs = {}
    for framework, prefix, evaluation_name in (("React", "react", "react_scenario_layout_evaluation.json"), ("Next", "next", "next_scenario_layout_evaluation.json")):
        evaluation = json.loads((artifact_root / evaluation_name).read_text(encoding="utf-8"))
        temporal = json.loads((artifact_root / f"{prefix}_temporal_v1.json").read_text(encoding="utf-8"))
        result = analyze_layout_semantics(evaluation, temporal, framework)
        outputs[prefix.lower()] = result
        (artifact_root / f"{prefix}_layout_semantics_diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (artifact_root / "layout_semantics_diagnostic_summary.json").write_text(json.dumps(build_summary(outputs["react"], outputs["next"]), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
