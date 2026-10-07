from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.api_anomaly_detection_service import summarize_api_anomalies


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge J.A.W.S API fuzz evaluation results.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    result_files = sorted(path for path in input_dir.rglob("result.json") if _looks_like_api_result(path))
    runs: Dict[str, Any] = {}
    sites: Dict[str, Any] = {}
    anomalies = []
    failed_result_files = []

    for path in result_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            run_id = str(data.get("run_id") or path.parent.name)
            run_anomalies = data.get("anomalies", []) if isinstance(data.get("anomalies"), list) else []
            runs[run_id] = {
                "path": str(path),
                "total_sites": data.get("total_sites", 0),
                "evaluated_sites": data.get("evaluated_sites", 0),
                "total_endpoints": data.get("total_endpoints", 0),
                "total_cases": data.get("total_cases", 0),
                "anomaly_count": len(run_anomalies),
            }
            for site_id, site in (data.get("sites", {}) or {}).items():
                if isinstance(site, Mapping):
                    row = dict(site)
                    row["run_id"] = run_id
                    row["result_path"] = str(path)
                    sites[str(site_id)] = row
            anomalies.extend(dict(item, run_id=run_id) for item in run_anomalies if isinstance(item, Mapping))
        except Exception as exc:
            failed_result_files.append({"path": str(path), "reason": str(exc)})

    summary = {
        "source": "api",
        "input_dir": str(input_dir),
        "run_count": len(runs),
        "total_sites": sum(int(run.get("total_sites", 0) or 0) for run in runs.values()),
        "evaluated_sites": sum(int(run.get("evaluated_sites", 0) or 0) for run in runs.values()),
        "total_endpoints": sum(int(run.get("total_endpoints", 0) or 0) for run in runs.values()),
        "total_cases": sum(int(run.get("total_cases", 0) or 0) for run in runs.values()),
        "anomalies": anomalies,
        "summary": summarize_api_anomalies(anomalies),
        "runs": runs,
        "sites": sites,
        "failed_result_files": failed_result_files,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("run_count", "total_sites", "evaluated_sites", "total_endpoints", "total_cases", "summary")}, ensure_ascii=False, indent=2))
    return 0


def _looks_like_api_result(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("source") == "api" or data.get("evaluation_mode") == "api_rule_based_fuzzing"


if __name__ == "__main__":
    raise SystemExit(main())
