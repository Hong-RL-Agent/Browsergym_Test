from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runners.evaluate_multisite_browsergym_agent import _write_openended_review_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Export site9800 open-ended review artifacts from evaluation JSON.")
    parser.add_argument("--evaluation-json", required=True, help="Path to evaluate_multisite_browsergym_agent JSON output.")
    parser.add_argument("--site-id", default="site9800")
    parser.add_argument("--base-url", default="http://localhost:9800")
    args = parser.parse_args()

    data = json.loads(Path(args.evaluation_json).read_text(encoding="utf-8"))
    sites = data.get("sites", {}) if isinstance(data, Mapping) else {}
    site_result: Dict[str, Any] = dict(sites.get(args.site_id, {}) if isinstance(sites, Mapping) else {})
    if not site_result:
        raise SystemExit(f"site result not found: {args.site_id}")
    exploratory = site_result.get("exploratory_anomalies") or site_result.get("openended_anomaly_discovery", {}).get("exploratory_anomalies") or []
    anomaly_types = site_result.get("anomaly_types") or site_result.get("openended_anomaly_discovery", {}).get("anomaly_types") or {}
    artifacts = _write_openended_review_artifacts(
        args.site_id,
        {"site_id": args.site_id, "base_url": args.base_url},
        site_result,
        exploratory if isinstance(exploratory, list) else [],
        anomaly_types if isinstance(anomaly_types, Mapping) else {},
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
