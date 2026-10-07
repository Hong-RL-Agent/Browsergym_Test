from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reports.generate_pdf_report import generate_pdf_report
from services.browsergym_report_data_builder import build_browsergym_report_data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", default="artifacts/reports")
    parser.add_argument("--artifact-root", default="")
    args = parser.parse_args()

    site_id = args.site_id
    artifact_root = Path(args.artifact_root) if args.artifact_root else _default_artifact_root(site_id)
    detected_bugs_path = artifact_root / site_id / "detected_bugs.json"
    training_summary_path = artifact_root / site_id / "training_summary.json"
    transition_log_path = artifact_root / site_id / "rl_transition_log.jsonl"
    bug_catalog_path = Path("datasets") / site_id / "bug_catalog.json"
    model_path = Path("artifacts") / "models" / f"{site_id}_browsergym_ppo.pt"

    _require_file(detected_bugs_path)
    _require_file(training_summary_path)
    _require_file(transition_log_path)
    _require_file(bug_catalog_path)
    model_path_arg = model_path if model_path.exists() else None

    report_data = build_browsergym_report_data(
        site_id=site_id,
        base_url=args.base_url,
        detected_bugs_path=detected_bugs_path,
        training_summary_path=training_summary_path,
        transition_log_path=transition_log_path,
        bug_catalog_path=bug_catalog_path,
        model_path=model_path_arg,
    )
    output_path = generate_pdf_report(report_data, args.output_dir)
    print(output_path)
    return 0


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required BrowserGym report input not found: {path}")


def _default_artifact_root(site_id: str) -> Path:
    multisite = Path("artifacts") / "multisite" / site_id / "detected_bugs.json"
    if multisite.exists():
        return Path("artifacts") / "multisite"
    return Path("artifacts") / "browsergym"


if __name__ == "__main__":
    raise SystemExit(main())
