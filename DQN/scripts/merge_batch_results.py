from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge BrowserGym batch evaluation JSON results.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reports-dir", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    result_files = sorted(input_dir.glob("batch-*-result.json"))
    sites: Dict[str, Any] = {}
    batch_results = {}
    failed_result_files = []

    for path in result_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            batch_results[path.stem.replace("-result", "")] = {
                "path": str(path),
                "site_count": len(data.get("sites", {})) if isinstance(data.get("sites"), Mapping) else 0,
                "average_reward": data.get("average_reward"),
            }
            for site_id, result in (data.get("sites", {}) or {}).items():
                if isinstance(result, Mapping):
                    site_result = dict(result)
                    site_result["batch_result_path"] = str(path)
                    sites[str(site_id)] = site_result
        except Exception as exc:
            failed_result_files.append({"path": str(path), "reason": str(exc)})

    recall_values = [float(site["recall"]) for site in sites.values() if site.get("recall") is not None]
    precision_values = [float(site["precision"]) for site in sites.values() if site.get("precision") is not None]
    matched_bug_ids = sorted(
        {
            str(bug_id)
            for site in sites.values()
            for bug_id in (site.get("matched_bug_ids") or [])
            if bug_id
        }
    )
    missed_bug_ids = sorted(
        {
            str(bug_id)
            for site in sites.values()
            for bug_id in (site.get("missed_bug_ids") or [])
            if bug_id
        }
    )
    failed_sites = {
        site_id: site
        for site_id, site in sites.items()
        if site.get("error") or site.get("status") == "failed"
    }

    summary = {
        "input_dir": str(input_dir),
        "batch_result_count": len(result_files),
        "total_sites": len(sites),
        "evaluated_sites": len(sites) - len(failed_sites),
        "failed_sites": len(failed_sites),
        "average_recall": mean(recall_values) if recall_values else None,
        "average_precision": mean(precision_values) if precision_values else None,
        "total_matched_bugs": len(matched_bug_ids),
        "total_missed_bugs": len(missed_bug_ids),
        "matched_bug_ids": matched_bug_ids,
        "missed_bug_ids": missed_bug_ids,
        "failed_result_files": failed_result_files,
        "batches": batch_results,
        "sites": sites,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.reports_dir:
        _write_batch_reports(summary, Path(args.reports_dir))
    print(json.dumps({k: summary[k] for k in ("total_sites", "evaluated_sites", "failed_sites", "average_recall", "average_precision")}, ensure_ascii=False, indent=2))
    return 0


def _write_batch_reports(summary: Mapping[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for batch_id, batch in (summary.get("batches", {}) or {}).items():
        sites = {
            site_id: site
            for site_id, site in (summary.get("sites", {}) or {}).items()
            if str(site.get("batch_result_path") or "") == str(batch.get("path") or "")
        }
        lines = [
            f"# {batch_id} Evaluation Report",
            "",
            f"- result_path: `{batch.get('path')}`",
            f"- site_count: `{len(sites)}`",
            "",
            "| Site | Recall | Precision | Matched | Missed |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for site_id, site in sorted(sites.items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        site_id,
                        _fmt(site.get("recall")),
                        _fmt(site.get("precision")),
                        str(len(site.get("matched_bug_ids") or [])),
                        str(len(site.get("missed_bug_ids") or [])),
                    ]
                )
                + " |"
            )
        (reports_dir / f"{batch_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
