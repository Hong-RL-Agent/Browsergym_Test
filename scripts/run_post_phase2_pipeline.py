"""Wait for phase 2, repair failures, run unguided evaluation, and gate models."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", default="artifacts/fleet/sites002-090-phase2.json")
    parser.add_argument("--expected-sites", type=int, default=89)
    parser.add_argument("--wait-timeout", type=int, default=43200)
    args = parser.parse_args()
    phase2 = Path(args.phase2)
    _wait(phase2, args.expected_sites, args.wait_timeout)

    # Resume retries only failed/missing sites because successful results are
    # retained in the same incremental summary.
    for attempt in range(1, 3):
        payload = _read(phase2)
        failed = [item for item in payload.get("results", []) if int(item.get("returncode", 1)) != 0]
        if not failed:
            break
        print(f"[post] retry_attempt={attempt} failed={len(failed)}", flush=True)
        subprocess.run([
            sys.executable, "-u", "scripts/run_site_fleet_rainbow.py",
            "--start-site", "2", "--end-site", "90", "--episodes", "3", "--max-steps", "25",
            "--site-timeout", "480", "--output", str(phase2), "--resume",
        ])

    evaluation = Path("artifacts/fleet/unguided_evaluation.json")
    subprocess.run([
        sys.executable, "-u", "scripts/run_fleet_evaluation.py",
        "--episodes", "3", "--max-steps", "20", "--seed", "1042",
        "--output", str(evaluation), "--resume",
    ])
    report = _gate_models(evaluation)
    report_path = Path("artifacts/fleet/model_promotion_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, Path("artifacts/fleet/final_fleet_report.md"))
    print(f"[post] accepted={report['accepted_count']} quarantined={report['quarantined_count']}", flush=True)
    return 0


def _wait(path: Path, expected: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        count = int(_read(path).get("site_count", 0) or 0) if path.exists() else 0
        print(f"[post] waiting phase2 {count}/{expected}", flush=True)
        if count >= expected:
            return
        time.sleep(30)
    raise TimeoutError("phase2 completion timeout")


def _gate_models(evaluation: Path) -> dict:
    rows = []
    quarantine = Path("artifacts/models/quarantine")
    quarantine.mkdir(parents=True, exist_ok=True)
    for result in _read(evaluation).get("results", []):
        site_id = str(result.get("site_id") or "")
        catalog = Path("datasets") / site_id / "bug_catalog.json"
        generated = False
        if catalog.exists():
            generated = bool(_read(catalog).get("generated_by"))
        recall_threshold = 0.7 if generated else 0.8
        precision = float(result.get("precision", 0.0) or 0.0)
        recall = float(result.get("recall", 0.0) or 0.0)
        accepted = int(result.get("returncode", 1)) == 0 and precision >= 0.9 and recall >= recall_threshold
        reason = "passed" if accepted else f"precision={precision:.3f},recall={recall:.3f}"
        model = Path("artifacts/models/fleet") / f"{site_id}_browsergym_rainbow_dqn.pt"
        if not accepted and model.exists():
            shutil.copy2(model, quarantine / model.name)
        rows.append({"site_id": site_id, "accepted": accepted, "precision": precision, "recall": recall,
                     "generated_catalog": generated, "reason": reason})
    return {"accepted_count": sum(row["accepted"] for row in rows),
            "quarantined_count": sum(not row["accepted"] for row in rows), "sites": rows}


def _write_markdown(report: dict, path: Path) -> None:
    lines = ["# 90-Site Fleet Final Report", "", f"- Accepted: {report['accepted_count']}",
             f"- Quarantined: {report['quarantined_count']}", "", "| Site | Status | Precision | Recall |", "|---|---:|---:|---:|"]
    for row in report["sites"]:
        lines.append(f"| {row['site_id']} | {'PASS' if row['accepted'] else 'QUARANTINE'} | {row['precision']:.3f} | {row['recall']:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
