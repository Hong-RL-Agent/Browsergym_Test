from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full J.A.W.S BrowserGym PPO port-batch pipeline.")
    parser.add_argument("--start-port", type=int, default=9220)
    parser.add_argument("--end-port", type=int, default=9309)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--total-updates", type=int, default=20)
    parser.add_argument("--episodes-per-site", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--model-path", default="artifacts/models/jaws_browsergym_shared_ppo_v2_browsergym_raw_obs.pt")
    parser.add_argument("--site-map", default="")
    parser.add_argument("--headless", default="true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-artifact-dir", default="artifacts")
    parser.add_argument("--enable-csv-logging", default="true")
    parser.add_argument("--csv-log-dir", default="")
    parser.add_argument("--log-observation-detail", default="true")
    parser.add_argument("--log-action-space", default="true")
    parser.add_argument("--log-raw-json", default="false")
    args = parser.parse_args()

    port_run_id = f"ports_{args.start_port}_{args.end_port}"
    run_id = f"{port_run_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    generated_config = Path("configs/generated") / f"port_sites_{args.start_port}_{args.end_port}.json"
    preflight_dir = Path(args.base_artifact_dir) / "preflight" / run_id
    batches_dir = Path("configs/generated/batches")
    evaluations_dir = Path(args.base_artifact_dir) / "evaluations" / run_id
    training_dir = Path(args.base_artifact_dir) / "training" / run_id
    final_summary = Path(args.base_artifact_dir) / "final" / f"{run_id}_summary.json"
    failed_batches_path = Path(args.base_artifact_dir) / "final" / "failed_batches.json"
    reports_dir = Path(args.base_artifact_dir) / "reports" / run_id
    failed_batches = []

    steps = [
        [
            sys.executable,
            "scripts/generate_port_sites_config.py",
            "--start-port",
            str(args.start_port),
            "--end-port",
            str(args.end_port),
            "--output",
            str(generated_config),
            "--shared-model-path",
            args.model_path,
        ],
        [
            sys.executable,
            "scripts/preflight_port_sites.py",
            "--config",
            str(generated_config),
            "--output-dir",
            str(preflight_dir),
            "--max-steps",
            "1",
            "--headless",
            args.headless,
        ],
        [
            sys.executable,
            "scripts/create_site_batches.py",
            "--input",
            str(preflight_dir / "active_sites.json"),
            "--batch-size",
            str(args.batch_size),
            "--output-dir",
            str(batches_dir),
        ],
    ]
    if args.site_map:
        steps[0].extend(["--site-map", args.site_map])

    for command in steps:
        completed = _run(command)
        if completed.returncode != 0:
            print(f"critical_step_failed returncode={completed.returncode}: {' '.join(command)}", file=sys.stderr)
            return completed.returncode

    manifest_path = batches_dir / "batches_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch_paths = [Path(path) for path in manifest.get("batches", [])]

    for batch_path in batch_paths:
        batch_id = batch_path.stem
        train_cmd = [
            sys.executable,
            "scripts/run_batch_training.py",
            "--batch-config",
            str(batch_path),
            "--total-updates",
            str(args.total_updates),
            "--episodes-per-site",
            str(args.episodes_per_site),
            "--max-steps",
            str(args.max_steps),
            "--seed",
            str(args.seed),
            "--headless",
            args.headless,
            "--artifact-root",
            str(training_dir),
            "--save-model",
            args.model_path,
            "--enable-csv-logging",
            args.enable_csv_logging,
            "--csv-log-dir",
            args.csv_log_dir or str(Path(args.base_artifact_dir) / "training"),
            "--log-observation-detail",
            args.log_observation_detail,
            "--log-action-space",
            args.log_action_space,
            "--log-raw-json",
            args.log_raw_json,
            "--run-id",
            run_id,
        ]
        train_result = _run(train_cmd)
        if train_result.returncode != 0:
            failed_batches.append(
                {
                    "batch_id": batch_id,
                    "stage": "training",
                    "returncode": train_result.returncode,
                    "command": train_cmd,
                }
            )

        eval_cmd = [
            sys.executable,
            "scripts/run_batch_evaluation.py",
            "--batch-config",
            str(batch_path),
            "--model-path",
            args.model_path,
            "--episodes",
            str(args.eval_episodes),
            "--max-steps",
            str(args.max_steps),
            "--seed",
            str(args.seed),
            "--headless",
            args.headless,
            "--artifact-root",
            str(evaluations_dir),
            "--enable-csv-logging",
            args.enable_csv_logging,
            "--csv-log-dir",
            args.csv_log_dir or str(Path(args.base_artifact_dir) / "evaluations"),
            "--log-observation-detail",
            args.log_observation_detail,
            "--log-action-space",
            args.log_action_space,
            "--log-raw-json",
            args.log_raw_json,
            "--run-id",
            run_id,
        ]
        eval_result = _run(eval_cmd)
        if eval_result.returncode != 0:
            failed_batches.append(
                {
                    "batch_id": batch_id,
                    "stage": "evaluation",
                    "returncode": eval_result.returncode,
                    "command": eval_cmd,
                }
            )

    failed_batches_path.parent.mkdir(parents=True, exist_ok=True)
    failed_batches_path.write_text(json.dumps(failed_batches, ensure_ascii=False, indent=2), encoding="utf-8")

    merge_result = _run(
        [
            sys.executable,
            "scripts/merge_batch_results.py",
            "--input-dir",
            str(evaluations_dir),
            "--output",
            str(final_summary),
            "--reports-dir",
            str(reports_dir),
        ]
    )
    if merge_result.returncode != 0:
        return merge_result.returncode

    final = json.loads(final_summary.read_text(encoding="utf-8"))
    automation_summary = {
        "run_id": run_id,
        "port_run_id": port_run_id,
        "generated_config": str(generated_config),
        "active_sites": str(preflight_dir / "active_sites.json"),
        "failed_sites": str(preflight_dir / "failed_sites.json"),
        "batch_count": len(batch_paths),
        "failed_batch_count": len(failed_batches),
        "failed_batches": str(failed_batches_path),
        "final_summary": str(final_summary),
        "total_sites": final.get("total_sites"),
        "evaluated_sites": final.get("evaluated_sites"),
        "average_recall": final.get("average_recall"),
        "average_precision": final.get("average_precision"),
    }
    print(json.dumps(automation_summary, ensure_ascii=False, indent=2))
    return 0


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"[run] {' '.join(command)}", flush=True)
    return subprocess.run(command, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
