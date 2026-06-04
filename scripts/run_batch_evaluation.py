from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PPO evaluation for one BrowserGym batch config.")
    parser.add_argument("--batch-config", required=True)
    parser.add_argument("--model-path", default="artifacts/models/jaws_browsergym_shared_ppo_v2_browsergym_raw_obs.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", default="true")
    parser.add_argument("--strict-site-validation", default="false")
    parser.add_argument("--artifact-root", default="artifacts/evaluations")
    parser.add_argument("--enable-csv-logging", default="true")
    parser.add_argument("--csv-log-dir", default="")
    parser.add_argument("--log-observation-detail", default="true")
    parser.add_argument("--log-action-space", default="true")
    parser.add_argument("--log-raw-json", default="false")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    batch_config = Path(args.batch_config)
    batch_id = batch_config.stem
    output_root = Path(args.artifact_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / f"{batch_id}-result.json"
    stdout_path = output_root / f"{batch_id}-stdout.txt"
    stderr_path = output_root / f"{batch_id}-stderr.txt"
    meta_path = output_root / f"{batch_id}-evaluation_summary.json"

    command = [
        sys.executable,
        "runners/evaluate_multisite_browsergym_agent.py",
        "--config",
        str(batch_config),
        "--model-path",
        args.model_path,
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(args.seed),
        "--headless",
        args.headless,
        "--strict-site-validation",
        args.strict_site_validation,
        "--output",
        str(result_path),
        "--enable-csv-logging",
        args.enable_csv_logging,
        "--csv-log-dir",
        args.csv_log_dir or str(output_root.parent if output_root.name.startswith("ports_") else output_root),
        "--log-observation-detail",
        args.log_observation_detail,
        "--log-action-space",
        args.log_action_space,
        "--log-raw-json",
        args.log_raw_json,
        "--run-id",
        args.run_id or (output_root.name if output_root.name else batch_id),
    ]

    started_at = datetime.now(timezone.utc).isoformat()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(command, stdout=stdout_handle, stderr=stderr_handle, text=True)
    finished_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "batch_id": batch_id,
        "batch_config": str(batch_config),
        "model_path": args.model_path,
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command": command,
        "returncode": completed.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "success": completed.returncode == 0 and result_path.exists(),
    }
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
