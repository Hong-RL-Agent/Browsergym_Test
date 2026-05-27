from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PPO training for one BrowserGym batch config.")
    parser.add_argument("--batch-config", required=True)
    parser.add_argument("--total-updates", type=int, default=20)
    parser.add_argument("--episodes-per-site", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", default="true")
    parser.add_argument("--load-model", default="")
    parser.add_argument("--save-model", default="")
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--strict-site-validation", default="false")
    parser.add_argument("--artifact-root", default="artifacts/training")
    args = parser.parse_args()

    batch_config = Path(args.batch_config)
    batch_id = batch_config.stem
    output_dir = Path(args.artifact_root) / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "runners/train_multisite_browsergym_agent.py",
        "--config",
        str(batch_config),
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
        "--entropy-coef",
        str(args.entropy_coef),
        "--strict-site-validation",
        args.strict_site_validation,
    ]
    if args.load_model:
        command.extend(["--load-model", args.load_model])
    if args.save_model:
        command.extend(["--save-model", args.save_model])

    result = _run(command, output_dir / "train_stdout.txt", output_dir / "train_stderr.txt")
    summary = {
        "batch_id": batch_id,
        "batch_config": str(batch_config),
        "command": command,
        "returncode": result.returncode,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "stdout_path": str(output_dir / "train_stdout.txt"),
        "stderr_path": str(output_dir / "train_stderr.txt"),
        "success": result.returncode == 0,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return result.returncode


class _Completed:
    def __init__(self, returncode: int, started_at: str, finished_at: str) -> None:
        self.returncode = returncode
        self.started_at = started_at
        self.finished_at = finished_at


def _run(command: list[str], stdout_path: Path, stderr_path: Path) -> _Completed:
    started_at = datetime.now(timezone.utc).isoformat()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.run(command, stdout=stdout_handle, stderr=stderr_handle, text=True)
    finished_at = datetime.now(timezone.utc).isoformat()
    return _Completed(process.returncode, started_at, finished_at)


if __name__ == "__main__":
    raise SystemExit(main())
