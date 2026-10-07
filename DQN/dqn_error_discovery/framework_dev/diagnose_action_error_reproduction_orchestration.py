"""Diagnostic-only launcher for action-error reproductions.

This module owns subprocess orchestration only. It does not alter the frozen
runner, action semantics, masks, rewards, or browser interaction behavior.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = ROOT.parent / "BrowserGym" / ".venv" / "Scripts" / "python.exe"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _read_episode_count(events_path: Path) -> tuple[int, int]:
    if not events_path.exists():
        return 0, 0
    episodes: set[int] = set()
    rows = 0
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("episode") is not None:
                episodes.add(int(value["episode"]))
    return rows, len(episodes)


def build_command(*, python: Path, site_id: str, base_url: str, method: str, seed: int, output_dir: Path,
                  episodes: int, max_steps: int, max_local_steps: int, canonical_artifact: str,
                  canonical_error_count: int) -> list[str]:
    name = f"{site_id}_{method}_seed{seed}"
    dqn_checkpoint = ROOT / "artifacts" / "models" / f"site001_dqn_histv2_final_seed{seed}_30ep.pt"
    command = [
        str(python), "-m", "dqn_error_discovery.runners.run_ppo_dqn_hybrid",
        "--site-id", site_id, "--base-url", base_url,
        "--episodes", str(episodes), "--max-steps", str(max_steps),
        "--max-local-steps", str(max_local_steps), "--seed", str(seed),
        "--ppo-checkpoint", str(ROOT / "artifacts/models/site001_ppo_rewardv2_sanity.pt"),
        "--dqn-checkpoint", str(dqn_checkpoint),
        "--dqn-manifest", str(ROOT / "artifacts/models/dqn_local_specialist_final_manifest.json"),
        "--ppo-selection-mode", "stochastic", "--temporal-evidence",
        "--functional-region-novelty-gate", "--action-error-diagnostic",
        "--action-error-diagnostic-output", str(output_dir / f"{name}_events.jsonl"),
        "--diagnostic-canonical-artifact", canonical_artifact,
        "--diagnostic-canonical-error-count", str(canonical_error_count),
        "--output", str(output_dir / f"{name}_result.json"),
    ]
    if method == "ppo":
        command.append("--ppo-only")
    return command


def run_one(*, site_id: str, base_url: str, method: str, seed: int, output_dir: Path,
            python: Path, episodes: int, max_steps: int, max_local_steps: int,
            canonical_artifact: str = "", canonical_error_count: int = 0,
            poll_seconds: float = 2.0) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{site_id}_{method}_seed{seed}"
    events_path = output_dir / f"{name}_events.jsonl"
    result_path = output_dir / f"{name}_result.json"
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    lifecycle_path = output_dir / f"{name}_orchestration.json"
    lifecycle: dict[str, Any] = {
        "diagnostic_reproduction": True,
        "canonical_evaluation": False,
        "performance_metrics_eligible": False,
        "site": site_id, "method": method, "seed": seed,
        "attempt_index": 1, "python": str(python), "cwd": str(ROOT),
        "stages": [], "heartbeat": [], "owned_processes": [],
    }

    def stage(name_: str, state: str, **extra: Any) -> None:
        lifecycle["stages"].append({"stage": name_, "state": state, "timestamp": _now(), **extra})
        _atomic_write(lifecycle_path, lifecycle)

    stage("preflight", "CREATED", python_exists=python.exists(), cwd_exists=ROOT.exists())
    if not python.exists():
        stage("preflight", "FAILED", reason="PYTHON_ENVIRONMENT_MISMATCH")
        return lifecycle
    command = build_command(
        python=python, site_id=site_id, base_url=base_url, method=method, seed=seed,
        output_dir=output_dir, episodes=episodes, max_steps=max_steps,
        max_local_steps=max_local_steps, canonical_artifact=canonical_artifact,
        canonical_error_count=canonical_error_count,
    )
    lifecycle["command"] = command
    lifecycle["environment"] = {
        "PYTHONIOENCODING": "utf-8",
        "RAWD_ACTION_ERROR_DIAGNOSTIC": os.environ.get("RAWD_ACTION_ERROR_DIAGNOSTIC"),
    }
    stage("runner", "RUNNER_STARTING")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
        )
        lifecycle["pid"] = process.pid
        lifecycle["owned_processes"].append({"pid": process.pid, "role": "runner"})
        stage("runner", "RUNNER_ALIVE", pid=process.pid)
        first_trace_seen = False
        last_rows = 0
        last_episodes = 0
        while process.poll() is None:
            rows, episode_count = _read_episode_count(events_path)
            first_trace_seen = first_trace_seen or rows > 0
            last_rows, last_episodes = rows, episode_count
            lifecycle["heartbeat"].append({
                "timestamp": _now(), "pid": process.pid, "rows": rows,
                "episodes_seen": episode_count, "stdout_bytes": stdout_path.stat().st_size,
                "stderr_bytes": stderr_path.stat().st_size,
            })
            _atomic_write(lifecycle_path, lifecycle)
            time.sleep(poll_seconds)
        lifecycle["exit_code"] = process.returncode
    rows, episode_count = _read_episode_count(events_path)
    lifecycle["trace_rows"] = rows
    lifecycle["episodes_seen"] = episode_count
    lifecycle["result_exists"] = result_path.exists()
    lifecycle["artifact_flushed"] = result_path.exists() and events_path.exists()
    if process.returncode == 0 and episode_count == episodes and result_path.exists():
        stage("completion", "RUN_COMPLETED", rows=rows, episodes_seen=episode_count)
        stage("flush", "ARTIFACT_FLUSHED")
        stage("exit", "PROCESS_EXITED", exit_code=process.returncode)
        lifecycle["status"] = "COMPLETE"
    elif process.returncode == 0:
        stage("completion", "FAILED_ARTIFACT_FLUSH", rows=rows, episodes_seen=episode_count)
        lifecycle["status"] = "INVALID_ARTIFACT"
    elif not first_trace_seen:
        stage("runner", "FAILED_BEFORE_FIRST_TRACE", exit_code=process.returncode)
        lifecycle["status"] = "FAILED_BEFORE_FIRST_TRACE"
    else:
        stage("completion", "STALLED_DURING_EPISODE", rows=rows, episodes_seen=episode_count,
              exit_code=process.returncode)
        lifecycle["status"] = "INCOMPLETE_DIAGNOSTIC_RUN"
    stage("cleanup", "CLEANUP_COMPLETE")
    _atomic_write(lifecycle_path, lifecycle)
    return lifecycle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--method", choices=("ppo", "hybrid"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-local-steps", type=int, default=5)
    parser.add_argument("--canonical-artifact", default="")
    parser.add_argument("--canonical-error-count", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    report = run_one(**vars(args))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    raise SystemExit(0 if report.get("status") == "COMPLETE" else 2)


if __name__ == "__main__":
    main()
