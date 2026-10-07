"""Run the frozen PPO-only and Framework-aware Hybrid v2 unseen evaluation.

Runtime-only runner: this module intentionally never imports BUGS.md,
scenario catalogs, or any post-hoc evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SITES = {
    "site002": "http://localhost:9221/",
    "site003": "http://localhost:9222/",
    "site004": "http://localhost:9223/",
    "site005": "http://localhost:9224/",
}
SEEDS = (42, 43, 44)
METHODS = ("ppo", "hybrid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(root: Path, output: Path, site_id: str, method: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = root / "artifacts" / "models" / f"site001_dqn_histv2_final_seed{seed}_30ep.pt"
    result = output / site_id.upper() / f"{method}_seed{seed}.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "dqn_error_discovery.runners.run_ppo_dqn_hybrid",
        "--site-id",
        site_id,
        "--base-url",
        SITES[site_id],
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--max-local-steps",
        str(args.max_local_steps),
        "--seed",
        str(seed),
        "--ppo-checkpoint",
        "artifacts/models/site001_ppo_rewardv2_sanity.pt",
        "--dqn-checkpoint",
        str(checkpoint.relative_to(root)).replace("/", "\\"),
        "--dqn-manifest",
        "artifacts/models/dqn_local_specialist_final_manifest.json",
        "--ppo-selection-mode",
        "stochastic",
        "--temporal-evidence",
        "--functional-region-novelty-gate",
        "--output",
        str(result.relative_to(root)).replace("/", "\\"),
        "--quiet",
    ]
    if method == "ppo":
        command.append("--ppo-only")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=root, check=False)
    elapsed = time.perf_counter() - started
    sidecar = {
        "site_id": site_id,
        "method": method,
        "seed": seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "max_local_steps": args.max_local_steps,
        "ppo_selection_mode": "stochastic",
        "python": sys.executable,
        "base_url": SITES[site_id],
        "exit_code": completed.returncode,
        "wall_time_sec": elapsed,
        "runtime_ground_truth_imported": False,
    }
    sidecar_path = result.with_name(result.stem + "_runtime.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"evaluation failed: {site_id} {method} seed{seed}: {completed.returncode}")
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-local-steps", type=int, default=5)
    parser.add_argument("--output", default="artifacts/final_unseen_eval")
    parser.add_argument("--only-site", choices=sorted(SITES), default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    selected_sites = [args.only_site] if args.only_site else list(SITES)
    schedule = [(site, method, seed) for site in selected_sites for seed in SEEDS for method in METHODS]
    sidecars = []
    for site_id, method, seed in schedule:
        sidecars.append(_run(root, output, site_id, method, seed, args))
    manifest = {
        "evaluation_only": True,
        "runtime_ground_truth_imported": False,
        "algorithm_changes": 0,
        "infrastructure_changes": "evaluation-only runner and PPO-only flag",
        "python": sys.executable,
        "sites": SITES,
        "methods": METHODS,
        "seeds": list(SEEDS),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "max_local_steps": args.max_local_steps,
        "ppo_selection_mode": "stochastic",
        "temporal_config": {
            "sample_interval_ms": 100,
            "minimum_observation_ms": 250,
            "stable_quiet_window_ms": 350,
            "maximum_observation_ms": 1500,
        },
        "schedule": schedule,
        "runs": sidecars,
        "gt_posthoc_reserved": True,
    }
    (output / "final_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
