"""Run equal-budget PPO/Rainbow-DQN comparisons and aggregate multiple seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--train-episodes", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--output-dir", default="artifacts/algorithm_ab_multiseed")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_root = ROOT / args.output_dir
    runs = []
    for seed in seeds:
        subprocess.run([
            sys.executable, str(ROOT / "scripts" / "run_algorithm_ab.py"),
            "--site-id", args.site_id, "--base-url", args.base_url,
            "--train-episodes", str(args.train_episodes), "--eval-episodes", str(args.eval_episodes),
            "--max-steps", str(args.max_steps), "--seed", str(seed),
            "--output-dir", args.output_dir,
        ], cwd=ROOT, check=True)
        comparison = output_root / f"{args.site_id}-seed-{seed}" / "comparison.json"
        runs.append(json.loads(comparison.read_text(encoding="utf-8")))

    algorithms = ("ppo", "rainbow-dqn")
    aggregate = {}
    for algorithm in algorithms:
        metrics = {}
        for key in ("recall", "precision", "average_reward", "false_positive_count"):
            values = [float(run["results"][algorithm].get(key, 0.0) or 0.0) for run in runs]
            metrics[key] = {
                "mean": statistics.fmean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        aggregate[algorithm] = metrics

    def rank(algorithm: str) -> tuple[float, float, float, float]:
        item = aggregate[algorithm]
        return (item["recall"]["mean"], item["precision"]["mean"],
                item["average_reward"]["mean"], -item["false_positive_count"]["mean"])

    selected = "rainbow-dqn" if rank("rainbow-dqn") > rank("ppo") else "ppo"
    report = {
        "site_id": args.site_id,
        "base_url": args.base_url,
        "seeds": seeds,
        "equal_budget": {"train_episodes": args.train_episodes,
                         "eval_episodes": args.eval_episodes, "max_steps": args.max_steps},
        "evaluation_mode": "pure-policy (guided actions disabled)",
        "aggregate": aggregate,
        "per_seed_winners": [run["selected_default_policy"] for run in runs],
        "selected_default_policy": selected,
        "selection_rule": "mean recall > precision > average reward > fewer false positives",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_root / "default_policy.json").write_text(json.dumps({"algorithm": selected}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
