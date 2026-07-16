"""Train and evaluate PPO and masked Rainbow DQN with identical budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", default="true")
    parser.add_argument("--output-dir", default="artifacts/algorithm_ab")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir / f"{args.site_id}-seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for algorithm, slug in (("ppo", "ppo"), ("rainbow-dqn", "rainbow_dqn")):
        model = output_dir / f"{slug}.pt"
        train_dir = output_dir / f"{slug}_training"
        evaluation = output_dir / f"{slug}_evaluation.json"
        _run([
            sys.executable, str(ROOT / "runners" / "train_browsergym_agent.py"),
            "--site-id", args.site_id, "--base-url", args.base_url,
            "--episodes", str(args.train_episodes), "--max-steps", str(args.max_steps),
            "--seed", str(args.seed), "--headless", args.headless,
            "--algorithm", algorithm, "--save-model", str(model),
        ], cwd=ROOT)
        _run([
            sys.executable, str(ROOT / "runners" / "evaluate_browsergym_agent.py"),
            "--site-id", args.site_id, "--base-url", args.base_url, "--model-path", str(model),
            "--episodes", str(args.eval_episodes), "--max-steps", str(args.max_steps),
            "--seed", str(args.seed), "--headless", args.headless,
            "--algorithm", algorithm, "--guided-actions", "false", "--output", str(evaluation),
        ], cwd=ROOT)
        results[algorithm] = json.loads(evaluation.read_text(encoding="utf-8"))

    selected, rationale = choose_default_policy(results["ppo"], results["rainbow-dqn"])
    report = {
        "site_id": args.site_id,
        "base_url": args.base_url,
        "equal_budget": {
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        "results": results,
        "selected_default_policy": selected,
        "selection_rationale": rationale,
        "selection_rule": "recall > precision > average_reward > fewer false positives",
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "default_policy.json").write_text(json.dumps({"algorithm": selected, "model": f"{selected.replace('-', '_')}.pt"}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def choose_default_policy(ppo: dict[str, Any], dqn: dict[str, Any]) -> tuple[str, list[str]]:
    keys = ("recall", "precision", "average_reward")
    ppo_tuple = tuple(float(ppo.get(key, 0.0) or 0.0) for key in keys) + (-int(ppo.get("false_positive_count", 0) or 0),)
    dqn_tuple = tuple(float(dqn.get(key, 0.0) or 0.0) for key in keys) + (-int(dqn.get("false_positive_count", 0) or 0),)
    selected = "rainbow-dqn" if dqn_tuple > ppo_tuple else "ppo"
    winner, loser = (dqn, ppo) if selected == "rainbow-dqn" else (ppo, dqn)
    rationale = [
        f"recall {float(winner.get('recall', 0) or 0):.4f} vs {float(loser.get('recall', 0) or 0):.4f}",
        f"precision {float(winner.get('precision', 0) or 0):.4f} vs {float(loser.get('precision', 0) or 0):.4f}",
        f"average_reward {float(winner.get('average_reward', 0) or 0):.4f} vs {float(loser.get('average_reward', 0) or 0):.4f}",
    ]
    return selected, rationale


def _run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


if __name__ == "__main__":
    raise SystemExit(main())
