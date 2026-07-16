from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.imitation_prior import save_imitation_prior


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a goal-conditioned action prior from transition logs.")
    parser.add_argument("logs", nargs="+", help="JSONL transition logs")
    parser.add_argument("--output", default="artifacts/imitation/goal_action_prior.json")
    args = parser.parse_args()
    prior = save_imitation_prior(args.logs, args.output)
    print(f"saved {args.output} ({len(prior)} goals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
