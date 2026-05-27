from __future__ import annotations

import argparse
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.browsergym_training_service import BrowserGymTrainingService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--load-model", default="")
    parser.add_argument("--save-model", default="")
    args = parser.parse_args()

    service = BrowserGymTrainingService(
        site_id=args.site_id,
        base_url=args.base_url,
        episodes=args.episodes,
        max_steps=args.max_steps,
        headless=args.headless,
        seed=args.seed,
        entropy_coef=args.entropy_coef,
        load_model_path=args.load_model or None,
        model_output_path=args.save_model or None,
    )
    summary = service.train()
    print("training_summary_path:", service.summary_path)
    print("model_path:", summary["model_path"])
    return 0


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
