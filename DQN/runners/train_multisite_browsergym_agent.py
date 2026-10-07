from __future__ import annotations

import argparse
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.multisite_training_service import MultiSiteTrainingService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--total-updates", type=int, default=20)
    parser.add_argument("--episodes-per-site", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    parser.add_argument("--load-model", default="")
    parser.add_argument("--save-model", default="")
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--strict-site-validation", type=_parse_bool, default=False)
    parser.add_argument("--enable-csv-logging", type=_parse_bool, default=True)
    parser.add_argument("--csv-log-dir", default="artifacts/training")
    parser.add_argument("--log-observation-detail", type=_parse_bool, default=True)
    parser.add_argument("--log-action-space", type=_parse_bool, default=True)
    parser.add_argument("--log-raw-json", type=_parse_bool, default=False)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    service = MultiSiteTrainingService(
        config_path=args.config,
        total_updates=args.total_updates,
        episodes_per_site=args.episodes_per_site,
        max_steps=args.max_steps,
        seed=args.seed,
        headless=args.headless,
        load_model_path=args.load_model or None,
        save_model_path=args.save_model or None,
        entropy_coef=args.entropy_coef,
        strict_site_validation=args.strict_site_validation,
        enable_csv_logging=args.enable_csv_logging,
        csv_log_dir=args.csv_log_dir,
        log_observation_detail=args.log_observation_detail,
        log_action_space=args.log_action_space,
        log_raw_json=args.log_raw_json,
        run_id=args.run_id or None,
    )
    summary = service.train()
    print("shared_model_path:", summary["shared_model_path"])
    print("multisite_summary_path:", os.path.join(summary["output_dir"], "multisite_training_summary.json"))
    return 0


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
