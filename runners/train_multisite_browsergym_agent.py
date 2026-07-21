from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.multisite_training_service import MultiSiteTrainingService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_sites_v4_blind_urls.json")
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--episodes-per-site", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--load-model", default="")
    parser.add_argument("--save-model", default="")
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--strict-site-validation", type=_parse_bool, default=False)
    parser.add_argument("--enable-csv-logging", type=_parse_bool, default=True)
    parser.add_argument("--csv-log-dir", default="")
    parser.add_argument("--log-observation-detail", type=_parse_bool, default=True)
    parser.add_argument("--log-action-space", type=_parse_bool, default=True)
    parser.add_argument("--log-raw-json", type=_parse_bool, default=False)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--enable-guided-actions", type=_parse_bool, default=None)
    parser.add_argument("--debug-action-mask", type=_parse_bool, default=None)
    parser.add_argument("--use-memory-encoder", type=_parse_bool, default=None)
    parser.add_argument("--memory-encoder-type", default="")
    parser.add_argument("--memory-hidden-size", type=int, default=None)
    parser.add_argument("--fresh-start", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--fallback-penalty", type=float, default=None)
    parser.add_argument("--fallback-reward-cap-enabled", type=_parse_bool, default=None)
    parser.add_argument("--fallback-reward-cap-value", type=float, default=None)
    parser.add_argument("--fallback-curriculum-enabled", type=_parse_bool, default=None)
    parser.add_argument("--fail-on-episode-exception", type=_parse_bool, default=None)
    args = parser.parse_args()
    config_defaults = _read_config_defaults(args.config)

    service = MultiSiteTrainingService(
        config_path=args.config,
        total_updates=_int_arg(args.total_updates, config_defaults, "total_updates", 20),
        episodes_per_site=_int_arg(args.episodes_per_site, config_defaults, "episodes_per_site", 1),
        max_steps=_int_arg(args.max_steps, config_defaults, "max_steps", 25),
        seed=args.seed,
        headless=args.headless,
        load_model_path=args.load_model or None,
        save_model_path=args.model_path or args.save_model or None,
        entropy_coef=args.entropy_coef,
        strict_site_validation=args.strict_site_validation,
        enable_csv_logging=args.enable_csv_logging,
        csv_log_dir=_str_arg(args.csv_log_dir, config_defaults, "training_log_dir", "artifacts/training_logs/v3_policy_safe"),
        log_observation_detail=args.log_observation_detail,
        log_action_space=args.log_action_space,
        log_raw_json=args.log_raw_json,
        run_id=args.run_id or str(config_defaults.get("run_id") or config_defaults.get("config_name") or "") or None,
        enable_guided_actions=_bool_arg(args.enable_guided_actions, config_defaults, "enable_guided_actions", False),
        debug_action_mask=_bool_arg(args.debug_action_mask, config_defaults, "debug_action_mask", False),
        use_memory_encoder=_bool_arg(args.use_memory_encoder, config_defaults, "use_memory_encoder", False),
        memory_encoder_type=_str_arg(args.memory_encoder_type, config_defaults, "memory_encoder_type", "gru"),
        memory_hidden_size=_int_arg(args.memory_hidden_size, config_defaults, "memory_hidden_size", 128),
        fresh_start=_optional_bool_arg(args.fresh_start, config_defaults, "fresh_start"),
        resume=_optional_bool_arg(args.resume, config_defaults, "resume"),
        fallback_penalty=_float_arg(args.fallback_penalty, config_defaults, "fallback_penalty", 0.25),
        fallback_reward_cap_enabled=_bool_arg(args.fallback_reward_cap_enabled, config_defaults, "fallback_reward_cap_enabled", True),
        fallback_reward_cap_value=_float_arg(args.fallback_reward_cap_value, config_defaults, "fallback_reward_cap_value", 0.0),
        fallback_curriculum_enabled=_bool_arg(args.fallback_curriculum_enabled, config_defaults, "fallback_curriculum_enabled", True),
        fail_on_episode_exception=_bool_arg(args.fail_on_episode_exception, config_defaults, "fail_on_episode_exception", False),
    )
    summary = service.train()
    print("shared_model_path:", summary["shared_model_path"])
    print("multisite_summary_path:", os.path.join(summary["output_dir"], "multisite_training_summary.json"))
    return 0


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _read_config_defaults(path: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _int_arg(value: int | None, config: dict, key: str, fallback: int) -> int:
    if value is not None:
        return int(value)
    try:
        return int(config.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _bool_arg(value: bool | None, config: dict, key: str, fallback: bool) -> bool:
    if value is not None:
        return bool(value)
    if key not in config:
        return fallback
    return _parse_bool(config.get(key))


def _float_arg(value: float | None, config: dict, key: str, fallback: float) -> float:
    if value is not None:
        return float(value)
    try:
        return float(config.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _optional_bool_arg(value: bool | None, config: dict, key: str) -> bool | None:
    if value is not None:
        return bool(value)
    if key in config:
        return _parse_bool(config.get(key))
    return None


def _str_arg(value: str | None, config: dict, key: str, fallback: str) -> str:
    if value:
        return str(value)
    if key in config and config.get(key):
        return str(config.get(key))
    return fallback


if __name__ == "__main__":
    raise SystemExit(main())
