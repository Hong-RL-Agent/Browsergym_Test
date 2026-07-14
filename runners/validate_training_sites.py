from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.ppo_agent import PPOAgent
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward
from services.browsergym_training_service import _enrich_action, _update_history
from services.policy_safe_metrics import TRAINING_LOG_DIR, VERSION, utc_now_iso, write_json


DEFAULT_OUTPUT = TRAINING_LOG_DIR / "site_validation_summary.json"
REQUIRED_RESULT_FIELDS = (
    "site_id",
    "base_url",
    "status",
    "reset_ok",
    "step_ok",
    "candidate_count",
    "valid_action_count",
    "valid_click_count",
    "valid_fill_count",
    "valid_global_action_count",
    "reward_total",
    "error_message",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--debug-action-mask", type=_parse_bool, default=False)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--headless", type=_parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = _read_json_dict(Path(args.config))
    configured_sites = _configured_sites(config.get("sites", []))
    sites = _validate_sites(config.get("sites", []))
    encoder = ObservationEncoder()
    action_space = ActionSpace()
    agent = PPOAgent(encoder.get_obs_dim(), action_space.get_action_dim())

    results = []
    for site in sites:
        result = validate_site(
            site,
            encoder,
            action_space,
            agent,
            max_steps=max(1, int(args.max_steps)),
            headless=args.headless,
            debug_action_mask=args.debug_action_mask,
        )
        results.append(result)

    summary = build_validation_summary(results, configured_sites=configured_sites, enabled_sites=sites)
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def validate_site(
    site: Mapping[str, Any],
    encoder: ObservationEncoder,
    action_space: ActionSpace,
    agent: PPOAgent,
    *,
    max_steps: int,
    headless: bool,
    debug_action_mask: bool,
) -> Dict[str, Any]:
    site_id = str(site.get("site_id") or "")
    base_url = str(site.get("base_url") or "")
    result = _empty_result(site_id, base_url)
    env = None
    history: Dict[str, Any] = {
        "clicked_bids": set(),
        "last_action_key": None,
        "last_action_type": None,
        "action_type_counts": {},
        "seen_anomaly_keys": set(),
    }
    try:
        if not site_id:
            raise ValueError("site_id is required")
        if not base_url:
            raise ValueError("base_url is required")
        env = BrowserGymJAWSEnv(
            site_id=site_id,
            base_url=base_url,
            max_steps=max_steps,
            headless=headless,
            requires_login=bool(site.get("requires_login")),
            login_config=site.get("login") if isinstance(site.get("login"), Mapping) else None,
        )
        observation, _ = env.reset()
        result["reset_ok"] = True
        action_mask = action_space.build_action_mask(observation)
        selected_probe = int(np.flatnonzero(action_mask)[0]) if np.count_nonzero(action_mask) else 0
        stats = action_space.build_action_mask_stats(observation, action_mask, selected_probe)
        _merge_stats(result, stats)
        if int(result["candidate_count"]) <= 0:
            raise RuntimeError("candidate_count is 0")
        if int(result["valid_action_count"]) <= 0:
            raise RuntimeError("valid_action_count is 0")

        reward_total = 0.0
        done = False
        for step in range(1, max_steps + 1):
            obs_vector = encoder.encode_observation(observation)
            action_mask = action_space.build_action_mask(observation)
            selected = agent.select_action(obs_vector, action_mask)
            action_id = int(selected["action_id"])
            stats = action_space.build_action_mask_stats(observation, action_mask, action_id)
            _merge_stats(result, stats)
            if debug_action_mask:
                _log_action_mask_stats(site_id, step, stats)

            action = action_space.decode(action_id)
            action["action_id"] = action_id
            action["site_id"] = site_id
            _enrich_action(action, observation)
            next_observation, _, done, step_info = env.step(action_id)
            result["step_ok"] = True
            action["failed"] = bool(step_info.get("last_action_error"))
            anomalies = detect_anomalies(observation, next_observation, {"action": action, **step_info})
            reward, reward_breakdown = calculate_autonomous_reward(
                observation,
                next_observation,
                action,
                anomalies,
                [],
                history,
            )
            if not math.isfinite(float(reward_breakdown.get("reward_total", reward))):
                raise RuntimeError("reward_total is not finite")
            reward_total += float(reward)
            _update_history(history, observation, action, anomalies)
            observation = next_observation
            if done:
                break

        result["reward_total"] = reward_total
        result["status"] = "ok" if result["reset_ok"] and result["step_ok"] else "failed"
    except Exception as exc:
        result["status"] = "failed"
        result["error_message"] = str(exc)
    finally:
        if env is not None:
            env.close()
    return _complete_result(result)


def build_validation_summary(
    results: list[Mapping[str, Any]],
    *,
    configured_sites: list[Mapping[str, Any]] | None = None,
    enabled_sites: list[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    completed = [_complete_result(dict(item)) for item in results]
    ok_site_ids = [str(item.get("site_id") or "") for item in completed if item.get("status") == "ok"]
    failed_site_ids = [str(item.get("site_id") or "") for item in completed if item.get("status") != "ok"]
    configured_sites = configured_sites or enabled_sites or []
    enabled_sites = enabled_sites or configured_sites
    disabled_site_ids = [
        str(site.get("site_id") or "")
        for site in configured_sites
        if isinstance(site, Mapping) and site.get("enabled") is False and site.get("site_id")
    ]
    return {
        "version": VERSION,
        "configured_site_ids": [str(site.get("site_id") or "") for site in configured_sites if site.get("site_id")],
        "enabled_site_ids": [str(site.get("site_id") or "") for site in enabled_sites if site.get("site_id")],
        "disabled_site_ids": disabled_site_ids,
        "skipped_site_ids": disabled_site_ids,
        "site_groups": {
            str(site.get("site_id") or ""): str(site.get("site_group") or "")
            for site in configured_sites
            if site.get("site_id")
        },
        "target_signal_types_by_site": {
            str(site.get("site_id") or ""): list(site.get("target_signal_types") or [])
            for site in configured_sites
            if site.get("site_id")
        },
        "total_sites": len(completed),
        "ok_sites": len(ok_site_ids),
        "failed_sites": len(failed_site_ids),
        "ok_site_ids": ok_site_ids,
        "failed_site_ids": failed_site_ids,
        "created_at": utc_now_iso(),
        "results": completed,
    }


def _empty_result(site_id: str, base_url: str) -> Dict[str, Any]:
    return {
        "site_id": site_id,
        "base_url": base_url,
        "status": "failed",
        "reset_ok": False,
        "step_ok": False,
        "candidate_count": 0,
        "valid_action_count": 0,
        "valid_click_count": 0,
        "valid_fill_count": 0,
        "valid_global_action_count": 0,
        "reward_total": 0.0,
        "error_message": "",
    }


def _complete_result(result: Dict[str, Any]) -> Dict[str, Any]:
    base = _empty_result(str(result.get("site_id") or ""), str(result.get("base_url") or ""))
    base.update(result)
    for key in REQUIRED_RESULT_FIELDS:
        base.setdefault(key, _empty_result("", "").get(key))
    return base


def _merge_stats(result: Dict[str, Any], stats: Mapping[str, Any]) -> None:
    for key in (
        "candidate_count",
        "valid_action_count",
        "valid_click_count",
        "valid_fill_count",
        "valid_global_action_count",
    ):
        result[key] = int(stats.get(key, result.get(key, 0)) or 0)


def _log_action_mask_stats(site_id: str, step: int, stats: Mapping[str, Any]) -> None:
    print(
        "[mask] "
        f"site={site_id} step={step} "
        f"candidates={stats.get('candidate_count', 0)} "
        f"valid={stats.get('valid_action_count', 0)} "
        f"click={stats.get('valid_click_count', 0)} "
        f"fill={stats.get('valid_fill_count', 0)} "
        f"global={stats.get('valid_global_action_count', 0)}"
    )


def _read_json_dict(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _validate_sites(raw_sites: Any) -> list[Dict[str, Any]]:
    if not isinstance(raw_sites, list):
        raise ValueError("config requires a sites list")
    sites = [dict(item) for item in raw_sites if isinstance(item, Mapping) and item.get("enabled") is not False]
    if not sites:
        raise ValueError("no valid sites found")
    return sites


def _configured_sites(raw_sites: Any) -> list[Dict[str, Any]]:
    if not isinstance(raw_sites, list):
        return []
    return [dict(item) for item in raw_sites if isinstance(item, Mapping)]


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
