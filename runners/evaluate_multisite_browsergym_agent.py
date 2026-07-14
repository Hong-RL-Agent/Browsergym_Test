from __future__ import annotations

import argparse
import base64
import json
import os
import random
import shutil
import sys
import textwrap
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.ppo_agent import PPOAgent
from envs.browsergym_jaws_env import BrowserGymJAWSEnv, _active_page
from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import apply_fallback_reward_policy, calculate_autonomous_reward
from services.browsergym_training_service import (
    _enrich_action,
    _has_openended_interactive_candidate,
    _target_bid,
    _update_history,
)
from services.episode_csv_logger import EpisodeCsvLogger, infer_run_id
from services.infra_anomaly_detection_service import detect_infra_anomalies
from services.infra_reward_service import calculate_infra_reward
from services.known_bug_matcher import load_known_bugs, match_anomalies_to_known_bugs
from services.policy_safe_metrics import (
    VERSION,
    aggregate_site_metrics,
    build_per_site_metric,
    split_site_metrics,
    strip_known_bug_metrics_for_signal_discovery,
    utc_now_iso,
    write_comparison_note,
    write_json,
)
from services.site_profile_service import build_site_profile, validate_site_identity


def _emit_event(**payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


_PROCESS_STARTED_AT = time.monotonic()
_STAGE_STATE: Dict[str, Any] = {
    "currentStage": "",
    "lastCompletedStage": "",
    "elapsedMs": 0,
}


def _emit_stage(stage: str, **payload: Any) -> None:
    elapsed_ms = int((time.monotonic() - _PROCESS_STARTED_AT) * 1000)
    _STAGE_STATE["currentStage"] = stage
    _STAGE_STATE["elapsedMs"] = elapsed_ms
    if stage.endswith("_completed") or stage in {"config_loaded", "result_written", "python_process_started"}:
        _STAGE_STATE["lastCompletedStage"] = stage
    _emit_event(
        event="stage",
        stage=stage,
        timestamp=datetime.now(timezone.utc).isoformat(),
        elapsedMs=elapsed_ms,
        **payload,
    )


def _failure_type_from_stage(stage: str) -> str:
    normalized = (stage or "").lower()
    if "model_loading" in normalized:
        return "MODEL_LOAD_TIMEOUT"
    if "browser_launch" in normalized:
        return "BROWSER_LAUNCH_TIMEOUT"
    if "site_navigation" in normalized:
        return "SITE_NAVIGATION_TIMEOUT"
    if "env_reset" in normalized:
        return "ENV_RESET_TIMEOUT"
    if "step_" in normalized:
        return "STEP_TIMEOUT"
    return "PYTHON_TIMEOUT"


def _action_target_text(action: Mapping[str, Any]) -> str:
    for key in ("clicked_text", "clicked_bid", "input_label", "target_text", "selector_hint"):
        value = action.get(key)
        if value:
            return str(value)
    return ""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-site-validation", type=_parse_bool, default=False)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--strict-metrics", type=_parse_bool, default=False)
    parser.add_argument("--output", default="")
    parser.add_argument("--enable-csv-logging", type=_parse_bool, default=True)
    parser.add_argument("--csv-log-dir", default="")
    parser.add_argument("--log-observation-detail", type=_parse_bool, default=True)
    parser.add_argument("--log-action-space", type=_parse_bool, default=True)
    parser.add_argument("--log-raw-json", type=_parse_bool, default=False)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--site-timeout-seconds", type=int, default=None)
    parser.add_argument("--episode-timeout-seconds", type=int, default=None)
    parser.add_argument("--no-progress-patience", type=int, default=None)
    parser.add_argument("--page-timeout-ms", "--page-load-timeout-ms", dest="page_timeout_ms", type=int, default=15000)
    parser.add_argument("--navigation-timeout-ms", type=int, default=15000)
    parser.add_argument("--env-reset-timeout-ms", "--reset-timeout-ms", dest="env_reset_timeout_ms", type=int, default=None)
    parser.add_argument("--reset-retry-count", type=int, default=None)
    parser.add_argument("--step-timeout-ms", type=int, default=10000)
    parser.add_argument("--step-timeout-seconds", type=int, default=None)
    parser.add_argument("--use-memory-encoder", type=_parse_bool, default=None)
    parser.add_argument("--memory-encoder-type", default="")
    parser.add_argument("--memory-hidden-size", type=int, default=None)
    parser.add_argument("--disable-fallback", type=_parse_bool, default=None)
    parser.add_argument("--fallback-mode", choices=["train", "eval", "strict"], default="")
    parser.add_argument("--fallback-penalty", type=float, default=None)
    parser.add_argument("--fallback-reward-cap-enabled", type=_parse_bool, default=None)
    parser.add_argument("--fallback-reward-cap-value", type=float, default=None)
    try:
        args = parser.parse_args()
    except SystemExit as exc:
        if exc.code:
            _emit_event(event="scan_failed", error_type="argparse_error", message=f"argparse exited with code {exc.code}")
        raise

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.step_timeout_seconds is not None:
        args.step_timeout_ms = int(args.step_timeout_seconds) * 1000

    try:
        config = _read_json_dict(Path(args.config))
        configured_sites = _configured_sites(config.get("sites", []))
        sites = _validate_sites(config.get("sites", []))
        _apply_evaluation_defaults_to_sites(sites, config)
        site_config_summary = _site_config_summary(configured_sites, sites)
        config_run_id = str(config.get("run_id") or config.get("config_name") or "")
        eval_dir = Path(str(config.get("evaluation_output_dir") or "artifacts/evaluation/v3_policy_safe"))
        args.output = args.output or str(eval_dir / "evaluation_summary.json")
        args.csv_log_dir = args.csv_log_dir or str(eval_dir)
        args.model_path = args.model_path or str(config.get("shared_model_path") or "artifacts/models/jaws_browsergym_shared_ppo_v3_policy_safe.pt")
        args.run_id = args.run_id or config_run_id
        args.episodes = _eval_episodes_arg(args.episodes, config, 3)
        args.max_steps = _int_arg(args.max_steps, config, "max_steps", 25)
        args.site_timeout_seconds = _int_arg(args.site_timeout_seconds, config, "site_timeout_seconds", 60)
        args.episode_timeout_seconds = _int_arg(args.episode_timeout_seconds, config, "episode_timeout_seconds", 0)
        args.no_progress_patience = _int_arg(args.no_progress_patience, config, "no_progress_patience", 0)
        args.env_reset_timeout_ms = _int_arg(args.env_reset_timeout_ms, config, "reset_timeout_ms", 30000)
        args.reset_retry_count = _int_arg(args.reset_retry_count, config, "reset_retry_count", 0)
        disable_fallback = _bool_arg(args.disable_fallback, config, "disable_fallback", False)
        fallback_mode = str(args.fallback_mode or config.get("fallback_mode") or ("strict" if disable_fallback else "eval"))
        fallback_penalty = _float_arg(args.fallback_penalty, config, "fallback_penalty", 0.25)
        fallback_reward_cap_enabled = _bool_arg(
            args.fallback_reward_cap_enabled,
            config,
            "fallback_reward_cap_enabled",
            True,
        )
        fallback_reward_cap_value = _float_arg(args.fallback_reward_cap_value, config, "fallback_reward_cap_value", 0.0)
        use_memory_encoder = _bool_arg(args.use_memory_encoder, config, "use_memory_encoder", False)
        memory_encoder_type = _str_arg(args.memory_encoder_type, config, "memory_encoder_type", "gru")
        memory_hidden_size = _int_arg(args.memory_hidden_size, config, "memory_hidden_size", 128)
        initial_scan_id = _run_id_from_output(args.output) or args.run_id or "scan"
        _emit_stage("python_process_started", output=args.output)
        _write_live_placeholder(
            args.output,
            scan_id=initial_scan_id,
            status="starting",
            stage="python_process_started",
            target_url="",
            message="Browser is starting...",
        )
        _emit_stage("config_loaded", config=args.config, site_count=len(sites))
        first_target_url = str(sites[0].get("base_url") or "") if sites else ""
        _write_live_placeholder(
            args.output,
            scan_id=initial_scan_id,
            status="starting",
            stage="config_loaded",
            target_url=first_target_url,
            message="Browser configuration loaded...",
        )
    except Exception as exc:
        _emit_event(event="scan_failed", error_type=exc.__class__.__name__, message=str(exc), config=args.config)
        raise
    run_id = args.run_id or _run_id_from_output(args.output) or infer_run_id(config, args.config, fallback_prefix="evaluation")
    batch_id = Path(args.config).stem
    encoder = ObservationEncoder()
    action_space = ActionSpace()
    agent = PPOAgent(
        encoder.get_obs_dim(),
        action_space.get_action_dim(),
        use_memory_encoder=use_memory_encoder,
        memory_encoder_type=memory_encoder_type,
        memory_hidden_size=memory_hidden_size,
    )
    try:
        _emit_stage("model_loading_started", model_path=args.model_path)
        agent.load(args.model_path)
        _emit_stage("model_loading_completed", model_path=args.model_path)
    except Exception as exc:
        _emit_event(event="scan_failed", error_type=exc.__class__.__name__, message=str(exc), model_path=args.model_path)
        raise
    csv_logger = (
        EpisodeCsvLogger(
            args.csv_log_dir,
            run_id,
            batch_id=batch_id,
            phase="evaluation",
            log_observation_detail=args.log_observation_detail,
            log_action_space=args.log_action_space,
            log_raw_json=args.log_raw_json,
        )
        if args.enable_csv_logging
        else None
    )

    _emit_event(
        event="scan_started",
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        site_count=len(sites),
    )

    site_results = {}
    rewards = []
    try:
        for site in sites:
            try:
                result = _evaluate_site(
                    site,
                    agent,
                    encoder,
                    action_space,
                    args.episodes,
                    args.max_steps,
                    args.headless,
                    args.strict_site_validation,
                    csv_logger=csv_logger,
                    site_timeout_seconds=args.site_timeout_seconds,
                    episode_timeout_seconds=args.episode_timeout_seconds,
                    no_progress_patience=args.no_progress_patience,
                    page_timeout_ms=args.page_timeout_ms,
                    navigation_timeout_ms=args.navigation_timeout_ms,
                    env_reset_timeout_ms=args.env_reset_timeout_ms,
                    reset_retry_count=args.reset_retry_count,
                    step_timeout_ms=args.step_timeout_ms,
                    output_path=args.output,
                    partial_transition_path=_partial_transition_path(args.output),
                    use_memory_encoder=use_memory_encoder,
                    memory_encoder_type=memory_encoder_type,
                    memory_hidden_size=memory_hidden_size,
                    fallback_mode=fallback_mode,
                    fallback_penalty=fallback_penalty,
                    fallback_reward_cap_enabled=fallback_reward_cap_enabled,
                    fallback_reward_cap_value=fallback_reward_cap_value,
                )
            except Exception as exc:
                site_id = str(site.get("site_id") or "")
                _write_live_status(
                    _live_dir(args.output),
                    scan_id=run_id,
                    status="failed",
                    stage="site_failed",
                    current_site=site_id,
                    last_screenshot_ok=False,
                    last_screenshot_error=str(exc),
                )
                _emit_event(
                    event="site_failed",
                    site_id=site_id,
                    base_url=str(site.get("base_url") or ""),
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                )
                _write_partial_output(args, run_id, site_results, rewards, exc, site)
                result = _failed_site_result(
                    site,
                    exc,
                    episodes=args.episodes,
                    completed_episodes=0,
                    reset_timeout_ms=args.env_reset_timeout_ms,
                    reset_retry_count=args.reset_retry_count,
                )
            site_results[site["site_id"]] = result
            rewards.append(result["average_reward"])
    finally:
        if csv_logger is not None:
            csv_logger.close()

    per_site_metrics = [
        build_per_site_metric(site_id, result)
        for site_id, result in site_results.items()
        if isinstance(result, Mapping)
    ]
    aggregate_metrics = aggregate_site_metrics(per_site_metrics)
    split_metrics = split_site_metrics(per_site_metrics)
    config_target_signal_types = list(config.get("target_signal_types", []) or [])
    aggregate_observed_signal_types = list(aggregate_metrics.get("observed_signal_types", []) or [])
    offline_known_bug_eval_metrics = {
        key: aggregate_metrics.get(key)
        for key in (
            "precision",
            "recall",
            "f1_score",
            "true_positive_count",
            "false_positive_count",
            "false_negative_count",
            "known_bug_total",
            "known_bug_matched_count",
        )
    }
    use_known_bug_for_evaluation = bool(config.get("use_known_bug_for_evaluation", False))
    top_level_metrics = dict(aggregate_metrics)
    if not use_known_bug_for_evaluation:
        for key in offline_known_bug_eval_metrics:
            top_level_metrics[key] = None
        split_metrics = strip_known_bug_metrics_for_signal_discovery(split_metrics)
    output_path = Path(args.output) if args.output else Path("artifacts/evaluation/v3_policy_safe/evaluation_summary.json")
    per_site_metrics_path = output_path.parent / "per_site_metrics.json"
    comparison_note_path = output_path.parent / "comparison_note.json"
    output = {
        "version": VERSION,
        "run_id": run_id,
        "config_name": str(config.get("config_name") or config.get("run_id") or Path(args.config).stem),
        "config_path": str(args.config),
        "status": "completed",
        "reportType": "Autonomous Anomaly Review Report",
        "model_path": args.model_path,
        "output_dir": str(output_path.parent),
        "evaluated_at": utc_now_iso(),
        "site_count": len(site_results),
        **site_config_summary,
        "episode_count": args.episodes * len(site_results),
        "eval_episodes": args.episodes,
        "total_updates": int(config.get("total_updates", 0) or 0),
        "episodes_per_site": int(config.get("episodes_per_site", 0) or 0),
        "failed_site_count": sum(1 for result in site_results.values() if result.get("status") == "failed"),
        "failed_site_ids": [site_id for site_id, result in site_results.items() if result.get("status") == "failed"],
        "partial": any(bool(result.get("partial")) for result in site_results.values()),
        "site_timeout_seconds": args.site_timeout_seconds,
        "episode_timeout_seconds": args.episode_timeout_seconds,
        "no_progress_patience": args.no_progress_patience,
        "reset_timeout_ms": args.env_reset_timeout_ms,
        "reset_retry_count": args.reset_retry_count,
        "fallback_mode": fallback_mode,
        "disable_fallback": disable_fallback,
        "fallback_penalty": fallback_penalty,
        "fallback_reward_cap_enabled": fallback_reward_cap_enabled,
        "fallback_reward_cap_value": fallback_reward_cap_value,
        "reward_mode": str(config.get("reward_mode") or "signal_based"),
        "allow_known_bug_reward_for_debug_only": False,
        "use_ground_truth_for_evaluation_only": use_known_bug_for_evaluation,
        "use_known_bug_reward": False,
        "use_known_bug_for_training": False,
        "use_known_bug_for_evaluation": use_known_bug_for_evaluation,
        "training_uses_site_specific_bug_catalog": bool(config.get("training_uses_site_specific_bug_catalog", False)),
        "policy_uses_bug_labels": False,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "memory_encoder_enabled": use_memory_encoder,
        "use_memory_encoder": use_memory_encoder,
        "memory_encoder_type": memory_encoder_type if use_memory_encoder else "",
        "memory_hidden_size": memory_hidden_size if use_memory_encoder else 0,
        "episode_memory_reset_count": sum(int(result.get("episode_memory_reset_count", 0) or 0) for result in site_results.values()),
        "memory_state_reset_count": sum(int(result.get("episode_memory_reset_count", 0) or 0) for result in site_results.values()),
        "memory_state_update_count": (
            sum(int(result.get("memory_state_update_count", 0) or 0) for result in site_results.values())
            if use_memory_encoder
            else 0
        ),
        "memory_state_norm_mean": _mean_memory_state_norm_from_results(site_results.values()) if use_memory_encoder else 0.0,
        "repeated_action_rate": _repeated_action_rate_from_results(site_results.values()),
        "repeated_click_target_rate": _repeated_click_target_rate_from_results(site_results.values()),
        "repeated_action_rate_before_memory": None,
        "repeated_action_rate_after_memory": _repeated_action_rate_from_results(site_results.values()) if use_memory_encoder else None,
        "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "fallback_applied_count": int(aggregate_metrics.get("fallback_applied_count") or 0),
        "fallback_applied_rate": float(aggregate_metrics.get("fallback_applied_rate") or 0.0),
        "fallback_penalty_total": float(aggregate_metrics.get("fallback_penalty_total") or 0.0),
        "fallback_reward_capped_count": int(aggregate_metrics.get("fallback_reward_capped_count") or 0),
        "policy_executed_action_match_count": int(aggregate_metrics.get("policy_executed_action_match_count") or 0),
        "policy_executed_action_mismatch_count": int(aggregate_metrics.get("policy_executed_action_mismatch_count") or 0),
        "per_site_metrics_path": str(per_site_metrics_path),
        "per_site_signal_coverage": {
            site_id: {
                "target_signal_types": result.get("target_signal_types", []),
                "observed_signal_types": result.get("observed_signal_types", []),
                "missing_signal_types": result.get("missing_signal_types", []),
                "signal_type_coverage": result.get("signal_type_coverage", 0.0),
            }
            for site_id, result in site_results.items()
            if isinstance(result, Mapping)
        },
        "offline_known_bug_eval_metrics": offline_known_bug_eval_metrics,
        "deprecated_known_bug_metrics": offline_known_bug_eval_metrics,
        **top_level_metrics,
        **split_metrics,
        "target_signal_types": config_target_signal_types,
        "observed_signal_types": aggregate_observed_signal_types,
        "missing_signal_types": sorted(set(config_target_signal_types) - set(aggregate_observed_signal_types)),
        "signal_type_coverage": _signal_type_coverage(config_target_signal_types, aggregate_observed_signal_types),
        "sites": site_results,
    }
    live_dir = _live_dir(args.output)
    if live_dir is not None:
        latest_screenshot = live_dir / "latest.png"
        screenshots = sorted(str(path) for path in live_dir.glob("episode_*_step_*.png"))
        if latest_screenshot.exists():
            output["latestScreenshotPath"] = str(latest_screenshot)
        if screenshots:
            output["screenshots"] = screenshots
    _emit_event(event="scan_completed", sites_evaluated=len(site_results))
    if args.output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(per_site_metrics_path, {"version": VERSION, "evaluated_at": output["evaluated_at"], "sites": per_site_metrics})
    write_comparison_note(comparison_note_path)
    _emit_stage("result_written", output=args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.min_recall is not None:
        failed = [
            site_id for site_id, result in site_results.items()
            if result.get("recall") is not None and float(result.get("recall") or 0.0) < args.min_recall
        ]
        if failed and args.strict_metrics:
            return 1
    return 0


def _partial_transition_path(output: str) -> Path | None:
    if not output:
        return None
    return Path(output).parent / "partial_transitions.jsonl"


def _live_dir(output: str) -> Path | None:
    if not output:
        return None
    return Path(output).parent / "live"


def _elapsed_ms() -> int:
    return int((time.monotonic() - _PROCESS_STARTED_AT) * 1000)


def _write_live_placeholder(
    output: str,
    *,
    scan_id: str,
    status: str,
    stage: str,
    target_url: str,
    message: str,
    current_site: str = "",
    current_episode: int | None = None,
    current_step: int | None = None,
) -> None:
    live_dir = _live_dir(output)
    if live_dir is None:
        return
    latest_path = live_dir / "latest.png"
    try:
        live_dir.mkdir(parents=True, exist_ok=True)
        width, height = 1280, 720
        if Image is None or ImageDraw is None:
            latest_path.write_bytes(base64.b64decode(_FALLBACK_PLACEHOLDER_PNG))
        else:
            image = Image.new("RGB", (width, height), "#eef2f7")
            draw = ImageDraw.Draw(image)
            title_font = _load_font(44)
            body_font = _load_font(24)
            small_font = _load_font(20)
            draw.rectangle((0, 0, width, height), fill="#eef2f7")
            draw.rectangle((0, 0, width, 86), fill="#d9e2ec")
            draw.text((56, 26), "J.A.W.S Live Browser Preview", fill="#1f2937", font=body_font)
            draw.text((56, 180), message, fill="#111827", font=title_font)
            y = 285
            for label, value in (
                ("targetUrl", target_url or "-"),
                ("scanId", scan_id or "-"),
                ("timestamp", datetime.now(timezone.utc).isoformat()),
            ):
                draw.text((56, y), f"{label}: ", fill="#334155", font=body_font)
                for line in textwrap.wrap(str(value), width=86)[:3]:
                    draw.text((190, y), line, fill="#475569", font=body_font)
                    y += 34
                y += 16
            draw.text((56, height - 70), "The first Playwright viewport screenshot will appear here.", fill="#64748b", font=small_font)
            image.save(latest_path, format="PNG", optimize=True)
        _write_live_status(
            live_dir,
            scan_id=scan_id,
            status=status,
            stage=stage,
            current_site=current_site,
            current_episode=current_episode,
            current_step=current_step,
            latest_screenshot_path=str(latest_path),
            last_screenshot_ok=True,
            last_screenshot_error=None,
            message=message,
        )
        _append_progress_event(
            output,
            event="live_placeholder_created",
            stage=stage,
            episode=current_episode or 0,
            step=current_step or 0,
            path=str(latest_path),
            ok=True,
        )
    except Exception as exc:
        _emit_event(event="warning", message=f"live placeholder write failed: {exc}")


def _load_font(size: int) -> Any:
    if ImageFont is None:
        return None
    for font_name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()


_FALLBACK_PLACEHOLDER_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP88OHDfwAJ6gP9"
    "nE5l2AAAAABJRU5ErkJggg=="
)


def _write_live_status(
    live_dir: Path | None,
    *,
    scan_id: str,
    status: str,
    stage: str = "",
    current_site: str = "",
    current_episode: int | None = None,
    current_step: int | None = None,
    latest_screenshot_path: str | None = None,
    last_screenshot_ok: bool | None = None,
    last_screenshot_error: str | None = None,
    message: str | None = None,
) -> None:
    if live_dir is None:
        return
    try:
        live_dir.mkdir(parents=True, exist_ok=True)
        latest_exists = bool(latest_screenshot_path and Path(latest_screenshot_path).exists())
        payload = {
            "scanId": scan_id,
            "available": latest_exists,
            "status": status,
            "stage": stage,
            "currentSite": current_site,
            "currentEpisode": current_episode,
            "currentStep": current_step,
            "latestScreenshotPath": latest_screenshot_path,
            "latestScreenshotUrl": f"/api/scans/{scan_id}/live-preview" if latest_exists else None,
            "latestScreenshotUpdatedAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "lastScreenshotOk": last_screenshot_ok,
            "lastScreenshotError": last_screenshot_error,
            "message": message or ("Live screenshot is available." if latest_exists else "Live screenshot is not available yet."),
        }
        (live_dir / "live_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        _emit_event(event="warning", message=f"live status write failed: {exc}")


def _append_progress_event(
    output: str,
    *,
    event: str,
    stage: str,
    episode: int,
    step: int,
    path: str | None = None,
    ok: bool = True,
    error: str | None = None,
) -> None:
    if not output:
        return
    progress_path = Path(output).parent / "progress.json"
    try:
        if progress_path.exists():
            root = json.loads(progress_path.read_text(encoding="utf-8") or "{}")
            if not isinstance(root, dict):
                root = {}
        else:
            root = {}
        events = root.get("screenshotEvents")
        if not isinstance(events, list):
            events = []
        item = {
            "event": event,
            "stage": stage,
            "episode": episode,
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsedMs": _elapsed_ms(),
            "ok": ok,
        }
        if path:
            item["path"] = path
        if error:
            item["error"] = error
        events.append(item)
        root["screenshotEvents"] = events[-200:]
        root["latestScreenshotEvent"] = item
        progress_path.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        _emit_event(event="warning", message=f"screenshot progress event write failed: {exc}")


def _resolve_live_page(env: BrowserGymJAWSEnv) -> Any:
    page = _active_page(env.env)
    if page is not None:
        return page
    candidates = [
        env,
        getattr(env, "env", None),
        getattr(getattr(env, "env", None), "unwrapped", None),
        getattr(getattr(env, "env", None), "task", None),
        getattr(getattr(getattr(env, "env", None), "unwrapped", None), "task", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        direct_page = getattr(candidate, "page", None)
        if direct_page is not None:
            return direct_page
        pages = getattr(candidate, "pages", None)
        if pages:
            return pages[0]
        for attr in ("context", "browser_context", "_context", "playwright_context"):
            context = getattr(candidate, attr, None)
            context_pages = getattr(context, "pages", None)
            if context_pages:
                return context_pages[0]
        task = getattr(candidate, "task", None)
        task_page = getattr(task, "page", None)
        if task_page is not None:
            return task_page
    return None


def _save_live_screenshot(
    env: BrowserGymJAWSEnv,
    output: str,
    *,
    scan_id: str,
    status: str,
    site_id: str,
    episode: int,
    step: int,
    stage: str,
) -> Dict[str, Any]:
    live_dir = _live_dir(output)
    if live_dir is None:
        return {"ok": False, "error": "output path is not configured"}
    latest_path = live_dir / "latest.png"
    step_path = live_dir / f"episode_{episode}_step_{step}.png"
    try:
        live_dir.mkdir(parents=True, exist_ok=True)
        page = _resolve_live_page(env)
        if page is None:
            raise RuntimeError("active page not found")
        page.screenshot(path=str(step_path), full_page=False)
        shutil.copyfile(step_path, latest_path)
        latest = str(latest_path)
        _write_live_status(
            live_dir,
            scan_id=scan_id,
            status=status,
            stage=stage,
            current_site=site_id,
            current_episode=episode,
            current_step=step,
            latest_screenshot_path=latest,
            last_screenshot_ok=True,
            last_screenshot_error=None,
        )
        _append_progress_event(
            output,
            event="screenshot_saved",
            stage=stage,
            episode=episode,
            step=step,
            path=str(step_path),
            ok=True,
        )
        return {
            "ok": True,
            "stepScreenshotPath": str(step_path),
            "latestScreenshotPath": latest,
        }
    except Exception as exc:
        _write_live_status(
            live_dir,
            scan_id=scan_id,
            status=status,
            stage=stage,
            current_site=site_id,
            current_episode=episode,
            current_step=step,
            latest_screenshot_path=str(latest_path) if latest_path.exists() else None,
            last_screenshot_ok=False,
            last_screenshot_error=str(exc),
        )
        _append_progress_event(
            output,
            event="screenshot_failed",
            stage=stage,
            episode=episode,
            step=step,
            ok=False,
            error=str(exc),
        )
        _emit_event(event="warning", site_id=site_id, episode=episode, step=step, message=f"live screenshot failed: {exc}")
        return {"ok": False, "error": str(exc)}


def _write_partial_output(
    args: argparse.Namespace,
    run_id: str,
    site_results: Mapping[str, Any],
    rewards: List[float],
    exc: Exception,
    site: Mapping[str, Any] | None = None,
) -> None:
    if not args.output:
        return
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_stage = str(_STAGE_STATE.get("currentStage") or "")
    merged_site_results = dict(site_results)
    if site is not None and not merged_site_results:
        site_id = str(site.get("site_id") or "")
        partial_site = _build_partial_site_result(site, exc, current_stage, _partial_transition_path(args.output))
        if site_id:
            merged_site_results[site_id] = partial_site
    output = {
        "run_id": run_id,
        "status": "partial_completed" if merged_site_results else "failed",
        "reportType": "Partial Scan Report" if merged_site_results else "Runtime Failure Report",
        "failureType": "SITE_TIMEOUT" if isinstance(exc, TimeoutError) else _failure_type_from_stage(current_stage),
        "failureReason": str(exc),
        "partial": True,
        "lastCompletedStage": str(_STAGE_STATE.get("lastCompletedStage") or ""),
        "currentStage": current_stage,
        "elapsedMs": int(_STAGE_STATE.get("elapsedMs") or 0),
        "sitesEvaluated": len(merged_site_results),
        "normalizedSites": [_normalized_partial_site(site_id, result) for site_id, result in merged_site_results.items()],
        "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "sites": merged_site_results,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _emit_stage("result_written", output=str(output_path), partial=True)


def _build_partial_site_result(site: Mapping[str, Any], exc: Exception, current_stage: str, transition_path: Path | None) -> Dict[str, Any]:
    site_id = str(site.get("site_id") or "")
    transitions = _read_partial_transitions(transition_path, site_id)
    steps_completed = max([int(item.get("step") or 0) for item in transitions] or [0])
    anomalies: List[Any] = []
    for item in transitions:
        anomalies.extend(item.get("detected_anomalies", []) or [])
    return {
        "status": "partial_failed",
        "failureType": "SITE_TIMEOUT" if isinstance(exc, TimeoutError) else _failure_type_from_stage(current_stage),
        "failureReason": str(exc),
        "base_url": str(site.get("base_url") or ""),
        "baseUrl": str(site.get("base_url") or ""),
        "stepsCompleted": steps_completed,
        "lastCompletedStage": str(_STAGE_STATE.get("lastCompletedStage") or ""),
        "currentStage": current_stage,
        "transitions": transitions,
        "anomalies": anomalies,
        "exploratory_anomalies": anomalies,
        "average_reward": float(np.mean([float(item.get("reward") or 0.0) for item in transitions])) if transitions else 0.0,
        "warnings": [f"Partial result generated after runtime exception: {exc}"],
    }


def _normalized_partial_site(site_id: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "siteId": site_id,
        "baseUrl": str(result.get("baseUrl") or result.get("base_url") or ""),
        "status": str(result.get("status") or "partial_failed"),
        "failureType": str(result.get("failureType") or ""),
        "stepsCompleted": int(result.get("stepsCompleted") or 0),
        "lastCompletedStage": str(result.get("lastCompletedStage") or ""),
        "currentStage": str(result.get("currentStage") or ""),
        "anomalyCandidates": list(result.get("anomalies") or []),
        "transitions": list(result.get("transitions") or []),
    }


def _read_partial_transitions(path: Path | None, site_id: str) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if not site_id or str(item.get("site_id") or item.get("siteId") or "") == site_id:
            rows.append(item)
    return rows


def _append_partial_transition(
    path: Path | None,
    *,
    site_id: str,
    base_url: str,
    episode: int,
    step: int,
    action: Mapping[str, Any],
    target: str,
    reward: float,
    anomalies: List[Mapping[str, Any]],
    url: str,
) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        elapsed_ms = int((time.monotonic() - _PROCESS_STARTED_AT) * 1000)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsedMs": elapsed_ms,
            "site_id": site_id,
            "siteId": site_id,
            "baseUrl": base_url,
            "episode": episode,
            "step": step,
            "action": action.get("action_type"),
            "target": target,
            "reward": reward,
            "detected_anomalies": anomalies,
            "url": url,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        _emit_event(event="warning", site_id=site_id, message=f"partial transition write failed: {exc}")


def _raise_if_site_timeout(site_started_at: float, site_timeout_seconds: int, site_id: str) -> None:
    if site_timeout_seconds <= 0:
        return
    elapsed = time.monotonic() - site_started_at
    if elapsed > site_timeout_seconds:
        raise TimeoutError(f"site timeout exceeded {site_timeout_seconds}s for {site_id}")


def _raise_if_episode_timeout(
    episode_started_at: float,
    episode_timeout_seconds: int,
    site_id: str,
    episode_index: int,
) -> None:
    if episode_timeout_seconds <= 0:
        return
    elapsed = time.monotonic() - episode_started_at
    if elapsed > episode_timeout_seconds:
        raise TimeoutError(f"episode timeout exceeded {episode_timeout_seconds}s for {site_id} episode={episode_index}")


def _make_eval_env(
    site: Mapping[str, Any],
    site_profile: Mapping[str, Any],
    *,
    max_steps: int,
    headless: bool,
    page_timeout_ms: int,
    navigation_timeout_ms: int,
) -> BrowserGymJAWSEnv:
    return BrowserGymJAWSEnv(
        site_id=str(site["site_id"]),
        base_url=str(site["base_url"]),
        max_steps=max_steps,
        headless=headless,
        site_profile=site_profile,
        requires_login=bool(site.get("requires_login")),
        login_config=site.get("login") if isinstance(site.get("login"), Mapping) else None,
        page_timeout_ms=page_timeout_ms,
        navigation_timeout_ms=navigation_timeout_ms,
    )


def _evaluate_site(
    site: Mapping[str, Any],
    agent: PPOAgent,
    encoder: ObservationEncoder,
    action_space: ActionSpace,
    episodes: int,
    max_steps: int,
    headless: bool,
    strict_site_validation: bool = False,
    csv_logger: EpisodeCsvLogger | None = None,
    site_timeout_seconds: int = 60,
    episode_timeout_seconds: int = 0,
    no_progress_patience: int = 0,
    page_timeout_ms: int = 15000,
    navigation_timeout_ms: int = 15000,
    env_reset_timeout_ms: int = 30000,
    reset_retry_count: int = 0,
    step_timeout_ms: int = 10000,
    output_path: str = "",
    partial_transition_path: Path | None = None,
    use_memory_encoder: bool = False,
    memory_encoder_type: str = "gru",
    memory_hidden_size: int = 128,
    fallback_mode: str = "eval",
    fallback_penalty: float = 0.25,
    fallback_reward_cap_enabled: bool = True,
    fallback_reward_cap_value: float = 0.0,
) -> Dict[str, Any]:
    site_id = str(site["site_id"])
    scan_id = _run_id_from_output(output_path) or ""
    live_status_dir = _live_dir(output_path)
    site_started_at = time.monotonic()
    _emit_event(event="site_started", site_id=site_id, base_url=str(site.get("base_url") or ""))
    _write_live_status(
        live_status_dir,
        scan_id=scan_id,
        status="running",
        stage="site_started",
        current_site=site_id,
        current_episode=0,
        current_step=0,
    )
    known_bugs = _load_known_bugs_for_site(site)
    site_profile = build_site_profile(site_id, [], exploration_profile=site.get("exploration_profile"))
    site_profile.update(
        {
            "reward_mode": str(site.get("reward_mode") or "signal_based"),
            "allow_known_bug_reward_for_debug_only": False,
            "use_ground_truth_for_evaluation_only": bool(
                site.get("use_ground_truth_for_evaluation_only", site.get("use_known_bug_for_evaluation", False))
            ),
            "use_known_bug_reward": False,
            "use_known_bug_for_training": False,
            "use_known_bug_for_evaluation": bool(site.get("use_known_bug_for_evaluation", False)),
            "training_uses_site_specific_bug_catalog": False,
            "target_signal_types": [],
            "policy_uses_bug_labels": False,
        }
    )
    has_catalog = bool(site.get("has_bug_catalog")) and bool(known_bugs)
    episode_rewards: List[float] = []
    scaled_episode_rewards: List[float] = []
    total_detected_candidates = 0
    unique_candidates: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    matched_by_bug_id: Dict[str, Dict[str, Any]] = {}
    suppressed_duplicates: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    action_counts: Counter[str] = Counter()
    fallback_action_count = 0
    warnings: List[str] = []
    workout_add_candidate_count = 0
    workout_add_click_count = 0
    functional_priority_candidate_count = 0
    functional_priority_click_count = 0
    high_value_functional_candidate_count = 0
    semantic_action_type_counts: Counter[str] = Counter()
    clicked_semantic_action_types: Counter[str] = Counter()
    semantic_no_effect_click_count = 0
    high_value_no_effect_click_count = 0
    semantic_button_no_response_count = 0
    functional_no_effect_anomaly_count = 0
    workout_add_no_effect_count = 0
    cart_no_effect_count = 0
    observed_functional_priority_targets: set[str] = set()
    clicked_functional_priority_targets: set[str] = set()
    empty_state_detected_count = 0
    layout_overflow_candidates = 0
    layout_overflow_candidates_raw = 0
    layout_overflow_unique: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    layout_overflow_match_reasons: Dict[str, str] = {}
    layout_overflow_top_evidence: List[Dict[str, Any]] = []
    catalog_candidate_count = 0
    catalog_selector_match_count = 0
    catalog_keyword_match_count = 0
    data_bug_ids_found: set[str] = set()
    button_no_response_candidates = 0
    openended_interactive_candidate_count = 0
    openended_keyword_match_count = 0
    first_click_step: int | None = None
    unique_clicked_targets: set[str] = set()
    observed_element_keys: set[str] = set()
    clicked_element_keys: set[str] = set()
    inspect_dom_repeat_count = 0
    layout_anomaly_count = 0
    interaction_anomaly_count = 0
    api_403_count = 0
    async_hang_candidates = 0
    cart_quantity_mismatch_candidates = 0
    sparse_data_candidates = 0
    attempted_bug_ids: set[str] = set()
    triggered_bug_ids: set[str] = set()
    reward_scale = _reward_scale(site)
    observed_url_sample = ""
    observed_title_sample = ""
    identity_summary: Dict[str, Any] = {
        "expected_bug_id_prefix": f"{site_id}-bug",
        "site_identity_match": True,
        "identity_warnings": [],
    }
    history: Dict[str, Any] = {
        "clicked_bids": set(),
        "last_action_key": None,
        "last_action_type": None,
        "action_type_counts": {},
        "purchase_click_counts": {},
        "workout_add_click_counts": {},
        "seen_anomaly_keys": set(),
        "mobile_viewport_seen": False,
        "inspected_cart_before_purchase": False,
        "matched_bug_ids": set(),
        "action_signature_counts": {},
    }
    completed_episodes = 0
    episode_errors: List[Dict[str, Any]] = []
    site_failed = False
    error_type = ""
    error_message = ""
    reset_elapsed_values: List[int] = []
    step_elapsed_values: List[int] = []
    episode_elapsed_values: List[int] = []
    early_stop_reasons: Counter[str] = Counter()
    timeout_count = 0
    episode_memory_reset_count = 0
    memory_state_norm_values: List[float] = []
    fallback_reason_counts: Counter[str] = Counter()
    diversity_reward_total = 0.0
    repeated_action_penalty_total = 0.0
    first_click_reward_count = 0
    new_action_type_reward_count = 0
    new_target_reward_count = 0
    inspect_reward_count = 0
    open_detail_panel_repeat_penalty_count = 0
    no_effect_action_penalty_count = 0
    signal_summary: Counter[str] = Counter()
    reset_retry_used = 0
    reset_failure_count = 0
    reset_failure_reason = ""
    no_functional_episode_count = 0
    filter_candidate_count = 0
    filter_click_count = 0
    category_candidate_count = 0
    category_click_count = 0
    tab_candidate_count = 0
    tab_click_count = 0
    enroll_candidate_count = 0
    enroll_click_count = 0
    search_input_click_count = 0
    search_input_fill_count = 0
    filtered_no_response_suppressed_count = 0
    no_response_suppressed_by_text_delta_count = 0
    no_response_suppressed_by_candidate_delta_count = 0
    no_response_suppressed_by_filter_semantics_count = 0
    search_click_no_effect_suppressed_count = 0
    no_progress_delayed_by_unclicked_functional_candidate_count = 0
    visited_target_suppression_count = 0
    visited_target_skipped_count = 0
    visited_element_key_suppression_count = 0
    unclicked_element_key_selected_count = 0
    element_key_collision_count = 0
    element_key_missing_count = 0
    unclicked_high_value_candidate_selected_count = 0
    unclicked_functional_candidate_selected_count = 0
    memory_guided_candidate_switch_count = 0
    repeated_action_redirect_count = 0
    repeated_target_redirect_count = 0
    finish_delayed_by_unclicked_candidate_count = 0
    add_no_effect_count = 0
    enroll_no_effect_count = 0
    high_value_enroll_no_response_count = 0
    high_value_button_no_response_count = 0
    low_value_generic_candidate_count = 0
    low_value_generic_click_count = 0
    low_value_generic_no_response_suppressed_count = 0
    policy_selected_low_value_generic_count = 0
    repeated_meta_action_suppression_count = 0
    inspect_console_repeat_suppression_count = 0
    inspect_network_repeat_suppression_count = 0
    meta_action_redirect_to_functional_count = 0
    verification_action_redirect_count = 0
    meta_action_repeated_priority_lowered_count = 0
    policy_selected_meta_action_suppressed_count = 0
    policy_selected_valid_action_count = 0
    policy_selected_invalid_action_count = 0
    policy_selected_visited_element_count = 0
    policy_selected_low_priority_action_count = 0
    fallback_applied_count = 0
    fallback_penalty_total = 0.0
    fallback_reward_capped_count = 0
    policy_executed_action_match_count = 0
    policy_executed_action_mismatch_count = 0

    for episode_index in range(1, episodes + 1):
        try:
            _raise_if_site_timeout(site_started_at, site_timeout_seconds, site_id)
        except Exception as exc:
            site_failed = True
            error_type = exc.__class__.__name__
            error_message = str(exc)
            episode_errors.append(
                {
                    "episode": episode_index,
                    "step": 0,
                    "error_type": error_type,
                    "error_message": error_message,
                }
            )
            _emit_event(
                event="site_failed",
                site_id=site_id,
                base_url=str(site.get("base_url") or ""),
                error_type=error_type,
                message=error_message,
            )
            break
        episode_started_at = time.monotonic()
        early_stop_reason = ""
        _emit_stage("episode_started", site_id=site_id, episode=episode_index)
        _emit_event(event="episode_started", site_id=site_id, episode=episode_index)
        _emit_stage("browser_launch_started", site_id=site_id, episode=episode_index)
        _write_live_placeholder(
            output_path,
            scan_id=scan_id,
            status="running",
            stage="browser_launch_started",
            target_url=str(site.get("base_url") or ""),
            message="Browser is launching...",
            current_site=site_id,
            current_episode=episode_index,
            current_step=0,
        )
        env = _make_eval_env(
            site,
            site_profile,
            max_steps=max_steps,
            headless=headless,
            page_timeout_ms=page_timeout_ms,
            navigation_timeout_ms=navigation_timeout_ms,
        )
        _emit_stage("browser_launch_completed", site_id=site_id, episode=episode_index)
        _write_live_placeholder(
            output_path,
            scan_id=scan_id,
            status="running",
            stage="browser_launch_completed",
            target_url=str(site.get("base_url") or ""),
            message="Browser is ready. Opening target...",
            current_site=site_id,
            current_episode=episode_index,
            current_step=0,
        )
        reward_total = 0.0
        scaled_reward_total = 0.0
        history: Dict[str, Any] = {
            "clicked_bids": set(),
            "last_action_key": None,
            "last_action_type": None,
            "action_type_counts": {},
            "purchase_click_counts": {},
            "workout_add_click_counts": {},
            "seen_anomaly_keys": set(),
            "mobile_viewport_seen": False,
            "inspected_cart_before_purchase": False,
            "matched_bug_ids": set(),
            "action_signature_counts": {},
            "memory_encoder_enabled": bool(use_memory_encoder),
        }
        try:
            step_index = -1
            _emit_stage("site_navigation_started", site_id=site_id, episode=episode_index)
            _write_live_placeholder(
                output_path,
                scan_id=scan_id,
                status="running",
                stage="site_navigation_started",
                target_url=str(site.get("base_url") or ""),
                message="Opening target page...",
                current_site=site_id,
                current_episode=episode_index,
                current_step=0,
            )
            observation = {}
            reset_info = {}
            reset_elapsed_ms = 0
            for reset_attempt in range(0, max(0, int(reset_retry_count or 0)) + 1):
                try:
                    _emit_stage("env_reset_started", site_id=site_id, episode=episode_index, resetAttempt=reset_attempt)
                    reset_started_at = time.monotonic()
                    observation, reset_info = env.reset()
                    reset_elapsed_ms = int((time.monotonic() - reset_started_at) * 1000)
                    reset_elapsed_values.append(reset_elapsed_ms)
                    if reset_elapsed_ms > env_reset_timeout_ms:
                        raise TimeoutError(f"env reset exceeded {env_reset_timeout_ms}ms for {site_id}")
                    reset_retry_used = max(reset_retry_used, reset_attempt)
                    break
                except Exception as reset_exc:
                    reset_failure_count += 1
                    reset_failure_reason = str(reset_exc)
                    if reset_attempt >= max(0, int(reset_retry_count or 0)):
                        raise
                    reset_retry_used = max(reset_retry_used, reset_attempt + 1)
                    _emit_event(
                        event="warning",
                        site_id=site_id,
                        episode=episode_index,
                        message=f"env reset failed; retrying {reset_attempt + 1}/{reset_retry_count}: {reset_exc}",
                    )
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = _make_eval_env(
                        site,
                        site_profile,
                        max_steps=max_steps,
                        headless=headless,
                        page_timeout_ms=page_timeout_ms,
                        navigation_timeout_ms=navigation_timeout_ms,
                    )
            _emit_stage("site_navigation_completed", site_id=site_id, episode=episode_index)
            _save_live_screenshot(
                env,
                output_path,
                scan_id=scan_id,
                status="running",
                site_id=site_id,
                episode=episode_index,
                step=0,
                stage="site_navigation_completed",
            )
            _emit_stage("env_reset_completed", site_id=site_id, episode=episode_index, resetElapsedMs=reset_elapsed_ms)
            _save_live_screenshot(
                env,
                output_path,
                scan_id=scan_id,
                status="running",
                site_id=site_id,
                episode=episode_index,
                step=0,
                stage="env_reset_completed",
            )
            if not observed_url_sample:
                observed_url_sample = str(observation.get("page_state", {}).get("url", "") or "")
                observed_title_sample = str(observation.get("page_state", {}).get("title", "") or "")
                identity_summary = validate_site_identity(site_id, observation)
                identity_summary["login_required"] = bool(reset_info.get("login_required"))
                identity_summary["login_attempted"] = bool(reset_info.get("login_attempted"))
                identity_summary["login_success"] = bool(reset_info.get("login_success"))
                identity_summary["post_login_url"] = str(reset_info.get("post_login_url") or "")
                if strict_site_validation and identity_summary.get("site_identity_match") is False:
                    raise RuntimeError(f"Site identity validation failed for {site_id}: {identity_summary}")
                for warning in identity_summary.get("identity_warnings", []) or []:
                    if warning not in warnings:
                        warnings.append(warning)
                        _emit_event(event="warning", site_id=site_id, episode=episode_index, message=warning)
                if has_catalog and bool(reset_info.get("login_success")) and not identity_summary.get("data_bug_ids_found"):
                    warning = "WARNING: bug_catalog loaded and login succeeded, but no data-bug-id was found. Check site9800 UI data-bug-id placement."
                    if warning not in warnings:
                        warnings.append(warning)
                        _emit_event(event="warning", site_id=site_id, episode=episode_index, message=warning)
            memory_state = agent.reset_memory_state()
            episode_memory_reset_count += 1
            for step_index in range(max_steps):
                _raise_if_site_timeout(site_started_at, site_timeout_seconds, site_id)
                _raise_if_episode_timeout(episode_started_at, episode_timeout_seconds, site_id, episode_index)
                obs_vector = encoder.encode_observation(observation)
                action_mask = action_space.build_action_mask(observation)
                policy_selected = agent.select_greedy_action(obs_vector, action_mask, memory_state=memory_state)
                policy_action_id = int(policy_selected["action_id"])
                policy_action = action_space.decode(policy_action_id)
                policy_action["action_id"] = policy_action_id
                policy_action["site_id"] = site_id
                _enrich_action(policy_action, observation)
                if 0 <= policy_action_id < len(action_mask) and float(action_mask[policy_action_id]) > 0.0:
                    policy_selected_valid_action_count += 1
                else:
                    policy_selected_invalid_action_count += 1
                if _policy_action_visited(policy_action, history):
                    policy_selected_visited_element_count += 1
                if _policy_action_low_priority(policy_action):
                    policy_selected_low_priority_action_count += 1
                fallback_before = _history_counter_snapshot(history)
                executed_mask = np.asarray(action_mask, dtype=np.float32).copy()
                fallback_warning = ""
                repeated = 0
                if str(fallback_mode or "eval") != "strict":
                    executed_mask, fallback_warning, repeated = _apply_eval_fallback_mask(
                        action_space,
                        executed_mask,
                        observation,
                        history,
                    )
                if fallback_warning and fallback_warning not in warnings:
                    warnings.append(fallback_warning)
                    _emit_event(event="warning", site_id=site_id, episode=episode_index, message=fallback_warning)
                if fallback_warning:
                    detailed_reasons = list(history.get("_fallback_reasons_step", []) or [])
                    if detailed_reasons:
                        for reason in detailed_reasons:
                            fallback_reason_counts[str(reason)] += 1
                    else:
                        fallback_reason_counts[_fallback_reason(fallback_warning)] += 1
                fallback_action_count += repeated
                deltas = _history_counter_delta(history, fallback_before)
                visited_target_suppression_count += deltas.get("visited_target_suppression_count", 0)
                visited_target_skipped_count += deltas.get("visited_target_skipped_count", 0)
                visited_element_key_suppression_count += deltas.get("visited_element_key_suppression_count", 0)
                memory_guided_candidate_switch_count += deltas.get("memory_guided_candidate_switch_count", 0)
                repeated_action_redirect_count += deltas.get("repeated_action_redirect_count", 0)
                repeated_target_redirect_count += deltas.get("repeated_target_redirect_count", 0)
                finish_delayed_by_unclicked_candidate_count += deltas.get("finish_delayed_by_unclicked_candidate_count", 0)
                repeated_meta_action_suppression_count += deltas.get("repeated_meta_action_suppression_count", 0)
                inspect_console_repeat_suppression_count += deltas.get("inspect_console_repeat_suppression_count", 0)
                inspect_network_repeat_suppression_count += deltas.get("inspect_network_repeat_suppression_count", 0)
                meta_action_redirect_to_functional_count += deltas.get("meta_action_redirect_to_functional_count", 0)
                verification_action_redirect_count += deltas.get("verification_action_redirect_count", 0)
                meta_action_repeated_priority_lowered_count += deltas.get("meta_action_repeated_priority_lowered_count", 0)
                policy_selected_meta_action_suppressed_count += deltas.get("policy_selected_meta_action_suppressed_count", 0)
                selected = (
                    policy_selected
                    if str(fallback_mode or "eval") == "strict"
                    else agent.select_greedy_action(obs_vector, executed_mask, memory_state=memory_state)
                )
                selected_memory_state = selected.get("memory_state", memory_state)
                memory_state_norm_values.append(float(selected.get("memory_state_norm", 0.0) or 0.0))
                action_id = int(selected["action_id"])
                action = action_space.decode(action_id)
                action["action_id"] = action_id
                action["site_id"] = site_id
                _enrich_action(action, observation)
                detailed_reasons = list(history.get("_fallback_reasons_step", []) or [])
                fallback_reason = ",".join(str(reason) for reason in detailed_reasons) if detailed_reasons else _fallback_reason(fallback_warning)
                fallback_applied = bool(str(fallback_mode or "eval") != "strict" and (repeated or fallback_warning or action_id != policy_action_id))
                if fallback_applied:
                    fallback_applied_count += 1
                if action_id == policy_action_id:
                    policy_executed_action_match_count += 1
                else:
                    policy_executed_action_mismatch_count += 1
                _attach_policy_execution_fields(
                    action,
                    policy_action_id=policy_action_id,
                    policy_action=policy_action,
                    executed_action_id=action_id,
                    fallback_applied=fallback_applied,
                    fallback_reason=fallback_reason if fallback_applied else "",
                )
                if csv_logger is not None:
                    csv_logger.log_observation(site_id, f"{site_id}-EP{episode_index:03d}", step_index + 1, step_index + 1, observation)
                    csv_logger.log_action_space(
                        site_id,
                        f"{site_id}-EP{episode_index:03d}",
                        step_index + 1,
                        step_index + 1,
                        observation,
                        action_space,
                        executed_mask,
                        action_id,
                    )
                _emit_event(
                    event="action",
                    site_id=site_id,
                    episode=episode_index,
                    step=step_index + 1,
                    action=action.get("action_type"),
                    target=_action_target_text(action),
                    target_text=_action_target_text(action),
                    action_signature=action.get("action_signature"),
                    action_element_key=action.get("action_element_key"),
                    action_bid=action.get("action_bid"),
                    action_text=action.get("action_text"),
                    action_semantic_type=action.get("action_semantic_type"),
                    dom_state_hash=action.get("dom_state_hash"),
                )
                if action.get("action_type") == "click_element":
                    if first_click_step is None:
                        first_click_step = step_index + 1
                    if action.get("clicked_bid") or action.get("clicked_text"):
                        unique_clicked_targets.add(str(action.get("clicked_bid") or action.get("clicked_text")))
                    if action.get("action_element_key"):
                        clicked_element_keys.add(str(action.get("action_element_key")))
                if action.get("action_type") == "inspect_dom" and history.get("last_action_type") == "inspect_dom":
                    inspect_dom_repeat_count += 1
                _emit_stage("step_started", site_id=site_id, episode=episode_index, step=step_index + 1)
                _save_live_screenshot(
                    env,
                    output_path,
                    scan_id=scan_id,
                    status="running",
                    site_id=site_id,
                    episode=episode_index,
                    step=step_index + 1,
                    stage="step_started",
                )
                step_started_at = time.monotonic()
                before_signature = _state_signature(observation)
                before_unique_count = len(unique_candidates)
                before_matched_count = len(matched_by_bug_id)
                next_observation, _, done, step_info = env.step(action_id)
                page_text_delta = abs(_page_text_length(next_observation) - _page_text_length(observation))
                next_candidates_for_delta = next_observation.get("candidate_elements", []) or []
                before_candidates_for_delta = observation.get("candidate_elements", []) or []
                candidate_delta = abs(
                    (len(next_candidates_for_delta) if isinstance(next_candidates_for_delta, list) else 0)
                    - (len(before_candidates_for_delta) if isinstance(before_candidates_for_delta, list) else 0)
                )
                step_elapsed_ms = int((time.monotonic() - step_started_at) * 1000)
                step_elapsed_values.append(step_elapsed_ms)
                if step_elapsed_ms > step_timeout_ms:
                    raise TimeoutError(f"step exceeded {step_timeout_ms}ms for {site_id} episode={episode_index} step={step_index + 1}")
                _emit_stage("step_completed", site_id=site_id, episode=episode_index, step=step_index + 1, stepElapsedMs=step_elapsed_ms)
                _save_live_screenshot(
                    env,
                    output_path,
                    scan_id=scan_id,
                    status="running",
                    site_id=site_id,
                    episode=episode_index,
                    step=step_index + 1,
                    stage="step_completed",
                )
                action["failed"] = bool(step_info.get("last_action_error"))
                anomalies = detect_anomalies(
                    observation,
                    next_observation,
                    {"action": action, "site_profile": site_profile, **step_info},
                    site_profile=site_profile,
                )
                infra_anomalies = detect_infra_anomalies(
                    next_observation,
                    {"action": action, "site_profile": site_profile, **step_info},
                )
                anomalies.extend(infra_anomalies)
                reward_matches: List[Dict[str, Any]] = []
                action["matched_bug_ids"] = []
                reward, reward_breakdown = calculate_autonomous_reward(
                    observation,
                    next_observation,
                    action,
                    anomalies,
                    reward_matches,
                    history,
                    site_profile,
                )
                infra_reward, infra_breakdown = calculate_infra_reward(
                    observation,
                    next_observation,
                    action,
                    infra_anomalies,
                    history,
                )
                reward += infra_reward
                reward_breakdown.update(infra_breakdown)
                reward, reward_breakdown = apply_fallback_reward_policy(
                    reward,
                    reward_breakdown,
                    fallback_applied=fallback_applied,
                    fallback_penalty=fallback_penalty,
                    fallback_reward_cap_enabled=fallback_reward_cap_enabled,
                    fallback_reward_cap_value=fallback_reward_cap_value,
                )
                fallback_penalty_total += float(reward_breakdown.get("fallback_penalty_total", 0.0) or 0.0)
                fallback_reward_capped_count += int(float(reward_breakdown.get("fallback_reward_capped_count", 0.0) or 0.0))
                matches = match_anomalies_to_known_bugs(anomalies, known_bugs, site_id=site_id)
                action["matched_bug_ids"] = [
                    str(match.get("matched_bug_id")) for match in matches if match.get("matched_bug_id")
                ]
                memory_state = selected_memory_state
                for key in (
                    "known_bug_reward_total",
                    "signal_reward_total",
                    "exploration_reward_total",
                    "ui_dom_signal_reward_total",
                    "console_runtime_signal_reward_total",
                    "network_api_signal_reward_total",
                    "cross_layer_signal_reward_total",
                    "security_signal_reward_total",
                    "repeated_penalty_total",
                    "signal_delta_reward_total",
                    "inspect_action_reward_total",
                    "functional_action_signal_reward_total",
                    "reward_functional_action_total",
                    "penalty_debug_meta_total",
                    "multi_signal_anomaly_count",
                    "console_error_count",
                    "runtime_exception_count",
                    "network_request_failed_count",
                    "api_4xx_count",
                    "api_5xx_count",
                    "api_timeout_count",
                    "api_ui_mismatch_count",
                    "security_signal_count",
                    "delta_console_error_count",
                    "delta_runtime_exception_count",
                    "delta_network_request_failed_count",
                    "delta_api_4xx_count",
                    "delta_api_5xx_count",
                    "delta_api_timeout_count",
                    "playwright_console_listener_enabled",
                    "playwright_pageerror_listener_enabled",
                    "playwright_network_listener_enabled",
                    "playwright_request_listener_enabled",
                    "playwright_response_listener_enabled",
                    "playwright_requestfailed_listener_enabled",
                    "functional_action_count",
                    "debug_meta_action_count",
                    "debug_meta_repeat_count",
                    "no_functional_action_episode_count",
                    "functional_action_signal_delta_count",
                    "functional_action_network_delta_count",
                ):
                    signal_summary[key] += float(reward_breakdown.get(key, 0.0) or 0.0)
                if reward_breakdown.get("playwright_listener_warning"):
                    signal_summary["playwright_listener_warning_count"] += 1
                diversity_reward_total += float(reward_breakdown.get("diversity_reward", 0.0) or 0.0)
                repeated_action_penalty_total += float(reward_breakdown.get("repeated_action_penalty", 0.0) or 0.0)
                first_click_reward_count += int(float(reward_breakdown.get("first_click_reward_count", 0.0) or 0.0))
                new_action_type_reward_count += int(float(reward_breakdown.get("new_action_type_reward_count", 0.0) or 0.0))
                new_target_reward_count += int(float(reward_breakdown.get("new_target_reward_count", 0.0) or 0.0))
                inspect_reward_count += int(float(reward_breakdown.get("inspect_reward_count", 0.0) or 0.0))
                open_detail_panel_repeat_penalty_count += int(float(reward_breakdown.get("open_detail_panel_repeat_penalty_count", 0.0) or 0.0))
                no_effect_action_penalty_count += int(float(reward_breakdown.get("no_effect_action_penalty_count", 0.0) or 0.0))
                reward_total += reward
                scaled_reward_total += _clamp_reward(reward * reward_scale)
                reward_breakdown["raw_final_reward"] = reward
                reward_breakdown["scaled_final_reward"] = _clamp_reward(reward * reward_scale)
                reward_breakdown["reward_scale"] = reward_scale
                _append_partial_transition(
                    partial_transition_path,
                    site_id=site_id,
                    base_url=str(site.get("base_url") or ""),
                    episode=episode_index,
                    step=step_index + 1,
                    action=action,
                    target=_action_target_text(action),
                    reward=_clamp_reward(reward * reward_scale),
                    anomalies=anomalies,
                    url=str(next_observation.get("page_state", {}).get("url", "") or ""),
                )
                if csv_logger is not None:
                    episode_id = f"{site_id}-EP{episode_index:03d}"
                    csv_logger.log_step(
                        site_id=site_id,
                        base_url=str(site.get("base_url") or ""),
                        episode_id=episode_id,
                        step_id=step_index + 1,
                        tick_id=step_index + 1,
                        before_observation=observation,
                        after_observation=next_observation,
                        action=action,
                        action_mask=action_mask,
                        step_info=step_info,
                        anomalies=anomalies,
                        known_matches=matches,
                        reward=_clamp_reward(reward * reward_scale),
                        reward_breakdown=reward_breakdown,
                        done=done,
                    )
                    csv_logger.log_observation(site_id, episode_id, step_index + 1, f"{step_index + 1}.after", next_observation)
                action_counts[str(action.get("action_type") or "")] += 1
                candidates = observation.get("candidate_elements", []) or []
                if isinstance(candidates, list):
                    catalog_candidate_count += sum(
                        1 for candidate in candidates if isinstance(candidate, Mapping) and candidate.get("catalog_bug_id_matches")
                    )
                    catalog_selector_match_count += sum(
                        1 for candidate in candidates if isinstance(candidate, Mapping) and candidate.get("catalog_selector_match")
                    )
                    catalog_keyword_match_count += sum(
                        len(candidate.get("catalog_keyword_matches", []) or [])
                        for candidate in candidates
                        if isinstance(candidate, Mapping)
                    )
                    data_bug_ids_found.update(
                        str(candidate.get("data_bug_id"))
                        for candidate in candidates
                        if isinstance(candidate, Mapping) and candidate.get("data_bug_id")
                    )
                    workout_add_candidate_count += sum(
                        1 for candidate in candidates if isinstance(candidate, Mapping) and candidate.get("is_workout_add_action")
                    )
                    functional_priority_candidate_count += sum(
                        1
                        for candidate in candidates
                        if isinstance(candidate, Mapping)
                        and (candidate.get("functional_priority_candidate") or candidate.get("functional_priority"))
                    )
                    high_value_functional_candidate_count += sum(
                        1
                        for candidate in candidates
                        if isinstance(candidate, Mapping) and candidate.get("is_high_value_functional_candidate")
                    )
                    current_observation_element_keys: set[str] = set()
                    for candidate in candidates:
                        if not isinstance(candidate, Mapping):
                            continue
                        semantic_type = str(candidate.get("semantic_action_type") or "")
                        if semantic_type:
                            semantic_action_type_counts[semantic_type] += 1
                        if candidate.get("is_low_value_generic_candidate"):
                            low_value_generic_candidate_count += 1
                        element_key = str(candidate.get("element_key") or "")
                        if element_key:
                            if element_key in current_observation_element_keys:
                                element_key_collision_count += 1
                            current_observation_element_keys.add(element_key)
                            observed_element_keys.add(element_key)
                        else:
                            element_key_missing_count += 1
                        if semantic_type == "filter":
                            filter_candidate_count += 1
                        elif semantic_type == "category":
                            category_candidate_count += 1
                        elif semantic_type == "tab":
                            tab_candidate_count += 1
                        elif semantic_type == "enroll":
                            enroll_candidate_count += 1
                        if candidate.get("functional_priority_candidate") or candidate.get("functional_priority"):
                            observed_functional_priority_targets.add(_candidate_key(candidate))
                    layout_overflow_candidates += sum(
                        1
                        for candidate in candidates
                        if isinstance(candidate, Mapping)
                        and (candidate.get("is_chart_related") or candidate.get("is_weekly_stats_related"))
                    )
                    openended_interactive_candidate_count += sum(
                        1
                        for candidate in candidates
                        if isinstance(candidate, Mapping)
                        and bool(candidate.get("is_interactive") or candidate.get("clickable"))
                    )
                    openended_keyword_match_count += sum(
                        len(candidate.get("openended_keyword_matches", []) or [])
                        for candidate in candidates
                        if isinstance(candidate, Mapping)
                    )
                if action.get("is_workout_add_action"):
                    workout_add_click_count += 1
                if action.get("functional_priority_candidate"):
                    selected_target_key = str(action.get("clicked_bid") or action.get("clicked_text") or "")
                    selected_element_key = str(action.get("action_element_key") or selected_target_key)
                    click_counts_before = history.get("click_target_counts", {})
                    element_counts_before = history.get("element_key_click_counts", {})
                    was_unclicked = (
                        not isinstance(element_counts_before, Mapping)
                        or int(element_counts_before.get(selected_element_key, 0) or 0) <= 0
                    )
                    if not selected_element_key:
                        was_unclicked = not isinstance(click_counts_before, Mapping) or int(click_counts_before.get(selected_target_key, 0) or 0) <= 0
                    functional_priority_click_count += 1
                    if was_unclicked:
                        unclicked_element_key_selected_count += 1
                        unclicked_functional_candidate_selected_count += 1
                        if action.get("is_high_value_functional_candidate"):
                            unclicked_high_value_candidate_selected_count += 1
                    if action.get("clicked_bid") or action.get("clicked_text"):
                        clicked_functional_priority_targets.add(str(action.get("clicked_bid") or action.get("clicked_text")))
                if action.get("semantic_action_type"):
                    clicked_semantic_action_types[str(action.get("semantic_action_type"))] += 1
                action_semantic_type = str(action.get("semantic_action_type") or "")
                if action.get("is_low_value_generic_candidate"):
                    low_value_generic_click_count += 1
                    policy_selected_low_value_generic_count += 1
                if action.get("action_type") == "click_element":
                    if action_semantic_type == "filter":
                        filter_click_count += 1
                    elif action_semantic_type == "category":
                        category_click_count += 1
                    elif action_semantic_type == "tab":
                        tab_click_count += 1
                    elif action_semantic_type == "enroll":
                        enroll_click_count += 1
                    elif action_semantic_type == "search_input":
                        search_input_click_count += 1
                elif action.get("action_type") == "fill_input" and action_semantic_type == "search_input":
                    search_input_fill_count += 1
                has_button_no_response = any(anomaly.get("type") == "button-no-response" for anomaly in anomalies)
                state_changed_after_action = before_signature != _state_signature(next_observation)
                if action.get("action_type") == "click_element" and action_semantic_type in {"filter", "category", "tab"}:
                    if not has_button_no_response:
                        filtered_no_response_suppressed_count += 1
                        no_response_suppressed_by_filter_semantics_count += 1
                        if page_text_delta > 0:
                            no_response_suppressed_by_text_delta_count += 1
                        if candidate_delta > 0:
                            no_response_suppressed_by_candidate_delta_count += 1
                if (
                    action.get("action_type") == "click_element"
                    and action_semantic_type == "search_input"
                    and not has_button_no_response
                ):
                    search_click_no_effect_suppressed_count += 1
                if (
                    action.get("action_type") == "click_element"
                    and action.get("is_low_value_generic_candidate")
                    and not has_button_no_response
                ):
                    low_value_generic_no_response_suppressed_count += 1
                empty_state_detected_count += sum(1 for anomaly in anomalies if anomaly.get("type") == "empty-state-rendering")
                button_no_response_candidates += sum(1 for anomaly in anomalies if anomaly.get("type") == "button-no-response")
                for anomaly in anomalies:
                    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
                    if evidence.get("semantic_no_effect_click"):
                        semantic_no_effect_click_count += 1
                    if evidence.get("high_value_functional_candidate"):
                        high_value_no_effect_click_count += 1
                    if anomaly.get("type") == "button-no-response" and evidence.get("semantic_action_type"):
                        semantic_button_no_response_count += 1
                    if anomaly.get("type") == "button-no-response" and evidence.get("high_value_functional_candidate"):
                        high_value_button_no_response_count += 1
                    if evidence.get("functional_no_effect_anomaly"):
                        functional_no_effect_anomaly_count += 1
                    if evidence.get("add_no_effect"):
                        add_no_effect_count += 1
                    if evidence.get("enroll_no_effect"):
                        enroll_no_effect_count += 1
                    if evidence.get("high_value_enroll_no_response"):
                        high_value_enroll_no_response_count += 1
                    if evidence.get("workout_add_no_effect"):
                        workout_add_no_effect_count += 1
                    if evidence.get("cart_no_effect"):
                        cart_no_effect_count += 1
                layout_anomaly_count += sum(
                    1 for anomaly in anomalies if anomaly.get("type") in {"layout-overlap", "layout-overflow"}
                )
                interaction_anomaly_count += sum(
                    1
                    for anomaly in anomalies
                    if anomaly.get("type")
                    in {
                        "button-no-response",
                        "form-no-feedback",
                        "async-hang",
                        "timeout-no-feedback",
                        "api-forbidden",
                        "cart-quantity-mismatch",
                        "sparse-data-rendering",
                        "broken-navigation",
                        "network-error",
                        "console-error",
                        "api-ui-mismatch",
                    }
                )
                api_403_count += sum(
                    int(anomaly.get("evidence", {}).get("api_403_count", 0) or 0)
                    for anomaly in anomalies
                    if isinstance(anomaly.get("evidence"), Mapping)
                )
                async_hang_candidates += sum(1 for anomaly in anomalies if anomaly.get("type") in {"async-hang", "timeout-no-feedback"})
                cart_quantity_mismatch_candidates += sum(1 for anomaly in anomalies if anomaly.get("type") == "cart-quantity-mismatch")
                sparse_data_candidates += sum(1 for anomaly in anomalies if anomaly.get("type") == "sparse-data-rendering")
                for anomaly in anomalies:
                    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
                    matched_bug_id = str(anomaly.get("matched_bug_id") or "")
                    _emit_event(
                        event="anomaly_detected",
                        site_id=site_id,
                        episode=episode_index,
                        step=step_index + 1,
                        type=str(anomaly.get("type") or ""),
                        matched_bug_id=matched_bug_id,
                        confidence=float(anomaly.get("confidence", 0.0) or 0.0),
                    )
                    _save_live_screenshot(
                        env,
                        output_path,
                        scan_id=scan_id,
                        status="running",
                        site_id=site_id,
                        episode=episode_index,
                        step=step_index + 1,
                        stage="anomaly_detected",
                    )
                    for bug_id in evidence.get("catalog_bug_id_matches", []) or []:
                        triggered_bug_ids.add(str(bug_id))
                layout_anomalies = [anomaly for anomaly in anomalies if anomaly.get("type") == "layout-overflow"]
                layout_overflow_candidates_raw += len(layout_anomalies)
                for anomaly in layout_anomalies:
                    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
                    layout_key = _layout_unique_key(anomaly)
                    previous = layout_overflow_unique.get(layout_key)
                    if previous is None or float(anomaly.get("confidence", 0.0) or 0.0) > float(previous.get("confidence", 0.0) or 0.0):
                        layout_overflow_unique[layout_key] = anomaly
                    if anomaly.get("matched_bug_id"):
                        layout_overflow_match_reasons[str(anomaly.get("matched_bug_id"))] = str(anomaly.get("match_reason") or "")
                    if _layout_evidence_key(evidence) not in {_layout_evidence_key(item) for item in layout_overflow_top_evidence} and len(layout_overflow_top_evidence) < 3:
                        layout_overflow_top_evidence.append(dict(evidence))
                total_detected_candidates += len(anomalies)
                _record_unique_candidates(unique_candidates, matched_by_bug_id, suppressed_duplicates, anomalies)
                progress = _episode_progress_made(
                    before_unique_count=before_unique_count,
                    after_unique_count=len(unique_candidates),
                    before_matched_count=before_matched_count,
                    after_matched_count=len(matched_by_bug_id),
                    before_signature=before_signature,
                    after_signature=_state_signature(next_observation),
                )
                if progress:
                    history["no_progress_steps"] = 0
                else:
                    history["no_progress_steps"] = int(history.get("no_progress_steps", 0) or 0) + 1
                _update_history(history, observation, action, anomalies)
                observation = next_observation
                repeated_signature_count = _current_action_signature_count(history, action)
                if no_progress_patience > 0 and int(history.get("no_progress_steps", 0) or 0) >= no_progress_patience:
                    unclicked_functional_count = len(observed_functional_priority_targets - clicked_functional_priority_targets)
                    if unclicked_functional_count > 0:
                        no_progress_delayed_by_unclicked_functional_candidate_count += 1
                        history["no_progress_steps"] = max(0, no_progress_patience - 1)
                        done = False
                    else:
                        early_stop_reason = "no_progress"
                        early_stop_reasons[early_stop_reason] += 1
                        done = True
                elif no_progress_patience > 0 and repeated_signature_count >= no_progress_patience + 1 and not progress:
                    early_stop_reason = "repeated_action"
                    early_stop_reasons[early_stop_reason] += 1
                    done = True
                if done and str(action.get("action_type") or "") == "finish_episode":
                    unclicked_functional_count = len(observed_functional_priority_targets - clicked_functional_priority_targets)
                    if unclicked_functional_count > 0:
                        finish_delayed_by_unclicked_candidate_count += 1
                        done = False
                if done:
                    break
            if not early_stop_reason and not done:
                early_stop_reason = "max_steps"
                early_stop_reasons[early_stop_reason] += 1
            if int(history.get("functional_action_count", 0) or 0) <= 0:
                no_functional_episode_count += 1
            _save_live_screenshot(
                env,
                output_path,
                scan_id=scan_id,
                status="running",
                site_id=site_id,
                episode=episode_index,
                step=step_index + 1 if "step_index" in locals() else 0,
                stage="scan_completed",
            )
            completed_episodes += 1
            episode_elapsed_values.append(int((time.monotonic() - episode_started_at) * 1000))
        except Exception as exc:
            site_failed = True
            error_type = exc.__class__.__name__
            error_message = str(exc)
            if isinstance(exc, TimeoutError):
                timeout_count += 1
                early_stop_reasons["timeout"] += 1
            episode_errors.append(
                {
                    "episode": episode_index,
                    "step": step_index + 1 if "step_index" in locals() else 0,
                    "error_type": error_type,
                    "error_message": error_message,
                }
            )
            _emit_event(event="warning", site_id=site_id, episode=episode_index, message=f"episode exception: {exc}")
            _write_live_status(
                live_status_dir,
                scan_id=scan_id,
                status="failed",
                stage="episode_failed",
                current_site=site_id,
                current_episode=episode_index,
                current_step=step_index + 1 if "step_index" in locals() else 0,
                last_screenshot_ok=False,
                last_screenshot_error=str(exc),
            )
            print(f"[multisite-eval] {site_id} episode {episode_index} exception: {exc}", file=sys.stderr, flush=True)
            break
        finally:
            env.close()
        episode_rewards.append(reward_total)
        scaled_episode_rewards.append(scaled_reward_total)

    matched_bug_ids = sorted(matched_by_bug_id)
    known_bug_ids = sorted(_known_bug_id(bug) for bug in known_bugs if _known_bug_id(bug))
    missed_bug_ids = [bug_id for bug_id in known_bug_ids if bug_id not in set(matched_bug_ids)]
    unique_detected = len(unique_candidates)
    confirmed_bugs, catalog_related_anomalies, exploratory_anomalies = _classify_candidates(unique_candidates.values())
    deduped_known_bug_candidates = len(confirmed_bugs) + len(catalog_related_anomalies)
    suppressed_duplicate_examples = list(suppressed_duplicates.values())[:3]
    action_diversity_score = _action_diversity_score(action_counts)
    anomaly_types = dict(Counter(str(item.get("type") or "") for item in unique_candidates.values()))
    early_stop_count = sum(int(value or 0) for value in early_stop_reasons.values())
    blocking_early_stop_count = sum(
        int(value or 0)
        for reason, value in early_stop_reasons.items()
        if str(reason) != "max_steps"
    )
    site_status = "failed" if site_failed else "partial_ok" if blocking_early_stop_count else "ok"
    site_elapsed_ms = int((time.monotonic() - site_started_at) * 1000)
    partial = site_failed or completed_episodes < episodes or bool(blocking_early_stop_count)
    exclusion_reason = _comparison_exclusion_reason(
        status=site_status,
        partial=partial,
        completed_episodes=completed_episodes,
        requested_episodes=episodes,
        timeout_count=timeout_count,
        observed_url_sample=observed_url_sample,
        action_counts=action_counts,
        openended_interactive_candidate_count=openended_interactive_candidate_count,
        error_message=error_message,
    )
    valid_for_comparison = not bool(exclusion_reason)
    result = {
        "status": site_status,
        "error_type": error_type,
        "error_message": error_message,
        "requested_episodes": episodes,
        "completed_episodes": completed_episodes,
        "valid_completed_episodes": completed_episodes if valid_for_comparison else 0,
        "valid_for_comparison": valid_for_comparison,
        "excluded_from_comparison_reason": exclusion_reason,
        "partial": partial,
        "episode_errors": episode_errors,
        "avg_step_elapsed_ms": _avg_int(step_elapsed_values),
        "max_step_elapsed_ms": max(step_elapsed_values) if step_elapsed_values else 0,
        "reset_elapsed_ms": _avg_int(reset_elapsed_values),
        "reset_timeout_ms": env_reset_timeout_ms,
        "reset_retry_count": reset_retry_count,
        "reset_retry_used": reset_retry_used,
        "reset_failure_count": reset_failure_count,
        "reset_failure_reason": reset_failure_reason,
        "site_elapsed_ms": site_elapsed_ms,
        "episode_elapsed_ms": _avg_int(episode_elapsed_values),
        "early_stop_count": early_stop_count,
        "early_stop_reasons": dict(early_stop_reasons),
        "timeout_count": timeout_count,
        "memory_encoder_enabled": bool(use_memory_encoder),
        "use_memory_encoder": bool(use_memory_encoder),
        "memory_encoder_type": memory_encoder_type if use_memory_encoder else "",
        "memory_hidden_size": memory_hidden_size if use_memory_encoder else 0,
        "episode_memory_reset_count": episode_memory_reset_count,
        "memory_state_reset_count": episode_memory_reset_count,
        "memory_state_update_count": len(memory_state_norm_values) if use_memory_encoder else 0,
        "memory_state_norm_mean": float(np.mean(memory_state_norm_values)) if use_memory_encoder and memory_state_norm_values else 0.0,
        "repeated_action_rate": _repeated_action_rate_from_action_counts(action_counts),
        "repeated_click_target_rate": _safe_rate(_repeated_click_target_count(history), action_counts.get("click_element", 0)),
        "repeated_action_rate_before_memory": None,
        "repeated_action_rate_after_memory": _repeated_action_rate_from_action_counts(action_counts) if use_memory_encoder else None,
        "known_bug_catalog_available": has_catalog,
        "requested_base_url": str(site.get("base_url") or ""),
        "observed_url_sample": observed_url_sample,
        "observed_title_sample": observed_title_sample,
        "expected_bug_id_prefix": identity_summary.get("expected_bug_id_prefix", f"{site_id}-bug"),
        "site_identity_match": bool(identity_summary.get("site_identity_match", True)),
        "identity_warnings": list(identity_summary.get("identity_warnings", []) or []),
        "login_required": bool(identity_summary.get("login_required", False)),
        "login_attempted": bool(identity_summary.get("login_attempted", False)),
        "login_success": bool(identity_summary.get("login_success", False)),
        "post_login_url": str(identity_summary.get("post_login_url") or ""),
        "evaluation_mode": "known_bug_catalog" if bool(site_profile.get("use_known_bug_for_evaluation", False)) and has_catalog else "signal_discovery",
        "episodes": episodes,
        "episode_count": completed_episodes,
        "average_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "raw_average_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "scaled_average_reward": float(np.mean(scaled_episode_rewards)) if scaled_episode_rewards else 0.0,
        "reward_scale": reward_scale,
        "detected_bug_count": unique_detected,
        "raw_detected_candidates": total_detected_candidates,
        "deduped_detected_candidates": unique_detected,
        "total_detected_candidates": total_detected_candidates,
        "unique_detected_candidates": unique_detected,
        "confirmed_bugs": confirmed_bugs,
        "catalog_related_anomalies": catalog_related_anomalies,
        "exploratory_anomalies": exploratory_anomalies,
        "suppressed_duplicate_count": len(suppressed_duplicates),
        "suppressed_duplicate_examples": suppressed_duplicate_examples,
        "action_counts": dict(action_counts),
        "click_element_count": int(action_counts.get("click_element", 0)),
        "inspect_cart_count": int(action_counts.get("inspect_cart", 0)),
        "workout_add_candidate_count": workout_add_candidate_count,
        "workout_add_click_count": workout_add_click_count,
        "functional_priority_candidate_count": functional_priority_candidate_count,
        "functional_priority_click_count": functional_priority_click_count,
        "filter_candidate_count": filter_candidate_count,
        "filter_click_count": filter_click_count,
        "category_candidate_count": category_candidate_count,
        "category_click_count": category_click_count,
        "tab_candidate_count": tab_candidate_count,
        "tab_click_count": tab_click_count,
        "enroll_candidate_count": enroll_candidate_count,
        "enroll_click_count": enroll_click_count,
        "search_input_click_count": search_input_click_count,
        "search_input_fill_count": search_input_fill_count,
        "low_value_generic_candidate_count": low_value_generic_candidate_count,
        "low_value_generic_click_count": low_value_generic_click_count,
        "low_value_generic_no_response_suppressed_count": low_value_generic_no_response_suppressed_count,
        "policy_selected_low_value_generic_count": policy_selected_low_value_generic_count,
        "filtered_no_response_suppressed_count": filtered_no_response_suppressed_count,
        "no_response_suppressed_by_text_delta_count": no_response_suppressed_by_text_delta_count,
        "no_response_suppressed_by_candidate_delta_count": no_response_suppressed_by_candidate_delta_count,
        "no_response_suppressed_by_filter_semantics_count": no_response_suppressed_by_filter_semantics_count,
        "search_click_no_effect_suppressed_count": search_click_no_effect_suppressed_count,
        "no_progress_delayed_by_unclicked_functional_candidate_count": no_progress_delayed_by_unclicked_functional_candidate_count,
        "visited_target_suppression_count": visited_target_suppression_count,
        "visited_target_skipped_count": visited_target_skipped_count,
        "visited_element_key_suppression_count": visited_element_key_suppression_count,
        "unclicked_element_key_selected_count": unclicked_element_key_selected_count,
        "element_key_collision_count": element_key_collision_count,
        "element_key_missing_count": element_key_missing_count,
        "unclicked_high_value_candidate_selected_count": unclicked_high_value_candidate_selected_count,
        "unclicked_functional_candidate_selected_count": unclicked_functional_candidate_selected_count,
        "memory_guided_candidate_switch_count": memory_guided_candidate_switch_count,
        "repeated_action_redirect_count": repeated_action_redirect_count,
        "repeated_target_redirect_count": repeated_target_redirect_count,
        "finish_delayed_by_unclicked_candidate_count": finish_delayed_by_unclicked_candidate_count,
        "repeated_meta_action_suppression_count": repeated_meta_action_suppression_count,
        "inspect_console_repeat_suppression_count": inspect_console_repeat_suppression_count,
        "inspect_network_repeat_suppression_count": inspect_network_repeat_suppression_count,
        "meta_action_redirect_to_functional_count": meta_action_redirect_to_functional_count,
        "verification_action_redirect_count": verification_action_redirect_count,
        "meta_action_repeated_priority_lowered_count": meta_action_repeated_priority_lowered_count,
        "policy_selected_meta_action_suppressed_count": policy_selected_meta_action_suppressed_count,
        "semantic_action_type_counts": dict(semantic_action_type_counts),
        "clicked_semantic_action_types": dict(clicked_semantic_action_types),
        "attempted_semantic_action_types": dict(clicked_semantic_action_types),
        "semantic_no_effect_click_count": semantic_no_effect_click_count,
        "high_value_no_effect_click_count": high_value_no_effect_click_count,
        "semantic_button_no_response_count": semantic_button_no_response_count,
        "functional_no_effect_anomaly_count": functional_no_effect_anomaly_count,
        "workout_add_no_effect_count": workout_add_no_effect_count,
        "cart_no_effect_count": cart_no_effect_count,
        "add_no_effect_count": add_no_effect_count,
        "enroll_no_effect_count": enroll_no_effect_count,
        "high_value_enroll_no_response_count": high_value_enroll_no_response_count,
        "high_value_button_no_response_count": high_value_button_no_response_count,
        "no_effect_click_to_new_candidate_fallback_count": int(fallback_reason_counts.get("no_effect_click_to_new_candidate", 0)),
        "action_fallback_invalid_action_count": int(fallback_reason_counts.get("invalid_action_index", 0)),
        "action_fallback_visited_element_key_count": int(fallback_reason_counts.get("visited_element_key", 0)),
        "action_fallback_low_value_candidate_count": int(fallback_reason_counts.get("low_value_generic_candidate", 0)),
        "action_fallback_exploration_redirect_count": int(fallback_reason_counts.get("exploration_redirect", 0)),
        "unclicked_functional_priority_candidate_count": len(observed_functional_priority_targets - clicked_functional_priority_targets),
        "high_value_functional_candidate_count": high_value_functional_candidate_count,
        "catalog_candidate_count": catalog_candidate_count,
        "catalog_selector_match_count": catalog_selector_match_count,
        "catalog_keyword_match_count": catalog_keyword_match_count,
        "data_bug_ids_found": sorted(data_bug_ids_found),
        "empty_state_detected_count": empty_state_detected_count,
        "layout_overflow_candidates": layout_overflow_candidates,
        "layout_overflow_candidates_raw": layout_overflow_candidates_raw,
        "layout_overflow_candidates_unique": len(layout_overflow_unique),
        "layout_overflow_matched_bug_ids": sorted(
            {
                str(item.get("matched_bug_id"))
                for item in layout_overflow_unique.values()
                if item.get("matched_bug_id")
            }
        ),
        "layout_overflow_unmatched_count": sum(1 for item in layout_overflow_unique.values() if not item.get("matched_bug_id")),
        "layout_overflow_match_reasons": layout_overflow_match_reasons,
        "layout_overflow_top_evidence": layout_overflow_top_evidence,
        "button_no_response_candidates": button_no_response_candidates,
        "policy_action_activity_count": sum(
            int(action_counts.get(action_name, 0))
            for action_name in ("click_element", "fill_input", "press_enter", "inspect_network", "inspect_console", "inspect_cart")
        ),
        "catalog_guided_action_count": 0,
        "fill_input_count": int(action_counts.get("fill_input", 0)),
        "inspect_network_count": int(action_counts.get("inspect_network", 0)),
        "inspect_console_count": int(action_counts.get("inspect_console", 0)),
        "api_403_count": api_403_count,
        "async_hang_candidates": async_hang_candidates,
        "cart_quantity_mismatch_candidates": cart_quantity_mismatch_candidates,
        "sparse_data_candidates": sparse_data_candidates,
        "triggered_bug_ids": sorted(triggered_bug_ids),
        "attempted_bug_ids": sorted(attempted_bug_ids),
        "missed_bug_reason": _missed_bug_reasons(
            known_bugs,
            matched_bug_ids,
            attempted_bug_ids,
            action_counts,
            api_403_count,
            clicked_semantic_action_types,
            semantic_button_no_response_count + functional_no_effect_anomaly_count,
        ),
        "missed_semantic_action_reason": _missed_semantic_action_reasons(
            known_bugs,
            matched_bug_ids,
            clicked_semantic_action_types,
            semantic_button_no_response_count + functional_no_effect_anomaly_count,
        ),
        "openended_interactive_candidate_count": openended_interactive_candidate_count,
        "openended_keyword_match_count": openended_keyword_match_count,
        "first_click_step": first_click_step,
        "unique_clicked_targets": sorted(unique_clicked_targets),
        "unique_element_keys": sorted(observed_element_keys),
        "unique_element_key_count": len(observed_element_keys),
        "unique_clicked_element_keys": sorted(clicked_element_keys),
        "unique_clicked_element_key_count": len(clicked_element_keys),
        "inspected_dom_count": int(action_counts.get("inspect_dom", 0)),
        "inspect_dom_repeat_count": inspect_dom_repeat_count,
        "layout_anomaly_count": layout_anomaly_count,
        "interaction_anomaly_count": interaction_anomaly_count,
        "anomaly_types": anomaly_types,
        "top_exploratory_anomalies": exploratory_anomalies[:5],
        "false_positive_filtered_count": _review_status_counts(exploratory_anomalies).get("likely_false_positive", 0),
        "action_diversity_score": action_diversity_score,
        "unique_action_type_count": len([key for key, value in action_counts.items() if int(value or 0) > 0]),
        "unique_clicked_target_count": len(unique_clicked_targets),
        "open_detail_panel_count": int(action_counts.get("open_detail_panel", 0)),
        "click_retry_button_count": int(action_counts.get("click_retry_button", 0)),
        "functional_action_count": int(signal_summary.get("functional_action_count", 0.0)),
        "first_functional_action_step": first_click_step if int(signal_summary.get("functional_action_count", 0.0)) > 0 else None,
        "unique_functional_action_type_count": len(
            [
                action_name
                for action_name in (
                    "click_element",
                    "fill_input",
                    "press_enter",
                )
                if int(action_counts.get(action_name, 0) or 0) > 0
            ]
        ),
        "unique_functional_target_count": len(unique_clicked_targets),
        "debug_meta_action_count": int(signal_summary.get("debug_meta_action_count", 0.0)),
        "debug_meta_repeat_count": int(signal_summary.get("debug_meta_repeat_count", 0.0)),
        "no_functional_action_episode_count": no_functional_episode_count,
        "functional_action_signal_delta_count": int(signal_summary.get("functional_action_signal_delta_count", 0.0)),
        "functional_action_network_delta_count": int(signal_summary.get("functional_action_network_delta_count", 0.0)),
        "verification_action_after_high_value_click_count": int(history.get("verification_action_after_high_value_click_count", 0) or 0),
        "verification_action_redirect_count": verification_action_redirect_count,
        "high_value_click_pending_verification_count": int(history.get("high_value_click_pending_verification_count", 0) or 0),
        "high_value_click_verified_count": int(history.get("high_value_click_verified_count", 0) or 0),
        "high_value_click_without_verification_count": int(history.get("high_value_click_without_verification_count", 0) or 0),
        "reward_functional_action_total": float(signal_summary.get("reward_functional_action_total", 0.0)),
        "penalty_debug_meta_total": float(signal_summary.get("penalty_debug_meta_total", 0.0)),
        "diversity_reward_total": diversity_reward_total,
        "repeated_action_penalty_total": repeated_action_penalty_total,
        "known_bug_reward_total": 0.0,
        "signal_reward_total": float(signal_summary.get("signal_reward_total", 0.0)),
        "exploration_reward_total": float(signal_summary.get("exploration_reward_total", 0.0)),
        "ui_dom_signal_reward_total": float(signal_summary.get("ui_dom_signal_reward_total", 0.0)),
        "console_runtime_signal_reward_total": float(signal_summary.get("console_runtime_signal_reward_total", 0.0)),
        "network_api_signal_reward_total": float(signal_summary.get("network_api_signal_reward_total", 0.0)),
        "cross_layer_signal_reward_total": float(signal_summary.get("cross_layer_signal_reward_total", 0.0)),
        "security_signal_reward_total": float(signal_summary.get("security_signal_reward_total", 0.0)),
        "repeated_penalty_total": float(signal_summary.get("repeated_penalty_total", 0.0)),
        "multi_signal_anomaly_count": int(signal_summary.get("multi_signal_anomaly_count", 0.0)),
        "console_error_count": int(signal_summary.get("console_error_count", 0.0)),
        "runtime_exception_count": int(signal_summary.get("runtime_exception_count", 0.0)),
        "network_error_count": int(signal_summary.get("network_request_failed_count", 0.0)),
        "api_4xx_count": int(signal_summary.get("api_4xx_count", 0.0)),
        "api_5xx_count": int(signal_summary.get("api_5xx_count", 0.0)),
        "api_timeout_count": int(signal_summary.get("api_timeout_count", 0.0)),
        "api_ui_mismatch_count": int(signal_summary.get("api_ui_mismatch_count", 0.0)),
        "security_signal_count": int(signal_summary.get("security_signal_count", 0.0)),
        "signal_delta_reward_total": float(signal_summary.get("signal_delta_reward_total", 0.0)),
        "inspect_action_reward_total": float(signal_summary.get("inspect_action_reward_total", 0.0)),
        "functional_action_signal_reward_total": float(signal_summary.get("functional_action_signal_reward_total", 0.0)),
        "target_signal_types": list(site_profile.get("target_signal_types") or []),
        "observed_signal_types": _observed_signal_types(signal_summary),
        "missing_signal_types": sorted(set(site_profile.get("target_signal_types") or []) - set(_observed_signal_types(signal_summary))),
        "signal_type_coverage": _signal_type_coverage(site_profile.get("target_signal_types") or [], _observed_signal_types(signal_summary)),
        "signal_discovery_score": _signal_discovery_score(signal_summary),
        "ui_dom_signal_count": int(signal_summary.get("multi_signal_anomaly_count", 0.0)),
        "console_runtime_signal_count": int(signal_summary.get("console_error_count", 0.0)) + int(signal_summary.get("runtime_exception_count", 0.0)),
        "network_api_signal_count": int(signal_summary.get("network_request_failed_count", 0.0))
        + int(signal_summary.get("api_4xx_count", 0.0))
        + int(signal_summary.get("api_5xx_count", 0.0))
        + int(signal_summary.get("api_timeout_count", 0.0)),
        "cross_layer_signal_count": int(signal_summary.get("api_ui_mismatch_count", 0.0)),
        "state_coverage_count": len(
            {
                str(item.get("evidence", {}).get("after_url") or item.get("evidence", {}).get("before_url") or "")
                for item in unique_candidates.values()
                if isinstance(item.get("evidence"), Mapping)
            }
        ),
        "new_url_count": len(
            {
                str(item.get("evidence", {}).get("after_url") or "")
                for item in unique_candidates.values()
                if isinstance(item.get("evidence"), Mapping) and item.get("evidence", {}).get("after_url")
            }
        ),
        "new_dom_state_count": layout_anomaly_count + interaction_anomaly_count,
        "signal_collector_enabled": True,
        "playwright_console_listener_enabled": bool(signal_summary.get("playwright_console_listener_enabled", 0.0)),
        "playwright_pageerror_listener_enabled": bool(signal_summary.get("playwright_pageerror_listener_enabled", 0.0)),
        "playwright_network_listener_enabled": bool(signal_summary.get("playwright_network_listener_enabled", 0.0)),
        "playwright_request_listener_enabled": bool(signal_summary.get("playwright_request_listener_enabled", 0.0)),
        "playwright_response_listener_enabled": bool(signal_summary.get("playwright_response_listener_enabled", 0.0)),
        "playwright_requestfailed_listener_enabled": bool(signal_summary.get("playwright_requestfailed_listener_enabled", 0.0)),
        "playwright_listener_warning": "listener unavailable" if int(signal_summary.get("playwright_listener_warning_count", 0) or 0) else "",
        "first_click_reward_count": first_click_reward_count,
        "new_action_type_reward_count": new_action_type_reward_count,
        "new_target_reward_count": new_target_reward_count,
        "inspect_reward_count": inspect_reward_count,
        "open_detail_panel_repeat_penalty_count": open_detail_panel_repeat_penalty_count,
        "no_effect_action_penalty_count": no_effect_action_penalty_count,
        "fallback_reason_counts": dict(fallback_reason_counts),
        "repeated_action_count": _repeated_action_signature_count(history),
        "repeated_action_type_count": _repeated_action_type_count(action_counts),
        "repeated_action_signature_count": _repeated_action_signature_count(history),
        "repeated_element_key_click_count": _repeated_element_key_click_count(history),
        "repeated_click_target_count": _repeated_click_target_count(history),
        "repeated_click_penalty_count": max(0, _repeated_click_target_count(history)),
        "matched_target_reclick_count": _matched_target_reclick_count(history),
        "action_fallback_count": fallback_action_count,
        "policy_selected_valid_action_count": policy_selected_valid_action_count,
        "policy_selected_invalid_action_count": policy_selected_invalid_action_count,
        "policy_selected_visited_element_count": policy_selected_visited_element_count,
        "policy_selected_low_priority_action_count": policy_selected_low_priority_action_count,
        "fallback_applied_count": fallback_applied_count,
        "fallback_applied_rate": _safe_rate(fallback_applied_count, policy_selected_valid_action_count + policy_selected_invalid_action_count),
        "fallback_penalty_total": fallback_penalty_total,
        "fallback_reward_capped_count": fallback_reward_capped_count,
        "policy_executed_action_match_count": policy_executed_action_match_count,
        "policy_executed_action_mismatch_count": policy_executed_action_mismatch_count,
        "fallback_mode": str(fallback_mode or "eval"),
        "disable_fallback": str(fallback_mode or "eval") == "strict",
        "click_suppression_count": max(0, int(action_counts.get("click_element", 0)) - 20),
        "warnings": warnings,
    }
    if float(result.get("fallback_applied_rate") or 0.0) >= 0.25:
        warning = "WARNING: policy relies on fallback"
        if warning not in result["warnings"]:
            result["warnings"].append(warning)
    _emit_event(
        event="site_completed",
        site_id=site_id,
        recall=result.get("recall"),
        precision=result.get("precision"),
        matched_bug_ids=result.get("matched_bug_ids", []),
        missed_bug_ids=result.get("missed_bug_ids", []),
        warnings=result.get("warnings", []),
        action_counts=result.get("action_counts", {}),
        anomaly_types=result.get("anomaly_types", {}),
    )
    if has_catalog:
        known_match_count = len(matched_bug_ids)
        result.update(
            {
                "known_bug_match_count": known_match_count,
                "matched_bug_ids": matched_bug_ids,
                "missed_bug_ids": missed_bug_ids,
                "precision": known_match_count / unique_detected if unique_detected else 0.0,
                "known_bug_precision": known_match_count / deduped_known_bug_candidates if deduped_known_bug_candidates else 0.0,
                "catalog_precision": len(confirmed_bugs) / (len(confirmed_bugs) + len(catalog_related_anomalies))
                if (len(confirmed_bugs) + len(catalog_related_anomalies)) else 0.0,
                "known_bug_candidate_count": deduped_known_bug_candidates,
                "exploratory_anomaly_count": len(exploratory_anomalies),
                "recall": min(1.0, known_match_count / max(1, len(known_bug_ids))),
            }
        )
        target_recall = site.get("target_recall")
        if target_recall is None and site_id in {"site001", "site003"}:
            target_recall = 1.0
        if target_recall is not None and result["recall"] < float(target_recall):
            result["warnings"].append(
                f"WARNING: recall target {float(target_recall):.3g} was not achieved; investigate before precision tuning."
            )
        elif result["recall"] >= 1.0:
            result["warnings"].append("INFO: recall target achieved; next optimization target is precision and evidence quality.")
        if unique_detected and result["precision"] < result["known_bug_precision"]:
            result["warnings"].append("WARNING: precision is low because exploratory anomalies are included in detected_bug_count.")
        if int(result["repeated_click_target_count"]) > 10:
            result["warnings"].append("WARNING: repeated click target count is high; click suppression should be improved.")
        if any(item.get("type") == "layout-overflow" and _uses_bbox_fallback(item.get("evidence", {})) for item in confirmed_bugs):
            result["warnings"].append("WARNING: layout-overflow matched using bbox fallback; evidence quality can be improved.")
        if site_id == "site003":
            if result["site_identity_match"] is False:
                result["warnings"].append(
                    "WARNING: site003 evaluation used a page whose data-bug-id prefix does not match site003. "
                    "Check configs/training_sites.json base_url."
                )
            if has_catalog and result["catalog_candidate_count"] == 0:
                result["warnings"].append("WARNING: known bug catalog is available but no catalog candidates were observed.")
            if result["click_element_count"] == 0:
                result["warnings"].append("WARNING: site003 click_element_count is 0; workout_add button was never tested.")
            if result["inspect_cart_count"] > 10:
                result["warnings"].append("WARNING: site003 policy selected inspect_cart repeatedly; action fallback applied.")
            if result["recall"] == 0.0 and total_detected_candidates == 0:
                result["warnings"].append("WARNING: site003 recall is 0.0 and anomaly count is 0; inspect observation/anomaly extraction.")
            if result["recall"] == 0.0 and int(result["click_element_count"]) > 30 and button_no_response_candidates == 0:
                result["warnings"].append("WARNING: site003 many clicks but no button-no-response anomalies; inspect click target evidence.")
            if result["recall"] == 0.0:
                result["warnings"].append("WARNING: site003 recall is 0.0; no known GUI bug was matched.")
            if (
                result["layout_overflow_candidates_raw"] > 0
                and not result["layout_overflow_matched_bug_ids"]
            ):
                result["warnings"].append(
                    "WARNING: layout-overflow candidates were generated but no layout-overflow known bug was matched. "
                    "Check catalog_bug_id_matches, selector_hint, and layout_check evidence."
                )
    else:
        review_counts = _review_status_counts(exploratory_anomalies)
        review_artifacts = _write_openended_review_artifacts(site_id, site, result, exploratory_anomalies, anomaly_types)
        result.update(
            {
                "known_bug_match_count": None,
                "matched_bug_ids": [],
                "missed_bug_ids": [],
                "precision": None,
                "recall": None,
                "review_candidate_count": len(exploratory_anomalies),
                "likely_true_positive_count": review_counts.get("likely_true_positive", 0),
                "likely_false_positive_count": review_counts.get("likely_false_positive", 0),
                "false_positive_filtered_count": review_counts.get("likely_false_positive", 0),
                "needs_review_count": review_counts.get("needs_review", 0),
                "review_artifacts": review_artifacts,
                "generated_review_doc_path": review_artifacts.get("review_doc_path"),
                "generated_candidate_catalog_path": review_artifacts.get("candidate_catalog_path"),
                "openended_anomaly_discovery": {
                    "unique_detected_candidates": unique_detected,
                    "total_detected_candidates": total_detected_candidates,
                    "confirmed_bugs": [],
                    "catalog_related_anomalies": catalog_related_anomalies,
                    "exploratory_anomalies": exploratory_anomalies,
                    "anomaly_types": anomaly_types,
                    "action_counts": dict(action_counts),
                    "openended_interactive_candidate_count": openended_interactive_candidate_count,
                    "openended_keyword_match_count": openended_keyword_match_count,
                    "click_element_count": int(action_counts.get("click_element", 0)),
                    "first_click_step": first_click_step,
                    "unique_clicked_targets": sorted(unique_clicked_targets),
                    "inspect_dom_repeat_count": inspect_dom_repeat_count,
                    "layout_anomaly_count": layout_anomaly_count,
                    "interaction_anomaly_count": interaction_anomaly_count,
                    "action_diversity_score": action_diversity_score,
                    "top_exploratory_anomalies": exploratory_anomalies[:5],
                    "review_candidate_count": len(exploratory_anomalies),
                    "likely_true_positive_count": review_counts.get("likely_true_positive", 0),
                    "likely_false_positive_count": review_counts.get("likely_false_positive", 0),
                    "false_positive_filtered_count": review_counts.get("likely_false_positive", 0),
                    "needs_review_count": review_counts.get("needs_review", 0),
                    "review_artifacts": review_artifacts,
                },
            }
        )
        if int(action_counts.get("click_element", 0)) == 0:
            result["warnings"].append("WARNING: open-ended site was not interacted with; click exploration is required.")
        if int(action_counts.get("inspect_dom", 0)) > 50:
            result["warnings"].append("WARNING: inspect_dom repeated excessively.")
        if action_diversity_score < 0.5:
            result["warnings"].append("WARNING: action diversity is low.")
    _write_live_status(
        live_status_dir,
        scan_id=scan_id,
        status="completed",
        stage="scan_completed",
        current_site=site_id,
        current_episode=episodes,
        current_step=max_steps,
        latest_screenshot_path=str(live_status_dir / "latest.png") if live_status_dir is not None and (live_status_dir / "latest.png").exists() else None,
        last_screenshot_ok=True,
        last_screenshot_error=None,
    )
    return result


def _apply_eval_fallback_mask(
    action_space: ActionSpace,
    action_mask: np.ndarray,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> tuple[np.ndarray, str, int]:
    mask = np.asarray(action_mask, dtype=np.float32).copy()
    site_id = _site_id_from_observation(observation)
    warning = ""
    repeated = 0
    if isinstance(history, dict):
        candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
        history["current_candidates"] = candidates if isinstance(candidates, list) else []
        history["_fallback_reasons_step"] = []
    if site_id not in {"site001", "site9800"}:
        mask[action_space.encode("inspect_cart", 0)] = 0.0
    if _prefer_verification_action(action_space, mask, history):
        repeated = 1
        _record_fallback_reason(history, "exploration_redirect")
        warning = warning or f"WARNING: {site_id} high-value click verification action selected."
    if _suppress_low_value_generic_candidates(action_space, mask, observation, history) > 0:
        repeated = 1
        _record_fallback_reason(history, "low_value_generic_candidate")
        _record_fallback_reason(history, "policy_selected_low_priority_candidate")
        warning = warning or f"WARNING: {site_id} low-value generic candidate suppressed; exploration fallback applied."
    suppressed_clicks = _suppress_repeated_clicks(action_space, mask, observation, history)
    if suppressed_clicks:
        _record_fallback_reason(history, "visited_element_key")
    blocked_signatures = _suppress_repeated_action_signatures(action_space, mask, history)
    last_action_type = str(history.get("last_action_type") or "")
    counts = history.get("consecutive_action_type_counts", {})
    consecutive = int(counts.get(last_action_type, 0) or 0) if isinstance(counts, Mapping) and last_action_type else 0
    action_counts = history.get("action_type_counts", {})
    inspect_dom_total = int(action_counts.get("inspect_dom", 0) or 0) if isinstance(action_counts, Mapping) else 0
    inspect_layout_total = int(action_counts.get("inspect_layout", 0) or 0) if isinstance(action_counts, Mapping) else 0
    if isinstance(action_counts, Mapping):
        meta_redirected = False
        for meta_action in ("inspect_console", "inspect_network", "inspect_dom"):
            total = int(action_counts.get(meta_action, 0) or 0)
            if total < 1:
                continue
            _disable_action_type(action_space, mask, meta_action)
            repeated = 1
            _record_fallback_reason(history, "policy_selected_meta_action")
            if isinstance(history, dict):
                history["repeated_meta_action_suppression_count"] = int(
                    history.get("repeated_meta_action_suppression_count", 0) or 0
                ) + 1
                _record_meta_suppression(history)
                if meta_action == "inspect_console":
                    history["inspect_console_repeat_suppression_count"] = int(
                        history.get("inspect_console_repeat_suppression_count", 0) or 0
                    ) + 1
                if meta_action == "inspect_network":
                    history["inspect_network_repeat_suppression_count"] = int(
                        history.get("inspect_network_repeat_suppression_count", 0) or 0
                    ) + 1
            warning = warning or f"WARNING: {site_id} repeated meta action {meta_action} blocked; functional fallback applied."
            meta_redirected = True
        if meta_redirected and _prefer_functional_actions(action_space, mask, observation, history):
            if isinstance(history, dict):
                history["meta_action_redirect_to_functional_count"] = int(
                    history.get("meta_action_redirect_to_functional_count", 0) or 0
                ) + 1
        for meta_action in ("open_detail_panel", "click_retry_button", "click_trigger_button", "click_recovery_button"):
            if int(action_counts.get(meta_action, 0) or 0) >= 1:
                _disable_action_type(action_space, mask, meta_action)
                repeated = 1
                _record_fallback_reason(history, "policy_selected_meta_action")
                if isinstance(history, dict):
                    _record_meta_suppression(history)
                warning = warning or f"WARNING: {site_id} {meta_action} repeat blocked; functional fallback applied."
        functional_count = int(history.get("functional_action_count", 0) or 0)
        debug_or_scroll_count = sum(
            int(action_counts.get(action_name, 0) or 0)
            for action_name in (
                "open_detail_panel",
                "click_retry_button",
                "click_trigger_button",
                "click_recovery_button",
                "scroll_down",
                "scroll_up",
                "inspect_dom",
                "inspect_layout",
                "inspect_network",
                "inspect_console",
            )
        )
        no_progress_steps = int(history.get("no_progress_steps", 0) or 0)
        if (
            (
                functional_count == 0
                and debug_or_scroll_count >= 2
            )
            or (
                last_action_type == "click_element"
                and no_progress_steps >= 1
            )
        ) and _prefer_functional_actions(action_space, mask, observation, history):
            repeated = 1
            _record_fallback_reason(history, "policy_selected_meta_action")
            if last_action_type == "click_element" and no_progress_steps >= 1:
                _record_fallback_reason(history, "exploration_redirect")
                warning = warning or f"WARNING: {site_id} no-effect click redirected to unclicked functional candidate."
            else:
                _record_fallback_reason(history, "exploration_redirect")
                warning = warning or f"WARNING: {site_id} debug/meta or scroll-only sequence blocked; functional fallback applied."
        if _suppress_visited_click_targets(action_space, mask, observation, history) > 0:
            repeated = 1
            _record_fallback_reason(history, "visited_element_key")
            warning = warning or f"WARNING: {site_id} visited target suppressed; unclicked candidate fallback applied."
    signature_scoped_actions = {"click_element", "fill_input", "press_enter"}
    if consecutive >= 2 and last_action_type and last_action_type not in signature_scoped_actions:
        if _prefer_functional_actions(action_space, mask, observation, history):
            history["repeated_action_redirect_count"] = int(history.get("repeated_action_redirect_count", 0) or 0) + 1
            _record_fallback_reason(history, "exploration_redirect")
        else:
            _disable_action_type(action_space, mask, last_action_type)
            _record_fallback_reason(history, "policy_selected_meta_action")
            if isinstance(history, dict):
                history["policy_selected_meta_action_suppressed_count"] = int(
                    history.get("policy_selected_meta_action_suppressed_count", 0) or 0
                ) + 1
        repeated = 1
        warning = f"WARNING: {site_id} policy selected {last_action_type} repeatedly; action fallback applied."
    if _is_openended_observation(observation) and last_action_type == "inspect_dom" and consecutive >= 2:
        _disable_action_type(action_space, mask, "inspect_dom")
        repeated = 1
        _record_fallback_reason(history, "policy_selected_meta_action")
        if isinstance(history, dict):
            _record_meta_suppression(history)
        warning = f"WARNING: {site_id} policy selected inspect_dom repeatedly; action fallback applied."
    if _is_openended_observation(observation) and inspect_dom_total >= 10:
        _disable_action_type(action_space, mask, "inspect_dom")
        _record_fallback_reason(history, "policy_selected_meta_action")
        if isinstance(history, dict):
            _record_meta_suppression(history)
        warning = warning or f"WARNING: {site_id} inspect_dom budget exhausted; action fallback applied."
    if site_id == "site9800" and inspect_layout_total >= 8:
        _disable_action_type(action_space, mask, "inspect_layout")
        _record_fallback_reason(history, "policy_selected_meta_action")
        if isinstance(history, dict):
            _record_meta_suppression(history)
        warning = warning or f"WARNING: {site_id} inspect_layout budget exhausted; action fallback applied."
    elif _is_openended_observation(observation) and inspect_layout_total >= 12:
        _disable_action_type(action_space, mask, "inspect_layout")
        _record_fallback_reason(history, "policy_selected_meta_action")
        if isinstance(history, dict):
            _record_meta_suppression(history)
        warning = warning or f"WARNING: {site_id} inspect_layout budget exhausted; action fallback applied."
    if not mask.any():
        _record_fallback_reason(history, "invalid_action_index")
        repeated = 1
        warning = warning or f"WARNING: {site_id} invalid action mask; global fallback applied."
        _apply_global_fallback(action_space, mask)
    if _delay_finish_when_unclicked_candidates(action_space, mask, observation, history) > 0:
        _record_fallback_reason(history, "exploration_redirect")
        warning = warning or f"WARNING: {site_id} finish_episode delayed; unclicked functional candidates remain."
    if blocked_signatures:
        repeated = 1
        _record_fallback_reason(history, "repeated_action_signature")
        warning = warning or f"WARNING: {site_id} repeated action signature blocked; action fallback applied."
    return mask, warning, repeated


def _fallback_reason(warning: str) -> str:
    normalized = warning.lower()
    if "repeated meta action" in normalized:
        return "policy_selected_meta_action"
    if "repeated action signature" in normalized:
        return "repeated_action_signature"
    if "low-value generic" in normalized:
        return "low_value_generic_candidate"
    if "visited target" in normalized:
        return "visited_element_key"
    if "verification action" in normalized:
        return "exploration_redirect"
    if "inspect_dom" in normalized and "repeated" in normalized:
        return "policy_selected_meta_action"
    if "repeatedly" in normalized:
        return "policy_selected_meta_action"
    if "budget exhausted" in normalized:
        return "policy_selected_meta_action"
    if "no-effect click" in normalized:
        return "exploration_redirect"
    if "finish_episode delayed" in normalized:
        return "exploration_redirect"
    return "fallback"


def _comparison_exclusion_reason(
    *,
    status: str,
    partial: bool,
    completed_episodes: int,
    requested_episodes: int,
    timeout_count: int,
    observed_url_sample: str,
    action_counts: Mapping[str, Any],
    openended_interactive_candidate_count: int,
    error_message: str,
) -> str:
    if str(status) == "failed":
        if int(timeout_count or 0) > 0 or "timeout" in str(error_message or "").lower():
            return "timeout"
        return "failed"
    if int(timeout_count or 0) > 0:
        return "timeout"
    if int(completed_episodes or 0) == 0:
        return "zero_completed_episodes"
    if bool(partial):
        return "partial"
    if int(completed_episodes or 0) < int(requested_episodes or 0):
        return "incomplete_episodes"
    if not str(observed_url_sample or ""):
        return "empty_observed_url"
    if int(openended_interactive_candidate_count or 0) <= 0:
        return "zero_interactive_candidates"
    if not isinstance(action_counts, Mapping) or not action_counts:
        return "zero_action_counts"
    return ""


def _history_counter_snapshot(history: Mapping[str, Any]) -> dict[str, int]:
    keys = (
        "visited_target_suppression_count",
        "visited_target_skipped_count",
        "visited_element_key_suppression_count",
        "memory_guided_candidate_switch_count",
        "repeated_action_redirect_count",
        "repeated_target_redirect_count",
        "finish_delayed_by_unclicked_candidate_count",
        "repeated_meta_action_suppression_count",
        "inspect_console_repeat_suppression_count",
        "inspect_network_repeat_suppression_count",
        "meta_action_redirect_to_functional_count",
        "verification_action_redirect_count",
        "meta_action_repeated_priority_lowered_count",
        "policy_selected_meta_action_suppressed_count",
    )
    return {key: int(history.get(key, 0) or 0) for key in keys}


def _history_counter_delta(history: Mapping[str, Any], before: Mapping[str, int]) -> dict[str, int]:
    return {
        key: max(0, int(history.get(key, 0) or 0) - int(before.get(key, 0) or 0))
        for key in before
    }


def _disable_action_type(action_space: ActionSpace, mask: np.ndarray, action_type: str) -> None:
    if action_type not in action_space.action_types:
        return
    for index in range(action_space.max_candidates):
        try:
            mask[action_space.encode(action_type, index)] = 0.0
        except ValueError:
            pass


def _record_fallback_reason(history: Mapping[str, Any], reason: str) -> None:
    if not isinstance(history, dict) or not reason:
        return
    reasons = history.setdefault("_fallback_reasons_step", [])
    if isinstance(reasons, list):
        reasons.append(str(reason))


def _record_meta_suppression(history: Mapping[str, Any]) -> None:
    if not isinstance(history, dict):
        return
    history["meta_action_repeated_priority_lowered_count"] = int(
        history.get("meta_action_repeated_priority_lowered_count", 0) or 0
    ) + 1
    history["policy_selected_meta_action_suppressed_count"] = int(
        history.get("policy_selected_meta_action_suppressed_count", 0) or 0
    ) + 1


def _attach_policy_execution_fields(
    action: Dict[str, Any],
    *,
    policy_action_id: int,
    policy_action: Mapping[str, Any],
    executed_action_id: int,
    fallback_applied: bool,
    fallback_reason: str,
) -> None:
    action.update(
        {
            "policy_action_index": int(policy_action_id),
            "policy_action_type": str(policy_action.get("action_type") or ""),
            "policy_action_signature": str(policy_action.get("action_signature") or ""),
            "policy_action_element_key": str(policy_action.get("action_element_key") or ""),
            "executed_action_index": int(executed_action_id),
            "executed_action_type": str(action.get("action_type") or ""),
            "executed_action_signature": str(action.get("action_signature") or ""),
            "executed_action_element_key": str(action.get("action_element_key") or ""),
            "fallback_applied": bool(fallback_applied),
            "fallback_reason": str(fallback_reason or ""),
        }
    )


def _policy_action_visited(action: Mapping[str, Any], history: Mapping[str, Any]) -> bool:
    element_key = str(action.get("action_element_key") or "")
    if element_key:
        clicked_keys = history.get("clicked_element_keys", set())
        if isinstance(clicked_keys, set) and element_key in clicked_keys:
            return True
        element_counts = history.get("element_key_click_counts", {})
        if isinstance(element_counts, Mapping) and int(element_counts.get(element_key, 0) or 0) > 0:
            return True
    bid = str(action.get("action_bid") or action.get("clicked_bid") or "")
    clicked_bids = history.get("clicked_bids", set())
    return bool(bid and isinstance(clicked_bids, set) and bid in clicked_bids)


def _policy_action_low_priority(action: Mapping[str, Any]) -> bool:
    action_type = str(action.get("action_type") or "")
    return bool(
        action.get("is_low_value_generic_candidate")
        or action_type in {"open_detail_panel", "click_retry_button", "click_recovery_button", "inspect_console", "inspect_network", "inspect_dom"}
    )


def _safe_visibility(candidate: Mapping[str, Any]) -> float:
    if "visibility" in candidate and candidate.get("visibility") is not None:
        try:
            value = float(candidate.get("visibility") or 0.0)
        except (TypeError, ValueError):
            value = 1.0
    elif "visible" in candidate:
        value = 1.0 if bool(candidate.get("visible")) else 0.0
    else:
        value = 1.0
    if value != value:
        return 1.0
    return max(0.0, min(1.0, value))


def _suppress_low_value_generic_candidates(
    action_space: ActionSpace,
    mask: np.ndarray,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> int:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    if not isinstance(candidates, list):
        return 0
    has_better_candidate = any(
        isinstance(candidate, Mapping)
        and not candidate.get("is_low_value_generic_candidate")
        and bool(candidate.get("clickable"))
        and _safe_visibility(candidate) > 0.0
        and (
            candidate.get("functional_priority_candidate")
            or candidate.get("functional_priority")
            or candidate.get("is_high_value_functional_candidate")
            or candidate.get("semantic_action_type")
        )
        for candidate in candidates[: action_space.max_candidates]
    )
    if not has_better_candidate:
        return 0
    suppressed = 0
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if not isinstance(candidate, Mapping) or not candidate.get("is_low_value_generic_candidate"):
            continue
        try:
            action_id = action_space.encode("click_element", index)
        except ValueError:
            continue
        if action_id < len(mask) and float(mask[action_id]) > 0.0:
            mask[action_id] = 0.0
            suppressed += 1
    if suppressed and isinstance(history, dict):
        history["action_fallback_low_value_candidate_count"] = int(history.get("action_fallback_low_value_candidate_count", 0) or 0) + suppressed
    return suppressed


def _prefer_verification_action(action_space: ActionSpace, mask: np.ndarray, history: Mapping[str, Any]) -> bool:
    if not history.get("verification_pending_after_high_value_click"):
        return False
    action_counts = history.get("action_type_counts", {})
    if not isinstance(action_counts, Mapping):
        action_counts = {}
    for action_type in ("inspect_dom", "inspect_network", "inspect_console"):
        if int(action_counts.get(action_type, 0) or 0) > 0:
            continue
        try:
            action_id = action_space.encode(action_type, 0)
        except ValueError:
            continue
        if action_id < len(mask):
            mask[:] = 0.0
            mask[action_id] = 1.0
            if isinstance(history, dict):
                history["verification_action_redirect_count"] = int(history.get("verification_action_redirect_count", 0) or 0) + 1
            return True
    return False


def _suppress_repeated_action_signatures(
    action_space: ActionSpace,
    mask: np.ndarray,
    history: Mapping[str, Any],
    threshold: int = 3,
) -> int:
    counts = history.get("action_signature_counts", {})
    if not isinstance(counts, Mapping):
        return 0
    blocked = 0
    for signature, count in counts.items():
        if int(count or 0) < threshold:
            continue
        action_type, candidate_index, element_key = _parse_action_signature(str(signature))
        if not action_type:
            continue
        if element_key:
            blocked += _disable_element_key_action(action_space, mask, action_type, element_key, history)
            continue
        if candidate_index is None:
            before = int(np.count_nonzero(mask))
            _disable_action_type(action_space, mask, action_type)
            blocked += max(0, before - int(np.count_nonzero(mask)))
            continue
        try:
            action_id = action_space.encode(action_type, candidate_index)
        except ValueError:
            continue
        if action_id < len(mask) and mask[action_id] > 0:
            mask[action_id] = 0.0
            blocked += 1
    return blocked


def _disable_element_key_action(
    action_space: ActionSpace,
    mask: np.ndarray,
    action_type: str,
    element_key: str,
    history: Mapping[str, Any],
) -> int:
    candidates = history.get("current_candidates", [])
    if not isinstance(candidates, list):
        return 0
    blocked = 0
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if not isinstance(candidate, Mapping) or str(candidate.get("element_key") or "") != element_key:
            continue
        try:
            action_id = action_space.encode(action_type, index)
        except ValueError:
            continue
        if action_id < len(mask) and float(mask[action_id]) > 0.0:
            mask[action_id] = 0.0
            blocked += 1
    return blocked


def _apply_global_fallback(action_space: ActionSpace, mask: np.ndarray) -> None:
    for action_type in ("inspect_dom", "inspect_network", "scroll_down", "finish_episode", "noop"):
        try:
            action_id = action_space.encode(action_type, 0)
        except ValueError:
            continue
        if action_id < len(mask):
            mask[action_id] = 1.0
            return


def _suppress_visited_click_targets(
    action_space: ActionSpace,
    mask: np.ndarray,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> int:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    if isinstance(history, dict):
        history["current_candidates"] = candidates if isinstance(candidates, list) else []
    if not isinstance(candidates, list):
        return 0
    click_counts = history.get("click_target_counts", {})
    element_counts = history.get("element_key_click_counts", {})
    if not isinstance(click_counts, Mapping):
        click_counts = {}
    has_unclicked = False
    visited_action_ids: list[int] = []
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if not isinstance(candidate, Mapping):
            continue
        if not bool(candidate.get("functional_priority_candidate") or candidate.get("functional_priority")):
            continue
        try:
            action_id = action_space.encode("click_element", index)
        except ValueError:
            continue
        if action_id >= len(mask) or float(mask[action_id]) <= 0.0:
            continue
        if _candidate_click_count(candidate, click_counts, element_counts) > 0:
            visited_action_ids.append(action_id)
        else:
            has_unclicked = True
    if not has_unclicked:
        return 0
    suppressed = 0
    for action_id in visited_action_ids:
        if float(mask[action_id]) > 0.0:
            mask[action_id] = 0.0
            suppressed += 1
    if suppressed:
        if isinstance(history, dict):
            history["visited_target_suppression_count"] = int(history.get("visited_target_suppression_count", 0) or 0) + suppressed
            history["visited_target_skipped_count"] = int(history.get("visited_target_skipped_count", 0) or 0) + suppressed
            history["visited_element_key_suppression_count"] = int(history.get("visited_element_key_suppression_count", 0) or 0) + suppressed
            if history.get("memory_encoder_enabled"):
                history["memory_guided_candidate_switch_count"] = int(history.get("memory_guided_candidate_switch_count", 0) or 0) + 1
            history["repeated_target_redirect_count"] = int(history.get("repeated_target_redirect_count", 0) or 0) + 1
    return suppressed


def _delay_finish_when_unclicked_candidates(
    action_space: ActionSpace,
    mask: np.ndarray,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> int:
    if not _has_unclicked_functional_candidate(action_space, mask, observation, history):
        return 0
    delayed = 0
    if "finish_episode" in action_space.action_types:
        try:
            finish_id = action_space.encode("finish_episode", 0)
        except ValueError:
            finish_id = -1
        if 0 <= finish_id < len(mask) and float(mask[finish_id]) > 0.0:
            mask[finish_id] = 0.0
            delayed = 1
    if delayed and isinstance(history, dict):
        history["finish_delayed_by_unclicked_candidate_count"] = int(
            history.get("finish_delayed_by_unclicked_candidate_count", 0) or 0
        ) + delayed
    if delayed and not mask.any():
        _prefer_functional_actions(action_space, mask, observation, history)
    return delayed


def _has_unclicked_functional_candidate(
    action_space: ActionSpace,
    mask: np.ndarray,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> bool:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    click_counts = history.get("click_target_counts", {})
    element_counts = history.get("element_key_click_counts", {})
    if not isinstance(candidates, list):
        return False
    if not isinstance(click_counts, Mapping):
        click_counts = {}
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if not isinstance(candidate, Mapping):
            continue
        if not bool(candidate.get("functional_priority_candidate") or candidate.get("functional_priority")):
            continue
        try:
            action_id = action_space.encode("click_element", index)
        except ValueError:
            continue
        if action_id < len(mask) and float(mask[action_id]) > 0.0 and _candidate_click_count(candidate, click_counts, element_counts) <= 0:
            return True
    return False


def _prefer_functional_actions(
    action_space: ActionSpace,
    mask: np.ndarray,
    observation: Mapping[str, Any] | None = None,
    history: Mapping[str, Any] | None = None,
) -> bool:
    prioritized_ids = _functional_priority_action_ids(action_space, mask, observation or {}, history or {})
    functional_ids: list[int] = list(prioritized_ids)
    if prioritized_ids:
        mask[:] = 0.0
        for action_id in functional_ids:
            mask[action_id] = 1.0
        return True
    for action_type in ("click_element", "fill_input", "press_enter"):
        if action_type not in action_space.action_types:
            continue
        for index in range(action_space.max_candidates):
            try:
                action_id = action_space.encode(action_type, index)
            except ValueError:
                continue
            if action_id < len(mask) and float(mask[action_id]) > 0.0:
                functional_ids.append(action_id)
    functional_ids = list(dict.fromkeys(functional_ids))
    if not functional_ids:
        return False
    mask[:] = 0.0
    for action_id in functional_ids:
        mask[action_id] = 1.0
    return True


def _functional_priority_action_ids(
    action_space: ActionSpace,
    mask: np.ndarray,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> list[int]:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    click_counts = history.get("click_target_counts", {})
    element_counts = history.get("element_key_click_counts", {})
    if not isinstance(click_counts, Mapping):
        click_counts = {}
    if not isinstance(candidates, list):
        return []
    has_unclicked = any(
        isinstance(candidate, Mapping)
        and bool(candidate.get("functional_priority_candidate") or candidate.get("functional_priority"))
        and _candidate_click_count(candidate, click_counts, element_counts) <= 0
        for candidate in candidates[: action_space.max_candidates]
    )
    ranked: list[tuple[float, int]] = []
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if not isinstance(candidate, Mapping):
            continue
        if not bool(candidate.get("functional_priority_candidate") or candidate.get("functional_priority")):
            continue
        key = _candidate_key(candidate)
        clicked_count = _candidate_click_count(candidate, click_counts, element_counts)
        if has_unclicked and clicked_count > 0:
            if isinstance(history, dict):
                history["visited_target_suppression_count"] = int(history.get("visited_target_suppression_count", 0) or 0) + 1
                history["visited_target_skipped_count"] = int(history.get("visited_target_skipped_count", 0) or 0) + 1
                history["visited_element_key_suppression_count"] = int(history.get("visited_element_key_suppression_count", 0) or 0) + 1
                if history.get("memory_encoder_enabled"):
                    history["memory_guided_candidate_switch_count"] = int(history.get("memory_guided_candidate_switch_count", 0) or 0) + 1
            continue
        try:
            action_id = action_space.encode("click_element", index)
        except ValueError:
            continue
        if action_id >= len(mask) or float(mask[action_id]) <= 0.0:
            continue
        score = float(candidate.get("openended_action_priority", 0.0) or 0.0)
        score += 6.0 if clicked_count <= 0 else -20.0
        score += 5.0 if bool(candidate.get("is_high_value_functional_candidate")) and clicked_count <= 0 else 0.0
        score += 4.0 if str(candidate.get("semantic_action_type") or "") == "workout_add" else 0.0
        score += 1.0 if str(candidate.get("role") or "").lower() == "button" else 0.0
        score += 0.5 if str(candidate.get("tag") or "").lower() == "button" else 0.0
        ranked.append((score, action_id))
    ranked.sort(reverse=True)
    return [action_id for _, action_id in ranked]


def _candidate_click_count(
    candidate: Mapping[str, Any],
    click_counts: Mapping[str, Any],
    element_counts: Mapping[str, Any] | None = None,
) -> int:
    keys = {
        str(candidate.get("element_key") or ""),
        _candidate_key(candidate),
        str(candidate.get("bid") or ""),
        str(candidate.get("name") or ""),
        str(candidate.get("text") or ""),
    }
    values = [int(click_counts.get(key, 0) or 0) for key in keys if key]
    if isinstance(element_counts, Mapping):
        values.extend(int(element_counts.get(key, 0) or 0) for key in keys if key)
    return max(values, default=0)


def _record_eval_action_signature(history: Dict[str, Any], action: Mapping[str, Any]) -> None:
    signature = _action_signature(action)
    counts = history.setdefault("action_signature_counts", {})
    if isinstance(counts, dict):
        counts[signature] = int(counts.get(signature, 0) or 0) + 1


def _current_action_signature_count(history: Mapping[str, Any], action: Mapping[str, Any]) -> int:
    counts = history.get("action_signature_counts", {})
    if not isinstance(counts, Mapping):
        return 0
    return int(counts.get(_action_signature(action), 0) or 0)


def _action_signature(action: Mapping[str, Any]) -> str:
    action_type = str(action.get("action_type") or "")
    if action.get("action_signature"):
        return str(action.get("action_signature"))
    element_key = str(action.get("action_element_key") or "")
    if element_key:
        return f"{action_type}::{element_key}"
    candidate_index = int(action.get("candidate_index", 0) or 0)
    target = str(action.get("clicked_bid") or action.get("clicked_text") or "")
    if target:
        return f"{action_type}:{candidate_index}:{target}"
    if action_type in {"click_element", "fill_input", "press_enter", "open_detail_panel", "click_trigger_button", "click_retry_button", "click_recovery_button"}:
        return f"{action_type}:{candidate_index}"
    return f"{action_type}:*"


def _state_signature(observation: Mapping[str, Any]) -> tuple[str, str, int, int, int]:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    runtime = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    url = str(page_state.get("url") or "") if isinstance(page_state, Mapping) else ""
    title = str(page_state.get("title") or "") if isinstance(page_state, Mapping) else ""
    text = str(page_state.get("text") or page_state.get("visible_text") or "") if isinstance(page_state, Mapping) else ""
    dom_node_count = int(page_state.get("dom_node_count") or 0) if isinstance(page_state, Mapping) else 0
    if not dom_node_count and isinstance(runtime, Mapping):
        dom_node_count = int(runtime.get("dom_node_count") or 0)
    candidate_count = len(candidates) if isinstance(candidates, list) else 0
    return (url, title, len(text), dom_node_count, candidate_count)


def _page_text_length(observation: Mapping[str, Any]) -> int:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    runtime_signals = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    if isinstance(page_state, Mapping):
        for key in ("page_text_length", "text_length", "visible_text_length"):
            value = page_state.get(key)
            if value is not None:
                try:
                    return int(value or 0)
                except (TypeError, ValueError):
                    pass
        text = str(page_state.get("page_text") or page_state.get("visible_text") or page_state.get("text") or "")
        if text:
            return len(text)
    if isinstance(runtime_signals, Mapping):
        text = str(runtime_signals.get("page_text") or "")
        if text:
            return len(text)
    return 0


def _episode_progress_made(
    *,
    before_unique_count: int,
    after_unique_count: int,
    before_matched_count: int,
    after_matched_count: int,
    before_signature: tuple[str, str, int, int, int],
    after_signature: tuple[str, str, int, int, int],
) -> bool:
    return (
        int(after_unique_count or 0) > int(before_unique_count or 0)
        or int(after_matched_count or 0) > int(before_matched_count or 0)
        or before_signature != after_signature
    )


def _parse_action_signature(signature: str) -> tuple[str, int | None, str]:
    if "::" in signature:
        action_type, element_key = signature.split("::", 1)
        return action_type, None, element_key
    parts = signature.split(":", 2)
    if not parts:
        return "", None, ""
    action_type = parts[0]
    if len(parts) < 2 or parts[1] == "*":
        return action_type, None, ""
    try:
        return action_type, int(parts[1]), ""
    except ValueError:
        return action_type, None, ""


def _suppress_repeated_clicks(
    action_space: ActionSpace,
    mask: np.ndarray,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> int:
    candidates = observation.get("candidate_elements", []) or []
    click_counts = history.get("click_target_counts", {})
    element_counts = history.get("element_key_click_counts", {})
    if not isinstance(candidates, list) or not isinstance(click_counts, Mapping):
        return 0
    if not isinstance(element_counts, Mapping):
        element_counts = {}
    suppressed = 0
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if not isinstance(candidate, Mapping):
            continue
        candidate_key = _candidate_key(candidate)
        element_key = str(candidate.get("element_key") or candidate_key)
        already_clicked = max(
            int(click_counts.get(candidate_key, 0) or 0),
            int(element_counts.get(element_key, 0) or 0),
        ) >= 2
        if already_clicked:
            try:
                action_id = action_space.encode("click_element", index)
            except ValueError:
                continue
            if mask[action_id] > 0:
                mask[action_id] = 0.0
                suppressed += 1
    return suppressed


def _site_id_from_observation(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation, Mapping) else {}
    runtime_signals = observation.get("runtime_signals", {}) if isinstance(observation, Mapping) else {}
    site_id = str(
        (page_state.get("site_id") if isinstance(page_state, Mapping) else "")
        or (runtime_signals.get("site_id") if isinstance(runtime_signals, Mapping) else "")
        or ""
    )
    if site_id:
        return site_id
    url = str(page_state.get("url") or "") if isinstance(page_state, Mapping) else ""
    return "site003" if ":9222" in url else "site001" if ":9220" in url else ""


def _missed_bug_reasons(
    known_bugs: List[Mapping[str, Any]],
    matched_bug_ids: List[str],
    attempted_bug_ids: set[str],
    action_counts: Mapping[str, int],
    api_403_count: int,
    attempted_semantic_action_types: Mapping[str, int] | None = None,
    semantic_interaction_anomaly_count: int = 0,
) -> Dict[str, str]:
    matched = set(matched_bug_ids)
    attempted_semantics = attempted_semantic_action_types or {}
    reasons: Dict[str, str] = {}
    for bug in known_bugs:
        bug_id = _known_bug_id(bug)
        if not bug_id or bug_id in matched:
            continue
        bug_type = str(bug.get("type") or "")
        expected_semantic = _semantic_type_for_bug(bug)
        if bug_type == "api-forbidden" and api_403_count <= 0:
            reasons[bug_id] = "no network 403 evidence observed"
        elif bug_type in {"async-hang", "timeout-no-feedback"} and int(action_counts.get("fill_input", 0) or 0) == 0:
            reasons[bug_id] = "async/hang input target was not filled"
        elif bug_type == "cart-quantity-mismatch" and int(action_counts.get("inspect_cart", 0) or 0) == 0:
            reasons[bug_id] = "cart quantity path was not inspected"
        elif expected_semantic and int(attempted_semantics.get(expected_semantic, 0) or 0) > 0:
            if int(semantic_interaction_anomaly_count or 0) > 0:
                reasons[bug_id] = f"semantic action {expected_semantic} was clicked and produced interaction anomaly, but no offline known-bug match was assigned"
            else:
                reasons[bug_id] = f"semantic action {expected_semantic} was clicked, but no interaction anomaly was produced"
        elif bug_id not in attempted_bug_ids:
            reasons[bug_id] = "no catalog candidate/action reached this bug"
        else:
            reasons[bug_id] = "catalog path attempted but no matching anomaly evidence was produced"
    return reasons


def _missed_semantic_action_reasons(
    known_bugs: List[Mapping[str, Any]],
    matched_bug_ids: List[str],
    attempted_semantic_action_types: Mapping[str, int],
    semantic_interaction_anomaly_count: int,
) -> Dict[str, str]:
    return _missed_bug_reasons(
        known_bugs,
        matched_bug_ids,
        set(),
        {},
        0,
        attempted_semantic_action_types,
        semantic_interaction_anomaly_count,
    )


def _semantic_type_for_bug(bug: Mapping[str, Any]) -> str:
    text = " ".join(
        str(value or "")
        for value in (
            bug.get("type"),
            bug.get("symptom"),
            bug.get("expected_agent_behavior"),
            " ".join(str(item) for item in bug.get("target_keywords", []) or []),
            " ".join(str(item) for item in bug.get("action_hints", []) or []),
        )
    ).lower()
    if any(token in text for token in ("운동", "workout")) and any(token in text for token in ("추가", "add")):
        return "workout_add"
    if any(token in text for token in ("장바구니", "cart", "basket")):
        return "cart"
    if any(token in text for token in ("추가", "add")):
        return "add"
    if any(token in text for token in ("검색", "search")):
        return "search"
    if any(token in text for token in ("설정", "setting")):
        return "settings"
    if any(token in text for token in ("저장", "save")):
        return "save"
    if any(token in text for token in ("제출", "submit")):
        return "submit"
    return ""


def _site003_eval_fallback_action_id(
    action_space: ActionSpace,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
    action_id: int,
) -> int:
    site_id = _site_id_from_observation(observation)
    if site_id == "site001":
        return action_id
    decoded = action_space.decode(action_id)
    if site_id != "site9800" and decoded.get("action_type") == "inspect_cart":
        return action_space.encode("inspect_dom", 0)
    infra_action = _guided_infra_eval_action_id(action_space, observation, history)
    if infra_action is not None:
        return infra_action
    candidates = observation.get("candidate_elements", []) or []
    counts = history.get("action_type_counts", {})
    if _is_openended_observation(observation):
        consecutive_counts = history.get("consecutive_action_type_counts", {})
        click_count = int(counts.get("click_element", 0) or 0) if isinstance(counts, Mapping) else 0
        inspect_dom_count = int(counts.get("inspect_dom", 0) or 0) if isinstance(counts, Mapping) else 0
        if click_count == 0:
            index = _first_openended_candidate_index(observation, history)
            if index is not None:
                return action_space.encode("click_element", index)
        if inspect_dom_count >= 5 or (
            isinstance(consecutive_counts, Mapping)
            and int(consecutive_counts.get("inspect_dom", 0) or 0) >= 2
        ):
            index = _first_openended_candidate_index(observation, history)
            if index is not None:
                return action_space.encode("click_element", index)
            return action_space.encode("inspect_layout", 0)
        if decoded.get("action_type") == "inspect_dom" and inspect_dom_count >= 3:
            index = _first_openended_candidate_index(observation, history)
            if index is not None:
                return action_space.encode("click_element", index)
            return action_space.encode("inspect_layout", 0)
        if decoded.get("action_type") == "inspect_layout" and int(counts.get("inspect_layout", 0) or 0) >= 5:
            index = _first_openended_candidate_index(observation, history)
            if index is not None:
                return action_space.encode("click_element", index)
            if inspect_dom_count < 5:
                return action_space.encode("inspect_dom", 0)
            return action_space.encode("scroll_down", 0)
        if decoded.get("action_type") == "finish_episode" and click_count == 0:
            index = _first_openended_candidate_index(observation, history)
            if index is not None:
                return action_space.encode("click_element", index)
    if isinstance(counts, Mapping) and int(counts.get("click_element", 0) or 0) >= 30:
        if _has_layout_signal_safe(observation):
            return action_space.encode("inspect_layout", 0)
        return action_space.encode("inspect_dom", 0)
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    if (
        isinstance(counts, Mapping)
        and int(counts.get("inspect_dom", 0) or 0) == 0
        and int(counts.get("click_element", 0) or 0) >= 1
        and bool(page_state.get("has_empty_state_text"))
    ):
        return action_space.encode("inspect_dom", 0)
    if (
        isinstance(counts, Mapping)
        and int(counts.get("inspect_layout", 0) or 0) == 0
        and _has_layout_signal_safe(observation)
    ):
        return action_space.encode("inspect_layout", 0)
    if isinstance(candidates, list) and isinstance(counts, Mapping) and int(counts.get("click_element", 0) or 0) == 0:
        for index, candidate in enumerate(candidates[: action_space.max_candidates]):
            if (
                isinstance(candidate, Mapping)
                and candidate.get("is_workout_add_action")
                and candidate.get("clickable")
                and _safe_visibility(candidate) > 0.0
            ):
                return action_space.encode("click_element", index)
    return action_id


def _guided_infra_eval_action_id(
    action_space: ActionSpace,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> int | None:
    infra = observation.get("infra_signals", {}) if isinstance(observation, Mapping) else {}
    if not isinstance(infra, Mapping):
        return None
    try:
        port = int(infra.get("port") or 0)
    except (TypeError, ValueError):
        return None
    if not 9000 <= port <= 9100:
        return None
    counts = history.get("action_type_counts", {})
    if not isinstance(counts, Mapping):
        counts = {}
    for action_type in (
        "inspect_port_status",
        "inspect_server_health",
        "inspect_latency",
        "inspect_server_logs",
        "inspect_runtime_metrics",
    ):
        if int(counts.get(action_type, 0) or 0) == 0:
            return action_space.encode(action_type, 0)
    return None


def _is_openended_observation(observation: Mapping[str, Any]) -> bool:
    return _has_openended_interactive_candidate(observation)


def _has_layout_signal_safe(observation: Mapping[str, Any]) -> bool:
    signals = observation.get("layout_signals", {}) if isinstance(observation.get("layout_signals"), Mapping) else {}
    if int(signals.get("layout_overlap_count", 0) or 0) > 0 or int(signals.get("layout_overflow_count", 0) or 0) > 0:
        return True
    candidates = observation.get("candidate_elements", []) or []
    return any(
        isinstance(candidate, Mapping) and bool(candidate.get("is_chart_like") or candidate.get("is_chart_related"))
        for candidate in candidates
    )


def _layout_unique_key(anomaly: Mapping[str, Any]) -> tuple[str, str, str, str]:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    matched_or_catalog = str(anomaly.get("matched_bug_id") or "")
    if not matched_or_catalog:
        catalog_ids = evidence.get("catalog_bug_id_matches", []) or []
        if isinstance(catalog_ids, list) and catalog_ids:
            matched_or_catalog = str(catalog_ids[0])
    bbox = evidence.get("child_bbox", []) if isinstance(evidence.get("child_bbox"), list) else []
    rounded_bbox = ",".join(str(int(round(float(value or 0) / 10.0) * 10)) for value in bbox[:4])
    return (
        str(anomaly.get("type") or ""),
        matched_or_catalog,
        str(evidence.get("selector_hint") or evidence.get("selector") or evidence.get("data_bug_id") or ""),
        rounded_bbox or str(evidence.get("candidate_text") or "")[:80],
    )


def _layout_evidence_key(evidence: Mapping[str, Any]) -> tuple[str, str, str]:
    catalog_ids = evidence.get("catalog_bug_id_matches", []) if isinstance(evidence.get("catalog_bug_id_matches"), list) else []
    primary = str(catalog_ids[0]) if catalog_ids else str(evidence.get("matched_bug_id") or "")
    section = ",".join(str(item) for item in evidence.get("section_keyword_matches", []) or [])
    bbox = evidence.get("child_bbox", []) if isinstance(evidence.get("child_bbox"), list) else []
    rounded_bbox = ",".join(str(int(round(float(value or 0) / 10.0) * 10)) for value in bbox[:4])
    return primary, section, rounded_bbox or str(evidence.get("bbox_source") or "")


def _repeated_click_target_count(history: Mapping[str, Any]) -> int:
    counts = history.get("element_key_click_counts", history.get("click_target_counts", {}))
    if not isinstance(counts, Mapping):
        return 0
    return sum(max(0, int(value or 0) - 1) for value in counts.values())


def _repeated_element_key_click_count(history: Mapping[str, Any]) -> int:
    counts = history.get("element_key_click_counts", {})
    if not isinstance(counts, Mapping):
        return 0
    return sum(max(0, int(value or 0) - 1) for value in counts.values())


def _repeated_action_signature_count(history: Mapping[str, Any]) -> int:
    counts = history.get("action_signature_counts", {})
    if not isinstance(counts, Mapping):
        return 0
    return sum(max(0, int(value or 0) - 1) for value in counts.values())


def _repeated_action_type_count(action_counts: Mapping[str, Any]) -> int:
    if not isinstance(action_counts, Mapping):
        return 0
    return sum(max(0, int(value or 0) - 1) for value in action_counts.values())


def _matched_target_reclick_count(history: Mapping[str, Any]) -> int:
    counts = history.get("catalog_click_counts", {})
    matched = set(history.get("matched_bug_ids", set()) or set())
    if not isinstance(counts, Mapping) or not matched:
        return 0
    total = 0
    for key, value in counts.items():
        if any(bug_id in str(key) for bug_id in matched):
            total += max(0, int(value or 0) - 1)
    return total


def _record_unique_candidates(
    unique_candidates: Dict[tuple[str, str, str, str], Dict[str, Any]],
    matched_by_bug_id: Dict[str, Dict[str, Any]],
    suppressed_duplicates: Dict[tuple[str, str, str, str], Dict[str, Any]],
    anomalies: List[Dict[str, Any]],
) -> None:
    for anomaly in anomalies:
        confidence = float(anomaly.get("confidence", 0.0) or 0.0)
        if confidence < 0.6 and not anomaly.get("matched_bug_id") and not anomaly.get("human_review_status"):
            continue
        evidence = anomaly.get("evidence", {})
        matched_bug_id = str(anomaly.get("matched_bug_id") or "")
        if matched_bug_id:
            previous = matched_by_bug_id.get(matched_bug_id)
            if previous is None or confidence > float(previous.get("confidence", 0.0) or 0.0):
                matched_by_bug_id[matched_bug_id] = anomaly
        key = _canonical_anomaly_key(anomaly)
        previous = unique_candidates.get(key)
        if previous is None or _anomaly_quality(anomaly) > _anomaly_quality(previous):
            unique_candidates[key] = anomaly
            if previous is not None:
                suppressed_duplicates[key] = previous
        else:
            suppressed_duplicates[key] = anomaly


def _canonical_anomaly_key(anomaly: Mapping[str, Any]) -> tuple[str, str, str, str]:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    matched_bug_id = str(anomaly.get("matched_bug_id") or "")
    catalog_ids = evidence.get("catalog_bug_id_matches", []) if isinstance(evidence.get("catalog_bug_id_matches"), list) else []
    primary_catalog = str(catalog_ids[0]) if catalog_ids else ""
    anomaly_type = str(anomaly.get("type") or "")
    if matched_bug_id:
        return ("matched", anomaly_type, matched_bug_id, "")
    if anomaly_type == "layout-overflow":
        section = ",".join(str(item) for item in evidence.get("section_keyword_matches", []) or [])
        return ("layout", primary_catalog, section, "")
    if anomaly_type == "empty-state-rendering":
        section = ",".join(str(item) for item in evidence.get("section_keyword_matches", []) or [])
        return ("empty", primary_catalog, section, "")
    if anomaly_type == "button-no-response":
        target = str(evidence.get("clicked_data_bug_id") or evidence.get("clicked_text") or evidence.get("clicked_name") or _target_bid(evidence))
        return ("button", primary_catalog, _normalize_text(target), "")
    selector = str(evidence.get("selector_hint") or evidence.get("selector") or evidence.get("data_bug_id") or "")
    bbox = evidence.get("child_bbox") or evidence.get("bbox") or []
    rounded_bbox = ",".join(str(int(round(float(value or 0) / 10.0) * 10)) for value in bbox[:4]) if isinstance(bbox, list) else ""
    text = _normalize_text(str(evidence.get("candidate_text") or evidence.get("clicked_text") or _target_bid(evidence)))[:80]
    return (anomaly_type, primary_catalog, selector or text, rounded_bbox)


def _anomaly_quality(anomaly: Mapping[str, Any]) -> tuple[int, float, int]:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    matched = 1 if anomaly.get("matched_bug_id") else 0
    confidence = float(anomaly.get("confidence", 0.0) or 0.0)
    richness = sum(
        1
        for key in (
            "selector_hint",
            "selector",
            "data_bug_id",
            "clicked_data_bug_id",
            "child_bbox",
            "parent_bbox",
            "catalog_bug_id_matches",
            "section_keyword_matches",
            "target_keyword_matches",
        )
        if evidence.get(key)
    )
    return matched, confidence, richness


def _classify_candidates(candidates: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    confirmed: List[Dict[str, Any]] = []
    catalog_related: List[Dict[str, Any]] = []
    exploratory: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        evidence = row.get("evidence", {}) if isinstance(row.get("evidence"), Mapping) else {}
        row["classification"] = "confirmed_known_bug" if row.get("matched_bug_id") else (
            "catalog_related_anomaly" if _is_catalog_related(evidence) else "exploratory_anomaly"
        )
        if row["classification"] == "exploratory_anomaly":
            _ensure_human_review_fields(row)
        if row.get("matched_bug_id"):
            confirmed.append(row)
        elif _is_catalog_related(evidence):
            catalog_related.append(row)
        else:
            exploratory.append(row)
    return confirmed, catalog_related, exploratory


def _ensure_human_review_fields(row: Dict[str, Any]) -> None:
    if row.get("human_review_status") and row.get("review_question"):
        return
    evidence = row.get("evidence", {}) if isinstance(row.get("evidence"), Mapping) else {}
    anomaly_type = str(row.get("type") or "")
    before_url = str(evidence.get("before_url") or "")
    after_url = str(evidence.get("after_url") or "")
    route_changed = bool(evidence.get("route_changed")) or bool(before_url and after_url and before_url != after_url)
    cart_changed = bool(evidence.get("cart_state_changed")) or (
        evidence.get("cart_text_before") is not None
        and evidence.get("cart_text_after") is not None
        and evidence.get("cart_text_before") != evidence.get("cart_text_after")
    )
    target = _review_target_text(evidence)
    lower_target = target.lower()
    if anomaly_type == "button-no-response" and (route_changed or cart_changed):
        row["human_review_status"] = "likely_false_positive"
        row["likely_false_positive_reason"] = "route or cart state changed after click"
        row["review_question"] = f"{target} changed route or cart state; verify whether this should be removed as a no-response false positive."
    elif anomaly_type == "button-no-response" and "cart" in lower_target and before_url.rstrip("/").endswith("/cart") and after_url.rstrip("/").endswith("/cart"):
        row["human_review_status"] = "likely_false_positive"
        row["likely_false_positive_reason"] = "already on cart page; same-page navigation click may be an expected no-op"
        row["review_question"] = "Already on cart page. Should clicking Cart again trigger any UI change?"
    elif "hang" in lower_target or "async" in lower_target:
        row["human_review_status"] = "likely_true_positive"
        row["review_question"] = f"{target} includes async/hang wording; reproduce input/submit flow and check for missing feedback or stuck UI."
    elif anomaly_type == "duplicated-rendering":
        if evidence.get("axtree_or_text_only") and not evidence.get("visible_duplicate_candidates"):
            row["human_review_status"] = "likely_false_positive"
            row["likely_false_positive_reason"] = "duplicate was observed only in text/AXTree and may be normal repeated UI"
        else:
            row["human_review_status"] = "needs_review"
        row["review_question"] = "Verify whether duplicated text/components are visible in the same parent/container, not repeated once per product card."
    elif anomaly_type == "form-no-feedback" and "checkout" in lower_target:
        if evidence.get("validation_message_visible") or evidence.get("error_message_visible"):
            row["human_review_status"] = "likely_false_positive"
            row["likely_false_positive_reason"] = "checkout showed validation or error feedback"
        else:
            row["human_review_status"] = "needs_review"
        row["review_question"] = "Should Checkout advance to the next step, or is the visible validation/error feedback the expected response?"
    elif anomaly_type == "form-no-feedback" and "search" in lower_target:
        if evidence.get("error_message_visible") or evidence.get("empty_state_visible"):
            row["human_review_status"] = "likely_false_positive"
            row["likely_false_positive_reason"] = "search showed error or empty-state feedback"
        elif not str(evidence.get("query_text") or "").strip():
            row["human_review_status"] = "likely_false_positive"
            row["likely_false_positive_reason"] = "search was evaluated without query text"
        else:
            row["human_review_status"] = "needs_review"
        row["review_question"] = "After entering a query, should Search update results or show validation/no-result feedback?"
    else:
        row["human_review_status"] = "needs_review"
        row["review_question"] = f"Verify whether {target or anomaly_type} is a real visible UI failure or an expected no-op."


def _review_target_text(evidence: Mapping[str, Any]) -> str:
    target = evidence.get("target")
    if isinstance(target, Mapping):
        for key in ("name", "text", "role", "bid"):
            if target.get(key):
                return str(target.get(key))
    for key in ("clicked_text", "clicked_name", "candidate_text", "selector_hint", "data_bug_id"):
        if evidence.get(key):
            return str(evidence.get(key))
    return ""


def _review_status_counts(anomalies: List[Dict[str, Any]]) -> Counter[str]:
    return Counter(str(item.get("human_review_status") or "needs_review") for item in anomalies)


def _write_openended_review_artifacts(
    site_id: str,
    site: Mapping[str, Any],
    result: Mapping[str, Any],
    exploratory_anomalies: List[Dict[str, Any]],
    anomaly_types: Mapping[str, Any],
) -> Dict[str, str]:
    if site_id != "site9800":
        return {}
    docs_path = Path("docs/site9800_openended_review.md")
    catalog_path = Path("datasets/site9800/bug_catalog.candidates.json")
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(
        _build_site9800_review_markdown(site_id, site, result, exploratory_anomalies, anomaly_types),
        encoding="utf-8",
    )
    catalog_path.write_text(
        json.dumps(
            _build_site9800_candidate_catalog(site_id, site, exploratory_anomalies),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "review_doc_path": str(docs_path),
        "candidate_catalog_path": str(catalog_path),
    }


def _build_site9800_review_markdown(
    site_id: str,
    site: Mapping[str, Any],
    result: Mapping[str, Any],
    exploratory_anomalies: List[Dict[str, Any]],
    anomaly_types: Mapping[str, Any],
) -> str:
    lines = [
        "# site9800 Open-ended Anomaly Review",
        "",
        "This document is generated from the latest open-ended BrowserGym evaluation. It is a review aid, not a confirmed bug catalog.",
        "",
        "## Summary",
        "",
        f"- site_id: `{site_id}`",
        f"- base_url: `{site.get('base_url')}`",
        f"- evaluation_mode: `{result.get('evaluation_mode')}`",
        f"- action_counts: `{json.dumps(result.get('action_counts', {}), ensure_ascii=False)}`",
        f"- anomaly_types: `{json.dumps(dict(anomaly_types), ensure_ascii=False)}`",
        f"- unique_detected_candidates: `{result.get('unique_detected_candidates')}`",
        f"- total_detected_candidates: `{result.get('total_detected_candidates')}`",
        "",
        "## Candidate Review Table",
        "",
        "| Candidate ID | Type | Target | URL | Confidence | Human Review Status | Evidence Summary | Review Question | Suggested Decision | Bug Catalog Candidate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, anomaly in enumerate(exploratory_anomalies, start=1):
        evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
        status = str(anomaly.get("human_review_status") or "needs_review")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"C{index}",
                    _md(str(anomaly.get("type") or "")),
                    _md(_review_target_text(evidence)),
                    _md(_review_url(evidence)),
                    f"{float(anomaly.get('confidence', 0.0) or 0.0):.2f}",
                    _md(status),
                    _md(_evidence_summary(anomaly)),
                    _md(str(anomaly.get("review_question") or "")),
                    _md(_suggested_decision(anomaly)),
                    _md("yes" if _candidate_should_be_promoted(anomaly) else "review/low-priority"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Review Instructions",
            "",
            "1. Reproduce each candidate in a real browser against `http://localhost:9800`.",
            "2. Confirm whether the issue is visible in the UI and not only in AXTree/page text.",
            "3. Treat same-page navigation clicks, empty-input clicks, and clicks that change URL/cart state as likely false positives unless the product spec says otherwise.",
            "4. Promote only confirmed true positives from `datasets/site9800/bug_catalog.candidates.json` to `datasets/site9800/bug_catalog.json`.",
            "5. After promotion, set `has_bug_catalog` to `true` and add `bug_catalog_path` in `configs/training_sites.json`.",
            "",
            "## True Positive Criteria",
            "",
            "- Expected UI, URL, toast, modal, result list, or validation change is clear and does not occur.",
            "- Validation feedback is absent when required user input is missing.",
            "- Duplicated text/components are visible in the rendered UI.",
            "- Async/hang behavior is reproducible with unresolved loading, timeout, or missing feedback.",
            "",
            "## False Positive Criteria",
            "",
            "- Already on the target page and clicking the current nav item is an expected no-op.",
            "- Empty input submission is intentionally ignored.",
            "- Duplicate only appears in AXTree/page_text but not the visible UI.",
            "- Route, page text, modal, form, toast, cart state, or candidate set changed after the click.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_site9800_candidate_catalog(
    site_id: str,
    site: Mapping[str, Any],
    exploratory_anomalies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    bugs = []
    for index, anomaly in enumerate(exploratory_anomalies, start=1):
        evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
        target = _review_target_text(evidence)
        status = str(anomaly.get("human_review_status") or "needs_review")
        bugs.append(
            {
                "bug_id": f"{site_id}-bug-candidate-{index:02d}",
                "site_id": site_id,
                "category": "frontend",
                "type": _candidate_catalog_type(anomaly),
                "severity": _candidate_severity(anomaly),
                "selector": "",
                "target_keywords": _candidate_keywords(target, anomaly),
                "section_keywords": _candidate_section_keywords(evidence),
                "negative_keywords": [],
                "expected_elements": _candidate_expected_elements(anomaly),
                "expected_change": _candidate_expected_change(anomaly),
                "anomaly_evidence_keywords": _candidate_evidence_keywords(anomaly),
                "action_hints": ["inspect_dom"] if anomaly.get("type") == "duplicated-rendering" else ["click"],
                "screen_location": _review_url(evidence),
                "symptom": _candidate_symptom(anomaly),
                "expected_agent_behavior": str(anomaly.get("review_question") or ""),
                "status": "candidate_requires_human_verification",
                "human_review_status": status,
                "suggested_decision": _suggested_decision(anomaly),
                "likely_false_positive_reason": anomaly.get("likely_false_positive_reason", ""),
                "source_evidence": evidence,
            }
        )
    return {
        "site_id": site_id,
        "site_name": "SmartCommerce Global",
        "base_url": str(site.get("base_url") or "http://localhost:9800"),
        "catalog_status": "candidate_requires_human_verification",
        "note": "This is not a confirmed known-bug catalog. Promote only human-confirmed true positives to bug_catalog.json.",
        "bugs": bugs,
    }


def _candidate_catalog_type(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    target = _review_target_text(evidence).lower()
    if "hang" in target or "async" in target:
        return "async-hang"
    return str(anomaly.get("type") or "unknown")


def _candidate_severity(anomaly: Mapping[str, Any]) -> str:
    status = str(anomaly.get("human_review_status") or "")
    anomaly_type = str(anomaly.get("type") or "")
    if status == "likely_true_positive":
        return "Medium"
    if anomaly_type == "duplicated-rendering":
        return "Low"
    return "Medium" if anomaly_type == "form-no-feedback" else "Low"


def _candidate_keywords(target: str, anomaly: Mapping[str, Any]) -> List[str]:
    words = [part.strip() for part in str(target or "").replace("/", " ").split() if part.strip()]
    lowered = sorted({word.lower() for word in words})
    return sorted(set(words + lowered))[:12]


def _candidate_section_keywords(evidence: Mapping[str, Any]) -> List[str]:
    values = []
    for key in ("before_url", "after_url", "checkout_step"):
        if evidence.get(key):
            values.append(str(evidence.get(key)))
    return values[:8]


def _candidate_expected_elements(anomaly: Mapping[str, Any]) -> List[str]:
    anomaly_type = str(anomaly.get("type") or "")
    if anomaly_type == "duplicated-rendering":
        return ["single visible component instance"]
    return ["validation message", "toast", "route change", "result update"]


def _candidate_expected_change(anomaly: Mapping[str, Any]) -> List[str]:
    anomaly_type = str(anomaly.get("type") or "")
    if anomaly_type == "duplicated-rendering":
        return []
    return ["route_change", "validation_message", "toast_visible", "page_text_change", "result_update"]


def _candidate_evidence_keywords(anomaly: Mapping[str, Any]) -> List[str]:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    keywords = [str(anomaly.get("type") or "")]
    for key in ("clicked_text", "clicked_name", "route_changed", "page_text_delta", "candidate_delta", "checkout_step"):
        if evidence.get(key) is not None:
            keywords.append(f"{key}:{evidence.get(key)}")
    return keywords[:16]


def _candidate_symptom(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    target = _review_target_text(evidence)
    if anomaly.get("type") == "duplicated-rendering":
        return f"{target or 'UI text'} appears duplicated in open-ended observation; visible UI confirmation required."
    return f"{target or 'Interactive target'} produced no route, text, validation, toast, modal, or result feedback in open-ended evaluation."


def _review_url(evidence: Mapping[str, Any]) -> str:
    before = str(evidence.get("before_url") or "")
    after = str(evidence.get("after_url") or "")
    if before and after and before != after:
        return f"{before} -> {after}"
    return before or after


def _evidence_summary(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    if anomaly.get("type") == "duplicated-rendering":
        return f"duplicated_titles={evidence.get('duplicated_titles', {})}"
    parts = []
    for key in (
        "page_text_delta",
        "candidate_delta",
        "route_changed",
        "cart_state_changed",
        "query_text",
        "checkout_step",
        "validation_message_visible",
        "async_keyword_match",
    ):
        if key in evidence:
            parts.append(f"{key}={evidence.get(key)}")
    return ", ".join(parts)


def _suggested_decision(anomaly: Mapping[str, Any]) -> str:
    status = str(anomaly.get("human_review_status") or "")
    if status == "likely_true_positive":
        return "high_priority_review"
    if status == "likely_false_positive":
        return "likely_false_positive"
    if anomaly.get("type") == "duplicated-rendering":
        return "verify_visible_ui_before_promotion"
    return "needs_human_review"


def _candidate_should_be_promoted(anomaly: Mapping[str, Any]) -> bool:
    return str(anomaly.get("human_review_status") or "") == "likely_true_positive"


def _md(value: str) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:500]


def _is_catalog_related(evidence: Mapping[str, Any]) -> bool:
    return bool(
        evidence.get("catalog_bug_id_matches")
        or evidence.get("catalog_keyword_matches")
        or evidence.get("section_keyword_matches")
        or evidence.get("target_keyword_matches")
        or evidence.get("selector_hint")
        or evidence.get("data_bug_id")
    )


def _uses_bbox_fallback(evidence: Mapping[str, Any]) -> bool:
    return str(evidence.get("bbox_source") or "") in {"viewport_fallback", "parent_fallback", "section_fallback"} or not evidence.get("child_bbox")


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("element_key")
        or candidate.get("bid")
        or candidate.get("name")
        or candidate.get("text")
        or candidate.get("selector_hint")
        or ""
    )


def _action_diversity_score(action_counts: Mapping[str, Any]) -> float:
    main_actions = {"click_element", "inspect_layout", "inspect_dom", "change_viewport_mobile"}
    used = sum(1 for action_type in main_actions if int(action_counts.get(action_type, 0) or 0) > 0)
    return round(used / len(main_actions), 4)


def _repeated_action_rate_from_action_counts(action_counts: Mapping[str, Any]) -> float:
    total = 0
    repeated = 0
    for count in action_counts.values():
        value = int(count or 0)
        total += value
        if value > 1:
            repeated += value - 1
    return round(repeated / total, 4) if total else 0.0


def _repeated_action_rate_from_results(results: Any) -> float:
    combined: Counter[str] = Counter()
    for result in results:
        if isinstance(result, Mapping) and isinstance(result.get("action_counts"), Mapping):
            combined.update({str(key): int(value or 0) for key, value in result["action_counts"].items()})
    return _repeated_action_rate_from_action_counts(combined)


def _safe_rate(numerator: Any, denominator: Any) -> float:
    try:
        denom = int(denominator or 0)
        if denom <= 0:
            return 0.0
        return round(int(numerator or 0) / denom, 4)
    except (TypeError, ValueError):
        return 0.0


def _repeated_click_target_rate_from_results(results: Any) -> float:
    repeated = 0
    clicks = 0
    for result in results:
        if not isinstance(result, Mapping):
            continue
        repeated += int(result.get("repeated_click_target_count", 0) or 0)
        clicks += int(result.get("click_element_count", result.get("action_counts", {}).get("click_element", 0) if isinstance(result.get("action_counts"), Mapping) else 0) or 0)
    return _safe_rate(repeated, clicks)


def _mean_memory_state_norm_from_results(results: Any) -> float:
    values = [
        float(result.get("memory_state_norm_mean", 0.0) or 0.0)
        for result in results
        if isinstance(result, Mapping) and float(result.get("memory_state_norm_mean", 0.0) or 0.0) > 0.0
    ]
    return float(np.mean(values)) if values else 0.0


def _observed_signal_types(signal_summary: Mapping[str, Any]) -> List[str]:
    observed = []
    if int(signal_summary.get("multi_signal_anomaly_count", 0) or 0):
        observed.extend(["ui-dom", "interaction", "layout"])
    if int(signal_summary.get("console_error_count", 0) or 0):
        observed.append("console-error")
    if int(signal_summary.get("runtime_exception_count", 0) or 0):
        observed.append("runtime-exception")
    if int(signal_summary.get("network_request_failed_count", 0) or 0):
        observed.append("network-error")
    if int(signal_summary.get("api_4xx_count", 0) or 0):
        observed.append("api-4xx")
    if int(signal_summary.get("api_5xx_count", 0) or 0):
        observed.append("api-5xx")
    if int(signal_summary.get("api_timeout_count", 0) or 0):
        observed.append("api-timeout")
    if int(signal_summary.get("api_ui_mismatch_count", 0) or 0):
        observed.append("api-ui-mismatch")
    if int(signal_summary.get("security_signal_count", 0) or 0):
        observed.extend(["auth-permission-anomaly", "sensitive-data-exposure", "token-exposure"])
    return sorted(set(observed))


def _signal_type_coverage(targets: Any, observed: Any) -> float:
    target_set = {str(item) for item in targets or [] if item}
    observed_set = {str(item) for item in observed or [] if item}
    if not target_set:
        return round(len(observed_set) / 5.0, 4) if observed_set else 0.0
    return round(len(target_set & observed_set) / len(target_set), 4)


def _signal_discovery_score(signal_summary: Mapping[str, Any]) -> float:
    return _signal_type_coverage([], _observed_signal_types(signal_summary))



def _load_known_bugs_for_site(site: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not bool(site.get("has_bug_catalog")):
        return []
    path = site.get("bug_catalog_path")
    if path:
        raw = json.loads(Path(str(path)).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [dict(item) for item in raw if isinstance(item, Mapping)]
        if isinstance(raw, Mapping):
            bugs = raw.get("bugs") or raw.get("items") or raw.get("bug_catalog") or raw.get("known_bugs") or []
            if isinstance(bugs, list):
                return [dict(item) for item in bugs if isinstance(item, Mapping)]
    return load_known_bugs(str(site.get("site_id") or ""))


def _reward_scale(site: Mapping[str, Any]) -> float:
    value = site.get("reward_scale")
    if value is None:
        site_id = str(site.get("site_id") or "")
        return 0.5 if site_id == "site001" else 1.2 if site_id == "site003" else 1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _clamp_reward(value: float) -> float:
    return max(-1.0, min(1.5, float(value)))


def _read_json_dict(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object config: {path}")
    return data


def _run_id_from_output(output: str) -> str:
    if not output:
        return ""
    path = Path(output)
    parent = path.parent
    if parent.name and parent.name not in {".", "evaluations", "artifacts"}:
        return parent.name
    return ""


def _validate_sites(raw_sites: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_sites, list) or not raw_sites:
        raise ValueError("Config requires a non-empty sites list.")
    return [dict(site) for site in raw_sites if isinstance(site, Mapping) and site.get("enabled") is not False]


def _apply_evaluation_defaults_to_sites(sites: List[Dict[str, Any]], config: Mapping[str, Any]) -> None:
    for site in sites:
        for key in (
            "reward_mode",
            "use_known_bug_reward",
            "use_known_bug_for_training",
            "use_known_bug_for_evaluation",
            "training_uses_site_specific_bug_catalog",
        ):
            if key not in site and key in config:
                site[key] = config[key]


def _configured_sites(raw_sites: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_sites, list):
        return []
    return [dict(site) for site in raw_sites if isinstance(site, Mapping)]


def _site_config_summary(configured_sites: List[Mapping[str, Any]], enabled_sites: List[Mapping[str, Any]]) -> Dict[str, Any]:
    configured_site_ids = [str(site.get("site_id") or "") for site in configured_sites if site.get("site_id")]
    enabled_site_ids = [str(site.get("site_id") or "") for site in enabled_sites if site.get("site_id")]
    disabled_site_ids = [
        str(site.get("site_id") or "")
        for site in configured_sites
        if site.get("enabled") is False and site.get("site_id")
    ]
    return {
        "configured_site_ids": configured_site_ids,
        "enabled_site_ids": enabled_site_ids,
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
    }


def _known_bug_id(bug: Mapping[str, Any]) -> str:
    return str(bug.get("id") or bug.get("bug_id") or bug.get("name") or "")


def _failed_site_result(
    site: Mapping[str, Any],
    exc: Exception,
    *,
    episodes: int,
    completed_episodes: int,
    reset_timeout_ms: int = 0,
    reset_retry_count: int = 0,
) -> Dict[str, Any]:
    error_type = exc.__class__.__name__
    error_message = str(exc)
    known_bug_ids = sorted(_known_bug_id(bug) for bug in _load_known_bugs_for_site(site) if _known_bug_id(bug))
    return {
        "status": "failed",
        "error_type": error_type,
        "error_message": error_message,
        "requested_episodes": episodes,
        "completed_episodes": completed_episodes,
        "valid_completed_episodes": 0,
        "valid_for_comparison": False,
        "excluded_from_comparison_reason": "timeout" if isinstance(exc, TimeoutError) or "timeout" in error_message.lower() else "failed",
        "partial": True,
        "avg_step_elapsed_ms": 0,
        "max_step_elapsed_ms": 0,
        "reset_elapsed_ms": 0,
        "reset_timeout_ms": reset_timeout_ms,
        "reset_retry_count": reset_retry_count,
        "reset_retry_used": 0,
        "reset_failure_count": 1 if "reset" in error_message.lower() or isinstance(exc, TimeoutError) else 0,
        "reset_failure_reason": error_message if "reset" in error_message.lower() or isinstance(exc, TimeoutError) else "",
        "site_elapsed_ms": 0,
        "episode_elapsed_ms": 0,
        "early_stop_count": 1 if isinstance(exc, TimeoutError) else 0,
        "early_stop_reasons": {"timeout": 1} if isinstance(exc, TimeoutError) else {},
        "timeout_count": 1 if isinstance(exc, TimeoutError) else 0,
        "known_bug_catalog_available": bool(site.get("has_bug_catalog")),
        "requested_base_url": str(site.get("base_url") or ""),
        "base_url": str(site.get("base_url") or ""),
        "evaluation_mode": "known_bug_catalog"
        if bool(site.get("use_known_bug_for_evaluation", False)) and bool(site.get("has_bug_catalog"))
        else "signal_discovery",
        "episodes": episodes,
        "episode_count": completed_episodes,
        "average_reward": 0.0,
        "raw_average_reward": 0.0,
        "scaled_average_reward": 0.0,
        "detected_bug_count": 0,
        "raw_detected_candidates": 0,
        "deduped_detected_candidates": 0,
        "total_detected_candidates": 0,
        "unique_detected_candidates": 0,
        "known_bug_match_count": 0,
        "matched_bug_ids": [],
        "missed_bug_ids": known_bug_ids,
        "known_bug_total": len(known_bug_ids),
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "action_counts": {},
        "target_signal_types": list(site.get("target_signal_types") or []),
        "observed_signal_types": [],
        "missing_signal_types": list(site.get("target_signal_types") or []),
        "signal_type_coverage": 0.0,
        "signal_discovery_score": 0.0,
        "signal_delta_reward_total": 0.0,
        "inspect_action_reward_total": 0.0,
        "functional_action_signal_reward_total": 0.0,
        "signal_collector_enabled": True,
        "playwright_console_listener_enabled": False,
        "playwright_pageerror_listener_enabled": False,
        "playwright_network_listener_enabled": False,
        "warnings": [f"{error_type}: {error_message}"],
    }


def _int_arg(value: int | None, config: Mapping[str, Any], key: str, fallback: int) -> int:
    if value is not None:
        return int(value)
    try:
        return int(config.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _bool_arg(value: bool | None, config: Mapping[str, Any], key: str, fallback: bool) -> bool:
    if value is not None:
        return bool(value)
    if key not in config:
        return fallback
    return _parse_bool(config.get(key))


def _float_arg(value: float | None, config: Mapping[str, Any], key: str, fallback: float) -> float:
    if value is not None:
        return float(value)
    try:
        return float(config.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _str_arg(value: str | None, config: Mapping[str, Any], key: str, fallback: str) -> str:
    if value:
        return str(value)
    if config.get(key):
        return str(config.get(key))
    return fallback


def _eval_episodes_arg(value: int | None, config: Mapping[str, Any], fallback: int) -> int:
    if value is not None:
        return int(value)
    for key in ("eval_episodes", "episodes"):
        if key not in config:
            continue
        try:
            return int(config.get(key, fallback))
        except (TypeError, ValueError):
            return fallback
    return fallback


def _avg_int(values: List[int]) -> int:
    return int(sum(values) / len(values)) if values else 0


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
