from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

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
from services.browsergym_training_service import (
    _enrich_action,
    _first_openended_candidate_index,
    _guided_action_id,
    _has_openended_interactive_candidate,
    _target_bid,
    _update_history,
)
from services.known_bug_matcher import load_known_bugs, match_anomalies_to_known_bugs
from services.site_profile_service import build_site_profile, validate_site_identity


def _emit_event(**payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


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
    parser.add_argument("--model-path", default="artifacts/models/jaws_browsergym_shared_ppo.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-site-validation", type=_parse_bool, default=False)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--strict-metrics", type=_parse_bool, default=False)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = _read_json_dict(Path(args.config))
    sites = _validate_sites(config.get("sites", []))
    encoder = ObservationEncoder()
    action_space = ActionSpace()
    agent = PPOAgent(encoder.get_obs_dim(), action_space.get_action_dim())
    agent.load(args.model_path)

    _emit_event(
        event="scan_started",
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        site_count=len(sites),
    )

    site_results = {}
    rewards = []
    for site in sites:
        result = _evaluate_site(
            site,
            agent,
            encoder,
            action_space,
            args.episodes,
            args.max_steps,
            args.headless,
            args.strict_site_validation,
        )
        site_results[site["site_id"]] = result
        rewards.append(result["average_reward"])

    output = {
        "model_path": args.model_path,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "sites": site_results,
    }
    _emit_event(event="scan_completed", sites_evaluated=len(site_results))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.min_recall is not None:
        failed = [
            site_id for site_id, result in site_results.items()
            if result.get("recall") is not None and float(result.get("recall") or 0.0) < args.min_recall
        ]
        if failed and args.strict_metrics:
            return 1
    return 0


def _evaluate_site(
    site: Mapping[str, Any],
    agent: PPOAgent,
    encoder: ObservationEncoder,
    action_space: ActionSpace,
    episodes: int,
    max_steps: int,
    headless: bool,
    strict_site_validation: bool = False,
) -> Dict[str, Any]:
    site_id = str(site["site_id"])
    _emit_event(event="site_started", site_id=site_id, base_url=str(site.get("base_url") or ""))
    known_bugs = _load_known_bugs_for_site(site)
    site_profile = build_site_profile(site_id, known_bugs, exploration_profile=site.get("exploration_profile"))
    has_catalog = bool(site.get("has_bug_catalog")) and bool(known_bugs)
    episode_rewards: List[float] = []
    scaled_episode_rewards: List[float] = []
    total_detected_candidates = 0
    unique_candidates: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    matched_by_bug_id: Dict[str, Dict[str, Any]] = {}
    suppressed_duplicates: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    action_counts: Counter[str] = Counter()
    repeated_action_count = 0
    warnings: List[str] = []
    workout_add_candidate_count = 0
    workout_add_click_count = 0
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

    for episode_index in range(1, episodes + 1):
        _emit_event(event="episode_started", site_id=site_id, episode=episode_index)
        env = BrowserGymJAWSEnv(
            site_id=site_id,
            base_url=str(site["base_url"]),
            max_steps=max_steps,
            headless=headless,
            site_profile=site_profile,
            requires_login=bool(site.get("requires_login")),
            login_config=site.get("login") if isinstance(site.get("login"), Mapping) else None,
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
        }
        try:
            observation, reset_info = env.reset()
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
            for step_index in range(max_steps):
                obs_vector = encoder.encode_observation(observation)
                action_mask = action_space.build_action_mask(observation)
                action_mask, fallback_warning, repeated = _apply_eval_fallback_mask(
                    action_space,
                    action_mask,
                    observation,
                    history,
                )
                if fallback_warning and fallback_warning not in warnings:
                    warnings.append(fallback_warning)
                    _emit_event(event="warning", site_id=site_id, episode=episode_index, message=fallback_warning)
                repeated_action_count += repeated
                selected = agent.select_greedy_action(obs_vector, action_mask)
                if site_id == "site003":
                    action_id = selected["action_id"]
                else:
                    action_id = _guided_action_id(action_space, observation, history, selected["action_id"])
                action_id = _site003_eval_fallback_action_id(action_space, observation, history, action_id)
                action = action_space.decode(action_id)
                action["action_id"] = action_id
                action["site_id"] = site_id
                _enrich_action(action, observation)
                _emit_event(
                    event="action",
                    site_id=site_id,
                    episode=episode_index,
                    step=step_index + 1,
                    action=action.get("action_type"),
                    target=_action_target_text(action),
                    target_text=_action_target_text(action),
                )
                if action.get("action_type") == "click_element":
                    if first_click_step is None:
                        first_click_step = step_index + 1
                    if action.get("clicked_bid") or action.get("clicked_text"):
                        unique_clicked_targets.add(str(action.get("clicked_bid") or action.get("clicked_text")))
                for bug_id in action.get("catalog_bug_id_matches", []) or []:
                    attempted_bug_ids.add(str(bug_id))
                if action.get("action_type") == "inspect_dom" and history.get("last_action_type") == "inspect_dom":
                    inspect_dom_repeat_count += 1
                next_observation, _, done, step_info = env.step(action_id)
                action["failed"] = bool(step_info.get("last_action_error"))
                anomalies = detect_anomalies(
                    observation,
                    next_observation,
                    {"action": action, "site_profile": site_profile, **step_info},
                    site_profile=site_profile,
                )
                matches = match_anomalies_to_known_bugs(anomalies, known_bugs, site_id=site_id)
                action["matched_bug_ids"] = [
                    str(match.get("matched_bug_id")) for match in matches if match.get("matched_bug_id")
                ]
                reward, _ = calculate_autonomous_reward(
                    observation,
                    next_observation,
                    action,
                    anomalies,
                    matches,
                    history,
                    site_profile,
                )
                reward_total += reward
                scaled_reward_total += _clamp_reward(reward * reward_scale)
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
                empty_state_detected_count += sum(1 for anomaly in anomalies if anomaly.get("type") == "empty-state-rendering")
                button_no_response_candidates += sum(1 for anomaly in anomalies if anomaly.get("type") == "button-no-response")
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
                _update_history(history, observation, action, anomalies)
                observation = next_observation
                if done:
                    break
        except Exception as exc:
            _emit_event(event="warning", site_id=site_id, episode=episode_index, message=f"episode exception: {exc}")
            print(f"[multisite-eval] {site_id} episode {episode_index} exception: {exc}", file=sys.stderr, flush=True)
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
    result = {
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
        "evaluation_mode": "known_bug_catalog" if has_catalog else "openended_anomaly_discovery",
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
        "catalog_guided_action_count": sum(
            int(action_counts.get(action_name, 0))
            for action_name in ("click_element", "fill_input", "press_enter", "inspect_network", "inspect_console", "inspect_cart")
        ),
        "fill_input_count": int(action_counts.get("fill_input", 0)),
        "inspect_network_count": int(action_counts.get("inspect_network", 0)),
        "inspect_console_count": int(action_counts.get("inspect_console", 0)),
        "api_403_count": api_403_count,
        "async_hang_candidates": async_hang_candidates,
        "cart_quantity_mismatch_candidates": cart_quantity_mismatch_candidates,
        "sparse_data_candidates": sparse_data_candidates,
        "triggered_bug_ids": sorted(triggered_bug_ids),
        "attempted_bug_ids": sorted(attempted_bug_ids),
        "missed_bug_reason": _missed_bug_reasons(known_bugs, matched_bug_ids, attempted_bug_ids, action_counts, api_403_count),
        "openended_interactive_candidate_count": openended_interactive_candidate_count,
        "openended_keyword_match_count": openended_keyword_match_count,
        "first_click_step": first_click_step,
        "unique_clicked_targets": sorted(unique_clicked_targets),
        "inspected_dom_count": int(action_counts.get("inspect_dom", 0)),
        "inspect_dom_repeat_count": inspect_dom_repeat_count,
        "layout_anomaly_count": layout_anomaly_count,
        "interaction_anomaly_count": interaction_anomaly_count,
        "anomaly_types": anomaly_types,
        "top_exploratory_anomalies": exploratory_anomalies[:5],
        "false_positive_filtered_count": _review_status_counts(exploratory_anomalies).get("likely_false_positive", 0),
        "action_diversity_score": action_diversity_score,
        "repeated_action_count": repeated_action_count,
        "repeated_click_target_count": _repeated_click_target_count(history),
        "repeated_click_penalty_count": max(0, _repeated_click_target_count(history)),
        "matched_target_reclick_count": _matched_target_reclick_count(history),
        "action_fallback_count": repeated_action_count,
        "click_suppression_count": max(0, int(action_counts.get("click_element", 0)) - 20),
        "warnings": warnings,
    }
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
    if site_id not in {"site001", "site9800"}:
        mask[action_space.encode("inspect_cart", 0)] = 0.0
    suppressed_clicks = _suppress_repeated_clicks(action_space, mask, observation, history)
    last_action_type = str(history.get("last_action_type") or "")
    counts = history.get("consecutive_action_type_counts", {})
    consecutive = int(counts.get(last_action_type, 0) or 0) if isinstance(counts, Mapping) and last_action_type else 0
    action_counts = history.get("action_type_counts", {})
    inspect_dom_total = int(action_counts.get("inspect_dom", 0) or 0) if isinstance(action_counts, Mapping) else 0
    inspect_layout_total = int(action_counts.get("inspect_layout", 0) or 0) if isinstance(action_counts, Mapping) else 0
    if consecutive >= 5 and last_action_type:
        _disable_action_type(action_space, mask, last_action_type)
        repeated = 1
        warning = f"WARNING: {site_id} policy selected {last_action_type} repeatedly; action fallback applied."
    if _is_openended_observation(observation) and last_action_type == "inspect_dom" and consecutive >= 2:
        _disable_action_type(action_space, mask, "inspect_dom")
        repeated = 1
        warning = f"WARNING: {site_id} policy selected inspect_dom repeatedly; action fallback applied."
    if _is_openended_observation(observation) and inspect_dom_total >= 10:
        _disable_action_type(action_space, mask, "inspect_dom")
        warning = warning or f"WARNING: {site_id} inspect_dom budget exhausted; action fallback applied."
    if site_id == "site9800" and inspect_layout_total >= 8:
        _disable_action_type(action_space, mask, "inspect_layout")
        warning = warning or f"WARNING: {site_id} inspect_layout budget exhausted; action fallback applied."
    elif _is_openended_observation(observation) and inspect_layout_total >= 12:
        _disable_action_type(action_space, mask, "inspect_layout")
        warning = warning or f"WARNING: {site_id} inspect_layout budget exhausted; action fallback applied."
    if not mask.any():
        mask[action_space.encode("noop", 0)] = 1.0
    return mask, warning, repeated


def _disable_action_type(action_space: ActionSpace, mask: np.ndarray, action_type: str) -> None:
    if action_type not in action_space.action_types:
        return
    for index in range(action_space.max_candidates):
        try:
            mask[action_space.encode(action_type, index)] = 0.0
        except ValueError:
            pass


def _suppress_repeated_clicks(
    action_space: ActionSpace,
    mask: np.ndarray,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> int:
    candidates = observation.get("candidate_elements", []) or []
    click_counts = history.get("catalog_click_counts", {})
    matched_bug_ids = set(history.get("matched_bug_ids", set()) or set())
    if not isinstance(candidates, list) or not isinstance(click_counts, Mapping):
        return 0
    suppressed = 0
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if not isinstance(candidate, Mapping):
            continue
        candidate_key = _candidate_key(candidate)
        catalog_ids = {str(item) for item in candidate.get("catalog_bug_id_matches", []) or []}
        already_clicked = int(click_counts.get(candidate_key, 0) or 0) >= 2
        already_matched = bool(catalog_ids.intersection(matched_bug_ids))
        if already_clicked or already_matched:
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
) -> Dict[str, str]:
    matched = set(matched_bug_ids)
    reasons: Dict[str, str] = {}
    for bug in known_bugs:
        bug_id = _known_bug_id(bug)
        if not bug_id or bug_id in matched:
            continue
        bug_type = str(bug.get("type") or "")
        if bug_type == "api-forbidden" and api_403_count <= 0:
            reasons[bug_id] = "no network 403 evidence observed"
        elif bug_type in {"async-hang", "timeout-no-feedback"} and int(action_counts.get("fill_input", 0) or 0) == 0:
            reasons[bug_id] = "async/hang input target was not filled"
        elif bug_type == "cart-quantity-mismatch" and int(action_counts.get("inspect_cart", 0) or 0) == 0:
            reasons[bug_id] = "cart quantity path was not inspected"
        elif bug_id not in attempted_bug_ids:
            reasons[bug_id] = "no catalog candidate/action reached this bug"
        else:
            reasons[bug_id] = "catalog path attempted but no matching anomaly evidence was produced"
    return reasons


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
    if site_id == "site9800":
        catalog_action = _catalog_guided_eval_action_id(action_space, observation, history)
        if catalog_action is not None:
            return catalog_action
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
        if _has_layout_target(observation):
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
        and _has_layout_target(observation)
    ):
        return action_space.encode("inspect_layout", 0)
    if isinstance(candidates, list) and isinstance(counts, Mapping) and int(counts.get("click_element", 0) or 0) == 0:
        for index, candidate in enumerate(candidates[: action_space.max_candidates]):
            if (
                isinstance(candidate, Mapping)
                and candidate.get("is_workout_add_action")
                and candidate.get("clickable")
                and float(candidate.get("visibility", 0.0) or 0.0) > 0.0
            ):
                return action_space.encode("click_element", index)
        for index, candidate in enumerate(candidates[: action_space.max_candidates]):
            if isinstance(candidate, Mapping) and candidate.get("catalog_bug_id_matches") and candidate.get("clickable"):
                return action_space.encode("click_element", index)
    return action_id


def _catalog_guided_eval_action_id(
    action_space: ActionSpace,
    observation: Mapping[str, Any],
    history: Mapping[str, Any],
) -> int | None:
    candidates = observation.get("candidate_elements", []) or []
    counts = history.get("action_type_counts", {})
    if not isinstance(candidates, list) or not isinstance(counts, Mapping):
        return None
    profile = observation.get("runtime_signals", {}).get("site_profile", {})
    bug_types = {
        str(item.get("type") or "")
        for item in profile.get("bugs", []) or []
        if isinstance(profile, Mapping) and isinstance(item, Mapping)
    }
    if {"api-forbidden", "api-ui-mismatch"}.intersection(bug_types) and int(counts.get("inspect_network", 0) or 0) == 0:
        return action_space.encode("inspect_network", 0)
    if {"api-forbidden", "api-ui-mismatch"}.intersection(bug_types) and int(counts.get("inspect_console", 0) or 0) == 0:
        return action_space.encode("inspect_console", 0)
    if "cart-quantity-mismatch" in bug_types and int(counts.get("inspect_cart", 0) or 0) == 0:
        return action_space.encode("inspect_cart", 0)
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("is_form_field") and (
            candidate.get("catalog_bug_id_matches")
            or candidate.get("is_async_related")
            or candidate.get("is_hang_related")
            or candidate.get("is_sparse_related")
            or candidate.get("is_search_related")
        ):
            bid = str(candidate.get("bid") or "")
            if bid not in set(history.get("filled_bids", set()) or set()):
                return action_space.encode("fill_input", index)
    filled = set(history.get("filled_bids", set()) or set())
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if (
            isinstance(candidate, Mapping)
            and candidate.get("is_form_field")
            and float(candidate.get("visibility", 0.0) or 0.0) > 0.0
            and str(candidate.get("bid") or "") not in filled
        ):
            return action_space.encode("fill_input", index)
    if int(counts.get("press_enter", 0) or 0) == 0:
        enter_index = None
        for index, candidate in enumerate(candidates[: action_space.max_candidates]):
            if isinstance(candidate, Mapping) and candidate.get("is_form_field"):
                enter_index = index
        if enter_index is not None:
            return action_space.encode("press_enter", enter_index)
    if int(counts.get("click_element", 0) or 0) < 8:
        for index, candidate in enumerate(candidates[: action_space.max_candidates]):
            if isinstance(candidate, Mapping) and candidate.get("catalog_bug_id_matches") and candidate.get("clickable"):
                return action_space.encode("click_element", index)
        for index, candidate in enumerate(candidates[: action_space.max_candidates]):
            if (
                isinstance(candidate, Mapping)
                and bool(candidate.get("is_interactive") or candidate.get("clickable"))
                and not candidate.get("is_form_field")
                and float(candidate.get("visibility", 0.0) or 0.0) > 0.0
            ):
                return action_space.encode("click_element", index)
    return None


def _is_openended_observation(observation: Mapping[str, Any]) -> bool:
    return _has_openended_interactive_candidate(observation) and not any(
        isinstance(candidate, Mapping) and candidate.get("catalog_bug_id_matches")
        for candidate in (observation.get("candidate_elements", []) or [])
    )


def _has_layout_target(observation: Mapping[str, Any]) -> bool:
    signals = observation.get("layout_signals", {}) if isinstance(observation.get("layout_signals"), Mapping) else {}
    if int(signals.get("catalog_layout_target_count", 0) or 0) > 0:
        return True
    candidates = observation.get("candidate_elements", []) or []
    return any(
        isinstance(candidate, Mapping) and bool(candidate.get("is_layout_target") or candidate.get("is_chart_like"))
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
    counts = history.get("catalog_click_counts", {})
    if not isinstance(counts, Mapping):
        return 0
    return sum(max(0, int(value or 0) - 1) for value in counts.values())


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
        candidate.get("bid")
        or candidate.get("data_bug_id")
        or candidate.get("selector_hint")
        or candidate.get("name")
        or candidate.get("text")
        or ""
    )


def _action_diversity_score(action_counts: Mapping[str, Any]) -> float:
    main_actions = {"click_element", "inspect_layout", "inspect_dom", "change_viewport_mobile"}
    used = sum(1 for action_type in main_actions if int(action_counts.get(action_type, 0) or 0) > 0)
    return round(used / len(main_actions), 4)



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


def _validate_sites(raw_sites: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_sites, list) or not raw_sites:
        raise ValueError("Config requires a non-empty sites list.")
    return [dict(site) for site in raw_sites if isinstance(site, Mapping)]


def _known_bug_id(bug: Mapping[str, Any]) -> str:
    return str(bug.get("id") or bug.get("bug_id") or bug.get("name") or "")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
