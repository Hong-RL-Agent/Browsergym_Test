"""Diagnose DQN candidate-anomaly reward attribution for repeated click loops.

This runner is diagnostic-only. It does not change reward weights, detector
rules, action masks, DQN architecture, PPO, or Hybrid behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.agents.dqn_agent import (
    FACTORIZED_DQN_ARCHITECTURE,
    FactorizedQNetwork,
    factorized_q_value_diagnostics,
    select_factorized_greedy_action,
)
from dqn_error_discovery.runners.dqn_click_bias_diagnostics import repeat_diagnostics
from dqn_error_discovery.runners.train_dqn_error_detector import (
    DQN_REWARD_CONFIG,
    _build_functional_local_state_signature,
    _build_raw_local_state_signature,
    _build_viewport_context_signature,
    _load_site_config,
    _target_identity,
    calculate_dqn_reward,
    dqn_no_effect_diagnostic,
    initialize_dqn_candidate_reward_baseline,
    safe_action_mask,
    sanitize_anomalies,
)
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies


CLICK = "click_element"
CAUSAL_RULES = {"R01", "R02", "R08", "R09", "R10", "R11", "R13", "R14", "R15"}
REVEALABLE_RULES = {"R04", "R05", "R06", "R07", "R12"}
GROUND_TRUTH_KEYS = {
    "known_bug",
    "expected_bug",
    "ground_truth",
    "data_bug_id",
    "catalog_bug_id_matches",
    "clicked_data_bug_id",
    "matched_bug_id",
}


def candidate_summary(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    candidate = candidate or {}
    return {
        "bid": _safe_str(candidate.get("bid")),
        "entity_id": _safe_str(_first_present(candidate, ("entity_id", "product_id", "item_id", "post_id", "order_id"))),
        "role": _safe_str(candidate.get("role")),
        "tag": _safe_str(candidate.get("tag")),
        "name": _sanitize_text(candidate.get("name") or candidate.get("accessible_name") or candidate.get("aria_label") or candidate.get("text")),
        "href": _url_path_from_value(candidate.get("href")),
        "visible": bool(candidate.get("visible", True)),
        "enabled": bool(candidate.get("enabled", True)),
        "clickable": bool(candidate.get("clickable", False)),
        "bbox": candidate.get("bbox", []),
        "logical_group": logical_candidate_group_key(candidate),
    }


def logical_candidate_group_key(anomaly: Mapping[str, Any]) -> str:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    target = evidence.get("target", {}) if isinstance(evidence.get("target"), Mapping) else {}
    rule_id = _safe_str(anomaly.get("rule_id") or evidence.get("rule_id"))
    url_path = _url_path_from_value(anomaly.get("url") or anomaly.get("url_path") or evidence.get("url") or evidence.get("url_path"))
    subtype = _safe_str(anomaly.get("issue_type") or evidence.get("issue_type") or anomaly.get("type") or "candidate")
    for source in (anomaly, evidence, target):
        value = _first_present(source, ("entity_id", "product_id", "item_id", "post_id", "order_id", "user_id", "cart_item_id"))
        if value not in (None, ""):
            return "|".join([rule_id, url_path, subtype, f"entity:{value}"])
    href = _url_path_from_value(evidence.get("clicked_href") or target.get("href") or anomaly.get("href"))
    if href:
        return "|".join([rule_id, url_path, subtype, f"href:{href}"])
    target_id = _safe_str(anomaly.get("target_id") or evidence.get("target_id") or evidence.get("clicked_bid") or target.get("bid"))
    if target_id:
        return "|".join([rule_id, url_path, subtype, f"target:{target_id}"])
    text = _sanitize_text(evidence.get("clicked_name") or evidence.get("clicked_text") or target.get("name") or target.get("text"))
    role = _safe_str(target.get("role") or evidence.get("clicked_role"))
    tag = _safe_str(target.get("tag") or evidence.get("clicked_tag"))
    return "|".join([rule_id, url_path, subtype, role, tag, text[:80]])


def classify_candidate_attribution(
    *,
    anomaly: Mapping[str, Any],
    action: Mapping[str, Any],
    before_signatures: set[str],
    functional_changed: bool,
    raw_changed: bool,
    viewport_changed: bool,
) -> str:
    signature = _safe_str(anomaly.get("signature"))
    rule_id = _safe_str(anomaly.get("rule_id") or (anomaly.get("evidence", {}) or {}).get("rule_id"))
    if signature and signature in before_signatures:
        return "PREEXISTING"
    action_type = _safe_str(action.get("action_type"))
    if rule_id == "R01" and action_type == CLICK:
        return "ACTION_CAUSED"
    if rule_id in CAUSAL_RULES and (functional_changed or raw_changed):
        return "ACTION_CAUSED"
    if rule_id in REVEALABLE_RULES and viewport_changed:
        return "ACTION_REVEALED"
    if not functional_changed and not raw_changed and not viewport_changed:
        return "PREEXISTING"
    return "UNKNOWN"


def reward_combination_key(components: Mapping[str, Any]) -> str:
    positive_candidate = float(components.get("candidate_anomaly", 0.0) or 0.0) > 0.0
    repeat = float(components.get("repeat_action_penalty", 0.0) or 0.0) < 0.0
    no_effect = float(components.get("no_effect_penalty", 0.0) or 0.0) < 0.0
    if positive_candidate and repeat and no_effect:
        return "candidate+repeat+no_effect"
    if positive_candidate and repeat:
        return "candidate+repeat"
    if positive_candidate and no_effect:
        return "candidate+no_effect"
    if positive_candidate:
        return "candidate_only"
    if repeat and no_effect:
        return "repeat+no_effect"
    if repeat:
        return "repeat_only"
    if no_effect:
        return "no_effect_only"
    return "none"


def click_effect_labels(
    *,
    components: Mapping[str, Any],
    functional_changed: bool,
    raw_changed: bool,
    viewport_changed: bool,
    attribution_counts: Mapping[str, int],
) -> list[str]:
    labels: list[str] = []
    if functional_changed or float(components.get("flow_progress", 0.0) or 0.0) > 0.0:
        labels.append("FUNCTIONAL_PROGRESS")
    if attribution_counts.get("ACTION_CAUSED", 0):
        labels.append("ANOMALY_CAUSAL")
    if viewport_changed and not functional_changed:
        labels.append("VIEWPORT_ONLY_CHANGE")
    if raw_changed and not functional_changed and not viewport_changed:
        labels.append("RAW_OBSERVATION_ONLY_CHANGE")
    if float(components.get("no_effect_penalty", 0.0) or 0.0) < 0.0:
        labels.append("NO_EFFECT")
    if not labels:
        labels.append("UNKNOWN")
    return labels


def summarize_click_rewards(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clicks = [record for record in records if record.get("action_type") == CLICK]
    rewards = [float(record.get("reward_total", 0.0) or 0.0) for record in clicks]
    positive = [value for value in rewards if value > 0.0]
    zero = [value for value in rewards if value == 0.0]
    negative = [value for value in rewards if value < 0.0]
    candidate_rewarded = [record for record in clicks if float((record.get("reward_components", {}) or {}).get("candidate_anomaly", 0.0) or 0.0) > 0.0]
    candidate_total = sum(float((record.get("reward_components", {}) or {}).get("candidate_anomaly", 0.0) or 0.0) for record in clicks)
    repeat_positive = [
        record for record in clicks
        if record.get("immediate_repeat") and float(record.get("reward_total", 0.0) or 0.0) > 0.0
    ]
    no_effect_positive = [
        record for record in clicks
        if float((record.get("reward_components", {}) or {}).get("no_effect_penalty", 0.0) or 0.0) < 0.0
        and float(record.get("reward_total", 0.0) or 0.0) > 0.0
    ]
    repeat_no_effect_positive = [
        record for record in clicks
        if record.get("immediate_repeat")
        and float((record.get("reward_components", {}) or {}).get("no_effect_penalty", 0.0) or 0.0) < 0.0
        and float(record.get("reward_total", 0.0) or 0.0) > 0.0
    ]
    return {
        "click_count": len(clicks),
        "positive": _reward_bucket_summary(positive, len(clicks)),
        "zero": _reward_bucket_summary(zero, len(clicks)),
        "negative": _reward_bucket_summary(negative, len(clicks)),
        "candidate_rewarded_click_count": len(candidate_rewarded),
        "candidate_reward_total": candidate_total,
        "candidate_reward_mean_per_click": _ratio(candidate_total, len(clicks)),
        "candidate_reward_mean_per_rewarded_click": _ratio(candidate_total, len(candidate_rewarded)),
        "repeat_positive_click_count": len(repeat_positive),
        "no_effect_positive_click_count": len(no_effect_positive),
        "repeat_no_effect_positive_click_count": len(repeat_no_effect_positive),
    }


def summarize_reward_combinations(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(reward_combination_key(record.get("reward_components", {}) or {}) for record in records))


def summarize_rule_rewards(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "observation_count": 0,
        "raw_signatures": set(),
        "logical_groups": set(),
        "rewarded_signature_count": 0,
        "reward_total": 0.0,
        "click_transition_count": 0,
    })
    for record in records:
        rewarded = float((record.get("reward_components", {}) or {}).get("candidate_anomaly", 0.0) or 0.0) > 0.0
        for anomaly in record.get("candidate_anomalies", []) or []:
            rule_id = _safe_str(anomaly.get("rule_id"))
            if not rule_id:
                continue
            row = rows[rule_id]
            row["observation_count"] += 1
            row["raw_signatures"].add(_safe_str(anomaly.get("signature")))
            row["logical_groups"].add(_safe_str(anomaly.get("logical_group")))
            row["click_transition_count"] += int(record.get("action_type") == CLICK)
            if rewarded and anomaly.get("first_seen"):
                row["rewarded_signature_count"] += 1
                row["reward_total"] += DQN_REWARD_CONFIG["candidate_anomaly"]
    return {
        rule_id: {
            "observation_count": int(row["observation_count"]),
            "unique_raw_signatures": len(row["raw_signatures"]),
            "logical_groups": len(row["logical_groups"]),
            "rewarded_signature_count": int(row["rewarded_signature_count"]),
            "reward_total": float(row["reward_total"]),
            "click_transition_count": int(row["click_transition_count"]),
        }
        for rule_id, row in sorted(rows.items())
    }


def summarize_candidate_fragmentation(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw_signatures: set[str] = set()
    logical_groups: set[str] = set()
    by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for anomaly in record.get("candidate_anomalies", []) or []:
            signature = _safe_str(anomaly.get("signature"))
            group = _safe_str(anomaly.get("logical_group"))
            if signature:
                raw_signatures.add(signature)
            if group:
                logical_groups.add(group)
                if signature:
                    by_group[group].add(signature)
    fragmented = [
        {"logical_group": group, "raw_signature_count": len(signatures), "signatures": sorted(signatures)[:8]}
        for group, signatures in by_group.items()
        if len(signatures) > 1
    ]
    fragmented.sort(key=lambda item: int(item["raw_signature_count"]), reverse=True)
    return {
        "raw_unique_candidate_signatures": len(raw_signatures),
        "logical_unique_candidate_groups": len(logical_groups),
        "candidate_signature_fragmentation_ratio": _ratio(len(raw_signatures), len(logical_groups)),
        "fragmented_groups": fragmented[:20],
    }


def summarize_candidate_attribution(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_counts: Counter[str] = Counter()
    click_counts: Counter[str] = Counter()
    rule_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for anomaly in record.get("candidate_anomalies", []) or []:
            attribution = _safe_str(anomaly.get("attribution") or "UNKNOWN")
            all_counts[attribution] += 1
            rule_counts[_safe_str(anomaly.get("rule_id"))][attribution] += 1
            if record.get("action_type") == CLICK:
                click_counts[attribution] += 1
    click_total = sum(click_counts.values())
    return {
        "all": dict(all_counts),
        "click": dict(click_counts),
        "click_ratios": {key: _ratio(value, click_total) for key, value in click_counts.items()},
        "by_rule": {rule: dict(counts) for rule, counts in sorted(rule_counts.items())},
    }


def candidate_reward_concentration(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    positive_mass = 0.0
    candidate_positive_mass = 0.0
    click_positive_mass = 0.0
    click_candidate_positive_mass = 0.0
    for record in records:
        reward = float(record.get("reward_total", 0.0) or 0.0)
        components = record.get("reward_components", {}) or {}
        candidate = max(0.0, float(components.get("candidate_anomaly", 0.0) or 0.0))
        if reward > 0.0:
            positive_mass += reward
            candidate_positive_mass += candidate
            if record.get("action_type") == CLICK:
                click_positive_mass += reward
                click_candidate_positive_mass += candidate
    return {
        "candidate_positive_mass": candidate_positive_mass,
        "all_positive_reward_mass": positive_mass,
        "candidate_positive_mass_ratio": _ratio(candidate_positive_mass, positive_mass),
        "click_candidate_positive_mass": click_candidate_positive_mass,
        "click_positive_reward_mass": click_positive_mass,
        "click_candidate_positive_mass_ratio": _ratio(click_candidate_positive_mass, click_positive_mass),
    }


def training_block_candidate_reward(manifest: Mapping[str, Any]) -> dict[str, Any]:
    episodes = manifest.get("episode_summaries", []) if isinstance(manifest.get("episode_summaries"), list) else []
    blocks = {"EP1-10": range(1, 11), "EP11-20": range(11, 21), "EP21-30": range(21, 31)}
    result: dict[str, Any] = {}
    for name, episode_range in blocks.items():
        items = [item for item in episodes if isinstance(item, Mapping) and int(item.get("episode", 0) or 0) in episode_range]
        click_count = sum(int((item.get("action_distribution", {}) or {}).get(CLICK, 0) or 0) for item in items)
        click_components = Counter()
        click_events = Counter()
        for item in items:
            totals = ((item.get("action_type_reward_component_totals", {}) or {}).get(CLICK, {}) or {})
            events = ((item.get("action_type_reward_component_event_counts", {}) or {}).get(CLICK, {}) or {})
            for key, value in totals.items():
                click_components[str(key)] += float(value or 0.0)
            for key, value in events.items():
                click_events[str(key)] += int(value or 0)
        result[name] = {
            "click_count": click_count,
            "candidate_reward_click_event_count": int(click_events.get("candidate_anomaly", 0)),
            "candidate_reward_total": float(click_components.get("candidate_anomaly", 0.0)),
            "mean_click_reward": _ratio(click_components.get("final_reward", 0.0), click_count),
            "repeat_click_count": int(click_events.get("repeat_action_penalty", 0)),
            "no_effect_click_count": int(click_events.get("no_effect_penalty", 0)),
        }
    return result


def confirmation_progress(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    useful_repeat = 0
    unproductive_repeat = 0
    for record in records:
        action_target = _safe_str(record.get("local_action_target"))
        for anomaly in record.get("candidate_anomalies", []) or []:
            signature = _safe_str(anomaly.get("signature"))
            if not signature:
                continue
            item = candidates.setdefault(
                signature,
                {
                    "rule_id": anomaly.get("rule_id"),
                    "confirmation_policy": anomaly.get("confirmation_policy"),
                    "required_action_target": anomaly.get("action_target"),
                    "seen_steps": [],
                    "followup_actions": [],
                    "confirmed": False,
                },
            )
            item["seen_steps"].append({"episode": record.get("episode"), "step": record.get("step")})
            if record.get("immediate_repeat") and action_target == _safe_str(item.get("required_action_target")):
                useful_repeat += 1
            elif record.get("immediate_repeat") and record.get("action_type") == CLICK:
                unproductive_repeat += 1
    confirmed = {sig for record in records for sig in record.get("confirmed_signatures", []) or []}
    for signature in confirmed:
        if signature in candidates:
            candidates[signature]["confirmed"] = True
    return {
        "candidate_count": len(candidates),
        "confirmed_count": sum(1 for item in candidates.values() if item.get("confirmed")),
        "useful_reproduction_repeat_count": useful_repeat,
        "unproductive_repeat_count": unproductive_repeat,
        "pending_candidates": [
            {"signature": sig, **item}
            for sig, item in list(candidates.items())[:30]
            if not item.get("confirmed")
        ],
    }


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    site = _load_site_config(args.config, args.site_id, args.base_url)
    env = BrowserGymJAWSEnv(
        site_id=str(site["site_id"]),
        base_url=str(site["base_url"]),
        max_steps=args.max_steps,
        max_candidates=args.max_candidates,
        headless=args.headless,
    )
    action_space = env.action_space
    network = FactorizedQNetwork(env.observation_encoder.get_obs_dim(), len(action_space.action_types), action_space.max_candidates)
    payload = torch.load(Path(args.model_path), map_location="cpu")
    manifest = _load_manifest_payload(args.model_path, payload)
    if manifest.get("dqn_architecture") != FACTORIZED_DQN_ARCHITECTURE:
        raise ValueError("candidate attribution diagnostic expects a factorized_v1 DQN checkpoint")
    network.load_state_dict(payload.get("model", payload))
    network.eval()

    records: list[dict[str, Any]] = []
    action_targets: list[str] = []
    history: dict[str, Any] = {}
    previous_candidate_signatures: set[str] = set()

    try:
        for episode in range(1, args.episodes + 1):
            raw, _ = env.reset()
            state = env.observation_encoder.encode_observation(raw)
            history = {}
            initial_anomalies = sanitize_anomalies(detect_anomalies(raw, raw, {"action": {"action_type": "noop", "candidate_index": 0}}))
            initialize_dqn_candidate_reward_baseline(history, initial_anomalies)
            previous_candidate_signatures = set()
            previous_action_target = ""
            for step in range(1, args.max_steps + 1):
                global_step = (episode - 1) * args.max_steps + step
                mask = safe_action_mask(action_space, raw)
                action_id = select_factorized_greedy_action(network, state, mask, action_space)
                q_diag = factorized_q_value_diagnostics(network, state, mask, action_space, action_id)
                decoded = action_space.decode(action_id)
                decoded["action_id"] = int(action_id)
                action_type = _safe_str(decoded.get("action_type"))
                candidate_index = int(decoded.get("candidate_index", 0) or 0)
                candidate = _candidate_at(raw, candidate_index) if action_type in {CLICK, "fill_input", "press_enter"} else {}
                local_action_target = f"{action_type}:{_target_identity(candidate) or candidate_index}"

                before_functional = _build_functional_local_state_signature(raw)
                before_raw = _build_raw_local_state_signature(raw)
                before_viewport = _build_viewport_context_signature(raw)
                next_raw, _, done, info = env.step(action_id)
                decoded["failed"] = bool(info.get("last_action_error"))
                decoded["timeout"] = bool(info.get("timeout") or info.get("timed_out") or info.get("timeout_occurred"))
                after_functional = _build_functional_local_state_signature(next_raw)
                after_raw = _build_raw_local_state_signature(next_raw)
                after_viewport = _build_viewport_context_signature(next_raw)
                functional_changed = before_functional != after_functional
                raw_changed = before_raw != after_raw
                viewport_changed = before_viewport != after_viewport

                anomalies = sanitize_anomalies(detect_anomalies(raw, next_raw, {**info, "action": decoded}))
                seen_before = set(history.get("seen_candidate_signatures", set()))
                reward, components = calculate_dqn_reward(raw, next_raw, decoded, anomalies, history)
                no_effect_diag = dqn_no_effect_diagnostic(raw, next_raw, decoded)
                repeat_diag = history.get("last_repeat_diagnostic", {}) if isinstance(history, Mapping) else {}
                seen_after = set(history.get("seen_candidate_signatures", set()))
                candidate_anomalies = []
                attribution_counter: Counter[str] = Counter()
                current_candidate_signatures = set()
                confirmed_signatures = []
                for anomaly in anomalies:
                    signature = _safe_str(anomaly.get("signature"))
                    status = _safe_str(anomaly.get("status") or "candidate")
                    if status == "candidate":
                        current_candidate_signatures.add(signature)
                        attribution = classify_candidate_attribution(
                            anomaly=anomaly,
                            action=decoded,
                            before_signatures=previous_candidate_signatures,
                            functional_changed=functional_changed,
                            raw_changed=raw_changed,
                            viewport_changed=viewport_changed,
                        )
                        attribution_counter[attribution] += 1
                        candidate_anomalies.append(
                            {
                                "rule_id": anomaly.get("rule_id"),
                                "status": status,
                                "signature": signature,
                                "logical_group": logical_candidate_group_key(anomaly),
                                "first_seen": bool(signature and signature not in seen_before and signature in seen_after),
                                "previous_seen": bool(signature and signature in seen_before),
                                "previous_seen_step": None,
                                "confirmation_policy": anomaly.get("confirmation_policy"),
                                "action_target": _anomaly_action_target_key(anomaly),
                                "attribution": attribution,
                            }
                        )
                    elif status == "confirmed":
                        confirmed_signatures.append(signature)

                immediate_repeat = bool(local_action_target and local_action_target == previous_action_target)
                action_targets.append(local_action_target)
                record = {
                    "episode": episode,
                    "step": step,
                    "global_step": global_step,
                    "action_type": action_type,
                    "action_id": int(action_id),
                    "candidate_index": candidate_index,
                    "target": candidate_summary(candidate),
                    "local_action_target": local_action_target,
                    "functional_state_before": before_functional,
                    "functional_state_after": after_functional,
                    "functional_state_changed": functional_changed,
                    "raw_state_changed": raw_changed,
                    "viewport_context_changed": viewport_changed,
                    "candidate_signatures_before": sorted(previous_candidate_signatures),
                    "candidate_signatures_after": sorted(current_candidate_signatures),
                    "pre_vs_post_candidates": {
                        "preexisting": sorted(previous_candidate_signatures & current_candidate_signatures),
                        "new_after_action": sorted(current_candidate_signatures - previous_candidate_signatures),
                        "disappeared": sorted(previous_candidate_signatures - current_candidate_signatures),
                        "unchanged": sorted(previous_candidate_signatures & current_candidate_signatures),
                    },
                    "reward_total": float(reward),
                    "reward_components": components,
                    "anomaly_present": bool(anomalies),
                    "r01_present": any(_safe_str(item.get("rule_id")) == "R01" for item in anomalies),
                    "meaningful_action_effect": bool(no_effect_diag.get("meaningful_action_effect")),
                    "no_effect": bool(no_effect_diag.get("no_effect")),
                    "no_effect_reason": no_effect_diag.get("no_effect_reasons", []),
                    "effect_reasons": no_effect_diag.get("effect_reasons", []),
                    "feedback_changed": bool(no_effect_diag.get("feedback_changed")),
                    "network_effect": bool(no_effect_diag.get("network_effect")),
                    "target_state_changed": bool(no_effect_diag.get("target_state_changed")),
                    "candidate_reward": float(components.get("candidate_anomaly", 0.0) or 0.0),
                    "confirmed_reward": float(components.get("confirmed_anomaly", 0.0) or 0.0),
                    "repeat_penalty": float(components.get("repeat_action_penalty", 0.0) or 0.0),
                    "no_effect_penalty": float(components.get("no_effect_penalty", 0.0) or 0.0),
                    "candidate_anomalies": candidate_anomalies,
                    "confirmed_signatures": confirmed_signatures,
                    "reward_combination": reward_combination_key(components),
                    "candidate_attribution_counts": dict(attribution_counter),
                    "effect_labels": click_effect_labels(
                        components=components,
                        functional_changed=functional_changed,
                        raw_changed=raw_changed,
                        viewport_changed=viewport_changed,
                        attribution_counts=attribution_counter,
                    ),
                    "immediate_repeat": immediate_repeat,
                    "action_identity": repeat_diag.get("action_identity", local_action_target) if isinstance(repeat_diag, Mapping) else local_action_target,
                    "recent_action_window": repeat_diag.get("recent_action_window", []) if isinstance(repeat_diag, Mapping) else [],
                    "recent_window_repeat": bool(repeat_diag.get("recent_window_repeat")) if isinstance(repeat_diag, Mapping) else False,
                    "cycle_repeat": bool(repeat_diag.get("cycle_repeat")) if isinstance(repeat_diag, Mapping) else False,
                    "cycle_length": int(repeat_diag.get("cycle_length", 0) or 0) if isinstance(repeat_diag, Mapping) else 0,
                    "repeat_kinds": repeat_diag.get("repeat_kinds", []) if isinstance(repeat_diag, Mapping) else [],
                    "productive_transition": bool(repeat_diag.get("productive_transition")) if isinstance(repeat_diag, Mapping) else False,
                    "productive_reasons": repeat_diag.get("productive_reasons", []) if isinstance(repeat_diag, Mapping) else [],
                    "pending_reproduction": bool(repeat_diag.get("pending_reproduction")) if isinstance(repeat_diag, Mapping) else False,
                    "reproduction_target_match": bool(repeat_diag.get("reproduction_target_match")) if isinstance(repeat_diag, Mapping) else False,
                    "reproduction_attempt": bool(repeat_diag.get("reproduction_attempt")) if isinstance(repeat_diag, Mapping) else False,
                    "confirmation_progress": bool(repeat_diag.get("confirmation_progress")) if isinstance(repeat_diag, Mapping) else False,
                    "useful_reproduction_repeat": bool(repeat_diag.get("useful_reproduction_repeat")) if isinstance(repeat_diag, Mapping) else False,
                    "unproductive_repeat": bool(repeat_diag.get("unproductive_repeat")) if isinstance(repeat_diag, Mapping) else False,
                    "repeat_penalty_applied": bool(repeat_diag.get("repeat_penalty_applied")) if isinstance(repeat_diag, Mapping) else False,
                    "q_diagnostic": q_diag,
                    "action_error": bool(decoded.get("failed")),
                }
                records.append(record)
                previous_candidate_signatures = current_candidate_signatures
                previous_action_target = local_action_target
                raw = next_raw
                state = env.observation_encoder.encode_observation(raw)
                if done:
                    break
    finally:
        env.close()

    repeat = repeat_diagnostics(action_targets, window=args.repeat_window)
    report = {
        "model_path": args.model_path,
        "site_id": site["site_id"],
        "url": site["base_url"],
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "dqn_architecture": FACTORIZED_DQN_ARCHITECTURE,
        "selection_mode": "factorized_type_then_candidate",
        "step_reward_trace": records,
        "click_reward_summary": summarize_click_rewards(records),
        "reward_combination_counts": summarize_reward_combinations(records),
        "rule_reward_summary": summarize_rule_rewards(records),
        "raw_vs_logical_candidate_signatures": summarize_candidate_fragmentation(records),
        "candidate_fragmentation": summarize_candidate_fragmentation(records),
        "pre_vs_post_candidates": summarize_pre_post(records),
        "candidate_attribution": summarize_candidate_attribution(records),
        "no_effect_analysis": summarize_no_effect(records),
        "repeat_positive_analysis": summarize_repeat_positive(records),
        "r01_transition_matrix": summarize_r01_transition_matrix(records),
        "candidate_reward_concentration": candidate_reward_concentration(records),
        "training_block_candidate_reward": training_block_candidate_reward(manifest),
        "confirmation_progress": confirmation_progress(records),
        "repeat_diagnostics": repeat,
        "repeat_breakdown": summarize_repeat_breakdown(records),
        "action_type_repeat_breakdown": summarize_action_type_repeat_breakdown(records),
        "action_distribution": dict(Counter(record["action_type"] for record in records)),
        "action_error_count": sum(int(record.get("action_error", False)) for record in records),
        "root_cause_summary": root_cause_summary(records, manifest),
        "manifest_context": {
            "dqn_architecture": manifest.get("dqn_architecture"),
            "episodes": manifest.get("episodes"),
            "greedy_action_distribution": manifest.get("greedy_action_distribution"),
            "action_type_reward_summary_click": (manifest.get("action_type_reward_summary", {}) or {}).get(CLICK, {}),
            "factorized_diagnostic_summary": manifest.get("factorized_diagnostic_summary"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def summarize_pre_post(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        pre_post = record.get("pre_vs_post_candidates", {}) or {}
        for key in ("preexisting", "new_after_action", "disappeared", "unchanged"):
            counts[key] += len(pre_post.get(key, []) or [])
    return dict(counts)


def summarize_no_effect(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    labels = Counter()
    for record in records:
        components = record.get("reward_components", {}) or {}
        is_no_effect = float(components.get("no_effect_penalty", 0.0) or 0.0) < 0.0
        if record.get("action_type") == CLICK:
            for label in record.get("effect_labels", []) or []:
                labels[str(label)] += 1
            if (
                not record.get("functional_state_changed")
                and not record.get("action_error")
                and not is_no_effect
            ):
                rows.append(
                    {
                        "episode": record.get("episode"),
                        "step": record.get("step"),
                        "action_type": record.get("action_type"),
                        "target": record.get("target"),
                        "reward_total": record.get("reward_total"),
                        "raw_state_changed": record.get("raw_state_changed"),
                        "viewport_context_changed": record.get("viewport_context_changed"),
                        "candidate_anomalies": record.get("candidate_anomalies", []),
                        "effect_labels": record.get("effect_labels", []),
                    }
                )
    return {
        "click_effect_label_counts": dict(labels),
        "functional_unchanged_without_no_effect_count": len(rows),
        "functional_unchanged_without_no_effect_examples": rows[:20],
    }


def summarize_repeat_positive(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    click_records = [record for record in records if record.get("action_type") == CLICK]
    repeat = [record for record in click_records if record.get("immediate_repeat")]
    repeat_positive = [record for record in repeat if float(record.get("reward_total", 0.0) or 0.0) > 0.0]
    no_effect_positive = [
        record for record in click_records
        if float((record.get("reward_components", {}) or {}).get("no_effect_penalty", 0.0) or 0.0) < 0.0
        and float(record.get("reward_total", 0.0) or 0.0) > 0.0
    ]
    repeat_no_effect_positive = [
        record for record in no_effect_positive
        if record.get("immediate_repeat")
    ]
    return {
        "click_count": len(click_records),
        "repeat_click_count": len(repeat),
        "repeat_positive_click_count": len(repeat_positive),
        "no_effect_positive_click_count": len(no_effect_positive),
        "repeat_no_effect_positive_click_count": len(repeat_no_effect_positive),
        "repeat_positive_examples": [
            {
                "episode": record.get("episode"),
                "step": record.get("step"),
                "reward_total": record.get("reward_total"),
                "reward_components": record.get("reward_components"),
                "candidate_anomalies": record.get("candidate_anomalies", []),
            }
            for record in repeat_positive[:20]
        ],
    }


def summarize_r01_transition_matrix(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if not bool(record.get("r01_present")):
            continue
        components = record.get("reward_components", {}) or {}
        has_no_effect = bool(record.get("no_effect")) or float(components.get("no_effect_penalty", 0.0) or 0.0) < 0.0
        has_candidate = float(components.get("candidate_anomaly", 0.0) or 0.0) > 0.0
        has_confirmed = (
            float(components.get("confirmed_anomaly", 0.0) or 0.0) > 0.0
            or float(components.get("confirmation_upgrade", 0.0) or 0.0) > 0.0
        )
        has_repeat = float(components.get("repeat_action_penalty", 0.0) or 0.0) < 0.0
        counts["R01"] += 1
        counts["R01+no_effect" if has_no_effect else "R01+meaningful_effect"] += 1
        if has_candidate:
            counts["R01+candidate_reward"] += 1
        if has_confirmed:
            counts["R01+confirmed"] += 1
        if has_repeat:
            counts["R01+repeat"] += 1
        if has_no_effect and has_repeat:
            counts["R01+no_effect+repeat"] += 1
        if has_no_effect and has_candidate:
            counts["R01+no_effect+candidate"] += 1
        if has_no_effect and has_confirmed:
            counts["R01+no_effect+confirmed"] += 1
    return dict(counts)


def summarize_repeat_breakdown(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for record in records:
        repeat_flags = 0
        if bool(record.get("immediate_repeat")):
            counts["immediate_repeat_count"] += 1
            repeat_flags += 1
        if bool(record.get("recent_window_repeat")):
            counts["recent_window_repeat_count"] += 1
            repeat_flags += 1
        if bool(record.get("cycle_repeat")):
            counts["cycle_repeat_count"] += 1
            repeat_flags += 1
            cycle_length = int(record.get("cycle_length", 0) or 0)
            if cycle_length:
                counts[f"cycle_length_{cycle_length}_count"] += 1
        if repeat_flags == 1 and bool(record.get("recent_window_repeat")):
            counts["recent_only_repeat_count"] += 1
        elif repeat_flags == 1 and bool(record.get("cycle_repeat")):
            counts["cycle_only_repeat_count"] += 1
        elif repeat_flags > 1:
            counts["multiple_repeat_count"] += 1
        if bool(record.get("useful_reproduction_repeat")):
            counts["useful_reproduction_repeat_count"] += 1
        if bool(record.get("unproductive_repeat")):
            counts["unproductive_repeat_count"] += 1
        if bool(record.get("repeat_penalty_applied")):
            counts["repeat_penalty_count"] += 1
        if bool(record.get("confirmation_progress")):
            counts["confirmation_progress_count"] += 1
    total = len(records)
    return {
        **dict(counts),
        "unproductive_repeat_ratio": _ratio(counts["unproductive_repeat_count"], total),
        "useful_reproduction_repeat_ratio": _ratio(counts["useful_reproduction_repeat_count"], total),
        "repeat_penalty_ratio": _ratio(counts["repeat_penalty_count"], total),
    }


def summarize_action_type_repeat_breakdown(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "unproductive_repeat_count": 0,
        "cycle_participation_count": 0,
        "useful_reproduction_repeat_count": 0,
        "rewards": [],
    })
    for record in records:
        action_type = _safe_str(record.get("action_type"))
        row = rows[action_type]
        row["count"] += 1
        row["unproductive_repeat_count"] += int(bool(record.get("unproductive_repeat")))
        row["cycle_participation_count"] += int(bool(record.get("cycle_repeat")))
        row["useful_reproduction_repeat_count"] += int(bool(record.get("useful_reproduction_repeat")))
        row["rewards"].append(float(record.get("reward_total", 0.0) or 0.0))
    return {
        action_type: {
            "count": int(row["count"]),
            "unproductive_repeat_count": int(row["unproductive_repeat_count"]),
            "cycle_participation_count": int(row["cycle_participation_count"]),
            "useful_reproduction_repeat_count": int(row["useful_reproduction_repeat_count"]),
            "mean_reward": float(np.mean(row["rewards"])) if row["rewards"] else 0.0,
        }
        for action_type, row in rows.items()
    }


def root_cause_summary(records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> list[str]:
    causes: list[str] = []
    click_summary = summarize_click_rewards(records)
    concentration = candidate_reward_concentration(records)
    attribution = summarize_candidate_attribution(records)
    fragmentation = summarize_candidate_fragmentation(records)
    no_effect = summarize_no_effect(records)
    repeat_positive = summarize_repeat_positive(records)
    if click_summary["candidate_rewarded_click_count"] > 0:
        causes.append("candidate_anomaly_reward_assigned_to_click_transitions")
    if concentration["click_candidate_positive_mass_ratio"] >= 0.5:
        causes.append("click_positive_reward_mass_dominated_by_candidate_reward")
    if attribution["click_ratios"].get("PREEXISTING", 0.0) >= 0.3:
        causes.append("preexisting_candidate_evidence_attributed_to_click")
    if fragmentation["candidate_signature_fragmentation_ratio"] > 1.2:
        causes.append("candidate_signature_fragmentation")
    if repeat_positive["repeat_positive_click_count"] > 0:
        causes.append("repeat_click_still_receives_positive_net_reward")
    if no_effect["functional_unchanged_without_no_effect_count"] > 0:
        causes.append("functional_unchanged_click_not_always_no_effect_penalized")
    click_training = ((manifest.get("action_type_reward_summary", {}) or {}).get(CLICK, {}) or {})
    if float(click_training.get("mean_reward", 0.0) or 0.0) > 0.0:
        causes.append("training_click_mean_reward_positive")
    return causes


def _reward_bucket_summary(values: Sequence[float], total_count: int) -> dict[str, float]:
    return {
        "count": len(values),
        "ratio": _ratio(len(values), total_count),
        "mean": _mean(values),
        "median": float(np.median(values)) if values else 0.0,
        "min": float(np.min(values)) if values else 0.0,
        "max": float(np.max(values)) if values else 0.0,
    }


def _candidate_at(observation: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    if isinstance(candidates, list) and 0 <= index < len(candidates) and isinstance(candidates[index], Mapping):
        return candidates[index]
    return {}


def _anomaly_action_target_key(anomaly: Mapping[str, Any]) -> str:
    action_type = _safe_str(anomaly.get("action_type"))
    target_id = _safe_str(anomaly.get("target_id"))
    if not action_type or not target_id:
        return ""
    if target_id.startswith(("bid:", "entity_id:", "product_id:", "fingerprint:")):
        return f"{action_type}:{target_id}"
    return f"{action_type}:bid:{target_id}"


def _load_manifest_payload(model_path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = Path(model_path).with_suffix(".manifest.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = payload.get("manifest", {}) if isinstance(payload, Mapping) else {}
    return dict(manifest) if isinstance(manifest, Mapping) else {}


def _first_present(source: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = source.get(key)
        if value not in (None, ""):
            return value
    return ""


def _url_path_from_value(value: Any) -> str:
    text = _safe_str(value)
    if not text:
        return ""
    parsed = urlparse(text)
    return parsed.path or text.split("?", 1)[0] or "/"


def _sanitize_text(value: Any) -> str:
    return " ".join(_safe_str(value).split())[:120]


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        den = float(denominator or 0.0)
        return float(numerator or 0.0) / den if den else 0.0
    except (TypeError, ValueError):
        return 0.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--site-id", default="site001")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model-path", default="artifacts/models/site001_dqn_factorized_v1_ep30.pt")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--repeat-window", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="artifacts/sanity/dqn_site001_factorized_candidate_reward_diagnostics.json")
    args = parser.parse_args()
    print(json.dumps(_jsonable(run_diagnostics(args)), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
