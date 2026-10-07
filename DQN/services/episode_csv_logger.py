"""CSV episode/tick logging for BrowserGym PPO training and evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


EPISODE_STEP_FIELDS = [
    "run_id", "batch_id", "site_id", "base_url", "episode_id", "step_id", "tick_id", "timestamp",
    "phase", "mode", "is_training", "is_evaluation",
    "before_raw_observation_keys", "after_raw_observation_keys", "before_raw_observation_key_count", "after_raw_observation_key_count",
    "before_url", "after_url", "url_changed",
    "before_title", "after_title", "title_changed",
    "before_text_hash", "after_text_hash", "text_changed", "before_text_length", "after_text_length", "text_delta_length",
    "before_candidate_count", "after_candidate_count", "candidate_count_delta",
    "viewport_width", "viewport_height",
    "selected_action_index", "selected_action_type", "selected_action_name", "selected_action_selector", "selected_action_text", "selected_action_role",
    "selected_action_bbox_x", "selected_action_bbox_y", "selected_action_bbox_width", "selected_action_bbox_height",
    "selected_action_bbox_center_x", "selected_action_bbox_center_y",
    "action_space_size", "valid_action_count", "invalid_action_count",
    "action_success", "action_error", "action_duration_ms",
    "network_request_count_before", "network_request_count_after", "network_error_count_before", "network_error_count_after", "network_status_codes",
    "console_error_count_before", "console_error_count_after", "console_errors",
    "layout_overlap_count", "layout_overflow_count", "duplicated_rendering_count",
    "port_open", "connection_refused", "health_check_ok", "health_status_code", "response_status", "response_latency_ms",
    "timeout_occurred", "server_5xx_count", "server_4xx_count", "server_log_exception_count", "process_alive",
    "cpu_usage_percent", "memory_usage_mb", "infra_anomaly_count", "infra_anomaly_types", "infra_reward_total",
    "detected_anomaly_count", "detected_anomaly_types", "detected_anomaly_ids", "anomaly_confidence_max", "anomaly_confidence_mean",
    "matched_bug_ids", "missed_bug_ids", "exploratory_anomaly_ids",
    "reward", "reward_bug_found", "reward_anomaly_found", "reward_penalty_repeat", "reward_penalty_invalid_action", "reward_penalty_noop", "reward_total",
    "done", "terminated", "truncated",
    "url_changed_diff", "text_changed_diff", "network_error_delta", "console_error_delta", "layout_overlap_delta", "layout_overflow_delta",
    "cart_state_changed", "form_state_changed", "success_message_appeared", "error_message_appeared", "validation_message_appeared", "loading_state_changed",
    "notes",
]

OBSERVATION_FIELDS = [
    "run_id", "batch_id", "site_id", "episode_id", "step_id", "tick_id", "timestamp",
    "url", "page_title", "viewport_width", "viewport_height",
    "candidate_count", "visible_candidate_count", "interactive_candidate_count",
    "candidate_index", "candidate_role", "candidate_text", "candidate_name", "candidate_tag", "candidate_selector", "candidate_href",
    "candidate_disabled", "candidate_visible", "candidate_bbox_x", "candidate_bbox_y", "candidate_bbox_width", "candidate_bbox_height",
    "candidate_bbox_center_x", "candidate_bbox_center_y", "candidate_in_viewport",
    "dom_text_length", "dom_hash", "page_text_hash",
    "layout_signal_count", "layout_overlap_count", "layout_overflow_count",
    "runtime_signal_count", "console_error_count", "network_error_count",
    "cart_state", "form_state", "loading_state", "error_message_visible", "success_message_visible", "validation_message_visible",
    "raw_observation_keys", "browsergym_raw_observation_keys", "browsergym_raw_observation_key_count", "browsergym_text_source",
    "browsergym_has_axtree", "browsergym_has_dom", "browsergym_has_screenshot",
    "infra_feature_name", "infra_feature_value", "health_endpoint", "health_response_status", "health_response_time_ms",
    "raw_observation_json",
]

ACTION_SPACE_FIELDS = [
    "run_id", "batch_id", "site_id", "episode_id", "step_id", "tick_id", "timestamp",
    "action_index", "action_type", "action_name", "action_description",
    "target_candidate_index", "target_selector", "target_text", "target_role", "target_tag",
    "target_bbox_x", "target_bbox_y", "target_bbox_width", "target_bbox_height",
    "is_valid", "is_enabled", "is_visible", "is_in_viewport",
    "policy_logit", "policy_probability", "selected", "selection_rank",
    "reason_if_invalid",
]

ANOMALY_FIELDS = [
    "run_id", "batch_id", "site_id", "episode_id", "step_id", "tick_id", "timestamp",
    "anomaly_id", "anomaly_type", "confidence", "severity",
    "before_url", "after_url", "target_selector", "target_text", "target_role",
    "target_bbox_x", "target_bbox_y", "target_bbox_width", "target_bbox_height",
    "evidence_summary", "evidence_json",
    "network_status", "network_error", "console_error", "layout_overlap", "layout_overflow", "duplicated_rendering",
    "matched_known_bug", "matched_bug_id", "match_score", "match_reason",
    "human_review_status", "likely_false_positive_reason",
    "infra_error_type", "infra_evidence_json",
]

REWARD_FIELDS = [
    "run_id", "batch_id", "site_id", "episode_id", "step_id", "tick_id", "timestamp",
    "action_type", "action_target",
    "reward_total", "reward_bug_found", "reward_anomaly_found", "reward_known_bug_match", "reward_exploration", "reward_state_change",
    "penalty_repeat_action", "penalty_invalid_action", "penalty_no_response", "penalty_noop", "penalty_timeout",
    "reward_infra_anomaly_found", "reward_server_timeout", "reward_server_5xx", "reward_port_closed", "penalty_false_positive_infra",
    "matched_bug_ids", "detected_anomaly_types", "reward_reason", "reward_breakdown_json",
]


class EpisodeCsvLogger:
    def __init__(
        self,
        output_dir: str | Path,
        run_id: str,
        batch_id: str = "",
        phase: str = "training",
        log_observation_detail: bool = True,
        log_action_space: bool = True,
        log_raw_json: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.batch_id = batch_id
        self.phase = phase
        self.log_observation_detail_enabled = log_observation_detail
        self.log_action_space_enabled = log_action_space
        self.log_raw_json = log_raw_json
        self._handles: Dict[str, Any] = {}
        self._writers: Dict[str, csv.DictWriter] = {}
        self._open("episode_step", "episode_step_logs.csv", EPISODE_STEP_FIELDS)
        self._open("observation", "observation_logs.csv", OBSERVATION_FIELDS)
        self._open("action_space", "action_space_logs.csv", ACTION_SPACE_FIELDS)
        self._open("anomaly", "anomaly_logs.csv", ANOMALY_FIELDS)
        self._open("reward", "reward_logs.csv", REWARD_FIELDS)

    def log_observation(
        self,
        site_id: str,
        episode_id: str,
        step_id: int,
        tick_id: str | int,
        observation: Mapping[str, Any],
    ) -> None:
        if not self.log_observation_detail_enabled:
            return
        timestamp = _timestamp()
        page_state = _mapping(observation.get("page_state"))
        raw_keys = _raw_observation_keys(observation)
        infra = _mapping(observation.get("infra_signals"))
        candidates = _candidate_list(observation)
        candidate_count = len(candidates)
        base = {
            "run_id": self.run_id,
            "batch_id": self.batch_id,
            "site_id": site_id,
            "episode_id": episode_id,
            "step_id": step_id,
            "tick_id": tick_id,
            "timestamp": timestamp,
            "url": page_state.get("url", ""),
            "page_title": page_state.get("title", ""),
            "viewport_width": page_state.get("viewport_width", ""),
            "viewport_height": page_state.get("viewport_height", ""),
            "candidate_count": candidate_count,
            "visible_candidate_count": sum(1 for item in candidates if _visible(item)),
            "interactive_candidate_count": sum(1 for item in candidates if _interactive(item)),
            "dom_text_length": page_state.get("dom_text_length", page_state.get("page_text_length", "")),
            "dom_hash": _hash(_page_text(observation)),
            "page_text_hash": _hash(_page_text(observation)),
            "layout_signal_count": len(_mapping(observation.get("layout_signals"))),
            "layout_overlap_count": _layout_count(observation, "layout_overlap_count"),
            "layout_overflow_count": _layout_count(observation, "layout_overflow_count"),
            "runtime_signal_count": len(_mapping(observation.get("runtime_signals"))),
            "console_error_count": _console_error_count(observation),
            "network_error_count": _network_error_count(observation),
            "cart_state": _cart_state(observation),
            "form_state": _form_state(observation),
            "loading_state": _loading_state(observation),
            "error_message_visible": _message_visible(observation, ("error", "failed", "forbidden")),
            "success_message_visible": _message_visible(observation, ("success", "complete", "added")),
            "validation_message_visible": _message_visible(observation, ("required", "invalid", "validation")),
            "raw_observation_keys": ";".join(raw_keys),
            "browsergym_raw_observation_keys": ";".join(raw_keys),
            "browsergym_raw_observation_key_count": len(raw_keys),
            "browsergym_text_source": page_state.get("browsergym_text_source", ""),
            "browsergym_has_axtree": page_state.get("browsergym_has_axtree", ""),
            "browsergym_has_dom": page_state.get("browsergym_has_dom", ""),
            "browsergym_has_screenshot": page_state.get("browsergym_has_screenshot", ""),
            "infra_feature_name": "",
            "infra_feature_value": "",
            "health_endpoint": infra.get("health_endpoint", ""),
            "health_response_status": infra.get("health_status_code", ""),
            "health_response_time_ms": infra.get("health_response_time_ms", infra.get("response_latency_ms", "")),
            "raw_observation_json": _json(observation, limit=12000) if self.log_raw_json else "",
        }
        if not candidates:
            row = dict(base)
            row.update(_candidate_row("", {}))
            self._write("observation", row)
        for index, candidate in enumerate(candidates):
            row = dict(base)
            row.update(_candidate_row(index, candidate))
            self._write("observation", row)
        for name, value in infra.items():
            row = dict(base)
            row.update(_candidate_row("", {}))
            row["infra_feature_name"] = name
            row["infra_feature_value"] = value
            self._write("observation", row)

    def log_action_space(
        self,
        site_id: str,
        episode_id: str,
        step_id: int,
        tick_id: str | int,
        observation: Mapping[str, Any],
        action_space: Any,
        action_mask: Sequence[float],
        selected_action_id: int | None,
    ) -> None:
        if not self.log_action_space_enabled:
            return
        timestamp = _timestamp()
        mask = np.asarray(action_mask, dtype=np.float32)
        action_dim = int(action_space.get_action_dim())
        action_indices = [index for index in range(min(action_dim, len(mask))) if mask[index] > 0]
        if selected_action_id is not None and 0 <= int(selected_action_id) < action_dim and int(selected_action_id) not in action_indices:
            action_indices.append(int(selected_action_id))
        for action_index in action_indices:
            decoded = action_space.decode(action_index)
            candidate_index = int(decoded.get("candidate_index", 0) or 0)
            candidate = _candidate_at(observation, candidate_index) if action_space.is_element_action(str(decoded.get("action_type"))) else {}
            bbox = _bbox(candidate)
            is_valid = bool(action_index < len(mask) and mask[action_index] > 0)
            self._write(
                "action_space",
                {
                    "run_id": self.run_id,
                    "batch_id": self.batch_id,
                    "site_id": site_id,
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "tick_id": tick_id,
                    "timestamp": timestamp,
                    "action_index": action_index,
                    "action_type": decoded.get("action_type", ""),
                    "action_name": decoded.get("action_type", ""),
                    "action_description": _action_description(decoded, candidate),
                    "target_candidate_index": candidate_index if candidate else "",
                    "target_selector": _selector(candidate),
                    "target_text": _candidate_text(candidate),
                    "target_role": candidate.get("role", "") if isinstance(candidate, Mapping) else "",
                    "target_tag": candidate.get("tag", "") if isinstance(candidate, Mapping) else "",
                    "target_bbox_x": bbox[0],
                    "target_bbox_y": bbox[1],
                    "target_bbox_width": bbox[2],
                    "target_bbox_height": bbox[3],
                    "is_valid": is_valid,
                    "is_enabled": not bool(candidate.get("disabled")) if isinstance(candidate, Mapping) else is_valid,
                    "is_visible": _visible(candidate) if candidate else is_valid,
                    "is_in_viewport": _in_viewport(candidate, observation) if candidate else is_valid,
                    "policy_logit": "",
                    "policy_probability": "",
                    "selected": selected_action_id is not None and int(selected_action_id) == action_index,
                    "selection_rank": 1 if selected_action_id is not None and int(selected_action_id) == action_index else "",
                    "reason_if_invalid": "" if is_valid else "masked_by_action_space",
                },
            )

    def log_step(
        self,
        site_id: str,
        base_url: str,
        episode_id: str,
        step_id: int,
        tick_id: str | int,
        before_observation: Mapping[str, Any],
        after_observation: Mapping[str, Any],
        action: Mapping[str, Any],
        action_mask: Sequence[float],
        step_info: Mapping[str, Any],
        anomalies: Sequence[Mapping[str, Any]],
        known_matches: Sequence[Mapping[str, Any]],
        reward: float,
        reward_breakdown: Mapping[str, Any],
        done: bool,
        mode: str = "BrowserGym PPO",
        notes: str = "",
    ) -> list[str]:
        timestamp = _timestamp()
        anomaly_ids = [
            str(item.get("anomaly_id") or f"{episode_id}-S{step_id:04d}-A{index:03d}")
            for index, item in enumerate(anomalies, start=1)
        ]
        confidences = [float(item.get("confidence", 0.0) or 0.0) for item in anomalies]
        matched_bug_ids = _matched_bug_ids(anomalies, known_matches)
        target = _selected_target(before_observation, action)
        bbox = _bbox(target)
        before_state = _mapping(before_observation.get("page_state"))
        after_state = _mapping(after_observation.get("page_state"))
        infra = _mapping(after_observation.get("infra_signals"))
        infra_anomalies = [item for item in anomalies if str(item.get("type") or "").startswith("server-")]
        diff = build_transition_diff(before_observation, after_observation)
        network_codes = _network_status_codes(before_observation, after_observation, step_info)
        mask = np.asarray(action_mask, dtype=np.float32)
        action_error = _action_error(step_info)
        row = {
            "run_id": self.run_id,
            "batch_id": self.batch_id,
            "site_id": site_id,
            "base_url": base_url,
            "episode_id": episode_id,
            "step_id": step_id,
            "tick_id": tick_id,
            "timestamp": timestamp,
            "phase": self.phase,
            "mode": mode,
            "is_training": self.phase == "training",
            "is_evaluation": self.phase == "evaluation",
            "before_raw_observation_keys": ";".join(_raw_observation_keys(before_observation)),
            "after_raw_observation_keys": ";".join(_raw_observation_keys(after_observation)),
            "before_raw_observation_key_count": len(_raw_observation_keys(before_observation)),
            "after_raw_observation_key_count": len(_raw_observation_keys(after_observation)),
            "before_url": before_state.get("url", ""),
            "after_url": after_state.get("url", ""),
            "url_changed": diff["url_changed"],
            "before_title": before_state.get("title", ""),
            "after_title": after_state.get("title", ""),
            "title_changed": before_state.get("title", "") != after_state.get("title", ""),
            "before_text_hash": _hash(_page_text(before_observation)),
            "after_text_hash": _hash(_page_text(after_observation)),
            "text_changed": diff["text_changed"],
            "before_text_length": _text_length(before_observation),
            "after_text_length": _text_length(after_observation),
            "text_delta_length": diff["text_delta_length"],
            "before_candidate_count": len(_candidate_list(before_observation)),
            "after_candidate_count": len(_candidate_list(after_observation)),
            "candidate_count_delta": diff["candidate_count_delta"],
            "viewport_width": before_state.get("viewport_width", ""),
            "viewport_height": before_state.get("viewport_height", ""),
            "selected_action_index": action.get("action_id", ""),
            "selected_action_type": action.get("action_type", ""),
            "selected_action_name": action.get("action_type", ""),
            "selected_action_selector": _selector(target) or action.get("selector_hint", ""),
            "selected_action_text": _candidate_text(target) or _action_target(action),
            "selected_action_role": target.get("role", "") if isinstance(target, Mapping) else "",
            "selected_action_bbox_x": bbox[0],
            "selected_action_bbox_y": bbox[1],
            "selected_action_bbox_width": bbox[2],
            "selected_action_bbox_height": bbox[3],
            "selected_action_bbox_center_x": _center(bbox)[0],
            "selected_action_bbox_center_y": _center(bbox)[1],
            "action_space_size": len(mask),
            "valid_action_count": int(mask.sum()) if len(mask) else 0,
            "invalid_action_count": int(len(mask) - mask.sum()) if len(mask) else 0,
            "action_success": not bool(action.get("failed")) and not bool(step_info.get("last_action_error")),
            "action_error": action_error,
            "action_duration_ms": step_info.get("action_duration_ms", step_info.get("duration_ms", "")),
            "network_request_count_before": _network_request_count(before_observation),
            "network_request_count_after": _network_request_count(after_observation),
            "network_error_count_before": _network_error_count(before_observation),
            "network_error_count_after": _network_error_count(after_observation),
            "network_status_codes": _json(network_codes, limit=1000),
            "console_error_count_before": _console_error_count(before_observation),
            "console_error_count_after": _console_error_count(after_observation),
            "console_errors": _json(_console_errors(before_observation, after_observation), limit=1000),
            "layout_overlap_count": _layout_count(after_observation, "layout_overlap_count"),
            "layout_overflow_count": _layout_count(after_observation, "layout_overflow_count"),
            "duplicated_rendering_count": sum(1 for item in anomalies if item.get("type") == "duplicated-rendering"),
            "port_open": infra.get("port_open", ""),
            "connection_refused": infra.get("connection_refused", ""),
            "health_check_ok": infra.get("health_check_ok", ""),
            "health_status_code": infra.get("health_status_code", ""),
            "response_status": infra.get("response_status", ""),
            "response_latency_ms": infra.get("response_latency_ms", ""),
            "timeout_occurred": infra.get("timeout_occurred", ""),
            "server_5xx_count": infra.get("server_5xx_count", ""),
            "server_4xx_count": infra.get("server_4xx_count", ""),
            "server_log_exception_count": infra.get("server_log_exception_count", ""),
            "process_alive": infra.get("process_alive", ""),
            "cpu_usage_percent": infra.get("cpu_usage_percent", ""),
            "memory_usage_mb": infra.get("memory_usage_mb", ""),
            "infra_anomaly_count": len(infra_anomalies),
            "infra_anomaly_types": ";".join(str(item.get("type") or "") for item in infra_anomalies),
            "infra_reward_total": reward_breakdown.get("infra_reward_total", 0.0),
            "detected_anomaly_count": len(anomalies),
            "detected_anomaly_types": ";".join(str(item.get("type") or "") for item in anomalies),
            "detected_anomaly_ids": ";".join(anomaly_ids),
            "anomaly_confidence_max": max(confidences) if confidences else "",
            "anomaly_confidence_mean": mean(confidences) if confidences else "",
            "matched_bug_ids": ";".join(matched_bug_ids),
            "missed_bug_ids": "",
            "exploratory_anomaly_ids": ";".join(
                anomaly_id for anomaly_id, item in zip(anomaly_ids, anomalies) if not item.get("matched_bug_id")
            ),
            "reward": reward,
            "reward_bug_found": reward_breakdown.get("known_bug_reward", 0.0),
            "reward_anomaly_found": float(reward_breakdown.get("anomaly_reward", 0.0) or 0.0)
            + float(reward_breakdown.get("error_signal_reward", 0.0) or 0.0),
            "reward_penalty_repeat": reward_breakdown.get("repetition_penalty", 0.0),
            "reward_penalty_invalid_action": reward_breakdown.get("penalty", 0.0) if action.get("failed") or action.get("invalid") else 0.0,
            "reward_penalty_noop": reward_breakdown.get("penalty", 0.0) if action.get("action_type") == "noop" else 0.0,
            "reward_total": reward_breakdown.get("scaled_final_reward", reward_breakdown.get("final_reward", reward)),
            "done": done,
            "terminated": done,
            "truncated": False,
            "notes": notes,
            **{f"{key}_diff" if key in {"url_changed", "text_changed"} else key: value for key, value in diff.items()},
        }
        self._write("episode_step", row)
        self.log_anomalies(site_id, episode_id, step_id, tick_id, before_observation, after_observation, anomalies, known_matches, anomaly_ids)
        self.log_reward(site_id, episode_id, step_id, tick_id, action, anomalies, known_matches, reward, reward_breakdown)
        return anomaly_ids

    def log_anomalies(
        self,
        site_id: str,
        episode_id: str,
        step_id: int,
        tick_id: str | int,
        before_observation: Mapping[str, Any],
        after_observation: Mapping[str, Any],
        anomalies: Sequence[Mapping[str, Any]],
        known_matches: Sequence[Mapping[str, Any]],
        anomaly_ids: Sequence[str] | None = None,
    ) -> None:
        timestamp = _timestamp()
        before_url = _mapping(before_observation.get("page_state")).get("url", "")
        after_url = _mapping(after_observation.get("page_state")).get("url", "")
        match_by_type = {str(item.get("type") or ""): item for item in known_matches}
        for index, anomaly in enumerate(anomalies, start=1):
            evidence = _mapping(anomaly.get("evidence"))
            target = evidence.get("target") if isinstance(evidence.get("target"), Mapping) else {}
            if not target:
                target = _target_from_evidence(before_observation, evidence)
            bbox = _bbox(target)
            anomaly_id = str((anomaly_ids or [])[index - 1]) if anomaly_ids and len(anomaly_ids) >= index else f"{episode_id}-S{step_id:04d}-A{index:03d}"
            match = match_by_type.get(str(anomaly.get("type") or ""), {})
            self._write(
                "anomaly",
                {
                    "run_id": self.run_id,
                    "batch_id": self.batch_id,
                    "site_id": site_id,
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "tick_id": tick_id,
                    "timestamp": timestamp,
                    "anomaly_id": anomaly_id,
                    "anomaly_type": anomaly.get("type", ""),
                    "confidence": anomaly.get("confidence", ""),
                    "severity": anomaly.get("severity", _severity(anomaly)),
                    "before_url": before_url,
                    "after_url": after_url,
                    "target_selector": _selector(target) or evidence.get("selector", ""),
                    "target_text": _candidate_text(target) or evidence.get("clicked_text", "") or evidence.get("candidate_text", ""),
                    "target_role": target.get("role", "") if isinstance(target, Mapping) else "",
                    "target_bbox_x": bbox[0],
                    "target_bbox_y": bbox[1],
                    "target_bbox_width": bbox[2],
                    "target_bbox_height": bbox[3],
                    "evidence_summary": _evidence_summary(evidence),
                    "evidence_json": _json(evidence, limit=6000),
                    "network_status": evidence.get("network_status", ""),
                    "network_error": evidence.get("network_error", ""),
                    "console_error": evidence.get("console_error", ""),
                    "layout_overlap": evidence.get("layout_overlap_count", ""),
                    "layout_overflow": evidence.get("layout_overflow_count", ""),
                    "duplicated_rendering": anomaly.get("type") == "duplicated-rendering",
                    "matched_known_bug": bool(anomaly.get("matched_bug_id") or match.get("matched_bug_id")),
                    "matched_bug_id": anomaly.get("matched_bug_id", match.get("matched_bug_id", "")),
                    "match_score": anomaly.get("match_score", match.get("match_score", "")),
                    "match_reason": anomaly.get("match_reason", match.get("match_reason", "")),
                    "human_review_status": anomaly.get("human_review_status", ""),
                    "likely_false_positive_reason": anomaly.get("likely_false_positive_reason", ""),
                    "infra_error_type": anomaly.get("type", "") if str(anomaly.get("type") or "").startswith("server-") else "",
                    "infra_evidence_json": _json(evidence.get("infra", evidence), limit=4000)
                    if str(anomaly.get("type") or "").startswith("server-")
                    else "",
                },
            )

    def log_reward(
        self,
        site_id: str,
        episode_id: str,
        step_id: int,
        tick_id: str | int,
        action: Mapping[str, Any],
        anomalies: Sequence[Mapping[str, Any]],
        known_matches: Sequence[Mapping[str, Any]],
        reward: float,
        reward_breakdown: Mapping[str, Any],
    ) -> None:
        matched_bug_ids = _matched_bug_ids(anomalies, known_matches)
        self._write(
            "reward",
            {
                "run_id": self.run_id,
                "batch_id": self.batch_id,
                "site_id": site_id,
                "episode_id": episode_id,
                "step_id": step_id,
                "tick_id": tick_id,
                "timestamp": _timestamp(),
                "action_type": action.get("action_type", ""),
                "action_target": _action_target(action),
                "reward_total": reward_breakdown.get("scaled_final_reward", reward_breakdown.get("final_reward", reward)),
                "reward_bug_found": reward_breakdown.get("known_bug_reward", 0.0),
                "reward_anomaly_found": reward_breakdown.get("anomaly_reward", 0.0),
                "reward_known_bug_match": reward_breakdown.get("known_bug_reward", 0.0),
                "reward_exploration": reward_breakdown.get("exploration_reward", 0.0),
                "reward_state_change": reward_breakdown.get("error_signal_reward", 0.0),
                "penalty_repeat_action": reward_breakdown.get("repetition_penalty", 0.0),
                "penalty_invalid_action": reward_breakdown.get("penalty", 0.0) if action.get("failed") or action.get("invalid") else 0.0,
                "penalty_no_response": reward_breakdown.get("penalty", 0.0)
                if any(item.get("type") == "button-no-response" for item in anomalies)
                else 0.0,
                "penalty_noop": reward_breakdown.get("penalty", 0.0) if action.get("action_type") == "noop" else 0.0,
                "penalty_timeout": reward_breakdown.get("penalty", 0.0)
                if any("timeout" in str(item.get("type") or "") for item in anomalies)
                else 0.0,
                "reward_infra_anomaly_found": reward_breakdown.get("reward_infra_anomaly_found", 0.0),
                "reward_server_timeout": reward_breakdown.get("reward_server_timeout", 0.0),
                "reward_server_5xx": reward_breakdown.get("reward_server_5xx", 0.0),
                "reward_port_closed": reward_breakdown.get("reward_port_closed", 0.0),
                "penalty_false_positive_infra": reward_breakdown.get("penalty_false_positive_infra", 0.0),
                "matched_bug_ids": ";".join(matched_bug_ids),
                "detected_anomaly_types": ";".join(str(item.get("type") or "") for item in anomalies),
                "reward_reason": _reward_reason(reward_breakdown, anomalies, matched_bug_ids),
                "reward_breakdown_json": _json(reward_breakdown, limit=4000),
            },
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._writers.clear()

    def _open(self, key: str, filename: str, fields: list[str]) -> None:
        path = self.output_dir / filename
        needs_header = not path.exists() or path.stat().st_size == 0
        handle = path.open("a", encoding="utf-8-sig", newline="")
        writer = csv.DictWriter(handle, fieldnames=fields, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
            handle.flush()
        self._handles[key] = handle
        self._writers[key] = writer

    def _write(self, key: str, row: Mapping[str, Any]) -> None:
        self._writers[key].writerow({field: _cell(row.get(field, "")) for field in self._writers[key].fieldnames or []})
        self._handles[key].flush()

    def log_anomaly(
        self,
        site_id: str,
        episode_id: str,
        step_id: int,
        tick_id: str | int,
        before_observation: Mapping[str, Any],
        after_observation: Mapping[str, Any],
        anomalies: Sequence[Mapping[str, Any]],
        known_matches: Sequence[Mapping[str, Any]],
        anomaly_ids: Sequence[str] | None = None,
    ) -> None:
        self.log_anomalies(
            site_id,
            episode_id,
            step_id,
            tick_id,
            before_observation,
            after_observation,
            anomalies,
            known_matches,
            anomaly_ids,
        )


def build_transition_diff(before_obs: Mapping[str, Any], after_obs: Mapping[str, Any]) -> Dict[str, Any]:
    before_text = _page_text(before_obs)
    after_text = _page_text(after_obs)
    before_form = _form_state(before_obs)
    after_form = _form_state(after_obs)
    before_loading = _loading_state(before_obs)
    after_loading = _loading_state(after_obs)
    return {
        "url_changed": _url(before_obs) != _url(after_obs),
        "text_changed": _hash(before_text) != _hash(after_text),
        "text_delta_length": len(after_text) - len(before_text),
        "candidate_count_delta": len(_candidate_list(after_obs)) - len(_candidate_list(before_obs)),
        "network_error_delta": _network_error_count(after_obs) - _network_error_count(before_obs),
        "console_error_delta": _console_error_count(after_obs) - _console_error_count(before_obs),
        "layout_overlap_delta": _layout_count(after_obs, "layout_overlap_count") - _layout_count(before_obs, "layout_overlap_count"),
        "layout_overflow_delta": _layout_count(after_obs, "layout_overflow_count") - _layout_count(before_obs, "layout_overflow_count"),
        "cart_state_changed": _cart_state(before_obs) != _cart_state(after_obs),
        "form_state_changed": before_form != after_form,
        "success_message_appeared": not _message_visible(before_obs, ("success", "complete", "added")) and _message_visible(after_obs, ("success", "complete", "added")),
        "error_message_appeared": not _message_visible(before_obs, ("error", "failed", "forbidden")) and _message_visible(after_obs, ("error", "failed", "forbidden")),
        "validation_message_appeared": not _message_visible(before_obs, ("required", "invalid", "validation")) and _message_visible(after_obs, ("required", "invalid", "validation")),
        "loading_state_changed": before_loading != after_loading,
    }


def infer_run_id(config: Mapping[str, Any], config_path: str | Path, fallback_prefix: str = "run") -> str:
    port_range = config.get("port_range") if isinstance(config.get("port_range"), Mapping) else {}
    start_port = port_range.get("start_port")
    end_port = port_range.get("end_port")
    if start_port is not None and end_port is not None:
        return f"ports_{start_port}_{end_port}"
    path = Path(config_path)
    batch_id = path.stem
    if batch_id:
        return f"{_timestamp_slug()}_{batch_id}"
    return f"{_timestamp_slug()}_{fallback_prefix}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _candidate_list(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    return [item for item in candidates if isinstance(item, Mapping)] if isinstance(candidates, list) else []


def _raw_observation_keys(observation: Mapping[str, Any]) -> list[str]:
    keys = observation.get("browsergym_raw_observation_keys", []) if isinstance(observation, Mapping) else []
    if isinstance(keys, list):
        return [str(item) for item in keys]
    if isinstance(keys, str):
        return [item for item in keys.split(";") if item]
    return []


def _candidate_at(observation: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    candidates = _candidate_list(observation)
    return candidates[index] if 0 <= index < len(candidates) else {}


def _selected_target(observation: Mapping[str, Any], action: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return _candidate_at(observation, int(action.get("candidate_index", 0) or 0))
    except (TypeError, ValueError):
        return {}


def _candidate_row(index: str | int, candidate: Mapping[str, Any]) -> Dict[str, Any]:
    bbox = _bbox(candidate)
    center = _center(bbox)
    return {
        "candidate_index": index,
        "candidate_role": candidate.get("role", ""),
        "candidate_text": _candidate_text(candidate),
        "candidate_name": candidate.get("name", ""),
        "candidate_tag": candidate.get("tag", ""),
        "candidate_selector": _selector(candidate),
        "candidate_href": candidate.get("href", ""),
        "candidate_disabled": candidate.get("disabled", ""),
        "candidate_visible": _visible(candidate) if candidate else "",
        "candidate_bbox_x": bbox[0],
        "candidate_bbox_y": bbox[1],
        "candidate_bbox_width": bbox[2],
        "candidate_bbox_height": bbox[3],
        "candidate_bbox_center_x": center[0],
        "candidate_bbox_center_y": center[1],
        "candidate_in_viewport": bool(candidate.get("in_viewport", candidate.get("is_in_viewport", ""))) if candidate else "",
    }


def _bbox(candidate: Mapping[str, Any]) -> list[Any]:
    raw = candidate.get("bbox", []) if isinstance(candidate, Mapping) else []
    if isinstance(raw, Mapping):
        values = [raw.get("x", 0), raw.get("y", 0), raw.get("width", raw.get("w", 0)), raw.get("height", raw.get("h", 0))]
    elif isinstance(raw, list) and len(raw) >= 4:
        values = raw[:4]
    else:
        values = ["", "", "", ""]
    return values


def _center(bbox: Sequence[Any]) -> tuple[Any, Any]:
    try:
        return (float(bbox[0]) + float(bbox[2]) / 2.0, float(bbox[1]) + float(bbox[3]) / 2.0)
    except (TypeError, ValueError):
        return ("", "")


def _visible(candidate: Mapping[str, Any]) -> bool:
    if not isinstance(candidate, Mapping):
        return False
    if "visible" in candidate:
        return bool(candidate.get("visible"))
    return float(candidate.get("visibility", 1.0) or 0.0) > 0.0


def _interactive(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("is_interactive") or candidate.get("clickable") or candidate.get("role") in {"button", "link", "textbox"})


def _in_viewport(candidate: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    if candidate.get("in_viewport") is not None:
        return bool(candidate.get("in_viewport"))
    bbox = _bbox(candidate)
    page = _mapping(observation.get("page_state"))
    try:
        x, y, w, h = [float(value) for value in bbox]
        vw = float(page.get("viewport_width", 0) or 0)
        vh = float(page.get("viewport_height", 0) or 0)
        return w > 0 and h > 0 and x + w >= 0 and y + h >= 0 and (not vw or x <= vw) and (not vh or y <= vh)
    except (TypeError, ValueError):
        return False


def _selector(candidate: Mapping[str, Any]) -> str:
    if not isinstance(candidate, Mapping):
        return ""
    return str(candidate.get("selector") or candidate.get("selector_hint") or candidate.get("data_bug_id") or candidate.get("bid") or "")


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    if not isinstance(candidate, Mapping):
        return ""
    return str(candidate.get("text") or candidate.get("name") or candidate.get("label") or "")[:500]


def _page_text(observation: Mapping[str, Any]) -> str:
    page = _mapping(observation.get("page_state"))
    return str(page.get("page_text") or page.get("page_text_sample") or page.get("text") or "")


def _text_length(observation: Mapping[str, Any]) -> int:
    page = _mapping(observation.get("page_state"))
    value = page.get("page_text_length")
    try:
        return int(value)
    except (TypeError, ValueError):
        return len(_page_text(observation))


def _url(observation: Mapping[str, Any]) -> str:
    return str(_mapping(observation.get("page_state")).get("url", "") or "")


def _layout_count(observation: Mapping[str, Any], key: str) -> int:
    layout = _mapping(observation.get("layout_signals"))
    try:
        return int(layout.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _network_error_count(observation: Mapping[str, Any]) -> int:
    runtime = _mapping(observation.get("runtime_signals"))
    for key in ("network_error_count", "network_errors", "api_error_count"):
        value = runtime.get(key)
        if isinstance(value, list):
            return len(value)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            pass
    return 0


def _network_request_count(observation: Mapping[str, Any]) -> int:
    runtime = _mapping(observation.get("runtime_signals"))
    for key in ("network_request_count", "request_count", "network_entries"):
        value = runtime.get(key)
        if isinstance(value, list):
            return len(value)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            pass
    return 0


def _network_status_codes(before: Mapping[str, Any], after: Mapping[str, Any], step_info: Mapping[str, Any]) -> list[Any]:
    values: list[Any] = []
    for source in (_mapping(before.get("runtime_signals")), _mapping(after.get("runtime_signals")), step_info):
        for key in ("network_status_codes", "status_codes"):
            value = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(value, list):
                values.extend(value)
            elif value not in (None, ""):
                values.append(value)
        entries = source.get("network_entries") if isinstance(source, Mapping) else None
        if isinstance(entries, list):
            for entry in entries[:20]:
                if isinstance(entry, Mapping) and entry.get("status") is not None:
                    values.append(entry.get("status"))
    return values


def _console_error_count(observation: Mapping[str, Any]) -> int:
    runtime = _mapping(observation.get("runtime_signals"))
    for key in ("console_error_count", "console_errors"):
        value = runtime.get(key)
        if isinstance(value, list):
            return len(value)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            pass
    return 0


def _console_errors(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[Any]:
    values: list[Any] = []
    for obs in (before, after):
        runtime = _mapping(obs.get("runtime_signals"))
        errors = runtime.get("console_errors", [])
        if isinstance(errors, list):
            values.extend(errors[:20])
    return values


def _cart_state(observation: Mapping[str, Any]) -> str:
    runtime = _mapping(observation.get("runtime_signals"))
    page = _mapping(observation.get("page_state"))
    return str(runtime.get("cart_text") or page.get("cart_text") or runtime.get("cart_count") or page.get("cart_count") or "")


def _form_state(observation: Mapping[str, Any]) -> str:
    candidates = _candidate_list(observation)
    return str(sum(1 for item in candidates if str(item.get("tag", "")).lower() in {"input", "textarea", "select", "form"}))


def _loading_state(observation: Mapping[str, Any]) -> str:
    runtime = _mapping(observation.get("runtime_signals"))
    page = _mapping(observation.get("page_state"))
    return str(runtime.get("loading") or page.get("loading") or _message_visible(observation, ("loading", "spinner")))


def _message_visible(observation: Mapping[str, Any], tokens: Iterable[str]) -> bool:
    text = _page_text(observation).lower()
    return any(token in text for token in tokens)


def _action_description(decoded: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    action_type = str(decoded.get("action_type") or "")
    if candidate:
        return f"{action_type}:{_candidate_text(candidate) or _selector(candidate)}"
    return action_type


def _action_target(action: Mapping[str, Any]) -> str:
    for key in ("clicked_text", "clicked_name", "clicked_bid", "input_label", "target_text", "selector_hint"):
        if action.get(key):
            return str(action.get(key))
    return str(action.get("candidate_index", ""))


def _action_error(step_info: Mapping[str, Any]) -> str:
    for key in ("action_error", "fallback_error", "error", "last_action_error_message"):
        value = step_info.get(key)
        if value:
            return str(value)
    return str(step_info.get("last_action_error") or "")


def _matched_bug_ids(anomalies: Sequence[Mapping[str, Any]], known_matches: Sequence[Mapping[str, Any]]) -> list[str]:
    values = {
        str(item.get("matched_bug_id"))
        for item in list(anomalies) + list(known_matches)
        if item.get("matched_bug_id")
    }
    return sorted(values)


def _target_from_evidence(observation: Mapping[str, Any], evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    target_bid = str(evidence.get("clicked_bid") or evidence.get("bid") or "")
    target_text = str(evidence.get("clicked_text") or evidence.get("candidate_text") or "")
    for candidate in _candidate_list(observation):
        if target_bid and str(candidate.get("bid") or "") == target_bid:
            return candidate
        if target_text and target_text in _candidate_text(candidate):
            return candidate
    return {}


def _evidence_summary(evidence: Mapping[str, Any]) -> str:
    parts = []
    for key in ("clicked_text", "clicked_name", "before_url", "after_url", "page_text_delta", "candidate_delta", "error", "network_status"):
        if evidence.get(key) not in (None, ""):
            parts.append(f"{key}={evidence.get(key)}")
    return ", ".join(parts)[:1000]


def _severity(anomaly: Mapping[str, Any]) -> str:
    confidence = float(anomaly.get("confidence", 0.0) or 0.0)
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.6:
        return "Medium"
    return "Low"


def _reward_reason(reward_breakdown: Mapping[str, Any], anomalies: Sequence[Mapping[str, Any]], matched_bug_ids: Sequence[str]) -> str:
    if matched_bug_ids:
        return "known_bug_match"
    if anomalies:
        return "anomaly_detected"
    positives = [key for key, value in reward_breakdown.items() if isinstance(value, (int, float)) and value > 0]
    return ",".join(positives[:5])


def _hash(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def _json(value: Any, limit: int = 4000) -> str:
    try:
        text = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    return text[:limit]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _cell(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple, set)):
        return _json(value)
    return value


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
