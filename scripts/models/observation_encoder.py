"""Numeric encoder for J.A.W.S raw observations."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


ROLE_ORDER = ["button", "link", "textbox", "combobox", "checkbox", "radio", "menuitem", "tab"]


class ObservationEncoder:
    page_feature_dim = 9
    raw_summary_feature_dim = 9
    candidate_feature_dim = len(ROLE_ORDER) + 12
    runtime_feature_dim = 3
    layout_feature_dim = 1
    infra_feature_dim = 11
    history_feature_dim = 3

    def __init__(self, max_candidates: int = 32) -> None:
        self.max_candidates = max_candidates

    def encode_observation(self, raw_observation: Mapping[str, Any]) -> np.ndarray:
        observation = self._structured_observation(raw_observation)
        features = []
        features.extend(self._page_features(observation.get("page_state", {})))
        features.extend(self._raw_summary_features(observation))

        candidates = observation.get("candidate_elements", [])
        if not isinstance(candidates, list):
            candidates = []
        for idx in range(self.max_candidates):
            if idx < len(candidates) and isinstance(candidates[idx], Mapping):
                features.extend(self._candidate_features(candidates[idx]))
            else:
                features.extend([0.0] * self.candidate_feature_dim)

        features.extend(self._runtime_features(observation.get("runtime_signals", {})))
        features.extend(self._layout_features(observation.get("layout_signals", {})))
        features.extend(self._infra_features(observation.get("infra_signals", {})))
        features.extend(self._history_features(observation.get("history", {})))

        vector = np.asarray(features, dtype=np.float32)
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def get_obs_dim(self) -> int:
        return (
            self.page_feature_dim
            + self.raw_summary_feature_dim
            + self.max_candidates * self.candidate_feature_dim
            + self.runtime_feature_dim
            + self.layout_feature_dim
            + self.infra_feature_dim
            + self.history_feature_dim
        )


    def _structured_observation(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(observation, Mapping) and isinstance(observation.get("structured_observation"), Mapping):
            return observation["structured_observation"]
        return observation if isinstance(observation, Mapping) else {}

    def _raw_summary_features(self, observation: Mapping[str, Any]) -> list[float]:
        summary = observation.get("raw_observation_summary") or observation.get("browsergym_raw_observation") or {}
        if not isinstance(summary, Mapping):
            summary = {}
        derived = observation.get("derived_features", {})
        if not isinstance(derived, Mapping):
            derived = {}
        return [
            _scale(len(summary.get("keys", []) or observation.get("raw_observation_keys", []) or []), 64.0),
            _scale(summary.get("text_length"), 20000.0),
            _scale(derived.get("candidate_count"), float(max(1, self.max_candidates))),
            _scale(derived.get("visible_candidate_count"), float(max(1, self.max_candidates))),
            _bool(derived.get("has_error_text")),
            _bool(derived.get("has_forbidden_text")),
            _bool(derived.get("has_timeout_text")),
            _bool(derived.get("has_network_error_text")),
            _bool(summary.get("has_screenshot") or derived.get("screenshot_available")),
        ]

    def _page_features(self, page_state: Mapping[str, Any]) -> list[float]:
        return [
            _scale(page_state.get("viewport_width"), 4096.0),
            _scale(page_state.get("viewport_height"), 4096.0),
            _scale(page_state.get("page_text_length"), 20000.0),
            _scale(page_state.get("dom_node_count"), 5000.0),
            _scale(page_state.get("elapsed_time"), 300.0),
            1.0 if page_state.get("url") else 0.0,
            1.0 if page_state.get("viewport_type") == "mobile" else 0.0,
            _bool(page_state.get("cart_count_detected")),
            _scale(page_state.get("cart_count"), 100.0),
        ]

    def _candidate_features(self, candidate: Mapping[str, Any]) -> list[float]:
        role = str(candidate.get("role", "")).lower()
        bbox = candidate.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(bbox, (list, tuple)):
            bbox = [0.0, 0.0, 0.0, 0.0]
        bbox = list(bbox[:4]) + [0.0] * max(0, 4 - len(bbox))

        features = [1.0 if role == expected else 0.0 for expected in ROLE_ORDER]
        features.extend(
            [
                _bool(candidate.get("visible")),
                _bool(candidate.get("enabled")),
                _bool(candidate.get("clickable")),
                _clamp(_float(candidate.get("visibility"))),
                _scale(bbox[0], 4096.0),
                _scale(bbox[1], 4096.0),
                _scale(bbox[2], 4096.0),
                _scale(bbox[3], 4096.0),
                _bool(candidate.get("has_text")),
                _scale(candidate.get("text_length"), 512.0),
                _bool(candidate.get("has_data_bug_id")),
                _bool(candidate.get("has_data_testid")),
            ]
        )
        return features

    def _runtime_features(self, signals: Mapping[str, Any]) -> list[float]:
        return [
            _bool(signals.get("url_changed")),
            _bool(signals.get("last_action_error")),
            _scale(signals.get("elapsed_time"), 300.0),
        ]

    def _layout_features(self, signals: Mapping[str, Any]) -> list[float]:
        return [_scale(signals.get("layout_overlap_count"), 128.0)]

    def _infra_features(self, signals: Mapping[str, Any]) -> list[float]:
        return [
            _bool(signals.get("port_open")),
            _bool(signals.get("health_check_ok")),
            _scale(signals.get("response_status"), 1000.0),
            _scale(signals.get("response_latency_ms"), 10000.0),
            _bool(signals.get("timeout_occurred")),
            _scale(signals.get("server_5xx_count"), 20.0),
            _scale(signals.get("server_4xx_count"), 20.0),
            _scale(signals.get("server_log_exception_count"), 50.0),
            _bool(signals.get("process_alive")),
            _scale(signals.get("cpu_usage_percent"), 100.0),
            _scale(signals.get("memory_usage_mb"), 4096.0),
        ]

    def _history_features(self, history: Mapping[str, Any]) -> list[float]:
        previous_action = str(history.get("previous_action_type", "noop"))
        return [
            _scale(history.get("step_index"), 1000.0),
            _scale(history.get("no_change_steps"), 100.0),
            _scale(_stable_action_hash(previous_action), 997.0),
        ]


def _stable_action_hash(value: str) -> int:
    total = 0
    for char in value:
        total = (total * 31 + ord(char)) % 997
    return total


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return number


def _scale(value: Any, denominator: float) -> float:
    return _clamp(_float(value) / denominator if denominator else 0.0)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, _float(value)))
