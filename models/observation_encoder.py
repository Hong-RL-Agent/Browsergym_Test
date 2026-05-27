"""Numeric encoder for J.A.W.S raw observations."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


ROLE_ORDER = ["button", "link", "textbox", "combobox", "checkbox", "radio", "menuitem", "tab"]


class ObservationEncoder:
    page_feature_dim = 9
    candidate_feature_dim = len(ROLE_ORDER) + 12
    runtime_feature_dim = 3
    layout_feature_dim = 1
    history_feature_dim = 3

    def __init__(self, max_candidates: int = 32) -> None:
        self.max_candidates = max_candidates

    def encode_observation(self, raw_observation: Mapping[str, Any]) -> np.ndarray:
        features = []
        features.extend(self._page_features(raw_observation.get("page_state", {})))

        candidates = raw_observation.get("candidate_elements", [])
        if not isinstance(candidates, list):
            candidates = []
        for idx in range(self.max_candidates):
            if idx < len(candidates) and isinstance(candidates[idx], Mapping):
                features.extend(self._candidate_features(candidates[idx]))
            else:
                features.extend([0.0] * self.candidate_feature_dim)

        features.extend(self._runtime_features(raw_observation.get("runtime_signals", {})))
        features.extend(self._layout_features(raw_observation.get("layout_signals", {})))
        features.extend(self._history_features(raw_observation.get("history", {})))

        vector = np.asarray(features, dtype=np.float32)
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def get_obs_dim(self) -> int:
        return (
            self.page_feature_dim
            + self.max_candidates * self.candidate_feature_dim
            + self.runtime_feature_dim
            + self.layout_feature_dim
            + self.history_feature_dim
        )

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
