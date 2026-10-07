"""Reward scoring for the rule-based API fuzzing baseline."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Mapping, Tuple

from services.api_anomaly_detection_service import anomaly_key


REWARD_BY_TYPE = {
    "api-5xx-error": 5.0,
    "api-4xx-unexpected": 2.0,
    "api-schema-mismatch": 3.0,
    "api-timeout": 4.0,
    "api-auth-bypass": 6.0,
    "api-latency-regression": 2.0,
    "api-validation-missing": 2.0,
    "api-response-body-invalid": 2.0,
    "api-forbidden": 1.0,
}


class ApiRewardScorer:
    def __init__(self) -> None:
        self._seen_anomalies: set[Tuple[str, str, str, str, int]] = set()
        self._endpoint_failures: Counter[tuple[str, str]] = Counter()

    def score(
        self,
        site_id: str,
        api_id: str,
        mutation: str,
        anomalies: Iterable[Mapping[str, object]],
        expected_error_handled: bool = False,
    ) -> tuple[float, Dict[str, float]]:
        reward = 0.0
        breakdown: Dict[str, float] = {}
        anomaly_list = list(anomalies)
        for anomaly in anomaly_list:
            anomaly_type = str(anomaly.get("anomaly_type") or anomaly.get("type") or "")
            value = REWARD_BY_TYPE.get(anomaly_type, 0.0)
            key = anomaly_key(anomaly)
            if key in self._seen_anomalies:
                value -= 1.0
                breakdown["duplicate_anomaly_penalty"] = breakdown.get("duplicate_anomaly_penalty", 0.0) - 1.0
            else:
                self._seen_anomalies.add(key)
            reward += value
            breakdown[anomaly_type] = breakdown.get(anomaly_type, 0.0) + value
        if not anomaly_list and expected_error_handled:
            reward += 0.1
            breakdown["expected_error_handling"] = 0.1
        endpoint_key = (site_id, api_id)
        if anomaly_list:
            self._endpoint_failures[endpoint_key] += 1
            if self._endpoint_failures[endpoint_key] > 5:
                reward -= 0.5
                breakdown["same_endpoint_repeat_failure_penalty"] = -0.5
        elif mutation in {"send_valid_request", "check_response_schema", "check_latency"}:
            self._endpoint_failures[endpoint_key] = 0
        return reward, breakdown
