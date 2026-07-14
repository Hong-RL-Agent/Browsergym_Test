from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.anomaly_detection_service import detect_anomalies
from services.policy_safe_metrics import build_per_site_metric


class SemanticNoEffectMetricsTests(unittest.TestCase):
    def test_add_no_effect_increments_semantic_metrics(self) -> None:
        anomalies = _click_no_effect(_candidate("add"))
        self.assertTrue(_semantic_button_no_response(anomalies, "add_no_effect"))
        metric = build_per_site_metric("site005", _site_result_from_anomalies(anomalies))
        self.assertEqual(1, metric["semantic_no_effect_click_count"])
        self.assertEqual(1, metric["high_value_no_effect_click_count"])
        self.assertEqual(1, metric["functional_no_effect_anomaly_count"])
        self.assertEqual(1, metric["semantic_button_no_response_count"])
        self.assertEqual(1, metric["add_no_effect_count"])

    def test_cart_no_effect_increments_semantic_metrics(self) -> None:
        anomalies = _click_no_effect(_candidate("cart"))
        self.assertTrue(_semantic_button_no_response(anomalies, "cart_no_effect"))

    def test_workout_add_no_effect_increments_semantic_metrics(self) -> None:
        anomalies = _click_no_effect(_candidate("workout_add"))
        self.assertTrue(_semantic_button_no_response(anomalies, "workout_add_no_effect"))


def _click_no_effect(candidate: dict) -> list[dict]:
    observation = _observation(candidate)
    return detect_anomalies(
        observation,
        _observation(candidate),
        {"action": {"action_type": "click_element", "candidate_index": 0}},
        {},
    )


def _semantic_button_no_response(anomalies: list[dict], flag: str) -> bool:
    return any(
        item.get("type") == "button-no-response"
        and item.get("evidence", {}).get("semantic_no_effect_click")
        and item.get("evidence", {}).get(flag)
        for item in anomalies
    )


def _candidate(semantic: str) -> dict:
    text = {"add": "Add", "cart": "Add to cart", "workout_add": "Workout add"}[semantic]
    return {
        "bid": semantic,
        "text": text,
        "name": text,
        "role": "button",
        "tag": "button",
        "visibility": 1.0,
        "clickable": True,
        "enabled": True,
        "semantic_action_type": semantic,
        "functional_priority_candidate": True,
        "functional_priority": True,
        "is_high_value_functional_candidate": True,
        "is_purchase_action": semantic in {"add", "cart"},
        "is_workout_add_action": semantic == "workout_add",
    }


def _observation(candidate: dict) -> dict:
    return {
        "page_state": {
            "url": "http://localhost/",
            "title": "Example",
            "page_text_length": 100,
            "page_text": "Example",
            "dom_node_count": 10,
            "has_modal_or_dialog": False,
            "has_form": False,
            "cart_count_detected": True,
            "cart_count": 1,
            "cart_text": "cart 1",
        },
        "candidate_elements": [dict(candidate)],
        "runtime_signals": {},
        "layout_signals": {},
    }


def _site_result_from_anomalies(anomalies: list[dict]) -> dict:
    semantic_count = 0
    high_value_count = 0
    functional_count = 0
    button_count = 0
    add_count = 0
    for anomaly in anomalies:
        evidence = anomaly.get("evidence", {})
        if evidence.get("semantic_no_effect_click"):
            semantic_count += 1
        if evidence.get("high_value_functional_candidate"):
            high_value_count += 1
        if evidence.get("functional_no_effect_anomaly"):
            functional_count += 1
        if anomaly.get("type") == "button-no-response" and evidence.get("semantic_action_type"):
            button_count += 1
        if evidence.get("add_no_effect"):
            add_count += 1
    return {
        "completed_episodes": 1,
        "semantic_no_effect_click_count": semantic_count,
        "high_value_no_effect_click_count": high_value_count,
        "functional_no_effect_anomaly_count": functional_count,
        "semantic_button_no_response_count": button_count,
        "add_no_effect_count": add_count,
    }


if __name__ == "__main__":
    unittest.main()
