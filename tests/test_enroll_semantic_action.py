from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.browsergym_observation_adapter import _make_candidate
from services.anomaly_detection_service import detect_anomalies
from services.policy_safe_metrics import build_per_site_metric


class EnrollSemanticActionTests(unittest.TestCase):
    def test_enroll_button_classified_as_high_value_functional(self) -> None:
        candidate = _candidate("수강신청")
        self.assertEqual("enroll", candidate["semantic_action_type"])
        self.assertTrue(candidate["functional_priority_candidate"])
        self.assertTrue(candidate["is_high_value_functional_candidate"])

    def test_enroll_no_effect_counts_semantic_button_no_response(self) -> None:
        candidate = _candidate("수강신청")
        anomalies = detect_anomalies(_observation(candidate), _observation(candidate), {"action": {"action_type": "click_element", "candidate_index": 0}}, {})
        self.assertTrue(any(item.get("type") == "button-no-response" for item in anomalies))
        metric = build_per_site_metric("site004", _site_result_from_anomalies(anomalies))
        self.assertEqual(1, metric["semantic_button_no_response_count"])
        self.assertEqual(1, metric["high_value_button_no_response_count"])
        self.assertEqual(1, metric["enroll_no_effect_count"])
        self.assertEqual(1, metric["high_value_enroll_no_response_count"])

    def test_no_bug_label_used_for_enroll_semantics(self) -> None:
        plain = _candidate("수강신청", data_bug_id="")
        labeled = _candidate("수강신청", data_bug_id="site004-bug01")
        for key in ("semantic_action_type", "functional_priority_candidate", "is_high_value_functional_candidate"):
            self.assertEqual(plain[key], labeled[key])


def _candidate(text: str, data_bug_id: str = "") -> dict:
    return _make_candidate(
        bid="enroll",
        text=text,
        name=text,
        role="button",
        tag="button",
        bbox=[0, 0, 100, 40],
        visibility=1.0,
        clickable=True,
        enabled=True,
        source="test",
        page_text="course list",
        data_bug_id=data_bug_id,
    )


def _observation(candidate: dict) -> dict:
    return {
        "page_state": {
            "url": "http://localhost/course",
            "title": "Courses",
            "page_text_length": 100,
            "page_text": "Courses",
            "dom_node_count": 10,
            "has_modal_or_dialog": False,
            "has_form": False,
        },
        "candidate_elements": [dict(candidate)],
        "runtime_signals": {},
        "layout_signals": {},
    }


def _site_result_from_anomalies(anomalies: list[dict]) -> dict:
    return {
        "completed_episodes": 1,
        "semantic_no_effect_click_count": sum(1 for item in anomalies if item.get("evidence", {}).get("semantic_no_effect_click")),
        "high_value_no_effect_click_count": sum(1 for item in anomalies if item.get("evidence", {}).get("high_value_functional_candidate")),
        "functional_no_effect_anomaly_count": sum(1 for item in anomalies if item.get("evidence", {}).get("functional_no_effect_anomaly")),
        "semantic_button_no_response_count": sum(
            1 for item in anomalies if item.get("type") == "button-no-response" and item.get("evidence", {}).get("semantic_action_type")
        ),
        "high_value_button_no_response_count": sum(
            1 for item in anomalies if item.get("type") == "button-no-response" and item.get("evidence", {}).get("high_value_functional_candidate")
        ),
        "enroll_no_effect_count": sum(1 for item in anomalies if item.get("evidence", {}).get("enroll_no_effect")),
        "high_value_enroll_no_response_count": sum(
            1 for item in anomalies if item.get("evidence", {}).get("high_value_enroll_no_response")
        ),
    }


if __name__ == "__main__":
    unittest.main()
