from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.browsergym_observation_adapter import _make_candidate
from services.anomaly_detection_service import detect_anomalies


class LowValueGenericFilteringTests(unittest.TestCase):
    def test_low_value_generic_no_response_suppressed(self) -> None:
        candidate = _generic_candidate()
        anomalies = detect_anomalies(_observation(candidate), _observation(candidate), {"action": {"action_type": "click_element", "candidate_index": 0}}, {})
        self.assertFalse(any(item.get("type") == "button-no-response" for item in anomalies))

    def test_element_dash_number_generic_not_high_confidence_anomaly(self) -> None:
        candidate = _generic_candidate()
        self.assertTrue(candidate["is_low_value_generic_candidate"])
        anomalies = detect_anomalies(_observation(candidate), _observation(candidate), {"action": {"action_type": "click_element", "candidate_index": 0}}, {})
        self.assertTrue(all(float(item.get("confidence", 0.0) or 0.0) < 0.6 for item in anomalies))

    def test_low_value_generic_candidate_priority_lowered(self) -> None:
        generic = _generic_candidate()
        add = _make_candidate(
            bid="add",
            text="담기",
            name="담기",
            role="button",
            tag="button",
            bbox=[0, 0, 100, 40],
            visibility=1.0,
            clickable=True,
            enabled=True,
            source="test",
            page_text="",
        )
        self.assertLess(float(generic["action_priority"]), float(add["action_priority"]))
        self.assertLess(float(generic["openended_action_priority"]), float(add["openended_action_priority"]))

    def test_no_bug_label_used_for_generic_filtering(self) -> None:
        plain = _generic_candidate(data_bug_id="")
        labeled = _generic_candidate(data_bug_id="site005-bug01")
        for key in ("is_low_value_generic_candidate", "action_priority", "openended_action_priority"):
            self.assertEqual(plain[key], labeled[key])


def _generic_candidate(data_bug_id: str = "") -> dict:
    return _make_candidate(
        bid="g",
        text="element-12",
        name="element-12",
        role="generic",
        tag="div",
        bbox=[0, 0, 100, 40],
        visibility=1.0,
        clickable=True,
        enabled=True,
        source="test",
        page_text="",
        data_bug_id=data_bug_id,
    )


def _observation(candidate: dict) -> dict:
    return {
        "page_state": {"url": "http://localhost", "title": "T", "page_text": "text", "page_text_length": 4, "dom_node_count": 10},
        "candidate_elements": [dict(candidate)],
        "runtime_signals": {},
        "layout_signals": {},
    }


if __name__ == "__main__":
    unittest.main()
