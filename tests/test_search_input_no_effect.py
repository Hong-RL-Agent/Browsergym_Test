from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.anomaly_detection_service import detect_anomalies


class SearchInputNoEffectTests(unittest.TestCase):
    def test_search_textbox_click_not_no_effect_anomaly(self) -> None:
        candidate = _search_candidate()
        anomalies = detect_anomalies(
            _observation(candidate),
            _observation(candidate),
            {"action": {"action_type": "click_element", "candidate_index": 0}},
            {},
        )
        self.assertFalse(any(item.get("type") in {"button-no-response", "form-no-feedback"} for item in anomalies))

    def test_search_textbox_fill_then_no_effect_can_be_anomaly(self) -> None:
        candidate = _search_candidate()
        anomalies = detect_anomalies(
            _observation(candidate),
            _observation(candidate),
            {"action": {"action_type": "fill_input", "candidate_index": 0}},
            {},
        )
        self.assertTrue(any(item.get("type") == "form-no-feedback" for item in anomalies))
        self.assertTrue(any(item.get("evidence", {}).get("search_input_no_effect") for item in anomalies))


def _search_candidate() -> dict:
    return {
        "bid": "search",
        "text": "Search",
        "name": "Search",
        "role": "searchbox",
        "tag": "input",
        "input_type": "search",
        "visibility": 1.0,
        "clickable": True,
        "enabled": True,
        "semantic_action_type": "search_input",
        "functional_priority_candidate": True,
        "functional_priority": True,
        "is_high_value_functional_candidate": False,
        "is_search_related": True,
        "is_form_field": True,
        "is_interactive": True,
    }


def _observation(candidate: dict) -> dict:
    return {
        "page_state": {
            "url": "http://localhost/",
            "title": "Example",
            "page_text_length": 100,
            "page_text": "Example page",
            "dom_node_count": 10,
            "has_modal_or_dialog": False,
            "has_form": True,
            "cart_count_detected": False,
        },
        "candidate_elements": [dict(candidate)],
        "runtime_signals": {},
        "layout_signals": {},
    }


if __name__ == "__main__":
    unittest.main()
