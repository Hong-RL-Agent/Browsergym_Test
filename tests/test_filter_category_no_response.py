from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.anomaly_detection_service import detect_anomalies


class FilterCategoryNoResponseTests(unittest.TestCase):
    def test_filter_click_with_text_delta_not_button_no_response(self) -> None:
        anomalies = _click_anomalies(_candidate("Programming", "filter"), text_delta=15)
        self.assertFalse(_has_type(anomalies, "button-no-response"))

    def test_category_click_with_text_delta_not_button_no_response(self) -> None:
        anomalies = _click_anomalies(_candidate("Design", "category"), text_delta=15)
        self.assertFalse(_has_type(anomalies, "button-no-response"))

    def test_tab_click_with_text_delta_not_button_no_response(self) -> None:
        anomalies = _click_anomalies(_candidate("Business", "tab", role="tab"), text_delta=15)
        self.assertFalse(_has_type(anomalies, "button-no-response"))

    def test_category_click_without_text_delta_not_button_no_response(self) -> None:
        anomalies = _click_anomalies(_candidate("Business", "category"), text_delta=0)
        self.assertFalse(_has_type(anomalies, "button-no-response"))

    def test_add_button_no_effect_still_button_no_response(self) -> None:
        anomalies = _click_anomalies(_candidate("Add", "add"), text_delta=0)
        self.assertTrue(_has_type(anomalies, "button-no-response"))

    def test_cart_button_no_effect_still_button_no_response(self) -> None:
        anomalies = _click_anomalies(_candidate("Add to cart", "cart"), text_delta=0)
        self.assertTrue(_has_type(anomalies, "button-no-response"))

    def test_no_bug_label_used_in_filter_semantics(self) -> None:
        source = _function_source(
            ROOT / "services" / "anomaly_detection_service.py",
            "def _is_high_value_semantic_action(",
            "def _looks_like_form_submission(",
        ).lower()
        for keyword in ("data-bug-id", "data_bug_id", "bug_id", "catalog_bug_id", "known_bug", "ground_truth", "oracle", "label"):
            self.assertNotIn(keyword, source)

    def test_no_catalog_used_in_no_response_filtering(self) -> None:
        source = _function_source(
            ROOT / "services" / "anomaly_detection_service.py",
            "def _is_semantic_no_effect_click(",
            "def _looks_like_form_submission(",
        ).lower()
        self.assertNotIn("catalog", source)


def _click_anomalies(candidate: dict, *, text_delta: int) -> list[dict]:
    before = _observation(candidate, text_length=100)
    after = _observation(candidate, text_length=100 + text_delta)
    return detect_anomalies(before, after, {"action": {"action_type": "click_element", "candidate_index": 0}}, {})


def _candidate(text: str, semantic: str, *, role: str = "button") -> dict:
    return {
        "bid": semantic,
        "text": text,
        "name": text,
        "role": role,
        "tag": "button",
        "visibility": 1.0,
        "clickable": True,
        "enabled": True,
        "semantic_action_type": semantic,
        "functional_priority_candidate": True,
        "functional_priority": True,
        "is_high_value_functional_candidate": semantic in {"add", "cart"},
        "is_interactive": True,
        "is_filter_related": semantic in {"filter", "category", "tab"},
    }


def _observation(candidate: dict, *, text_length: int) -> dict:
    return {
        "page_state": {
            "url": "http://localhost/",
            "title": "Example",
            "page_text_length": text_length,
            "page_text": "x" * text_length,
            "dom_node_count": 10,
            "has_modal_or_dialog": False,
            "has_form": False,
            "cart_count_detected": False,
        },
        "candidate_elements": [dict(candidate)],
        "runtime_signals": {},
        "layout_signals": {},
    }


def _has_type(anomalies: list[dict], anomaly_type: str) -> bool:
    return any(item.get("type") == anomaly_type for item in anomalies)


def _function_source(path: Path, start_marker: str, end_marker: str) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
