from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.browsergym_training_service import _enrich_action, _update_history
from services.policy_safe_metrics import build_per_site_metric


class ActionSignatureRepeatDetectionTests(unittest.TestCase):
    def test_same_action_type_different_element_key_not_repeated(self) -> None:
        history: dict = {}
        observation = _observation()
        first = {"action_type": "click_element", "candidate_index": 0}
        second = {"action_type": "click_element", "candidate_index": 1}
        _enrich_action(first, observation)
        _update_history(history, observation, first, [])
        _enrich_action(second, observation)
        _update_history(history, observation, second, [])

        counts = history["action_signature_counts"]
        self.assertEqual(1, counts[first["action_signature"]])
        self.assertEqual(1, counts[second["action_signature"]])
        self.assertNotEqual(first["action_signature"], second["action_signature"])

    def test_same_action_signature_is_repeated(self) -> None:
        history: dict = {}
        observation = _observation()
        for _ in range(2):
            action = {"action_type": "click_element", "candidate_index": 0}
            _enrich_action(action, observation)
            _update_history(history, observation, action, [])

        signature = "click_element::button|add|담기|restaurant-card-1|pos-0-0-2|aaa"
        self.assertEqual(2, history["action_signature_counts"][signature])
        self.assertEqual(1, history["element_key_click_counts"]["button|add|담기|restaurant-card-1|pos-0-0-2|aaa"] - 1)

    def test_action_log_records_action_signature(self) -> None:
        action = {"action_type": "click_element", "candidate_index": 0}
        _enrich_action(action, _observation())
        self.assertEqual("button|add|담기|restaurant-card-1|pos-0-0-2|aaa", action["action_element_key"])
        self.assertEqual("b1", action["action_bid"])
        self.assertEqual("담기", action["action_text"])
        self.assertEqual("add", action["action_semantic_type"])
        self.assertTrue(action["dom_state_hash"])
        self.assertTrue(action["action_signature"].startswith("click_element::button|add|담기"))

    def test_repeated_action_count_uses_action_signature(self) -> None:
        metric = build_per_site_metric(
            "site005",
            {
                "completed_episodes": 1,
                "repeated_action_count": 1,
                "repeated_action_signature_count": 1,
                "repeated_action_type_count": 4,
            },
        )
        self.assertEqual(1, metric["repeated_action_signature_count"])
        self.assertEqual(4, metric["repeated_action_type_count"])

    def test_repeated_action_type_count_is_separate_metric(self) -> None:
        metric = build_per_site_metric(
            "site005",
            {
                "completed_episodes": 1,
                "repeated_action_count": 0,
                "repeated_action_signature_count": 0,
                "repeated_action_type_count": 5,
            },
        )
        self.assertEqual(0, metric["repeated_action_signature_count"])
        self.assertEqual(5, metric["repeated_action_type_count"])


def _observation() -> dict:
    return {
        "page_state": {"url": "http://localhost", "title": "T", "page_text": "text", "page_text_length": 4},
        "candidate_elements": [
            _candidate("b1", "button|add|담기|restaurant-card-1|pos-0-0-2|aaa"),
            _candidate("b2", "button|add|담기|restaurant-card-2|pos-0-1-2|bbb"),
        ],
    }


def _candidate(bid: str, element_key: str) -> dict:
    return {
        "bid": bid,
        "text": "담기",
        "name": "담기",
        "role": "button",
        "tag": "button",
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "clickable": True,
        "semantic_action_type": "add",
        "element_key": element_key,
        "functional_priority_candidate": True,
        "functional_priority": True,
        "is_high_value_functional_candidate": True,
    }


if __name__ == "__main__":
    unittest.main()
