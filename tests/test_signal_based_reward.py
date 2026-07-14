from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.autonomous_reward_service import calculate_autonomous_reward


class SignalBasedRewardTests(unittest.TestCase):
    def test_known_bug_reward_disabled_by_default(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _obs(),
            _obs(url="http://local/next"),
            {"action_type": "click_element", "candidate_index": 0},
            [{"type": "button-no-response", "confidence": 0.9, "matched_bug_id": "site001-bug01"}],
            [{"matched_bug_id": "site001-bug01"}],
            {},
            {},
        )
        self.assertEqual("signal_based", breakdown["reward_mode"])
        self.assertFalse(breakdown["use_known_bug_reward"])
        self.assertEqual(0.0, breakdown["known_bug_reward_total"])
        self.assertEqual(0.0, breakdown["reward_known_bug_match"])

    def test_training_reward_never_uses_known_bug_match(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _obs(),
            _obs(url="http://local/next"),
            {"action_type": "click_element", "candidate_index": 0},
            [{"type": "button-no-response", "confidence": 0.9, "matched_bug_id": "site001-bug01"}],
            [{"matched_bug_id": "site001-bug01"}],
            {},
            {"reward_mode": "known_bug_catalog", "use_known_bug_reward": True},
        )
        self.assertFalse(breakdown["use_known_bug_reward"])
        self.assertEqual(0.0, breakdown["reward_known_bug_match"])
        self.assertEqual(0.0, breakdown["known_bug_reward_total"])

    def test_reward_known_bug_match_always_zero_in_training(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _obs(),
            _obs(),
            {"action_type": "click_element", "candidate_index": 0},
            [{"type": "button-no-response", "confidence": 0.9, "matched_bug_id": "site001-bug01"}],
            [{"id": "site001-bug01"}],
            {},
            {"allow_known_bug_reward_for_debug_only": True, "use_known_bug_reward": True},
        )
        self.assertEqual(0.0, breakdown["reward_known_bug_match"])

    def test_post_action_matched_ids_not_called_for_reward(self) -> None:
        with patch(
            "services.autonomous_reward_service._post_action_matched_ids",
            side_effect=AssertionError("_post_action_matched_ids must not be used for reward"),
        ):
            _, breakdown = calculate_autonomous_reward(
                _obs(),
                _obs(),
                {"action_type": "click_element", "candidate_index": 0},
                [{"type": "button-no-response", "confidence": 0.9, "matched_bug_id": "site001-bug01"}],
                [{"id": "site001-bug01"}],
                {},
                {"use_known_bug_reward": True},
            )
        self.assertEqual(0.0, breakdown["known_bug_reward_total"])

    def test_catalog_fields_not_used_in_reward(self) -> None:
        source = _calculate_reward_source().lower()
        for keyword in ("catalog_bug_id_matches", "catalog_selector_match", "catalog_keyword_matches"):
            self.assertNotIn(keyword, source)

    def test_bug_id_not_used_in_reward(self) -> None:
        self.assertNotIn("bug_id", _calculate_reward_source().lower())

    def test_data_bug_id_not_used_in_reward(self) -> None:
        self.assertNotIn("data_bug_id", _calculate_reward_source().lower())

    def test_signal_based_reward_uses_console_runtime_network_signals(self) -> None:
        after = _obs()
        after["runtime_signals"].update(
            {
                "console_error_count": 1,
                "runtime_exception_count": 1,
                "network_status_codes": [404, 500],
            }
        )
        _, breakdown = calculate_autonomous_reward(_obs(), after, {"action_type": "inspect_console"}, [], [], {}, {})
        self.assertGreater(breakdown["console_runtime_signal_reward_total"], 0.0)
        self.assertGreater(breakdown["network_api_signal_reward_total"], 0.0)
        self.assertEqual(1, breakdown["console_error_count"])
        self.assertEqual(1, breakdown["runtime_exception_count"])
        self.assertEqual(1, breakdown["api_4xx_count"])
        self.assertEqual(1, breakdown["api_5xx_count"])

    def test_console_error_delta_reward(self) -> None:
        before = _obs()
        before["runtime_signals"]["console_error_count"] = 1
        after = _obs()
        after["runtime_signals"]["console_error_count"] = 2
        _, breakdown = calculate_autonomous_reward(before, after, {"action_type": "click_element"}, [], [], {}, {})
        self.assertEqual(1, breakdown["delta_console_error_count"])
        self.assertGreater(breakdown["reward_console_error_detected"], 0.0)

    def test_inspect_console_alone_does_not_create_signal_reward(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _obs(),
            _obs(),
            {"action_type": "inspect_console"},
            [],
            [],
            {"action_type_counts": {"inspect_console": 0}},
            {},
        )
        self.assertEqual(0.0, breakdown["console_runtime_signal_reward_total"])
        self.assertEqual(0.0, breakdown["signal_delta_reward_total"])

    def test_network_5xx_delta_reward(self) -> None:
        after = _obs()
        after["runtime_signals"]["network_entries"] = [{"status": 500}]
        _, breakdown = calculate_autonomous_reward(_obs(), after, {"action_type": "click_element"}, [], [], {}, {})
        self.assertEqual(1, breakdown["delta_api_5xx_count"])
        self.assertGreater(breakdown["reward_api_5xx_detected"], 0.0)

    def test_requestfailed_delta_reward(self) -> None:
        after = _obs()
        after["runtime_signals"]["network_request_failed_count"] = 1
        _, breakdown = calculate_autonomous_reward(_obs(), after, {"action_type": "click_element"}, [], [], {}, {})
        self.assertEqual(1, breakdown["delta_network_request_failed_count"])
        self.assertGreater(breakdown["reward_network_request_failed"], 0.0)

    def test_signal_delta_reward_can_exist_without_functional_action_reward(self) -> None:
        after = _obs()
        after["runtime_signals"]["network_entries"] = [{"status": 500}]
        _, breakdown = calculate_autonomous_reward(
            _obs(),
            after,
            {"action_type": "inspect_network"},
            [],
            [],
            {"action_type_counts": {"inspect_network": 0}, "functional_action_count": 0},
            {},
        )
        self.assertGreater(breakdown["signal_delta_reward_total"], 0.0)
        self.assertEqual(0.0, breakdown["functional_action_signal_reward_total"])

    def test_signal_based_reward_does_not_use_bug_labels(self) -> None:
        source = (ROOT / "services" / "autonomous_reward_service.py").read_text(encoding="utf-8").lower()
        for keyword in (
            "data-bug-id",
            "data_bug_id",
            "has_data_bug_id",
            "known_bug_id",
            "catalog_bug_id_matches",
            "ground_truth",
            "oracle",
        ):
            self.assertNotIn(keyword, source)

    def test_reward_summary_marks_policy_uses_bug_labels_false(self) -> None:
        _, breakdown = calculate_autonomous_reward(_obs(), _obs(), {"action_type": "noop"}, [], [], {}, {})
        self.assertFalse(breakdown["policy_uses_bug_labels"])
        self.assertFalse(breakdown["training_uses_site_specific_bug_catalog"])


def _obs(url: str = "http://local") -> dict:
    return {
        "page_state": {"url": url, "title": "Local", "page_text_length": 10, "dom_node_count": 5},
        "candidate_elements": [{"bid": "b1", "text": "Open", "visible": True, "enabled": True, "clickable": True}],
        "runtime_signals": {},
        "layout_signals": {},
    }


def _calculate_reward_source() -> str:
    source = (ROOT / "services" / "autonomous_reward_service.py").read_text(encoding="utf-8")
    start = source.index("def calculate_autonomous_reward(")
    end = source.index("def apply_fallback_reward_policy(", start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
