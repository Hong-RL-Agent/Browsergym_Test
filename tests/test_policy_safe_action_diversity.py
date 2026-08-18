from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.autonomous_reward_service import calculate_autonomous_reward
from services.policy_safe_metrics import build_per_site_metric


class PolicySafeActionDiversityRewardTests(unittest.TestCase):
    def test_diversity_reward_does_not_use_bug_labels(self) -> None:
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

    def test_first_click_element_reward(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(url="http://local/detail"),
            {"action_type": "click_element", "candidate_index": 0},
            [],
            [],
            {"action_type_counts": {}, "clicked_bids": set(), "target_signatures": set()},
        )
        self.assertEqual(0.5, breakdown["reward_first_click_element"])
        self.assertEqual(1.0, breakdown["first_click_reward_count"])

    def test_new_action_type_reward(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(dom_node_count=12),
            {"action_type": "inspect_dom", "candidate_index": 0},
            [],
            [],
            {"action_type_counts": {}},
        )
        self.assertEqual(0.2, breakdown["reward_new_action_type"])
        self.assertEqual(0.2, breakdown["reward_first_inspect_dom"])
        self.assertEqual(1.0, breakdown["new_action_type_reward_count"])
        self.assertEqual(1.0, breakdown["inspect_reward_count"])

    def test_new_target_reward(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(url="http://local/next"),
            {"action_type": "click_element", "candidate_index": 0},
            [],
            [],
            {"action_type_counts": {"click_element": 1}, "target_signatures": set()},
        )
        self.assertEqual(0.3, breakdown["reward_new_target"])
        self.assertEqual(1.0, breakdown["new_target_reward_count"])

    def test_open_detail_panel_repeat_penalty(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(),
            {"action_type": "open_detail_panel", "candidate_index": 0},
            [],
            [],
            {"action_type_counts": {"open_detail_panel": 2}},
        )
        self.assertEqual(-0.5, breakdown["penalty_open_detail_panel_repeat"])
        self.assertEqual(1.0, breakdown["open_detail_panel_repeat_penalty_count"])
        self.assertLess(breakdown["penalty_debug_meta_total"], 0.0)

    def test_click_retry_button_repeat_penalized(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(),
            {"action_type": "click_retry_button", "candidate_index": 0},
            [],
            [],
            {"action_type_counts": {"click_retry_button": 1}, "step_index": 3},
        )
        self.assertLess(breakdown["penalty_retry_button_repeat"], 0.0)
        self.assertEqual(1.0, breakdown["click_retry_button_repeat_penalty_count"])

    def test_debug_meta_actions_not_counted_as_functional_actions(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(),
            {"action_type": "open_detail_panel", "candidate_index": 0},
            [],
            [],
            {"action_type_counts": {}, "step_index": 0},
        )
        self.assertTrue(breakdown["is_debug_meta_action"])
        self.assertFalse(breakdown["is_functional_action"])
        self.assertEqual(0.0, breakdown["functional_action_count"])
        self.assertEqual(1.0, breakdown["debug_meta_action_count"])

    def test_no_functional_action_episode_penalized(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(),
            {"action_type": "inspect_dom", "candidate_index": 0},
            [],
            [],
            {"action_type_counts": {"inspect_dom": 2}, "functional_action_count": 0, "step_index": 3},
        )
        self.assertLess(breakdown["penalty_no_functional_action_episode"], 0.0)
        self.assertEqual(1.0, breakdown["no_functional_action_episode_count"])

    def test_debug_meta_and_scroll_only_episode_penalized(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(),
            {"action_type": "scroll_up", "candidate_index": 0},
            [],
            [],
            {"action_type_counts": {"scroll_up": 2}, "functional_action_count": 0, "step_index": 3},
        )
        self.assertTrue(breakdown["is_debug_meta_action"])
        self.assertEqual(0.0, breakdown["functional_action_count"])
        self.assertEqual(1.0, breakdown["no_functional_action_episode_count"])
        self.assertLess(breakdown["penalty_no_functional_action_episode"], 0.0)

    def test_functional_action_reward_does_not_use_bug_labels(self) -> None:
        after = _observation(url="http://local/next")
        after["runtime_signals"] = {"delta_api_5xx_count": 1}
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            after,
            {"action_type": "click_element", "candidate_index": 0},
            [],
            [],
            {"action_type_counts": {}, "functional_action_count": 0, "functional_target_signatures": set()},
        )
        self.assertGreater(breakdown["reward_first_functional_action"], 0.0)
        self.assertGreater(breakdown["reward_functional_action_caused_signal_delta"], 0.0)
        self.assertGreater(breakdown["reward_functional_action_total"], 0.0)

    def test_functional_action_signal_reward_requires_functional_action(self) -> None:
        after = _observation()
        after["runtime_signals"] = {"delta_api_5xx_count": 1}
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            after,
            {"action_type": "open_detail_panel", "candidate_index": 0},
            [],
            [],
            {"action_type_counts": {"open_detail_panel": 0}, "functional_action_count": 0},
        )
        self.assertEqual(0.0, breakdown["functional_action_count"])
        self.assertEqual(0.0, breakdown["functional_action_signal_reward_total"])
        self.assertEqual(0.0, breakdown["signal_delta_reward_total"])
        self.assertGreater(breakdown["raw_signal_reward_metric_total"], 0.0)
        self.assertFalse(breakdown["direct_signal_reward_used_for_policy"])

    def test_same_action_signature_repeat_penalty(self) -> None:
        history = {
            "action_type_counts": {"click_element": 1},
            "action_signature_counts": {"click_element:0:click_element:btn-1": 1},
            "target_signatures": {"click_element:btn-1"},
        }
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(),
            {"action_type": "click_element", "candidate_index": 0},
            [],
            [],
            history,
        )
        self.assertEqual(-0.7, breakdown["penalty_same_action_signature_repeat"])

    def test_reused_click_type_not_penalized_for_new_target(self) -> None:
        history = {
            "action_type_counts": {"click_element": 3},
            "action_signature_counts": {
                "click_element:0:click_element:btn-1": 1,
                "click_element:1:click_element:btn-2": 1,
            },
            "target_signatures": {"click_element:btn-1", "click_element:btn-2"},
        }
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(url="http://local/next"),
            {"action_type": "click_element", "candidate_index": 2},
            [],
            [],
            history,
        )
        self.assertEqual(0.3, breakdown["reward_new_target"])
        self.assertEqual(0.0, breakdown["penalty_same_action_type_repeat"])
        self.assertEqual(0.0, breakdown["penalty_same_action_signature_repeat"])

    def test_reused_click_type_still_penalized_for_seen_signature(self) -> None:
        history = {
            "action_type_counts": {"click_element": 3},
            "action_signature_counts": {"click_element:0:click_element:btn-1": 1},
            "target_signatures": {"click_element:btn-1"},
        }
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(),
            {"action_type": "click_element", "candidate_index": 0},
            [],
            [],
            history,
        )
        self.assertEqual(-0.5, breakdown["penalty_same_action_type_repeat"])
        self.assertEqual(-0.7, breakdown["penalty_same_action_signature_repeat"])

    def test_no_effect_open_detail_penalty(self) -> None:
        _, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(),
            {"action_type": "open_detail_panel", "candidate_index": 0},
            [],
            [],
            {"action_type_counts": {"open_detail_panel": 1}},
        )
        self.assertEqual(-0.5, breakdown["penalty_no_effect_open_detail"])
        self.assertLessEqual(breakdown["penalty_no_effect"], -0.3)
        self.assertEqual(1.0, breakdown["no_effect_action_penalty_count"])

    def test_action_diversity_metrics_written(self) -> None:
        metrics = build_per_site_metric(
            "site001",
            {
                "action_diversity_score": 0.5,
                "unique_action_type_count": 3,
                "unique_clicked_target_count": 2,
                "first_click_step": 4,
                "open_detail_panel_count": 1,
                "diversity_reward_total": 1.2,
                "repeated_action_penalty_total": -0.7,
                "first_click_reward_count": 1,
                "new_action_type_reward_count": 3,
                "new_target_reward_count": 2,
                "inspect_reward_count": 1,
                "open_detail_panel_repeat_penalty_count": 0,
                "no_effect_action_penalty_count": 2,
                "fallback_reason_counts": {"repeated_action_type": 1},
            },
        )
        self.assertEqual(0.5, metrics["action_diversity_score"])
        self.assertEqual(3, metrics["unique_action_type_count"])
        self.assertEqual(2, metrics["unique_clicked_target_count"])
        self.assertEqual(4, metrics["first_click_step"])
        self.assertEqual(1, metrics["open_detail_panel_count"])
        self.assertEqual({"repeated_action_type": 1}, metrics["fallback_reason_counts"])

    def test_no_functional_action_episode_count_is_one_when_no_functional_action(self) -> None:
        metrics = build_per_site_metric(
            "site003",
            {
                "status": "ok",
                "completed_episodes": 1,
                "episodes": 1,
                "functional_action_count": 0,
                "functional_action_signal_reward_total": 3.0,
            },
        )
        self.assertEqual(1, metrics["no_functional_action_episode_count"])
        self.assertEqual(0.0, metrics["functional_action_signal_reward_total"])


def _observation(*, url: str = "http://local", dom_node_count: int = 10) -> dict:
    return {
        "page_state": {
            "url": url,
            "title": "Example",
            "page_text_length": 100,
            "dom_node_count": dom_node_count,
        },
        "candidate_elements": [
            {
                "bid": "btn-1",
                "text": "Open",
                "name": "Open",
                "visible": True,
                "enabled": True,
                "clickable": True,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
