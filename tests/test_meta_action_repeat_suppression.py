from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from runners.evaluate_multisite_browsergym_agent import _apply_eval_fallback_mask
from services.autonomous_reward_service import calculate_autonomous_reward


class MetaActionRepeatSuppressionTests(unittest.TestCase):
    def test_inspect_console_repeat_suppressed(self) -> None:
        action_space = ActionSpace(max_candidates=2)
        observation = _observation()
        mask = action_space.build_action_mask(observation)
        history = {
            "last_action_type": "inspect_console",
            "action_type_counts": {"inspect_console": 1},
            "consecutive_action_type_counts": {"inspect_console": 1},
            "click_target_counts": {},
        }

        filtered, warning, repeated = _apply_eval_fallback_mask(action_space, mask, observation, history)

        self.assertEqual(0.0, float(filtered[action_space.encode("inspect_console", 0)]))
        self.assertGreater(float(filtered[action_space.encode("click_element", 0)]), 0.0)
        self.assertIn("repeated meta action", warning)
        self.assertEqual(1, repeated)
        self.assertEqual(1, int(history.get("inspect_console_repeat_suppression_count", 0)))

    def test_meta_action_redirects_to_unclicked_functional_candidate(self) -> None:
        action_space = ActionSpace(max_candidates=2)
        observation = _observation()
        mask = action_space.build_action_mask(observation)
        history = {
            "last_action_type": "inspect_network",
            "action_type_counts": {"inspect_network": 1},
            "consecutive_action_type_counts": {"inspect_network": 1},
            "click_target_counts": {"visited": 1},
            "memory_encoder_enabled": True,
        }

        filtered, _, _ = _apply_eval_fallback_mask(action_space, mask, observation, history)

        self.assertEqual(0.0, float(filtered[action_space.encode("click_element", 0)]))
        self.assertGreater(float(filtered[action_space.encode("click_element", 1)]), 0.0)
        self.assertGreaterEqual(int(history.get("meta_action_redirect_to_functional_count", 0)), 1)
        self.assertEqual(1, int(history.get("inspect_network_repeat_suppression_count", 0)))

    def test_repeated_action_penalty_applied_to_meta_actions(self) -> None:
        reward, breakdown = calculate_autonomous_reward(
            _observation(),
            _observation(),
            {"action_type": "inspect_console"},
            [],
            history={"action_type_counts": {"inspect_console": 1}, "functional_action_count": 1},
            site_profile={"reward_mode": "signal_based"},
        )

        self.assertLess(reward, 0.0)
        self.assertLess(breakdown["penalty_repeated_meta_action"], 0.0)
        self.assertEqual(1.0, breakdown["repeated_meta_action_penalty_count"])


def _observation() -> dict:
    return {
        "page_state": {"site_id": "site004", "url": "http://localhost:9223/", "title": "Courses", "page_text": "Courses"},
        "candidate_elements": [
            _candidate("visited", "Visited add"),
            _candidate("fresh", "Fresh add"),
        ],
        "runtime_signals": {},
    }


def _candidate(bid: str, text: str) -> dict:
    return {
        "bid": bid,
        "text": text,
        "name": text,
        "role": "button",
        "tag": "button",
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "clickable": True,
        "functional_priority_candidate": True,
        "functional_priority": True,
        "is_high_value_functional_candidate": True,
        "semantic_action_type": "add",
    }


if __name__ == "__main__":
    unittest.main()
