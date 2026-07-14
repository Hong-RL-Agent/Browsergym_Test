from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from runners.evaluate_multisite_browsergym_agent import _apply_eval_fallback_mask
from services.browsergym_training_service import _enrich_action, _update_history


class ActionDiversityVerificationTests(unittest.TestCase):
    def test_high_value_click_followed_by_verification_action(self) -> None:
        action_space = ActionSpace(max_candidates=1)
        observation = _observation()
        history = {"verification_pending_after_high_value_click": True, "action_type_counts": {"click_element": 1}}
        mask = action_space.build_action_mask(observation)

        filtered, warning, repeated = _apply_eval_fallback_mask(action_space, mask, observation, history)

        self.assertEqual(1, repeated)
        self.assertIn("verification action", warning)
        self.assertEqual(1.0, float(filtered[action_space.encode("inspect_dom", 0)]))
        self.assertEqual(1, int(history.get("verification_action_redirect_count", 0)))

    def test_action_diversity_redirect_after_repeated_click_type(self) -> None:
        action_space = ActionSpace(max_candidates=1)
        observation = _observation()
        history = {
            "verification_pending_after_high_value_click": True,
            "last_action_type": "click_element",
            "consecutive_action_type_counts": {"click_element": 3},
            "action_type_counts": {"click_element": 3},
        }
        filtered, _, _ = _apply_eval_fallback_mask(action_space, action_space.build_action_mask(observation), observation, history)
        self.assertEqual(1.0, float(filtered[action_space.encode("inspect_dom", 0)]))

    def test_verification_action_not_repeated(self) -> None:
        action_space = ActionSpace(max_candidates=1)
        observation = _observation()
        history = {
            "verification_pending_after_high_value_click": True,
            "action_type_counts": {"click_element": 1, "inspect_dom": 1},
        }
        filtered, _, _ = _apply_eval_fallback_mask(action_space, action_space.build_action_mask(observation), observation, history)
        self.assertEqual(0.0, float(filtered[action_space.encode("inspect_dom", 0)]))
        self.assertEqual(1.0, float(filtered[action_space.encode("inspect_network", 0)]))

    def test_history_records_verification_after_high_value_click(self) -> None:
        history: dict = {}
        observation = _observation()
        click = {"action_type": "click_element", "candidate_index": 0}
        _enrich_action(click, observation)
        _update_history(history, observation, click, [])
        self.assertTrue(history["verification_pending_after_high_value_click"])
        self.assertEqual(1, history["high_value_click_pending_verification_count"])
        self.assertEqual(1, history["high_value_click_without_verification_count"])
        inspect = {"action_type": "inspect_dom", "candidate_index": 0}
        _enrich_action(inspect, observation)
        _update_history(history, observation, inspect, [])
        self.assertFalse(history["verification_pending_after_high_value_click"])
        self.assertEqual(1, history["verification_action_after_high_value_click_count"])
        self.assertEqual(1, history["high_value_click_verified_count"])
        self.assertEqual(0, history["high_value_click_pending_verification_count"])
        self.assertEqual(0, history["high_value_click_without_verification_count"])

    def test_high_value_click_verification_only_once(self) -> None:
        action_space = ActionSpace(max_candidates=1)
        observation = _observation()
        history: dict = {}
        click = {"action_type": "click_element", "candidate_index": 0}
        _enrich_action(click, observation)
        _update_history(history, observation, click, [])
        inspect = {"action_type": "inspect_dom", "candidate_index": 0}
        _enrich_action(inspect, observation)
        _update_history(history, observation, inspect, [])

        filtered, warning, _ = _apply_eval_fallback_mask(
            action_space,
            action_space.build_action_mask(observation),
            observation,
            history,
        )

        self.assertNotIn("verification action", warning)
        self.assertEqual(0, int(history.get("verification_action_redirect_count", 0)))
        self.assertEqual(1.0, float(filtered[action_space.encode("click_element", 0)]))

    def test_after_verification_returns_to_functional_exploration(self) -> None:
        action_space = ActionSpace(max_candidates=1)
        observation = _observation()
        history = {
            "verification_pending_after_high_value_click": False,
            "action_type_counts": {"click_element": 1, "inspect_dom": 1},
        }

        filtered, _, _ = _apply_eval_fallback_mask(
            action_space,
            action_space.build_action_mask(observation),
            observation,
            history,
        )

        self.assertEqual(1.0, float(filtered[action_space.encode("click_element", 0)]))
        self.assertEqual(0.0, float(filtered[action_space.encode("inspect_dom", 0)]))

    def test_repeated_meta_action_priority_lowered(self) -> None:
        action_space = ActionSpace(max_candidates=1)
        observation = _observation()
        history = {"action_type_counts": {"inspect_console": 1}}

        filtered, warning, repeated = _apply_eval_fallback_mask(
            action_space,
            action_space.build_action_mask(observation),
            observation,
            history,
        )

        self.assertEqual(1, repeated)
        self.assertIn("repeated meta action", warning)
        self.assertEqual(0.0, float(filtered[action_space.encode("inspect_console", 0)]))
        self.assertEqual(1, int(history.get("meta_action_repeated_priority_lowered_count", 0)))
        self.assertEqual(1, int(history.get("policy_selected_meta_action_suppressed_count", 0)))

    def test_policy_selected_meta_action_suppressed_count(self) -> None:
        action_space = ActionSpace(max_candidates=1)
        observation = _observation()
        history = {
            "last_action_type": "inspect_console",
            "consecutive_action_type_counts": {"inspect_console": 2},
            "action_type_counts": {"inspect_console": 1},
        }

        _apply_eval_fallback_mask(action_space, action_space.build_action_mask(observation), observation, history)

        self.assertGreaterEqual(int(history.get("policy_selected_meta_action_suppressed_count", 0)), 1)

    def test_no_bug_label_used_for_verification_routing(self) -> None:
        source = (ROOT / "runners" / "evaluate_multisite_browsergym_agent.py").read_text(encoding="utf-8")
        start = source.index("def _prefer_verification_action(")
        end = source.index("def _suppress_repeated_action_signatures(", start)
        snippet = source[start:end].lower()
        for forbidden in ("bug_id", "data-bug-id", "catalog", "site_group", "target_signal_types"):
            self.assertNotIn(forbidden, snippet)


def _observation() -> dict:
    return {
        "page_state": {"site_id": "site005", "url": "http://localhost", "title": "T"},
        "candidate_elements": [
            {
                "bid": "add",
                "text": "담기",
                "name": "담기",
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
                "element_key": "button|add|담기|test|pos-0-0-2|bbb",
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
