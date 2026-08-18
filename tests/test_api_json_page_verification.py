from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from runners.browsergym_api_server import _looks_like_api_json_url
from runners.evaluate_multisite_browsergym_agent import _required_observation_work_pending
from services.autonomous_reward_service import calculate_autonomous_reward
from services.browsergym_training_service import _attach_action_history_to_observation, _update_page_verification_state


class ApiJsonPageVerificationTests(unittest.TestCase):
    def test_api_json_page_requires_minimum_verifications(self) -> None:
        history: dict = {}
        state = _update_page_verification_state(history, observation=_api_json_observation(), action=None)

        self.assertEqual("api_json_page", state["page_type"])
        self.assertIn("api_response_check", state["required_verifications_remaining"])
        self.assertIn("schema_contract_check", state["required_verifications_remaining"])
        self.assertFalse(state["required_verifications_completed"])

    def test_api_json_page_not_completed_after_only_layout_checks(self) -> None:
        observation = _api_json_observation()
        history = {"step_index": 12, "min_steps": 20}
        _update_page_verification_state(history, observation=observation, action={"action_type": "inspect_layout"})
        _attach_action_history_to_observation(observation, history)
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)

        self.assertEqual(0, action_space._valid_count_for_type(mask, "finish_episode"))
        self.assertNotIn("inspect_layout", action_space.enabled_action_types(mask))

    def test_early_stop_blocked_when_required_verifications_remaining(self) -> None:
        observation = _api_json_observation()
        history = {"step_index": 21, "min_steps": 20}
        _update_page_verification_state(history, observation=observation, action=None)

        self.assertEqual("required_verification", _required_observation_work_pending(observation, history))

    def test_api_json_page_prioritizes_network_and_schema_checks(self) -> None:
        observation = _api_json_observation()
        history = {"step_index": 21, "min_steps": 20}
        _attach_action_history_to_observation(observation, history)
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)
        enabled = action_space.enabled_action_types(mask)

        self.assertIn("inspect_last_api_response", enabled)
        self.assertNotIn("inspect_layout", enabled)

        _update_page_verification_state(history, observation=observation, action={"action_type": "inspect_last_api_response"})
        _attach_action_history_to_observation(observation, history)
        mask = action_space.build_action_mask(observation)
        self.assertIn("validate_last_api_schema", action_space.enabled_action_types(mask))

    def test_opportunity_work_not_min_steps_blocks_completion(self) -> None:
        observation = _api_json_observation()
        history = {"step_index": 6, "min_steps": 0}
        _attach_action_history_to_observation(observation, history)
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)

        self.assertEqual(0, action_space._valid_count_for_type(mask, "finish_episode"))
        self.assertIn(
            _required_observation_work_pending(observation, history),
            {"required_verification", "required_opportunity", "opportunity_coverage"},
        )

    def test_inspect_layout_repetition_penalized_same_context(self) -> None:
        observation = _api_json_observation()
        history = {
            "page_verification": {"page_type": "api_json_page"},
            "repeated_inspect_layout_same_context_count": 1,
        }

        _, breakdown = calculate_autonomous_reward(
            observation,
            observation,
            {"action_type": "inspect_layout", "candidate_index": 0},
            [],
            [],
            history,
            {"reward_mode": "signal_based"},
        )

        self.assertEqual(-0.2, breakdown["penalty_repeated_inspect_layout_same_context"])

    def test_change_viewport_mobile_failure_records_reason_shape(self) -> None:
        # Unit-level guard for the report fields expected from env step_info rows.
        row = {
            "action": "change_viewport_mobile",
            "action_success": False,
            "failure_reason": "viewport change failed",
            "exception_type": "TimeoutError",
            "attempted_viewport_width": 375,
            "attempted_viewport_height": 812,
        }

        self.assertEqual("viewport change failed", row["failure_reason"])
        self.assertEqual(375, row["attempted_viewport_width"])

    def test_api_json_url_detection(self) -> None:
        self.assertTrue(_looks_like_api_json_url("https://my-json-server.typicode.com/typicode/demo"))
        self.assertTrue(_looks_like_api_json_url("https://my-json-server.typicode.com/typicode/demo/posts"))


def _api_json_observation() -> dict:
    return {
        "page_state": {
            "url": "https://my-json-server.typicode.com/typicode/demo/posts",
            "title": "JSON",
            "page_text": '[{"id":1,"title":"Post 1"}]',
            "page_text_length": 27,
            "viewport_type": "desktop",
            "viewport_width": 1280,
            "viewport_height": 720,
        },
        "runtime_signals": {},
        "infra_signals": {},
        "candidate_elements": [],
    }


if __name__ == "__main__":
    unittest.main()
