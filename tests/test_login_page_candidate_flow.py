from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.browsergym_action_adapter import BrowserGymActionAdapter
from adapters.browsergym_observation_adapter import BrowserGymObservationAdapter
from models.action_space import ActionSpace
from runners.evaluate_multisite_browsergym_agent import (
    _action_budget_status,
    _login_form_unprocessed,
    _required_observation_work_pending,
    _scan_completed_reason,
    _valid_scan_run,
)
from services.browsergym_training_service import _attach_action_history_to_observation, _update_login_flow_state
from services.autonomous_reward_service import calculate_autonomous_reward


class LoginPageCandidateFlowTests(unittest.TestCase):
    def test_playwright_extracts_login_page_candidates(self) -> None:
        observation = _login_observation()

        candidates = observation["candidate_elements"]

        self.assertGreaterEqual(len(candidates), 3)
        self.assertTrue(any(item.get("name") == "username" for item in candidates))
        self.assertTrue(any(item.get("is_password") for item in candidates))
        self.assertTrue(any(item.get("is_submit") for item in candidates))

    def test_observation_contains_username_password_submit(self) -> None:
        candidates = _login_observation()["candidate_elements"]
        by_name = {str(item.get("name") or ""): item for item in candidates}

        self.assertTrue(by_name["username"]["fillable"])
        self.assertTrue(by_name["password"]["is_password"])
        self.assertTrue(any(item["is_submit"] and item["clickable"] for item in candidates))

    def test_action_mask_enables_fill_input_when_input_exists(self) -> None:
        observation = _login_observation()
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)

        self.assertGreater(action_space._valid_count_for_type(mask, "fill_input"), 0)

    def test_action_mask_enables_click_submit_when_submit_exists(self) -> None:
        observation = _login_observation()
        history = {}
        _update_login_flow_state(history, observation=observation, action={"action_type": "fill_input", "candidate_index": _first_index(observation, "name", "username")})
        _update_login_flow_state(history, observation=observation, action={"action_type": "fill_input", "candidate_index": _first_index(observation, "is_password")})
        _attach_action_history_to_observation(observation, history)
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)

        self.assertGreater(action_space._valid_count_for_type(mask, "click_submit"), 0)

    def test_action_mask_disables_click_when_candidates_zero(self) -> None:
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask({"candidate_elements": [], "page_state": {}, "runtime_signals": {}})

        self.assertEqual(0, action_space._valid_count_for_type(mask, "click_element"))
        self.assertEqual(0, action_space._valid_count_for_type(mask, "fill_input"))

    def test_fill_input_requires_fillable_candidate(self) -> None:
        observation = {"candidate_elements": [{"visible": True, "enabled": True, "tag": "button", "clickable": True, "fillable": False}]}
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)

        self.assertEqual(0, action_space._valid_count_for_type(mask, "fill_input"))

    def test_click_element_requires_clickable_candidate(self) -> None:
        observation = {"candidate_elements": [{"visible": True, "enabled": True, "tag": "input", "fillable": True, "clickable": False}]}
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)

        self.assertEqual(0, action_space._valid_count_for_type(mask, "click_element"))

    def test_click_without_target_is_not_success(self) -> None:
        adapter = BrowserGymActionAdapter()
        action = adapter.adapt({"action_type": "click_element", "candidate_index": 0}, {"candidate_elements": []})

        self.assertFalse(action.executable)
        self.assertEqual("click_element", action.action_type)

    def test_login_flow_requires_fill_before_submit(self) -> None:
        observation = _login_observation()
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)

        self.assertIn("fill_input", action_space.enabled_action_types(mask))
        self.assertNotIn("click_submit", action_space.enabled_action_types(mask))

    def test_action_timeline_records_candidate_counts(self) -> None:
        observation = _login_observation()
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)
        action_id = action_space.encode("fill_input", 0)
        stats = action_space.build_action_mask_stats(observation, mask, action_id)

        self.assertEqual(3, stats["candidate_count"])
        self.assertEqual(2, stats["fillable_count"])
        self.assertEqual(1, stats["password_input_count"])
        self.assertEqual(1, stats["submit_count"])
        self.assertIn("fill_input", stats["action_mask_enabled_actions"])

    def test_candidate_count_logged_from_observation(self) -> None:
        observation = _login_observation()
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)
        submit_index = _first_index(observation, "is_submit")
        stats = action_space.build_action_mask_stats(observation, mask, action_space.encode("click_submit", submit_index))

        self.assertEqual(len(observation["candidate_elements"]), stats["candidate_count"])

    def test_action_timeline_records_selected_target(self) -> None:
        observation = _login_observation()
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)
        submit_index = _first_index(observation, "is_submit")
        stats = action_space.build_action_mask_stats(observation, mask, action_space.encode("click_submit", submit_index))

        self.assertTrue(stats["selected_target_element_key"])
        self.assertEqual("Login", stats["selected_target_name"])
        self.assertEqual("button", stats["selected_target_role"])

    def test_candidate_count_not_zero_on_quotes_login_fixture(self) -> None:
        self.assertGreater(len(_login_observation()["candidate_elements"]), 0)

    def test_early_stop_blocked_when_login_form_unprocessed(self) -> None:
        self.assertTrue(_login_form_unprocessed(_login_observation(), {"action_type_counts": {}}))
        self.assertTrue(_login_form_unprocessed(_login_observation(), {"action_type_counts": {"fill_input": 1}}))
        history = {"login_flow": {"has_login_form": True, "submit_clicked": True, "submit_result_checked": False}}
        self.assertTrue(_login_form_unprocessed(_login_observation(), history))
        self.assertFalse(
            _login_form_unprocessed(
                _login_observation(),
                {
                    "action_type_counts": {"fill_input": 2, "click_submit": 1, "inspect_network": 1},
                    "login_flow": {"has_login_form": True, "login_flow_completed": True, "required_actions_remaining": []},
                },
            )
        )

    def test_early_stop_allowed_after_no_new_observation_and_required_actions_done(self) -> None:
        self.assertEqual(
            "",
            _required_observation_work_pending(
                _login_observation(),
                {"action_type_counts": {"fill_input": 2, "click_submit": 1, "inspect_network": 1, "inspect_console": 1}},
            ),
        )

    def test_login_form_flow_not_completed_after_only_9_steps(self) -> None:
        observation = _login_observation()
        observation["history"]["step_index"] = 9
        observation["history"]["action_type_counts"] = {}
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)

        self.assertEqual(0, action_space._valid_count_for_type(mask, "finish_episode"))

    def test_completed_invalid_when_login_flow_incomplete(self) -> None:
        summary = {
            "has_login_form": True,
            "login_flow_completed": False,
            "remaining_required_opportunity_count": 3,
            "required_opportunity_completion_rate": 0.0,
            "unverified_anomaly_count": 0,
            "failed_opportunity_count": 0,
            "action_opportunity_coverage_rate": 0.2,
            "coverage_threshold": 0.7,
            "finish_allowed": False,
        }

        reason = _scan_completed_reason(
            site_failed=False,
            early_stop_reasons={},
            max_steps_reached=False,
            opportunity_summary=summary,
        )

        self.assertEqual("invalid_completed_with_login_flow_incomplete", reason)
        self.assertEqual(
            "insufficient_login_flow",
            _action_budget_status(max_steps_reached=False, completed_reason=reason, opportunity_summary=summary),
        )
        self.assertFalse(_valid_scan_run(summary, completed_reason=reason, site_failed=False))

    def test_login_form_tracks_email_password_submit_state(self) -> None:
        observation = _login_observation()
        history = {}

        state = _update_login_flow_state(history, observation=observation, action=None)
        self.assertEqual("input_detected", state["login_flow_status"])
        self.assertIn("fill_username_or_email", state["required_actions_remaining"])
        self.assertTrue(state["has_username_or_email_input"])
        self.assertTrue(state["has_login_submit"])

        _update_login_flow_state(history, observation=observation, action={"action_type": "fill_input", "candidate_index": _first_index(observation, "name", "username")})
        _update_login_flow_state(history, observation=observation, action={"action_type": "fill_input", "candidate_index": _first_index(observation, "is_password")})
        state = _update_login_flow_state(history, observation=observation, action={"action_type": "click_submit", "candidate_index": _first_index(observation, "is_submit")})

        self.assertTrue(state["email_input_filled"])
        self.assertTrue(state["username_or_email_filled"])
        self.assertTrue(state["password_input_filled"])
        self.assertTrue(state["password_filled"])
        self.assertTrue(state["submit_clicked"])
        self.assertTrue(state["login_submit_clicked"])
        self.assertFalse(state.get("login_flow_completed", False))
        self.assertEqual(["verify_login_result"], state["required_actions_remaining"])

    def test_login_flow_completed_only_after_result_verification(self) -> None:
        observation = _login_observation()
        history = {}
        _update_login_flow_state(history, observation=observation, action={"action_type": "fill_input", "candidate_index": _first_index(observation, "name", "username")})
        _update_login_flow_state(history, observation=observation, action={"action_type": "fill_input", "candidate_index": _first_index(observation, "is_password")})
        _update_login_flow_state(history, observation=observation, action={"action_type": "click_submit", "candidate_index": _first_index(observation, "is_submit")})

        state = _update_login_flow_state(history, observation=observation, action={"action_type": "inspect_network", "candidate_index": 0})

        self.assertTrue(state["submit_result_checked"])
        self.assertTrue(state["login_result_checked"])
        self.assertTrue(state["login_flow_completed"])
        self.assertEqual("verified", state["login_flow_status"])

    def test_password_input_prioritized_after_email_filled(self) -> None:
        observation = _login_observation()
        history = {}
        _update_login_flow_state(history, observation=observation, action={"action_type": "fill_input", "candidate_index": _first_index(observation, "name", "username")})
        _attach_action_history_to_observation(observation, history)
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)

        self.assertGreater(action_space._valid_count_for_type(mask, "fill_input"), 0)
        self.assertEqual(0, action_space._valid_count_for_type(mask, "click_submit"))

    def test_submit_prioritized_after_required_inputs_filled(self) -> None:
        observation = _login_observation()
        history = {}
        _update_login_flow_state(history, observation=observation, action={"action_type": "fill_input", "candidate_index": _first_index(observation, "name", "username")})
        _update_login_flow_state(history, observation=observation, action={"action_type": "fill_input", "candidate_index": _first_index(observation, "is_password")})
        _attach_action_history_to_observation(observation, history)
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)

        self.assertEqual(["click_submit"], action_space.enabled_action_types(mask))

    def test_submit_result_verification_required_before_completed(self) -> None:
        observation = _login_observation()
        history = {}
        _update_login_flow_state(history, observation=observation, action={"action_type": "fill_input", "candidate_index": _first_index(observation, "name", "username")})
        _update_login_flow_state(history, observation=observation, action={"action_type": "fill_input", "candidate_index": _first_index(observation, "is_password")})
        _update_login_flow_state(history, observation=observation, action={"action_type": "click_submit", "candidate_index": _first_index(observation, "is_submit")})
        _attach_action_history_to_observation(observation, history)
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(observation)

        self.assertIn("inspect_network", action_space.enabled_action_types(mask))
        self.assertEqual(0, action_space._valid_count_for_type(mask, "finish_episode"))

    def test_reward_login_form_coverage_and_penalty(self) -> None:
        observation = _login_observation()
        history = {"login_flow": {"has_login_form": True, "has_email_or_text_input": True, "has_password_input": True, "has_submit_button": True}}
        reward, breakdown = calculate_autonomous_reward(
            observation,
            observation,
            {"action_type": "fill_input", "candidate_index": _first_index(observation, "name", "username")},
            [],
            [],
            history,
            {"reward_mode": "signal_based"},
        )

        self.assertGreaterEqual(reward, 0.0)
        self.assertEqual(0.1, breakdown["reward_email_input_filled"])

    def test_early_finish_gets_login_flow_penalty(self) -> None:
        observation = _login_observation()
        _, breakdown = calculate_autonomous_reward(
            observation,
            observation,
            {"action_type": "finish_episode", "candidate_index": 0},
            [],
            [],
            {"login_flow": {"has_login_form": True, "login_flow_completed": False}},
            {"reward_mode": "signal_based"},
        )

        self.assertEqual(-1.0, breakdown["penalty_login_flow_incomplete_early_stop"])


def _login_observation() -> dict:
    raw = {
        "url": "https://quotes.toscrape.com/login",
        "title": "Login",
        "text": "Login Username Password",
        "playwright_candidates": [
            {
                "bid": "pw-username",
                "selector": 'input[name="username"]',
                "tag": "input",
                "role": "textbox",
                "type": "text",
                "name": "username",
                "text": "username",
                "placeholder": "Username",
                "visible": True,
                "enabled": True,
                "fillable": True,
                "clickable": False,
                "bounding_box": {"x": 10, "y": 10, "width": 180, "height": 24},
            },
            {
                "bid": "pw-password",
                "selector": 'input[name="password"]',
                "tag": "input",
                "role": "textbox",
                "type": "password",
                "name": "password",
                "text": "password",
                "placeholder": "Password",
                "visible": True,
                "enabled": True,
                "fillable": True,
                "is_password": True,
                "clickable": False,
                "bounding_box": {"x": 10, "y": 40, "width": 180, "height": 24},
            },
            {
                "bid": "pw-submit",
                "selector": 'input[type="submit"]',
                "tag": "input",
                "role": "button",
                "type": "submit",
                "name": "Login",
                "text": "Login",
                "visible": True,
                "enabled": True,
                "clickable": True,
                "is_submit": True,
                "bounding_box": {"x": 10, "y": 70, "width": 80, "height": 28},
            },
        ],
    }
    return BrowserGymObservationAdapter(max_candidates=8).convert(raw)


def _first_index(observation: dict, key: str, value: object = True) -> int:
    for index, candidate in enumerate(observation.get("candidate_elements", [])):
        if candidate.get(key) == value or (value is True and candidate.get(key)):
            return index
    raise AssertionError(f"candidate with {key} not found")


if __name__ == "__main__":
    unittest.main()
