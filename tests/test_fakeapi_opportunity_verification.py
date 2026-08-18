import unittest

from models.action_space import ActionSpace
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from runners.browsergym_api_server import _is_reportable_anomaly
from runners.evaluate_multisite_browsergym_agent import _action_budget_status, _opportunity_debug_summary, _scan_completed_reason
from services.action_opportunity_service import build_action_opportunities, update_opportunity_history
from services.anomaly_detection_service import detect_anomalies
from services.browsergym_training_service import _attach_action_history_to_observation, _update_history
from services.policy_safe_metrics import build_per_site_metric


def _candidate(**overrides):
    base = {
        "element_key": "button|submit|hero",
        "tag": "button",
        "role": "button",
        "type": "submit",
        "text": "Submit",
        "name": "Submit",
        "visible": True,
        "enabled": True,
        "clickable": True,
        "fillable": False,
        "is_submit": True,
        "is_form_control": True,
        "semantic_action_type": "submit",
        "bbox": [0, 0, 100, 32],
    }
    base.update(overrides)
    return base


def _fakeapi_obs(candidates=None, history=None):
    return {
        "candidate_elements": list(candidates or []),
        "page_state": {
            "url": "https://fakeapi.platzi.com/",
            "title": "Fake API",
            "page_text": "Fake API documentation products users search",
            "dom_hash": "dom-1",
        },
        "runtime_signals": {"network_request_count": 1},
        "history": dict(history or {}),
    }


class FakeApiOpportunityVerificationTests(unittest.TestCase):
    def test_fakeapi_like_page_requires_network_or_search_verification(self):
        obs = _fakeapi_obs([
            _candidate(tag="input", role="searchbox", type="search", fillable=True, clickable=False, is_submit=False, text=""),
            _candidate(tag="a", role="link", href="/api/v1/products", clickable=True, is_submit=False, text="/api/v1/products"),
        ])
        state = build_action_opportunities(obs)
        required = {item["opportunity_type"] for item in state["opportunities"] if item.get("required")}
        self.assertIn("inspect_network", required)
        self.assertIn("inspect_console", required)
        self.assertIn("check_api_response", required)
        self.assertIn("check_search_result_after_input", {item["opportunity_type"] for item in state["opportunities"]})

    def test_candidates_32_generates_expected_opportunities(self):
        candidates = [
            _candidate(element_key=f"button|endpoint|{index}", text=f"Endpoint {index}", name=f"Endpoint {index}")
            for index in range(32)
        ]
        obs = _fakeapi_obs(candidates)
        state = build_action_opportunities(obs)
        self.assertEqual(32, state["summary"]["detected_candidate_count"])
        self.assertGreaterEqual(state["summary"]["generated_opportunity_count"], 32)
        self.assertNotIn("candidates_exist_but_low_opportunity_generation", state["summary"]["opportunity_diagnostics"])

    def test_click_submit_not_repeated_consecutively_same_target(self):
        obs = _fakeapi_obs([_candidate()])
        history = {"step_index": 1}
        _attach_action_history_to_observation(obs, history)
        _update_history(
            history,
            obs,
            {"action_type": "click_submit", "candidate_index": 0, "action_success": False, "failed": True, "failure_reason": "timeout"},
            anomalies=[{"type": "action-error", "evidence": {"error": "timeout"}}],
        )
        obs["history"] = {}
        _attach_action_history_to_observation(obs, history)
        mask = ActionSpace(max_candidates=8).build_action_mask(obs)
        enabled = ActionSpace(max_candidates=8).enabled_action_types(mask)
        self.assertIn("inspect_network", enabled)
        self.assertNotIn("click_submit", enabled)

    def test_click_submit_not_repeated_before_verification_even_with_other_submit_candidate(self):
        obs = _fakeapi_obs([
            _candidate(element_key="button|submit|one"),
            _candidate(element_key="button|submit|two", text="Submit 2", name="Submit 2"),
        ])
        history = {"step_index": 1}
        _attach_action_history_to_observation(obs, history)
        _update_history(
            history,
            obs,
            {"action_type": "click_submit", "candidate_index": 0, "action_success": True},
            anomalies=[],
        )
        obs["history"] = {}
        _attach_action_history_to_observation(obs, history)
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(obs)
        self.assertEqual(0, action_space._valid_count_for_type(mask, "click_submit"))

    def test_check_api_response_opportunity_enables_network_verification(self):
        obs = _fakeapi_obs()
        history = {"step_index": 1}
        _attach_action_history_to_observation(obs, history)
        mask = ActionSpace(max_candidates=8).build_action_mask(obs)
        enabled = ActionSpace(max_candidates=8).enabled_action_types(mask)
        self.assertIn("inspect_network", enabled)
        self.assertNotIn("click_submit", enabled)

    def test_failed_submit_not_satisfied_by_execution_only(self):
        obs = _fakeapi_obs([_candidate()])
        history = {"step_index": 1}
        _attach_action_history_to_observation(obs, history)
        _update_history(
            history,
            obs,
            {"action_type": "click_submit", "candidate_index": 0, "action_success": False, "failed": True},
            anomalies=[{"type": "action-error", "evidence": {"error": "timeout"}}],
        )
        state = build_action_opportunities(obs, history)
        self.assertGreater(state["summary"]["remaining_required_opportunity_count"], 0)
        self.assertFalse(state["summary"]["finish_allowed"])

    def test_inspect_network_failure_records_failure_reason(self):
        env = BrowserGymJAWSEnv.__new__(BrowserGymJAWSEnv)
        env.env = None
        info = env._inspect_network()
        self.assertFalse(info["action_success"])
        self.assertEqual("active page not found", info["failure_reason"])
        self.assertFalse(info["network_log_available"])
        self.assertTrue(info["network_capture_enabled"])

    def test_network_failure_requires_successful_verification_or_incomplete(self):
        obs = _fakeapi_obs()
        history = {"step_index": 1}
        _attach_action_history_to_observation(obs, history)
        _update_history(
            history,
            obs,
            {"action_type": "inspect_network", "candidate_index": 0, "action_success": False, "failed": True, "failure_reason": "network capture failed"},
            anomalies=[{"type": "network-error", "signature": "net"}],
        )
        state = build_action_opportunities(obs, history)
        self.assertGreater(state["summary"]["remaining_required_opportunity_count"], 0)
        self.assertFalse(state["summary"]["finish_allowed"])

    def test_action_budget_status_insufficient_when_verification_incomplete(self):
        summary = {
            "remaining_required_opportunity_count": 1,
            "unverified_anomaly_count": 1,
            "failed_opportunity_count": 1,
            "action_opportunity_coverage_rate": 0.5,
            "coverage_threshold": 0.7,
        }
        reason = _scan_completed_reason(
            site_failed=False,
            early_stop_reasons={},
            max_steps_reached=False,
            opportunity_summary=summary,
        )
        self.assertEqual("invalid_completed_with_remaining_required_or_unverified_issue", reason)
        self.assertEqual(
            "insufficient_verification",
            _action_budget_status(max_steps_reached=False, completed_reason=reason, opportunity_summary=summary),
        )

    def test_six_actions_valid_only_when_remaining_required_zero(self):
        incomplete = {
            "remaining_required_opportunity_count": 1,
            "unverified_anomaly_count": 0,
            "failed_opportunity_count": 0,
            "required_opportunity_completion_rate": 0.8,
            "action_opportunity_coverage_rate": 0.8,
            "coverage_threshold": 0.7,
        }
        reason = _scan_completed_reason(
            site_failed=False,
            early_stop_reasons={},
            max_steps_reached=False,
            opportunity_summary=incomplete,
        )
        self.assertEqual("invalid_completed_with_remaining_required_or_unverified_issue", reason)
        complete = {
            "remaining_required_opportunity_count": 0,
            "unverified_anomaly_count": 0,
            "failed_opportunity_count": 0,
            "required_opportunity_completion_rate": 1.0,
            "action_opportunity_coverage_rate": 1.0,
            "coverage_threshold": 0.7,
            "finish_allowed": True,
        }
        self.assertIn(
            _scan_completed_reason(
                site_failed=False,
                early_stop_reasons={},
                max_steps_reached=False,
                opportunity_summary=complete,
            ),
            {"completed_by_coverage_saturation", "completed_by_required_opportunities_done"},
        )

    def test_failure_reason_missing_marks_invalid_scan(self):
        reason = _scan_completed_reason(
            site_failed=False,
            early_stop_reasons={},
            max_steps_reached=False,
            opportunity_summary={"remaining_required_opportunity_count": 0, "required_opportunity_completion_rate": 1.0, "action_opportunity_coverage_rate": 1.0, "finish_allowed": True},
            failed_action_count=1,
            missing_failure_reason_count=1,
        )
        self.assertEqual("invalid_completed_with_unexplained_action_failure", reason)

    def test_console_error_requires_console_verification(self):
        reason = _scan_completed_reason(
            site_failed=False,
            early_stop_reasons={},
            max_steps_reached=False,
            opportunity_summary={"remaining_required_opportunity_count": 0, "required_opportunity_completion_rate": 1.0, "action_opportunity_coverage_rate": 1.0, "finish_allowed": True},
            console_error_detected=True,
            console_verification_performed=False,
            console_verification_success=False,
        )
        self.assertEqual("invalid_completed_with_remaining_required_or_unverified_issue", reason)

    def test_submit_failure_triggers_verification_not_immediate_critical(self):
        before = _fakeapi_obs([_candidate()])
        after = _fakeapi_obs([_candidate()])
        anomalies = detect_anomalies(
            before,
            after,
            {
                "action": {"action_type": "click_submit", "candidate_index": 0},
                "last_action_error": True,
                "action_error": "selector timeout",
            },
        )
        action_errors = [item for item in anomalies if item.get("type") == "action-error"]
        self.assertTrue(action_errors)
        self.assertFalse(_is_reportable_anomaly(action_errors[0]))
        self.assertTrue(action_errors[0]["evidence"]["requires_followup_verification"])

    def test_failed_action_requires_failure_reason(self):
        obs = _fakeapi_obs([_candidate()])
        history = {"step_index": 1}
        _attach_action_history_to_observation(obs, history)
        _update_history(
            history,
            obs,
            {"action_type": "click_submit", "candidate_index": 0, "failed": True},
            anomalies=[{"type": "action-error", "evidence": {}}],
        )
        state = build_action_opportunities(obs, history)
        failed = [item for item in state["opportunities"] if item.get("failed")]
        self.assertTrue(failed)
        self.assertEqual("action_failed_without_reason", failed[0]["failure_reason"])

    def test_unverified_anomaly_blocks_finish(self):
        obs = _fakeapi_obs()
        history = {"unverified_anomalies": [{"type": "form-no-feedback", "signature": "search"}]}
        state = build_action_opportunities(obs, history)
        self.assertFalse(state["summary"]["finish_allowed"])
        self.assertGreater(state["summary"]["unverified_anomaly_count"], 0)

    def test_console_error_generates_required_inspect_console_and_runtime_verification(self):
        obs = _fakeapi_obs(history={})
        obs["runtime_signals"] = {"console_error_count": 1}
        state = build_action_opportunities(obs)
        required = {item["opportunity_type"] for item in state["opportunities"] if item.get("required")}
        self.assertIn("inspect_console", required)
        self.assertIn("verify_runtime_error", required)
        self.assertFalse(state["summary"]["finish_allowed"])

    def test_finish_blocked_when_console_error_unverified(self):
        obs = _fakeapi_obs(history={})
        obs["runtime_signals"] = {"console_error_count": 1}
        state = build_action_opportunities(obs)
        self.assertFalse(state["summary"]["finish_allowed"])
        self.assertEqual("unverified_anomaly_remaining", state["summary"]["finish_blocked_reason"])

    def test_detected_issue_requires_verification_action(self):
        obs = _fakeapi_obs()
        state = build_action_opportunities(obs, anomalies=[{"type": "duplicated-rendering", "signature": "dup"}])
        required = {item["opportunity_type"] for item in state["opportunities"] if item.get("required")}
        self.assertIn("verify_duplicated_rendering", required)
        self.assertGreater(state["summary"]["unverified_anomaly_count"], 0)
        self.assertFalse(state["summary"]["finish_allowed"])

    def test_duplicated_rendering_not_verified_without_followup(self):
        obs = _fakeapi_obs()
        history = {"unverified_anomalies": [{"type": "duplicated-rendering", "signature": "dup"}]}
        update_opportunity_history(
            history,
            obs,
            {"action_type": "click_element", "candidate_index": 0, "action_success": True},
            success=True,
        )
        state = build_action_opportunities(obs, history)
        self.assertGreater(state["summary"]["unverified_anomaly_count"], 0)
        self.assertFalse(state["summary"]["anomaly_verification_completed"])

    def test_duplicated_rendering_verified_by_followup_dom_inspection(self):
        obs = _fakeapi_obs()
        history = {"unverified_anomalies": [{"type": "duplicated-rendering", "signature": "dup"}]}
        update_opportunity_history(
            history,
            obs,
            {"action_type": "inspect_dom", "candidate_index": 0, "action_success": True},
            success=True,
        )
        state = build_action_opportunities(obs, history)
        self.assertEqual(0, state["summary"]["unverified_anomaly_count"])
        self.assertTrue(state["summary"]["anomaly_verification_completed"])

    def test_inspect_dom_repetition_penalized_same_context(self):
        obs = _fakeapi_obs()
        history = {}
        update_opportunity_history(history, obs, {"action_type": "inspect_dom", "action_success": True}, success=True)
        update_opportunity_history(history, obs, {"action_type": "inspect_dom", "action_success": True}, success=True)
        self.assertEqual(1, history["repeated_inspect_dom_count"])
        self.assertEqual(1, history["repeated_inspect_dom_penalty_applied"])

    def test_inspect_dom_repetition_blocked_same_context_without_verification(self):
        obs = _fakeapi_obs()
        history = {"step_index": 6}
        update_opportunity_history(history, obs, {"action_type": "inspect_dom", "action_success": True}, success=True)
        obs["history"] = history
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(obs)
        self.assertNotIn("inspect_dom", action_space.enabled_action_types(mask))

    def test_required_opportunity_mask_prefers_highest_priority_verification(self):
        obs = _fakeapi_obs()
        obs["runtime_signals"] = {"console_error_count": 1}
        history = {"step_index": 6}
        _attach_action_history_to_observation(obs, history)
        action_space = ActionSpace(max_candidates=8)
        mask = action_space.build_action_mask(obs)
        enabled = action_space.enabled_action_types(mask)
        self.assertIn("inspect_console", enabled)
        self.assertNotIn("inspect_dom", enabled)

    def test_form_no_feedback_generates_verification_opportunity(self):
        obs = _fakeapi_obs()
        state = build_action_opportunities(obs, anomalies=[{"type": "form-no-feedback", "signature": "search"}])
        self.assertIn("verify_form_no_feedback", {item["opportunity_type"] for item in state["opportunities"]})

    def test_action_error_not_promoted_to_critical_without_evidence(self):
        anomaly = {"type": "action-error", "confidence": 0.55, "evidence": {"requires_followup_verification": True}}
        self.assertFalse(_is_reportable_anomaly(anomaly))

    def test_required_opportunity_remaining_marks_invalid_scan(self):
        obs = _fakeapi_obs()
        state = build_action_opportunities(obs)
        self.assertGreater(state["summary"]["remaining_required_opportunity_count"], 0)
        self.assertFalse(state["summary"]["valid_scan_run"])

    def test_report_shows_opportunity_coverage_metrics(self):
        obs = _fakeapi_obs()
        summary = build_action_opportunities(obs)["summary"]
        for key in (
            "generated_opportunity_count",
            "required_opportunity_count",
            "optional_opportunity_count",
            "action_opportunity_coverage_rate",
            "required_opportunity_completion_rate",
            "finish_allowed",
            "finish_blocked_reason",
        ):
            self.assertIn(key, summary)

    def test_opportunity_debug_flags_recorded(self):
        obs = _fakeapi_obs()
        history = {}
        state = build_action_opportunities(obs, history)
        history["action_opportunities"] = state["opportunities"]
        debug = _opportunity_debug_summary(
            history,
            state["summary"],
            finish_guard_checked_steps=3,
            finish_guard_blocked_count=2,
            ppo_selected_opportunity_ids=["opp-1"],
        )
        self.assertTrue(debug["opportunity_engine_enabled"])
        self.assertTrue(debug["finish_guard_enabled"])
        self.assertEqual(3, debug["finish_guard_checked_steps"])
        self.assertEqual(2, debug["finish_guard_blocked_count"])
        self.assertIn("check_api_response", debug["opportunity_count_by_type"])

    def test_policy_safe_metric_preserves_opportunity_debug(self):
        metric = build_per_site_metric(
            "site-x",
            {
                "opportunity_debug": {
                    "opportunity_engine_enabled": True,
                    "finish_guard_blocked_count": 2,
                }
            },
        )
        self.assertTrue(metric["opportunity_debug"]["opportunity_engine_enabled"])
        self.assertEqual(2, metric["opportunity_debug"]["finish_guard_blocked_count"])

    def test_valid_scan_run_false_when_only_five_actions_and_required_remaining(self):
        obs = _fakeapi_obs()
        state = build_action_opportunities(obs, {"executed_opportunity_ids": {"a", "b", "c", "d", "e"}})
        self.assertFalse(state["summary"]["valid_scan_run"])

    def test_finish_blocked_when_network_issue_unverified(self):
        obs = _fakeapi_obs()
        state = build_action_opportunities(obs, {"unverified_anomalies": [{"type": "network-error", "signature": "net"}]})
        self.assertFalse(state["summary"]["finish_allowed"])
        self.assertEqual("unverified_anomaly_remaining", state["summary"]["finish_blocked_reason"])

    def test_search_input_requires_result_or_no_feedback_verification(self):
        obs = _fakeapi_obs([
            _candidate(tag="input", role="searchbox", type="search", fillable=True, clickable=False, is_submit=False, text=""),
        ])
        state = build_action_opportunities(obs)
        self.assertIn("check_search_result_after_input", {item["opportunity_type"] for item in state["opportunities"]})


if __name__ == "__main__":
    unittest.main()
