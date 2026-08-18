import unittest

from models.action_space import ActionSpace
from runners.evaluate_multisite_browsergym_agent import (
    _action_budget_status,
    _append_partial_transition,
    _scan_completed_reason,
    _select_dom_exhaustive_action_id,
    _valid_scan_run,
)
from services.action_opportunity_service import (
    action_ids_for_opportunities,
    build_action_opportunities,
    update_opportunity_history,
)
from services.autonomous_reward_service import calculate_autonomous_reward
from services.browsergym_training_service import _attach_action_history_to_observation
from tempfile import TemporaryDirectory
from pathlib import Path
import json


def _candidate(**overrides):
    base = {
        "candidate_id": "c0",
        "element_key": overrides.get("element_key", "button|submit|form-1"),
        "tag": "button",
        "role": "button",
        "type": "button",
        "name": "Submit",
        "text": "Submit",
        "visible": True,
        "enabled": True,
        "clickable": True,
        "fillable": False,
        "is_password": False,
        "is_submit": False,
        "is_form_control": False,
        "selector": "button",
        "bounding_box": {"x": 1, "y": 1, "width": 100, "height": 32},
        "semantic_action_type": "",
    }
    base.update(overrides)
    return base


def _observation(candidates=None, runtime=None):
    return {
        "candidate_elements": list(candidates or []),
        "page_state": {
            "url": "https://example.test/login",
            "dom_hash": "dom-1",
            "page_type": "web_page",
        },
        "runtime_signals": dict(runtime or {}),
        "history": {"step_index": 8},
    }


class ActionOpportunityEngineTests(unittest.TestCase):
    def test_action_opportunities_generated_from_dom_candidates(self):
        obs = _observation([_candidate(text="Add", semantic_action_type="add")])
        state = build_action_opportunities(obs)
        types = {item["opportunity_type"] for item in state["opportunities"]}
        self.assertIn("click_button", types)
        self.assertGreater(state["summary"]["generated_opportunity_count"], 0)

    def test_fillable_input_generates_fill_opportunity(self):
        obs = _observation([
            _candidate(
                element_key="textbox|username|form-1",
                tag="input",
                role="textbox",
                type="text",
                clickable=False,
                fillable=True,
                is_form_control=True,
            )
        ])
        types = {item["opportunity_type"] for item in build_action_opportunities(obs)["opportunities"]}
        self.assertIn("fill_username_or_email", types)

    def test_password_input_generates_required_fill_password_opportunity(self):
        obs = _observation([
            _candidate(
                element_key="textbox|password|form-1",
                tag="input",
                role="textbox",
                type="password",
                clickable=False,
                fillable=True,
                is_password=True,
                is_form_control=True,
            )
        ])
        state = build_action_opportunities(obs)
        password = [item for item in state["opportunities"] if item["opportunity_type"] == "fill_password"]
        self.assertEqual(1, len(password))
        self.assertTrue(password[0]["required"])

    def test_login_form_generates_named_required_opportunities(self):
        obs = _observation([
            _candidate(
                element_key="input|username|form-1",
                tag="input",
                role="textbox",
                type="text",
                name="username",
                text="Username",
                clickable=False,
                fillable=True,
                is_form_control=True,
            ),
            _candidate(
                element_key="input|password|form-1",
                tag="input",
                role="textbox",
                type="password",
                name="password",
                text="Password",
                clickable=False,
                fillable=True,
                is_password=True,
                is_form_control=True,
            ),
            _candidate(
                element_key="button|login|form-1",
                tag="button",
                role="button",
                type="submit",
                name="Login",
                text="Login",
                is_submit=True,
                clickable=True,
                is_form_control=True,
            ),
        ])
        state = build_action_opportunities(obs)
        required_types = {item["opportunity_type"] for item in state["opportunities"] if item["required"]}

        self.assertIn("fill_username_or_email", required_types)
        self.assertIn("fill_password", required_types)
        self.assertIn("click_login_submit", required_types)
        self.assertTrue(state["summary"]["has_login_form"])

    def test_form_generates_submit_form_opportunity(self):
        obs = _observation([
            _candidate(
                element_key="button|login|form-1",
                tag="button",
                role="button",
                type="submit",
                is_submit=True,
                is_form_control=True,
                semantic_action_type="submit",
            )
        ])
        state = build_action_opportunities(obs)
        submit = [item for item in state["opportunities"] if item["opportunity_type"] == "click_login_submit"]
        self.assertEqual(1, len(submit))
        self.assertFalse(submit[0]["required"])

    def test_network_request_generates_inspect_network_opportunity(self):
        obs = _observation(runtime={"network_request_count": 2})
        types = {item["opportunity_type"] for item in build_action_opportunities(obs)["opportunities"]}
        self.assertIn("inspect_network", types)

    def test_console_error_generates_inspect_console_opportunity(self):
        obs = _observation(runtime={"console_error_count": 1})
        types = {item["opportunity_type"] for item in build_action_opportunities(obs)["opportunities"]}
        self.assertIn("inspect_console", types)

    def test_anomaly_generates_verification_opportunity(self):
        obs = _observation()
        anomalies = [{"type": "button-no-response", "target_signature": "button|add"}]
        state = build_action_opportunities(obs, anomalies=anomalies)
        types = {item["opportunity_type"] for item in state["opportunities"]}
        self.assertIn("verify_anomaly_reproduction", types)
        self.assertEqual(1, state["summary"]["unverified_anomaly_count"])

    def test_finish_blocked_when_required_opportunity_remaining(self):
        obs = _observation([
            _candidate(
                element_key="textbox|password|form-1",
                tag="input",
                role="textbox",
                type="password",
                clickable=False,
                fillable=True,
                is_password=True,
            )
        ])
        _attach_action_history_to_observation(obs, obs["history"])
        space = ActionSpace(max_candidates=8)
        mask = space.build_action_mask(obs)
        self.assertEqual(0.0, mask[space.encode("finish_episode", 0)])

    def test_finish_allowed_when_required_opportunities_completed(self):
        obs = _observation([
            _candidate(
                element_key="button|login|form-1",
                type="submit",
                is_submit=True,
                semantic_action_type="submit",
            )
        ])
        first = build_action_opportunities(obs)
        ids = {item["opportunity_id"] for item in first["opportunities"]}
        history = {
            "step_index": 20,
            "executed_opportunity_ids": set(ids),
            "verified_opportunity_ids": set(ids),
        }
        state = build_action_opportunities(obs, history)
        self.assertTrue(state["summary"]["finish_allowed"])
        self.assertEqual("", state["summary"]["finish_blocked_reason"])

    def test_opportunity_coverage_rate_recorded(self):
        obs = _observation([_candidate(text="Add", semantic_action_type="add")])
        state = build_action_opportunities(obs)
        self.assertIn("action_opportunity_coverage_rate", state["summary"])
        self.assertEqual(0.0, state["summary"]["action_opportunity_coverage_rate"])

    def test_valid_scan_run_false_before_first_opportunity_action(self):
        obs = _observation([_candidate(text="Add", semantic_action_type="add")])
        state = build_action_opportunities(obs)
        self.assertFalse(state["summary"]["valid_scan_run"])
        self.assertEqual("no_opportunity_action_executed", state["summary"]["finish_blocked_reason"])
        self.assertTrue(state["summary"]["optional_opportunity_coverage_below_threshold"])

    def test_plain_click_button_opportunity_does_not_block_finish_after_some_action(self):
        obs = _observation([_candidate(element_key="button|plain|main", text="Plain", semantic_action_type="")])
        history = {"step_index": 3, "executed_action_signatures": {"inspect_dom::global"}}
        state = build_action_opportunities(obs, history)

        self.assertGreaterEqual(state["summary"]["remaining_opportunity_count"], 1)
        self.assertEqual(0, state["summary"]["remaining_blocking_opportunity_count"])
        self.assertTrue(state["summary"]["finish_allowed"])

    def test_high_value_click_button_opportunity_prioritized_but_does_not_block_finish(self):
        obs = _observation([_candidate(element_key="button|add|main", text="Add", semantic_action_type="add")])
        history = {"step_index": 3, "executed_action_signatures": {"inspect_dom::global"}}
        state = build_action_opportunities(obs, history)

        self.assertGreaterEqual(state["summary"]["remaining_opportunity_count"], 1)
        self.assertEqual(0, state["summary"]["remaining_blocking_opportunity_count"])
        self.assertTrue(state["summary"]["finish_allowed"])

    def test_ppo_selects_from_opportunity_action_mask(self):
        obs = _observation([
            _candidate(
                element_key="textbox|username|form-1",
                tag="input",
                role="textbox",
                type="text",
                clickable=False,
                fillable=True,
            ),
        ])
        _attach_action_history_to_observation(obs, obs["history"])
        space = ActionSpace(max_candidates=8)
        mask = space.build_action_mask(obs)
        self.assertGreater(mask[space.encode("fill_input", 0)], 0.0)
        # A real scan showed fillable_count staying >= 1 for 60 straight
        # steps while fill_input was never once selected: an unfilled
        # fillable field is now a required opportunity (like clickable
        # functional-priority candidates already were), so finish stays
        # blocked until the policy actually tries it at least once.
        self.assertEqual(mask[space.encode("finish_episode", 0)], 0.0)

        action = {"action_type": "fill_input", "candidate_index": 0, "action_signature": "fill_input::textbox|username|form-1"}
        update_opportunity_history(obs["history"], obs, action, step=1, success=True)
        _attach_action_history_to_observation(obs, obs["history"])
        mask_after_fill = space.build_action_mask(obs)
        self.assertGreater(mask_after_fill[space.encode("finish_episode", 0)], 0.0)

    def test_rule_engine_does_not_execute_actions_directly(self):
        history = {"step_index": 1}
        obs = _observation([_candidate(text="Add", semantic_action_type="add")])
        build_action_opportunities(obs, history)
        self.assertNotIn("executed_opportunity_ids", history)

    def test_verified_opportunity_removed_from_enabled_opportunities(self):
        obs = _observation([_candidate(element_key="button|add|card-1", text="Add", semantic_action_type="add")])
        initial = build_action_opportunities(obs)
        click_ids = [item["opportunity_id"] for item in initial["opportunities"] if item["opportunity_type"] == "click_button"]
        history = {
            "step_index": 10,
            "executed_opportunity_ids": set(click_ids),
            "verified_opportunity_ids": set(click_ids),
        }
        obs["history"] = history
        _attach_action_history_to_observation(obs, history)
        enabled = action_ids_for_opportunities(
            [item for item in obs["history"]["action_opportunities"] if not item.get("verified")],
            ActionSpace(max_candidates=8),
        )
        self.assertNotIn(ActionSpace(max_candidates=8).encode("click_element", 0), enabled)

    def test_opportunity_engine_ignores_metadata_fields(self):
        clean = _observation([_candidate(element_key="button|add|card-1", text="Add", semantic_action_type="add")])
        noisy = _observation([
            _candidate(
                element_key="button|add|card-1",
                text="Add",
                semantic_action_type="add",
                **{
                    "data_bug_id": "hidden",
                    "catalog_bug_id_matches": True,
                    "known_bug_id": "hidden",
                },
            )
        ])
        clean_state = build_action_opportunities(clean)
        noisy_state = build_action_opportunities(noisy)
        clean_pairs = [(item["opportunity_type"], item["target_element_key"]) for item in clean_state["opportunities"]]
        noisy_pairs = [(item["opportunity_type"], item["target_element_key"]) for item in noisy_state["opportunities"]]
        self.assertEqual(clean_pairs, noisy_pairs)

    def test_max_steps_with_remaining_required_marks_invalid_scan(self):
        summary = {
            "remaining_required_opportunity_count": 1,
            "required_opportunity_completion_rate": 0.5,
            "action_opportunity_coverage_rate": 0.6,
            "coverage_threshold": 0.7,
            "unverified_anomaly_count": 0,
            "finish_allowed": False,
        }
        reason = _scan_completed_reason(
            site_failed=False,
            early_stop_reasons={"max_steps": 1},
            max_steps_reached=True,
            opportunity_summary=summary,
        )
        self.assertEqual("invalid_completed_with_remaining_required_or_unverified_issue", reason)
        self.assertFalse(_valid_scan_run(summary, completed_reason=reason, site_failed=False))
        self.assertEqual(
            "insufficient_verification",
            _action_budget_status(max_steps_reached=True, completed_reason=reason, opportunity_summary=summary),
        )

    def test_max_steps_with_all_required_done_marks_valid_scan(self):
        summary = {
            "remaining_required_opportunity_count": 0,
            "required_opportunity_completion_rate": 1.0,
            "action_opportunity_coverage_rate": 0.9,
            "coverage_threshold": 0.7,
            "unverified_anomaly_count": 0,
            "finish_allowed": True,
        }
        reason = _scan_completed_reason(
            site_failed=False,
            early_stop_reasons={"max_steps": 1},
            max_steps_reached=True,
            opportunity_summary=summary,
        )
        self.assertEqual("completed_at_max_steps_with_all_required_done", reason)
        self.assertTrue(_valid_scan_run(summary, completed_reason=reason, site_failed=False))

    def test_required_opportunity_completion_rate_recorded(self):
        obs = _observation([
            _candidate(
                element_key="button|submit|form-1",
                type="submit",
                is_submit=True,
                semantic_action_type="submit",
            )
        ])
        state = build_action_opportunities(obs)
        self.assertIn("required_opportunity_completion_rate", state["summary"])

    def test_valid_scan_run_false_when_finish_not_allowed(self):
        summary = {
            "remaining_required_opportunity_count": 0,
            "required_opportunity_completion_rate": 1.0,
            "action_opportunity_coverage_rate": 0.2,
            "coverage_threshold": 0.7,
            "unverified_anomaly_count": 0,
            "finish_allowed": False,
        }
        self.assertFalse(
            _valid_scan_run(
                summary,
                completed_reason="invalid_completed_with_remaining_required_or_unverified_issue",
                site_failed=False,
            )
        )

    def test_completed_reason_standardized(self):
        summary = {
            "remaining_required_opportunity_count": 0,
            "required_opportunity_completion_rate": 1.0,
            "action_opportunity_coverage_rate": 0.8,
            "coverage_threshold": 0.7,
            "unverified_anomaly_count": 0,
            "finish_allowed": True,
        }
        reason = _scan_completed_reason(
            site_failed=False,
            early_stop_reasons={},
            max_steps_reached=False,
            opportunity_summary=summary,
        )
        self.assertEqual("completed_by_required_opportunities_done", reason)

    def test_action_timeline_records_selected_opportunity(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial_transitions.jsonl"
            _append_partial_transition(
                path,
                site_id="site-test",
                base_url="https://example.test",
                episode=1,
                step=1,
                action={"action_type": "fill_input"},
                target="username",
                reward=0.0,
                anomalies=[],
                url="https://example.test",
                mask_stats={
                    "selected_opportunity_id": "opp-1",
                    "selected_opportunity_type": "fill_text_input",
                    "finish_blocked_reason": "required_opportunity_remaining",
                },
            )
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual("opp-1", row["selected_opportunity_id"])
            self.assertEqual("fill_text_input", row["selected_opportunity_type"])

    def test_action_timeline_records_finish_blocked_reason(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial_transitions.jsonl"
            _append_partial_transition(
                path,
                site_id="site-test",
                base_url="https://example.test",
                episode=1,
                step=1,
                action={"action_type": "finish_episode"},
                target="",
                reward=-0.5,
                anomalies=[],
                url="https://example.test",
                mask_stats={"finish_allowed": False, "finish_blocked_reason": "opportunity_coverage_below_threshold"},
            )
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(row["finish_allowed"])
            self.assertEqual("opportunity_coverage_below_threshold", row["finish_blocked_reason"])

    def test_opportunity_status_transitions(self):
        obs = _observation([_candidate(element_key="button|add|card-1", text="Add", semantic_action_type="add")])
        history = {"step_index": 1}
        update_opportunity_history(
            history,
            obs,
            {"action_type": "click_element", "candidate_index": 0, "action_success": True},
            step=1,
            success=True,
        )
        state = build_action_opportunities(obs, history)
        click = [item for item in state["opportunities"] if item["opportunity_type"] == "click_button"][0]
        self.assertEqual("verified", click["status"])
        self.assertEqual(1, click["selected_step"])
        self.assertEqual(1, click["executed_step"])

    def test_finish_blocked_action_not_counted_as_successful_execution(self):
        obs = _observation([_candidate(type="submit", is_submit=True)])
        _attach_action_history_to_observation(obs, obs["history"])
        space = ActionSpace(max_candidates=8)
        mask = space.build_action_mask(obs)
        self.assertGreater(mask[space.encode("finish_episode", 0)], 0.0)

    def test_report_distinguishes_executed_action_count_and_opportunity_count(self):
        obs = _observation([_candidate(text="Add"), _candidate(text="Cart", element_key="button|cart|card-1")])
        state = build_action_opportunities(obs)
        self.assertGreater(state["summary"]["generated_opportunity_count"], 2)
        self.assertNotEqual(state["summary"]["generated_opportunity_count"], 1)

    def test_failed_action_requires_failure_reason(self):
        obs = _observation([_candidate(text="Add")])
        history = {"step_index": 1}
        update_opportunity_history(
            history,
            obs,
            {"action_type": "click_element", "candidate_index": 0, "failed": True},
            step=1,
            success=False,
        )
        state = build_action_opportunities(obs, history)
        failed = [item for item in state["opportunities"] if item.get("failed")]
        self.assertEqual("action_failed_without_reason", failed[0]["failure_reason"])

    def test_unverified_anomaly_blocks_finish(self):
        obs = _observation()
        state = build_action_opportunities(obs, anomalies=[{"type": "button-no-response", "signature": "a"}])
        self.assertFalse(state["summary"]["finish_allowed"])
        self.assertEqual("unverified_anomaly_remaining", state["summary"]["finish_blocked_reason"])

    def test_verified_anomaly_type_context_not_readded_with_new_evidence(self):
        obs = _observation()
        history = {"step_index": 1}
        update_opportunity_history(
            history,
            obs,
            {"action_type": "inspect_dom", "candidate_index": 0, "action_success": True},
            step=1,
            success=True,
            anomalies=[{"type": "button-no-response", "signature": "a", "evidence": "first"}],
        )
        state = build_action_opportunities(
            obs,
            history,
            anomalies=[{"type": "button-no-response", "signature": "a", "evidence": "second"}],
        )

        self.assertEqual(0, state["summary"]["unverified_anomaly_count"])
        self.assertTrue(state["summary"]["finish_allowed"])

    def test_shaping_reward_for_required_opportunity_completion(self):
        obs = _observation([
            _candidate(
                element_key="input|username|form-1",
                tag="input",
                role="textbox",
                type="text",
                name="username",
                clickable=False,
                fillable=True,
                is_form_control=True,
            ),
            _candidate(
                element_key="input|password|form-1",
                tag="input",
                role="textbox",
                type="password",
                name="password",
                clickable=False,
                fillable=True,
                is_password=True,
                is_form_control=True,
            ),
            _candidate(
                element_key="button|login|form-1",
                type="submit",
                name="Login",
                text="Login",
                is_submit=True,
                semantic_action_type="submit",
            )
        ])
        _attach_action_history_to_observation(obs, obs["history"])
        reward, breakdown = calculate_autonomous_reward(
            obs,
            obs,
            {"action_type": "fill_input", "candidate_index": 0, "action_success": True},
            [],
            history=obs["history"],
            site_profile={"reward_mode": "signal_based"},
        )
        self.assertGreaterEqual(breakdown["reward_required_opportunity_executed"], 0.1)
        self.assertGreater(reward, -0.01)

    def test_no_large_reward_for_opportunity_execution_without_verified_finding(self):
        obs = _observation([_candidate(text="Add", semantic_action_type="add")])
        _attach_action_history_to_observation(obs, obs["history"])
        _reward, breakdown = calculate_autonomous_reward(
            obs,
            obs,
            {"action_type": "click_element", "candidate_index": 0, "action_success": True},
            [],
            history=obs["history"],
            site_profile={"reward_mode": "signal_based"},
        )
        self.assertLessEqual(breakdown["opportunity_shaping_reward_total"], 0.05)
        self.assertEqual(0.0, breakdown["verified_finding_reward_total"])

    def test_dom_candidates_generate_action_opportunities(self):
        obs = _observation([
            _candidate(element_key="button|one|main", text="One"),
            _candidate(element_key="link|docs|nav", tag="a", role="link", href="/docs", text="Docs"),
            _candidate(element_key="input|search|header", tag="input", role="searchbox", type="search", fillable=True),
        ])
        state = build_action_opportunities(obs)
        types = {item["opportunity_type"] for item in state["opportunities"]}
        self.assertIn("click_button", types)
        self.assertIn("click_link", types)
        self.assertIn("fill_search_input", types)

    def test_opportunity_records_executable_element_fields(self):
        obs = _observation([
            _candidate(
                element_key="input|email|form",
                tag="input",
                role="textbox",
                type="email",
                name="email",
                placeholder="Email address",
                fillable=True,
                clickable=False,
            )
        ])
        opportunity = [item for item in build_action_opportunities(obs)["opportunities"] if item["opportunity_type"] == "fill_email_input"][0]
        self.assertEqual("input", opportunity["target_tag"])
        self.assertEqual("email", opportunity["target_type"])
        self.assertEqual("Email address", opportunity["target_placeholder"])
        self.assertTrue(opportunity["visible"])
        self.assertTrue(opportunity["enabled"])
        self.assertTrue(opportunity["fillable"])
        self.assertIn("action_signature", opportunity)

    def test_submit_button_generates_click_submit_opportunity(self):
        obs = _observation([_candidate(element_key="button|submit|form", type="submit", is_submit=True, text="Submit")])
        types = {item["opportunity_type"] for item in build_action_opportunities(obs)["opportunities"]}
        self.assertIn("click_submit", types)
        self.assertIn("submit_form", types)

    def test_same_action_signature_not_repeated(self):
        obs = _observation([_candidate(element_key="button|add|card-1", text="Add")])
        state = build_action_opportunities(obs)
        opportunity = [item for item in state["opportunities"] if item["opportunity_type"] == "click_button"][0]
        history = {
            "executed_action_signatures": {opportunity["action_signature"]},
            "step_index": 2,
        }
        repeated_state = build_action_opportunities(obs, history)
        repeated = [item for item in repeated_state["opportunities"] if item["opportunity_type"] == "click_button"][0]
        self.assertTrue(repeated["executed"])
        self.assertNotIn(repeated, [item for item in repeated_state["opportunities"] if not item.get("executed")])

    def test_failed_action_signature_retried_once_then_blacklisted(self):
        obs = _observation([_candidate(element_key="button|add|card-1", text="Add")])
        state = build_action_opportunities(obs)
        signature = [item for item in state["opportunities"] if item["opportunity_type"] == "click_button"][0]["action_signature"]
        history = {"failed_action_signature_counts": {signature: 2}, "step_index": 3}
        blacklisted = [item for item in build_action_opportunities(obs, history)["opportunities"] if item["opportunity_type"] == "click_button"][0]
        self.assertEqual("failed_action_signature_blacklisted", blacklisted["skipped_reason"])

    def test_dom_exhaustive_selects_required_opportunity_before_policy(self):
        obs = _observation([
            _candidate(
                element_key="input|password|form",
                tag="input",
                role="textbox",
                type="password",
                fillable=True,
                clickable=False,
                is_password=True,
            ),
            _candidate(element_key="button|later|main", text="Later"),
        ])
        history = obs["history"]
        _attach_action_history_to_observation(obs, history)
        space = ActionSpace(max_candidates=8)
        mask = space.build_action_mask(obs)
        action_id = _select_dom_exhaustive_action_id(space, obs, mask, history)
        self.assertEqual(space.encode("fill_input", 0), action_id)

    def test_dom_exhaustive_forces_optional_click_when_coverage_is_low(self):
        obs = _observation([_candidate(element_key="button|plain|main", text="Plain")])
        history = obs["history"]
        _attach_action_history_to_observation(obs, history)
        space = ActionSpace(max_candidates=8)
        mask = space.build_action_mask(obs)

        action_id = _select_dom_exhaustive_action_id(space, obs, mask, history)
        self.assertEqual(space.encode("click_element", 0), action_id)

    def test_network_failure_not_counted_as_failed_action_semantics(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial_transitions.jsonl"
            _append_partial_transition(
                path,
                site_id="site-test",
                base_url="https://example.test",
                episode=1,
                step=1,
                action={"action_type": "click_element"},
                target="submit",
                reward=0.0,
                anomalies=[{"type": "network-request-failed"}],
                url="https://example.test",
                step_info={"action_success": True, "network_request_failed_count": 1},
            )
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row["action_success"])
            self.assertEqual("", row["failure_reason"])


if __name__ == "__main__":
    unittest.main()
