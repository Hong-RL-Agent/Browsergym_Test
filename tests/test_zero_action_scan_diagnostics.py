import unittest

from models.action_space import ActionSpace
from runners.browsergym_api_server import _job_progress
from runners.evaluate_multisite_browsergym_agent import (
    _initial_scan_counts,
    _zero_action_diagnosis,
)
from services.policy_safe_metrics import build_per_site_metric


def _obs(candidates=None):
    return {
        "candidate_elements": list(candidates or []),
        "page_state": {"url": "https://example.test", "dom_hash": "dom-1"},
        "runtime_signals": {},
        "history": {},
    }


def _button():
    return {
        "element_key": "button|start|hero",
        "tag": "button",
        "role": "button",
        "text": "Start",
        "visible": True,
        "enabled": True,
        "clickable": True,
        "fillable": False,
        "bbox": [0, 0, 80, 32],
    }


class ZeroActionScanDiagnosticsTests(unittest.TestCase):
    def test_scan_invalid_when_executed_action_count_zero(self):
        metric = build_per_site_metric(
            "site-x",
            {
                "status": "ok",
                "scan_status": "failed_no_action_executed",
                "executed_action_count": 0,
                "valid_scan_run": False,
                "zero_action_diagnosis": "opportunities_enabled_but_no_action_executed",
            },
        )
        self.assertEqual("failed_no_action_executed", metric["scan_status"])
        self.assertFalse(metric["valid_scan_run"])
        self.assertEqual("opportunities_enabled_but_no_action_executed", metric["zero_action_diagnosis"])

    def test_scan_invalid_when_event_log_count_zero(self):
        diagnosis = _zero_action_diagnosis(
            executed_action_count=0,
            raw_event_log_count=0,
            action_loop_entered=True,
            action_loop_iteration_count=1,
            initial_observation_collected=True,
            candidate_count=1,
            generated_opportunity_count=1,
            enabled_opportunity_count=1,
            finish_only_mask=False,
        )
        self.assertEqual("opportunities_enabled_but_no_action_executed", diagnosis)

    def test_finish_blocked_before_first_action(self):
        obs = _obs([_button()])
        from runners.evaluate_multisite_browsergym_agent import _attach_action_history_to_observation

        history = {"step_index": 0, "min_steps": 0}
        _attach_action_history_to_observation(obs, history)
        mask = ActionSpace(max_candidates=8).build_action_mask(obs)
        self.assertEqual(0.0, mask[ActionSpace(max_candidates=8).encode("finish_episode", 0)])

    def test_action_loop_entered_recorded(self):
        metric = build_per_site_metric("site-x", {"action_loop_entered": True, "action_loop_iteration_count": 2})
        self.assertTrue(metric["action_loop_entered"])
        self.assertEqual(2, metric["action_loop_iteration_count"])

    def test_initial_observation_count_recorded(self):
        obs = _obs([_button()])
        from runners.evaluate_multisite_browsergym_agent import _attach_action_history_to_observation

        history = {"step_index": 0}
        _attach_action_history_to_observation(obs, history)
        counts = _initial_scan_counts(obs, history, ActionSpace(max_candidates=8))
        self.assertEqual(1, counts["initial_candidate_count"])
        self.assertEqual(1, counts["initial_clickable_count"])

    def test_candidates_exist_but_no_opportunities_diagnosed(self):
        diagnosis = _zero_action_diagnosis(
            executed_action_count=0,
            raw_event_log_count=0,
            action_loop_entered=True,
            action_loop_iteration_count=1,
            initial_observation_collected=True,
            candidate_count=3,
            generated_opportunity_count=0,
            enabled_opportunity_count=0,
            finish_only_mask=False,
        )
        self.assertEqual("candidates_exist_but_no_opportunities_generated", diagnosis)

    def test_opportunities_exist_but_all_masked_diagnosed(self):
        diagnosis = _zero_action_diagnosis(
            executed_action_count=0,
            raw_event_log_count=0,
            action_loop_entered=True,
            action_loop_iteration_count=1,
            initial_observation_collected=True,
            candidate_count=3,
            generated_opportunity_count=2,
            enabled_opportunity_count=0,
            finish_only_mask=False,
        )
        self.assertEqual("opportunities_exist_but_all_masked", diagnosis)

    def test_enabled_opportunities_but_no_action_executed_diagnosed(self):
        diagnosis = _zero_action_diagnosis(
            executed_action_count=0,
            raw_event_log_count=0,
            action_loop_entered=True,
            action_loop_iteration_count=1,
            initial_observation_collected=True,
            candidate_count=3,
            generated_opportunity_count=2,
            enabled_opportunity_count=2,
            finish_only_mask=False,
        )
        self.assertEqual("opportunities_enabled_but_no_action_executed", diagnosis)

    def test_executed_actions_do_not_get_zero_action_diagnosis(self):
        diagnosis = _zero_action_diagnosis(
            executed_action_count=7,
            raw_event_log_count=7,
            action_loop_entered=True,
            action_loop_iteration_count=8,
            initial_observation_collected=True,
            candidate_count=32,
            generated_opportunity_count=33,
            enabled_opportunity_count=33,
            finish_only_mask=False,
        )
        self.assertEqual("", diagnosis)

    def test_report_records_log_pipeline_counts(self):
        metric = build_per_site_metric(
            "site-x",
            {
                "runner_action_count": 2,
                "episode_action_count": 2,
                "result_json_action_count": 2,
                "report_json_action_count": 0,
            },
        )
        self.assertEqual(2, metric["runner_action_count"])
        self.assertEqual(0, metric["report_json_action_count"])

    def test_log_pipeline_mismatch_detected(self):
        metric = build_per_site_metric(
            "site-x",
            {
                "log_pipeline_mismatch": True,
                "log_pipeline_mismatch_stage": "result_json_to_report_json",
            },
        )
        self.assertTrue(metric["log_pipeline_mismatch"])
        self.assertEqual("result_json_to_report_json", metric["log_pipeline_mismatch_stage"])

    def test_completed_valid_requires_at_least_one_action(self):
        metric = build_per_site_metric(
            "site-x",
            {"scan_status": "failed_no_action_executed", "executed_action_count": 0, "valid_scan_run": False},
        )
        self.assertNotEqual("completed_valid", metric["scan_status"])

    def test_zero_action_scan_status_failed_or_invalid(self):
        metric = build_per_site_metric("site-x", {"scan_status": "completed_invalid", "executed_action_count": 0})
        self.assertIn(metric["scan_status"], {"completed_invalid", "failed_no_action_executed", "failed_no_observation"})

    def test_step_zero_finish_not_allowed(self):
        self.test_finish_blocked_before_first_action()

    def test_runner_exception_not_reported_as_completed_valid(self):
        metric = build_per_site_metric(
            "site-x",
            {
                "scan_status": "failed_runner_exception",
                "runner_exception": "boom",
                "valid_scan_run": False,
            },
        )
        self.assertEqual("failed_runner_exception", metric["scan_status"])
        self.assertFalse(metric["valid_scan_run"])

    def test_completed_invalid_progress_is_terminal(self):
        self.assertEqual(100, _job_progress(None, "completed_invalid"))
        self.assertEqual(100, _job_progress(None, "failed_no_action_executed"))


if __name__ == "__main__":
    unittest.main()
