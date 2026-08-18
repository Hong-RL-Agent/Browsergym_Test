import unittest

from exporters.pdf_report_exporter import render_report_html


class ReportOpportunityMetricsTests(unittest.TestCase):
    def test_report_template_shows_opportunity_metrics_for_fakeapi_run(self):
        html = render_report_html(
            {
                "title": "Report",
                "mode": "BrowserGym Autonomous PPO",
                "report_id": "r1",
                "created_at": "now",
                "site_id": "openended",
                "page_name": "FakeAPI",
                "base_url": "https://fakeapi.platzi.com/",
                "author": "test",
                "environment": {"execution_mode": "Autonomous", "scenario_used": False, "engine": "BrowserGym"},
                "summary": {
                    "recall": None,
                    "precision": None,
                    "known_bug_match_count": 0,
                    "average_reward": 0,
                    "total_detected_candidates": 1,
                    "unique_detected_candidates": 1,
                    "matched_bug_ids": [],
                    "missed_bug_ids": [],
                    "generated_opportunity_count": 9,
                    "required_opportunity_count": 4,
                    "executed_opportunity_count": 3,
                    "verified_opportunity_count": 1,
                    "failed_opportunity_count": 2,
                    "remaining_required_opportunity_count": 2,
                    "action_opportunity_coverage_rate": 0.33,
                    "required_opportunity_completion_rate": 0.5,
                    "unverified_anomaly_count": 1,
                    "finish_allowed": False,
                    "valid_scan_run": False,
                    "completed_reason": "invalid_completed_with_remaining_required_or_unverified_issue",
                    "verified_finding_count": 0,
                    "anomaly_verification_required": True,
                    "anomaly_verification_completed": False,
                    "finish_blocked_reason": "required_opportunity_remaining",
                    "opportunity_diagnostics": ["repeated_dom_inspection_without_new_signal"],
                    "action_budget_status": "insufficient_verification",
                    "failed_action_count": 1,
                    "executed_action_count": 9,
                    "successful_action_count": 8,
                    "action_log_count": 9,
                    "raw_event_log_count": 25,
                    "runner_failed_action_count": 5,
                    "timeline_failed_action_count": 1,
                    "summary_failed_action_count": 1,
                    "exploration_failed_action_count": 1,
                    "failed_action_count_mismatch": True,
                    "failed_action_count_mismatch_reason": "runner_summary_differs_from_action_timeline",
                    "valid_report_metrics": False,
                    "report_metric_warning": "failed_action_count_mismatch",
                    "network_failure_count": 6,
                    "console_error_count": 1,
                    "runtime_error_count": 0,
                    "warning_issue_count": 5,
                    "critical_issue_count": 1,
                    "total_problem_signal_count": 14,
                    "missing_failure_reason_count": 0,
                    "network_failure_detected": True,
                    "network_verification_performed": True,
                    "network_verification_success": False,
                    "console_error_detected": True,
                    "console_verification_performed": False,
                    "console_verification_success": False,
                    "raw_anomaly_candidate_count": 6,
                    "verified_finding_count": 1,
                    "filtered_false_positive_count": 3,
                    "duplicate_anomaly_count": 2,
                    "anomaly_filter_reason_counts": {"duplicate_anomaly": 2, "low_confidence_below_threshold": 1},
                    "finding_promotion_reason": "high_confidence_observed_signal",
                    "finding_rejection_reason": "duplicate_anomaly",
                    "action_diversity_score": 0.5,
                    "unique_action_type_count": 3,
                    "repeated_action_type_count": 2,
                    "repeated_strategy_count": 2,
                    "consecutive_same_strategy_count": 1,
                    "action_counts": {},
                },
                "model": None,
                "transition_summary": None,
                "api_anomalies": [],
                "bugs": [],
                "analysis": "",
            },
            "reports/templates/error_report_template.html",
        )
        self.assertIn("Opportunity / Finish Decision", html)
        self.assertIn("Remaining Required", html)
        self.assertIn("invalid_completed_with_remaining_required_or_unverified_issue", html)
        self.assertIn("insufficient_verification", html)
        self.assertIn("Network Verification", html)
        self.assertIn("Console Verification", html)
        self.assertIn("Action Execution", html)
        self.assertIn("Observed Error Signals", html)
        self.assertIn("failed_action_count_mismatch", html)
        self.assertIn("Total Problem Signals", html)
        self.assertIn("Anomaly Verification Required", html)
        self.assertIn("Opportunity Diagnostics", html)
        self.assertIn("repeated_dom_inspection_without_new_signal", html)
        self.assertIn("Raw Anomaly Candidates", html)
        self.assertIn("high_confidence_observed_signal", html)

    def test_report_contains_action_diversity_and_issue_filter_metrics(self):
        html = render_report_html(
            {
                "title": "Report",
                "mode": "BrowserGym Autonomous PPO",
                "report_id": "r2",
                "created_at": "now",
                "site_id": "openended",
                "page_name": "FakeAPI",
                "base_url": "https://fakeapi.platzi.com/",
                "author": "test",
                "environment": {"execution_mode": "Autonomous", "scenario_used": False, "engine": "BrowserGym"},
                "summary": {
                    "raw_anomaly_candidate_count": 4,
                    "verified_finding_count": 1,
                    "filtered_false_positive_count": 2,
                    "duplicate_anomaly_count": 1,
                    "action_diversity_score": 0.25,
                    "unique_action_type_count": 2,
                    "repeated_strategy_count": 3,
                    "anomaly_filter_reason_counts": {"duplicate_anomaly": 1},
                    "action_counts": {},
                },
                "model": None,
                "transition_summary": None,
                "api_anomalies": [],
                "bugs": [],
                "analysis": "",
            },
            "reports/templates/error_report_template.html",
        )
        self.assertIn("Action Diversity", html)
        self.assertIn("Filter Reason Counts", html)
        self.assertIn("duplicate_anomaly", html)


if __name__ == "__main__":
    unittest.main()
