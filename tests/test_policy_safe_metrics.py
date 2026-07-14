from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.policy_safe_metrics import (
    COMPARISON_NOTE_PATH,
    EVALUATION_DIR,
    TRAINING_LOG_DIR,
    build_metric_counts,
    build_per_site_metric,
    comparison_note,
    compute_signal_discovery_metrics,
    compute_precision_recall_f1,
    strip_known_bug_metrics_for_signal_discovery,
)
from runners.evaluate_multisite_browsergym_agent import _missed_bug_reasons


class PolicySafeMetricTests(unittest.TestCase):
    def test_no_memory_does_not_record_after_memory_rate(self) -> None:
        metric = build_per_site_metric(
            "site001",
            {
                "memory_encoder_enabled": False,
                "memory_state_update_count": 9,
                "memory_state_norm_mean": 3.0,
                "repeated_action_rate_after_memory": 0.5,
                "action_counts": {"click_element": 3},
            },
        )
        self.assertFalse(metric["memory_encoder_enabled"])
        self.assertEqual(0, metric["memory_state_update_count"])
        self.assertEqual(0.0, metric["memory_state_norm_mean"])
        self.assertIsNone(metric["repeated_action_rate_after_memory"])
        self.assertEqual(0.6667, metric["repeated_action_rate"])

    def test_memory_records_memory_state_updates(self) -> None:
        metric = build_per_site_metric(
            "site001",
            {
                "memory_encoder_enabled": True,
                "memory_encoder_type": "gru",
                "memory_hidden_size": 128,
                "memory_state_update_count": 9,
                "memory_state_norm_mean": 1.25,
                "repeated_action_rate_after_memory": 0.25,
            },
        )
        self.assertTrue(metric["memory_encoder_enabled"])
        self.assertEqual(9, metric["memory_state_update_count"])
        self.assertEqual(1.25, metric["memory_state_norm_mean"])
        self.assertEqual(0.25, metric["repeated_action_rate_after_memory"])

    def test_precision_recall_f1_safe_zero_division(self) -> None:
        metrics = compute_precision_recall_f1(
            true_positive_count=0,
            false_positive_count=0,
            false_negative_count=0,
        )
        self.assertEqual({"precision": 0.0, "recall": 0.0, "f1_score": 0.0}, metrics)

    def test_precision_recall_f1_counts(self) -> None:
        counts = build_metric_counts(
            known_bug_total=5,
            known_bug_matched_count=3,
            unique_detected_anomaly_count=7,
        )
        self.assertEqual(3, counts["true_positive_count"])
        self.assertEqual(4, counts["false_positive_count"])
        self.assertEqual(2, counts["false_negative_count"])
        metrics = compute_precision_recall_f1(
            true_positive_count=counts["true_positive_count"],
            false_positive_count=counts["false_positive_count"],
            false_negative_count=counts["false_negative_count"],
        )
        self.assertAlmostEqual(3 / 7, metrics["precision"])
        self.assertAlmostEqual(3 / 5, metrics["recall"])
        self.assertAlmostEqual(0.5, metrics["f1_score"])

    def test_v3_summary_paths_do_not_overwrite_v2(self) -> None:
        self.assertEqual(Path("artifacts/evaluation/v3_policy_safe"), EVALUATION_DIR)
        self.assertEqual(Path("artifacts/training_logs/v3_policy_safe"), TRAINING_LOG_DIR)
        self.assertEqual(Path("artifacts/evaluation/v3_policy_safe/comparison_note.json"), COMPARISON_NOTE_PATH)
        self.assertNotIn("v2", str(EVALUATION_DIR))
        self.assertNotIn("v2", str(TRAINING_LOG_DIR))

    def test_comparison_note_marks_v2_excluded(self) -> None:
        note = comparison_note()
        self.assertEqual("v2_browsergym_raw_obs", note["previous_version"])
        self.assertEqual("excluded_from_final_metrics", note["previous_result_status"])
        self.assertEqual("v3_policy_safe", note["current_version"])
        self.assertEqual("valid_after_policy_safe_audit", note["current_result_status"])
        self.assertIn("catalog_bug_id_matches", note["exclusion_reason"])

    def test_signal_discovery_metrics_used_when_known_bug_eval_disabled(self) -> None:
        metrics = compute_signal_discovery_metrics(
            [
                {
                    "multi_signal_anomaly_count": 2,
                    "console_error_count": 1,
                    "api_5xx_count": 1,
                    "security_signal_count": 0,
                }
            ],
            target_signal_types=["ui-dom", "console-error", "api-5xx"],
        )
        self.assertGreaterEqual(metrics["unique_signal_type_count"], 3)
        self.assertEqual(1.0, metrics["signal_type_coverage"])
        self.assertGreater(metrics["signal_discovery_score"], 0.0)

    def test_known_bug_metrics_moved_to_offline_eval_when_disabled(self) -> None:
        known_bug_metrics = {
            "precision": 1.0,
            "recall": 0.5,
            "f1_score": 2 / 3,
            "known_bug_total": 4,
        }
        output = {
            "use_known_bug_for_evaluation": False,
            "offline_known_bug_eval_metrics": known_bug_metrics,
            "precision": None,
            "recall": None,
            "f1_score": None,
        }
        self.assertIsNone(output["precision"])
        self.assertEqual(known_bug_metrics, output["offline_known_bug_eval_metrics"])

    def test_nested_known_bug_metrics_removed_when_known_bug_eval_disabled(self) -> None:
        stripped = strip_known_bug_metrics_for_signal_discovery(
            {
                "all_site_metrics": {"precision": 1.0, "recall": 0.5, "f1_score": 0.66, "known_bug_total": 4},
                "valid_site_metrics": {"precision": 1.0, "recall": 0.5, "f1_score": 0.66, "known_bug_total": 4},
                "failed_site_metrics": {"failed_site_count": 0},
            }
        )
        self.assertIsNone(stripped["all_site_metrics"]["precision"])
        self.assertIsNone(stripped["all_site_metrics"]["recall"])
        self.assertIsNone(stripped["valid_site_metrics"]["f1_score"])
        self.assertIsNone(stripped["valid_site_metrics"]["known_bug_total"])

    def test_missed_bug_reason_uses_semantic_attempt_when_catalog_disabled(self) -> None:
        reasons = _missed_bug_reasons(
            [
                {
                    "bug_id": "site003-bug01",
                    "type": "button-no-response",
                    "target_keywords": ["운동 추가"],
                    "action_hints": ["click"],
                }
            ],
            [],
            set(),
            {"click_element": 1},
            0,
            {"workout_add": 1},
            0,
        )
        self.assertIn("semantic action workout_add was clicked", reasons["site003-bug01"])
        self.assertNotEqual("no catalog candidate/action reached this bug", reasons["site003-bug01"])


if __name__ == "__main__":
    unittest.main()
