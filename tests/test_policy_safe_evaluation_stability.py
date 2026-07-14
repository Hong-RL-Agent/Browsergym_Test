from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from runners.evaluate_multisite_browsergym_agent import (
    _apply_eval_fallback_mask,
    _apply_evaluation_defaults_to_sites,
    _episode_progress_made,
    _eval_episodes_arg,
    _failed_site_result,
    _int_arg,
    _record_unique_candidates,
    _record_eval_action_signature,
)
from services.policy_safe_metrics import build_per_site_metric
from services.policy_safe_metrics import split_site_metrics


class PolicySafeEvaluationStabilityTests(unittest.TestCase):
    def test_multisite_eval_continues_after_site_timeout(self) -> None:
        result = _failed_site_result(
            {"site_id": "site001", "base_url": "http://localhost:9220", "has_bug_catalog": True},
            TimeoutError("site timeout exceeded 60s for site001"),
            episodes=3,
            completed_episodes=1,
        )
        self.assertEqual("failed", result["status"])
        self.assertTrue(result["partial"])
        self.assertEqual("TimeoutError", result["error_type"])
        self.assertEqual(1, result["completed_episodes"])

    def test_eval_timeout_cli_overrides_config(self) -> None:
        config = {"site_timeout_seconds": 180, "episode_timeout_seconds": 120}
        self.assertEqual(30, _int_arg(30, config, "site_timeout_seconds", 60))
        self.assertEqual(180, _int_arg(None, config, "site_timeout_seconds", 60))
        self.assertEqual(120, _int_arg(None, config, "episode_timeout_seconds", 0))

    def test_reset_timeout_cli_overrides_config(self) -> None:
        config = {"reset_timeout_ms": 60000, "reset_retry_count": 1}
        self.assertEqual(45000, _int_arg(45000, config, "reset_timeout_ms", 30000))
        self.assertEqual(60000, _int_arg(None, config, "reset_timeout_ms", 30000))
        self.assertEqual(2, _int_arg(2, config, "reset_retry_count", 0))
        self.assertEqual(1, _int_arg(None, config, "reset_retry_count", 0))

    def test_eval_episodes_cli_overrides_config(self) -> None:
        config = {"eval_episodes": 1, "episodes": 9}
        self.assertEqual(2, _eval_episodes_arg(2, config, 3))
        self.assertEqual(1, _eval_episodes_arg(None, config, 3))
        self.assertEqual(4, _eval_episodes_arg(None, {"episodes": 4}, 3))

    def test_failed_site_metrics_are_written(self) -> None:
        site_result = _failed_site_result(
            {"site_id": "site003", "base_url": "http://localhost:9222", "has_bug_catalog": True},
            RuntimeError("boom"),
            episodes=3,
            completed_episodes=0,
        )
        metrics = build_per_site_metric("site003", site_result)
        self.assertEqual("failed", metrics["status"])
        self.assertEqual("RuntimeError", metrics["error_type"])
        self.assertEqual("boom", metrics["error_message"])
        self.assertEqual(0, metrics["completed_episodes"])
        self.assertTrue(metrics["partial"])

    def test_reset_retry_success_marks_site_partial_ok(self) -> None:
        metrics = build_per_site_metric(
            "site001",
            {
                "status": "partial_ok",
                "completed_episodes": 1,
                "partial": True,
                "episodes": 1,
                "reset_timeout_ms": 60000,
                "reset_retry_count": 1,
                "reset_retry_used": 1,
                "reset_failure_count": 1,
                "reset_failure_reason": "env reset exceeded 30000ms for site001",
            },
        )
        self.assertEqual("partial_ok", metrics["status"])
        self.assertEqual(60000, metrics["reset_timeout_ms"])
        self.assertEqual(1, metrics["reset_retry_count"])
        self.assertEqual(1, metrics["reset_retry_used"])
        self.assertEqual(1, metrics["reset_failure_count"])

    def test_failed_site_known_bugs_excluded_from_valid_metrics(self) -> None:
        split = split_site_metrics(
            [
                build_per_site_metric(
                    "site001",
                    {
                        "status": "failed",
                        "known_bug_total": 3,
                        "known_bug_match_count": 0,
                        "unique_detected_candidates": 0,
                        "missed_bug_ids": ["site001-bug01", "site001-bug02", "site001-bug03"],
                    },
                ),
                build_per_site_metric(
                    "site003",
                    {
                        "status": "partial_ok",
                        "known_bug_total": 3,
                        "known_bug_match_count": 2,
                        "unique_detected_candidates": 2,
                        "matched_bug_ids": ["site003-bug02", "site003-bug03"],
                        "missed_bug_ids": ["site003-bug01"],
                    },
                ),
            ]
        )
        self.assertEqual(3, split["valid_site_metrics"]["known_bug_total"])
        self.assertEqual(1, split["valid_site_metrics"]["false_negative_count"])
        self.assertEqual(["site003"], split["valid_site_metrics"]["valid_site_ids"])
        self.assertEqual(3, split["failed_site_metrics"]["excluded_known_bug_count"])

    def test_all_site_metrics_keep_failed_site_as_conservative_fn(self) -> None:
        split = split_site_metrics(
            [
                build_per_site_metric("site001", {"status": "failed", "known_bug_total": 3, "known_bug_match_count": 0}),
                build_per_site_metric("site003", {"status": "ok", "known_bug_total": 3, "known_bug_match_count": 2, "unique_detected_candidates": 2}),
            ]
        )
        self.assertEqual(6, split["all_site_metrics"]["known_bug_total"])
        self.assertEqual(4, split["all_site_metrics"]["false_negative_count"])
        self.assertEqual(3, split["valid_site_metrics"]["known_bug_total"])
        self.assertEqual(1, split["valid_site_metrics"]["false_negative_count"])

    def test_valid_site_metrics_written(self) -> None:
        split = split_site_metrics([build_per_site_metric("site003", {"status": "ok", "known_bug_total": 3, "known_bug_match_count": 2, "unique_detected_candidates": 2})])
        self.assertIn("all_site_metrics", split)
        self.assertIn("valid_site_metrics", split)
        self.assertIn("failed_site_metrics", split)
        self.assertEqual(1, split["valid_site_metrics"]["valid_site_count"])

    def test_repeated_action_fallback_does_not_use_bug_labels(self) -> None:
        source = _function_source(
            ROOT / "runners" / "evaluate_multisite_browsergym_agent.py",
            "def _suppress_repeated_action_signatures(",
            "def _suppress_repeated_clicks(",
        ).lower()
        for keyword in (
            "data-bug-id",
            "bug_id",
            "known_bug",
            "catalog_bug_id",
            "catalog_bug_id_matches",
            "has_data_bug_id",
            "ground_truth",
            "oracle",
            "label",
        ):
            self.assertNotIn(keyword, source)

        action_space = ActionSpace(max_candidates=4)
        mask = np.ones(action_space.get_action_dim(), dtype=np.float32)
        history = {"action_signature_counts": {}}
        action = {"action_type": "open_detail_panel", "candidate_index": 2, "clicked_bid": "target-2"}
        for _ in range(3):
            _record_eval_action_signature(history, action)
        filtered, warning, repeated = _apply_eval_fallback_mask(
            action_space,
            mask,
            {"candidate_elements": [{"bid": "target-2"}], "page_state": {"site_id": "site001"}},
            history,
        )
        self.assertEqual(0.0, float(filtered[action_space.encode("open_detail_panel", 2)]))
        self.assertEqual(1, repeated)
        self.assertIn("repeated action signature", warning)

    def test_no_progress_early_stop_marks_episode_completed(self) -> None:
        self.assertFalse(
            _episode_progress_made(
                before_unique_count=1,
                after_unique_count=1,
                before_matched_count=0,
                after_matched_count=0,
                before_signature=("url", "title", 10, 5, 3),
                after_signature=("url", "title", 10, 5, 3),
            )
        )
        metrics = build_per_site_metric(
            "site001",
            {
                "status": "partial_ok",
                "completed_episodes": 1,
                "partial": True,
                "episodes": 1,
                "early_stop_count": 1,
                "early_stop_reasons": {"no_progress": 1},
            },
        )
        self.assertEqual("partial_ok", metrics["status"])
        self.assertEqual(1, metrics["completed_episodes"])
        self.assertEqual({"no_progress": 1}, metrics["early_stop_reasons"])

    def test_repeated_anomaly_dedup_for_metrics(self) -> None:
        unique = {}
        matched = {}
        suppressed = {}
        anomaly = {
            "type": "button-no-response",
            "confidence": 0.9,
            "evidence": {"clicked_text": "Open details"},
        }
        _record_unique_candidates(unique, matched, suppressed, [dict(anomaly), dict(anomaly)])
        self.assertEqual(1, len(unique))
        counts = build_per_site_metric(
            "site001",
            {
                "known_bug_total": 2,
                "known_bug_match_count": 0,
                "unique_detected_candidates": len(unique),
                "total_detected_candidates": 2,
            },
        )
        self.assertEqual(2, counts["detected_anomaly_count"])
        self.assertEqual(1, counts["unique_detected_anomaly_count"])
        self.assertEqual(1, counts["false_positive_count"])

    def test_early_stop_does_not_use_bug_labels(self) -> None:
        source = _function_source(
            ROOT / "runners" / "evaluate_multisite_browsergym_agent.py",
            "def _current_action_signature_count(",
            "def _parse_action_signature(",
        ).lower()
        for keyword in (
            "data-bug-id",
            "bug_id",
            "known_bug",
            "catalog_bug_id",
            "catalog_bug_id_matches",
            "has_data_bug_id",
            "ground_truth",
            "oracle",
            "label",
        ):
            self.assertNotIn(keyword, source)

    def test_timing_fields_are_written(self) -> None:
        metrics = build_per_site_metric(
            "site003",
            {
                "status": "ok",
                "completed_episodes": 1,
                "episodes": 1,
                "avg_step_elapsed_ms": 25,
                "max_step_elapsed_ms": 50,
                "reset_elapsed_ms": 100,
                "site_elapsed_ms": 250,
                "episode_elapsed_ms": 200,
                "early_stop_count": 1,
                "early_stop_reasons": {"max_steps": 1},
                "timeout_count": 0,
            },
        )
        self.assertEqual(25, metrics["avg_step_elapsed_ms"])
        self.assertEqual(50, metrics["max_step_elapsed_ms"])
        self.assertEqual(100, metrics["reset_elapsed_ms"])
        self.assertEqual(250, metrics["site_elapsed_ms"])
        self.assertEqual(200, metrics["episode_elapsed_ms"])
        self.assertEqual(1, metrics["early_stop_count"])
        self.assertEqual({"max_steps": 1}, metrics["early_stop_reasons"])
        self.assertEqual(0, metrics["timeout_count"])

    def test_site_evaluation_mode_is_signal_discovery_when_known_bug_eval_disabled(self) -> None:
        sites = [{"site_id": "site003", "base_url": "http://localhost:9222", "has_bug_catalog": True}]
        _apply_evaluation_defaults_to_sites(sites, {"use_known_bug_for_evaluation": False})
        result = _failed_site_result(sites[0], RuntimeError("boom"), episodes=1, completed_episodes=0)
        self.assertEqual("signal_discovery", result["evaluation_mode"])

    def test_functional_fallback_prefers_clickable_candidates(self) -> None:
        action_space = ActionSpace(max_candidates=4)
        observation = {
            "page_state": {"site_id": "site003"},
            "candidate_elements": [
                {"bid": "btn-1", "visible": True, "enabled": True, "clickable": True, "visibility": 1.0},
                {"bid": "btn-2", "visible": True, "enabled": True, "clickable": True, "visibility": 1.0},
            ],
        }
        mask = action_space.build_action_mask(observation)
        filtered, warning, repeated = _apply_eval_fallback_mask(
            action_space,
            mask,
            observation,
            {
                "functional_action_count": 0,
                "action_type_counts": {"click_retry_button": 1, "scroll_up": 1},
            },
        )
        self.assertEqual(1, repeated)
        self.assertIn("functional fallback", warning)
        self.assertGreater(float(filtered[action_space.encode("click_element", 0)]), 0.0)
        self.assertGreater(float(filtered[action_space.encode("click_element", 1)]), 0.0)
        self.assertEqual(0.0, float(filtered[action_space.encode("scroll_up", 0)]))
        self.assertEqual(0.0, float(filtered[action_space.encode("open_detail_panel", 0)]))


def _function_source(path: Path, start_marker: str, end_marker: str) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
