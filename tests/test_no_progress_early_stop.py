from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.policy_safe_metrics import build_per_site_metric
from runners.evaluate_multisite_browsergym_agent import (
    _effective_eval_max_steps,
    _effective_no_progress_patience,
    _progress_policy_allows_finish,
)


class NoProgressEarlyStopTests(unittest.TestCase):
    def test_no_progress_delayed_when_functional_candidates_remain(self) -> None:
        metric = build_per_site_metric(
            "site005",
            {
                "completed_episodes": 1,
                "episodes": 1,
                "functional_priority_candidate_count": 5,
                "unclicked_functional_priority_candidate_count": 2,
                "no_progress_delayed_by_unclicked_functional_candidate_count": 1,
            },
        )
        self.assertEqual(1, metric["no_progress_delayed_by_unclicked_functional_candidate_count"])
        self.assertEqual(2, metric["unclicked_functional_priority_candidate_count"])

    def test_no_progress_delay_logic_does_not_use_bug_labels(self) -> None:
        source = _function_source(
            ROOT / "runners" / "evaluate_multisite_browsergym_agent.py",
            "effective_no_progress_patience = _effective_no_progress_patience(no_progress_patience, history)",
            "if str(action.get(\"action_type\") or \"\") == \"finish_episode\":",
        ).lower()
        for keyword in ("data-bug-id", "data_bug_id", "bug_id", "catalog_bug_id", "known_bug", "ground_truth", "oracle", "label"):
            self.assertNotIn(keyword, source)

    def test_dynamic_no_progress_patience_enabled_when_unconfigured(self) -> None:
        history = {
            "opportunity_summary": {
                "generated_opportunity_count": 22,
                "detected_candidate_count": 32,
            }
        }

        patience = _effective_no_progress_patience(0, history)

        self.assertGreaterEqual(patience, 6)
        self.assertLessEqual(patience, 14)
        self.assertTrue(history["dynamic_no_progress_patience_enabled"])

    def test_eval_max_steps_defaults_to_finite_limit(self) -> None:
        self.assertEqual(60, _effective_eval_max_steps(0))
        self.assertEqual(60, _effective_eval_max_steps(None))  # type: ignore[arg-type]
        self.assertEqual(7, _effective_eval_max_steps(7))

    def test_progress_policy_blocks_finish_until_optional_coverage_or_stall(self) -> None:
        history = {
            "no_progress_steps": 0,
            "opportunity_summary": {
                "remaining_required_opportunity_count": 0,
                "unverified_anomaly_count": 0,
                "generated_opportunity_count": 10,
                "remaining_opportunity_count": 10,
                "action_opportunity_coverage_rate": 0.0,
                "coverage_threshold": 0.7,
            },
        }

        self.assertFalse(_progress_policy_allows_finish({"runtime_signals": {}}, history))
        history["no_progress_steps"] = _effective_no_progress_patience(0, history)
        self.assertTrue(_progress_policy_allows_finish({"runtime_signals": {}}, history))

    def test_progress_policy_allows_finish_after_optional_coverage_threshold(self) -> None:
        history = {
            "no_progress_steps": 0,
            "opportunity_summary": {
                "remaining_required_opportunity_count": 0,
                "unverified_anomaly_count": 0,
                "generated_opportunity_count": 10,
                "remaining_opportunity_count": 3,
                "action_opportunity_coverage_rate": 0.7,
                "coverage_threshold": 0.7,
            },
        }

        self.assertTrue(_progress_policy_allows_finish({"runtime_signals": {}}, history))


def _function_source(path: Path, start_marker: str, end_marker: str) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
