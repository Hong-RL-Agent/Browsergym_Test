from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.compare_memory_vs_no_memory import build_comparison_summary, write_comparison_table


class PaperMetricsSummaryTests(unittest.TestCase):
    def test_paper_metrics_present_in_comparison_summary(self) -> None:
        comparison = build_comparison_summary(
            _summary("no_memory"),
            _summary("memory"),
            no_memory_model_path="no.pt",
            memory_model_path="mem.pt",
            seed=42,
            episodes=3,
            max_steps=10,
            strict_preflight=True,
            allow_partial_sites=False,
        )

        for prefix in ("no_memory", "memory"):
            self.assertIn(f"{prefix}_paper_unique_clicked_element_key_count", comparison)
            self.assertIn(f"{prefix}_paper_repeated_action_signature_count", comparison)
            self.assertIn(f"{prefix}_paper_action_fallback_count", comparison)
            self.assertIn(f"{prefix}_paper_action_diversity_score", comparison)
            self.assertIn(f"{prefix}_paper_verification_after_high_value_count", comparison)
            self.assertIn(f"{prefix}_paper_unique_detected_candidates", comparison)
            self.assertIn(f"{prefix}_paper_known_bug_reward_total", comparison)
            self.assertIn(f"{prefix}_paper_catalog_guided_action_count", comparison)
        self.assertEqual(4, comparison["memory_paper_unique_clicked_element_key_count"])
        self.assertEqual(0.0, comparison["memory_paper_known_bug_reward_total"])
        self.assertEqual(0, comparison["memory_paper_catalog_guided_action_count"])

    def test_paper_metrics_present_in_comparison_table(self) -> None:
        comparison = build_comparison_summary(
            _summary("no_memory"),
            _summary("memory"),
            no_memory_model_path="no.pt",
            memory_model_path="mem.pt",
            seed=42,
            episodes=3,
            max_steps=10,
            strict_preflight=True,
            allow_partial_sites=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = Path(tmpdir) / "comparison_table.csv"
            write_comparison_table(table_path, comparison)
            with table_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        metrics = {row["metric"] for row in rows}
        self.assertIn("memory_paper_unique_clicked_element_key_count", metrics)
        self.assertIn("memory_paper_verification_after_high_value_count", metrics)
        self.assertIn("memory_paper_catalog_guided_action_count", metrics)


def _summary(run_label: str) -> dict:
    is_memory = run_label == "memory"
    return {
        "per_site_metrics": [
            {
                "site_id": "site005",
                "status": "ok",
                "valid_for_comparison": True,
                "completed_episodes": 3,
                "valid_completed_episodes": 3,
                "observed_url_sample": "http://localhost:9224",
                "openended_interactive_candidate_count": 12,
                "policy_action_activity_count": 6,
                "action_counts": {"click_element": 4, "inspect_dom": 1},
                "avg_reward": 1.0 if is_memory else 0.5,
                "repeated_action_rate": 0.1 if is_memory else 0.2,
                "action_diversity_score": 0.5 if is_memory else 0.25,
                "unique_clicked_target_count": 3 if is_memory else 2,
                "unique_clicked_element_key_count": 4 if is_memory else 2,
                "repeated_action_signature_count": 0,
                "repeated_action_type_count": 2,
                "repeated_element_key_click_count": 0,
                "action_fallback_count": 1 if is_memory else 3,
                "verification_action_after_high_value_click_count": 1 if is_memory else 0,
                "functional_action_count": 4,
                "state_coverage_count": 2,
                "unique_detected_anomaly_count": 3,
                "known_bug_reward_total": 0.0,
                "catalog_guided_action_count": 0,
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()
