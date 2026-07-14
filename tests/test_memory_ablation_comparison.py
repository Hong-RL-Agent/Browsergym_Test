from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MemoryAblationComparisonTests(unittest.TestCase):
    def test_comparison_script_outputs_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_summaries(tmpdir)
            self._run_compare(tmpdir, paths)
            summary = json.loads((Path(tmpdir) / "comparison_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(42, summary["seed"])
            self.assertEqual(["site001"], summary["compared_site_ids"])

    def test_comparison_script_outputs_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_summaries(tmpdir)
            self._run_compare(tmpdir, paths)
            table = (Path(tmpdir) / "comparison_table.csv").read_text(encoding="utf-8")
            self.assertIn("memory_repeated_action_rate", table)
            self.assertIn("action_diversity_delta", table)

    def test_comparison_uses_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_summaries(tmpdir)
            self._run_compare(tmpdir, paths, seed=123)
            summary = json.loads((Path(tmpdir) / "comparison_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(123, summary["seed"])

    def test_comparison_includes_repetition_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_summaries(tmpdir)
            self._run_compare(tmpdir, paths)
            summary = json.loads((Path(tmpdir) / "comparison_summary.json").read_text(encoding="utf-8"))
            self.assertIn("no_memory_repeated_action_rate", summary)
            self.assertIn("memory_repeated_action_rate", summary)
            self.assertTrue(summary["memory_better_on_repetition"])

    def test_comparison_includes_diversity_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_summaries(tmpdir)
            self._run_compare(tmpdir, paths)
            summary = json.loads((Path(tmpdir) / "comparison_summary.json").read_text(encoding="utf-8"))
            self.assertIn("action_diversity_delta", summary)
            self.assertTrue(summary["memory_better_on_diversity"])

    def test_comparison_does_not_use_known_bug_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._write_summaries(tmpdir)
            self._run_compare(tmpdir, paths)
            summary = json.loads((Path(tmpdir) / "comparison_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["policy_uses_bug_labels"])
            self.assertFalse(summary["known_bug_catalog_used"])

    def _write_summaries(self, tmpdir: str) -> dict[str, Path]:
        root = Path(tmpdir)
        no_memory = {
            "enabled_site_ids": ["site001"],
            "avg_reward": 1.0,
            "repeated_action_rate": 0.6,
            "action_diversity_score": 0.25,
            "unique_clicked_target_count": 1,
            "functional_action_count": 1,
            "state_coverage_count": 2,
            "unique_detected_anomaly_count": 1,
            "sites": {
                "site001": _site_result(
                    avg_reward=1.0,
                    repeated_action_rate=0.6,
                    action_diversity_score=0.25,
                    unique_clicked_target_count=1,
                    functional_action_count=1,
                    state_coverage_count=2,
                    unique_detected_anomaly_count=1,
                )
            },
        }
        memory = {
            "enabled_site_ids": ["site001"],
            "avg_reward": 1.2,
            "repeated_action_rate": 0.2,
            "action_diversity_score": 0.5,
            "unique_clicked_target_count": 3,
            "functional_action_count": 4,
            "state_coverage_count": 5,
            "unique_detected_anomaly_count": 2,
            "sites": {
                "site001": _site_result(
                    avg_reward=1.2,
                    repeated_action_rate=0.2,
                    action_diversity_score=0.5,
                    unique_clicked_target_count=3,
                    functional_action_count=4,
                    state_coverage_count=5,
                    unique_detected_anomaly_count=2,
                )
            },
        }
        no_path = root / "no_memory_summary.json"
        mem_path = root / "memory_summary.json"
        no_path.write_text(json.dumps(no_memory), encoding="utf-8")
        mem_path.write_text(json.dumps(memory), encoding="utf-8")
        return {"no": no_path, "memory": mem_path}

    def _run_compare(self, tmpdir: str, paths: dict[str, Path], seed: int = 42) -> None:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "runners" / "compare_memory_vs_no_memory.py"),
                "--no-memory-config",
                "configs/training_sites_v4_blind_no_memory.json",
                "--memory-config",
                "configs/training_sites_v4_blind_gru_memory.json",
                "--no-memory-model",
                "artifacts/models/jaws_ppo_v4_blind_no_memory.pt",
                "--memory-model",
                "artifacts/models/jaws_ppo_v4_blind_gru_memory.pt",
                "--episodes",
                "3",
                "--max-steps",
                "10",
                "--seed",
                str(seed),
                "--output-dir",
                tmpdir,
                "--no-memory-summary",
                str(paths["no"]),
                "--memory-summary",
                str(paths["memory"]),
                "--skip-evaluation",
            ],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.DEVNULL,
        )


def _site_result(
    *,
    avg_reward: float,
    repeated_action_rate: float,
    action_diversity_score: float,
    unique_clicked_target_count: int,
    functional_action_count: int,
    state_coverage_count: int,
    unique_detected_anomaly_count: int,
) -> dict:
    return {
        "status": "ok",
        "completed_episodes": 1,
        "observed_url_sample": "http://localhost:9220/",
        "openended_interactive_candidate_count": 5,
        "policy_action_activity_count": 2,
        "action_counts": {"click_element": 2},
        "avg_reward": avg_reward,
        "average_reward": avg_reward,
        "repeated_action_rate": repeated_action_rate,
        "action_diversity_score": action_diversity_score,
        "unique_clicked_target_count": unique_clicked_target_count,
        "functional_action_count": functional_action_count,
        "state_coverage_count": state_coverage_count,
        "unique_detected_anomaly_count": unique_detected_anomaly_count,
    }


if __name__ == "__main__":
    unittest.main()
