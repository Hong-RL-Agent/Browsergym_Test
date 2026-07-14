from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.policy_safe_metrics import build_per_site_metric


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
            "if no_progress_patience > 0 and int(history.get(\"no_progress_steps\", 0) or 0) >= no_progress_patience:",
            "elif no_progress_patience > 0 and repeated_signature_count >= no_progress_patience + 1 and not progress:",
        ).lower()
        for keyword in ("data-bug-id", "data_bug_id", "bug_id", "catalog_bug_id", "known_bug", "ground_truth", "oracle", "label"):
            self.assertNotIn(keyword, source)


def _function_source(path: Path, start_marker: str, end_marker: str) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
