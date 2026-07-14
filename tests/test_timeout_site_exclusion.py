from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.compare_memory_vs_no_memory import build_comparison_summary, validate_comparison_inputs
from services.policy_safe_metrics import build_per_site_metric


class TimeoutSiteExclusionTests(unittest.TestCase):
    def test_timeout_site_not_included_in_comparison_average(self) -> None:
        no_memory = _summary({"site001": _valid_site(1.0), "site005": _timeout_site()})
        memory = _summary({"site001": _valid_site(2.0), "site005": _timeout_site()})

        summary = build_comparison_summary(
            no_memory,
            memory,
            no_memory_model_path="no.pt",
            memory_model_path="mem.pt",
            seed=42,
            episodes=3,
            max_steps=10,
            allow_partial_sites=True,
        )

        self.assertEqual(["site001"], summary["compared_site_ids"])
        self.assertEqual(1.0, summary["no_memory_avg_reward"])
        self.assertEqual(2.0, summary["memory_avg_reward"])
        self.assertIn("site005", summary["invalid_site_ids"])
        self.assertGreaterEqual(summary["timeout_excluded_from_comparison_count"], 1)

    def test_completed_episodes_zero_sets_valid_completed_episodes_zero(self) -> None:
        metric = build_per_site_metric("site005", _timeout_site())
        self.assertEqual(0, metric["episode_count"])
        self.assertEqual(3, metric["requested_episodes"])
        self.assertEqual(0, metric["valid_completed_episodes"])
        self.assertFalse(metric["valid_for_comparison"])

    def test_partial_site_valid_for_comparison_false(self) -> None:
        site = _valid_site(1.0)
        site.update({"partial": True, "valid_for_comparison": False, "excluded_from_comparison_reason": "partial"})
        validation = validate_comparison_inputs(_summary({"site004": site}), _summary({"site004": site}), allow_partial_sites=True)
        self.assertIn("site004", validation["invalid_site_ids"])
        self.assertEqual([], validation["compared_site_ids"])
        self.assertEqual(1, validation["partial_site_excluded_count"])

    def test_comparison_excludes_timeout_site_by_default(self) -> None:
        validation = validate_comparison_inputs(_summary({"site005": _timeout_site()}), _summary({"site005": _timeout_site()}))
        self.assertFalse(validation["comparison_valid"])
        self.assertEqual([], validation["compared_site_ids"])
        self.assertIn("site005", validation["invalid_site_ids"])

    def test_max_steps_episode_is_valid_for_comparison(self) -> None:
        site = _valid_site(1.0)
        site.update(
            {
                "early_stop_count": 3,
                "early_stop_reasons": {"max_steps": 3},
                "partial": False,
                "valid_for_comparison": True,
                "valid_completed_episodes": 3,
                "excluded_from_comparison_reason": "",
            }
        )
        validation = validate_comparison_inputs(_summary({"site005": site}), _summary({"site005": site}))
        self.assertTrue(validation["comparison_valid"])
        self.assertEqual(["site005"], validation["compared_site_ids"])

    def test_timeout_episode_is_invalid_for_comparison(self) -> None:
        validation = validate_comparison_inputs(_summary({"site005": _timeout_site()}), _summary({"site005": _timeout_site()}))
        self.assertFalse(validation["comparison_valid"])
        self.assertIn("site005", validation["invalid_site_ids"])

    def test_completed_episodes_three_max_steps_valid_completed_episodes_three(self) -> None:
        site = _valid_site(1.0)
        site.update({"early_stop_reasons": {"max_steps": 3}, "early_stop_count": 3})
        metric = build_per_site_metric("site005", site)
        self.assertEqual(3, metric["completed_episodes"])
        self.assertEqual(3, metric["valid_completed_episodes"])
        self.assertTrue(metric["valid_for_comparison"])


def _summary(sites: dict[str, dict]) -> dict:
    return {"enabled_site_ids": list(sites), "sites": sites}


def _valid_site(avg_reward: float) -> dict:
    return {
        "status": "ok",
        "requested_episodes": 3,
        "completed_episodes": 3,
        "valid_completed_episodes": 3,
        "valid_for_comparison": True,
        "episode_count": 3,
        "partial": False,
        "observed_url_sample": "http://localhost:9220/",
        "openended_interactive_candidate_count": 5,
        "policy_action_activity_count": 2,
        "action_counts": {"click_element": 2},
        "avg_reward": avg_reward,
        "average_reward": avg_reward,
        "repeated_action_rate": 0.1,
        "action_diversity_score": 0.5,
    }


def _timeout_site() -> dict:
    return {
        "status": "failed",
        "requested_episodes": 3,
        "completed_episodes": 0,
        "valid_completed_episodes": 0,
        "valid_for_comparison": False,
        "excluded_from_comparison_reason": "timeout",
        "episode_count": 0,
        "partial": True,
        "timeout_count": 1,
        "error_message": "site timeout exceeded 240s",
        "observed_url_sample": "",
        "openended_interactive_candidate_count": 0,
        "policy_action_activity_count": 0,
        "action_counts": {},
        "avg_reward": 0.0,
        "average_reward": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
