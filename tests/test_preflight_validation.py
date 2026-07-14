from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.compare_memory_vs_no_memory import build_comparison_summary, validate_comparison_inputs


class PreflightValidationTests(unittest.TestCase):
    def test_comparison_fails_when_site_connection_refused(self) -> None:
        no_memory = _summary({"site001": _valid_site(), "site003": _failed_site("ERR_CONNECTION_REFUSED")})
        memory = _summary({"site001": _valid_site(), "site003": _failed_site("ERR_CONNECTION_REFUSED")})
        summary = _comparison(no_memory, memory)
        self.assertFalse(summary["comparison_valid"])
        self.assertTrue(summary["comparison_failed_fast"])
        self.assertIn("site003", summary["connection_refused_site_ids"])

    def test_comparison_excludes_failed_sites_when_allow_partial_true(self) -> None:
        no_memory = _summary({"site001": _valid_site(), "site003": _failed_site("ERR_CONNECTION_REFUSED")})
        memory = _summary({"site001": _valid_site(avg_reward=2.0), "site003": _failed_site("ERR_CONNECTION_REFUSED")})
        summary = _comparison(no_memory, memory, allow_partial_sites=True)
        self.assertTrue(summary["comparison_valid"])
        self.assertEqual(["site001"], summary["compared_site_ids"])
        self.assertEqual(["site003"], summary["excluded_site_ids"])

    def test_comparison_does_not_average_zero_reward_failed_sites(self) -> None:
        no_memory = _summary({"site001": _valid_site(avg_reward=10.0), "site003": _failed_site("ERR_CONNECTION_REFUSED", avg_reward=0.0)})
        memory = _summary({"site001": _valid_site(avg_reward=12.0), "site003": _failed_site("ERR_CONNECTION_REFUSED", avg_reward=0.0)})
        summary = _comparison(no_memory, memory, allow_partial_sites=True)
        self.assertEqual(10.0, summary["no_memory_avg_reward"])
        self.assertEqual(12.0, summary["memory_avg_reward"])

    def test_zero_interaction_site_marked_invalid(self) -> None:
        validation = validate_comparison_inputs(
            _summary({"site001": _zero_interaction_site()}),
            _summary({"site001": _valid_site()}),
        )
        self.assertIn("site001", validation["zero_interaction_site_ids"])
        self.assertFalse(validation["comparison_valid"])

    def test_strict_preflight_requires_candidate_count(self) -> None:
        site = _valid_site()
        site["openended_interactive_candidate_count"] = 0
        site["functional_priority_candidate_count"] = 0
        validation = validate_comparison_inputs(_summary({"site001": site}), _summary({"site001": _valid_site()}))
        reasons = " ".join(validation["invalid_site_reasons"]["site001"])
        self.assertIn("candidate_count == 0", reasons)

    def test_valid_comparison_requires_policy_action_activity(self) -> None:
        site = _valid_site()
        site["policy_action_activity_count"] = 0
        validation = validate_comparison_inputs(_summary({"site001": site}), _summary({"site001": _valid_site()}))
        reasons = " ".join(validation["invalid_site_reasons"]["site001"])
        self.assertIn("policy_action_activity_count == 0", reasons)

    def test_invalid_site_reasons_recorded(self) -> None:
        validation = validate_comparison_inputs(
            _summary({"site001": _failed_site("ERR_CONNECTION_REFUSED")}),
            _summary({"site001": _valid_site()}),
        )
        self.assertIn("site001", validation["invalid_site_reasons"])
        self.assertGreater(len(validation["invalid_site_reasons"]["site001"]), 0)

    def test_comparison_valid_true_only_when_all_sites_valid(self) -> None:
        valid = _comparison(_summary({"site001": _valid_site()}), _summary({"site001": _valid_site()}))
        invalid = _comparison(_summary({"site001": _zero_interaction_site()}), _summary({"site001": _valid_site()}))
        self.assertTrue(valid["comparison_valid"])
        self.assertFalse(invalid["comparison_valid"])


def _comparison(no_memory: dict, memory: dict, *, allow_partial_sites: bool = False) -> dict:
    return build_comparison_summary(
        no_memory,
        memory,
        no_memory_model_path="no.pt",
        memory_model_path="mem.pt",
        seed=42,
        episodes=1,
        max_steps=3,
        strict_preflight=True,
        allow_partial_sites=allow_partial_sites,
    )


def _summary(sites: dict[str, dict]) -> dict:
    return {
        "enabled_site_ids": list(sites),
        "sites": sites,
    }


def _valid_site(
    *,
    avg_reward: float = 1.0,
    repeated_action_rate: float = 0.5,
    action_diversity_score: float = 0.25,
) -> dict:
    return {
        "status": "ok",
        "completed_episodes": 1,
        "observed_url_sample": "http://localhost:9220/",
        "openended_interactive_candidate_count": 5,
        "functional_priority_candidate_count": 2,
        "policy_action_activity_count": 2,
        "action_counts": {"click_element": 2},
        "avg_reward": avg_reward,
        "average_reward": avg_reward,
        "repeated_action_rate": repeated_action_rate,
        "action_diversity_score": action_diversity_score,
        "unique_clicked_target_count": 2,
        "functional_action_count": 2,
        "state_coverage_count": 1,
        "unique_detected_anomaly_count": 1,
    }


def _failed_site(error_message: str, *, avg_reward: float = 0.0) -> dict:
    return {
        "status": "failed",
        "error_message": error_message,
        "completed_episodes": 0,
        "observed_url_sample": "",
        "openended_interactive_candidate_count": 0,
        "policy_action_activity_count": 0,
        "action_counts": {},
        "avg_reward": avg_reward,
    }


def _zero_interaction_site() -> dict:
    site = _valid_site()
    site.update(
        {
            "observed_url_sample": "",
            "openended_interactive_candidate_count": 0,
            "functional_priority_candidate_count": 0,
            "policy_action_activity_count": 0,
            "action_counts": {},
        }
    )
    return site


if __name__ == "__main__":
    unittest.main()
