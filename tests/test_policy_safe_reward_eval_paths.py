from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PolicySafeRewardEvalPathTests(unittest.TestCase):
    def test_known_bug_matching_is_post_action_only(self) -> None:
        multisite_source = (ROOT / "services" / "multisite_training_service.py").read_text(encoding="utf-8")
        collect_episode = _function_source(multisite_source, "    def _collect_episode(", "    def _record_detected_bug(")
        self.assertLess(collect_episode.index("env.step(action_id)"), collect_episode.index("detect_anomalies("))
        self.assertNotIn("match_anomalies_to_known_bugs(", collect_episode)

        eval_source = (ROOT / "runners" / "evaluate_multisite_browsergym_agent.py").read_text(encoding="utf-8")
        eval_site = _function_source(eval_source, "def _evaluate_site(", "def _apply_eval_fallback_mask(")
        self.assertLess(eval_site.index("env.step(action_id)"), eval_site.index("detect_anomalies("))
        self.assertLess(eval_site.index("detect_anomalies("), eval_site.index("calculate_autonomous_reward("))
        self.assertLess(eval_site.index("calculate_autonomous_reward("), eval_site.index("match_anomalies_to_known_bugs("))

    def test_ground_truth_used_only_for_evaluation_metrics(self) -> None:
        eval_source = (ROOT / "runners" / "evaluate_multisite_browsergym_agent.py").read_text(encoding="utf-8")
        eval_site = _function_source(eval_source, "def _evaluate_site(", "def _apply_eval_fallback_mask(")
        self.assertIn("match_anomalies_to_known_bugs(", eval_site)
        self.assertLess(eval_site.index("calculate_autonomous_reward("), eval_site.index("match_anomalies_to_known_bugs("))

    def test_reward_service_does_not_select_action_from_bug_label(self) -> None:
        source = (ROOT / "services" / "autonomous_reward_service.py").read_text(encoding="utf-8").lower()
        forbidden = (
            "select_action",
            "select_greedy_action",
            "action_space",
            "build_action_mask",
            "catalog_bug_id_matches",
            "data_bug_id",
            "has_data_bug_id",
            "known_bug_id",
        )
        hits = [item for item in forbidden if item in source]
        self.assertEqual([], hits)

    def test_v3_policy_safe_paths_do_not_overwrite_v2(self) -> None:
        sources = {
            "services/multisite_training_service.py": (ROOT / "services" / "multisite_training_service.py").read_text(encoding="utf-8"),
            "runners/train_multisite_browsergym_agent.py": (ROOT / "runners" / "train_multisite_browsergym_agent.py").read_text(encoding="utf-8"),
            "runners/evaluate_multisite_browsergym_agent.py": (ROOT / "runners" / "evaluate_multisite_browsergym_agent.py").read_text(encoding="utf-8"),
            "runners/evaluate_browsergym_agent.py": (ROOT / "runners" / "evaluate_browsergym_agent.py").read_text(encoding="utf-8"),
            "runners/smoke_test_policy_safe_training.py": (ROOT / "runners" / "smoke_test_policy_safe_training.py").read_text(encoding="utf-8"),
            "services/scan_backend_service.py": (ROOT / "services" / "scan_backend_service.py").read_text(encoding="utf-8"),
        }
        joined = "\n".join(sources.values())
        self.assertIn("artifacts/models/jaws_browsergym_shared_ppo_v3_policy_safe.pt", joined)
        self.assertIn("artifacts/evaluation/v3_policy_safe", joined)
        self.assertIn("artifacts/training_logs/v3_policy_safe", joined)
        for name, source in sources.items():
            self.assertNotIn("jaws_browsergym_shared_ppo_v2_browsergym_raw_obs.pt", source, name)


def _function_source(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
