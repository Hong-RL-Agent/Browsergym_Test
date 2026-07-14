from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.autonomous_reward_service import apply_fallback_reward_policy, fallback_curriculum_penalty


class FallbackRewardPenaltyTests(unittest.TestCase):
    def test_fallback_applied_gets_penalty(self) -> None:
        reward, breakdown = apply_fallback_reward_policy(
            -0.1,
            {},
            fallback_applied=True,
            fallback_penalty=0.25,
            fallback_reward_cap_enabled=True,
            fallback_reward_cap_value=0.0,
        )
        self.assertLess(reward, -0.1)
        self.assertEqual(-0.25, breakdown["fallback_penalty_total"])

    def test_fallback_reward_is_capped(self) -> None:
        reward, breakdown = apply_fallback_reward_policy(
            1.0,
            {},
            fallback_applied=True,
            fallback_penalty=0.25,
            fallback_reward_cap_enabled=True,
            fallback_reward_cap_value=0.0,
        )
        self.assertEqual(-0.25, reward)
        self.assertEqual(1.0, breakdown["fallback_reward_capped_count"])

    def test_fallback_positive_reward_not_fully_given_to_policy(self) -> None:
        reward, _ = apply_fallback_reward_policy(
            0.8,
            {},
            fallback_applied=True,
            fallback_penalty=0.1,
            fallback_reward_cap_enabled=True,
            fallback_reward_cap_value=0.0,
        )
        self.assertLessEqual(reward, 0.0)

    def test_no_bug_label_used_in_fallback_penalty(self) -> None:
        source = (ROOT / "services" / "autonomous_reward_service.py").read_text(encoding="utf-8")
        start = source.index("def apply_fallback_reward_policy(")
        end = source.index("def fallback_curriculum_penalty(", start)
        snippet = source[start:end].lower()
        for forbidden in ("bug_id", "data-bug-id", "catalog", "site_group", "target_signal_types"):
            self.assertNotIn(forbidden, snippet)


class FallbackCurriculumTests(unittest.TestCase):
    def test_fallback_curriculum_increases_penalty(self) -> None:
        early = fallback_curriculum_penalty(0.25, curriculum_enabled=True, episode_index=0, warmup_episodes=4)
        later = fallback_curriculum_penalty(0.25, curriculum_enabled=True, episode_index=4, warmup_episodes=4)
        self.assertGreater(later, early)


if __name__ == "__main__":
    unittest.main()
