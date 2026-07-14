from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.autonomous_reward_service import fallback_curriculum_penalty


class FallbackCurriculumTests(unittest.TestCase):
    def test_fallback_curriculum_increases_penalty(self) -> None:
        early = fallback_curriculum_penalty(0.2, curriculum_enabled=True, episode_index=0, warmup_episodes=4)
        later = fallback_curriculum_penalty(0.2, curriculum_enabled=True, episode_index=4, warmup_episodes=4)
        self.assertGreater(later, early)


if __name__ == "__main__":
    unittest.main()
