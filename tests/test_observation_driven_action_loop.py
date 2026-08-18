import unittest

from envs.browsergym_jaws_env import _action_count_limit_reached


class ObservationDrivenActionLoopTests(unittest.TestCase):
    def test_max_steps_zero_disables_action_count_limit(self) -> None:
        self.assertFalse(_action_count_limit_reached(1, 0))
        self.assertFalse(_action_count_limit_reached(100, 0))

    def test_positive_max_steps_keeps_legacy_limit(self) -> None:
        self.assertFalse(_action_count_limit_reached(1, 2))
        self.assertTrue(_action_count_limit_reached(2, 2))


if __name__ == "__main__":
    unittest.main()
