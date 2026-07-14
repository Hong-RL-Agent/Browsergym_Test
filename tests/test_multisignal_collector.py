from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.multisignal_collector import collect_multi_signal_counts
from envs.browsergym_jaws_env import BrowserGymJAWSEnv


class MultiSignalCollectorTests(unittest.TestCase):
    def test_console_error_count_collected(self) -> None:
        counts = collect_multi_signal_counts(after_observation={"runtime_signals": {"console_errors": ["boom"]}})
        self.assertEqual(1, counts["console_error_count"])

    def test_page_error_count_collected(self) -> None:
        counts = collect_multi_signal_counts(after_observation={"runtime_signals": {"page_error_count": 2}})
        self.assertEqual(2, counts["page_error_count"])

    def test_network_4xx_5xx_counts_collected(self) -> None:
        counts = collect_multi_signal_counts(
            after_observation={"runtime_signals": {"network_entries": [{"status": 404}, {"status": 500}]}}
        )
        self.assertEqual(1, counts["api_4xx_count"])
        self.assertEqual(1, counts["api_5xx_count"])

    def test_security_signal_fields_exist(self) -> None:
        counts = collect_multi_signal_counts(after_observation={"runtime_signals": {"page_text": "Bearer token leaked"}})
        self.assertIn("security_signal_count", counts)
        self.assertIn("token_exposure_signal_count", counts)
        self.assertGreaterEqual(counts["security_signal_count"], 1)

    def test_console_error_delta_reward_fields(self) -> None:
        counts = collect_multi_signal_counts(
            before_observation={"runtime_signals": {"console_error_count": 1}},
            after_observation={"runtime_signals": {"console_error_count": 3}},
        )
        self.assertEqual(1, counts["before_console_error_count"])
        self.assertEqual(3, counts["after_console_error_count"])
        self.assertEqual(2, counts["delta_console_error_count"])

    def test_page_error_delta_reward_fields(self) -> None:
        counts = collect_multi_signal_counts(
            before_observation={"runtime_signals": {"page_error_count": 0}},
            after_observation={"runtime_signals": {"page_error_count": 1}},
        )
        self.assertEqual(1, counts["delta_page_error_count"])

    def test_network_5xx_delta_reward_fields(self) -> None:
        counts = collect_multi_signal_counts(
            before_observation={"runtime_signals": {"network_entries": [{"status": 200}]}},
            after_observation={"runtime_signals": {"network_entries": [{"status": 200}, {"status": 500}]}},
        )
        self.assertEqual(1, counts["delta_api_5xx_count"])

    def test_requestfailed_delta_reward_fields(self) -> None:
        counts = collect_multi_signal_counts(
            after_observation={"runtime_signals": {"network_request_failed_count": 1}},
        )
        self.assertEqual(1, counts["delta_network_request_failed_count"])

    def test_security_signal_delta_fields_exist(self) -> None:
        counts = collect_multi_signal_counts(
            after_observation={"runtime_signals": {"page_text": "Bearer token leaked"}},
        )
        self.assertIn("delta_token_exposure_signal_count", counts)
        self.assertEqual(1, counts["delta_token_exposure_signal_count"])

    def test_playwright_console_listener_enabled_when_page_available(self) -> None:
        env = _fake_env_with_page()
        env._install_playwright_listeners()
        self.assertTrue(env._playwright_listener_flags["playwright_console_listener_enabled"])

    def test_playwright_pageerror_listener_enabled_when_page_available(self) -> None:
        env = _fake_env_with_page()
        env._install_playwright_listeners()
        self.assertTrue(env._playwright_listener_flags["playwright_pageerror_listener_enabled"])

    def test_playwright_network_listeners_enabled_when_page_available(self) -> None:
        env = _fake_env_with_page()
        env._install_playwright_listeners()
        self.assertTrue(env._playwright_listener_flags["playwright_network_listener_enabled"])
        self.assertTrue(env._playwright_listener_flags["playwright_request_listener_enabled"])
        self.assertTrue(env._playwright_listener_flags["playwright_response_listener_enabled"])
        self.assertTrue(env._playwright_listener_flags["playwright_requestfailed_listener_enabled"])

    def test_signal_delta_counts_from_listener_buffer(self) -> None:
        env = _fake_env_with_page()
        env._install_playwright_listeners()
        env._playwright_events.extend(
            [
                {"event": "console", "type": "error", "message": "boom"},
                {"event": "pageerror", "message": "runtime boom"},
                {"event": "response", "status": 500, "url": "http://local/api", "resource_type": "fetch"},
                {"event": "requestfailed", "message": "timeout", "url": "http://local/api", "resource_type": "fetch"},
            ]
        )
        counts = env._playwright_signal_counts(0)
        self.assertEqual(1, counts["delta_console_error_count"])
        self.assertEqual(1, counts["delta_runtime_exception_count"])
        self.assertEqual(1, counts["delta_api_5xx_count"])
        self.assertEqual(1, counts["delta_network_request_failed_count"])


class _FakePage:
    def __init__(self) -> None:
        self.handlers = {}

    def on(self, event_name, handler):
        self.handlers[event_name] = handler

    def locator(self, *args, **kwargs):
        return self

    def evaluate(self, *args, **kwargs):
        return None


class _FakeBrowserEnv:
    def __init__(self, page) -> None:
        self.page = page


def _fake_env_with_page() -> BrowserGymJAWSEnv:
    env = BrowserGymJAWSEnv.__new__(BrowserGymJAWSEnv)
    env.env = _FakeBrowserEnv(_FakePage())
    env.step_index = 0
    env.previous_action_type = "noop"
    env._playwright_events = []
    env._playwright_listener_page = None
    env._playwright_listener_warning = ""
    env._playwright_listener_flags = {
        "playwright_console_listener_enabled": False,
        "playwright_pageerror_listener_enabled": False,
        "playwright_network_listener_enabled": False,
        "playwright_request_listener_enabled": False,
        "playwright_response_listener_enabled": False,
        "playwright_requestfailed_listener_enabled": False,
        "signal_collector_enabled": True,
    }
    return env


if __name__ == "__main__":
    unittest.main()
