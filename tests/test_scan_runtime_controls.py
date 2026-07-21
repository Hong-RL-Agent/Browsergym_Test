from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.evaluate_multisite_browsergym_agent import _target_url_boundary_violation


class ScanRuntimeControlsTests(unittest.TestCase):
    def test_target_url_boundary_allows_same_host_and_path_prefix(self) -> None:
        site = {
            "base_url": "https://www.scrapethissite.com/pages/simple/",
            "allowed_hosts": ["www.scrapethissite.com"],
            "allowed_path_prefixes": ["/pages/simple/"],
            "blocked_url_keywords": ["checkout", "gumroad.com"],
        }
        observation = {"page_state": {"url": "https://www.scrapethissite.com/pages/simple/?page=2"}}

        self.assertEqual("", _target_url_boundary_violation(observation, site))

    def test_target_url_boundary_blocks_external_checkout_host(self) -> None:
        site = {
            "base_url": "https://www.scrapethissite.com/pages/simple/",
            "allowed_hosts": ["www.scrapethissite.com"],
            "allowed_path_prefixes": ["/pages/simple/"],
            "blocked_url_keywords": ["checkout", "gumroad.com"],
        }
        observation = {"page_state": {"url": "https://gumroad.com/checkout"}}

        reason = _target_url_boundary_violation(observation, site)

        self.assertIn("outside allowed hosts", reason)

    def test_target_url_boundary_blocks_outside_path_prefix(self) -> None:
        site = {
            "base_url": "https://www.scrapethissite.com/pages/simple/",
            "allowed_hosts": ["www.scrapethissite.com"],
            "allowed_path_prefixes": ["/pages/simple/"],
            "blocked_url_keywords": ["checkout", "gumroad.com"],
        }
        observation = {"page_state": {"url": "https://www.scrapethissite.com/lessons/sign-up/"}}

        reason = _target_url_boundary_violation(observation, site)

        self.assertTrue("blocked URL keyword" in reason or "outside allowed prefixes" in reason)


if __name__ == "__main__":
    unittest.main()
