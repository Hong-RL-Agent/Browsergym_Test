from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.evaluate_multisite_browsergym_agent import _site_config_summary, _validate_sites
from services.multisite_training_service import _signal_type_coverage


CONFIG_PATH = ROOT / "configs" / "training_sites_v3_multisignal.json"


class V3MultiSignalTrainingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_multisignal_config_exists(self) -> None:
        self.assertTrue(CONFIG_PATH.exists())
        self.assertEqual("v3_policy_safe_multisignal", self.config["run_id"])

    def test_multisignal_config_has_non_ui_site_groups(self) -> None:
        groups = {site.get("site_group") for site in self.config["sites"]}
        self.assertIn("console_runtime", groups)
        self.assertIn("network_api", groups)
        self.assertIn("cross_layer", groups)
        self.assertIn("db", groups)
        self.assertIn("security", groups)
        self.assertIn("server_infra", groups)

    def test_disabled_sites_are_skipped(self) -> None:
        enabled_sites = _validate_sites(self.config["sites"])
        enabled_site_ids = [site["site_id"] for site in enabled_sites]
        self.assertEqual(["site001", "site003"], enabled_site_ids)
        self.assertNotIn("site_console_001", enabled_site_ids)

    def test_enabled_site_ids_recorded(self) -> None:
        enabled_sites = _validate_sites(self.config["sites"])
        summary = _site_config_summary(self.config["sites"], enabled_sites)
        self.assertEqual(["site001", "site003"], summary["enabled_site_ids"])
        self.assertIn("site_console_001", summary["disabled_site_ids"])
        self.assertEqual(summary["disabled_site_ids"], summary["skipped_site_ids"])

    def test_target_signal_types_recorded(self) -> None:
        summary = _site_config_summary(self.config["sites"], _validate_sites(self.config["sites"]))
        self.assertIn("site_network_001", summary["target_signal_types_by_site"])
        self.assertIn("api-5xx", summary["target_signal_types_by_site"]["site_network_001"])
        self.assertIn("token-exposure", self.config["target_signal_types"])

    def test_signal_type_coverage_computed(self) -> None:
        coverage = _signal_type_coverage(
            ["ui-dom", "layout", "console-error", "api-5xx"],
            ["ui-dom", "layout"],
        )
        self.assertEqual(0.5, coverage)

    def test_multisignal_config_uses_signal_based_reward(self) -> None:
        self.assertEqual("signal_based", self.config["reward_mode"])
        self.assertFalse(self.config["policy_uses_bug_labels"])

    def test_multisignal_config_does_not_use_known_bug_reward(self) -> None:
        self.assertFalse(self.config["use_known_bug_reward"])
        self.assertFalse(self.config["use_known_bug_for_training"])
        self.assertFalse(self.config["use_known_bug_for_evaluation"])
        self.assertFalse(self.config["training_uses_site_specific_bug_catalog"])


if __name__ == "__main__":
    unittest.main()
