from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.train_multisite_browsergym_agent import _bool_arg, _int_arg, _read_config_defaults
from runners.evaluate_multisite_browsergym_agent import _validate_sites as _validate_eval_sites


SMALL_CONFIG = ROOT / "configs" / "training_sites_v3_small.json"
MEDIUM_CONFIG = ROOT / "configs" / "training_sites_v3_medium.json"
MULTISIGNAL_CONFIG = ROOT / "configs" / "training_sites_v3_multisignal.json"


class V3SmallTrainingConfigTests(unittest.TestCase):
    def test_small_config_contains_only_validated_sites(self) -> None:
        config = json.loads(SMALL_CONFIG.read_text(encoding="utf-8"))
        site_ids = [site["site_id"] for site in config["sites"]]
        self.assertEqual(["site001", "site003"], site_ids)
        self.assertNotIn("site9800", site_ids)

    def test_small_config_uses_v3_policy_safe_paths(self) -> None:
        config = json.loads(SMALL_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual("v3_policy_safe", config["version"])
        self.assertEqual(
            "artifacts/models/jaws_browsergym_shared_ppo_v3_policy_safe.pt",
            config["shared_model_path"],
        )
        self.assertEqual("artifacts/multisite/v3_policy_safe", config["output_dir"])

    def test_train_runner_uses_config_defaults_for_small_run(self) -> None:
        config = _read_config_defaults(str(SMALL_CONFIG))
        self.assertEqual(2, _int_arg(None, config, "total_updates", 20))
        self.assertEqual(1, _int_arg(None, config, "episodes_per_site", 99))
        self.assertEqual(5, _int_arg(None, config, "max_steps", 25))
        self.assertFalse(_bool_arg(None, config, "enable_guided_actions", True))
        self.assertTrue(_bool_arg(None, config, "debug_action_mask", False))

    def test_cli_values_override_config_defaults(self) -> None:
        config = _read_config_defaults(str(SMALL_CONFIG))
        self.assertEqual(3, _int_arg(3, config, "total_updates", 20))
        self.assertFalse(_bool_arg(False, config, "debug_action_mask", False))

    def test_medium_config_uses_separate_v3_policy_safe_paths(self) -> None:
        config = json.loads(MEDIUM_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual("v3_policy_safe_medium", config["run_id"])
        self.assertEqual("v3_policy_safe_medium", config["config_name"])
        self.assertEqual("artifacts/models/jaws_browsergym_shared_ppo_v3_policy_safe_medium.pt", config["shared_model_path"])
        self.assertEqual("artifacts/multisite/v3_policy_safe_medium", config["output_dir"])
        self.assertEqual("artifacts/training_logs/v3_policy_safe_medium", config["training_log_dir"])
        self.assertEqual("artifacts/evaluation/v3_policy_safe_medium", config["evaluation_output_dir"])
        self.assertEqual(10, _int_arg(None, config, "total_updates", 20))
        self.assertEqual(2, _int_arg(None, config, "episodes_per_site", 1))
        self.assertEqual(15, _int_arg(None, config, "max_steps", 25))
        self.assertEqual(60000, _int_arg(None, config, "reset_timeout_ms", 30000))
        self.assertEqual(1, _int_arg(None, config, "reset_retry_count", 0))
        self.assertFalse(_bool_arg(None, config, "enable_guided_actions", True))
        self.assertFalse(_bool_arg(None, config, "debug_action_mask", True))

    def test_disabled_multisignal_sites_are_skipped(self) -> None:
        config = json.loads(MULTISIGNAL_CONFIG.read_text(encoding="utf-8"))
        sites = _validate_eval_sites(config["sites"])
        site_ids = [site["site_id"] for site in sites]
        self.assertEqual(["site001", "site003"], site_ids)
        disabled_ids = [site["site_id"] for site in config["sites"] if site.get("enabled") is False]
        self.assertIn("site_console_001", disabled_ids)


if __name__ == "__main__":
    unittest.main()
