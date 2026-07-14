from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.multisite_training_service import (
    MultiSiteTrainingService,
    _is_blind_url_training_config,
    _load_known_bugs_for_site,
    _policy_safety_audit,
    _reject_known_bug_reward_for_training,
    _reward_mode_config,
    _validate_sites,
)


BLIND_CONFIG_PATH = ROOT / "configs" / "training_sites_v4_blind_urls.json"
EVAL_REFERENCE_PATH = ROOT / "configs" / "evaluation_reference_sites.json"
NO_MEMORY_CONFIG_PATH = ROOT / "configs" / "training_sites_v4_blind_no_memory.json"
GRU_MEMORY_CONFIG_PATH = ROOT / "configs" / "training_sites_v4_blind_gru_memory.json"


class BlindUrlTrainingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(BLIND_CONFIG_PATH.read_text(encoding="utf-8"))

    def test_blind_training_config_exists(self) -> None:
        self.assertTrue(BLIND_CONFIG_PATH.exists())
        self.assertTrue(NO_MEMORY_CONFIG_PATH.exists())
        self.assertTrue(GRU_MEMORY_CONFIG_PATH.exists())
        self.assertEqual("v4_blind_url_signal_training", self.config["run_id"])
        self.assertTrue(_is_blind_url_training_config(self.config))
        self.assertFalse(self.config["allow_known_bug_reward_for_debug_only"])
        self.assertFalse(self.config["use_ground_truth_for_evaluation_only"])

    def test_blind_training_config_has_no_site_group(self) -> None:
        self.assertNotIn("site_group", self.config)
        for site in self.config["sites"]:
            self.assertNotIn("site_group", site)

    def test_blind_training_config_has_no_target_signal_types(self) -> None:
        self.assertNotIn("target_signal_types", self.config)
        self.assertNotIn("expected_signal_types", self.config)
        for site in self.config["sites"]:
            self.assertNotIn("target_signal_types", site)
            self.assertNotIn("expected_signal_types", site)

    def test_blind_training_config_has_no_known_bug_catalog(self) -> None:
        for site in self.config["sites"]:
            self.assertNotIn("has_bug_catalog", site)
            self.assertNotIn("bug_catalog_path", site)
            self.assertNotIn("known_bug_catalog", site)

    def test_blind_training_config_site_entries_are_url_only(self) -> None:
        for site in self.config["sites"]:
            self.assertEqual({"site_id", "base_url", "enabled"}, set(site.keys()))

    def test_evaluation_reference_is_separate_and_not_training(self) -> None:
        reference = json.loads(EVAL_REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertFalse(reference["used_for_training"])
        self.assertEqual([], reference["sites"])

    def test_training_does_not_load_evaluation_reference(self) -> None:
        audit = _policy_safety_audit(self.config)
        self.assertFalse(audit["evaluation_reference_loaded_in_training"])

    def test_known_bug_catalog_not_loaded_in_blind_training(self) -> None:
        site = {"site_id": "site001", "base_url": "http://localhost:9220", "enabled": True}
        self.assertEqual([], _load_known_bugs_for_site(site))

    def test_blind_reward_mode_config_ignores_accidental_metadata(self) -> None:
        config = dict(self.config)
        site = {
            "site_id": "site001",
            "base_url": "http://localhost:9220",
            "enabled": True,
            "has_bug_catalog": True,
            "bug_catalog_path": "datasets/site001/bug_catalog.json",
            "target_signal_types": ["ui-dom"],
            "training_uses_site_specific_bug_catalog": True,
            "use_known_bug_reward": True,
            "use_known_bug_for_training": True,
        }
        profile = _reward_mode_config(config, site)
        self.assertFalse(profile["use_known_bug_reward"])
        self.assertFalse(profile["use_known_bug_for_training"])
        self.assertFalse(profile["allow_known_bug_reward_for_debug_only"])
        self.assertFalse(profile["training_uses_site_specific_bug_catalog"])
        self.assertEqual([], profile["target_signal_types"])

    def test_config_use_known_bug_reward_true_rejected_in_training(self) -> None:
        config = dict(self.config)
        config["use_known_bug_reward"] = True
        with self.assertRaises(ValueError):
            _reject_known_bug_reward_for_training(config)

    def test_known_bug_reward_total_always_zero_in_blind_training(self) -> None:
        config = dict(self.config)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "blind.json"
            config["sites"] = [config["sites"][0]]
            path.write_text(json.dumps(config), encoding="utf-8")
            service = MultiSiteTrainingService(
                config_path=path,
                total_updates=0,
                episodes_per_site=0,
                max_steps=1,
                enable_csv_logging=False,
                log_observation_detail=False,
                log_action_space=False,
            )
            summary = service.train()
        self.assertEqual(0.0, summary["known_bug_reward_total"])
        self.assertFalse(summary["use_known_bug_reward"])
        self.assertFalse(summary["allow_known_bug_reward_for_debug_only"])

    def test_policy_safety_audit_passes_for_blind_config(self) -> None:
        audit = _policy_safety_audit(self.config)
        self.assertTrue(audit["blind_url_training"])
        self.assertFalse(audit["metadata_used_for_training"])
        self.assertFalse(audit["site_group_used_in_training"])
        self.assertFalse(audit["target_signal_types_used_in_training"])
        self.assertFalse(audit["known_bug_catalog_used_in_training"])
        self.assertFalse(audit["evaluation_reference_loaded_in_training"])
        self.assertTrue(audit["reward_uses_only_observed_signals"])
        self.assertFalse(audit["blind_config_has_site_group"])
        self.assertFalse(audit["blind_config_has_target_signal_types"])
        self.assertFalse(audit["blind_config_has_known_bug_catalog"])

    def test_multisite_service_blind_summary_hides_metadata_maps(self) -> None:
        service = MultiSiteTrainingService(
            config_path=BLIND_CONFIG_PATH,
            total_updates=0,
            episodes_per_site=0,
            max_steps=1,
            enable_csv_logging=False,
            log_observation_detail=False,
            log_action_space=False,
        )
        summary = service._site_config_summary()
        self.assertEqual({}, summary["site_groups"])
        self.assertEqual({}, summary["target_signal_types_by_site"])
        self.assertEqual(["site001", "site002", "site003", "site004", "site005"], summary["enabled_site_ids"])

    def test_training_service_does_not_call_known_bug_loader_in_blind_mode(self) -> None:
        with patch(
            "services.multisite_training_service._load_known_bugs_for_site",
            side_effect=AssertionError("known bug catalog loader must not run in blind training"),
        ):
            service = MultiSiteTrainingService(
                config_path=BLIND_CONFIG_PATH,
                total_updates=0,
                episodes_per_site=0,
                max_steps=1,
                enable_csv_logging=False,
                log_observation_detail=False,
                log_action_space=False,
            )
        for state in service.site_states.values():
            self.assertEqual([], state["known_bugs"])

    def test_validate_sites_accepts_blind_sites(self) -> None:
        sites = _validate_sites(self.config["sites"])
        self.assertEqual(5, len(sites))
        self.assertEqual("site001", sites[0]["site_id"])


if __name__ == "__main__":
    unittest.main()
