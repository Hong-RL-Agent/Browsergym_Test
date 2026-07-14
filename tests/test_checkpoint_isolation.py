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

from runners.evaluate_multisite_browsergym_agent import _bool_arg, _int_arg, _str_arg
from services.multisite_training_service import MultiSiteTrainingService


NO_MEMORY_CONFIG = ROOT / "configs" / "training_sites_v4_blind_no_memory.json"
GRU_MEMORY_CONFIG = ROOT / "configs" / "training_sites_v4_blind_gru_memory.json"


class CheckpointIsolationTests(unittest.TestCase):
    def test_memory_and_no_memory_use_different_model_paths(self) -> None:
        no_memory = json.loads(NO_MEMORY_CONFIG.read_text(encoding="utf-8"))
        gru_memory = json.loads(GRU_MEMORY_CONFIG.read_text(encoding="utf-8"))
        self.assertNotEqual(no_memory["run_id"], gru_memory["run_id"])
        self.assertNotEqual(no_memory["shared_model_path"], gru_memory["shared_model_path"])
        self.assertNotEqual(no_memory["output_dir"], gru_memory["output_dir"])
        self.assertNotEqual(no_memory["training_log_dir"], gru_memory["training_log_dir"])
        self.assertNotEqual(no_memory["evaluation_output_dir"], gru_memory["evaluation_output_dir"])

    def test_no_memory_config_disables_memory_encoder(self) -> None:
        config = json.loads(NO_MEMORY_CONFIG.read_text(encoding="utf-8"))
        self.assertFalse(config["use_memory_encoder"])
        self.assertIn(config["run_id"], config["shared_model_path"])
        self.assertIn(config["run_id"], config["output_dir"])

    def test_gru_memory_config_enables_memory_encoder(self) -> None:
        config = json.loads(GRU_MEMORY_CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(config["use_memory_encoder"])
        self.assertEqual("gru", config["memory_encoder_type"])
        self.assertEqual(128, config["memory_hidden_size"])
        self.assertIn(config["run_id"], config["shared_model_path"])
        self.assertIn(config["run_id"], config["output_dir"])

    def test_fresh_start_does_not_load_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(tmpdir, fresh_start=True, resume=False)
            with patch("agents.ppo_agent.PPOAgent.load", side_effect=AssertionError("load should not run")):
                service = MultiSiteTrainingService(config_path=config_path, total_updates=0, episodes_per_site=0)
            self.assertFalse(service.checkpoint_loaded)
            self.assertTrue(service.fresh_start)
            self.assertFalse(service.resume)

    def test_resume_loads_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(tmpdir, fresh_start=False, resume=True)
            loaded = []

            def _fake_load(_agent, path):
                loaded.append(str(path))

            with patch("agents.ppo_agent.PPOAgent.load", _fake_load):
                service = MultiSiteTrainingService(config_path=config_path, total_updates=0, episodes_per_site=0)
            self.assertTrue(service.checkpoint_loaded)
            self.assertTrue(service.resume)
            self.assertFalse(service.fresh_start)
            self.assertEqual([service.checkpoint_path], loaded)

    def test_summary_records_memory_encoder_enabled(self) -> None:
        service = MultiSiteTrainingService(config_path=GRU_MEMORY_CONFIG, total_updates=0, episodes_per_site=0)
        summary = service._build_multisite_summary([], {})
        self.assertTrue(summary["memory_encoder_enabled"])
        self.assertTrue(summary["use_memory_encoder"])
        self.assertEqual(128, summary["memory_hidden_size"])

    def test_summary_records_checkpoint_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(tmpdir, fresh_start=False, resume=True)
            with patch("agents.ppo_agent.PPOAgent.load", lambda _agent, _path: None):
                service = MultiSiteTrainingService(config_path=config_path, total_updates=0, episodes_per_site=0)
            summary = service._build_multisite_summary([], {})
            self.assertTrue(summary["checkpoint_loaded"])
            self.assertTrue(summary["resume"])
            self.assertFalse(summary["fresh_start"])
            self.assertTrue(summary["checkpoint_isolated_by_run_id"])

    def test_evaluation_accepts_explicit_model_path(self) -> None:
        config = {"use_memory_encoder": True, "memory_encoder_type": "gru", "memory_hidden_size": 64}
        self.assertTrue(_bool_arg(False, config, "use_memory_encoder", True) is False)
        self.assertTrue(_bool_arg(None, config, "use_memory_encoder", False))
        self.assertEqual("gru", _str_arg("", config, "memory_encoder_type", "gru"))
        self.assertEqual(64, _int_arg(None, config, "memory_hidden_size", 128))

    def _write_config(self, tmpdir: str, *, fresh_start: bool, resume: bool) -> Path:
        root = Path(tmpdir)
        model_path = root / "artifacts" / "models" / "v4_test_run.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"checkpoint placeholder")
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "run_id": "v4_test_run",
                    "blind_url_training": True,
                    "fresh_start": fresh_start,
                    "resume": resume,
                    "shared_model_path": str(model_path),
                    "output_dir": str(root / "artifacts" / "multisite" / "v4_test_run"),
                    "training_log_dir": str(root / "artifacts" / "training_logs" / "v4_test_run"),
                    "use_memory_encoder": False,
                    "sites": [{"site_id": "site001", "base_url": "http://localhost:9220", "enabled": True}],
                }
            ),
            encoding="utf-8",
        )
        return config_path


if __name__ == "__main__":
    unittest.main()
