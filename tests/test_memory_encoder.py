from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import PPOAgent
from models.memory_encoder import MemoryEncoder
from runners.evaluate_multisite_browsergym_agent import _evaluate_site
from services.multisite_training_service import MultiSiteTrainingService


class MemoryEncoderTests(unittest.TestCase):
    def test_memory_encoder_output_shape(self) -> None:
        encoder = MemoryEncoder(input_size=8, hidden_size=16)
        output = encoder(torch.ones(8), encoder.initial_state())
        self.assertEqual((1, 16), tuple(output.shape))

    def test_memory_hidden_state_resets_per_episode(self) -> None:
        encoder = MemoryEncoder(input_size=4, hidden_size=8)
        state_a = encoder.initial_state()
        next_state = encoder(torch.ones(4), state_a)
        state_b = encoder.initial_state()
        self.assertFalse(torch.equal(next_state, state_b))
        self.assertTrue(torch.equal(state_b, torch.zeros_like(state_b)))

    def test_memory_hidden_state_updates_per_step(self) -> None:
        encoder = MemoryEncoder(input_size=4, hidden_size=8)
        state = encoder.initial_state()
        next_state = encoder(torch.ones(4), state)
        newer_state = encoder(torch.ones(4) * 2, next_state)
        self.assertFalse(torch.equal(state, next_state))
        self.assertFalse(torch.equal(next_state, newer_state))

    def test_ppo_accepts_memory_state(self) -> None:
        agent = PPOAgent(8, 5, use_memory_encoder=True, memory_hidden_size=16)
        obs = np.ones(8, dtype=np.float32)
        mask = np.ones(5, dtype=np.float32)
        memory = agent.reset_memory_state()
        selected = agent.select_action(obs, mask, memory_state=memory)
        self.assertIn("memory_state", selected)
        self.assertEqual((16,), selected["memory_state"].shape)
        scored = agent.score_action(obs, mask, selected["action_id"], memory_state=selected["memory_state"])
        self.assertEqual(selected["action_id"], scored["action_id"])

    def test_memory_encoder_can_be_disabled(self) -> None:
        agent = PPOAgent(8, 5, use_memory_encoder=False)
        obs = np.ones(8, dtype=np.float32)
        mask = np.ones(5, dtype=np.float32)
        selected = agent.select_action(obs, mask)
        self.assertEqual((0,), selected["memory_state"].shape)
        self.assertEqual(0.0, selected["memory_state_norm"])

    def test_training_runs_with_memory_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_id": "v4_blind_url_signal_training_test",
                        "blind_url_training": True,
                        "reward_mode": "signal_based",
                        "use_memory_encoder": True,
                        "memory_encoder_type": "gru",
                        "memory_hidden_size": 16,
                        "shared_model_path": str(Path(tmpdir) / "model.pt"),
                        "output_dir": str(Path(tmpdir) / "out"),
                        "training_log_dir": str(Path(tmpdir) / "logs"),
                        "sites": [
                            {"site_id": "site001", "base_url": "http://localhost:9220", "enabled": True}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            service = MultiSiteTrainingService(
                config_path=config_path,
                total_updates=0,
                episodes_per_site=0,
                max_steps=1,
                enable_csv_logging=False,
                log_observation_detail=False,
                log_action_space=False,
            )
            summary = service._build_multisite_summary([], {})
            self.assertTrue(summary["memory_encoder_enabled"])
            self.assertEqual("gru", summary["memory_encoder_type"])
            self.assertEqual(16, summary["memory_hidden_size"])

    def test_evaluation_runs_with_memory_encoder(self) -> None:
        site = {"site_id": "site001", "base_url": "http://localhost:9220", "enabled": True}
        agent = PPOAgent(8, 5, use_memory_encoder=True, memory_hidden_size=16)
        with patch.object(agent, "reset_memory_state", wraps=agent.reset_memory_state) as reset_mock:
            state = agent.reset_memory_state()
            selected = agent.select_greedy_action(np.ones(8, dtype=np.float32), np.ones(5, dtype=np.float32), state)
        self.assertEqual((16,), selected["memory_state"].shape)
        self.assertGreaterEqual(reset_mock.call_count, 1)
        self.assertEqual("site001", site["site_id"])


if __name__ == "__main__":
    unittest.main()
