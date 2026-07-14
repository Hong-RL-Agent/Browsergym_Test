from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from runners import compare_memory_vs_no_memory
from runners.evaluate_multisite_browsergym_agent import _apply_eval_fallback_mask
from services.autonomous_reward_service import apply_fallback_reward_policy


class StrictEvaluationNoFallbackTests(unittest.TestCase):
    def test_strict_eval_disables_fallback(self) -> None:
        action_space = ActionSpace(max_candidates=1)
        observation = {
            "page_state": {"site_id": "site005"},
            "candidate_elements": [
                {
                    "bid": "a",
                    "role": "button",
                    "tag": "button",
                    "visible": True,
                    "enabled": True,
                    "clickable": True,
                }
            ],
        }
        mask = action_space.build_action_mask(observation)
        strict_mask = mask.copy()

        self.assertEqual(mask.tolist(), strict_mask.tolist())
        fallback_mask, warning, repeated = _apply_eval_fallback_mask(
            action_space,
            mask,
            observation,
            {"action_type_counts": {"inspect_console": 1}},
        )
        self.assertNotEqual(fallback_mask.tolist(), strict_mask.tolist())
        self.assertIn("repeated meta action", warning)
        self.assertEqual(1, repeated)

    def test_strict_eval_invalid_action_gets_penalty(self) -> None:
        reward, breakdown = apply_fallback_reward_policy(
            0.5,
            {"penalty_invalid_action": -1.0},
            fallback_applied=True,
            fallback_penalty=0.25,
            fallback_reward_cap_enabled=True,
            fallback_reward_cap_value=0.0,
        )
        self.assertLess(reward, 0.0)
        self.assertLess(breakdown["fallback_penalty_total"], 0.0)

    def test_compare_accepts_fallback_mode_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            no_summary = Path(tmpdir) / "no.json"
            mem_summary = Path(tmpdir) / "mem.json"
            no_summary.write_text(_summary_json("no_memory"), encoding="utf-8")
            mem_summary.write_text(_summary_json("memory"), encoding="utf-8")
            argv = [
                "compare_memory_vs_no_memory.py",
                "--no-memory-config",
                "configs/training_sites_v4_blind_no_memory.json",
                "--memory-config",
                "configs/training_sites_v4_blind_gru_memory.json",
                "--no-memory-model",
                "no.pt",
                "--memory-model",
                "mem.pt",
                "--no-memory-summary",
                str(no_summary),
                "--memory-summary",
                str(mem_summary),
                "--skip-evaluation",
                "--output-dir",
                tmpdir,
                "--fallback-mode",
                "strict",
            ]
            with patch.object(sys, "argv", argv):
                code = compare_memory_vs_no_memory.main()
            output = Path(tmpdir) / "comparison_summary.json"
            self.assertEqual(0, code)
            self.assertTrue(output.exists())
            data = __import__("json").loads(output.read_text(encoding="utf-8"))
            self.assertEqual("strict", data["fallback_mode"])
            self.assertTrue(data["disable_fallback"])

    def test_compare_passes_fallback_mode_to_evaluation(self) -> None:
        calls: list[list[str]] = []

        def fake_run(cmd, cwd=None, check=False):
            calls.append(list(cmd))

        with patch("runners.compare_memory_vs_no_memory.subprocess.run", side_effect=fake_run):
            compare_memory_vs_no_memory._run_evaluation(
                config="config.json",
                model="model.pt",
                output=Path("out.json"),
                use_memory=False,
                episodes=1,
                max_steps=1,
                seed=42,
                fallback_mode="strict",
            )

        self.assertIn("--fallback-mode", calls[0])
        self.assertIn("strict", calls[0])
        self.assertIn("--disable-fallback", calls[0])
        self.assertIn("true", calls[0])

    def test_strict_comparison_records_fallback_mode_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            no_summary = Path(tmpdir) / "no.json"
            mem_summary = Path(tmpdir) / "mem.json"
            no_summary.write_text(_summary_json("no_memory"), encoding="utf-8")
            mem_summary.write_text(_summary_json("memory"), encoding="utf-8")
            summary = compare_memory_vs_no_memory.build_comparison_summary(
                json.loads(no_summary.read_text(encoding="utf-8")),
                json.loads(mem_summary.read_text(encoding="utf-8")),
                no_memory_model_path="no.pt",
                memory_model_path="mem.pt",
                seed=42,
                episodes=1,
                max_steps=1,
                fallback_mode="strict",
                strict_preflight=True,
                allow_partial_sites=False,
            )
            self.assertEqual("strict", summary["fallback_mode"])

    def test_strict_comparison_sets_disable_fallback_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            no_summary = Path(tmpdir) / "no.json"
            mem_summary = Path(tmpdir) / "mem.json"
            no_summary.write_text(_summary_json("no_memory"), encoding="utf-8")
            mem_summary.write_text(_summary_json("memory"), encoding="utf-8")
            summary = compare_memory_vs_no_memory.build_comparison_summary(
                json.loads(no_summary.read_text(encoding="utf-8")),
                json.loads(mem_summary.read_text(encoding="utf-8")),
                no_memory_model_path="no.pt",
                memory_model_path="mem.pt",
                seed=42,
                episodes=1,
                max_steps=1,
                fallback_mode="strict",
                strict_preflight=True,
                allow_partial_sites=False,
            )
            self.assertTrue(summary["disable_fallback"])


def _summary_json(label: str) -> str:
    return __import__("json").dumps(
        {
            "per_site_metrics": [
                {
                    "site_id": "site001",
                    "status": "ok",
                    "valid_for_comparison": True,
                    "completed_episodes": 1,
                    "valid_completed_episodes": 1,
                    "observed_url_sample": "http://localhost",
                    "openended_interactive_candidate_count": 1,
                    "policy_action_activity_count": 1,
                    "action_counts": {"click_element": 1},
                }
            ],
            "fallback_mode": "strict",
        }
    )


if __name__ == "__main__":
    unittest.main()
