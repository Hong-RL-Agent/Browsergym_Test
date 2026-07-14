from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder
from agents.ppo_agent import PPOAgent
from services.autonomous_reward_service import calculate_autonomous_reward


def _candidate() -> dict:
    return {
        "bid": "candidate-1",
        "tag": "button",
        "role": "button",
        "text": "Add",
        "name": "Add",
        "visible": True,
        "enabled": True,
        "clickable": True,
        "visibility": 1.0,
        "bbox": [10.0, 20.0, 120.0, 32.0],
        "text_length": 3,
        "has_text": True,
        "is_form_field": False,
        "href": "",
        "has_href": False,
    }


def _observation(with_metadata: bool = False) -> dict:
    page_state = {
        "site_id": "site001",
        "url": "http://localhost:9220/",
        "title": "Blind safety test",
        "viewport_width": 1280,
        "viewport_height": 720,
        "page_text_length": 100,
        "dom_node_count": 10,
        "elapsed_time": 1,
    }
    if with_metadata:
        page_state.update(
            {
                "site_group": "ui_dom",
                "target_signal_types": ["ui-dom", "interaction"],
                "expected_signal_types": ["ui-dom"],
                "expected_bug_type": "button-no-response",
                "known_bug_catalog": [{"bug_id": "site001-bug01"}],
            }
        )
    return {
        "page_state": page_state,
        "candidate_elements": [_candidate()],
        "runtime_signals": {},
        "layout_signals": {},
        "infra_signals": {},
        "history": {"step_index": 6, "no_change_steps": 0, "previous_action_type": "noop"},
    }


class PolicySafetyNoMetadataLeakageTests(unittest.TestCase):
    def test_observation_does_not_include_site_metadata(self) -> None:
        encoder = ObservationEncoder(max_candidates=4)
        without_metadata = encoder.encode_observation(_observation(False))
        with_metadata = encoder.encode_observation(_observation(True))
        np.testing.assert_array_equal(without_metadata, with_metadata)

    def test_action_selection_does_not_use_site_metadata(self) -> None:
        action_space = ActionSpace(max_candidates=4)
        without_metadata = action_space.build_action_mask(_observation(False))
        with_metadata = action_space.build_action_mask(_observation(True))
        np.testing.assert_array_equal(without_metadata, with_metadata)

    def test_reward_does_not_use_site_metadata(self) -> None:
        before = _observation(False)
        after = _observation(False)
        action = {"action_type": "click_element", "candidate_index": 0}
        reward_without_metadata, breakdown_without_metadata = calculate_autonomous_reward(
            before,
            after,
            action,
            [],
            known_bugs=[],
            history={},
            site_profile={"reward_mode": "signal_based"},
        )
        reward_with_metadata, breakdown_with_metadata = calculate_autonomous_reward(
            before,
            after,
            action,
            [],
            known_bugs=[],
            history={},
            site_profile={
                "reward_mode": "signal_based",
                "site_group": "ui_dom",
                "target_signal_types": ["ui-dom", "interaction"],
                "expected_signal_types": ["ui-dom"],
                "expected_bug_type": "button-no-response",
                "known_bug_catalog": [{"bug_id": "site001-bug01"}],
            },
        )
        self.assertEqual(reward_without_metadata, reward_with_metadata)
        self.assertEqual(
            breakdown_without_metadata["reward_total"],
            breakdown_with_metadata["reward_total"],
        )
        self.assertFalse(breakdown_with_metadata["use_known_bug_reward"])
        self.assertFalse(breakdown_with_metadata["policy_uses_bug_labels"])

    def test_memory_encoder_does_not_receive_site_metadata(self) -> None:
        encoder = ObservationEncoder(max_candidates=4)
        agent = PPOAgent(encoder.get_obs_dim(), 8, use_memory_encoder=True, memory_hidden_size=16)
        vector_without_metadata = encoder.encode_observation(_observation(False))
        vector_with_metadata = encoder.encode_observation(_observation(True))
        memory = agent.reset_memory_state()
        next_without = agent.update_memory_state(vector_without_metadata, memory)
        next_with = agent.update_memory_state(vector_with_metadata, memory)
        np.testing.assert_allclose(next_without, next_with, rtol=1e-6, atol=1e-6)

    def test_memory_encoder_does_not_receive_bug_labels(self) -> None:
        safe = _observation(False)
        labeled = _observation(False)
        labeled["candidate_elements"][0].update(
            {
                "data-bug-id": "site001-bug01",
                "bug_id": "site001-bug01",
                "known_bug_id": "site001-bug01",
                "catalog_bug_id_matches": ["site001-bug01"],
                "has_data_bug_id": True,
            }
        )
        encoder = ObservationEncoder(max_candidates=4)
        agent = PPOAgent(encoder.get_obs_dim(), 8, use_memory_encoder=True, memory_hidden_size=16)
        memory = agent.reset_memory_state()
        next_safe = agent.update_memory_state(encoder.encode_observation(safe), memory)
        next_labeled = agent.update_memory_state(encoder.encode_observation(labeled), memory)
        np.testing.assert_allclose(next_safe, next_labeled, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
