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


FORBIDDEN_POLICY_KEYWORDS = (
    "data-bug-id",
    "bug_id",
    "known_bug",
    "catalog_bug_id",
    "catalog_bug_id_matches",
    "has_data_bug_id",
    "ground_truth",
    "oracle",
    "label",
    "bug_type",
)


def _candidate(**overrides):
    candidate = {
        "bid": "candidate-1",
        "tag": "button",
        "role": "button",
        "text": "Save",
        "name": "Save",
        "visible": True,
        "enabled": True,
        "clickable": True,
        "visibility": 1.0,
        "bbox": [10.0, 20.0, 120.0, 32.0],
        "text_length": 4,
        "has_text": True,
        "is_form_field": False,
        "href": "",
        "has_href": False,
    }
    candidate.update(overrides)
    return candidate


def _observation(candidate):
    return {
        "page_state": {
            "site_id": "site001",
            "url": "http://localhost:9220/",
            "title": "Leakage test",
            "viewport_width": 1280,
            "viewport_height": 720,
            "page_text_length": 100,
            "dom_node_count": 10,
            "elapsed_time": 1,
            "cart_count_detected": False,
            "cart_count": 0,
        },
        "candidate_elements": [candidate],
        "runtime_signals": {},
        "layout_signals": {},
        "infra_signals": {},
        "history": {"step_index": 6, "no_change_steps": 0, "previous_action_type": "noop"},
    }


class PolicyLabelLeakageTests(unittest.TestCase):
    def test_action_mask_ignores_bug_label_metadata(self) -> None:
        action_space = ActionSpace(max_candidates=4)
        safe_candidate = _candidate()
        labeled_candidate = _candidate(
            **{
                "catalog_bug_id_matches": ["site001-bug01"],
                "has_data_bug_id": True,
                "known_bug_id": "site001-bug01",
                "data-bug-id": "site001-bug01",
                "bug_id": "site001-bug01",
                "bug_type": "known",
                "ground_truth": True,
                "oracle": True,
                "label": "positive",
            }
        )

        mask_without_labels = action_space.build_action_mask(_observation(safe_candidate))
        mask_with_labels = action_space.build_action_mask(_observation(labeled_candidate))

        np.testing.assert_array_equal(mask_without_labels, mask_with_labels)

    def test_observation_vector_ignores_bug_label_metadata(self) -> None:
        encoder = ObservationEncoder(max_candidates=4)
        safe_candidate = _candidate()
        labeled_candidate = _candidate(
            **{
                "catalog_bug_id_matches": ["site001-bug01"],
                "has_data_bug_id": True,
                "known_bug_id": "site001-bug01",
                "data-bug-id": "site001-bug01",
                "bug_id": "site001-bug01",
                "bug_type": "known",
                "ground_truth": True,
                "oracle": True,
                "label": "positive",
            }
        )

        vector_without_labels = encoder.encode_observation(_observation(safe_candidate))
        vector_with_labels = encoder.encode_observation(_observation(labeled_candidate))

        np.testing.assert_array_equal(vector_without_labels, vector_with_labels)

    def test_action_mask_stats_ignore_bug_label_metadata(self) -> None:
        action_space = ActionSpace(max_candidates=4)
        safe_candidate = _candidate()
        labeled_candidate = _candidate(
            **{
                "catalog_bug_id_matches": ["site001-bug01"],
                "has_data_bug_id": True,
                "known_bug_id": "site001-bug01",
                "data-bug-id": "site001-bug01",
                "bug_id": "site001-bug01",
                "bug_type": "known",
                "ground_truth": True,
                "oracle": True,
                "label": "positive",
            }
        )
        safe_observation = _observation(safe_candidate)
        labeled_observation = _observation(labeled_candidate)
        selected_action_id = action_space.encode("click_element", 0)

        stats_without_labels = action_space.build_action_mask_stats(
            safe_observation,
            action_space.build_action_mask(safe_observation),
            selected_action_id,
        )
        stats_with_labels = action_space.build_action_mask_stats(
            labeled_observation,
            action_space.build_action_mask(labeled_observation),
            selected_action_id,
        )

        self.assertEqual(stats_without_labels, stats_with_labels)

    def test_policy_path_keyword_audit(self) -> None:
        policy_sources = {
            "models/action_space.py": (ROOT / "models" / "action_space.py").read_text(encoding="utf-8"),
            "models/observation_encoder.py": (ROOT / "models" / "observation_encoder.py").read_text(encoding="utf-8"),
            "agents/ppo_agent.py": (ROOT / "agents" / "ppo_agent.py").read_text(encoding="utf-8"),
            "services/multisite_training_service.py guided action": _guided_action_source(),
        }
        hits = []
        for source_name, source in policy_sources.items():
            lowered = source.lower()
            for keyword in FORBIDDEN_POLICY_KEYWORDS:
                if keyword in lowered:
                    hits.append(f"{source_name}: {keyword}")

        self.assertEqual([], hits)


def _guided_action_source() -> str:
    path = ROOT / "services" / "multisite_training_service.py"
    source = path.read_text(encoding="utf-8")
    start = source.index("def _policy_safe_guided_action_id(")
    end = source.index("def _reward_scale(", start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
