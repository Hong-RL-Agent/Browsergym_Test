from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from runners.evaluate_multisite_browsergym_agent import _apply_eval_fallback_mask


class MemoryGuidedActionSelectionTests(unittest.TestCase):
    def test_memory_guided_action_selection_avoids_visited_target(self) -> None:
        action_space = ActionSpace(max_candidates=3)
        observation = {
            "page_state": {"site_id": "site005"},
            "candidate_elements": [_candidate("visited"), _candidate("fresh")],
        }
        mask = action_space.build_action_mask(observation)
        history = {
            "memory_encoder_enabled": True,
            "last_action_type": "click_element",
            "no_progress_steps": 1,
            "functional_action_count": 1,
            "click_target_counts": {"visited": 1},
            "element_key_click_counts": {"button|add|visited|card-1|pos-0-0-1|111": 1},
            "action_type_counts": {"click_element": 1},
        }
        filtered, _, _ = _apply_eval_fallback_mask(action_space, mask, observation, history)
        self.assertEqual(0.0, float(filtered[action_space.encode("click_element", 0)]))
        self.assertGreater(float(filtered[action_space.encode("click_element", 1)]), 0.0)
        self.assertGreater(int(history.get("memory_guided_candidate_switch_count", 0)), 0)

    def test_visited_element_key_suppression(self) -> None:
        action_space = ActionSpace(max_candidates=3)
        observation = {
            "page_state": {"site_id": "site005"},
            "candidate_elements": [_candidate("visited"), _candidate("fresh")],
        }
        mask = action_space.build_action_mask(observation)
        history = {
            "memory_encoder_enabled": True,
            "last_action_type": "click_element",
            "no_progress_steps": 1,
            "functional_action_count": 1,
            "click_target_counts": {},
            "element_key_click_counts": {"button|add|visited|card-1|pos-0-0-1|111": 1},
            "action_type_counts": {"click_element": 1},
        }
        filtered, _, _ = _apply_eval_fallback_mask(action_space, mask, observation, history)
        self.assertEqual(0.0, float(filtered[action_space.encode("click_element", 0)]))
        self.assertGreater(float(filtered[action_space.encode("click_element", 1)]), 0.0)
        self.assertGreater(int(history.get("visited_element_key_suppression_count", 0)), 0)

    def test_unclicked_element_key_priority(self) -> None:
        action_space = ActionSpace(max_candidates=3)
        observation = {
            "page_state": {"site_id": "site005"},
            "candidate_elements": [_candidate("visited"), _candidate("fresh")],
        }
        mask = action_space.build_action_mask(observation)
        history = {
            "memory_encoder_enabled": True,
            "last_action_type": "inspect_console",
            "action_type_counts": {"inspect_console": 1},
            "element_key_click_counts": {"button|add|visited|card-1|pos-0-0-1|111": 1},
        }
        filtered, _, _ = _apply_eval_fallback_mask(action_space, mask, observation, history)
        self.assertEqual(0.0, float(filtered[action_space.encode("click_element", 0)]))
        self.assertGreater(float(filtered[action_space.encode("click_element", 1)]), 0.0)

    def test_no_bug_label_used_for_memory_guided_selection(self) -> None:
        source = _function_source(
            ROOT / "runners" / "evaluate_multisite_browsergym_agent.py",
            "def _functional_priority_action_ids(",
            "def _candidate_click_count(",
        ).lower()
        for keyword in ("data-bug-id", "data_bug_id", "bug_id", "known_bug", "catalog", "label", "ground_truth", "oracle"):
            self.assertNotIn(keyword, source)


def _candidate(bid: str) -> dict:
    suffix = "111" if bid == "visited" else "222"
    context = "card-1" if bid == "visited" else "card-2"
    pos = "pos-0-0-1" if bid == "visited" else "pos-0-1-1"
    return {
        "bid": bid,
        "text": bid,
        "name": bid,
        "role": "button",
        "tag": "button",
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "clickable": True,
        "functional_priority_candidate": True,
        "functional_priority": True,
        "is_high_value_functional_candidate": True,
        "semantic_action_type": "add",
        "element_key": f"button|add|{bid}|{context}|{pos}|{suffix}",
    }


def _function_source(path: Path, start_marker: str, end_marker: str) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
