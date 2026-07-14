from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from runners.evaluate_multisite_browsergym_agent import (
    _apply_eval_fallback_mask,
    _functional_priority_action_ids,
)


class ActionRepeatSuppressionTests(unittest.TestCase):
    def test_repeated_target_priority_lowered(self) -> None:
        action_space = ActionSpace(max_candidates=3)
        observation = _observation([_candidate("already", high=True), _candidate("new", high=True)])
        mask = action_space.build_action_mask(observation)
        ids = _functional_priority_action_ids(
            action_space,
            mask,
            observation,
            {"click_target_counts": {"already": 1}},
        )
        self.assertEqual(action_space.encode("click_element", 1), ids[0])
        self.assertNotIn(action_space.encode("click_element", 0), ids)

    def test_fallback_prefers_unclicked_high_value_candidate(self) -> None:
        action_space = ActionSpace(max_candidates=3)
        observation = _observation([_candidate("visited", high=True), _candidate("fresh", high=True)])
        mask = action_space.build_action_mask(observation)
        filtered, warning, repeated = _apply_eval_fallback_mask(
            action_space,
            mask,
            observation,
            {
                "last_action_type": "click_element",
                "no_progress_steps": 1,
                "functional_action_count": 1,
                "click_target_counts": {"visited": 1},
                "action_type_counts": {"click_element": 1},
            },
        )
        self.assertEqual(1, repeated)
        self.assertIn("no-effect click", warning)
        self.assertEqual(0.0, float(filtered[action_space.encode("click_element", 0)]))
        self.assertGreater(float(filtered[action_space.encode("click_element", 1)]), 0.0)

    def test_repeated_action_redirects_to_new_candidate(self) -> None:
        action_space = ActionSpace(max_candidates=3)
        observation = _observation([_candidate("visited", high=True), _candidate("fresh", high=True)])
        mask = action_space.build_action_mask(observation)
        history = {
            "last_action_type": "click_element",
            "consecutive_action_type_counts": {"click_element": 2},
            "click_target_counts": {"visited": 1},
            "action_type_counts": {"click_element": 2},
        }
        filtered, warning, repeated = _apply_eval_fallback_mask(action_space, mask, observation, history)
        self.assertEqual(1, repeated)
        self.assertIn("visited target suppressed", warning)
        self.assertEqual(0, int(history.get("repeated_action_redirect_count", 0) or 0))
        self.assertGreater(int(history.get("visited_element_key_suppression_count", 0)), 0)
        self.assertGreater(float(filtered[action_space.encode("click_element", 1)]), 0.0)

    def test_finish_delayed_when_unclicked_functional_candidates_remain(self) -> None:
        action_space = ActionSpace(max_candidates=3)
        observation = _observation([_candidate("fresh", high=True)])
        mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
        mask[action_space.encode("finish_episode", 0)] = 1.0
        mask[action_space.encode("click_element", 0)] = 1.0
        history = {"action_type_counts": {}}
        filtered, warning, _ = _apply_eval_fallback_mask(action_space, mask, observation, history)
        self.assertIn("finish_episode delayed", warning)
        self.assertEqual(0.0, float(filtered[action_space.encode("finish_episode", 0)]))
        self.assertEqual(1, int(history.get("finish_delayed_by_unclicked_candidate_count", 0)))

    def test_no_catalog_used_for_repeat_suppression(self) -> None:
        source = _function_source(
            ROOT / "runners" / "evaluate_multisite_browsergym_agent.py",
            "def _suppress_visited_click_targets(",
            "def _delay_finish_when_unclicked_candidates(",
        ).lower()
        self.assertNotIn("catalog", source)


def _candidate(bid: str, *, high: bool) -> dict:
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
        "is_high_value_functional_candidate": high,
        "semantic_action_type": "add" if high else "category",
        "element_key": f"button|{'add' if high else 'category'}|{bid}|test|pos-0-0-1|{bid}",
    }


def _observation(candidates: list[dict]) -> dict:
    return {"page_state": {"site_id": "site005"}, "candidate_elements": candidates}


def _function_source(path: Path, start_marker: str, end_marker: str) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
