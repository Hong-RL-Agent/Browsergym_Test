from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from runners.evaluate_multisite_browsergym_agent import _apply_eval_fallback_mask, _fallback_reason


class ActionFallbackReasonDetailTests(unittest.TestCase):
    def test_fallback_reason_not_generic_fallback_only(self) -> None:
        action_space = ActionSpace(max_candidates=2)
        observation = _observation([_generic_candidate(), _high_value_candidate()])
        mask = action_space.build_action_mask(observation)
        history: dict = {"action_type_counts": {}, "click_target_counts": {}, "element_key_click_counts": {}}

        _, warning, repeated = _apply_eval_fallback_mask(action_space, mask, observation, history)

        self.assertEqual(1, repeated)
        self.assertIn("low-value generic", warning)
        self.assertNotEqual(["fallback"], history.get("_fallback_reasons_step"))
        self.assertIn("low_value_generic_candidate", history.get("_fallback_reasons_step", []))

    def test_fallback_reason_counts_are_detailed(self) -> None:
        self.assertEqual("visited_element_key", _fallback_reason("WARNING: site visited target suppressed; unclicked candidate fallback applied."))
        self.assertEqual("policy_selected_meta_action", _fallback_reason("WARNING: site repeated meta action inspect_console blocked; functional fallback applied."))
        self.assertEqual("repeated_action_signature", _fallback_reason("WARNING: site repeated action signature blocked; action fallback applied."))
        self.assertEqual("exploration_redirect", _fallback_reason("WARNING: site high-value click verification action selected."))

    def test_no_catalog_used_for_fallback_reasoning(self) -> None:
        source = _function_source(
            ROOT / "runners" / "evaluate_multisite_browsergym_agent.py",
            "def _suppress_low_value_generic_candidates(",
            "def _prefer_verification_action(",
        ).lower()
        self.assertNotIn("catalog", source)


def _observation(candidates: list[dict]) -> dict:
    return {"page_state": {"site_id": "site005"}, "candidate_elements": candidates}


def _generic_candidate() -> dict:
    return {
        "bid": "g",
        "text": "element-12",
        "name": "element-12",
        "role": "generic",
        "tag": "div",
        "visible": True,
        "visibility": 1.0,
        "enabled": True,
        "clickable": True,
        "is_low_value_generic_candidate": True,
        "element_key": "generic|none|element-12|test|pos-0-0-2|aaa",
    }


def _high_value_candidate() -> dict:
    return {
        "bid": "add",
        "text": "담기",
        "name": "담기",
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
        "element_key": "button|add|담기|test|pos-0-0-2|bbb",
    }


def _function_source(path: Path, start_marker: str, end_marker: str) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
