from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.browsergym_observation_adapter import _candidate_sort_key, _make_candidate, _semantic_action_type
from models.action_space import ActionSpace
from runners.evaluate_multisite_browsergym_agent import _apply_eval_fallback_mask
from services.anomaly_detection_service import detect_anomalies


class CandidateSemanticPriorityTests(unittest.TestCase):
    def test_workout_add_button_classified_as_functional_priority(self) -> None:
        candidate = _candidate("운동 추가", role="button")
        self.assertEqual("workout_add", candidate["semantic_action_type"])
        self.assertTrue(candidate["functional_priority_candidate"])
        self.assertTrue(candidate["is_workout_add_action"])
        self.assertTrue(candidate["is_high_value_functional_candidate"])

    def test_cart_or_add_button_classified_as_functional_priority(self) -> None:
        cart = _candidate("장바구니 담기", role="button")
        add = _candidate("추가", role="button")
        self.assertIn(cart["semantic_action_type"], {"cart", "add"})
        self.assertTrue(cart["functional_priority_candidate"])
        self.assertEqual("add", add["semantic_action_type"])
        self.assertTrue(add["functional_priority_candidate"])

    def test_functional_priority_candidate_selected_before_scroll(self) -> None:
        action_space = ActionSpace(max_candidates=4)
        observation = {
            "page_state": {"site_id": "site003"},
            "candidate_elements": [
                _candidate("운동 추가", bid="workout", role="button"),
                _candidate("일반 텍스트", bid="plain", role="button"),
            ],
        }
        mask = action_space.build_action_mask(observation)
        filtered, warning, repeated = _apply_eval_fallback_mask(
            action_space,
            mask,
            observation,
            {"functional_action_count": 0, "action_type_counts": {"scroll_down": 1, "click_retry_button": 1}},
        )
        self.assertEqual(1, repeated)
        self.assertIn("functional fallback", warning)
        self.assertGreater(float(filtered[action_space.encode("click_element", 0)]), 0.0)
        self.assertEqual(0.0, float(filtered[action_space.encode("scroll_down", 0)]))

    def test_no_effect_click_fallback_moves_to_unclicked_functional_candidate(self) -> None:
        action_space = ActionSpace(max_candidates=4)
        observation = {
            "page_state": {"site_id": "site003"},
            "candidate_elements": [
                _candidate("운동 추가", bid="already", role="button"),
                _candidate("저장", bid="save", role="button"),
            ],
        }
        mask = action_space.build_action_mask(observation)
        filtered, warning, repeated = _apply_eval_fallback_mask(
            action_space,
            mask,
            observation,
            {
                "last_action_type": "click_element",
                "no_progress_steps": 1,
                "functional_action_count": 1,
                "click_target_counts": {"already": 1},
                "action_type_counts": {"click_element": 1},
            },
        )
        self.assertEqual(1, repeated)
        self.assertIn("no-effect click", warning)
        self.assertEqual(0.0, float(filtered[action_space.encode("click_element", 0)]))
        self.assertGreater(float(filtered[action_space.encode("click_element", 1)]), 0.0)

    def test_semantic_priority_does_not_use_bug_labels(self) -> None:
        source = _function_source(
            ROOT / "adapters" / "browsergym_observation_adapter.py",
            "def _semantic_action_type(",
            "def _normalize_semantic_text(",
        ).lower()
        for keyword in (
            "data-bug-id",
            "data_bug_id",
            "bug_id",
            "known_bug",
            "catalog_bug_id",
            "catalog_bug_id_matches",
            "ground_truth",
            "oracle",
            "label",
        ):
            self.assertNotIn(keyword, source)

    def test_candidate_priority_ignores_data_bug_id(self) -> None:
        without_label = _candidate("운동 추가", bid="same", data_bug_id="")
        with_label = _candidate("운동 추가", bid="same", data_bug_id="site003-bug01")
        self.assertEqual(without_label["semantic_action_type"], with_label["semantic_action_type"])
        self.assertEqual(without_label["functional_priority_candidate"], with_label["functional_priority_candidate"])
        self.assertEqual(_candidate_sort_key(without_label), _candidate_sort_key(with_label))

    def test_workout_add_no_effect_creates_button_no_response_anomaly(self) -> None:
        observation = _observation_with_candidate(_candidate("운동 추가", bid="workout"))
        anomalies = detect_anomalies(
            observation,
            _observation_with_candidate(_candidate("운동 추가", bid="workout")),
            {"action": {"action_type": "click_element", "candidate_index": 0}},
            {},
        )
        self.assertTrue(any(item.get("type") == "button-no-response" for item in anomalies))
        self.assertTrue(any(item.get("evidence", {}).get("workout_add_no_effect") for item in anomalies))

    def test_cart_no_effect_creates_interaction_anomaly(self) -> None:
        observation = _observation_with_candidate(_candidate("장바구니 담기", bid="cart"))
        anomalies = detect_anomalies(
            observation,
            _observation_with_candidate(_candidate("장바구니 담기", bid="cart")),
            {"action": {"action_type": "click_element", "candidate_index": 0}},
            {},
        )
        self.assertTrue(any(item.get("type") == "button-no-response" for item in anomalies))
        self.assertTrue(any(item.get("evidence", {}).get("semantic_no_effect_click") for item in anomalies))

    def test_high_value_functional_click_no_effect_recorded(self) -> None:
        observation = _observation_with_candidate(_candidate("저장", bid="save"))
        anomalies = detect_anomalies(
            observation,
            _observation_with_candidate(_candidate("저장", bid="save")),
            {"action": {"action_type": "click_element", "candidate_index": 0}},
            {},
        )
        self.assertTrue(any(item.get("evidence", {}).get("functional_no_effect_anomaly") for item in anomalies))

    def test_low_value_click_no_effect_not_overreported(self) -> None:
        candidate = _candidate("섹션 제목", bid="title", role="generic", tag="div")
        candidate["semantic_action_type"] = ""
        candidate["functional_priority_candidate"] = False
        candidate["functional_priority"] = False
        candidate["is_high_value_functional_candidate"] = False
        observation = _observation_with_candidate(candidate)
        anomalies = detect_anomalies(
            observation,
            _observation_with_candidate(candidate),
            {"action": {"action_type": "click_element", "candidate_index": 0}},
            {},
        )
        self.assertFalse(any(item.get("evidence", {}).get("semantic_no_effect_click") for item in anomalies))

    def test_semantic_no_effect_detection_does_not_use_bug_labels(self) -> None:
        source = _function_source(
            ROOT / "services" / "anomaly_detection_service.py",
            "def _is_high_value_semantic_action(",
            "def _looks_like_form_submission(",
        ).lower()
        for keyword in (
            "data-bug-id",
            "data_bug_id",
            "bug_id",
            "known_bug",
            "catalog_bug_id",
            "catalog_bug_id_matches",
            "ground_truth",
            "oracle",
            "label",
        ):
            self.assertNotIn(keyword, source)


def _candidate(
    text: str,
    *,
    bid: str = "b1",
    role: str = "button",
    tag: str = "button",
    data_bug_id: str = "",
) -> dict:
    return _make_candidate(
        bid=bid,
        text=text,
        name=text,
        role=role,
        tag=tag,
        bbox=[0, 0, 100, 30],
        visibility=1.0,
        clickable=True,
        enabled=True,
        source="test",
        page_text="",
        data_bug_id=data_bug_id,
    )


def _observation_with_candidate(candidate: dict) -> dict:
    return {
        "page_state": {
            "url": "http://localhost:9222/",
            "title": "Example",
            "page_text_length": 100,
            "page_text": "Example page",
            "page_text_sample": "Example page",
            "dom_node_count": 10,
            "has_modal_or_dialog": False,
            "has_form": False,
            "cart_count_detected": False,
        },
        "candidate_elements": [dict(candidate)],
        "runtime_signals": {},
        "layout_signals": {},
    }


def _function_source(path: Path, start_marker: str, end_marker: str) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
