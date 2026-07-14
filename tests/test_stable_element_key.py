from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.browsergym_observation_adapter import _make_candidate


class StableElementKeyTests(unittest.TestCase):
    def test_candidate_has_stable_element_key(self) -> None:
        candidate = _candidate(text="수강신청", class_name="course-card-3")
        self.assertTrue(candidate["element_key"])
        self.assertEqual("role|semantic|text|context|position", candidate["element_key_source"])
        self.assertEqual("button", candidate["element_role"])
        self.assertIn("수강신청", candidate["element_text_norm"])
        self.assertTrue(candidate["element_context_key"])
        self.assertTrue(candidate["element_position_key"])

    def test_element_key_does_not_use_bug_label(self) -> None:
        plain = _candidate(text="수강신청", class_name="course-card-3")
        labeled = _candidate(text="수강신청", class_name="course-card-3", data_bug_id="site004-bug01")
        self.assertEqual(plain["element_key"], labeled["element_key"])

    def test_element_key_does_not_use_data_bug_id(self) -> None:
        first = _candidate(text="담기", class_name="restaurant-card-2", data_bug_id="site005-bug01")
        second = _candidate(text="담기", class_name="restaurant-card-2", data_bug_id="site005-bug99")
        self.assertEqual(first["element_key"], second["element_key"])

    def test_same_bid_generates_same_element_key(self) -> None:
        first = _candidate(bid="button-1", text="담기", class_name="restaurant-card-2")
        second = _candidate(bid="button-1", text="담기", class_name="restaurant-card-2")
        self.assertEqual(first["element_key"], second["element_key"])

    def test_same_text_different_context_generates_different_element_key(self) -> None:
        first = _candidate(text="담기", class_name="restaurant-card-1", bbox=[0, 0, 100, 40])
        second = _candidate(text="담기", class_name="restaurant-card-2", bbox=[0, 120, 100, 40])
        self.assertNotEqual(first["element_key"], second["element_key"])


def _candidate(
    *,
    bid: str = "button",
    text: str,
    class_name: str,
    bbox: list[float] | None = None,
    data_bug_id: str = "",
) -> dict:
    return _make_candidate(
        bid=bid,
        text=text,
        name=text,
        role="button",
        tag="button",
        bbox=bbox or [0, 0, 100, 40],
        visibility=1.0,
        clickable=True,
        enabled=True,
        source="test",
        page_text="",
        class_name=class_name,
        data_bug_id=data_bug_id,
    )


if __name__ == "__main__":
    unittest.main()
