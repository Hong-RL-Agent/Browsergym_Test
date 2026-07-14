from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.browsergym_observation_adapter import _make_candidate


class CandidateSemanticFilterSearchTests(unittest.TestCase):
    def test_filter_category_tab_search_input_classification(self) -> None:
        self.assertIn(_candidate("Programming")["semantic_action_type"], {"filter", "category"})
        self.assertIn(_candidate("Design")["semantic_action_type"], {"filter", "category"})
        self.assertIn(_candidate("Business")["semantic_action_type"], {"filter", "category"})
        self.assertEqual("tab", _candidate("Details", role="tab")["semantic_action_type"])
        self.assertEqual("search_input", _candidate("Search", role="searchbox", tag="input")["semantic_action_type"])


def _candidate(text: str, *, role: str = "button", tag: str = "button") -> dict:
    return _make_candidate(
        bid=text.lower().replace(" ", "-"),
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
    )


if __name__ == "__main__":
    unittest.main()
