from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.browsergym_observation_adapter import BrowserGymObservationAdapter, _make_candidate, _safe_visibility
from models.action_space import ActionSpace
from services.autonomous_reward_service import _safe_visibility as _reward_safe_visibility
from services.browsergym_training_service import _has_openended_interactive_candidate


class VisibilityDefaultHandlingTests(unittest.TestCase):
    def test_candidate_visibility_default_when_missing(self) -> None:
        self.assertEqual(1.0, _safe_visibility({}, default=1.0))
        candidate = _candidate({})
        self.assertEqual(1.0, candidate["visibility"])
        self.assertTrue(candidate["visible"])

    def test_candidate_visibility_from_visible_boolean_true(self) -> None:
        self.assertEqual(1.0, _safe_visibility({"visible": True}))

    def test_candidate_visibility_from_visible_boolean_false(self) -> None:
        self.assertEqual(0.0, _safe_visibility({"visible": False}))

    def test_action_selection_no_visibility_unboundlocal_error(self) -> None:
        action_space = ActionSpace(max_candidates=1)
        observation = {"candidate_elements": [_candidate({"visibility": None})]}
        mask = action_space.build_action_mask(observation)
        self.assertEqual(1.0, float(mask[action_space.encode("click_element", 0)]))

    def test_element_key_generation_no_visibility_unboundlocal_error(self) -> None:
        candidate = _candidate({"visible": True})
        self.assertIn("element_key", candidate)
        self.assertNotEqual("", candidate["element_key"])

    def test_training_episode_no_visibility_unboundlocal_error(self) -> None:
        observation = {"candidate_elements": [_candidate({"visible": True})]}
        self.assertTrue(_has_openended_interactive_candidate(observation))

    def test_adapter_extra_prop_missing_visibility_does_not_crash(self) -> None:
        adapter = BrowserGymObservationAdapter(max_candidates=4)
        obs = {
            "url": "http://local",
            "title": "Local",
            "extra_element_properties": {
                "b1": {
                    "bbox": [0, 0, 100, 40],
                    "text": "Add",
                    "role": "button",
                    "clickable": True,
                    "visible": True,
                }
            },
        }
        converted = adapter.adapt(obs, info={"site_id": "site001"})
        self.assertGreaterEqual(len(converted["candidate_elements"]), 1)
        self.assertEqual(1.0, converted["candidate_elements"][0]["visibility"])

    def test_no_bug_label_used_in_visibility_handling(self) -> None:
        clean = {"visible": True}
        labeled = {
            "visible": True,
            "data-bug-id": "site001-bug01",
            "bug_id": "site001-bug01",
            "catalog_bug_id_matches": True,
            "known_bug_id": "site001-bug01",
            "site_group": "known-bug-site",
            "target_signal_types": ["console"],
        }
        self.assertEqual(_safe_visibility(clean), _safe_visibility(labeled))
        self.assertEqual(_reward_safe_visibility(clean), _reward_safe_visibility(labeled))


def _candidate(extra: dict) -> dict:
    return _make_candidate(
        bid="b1",
        text="Add",
        name="Add",
        role="button",
        tag="button",
        bbox=[0, 0, 100, 40],
        visibility=_safe_visibility(extra),
        clickable=True,
        enabled=True,
        source="test",
        page_text="Add",
    )


if __name__ == "__main__":
    unittest.main()
