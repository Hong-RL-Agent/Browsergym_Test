from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.browsergym_observation_adapter import BrowserGymObservationAdapter, _make_candidate, _safe_visibility
from models.action_space import ActionSpace, _safe_visibility as _action_safe_visibility
from services.anomaly_detection_service import detect_anomalies, _visibility as _anomaly_visibility
from services.autonomous_reward_service import _safe_visibility as _reward_safe_visibility
from services.browsergym_training_service import _has_openended_interactive_candidate
from services.multisite_training_service import MultiSiteTrainingService
from services.policy_safe_metrics import safe_visibility as _metrics_safe_visibility


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
        self.assertEqual(_action_safe_visibility(clean), _action_safe_visibility(labeled))
        self.assertEqual(_reward_safe_visibility(clean), _reward_safe_visibility(labeled))
        self.assertEqual(_metrics_safe_visibility(clean), _metrics_safe_visibility(labeled))
        self.assertEqual(_anomaly_visibility(clean), _anomaly_visibility(labeled))

    def test_safe_visibility_used_in_action_space(self) -> None:
        self.assertEqual(1.0, _action_safe_visibility({}))
        self.assertEqual(0.0, _action_safe_visibility({"visible": False}))

    def test_safe_visibility_used_in_reward_service(self) -> None:
        self.assertEqual(1.0, _reward_safe_visibility({}))
        self.assertEqual(0.0, _reward_safe_visibility({"visible": False}))

    def test_safe_visibility_used_in_policy_safe_metrics(self) -> None:
        self.assertEqual(1.0, _metrics_safe_visibility({}))
        self.assertEqual(0.0, _metrics_safe_visibility({"visible": False}))

    def test_safe_visibility_used_in_anomaly_detector(self) -> None:
        self.assertEqual(1.0, _anomaly_visibility({}))
        self.assertEqual(0.0, _anomaly_visibility({"visible": False}))

    def test_episode_exception_records_full_traceback(self) -> None:
        service = object.__new__(MultiSiteTrainingService)
        state = {"errors": [], "episode_errors": []}
        try:
            raise UnboundLocalError("cannot access local variable 'visibility' where it is not associated with a value")
        except Exception as exc:
            record = service._record_episode_exception(
                state,
                site_id="site001",
                update_id=2,
                episode_id="U0002-site001-EP001",
                local_episode_id=1,
                step=3,
                exc=exc,
            )
        self.assertEqual("site001", record["site_id"])
        self.assertEqual(2, record["update_id"])
        self.assertEqual("UnboundLocalError", record["exception_type"])
        self.assertIn("Traceback", record["traceback"])
        self.assertIn("visibility", record["traceback"])

    def test_training_summary_records_episode_exception_count(self) -> None:
        service = _service_with_recorded_exception()
        summary = service._build_multisite_summary([], {})
        self.assertEqual(1, summary["episode_exception_count"])
        self.assertEqual(1, len(summary["episode_errors"]))

    def test_training_run_invalid_when_episode_exception_occurs(self) -> None:
        service = _service_with_recorded_exception()
        summary = service._build_multisite_summary([], {})
        self.assertFalse(summary["valid_training_run"])

    def test_training_runtime_path_no_visibility_unboundlocal_error(self) -> None:
        candidate = _candidate({"visible": True})
        observation = {"candidate_elements": [candidate]}
        action_space = ActionSpace(max_candidates=1)
        mask = action_space.build_action_mask(observation)
        self.assertEqual(1.0, float(mask[action_space.encode("click_element", 0)]))
        self.assertEqual(1.0, _reward_safe_visibility(candidate))
        self.assertEqual(1.0, _metrics_safe_visibility(candidate))
        self.assertEqual(1.0, _anomaly_visibility(candidate))

    def test_detect_anomalies_visibility_default_when_target_missing(self) -> None:
        anomalies = detect_anomalies(
            _observation([]),
            _observation([]),
            {"action": {"action_type": "fill_input", "candidate_index": 0}, "last_action_error": True},
        )
        self.assertIsInstance(anomalies, list)

    def test_detect_anomalies_visibility_default_when_visibility_missing(self) -> None:
        candidate = _candidate_without_visibility({"visible": True})
        anomalies = detect_anomalies(
            _observation([candidate]),
            _observation([candidate]),
            {"action": {"action_type": "fill_input", "candidate_index": 0}, "last_action_error": True},
        )
        self.assertIsInstance(anomalies, list)

    def test_detect_anomalies_visibility_from_visible_boolean_true(self) -> None:
        candidate = _candidate_without_visibility({"visible": True})
        detect_anomalies(
            _observation([candidate]),
            _observation([candidate]),
            {"action": {"action_type": "click_element", "candidate_index": 0}, "last_action_error": True},
        )
        self.assertEqual(1.0, _anomaly_visibility(candidate))

    def test_detect_anomalies_visibility_from_visible_boolean_false(self) -> None:
        candidate = _candidate_without_visibility({"visible": False})
        anomalies = detect_anomalies(
            _observation([candidate]),
            _observation([candidate]),
            {"action": {"action_type": "click_element", "candidate_index": 0}, "last_action_error": True},
        )
        self.assertEqual(0.0, _anomaly_visibility(candidate))
        self.assertTrue(any(item.get("type") == "low-visibility-interaction" for item in anomalies))

    def test_detect_anomalies_no_unboundlocal_when_last_action_error_and_no_state_change(self) -> None:
        candidate = _candidate_without_visibility({"visible": True})
        anomalies = detect_anomalies(
            _observation([candidate]),
            _observation([candidate]),
            {"action": {"action_type": "submit_form", "candidate_index": 0}, "last_action_error": True},
        )
        self.assertIsInstance(anomalies, list)

    def test_detect_anomalies_no_unboundlocal_when_target_candidate_none(self) -> None:
        anomalies = detect_anomalies(
            _observation([]),
            _observation([]),
            {"action": {"action_type": "click_element", "candidate_index": 99}, "last_action_error": True},
        )
        self.assertIsInstance(anomalies, list)

    def test_detect_anomalies_no_unboundlocal_when_action_target_empty(self) -> None:
        anomalies = detect_anomalies(
            _observation([]),
            _observation([]),
            {"action": {"action_type": "press_enter"}, "last_action_error": True},
        )
        self.assertIsInstance(anomalies, list)

    def test_fail_on_episode_exception_stops_training(self) -> None:
        service = _minimal_service(fail_on_episode_exception=True)

        class FailingEnv:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def reset(self):
                raise RuntimeError("synthetic reset failure")

            def close(self) -> None:
                pass

        try:
            with patch("builtins.print"), patch.object(service, "_run_preflight_checks", return_value=None), patch(
                "services.multisite_training_service.BrowserGymJAWSEnv",
                FailingEnv,
            ):
                with self.assertRaises(RuntimeError):
                    service.train()
        finally:
            service._tmpdir.cleanup()


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


def _candidate_without_visibility(extra: dict) -> dict:
    candidate = _candidate(extra)
    candidate.pop("visibility", None)
    if "visible" in extra:
        candidate["visible"] = extra["visible"]
    return candidate


def _observation(candidates: list[dict]) -> dict:
    return {
        "page_state": {
            "url": "http://localhost/test",
            "title": "Test",
            "page_text_length": 10,
            "dom_node_count": 3,
        },
        "candidate_elements": candidates,
        "runtime_signals": {},
        "layout_signals": {},
    }


def _minimal_service(*, fail_on_episode_exception: bool = False) -> MultiSiteTrainingService:
    tmpdir = tempfile.TemporaryDirectory()
    root = Path(tmpdir.name)
    config_path = root / "config.json"
    config_path.write_text(
        __import__("json").dumps(
            {
                "run_id": "v4_blind_url_visibility_fail_test",
                "blind_url_training": True,
                "output_dir": str(root / "out"),
                "shared_model_path": str(root / "model.pt"),
                "sites": [{"site_id": "site001", "base_url": "http://localhost:9220", "enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    service = MultiSiteTrainingService(
        config_path=config_path,
        total_updates=1,
        episodes_per_site=1,
        max_steps=1,
        enable_csv_logging=False,
        fail_on_episode_exception=fail_on_episode_exception,
    )
    service._tmpdir = tmpdir
    return service


def _service_with_recorded_exception() -> MultiSiteTrainingService:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        config_path.write_text(
            __import__("json").dumps(
                {
                    "run_id": "v4_blind_url_visibility_test",
                    "blind_url_training": True,
                    "output_dir": str(Path(tmpdir) / "out"),
                    "shared_model_path": str(Path(tmpdir) / "model.pt"),
                    "sites": [{"site_id": "site001", "base_url": "http://localhost:9220", "enabled": True}],
                }
            ),
            encoding="utf-8",
        )
        service = MultiSiteTrainingService(
            config_path=config_path,
            total_updates=1,
            episodes_per_site=1,
            max_steps=1,
            enable_csv_logging=False,
        )
        state = service.site_states["site001"]
        try:
            raise RuntimeError("synthetic episode failure")
        except Exception as exc:
            service._record_episode_exception(
                state,
                site_id="site001",
                update_id=1,
                episode_id="U0001-site001-EP001",
                local_episode_id=1,
                step=1,
                exc=exc,
            )
        return service


if __name__ == "__main__":
    unittest.main()
