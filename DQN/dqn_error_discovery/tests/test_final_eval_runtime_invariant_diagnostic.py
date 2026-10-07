import json
from pathlib import Path

from dqn_error_discovery.framework_dev.diagnose_final_runtime_invariants import (
    _run_row,
    _timeline_integrity,
    build_diagnostic,
)


def _write_run(root: Path, *, action_errors: int = 0, duplicate: bool = False) -> None:
    site = root / "SITE002"
    site.mkdir(parents=True)
    timeline = [{"episode": episode, "step": 1} for episode in range(1, 11)]
    if duplicate:
        timeline.append({"episode": 1, "step": 1})
    payload = {
        "diagnostic": {"action_error_count": action_errors},
        "hard_invariants": {"all_pass": action_errors == 0, "ground_truth_used": False},
        "environment_continuity": {"unexpected_reset_count": 0, "reset_at_switch_count": 0},
        "timeline": timeline,
    }
    (site / "ppo_seed42.json").write_text(json.dumps(payload), encoding="utf-8")
    (site / "ppo_seed42_runtime.json").write_text(
        json.dumps({"runtime_ground_truth_imported": False}), encoding="utf-8"
    )


def test_action_error_extraction_and_event_context_is_explicitly_unknown(tmp_path):
    _write_run(tmp_path, action_errors=2)
    row = _run_row(tmp_path, "SITE002", "ppo", 42)
    assert row["action_error_count"] == 2
    assert row["event_level_context_available"] is False
    assert row["root_cause"] == "UNAVAILABLE_EVENT_CONTEXT"
    assert row["validity"] == "REVIEW"


def test_timeline_duplicate_detection():
    result = _timeline_integrity([{"episode": 1, "step": 1}, {"episode": 1, "step": 1}])
    assert result["duplicate_step_pairs"] == 1


def test_artifact_integrity_does_not_use_ground_truth(tmp_path):
    _write_run(tmp_path)
    payload = build_diagnostic(tmp_path)
    assert payload["ground_truth_used_for_validity"] is False
    assert payload["runs"][0]["ground_truth_used"] is False


def test_action_error_does_not_overwrite_or_mutate_source_artifact(tmp_path):
    _write_run(tmp_path, action_errors=1)
    source = (tmp_path / "SITE002" / "ppo_seed42.json").read_bytes()
    build_diagnostic(tmp_path)
    assert (tmp_path / "SITE002" / "ppo_seed42.json").read_bytes() == source


def test_missing_episode_is_invalid_artifact(tmp_path):
    _write_run(tmp_path)
    path = tmp_path / "SITE002" / "ppo_seed42.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["timeline"] = [row for row in payload["timeline"] if row["episode"] != 10]
    path.write_text(json.dumps(payload), encoding="utf-8")
    row = _run_row(tmp_path, "SITE002", "ppo", 42)
    assert row["validity"] == "INVALID_ARTIFACT"
