import json

from dqn_error_discovery.framework_dev.action_error_event_recorder import ActionErrorEventRecorder


def test_recorder_writes_selection_execution_and_reset_events(tmp_path):
    path = tmp_path / "events.jsonl"
    with ActionErrorEventRecorder(
        path,
        site="site002",
        method="ppo",
        seed=42,
        canonical_path="canonical.json",
        canonical_error_count=1,
    ) as recorder:
        recorder.record_reset(episode=1, reason="episode_boundary")
        recorder.record_step({
            "action_trace_id": "trace-1",
            "agent_mode": "PPO_GLOBAL",
            "selected_action_mask_valid": True,
            "execution": {"exception_class": "TimeoutError"},
        })
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event"] == "reset"
    assert rows[1]["event"] == "step"
    assert rows[1]["action_trace_id"] == "trace-1"
    assert rows[1]["execution"]["exception_class"] == "TimeoutError"


def test_recorder_is_opt_in_at_call_site(tmp_path):
    assert not (tmp_path / "events.jsonl").exists()
