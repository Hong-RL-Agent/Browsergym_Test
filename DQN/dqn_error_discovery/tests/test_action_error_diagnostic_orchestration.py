import json
from pathlib import Path

from dqn_error_discovery.framework_dev.diagnose_action_error_reproduction_orchestration import (
    build_command,
    _read_episode_count,
)


def test_orchestration_command_uses_frozen_runtime_and_unique_diagnostic_outputs(tmp_path: Path):
    command = build_command(
        python=Path(r"C:\workspace\browsergym\BrowserGym\.venv\Scripts\python.exe"),
        site_id="site002",
        base_url="http://localhost:9221/",
        method="hybrid",
        seed=43,
        output_dir=tmp_path,
        episodes=10,
        max_steps=60,
        max_local_steps=5,
        canonical_artifact="canonical.json",
        canonical_error_count=5,
    )
    assert command[0].endswith(r"BrowserGym\.venv\Scripts\python.exe")
    assert "--action-error-diagnostic" in command
    assert "--functional-region-novelty-gate" in command
    assert "--max-steps" in command and command[command.index("--max-steps") + 1] == "60"
    assert str(tmp_path / "site002_hybrid_seed43_events.jsonl") in command
    assert "--ppo-only" not in command


def test_partial_trace_is_not_an_episode_completion(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps({"event": "reset", "episode": 1}) + "\n"
        + json.dumps({"event": "step", "episode": 1, "episode_step": 1}) + "\n",
        encoding="utf-8",
    )
    rows, episodes = _read_episode_count(path)
    assert rows == 2
    assert episodes == 1
