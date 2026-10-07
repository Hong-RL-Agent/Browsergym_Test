from pathlib import Path

from dqn_error_discovery.v3 import evaluate_final_v3_posthoc as posthoc
from dqn_error_discovery.v3 import run_final_v3_evaluation as runner


def test_canonical_run_matrix_generation():
    runs = runner.canonical_runs()
    assert len(runs) == 24
    assert runs[0] == {"site": "EVAL001", "method": "ppo", "seed": 42}
    assert runs[5] == {"site": "EVAL001", "method": "hybrid", "seed": 44}
    assert runs[-1] == {"site": "EVAL004", "method": "hybrid", "seed": 44}
    assert {run["site"] for run in runs} == {"EVAL001", "EVAL002", "EVAL003", "EVAL004"}
    assert {run["method"] for run in runs} == {"ppo", "hybrid"}
    assert {run["seed"] for run in runs} == {42, 43, 44}


def test_runner_does_not_reference_gt_catalog():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "heldout_gt_catalog" not in source
    assert "GT_PATH" not in source


def test_protocol_budget_is_fair_and_frozen():
    protocol = runner.json.loads((runner.OUT / "evaluation_protocol.json").read_text(encoding="utf-8"))
    assert protocol["planned_total_episodes"] == 240
    assert protocol["episodes_per_site_method_seed"] == 10
    assert protocol["max_total_env_steps_per_episode"] == 60
    assert protocol["max_local_dqn_steps"] == 5
    assert protocol["fair_budget"]["hybrid_gets_extra_steps"] is False


def test_ppo_only_disables_only_dqn_handoff():
    protocol = runner.json.loads((runner.OUT / "evaluation_protocol.json").read_text(encoding="utf-8"))
    ppo = protocol["methods"]["ppo_only_v3"]
    hybrid = protocol["methods"]["ppo_dqn_hybrid_v3"]
    assert ppo["dqn_handoff_enabled"] is False
    assert hybrid["dqn_handoff_enabled"] is True
    assert ppo["ppo_sha256"] == hybrid["ppo_sha256"]


def test_hybrid_uses_seed_matched_dqn():
    protocol = runner.json.loads((runner.OUT / "evaluation_protocol.json").read_text(encoding="utf-8"))
    checkpoints = protocol["methods"]["ppo_dqn_hybrid_v3"]["dqn_checkpoints"]
    assert set(checkpoints) == {"42", "43", "44"}
    assert checkpoints["42"]["path"].endswith("seed42.pt")
    assert checkpoints["43"]["path"].endswith("seed43.pt")
    assert checkpoints["44"]["path"].endswith("seed44.pt")


def test_freeze_hash_enforcement_passes_current_artifacts():
    result = runner.verify_freeze_hashes()
    assert result["model_hash_precheck"] is True
    assert result["artifact_hash_precheck"] is True
    assert result["site_source_hash_precheck"] is True
    assert result["gt_runtime_access"] is False


def test_posthoc_outcome_classification_states():
    assert posthoc.classify_gt(False, False, False) == "NOT_REACHED"
    assert posthoc.classify_gt(True, False, False) == "REACHED_NOT_DETECTED"
    assert posthoc.classify_gt(True, True, False) == "DETECTED_NOT_CONFIRMED"
    assert posthoc.classify_gt(True, True, True) == "CONFIRMED"


def test_precision_recall_helpers_and_candidate_burden():
    assert posthoc.safe_div(1, 0) == 0.0
    assert posthoc.safe_div(3, 6) == 0.5
    keys = {"R05|/|layout-overlap|target:1", "R14|/api/network|transport"}
    assert posthoc.matched_candidate_keys(keys, {"R05"}) == {"R05|/|layout-overlap|target:1"}


def test_seed_robustness_counts_shape():
    values = posthoc.mean_std([0.0, 0.5, 1.0])
    assert values["mean"] == 0.5
    assert values["std"] > 0.0

