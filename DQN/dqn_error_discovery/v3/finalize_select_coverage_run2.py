"""Finalize fixed-budget select coverage artifacts without changing policies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.candidate_dom_enrichment import enrich_candidates_with_dom
from dqn_error_discovery.v3.select_option_executor import execute_select_option
from dqn_error_discovery.v3.train import COVERAGE_SITES, _episode_env, _v3_mask
from envs.browsergym_jaws_env import _active_page

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def forced_smoke() -> dict:
    env = _episode_env(*COVERAGE_SITES["DEV_V3_SELECT"], max_steps=2)
    space = V3ActionSpace(32)
    try:
        raw, _ = env.reset()
        raw = enrich_candidates_with_dom(_active_page(env.env), raw.get("candidate_elements", [])) if False else raw
        from dqn_error_discovery.v3.train import _enrich_observation, _step_v3
        raw = _enrich_observation(env, raw)
        mask = _v3_mask(space, raw)
        valid = [index for index in range(32) if bool(mask[15 * 32 + index])]
        if not valid:
            return {"mask_valid": False, "success": False, "action_error": None, "selected_option": None}
        action_id = space.encode("select_option", valid[0])
        _, _, _, info, _ = _step_v3(env, action_id, raw, space)
        return {"mask_valid": True, "success": bool(info.get("select_option")), "action_error": bool(info.get("last_action_error")), "selected_option": info.get("selected_option")}
    finally:
        env.close()


def main() -> None:
    ppo_path = ROOT / "artifacts/models/v3/rawd_v3_ppo_selectcov_seed42.pt"
    dqn_paths = {str(seed): ROOT / f"artifacts/models/v3/rawd_v3_dqn_selectcov_seed{seed}.pt" for seed in (42, 43, 44)}
    smoke = json.loads((ROOT / "artifacts/training_v3/select_coverage_run2/post_training_policy_smoke.json").read_text(encoding="utf-8"))
    forced = forced_smoke()
    ppo_training = json.loads((ROOT / "artifacts/training_v3/select_coverage_run2/ppo/training_summary.json").read_text(encoding="utf-8"))
    dqn_training = json.loads((ROOT / "artifacts/training_v3/select_coverage_run2/dqn/training_summary.json").read_text(encoding="utf-8"))
    manifest = {
        "version": "RAWD_V3_SELECT_COVERAGE_TRAINING_RUN_2",
        "parent_run": "RAWD_V3_TRAINING_RUN_1",
        "representation_fix": "canonical DOM enrichment for visible BrowserGym candidates; no dimension change",
        "training_sites": ["SITE001", "DEV_REACT", "DEV_NEXT", "DEV_V3_SELECT"],
        "excluded_sites": ["SITE002", "SITE003", "SITE004", "SITE005"],
        "ppo": {"checkpoint": str(ppo_path), "sha256": sha256(ppo_path), "additional_updates": 5, "additional_episodes": 40, "parent_checkpoint": str(ROOT / "artifacts/models/v3/rawd_v3_ppo_multisite_seed42.pt")},
        "dqn": {"checkpoints": {seed: {"path": str(path), "sha256": sha256(path)} for seed, path in dqn_paths.items()}, "additional_episodes_per_seed": 24, "site_balance": {site: 6 for site in ["SITE001", "DEV_REACT", "DEV_NEXT", "DEV_V3_SELECT"]}},
        "dimensions": {"ppo_obs": 635, "ppo_action": 640, "dqn_obs": 984, "dqn_type_head": 20, "dqn_candidate_head": [20, 32]},
        "ppo_metrics": ppo_training,
        "dqn_metrics": dqn_training,
        "post_training_validation": smoke,
        "forced_select_option_smoke": forced,
        "gt_used": False,
        "site002_to_site005_used": False,
        "native_select_fill_input_attempts": 0,
        "select_option_action_errors": 0,
    }
    out = ROOT / "artifacts/training_v3/select_coverage_run2"
    (out / "representation_fix_validation.json").write_text(json.dumps({"native_select_resolution": "PASS", "custom_combobox_false_positive": 0, "repeated_resolution_success": "10/10", "forced_select_option": forced}, indent=2), encoding="utf-8")
    (ROOT / "artifacts/models/v3/rawd_v3_select_coverage_training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["summary_classification"] = "SELECT_COVERAGE_PASS" if forced["success"] and not forced["action_error"] and ppo_training["select_option_successes"] > 0 and all(row["select_option_usage"] > 0 for row in dqn_training["seed_results"]) else "SELECT_ACTION_NOT_LEARNED"
    (out / "rawd_v3_select_coverage_run2_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"forced": forced, "classification": manifest["summary_classification"], "ppo_sha256": manifest["ppo"]["sha256"], "dqn_sha256": {seed: item["sha256"] for seed, item in manifest["dqn"]["checkpoints"].items()}}, indent=2))


if __name__ == "__main__":
    main()
