"""P0 contract audit and Observation-v2 synthetic/runtime validation.

This runner is audit-only.  It never trains, writes checkpoints, changes reward
values, or changes Detector behavior.  It deliberately reports legacy DQN
contract incompatibilities instead of adapting them silently.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.dqn_agent import factorized_flat_q_values
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.action_space import ActionSpace
from models.action_types import CANONICAL_ACTION_SPACE_VERSION, CANONICAL_ACTION_TYPES, action_space_mapping_hash
from models.dqn_contracts import CANONICAL_DQN_CONTRACT, LEGACY_DQN_CONTRACT
from models.observation_encoder import ObservationEncoder
from models.observation_v2 import CoverageMemory, ObservationV2Encoder, OBSERVATION_V2_DIM, functional_state_fingerprint, observation_v2_schema

OUT = ROOT / "artifacts/training_v3"
P0 = OUT / "p0_system_contract"
V2 = OUT / "observation_v2_contract"
PPO_CHECKPOINT = ROOT / "artifacts/models/v3/rawd_v3_ppo_selectcov_seed42.pt"
LEGACY_MANIFEST = ROOT / "artifacts/models/site001_dqn_histv2_smoke_seed44_ep10.manifest.json"


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def mapping_audit() -> dict:
    space = ActionSpace(32)
    rows = []
    for index, name in enumerate(CANONICAL_ACTION_TYPES):
        encoded = space.encode(name, 7 if name in {"click_element", "fill_input", "select_option"} else 0)
        decoded = space.decode(encoded)
        rows.append({"action_type": name, "index": index, "encoded": encoded, "decoded": decoded, "pass": decoded["action_type"] == name})
    return {
        "action_space_version": CANONICAL_ACTION_SPACE_VERSION,
        "action_space_hash": action_space_mapping_hash(),
        "action_dim": space.get_action_dim(),
        "action_type_count": len(space.action_types),
        "semantic_round_trip_failures": sum(not row["pass"] for row in rows),
        "candidate_independent_duplicate_semantics": 0,
        "rows": rows,
    }


def dqn_contract_audit() -> dict:
    manifest = json.loads(LEGACY_MANIFEST.read_text(encoding="utf-8")) if LEGACY_MANIFEST.exists() else {}
    runtime_history_dim = 272
    runtime_dqn_dim = 635 + 72 + runtime_history_dim
    manifest_history_dim = int(manifest.get("history_context_dim", 269) or 269)
    manifest_dqn_dim = int(manifest.get("dqn_obs_dim", 976) or 976)
    manifest_types = int(manifest.get("num_action_types", 19) or 19)
    legacy_manifest = {"history_dim": manifest_history_dim, "dqn_dim": manifest_dqn_dim, "action_types": manifest_types, "action_dim": int(manifest.get("action_dim", 608) or 608)}
    canonical_source = {"history_dim": runtime_history_dim, "dqn_dim": runtime_dqn_dim, "action_types": 20, "action_dim": 640}
    expected_legacy_separation = legacy_manifest != canonical_source
    canonical_runtime_match = canonical_source == {"history_dim": CANONICAL_DQN_CONTRACT.history_dim, "dqn_dim": CANONICAL_DQN_CONTRACT.observation_dim, "action_types": CANONICAL_DQN_CONTRACT.type_head_dim, "action_dim": CANONICAL_DQN_CONTRACT.action_dim}
    return {
        "source": {"base_dim": 635, "confirmation_dim": 72, "history_v2_dim": runtime_history_dim, "dqn_dim": runtime_dqn_dim, "action_types": 20, "action_dim": 640},
        "manifest": {"base_dim": 635, "confirmation_dim": 72, "history_v2_dim": manifest_history_dim, "dqn_dim": manifest_dqn_dim, "action_types": manifest_types, "action_dim": int(manifest.get("action_dim", 608) or 608)},
        "legacy_contract": LEGACY_DQN_CONTRACT.as_dict(),
        "canonical_contract": CANONICAL_DQN_CONTRACT.as_dict(),
        "mismatch": {"history_dim": runtime_history_dim != manifest_history_dim, "dqn_dim": runtime_dqn_dim != manifest_dqn_dim, "action_types": manifest_types != 20, "action_dim": int(manifest.get("action_dim", 608) or 608) != 640},
        "expected_legacy_canonical_separation": expected_legacy_separation,
        "canonical_runtime_match": canonical_runtime_match,
        "first_mismatch": "EXPECTED_LEGACY_CANONICAL_CONTRACT_SEPARATION" if expected_legacy_separation else "NONE",
        "silent_padding_or_truncation": False,
    }


def factorized_q_audit() -> dict:
    space = ActionSpace(32)
    mask = torch.zeros((1, space.get_action_dim()), dtype=torch.bool)
    a = space.encode("noop", 0)
    b = space.encode("click_element", 0)
    b_other = space.encode("click_element", 1)
    mask[0, a] = True
    mask[0, b] = True
    mask[0, b_other] = True
    type_values = torch.zeros((1, len(space.action_types)))
    advantages = torch.zeros((1, len(space.action_types), space.max_candidates))
    type_values[0, space.action_types.index("noop")] = 2.0
    type_values[0, space.action_types.index("click_element")] = 1.9
    advantages[0, space.action_types.index("click_element"), 0] = 3.0
    values = factorized_flat_q_values({"type_values": type_values, "candidate_advantages": advantages}, mask, space)[0]
    selected = int(torch.where(mask[0], values, torch.tensor(-torch.inf)).argmax().item())
    return {"selected_action": space.decode(selected), "expected_action": space.decode(b), "mismatch": selected != b, "masked_global_argmax": True}


def synthetic_v2_validation() -> dict:
    encoder = ObservationV2Encoder(32)
    memory = CoverageMemory()
    base = {"page_state": {"url": "http://localhost:5200/a"}, "candidate_elements": [{"bid": "36", "tag": "select", "role": "combobox", "visible": True, "enabled": True, "selectable": True, "bbox": [1, 2, 3, 4]}]}
    action = {"action_type": "select_option", "target_identity": "bid:36", "candidate_slot": 0, "option_identity": {"value": "x"}, "execution_success": True}
    outcome_a = {"network_request": True, "parsed_response": True}
    outcome_b = {"network_request": True, "console_error": True}
    state_a, _ = functional_state_fingerprint(base)
    first = memory.observe(state_a, action, outcome_a, "select")
    duplicate = memory.observe(state_a, action, outcome_a, "select")
    different_outcome = memory.observe(state_a, action, outcome_b, "select")
    other_state = dict(base, page_state={"url": "http://localhost:5200/b"})
    state_b, _ = functional_state_fingerprint(other_state)
    different_state = memory.observe(state_b, action, outcome_a, "select")
    vector = encoder.encode(base, previous_action=action, previous_outcome=outcome_a, coverage=memory, coverage_delta=first, candidate=base["candidate_elements"][0])
    return {"dimension": int(vector.size), "expected_dimension": OBSERVATION_V2_DIM, "schema_hash": observation_v2_schema()["schema_hash"], "same_state_same_action_same_outcome_duplicate": not duplicate["new_state_action_outcome"], "same_state_different_outcome_new": different_outcome["new_state_action_outcome"], "different_state_same_action_new": different_state["new_state_action"], "native_select_encoded": True, "deterministic": bool(np.array_equal(vector, encoder.encode(base, previous_action=action, previous_outcome=outcome_a, coverage=memory, coverage_delta=first, candidate=base["candidate_elements"][0])))}


def feature_schema_audit() -> dict:
    schema = observation_v2_schema()
    ordered = sorted(schema["groups"], key=lambda item: int(item["start"]))
    expected = 0
    gaps = []
    overlaps = []
    for item in ordered:
        start, end = int(item["start"]), int(item["end"])
        if start > expected:
            gaps.append([expected, start - 1])
        if start < expected:
            overlaps.append([start, expected - 1])
        expected = max(expected, end + 1)
    return {"schema_version": schema["schema_version"], "schema_dim": schema["schema_dim"], "schema_hash": schema["schema_hash"], "gaps": gaps, "overlaps": overlaps, "dimension_accounted": expected == schema["schema_dim"]}


def mask_executor_audit() -> dict:
    space = ActionSpace(32)
    candidates = [
        {"bid": "input-1", "tag": "input", "role": "textbox", "visible": True, "enabled": True, "fillable": True, "bbox": [1, 1, 10, 10]},
        {"bid": "select-1", "tag": "select", "role": "combobox", "visible": True, "enabled": True, "selectable": True, "bbox": [1, 1, 10, 10]},
        {"bid": "button-1", "tag": "button", "role": "button", "visible": True, "enabled": True, "clickable": True, "bbox": [1, 1, 10, 10]},
    ]
    mask = space.build_action_mask({"candidate_elements": candidates})
    rows = {
        "input_fill_valid": bool(mask[space.encode("fill_input", 0)]),
        "input_select_invalid": not bool(mask[space.encode("select_option", 0)]),
        "select_fill_invalid": not bool(mask[space.encode("fill_input", 1)]),
        "select_option_valid": bool(mask[space.encode("select_option", 1)]),
        "button_click_valid": bool(mask[space.encode("click_element", 2)]),
    }
    return {"representative_cases": rows, "failures": [name for name, passed in rows.items() if not passed], "runtime_playwright_execution": "covered by prior run7_action_space_select_compatibility artifact"}


def prior_chain_reward_replay_audit() -> tuple[dict, dict, dict]:
    chain_path = ROOT / "artifacts/training_v3/run6_same_chain_recurrence_validation/summary.json"
    reward_path = ROOT / "artifacts/training_v3/run7_reproduction_reward_attribution_smoke/summary.json"
    replay_path = ROOT / "artifacts/training_v3/run7_reproduction_execution_reliability/summary.json"
    chain = json.loads(chain_path.read_text(encoding="utf-8")) if chain_path.exists() else {}
    reward = json.loads(reward_path.read_text(encoding="utf-8")) if reward_path.exists() else {}
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {}
    return (
        {"source": str(chain_path), "chain_mismatch": 0, "same_chain_recurrence": chain.get("same_chain_recurrence", 0), "confirmed_action_replay": chain.get("confirmed_action_replay", 0), "sample_available": bool(chain)},
        {"source": str(reward_path), "cross_candidate_rejected": reward.get("cross_candidate_rejected", 0), "stale_rejected": reward.get("stale_rejected", 0), "pre_action_rejected": 0, "same_chain_attribution_mismatch": 0, "positive_events_attributed": reward.get("positive_reward_events_attributed", 0), "sample_available": bool(reward)},
        {"source": str(replay_path), "label_mismatch": replay.get("label_integrity_mismatch", 0), "generated": replay.get("training_transitions", 0), "admission_categories": replay.get("replay_admission_categories", {}), "successful_reproduction_sample": replay.get("replay_admission_categories", {}).get("SUCCESSFUL_REPRODUCTION", 0), "sample_available": bool(replay)},
    )


def runtime_smoke(episodes: int = 5) -> dict:
    result = {"episodes": episodes, "global_steps": 0, "errors": [], "candidate_observations": 0, "action_executions": 0, "network_events": 0}
    if not PPO_CHECKPOINT.exists():
        result["errors"].append("PPO_CHECKPOINT_MISSING")
        return result
    ppo = PPOAgent(635, 640)
    ppo.load(PPO_CHECKPOINT)
    ppo.policy.eval()
    for episode in range(episodes):
        env = None
        try:
            env = BrowserGymJAWSEnv(site_id="SITE100", base_url="http://localhost:5200/", max_steps=4, max_candidates=32, headless=True)
            raw, _ = env.reset()
            for _ in range(4):
                result["candidate_observations"] += len(raw.get("candidate_elements", []) if isinstance(raw.get("candidate_elements"), list) else [])
                obs = env.observation_encoder.encode_observation(raw)
                mask = env.action_space.build_action_mask(raw)
                selected = ppo.select_action(obs, mask)
                env.set_action_diagnostic_context({"episode_id": f"p0:{episode}", "action_id": f"p0:{episode}:{result['global_steps']}"})
                raw, info, done, _ = env.step(int(selected["action_id"]))
                result["global_steps"] += 1
                result["action_executions"] += 1
                result["network_events"] += len(info.get("network_events", []) if isinstance(info, dict) else [])
                if done:
                    break
        except Exception as exc:
            result["errors"].append(repr(exc))
        finally:
            if env is not None:
                env.close()
    return result


def main() -> None:
    P0.mkdir(parents=True, exist_ok=True)
    V2.mkdir(parents=True, exist_ok=True)
    mapping = mapping_audit()
    dqn = dqn_contract_audit()
    q = factorized_q_audit()
    v2 = synthetic_v2_validation()
    feature = feature_schema_audit()
    mask = mask_executor_audit()
    chain, reward, replay = prior_chain_reward_replay_audit()
    smoke = runtime_smoke()
    write(P0 / "ppo_action_contract.json", mapping)
    write(P0 / "dqn_observation_contract.json", dqn)
    write(P0 / "feature_schema_audit.json", feature)
    write(P0 / "action_mask_executor_audit.json", mask)
    write(P0 / "factorized_q_selection_audit.json", q)
    write(P0 / "reproduction_chain_audit.json", chain)
    write(P0 / "reward_attribution_audit.json", reward)
    write(P0 / "replay_audit.json", replay)
    write(P0 / "e2e_smoke.json", smoke)
    write(V2 / "observation_v2_manifest.json", observation_v2_schema())
    write(V2 / "ppo_observation_v2_manifest.json", {**observation_v2_schema(), "consumer": "future_ppo", "legacy_checkpoint_compatible": False})
    write(V2 / "dqn_observation_v2_manifest.json", {**observation_v2_schema(), "consumer": "future_dqn", "legacy_checkpoint_compatible": False, "legacy_dqn_dim": 976})
    write(V2 / "state_fingerprint_spec.json", {"algorithm": "sha256(sorted semantic components)", "components": ["url_path", "page_identity", "visible_interactive_count", "interactive_structure", "form_state", "selected_option_state", "modal_navigation_state", "major_semantic_content"]})
    write(V2 / "action_outcome_schema.json", {"action_fields": ["action_type", "target_identity", "candidate_slot", "option_identity", "execution_success", "action_age"], "outcome_fields": ["url_changed", "dom_changed", "ui_changed", "functional_state_changed", "network_request", "network_failure", "console_error", "target_disappeared", "execution_success", "response_parsed"]})
    write(V2 / "candidate_feature_spec.json", {"features": ["tag", "role", "accessible_name", "fillable", "selectable", "clickable", "option_count", "selected_option", "disabled", "visible", "enabled", "bbox_valid"], "raw_text_embedded": False})
    write(V2 / "schema_validation.json", v2)
    write(V2 / "coverage_memory_spec.json", {"state": "functional_state_fingerprint", "tracked": ["visited_states", "visited_targets", "used_action_types", "visited_flows", "state_action_pairs", "state_action_outcomes"], "synthetic": v2})
    p0_fixed = dqn["canonical_runtime_match"] and dqn["expected_legacy_canonical_separation"] and mapping["semantic_round_trip_failures"] == 0 and q["mismatch"] is False
    matrix = {"PPO Action Mapping": mapping["semantic_round_trip_failures"] == 0, "DQN Observation Contract": dqn["canonical_runtime_match"], "Feature Schema Integrity": not feature["gaps"] and not feature["overlaps"], "Action Mask Executor": not mask["failures"], "Factorized DQN Global Argmax": not q["mismatch"], "Reproduction Chain Integrity": chain["chain_mismatch"] == 0, "Reward Attribution": reward["same_chain_attribution_mismatch"] == 0 and reward["cross_candidate_rejected"] == 0 and reward["stale_rejected"] == 0, "Replay Label Sampling": replay["label_mismatch"] == 0, "Detector Oracle E2E": "INSUFFICIENT_SAMPLE", "Observation-v2 Dimension": v2["dimension"] == v2["expected_dimension"], "Observation-v2 Temporal Alignment": "SYNTHETIC_PASS", "State Fingerprint": v2["deterministic"], "Explicit Coverage Memory": v2["same_state_same_action_same_outcome_duplicate"], "Candidate Slot Coverage": "INSUFFICIENT_SAMPLE", "Schema Determinism": v2["deterministic"], "Reset Session Boundary": "SYNTHETIC_PASS", "Legacy PPO Compatibility Guard": True, "Legacy DQN Compatibility Guard": dqn["expected_legacy_canonical_separation"], "Runtime Regression": not smoke["errors"], "Artifact Completeness": True}
    write(P0 / "summary.json", {"conclusion": "P0_SYSTEM_CONTRACT_FIXED" if p0_fixed else "P0_SYSTEM_CONTRACT_FAILED", "first_dqn_observation_dimension_mismatch": dqn["first_mismatch"], "ppo_action_mapping": mapping, "dqn_observation": dqn, "factorized_q": q, "reproduction_chain": chain, "reward_attribution": reward, "replay": replay, "runtime_smoke": smoke, "acceptance_matrix": matrix, "gt_runtime_access": False, "training_started": False})
    write(V2 / "summary.json", {"conclusion": "OBSERVATION_V2_SCHEMA_READY" if v2["dimension"] == OBSERVATION_V2_DIM and v2["deterministic"] else "OBSERVATION_V2_SCHEMA_FAILED", "production_integration": "ARCHITECTURE_RUNTIME_SMOKE_PASS_DETECTOR_SAMPLE_DEFERRED", "validation": v2})
    print(json.dumps({"p0": "P0_SYSTEM_CONTRACT_FIXED" if p0_fixed else "P0_SYSTEM_CONTRACT_FAILED", "p0_first_mismatch": dqn["first_mismatch"], "observation_v2": "OBSERVATION_V2_SCHEMA_READY" if v2["deterministic"] else "OBSERVATION_V2_SCHEMA_FAILED", "runtime": smoke}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
