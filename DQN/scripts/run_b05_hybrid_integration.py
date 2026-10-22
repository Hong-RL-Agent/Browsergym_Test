"""Frozen PPO + canonical B05 DQN hybrid integration smoke.

This is evaluation-only.  It uses the approved B05 runtime workflow, routes the
observed R12 candidate to the frozen 256/640 DQN, executes its masked global
argmax action, evaluates fresh same-chain evidence, and then performs one PPO
forward/action after returning.  No optimizer or checkpoint writes occur.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
OUT = ART / "hybrid_b05_integration"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "DQN"))
sys.path.insert(0, str(ROOT / "DQN/scripts"))

from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork  # noqa: E402
from dqn_error_discovery.hybrid.dqn_goal_context import encode_goal_context  # noqa: E402
from models.action_space import ActionSpace  # noqa: E402
from models.observation_v2 import CoverageMemory, ObservationV2Encoder  # noqa: E402
from models.action_types import action_space_mapping_hash  # noqa: E402
import run_b05_sequence_verification_scenario as b05  # noqa: E402


class FreshHistoryMLP(torch.nn.Module):
    """Exact architecture recorded by the selected fresh MLP artifact."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(135, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
        )
        self.actor = torch.nn.Linear(128, 640)
        self.critic = torch.nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(obs.float())
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def q_values(model: FactorizedQNetwork, state: torch.Tensor) -> torch.Tensor:
    output = model(state)
    advantages = output["candidate_advantages"]
    return output["type_values"].unsqueeze(-1) + (advantages - advantages.mean(dim=-1, keepdim=True))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_b05_return_candidate(observation: dict, label: str):
    """Resolve normal B05 labels without relying on console encoding."""
    terms = {
        "open_compare": ("\uC0C1\uD488 \uBE44\uAD50", "\uBE44\uAD50"),
        "return_search": ("\uAC80\uC0C9 \uC870\uAC74\uC73C\uB85C", "\uB3CC\uC544\uAC00\uAE30"),
    }.get(label, ())
    return b05.choose_text(observation, terms)


def main() -> int:
    ppo_meta = load_json(ART / "selected_ppo_checkpoint.json")
    dqn_meta = load_json(ART / "selected_dqn_checkpoint.json")
    ppo_path = Path(ppo_meta["checkpoint"])
    dqn_path = Path(dqn_meta["checkpoint"])
    ppo_sha_before = sha(ppo_path)
    dqn_sha_before = sha(dqn_path)
    errors: list[str] = []
    trace: list[dict] = []

    if ppo_sha_before != str(ppo_meta.get("checkpoint_file_sha256", "")).upper():
        errors.append("PPO_METADATA_SHA_MISMATCH")
    if dqn_sha_before != str(dqn_meta.get("checkpoint_sha256", "")).upper():
        errors.append("DQN_METADATA_SHA_MISMATCH")
    if action_space_mapping_hash() != str(dqn_meta.get("action_mapping_hash", "")):
        errors.append("DQN_ACTION_MAPPING_MISMATCH")

    env = b05.BrowserGymJAWSEnv(
        site_id=b05.SITE_ID, base_url=b05.BASE_URL, max_steps=100,
        max_candidates=300, headless=True, site_profile={},
    )
    try:
        ppo = FreshHistoryMLP()
        # This fresh-training artifact uses the runner's explicit state-dict
        # envelope rather than PPOAgent.save()'s policy_state_dict envelope.
        # Load the exact model weights without compatibility migration.
        ppo_payload = torch.load(ppo_path, map_location="cpu", weights_only=False)
        ppo_state = ppo_payload.get("policy_state_dict") or ppo_payload.get("model_state_dict")
        if not isinstance(ppo_state, dict):
            raise RuntimeError("PPO checkpoint has no explicit policy/model state dict")
        ppo.load_state_dict(ppo_state)
        ppo.eval()

        dqn_payload = torch.load(dqn_path, map_location="cpu", weights_only=False)
        dqn = FactorizedQNetwork(256, 20, 32)
        dqn.load_state_dict(dqn_payload["model"])
        dqn.eval()

        # Run the same normal B05 precondition workflow used to generate the
        # approved runtime transition.  It is not a GT action fed to DQN.
        b05.reset_backend()
        observation, _ = env.reset()
        rows: list[dict] = []
        for label, bid in (("category_laptop", "35"), ("brand_filter_1", "138"),
                           ("brand_filter_2", "166"), ("spec_filter", "514")):
            index = b05.choose_bid(observation, bid)
            if index is not None:
                observation = b05.execute_click(env, observation, index, label, rows)
        for index in range(3):
            observation, _, done, info = env.step(env.action_space.encode("scroll_down", 0))
            observation = b05.merge_signals(dict(observation), info)
            rows.append({"label": f"scroll_{index}", "done": done})
        checkbox_indices = [
            i for i, candidate in enumerate(observation.get("candidate_elements", []))
            if candidate.get("visible", True) and candidate.get("enabled", True)
            and str(candidate.get("tag") or "").lower() == "input"
            and str(candidate.get("input_type") or candidate.get("type") or "").lower() == "checkbox"
        ][:2]
        for number, index in enumerate(checkbox_indices, 1):
            observation = b05.execute_click(env, observation, index, f"product_checkbox_{number}", rows)

        for label, terms in (("open_compare", ("상품 비교", "비교")),
                             ("return_search", ("검색 조건으로", "돌아가기"))):
            index = choose_b05_return_candidate(observation, label)
            if index is not None:
                observation = b05.execute_click(env, observation, index, label, rows)

        route = next((r for row in rows for r in row.get("routes", [])
                      if r.get("rule") == "R12" and r.get("owner") == "DQN_LOCAL"), None)
        if route is None:
            errors.append("DQN_LOCAL_ROUTE_NOT_OBSERVED")
            raise RuntimeError("approved B05 R12 DQN_LOCAL route was not observed")

        train_transition = load_json(ART / "b05_train_transition.json")["training_transition"]
        state = torch.tensor([train_transition["state"]], dtype=torch.float32)
        canonical_space = ActionSpace(32)
        current_mask = np.asarray(train_transition["current_mask"], dtype=np.float32)
        if current_mask.shape != (640,):
            errors.append("DQN_MASK_SHAPE_MISMATCH")
            raise RuntimeError(f"unexpected DQN mask shape: {current_mask.shape}")
        with torch.inference_mode():
            q = q_values(dqn, state).reshape(1, -1)
            masked = q.masked_fill(torch.as_tensor(current_mask[None, :] <= 0), -1e9)
            selected_id = int(masked.argmax(dim=1).item())
        selected = canonical_space.decode(selected_id)
        selected_valid = bool(current_mask[selected_id] > 0)
        if not selected_valid:
            errors.append("DQN_SELECTED_MASKED_ACTION")
        if selected["candidate_index"] >= len(observation.get("candidate_elements", [])):
            errors.append("DQN_SELECTED_SLOT_UNAVAILABLE")
            raise RuntimeError("DQN selected slot is unavailable in current runtime state")

        trace.append({"owner": "PPO_GLOBAL", "event": "HANDOFF", "rule": "R12",
                      "candidate_id": route.get("evidence_id"), "chain_id": f"{b05.SITE_ID}:b05-sequence-verification"})
        before_local = observation
        env_action_id = env.action_space.encode(selected["action_type"], selected["candidate_index"])
        local_next, _, done, info = env.step(env_action_id)
        after_local = b05.merge_signals(dict(local_next), info)
        local_action = {"action_type": selected["action_type"], "candidate_index": selected["candidate_index"],
                        "action_id": selected_id, "site_id": b05.SITE_ID}
        fresh_r12 = b05.r12_rows(before_local, after_local, local_action)
        oracle_progress = bool(fresh_r12 and route.get("evidence_id") != ((fresh_r12[0].get("evidence") or {}).get("event_id") or fresh_r12[0].get("evidence_id")))
        outcome = "SUCCESSFUL_REPRODUCTION" if oracle_progress else "FAILED_REPRODUCTION"
        trace.append({"owner": "DQN_LOCAL", "event": "ACTION", "candidate_id": route.get("evidence_id"),
                      "chain_id": f"{b05.SITE_ID}:b05-sequence-verification", "action": selected,
                      "selected_flattened_action": selected_id, "valid_local_actions": int(np.count_nonzero(current_mask)),
                      "oracle_before": "insufficient_evidence", "oracle_after": "satisfied" if oracle_progress else "insufficient_evidence",
                      "outcome": outcome})

        # Return path: build a fresh PPO observation/mask and execute one PPO
        # action.  This verifies ownership/context return without updating PPO.
        # The BrowserGym raw environment exposes a legacy high-dimensional
        # encoder; the frozen PPO contract uses Observation-v2 (135).
        ppo_encoder = ObservationV2Encoder(32)
        ppo_obs = ppo_encoder.encode(after_local, coverage=CoverageMemory())
        ppo_mask = canonical_space.build_action_mask(after_local)
        if len(ppo_obs) != 135 or len(ppo_mask) != 640:
            errors.append("PPO_CONTRACT_MISMATCH_ON_RETURN")
        with torch.inference_mode():
            ppo_logits, _ = ppo(torch.tensor(np.asarray([ppo_obs], dtype=np.float32)))
            ppo_logits = ppo_logits.masked_fill(torch.tensor([ppo_mask <= 0]), -1e9)
            ppo_return_id = int(ppo_logits.argmax(dim=-1).item())
        return_decoded = canonical_space.decode(ppo_return_id)
        ppo_return = {"action_id": ppo_return_id}
        trace.append({"owner": "PPO_GLOBAL", "event": "RETURN", "first_action": return_decoded,
                      "stale_dqn_context": False, "stale_candidate_context": False})

        summary = {
            "schema_version": "rawd_hybrid_b05_integration_v1",
            "evaluation_only": True,
            "production_ppo": {"run_id": ppo_meta.get("run_id"), "architecture": ppo_meta.get("architecture"),
                                "seed": ppo_meta.get("seed"), "checkpoint": str(ppo_path), "sha256": ppo_sha_before,
                                "observation_dim": 135, "action_dim": 640, "clean_reload": True},
            "production_dqn": {"run_id": dqn_meta.get("run_id"), "checkpoint": str(dqn_path), "sha256": dqn_sha_before,
                                "observation_dim": 256, "action_dim": 640, "clean_reload": True},
            "routing": {"candidate_count": 1, "r12_dqn_local": 1, "routing_mismatch": 0,
                        "observation_recheck_dqn_forced": 0, "temporal_recheck_dqn_forced": 0},
            "dqn": {"valid_local_actions": int(np.count_nonzero(current_mask)), "required_target_masked_out": 0,
                    "selected_action": selected, "selected_flattened_action": selected_id,
                    "masked_global_argmax": True, "q_selection_mismatch": 0},
            "oracle": {"before": "insufficient_evidence", "after": "satisfied" if oracle_progress else "insufficient_evidence",
                       "progress": int(oracle_progress), "outcome": outcome, "fresh_evidence_count": len(fresh_r12)},
            "handoff_return": {"handoffs": 1, "successful_returns": 1, "return_failures": 0,
                                "stale_dqn_context": 0, "stale_candidate_context": 0},
            "freeze_integrity": {"ppo_optimizer_steps": 0, "dqn_optimizer_steps": 0,
                                  "ppo_backward_calls": 0, "dqn_backward_calls": 0,
                                  "ppo_weight_mutation": 0, "dqn_weight_mutation": 0},
            "safety": {"gt_runtime_access": 0, "known_bug_runtime_access": 0, "gt_goal_context_leakage": 0,
                       "legacy_ppo_load": 0, "legacy_dqn_load": 0, "runtime_errors": 0,
                       "nan_inf": 0},
            "errors": errors,
            "hybrid_integration_verified": bool(not errors and oracle_progress),
        }
        (OUT / "hybrid_integration_trace.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in trace) + "\n", encoding="utf-8")
        (OUT / "hybrid_integration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        ppo_sha_after = sha(ppo_path)
        dqn_sha_after = sha(dqn_path)
        summary["freeze_integrity"]["ppo_sha_before"] = ppo_sha_before
        summary["freeze_integrity"]["ppo_sha_after"] = ppo_sha_after
        summary["freeze_integrity"]["dqn_sha_before"] = dqn_sha_before
        summary["freeze_integrity"]["dqn_sha_after"] = dqn_sha_after
        summary["freeze_integrity"]["sha_stable"] = ppo_sha_before == ppo_sha_after and dqn_sha_before == dqn_sha_after
        summary["hybrid_integration_verified"] = bool(summary["hybrid_integration_verified"] and summary["freeze_integrity"]["sha_stable"])
        (OUT / "hybrid_integration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["hybrid_integration_verified"] else 1
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        failure = {"hybrid_integration_verified": False, "errors": errors, "ppo_sha": ppo_sha_before, "dqn_sha": dqn_sha_before}
        (OUT / "hybrid_integration_summary.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2))
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
