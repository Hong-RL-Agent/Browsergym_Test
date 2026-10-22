"""Run approved normal workflows with runtime-adaptive selectors.

Diagnostic-only: no GT targets are injected and DQN routing is never forced.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DQN"))
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from models.observation_v2 import ObservationV2Encoder, CoverageMemory
from models.action_space import ActionSpace
from services.anomaly_detection_service import detect_anomalies
from dqn_error_discovery.agents.confirmation_reproduction import derive_reproduction_context
from dqn_error_discovery.hybrid.strategy_router import route_candidate
from dqn_error_discovery.hybrid.dqn_goal_context import encode_goal_context
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork
from models.recurrent_ppo import ObservationV2HistoryMLP

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"
OUT = BASE / "approved_multisite_adapter_probe.json"

ADAPTERS = {
    "B01_SNS": {"url": "http://127.0.0.1:5316", "actions": [("click", ["검색"]), ("fill", ["검색"]), ("click", ["검색"]), ("click", []), ("click", ["댓글"]), ("fill", ["댓글"]), ("click", [])]},
    "B02_중고거래": {"url": "http://127.0.0.1:5317", "actions": [("click", ["검색"]), ("fill", ["검색"]), ("click", ["검색"]), ("click", []), ("click", ["찜", "예약"]), ("click", ["예약"])]},
    "B03_병원예약": {"url": "http://127.0.0.1:5318", "actions": [("click", ["내 예약", "병원", "예약"]), ("click", []), ("click", ["진료", "시간", "예약"]), ("click", []), ("scroll", [])]},
    "B04_영화예매": {"url": "http://127.0.0.1:5319", "actions": [("click", ["검색"]), ("fill", ["검색", "영화"]), ("click", ["검색"]), ("click", ["상영", "영화"]), ("click", []), ("click", ["예매", "선택"])]},
    "B05_상품검색비교": {"url": "http://127.0.0.1:5320", "actions": [("click", ["노트북", "laptop"]), ("click", ["가격", "필터"]), ("click", ["비교", "compare"]), ("click", []), ("scroll", []), ("click", ["검색 조건"])]},
    "B06_회원가입계정": {"url": "http://127.0.0.1:5321", "actions": [("click", ["가입 시작", "회원가입"]), ("fill", ["이메일"]), ("fill", ["계정 ID"]), ("fill", ["이름"]), ("click", ["저장 후 인증", "다음"]), ("fill", ["인증"]), ("click", ["인증 확인", "다음"]), ("click", ["가입 완료"])]},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def q_values(model, state: torch.Tensor) -> torch.Tensor:
    out = model(state)
    adv = out["candidate_advantages"]
    return out["type_values"].unsqueeze(-1) + (adv - adv.mean(dim=-1, keepdim=True))


def candidate_text(candidate: dict) -> str:
    return " ".join(str(candidate.get(k) or "") for k in ("text", "name", "aria_label", "label")).strip()


def state_signature(observation: dict) -> str:
    """Stable diagnostic signature; excludes GT and defect identifiers."""
    candidates = observation.get("candidate_elements", []) if isinstance(observation, dict) else []
    rows = []
    for candidate in candidates[:32]:
        if not isinstance(candidate, dict):
            continue
        rows.append((str(candidate.get("bid") or ""), candidate_text(candidate),
                     str(candidate.get("role") or ""), bool(candidate.get("visible", True)),
                     bool(candidate.get("enabled", True))))
    signals = observation.get("runtime_signals", {}) if isinstance(observation, dict) else {}
    return hashlib.sha256(json.dumps({"candidates": rows, "signals": signals}, sort_keys=True, default=str).encode()).hexdigest()


def find_runtime_candidate(candidates: list[dict], terms: list[str], used: set[int], kind: str = "click") -> int | None:
    """Prefer semantic labels; otherwise choose a current normal UI candidate."""
    eligible = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        if index in used or not candidate.get("visible", True) or not candidate.get("enabled", True):
            continue
        if kind == "click" and not candidate.get("clickable", False):
            continue
        if kind == "fill" and not (candidate.get("fillable", False) or str(candidate.get("role") or "").lower() in {"textbox", "searchbox"} or str(candidate.get("tag") or "").lower() == "input" or str(candidate.get("input_type") or candidate.get("type") or "").lower() in {"text", "email", "password", "search", "tel"}):
            continue
        if kind == "select" and not (candidate.get("selectable", False) or str(candidate.get("tag") or "").lower() == "select"):
            continue
        bbox = candidate.get("bbox") or {}
        if isinstance(bbox, dict) and (float(bbox.get("width", 1) or 0) <= 0 or float(bbox.get("height", 1) or 0) <= 0):
            continue
        text = candidate_text(candidate)
        lowered = text.lower()
        if any(token in lowered for token in ("reset", "초기화", "logout", "로그아웃", "delete all", "모두 삭제")):
            continue
        eligible.append((index, candidate, text))
    if terms:
        for index, _, text in eligible:
            if any(str(term).lower() in text.lower() for term in terms):
                return index
    if not eligible:
        return None

    def score(item):
        index, candidate, text = item
        role = str(candidate.get("role") or "").lower()
        tag = str(candidate.get("tag") or "").lower()
        return (3 if role in {"button", "textbox", "combobox"} or tag in {"button", "input", "select"} else 1,
                0 if role == "link" or tag == "a" else 1, -len(text), -index)

    return max(eligible, key=score)[0]


def main() -> int:
    pmeta = json.loads((BASE / "selected_ppo_checkpoint.json").read_text(encoding="utf-8"))
    dmeta = json.loads((BASE / "selected_dqn_checkpoint.json").read_text(encoding="utf-8"))
    ppath, dpath = Path(pmeta["checkpoint"]), Path(dmeta["checkpoint"])
    psha_before, dsha_before = sha(ppath), sha(dpath)
    ppo = ObservationV2HistoryMLP(135, 640)
    ppo.load_state_dict(torch.load(ppath, map_location="cpu", weights_only=False)["model_state_dict"])
    ppo.eval()
    dqn = FactorizedQNetwork(256, 20, 32)
    dqn.load_state_dict(torch.load(dpath, map_location="cpu", weights_only=False)["model"])
    dqn.eval()
    encoder, action_space = ObservationV2Encoder(32), ActionSpace(32)
    rows, errors = [], []
    totals = {"steps": 0, "adapter_actions": 0, "candidates": 0, "dqn_local": 0, "dqn_executed": 0,
              "valid_selected": 0, "post_action_fresh_evidence": 0, "partial": 0, "successful": 0, "failed": 0}

    for site_id, spec in ADAPTERS.items():
        env = BrowserGymJAWSEnv(site_id=site_id, base_url=spec["url"], max_steps=50,
                                max_candidates=32, headless=True, site_profile={})
        raw, _ = env.reset()
        memory, previous_action, previous_outcome, used = CoverageMemory(), {}, {}, set()
        row = {"site_id": site_id, "steps": 0, "adapter_actions": 0, "candidates": 0,
               "dqn_local": 0, "dqn_executed": 0, "valid_selected": 0,
               "post_action_fresh_evidence": 0, "partial": 0, "successful": 0,
               "failed": 0, "unmatched": [], "chains": [], "transition_trace": [], "route_counts": {}}
        try:
            for kind, terms in spec["actions"]:
                candidates = raw.get("candidate_elements", []) if isinstance(raw, dict) else []
                index = find_runtime_candidate(candidates, terms, used, kind) if kind in {"click", "fill", "select"} else 0
                if kind in {"click", "fill", "select"} and index is None:
                    row["unmatched"].append(terms)
                    continue
                if kind in {"click", "fill", "select"}:
                    used.add(index)
                action_type = {"click": "click_element", "fill": "fill_input", "select": "select_option"}.get(kind, "scroll_down")
                action_id = env.action_space.encode(action_type, index)
                before = raw
                before_signature = state_signature(before)
                after, _, _, info = env.step(action_id)
                after = dict(after)
                signals = dict(after.get("runtime_signals") or {})
                if isinstance(info, dict):
                    signals.update(info)
                after["runtime_signals"] = signals
                after_signature = state_signature(after)
                row["steps"] += 1
                row["adapter_actions"] += 1
                totals["steps"] += 1
                totals["adapter_actions"] += 1
                target = str(candidates[index].get("bid") or "") if kind == "click" and index < len(candidates) else ""
                action = {"action_type": action_type, "candidate_index": index, "action_id": action_id,
                          "target_identity": target, "execution_success": True}
                anomalies = detect_anomalies(before, after, {"action": action, "action_type": action_type,
                    "trigger_action_type": action_type, "trigger_target_identity": target,
                    "site_id": site_id, "last_action_error": False}, site_profile={})
                row["candidates"] += len(anomalies)
                totals["candidates"] += len(anomalies)
                row["transition_trace"].append({"step": row["steps"], "action_type": action_type,
                    "target": target, "before_state_signature": before_signature,
                    "after_state_signature": after_signature, "state_changed": before_signature != after_signature,
                    "candidate_count_before": len(candidates),
                    "candidate_count_after": len(after.get("candidate_elements", [])),
                    "candidate_changed": len(candidates) != len(after.get("candidate_elements", [])),
                    "fresh_anomaly_count": len(anomalies)})
                for anomaly_index, anomaly in enumerate(anomalies):
                    anomaly.setdefault("trigger_action_type", action_type)
                    anomaly.setdefault("trigger_target_identity", target)
                    context = derive_reproduction_context(anomaly).as_dict()
                    route = route_candidate({
                        "reproduction_strategy": context.get("reproduction_strategy"),
                        "requires_action_reproduction": bool(context.get("requires_action_reproduction")),
                        "reproduction_action_type": context.get("reproduction_action_type"),
                        "reproduction_target_identity": context.get("reproduction_target_identity"),
                        "target_actionable": bool(context.get("requires_action_reproduction") and context.get("reproduction_target_identity")),
                        "exact_action_known": False,
                        "chain_id": f"{site_id}:{row['steps']}:{anomaly_index}",
                    })
                    if not route.dqn_eligible:
                        row["route_counts"][str(route.owner)] = row["route_counts"].get(str(route.owner), 0) + 1
                        continue
                    totals["dqn_local"] += 1
                    row["dqn_local"] += 1
                    row["route_counts"][str(route.owner)] = row["route_counts"].get(str(route.owner), 0) + 1
                    observation = encoder.encode(before, previous_action=previous_action,
                                                 previous_outcome=previous_outcome, coverage=memory)
                    goal = encode_goal_context({"active_candidate": True, "strategy": str(route.strategy),
                        "candidate_family": str(anomaly.get("rule_id") or "unknown"),
                        "oracle_family": "RUNTIME_EVIDENCE_ORACLE", "confirmation_progress": 0.0,
                        "same_chain_progress": 0.0, "remaining_local_budget": 1.0,
                        "prior_execution_success": 1.0})
                    state = np.concatenate([observation, goal, np.zeros(62, dtype=np.float32)]).astype(np.float32)
                    mask = action_space.build_action_mask(before).astype(bool)
                    with torch.inference_mode():
                        values = q_values(dqn, torch.from_numpy(state).reshape(1, -1)).reshape(1, -1)
                        selected = int(values.masked_fill(~torch.from_numpy(mask).reshape(1, -1), -1e9).argmax().item())
                    decoded = action_space.decode(selected)
                    valid = bool(mask[selected])
                    row["dqn_executed"] += 1
                    row["valid_selected"] += int(valid)
                    totals["dqn_executed"] += 1
                    totals["valid_selected"] += int(valid)
                    fresh = []
                    if valid and decoded["candidate_index"] < len(candidates):
                        local_id = env.action_space.encode(decoded["action_type"], decoded["candidate_index"])
                        local_after, _, _, _ = env.step(local_id)
                        local_after = dict(local_after)
                        fresh = detect_anomalies(after, local_after, {"action_type": decoded["action_type"],
                            "trigger_action_type": decoded["action_type"], "trigger_target_identity": target,
                            "site_id": site_id}, site_profile={})
                        raw = local_after
                    else:
                        raw = after
                    outcome = "SUCCESSFUL_REPRODUCTION" if fresh else "FAILED_REPRODUCTION"
                    row["post_action_fresh_evidence"] += int(bool(fresh))
                    totals["post_action_fresh_evidence"] += int(bool(fresh))
                    row["successful"] += int(outcome == "SUCCESSFUL_REPRODUCTION")
                    row["failed"] += int(outcome == "FAILED_REPRODUCTION")
                    totals["successful"] += int(outcome == "SUCCESSFUL_REPRODUCTION")
                    totals["failed"] += int(outcome == "FAILED_REPRODUCTION")
                    row["chains"].append({"rule": anomaly.get("rule_id"), "strategy": str(route.strategy),
                        "valid_local_actions": int(mask.sum()), "selected_action": decoded,
                        "selected_flattened_action": selected, "post_action_fresh_evidence": len(fresh),
                        "outcome": outcome, "runtime": True, "synthetic": False})
                raw = after
                previous_action = {"action_type": action_type, "candidate_slot": index, "execution_success": True}
                previous_outcome = {"execution_success": True}
        except Exception as exc:
            row["error"] = repr(exc)
            errors.append({"site_id": site_id, "error": repr(exc)})
        finally:
            env.close()
        rows.append(row)

    psha_after, dsha_after = sha(ppath), sha(dpath)
    result = {"diagnostic_only": True, "selector_version": "runtime_candidate_adaptive_v2",
              "site_rows": rows, "counts": totals, "runtime_errors": errors,
              "ppo_sha_before": psha_before, "ppo_sha_after": psha_after,
              "dqn_sha_before": dsha_before, "dqn_sha_after": dsha_after,
              "sha_stable": psha_before == psha_after and dsha_before == dsha_after,
              "optimizer_steps": 0, "gt_leakage": 0, "test_access": 0,
              "dqn_readiness": totals["dqn_local"] > 0 and totals["valid_selected"] == totals["dqn_executed"]
              and totals["successful"] + totals["partial"] > 0}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
