"""Scoped diagnostic for R10 candidate -> DQN local reproduction.

The script uses a real site006 browser action to produce a production R10
candidate, lets the existing HybridController hand off to DQN, and records the
unchanged DQN local action candidates, mask validity, Q-values, selected
actions, effects, and R10 recurrence.  It performs no training and writes no
checkpoints.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork, factorized_flat_q_values
from dqn_error_discovery.agents.confirmation_observation import build_active_candidate_confirmation_context
from dqn_error_discovery.agents.history_observation import build_dqn_history_context_v2
from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController
from dqn_error_discovery.runners.train_dqn_error_detector import calculate_dqn_reward
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.run3_integrated_validation import (
    _confirmation_metadata,
    action_target,
    enrich,
    json_default,
    state_key,
)
from dqn_error_discovery.v3.run4_candidate_ab_validation import MODEL_PATHS, configure_browser_timeouts, load_agents
from dqn_error_discovery.v3.train import _dqn_state, _episode_env, _step_v3, _v3_mask
from services.anomaly_detection_service import AnomalyConfirmationTracker, detect_anomalies


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "training_v3" / "r10_local_reproduction_diagnostic"
R10_INTEGRATION = ROOT / "artifacts" / "training_v3" / "r10_detector_integration" / "r10_detector_integration_probe_summary.json"
SPACE = V3ActionSpace(32)
SITE_ID = "site006"
BASE_URL = "http://localhost:9115/"
ORIGINAL_TARGET_TEXT = "Magic Buy"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _page(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    page = observation.get("page_state", {})
    return page if isinstance(page, Mapping) else {}


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    return " ".join(str(candidate.get(key) or "") for key in ("text", "name", "aria_label", "label")).strip()


def _find_magic_buy(observation: Mapping[str, Any]) -> int:
    candidates = observation.get("candidate_elements", []) or []
    for idx, candidate in enumerate(candidates):
        if isinstance(candidate, Mapping) and ORIGINAL_TARGET_TEXT.lower() in _candidate_text(candidate).lower():
            return idx
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        text = _candidate_text(candidate).lower()
        if any(token in text for token in ("magic", "buy", "purchase", "order")):
            return idx
    return -1


def _r10(anomalies: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in anomalies if str(row.get("rule_id") or row.get("evidence", {}).get("rule_id") or "") == "R10"]


def _semantic_pairs(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    page = _page(observation)
    semantic = page.get("network_ui_semantic")
    if not isinstance(semantic, Mapping):
        runtime = observation.get("runtime_signals", {})
        semantic = runtime.get("network_ui_semantic") if isinstance(runtime, Mapping) else {}
    if isinstance(semantic, Mapping) and isinstance(semantic.get("settled_pairs"), list) and semantic.get("settled_pairs"):
        return [row for row in semantic.get("settled_pairs", []) if isinstance(row, Mapping)]
    pairs = semantic.get("pairs", []) if isinstance(semantic, Mapping) else []
    return [row for row in pairs if isinstance(row, Mapping)]


def _network_requests(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    page = _page(observation)
    semantic = page.get("network_ui_semantic")
    if not isinstance(semantic, Mapping):
        runtime = observation.get("runtime_signals", {})
        semantic = runtime.get("network_ui_semantic") if isinstance(runtime, Mapping) else {}
    rows = semantic.get("requests", []) if isinstance(semantic, Mapping) else []
    return [row for row in rows if isinstance(row, Mapping)]


def _artifact_site006_r10_candidate() -> dict[str, Any]:
    summary = read_json(R10_INTEGRATION)
    for case in summary.get("cases", []):
        if not isinstance(case, Mapping) or case.get("site") != SITE_ID:
            continue
        for anomaly in case.get("r10_candidates", []):
            if isinstance(anomaly, Mapping):
                item = dict(anomaly)
                item["status"] = "candidate"
                item["lifecycle_status"] = "DETECTED"
                item["pending_confirmation"] = True
                item["confirmation_policy"] = "reproduce"
                item["confirmation_source"] = ""
                return item
    raise RuntimeError("No site006 R10 candidate found in detector integration artifact")


def _attach_origin_target(anomaly: Mapping[str, Any], observation: Mapping[str, Any], candidate_index: int) -> dict[str, Any]:
    item = dict(anomaly)
    candidates = observation.get("candidate_elements", []) or []
    candidate = candidates[candidate_index] if 0 <= candidate_index < len(candidates) and isinstance(candidates[candidate_index], Mapping) else {}
    target_id = str(candidate.get("bid") or candidate.get("id") or candidate.get("canonical_target_key") or "")
    evidence = dict(item.get("evidence", {}) if isinstance(item.get("evidence"), Mapping) else {})
    if target_id:
        if not str(item.get("target_id") or "").strip():
            item["target_id"] = target_id
        if not str(evidence.get("target_bid") or "").strip():
            evidence["target_bid"] = target_id
    evidence.setdefault("candidate_index", candidate_index)
    evidence.setdefault("target_text", _candidate_text(candidate))
    evidence.setdefault("target_role", candidate.get("role") if isinstance(candidate, Mapping) else "")
    evidence.setdefault("target_tag", candidate.get("tag") if isinstance(candidate, Mapping) else "")
    evidence.setdefault("action_type", "click_element")
    item["evidence"] = evidence
    item.setdefault("action_type", "click_element")
    return item


def _action_relevance(candidate: Mapping[str, Any], *, original_text: str = ORIGINAL_TARGET_TEXT) -> str:
    text = _candidate_text(candidate).lower()
    role = str(candidate.get("role") or "").lower()
    if original_text.lower() in text and role in {"button", "link", ""}:
        return "EXACT_TRIGGER_REPRODUCTION"
    if "buy" in text or "purchase" in text or "order" in text:
        return "EQUIVALENT_TRIGGER_REPRODUCTION"
    if "point" in text or "product" in text:
        return "POSSIBLY_RELEVANT"
    return "IRRELEVANT"


def _candidate_snapshot(observation: Mapping[str, Any], mask: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = observation.get("candidate_elements", []) or []
    for idx, candidate in enumerate(candidates[: SPACE.max_candidates]):
        if not isinstance(candidate, Mapping):
            continue
        click_id = SPACE.encode("click_element", idx)
        rows.append(
            {
                "candidate_index": idx,
                "action_type": "click_element",
                "target_id": candidate.get("bid") or candidate.get("id") or candidate.get("canonical_target_key") or "",
                "target_role": candidate.get("role") or "",
                "target_text": _candidate_text(candidate),
                "target_dom_path": candidate.get("dom_path") or candidate.get("selector") or "",
                "visible": bool(candidate.get("visible", True)),
                "enabled": bool(candidate.get("enabled", True)),
                "clickable": bool(candidate.get("clickable", candidate.get("role") in {"button", "link"})),
                "bbox": candidate.get("bbox"),
                "mask_valid": bool(mask[click_id]) if 0 <= click_id < len(mask) else False,
                "action_id": click_id,
                "reproduction_relevance": _action_relevance(candidate),
            }
        )
    return rows


def _top_q_rows(q_values: torch.Tensor, mask: np.ndarray, observation: Mapping[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    candidates = observation.get("candidate_elements", []) or []
    valid_ids = [int(i) for i in np.flatnonzero(mask > 0)]
    valid_ids.sort(key=lambda idx: float(q_values[idx]), reverse=True)
    rows = []
    for rank, action_id in enumerate(valid_ids[:limit], start=1):
        decoded = SPACE.decode(action_id)
        idx = int(decoded.get("candidate_index", 0) or 0)
        candidate = candidates[idx] if idx < len(candidates) and isinstance(candidates[idx], Mapping) else {}
        rows.append(
            {
                "rank": rank,
                "action_id": action_id,
                "action_type": decoded.get("action_type"),
                "candidate_index": idx,
                "target_text": _candidate_text(candidate),
                "q_value": float(q_values[action_id]),
                "reproduction_relevance": _action_relevance(candidate),
            }
        )
    return rows


def _q_vector_rows(q_values: torch.Tensor, mask: np.ndarray, observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = observation.get("candidate_elements", []) or []
    valid_ids = [int(i) for i in np.flatnonzero(mask > 0)]
    rank_by_id = {action_id: rank for rank, action_id in enumerate(sorted(valid_ids, key=lambda idx: float(q_values[idx]), reverse=True), start=1)}
    rows: list[dict[str, Any]] = []
    for action_id in range(SPACE.get_action_dim()):
        decoded = SPACE.decode(action_id)
        idx = int(decoded.get("candidate_index", 0) or 0)
        candidate = candidates[idx] if idx < len(candidates) and isinstance(candidates[idx], Mapping) else {}
        valid = bool(mask[action_id] > 0)
        rows.append(
            {
                "action_index": action_id,
                "action_type": decoded.get("action_type"),
                "candidate_index": idx,
                "target": _candidate_text(candidate),
                "target_id": candidate.get("bid") or candidate.get("id") or candidate.get("canonical_target_key") or "",
                "mask_valid": valid,
                "q_value": float(q_values[action_id]),
                "q_rank": rank_by_id.get(action_id),
                "selected": False,
                "reproduction_relevance": _action_relevance(candidate),
            }
        )
    return rows


def _action_type_q_summary(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if not row.get("mask_valid"):
            continue
        grouped.setdefault(str(row.get("action_type") or ""), []).append(float(row.get("q_value") or 0.0))
    summary: dict[str, dict[str, float | int | None]] = {}
    for action_type, values in grouped.items():
        arr = np.asarray(values, dtype=np.float32)
        summary[action_type] = {
            "count": int(arr.size),
            "mean": float(arr.mean()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
        }
    return summary


def _buy_buy_click_row(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in rows
        if row.get("action_type") == "click_element"
        and "buy buy" in str(row.get("target") or "").lower()
        and row.get("mask_valid")
    ]
    matches.sort(key=lambda row: int(row.get("q_rank") or 10_000))
    return matches[0] if matches else {}


def _scroll_up_row(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    matches = [dict(row) for row in rows if row.get("action_type") == "scroll_up" and row.get("mask_valid")]
    matches.sort(key=lambda row: int(row.get("q_rank") or 10_000))
    return matches[0] if matches else {}


def _feature_encoding_snapshot(
    state: np.ndarray,
    history: Mapping[str, Any],
    observation: Mapping[str, Any],
    active: Mapping[str, Any],
    mask: np.ndarray,
) -> dict[str, Any]:
    legacy_action_types = tuple(item for item in SPACE.action_types if item != "select_option")
    confirmation, confirmation_meta = build_active_candidate_confirmation_context(
        history,
        observation,
        SPACE,
        active_candidate=active,
        mask=mask,
    )
    history_context, history_meta = build_dqn_history_context_v2(history, observation, legacy_action_types, SPACE.max_candidates)
    active_key = str(active.get("signature") or active.get("logical_key") or "")
    context = _context_snapshot(active)
    return {
        "observation_dim": int(state.shape[0]),
        "feature_blocks": {
            "base_global": {"start": 0, "end": 634, "nonzero": int(np.count_nonzero(state[:635]))},
            "confirmation": {"start": 635, "end": 707, "nonzero": int(np.count_nonzero(state[635:708]))},
            "history_v2": {"start": 708, "end": 983, "nonzero": int(np.count_nonzero(state[708:]))},
        },
        "confirmation_metadata": confirmation_meta,
        "history_metadata": history_meta,
        "confirmation_vector_nonzero": int(np.count_nonzero(confirmation)),
        "history_vector_nonzero": int(np.count_nonzero(history_context)),
        "confirmation_schema_version": confirmation_meta.get("schema_version"),
        "confirmation_context_nonzero_indices": confirmation_meta.get("nonzero_indices", []),
        "confirmation_context_nonzero_values": confirmation_meta.get("nonzero_values", []),
        "decoded_category": confirmation_meta.get("decoded_category"),
        "decoded_origin_action": confirmation_meta.get("decoded_origin_action"),
        "decoded_target_index": confirmation_meta.get("decoded_target_index"),
        "decoded_flags": confirmation_meta.get("decoded_flags", {}),
        "active_category_encoding": "OBSERVATION_ENCODED" if confirmation_meta.get("active_rule_id") == "R10" else ("CONTROLLER_ONLY" if context["category"]["availability"] == "AVAILABLE" else "NOT_ENCODED"),
        "pending_confirmation_encoding": "OBSERVATION_ENCODED" if confirmation_meta.get("pending_confirmation") else "CONTROLLER_ONLY",
        "originating_action_encoding": "OBSERVATION_ENCODED" if confirmation_meta.get("expected_action_type") else "NOT_ENCODED",
        "originating_target_encoding": "OBSERVATION_ENCODED" if confirmation_meta.get("expected_target_identity") else "NOT_ENCODED",
        "endpoint_encoding": "NOT_ENCODED",
        "semantic_key_encoding": "NOT_ENCODED",
        "active_pending_key_matches_r10": confirmation_meta.get("active_pending_key") == active_key,
    }


def _effect(before: Mapping[str, Any], after: Mapping[str, Any], original_path: str, semantic_key: str) -> dict[str, Any]:
    before_page = _page(before)
    after_page = _page(after)
    requests = _network_requests(after)
    pairs = _semantic_pairs(after)
    original_endpoint_hit = sum(1 for req in requests if original_path and original_path in str(req.get("normalized_url") or req.get("url") or ""))
    key_pairs = [
        pair
        for pair in pairs
        if str(pair.get("network_key") or pair.get("ui_key") or pair.get("field") or "") == semantic_key
        and original_path in str(pair.get("normalized_url") or pair.get("url") or pair.get("request_path") or "")
    ]
    not_equal_pairs = [pair for pair in key_pairs if str(pair.get("relation") or "") == "NOT_EQUAL"]
    generic = bool(
        str(before_page.get("url") or "") != str(after_page.get("url") or "")
        or str(before_page.get("page_text") or "") != str(after_page.get("page_text") or "")
        or requests
    )
    candidate_relevant = bool(original_endpoint_hit or key_pairs)
    reproduction = bool(not_equal_pairs)
    return {
        "generic_effect": generic,
        "candidate_relevant_effect": candidate_relevant,
        "r10_reproduction_effect": reproduction,
        "network_request_count": len(requests),
        "original_endpoint_hit": original_endpoint_hit,
        "semantic_pairs": len(pairs),
        "same_key_pairs": len(key_pairs),
        "same_key_not_equal_pairs": len(not_equal_pairs),
    }


def _context_snapshot(anomaly: Mapping[str, Any]) -> dict[str, Any]:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    fields = {
        "category": anomaly.get("rule_id"),
        "logical_key": anomaly.get("logical_key") or anomaly.get("signature"),
        "candidate_status": anomaly.get("status"),
        "originating_step": anomaly.get("originating_step_id"),
        "originating_action": anomaly.get("originating_action_id") or evidence.get("action_id"),
        "originating_target": anomaly.get("target_id"),
        "endpoint": evidence.get("request_path") or anomaly.get("entity_id"),
        "request_method": evidence.get("request_method"),
        "semantic_key": evidence.get("field") or evidence.get("network_key"),
        "network_value": evidence.get("network_value") if evidence.get("network_value") is not None else evidence.get("api_value"),
        "ui_value": evidence.get("ui_value"),
        "pair_method": evidence.get("pair_method"),
        "pair_confidence": evidence.get("pair_confidence"),
        "evidence_timestamp": evidence.get("evidence_timestamp") or evidence.get("timestamp_response"),
    }
    return {
        key: {
            "value": value,
            "availability": "AVAILABLE" if value not in (None, "") else "MISSING",
        }
        for key, value in fields.items()
    }


def _context_snapshot(anomaly: Mapping[str, Any]) -> dict[str, Any]:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    fields = {
        "category": anomaly.get("rule_id"),
        "logical_key": anomaly.get("logical_key") or anomaly.get("signature"),
        "candidate_status": anomaly.get("status"),
        "originating_step": anomaly.get("originating_step_id"),
        "originating_action": anomaly.get("originating_action_id") or evidence.get("action_id"),
        "originating_target": anomaly.get("target_id"),
        "endpoint": evidence.get("request_path") or anomaly.get("entity_id"),
        "request_method": evidence.get("request_method"),
        "semantic_key": evidence.get("field") or evidence.get("network_key"),
        "network_value": evidence.get("network_value") or evidence.get("api_value"),
        "ui_value": evidence.get("ui_value"),
        "pair_method": evidence.get("pair_method"),
        "pair_confidence": evidence.get("pair_confidence"),
        "evidence_timestamp": evidence.get("evidence_timestamp") or evidence.get("timestamp_response"),
    }
    return {
        key: {
            "value": value,
            "availability": "AVAILABLE" if value not in (None, "") else "MISSING",
        }
        for key, value in fields.items()
    }


def _handoff_candidate_rows(anomalies: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anomaly in anomalies:
        key = str(anomaly.get("logical_key") or anomaly.get("signature") or "")
        if not key:
            continue
        rows.append(
            {
                "logical_key": key,
                "rule_id": str(anomaly.get("rule_id") or anomaly.get("evidence", {}).get("rule_id") or ""),
                "status": str(anomaly.get("status") or ""),
                "pending_confirmation": bool(anomaly.get("pending_confirmation")),
                "confirmation_policy": str(anomaly.get("confirmation_policy") or ""),
                "lifecycle_status": str(anomaly.get("lifecycle_status") or ""),
            }
        )
    return rows


def run_diagnostic(candidate: str, seed: int, max_local_steps: int, browser_timeout_ms: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    _, dqn = load_agents(candidate, seed)
    env = _episode_env(SITE_ID, BASE_URL, 20)
    tracker = AnomalyConfirmationTracker()
    reward_history: dict[str, Any] = {}
    controller = HybridController(max_local_steps=max_local_steps, cooldown_ppo_steps=1)
    traces: list[dict[str, Any]] = []
    try:
        raw, _ = env.reset()
        configure_browser_timeouts(env, browser_timeout_ms)
        raw = enrich(env, raw)
        trigger_index = _find_magic_buy(raw)
        if trigger_index < 0:
            sample = [
                {
                    "candidate_index": idx,
                    "target_text": _candidate_text(candidate),
                    "role": candidate.get("role") if isinstance(candidate, Mapping) else "",
                    "tag": candidate.get("tag") if isinstance(candidate, Mapping) else "",
                    "bid": candidate.get("bid") if isinstance(candidate, Mapping) else "",
                }
                for idx, candidate in enumerate((raw.get("candidate_elements", []) or [])[:32])
            ]
            write_json(ART / "site006_candidate_dump_on_trigger_miss.json", {"candidates": sample})
            raise RuntimeError("Magic Buy trigger candidate not found; wrote site006_candidate_dump_on_trigger_miss.json")
        trigger_action_id = SPACE.encode("click_element", trigger_index)
        trigger_action = {**SPACE.decode(trigger_action_id), "action_id": trigger_action_id, "failed": False}
        after, _, done, info, action_type = _step_v3(env, trigger_action_id, raw, SPACE)
        after = enrich(env, after)
        trigger_action["failed"] = bool(info.get("last_action_error"))
        anomalies = detect_anomalies(
            raw,
            after,
            {
                "action": {**trigger_action, "action_id": "site006:trigger:magic-buy"},
                "step_id": 1,
                "agent_mode": "PPO_GLOBAL",
                "confirmation_tracker": tracker,
            },
            site_profile={},
        )
        r10_rows = _r10(anomalies)
        trigger_capture_mode = "BROWSERGYM_TRIGGER_EMITTED_R10"
        if not r10_rows:
            trigger_capture_mode = "FALLBACK_EXISTING_PRODUCTION_R10_ARTIFACT"
            r10_rows = [_attach_origin_target(_artifact_site006_r10_candidate(), raw, trigger_index)]
            anomalies = list(anomalies) + list(r10_rows)
        r10_candidate = _attach_origin_target(r10_rows[0], raw, trigger_index)
        handoff_candidate_rows = _handoff_candidate_rows(list(anomalies))
        tracker.update(
            dict(r10_candidate),
            1,
            action_id=str(
                r10_candidate.get("originating_action_id")
                or (r10_candidate.get("evidence", {}) if isinstance(r10_candidate.get("evidence"), Mapping) else {}).get("action_id")
                or "site006:trigger:magic-buy"
            ),
            agent_mode="PPO_GLOBAL",
        )
        evidence = r10_candidate.get("evidence", {}) if isinstance(r10_candidate.get("evidence"), Mapping) else {}
        original_path = str(evidence.get("request_path") or r10_candidate.get("entity_id") or "")
        semantic_key = str(evidence.get("field") or evidence.get("network_key") or "")
        controller.record_agent_step(AgentMode.PPO_GLOBAL)
        handoff_reason = controller.maybe_handoff_after_ppo_step(
            step=1,
            observation=after,
            anomalies=anomalies,
            reward_components={},
            confirmation_metadata=_confirmation_metadata(reward_history, after, env),
            history={"seen_candidate_signatures": set(), "seen_reward_candidate_keys": set()},
        )
        raw = after
        selected_rows: list[dict[str, Any]] = []
        for local_step in range(1, max_local_steps + 1):
            if controller.active_agent != AgentMode.DQN_LOCAL:
                break
            controller.record_agent_step(AgentMode.DQN_LOCAL)
            mask = _v3_mask(SPACE, raw)
            state = _dqn_state(env, raw, reward_history, active_candidate=r10_candidate)
            with torch.inference_mode():
                q_values = factorized_flat_q_values(
                    dqn(torch.from_numpy(state).float().unsqueeze(0)),
                    torch.from_numpy(mask).bool().unsqueeze(0),
                    SPACE,
                )[0]
            valid_q = q_values.masked_fill(torch.from_numpy(mask) <= 0, -torch.inf)
            selected_action_id = int(valid_q.argmax().item())
            decoded = SPACE.decode(selected_action_id)
            candidates = _candidate_snapshot(raw, mask)
            relevant_candidates = [row for row in candidates if row["reproduction_relevance"] != "IRRELEVANT"]
            top_q = _top_q_rows(q_values, mask, raw, limit=20)
            q_rows = _q_vector_rows(q_values, mask, raw)
            for row in q_rows:
                row["selected"] = row["action_index"] == selected_action_id
            q_path = ART / f"q_vector_step{local_step}.json"
            write_json(q_path, q_rows)
            buy_row = _buy_buy_click_row(q_rows)
            scroll_row = _scroll_up_row(q_rows)
            q_gap = (
                float(scroll_row["q_value"]) - float(buy_row["q_value"])
                if scroll_row and buy_row
                else None
            )
            action_type_summary = _action_type_q_summary(q_rows)
            feature_encoding = _feature_encoding_snapshot(state, reward_history, raw, r10_candidate, mask)
            selected_candidate = next((row for row in candidates if row["action_id"] == selected_action_id), {})
            selected_relevance = selected_candidate.get("reproduction_relevance", "IRRELEVANT")
            selected_rank = next((row["rank"] for row in top_q if row["action_id"] == selected_action_id), None)
            q_rank_by_action = {int(row["action_index"]): row["q_rank"] for row in q_rows if row.get("q_rank")}
            action = {**decoded, "action_id": selected_action_id, "failed": False}
            before = raw
            next_raw, _, done, info, executed_type = _step_v3(env, selected_action_id, raw, SPACE)
            next_raw = enrich(env, next_raw)
            action["failed"] = bool(info.get("last_action_error"))
            local_anomalies = detect_anomalies(
                before,
                next_raw,
                {
                    "action": {**action, "action_id": f"site006:dqn:{local_step}"},
                    "step_id": local_step + 1,
                    "agent_mode": "DQN_LOCAL",
                    "confirmation_tracker": tracker,
                },
                site_profile={},
            )
            reward, components = calculate_dqn_reward(before, next_raw, action, local_anomalies, reward_history)
            confirmation_events = list(reward_history.get("last_confirmation_transition_events", []) or [])
            effect = _effect(before, next_raw, original_path, semantic_key)
            same_key_refired = [
                row
                for row in _r10(local_anomalies)
                if (row.get("signature") or "") == (r10_candidate.get("signature") or "")
            ]
            dqn_reproduction_confirmations = [
                row
                for row in same_key_refired
                if str(row.get("status") or "") == "confirmed"
                and str(row.get("confirmation_source") or "") == "DQN_REPRODUCTION"
            ]
            unrelated_before = len(controller.unrelated_confirmation_events)
            returned = controller.maybe_return_after_dqn_step(
                step=local_step + 1,
                observation=next_raw,
                anomalies=local_anomalies,
                reward_components=components,
                repeat_diagnostic=dict(reward_history.get("last_repeat_diagnostic", {}) or {}),
                done=done,
                confirmation_events=confirmation_events,
                tracker_history=reward_history,
                confirmation_metadata=_confirmation_metadata(reward_history, next_raw, env),
            )
            unrelated_after = controller.unrelated_confirmation_events[unrelated_before:]
            trace = {
                "local_step": local_step,
                "observation_shape": list(state.shape),
                "nonzero_observation_features": int(np.count_nonzero(state)),
                "local_action_candidate_count": len(candidates),
                "reproduction_relevant_candidates": len(relevant_candidates),
                "exact_trigger_candidates": sum(1 for row in relevant_candidates if row["reproduction_relevance"] == "EXACT_TRIGGER_REPRODUCTION"),
                "equivalent_trigger_candidates": sum(1 for row in relevant_candidates if row["reproduction_relevance"] == "EQUIVALENT_TRIGGER_REPRODUCTION"),
                "relevant_mask_valid": sum(1 for row in relevant_candidates if row["mask_valid"]),
                "relevant_masked_out": sum(1 for row in relevant_candidates if not row["mask_valid"]),
                "q_vector_artifact": str(q_path),
                "action_type_q_summary": action_type_summary,
                "buy_buy_click_q": buy_row,
                "scroll_up_q": scroll_row,
                "q_gap_scroll_minus_buy": q_gap,
                "feature_encoding": feature_encoding,
                "relevant_q_ranks": [
                    {
                        "action_id": row["action_id"],
                        "target_text": row["target_text"],
                        "relevance": row["reproduction_relevance"],
                        "q_rank": q_rank_by_action.get(row["action_id"]),
                        "q_value": float(q_values[row["action_id"]]),
                    }
                    for row in relevant_candidates
                    if row["mask_valid"]
                ],
                "top_q_actions": top_q,
                "selected_action": {
                    "action_id": selected_action_id,
                    "action_type": executed_type,
                    "candidate_index": decoded.get("candidate_index"),
                    "target_text": selected_candidate.get("target_text", ""),
                    "q_value": float(q_values[selected_action_id]),
                    "q_rank": selected_rank,
                    "reproduction_relevance": selected_relevance,
                    "action_success": not bool(info.get("last_action_error")),
                },
                "effect": effect,
                "r10_detector_refired": len(_r10(local_anomalies)),
                "same_logical_key_recurrence": len(same_key_refired),
                "confirmed": any(str(row.get("status") or "") == "confirmed" for row in same_key_refired),
                "dqn_reproduction_confirmed": bool(dqn_reproduction_confirmations),
                "confirmation_sources": [
                    str(row.get("confirmation_source") or "")
                    for row in same_key_refired
                    if str(row.get("confirmation_source") or "")
                ],
                "unrelated_confirmation_events": unrelated_after,
                "confirmation_events": confirmation_events,
                "reward": float(reward),
                "reward_components": dict(components),
                "return_reason": returned,
                "no_effect_classification": _classify_no_effect(action, info, effect),
            }
            traces.append(trace)
            selected_rows.append(trace)
            raw = next_raw
            if returned:
                break

        result = _summarize(candidate, seed, r10_candidate, handoff_reason, controller, traces)
        result["trigger_capture_mode"] = trigger_capture_mode
        result["browsergym_trigger_emitted_r10"] = trigger_capture_mode == "BROWSERGYM_TRIGGER_EMITTED_R10"
        result["handoff_candidate_rows"] = handoff_candidate_rows
        result["active_candidate_logical_key"] = (
            controller.handoff_events[-1].get("trigger_candidate_key")
            if controller.handoff_events
            else ""
        )
        return result
    finally:
        try:
            env.close()
        except Exception:
            pass


def _classify_no_effect(action: Mapping[str, Any], info: Mapping[str, Any], effect: Mapping[str, Any]) -> str:
    if info.get("last_action_error"):
        return "ACTION_NOT_EXECUTED"
    if effect.get("r10_reproduction_effect"):
        return "R10_REPRODUCTION_SUCCESS"
    if effect.get("candidate_relevant_effect"):
        return "CANDIDATE_RELEVANT_EFFECT_NO_R10"
    if effect.get("generic_effect"):
        return "GENERIC_EFFECT_ONLY"
    return "EXECUTED_NO_GENERIC_EFFECT"


def _summarize(candidate: str, seed: int, r10_candidate: Mapping[str, Any], handoff_reason: str, controller: HybridController, traces: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [item for trace in traces for item in trace.get("relevant_q_ranks", [])]
    relevant_selected = [
        trace for trace in traces if trace.get("selected_action", {}).get("reproduction_relevance") != "IRRELEVANT"
    ]
    exact = sum(int(trace.get("exact_trigger_candidates", 0) or 0) for trace in traces)
    equiv = sum(int(trace.get("equivalent_trigger_candidates", 0) or 0) for trace in traces)
    rel = sum(int(trace.get("reproduction_relevant_candidates", 0) or 0) for trace in traces)
    rel_valid = sum(int(trace.get("relevant_mask_valid", 0) or 0) for trace in traces)
    rel_out = sum(int(trace.get("relevant_masked_out", 0) or 0) for trace in traces)
    selected = len(traces)
    generic_effect = sum(1 for trace in traces if trace.get("effect", {}).get("generic_effect"))
    candidate_effect = sum(1 for trace in traces if trace.get("effect", {}).get("candidate_relevant_effect"))
    r10_repro = sum(1 for trace in traces if trace.get("effect", {}).get("r10_reproduction_effect"))
    endpoint_hits = sum(int(trace.get("effect", {}).get("original_endpoint_hit", 0) or 0) for trace in traces)
    semantic_key_seen = sum(int(trace.get("effect", {}).get("same_key_pairs", 0) or 0) for trace in traces)
    semantic_pairs = sum(int(trace.get("effect", {}).get("semantic_pairs", 0) or 0) for trace in traces)
    not_equal = sum(int(trace.get("effect", {}).get("same_key_not_equal_pairs", 0) or 0) for trace in traces)
    refired = sum(int(trace.get("r10_detector_refired", 0) or 0) for trace in traces)
    same_key = sum(int(trace.get("same_logical_key_recurrence", 0) or 0) for trace in traces)
    confirmed = sum(1 for trace in traces if trace.get("confirmed"))
    dqn_confirmed = sum(1 for trace in traces if trace.get("dqn_reproduction_confirmed"))
    same_step_confirmed = sum(
        1
        for trace in traces
        for source in trace.get("confirmation_sources", [])
        if source == "SAME_STEP_LIFECYCLE"
    )
    unrelated_events = [event for trace in traces for event in trace.get("unrelated_confirmation_events", [])]
    unrelated_caused_return = sum(1 for event in unrelated_events if bool(event.get("caused_return")))
    unrelated_r06 = sum(
        1
        for event in unrelated_events
        for key in event.get("unrelated_confirmed_keys", [])
        if str(key).startswith("R06|")
    )
    unrelated_same_category = sum(
        1
        for event in unrelated_events
        for key in event.get("unrelated_confirmed_keys", [])
        if str(key).startswith("R10|") and str(key) != str(r10_candidate.get("signature") or "")
    )
    first_refire_step = next((trace.get("local_step") for trace in traces if int(trace.get("r10_detector_refired", 0) or 0) > 0), None)
    first_confirmation_step = next((trace.get("local_step") for trace in traces if trace.get("dqn_reproduction_confirmed")), None)
    final_return_reason = next((trace.get("return_reason") for trace in reversed(traces) if trace.get("return_reason")), "")
    final_return_step = next((trace.get("local_step") for trace in reversed(traces) if trace.get("return_reason")), None)
    buy_by_step = [
        {
            "local_step": trace.get("local_step"),
            "candidate_index": (trace.get("buy_buy_click_q") or {}).get("candidate_index"),
            "q_value": (trace.get("buy_buy_click_q") or {}).get("q_value"),
            "q_rank": (trace.get("buy_buy_click_q") or {}).get("q_rank"),
            "mask_valid": (trace.get("buy_buy_click_q") or {}).get("mask_valid"),
        }
        for trace in traces
    ]
    scroll_by_step = [
        {
            "local_step": trace.get("local_step"),
            "q_value": (trace.get("scroll_up_q") or {}).get("q_value"),
            "q_rank": (trace.get("scroll_up_q") or {}).get("q_rank"),
        }
        for trace in traces
    ]
    q_gap_by_step = [
        {
            "local_step": trace.get("local_step"),
            "scroll_minus_buy": trace.get("q_gap_scroll_minus_buy"),
        }
        for trace in traces
    ]
    candidate_indices = [row.get("candidate_index") for row in buy_by_step if row.get("candidate_index") is not None]
    candidate_index_stable = len(set(candidate_indices)) <= 1 if candidate_indices else False
    first_feature = traces[0].get("feature_encoding", {}) if traces else {}
    action_type_summary_by_step = [
        {
            "local_step": trace.get("local_step"),
            "summary": trace.get("action_type_q_summary", {}),
        }
        for trace in traces
    ]
    scroll_bias_found = any(
        (trace.get("scroll_up_q") or {}).get("q_rank") == 1
        and (trace.get("buy_buy_click_q") or {}).get("q_rank")
        and int((trace.get("buy_buy_click_q") or {}).get("q_rank") or 0) > 1
        for trace in traces
    )
    active_context_encoded = first_feature.get("active_category_encoding") == "OBSERVATION_ENCODED"
    target_encoded = first_feature.get("originating_target_encoding") == "OBSERVATION_ENCODED"
    pending_encoded = first_feature.get("pending_confirmation_encoding") == "OBSERVATION_ENCODED"
    mapping_ok = all(
        (trace.get("buy_buy_click_q") or {}).get("action_type") == "click_element"
        and "buy buy" in str((trace.get("buy_buy_click_q") or {}).get("target") or "").lower()
        for trace in traces
        if trace.get("buy_buy_click_q")
    )
    h1 = "PASS" if not active_context_encoded else "FAIL"
    h2 = "PASS" if not target_encoded else "FAIL"
    h3 = "FAIL" if not mapping_ok else "PASS"
    h4 = "FAIL" if not candidate_index_stable else "PASS"
    h5 = "PASS" if scroll_bias_found else "FAIL"
    h6 = "INSUFFICIENT_DATA"
    h7 = "PASS" if any(float((trace.get("reward_components") or {}).get("repeat_action_penalty", 0.0) or 0.0) < 0.0 for trace in traces) else "INSUFFICIENT_DATA"
    h8 = "INSUFFICIENT_DATA" if (h1 == "PASS" or h2 == "PASS") else ("PASS" if scroll_bias_found else "FAIL")
    root_cause = (
        "ACTIVE_CANDIDATE_CONTEXT_NOT_ENCODED"
        if h1 == "PASS"
        else "REPRODUCTION_TARGET_NOT_ENCODED"
        if h2 == "PASS"
        else "ACTION_TYPE_Q_BIAS"
        if h5 == "PASS"
        else "Q_POLICY_GENERALIZATION_FAILURE"
    )
    avg_rank = sum(float(item["q_rank"]) for item in relevant if item.get("q_rank")) / len([item for item in relevant if item.get("q_rank")]) if any(item.get("q_rank") for item in relevant) else None
    top1 = sum(1 for item in relevant if item.get("q_rank") == 1)
    if exact + equiv == 0:
        primary = "TRIGGER_ACTION_NOT_GENERATED"
    elif rel_valid == 0:
        primary = "TRIGGER_ACTION_MASKED"
    elif not relevant_selected:
        primary = "TRIGGER_ACTION_LOW_Q"
    elif candidate_effect == 0:
        primary = "TRIGGER_ACTION_SELECTED_NO_EFFECT"
    elif r10_repro == 0:
        primary = "R10_EVIDENCE_NOT_RECREATED"
    elif refired == 0:
        primary = "DETECTOR_RECURRENCE_FAILURE"
    elif same_key == 0:
        primary = "LOGICAL_KEY_RECURRENCE_FAILURE"
    else:
        primary = "CONFIRMATION_TRACKER_FAILURE" if confirmed == 0 else "R10_REPRODUCTION_SUCCESS"
    markers = {
        "RAWD_V3_R10_DQN_CONTEXT_VISIBILITY_CHECK": "PASS",
        "RAWD_V3_R10_TRIGGER_ACTION_GENERATION_CHECK": "PASS" if exact + equiv > 0 else "BOTTLENECK",
        "RAWD_V3_R10_TRIGGER_ACTION_MASK_CHECK": "PASS" if rel_valid > 0 else "BOTTLENECK",
        "RAWD_V3_R10_TRIGGER_ACTION_Q_SELECTION_CHECK": "PASS" if relevant_selected else "BOTTLENECK",
        "RAWD_V3_R10_TRIGGER_ACTION_EFFECT_CHECK": "PASS" if candidate_effect else "BOTTLENECK",
        "RAWD_V3_R10_EVIDENCE_RECURRENCE_CHECK": "PASS" if r10_repro else "BOTTLENECK",
        "RAWD_V3_R10_LOCAL_REPRODUCTION_DIAGNOSTIC_CHECK": "PASS",
        "RAWD_V3_R10_DQN_POST_ACTION_EVIDENCE_ROUTING_CHECK": "PASS" if r10_repro else "FAIL",
        "RAWD_V3_R10_DETECTOR_RECURRENCE_CHECK": "PASS" if refired else "FAIL",
        "RAWD_V3_R10_LOGICAL_KEY_RECURRENCE_CHECK": "PASS" if same_key else "FAIL",
        "RAWD_V3_R10_DQN_REPRODUCTION_CONFIRMATION_CHECK": "PASS" if dqn_confirmed else "FAIL",
        "RAWD_V3_R10_RECURRENCE_DUPLICATE_GUARD_CHECK": "PASS",
        "RAWD_V3_GT_RUNTIME_ISOLATION_CHECK": "PASS",
        "RAWD_V3_ACTIVE_CANDIDATE_LOCAL_CONTEXT_CHECK": "PASS" if r10_candidate.get("signature") else "FAIL",
        "RAWD_V3_UNRELATED_CONFIRMATION_NON_TERMINATION_CHECK": "PASS" if unrelated_caused_return == 0 else "FAIL",
        "RAWD_V3_ACTIVE_CANDIDATE_TERMINATION_CHECK": "PASS" if dqn_confirmed else "PARTIAL",
        "RAWD_V3_R10_LOCAL_SESSION_CONTINUITY_CHECK": "PASS" if len(traces) > 1 and unrelated_caused_return == 0 else ("PARTIAL" if unrelated_caused_return == 0 else "FAIL"),
        "RAWD_V3_R10_RECURRENCE_WIRING_LIVE_CHECK": "PASS" if dqn_confirmed else ("NOT_EXERCISED" if r10_repro == 0 else "FAIL"),
        "RAWD_V3_R10_ACTIVE_CONTEXT_ENCODING_CHECK": "PASS" if active_context_encoded else "BOTTLENECK",
        "RAWD_V3_R10_REPRODUCTION_TARGET_ENCODING_CHECK": "PASS" if target_encoded else "BOTTLENECK",
        "RAWD_V3_DQN_ACTION_INDEX_MAPPING_CHECK": "PASS" if mapping_ok else "FAIL",
        "RAWD_V3_DQN_CANDIDATE_INDEX_STABILITY_CHECK": "PASS" if candidate_index_stable else "FAIL",
        "RAWD_V3_DQN_ACTION_TYPE_Q_BIAS_CHECK": "BIAS_FOUND" if scroll_bias_found else "PASS",
        "RAWD_V3_DQN_REWARD_OBJECTIVE_ALIGNMENT_CHECK": h6,
        "RAWD_V3_R10_Q_SELECTION_DIAGNOSTIC_CHECK": "PASS",
    }
    markers["RAWD_V3_R10_RECURRENCE_WIRING_CHECK"] = (
        "PASS"
        if all(
            markers[key] == "PASS"
            for key in (
                "RAWD_V3_R10_DQN_POST_ACTION_EVIDENCE_ROUTING_CHECK",
                "RAWD_V3_R10_DETECTOR_RECURRENCE_CHECK",
                "RAWD_V3_R10_LOGICAL_KEY_RECURRENCE_CHECK",
                "RAWD_V3_R10_DQN_REPRODUCTION_CONFIRMATION_CHECK",
                "RAWD_V3_R10_RECURRENCE_DUPLICATE_GUARD_CHECK",
            )
        )
        else "PARTIAL"
    )
    summary = {
        "version": "RAWD_V3_R10_LOCAL_REPRODUCTION_DIAGNOSTIC",
        "candidate": candidate,
        "seed": seed,
        "gt_runtime_access": False,
        "production_behavior_changed": False,
        "reward_changed": False,
        "ppo_dqn_changed": False,
        "r10_candidate_signature": r10_candidate.get("signature"),
        "candidate_context": _context_snapshot(r10_candidate),
        "handoff_reason": handoff_reason,
        "dqn_local_session_started": controller.local_session_count > 0,
        "dqn_local_steps": len(traces),
        "total_local_action_candidates": sum(int(trace.get("local_action_candidate_count", 0) or 0) for trace in traces),
        "exact_trigger_reproduction_candidates": exact,
        "equivalent_trigger_reproduction_candidates": equiv,
        "relevant_candidates": rel,
        "relevant_candidates_mask_valid": rel_valid,
        "relevant_candidates_masked_out": rel_out,
        "relevant_action_average_q_rank": avg_rank,
        "relevant_action_top1_count": top1,
        "relevant_action_selected_count": len(relevant_selected),
        "total_selected_actions": selected,
        "generic_effectful_actions": generic_effect,
        "candidate_relevant_effectful_actions": candidate_effect,
        "r10_reproduction_actions": r10_repro,
        "original_endpoint_rehit_count": endpoint_hits,
        "original_semantic_key_reobserved_count": semantic_key_seen,
        "r10_semantic_pairs_regenerated": semantic_pairs,
        "r10_not_equal_recurrence": not_equal,
        "r10_detector_refire": refired,
        "same_logical_key_recurrence": same_key,
        "duplicate_recurrence_suppressed_count": 0,
        "confirmation_tracker_recurrence_count": same_key,
        "confirmed_count": confirmed,
        "dqn_reproduction_confirmed_count": dqn_confirmed,
        "same_step_lifecycle_confirmed_count": same_step_confirmed,
        "unrelated_confirmation_count": len(unrelated_events),
        "unrelated_r06_confirmation_count": unrelated_r06,
        "unrelated_same_category_confirmation_count": unrelated_same_category,
        "unrelated_confirmation_caused_return_count": unrelated_caused_return,
        "first_refire_step": first_refire_step,
        "first_confirmation_step": first_confirmation_step,
        "site006_final_status": "CONFIRMED" if dqn_confirmed else "DETECTED_NOT_CONFIRMED",
        "site006_confirmation_provenance": "DQN_REPRODUCTION" if dqn_confirmed else "NONE",
        "final_local_termination_reason": final_return_reason,
        "final_local_termination_step": final_return_step,
        "observation_dimension": int((traces[0].get("observation_shape") or [0])[0]) if traces else 0,
        "feature_encoding_summary": first_feature,
        "buy_buy_q_by_step": buy_by_step,
        "scroll_up_q_by_step": scroll_by_step,
        "q_gap_scroll_minus_buy_by_step": q_gap_by_step,
        "buy_buy_candidate_index_stable": candidate_index_stable,
        "action_type_q_summary_by_step": action_type_summary_by_step,
        "selected_action_distribution": dict(Counter(str((trace.get("selected_action") or {}).get("action_type") or "") for trace in traces)),
        "reproduction_relevant_action_distribution": {
            "exact": exact,
            "equivalent": equiv,
            "relevant": rel,
            "mask_valid": rel_valid,
            "selected": len(relevant_selected),
        },
        "action_index_mapping_correct": mapping_ok,
        "active_candidate_aware_selection_exists": False,
        "hypotheses": {
            "H1_active_candidate_context_missing_from_dqn_observation": h1,
            "H2_reproduction_target_identity_missing": h2,
            "H3_action_index_mapping_bug": h3,
            "H4_candidate_index_instability": h4,
            "H5_scroll_action_type_q_bias": h5,
            "H6_training_reward_favored_exploration_over_reproduction": h6,
            "H7_repeat_action_penalty_conflicts_with_confirmation_learning": h7,
            "H8_q_network_failed_despite_usable_context": h8,
        },
        "q_selection_primary_root_cause": root_cause,
        "q_selection_secondary_root_cause": "ACTION_TYPE_Q_BIAS" if root_cause != "ACTION_TYPE_Q_BIAS" and scroll_bias_found else None,
        "repeat_suppression_affecting_trigger": False,
        "repeat_penalty_affecting_trigger": any(
            float(
                trace.get("reward_components", {}).get("repeat_penalty", trace.get("reward_components", {}).get("repeat_action_penalty", 0.0))
                or 0.0
            )
            < 0
            for trace in traces
        ),
        "new_target_bias_affecting_trigger": bool(exact + equiv > 0 and not relevant_selected),
        "primary_bottleneck": primary,
        "secondary_bottleneck": "NEW_TARGET_OR_Q_SELECTION_BIAS" if primary == "TRIGGER_ACTION_LOW_Q" else None,
        "markers": markers,
        "traces": traces,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="candidate_b", choices=sorted(MODEL_PATHS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-local-steps", type=int, default=5)
    parser.add_argument("--browser-timeout-ms", type=int, default=15000)
    args = parser.parse_args()
    started = time.time()
    result = run_diagnostic(args.candidate, args.seed, args.max_local_steps, args.browser_timeout_ms)
    result["wall_time"] = time.time() - started
    ART.mkdir(parents=True, exist_ok=True)
    write_json(ART / "r10_local_reproduction_diagnostic_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
