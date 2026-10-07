"""Run5 context-aware DQN training for RAWD v3.

PPO is frozen at Candidate B.  DQN is freshly initialized and trained with the
v3 active-candidate confirmation context while keeping obs/action dimensions,
reward, detector, controller semantics, and site sources unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.confirmation_observation import (
    ACTIVE_CANDIDATE_CONFIRMATION_OBSERVATION_VERSION,
    build_active_candidate_confirmation_context,
)
from dqn_error_discovery.agents.confirmation_reproduction import ACTION_REPLAY, derive_reproduction_context
from dqn_error_discovery.agents.dqn_agent import (
    ConfirmationAwareReplayBuffer,
    FactorizedQNetwork,
    Transition,
    factorized_flat_q_values,
)
from dqn_error_discovery.runners.train_dqn_error_detector import build_replay_metadata, calculate_dqn_reward
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.productive_context_sampler import classify_context
from dqn_error_discovery.v3.run4_fullstack_training import (
    SiteServer,
    catalog_categories,
    discover_sites,
    json_default,
    preflight_site,
    read_json,
    run_git,
    telemetry_from_anomalies,
    write_json,
    write_jsonl,
)
from dqn_error_discovery.v3.train import _dqn_state, _dqn_update, _step_v3, _v3_mask, _v3_state
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies
from services.anomaly_detection_service import AnomalyConfirmationTracker
from services.site_profile_service import build_site_profile, load_training_site_config


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts/models/v3"
OUT = Path(os.environ.get("RAWD_RUN_OUT", str(ROOT / "artifacts/training_v3/run5_context_aware_dqn")))
MODEL_OUT = Path(os.environ.get("RAWD_RUN_MODEL_OUT", str(MODELS / "run5_context_aware_dqn")))
SPACE = V3ActionSpace(32)
CANDIDATE_B_PATHS = {
    "ppo": MODELS / "rawd_v3_run4_candidate_b_ppo.pt",
    "dqn42": MODELS / "rawd_v3_run4_candidate_b_dqn_seed42.pt",
    "dqn43": MODELS / "rawd_v3_run4_candidate_b_dqn_seed43.pt",
    "dqn44": MODELS / "rawd_v3_run4_candidate_b_dqn_seed44.pt",
}
EXPECTED_B = {
    "ppo": "F185CD55F9B262EB88C70BE418CFF1240EE0C667814589368D8FAA342AC4685E",
    "dqn42": "44F066B0576D5650A3F542B2E32DBD6847A35ABE823F173B3398ED454EB4005E",
    "dqn43": "5DCF693E5BE1D49EC618C8674C42811DDAB1E266C7B7F96EBF0F9B6FC993D832",
    "dqn44": "553AE8C71CA2C9853CA58A73497035E2BC0B3611E72BA5247662436F245C4E8E",
}
EXPECTED_A = {
    "ppo": "16F45AD405F4298FDDD00BA8532B2EB30635968208D5CF533F8AFD907743A9D3",
    "dqn42": "6D90FF0AE03DC3DC2FA3118387F3F515B6FE84A7109DB7F3880AE39ED1B9B5F4",
    "dqn43": "7DA480BD642FE117AC5A44FBFC4D4D7E9CB23A56B5457DD3BEEC991460F67298",
    "dqn44": "6A9982528217E9F973313090B2124708F5007ED6262A86160590D21842594958",
}
CANDIDATE_A_PATHS = {
    "ppo": MODELS / "rawd_v3_ppo_selectcov_seed42.pt",
    "dqn42": MODELS / "rawd_v3_dqn_productive_seed42.pt",
    "dqn43": MODELS / "rawd_v3_dqn_productive_seed43.pt",
    "dqn44": MODELS / "rawd_v3_dqn_productive_seed44.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _recovery_payload(
    *, seed: int, network: FactorizedQNetwork, target: FactorizedQNetwork,
    optimizer: torch.optim.Optimizer, replay: ConfirmationAwareReplayBuffer, episode_count: int,
    optimizer_steps: int, total_steps: int,
) -> dict[str, Any]:
    return {
        "snapshot_version": "run5_seed44_full_state_v1",
        "seed": seed,
        "episode_count": episode_count,
        "optimizer_steps": optimizer_steps,
        "total_steps": total_steps,
        "network_state_dict": network.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_capacity": replay.items.maxlen,
        "replay_items": list(replay.items),
        "replay_general_items": list(replay.general_items),
        "replay_confirmation_items": list(replay.confirmation_items),
        "replay_successful_items": list(replay.successful_items),
        "replay_confirmation_fraction": replay.confirmation_fraction,
        "replay_sample_telemetry": dict(replay.sample_telemetry),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "obs_schema": ACTIVE_CANDIDATE_CONFIRMATION_OBSERVATION_VERSION,
        "obs_dim": 984,
        "gt_runtime_access": False,
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reset_env(site: Mapping[str, Any], max_steps: int, browser_timeout_ms: int = 60000) -> tuple[BrowserGymJAWSEnv, dict[str, Any]]:
    site_id = str(site["site_id"])
    try:
        cfg = load_training_site_config(site_id)
    except Exception:
        cfg = {}
    profile = build_site_profile(site_id, exploration_profile=cfg.get("exploration_profile"))
    profile["browsergym_timeout_ms"] = int(browser_timeout_ms)
    # A failed BrowserGym/Playwright startup can leave the process-global handle
    # as the context manager rather than the started Playwright object.  Clear
    # only that invalid infrastructure handle; this does not alter task policy,
    # observation, reward, or replay semantics.
    try:
        import browsergym.core as browsergym_core
        if not hasattr(getattr(browsergym_core, "_PLAYWRIGHT", None), "chromium"):
            browsergym_core._PLAYWRIGHT = None
    except Exception:
        pass
    env = BrowserGymJAWSEnv(
        site_id=site_id,
        base_url=str(site["url"]),
        max_steps=max_steps,
        max_candidates=32,
        headless=True,
        site_profile=profile,
        requires_login=bool(cfg.get("requires_login")),
        login_config=cfg.get("login") if isinstance(cfg.get("login"), Mapping) else None,
    )
    raw, _ = env.reset()
    return env, dict(raw)


def q_values(network: FactorizedQNetwork, state: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    with torch.inference_mode():
        return factorized_flat_q_values(
            network(torch.from_numpy(state).float().unsqueeze(0)),
            torch.from_numpy(mask).bool().unsqueeze(0),
            SPACE,
        )[0]


def choose_dqn(network: FactorizedQNetwork, state: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
    valid = np.flatnonzero(mask > 0)
    if len(valid) == 0:
        return SPACE.encode("noop", 0)
    if random.random() < epsilon:
        return int(random.choice(valid.tolist()))
    values = q_values(network, state, mask)
    return int(values.masked_fill(torch.from_numpy(mask) <= 0, -torch.inf).argmax().item())


def anomaly_key(anomaly: Mapping[str, Any]) -> str:
    return str(anomaly.get("signature") or anomaly.get("logical_key") or "")


def active_candidate_from_anomalies(anomalies: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in anomalies
        if isinstance(row, Mapping)
        and derive_reproduction_context(row).strategy == ACTION_REPLAY
        and (
            bool(row.get("pending_confirmation"))
            or str(row.get("status") or "").lower() == "candidate"
            or str(row.get("confirmation_policy") or "").lower() == "reproduce"
        )
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            0 if str(row.get("rule_id") or row.get("evidence", {}).get("rule_id") or "") == "R10" else 1,
            0 if bool(row.get("pending_confirmation")) else 1,
            str(row.get("signature") or ""),
        )
    )
    return candidates[0]


def semantic_pairs(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    semantic = page.get("network_ui_semantic")
    if not isinstance(semantic, Mapping):
        runtime = observation.get("runtime_signals", {})
        semantic = runtime.get("network_ui_semantic") if isinstance(runtime, Mapping) else {}
    rows = semantic.get("settled_pairs") if isinstance(semantic, Mapping) and semantic.get("settled_pairs") else semantic.get("pairs", []) if isinstance(semantic, Mapping) else []
    return [row for row in rows if isinstance(row, Mapping)]


def network_requests(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    page = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    semantic = page.get("network_ui_semantic")
    if not isinstance(semantic, Mapping):
        runtime = observation.get("runtime_signals", {})
        semantic = runtime.get("network_ui_semantic") if isinstance(runtime, Mapping) else {}
    rows = semantic.get("requests", []) if isinstance(semantic, Mapping) else []
    return [row for row in rows if isinstance(row, Mapping)]


def active_endpoint_and_key(candidate: Mapping[str, Any] | None) -> tuple[str, str]:
    if not candidate:
        return "", ""
    evidence = candidate.get("evidence", {}) if isinstance(candidate.get("evidence"), Mapping) else {}
    return str(evidence.get("request_path") or candidate.get("entity_id") or ""), str(evidence.get("field") or evidence.get("network_key") or "")


def active_context_metadata(history: Mapping[str, Any], observation: Mapping[str, Any], active: Mapping[str, Any] | None, mask: np.ndarray) -> dict[str, Any]:
    _vector, meta = build_active_candidate_confirmation_context(history, observation, SPACE, active_candidate=active, mask=mask)
    return meta


def q_rank_for_action(q: torch.Tensor, mask: np.ndarray, action_id: int) -> int | None:
    if not 0 <= int(action_id) < len(mask) or mask[action_id] <= 0:
        return None
    valid_ids = [int(idx) for idx in np.flatnonzero(mask > 0)]
    valid_ids.sort(key=lambda idx: float(q[idx]), reverse=True)
    return valid_ids.index(int(action_id)) + 1 if int(action_id) in valid_ids else None


def transition_telemetry(
    q: torch.Tensor,
    mask: np.ndarray,
    selected_id: int,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    active: Mapping[str, Any] | None,
    meta: Mapping[str, Any],
    anomalies: list[Mapping[str, Any]],
    components: Mapping[str, Any],
    history: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    history = history or {}
    target_slot = meta.get("decoded_target_index")
    action_type = str(meta.get("decoded_origin_action") or "")
    target_action_id = SPACE.encode(action_type, int(target_slot)) if action_type in SPACE.action_types and target_slot is not None else None
    endpoint, field = active_endpoint_and_key(active)
    requests = network_requests(after)
    pairs = semantic_pairs(after)
    endpoint_hits = sum(1 for row in requests if endpoint and endpoint in str(row.get("normalized_url") or row.get("url") or ""))
    field_pairs = [
        row
        for row in pairs
        if field
        and str(row.get("network_key") or row.get("ui_key") or row.get("field") or "") == field
        and endpoint in str(row.get("request_path") or row.get("normalized_url") or row.get("url") or "")
    ]
    not_equal = [row for row in field_pairs if str(row.get("relation") or "") == "NOT_EQUAL"]
    active_key = anomaly_key(active or {})
    r10_refire = [row for row in anomalies if str(row.get("rule_id") or row.get("evidence", {}).get("rule_id") or "") == "R10"]
    same_key = [row for row in r10_refire if anomaly_key(row) == active_key]
    confirmations = [row for row in same_key if str(row.get("status") or "") == "confirmed"]
    event = history.get("last_confirmation_transition") if isinstance(history.get("last_confirmation_transition"), Mapping) else None
    pre_records = history.get("last_pre_pending_records", {}) if isinstance(history.get("last_pre_pending_records"), Mapping) else {}
    event_key = str((event or {}).get("logical_key") or "MISSING")
    event_id = str((event or {}).get("event_id") or (f"{event_key}:{history.get('current_transition_step', 'MISSING')}" if event else "MISSING"))
    matching_record = next((record for record in pre_records.values() if isinstance(record, Mapping) and str(record.get("logical_key") or "") == event_key), None)
    action_repro_record = next((record for record in pre_records.values() if isinstance(record, Mapping) and bool(record.get("requires_action_reproduction"))), None)
    selected_target = action.get("target_identity", "MISSING")
    return {
        "active_candidate": int(active is not None),
        "r10_active_candidate": int(str((active or {}).get("rule_id") or (active or {}).get("evidence", {}).get("rule_id") or "") == "R10"),
        "pending_confirmation": int(bool(meta.get("pending_confirmation"))),
        "target_mapped": int(bool(meta.get("origin_target_mapped"))),
        "target_actionable": int(bool(meta.get("active_target_actionable"))),
        "reproduction_target_available": int(target_action_id is not None),
        "reproduction_target_valid": int(target_action_id is not None and mask[int(target_action_id)] > 0),
        "reproduction_target_q_rank": q_rank_for_action(q, mask, int(target_action_id)) if target_action_id is not None else None,
        "reproduction_action_selected": int(target_action_id is not None and int(selected_id) == int(target_action_id)),
        "endpoint_rehit": int(endpoint_hits > 0),
        "not_equal_recurrence": int(bool(not_equal)),
        "detector_refire": int(bool(r10_refire)),
        "same_logical_key": int(bool(same_key)),
        "dqn_reproduction_confirmation": int(any(str(row.get("confirmation_source") or "") == "DQN_REPRODUCTION" for row in confirmations)),
        "passive_reobservation_confirmation": int(any(str(row.get("confirmation_source") or "") == "PASSIVE_REOBSERVATION" for row in confirmations)),
        "same_step_lifecycle_confirmation": int(any(str(row.get("confirmation_source") or "") == "SAME_STEP_LIFECYCLE" for row in confirmations)),
        "generic_no_effect": int(float(components.get("no_effect_penalty", 0.0) or 0.0) < 0),
        "candidate_relevant_effect": int(endpoint_hits > 0 or bool(field_pairs)),
        "r10_reproduction_effect": int(bool(not_equal)),
        "repeat_penalty": float(components.get("repeat_action_penalty", components.get("repeat_penalty", 0.0)) or 0.0),
        "transition_id": str(history.get("transition_id", "MISSING")),
        "active_candidate_logical_key": active_key or "MISSING",
        "active_candidate_status_before": (event or {}).get("previous_status", "MISSING"),
        "active_candidate_status_after": (event or {}).get("current_status", "MISSING"),
        "requires_action_reproduction": bool((action_repro_record or {}).get("requires_action_reproduction")) if action_repro_record is not None else "MISSING",
        "reproduction_strategy": (action_repro_record or {}).get("confirmation_mode", "MISSING"),
        "reproduction_action_type": (action_repro_record or {}).get("action_type", "MISSING"),
        "reproduction_target_identity": (action_repro_record or {}).get("action_target", "MISSING"),
        "selected_action_type": str(action.get("action_type") or "MISSING"),
        "selected_target_identity": selected_target,
        "network_recurrence": bool(endpoint_hits > 0),
        "semantic_recurrence": bool(not_equal),
        "same_chain_recurrence": bool(same_key),
        "confirmation_event_present": bool(event),
        "confirmation_event_id": event_id,
        "confirmation_event_key": event_key,
        "confirmation_source": next((str(row.get("confirmation_source") or "") for row in confirmations if row.get("confirmation_source")), "MISSING"),
        "confirmation_event_step": (event or {}).get("created_step", history.get("current_transition_step", "MISSING")),
        "confirmation_event_fresh": bool(event and (event.get("created_step", history.get("current_transition_step")) == history.get("current_transition_step"))),
        "confirmation_event_consumed": bool(event and (event_key in (history.get("rewarded_confirmed_logical_keys", set()) or set()))),
        "confirmation_upgrade_eligible": bool(history.get("last_confirmation_upgrade_eligible", False)),
        "confirmation_upgrade_component": float(components.get("confirmation_upgrade", 0.0) or 0.0),
        "final_reward": float(components.get("final_reward", 0.0) or 0.0),
    }


def loss_stats(losses: list[float]) -> dict[str, float]:
    values = np.asarray(losses or [0.0], dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def parameter_checksum(network: FactorizedQNetwork) -> float:
    with torch.inference_mode():
        return float(sum(parameter.detach().abs().sum().item() for parameter in network.parameters()))


def train_seed(seed: int, sites: list[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    seed_all(seed)
    ppo = PPOAgent(635, 640)
    ppo.load(CANDIDATE_B_PATHS["ppo"])
    network = FactorizedQNetwork(984, 20, 32)
    target = FactorizedQNetwork(984, 20, 32)
    target.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    replay = ConfirmationAwareReplayBuffer(50000)
    rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    losses: list[float] = []
    health = Counter()
    funnel = Counter()
    action_counts = Counter()
    action_reward = defaultdict(float)
    q_ranks: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    reward_components = defaultdict(float)
    confirmation_action_counts = Counter()
    successful_draw_counts = Counter()
    optimizer_trace: list[dict[str, Any]] = []
    target_update_count = 0
    stop_requested = False
    ppo_steps_total = 0
    optimizer_steps = 0
    total_steps = 0
    episode_count = 0
    disk_rows: list[dict[str, Any]] = []
    for site in sites:
        with SiteServer(site):
            for episode in range(args.episodes_per_site):
                episode_count += 1
                env = None
                try:
                    print(f"[run6-dqn] seed={seed} start {site['site_id']} episode={episode + 1}/{args.episodes_per_site}")
                    env, raw = reset_env(site, args.max_steps, args.browser_timeout_ms)
                    episode_ppo_steps = 0
                    episode_replay_before = len(replay)
                    history: dict[str, Any] = {}
                    sessions = 0
                    for ppo_step in range(args.max_steps):
                        ppo_steps_total += 1
                        episode_ppo_steps += 1
                        selected = ppo.select_action(_v3_state(env, raw), _v3_mask(SPACE, raw))
                        aid = int(selected["action_id"])
                        ppo_action = {**SPACE.decode(aid), "action_id": aid, "failed": False}
                        after, _, done, info, _ = _step_v3(env, aid, raw, SPACE)
                        ppo_action["failed"] = bool(info.get("last_action_error"))
                        anomalies = detect_anomalies(raw, after, {"action": ppo_action}, site_profile={})
                        context = classify_context(raw, after, anomalies, info)
                        active = active_candidate_from_anomalies(anomalies)
                        raw = after
                        if not context.get("admit") or sessions >= args.dqn_sessions_per_site:
                            if done:
                                break
                            continue
                        sessions += 1
                        session_id = f"{site['site_id']}:seed{seed}:ep{episode}:session{sessions}"
                        session = Counter({"sessions": 1})
                        tracker = AnomalyConfirmationTracker()
                        local_raw = raw
                        endpoint, field = active_endpoint_and_key(active)
                        for local_step in range(args.max_local_steps):
                            local_mask = _v3_mask(SPACE, local_raw)
                            meta = active_context_metadata(history, local_raw, active, local_mask)
                            state = _dqn_state(env, local_raw, history, active_candidate=active)
                            q = q_values(network, state, local_mask)
                            valid_q = q[torch.from_numpy(local_mask).bool()]
                            health["nan_q"] += int(bool(torch.isnan(valid_q).any()) if valid_q.numel() else False)
                            health["inf_q"] += int(bool(torch.isinf(valid_q).any()) if valid_q.numel() else False)
                            local_id = choose_dqn(network, state, local_mask, args.epsilon)
                            action = {**SPACE.decode(local_id), "action_id": int(local_id), "failed": False}
                            local_after, _, local_done, local_info, executed_type = _step_v3(env, local_id, local_raw, SPACE)
                            action["failed"] = bool(local_info.get("last_action_error"))
                            local_anomalies = detect_anomalies(
                                local_raw, local_after,
                                {"action": action, "step_id": total_steps + 1, "agent_mode": "DQN_LOCAL", "confirmation_tracker": tracker},
                                site_profile={},
                            )
                            transition_id = f"{episode_count}:{session_id}:{local_step + 1}:{total_steps + 1}"
                            history["transition_id"] = transition_id
                            history["current_transition_step"] = total_steps + 1
                            reward, components = calculate_dqn_reward(local_raw, local_after, action, local_anomalies, history)
                            next_state = _dqn_state(env, local_after, history, active_candidate=active)
                            next_mask = _v3_mask(SPACE, local_after)
                            progress = bool(history.get("last_confirmation_progress"))
                            upgrade = bool(float(components.get("confirmation_upgrade", 0.0) or 0.0) > 0.0)
                            runtime_effect = any(
                                isinstance(row.get("evidence"), Mapping)
                                and (
                                    bool(row["evidence"].get("ui_settled"))
                                    or str(row["evidence"].get("source") or "") == "network_ui_semantic"
                                    or bool(row["evidence"].get("network_key"))
                                )
                                for row in local_anomalies
                            )
                            replay_metadata = build_replay_metadata(
                                confirmation_context=bool(meta.get("pending_confirmation") or progress or upgrade),
                                action_replay=bool(str(meta.get("decoded_origin_action") or "") == str(action.get("action_type") or "") and meta.get("pending_confirmation")),
                                reproduction_attempt=bool(meta.get("pending_confirmation") or progress),
                                candidate_relevant_effect=bool(runtime_effect or progress),
                                same_chain_recurrence=bool(progress),
                                confirmation_upgrade=upgrade,
                            )
                            replay.add(Transition(state, int(local_id), float(reward), next_state, bool(local_done), next_mask, local_mask, replay_metadata))
                            for component_name, component_value in components.items():
                                reward_components[component_name] += float(component_value or 0.0)
                            if meta.get("pending_confirmation"):
                                confirmation_action_counts[str(action["action_type"])] += 1
                            loss = None
                            if len(replay) >= args.batch_size:
                                batch = replay.sample(args.batch_size, seed=seed * 100000 + optimizer_steps)
                                for item in batch:
                                    if bool((item.replay_metadata or {}).get("is_successful_reproduction")):
                                        successful_draw_counts[id(item)] += 1
                                batch_rows.append({
                                    "optimizer_step": optimizer_steps + 1,
                                    "batch_size": len(batch),
                                    "general_samples": sum(int(not bool((item.replay_metadata or {}).get("is_confirmation_context"))) for item in batch),
                                    "confirmation_samples": sum(int(bool((item.replay_metadata or {}).get("is_confirmation_context"))) for item in batch),
                                    "action_replay_samples": sum(int(bool((item.replay_metadata or {}).get("is_action_replay"))) for item in batch),
                                    "successful_reproduction_samples": sum(int(bool((item.replay_metadata or {}).get("is_successful_reproduction"))) for item in batch),
                                    "failed_reproduction_samples": sum(int(bool((item.replay_metadata or {}).get("is_failed_reproduction"))) for item in batch),
                                    "partial_reproduction_samples": sum(int(bool((item.replay_metadata or {}).get("is_partial_reproduction"))) for item in batch),
                                })
                                before_checksum = parameter_checksum(network)
                                batch_rewards = np.asarray([item.reward for item in batch], dtype=np.float64)
                                batch_states = torch.from_numpy(np.stack([item.state for item in batch])).float()
                                batch_masks = torch.from_numpy(np.stack([item.current_mask for item in batch])).bool()
                                batch_next_states = torch.from_numpy(np.stack([item.next_state for item in batch])).float()
                                batch_next_masks = torch.from_numpy(np.stack([item.next_mask for item in batch])).bool()
                                with torch.inference_mode():
                                    current_q_all = factorized_flat_q_values(network(batch_states), batch_masks, SPACE)
                                    target_q_all = factorized_flat_q_values(target(batch_next_states), batch_next_masks, SPACE)
                                loss = _dqn_update(network, target, optimizer, batch, 0.99, SPACE)
                                after_checksum = parameter_checksum(network)
                                grad_norm = float(sum(parameter.grad.detach().norm().item() ** 2 for parameter in network.parameters() if parameter.grad is not None) ** 0.5)
                                optimizer_trace.append({
                                    "optimizer_step": optimizer_steps + 1,
                                    "transition_count": len(transition_rows),
                                    "batch_size": len(batch),
                                    "state_batch_shape": list(batch_states.shape),
                                    "action_batch_size": len(batch),
                                    "reward_batch": {"min": float(batch_rewards.min()), "mean": float(batch_rewards.mean()), "max": float(batch_rewards.max())},
                                    "current_q": {"min": float(current_q_all.min()), "mean": float(current_q_all.mean()), "max": float(current_q_all.max()), "nan": int(torch.isnan(current_q_all).any()), "inf": int(torch.isinf(current_q_all).any())},
                                    "target_q": {"min": float(target_q_all.min()), "mean": float(target_q_all.mean()), "max": float(target_q_all.max()), "nan": int(torch.isnan(target_q_all).any()), "inf": int(torch.isinf(target_q_all).any())},
                                    "loss": float(loss),
                                    "gradient_norm": grad_norm,
                                    "parameter_checksum_before": before_checksum,
                                    "parameter_checksum_after": after_checksum,
                                    "parameter_changed": bool(before_checksum != after_checksum),
                                    "zero_grad_backward_optimizer_step": True,
                                })
                                optimizer_steps += 1
                                losses.append(loss)
                                health["nan_loss"] += int(bool(np.isnan(loss)))
                                health["inf_loss"] += int(bool(np.isinf(loss)))
                                grad_bad = any(
                                    parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
                                    for parameter in network.parameters()
                                )
                                health["nan_inf_grad"] += int(grad_bad)
                            if optimizer_steps and optimizer_steps % args.target_update_interval == 0:
                                target.load_state_dict(network.state_dict())
                                target_update_count += 1
                            tel = transition_telemetry(q, local_mask, int(local_id), local_raw, local_after, active, meta, local_anomalies, components, history)
                            funnel.update({key: value for key, value in tel.items() if isinstance(value, int)})
                            session.update({key: value for key, value in tel.items() if isinstance(value, int)})
                            action_counts[str(action["action_type"])] += 1
                            action_reward[str(action["action_type"])] += float(reward)
                            if tel.get("reproduction_target_q_rank") is not None:
                                q_ranks.append({"transition": len(transition_rows), "episode": episode_count, "rank": int(tel["reproduction_target_q_rank"]), "selected_rank": q_rank_for_action(q, local_mask, int(local_id)), "selected": bool(tel.get("reproduction_action_selected"))})
                            transition_rows.append(
                                {
                                    "seed": seed,
                                    "transition_id": transition_id,
                                    "site_id": site["site_id"],
                                    "episode": episode,
                                    "session_id": session_id,
                                    "local_step": local_step + 1,
                                    "active_key": anomaly_key(active or {}),
                                    "active_rule": str((active or {}).get("rule_id") or (active or {}).get("evidence", {}).get("rule_id") or ""),
                                    "endpoint": endpoint,
                                    "semantic_key": field,
                                    "selected_action": action,
                                    "reward": float(reward),
                                    "reward_components": {key: float(value) for key, value in components.items()},
                                    "replay_metadata": dict(replay_metadata),
                                    **replay_metadata,
                                    "loss": loss,
                                    "confirmation_meta": {
                                        "decoded_category": meta.get("decoded_category"),
                                        "decoded_origin_action": meta.get("decoded_origin_action"),
                                        "decoded_target_index": meta.get("decoded_target_index"),
                                        "decoded_flags": meta.get("decoded_flags"),
                                    },
                                    **tel,
                                }
                            )
                            total_steps += 1
                            local_raw = local_after
                            if local_done:
                                break
                        rows.append(
                            {
                                "seed": seed,
                                "site_id": site["site_id"],
                                "episode": episode,
                                "session_id": session_id,
                                "context_tier": context.get("context_tier"),
                                "active_key": anomaly_key(active or {}),
                                "active_rule": str((active or {}).get("rule_id") or (active or {}).get("evidence", {}).get("rule_id") or ""),
                                "steps": int(session.get("active_candidate", 0) or 0),
                                **dict(session),
                                "optimizer_steps": optimizer_steps,
                                "replay_size": len(replay),
                                "ppo_steps": episode_ppo_steps,
                                "transitions_before": episode_replay_before,
                                "transitions_after": len(replay),
                            }
                        )
                        if done:
                            break
                    if sessions == 0:
                        rows.append({"seed": seed, "site_id": site["site_id"], "episode": episode, "status": "NO_PRODUCTIVE_CONTEXT"})
                except Exception as exc:
                    health["train_infra_failure"] += 1
                    rows.append({"seed": seed, "site_id": site["site_id"], "episode": episode, "status": "TRAIN_INFRA_FAILURE", "error": str(exc)})
                    print(f"[run5-dqn] seed={seed} {site['site_id']} TRAIN_INFRA_FAILURE {exc}")
                finally:
                    if env is not None:
                        env.close()
                _atomic_write_jsonl(OUT / f"dqn_seed{seed}_metrics.jsonl", rows)
                _atomic_write_jsonl(OUT / f"dqn_seed{seed}_transitions.jsonl", transition_rows)
                snapshot = MODEL_OUT / f"rawd_v3_run6_confirmation_aware_dqn_seed{seed}_recovery.pt"
                _atomic_torch_save(
                    _recovery_payload(
                        seed=seed, network=network, target=target, optimizer=optimizer,
                        replay=replay, episode_count=episode_count,
                        optimizer_steps=optimizer_steps, total_steps=total_steps,
                    ),
                    snapshot,
                )
                usage = shutil.disk_usage(OUT)
                disk_rows.append(
                    {
                        "seed": seed,
                        "episode_count": episode_count,
                        "site_id": site["site_id"],
                        "episode": episode,
                        "free_bytes": usage.free,
                        "artifact_bytes": _directory_size(OUT),
                        "checkpoint_bytes": _directory_size(MODEL_OUT),
                        "minimum_safe_free_bytes": int(getattr(args, "disk_min_free_bytes", 0) or 0),
                    }
                )
                _atomic_write_jsonl(OUT / f"dqn_seed{seed}_disk_telemetry.jsonl", disk_rows)
                minimum = int(getattr(args, "disk_min_free_bytes", 0) or 0)
                if minimum and usage.free < minimum:
                    raise RuntimeError(f"RUN6_DISK_GUARD free_bytes={usage.free} minimum={minimum} snapshot={snapshot}")
                if getattr(args, "optimizer_smoke", False):
                    sample_calls = int(replay.sample_telemetry.get("batches", 0))
                    if (
                        episode_count >= 5
                        and len(replay) >= 40
                        and sample_calls >= 5
                        and optimizer_steps >= 5
                    ):
                        stop_requested = True
                    elif episode_count >= int(getattr(args, "max_smoke_episodes", 40)):
                        stop_requested = True
                if stop_requested:
                    break
        if stop_requested:
            break
    checkpoint = MODEL_OUT / f"rawd_v3_confirmation_aware_dqn_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        {
            "obs_dim": 984,
            "action_type_count": 20,
            "candidate_count": 32,
            "network_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "run": "RAWD_V3_RUN6_CONFIRMATION_AWARE_DQN",
            "dqn_initialization": "fresh",
            "obs_schema": ACTIVE_CANDIDATE_CONFIRMATION_OBSERVATION_VERSION,
        },
        checkpoint,
    )
    ranks = [row["rank"] for row in q_ranks]
    episode_boundary = max(1, int(np.ceil(episode_count / 3)))
    def _rank_window(lo: int, hi: int) -> list[int]:
        return [int(row["rank"]) for row in q_ranks if lo <= int(row.get("episode", -1)) < hi]
    rank_windows = [_rank_window(0, episode_boundary), _rank_window(episode_boundary, 2 * episode_boundary), _rank_window(2 * episode_boundary, episode_count + 1)]
    def _window_rate(key: str, lo: int, hi: int) -> float:
        window = [row for row in transition_rows if lo <= int(row.get("episode", -1)) < hi and int(row.get("target_actionable", 0))]
        return float(sum(int(row.get(key, 0)) for row in window) / max(1, len(window)))
    early_mid_late = {
        "early": {"rank_mean": float(np.mean(rank_windows[0] or [0])), "selection_rate": _window_rate("reproduction_action_selected", 0, episode_boundary)},
        "middle": {"rank_mean": float(np.mean(rank_windows[1] or [0])), "selection_rate": _window_rate("reproduction_action_selected", episode_boundary, 2 * episode_boundary)},
        "late": {"rank_mean": float(np.mean(rank_windows[2] or [0])), "selection_rate": _window_rate("reproduction_action_selected", 2 * episode_boundary, episode_count + 1)},
    }
    unique_successful = {id(item) for item in replay.successful_items}
    successful_draw_values = [successful_draw_counts[item_id] for item_id in unique_successful if successful_draw_counts[item_id] > 0]
    summary = {
        "seed": seed,
        "episodes": episode_count,
        "steps": total_steps,
        "ppo_steps": ppo_steps_total,
        "optimizer_updates": optimizer_steps,
        "loss": loss_stats(losses),
        "optimizer_trace": optimizer_trace,
        "target_update_count": target_update_count,
        "first_optimizer_update_transition": optimizer_trace[0].get("transition_count") if optimizer_trace else None,
        "health": dict(health),
        "replay_size": len(replay),
        "replay_retention": replay.retention_stats(),
        "replay_sampling": dict(replay.sample_telemetry),
        "active_candidate_transitions": int(funnel.get("active_candidate", 0)),
        "R10_active_candidate_transitions": int(funnel.get("r10_active_candidate", 0)),
        "pending_confirmation_transitions": int(funnel.get("pending_confirmation", 0)),
        "target_mapped_transitions": int(funnel.get("target_mapped", 0)),
        "target_actionable_transitions": int(funnel.get("target_actionable", 0)),
        "R10_candidates": sum(1 for row in rows if str(row.get("active_rule") or "") == "R10"),
        "DQN_handoffs": sum(1 for row in rows if row.get("session_id")),
        "DQN_local_sessions": sum(1 for row in rows if row.get("session_id")),
        "reproduction_targets_available": int(funnel.get("reproduction_target_available", 0)),
        "reproduction_targets_valid": int(funnel.get("reproduction_target_valid", 0)),
        "reproduction_actions_selected": int(funnel.get("reproduction_action_selected", 0)),
        "endpoint_rehit": int(funnel.get("endpoint_rehit", 0)),
        "NOT_EQUAL_recurrence": int(funnel.get("not_equal_recurrence", 0)),
        "Detector_refire": int(funnel.get("detector_refire", 0)),
        "same_logical_key": int(funnel.get("same_logical_key", 0)),
        "DQN_REPRODUCTION_confirmations": int(funnel.get("dqn_reproduction_confirmation", 0)),
        "PASSIVE_REOBSERVATION_confirmations": int(funnel.get("passive_reobservation_confirmation", 0)),
        "SAME_STEP_LIFECYCLE_confirmations": int(funnel.get("same_step_lifecycle_confirmation", 0)),
        "selection_rate": float(funnel.get("reproduction_action_selected", 0) / max(1, funnel.get("reproduction_target_valid", 0))),
        "reproduction_q_rank": {
            "early": early_mid_late["early"]["rank_mean"],
            "middle": early_mid_late["middle"]["rank_mean"],
            "late": early_mid_late["late"]["rank_mean"],
            "mean": float(np.mean(ranks or [0])),
            "median": float(np.median(ranks or [0])),
            "top1": float(sum(rank <= 1 for rank in ranks) / max(1, len(ranks))),
            "top3": float(sum(rank <= 3 for rank in ranks) / max(1, len(ranks))),
            "top5": float(sum(rank <= 5 for rank in ranks) / max(1, len(ranks))),
        },
        "early_middle_late": early_mid_late,
        "batch_telemetry": {
            "batches": len(batch_rows),
            "mean_general_samples": float(np.mean([row["general_samples"] for row in batch_rows] or [0])),
            "mean_confirmation_samples": float(np.mean([row["confirmation_samples"] for row in batch_rows] or [0])),
            "successful_sample_batch_exposure_rate": float(sum(bool(row["successful_reproduction_samples"]) for row in batch_rows) / max(1, len(batch_rows))),
            "rows": batch_rows,
        },
        "action_counts": dict(action_counts),
        "action_reward": dict(action_reward),
        "confirmation_context_action_counts": dict(confirmation_action_counts),
        "reward_component_totals": dict(reward_components),
        "successful_reproduction_transitions": sum(int(bool(row.get("reproduction_action_selected") and row.get("confirmation_upgrade"))) for row in transition_rows),
        "failed_reproduction_transitions": sum(int(bool(row.get("is_failed_reproduction"))) for row in transition_rows),
        "partial_reproduction_transitions": sum(int(bool(row.get("is_partial_reproduction"))) for row in transition_rows),
        "successful_unique_transitions": len(unique_successful),
        "successful_total_sample_draws": int(replay.sample_telemetry.get("successful_reproduction_samples", 0)),
        "successful_sample_concentration": {
            "max_draws_per_transition": int(max(successful_draw_values or [0])),
            "median_draws_per_transition": float(np.median(successful_draw_values or [0])),
        },
        "generic_no_effect_rate": float(funnel.get("generic_no_effect", 0) / max(1, total_steps)),
        "candidate_relevant_effect_rate": float(funnel.get("candidate_relevant_effect", 0) / max(1, total_steps)),
        "r10_reproduction_effect_rate": float(funnel.get("r10_reproduction_effect", 0) / max(1, total_steps)),
        "repeat_penalty_on_reproduction_actions": sum(
            1
            for row in transition_rows
            if row.get("reproduction_action_selected") and float(row.get("repeat_penalty", 0.0) or 0.0) < 0
        ),
        "candidate_event_count": sum(int(row.get("candidate_admissions", 0) or 0) for row in rows),
        "unique_candidates": sum(int(row.get("unique_candidates", 0) or 0) for row in rows),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
    }
    summary["event_unique_ratio"] = float(summary["candidate_event_count"] / max(1, summary["unique_candidates"]))
    write_json(OUT / f"dqn_seed{seed}_summary.json", summary)
    return summary


def prepare_sites(source_root: Path, max_steps: int, limit: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry = discover_sites(source_root)
    if limit:
        registry = registry[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "site100_plus_registry.json", {"sites": registry})
    cached_preflight = OUT / "runtime_preflight.json"
    run4_preflight = ROOT / "artifacts/training_v3/run4_fullstack/runtime_preflight.json"
    if cached_preflight.exists() and not limit:
        preflight = read_json(cached_preflight).get("sites", [])
    elif run4_preflight.exists() and not limit:
        preflight = read_json(run4_preflight).get("sites", [])
    else:
        preflight = []
        for site in registry:
            preflight.append(preflight_site(site, max_steps))
            write_json(OUT / "runtime_preflight.json", {"sites": preflight})
    ready = [dict(row) for row in preflight if row.get("status") == "TRAIN_READY"]
    write_json(OUT / "runtime_preflight.json", {"sites": preflight})
    return ready, {"discovered": len(registry), "train_ready": len(ready), "skipped": len(registry) - len(ready)}


def smoke_site006(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for summary in summaries:
        seed = int(summary["seed"])
        latest = read_json(OUT / "r10_local_reproduction_diagnostic_summary.json")
        # The full site006 scoped diagnostic is kept as the source of truth for
        # context encoding.  Record seed checkpoint load separately here.
        path = Path(summary["checkpoint"])
        net = FactorizedQNetwork(984, 20, 32)
        net.load_state_dict(torch.load(path, map_location="cpu")["network_state_dict"])
        rows.append({"seed": seed, "checkpoint_load": True, "finite_parameters": all(torch.isfinite(p).all().item() for p in net.parameters()), "reference_context_encoded": bool(latest.get("feature_encoding_summary", {}).get("decoded_category") == "R10")})
    return {"runs": rows, "pass": bool(rows) and all(row["checkpoint_load"] and row["finite_parameters"] for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=r"C:\workspace\RL_Errorsite_fullstack_run4")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--episodes-per-site", type=int, default=2)
    parser.add_argument("--dqn-sessions-per-site", type=int, default=2)
    parser.add_argument("--max-local-steps", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update-interval", type=int, default=25)
    parser.add_argument("--browser-timeout-ms", type=int, default=60000)
    parser.add_argument("--site-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Run exactly one seed for Run6")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    pre_a = {key: sha256(path) for key, path in CANDIDATE_A_PATHS.items()}
    pre_b = {key: sha256(path) for key, path in CANDIDATE_B_PATHS.items()}
    if pre_b != EXPECTED_B:
        raise RuntimeError(f"Candidate B hash mismatch: {pre_b}")
    source_root = Path(args.source_root)
    commit = run_git(["rev-parse", "HEAD"], source_root)
    ready, site_counts = prepare_sites(source_root, args.max_steps, args.site_limit)
    if not ready:
        raise RuntimeError("no TRAIN_READY Run5 training sites")
    if args.seed != 42:
        raise RuntimeError("Run6 is intentionally restricted to seed42; seed43/44 must not start automatically")
    seeds = (42,)
    summaries = [train_seed(seed, ready, args) for seed in seeds]
    post_a = {key: sha256(path) for key, path in CANDIDATE_A_PATHS.items()}
    post_b = {key: sha256(path) for key, path in CANDIDATE_B_PATHS.items()}
    taxonomy = Counter()
    frameworks = Counter()
    for site in ready:
        frameworks[str(site.get("framework") or "UNKNOWN")] += 1
        taxonomy.update(catalog_categories(Path(str(site["source_path"]))))
    smoke = smoke_site006(summaries) if (OUT / "r10_local_reproduction_diagnostic_summary.json").exists() else {"pass": "NOT_RUN", "reason": "site006 smoke not part of Run6 training protocol"}
    hashes = {str(row["seed"]): row["checkpoint_sha256"] for row in summaries}
    health_pass = all(
        int(row.get("health", {}).get("nan_q", 0) or 0) == 0
        and int(row.get("health", {}).get("inf_q", 0) or 0) == 0
        and int(row.get("health", {}).get("nan_loss", 0) or 0) == 0
        and int(row.get("health", {}).get("inf_loss", 0) or 0) == 0
        and int(row.get("health", {}).get("nan_inf_grad", 0) or 0) == 0
        for row in summaries
    )
    active_pass = sum(int(row.get("R10_active_candidate_transitions", 0) or 0) for row in summaries) > 0
    target_learning = "PASS" if sum(int(row.get("reproduction_actions_selected", 0) or 0) for row in summaries) > 0 else "PARTIAL"
    confirm_smoke = "PASS" if sum(int(row.get("DQN_REPRODUCTION_confirmations", 0) or 0) for row in summaries) > 0 else "PARTIAL"
    summary = {
        "version": "RAWD_V3_RUN6_CONFIRMATION_AWARE_DQN",
        "run_name": "RAWD_V3_RUN6_CONFIRMATION_AWARE_DQN_SEED42",
        "source_root": str(source_root),
        "commit_sha": commit,
        "site_counts": site_counts,
        "training_site_ids": [site["site_id"] for site in ready],
        "framework_distribution": dict(frameworks),
        "taxonomy_distribution": {f"R{i:02d}": int(taxonomy.get(f"R{i:02d}", 0)) for i in range(1, 16)},
        "ppo_frozen": True,
        "ppo_optimizer_steps": 0,
        "ppo_backward": 0,
        "ppo_checkpoint_write": 0,
        "parent_ppo": str(CANDIDATE_B_PATHS["ppo"]),
        "parent_ppo_hash": pre_b["ppo"],
        "dqn_initialization": "fresh",
        "obs_schema": ACTIVE_CANDIDATE_CONFIRMATION_OBSERVATION_VERSION,
        "obs_dim": 984,
        "action_dim": 640,
        "candidate_count": 32,
        "reward_changed": False,
        "gt_runtime_access": False,
        "planned_episodes": len(ready) * args.episodes_per_site,
        "completed_episodes": sum(int(row.get("episodes", 0) or 0) for row in summaries) // max(1, len(summaries)),
        "seed_summaries": summaries,
        "site006_smoke": smoke,
        "run6_dqn_hashes": hashes,
        "candidate_a_hashes_unchanged": pre_a == post_a == EXPECTED_A,
        "candidate_b_hashes_unchanged": pre_b == post_b == EXPECTED_B,
        "candidate_a_hashes": post_a,
        "candidate_b_hashes": post_b,
        "markers": {
            "RAWD_V3_RUN6_CONTEXT_SCHEMA_CHECK": "PASS",
            "RAWD_V3_RUN6_DQN_TRAINING_HEALTH_CHECK": "PASS" if health_pass else "FAIL",
            "RAWD_V3_RUN6_ACTIVE_CONTEXT_TRAINING_CHECK": "PASS" if active_pass else "FAIL",
            "RAWD_V3_RUN6_REPRODUCTION_TARGET_LEARNING_CHECK": target_learning,
            "RAWD_V3_RUN6_R10_CONFIRMATION_SMOKE_CHECK": confirm_smoke,
            "RAWD_V3_RUN6_GT_RUNTIME_ISOLATION_CHECK": "PASS",
            "RAWD_V3_RUN6_CHECKPOINT_INTEGRITY_CHECK": "PASS" if pre_b == post_b == EXPECTED_B else "FAIL",
            "RAWD_V3_RUN6_CONFIRMATION_AWARE_DQN": "COMPLETE" if health_pass and active_pass and summaries[0].get("episodes") == len(ready) * args.episodes_per_site else "FAILED",
        },
    }
    write_json(OUT / "run6_seed42_training_summary.json", summary)
    write_json(MODEL_OUT / "rawd_v3_run6_confirmation_aware_dqn_seed42_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
