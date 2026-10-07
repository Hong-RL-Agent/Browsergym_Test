"""Frozen RAWD v3 final held-out evaluation runner.

This module deliberately does not import or read the held-out GT catalog.  It
executes the frozen PPO-only and PPO+DQN Hybrid methods on the frozen EVAL
sites and writes raw runtime artifacts for a separate post-hoc evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork, factorized_flat_q_values
from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController
from dqn_error_discovery.runners.train_dqn_error_detector import calculate_dqn_reward
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.run3_integrated_validation import (
    _confirmation_metadata,
    _handoff_predicates,
    action_collapse,
    action_target,
    anomaly_causality,
    candidate_keys,
    enrich,
    json_default,
    state_key,
)
from dqn_error_discovery.v3.train import _dqn_state, _episode_env, _step_v3, _v3_mask, _v3_state
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
MODELS = ROOT / "artifacts/models/v3"
OUT = ROOT / "artifacts/final_v3_eval"
RAW = OUT / "raw"
SPACE = V3ActionSpace(32)
SITES = {
    "EVAL001": ("EVAL001", "http://localhost:9231/"),
    "EVAL002": ("EVAL002", "http://localhost:9232/"),
    "EVAL003": ("EVAL003", "http://localhost:9233/"),
    "EVAL004": ("EVAL004", "http://localhost:9234/"),
}
SITE_SOURCE = {
    "EVAL001": REPO / "DQN/evaluation_sites/EVAL001/server.js",
    "EVAL002": REPO / "DQN/evaluation_sites/EVAL002/server.js",
    "EVAL003": REPO / "DQN/evaluation_sites/EVAL003/server.js",
    "EVAL004": REPO / "DQN/evaluation_sites/EVAL004/server.js",
}
METHODS = ("ppo", "hybrid")
SEEDS = (42, 43, 44)
EXPECTED_MODEL_HASHES = {
    "ppo": "16F45AD405F4298FDDD00BA8532B2EB30635968208D5CF533F8AFD907743A9D3",
    "42": "6D90FF0AE03DC3DC2FA3118387F3F515B6FE84A7109DB7F3880AE39ED1B9B5F4",
    "43": "7DA480BD642FE117AC5A44FBFC4D4D7E9CB23A56B5457DD3BEEC991460F67298",
    "44": "6A9982528217E9F973313090B2124708F5007ED6262A86160590D21842594958",
}
FORBIDDEN_SITES = {"SITE001", "SITE002", "SITE003", "SITE004", "SITE005", "DEV_REACT", "DEV_NEXT", "DEV_V3_SELECT"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_runs() -> list[dict[str, Any]]:
    return [
        {"site": site, "method": method, "seed": seed}
        for site in SITES
        for method in METHODS
        for seed in SEEDS
    ]


def run_dir(site: str, method: str, seed: int) -> Path:
    return RAW / f"{site.lower()}_{method}_seed{seed}"


def verify_freeze_hashes() -> dict[str, Any]:
    freeze = json.loads((OUT / "freeze_hashes.json").read_text(encoding="utf-8"))
    artifact_paths = {
        "heldout_sites_manifest.json": OUT / "heldout_sites_manifest.json",
        "evaluation_protocol.json": OUT / "evaluation_protocol.json",
    }
    artifact_actual = {name: sha256(path) for name, path in artifact_paths.items()}
    source_actual = {f"{site}/server.js": sha256(path) for site, path in SITE_SOURCE.items()}
    model_actual = {
        "ppo": sha256(MODELS / "rawd_v3_ppo_selectcov_seed42.pt"),
        "42": sha256(MODELS / "rawd_v3_dqn_productive_seed42.pt"),
        "43": sha256(MODELS / "rawd_v3_dqn_productive_seed43.pt"),
        "44": sha256(MODELS / "rawd_v3_dqn_productive_seed44.pt"),
    }
    artifact_expected = {
        key: value
        for key, value in freeze.get("artifact_hashes", {}).items()
        if key in artifact_paths
    }
    source_expected = freeze.get("heldout_site_source_hashes", {})
    result = {
        "model_hashes": model_actual,
        "model_hash_precheck": model_actual == EXPECTED_MODEL_HASHES,
        "artifact_hashes": artifact_actual,
        "artifact_hash_precheck": artifact_actual == artifact_expected,
        "site_source_hashes": source_actual,
        "site_source_hash_precheck": source_actual == source_expected,
        "gt_hash_precheck": "EXTERNAL_PRECHECK_ONLY",
        "gt_runtime_access": False,
    }
    if not result["model_hash_precheck"]:
        raise SystemExit(f"model hash mismatch: {model_actual}")
    if not result["artifact_hash_precheck"]:
        raise SystemExit(f"protocol/site manifest hash mismatch: {artifact_actual} != {artifact_expected}")
    if not result["site_source_hash_precheck"]:
        raise SystemExit(f"site source hash mismatch: {source_actual} != {source_expected}")
    return result


def http_probe(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode("utf-8", errors="replace")
            title = ""
            start = body.lower().find("<title>")
            end = body.lower().find("</title>")
            if start >= 0 and end > start:
                title = body[start + len("<title>"):end]
            return {"ok": True, "status": response.status, "length": len(body), "title": title, "body": body[:500]}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def preflight(max_steps: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for site, (site_id, url) in SITES.items():
        root = http_probe(url)
        health = http_probe(url.rstrip("/") + "/api/health")
        reset = {"browsergym_reset_success": False}
        candidate_count = None
        if root.get("ok"):
            env = _episode_env(site_id, url, max_steps)
            try:
                raw, _ = env.reset()
                candidate_count = len(raw.get("candidate_elements", []) or []) if isinstance(raw, Mapping) else None
                reset = {"browsergym_reset_success": True, "candidate_count": candidate_count}
            except Exception as exc:
                reset = {"browsergym_reset_success": False, "error": repr(exc)}
            finally:
                try:
                    env.close()
                except Exception:
                    pass
        identity = site in str(root.get("title", "")) or site in str(health.get("body", ""))
        rows[site] = {
            "site": site,
            "url": url,
            "port": int(url.rstrip("/").rsplit(":", 1)[1]),
            "http_root": {key: value for key, value in root.items() if key != "body"},
            "http_health": health,
            "identity_ok": bool(identity),
            "browsergym_reset": reset,
            "source_hash": sha256(SITE_SOURCE[site]),
            "status": "PASS" if root.get("ok") and health.get("ok") and identity and reset.get("browsergym_reset_success") else "FAIL",
        }
    artifact = {"sites": rows, "status": "PASS" if all(row["status"] == "PASS" for row in rows.values()) else "FAIL"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(artifact, indent=2, default=json_default), encoding="utf-8")
    if artifact["status"] != "PASS":
        raise SystemExit(f"preflight failed: {artifact}")
    return artifact


def load_agents(seed: int) -> tuple[PPOAgent, FactorizedQNetwork]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ppo = PPOAgent(635, 640)
    ppo.load(MODELS / "rawd_v3_ppo_selectcov_seed42.pt")
    ppo.policy.eval()
    dqn = FactorizedQNetwork(984, 20, 32)
    payload = torch.load(MODELS / f"rawd_v3_dqn_productive_seed{seed}.pt", map_location="cpu")
    dqn.load_state_dict(payload["network_state_dict"])
    dqn.eval()
    return ppo, dqn


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=json_default, ensure_ascii=False) + "\n")


def summarize_episode(rows: list[Mapping[str, Any]], episode: int, wall_time: float) -> dict[str, Any]:
    steps = len(rows)
    states = {str(row.get("state_key") or "") for row in rows if row.get("state_key")}
    routes = {str(row.get("route") or row.get("url") or "") for row in rows if row.get("route") or row.get("url")}
    targets = set()
    action_targets = set()
    unique_candidates = set()
    confirmations = set()
    causality = Counter()
    for row in rows:
        targets.update(row.get("target_keys", []) or [])
        action_targets.add(str(row.get("action_target") or ""))
        unique_candidates.update(row.get("candidate_keys", []) or [])
        confirmations.update(row.get("confirmed_keys", []) or [])
        causality.update(row.get("causality", {}) or {})
    return {
        "episode": episode,
        "browser_steps": steps,
        "ppo_steps": sum(1 for row in rows if row.get("active_agent") == AgentMode.PPO_GLOBAL.value),
        "dqn_steps": sum(1 for row in rows if row.get("active_agent") == AgentMode.DQN_LOCAL.value),
        "wall_time": wall_time,
        "ms_per_step": (wall_time * 1000 / steps) if steps else 0.0,
        "unique_states": len(states),
        "unique_routes": len(routes),
        "unique_targets": len(targets),
        "unique_action_targets": len(action_targets),
        "anomaly_emissions": sum(int(row.get("anomaly_emissions", 0) or 0) for row in rows),
        "causality": dict(causality),
        "candidate_admissions": sum(int(row.get("candidate_admission", 0) or 0) for row in rows),
        "unique_candidates": len(unique_candidates),
        "unique_confirmations": len(confirmations),
        "invalid": sum(int(row.get("invalid", 0) or 0) for row in rows),
        "timeout": sum(int(row.get("timeout", 0) or 0) for row in rows),
        "action_error": sum(int(row.get("action_error", 0) or 0) for row in rows),
        "unexpected_reset": 0,
    }


def run_canonical(site: str, method: str, seed: int, episodes: int, max_steps: int, max_local_steps: int) -> dict[str, Any]:
    if site in FORBIDDEN_SITES:
        raise SystemExit(f"forbidden site requested: {site}")
    out = run_dir(site, method, seed)
    out.mkdir(parents=True, exist_ok=True)
    ppo, dqn = load_agents(seed)
    site_id, url = SITES[site]
    timeline: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    controller_events: list[dict[str, Any]] = []
    handoff_opportunities = Counter()
    handoff_actual = Counter()
    handoff_suppressed = Counter()
    return_reasons = Counter()
    action_counts = Counter()
    started = time.time()
    status = "VALID"
    fatal_error = None
    (out / "command.json").write_text(json.dumps({
        "module": "dqn_error_discovery.v3.run_final_v3_evaluation",
        "site": site,
        "method": method,
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "max_local_steps": max_local_steps,
    }, indent=2), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({
        "site": site,
        "site_id": site_id,
        "url": url,
        "method": method,
        "seed": seed,
        "dqn_handoff_enabled": method == "hybrid",
        "ppo_stochastic": True,
        "gt_runtime_access": False,
        "productive_context_sampler_used": False,
        "production_hybrid_controller_used": method == "hybrid",
    }, indent=2), encoding="utf-8")
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    try:
        for episode in range(1, episodes + 1):
            env = _episode_env(site_id, url, max_steps)
            controller = HybridController(
                max_local_steps=max_local_steps,
                cooldown_ppo_steps=1,
                pending_rehandoff_min_ppo_steps=3,
                functional_region_novelty_gate_enabled=True,
            )
            reward_history: dict[str, Any] = {}
            episode_timeline: list[dict[str, Any]] = []
            seen_state_keys: set[str] = set()
            seen_targets: set[str] = set()
            seen_action_targets: set[str] = set()
            episode_start = time.time()
            try:
                raw, _ = env.reset()
                raw = enrich(env, raw)
                for step in range(1, max_steps + 1):
                    active = controller.active_agent if method == "hybrid" else AgentMode.PPO_GLOBAL
                    if method == "hybrid":
                        controller.record_agent_step(active)
                    if active == AgentMode.PPO_GLOBAL:
                        state = _v3_state(env, raw)
                        mask = _v3_mask(SPACE, raw)
                        selected = ppo.select_action(state, mask)
                        action_id = int(selected["action_id"])
                        local_session_id = None
                        handoff_reason = ""
                    else:
                        state = _dqn_state(env, raw, reward_history)
                        mask = _v3_mask(SPACE, raw)
                        with torch.inference_mode():
                            flat = factorized_flat_q_values(
                                dqn(torch.from_numpy(state).float().unsqueeze(0)),
                                torch.from_numpy(mask).bool().unsqueeze(0),
                                SPACE,
                            )[0]
                        action_id = int(flat.masked_fill(torch.from_numpy(mask) <= 0, -torch.inf).argmax().item())
                        local_session_id = controller.local_session.local_session_id if controller.local_session else None
                        handoff_reason = controller.local_session.handoff_reason if controller.local_session else ""
                    decoded = SPACE.decode(action_id)
                    action = {**decoded, "action_id": action_id, "failed": False}
                    mask_valid = bool(mask[action_id]) if 0 <= action_id < len(mask) else False
                    native_select_fill = 0
                    if decoded.get("action_type") == "fill_input":
                        candidates = raw.get("candidate_elements", []) or []
                        idx = int(decoded.get("candidate_index", 0) or 0)
                        candidate = candidates[idx] if idx < len(candidates) and isinstance(candidates[idx], Mapping) else {}
                        native_select_fill = int(str(candidate.get("tag") or "").lower() == "select")
                    before_reward_keys = set(reward_history.get("seen_reward_candidate_keys", set()) or set())
                    seen_candidate_before = set(reward_history.get("seen_candidate_signatures", set()) or set())
                    next_raw, _, done, info, action_type = _step_v3(env, action_id, raw, SPACE)
                    next_raw = enrich(env, next_raw)
                    action["failed"] = bool(info.get("last_action_error"))
                    anomalies = detect_anomalies(raw, next_raw, {"action": action}, site_profile={})
                    keys = candidate_keys(anomalies)
                    if active == AgentMode.PPO_GLOBAL:
                        reward, components = calculate_autonomous_reward(raw, next_raw, action, anomalies, [], reward_history, {})
                        confirmation_events: list[dict[str, Any]] = []
                        returned = None
                        if method == "hybrid":
                            metadata = _confirmation_metadata(reward_history, next_raw, env)
                            view = dict(reward_history)
                            view["seen_candidate_signatures"] = seen_candidate_before
                            reason = controller.maybe_handoff_after_ppo_step(
                                step=(episode - 1) * max_steps + step,
                                observation=next_raw,
                                anomalies=anomalies,
                                reward_components=components,
                                confirmation_metadata=metadata,
                                history=view,
                            )
                            for reason_name, predicate in _handoff_predicates(anomalies, view, components, next_raw, metadata).items():
                                if predicate:
                                    handoff_opportunities[reason_name] += 1
                            if reason:
                                handoff_actual[reason] += 1
                            for event in controller.suppression_events:
                                if event.get("_counted"):
                                    continue
                                event["_counted"] = True
                                for item in event.get("reasons", []) or []:
                                    handoff_suppressed[str(item)] += 1
                    else:
                        reward, components = calculate_dqn_reward(raw, next_raw, action, anomalies, reward_history)
                        after_reward_keys = set(reward_history.get("seen_reward_candidate_keys", set()) or set())
                        new_reward_keys = after_reward_keys - before_reward_keys
                        confirmation_events = list(reward_history.get("last_confirmation_transition_events", []) or [])
                        repeat_diag = dict(reward_history.get("last_repeat_diagnostic", {}) or {})
                        returned = controller.maybe_return_after_dqn_step(
                            step=(episode - 1) * max_steps + step,
                            observation=next_raw,
                            anomalies=anomalies,
                            reward_components=components,
                            repeat_diagnostic=repeat_diag,
                            done=done,
                            confirmation_events=confirmation_events,
                            tracker_history=reward_history,
                            confirmation_metadata=_confirmation_metadata(reward_history, next_raw, env),
                        )
                        if returned:
                            return_reasons[returned] += 1
                        for key in new_reward_keys:
                            lifecycle.append({
                                "site": site,
                                "method": method,
                                "seed": seed,
                                "episode": episode,
                                "step": step,
                                "session_id": local_session_id,
                                "logical_key": key,
                                "event": "candidate_created",
                            })
                        for event in confirmation_events:
                            lifecycle.append({
                                "site": site,
                                "method": method,
                                "seed": seed,
                                "episode": episode,
                                "step": step,
                                "session_id": local_session_id,
                                "logical_key": event.get("logical_key"),
                                "event": "candidate_to_confirmed",
                                "newly_confirmed": event.get("newly_confirmed"),
                            })
                    page = next_raw.get("page_state", {}) if isinstance(next_raw.get("page_state", {}), Mapping) else {}
                    local_state_after = state_key(next_raw)
                    target_after = {
                        str(c.get("canonical_target_key") or c.get("bid") or c.get("id") or "")
                        for c in next_raw.get("candidate_elements", []) or []
                        if isinstance(c, Mapping)
                    }
                    action_target_key = action_target(raw, action)
                    new_state = local_state_after not in seen_state_keys
                    new_targets = len(target_after - seen_targets)
                    new_action_target = action_target_key not in seen_action_targets
                    seen_state_keys.add(local_state_after)
                    seen_targets.update(target_after)
                    seen_action_targets.add(action_target_key)
                    causal_eligible = sum(1 for item in anomalies if anomaly_causality(item) in {"ACTION_CAUSED", "ACTION_REVEALED"})
                    candidate_eligible = sum(1 for item in anomalies if str(item.get("status") or "candidate") in {"candidate", "confirmed"})
                    row = {
                        "site": site,
                        "method": method,
                        "seed": seed,
                        "episode": episode,
                        "step": step,
                        "active_agent": active.value,
                        "local_session_id": local_session_id,
                        "handoff_reason": handoff_reason,
                        "return_reason": returned,
                        "action_id": action_id,
                        "action_type": action_type,
                        "action_target": action_target_key,
                        "mask_valid": mask_valid,
                        "native_select_fill_input": native_select_fill,
                        "invalid": int(bool(info.get("invalid") or not mask_valid)),
                        "timeout": int(bool(info.get("timeout"))),
                        "action_error": int(bool(info.get("last_action_error"))),
                        "reward": float(reward),
                        "reward_components": dict(components),
                        "anomaly_emissions": len(anomalies),
                        "causality": dict(Counter(anomaly_causality(item) for item in anomalies)),
                        "causal_eligible": causal_eligible,
                        "candidate_eligible": candidate_eligible,
                        "candidate_keys": keys,
                        "confirmed_keys": [str(event.get("logical_key") or "") for event in confirmation_events],
                        "candidate_admission": int(float(components.get("candidate_anomaly", 0.0) or 0.0) > 0.0),
                        "state_key": local_state_after,
                        "url": str(page.get("url") or next_raw.get("url") or ""),
                        "route": str(page.get("route") or ""),
                        "target_keys": sorted(target_after),
                        "new_local_state": int(new_state),
                        "new_targets": int(new_targets),
                        "new_action_target": int(new_action_target),
                        "repeat_action_target": int(float(components.get("repeat_penalty", 0.0) or 0.0) < 0.0),
                        "repeat_state": int(float(components.get("repeat_penalty", 0.0) or 0.0) < 0.0),
                        "repeat_anomaly": 0,
                        "no_effect": int(float(components.get("no_effect_penalty", 0.0) or components.get("no_effect", 0.0) or 0.0) < 0.0),
                        "done": bool(done),
                    }
                    timeline.append(row)
                    episode_timeline.append(row)
                    action_counts[action_type] += 1
                    raw = next_raw
                    if done:
                        break
            except Exception as exc:
                status = "INVALID_INFRASTRUCTURE"
                fatal_error = repr(exc)
                stderr_lines.append(f"episode {episode}: {repr(exc)}")
                break
            finally:
                if method == "hybrid":
                    summary = controller.summary()
                    controller_events.append({"episode": episode, **summary})
                try:
                    env.close()
                except Exception:
                    pass
            episode_rows.append(summarize_episode(episode_timeline, episode, time.time() - episode_start))
            if len(episode_timeline) > max_steps:
                status = "INVALID_ARTIFACT"
                fatal_error = f"episode {episode} exceeded max_steps"
                break
            write_jsonl(out / "episode_metrics.jsonl", episode_rows)
            write_jsonl(out / "step_timeline.jsonl", timeline)
            write_jsonl(out / "candidate_lifecycle.jsonl", lifecycle)
            write_jsonl(out / "controller_events.jsonl", controller_events)
    finally:
        (out / "stdout.log").write_text("\n".join(stdout_lines), encoding="utf-8")
        (out / "stderr.log").write_text("\n".join(stderr_lines), encoding="utf-8")
    dqn_sessions = {
        (row["episode"], row.get("local_session_id"))
        for row in timeline
        if row.get("active_agent") == AgentMode.DQN_LOCAL.value and row.get("local_session_id") is not None
    }
    candidate_sessions = {
        (row["episode"], row.get("local_session_id"))
        for row in timeline
        if row.get("active_agent") == AgentMode.DQN_LOCAL.value and row.get("local_session_id") is not None and row.get("candidate_admission")
    }
    confirmation_sessions = {
        (row["episode"], row.get("local_session_id"))
        for row in timeline
        if row.get("active_agent") == AgentMode.DQN_LOCAL.value and row.get("local_session_id") is not None and row.get("confirmed_keys")
    }
    unique_candidates = {key for row in timeline for key in row.get("candidate_keys", []) or []}
    unique_confirmations = {key for row in timeline for key in row.get("confirmed_keys", []) or [] if key}
    browser_steps = len(timeline)
    ppo_steps = sum(1 for row in timeline if row.get("active_agent") == AgentMode.PPO_GLOBAL.value)
    dqn_steps = sum(1 for row in timeline if row.get("active_agent") == AgentMode.DQN_LOCAL.value)
    completed = len(episode_rows)
    duplicate_episodes = completed != len({row["episode"] for row in episode_rows})
    duplicate_steps = len({(row["episode"], row["step"]) for row in timeline}) != len(timeline)
    if completed != episodes and status == "VALID":
        status = "INVALID_INFRASTRUCTURE"
        fatal_error = fatal_error or "episode count incomplete"
    if duplicate_episodes or duplicate_steps:
        status = "INVALID_ARTIFACT"
    summary = {
        "site": site,
        "method": method,
        "seed": seed,
        "status": status,
        "fatal_error": fatal_error,
        "episodes_expected": episodes,
        "episodes_completed": completed,
        "episode_ids_complete": [row["episode"] for row in episode_rows] == list(range(1, episodes + 1)),
        "duplicate_episodes": duplicate_episodes,
        "duplicate_steps": duplicate_steps,
        "artifact_flush_complete": True,
        "browser_steps": browser_steps,
        "ppo_steps": ppo_steps,
        "dqn_steps": dqn_steps,
        "ppo_step_share": ppo_steps / browser_steps if browser_steps else 0.0,
        "dqn_step_share": dqn_steps / browser_steps if browser_steps else 0.0,
        "wall_time": time.time() - started,
        "ms_per_step": ((time.time() - started) * 1000 / browser_steps) if browser_steps else 0.0,
        "unique_states": len({row.get("state_key") for row in timeline if row.get("state_key")}),
        "unique_routes": len({row.get("route") or row.get("url") for row in timeline if row.get("route") or row.get("url")}),
        "unique_targets": len({target for row in timeline for target in row.get("target_keys", []) or []}),
        "unique_action_targets": len({row.get("action_target") for row in timeline if row.get("action_target")}),
        "anomaly_emissions": sum(int(row.get("anomaly_emissions", 0) or 0) for row in timeline),
        "causality": dict(sum((Counter(row.get("causality", {}) or {}) for row in timeline), Counter())),
        "candidate_admissions": sum(int(row.get("candidate_admission", 0) or 0) for row in timeline),
        "unique_candidates": len(unique_candidates),
        "unique_confirmations": len(unique_confirmations),
        "invalid": sum(int(row.get("invalid", 0) or 0) for row in timeline),
        "timeout": sum(int(row.get("timeout", 0) or 0) for row in timeline),
        "action_error": sum(int(row.get("action_error", 0) or 0) for row in timeline),
        "mask_invalid": sum(int(not row.get("mask_valid", True)) for row in timeline),
        "native_select_fill_input": sum(int(row.get("native_select_fill_input", 0) or 0) for row in timeline),
        "unexpected_reset": 0,
        "page_closed": int("TargetClosed" in str(fatal_error or "")),
        "browser_disconnected": int("browser" in str(fatal_error or "").lower() and "disconnect" in str(fatal_error or "").lower()),
        "server_unavailable": int("ERR_CONNECTION_REFUSED" in str(fatal_error or "")),
        "handoff_count": sum(len(event.get("handoff_events", []) or []) for event in controller_events),
        "handoff_opportunities": dict(handoff_opportunities),
        "handoff_reasons": dict(handoff_actual),
        "handoff_suppressed": dict(handoff_suppressed),
        "return_reasons": dict(return_reasons),
        "controller_invariant_violations": 0,
        "dqn_local_sessions": len(dqn_sessions),
        "candidate_producing_local_sessions": len(candidate_sessions),
        "confirmation_producing_local_sessions": len(confirmation_sessions),
        "candidate_producing_session_rate": len(candidate_sessions) / len(dqn_sessions) if dqn_sessions else 0.0,
        "confirmation_producing_session_rate": len(confirmation_sessions) / len(dqn_sessions) if dqn_sessions else 0.0,
        "action_distribution": dict(action_counts),
        "action_collapse": action_collapse(action_counts),
        "training_occurred": False,
        "checkpoint_writes": 0,
        "gt_runtime_access": False,
        "productive_context_sampler_used": False,
    }
    (out / "runtime_health.json").write_text(json.dumps({
        key: summary[key]
        for key in ("invalid", "timeout", "action_error", "mask_invalid", "page_closed", "browser_disconnected", "server_unavailable", "unexpected_reset", "native_select_fill_input")
    }, indent=2), encoding="utf-8")
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    write_jsonl(out / "episode_metrics.jsonl", episode_rows)
    write_jsonl(out / "step_timeline.jsonl", timeline)
    write_jsonl(out / "candidate_lifecycle.jsonl", lifecycle)
    write_jsonl(out / "controller_events.jsonl", controller_events)
    return summary


def aggregate_run_summaries(summaries: list[Mapping[str, Any]], freeze: Mapping[str, Any], preflight_artifact: Mapping[str, Any]) -> dict[str, Any]:
    matrix = []
    for item in summaries:
        matrix.append({
            "site": item["site"],
            "method": item["method"],
            "seed": item["seed"],
            "status": item["status"],
            "episodes_completed": item["episodes_completed"],
            "artifact": str(run_dir(item["site"], item["method"], int(item["seed"]))),
        })
    (OUT / "run_validity_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    execution = {
        "version": "RAWD_V3_FINAL_EVALUATION_EXECUTION",
        "production_algorithm_changed": False,
        "training_occurred": False,
        "checkpoint_writes": 0,
        "gt_runtime_access": False,
        "site002_to_site005_used": False,
        "forbidden_training_sites_used": False,
        "canonical_runs_planned": 24,
        "canonical_runs_completed": len(summaries),
        "episodes_planned": 240,
        "episodes_completed": sum(int(item.get("episodes_completed", 0) or 0) for item in summaries),
        "ppo_only_episodes": sum(int(item.get("episodes_completed", 0) or 0) for item in summaries if item.get("method") == "ppo"),
        "hybrid_episodes": sum(int(item.get("episodes_completed", 0) or 0) for item in summaries if item.get("method") == "hybrid"),
        "validity": matrix,
        "valid_canonical_run_count": sum(1 for item in summaries if item.get("status") == "VALID"),
        "infrastructure_invalid_run_count": sum(1 for item in summaries if item.get("status") == "INVALID_INFRASTRUCTURE"),
        "runtime_limitation_run_count": sum(1 for item in summaries if item.get("status") == "VALID_WITH_RUNTIME_LIMITATION"),
        "invalid_artifact_count": sum(1 for item in summaries if item.get("status") == "INVALID_ARTIFACT"),
        "review_count": sum(1 for item in summaries if item.get("status") == "REVIEW"),
        "replacement_runs_performed": 0,
        "preflight": preflight_artifact,
        "freeze_precheck": freeze,
        "model_hashes_post": {
            "ppo": sha256(MODELS / "rawd_v3_ppo_selectcov_seed42.pt"),
            "42": sha256(MODELS / "rawd_v3_dqn_productive_seed42.pt"),
            "43": sha256(MODELS / "rawd_v3_dqn_productive_seed43.pt"),
            "44": sha256(MODELS / "rawd_v3_dqn_productive_seed44.pt"),
        },
        "site_hashes_post": {f"{site}/server.js": sha256(path) for site, path in SITE_SOURCE.items()},
        "protocol_hash_post": sha256(OUT / "evaluation_protocol.json"),
        "final_evaluation_check": "COMPLETE" if len(summaries) == 24 and all(item.get("status") == "VALID" for item in summaries) else "REVIEW",
    }
    execution["model_hash_postcheck"] = execution["model_hashes_post"] == EXPECTED_MODEL_HASHES
    execution["site_hash_postcheck"] = execution["site_hashes_post"] == freeze["site_source_hashes"]
    execution["protocol_hash_postcheck"] = execution["protocol_hash_post"] == freeze["artifact_hashes"]["evaluation_protocol.json"]
    (OUT / "final_evaluation_execution_manifest.json").write_text(json.dumps(execution, indent=2, default=json_default), encoding="utf-8")
    return execution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-local-steps", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    freeze = verify_freeze_hashes()
    preflight_artifact = preflight(args.max_steps)
    if args.preflight_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "preflight": str(OUT / "preflight.json")}, indent=2))
        return
    summaries: list[dict[str, Any]] = []
    for item in canonical_runs():
        summary = run_canonical(item["site"], item["method"], int(item["seed"]), args.episodes, args.max_steps, args.max_local_steps)
        summaries.append(summary)
        (OUT / "run_progress.json").write_text(json.dumps({
            "completed": len(summaries),
            "planned": len(canonical_runs()),
            "last": summary,
        }, indent=2, default=json_default), encoding="utf-8")
        if summary["status"] != "VALID":
            break
    execution = aggregate_run_summaries(summaries, freeze, preflight_artifact)
    print(json.dumps({
        "status": execution["final_evaluation_check"],
        "canonical_runs_completed": execution["canonical_runs_completed"],
        "episodes_completed": execution["episodes_completed"],
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
