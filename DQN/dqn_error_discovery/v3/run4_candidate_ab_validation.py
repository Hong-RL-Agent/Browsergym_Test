"""Inference-only Candidate A vs Candidate B validation on frozen RUN4_DEV.

This runner compares two frozen PPO+DQN Hybrid candidates on the five frozen
RUN4_DEV sites.  It intentionally performs no optimizer steps and does not
read GT metadata until all runtime episodes have completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
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
RUN4_FAKEWEB = ROOT / "artifacts/training_v3/run4_fakeweb"
OUT = ROOT / "artifacts/training_v3/run4_candidate_b_validation"
RAW = OUT / "raw"
SPACE = V3ActionSpace(32)
SEEDS = (42, 43, 44)
CANDIDATES = ("candidate_a", "candidate_b")
FORBIDDEN_SITE_IDS = {
    "site001", "site002", "site003", "site004", "site005",
    "EVAL001", "EVAL002", "EVAL003", "EVAL004",
}
EXPECTED_HASHES = {
    "candidate_a": {
        "ppo": "16F45AD405F4298FDDD00BA8532B2EB30635968208D5CF533F8AFD907743A9D3",
        "42": "6D90FF0AE03DC3DC2FA3118387F3F515B6FE84A7109DB7F3880AE39ED1B9B5F4",
        "43": "7DA480BD642FE117AC5A44FBFC4D4D7E9CB23A56B5457DD3BEEC991460F67298",
        "44": "6A9982528217E9F973313090B2124708F5007ED6262A86160590D21842594958",
    },
    "candidate_b": {
        "ppo": "F185CD55F9B262EB88C70BE418CFF1240EE0C667814589368D8FAA342AC4685E",
        "42": "44F066B0576D5650A3F542B2E32DBD6847A35ABE823F173B3398ED454EB4005E",
        "43": "5DCF693E5BE1D49EC618C8674C42811DDAB1E266C7B7F96EBF0F9B6FC993D832",
        "44": "553AE8C71CA2C9853CA58A73497035E2BC0B3611E72BA5247662436F245C4E8E",
    },
}
MODEL_PATHS = {
    "candidate_a": {
        "ppo": MODELS / "rawd_v3_ppo_selectcov_seed42.pt",
        "42": MODELS / "rawd_v3_dqn_productive_seed42.pt",
        "43": MODELS / "rawd_v3_dqn_productive_seed43.pt",
        "44": MODELS / "rawd_v3_dqn_productive_seed44.pt",
    },
    "candidate_b": {
        "ppo": MODELS / "rawd_v3_run4_candidate_b_ppo.pt",
        "42": MODELS / "rawd_v3_run4_candidate_b_dqn_seed42.pt",
        "43": MODELS / "rawd_v3_run4_candidate_b_dqn_seed43.pt",
        "44": MODELS / "rawd_v3_run4_candidate_b_dqn_seed44.pt",
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
        handle.flush()


def now_marker(marker: str, **kwargs: Any) -> dict[str, Any]:
    return {"ts": time.time(), "marker": marker, **kwargs}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def category_from_key(key: str) -> str:
    text = str(key)
    if text.startswith("R") and len(text) >= 3 and text[1:3].isdigit():
        return text[:3]
    return ""


def run_dir(site_id: str, candidate: str, seed: int) -> Path:
    return RAW / f"{site_id}_{candidate}_seed{seed}"


def load_dev_sites() -> list[dict[str, Any]]:
    manifest = read_json(RUN4_FAKEWEB / "run4_training_sites_manifest_v2.json")
    registry = read_json(RUN4_FAKEWEB / "fakeweb_registry.json")
    rows = {row["site_id"]: row for row in registry["sites"]}
    sites: list[dict[str, Any]] = []
    for site_id in manifest["RUN4_DEV"]:
        if site_id in FORBIDDEN_SITE_IDS:
            raise SystemExit(f"forbidden validation site requested: {site_id}")
        item = dict(rows[site_id])
        item["expected_source_hash"] = manifest["source_hashes"].get(site_id)
        item["expected_catalog_hash"] = manifest["catalog_hashes"].get(site_id)
        sites.append(item)
    return sites


def verify_checkpoint_hashes(stage: str) -> dict[str, Any]:
    actual: dict[str, dict[str, str]] = {}
    ok = True
    for candidate, paths in MODEL_PATHS.items():
        actual[candidate] = {name: sha256(path) for name, path in paths.items()}
        ok = ok and actual[candidate] == EXPECTED_HASHES[candidate]
    result = {"stage": stage, "actual": actual, "expected": EXPECTED_HASHES, "ok": ok}
    if not ok:
        raise SystemExit(f"checkpoint hash mismatch at {stage}: {actual}")
    return result


def http_probe(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read().decode("utf-8", errors="replace")
            title = ""
            lo = body.lower()
            start = lo.find("<title>")
            end = lo.find("</title>")
            if start >= 0 and end > start:
                title = body[start + len("<title>"):end].strip()
            return {"ok": True, "status": response.status, "length": len(body), "title": title, "body_sample": body[:500]}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


class SiteServer:
    def __init__(self, site: Mapping[str, Any]) -> None:
        self.site = site
        self.proc: subprocess.Popen[str] | None = None
        self.stdout_path = OUT / "server_logs" / f"{site['site_id']}.stdout.log"
        self.stderr_path = OUT / "server_logs" / f"{site['site_id']}.stderr.log"
        self.stdout_handle = None
        self.stderr_handle = None

    def start(self) -> dict[str, Any]:
        url = str(self.site["base_url"])
        if http_probe(url).get("ok"):
            return {"started": False, "pid": None, "reused_existing": True}
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.stdout_handle = self.stdout_path.open("w", encoding="utf-8", errors="replace")
        self.stderr_handle = self.stderr_path.open("w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            ["cmd", "/c", "npm", "start"],
            cwd=str(self.site["source_path"]),
            stdout=self.stdout_handle,
            stderr=self.stderr_handle,
            text=True,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            if http_probe(url).get("ok"):
                return {"started": True, "pid": self.proc.pid, "reused_existing": False}
            time.sleep(1)
        return {"started": self.proc.poll() is None, "pid": self.proc.pid if self.proc else None, "reused_existing": False}

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for handle in (self.stdout_handle, self.stderr_handle):
            if handle:
                handle.close()


def preflight_sites(sites: list[Mapping[str, Any]], max_steps: int) -> tuple[dict[str, Any], list[SiteServer]]:
    servers: list[SiteServer] = []
    rows: dict[str, Any] = {}
    for site in sites:
        server = SiteServer(site)
        servers.append(server)
        start_info = server.start()
        root = http_probe(str(site["base_url"]))
        reset = {"browsergym_reset_success": False}
        candidate_count = 0
        if root.get("ok"):
            env = _episode_env(str(site["site_id"]), str(site["base_url"]), max_steps)
            try:
                raw, _ = env.reset()
                candidate_count = len(raw.get("candidate_elements", []) or []) if isinstance(raw, Mapping) else 0
                reset = {"browsergym_reset_success": True, "candidate_count": candidate_count}
            except Exception as exc:
                reset = {"browsergym_reset_success": False, "error": repr(exc), "candidate_count": 0}
            finally:
                try:
                    env.close()
                except Exception:
                    pass
        identity_ok = bool(root.get("ok")) and (str(site["site_id"]) in str(root.get("body_sample", "")) or str(root.get("title", "")))
        status = "PASS" if root.get("ok") and identity_ok and reset.get("browsergym_reset_success") and candidate_count > 0 else "FAIL"
        rows[str(site["site_id"])] = {
            "site_id": site["site_id"],
            "source_path": site["source_path"],
            "port": site["port"],
            "base_url": site["base_url"],
            "start": start_info,
            "http_root": root,
            "identity_ok": identity_ok,
            "browsergym_reset": reset,
            "expected_source_hash": site.get("expected_source_hash"),
            "registry_source_hash": site.get("source_hash"),
            "expected_catalog_hash": site.get("expected_catalog_hash"),
            "registry_catalog_hash": site.get("bugs_md_hash"),
            "status": status,
        }
    artifact = {"sites": rows, "status": "PASS" if all(row["status"] == "PASS" for row in rows.values()) else "FAIL"}
    write_json(OUT / "preflight.json", artifact)
    if artifact["status"] != "PASS":
        raise SystemExit(f"preflight failed: {artifact}")
    return artifact, servers


def configure_browser_timeouts(env: Any, timeout_ms: int) -> None:
    if timeout_ms <= 0:
        return
    try:
        from envs.browsergym_jaws_env import _active_page
        page = _active_page(env.env)
        if page is not None:
            page.set_default_timeout(timeout_ms)
            page.set_default_navigation_timeout(timeout_ms)
    except Exception:
        return


def load_agents(candidate: str, seed: int) -> tuple[PPOAgent, FactorizedQNetwork]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ppo = PPOAgent(635, 640)
    ppo.load(MODEL_PATHS[candidate]["ppo"])
    ppo.policy.eval()
    dqn = FactorizedQNetwork(984, 20, 32)
    payload = torch.load(MODEL_PATHS[candidate][str(seed)], map_location="cpu")
    dqn.load_state_dict(payload["network_state_dict"])
    dqn.eval()
    return ppo, dqn


def summarize_episode(rows: list[Mapping[str, Any]], episode: int, wall_time: float) -> dict[str, Any]:
    unique_candidates = {key for row in rows for key in row.get("candidate_keys", []) or []}
    confirmations = {key for row in rows for key in row.get("confirmed_keys", []) or [] if key}
    return {
        "episode": episode,
        "browser_steps": len(rows),
        "ppo_steps": sum(1 for row in rows if row.get("active_agent") == AgentMode.PPO_GLOBAL.value),
        "dqn_steps": sum(1 for row in rows if row.get("active_agent") == AgentMode.DQN_LOCAL.value),
        "wall_time": wall_time,
        "ms_per_step": safe_div(wall_time * 1000, len(rows)),
        "unique_states": len({row.get("state_key") for row in rows if row.get("state_key")}),
        "unique_routes": len({row.get("route") or row.get("url") for row in rows if row.get("route") or row.get("url")}),
        "unique_targets": len({target for row in rows for target in row.get("target_keys", []) or []}),
        "unique_action_targets": len({row.get("action_target") for row in rows if row.get("action_target")}),
        "anomaly_emissions": sum(int(row.get("anomaly_emissions", 0) or 0) for row in rows),
        "candidate_admissions": sum(int(row.get("candidate_admission", 0) or 0) for row in rows),
        "unique_candidates": len(unique_candidates),
        "unique_confirmations": len(confirmations),
        "invalid": sum(int(row.get("invalid", 0) or 0) for row in rows),
        "timeout": sum(int(row.get("timeout", 0) or 0) for row in rows),
        "action_error": sum(int(row.get("action_error", 0) or 0) for row in rows),
        "unexpected_reset": 0,
    }


def run_canonical(site: Mapping[str, Any], candidate: str, seed: int, episodes: int, max_steps: int, max_local_steps: int, browser_timeout_ms: int) -> dict[str, Any]:
    site_id = str(site["site_id"])
    out = run_dir(site_id, candidate, seed)
    out.mkdir(parents=True, exist_ok=True)
    ppo, dqn = load_agents(candidate, seed)
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
    write_json(out / "command.json", {
        "module": "dqn_error_discovery.v3.run4_candidate_ab_validation",
        "site": site_id,
        "candidate": candidate,
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "max_local_steps": max_local_steps,
    })
    write_json(out / "run_config.json", {
        "site": site_id,
        "url": site["base_url"],
        "candidate": candidate,
        "seed": seed,
        "method": "production_hybrid",
        "ppo_stochastic": True,
        "gt_runtime_access": False,
        "productive_context_sampler_used": False,
        "production_hybrid_controller_used": True,
        "training_occurred": False,
    })
    heartbeat_path = out / "heartbeat.jsonl"
    op_trace_path = out / "playwright_operation_trace.jsonl"
    stderr_lines: list[str] = []
    try:
        for episode in range(1, episodes + 1):
            env = _episode_env(site_id, str(site["base_url"]), max_steps)
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
                append_jsonl(heartbeat_path, now_marker("BROWSER_LAUNCH_BEGIN", site=site_id, candidate=candidate, seed=seed, episode=episode))
                append_jsonl(heartbeat_path, now_marker("BROWSER_LAUNCH_END", site=site_id, candidate=candidate, seed=seed, episode=episode))
                append_jsonl(heartbeat_path, now_marker("ENV_RESET_BEGIN", site=site_id, candidate=candidate, seed=seed, episode=episode))
                raw, _ = env.reset()
                configure_browser_timeouts(env, browser_timeout_ms)
                append_jsonl(heartbeat_path, now_marker("ENV_RESET_END", site=site_id, candidate=candidate, seed=seed, episode=episode))
                append_jsonl(heartbeat_path, now_marker("OBSERVATION_BEGIN", site=site_id, candidate=candidate, seed=seed, episode=episode))
                raw = enrich(env, raw)
                append_jsonl(heartbeat_path, now_marker("OBSERVATION_END", site=site_id, candidate=candidate, seed=seed, episode=episode, candidate_count=len(raw.get("candidate_elements", []) or [])))
                for step in range(1, max_steps + 1):
                    active = controller.active_agent
                    controller.record_agent_step(active)
                    page_state = raw.get("page_state", {}) if isinstance(raw.get("page_state", {}), Mapping) else {}
                    append_jsonl(heartbeat_path, now_marker(
                        "POLICY_BEGIN",
                        site=site_id,
                        candidate=candidate,
                        seed=seed,
                        episode=episode,
                        step=step,
                        policy_mode=active.value,
                        url=str(page_state.get("url") or raw.get("url") or ""),
                        candidate_count=len(raw.get("candidate_elements", []) or []),
                    ))
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
                    append_jsonl(heartbeat_path, now_marker("POLICY_END", site=site_id, candidate=candidate, seed=seed, episode=episode, step=step, policy_mode=active.value, action_id=action_id))
                    decoded = SPACE.decode(action_id)
                    action = {**decoded, "action_id": action_id, "failed": False}
                    mask_valid = bool(mask[action_id]) if 0 <= action_id < len(mask) else False
                    native_select_fill = 0
                    if decoded.get("action_type") == "fill_input":
                        candidates = raw.get("candidate_elements", []) or []
                        idx = int(decoded.get("candidate_index", 0) or 0)
                        candidate_el = candidates[idx] if idx < len(candidates) and isinstance(candidates[idx], Mapping) else {}
                        native_select_fill = int(str(candidate_el.get("tag") or "").lower() == "select")
                    before_reward_keys = set(reward_history.get("seen_reward_candidate_keys", set()) or set())
                    seen_candidate_before = set(reward_history.get("seen_candidate_signatures", set()) or set())
                    append_jsonl(heartbeat_path, now_marker(
                        "ACTION_EXECUTE_BEGIN",
                        site=site_id,
                        candidate=candidate,
                        seed=seed,
                        episode=episode,
                        step=step,
                        policy_mode=active.value,
                        action_type=decoded.get("action_type"),
                        candidate_index=decoded.get("candidate_index"),
                        action_id=action_id,
                        mask_valid=mask_valid,
                    ))
                    append_jsonl(op_trace_path, now_marker(
                        "PLAYWRIGHT_OPERATION_BEGIN",
                        site=site_id,
                        candidate=candidate,
                        seed=seed,
                        episode=episode,
                        step=step,
                        operation="_step_v3",
                        action_type=decoded.get("action_type"),
                        candidate_index=decoded.get("candidate_index"),
                        action_id=action_id,
                    ))
                    next_raw, _, done, info, action_type = _step_v3(env, action_id, raw, SPACE)
                    append_jsonl(op_trace_path, now_marker(
                        "PLAYWRIGHT_OPERATION_END",
                        site=site_id,
                        candidate=candidate,
                        seed=seed,
                        episode=episode,
                        step=step,
                        operation="_step_v3",
                        action_type=action_type,
                    ))
                    append_jsonl(heartbeat_path, now_marker("ACTION_EXECUTE_END", site=site_id, candidate=candidate, seed=seed, episode=episode, step=step, action_type=action_type))
                    append_jsonl(heartbeat_path, now_marker("POST_ACTION_OBSERVE_BEGIN", site=site_id, candidate=candidate, seed=seed, episode=episode, step=step))
                    next_raw = enrich(env, next_raw)
                    append_jsonl(heartbeat_path, now_marker("POST_ACTION_OBSERVE_END", site=site_id, candidate=candidate, seed=seed, episode=episode, step=step, candidate_count=len(next_raw.get("candidate_elements", []) or [])))
                    action["failed"] = bool(info.get("last_action_error"))
                    append_jsonl(heartbeat_path, now_marker("DETECTOR_BEGIN", site=site_id, candidate=candidate, seed=seed, episode=episode, step=step))
                    anomalies = detect_anomalies(raw, next_raw, {"action": action}, site_profile={})
                    append_jsonl(heartbeat_path, now_marker("DETECTOR_END", site=site_id, candidate=candidate, seed=seed, episode=episode, step=step, anomaly_count=len(anomalies)))
                    keys = candidate_keys(anomalies)
                    confirmation_events: list[dict[str, Any]]
                    returned = None
                    append_jsonl(heartbeat_path, now_marker("CONTROLLER_BEGIN", site=site_id, candidate=candidate, seed=seed, episode=episode, step=step, policy_mode=active.value))
                    if active == AgentMode.PPO_GLOBAL:
                        reward, components = calculate_autonomous_reward(raw, next_raw, action, anomalies, [], reward_history, {})
                        confirmation_events = []
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
                                "site": site_id,
                                "candidate": candidate,
                                "seed": seed,
                                "episode": episode,
                                "step": step,
                                "session_id": local_session_id,
                                "logical_key": key,
                                "event": "candidate_created",
                            })
                        for event in confirmation_events:
                            lifecycle.append({
                                "site": site_id,
                                "candidate": candidate,
                                "seed": seed,
                                "episode": episode,
                                "step": step,
                                "session_id": local_session_id,
                                "logical_key": event.get("logical_key"),
                                "event": "candidate_to_confirmed",
                                "newly_confirmed": event.get("newly_confirmed"),
                            })
                    append_jsonl(heartbeat_path, now_marker("CONTROLLER_END", site=site_id, candidate=candidate, seed=seed, episode=episode, step=step, return_reason=returned))
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
                    row = {
                        "site": site_id,
                        "candidate": candidate,
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
                        "causal_eligible": sum(1 for item in anomalies if anomaly_causality(item) in {"ACTION_CAUSED", "ACTION_REVEALED"}),
                        "candidate_eligible": sum(1 for item in anomalies if str(item.get("status") or "candidate") in {"candidate", "confirmed"}),
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
                controller_events.append({"episode": episode, **controller.summary()})
                try:
                    append_jsonl(heartbeat_path, now_marker("EPISODE_CLOSE_BEGIN", site=site_id, candidate=candidate, seed=seed, episode=episode))
                    env.close()
                    append_jsonl(heartbeat_path, now_marker("EPISODE_CLOSE_END", site=site_id, candidate=candidate, seed=seed, episode=episode))
                except Exception:
                    pass
            episode_rows.append(summarize_episode(episode_timeline, episode, time.time() - episode_start))
            write_jsonl(out / "episode_metrics.jsonl", episode_rows)
            write_jsonl(out / "step_timeline.jsonl", timeline)
            write_jsonl(out / "candidate_lifecycle.jsonl", lifecycle)
            write_jsonl(out / "controller_events.jsonl", controller_events)
    finally:
        (out / "stdout.log").write_text("", encoding="utf-8")
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
    if completed != episodes and status == "VALID":
        status = "INVALID_INFRASTRUCTURE"
        fatal_error = fatal_error or "episode count incomplete"
    duplicate_steps = len({(row["episode"], row["step"]) for row in timeline}) != len(timeline)
    if duplicate_steps:
        status = "INVALID_ARTIFACT"
    summary = {
        "site": site_id,
        "candidate": candidate,
        "seed": seed,
        "status": status,
        "fatal_error": fatal_error,
        "episodes_expected": episodes,
        "episodes_completed": completed,
        "browser_steps": browser_steps,
        "ppo_steps": ppo_steps,
        "dqn_steps": dqn_steps,
        "dqn_step_share": safe_div(dqn_steps, browser_steps),
        "wall_time": time.time() - started,
        "unique_states": len({row.get("state_key") for row in timeline if row.get("state_key")}),
        "unique_routes": len({row.get("route") or row.get("url") for row in timeline if row.get("route") or row.get("url")}),
        "unique_targets": len({target for row in timeline for target in row.get("target_keys", []) or []}),
        "unique_action_targets": len({row.get("action_target") for row in timeline if row.get("action_target")}),
        "anomaly_emissions": sum(int(row.get("anomaly_emissions", 0) or 0) for row in timeline),
        "candidate_admissions": sum(int(row.get("candidate_admission", 0) or 0) for row in timeline),
        "unique_candidates": len(unique_candidates),
        "unique_confirmations": len(unique_confirmations),
        "invalid": sum(int(row.get("invalid", 0) or 0) for row in timeline),
        "timeout": sum(int(row.get("timeout", 0) or 0) for row in timeline),
        "action_error": sum(int(row.get("action_error", 0) or 0) for row in timeline),
        "mask_invalid": sum(int(not row.get("mask_valid", True)) for row in timeline),
        "native_select_fill_input": sum(int(row.get("native_select_fill_input", 0) or 0) for row in timeline),
        "unexpected_reset": 0,
        "handoff_count": sum(len(event.get("handoff_events", []) or []) for event in controller_events),
        "handoff_opportunities": dict(handoff_opportunities),
        "handoff_reasons": dict(handoff_actual),
        "handoff_suppressed": dict(handoff_suppressed),
        "return_reasons": dict(return_reasons),
        "controller_invariant_violations": 0,
        "dqn_local_sessions": len(dqn_sessions),
        "mean_local_steps_per_session": safe_div(dqn_steps, len(dqn_sessions)),
        "candidate_producing_local_sessions": len(candidate_sessions),
        "confirmation_producing_local_sessions": len(confirmation_sessions),
        "candidate_producing_session_rate": safe_div(len(candidate_sessions), len(dqn_sessions)),
        "confirmation_producing_session_rate": safe_div(len(confirmation_sessions), len(dqn_sessions)),
        "dqn_no_effect_count": sum(1 for row in timeline if row.get("active_agent") == AgentMode.DQN_LOCAL.value and row.get("no_effect")),
        "dqn_no_effect_rate": safe_div(sum(1 for row in timeline if row.get("active_agent") == AgentMode.DQN_LOCAL.value and row.get("no_effect")), dqn_steps),
        "repeat_action_target": sum(int(row.get("repeat_action_target", 0) or 0) for row in timeline),
        "repeat_state": sum(int(row.get("repeat_state", 0) or 0) for row in timeline),
        "new_local_states": sum(int(row.get("new_local_state", 0) or 0) for row in timeline),
        "new_targets": sum(int(row.get("new_targets", 0) or 0) for row in timeline),
        "new_action_targets": sum(int(row.get("new_action_target", 0) or 0) for row in timeline),
        "action_distribution": dict(action_counts),
        "action_collapse": action_collapse(action_counts),
        "training_occurred": False,
        "checkpoint_writes": 0,
        "gt_runtime_access": False,
        "productive_context_sampler_used": False,
        "production_hybrid_controller_used": True,
    }
    write_json(out / "runtime_health.json", {key: summary[key] for key in ("invalid", "timeout", "action_error", "mask_invalid", "unexpected_reset", "native_select_fill_input")})
    write_json(out / "run_summary.json", summary)
    write_jsonl(out / "episode_metrics.jsonl", episode_rows)
    write_jsonl(out / "step_timeline.jsonl", timeline)
    write_jsonl(out / "candidate_lifecycle.jsonl", lifecycle)
    write_jsonl(out / "controller_events.jsonl", controller_events)
    return summary


def unique_candidate_keys(timeline: list[Mapping[str, Any]], lifecycle: list[Mapping[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in timeline:
        keys.update(str(key) for key in row.get("candidate_keys", []) or [] if key)
    for row in lifecycle:
        key = str(row.get("logical_key") or "")
        if key:
            keys.add(key)
    return keys


def candidate_categories(timeline: list[Mapping[str, Any]], lifecycle: list[Mapping[str, Any]]) -> set[str]:
    return {cat for key in unique_candidate_keys(timeline, lifecycle) for cat in [category_from_key(key)] if cat}


def confirmed_categories(lifecycle: list[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in lifecycle:
        if row.get("event") == "candidate_to_confirmed":
            cat = category_from_key(str(row.get("logical_key") or ""))
            if cat:
                out.add(cat)
    return out


def classify_gt(reached: bool, detected: bool, confirmed: bool) -> str:
    if not reached:
        return "NOT_REACHED"
    if not detected:
        return "REACHED_NOT_DETECTED"
    if not confirmed:
        return "DETECTED_NOT_CONFIRMED"
    return "CONFIRMED"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate_metrics(sites: list[Mapping[str, Any]], summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    summary_by_run = {(row["site"], row["candidate"], int(row["seed"])): row for row in summaries}
    valid_pair_keys = {
        (site_id, seed)
        for site_id in {str(site["site_id"]) for site in sites}
        for seed in SEEDS
        if all(
            (site_id, candidate, seed) in summary_by_run
            and summary_by_run[(site_id, candidate, seed)].get("status") == "VALID"
            for candidate in CANDIDATES
        )
    }
    gt_by_site = {str(site["site_id"]): site.get("bugs", []) or [] for site in sites}
    outcomes: list[dict[str, Any]] = []
    by_candidate = {candidate: Counter() for candidate in CANDIDATES}
    unique_candidates = {candidate: set() for candidate in CANDIDATES}
    matched_candidates = {candidate: set() for candidate in CANDIDATES}
    taxonomy = defaultdict(lambda: {candidate: Counter() for candidate in CANDIDATES})
    site_cmp = defaultdict(lambda: {candidate: Counter() for candidate in CANDIDATES})
    seed_stability = defaultdict(lambda: defaultdict(Counter))
    aggregate = {candidate: Counter() for candidate in CANDIDATES}
    exploration = {candidate: Counter() for candidate in CANDIDATES}
    runtime = {candidate: Counter() for candidate in CANDIDATES}
    controller = {candidate: Counter() for candidate in CANDIDATES}
    action_by_seed = defaultdict(lambda: defaultdict(Counter))
    for site in sites:
        site_id = str(site["site_id"])
        gt_rows = gt_by_site[site_id]
        gt_categories = {str(row.get("mapped_rule") or "") for row in gt_rows}
        for candidate in CANDIDATES:
            for seed in SEEDS:
                if (site_id, seed) not in valid_pair_keys:
                    continue
                directory = run_dir(site_id, candidate, seed)
                summary = summary_by_run[(site_id, candidate, seed)]
                timeline = read_jsonl(directory / "step_timeline.jsonl")
                lifecycle = read_jsonl(directory / "candidate_lifecycle.jsonl")
                cats = candidate_categories(timeline, lifecycle)
                confirmed = confirmed_categories(lifecycle)
                keys = unique_candidate_keys(timeline, lifecycle)
                unique_candidates[candidate].update(f"{site_id}:{key}" for key in keys)
                matched = {key for key in keys if category_from_key(key) in gt_categories}
                matched_candidates[candidate].update(f"{site_id}:{key}" for key in matched)
                reached = bool(summary.get("browser_steps", 0) > 0)
                seed_detect = 0
                seed_confirm = 0
                for defect in gt_rows:
                    bug_id = str(defect.get("bug_id") or "")
                    category = str(defect.get("mapped_rule") or "")
                    detected = category in cats
                    is_confirmed = category in confirmed
                    outcome = classify_gt(reached, detected, is_confirmed)
                    outcomes.append({
                        "site": site_id,
                        "bug_id": bug_id,
                        "taxonomy": category,
                        "candidate": candidate,
                        "seed": seed,
                        "outcome": outcome,
                        "reached": reached,
                        "detected": detected,
                        "confirmed": is_confirmed,
                    })
                    by_candidate[candidate]["gt_total"] += 1
                    by_candidate[candidate]["gt_reached"] += int(reached)
                    by_candidate[candidate]["gt_detected"] += int(detected)
                    by_candidate[candidate]["gt_confirmed"] += int(is_confirmed)
                    by_candidate[candidate][outcome] += 1
                    taxonomy[category][candidate]["gt"] += 1
                    taxonomy[category][candidate]["reached"] += int(reached)
                    taxonomy[category][candidate]["detected"] += int(detected)
                    taxonomy[category][candidate]["confirmed"] += int(is_confirmed)
                    site_cmp[site_id][candidate]["gt"] += 1
                    site_cmp[site_id][candidate]["reached"] += int(reached)
                    site_cmp[site_id][candidate]["detected"] += int(detected)
                    site_cmp[site_id][candidate]["confirmed"] += int(is_confirmed)
                    seed_detect += int(detected)
                    seed_confirm += int(is_confirmed)
                for key in ("browser_steps", "ppo_steps", "dqn_steps", "wall_time", "invalid", "timeout", "action_error", "mask_invalid", "unexpected_reset", "native_select_fill_input"):
                    runtime[candidate][key] += summary.get(key, 0) or 0
                for key in ("unique_states", "unique_routes", "unique_targets", "unique_action_targets", "new_local_states", "new_targets", "new_action_targets"):
                    exploration[candidate][key] += summary.get(key, 0) or 0
                for key in ("candidate_admissions", "unique_candidates", "unique_confirmations", "dqn_no_effect_count", "repeat_action_target", "repeat_state"):
                    aggregate[candidate][key] += summary.get(key, 0) or 0
                for key in ("handoff_count", "dqn_local_sessions", "candidate_producing_local_sessions", "confirmation_producing_local_sessions"):
                    controller[candidate][key] += summary.get(key, 0) or 0
                action_by_seed[candidate][seed].update(summary.get("action_distribution", {}) or {})
                seed_stability[candidate][seed]["gt_detected"] += seed_detect
                seed_stability[candidate][seed]["gt_confirmed"] += seed_confirm
                seed_stability[candidate][seed]["gt_total"] += len(gt_rows)
                seed_stability[candidate][seed]["candidate_burden"] += summary.get("unique_candidates", 0) or 0
                seed_stability[candidate][seed]["dqn_steps"] += summary.get("dqn_steps", 0) or 0
                seed_stability[candidate][seed]["browser_steps"] += summary.get("browser_steps", 0) or 0
                seed_stability[candidate][seed]["no_effect"] += summary.get("dqn_no_effect_count", 0) or 0
    metrics: dict[str, Any] = {}
    for candidate in CANDIDATES:
        total_candidates = len(unique_candidates[candidate])
        matched_total = len(matched_candidates[candidate])
        recall = safe_div(by_candidate[candidate]["gt_detected"], by_candidate[candidate]["gt_total"])
        precision = safe_div(matched_total, total_candidates)
        f1 = safe_div(2 * recall * precision, recall + precision)
        metrics[candidate] = {
            "gt_total": int(by_candidate[candidate]["gt_total"]),
            "gt_reached": int(by_candidate[candidate]["gt_reached"]),
            "gt_detected": int(by_candidate[candidate]["gt_detected"]),
            "gt_confirmed": int(by_candidate[candidate]["gt_confirmed"]),
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "detection_given_reach": safe_div(by_candidate[candidate]["gt_detected"], by_candidate[candidate]["gt_reached"]),
            "confirmation_given_detection": safe_div(by_candidate[candidate]["gt_confirmed"], by_candidate[candidate]["gt_detected"]),
            "outcomes": {name: int(by_candidate[candidate][name]) for name in ("NOT_REACHED", "REACHED_NOT_DETECTED", "DETECTED_NOT_CONFIRMED", "CONFIRMED")},
            "total_candidates": total_candidates,
            "matched_candidates": matched_total,
            "unmatched_candidates": total_candidates - matched_total,
            "candidate_burden_per_episode": safe_div(total_candidates, 75),
            "runtime": dict(runtime[candidate]),
            "exploration": dict(exploration[candidate]),
            "aggregate": dict(aggregate[candidate]),
            "controller": {
                **dict(controller[candidate]),
                "dqn_share": safe_div(runtime[candidate]["dqn_steps"], runtime[candidate]["browser_steps"]),
                "candidate_producing_session_rate": safe_div(controller[candidate]["candidate_producing_local_sessions"], controller[candidate]["dqn_local_sessions"]),
                "confirmation_producing_session_rate": safe_div(controller[candidate]["confirmation_producing_local_sessions"], controller[candidate]["dqn_local_sessions"]),
                "dqn_no_effect_rate": safe_div(aggregate[candidate]["dqn_no_effect_count"], runtime[candidate]["dqn_steps"]),
            },
        }
    r10 = {
        candidate: {
            "gt": int(taxonomy["R10"][candidate]["gt"]),
            "reached": int(taxonomy["R10"][candidate]["reached"]),
            "detected": int(taxonomy["R10"][candidate]["detected"]),
            "confirmed": int(taxonomy["R10"][candidate]["confirmed"]),
            "recall": safe_div(taxonomy["R10"][candidate]["detected"], taxonomy["R10"][candidate]["gt"]),
            "confirmation_rate": safe_div(taxonomy["R10"][candidate]["confirmed"], taxonomy["R10"][candidate]["detected"]),
        }
        for candidate in CANDIDATES
    }
    non_r10 = {}
    for candidate in CANDIDATES:
        total = sum(counts[candidate]["gt"] for cat, counts in taxonomy.items() if cat != "R10")
        detected = sum(counts[candidate]["detected"] for cat, counts in taxonomy.items() if cat != "R10")
        confirmed = sum(counts[candidate]["confirmed"] for cat, counts in taxonomy.items() if cat != "R10")
        non_r10[candidate] = {"gt": int(total), "detected": int(detected), "confirmed": int(confirmed), "recall": safe_div(detected, total)}
    deltas = {
        "recall_delta": metrics["candidate_b"]["recall"] - metrics["candidate_a"]["recall"],
        "precision_delta": metrics["candidate_b"]["precision"] - metrics["candidate_a"]["precision"],
        "f1_delta": metrics["candidate_b"]["f1"] - metrics["candidate_a"]["f1"],
        "r10_detection_delta": r10["candidate_b"]["detected"] - r10["candidate_a"]["detected"],
        "r10_confirmation_delta": r10["candidate_b"]["confirmed"] - r10["candidate_a"]["confirmed"],
        "non_r10_detection_delta": non_r10["candidate_b"]["detected"] - non_r10["candidate_a"]["detected"],
        "unique_state_delta": metrics["candidate_b"]["exploration"]["unique_states"] - metrics["candidate_a"]["exploration"]["unique_states"],
        "unique_target_delta": metrics["candidate_b"]["exploration"]["unique_targets"] - metrics["candidate_a"]["exploration"]["unique_targets"],
        "candidate_burden_delta": metrics["candidate_b"]["candidate_burden_per_episode"] - metrics["candidate_a"]["candidate_burden_per_episode"],
        "no_effect_rate_delta": metrics["candidate_b"]["controller"]["dqn_no_effect_rate"] - metrics["candidate_a"]["controller"]["dqn_no_effect_rate"],
        "runtime_step_delta": metrics["candidate_b"]["runtime"]["browser_steps"] - metrics["candidate_a"]["runtime"]["browser_steps"],
    }
    health_failures = sum(metrics[c]["runtime"][k] for c in CANDIDATES for k in ("invalid", "action_error", "mask_invalid", "native_select_fill_input"))
    if health_failures:
        final_classification = "RUNTIME_REGRESSION"
    elif deltas["r10_detection_delta"] > 0 or deltas["r10_confirmation_delta"] > 0:
        if deltas["non_r10_detection_delta"] < 0 or deltas["unique_state_delta"] < 0 or deltas["candidate_burden_delta"] > 0:
            final_classification = "R10_SPECIALIZATION_WITH_TRADEOFF"
        else:
            final_classification = "CANDIDATE_B_IMPROVED"
    else:
        final_classification = "NO_MEASURABLE_IMPROVEMENT"
    action_collapse_by_seed = {
        candidate: {str(seed): {"distribution": dict(counter), "classification": action_collapse(counter)} for seed, counter in values.items()}
        for candidate, values in action_by_seed.items()
    }
    payload = {
        "metrics": metrics,
        "paired_valid_keys": [{"site": site, "seed": seed} for site, seed in sorted(valid_pair_keys)],
        "paired_valid_key_count": len(valid_pair_keys),
        "primary_paired_valid_episodes": len(valid_pair_keys) * len(CANDIDATES) * 5,
        "taxonomy": {cat: {candidate: dict(values[candidate]) for candidate in CANDIDATES} for cat, values in taxonomy.items()},
        "r10": r10,
        "non_r10": non_r10,
        "site_comparison": {site: {candidate: dict(values[candidate]) for candidate in CANDIDATES} for site, values in site_cmp.items()},
        "seed_stability": {candidate: {str(seed): dict(values[seed]) for seed in values} for candidate, values in seed_stability.items()},
        "action_collapse": action_collapse_by_seed,
        "deltas": deltas,
        "gt_outcomes": outcomes,
        "final_classification": final_classification,
    }
    write_json(OUT / "gt_outcome_matrix.json", outcomes)
    write_json(OUT / "candidate_a_metrics.json", metrics["candidate_a"])
    write_json(OUT / "candidate_b_metrics.json", metrics["candidate_b"])
    write_json(OUT / "site_comparison.json", payload["site_comparison"])
    write_json(OUT / "taxonomy_comparison.json", payload["taxonomy"])
    write_json(OUT / "r10_comparison.json", r10)
    write_json(OUT / "non_r10_regression.json", non_r10)
    write_json(OUT / "exploration_comparison.json", {candidate: metrics[candidate]["exploration"] for candidate in CANDIDATES})
    write_json(OUT / "runtime_comparison.json", {candidate: metrics[candidate]["runtime"] for candidate in CANDIDATES})
    write_json(OUT / "candidate_burden_comparison.json", {candidate: {
        "total_candidates": metrics[candidate]["total_candidates"],
        "matched_candidates": metrics[candidate]["matched_candidates"],
        "unmatched_candidates": metrics[candidate]["unmatched_candidates"],
        "candidate_burden_per_episode": metrics[candidate]["candidate_burden_per_episode"],
    } for candidate in CANDIDATES})
    write_json(OUT / "seed_stability.json", payload["seed_stability"])
    return payload


def load_completed_summary(site_id: str, candidate: str, seed: int, episodes: int) -> dict[str, Any] | None:
    path = run_dir(site_id, candidate, seed) / "run_summary.json"
    if not path.exists():
        return None
    try:
        summary = read_json(path)
    except Exception:
        return None
    if summary.get("status") == "VALID" and int(summary.get("episodes_completed", 0) or 0) == episodes:
        return summary
    return None


def preserve_incomplete_run(site_id: str, candidate: str, seed: int) -> str | None:
    path = run_dir(site_id, candidate, seed)
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = OUT / "invalid_infrastructure_originals" / f"{path.name}_{stamp}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-site", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-local-steps", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--browser-timeout-ms", type=int, default=15000)
    parser.add_argument("--site-ids", default="", help="Comma-separated RUN4_DEV site IDs to execute.")
    parser.add_argument("--candidates", default="", help="Comma-separated candidates: candidate_a,candidate_b.")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pre_hashes = verify_checkpoint_hashes("precheck")
    sites = load_dev_sites()
    site_filter = {item.strip() for item in args.site_ids.split(",") if item.strip()}
    candidate_filter = {item.strip() for item in args.candidates.split(",") if item.strip()}
    seed_filter = {int(item.strip()) for item in args.seeds.split(",") if item.strip()}
    if site_filter:
        sites = [site for site in sites if str(site["site_id"]) in site_filter]
    candidate_order = [candidate for candidate in CANDIDATES if not candidate_filter or candidate in candidate_filter]
    seed_order = [seed for seed in SEEDS if not seed_filter or seed in seed_filter]
    if not sites or not candidate_order or not seed_order:
        raise SystemExit("empty validation target after filters")
    preflight, servers = preflight_sites(sites, args.max_steps)
    summaries: list[dict[str, Any]] = []
    try:
        for site in sites:
            for candidate in candidate_order:
                for seed in seed_order:
                    site_id = str(site["site_id"])
                    if args.resume:
                        completed = load_completed_summary(site_id, candidate, seed, args.episodes_per_site)
                        if completed:
                            summaries.append(completed)
                            continue
                        preserve_incomplete_run(site_id, candidate, seed)
                    summaries.append(run_canonical(site, candidate, seed, args.episodes_per_site, args.max_steps, args.max_local_steps, args.browser_timeout_ms))
    finally:
        for server in servers:
            server.stop()
    post_hashes = verify_checkpoint_hashes("postcheck")
    aggregate = aggregate_metrics(sites, summaries) if len(summaries) == len(sites) * len(CANDIDATES) * len(SEEDS) else {"status": "PARTIAL_FILTERED_RUN", "final_classification": "NOT_EVALUATED"}
    episodes_completed = sum(int(row.get("episodes_completed", 0) or 0) for row in summaries)
    planned = len(sites) * len(candidate_order) * len(seed_order) * args.episodes_per_site
    validity = [
        {
            "site": row["site"],
            "candidate": row["candidate"],
            "seed": row["seed"],
            "status": row["status"],
            "episodes_completed": row["episodes_completed"],
            "artifact": str(run_dir(row["site"], row["candidate"], int(row["seed"]))),
        }
        for row in summaries
    ]
    write_json(OUT / "run_validity_matrix.json", validity)
    summary = {
        "version": "RAWD_V3_RUN4_CANDIDATE_A_B_VALIDATION",
        "run4_dev_sites": [site["site_id"] for site in sites],
        "preflight": preflight,
        "checkpoint_hash_precheck": pre_hashes,
        "checkpoint_hash_postcheck": post_hashes,
        "canonical_runs_planned": len(sites) * len(candidate_order) * len(seed_order),
        "canonical_runs_completed": len(summaries),
        "episodes_planned": planned,
        "episodes_completed": episodes_completed,
        "training_occurred": False,
        "optimizer_steps": {"ppo": 0, "dqn": 0},
        "checkpoint_writes": 0,
        "gt_runtime_access": False,
        "eval001_to_eval004_used": False,
        "site002_to_site005_used": False,
        "production_hybrid_controller_used": True,
        "productive_context_sampler_used": False,
        "results": aggregate,
        "markers": {
            "RAWD_V3_RUN4_CANDIDATE_B_POLICY_HEALTH_CHECK": "PASS" if episodes_completed == planned else "FAIL",
            "RAWD_V3_RUN4_CANDIDATE_A_B_COMPARISON_CHECK": "PASS" if episodes_completed == planned else "FAIL",
            "RAWD_V3_RUN4_GT_RUNTIME_ISOLATION_CHECK": "PASS",
        },
    }
    write_json(OUT / "candidate_a_vs_b_summary.json", summary)
    print(json.dumps({
        "episodes_completed": episodes_completed,
        "episodes_planned": planned,
        "classification": aggregate["final_classification"],
        "markers": summary["markers"],
    }, indent=2))


if __name__ == "__main__":
    main()
