"""GT-blind, inference-only candidate-context harvest for Run6 lifecycle telemetry."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork
from dqn_error_discovery.runners.train_dqn_error_detector import build_replay_metadata, calculate_dqn_reward
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.productive_context_sampler import classify_context
from dqn_error_discovery.v3.run5_context_aware_dqn_training import (
    CANDIDATE_B_PATHS,
    _dqn_state,
    _step_v3,
    _v3_mask,
    _v3_state,
    active_candidate_from_anomalies,
    active_context_metadata,
    choose_dqn,
    discover_sites,
    q_values,
    read_json,
    reset_env,
    transition_telemetry,
)
from services.anomaly_detection_service import AnomalyConfirmationTracker, detect_anomalies


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(r"C:\workspace\RL_Errorsite_fullstack_run4")
DEFAULT_RUN = ROOT / "artifacts/training_v3/run6_candidate_context_harvest_validation"
DEFAULT_CHECKPOINT = ROOT / "artifacts/models/v3/run6_confirmation_aware_dqn/seed42_full/rawd_v3_confirmation_aware_dqn_seed42.pt"
VITE_SERVER = ROOT / "dqn_error_discovery/v3/run6_vite_programmatic_server.mjs"
SPACE = V3ActionSpace(32)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def is_listening(port: int) -> bool:
    import socket
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def http_health(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            body = response.read(512)
            return {"ok": response.status == 200, "status": response.status, "body_bytes": len(body)}
    except Exception as exc:
        return {"ok": False, "status": None, "body_bytes": 0, "error": repr(exc)}


class ManagedSite:
    def __init__(self, site: Mapping[str, Any], out: Path) -> None:
        self.site = site
        self.out = out
        self.started: list[subprocess.Popen[str]] = []
        self.logs: dict[str, Any] = {}

    def __enter__(self) -> "ManagedSite":
        source = Path(str(self.site["source_path"]))
        site_id = str(self.site["site_id"])
        self.out.mkdir(parents=True, exist_ok=True)
        frontend_port = int(self.site["port"])
        backend_port = int(self.site["backend_port"])
        log_dir = self.out / "server_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if not is_listening(backend_port):
            log = (log_dir / f"{site_id}_backend.log").open("a", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(
                [r"C:\Program Files\nodejs\npm.cmd", "start"],
                cwd=str(source / "backend"), stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            self.started.append(proc)
            self.logs["backend_pid"] = proc.pid
        else:
            self.logs["backend_existing"] = True
        if not is_listening(frontend_port):
            log = (log_dir / f"{site_id}_frontend.log").open("a", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(
                [r"C:\Program Files\nodejs\node.exe", str(VITE_SERVER), "--root", str(source / "frontend"),
                 "--port", str(frontend_port), "--target", f"http://localhost:{backend_port}"],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            self.started.append(proc)
            self.logs["frontend_pid"] = proc.pid
        else:
            self.logs["frontend_existing"] = True
        deadline = time.time() + 25
        health = {"ok": False}
        while time.time() < deadline:
            health = http_health(str(self.site["url"]))
            if health["ok"]:
                break
            time.sleep(0.5)
        self.logs["http"] = health
        self.logs["backend_listening"] = is_listening(backend_port)
        self.logs["frontend_listening"] = is_listening(frontend_port)
        return self

    def __exit__(self, *_exc: object) -> None:
        for proc in reversed(self.started):
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


def manifest_order(source_sites: list[dict[str, Any]], run_root: Path) -> list[dict[str, Any]]:
    by_id = {str(site["site_id"]): site for site in source_sites}
    preflight = read_json(run_root / "runtime_preflight.json").get("sites", [])
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in preflight:
        site_id = str(row.get("site_id") or "")
        if row.get("status") == "TRAIN_READY" and site_id in by_id and site_id not in seen:
            ordered.append(by_id[site_id])
            seen.add(site_id)
    for site in source_sites:
        if str(site["site_id"]) not in seen:
            ordered.append(site)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--run-root", type=Path, default=ROOT / "artifacts/training_v3/run6_confirmation_aware_dqn/seed42_full")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--quota", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--max-sites", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-local-steps", type=int, default=5)
    args = parser.parse_args()

    out = args.artifact_dir
    out.mkdir(parents=True, exist_ok=True)
    network = FactorizedQNetwork(984, 20, 32)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["network_state_dict"])
    network.eval()
    ppo = PPOAgent(635, 640)
    ppo.load(CANDIDATE_B_PATHS["ppo"])

    sites = manifest_order(discover_sites(args.source_root), args.run_root)[:args.max_sites]
    transition_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    infrastructure_errors: list[dict[str, Any]] = []
    funnel = Counter()
    field_present = Counter()
    field_missing = Counter()
    action_counts = Counter()
    episodes = 0
    ppo_steps = 0
    global_step = 0
    server_manifest: list[dict[str, Any]] = []

    for site in sites:
        if episodes >= args.max_episodes or funnel["action_replay_opportunities"] >= args.quota:
            break
        site_row = {"site_id": site["site_id"], "url": site["url"], "episodes": 0, "ppo_steps": 0,
                    "detector_fires": 0, "candidate_proposals": 0, "candidate_admissions": 0,
                    "action_replay_candidates": 0, "observation_recheck_candidates": 0,
                    "dqn_handoffs": 0, "local_transitions": 0}
        with ManagedSite(site, out) as managed:
            server_manifest.append({"site_id": site["site_id"], "url": site["url"], "server": managed.logs})
            if not managed.logs.get("http", {}).get("ok"):
                infrastructure_errors.append({"site_id": site["site_id"], "error": "HTTP_NOT_REACHABLE", "server": managed.logs})
                site_rows["error"] = "HTTP_NOT_REACHABLE"
                site_rows["episodes"] = 0
                site_rows["server"] = managed.logs
                site_rows["episodes"] = site_row["episodes"]
                site_rows["ppo_steps"] = site_row["ppo_steps"]
                site_rows["detector_fires"] = site_row["detector_fires"]
                site_rows["candidate_proposals"] = site_row["candidate_proposals"]
                site_rows["candidate_admissions"] = site_row["candidate_admissions"]
                site_rows["action_replay_candidates"] = site_row["action_replay_candidates"]
                site_rows["observation_recheck_candidates"] = site_row["observation_recheck_candidates"]
                site_rows["dqn_handoffs"] = site_row["dqn_handoffs"]
                site_rows["local_transitions"] = site_row["local_transitions"]
                site_rows.append if False else None
                continue
            for _ in range(2):
                if episodes >= args.max_episodes or funnel["action_replay_opportunities"] >= args.quota:
                    break
                episodes += 1
                site_row["episodes"] += 1
                env = None
                try:
                    env, raw = reset_env(site, args.max_steps, 60000)
                    history: dict[str, Any] = {}
                    sessions = 0
                    for _ppo_step in range(args.max_steps):
                        ppo_steps += 1
                        site_row["ppo_steps"] += 1
                        selected = ppo.select_action(_v3_state(env, raw), _v3_mask(SPACE, raw))
                        aid = int(selected["action_id"])
                        action = {**SPACE.decode(aid), "action_id": aid, "failed": False}
                        after, _, done, info, _ = _step_v3(env, aid, raw, SPACE)
                        action["failed"] = bool(info.get("last_action_error"))
                        anomalies = detect_anomalies(raw, after, {"action": action, "step_id": global_step + 1, "agent_mode": "PPO_GLOBAL"}, site_profile={})
                        funnel["detector_fires"] += len(anomalies)
                        site_row["detector_fires"] += len(anomalies)
                        context = classify_context(raw, after, anomalies, info)
                        active = active_candidate_from_anomalies(anomalies)
                        site_row["candidate_proposals"] += sum(int(bool(a.get("candidate") or a.get("is_candidate") or a.get("status") in {"candidate", "confirmed"})) for a in anomalies)
                        site_row["candidate_admissions"] += int(active is not None)
                        raw = after
                        if active is not None:
                            req = bool(active.get("requires_action_reproduction") or active.get("reproduction_strategy") == "ACTION_REPLAY")
                            if req:
                                site_row["action_replay_candidates"] += 1
                                funnel["action_replay_candidates"] += 1
                            else:
                                site_row["observation_recheck_candidates"] += 1
                                funnel["observation_recheck_candidates"] += 1
                        if not context.get("admit") or sessions >= 2:
                            if done:
                                break
                            continue
                        sessions += 1
                        site_row["dqn_handoffs"] += 1
                        funnel["dqn_handoffs"] += 1
                        tracker = AnomalyConfirmationTracker()
                        local_raw = raw
                        for local_step in range(args.max_local_steps):
                            mask = _v3_mask(SPACE, local_raw)
                            meta = active_context_metadata(history, local_raw, active, mask)
                            state = _dqn_state(env, local_raw, history, active_candidate=active)
                            q = q_values(network, state, mask)
                            local_id = choose_dqn(network, state, mask, 0.0)
                            local_action = {**SPACE.decode(local_id), "action_id": int(local_id), "failed": False}
                            local_after, _, local_done, local_info, _ = _step_v3(env, local_id, local_raw, SPACE)
                            local_action["failed"] = bool(local_info.get("last_action_error"))
                            local_anomalies = detect_anomalies(local_raw, local_after, {"action": local_action, "step_id": global_step + 1, "agent_mode": "DQN_LOCAL", "confirmation_tracker": tracker}, site_profile={})
                            transition_id = f"{episodes}:{site['site_id']}:session{sessions}:{local_step + 1}:{global_step + 1}"
                            history["transition_id"] = transition_id
                            history["current_transition_step"] = global_step + 1
                            reward, components = calculate_dqn_reward(local_raw, local_after, local_action, local_anomalies, history)
                            tel = transition_telemetry(q, mask, int(local_id), local_raw, local_after, active, meta, local_anomalies, components, history)
                            metadata = build_replay_metadata(
                                confirmation_context=bool(meta.get("pending_confirmation") or history.get("last_confirmation_progress") or components.get("confirmation_upgrade", 0.0)),
                                action_replay=bool(meta.get("pending_confirmation") and meta.get("decoded_origin_action") == local_action.get("action_type")),
                                reproduction_attempt=bool(meta.get("pending_confirmation") or history.get("last_confirmation_progress")),
                                candidate_relevant_effect=bool(tel.get("candidate_relevant_effect")),
                                same_chain_recurrence=bool(tel.get("same_chain_recurrence")),
                                confirmation_upgrade=bool(components.get("confirmation_upgrade", 0.0)),
                            )
                            row = {"transition_id": transition_id, "site_id": site["site_id"], "episode": episodes, "local_step": local_step + 1,
                                   "selected_action": local_action, "reward_input": {"pre_pending_records": history.get("last_pre_pending_records", "MISSING"), "event": history.get("last_confirmation_transition", "MISSING")},
                                   "reward_output": {"eligible": history.get("last_confirmation_upgrade_eligible", "MISSING"), "components": components, "final_reward": reward},
                                   "replay_metadata": metadata, **tel}
                            transition_rows.append(row)
                            site_row["local_transitions"] += 1
                            funnel["local_transitions"] += 1
                            for field in ("transition_id", "active_candidate_logical_key", "active_candidate_status_before", "active_candidate_status_after", "requires_action_reproduction", "reproduction_strategy", "reproduction_action_type", "reproduction_target_identity", "selected_action_type", "selected_target_identity", "candidate_relevant_effect", "network_recurrence", "semantic_recurrence", "detector_recurrence", "same_chain_recurrence", "confirmation_event_present", "confirmation_event_id", "confirmation_event_key", "confirmation_source", "confirmation_event_step", "confirmation_event_fresh", "confirmation_upgrade_eligible", "confirmation_upgrade_component", "final_reward"):
                                if row.get(field) == "MISSING" or row.get(field) is None:
                                    field_missing[field] += 1
                                else:
                                    field_present[field] += 1
                            action_counts[str(local_action.get("action_type"))] += 1
                            if row.get("requires_action_reproduction") is True and row.get("target_actionable"):
                                funnel["action_replay_opportunities"] += 1
                            if row.get("is_reproduction_action_selected"):
                                funnel["reproduction_selected"] += 1
                            if row.get("same_chain_recurrence"):
                                funnel["same_chain_confirmations"] += 1
                            if row.get("is_failed_reproduction"):
                                funnel["failed_reproductions"] += 1
                            global_step += 1
                            local_raw = local_after
                            if local_done:
                                break
                        if done:
                            break
                except Exception as exc:
                    infrastructure_errors.append({"site_id": site["site_id"], "episode": episodes, "error": repr(exc)})
                finally:
                    if env is not None:
                        env.close()
        site_rows.append(site_row)

    ar = [r for r in transition_rows if r.get("requires_action_reproduction") is True]
    obs = [r for r in transition_rows if r.get("requires_action_reproduction") is False and r.get("confirmation_event_present")]
    def matrix(group: list[dict[str, Any]]) -> dict[str, Any]:
        tp = sum(bool(r.get("confirmation_upgrade_component", 0.0)) and bool(r.get("same_chain_recurrence")) for r in group)
        fp = sum(bool(r.get("confirmation_upgrade_component", 0.0)) and not bool(r.get("same_chain_recurrence")) for r in group)
        fn = sum(not bool(r.get("confirmation_upgrade_component", 0.0)) and bool(r.get("same_chain_recurrence")) for r in group)
        tn = len(group) - tp - fp - fn
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
    eligible_mismatch = sum((r.get("reward_output", {}).get("eligible") is True) != (float((r.get("reward_output", {}).get("components") or {}).get("confirmation_upgrade", 0.0) or 0.0) > 0) for r in transition_rows)
    duplicate_events = [r for r in transition_rows if r.get("confirmation_event_id") not in {None, "MISSING"}]
    seen_events: set[str] = set()
    duplicate_reward = 0
    for row in duplicate_events:
        event_id = str(row.get("confirmation_event_id"))
        if float((row.get("reward_output", {}).get("components") or {}).get("confirmation_upgrade", 0.0) or 0.0) > 0 and event_id in seen_events:
            duplicate_reward += 1
        seen_events.add(event_id)
    summary = {
        "episodes": episodes, "sites_attempted": len(site_rows), "sites_started": sum(int(bool(row.get("server", {}).get("http", {}).get("ok"))) for row in server_manifest),
        "ppo_global_steps": ppo_steps, "detector_fires": funnel["detector_fires"], "candidate_proposals": sum(r["candidate_proposals"] for r in site_rows),
        "candidate_admissions": sum(r["candidate_admissions"] for r in site_rows), "action_replay_candidates": funnel["action_replay_candidates"],
        "observation_recheck_candidates": funnel["observation_recheck_candidates"], "dqn_handoffs": funnel["dqn_handoffs"], "local_transitions": len(transition_rows),
        "action_replay_opportunities": funnel["action_replay_opportunities"], "reproduction_selections": funnel["reproduction_selected"],
        "same_chain_confirmations": funnel["same_chain_confirmations"], "failed_reproductions": funnel["failed_reproductions"],
        "observation_confirmations": len(obs), "field_present": dict(field_present), "field_missing": dict(field_missing),
        "eligibility_component_mismatch": eligible_mismatch, "duplicate_event_rewarded": duplicate_reward,
        "action_replay_confusion": matrix(ar), "observation_confusion": matrix(obs), "action_counts": dict(action_counts),
        "infrastructure_errors": infrastructure_errors, "optimizer_updates": 0, "backward": 0, "replay_training": 0, "checkpoint_writes": 0,
        "gt_runtime_access": False, "checkpoint_sha256": "23CFDA0E7C752BC70185C212CE50856E957F027625B877BF897D9B3682F53FB9",
        "coverage_classification": (
            "SUFFICIENT_RUNTIME_COVERAGE" if funnel["action_replay_opportunities"] >= args.quota else
            "DETECTOR_FIRE_SCARCE" if funnel["detector_fires"] == 0 else
            "CANDIDATE_ADMISSION_SCARCE" if sum(r["candidate_admissions"] for r in site_rows) == 0 else
            "ACTION_REPLAY_DERIVATION_SCARCE" if funnel["action_replay_candidates"] == 0 else
            "HANDOFF_SCARCE" if funnel["dqn_handoffs"] == 0 else
            "FROZEN_POLICY_LOCAL_CONTEXT_SCARCE"
        ),
    }
    write_json(out / "summary.json", summary)
    write_json(out / "site_funnel.json", site_rows)
    write_json(out / "attribution_matrix.json", {"action_replay": summary["action_replay_confusion"], "observation": summary["observation_confusion"]})
    write_json(out / "server_manifest.json", server_manifest)
    with (out / "runtime_transitions.jsonl").open("w", encoding="utf-8") as handle:
        for row in transition_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("episodes", "sites_attempted", "sites_started", "detector_fires", "candidate_admissions", "action_replay_candidates", "dqn_handoffs", "local_transitions", "action_replay_opportunities", "same_chain_confirmations", "infrastructure_errors", "coverage_classification")}, indent=2))


if __name__ == "__main__":
    main()
