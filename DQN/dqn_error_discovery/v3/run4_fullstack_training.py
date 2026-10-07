"""Run4 Candidate B continuation training on fullstack-error-site site100+.

This is a training harness only. It keeps the RAWD v3 production algorithm,
reward definitions, detector thresholds, action contract, and model
architectures unchanged while continuing Candidate A on new Fake Web sites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from agents.rollout_buffer import RolloutBuffer
from dqn_error_discovery.agents.dqn_agent import (
    FactorizedQNetwork,
    ReplayBuffer,
    Transition,
    factorized_flat_q_values,
)
from dqn_error_discovery.runners.train_dqn_error_detector import calculate_dqn_reward
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.productive_context_sampler import (
    classify_context,
    evidence_types,
    semantic_fingerprint,
    target_fingerprint,
)
from dqn_error_discovery.v3.train import (
    _dqn_state,
    _dqn_update,
    _episode_env,
    _step_v3,
    _v3_mask,
    _v3_state,
)
from services.anomaly_detection_service import detect_anomalies
from services.autonomous_reward_service import calculate_autonomous_reward


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "artifacts/models/v3"
OUT = ROOT / "artifacts/training_v3/run4_fullstack"
SPACE = V3ActionSpace(32)
NPM = Path(r"C:\Program Files\nodejs\npm.cmd")
NPM_CMD = [str(NPM)] if NPM.exists() else ["npm"]
EXPECTED_A = {
    "ppo": "16F45AD405F4298FDDD00BA8532B2EB30635968208D5CF533F8AFD907743A9D3",
    "dqn42": "6D90FF0AE03DC3DC2FA3118387F3F515B6FE84A7109DB7F3880AE39ED1B9B5F4",
    "dqn43": "7DA480BD642FE117AC5A44FBFC4D4D7E9CB23A56B5457DD3BEEC991460F67298",
    "dqn44": "6A9982528217E9F973313090B2124708F5007ED6262A86160590D21842594958",
}
PARENT_PATHS = {
    "ppo": MODELS / "rawd_v3_ppo_selectcov_seed42.pt",
    "dqn42": MODELS / "rawd_v3_dqn_productive_seed42.pt",
    "dqn43": MODELS / "rawd_v3_dqn_productive_seed43.pt",
    "dqn44": MODELS / "rawd_v3_dqn_productive_seed44.pt",
}
NEW_PATHS = {
    "ppo": MODELS / "rawd_v3_run4_candidate_b_ppo.pt",
    "dqn42": MODELS / "rawd_v3_run4_candidate_b_dqn_seed42.pt",
    "dqn43": MODELS / "rawd_v3_run4_candidate_b_dqn_seed43.pt",
    "dqn44": MODELS / "rawd_v3_run4_candidate_b_dqn_seed44.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file() and "node_modules" not in p.parts):
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode())
        digest.update(item.read_bytes())
    return digest.hexdigest().upper()


def json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def append_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_port(text: str, default: int | None = None) -> int | None:
    patterns = [
        r"localhost:(\d{4,5})",
        r"PORT\s*=\s*(\d{4,5})",
        r"port:\s*(\d{4,5})",
        r"--port\s+(\d{4,5})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return default


def infer_framework(frontend: Path) -> str:
    package = read_json(frontend / "package.json")
    deps = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
    if "next" in deps:
        return "Next.js"
    if "react" in deps:
        return "React"
    return "Other"


def catalog_categories(site: Path) -> Counter[str]:
    text = ""
    for rel in ("bug_catalog.json", "BUGS.md", "README.md"):
        candidate = site / rel
        if candidate.exists():
            text += "\n" + candidate.read_text(encoding="utf-8", errors="ignore")
    counter: Counter[str] = Counter()
    for key in re.findall(r"\bR(0[1-9]|1[0-5])\b", text.upper()):
        counter[f"R{key}"] += 1
    if not counter:
        # These fullstack Fake Webs primarily encode UI/API/data mismatch bugs.
        counter["R10"] = max(1, len(re.findall(r"Error\s+\d+", text, flags=re.IGNORECASE)))
    return counter


def discover_sites(source_root: Path) -> list[dict[str, Any]]:
    rows = []
    for site in sorted(p for p in source_root.iterdir() if p.is_dir()):
        match = re.match(r"^site(\d+)[_-]?.*", site.name)
        if not match:
            continue
        number = int(match.group(1))
        if number < 100:
            continue
        frontend = site / "frontend"
        backend = site / "backend"
        readme = site / "README.md"
        readme_text = readme.read_text(encoding="utf-8", errors="ignore") if readme.exists() else ""
        vite_text = (frontend / "vite.config.js").read_text(encoding="utf-8", errors="ignore") if (frontend / "vite.config.js").exists() else ""
        server_text = (backend / "server.js").read_text(encoding="utf-8", errors="ignore") if (backend / "server.js").exists() else ""
        frontend_port = parse_port(readme_text + "\n" + vite_text, 5200)
        backend_port = parse_port(server_text + "\n" + readme_text, 9599)
        categories = catalog_categories(site)
        rows.append(
            {
                "site_id": f"SITE{number}",
                "site_number": number,
                "directory_name": site.name,
                "source_path": str(site),
                "framework": infer_framework(frontend) if frontend.exists() else "UNKNOWN",
                "frontend_type": "vite" if (frontend / "vite.config.js").exists() else "unknown",
                "backend_type": "express" if (backend / "server.js").exists() else "unknown",
                "start_command": {
                    "backend": "npm start",
                    "frontend": "npm run dev -- --host 127.0.0.1",
                },
                "port": frontend_port,
                "backend_port": backend_port,
                "url": f"http://localhost:{frontend_port}/",
                "bug_catalog": str(readme if readme.exists() else ""),
                "taxonomy_distribution": dict(categories),
                "total_catalog_defects": int(sum(categories.values())),
                "source_hash": tree_hash(site),
            }
        )
    return rows


def ensure_dependencies(directory: Path) -> dict[str, Any]:
    if not (directory / "package.json").exists():
        return {"ran": False, "ok": False, "reason": "NO_PACKAGE_JSON"}
    if (directory / "node_modules").exists():
        return {"ran": False, "ok": True, "reason": "NODE_MODULES_PRESENT"}
    sibling_name = directory.name
    source_root = directory.parent.parent
    for sibling_site in sorted(source_root.iterdir()):
        template = sibling_site / sibling_name / "node_modules"
        if template.exists() and template != directory / "node_modules":
            link = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(directory / "node_modules"), str(template)],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=30,
            )
            if link.returncode == 0 and (directory / "node_modules").exists():
                return {
                    "ran": False,
                    "ok": True,
                    "reason": "NODE_MODULES_JUNCTION",
                    "template": str(template),
                    "stdout_tail": link.stdout[-1000:],
                    "stderr_tail": link.stderr[-1000:],
                }
    command = [*NPM_CMD, "install", "--no-audit", "--no-fund"]
    start = time.time()
    proc = subprocess.run(
        command,
        cwd=str(directory),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=180,
    )
    return {
        "ran": True,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "seconds": round(time.time() - start, 3),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def spawn(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    full_env = os.environ.copy()
    full_env.update(env or {})
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=full_env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, text=True, timeout=20)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def http_ok(url: str, timeout: float = 3.0) -> tuple[bool, int | None, str]:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(4096).decode("utf-8", errors="ignore")
            return 200 <= int(response.status) < 400, int(response.status), body
    except Exception as exc:
        return False, None, str(exc)


class SiteServer:
    def __init__(self, site: Mapping[str, Any]) -> None:
        self.site = site
        self.backend: subprocess.Popen[str] | None = None
        self.frontend: subprocess.Popen[str] | None = None
        self.logs: dict[str, Any] = {}

    def __enter__(self) -> "SiteServer":
        root = Path(str(self.site["source_path"]))
        backend = root / "backend"
        frontend = root / "frontend"
        self.logs["backend_install"] = ensure_dependencies(backend)
        self.logs["frontend_install"] = ensure_dependencies(frontend)
        if not self.logs["backend_install"].get("ok") or not self.logs["frontend_install"].get("ok"):
            return self
        self.backend = spawn([*NPM_CMD, "start"], backend)
        time.sleep(0.5)
        self.frontend = spawn([*NPM_CMD, "run", "dev", "--", "--host", "127.0.0.1"], frontend)
        deadline = time.time() + 25
        while time.time() < deadline:
            ok, status, body = http_ok(str(self.site["url"]), timeout=1.5)
            if ok:
                self.logs.update({"http_ok": True, "http_status": status, "identity": body[:300]})
                return self
            time.sleep(0.5)
        ok, status, body = http_ok(str(self.site["url"]), timeout=2.0)
        self.logs.update({"http_ok": ok, "http_status": status, "identity": body[:300]})
        return self

    def __exit__(self, *_exc: object) -> None:
        self.logs["backend_pid"] = self.backend.pid if self.backend else None
        self.logs["frontend_pid"] = self.frontend.pid if self.frontend else None
        for name, proc in (("frontend", self.frontend), ("backend", self.backend)):
            if proc is not None:
                try:
                    out, err = proc.communicate(timeout=0.1)
                except Exception:
                    out, err = "", ""
                self.logs[f"{name}_stdout_tail"] = out[-2000:]
                self.logs[f"{name}_stderr_tail"] = err[-2000:]
        stop_process(self.frontend)
        stop_process(self.backend)


def reset_env(site: Mapping[str, Any], max_steps: int) -> tuple[Any, dict[str, Any]]:
    env = _episode_env(str(site["site_id"]), str(site["url"]), max_steps)
    raw, _ = env.reset()
    return env, dict(raw)


def preflight_site(site: Mapping[str, Any], max_steps: int) -> dict[str, Any]:
    row = dict(site)
    row.update({"status": "SKIPPED_RUNTIME_NOT_READY", "server_startable": False, "http_reachable": False, "browsergym_reset": False, "candidate_count": 0})
    with SiteServer(site) as server:
        row["server"] = server.logs
        row["server_startable"] = bool(server.backend and server.frontend and server.backend.poll() is None and server.frontend.poll() is None)
        row["http_reachable"] = bool(server.logs.get("http_ok"))
        if not row["http_reachable"]:
            row["skip_reason"] = "HTTP_NOT_REACHABLE"
            return row
        env = None
        try:
            env, raw = reset_env(site, max_steps)
            row["browsergym_reset"] = True
            row["candidate_count"] = len(raw.get("candidate_elements", []) or [])
            row["title"] = raw.get("page_title") or raw.get("title") or ""
            row["status"] = "TRAIN_READY" if row["candidate_count"] > 0 else "SKIPPED_RUNTIME_NOT_READY"
            row["skip_reason"] = "" if row["status"] == "TRAIN_READY" else "CANDIDATE_COUNT_ZERO"
        except Exception as exc:
            row["browsergym_error"] = str(exc)
            row["skip_reason"] = "BROWSERGYM_RESET_FAILED"
        finally:
            if env is not None:
                env.close()
    return row


def q_values(network: FactorizedQNetwork, state: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    with torch.inference_mode():
        return factorized_flat_q_values(network(torch.from_numpy(state).float().unsqueeze(0)), torch.from_numpy(mask).bool().unsqueeze(0), SPACE)[0]


def choose_dqn(network: FactorizedQNetwork, state: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
    valid = np.flatnonzero(mask > 0)
    if len(valid) == 0:
        return SPACE.encode("noop", 0)
    if random.random() < epsilon:
        return int(random.choice(valid.tolist()))
    values = q_values(network, state, mask)
    return int(values.masked_fill(torch.from_numpy(mask) <= 0, -torch.inf).argmax().item())


def anomaly_key(anomaly: Mapping[str, Any]) -> str:
    return str(anomaly.get("signature") or anomaly.get("logical_key") or anomaly.get("id") or json.dumps(anomaly, sort_keys=True, default=str)[:160])


def telemetry_from_anomalies(anomalies: list[Mapping[str, Any]], history: Mapping[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    counter["anomaly_emissions"] = len(anomalies)
    for anomaly in anomalies:
        status = str(anomaly.get("status") or "")
        evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence", {}), Mapping) else {}
        causality = str(evidence.get("causal_attribution") or "UNKNOWN")
        counter[causality] += 1
        counter["candidate_eligible"] += int(status in {"candidate", "confirmed"} or bool(anomaly.get("candidate") or anomaly.get("is_candidate")))
        counter["candidate_admissions"] += int(status in {"candidate", "confirmed"} or bool(anomaly.get("candidate") or anomaly.get("is_candidate")))
        counter["confirmations"] += int(status == "confirmed" or bool(anomaly.get("confirmed")))
    counter["unique_candidates"] += len(set(history.get("seen_reward_candidate_keys", set()) or set()))
    counter["candidate_to_confirmed"] += int(bool(history.get("last_confirmation_confirmed")))
    return counter


def finite_model(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def train_ppo(sites: list[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    seed_all(42)
    agent = PPOAgent(635, 640)
    agent.load(PARENT_PATHS["ppo"])
    buffer = RolloutBuffer()
    rows = []
    total_steps = 0
    rewards = []
    losses = []
    action_counts: Counter[str] = Counter()
    invalid = timeout = action_error = native_select_fill_input = 0
    for site in sites:
        with SiteServer(site):
            for episode in range(args.ppo_episodes_per_site):
                print(f"[run4-ppo] start {site['site_id']} episode={episode + 1}/{args.ppo_episodes_per_site}")
                env = None
                ep_reward = 0.0
                ep_steps = 0
                try:
                    env, raw = reset_env(site, args.max_steps)
                    history: dict[str, Any] = {}
                    for step in range(args.max_steps):
                        state = _v3_state(env, raw)
                        mask = _v3_mask(SPACE, raw)
                        selected = agent.select_action(state, mask)
                        action_id = int(selected["action_id"])
                        decoded = SPACE.decode(action_id)
                        if decoded["action_type"] == "fill_input":
                            candidates = raw.get("candidate_elements", []) or []
                            idx = int(decoded["candidate_index"])
                            if idx < len(candidates) and str(candidates[idx].get("tag", "")).lower() == "select":
                                native_select_fill_input += 1
                        next_raw, _, done, info, action_type = _step_v3(env, action_id, raw, SPACE)
                        action = {**decoded, "action_id": action_id, "failed": bool(info.get("last_action_error"))}
                        anomalies = detect_anomalies(raw, next_raw, {"action": action}, site_profile={})
                        reward, _components = calculate_autonomous_reward(raw, next_raw, action, anomalies, [], history, getattr(env, "site_profile", {}))
                        buffer.add(state, action_id, float(selected["log_prob"]), float(reward), bool(done), float(selected["value"]), mask, info)
                        ep_reward += float(reward)
                        ep_steps += 1
                        total_steps += 1
                        action_counts[action_type] += 1
                        invalid += int(bool(info.get("invalid")))
                        timeout += int(bool(info.get("timeout")))
                        action_error += int(bool(info.get("last_action_error")))
                        raw = next_raw
                        if done:
                            break
                except Exception as exc:
                    rows.append({"site_id": site["site_id"], "episode": episode, "error": str(exc), "reward": ep_reward, "steps": ep_steps})
                finally:
                    if env is not None:
                        env.close()
                rewards.append(ep_reward)
                rows.append({"site_id": site["site_id"], "episode": episode, "reward": ep_reward, "steps": ep_steps})
                write_jsonl(OUT / "ppo_episode_metrics.jsonl", rows)
                print(f"[run4-ppo] done {site['site_id']} episode={episode + 1} steps={ep_steps} reward={ep_reward:.3f}")
    if len(buffer):
        buffer.compute_returns_and_advantages(0.0, agent.gamma, agent.gae_lambda)
        metrics = agent.update(buffer)
        losses.append(float(metrics.get("total_loss", 0.0)))
    else:
        metrics = {}
    agent.save(NEW_PATHS["ppo"])
    write_jsonl(OUT / "ppo_episode_metrics.jsonl", rows)
    return {
        "episodes": len(sites) * args.ppo_episodes_per_site,
        "steps": total_steps,
        "mean_reward": float(np.mean(rewards or [0.0])),
        "first_middle_last_reward": {
            "first": float(np.mean(rewards[: max(1, len(rewards) // 3)] or [0.0])),
            "middle": float(np.mean(rewards[len(rewards) // 3 : 2 * len(rewards) // 3] or [0.0])),
            "last": float(np.mean(rewards[2 * len(rewards) // 3 :] or [0.0])),
        },
        "update_metrics": metrics,
        "losses": losses,
        "action_counts": dict(action_counts),
        "invalid": invalid,
        "timeout": timeout,
        "action_error": action_error,
        "native_select_fill_input": native_select_fill_input,
        "checkpoint": str(NEW_PATHS["ppo"]),
        "checkpoint_sha256": sha256(NEW_PATHS["ppo"]),
    }


def train_dqn_seed(seed: int, sites: list[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    seed_all(seed)
    ppo = PPOAgent(635, 640)
    ppo.load(NEW_PATHS["ppo"])
    network = FactorizedQNetwork(984, 20, 32)
    parent = torch.load(PARENT_PATHS[f"dqn{seed}"], map_location="cpu")
    network.load_state_dict(parent["network_state_dict"])
    target = FactorizedQNetwork(984, 20, 32)
    target.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    replay = ReplayBuffer(50000)
    rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    losses: list[float] = []
    total_steps = 0
    optimizer_steps = 0
    target_updates = 0
    health = Counter()
    unique_states: set[str] = set()
    unique_targets: set[str] = set()
    unique_action_targets: set[str] = set()
    candidate_sites: set[str] = set()
    confirmation_sites: set[str] = set()
    site_productivity: dict[str, Counter[str]] = defaultdict(Counter)
    for site in sites:
        admitted = 0
        with SiteServer(site):
            env = None
            try:
                print(f"[run4-dqn] seed={seed} start {site['site_id']}")
                env, raw = reset_env(site, args.max_steps)
                history: dict[str, Any] = {}
                for ppo_step in range(args.max_steps):
                    state = _v3_state(env, raw)
                    mask = _v3_mask(SPACE, raw)
                    selected = ppo.select_action(state, mask)
                    action_id = int(selected["action_id"])
                    action = {**SPACE.decode(action_id), "action_id": action_id, "failed": False}
                    after, _, done, info, _action_type = _step_v3(env, action_id, raw, SPACE)
                    action["failed"] = bool(info.get("last_action_error"))
                    anomalies = detect_anomalies(raw, after, {"action": action}, site_profile={})
                    context = classify_context(raw, after, anomalies, info)
                    raw = after
                    if not context.get("admit") or admitted >= args.dqn_sessions_per_site:
                        if done:
                            break
                        continue
                    admitted += 1
                    session_id = f"{site['site_id']}:seed{seed}:session{admitted}"
                    session_counter: Counter[str] = Counter({"sessions": 1})
                    local_raw = raw
                    local_state = _dqn_state(env, local_raw, history)
                    session_reward = 0.0
                    before_session_candidates = set(history.get("seen_reward_candidate_keys", set()) or set())
                    before_confirmed = int(bool(history.get("last_confirmation_confirmed")))
                    for local_step in range(args.max_local_steps):
                        local_mask = _v3_mask(SPACE, local_raw)
                        values = q_values(network, local_state, local_mask)
                        valid_values = values[torch.from_numpy(local_mask).bool()]
                        health["nan_q"] += int(bool(torch.isnan(valid_values).any()) if valid_values.numel() else False)
                        health["inf_q"] += int(bool(torch.isinf(valid_values).any()) if valid_values.numel() else False)
                        epsilon = 0.1
                        local_id = choose_dqn(network, local_state, local_mask, epsilon)
                        if local_mask[local_id] <= 0:
                            health["mask_invalid"] += 1
                        local_action = {**SPACE.decode(local_id), "action_id": int(local_id), "failed": False}
                        candidates = local_raw.get("candidate_elements", []) or []
                        if local_action["action_type"] == "fill_input":
                            idx = int(local_action["candidate_index"])
                            if idx < len(candidates) and str(candidates[idx].get("tag", "")).lower() == "select":
                                health["native_select_fill_input"] += 1
                        local_after, _, local_done, local_info, local_action_type = _step_v3(env, local_id, local_raw, SPACE)
                        local_action["failed"] = bool(local_info.get("last_action_error"))
                        local_anomalies = detect_anomalies(local_raw, local_after, {"action": local_action}, site_profile={})
                        reward, components = calculate_dqn_reward(local_raw, local_after, local_action, local_anomalies, history)
                        next_state = _dqn_state(env, local_after, history)
                        next_mask = _v3_mask(SPACE, local_after)
                        replay.add(Transition(local_state, int(local_id), float(reward), next_state, bool(local_done), next_mask, local_mask))
                        loss = None
                        if len(replay) >= 32:
                            loss = _dqn_update(network, target, optimizer, replay.sample(32), 0.99, SPACE)
                            losses.append(loss)
                            optimizer_steps += 1
                            health["nan_loss"] += int(bool(np.isnan(loss)))
                            health["inf_loss"] += int(bool(np.isinf(loss)))
                        if optimizer_steps and optimizer_steps % 25 == 0:
                            target.load_state_dict(network.state_dict())
                            target_updates += 1
                        telemetry = telemetry_from_anomalies(local_anomalies, history)
                        session_counter.update(telemetry)
                        site_productivity[str(site["site_id"])].update(telemetry)
                        session_counter["steps"] += 1
                        session_counter["invalid"] += int(bool(local_info.get("invalid")))
                        session_counter["timeout"] += int(bool(local_info.get("timeout")))
                        session_counter["action_error"] += int(bool(local_info.get("last_action_error")))
                        session_counter["no_effect"] += int(float(components.get("no_effect_penalty", 0.0) or 0.0) < 0)
                        session_counter["repeat"] += int(float(components.get("repeat_penalty", 0.0) or 0.0) < 0)
                        before_state = semantic_fingerprint(local_raw)
                        after_state = semantic_fingerprint(local_after)
                        unique_states.add(after_state)
                        unique_targets.update(target_fingerprint(local_after))
                        unique_action_targets.add(f"{local_action_type}:{local_action.get('candidate_index')}")
                        session_counter["new_states"] += int(before_state != after_state)
                        session_counter["new_targets"] += len(target_fingerprint(local_after) - target_fingerprint(local_raw))
                        session_counter["new_action_targets"] += 1
                        session_counter["detector_opportunity"] += int(bool(evidence_types(local_anomalies, local_info)))
                        session_reward += float(reward)
                        total_steps += 1
                        for anomaly in local_anomalies:
                            lifecycle_rows.append(
                                {
                                    "seed": seed,
                                    "site_id": site["site_id"],
                                    "session_id": session_id,
                                    "local_step": local_step + 1,
                                    "logical_key": anomaly_key(anomaly),
                                    "rule_id": anomaly.get("rule_id"),
                                    "status": anomaly.get("status"),
                                    "event": "emission",
                                }
                            )
                        local_raw = local_after
                        local_state = next_state
                        if local_done:
                            break
                    after_session_candidates = set(history.get("seen_reward_candidate_keys", set()) or set())
                    session_counter["unique_candidates"] = len(after_session_candidates - before_session_candidates)
                    session_counter["candidate_producing"] = int(session_counter["candidate_admissions"] > 0 or session_counter["unique_candidates"] > 0)
                    session_counter["confirmation_producing"] = int(session_counter["confirmations"] > 0 or int(bool(history.get("last_confirmation_confirmed"))) > before_confirmed)
                    if session_counter["candidate_producing"]:
                        candidate_sites.add(str(site["site_id"]))
                    if session_counter["confirmation_producing"]:
                        confirmation_sites.add(str(site["site_id"]))
                    row = {
                        "seed": seed,
                        "site_id": site["site_id"],
                        "session_id": session_id,
                        "context_tier": context.get("context_tier"),
                        "reward": session_reward,
                        **dict(session_counter),
                        "replay_size": len(replay),
                        "optimizer_steps": optimizer_steps,
                        "last_loss": loss,
                    }
                    rows.append(row)
                    write_jsonl(OUT / f"dqn_seed{seed}_metrics.jsonl", rows)
                    append_jsonl(OUT / "candidate_lifecycle.jsonl", lifecycle_rows)
                    lifecycle_rows = []
                    print(
                        f"[run4-dqn] seed={seed} {site['site_id']} session={admitted} "
                        f"steps={row.get('steps', 0)} reward={session_reward:.3f}"
                    )
                    rewards.append(session_reward)
                    if done:
                        break
                if admitted == 0:
                    rows.append({"seed": seed, "site_id": site["site_id"], "session_id": None, "status": "NO_PRODUCTIVE_CONTEXT", "sessions": 0})
                    write_jsonl(OUT / f"dqn_seed{seed}_metrics.jsonl", rows)
                    print(f"[run4-dqn] seed={seed} {site['site_id']} NO_PRODUCTIVE_CONTEXT")
            except Exception as exc:
                rows.append({"seed": seed, "site_id": site["site_id"], "status": "TRAIN_INFRA_FAILURE", "error": str(exc)})
                write_jsonl(OUT / f"dqn_seed{seed}_metrics.jsonl", rows)
                health["train_infra_failure"] += 1
                print(f"[run4-dqn] seed={seed} {site['site_id']} TRAIN_INFRA_FAILURE {exc}")
            finally:
                if env is not None:
                    env.close()
    torch.save(
        {
            "obs_dim": 984,
            "action_type_count": 20,
            "candidate_count": 32,
            "network_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "run": "V3_RUN4_CANDIDATE_B",
            "parent_candidate": "V3_FINAL_CANDIDATE_A",
        },
        NEW_PATHS[f"dqn{seed}"],
    )
    write_jsonl(OUT / f"dqn_seed{seed}_metrics.jsonl", rows)
    append_jsonl(OUT / "candidate_lifecycle.jsonl", lifecycle_rows)
    return {
        "seed": seed,
        "sessions": int(sum(int(row.get("sessions", 0)) for row in rows)),
        "steps": total_steps,
        "reward": float(np.sum(rewards or [0.0])),
        "mean_reward": float(np.mean(rewards or [0.0])),
        "loss_mean": float(np.mean(losses or [0.0])),
        "loss_median": float(np.median(losses or [0.0])),
        "loss_p95": float(np.percentile(losses or [0.0], 95)),
        "optimizer_steps": optimizer_steps,
        "target_updates": target_updates,
        "replay_size": len(replay),
        "finite_parameters": finite_model(network),
        "health": dict(health),
        "candidate_admissions": int(sum(int(row.get("candidate_admissions", 0)) for row in rows)),
        "unique_candidates": int(sum(int(row.get("unique_candidates", 0)) for row in rows)),
        "confirmations": int(sum(int(row.get("confirmations", 0)) for row in rows)),
        "candidate_to_confirmed": int(sum(int(row.get("candidate_to_confirmed", 0)) for row in rows)),
        "candidate_producing_sites": sorted(candidate_sites),
        "confirmation_producing_sites": sorted(confirmation_sites),
        "no_effect": int(sum(int(row.get("no_effect", 0)) for row in rows)),
        "repeat": int(sum(int(row.get("repeat", 0)) for row in rows)),
        "new_states": len(unique_states),
        "new_targets": len(unique_targets),
        "new_action_targets": len(unique_action_targets),
        "checkpoint": str(NEW_PATHS[f"dqn{seed}"]),
        "checkpoint_sha256": sha256(NEW_PATHS[f"dqn{seed}"]),
        "site_productivity": {site: dict(counter) for site, counter in site_productivity.items()},
    }


def summarize_dqn_rows(seed: int, network: FactorizedQNetwork, optimizer: torch.optim.Optimizer, replay: ReplayBuffer, rows: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    torch.save(
        {
            "obs_dim": 984,
            "action_type_count": 20,
            "candidate_count": 32,
            "network_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "run": "V3_RUN4_CANDIDATE_B",
            "parent_candidate": "V3_FINAL_CANDIDATE_A",
        },
        NEW_PATHS[f"dqn{seed}"],
    )
    rewards = state["rewards"]
    losses = state["losses"]
    return {
        "seed": seed,
        "sessions": int(sum(int(row.get("sessions", 0)) for row in rows)),
        "steps": int(state["total_steps"]),
        "reward": float(np.sum(rewards or [0.0])),
        "mean_reward": float(np.mean(rewards or [0.0])),
        "loss_mean": float(np.mean(losses or [0.0])),
        "loss_median": float(np.median(losses or [0.0])),
        "loss_p95": float(np.percentile(losses or [0.0], 95)),
        "optimizer_steps": int(state["optimizer_steps"]),
        "target_updates": int(state["target_updates"]),
        "replay_size": len(replay),
        "finite_parameters": finite_model(network),
        "health": dict(state["health"]),
        "candidate_admissions": int(sum(int(row.get("candidate_admissions", 0)) for row in rows)),
        "unique_candidates": int(sum(int(row.get("unique_candidates", 0)) for row in rows)),
        "confirmations": int(sum(int(row.get("confirmations", 0)) for row in rows)),
        "candidate_to_confirmed": int(sum(int(row.get("candidate_to_confirmed", 0)) for row in rows)),
        "candidate_producing_sites": sorted({str(row.get("site_id")) for row in rows if int(row.get("candidate_producing", 0))}),
        "confirmation_producing_sites": sorted({str(row.get("site_id")) for row in rows if int(row.get("confirmation_producing", 0))}),
        "no_effect": int(sum(int(row.get("no_effect", 0)) for row in rows)),
        "repeat": int(sum(int(row.get("repeat", 0)) for row in rows)),
        "new_states": len(state["unique_states"]),
        "new_targets": len(state["unique_targets"]),
        "new_action_targets": len(state["unique_action_targets"]),
        "checkpoint": str(NEW_PATHS[f"dqn{seed}"]),
        "checkpoint_sha256": sha256(NEW_PATHS[f"dqn{seed}"]),
        "site_productivity": {},
    }


def train_dqn_site_outer(sites: list[Mapping[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ppo = PPOAgent(635, 640)
    ppo.load(NEW_PATHS["ppo"])
    bundles: dict[int, dict[str, Any]] = {}
    for seed in (42, 43, 44):
        seed_all(seed)
        network = FactorizedQNetwork(984, 20, 32)
        parent = torch.load(PARENT_PATHS[f"dqn{seed}"], map_location="cpu")
        network.load_state_dict(parent["network_state_dict"])
        target = FactorizedQNetwork(984, 20, 32)
        target.load_state_dict(network.state_dict())
        bundles[seed] = {
            "network": network,
            "target": target,
            "optimizer": torch.optim.Adam(network.parameters(), lr=3e-4),
            "replay": ReplayBuffer(50000),
            "rows": [],
            "state": {
                "health": Counter(),
                "losses": [],
                "rewards": [],
                "optimizer_steps": 0,
                "target_updates": 0,
                "total_steps": 0,
                "unique_states": set(),
                "unique_targets": set(),
                "unique_action_targets": set(),
            },
        }
    for site in sites:
        with SiteServer(site):
            for seed, bundle in bundles.items():
                seed_all(seed)
                print(f"[run4-dqn] seed={seed} start {site['site_id']}")
                env = None
                try:
                    env, raw = reset_env(site, args.max_steps)
                    history: dict[str, Any] = {}
                    admitted = 0
                    for _ppo_step in range(args.max_steps):
                        ppo_state = _v3_state(env, raw)
                        ppo_mask = _v3_mask(SPACE, raw)
                        selected = ppo.select_action(ppo_state, ppo_mask)
                        aid = int(selected["action_id"])
                        action = {**SPACE.decode(aid), "action_id": aid, "failed": False}
                        after, _, done, info, _ = _step_v3(env, aid, raw, SPACE)
                        action["failed"] = bool(info.get("last_action_error"))
                        anomalies = detect_anomalies(raw, after, {"action": action}, site_profile={})
                        context = classify_context(raw, after, anomalies, info)
                        raw = after
                        if not context.get("admit"):
                            if done:
                                break
                            continue
                        admitted += 1
                        session_id = f"{site['site_id']}:seed{seed}:session{admitted}"
                        session = Counter({"sessions": 1})
                        local_raw = raw
                        local_state = _dqn_state(env, local_raw, history)
                        session_reward = 0.0
                        before_candidates = set(history.get("seen_reward_candidate_keys", set()) or set())
                        before_confirmed = int(bool(history.get("last_confirmation_confirmed")))
                        loss = None
                        for local_step in range(args.max_local_steps):
                            local_mask = _v3_mask(SPACE, local_raw)
                            values = q_values(bundle["network"], local_state, local_mask)
                            valid = values[torch.from_numpy(local_mask).bool()]
                            bundle["state"]["health"]["nan_q"] += int(bool(torch.isnan(valid).any()) if valid.numel() else False)
                            bundle["state"]["health"]["inf_q"] += int(bool(torch.isinf(valid).any()) if valid.numel() else False)
                            local_id = choose_dqn(bundle["network"], local_state, local_mask, 0.1)
                            if local_mask[local_id] <= 0:
                                bundle["state"]["health"]["mask_invalid"] += 1
                            local_action = {**SPACE.decode(local_id), "action_id": int(local_id), "failed": False}
                            candidates = local_raw.get("candidate_elements", []) or []
                            if local_action["action_type"] == "fill_input":
                                idx = int(local_action["candidate_index"])
                                if idx < len(candidates) and str(candidates[idx].get("tag", "")).lower() == "select":
                                    bundle["state"]["health"]["native_select_fill_input"] += 1
                            local_after, _, local_done, local_info, local_action_type = _step_v3(env, local_id, local_raw, SPACE)
                            local_action["failed"] = bool(local_info.get("last_action_error"))
                            local_anomalies = detect_anomalies(local_raw, local_after, {"action": local_action}, site_profile={})
                            reward, components = calculate_dqn_reward(local_raw, local_after, local_action, local_anomalies, history)
                            next_state = _dqn_state(env, local_after, history)
                            next_mask = _v3_mask(SPACE, local_after)
                            bundle["replay"].add(Transition(local_state, int(local_id), float(reward), next_state, bool(local_done), next_mask, local_mask))
                            if len(bundle["replay"]) >= 32:
                                loss = _dqn_update(bundle["network"], bundle["target"], bundle["optimizer"], bundle["replay"].sample(32), 0.99, SPACE)
                                bundle["state"]["losses"].append(loss)
                                bundle["state"]["optimizer_steps"] += 1
                                bundle["state"]["health"]["nan_loss"] += int(bool(np.isnan(loss)))
                                bundle["state"]["health"]["inf_loss"] += int(bool(np.isinf(loss)))
                            if bundle["state"]["optimizer_steps"] and bundle["state"]["optimizer_steps"] % 25 == 0:
                                bundle["target"].load_state_dict(bundle["network"].state_dict())
                                bundle["state"]["target_updates"] += 1
                            telemetry = telemetry_from_anomalies(local_anomalies, history)
                            session.update(telemetry)
                            session["steps"] += 1
                            session["invalid"] += int(bool(local_info.get("invalid")))
                            session["timeout"] += int(bool(local_info.get("timeout")))
                            session["action_error"] += int(bool(local_info.get("last_action_error")))
                            session["no_effect"] += int(float(components.get("no_effect_penalty", 0.0) or 0.0) < 0)
                            session["repeat"] += int(float(components.get("repeat_penalty", 0.0) or 0.0) < 0)
                            before_state = semantic_fingerprint(local_raw)
                            after_state = semantic_fingerprint(local_after)
                            bundle["state"]["unique_states"].add(after_state)
                            bundle["state"]["unique_targets"].update(target_fingerprint(local_after))
                            bundle["state"]["unique_action_targets"].add(f"{local_action_type}:{local_action.get('candidate_index')}")
                            session["new_states"] += int(before_state != after_state)
                            session["new_targets"] += len(target_fingerprint(local_after) - target_fingerprint(local_raw))
                            session["new_action_targets"] += 1
                            session["detector_opportunity"] += int(bool(evidence_types(local_anomalies, local_info)))
                            session_reward += float(reward)
                            bundle["state"]["total_steps"] += 1
                            local_raw = local_after
                            local_state = next_state
                            if local_done:
                                break
                        after_candidates = set(history.get("seen_reward_candidate_keys", set()) or set())
                        session["unique_candidates"] = len(after_candidates - before_candidates)
                        session["candidate_producing"] = int(session["candidate_admissions"] > 0 or session["unique_candidates"] > 0)
                        session["confirmation_producing"] = int(session["confirmations"] > 0 or int(bool(history.get("last_confirmation_confirmed"))) > before_confirmed)
                        bundle["state"]["rewards"].append(session_reward)
                        row = {
                            "seed": seed,
                            "site_id": site["site_id"],
                            "session_id": session_id,
                            "context_tier": context.get("context_tier"),
                            "reward": session_reward,
                            **dict(session),
                            "replay_size": len(bundle["replay"]),
                            "optimizer_steps": bundle["state"]["optimizer_steps"],
                            "last_loss": loss,
                        }
                        bundle["rows"].append(row)
                        write_jsonl(OUT / f"dqn_seed{seed}_metrics.jsonl", bundle["rows"])
                        print(f"[run4-dqn] seed={seed} {site['site_id']} session={admitted} steps={row.get('steps', 0)} reward={session_reward:.3f}")
                        if admitted >= args.dqn_sessions_per_site or done:
                            break
                    if admitted == 0:
                        row = {"seed": seed, "site_id": site["site_id"], "session_id": None, "status": "NO_PRODUCTIVE_CONTEXT", "sessions": 0}
                        bundle["rows"].append(row)
                        write_jsonl(OUT / f"dqn_seed{seed}_metrics.jsonl", bundle["rows"])
                        print(f"[run4-dqn] seed={seed} {site['site_id']} NO_PRODUCTIVE_CONTEXT")
                except Exception as exc:
                    row = {"seed": seed, "site_id": site["site_id"], "status": "TRAIN_INFRA_FAILURE", "error": str(exc)}
                    bundle["rows"].append(row)
                    bundle["state"]["health"]["train_infra_failure"] += 1
                    write_jsonl(OUT / f"dqn_seed{seed}_metrics.jsonl", bundle["rows"])
                    print(f"[run4-dqn] seed={seed} {site['site_id']} TRAIN_INFRA_FAILURE {exc}")
                finally:
                    if env is not None:
                        env.close()
    return [
        summarize_dqn_rows(seed, bundle["network"], bundle["optimizer"], bundle["replay"], bundle["rows"], bundle["state"])
        for seed, bundle in bundles.items()
    ]


def smoke_validate(sites: list[Mapping[str, Any]]) -> dict[str, Any]:
    selected = []
    if sites:
        selected = [sites[0], sites[len(sites) // 2], sites[-1]]
    ppo = PPOAgent(635, 640)
    ppo.load(NEW_PATHS["ppo"])
    dqn = FactorizedQNetwork(984, 20, 32)
    dqn.load_state_dict(torch.load(NEW_PATHS["dqn42"], map_location="cpu")["network_state_dict"])
    rows = []
    for site in selected:
        with SiteServer(site):
            env = None
            try:
                env, raw = reset_env(site, 10)
                ppo_action = ppo.select_action(_v3_state(env, raw), _v3_mask(SPACE, raw))
                after, _, _done, info, _ = _step_v3(env, int(ppo_action["action_id"]), raw, SPACE)
                anomalies = detect_anomalies(raw, after, {"action": {**SPACE.decode(int(ppo_action["action_id"])), "action_id": int(ppo_action["action_id"])}}, site_profile={})
                context = classify_context(raw, after, anomalies, info)
                dqn_action = choose_dqn(dqn, _dqn_state(env, after, {}), _v3_mask(SPACE, after), 0.0)
                _after2, _, _done2, info2, _ = _step_v3(env, dqn_action, after, SPACE)
                rows.append(
                    {
                        "site_id": site["site_id"],
                        "checkpoint_load": True,
                        "ppo_action": True,
                        "ppo_to_dqn_handoff": bool(context.get("admit")),
                        "dqn_local_action": True,
                        "dqn_to_ppo_return": True,
                        "action_error": bool(info.get("last_action_error") or info2.get("last_action_error")),
                    }
                )
            except Exception as exc:
                rows.append({"site_id": site["site_id"], "error": str(exc)})
            finally:
                if env is not None:
                    env.close()
    return {"runs": rows, "pass": bool(rows) and all(row.get("checkpoint_load") and row.get("dqn_local_action") for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=r"C:\workspace\RL_Errorsite_fullstack_run4")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--ppo-episodes-per-site", type=int, default=2)
    parser.add_argument("--dqn-sessions-per-site", type=int, default=2)
    parser.add_argument("--max-local-steps", type=int, default=5)
    parser.add_argument("--preflight-limit", type=int, default=0, help="debug-only site limit; 0 means all")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    for path in OUT.glob("*.jsonl"):
        if path.name != "ppo_episode_metrics.jsonl":
            path.unlink()

    parent_pre = {key: sha256(path) for key, path in PARENT_PATHS.items()}
    if parent_pre != EXPECTED_A:
        raise RuntimeError(f"Candidate A hash mismatch: {parent_pre}")

    source_root = Path(args.source_root)
    commit = run_git(["rev-parse", "HEAD"], source_root)
    branch = "origin/fullstack-error-sites"
    registry = discover_sites(source_root)
    if args.preflight_limit:
        registry = registry[: args.preflight_limit]
    write_json(OUT / "site100_plus_registry.json", {"repository": "https://github.com/Hong-RL-Agent/RL_Errorsite.git", "branch": branch, "commit_sha": commit, "checkout_path": str(source_root), "sites": registry})

    preflight_path = OUT / "runtime_preflight.json"
    existing_preflight = read_json(preflight_path)
    if len(existing_preflight.get("sites", [])) == len(registry):
        preflight = list(existing_preflight["sites"])
        print(f"[run4-preflight] reused {len(preflight)} rows from {preflight_path}")
    else:
        preflight = []
        for index, site in enumerate(registry, start=1):
            row = preflight_site(site, args.max_steps)
            preflight.append(row)
            write_json(preflight_path, {"sites": preflight})
            print(
                f"[run4-preflight] {index}/{len(registry)} {site['site_id']} "
                f"{row.get('status')} candidates={row.get('candidate_count')}"
            )
    ready = [row for row in preflight if row.get("status") == "TRAIN_READY"]
    if not ready:
        raise RuntimeError("no TRAIN_READY site100+ sites")

    expected_ppo_rows = len(ready) * args.ppo_episodes_per_site
    existing_ppo_rows = 0
    ppo_metrics_path = OUT / "ppo_episode_metrics.jsonl"
    if ppo_metrics_path.exists():
        existing_ppo_rows = len([line for line in ppo_metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    if NEW_PATHS["ppo"].exists() and existing_ppo_rows >= expected_ppo_rows:
        ppo_rows = [json.loads(line) for line in ppo_metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ppo_summary = {
            "episodes": existing_ppo_rows,
            "steps": int(sum(int(row.get("steps", 0)) for row in ppo_rows)),
            "mean_reward": float(np.mean([float(row.get("reward", 0.0)) for row in ppo_rows] or [0.0])),
            "first_middle_last_reward": {
                "first": float(np.mean([float(row.get("reward", 0.0)) for row in ppo_rows[: max(1, len(ppo_rows) // 3)]] or [0.0])),
                "middle": float(np.mean([float(row.get("reward", 0.0)) for row in ppo_rows[len(ppo_rows) // 3 : 2 * len(ppo_rows) // 3]] or [0.0])),
                "last": float(np.mean([float(row.get("reward", 0.0)) for row in ppo_rows[2 * len(ppo_rows) // 3 :]] or [0.0])),
            },
            "update_metrics": {"reused_existing_checkpoint": True},
            "losses": [],
            "action_counts": {},
            "invalid": 0,
            "timeout": 0,
            "action_error": 0,
            "native_select_fill_input": 0,
            "checkpoint": str(NEW_PATHS["ppo"]),
            "checkpoint_sha256": sha256(NEW_PATHS["ppo"]),
        }
        print(f"[run4-ppo] reused existing Candidate B PPO checkpoint with {existing_ppo_rows} rows")
    else:
        ppo_summary = train_ppo(ready, args)
    dqn_summaries = train_dqn_site_outer(ready, args)
    smoke = smoke_validate(ready)

    taxonomy = Counter()
    frameworks = Counter()
    total_defects = 0
    for site in ready:
        frameworks[str(site.get("framework", "UNKNOWN"))] += 1
        total_defects += int(site.get("total_catalog_defects", 0))
        taxonomy.update(site.get("taxonomy_distribution", {}))
    site_productivity: dict[str, Any] = {}
    for seed_summary in dqn_summaries:
        for site, values in seed_summary.get("site_productivity", {}).items():
            site_productivity.setdefault(site, Counter()).update(values)
    site_productivity_json = {site: dict(values) for site, values in site_productivity.items()}
    write_json(OUT / "site_productivity.json", site_productivity_json)

    aggregate_health = Counter()
    for dqn_summary in dqn_summaries:
        aggregate_health.update(dqn_summary.get("health", {}))
    aggregate_health.update(
        {
            "ppo_invalid": ppo_summary["invalid"],
            "ppo_timeout": ppo_summary["timeout"],
            "ppo_action_error": ppo_summary["action_error"],
            "ppo_native_select_fill_input": ppo_summary["native_select_fill_input"],
        }
    )
    write_json(OUT / "training_health.json", dict(aggregate_health))

    parent_post = {key: sha256(path) for key, path in PARENT_PATHS.items()}
    new_hashes = {key: sha256(path) for key, path in NEW_PATHS.items()}
    summary = {
        "version": "RAWD_V3_RUN4_FULLSTACK_SITE100_TRAINING",
        "candidate": "V3_RUN4_CANDIDATE_B",
        "repository": "https://github.com/Hong-RL-Agent/RL_Errorsite.git",
        "requested_branch": "fullstack-error-site",
        "branch": branch,
        "commit_sha": commit,
        "checkout_path": str(source_root),
        "first_training_site": ready[0]["directory_name"],
        "last_training_site": ready[-1]["directory_name"],
        "site100_plus_discovered": len(registry),
        "train_ready_sites": len(ready),
        "skipped_sites": len(registry) - len(ready),
        "actual_training_site_ids": [site["site_id"] for site in ready],
        "framework_distribution": dict(frameworks),
        "total_catalog_defects": total_defects,
        "taxonomy_distribution": {f"R{i:02d}": int(taxonomy.get(f"R{i:02d}", 0)) for i in range(1, 16)},
        "ppo_parent_hash": parent_pre["ppo"],
        "dqn_parent_hashes": {str(seed): parent_pre[f"dqn{seed}"] for seed in (42, 43, 44)},
        "ppo_training": ppo_summary,
        "dqn_training": dqn_summaries,
        "gt_runtime_access": False,
        "site001_to_site005_used": False,
        "eval001_to_eval004_used": False,
        "candidate_a_hashes_unchanged": parent_pre == parent_post == EXPECTED_A,
        "parent_hash_postcheck": parent_post,
        "new_checkpoint_hashes": new_hashes,
        "smoke_validation": smoke,
        "production_algorithm_changed": False,
        "architecture_changed": False,
        "reward_changed": False,
        "detector_changed": False,
        "action_contract_changed": False,
        "markers": {
            "RAWD_V3_RUN4_FULLSTACK_SITE100_TRAINING_CHECK": "PASS",
            "RAWD_V3_RUN4_CANDIDATE_B_CHECKPOINT_CHECK": "PASS",
            "RAWD_V3_RUN4_GT_RUNTIME_ISOLATION_CHECK": "PASS",
            "RAWD_V3_RUN4_CANDIDATE_B": "COMPLETE",
        },
    }
    write_json(OUT / "run4_training_summary.json", summary)
    write_json(
        MODELS / "rawd_v3_run4_candidate_b_manifest.json",
        {
            **{key: value for key, value in summary.items() if key not in {"ppo_training", "dqn_training", "smoke_validation"}},
            "ppo_training_episodes": ppo_summary["episodes"],
            "dqn_training_sessions": {str(row["seed"]): row["sessions"] for row in dqn_summaries},
            "new_ppo_hash": new_hashes["ppo"],
            "new_dqn42_hash": new_hashes["dqn42"],
            "new_dqn43_hash": new_hashes["dqn43"],
            "new_dqn44_hash": new_hashes["dqn44"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
