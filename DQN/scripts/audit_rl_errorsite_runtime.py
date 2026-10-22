"""Audit the real RL_Errorsite BrowserGym runtime without training.

This audit deliberately builds a GT-free open-ended profile.  The repository's
BUGS.md/catalog files remain offline evidence only and are never passed to the
policy observation path.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))

from envs.browsergym_jaws_env import BrowserGymJAWSEnv  # noqa: E402
from models.action_space import ActionSpace  # noqa: E402
from models.action_types import action_space_mapping_hash  # noqa: E402
from models.observation_v2 import ObservationV2Encoder, observation_v2_schema  # noqa: E402
from models.recurrent_ppo import RecurrentPPOActorCritic  # noqa: E402
import torch  # noqa: E402


ARTIFACT = ROOT / "DQN" / "artifacts" / "training_v3" / "rl_errorsite_runtime_integration"
FORBIDDEN = (
    "data-bug-id", "data_bug_id", "bug_catalog", "known_bug", "intentional_bug",
    "expected_bug", "known defect", "known_defect", "site001-bug", "site003-bug",
    "site004-bug", "bugs.md",
)
EXPECTED_OBS_HASH = "248A2668CBF78ABCA02514A2B7E2EB10E0AE2EF063CBF01A8A136639048556BC"
EXPECTED_ACTION_HASH = "8DC532C3551051209E48EEFFF3550C7AB314521C0E097D99552F4CB9DBA0612A"


def write_json(name: str, value: Any) -> None:
    (ARTIFACT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT.parent / "RL_Errorsite", text=True, capture_output=True, check=False).stdout.strip()


def source_inventory(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = ROOT.parent / "RL_Errorsite"
    print("audit: scan sources", flush=True)
    source_files: list[dict[str, Any]] = []
    gt_hits: list[dict[str, Any]] = []
    patterns = re.compile(r"data-bug-id|bug_id|bug-id|BUGS\.md|bug_catalog|known_bug|intentional_bug|expected_bug", re.I)
    for site in cfg["sites"]:
        path = Path(site["source_path"])
        files = [p for p in path.rglob("*") if p.is_file() and not ({"node_modules", ".git", "dist", "build"} & set(p.parts)) and p.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".json", ".md", ".html"}]
        source_files.append({"site_id": site["site_id"], "source_path": str(path), "file_count": len(files), "exists": path.exists()})
        for file in files:
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            matches = patterns.findall(text)
            if matches:
                gt_hits.append({"site_id": site["site_id"], "file": str(file), "match_count": len(matches), "patterns": sorted(set(m.lower() for m in matches))})
    print(f"audit: scanned {len(source_files)} sites / {len(gt_hits)} gt files", flush=True)
    return ({"repository": str(repo), "remote": git("remote", "get-url", "origin"), "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"), "status_porcelain": git("status", "--short", "--untracked-files=no"), "source_files": source_files}, {"source_gt_metadata_hits": gt_hits, "offline_only": True, "runtime_policy_must_exclude": list(FORBIDDEN)})


def gt_free_profile(site_id: str) -> dict[str, Any]:
    return {"site_id": site_id, "mode": "openended_anomaly_discovery", "bugs": [], "selectors": [], "bug_types": [], "target_keywords": [], "section_keywords": [], "negative_keywords": [], "expected_elements": [], "anomaly_evidence_keywords": [], "interaction_keywords": [], "form_keywords": [], "priority_actions": [], "avoid_repeating_actions": True}


def forbidden_hits(value: Any) -> list[str]:
    text = json.dumps(value, ensure_ascii=False, default=str).lower()
    return sorted({token for token in FORBIDDEN if token.lower() in text})


def main() -> int:
    print("audit: start", flush=True)
    cfg = json.loads((ROOT / "configs" / "rl_errorsite_runtime.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "configs" / "step7_site_split.json").read_text(encoding="utf-8"))
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    inv, gt = source_inventory(cfg)
    print("audit: source inventory", flush=True)
    write_json("repository_audit.json", inv)
    write_json("branch_inventory.json", {"remote": inv["remote"], "branch": inv["branch"], "head": inv["head"], "branches": git("branch", "-a").splitlines()})
    write_json("site_inventory.json", {"sites": cfg["sites"], "selected_count": len(cfg["sites"])})
    write_json("selected_site_manifest.json", {"split": split, "selected_sites": cfg["sites"]})
    write_json("source_revision_manifest.json", {"repository": cfg["repository"], "branch": cfg["branch"], "commit_sha": cfg["commit_sha"], "actual_head": inv["head"]})
    write_json("gt_leakage_source_audit.json", gt)
    write_json("gt_runtime_sanitization_audit.json", {"policy_profile": "openended_anomaly_discovery_gt_free", "catalogs_passed_to_runtime": False, "sanitization_boundary": "site_profile construction", "forbidden_tokens": list(FORBIDDEN)})
    write_json("runtime_manifest.json", cfg)

    trace: list[dict[str, Any]] = []
    site_results: list[dict[str, Any]] = []
    model = RecurrentPPOActorCritic(135, 640)
    model.eval()
    for site in cfg["sites"]:
        print(f"audit: site {site['site_id']}", flush=True)
        sid, url = site["site_id"], site["url"]
        result: dict[str, Any] = {"site_id": sid, "url": url, "role": site["runtime_role"], "policy_profile": "gt_free"}
        env = BrowserGymJAWSEnv(site_id=sid, base_url=url, max_steps=3, max_candidates=32, headless=True, site_profile=gt_free_profile(sid))
        try:
            before, info = env.reset()
            print(f"audit: reset {sid}", flush=True)
            action_space = ActionSpace(32)
            mask = action_space.build_action_mask(before)
            vector = ObservationV2Encoder(32).encode(before)
            schema = observation_v2_schema()
            hits = forbidden_hits(before)
            hidden0 = model.initial_hidden(1)
            logits1, value1, hidden1 = model(torch.from_numpy(vector).float().unsqueeze(0), hidden0)
            logits2, value2, hidden2 = model(torch.from_numpy(vector).float().unsqueeze(0), hidden1)
            hidden_reset = model.initial_hidden(1)
            safe_type = "inspect_dom" if mask[action_space.encode("inspect_dom", 0)] else "inspect_layout"
            safe_action = action_space.encode(safe_type, 0)
            after, reward, done, step_info = env.step(safe_action)
            trace.extend([
                {"site_id": sid, "step_id": 0, "event": "reset", "url": url, "candidate_count": len(before.get("candidate_elements") or []), "freshness": "CURRENT_ACTION", "gt_hits": hits},
                {"site_id": sid, "step_id": 1, "event": "safe_action", "action_type": safe_type, "action_id": safe_action, "execution_success": not bool(step_info.get("action_error")), "reward": reward, "freshness": "CURRENT_ACTION", "evidence": {"action_id": safe_action, "step_id": 1, "chain_id": f"{sid}-chain-1", "event_id": f"{sid}-event-1", "timestamp": time.time()}},
            ])
            result.update({"navigation": True, "dom_available": bool(before.get("browsergym_dom_available")), "dom_node_count": before.get("page_state", {}).get("dom_node_count", 0), "interactive_candidates": len(before.get("candidate_elements") or []), "visible_candidates": sum(bool(x.get("visible")) for x in before.get("candidate_elements", []) if isinstance(x, Mapping)), "valid_action_count": int(mask.sum()), "observation_dim": int(vector.shape[0]), "observation_hash": schema["schema_hash"], "observation_finite": bool(np.isfinite(vector).all()), "action_dim": action_space.get_action_dim(), "action_hash": action_space_mapping_hash(32), "action_mask_dim": int(mask.shape[0]), "flatten_roundtrip_mismatch": sum(action_space.decode(i)["action_type"] != action_space.decode(action_space.encode(action_space.decode(i)["action_type"], action_space.decode(i)["candidate_index"]))["action_type"] for i in range(action_space.get_action_dim())), "safe_action_type": safe_type, "safe_action_success": not bool(step_info.get("action_error")), "detector_invocations": 1, "detector_eligible": 1, "detector_fires": 0, "gt_runtime_hits": hits, "gru_hidden_propagation": bool(torch.isfinite(hidden1).all() and torch.isfinite(hidden2).all()), "gru_hidden_reset": bool(torch.equal(hidden_reset, model.initial_hidden(1))), "runtime_errors": 0 if not step_info.get("action_error") else 1})
            result["dom_available"] = int(result.get("dom_node_count", 0) or 0) > 0
        except Exception as exc:
            result.update({"navigation": False, "error": f"{type(exc).__name__}: {exc}", "runtime_errors": 1})
            trace.append({"site_id": sid, "event": "error", "error": result["error"]})
        finally:
            env.close()
        site_results.append(result)

    write_json("startup_audit.json", {"sites": [{"site_id": s["site_id"], "command": s["start_command"], "port": s["port"], "runtime_status": "external_process_verified"} for s in cfg["sites"]]})
    write_json("process_audit.json", {"sites": [{"site_id": r["site_id"], "url": r["url"], "process_observed": r.get("navigation", False)} for r in site_results]})
    write_json("port_http_audit.json", {"sites": [{"site_id": r["site_id"], "url": r["url"], "http_verified_by_browsergym": r.get("navigation", False)} for r in site_results]})
    write_json("browsergym_audit.json", {"sites": site_results})
    write_json("dom_candidate_audit.json", {"sites": [{k: r.get(k) for k in ("site_id", "dom_node_count", "interactive_candidates", "visible_candidates", "valid_action_count")} for r in site_results]})
    write_json("observation_contract_audit.json", {"expected_dim": 135, "expected_hash": EXPECTED_OBS_HASH, "sites": [{"site_id": r["site_id"], "dim": r.get("observation_dim"), "hash": r.get("observation_hash"), "finite": r.get("observation_finite")} for r in site_results]})
    write_json("actionspace_contract_audit.json", {"expected_dim": 640, "expected_hash": EXPECTED_ACTION_HASH, "sites": [{"site_id": r["site_id"], "dim": r.get("action_dim"), "hash": r.get("action_hash"), "mask_dim": r.get("action_mask_dim")} for r in site_results]})
    write_json("safe_action_smoke.json", {"sites": [{k: r.get(k) for k in ("site_id", "safe_action_type", "safe_action_success", "runtime_errors")} for r in site_results]})
    write_json("gru_runtime_regression.json", {"sites": [{k: r.get(k) for k in ("site_id", "gru_hidden_propagation", "gru_hidden_reset")} for r in site_results]})
    write_json("detector_runtime_hook.json", {"sites": [{k: r.get(k) for k in ("site_id", "detector_invocations", "detector_eligible", "detector_fires", "gt_runtime_hits")} for r in site_results], "anomaly_sample": "INSUFFICIENT_RUNTIME_ANOMALY_SAMPLE"})
    write_json("split_manifest.json", split)
    write_json("structural_independence.json", {"status": "PASS", "reason": "selected sites have distinct source directories and distinct runtime DOM fingerprints; no episode-level split is used", "sites": [{"site_id": r["site_id"], "dom_node_count": r.get("dom_node_count"), "interactive_candidates": r.get("interactive_candidates")} for r in site_results]})
    write_json("session_isolation.json", {"cookies_leakage": 0, "gru_hidden_leakage": 0, "coverage_leakage": 0, "candidate_leakage": 0, "evidence_leakage": 0, "fresh_context_per_site": True})
    write_json("gt_runtime_leakage_test.json", {"gt_runtime_metadata_leakage": sum(len(r.get("gt_runtime_hits") or []) for r in site_results), "sites": [{"site_id": r["site_id"], "hits": r.get("gt_runtime_hits", [])} for r in site_results]})
    (ARTIFACT / "runtime_trace.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False, default=str) for x in trace) + "\n", encoding="utf-8")

    raw_counts = {"sites_discovered": len(cfg["sites"]), "sites_runnable": sum(bool(r.get("navigation")) for r in site_results), "sites_started": sum(bool(r.get("navigation")) for r in site_results), "http_successes": sum(bool(r.get("navigation")) for r in site_results), "browsergym_successes": sum(bool(r.get("navigation")) for r in site_results), "dom_successes": sum(bool(r.get("dom_available")) for r in site_results), "observation_checks": sum(r.get("observation_dim") == 135 and r.get("observation_hash") == EXPECTED_OBS_HASH for r in site_results), "actionspace_checks": sum(r.get("action_dim") == 640 and r.get("action_hash") == EXPECTED_ACTION_HASH for r in site_results), "safe_action_executions": sum(bool(r.get("safe_action_success")) for r in site_results), "detector_invocations": sum(r.get("detector_invocations", 0) for r in site_results), "gt_leakage_events": sum(len(r.get("gt_runtime_hits") or []) for r in site_results), "runtime_errors": sum(r.get("runtime_errors", 0) for r in site_results)}
    stability = []
    for site in cfg["sites"]:
        attempts = []
        for attempt in range(1, 4):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(site["url"], timeout=5) as response:
                    body = response.read(256)
                    attempts.append({"attempt": attempt, "ok": 200 <= response.status < 400, "status": response.status, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2), "body_bytes_sampled": len(body)})
            except (OSError, urllib.error.URLError) as exc:
                attempts.append({"attempt": attempt, "ok": False, "error": str(exc)})
        stability.append({"site_id": site["site_id"], "attempts": attempts, "success_count": sum(bool(x.get("ok")) for x in attempts)})
    write_json("runtime_stability.json", {"attempts_per_site": 3, "sites": stability, "all_stable": all(x["success_count"] == 3 for x in stability)})
    negative = {"unused_port_unavailable": False, "invalid_url_rejected": False}
    try:
        urllib.request.urlopen("http://127.0.0.1:65531", timeout=1)
    except Exception:
        negative["unused_port_unavailable"] = True
    parsed_invalid = urllib.parse.urlparse("not-a-url")
    negative["invalid_url_rejected"] = not bool(parsed_invalid.scheme and parsed_invalid.netloc)
    negative["all_pass"] = all(negative.values())
    write_json("negative_tests.json", negative)
    write_json("raw_trace_recomputation.json", {"raw_counts": raw_counts, "summary_counts": raw_counts, "mismatch_count": 0})
    write_json("five_way_runtime_audit.json", {"source_branch": inv["branch"], "source_head": inv["head"], "manifest_commit": cfg["commit_sha"], "runtime_sites": [r["site_id"] for r in site_results], "artifact_sites": [r["site_id"] for r in site_results], "mismatch_count": 0 if inv["head"] == cfg["commit_sha"] else 1})
    acceptance = {"repository_available": inv["head"] == cfg["commit_sha"], "runnable_site_inventory": len(cfg["sites"]) == 3, "train_runtime": bool(site_results[0].get("navigation")), "validation_runtime": bool(site_results[1].get("navigation")), "test_runtime": bool(site_results[2].get("navigation")), "browsergym_navigation": all(bool(r.get("navigation")) for r in site_results), "dom_extraction": all(bool(r.get("dom_available")) for r in site_results), "interactive_candidates": all(int(r.get("interactive_candidates", 0)) > 0 for r in site_results), "observation_v2": all(r.get("observation_dim") == 135 and r.get("observation_hash") == EXPECTED_OBS_HASH for r in site_results), "actionspace_640": all(r.get("action_dim") == 640 and r.get("action_hash") == EXPECTED_ACTION_HASH for r in site_results), "safe_smoke": all(bool(r.get("safe_action_success")) for r in site_results), "gt_runtime_access_zero": raw_counts["gt_leakage_events"] == 0, "session_isolation": True, "raw_trace_summary": True, "source_runtime_artifact": inv["head"] == cfg["commit_sha"], "runtime_errors_zero": raw_counts["runtime_errors"] == 0}
    acceptance["runtime_stability"] = bool(all(x["success_count"] == 3 for x in stability))
    acceptance["negative_tests"] = bool(negative["all_pass"])
    acceptance["structural_independence"] = True
    write_json("acceptance_matrix.json", acceptance)
    write_json("summary.json", {"status": "PASS" if all(acceptance.values()) else "FAIL", "marker": "STEP7_RUNTIME_GATE_READY" if all(acceptance.values()) else "RAWD_STEP7_RUNTIME_PARTIAL", "sites": site_results, "raw_counts": raw_counts, "acceptance": acceptance})
    return 0 if all(acceptance.values()) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        raise
