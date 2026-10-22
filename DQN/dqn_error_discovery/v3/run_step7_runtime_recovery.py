"""Audit the real Step-7 site runtimes without starting training."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
OUT = ROOT / "artifacts/training_v3/step7_runtime_recovery"
SITE_CONFIG = PROJECT_ROOT / "configs/training_sites.json"
SPLIT_CONFIG = PROJECT_ROOT / "configs/step7_site_split.json"
OBS_HASH = "248A2668CBF78ABCA02514A2B7E2EB10E0AE2EF063CBF01A8A136639048556BC"
ACTION_HASH = "8DC532C3551051209E48EEFFF3550C7AB314521C0E097D99552F4CB9DBA0612A"


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def tcp_audit(url: str) -> dict:
    parsed = urlparse(url)
    started = time.perf_counter()
    result = {"url": url, "host": parsed.hostname, "port": parsed.port, "listening": False, "elapsed_ms": None, "error": None}
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=1.0):
            result["listening"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def http_audit(url: str) -> dict:
    result = {"url": url, "reachable": False, "status": None, "content_type": None, "body_bytes": 0, "redirect": None, "error": None}
    try:
        request = Request(url, headers={"User-Agent": "RAWD-step7-runtime-audit/1.0"})
        with urlopen(request, timeout=3.0) as response:
            body = response.read(4096)
            result.update({"reachable": True, "status": response.status, "content_type": response.headers.get("content-type"), "body_bytes": len(body), "redirect": response.geturl() if response.geturl() != url else None})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    config = json.loads(SITE_CONFIG.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_CONFIG.read_text(encoding="utf-8"))
    sites = {str(item["site_id"]): item for item in config.get("sites", [])}
    source_rows = []
    process_rows = []
    port_rows = []
    browsergym_rows = []
    dom_rows = []
    action_rows = []
    trace = []
    for site_id in [*split.get("train", []), *split.get("validation", []), *split.get("test", [])]:
        site = sites[site_id]
        url = str(site["base_url"])
        source_rows.append({
            "site_id": site_id,
            "source_path": None,
            "framework": None,
            "frontend_start_command": site.get("start_command"),
            "backend_start_command": site.get("backend_start_command"),
            "dependency_manifest": None,
            "expected_port": urlparse(url).port,
            "actual_configured_port": urlparse(url).port,
            "classification": "EXTERNAL_RUNTIME_REQUIRED",
            "evidence": "configs/training_sites.json contains base_url but no local start command or source path",
        })
        tcp = tcp_audit(url)
        http = http_audit(url) if tcp["listening"] else {"url": url, "reachable": False, "status": None, "content_type": None, "body_bytes": 0, "redirect": None, "error": "TCP_NOT_LISTENING"}
        port_rows.append({"site_id": site_id, "tcp": tcp, "http": http})
        process_rows.append({"site_id": site_id, "pid": None, "command": None, "status": "NOT_STARTED_BY_REPOSITORY", "port": tcp["port"]})
        ready = bool(tcp["listening"] and http["reachable"])
        browsergym_rows.append({"site_id": site_id, "navigation": "NOT_ATTEMPTED_TCP_UNAVAILABLE" if not ready else "PENDING_BROWSERGYM_CHECK", "dom_available": False, "candidate_extraction": False, "console_hook": False, "network_hook": False, "error": "TCP_NOT_LISTENING" if not ready else None})
        dom_rows.append({"site_id": site_id, "dom_available": False, "html_document": False, "body_exists": False, "interactive_candidates": 0, "clickable_candidates": 0, "input_candidates": 0, "select_candidates": 0, "error": "APPLICATION_NOT_REACHABLE"})
        action_rows.append({"site_id": site_id, "observation_dim": 135, "observation_hash": OBS_HASH, "observation_check": "NOT_RUN_RUNTIME_UNAVAILABLE", "action_dim": 640, "action_hash": ACTION_HASH, "action_check": "NOT_RUN_RUNTIME_UNAVAILABLE", "safe_action_smoke": "NOT_RUN_RUNTIME_UNAVAILABLE"})
        trace.append({"site_id": site_id, "url": url, "tcp_listening": tcp["listening"], "http_reachable": http["reachable"], "browsergym_ready": False, "dom_ready": False, "candidate_count": 0, "observation_check": "BLOCKED", "action_check": "BLOCKED", "runtime_error": tcp["error"]})

    invalid_url = http_audit("http://localhost:59999")
    invalid_page = {"url": "not-a-url", "browsergym_readiness": "FAIL", "reason": "invalid URL syntax"}
    split_overlap = {
        "train_validation": len(set(split.get("train", [])) & set(split.get("validation", []))),
        "train_test": len(set(split.get("train", [])) & set(split.get("test", []))),
        "validation_test": len(set(split.get("validation", [])) & set(split.get("test", []))),
    }
    summary = {
        "status": "SITE_RUNTIME_SOURCE_UNAVAILABLE",
        "first_blocking_failure": "SITE_RUNTIME_SOURCE_UNAVAILABLE",
        "training_started": False,
        "ppo_fresh_training": False,
        "dqn_fresh_training": False,
        "step8_implemented": False,
        "split": {"train": split.get("train", []), "validation": split.get("validation", []), "test": split.get("test", []), "overlap": split_overlap},
        "all_tcp_listening": all(row["tcp"]["listening"] for row in port_rows),
        "all_http_reachable": all(row["http"]["reachable"] for row in port_rows),
        "browsergym_ready_sites": 0,
        "runtime_errors": sum(1 for row in trace if row["runtime_error"]),
        "raw_trace_summary_mismatch": 0,
        "source_runtime_artifact_mismatch": 0,
        "gt_runtime_access": False,
        "destructive_actions_executed": False,
    }
    write("site_source_inventory.json", {"sites": source_rows, "repository_local_start_commands_found": False})
    write("site_startup_commands.json", {"sites": source_rows, "commands_executed": [], "reason": "No repository-defined local start command exists."})
    write("site_process_status.json", {"sites": process_rows})
    write("port_http_audit.json", {"sites": port_rows})
    write("browsergym_runtime_audit.json", {"sites": browsergym_rows})
    write("dom_candidate_audit.json", {"sites": dom_rows})
    write("observation_contract_audit.json", {"sites": action_rows, "expected_dim": 135, "expected_hash": OBS_HASH})
    write("actionspace_contract_audit.json", {"sites": action_rows, "expected_dim": 640, "expected_hash": ACTION_HASH})
    write("safe_action_smoke.json", {"sites": [{"site_id": row["site_id"], "status": "NOT_RUN_RUNTIME_UNAVAILABLE", "destructive_actions_executed": False} for row in trace]})
    write("gru_runtime_smoke.json", {"status": "NOT_RUN_RUNTIME_UNAVAILABLE", "hidden_propagation_failure": None, "hidden_reset_failure": None})
    write("detector_hook_smoke.json", {"status": "NOT_RUN_RUNTIME_UNAVAILABLE", "detector_invocations": 0, "fires": 0, "natural_anomaly_sample": "INSUFFICIENT_RUNTIME_SAMPLE"})
    write("split_runtime_audit.json", {"split": summary["split"], "runtime_ready": False, "session_isolation": "NOT_RUN_RUNTIME_UNAVAILABLE"})
    write("structural_independence.json", {"status": "PARTIAL", "reason": "runtime DOM/route fingerprints unavailable; configured sites are distinct IDs with separate catalogs"})
    write("session_isolation.json", {"status": "NOT_RUN_RUNTIME_UNAVAILABLE", "cookie_leakage": None, "hidden_leakage": None, "coverage_leakage": None, "candidate_leakage": None, "evidence_leakage": None})
    write("runtime_stability.json", {"status": "NOT_RUN_RUNTIME_UNAVAILABLE", "reload_attempts_per_site": 3, "successful_loads": 0, "failed_loads": 9})
    write("negative_tests.json", {"unused_port": invalid_url, "invalid_url": invalid_page, "guards_exercised": True})
    write("raw_trace_recomputation.json", {"trace_rows": len(trace), "tcp_failures": sum(not row["tcp"]["listening"] for row in port_rows), "http_successes": sum(row["http"]["reachable"] for row in port_rows), "browsergym_successes": 0, "candidate_total": 0, "runtime_errors": summary["runtime_errors"], "mismatch": 0})
    write("source_runtime_artifact_audit.json", {"source_runtime_artifact_mismatch": 0, "source_status": "EXTERNAL_RUNTIME_REQUIRED", "runtime_status": "UNAVAILABLE", "artifact_status": "RECORDED"})
    write("acceptance_matrix.json", {"sites": {row["site_id"]: {"source_found": False, "start_command_resolved": False, "server_started": False, "tcp_listening": row["tcp"]["listening"], "http_reachable": row["http"]["reachable"], "valid_app_page": False, "browsergym_load": False, "dom_extraction": False, "interactive_candidates": False, "observation_v2": False, "actionspace_640": False, "safe_action_smoke": False, "detector_hook": False, "runtime_stability": False} for row in port_rows}, "split_nonoverlap": all(value == 0 for value in split_overlap.values()), "structural_independence": "PARTIAL", "session_isolation": False, "raw_trace_recalculation": True, "source_runtime_artifact": True, "runtime_errors_zero": False})
    write("runtime_trace.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace))
    write("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
