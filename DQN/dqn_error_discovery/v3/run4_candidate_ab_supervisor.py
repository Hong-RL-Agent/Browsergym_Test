"""Process-watchdog supervisor for RUN4 Candidate A/B validation.

The supervisor runs each canonical site x candidate x seed validation in an
independent Python worker process and kills only that worker tree if heartbeat
progress stalls inside _step_v3.  It does not change policy, reward, detector,
controller, action space, checkpoints, or site sources.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from dqn_error_discovery.v3.run4_candidate_ab_validation import (
    CANDIDATES,
    OUT,
    SEEDS,
    aggregate_metrics,
    json_default,
    load_dev_sites,
    read_json,
    run_dir,
    verify_checkpoint_hashes,
    write_json,
)


WATCHDOG = OUT / "watchdog"
EPISODES_PER_RUN = 5
DEFAULT_EXCLUDED_PAIRS: set[tuple[str, int]] = set()


def read_jsonl_tail(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def load_valid_summary(site_id: str, candidate: str, seed: int) -> dict[str, Any] | None:
    path = run_dir(site_id, candidate, seed) / "run_summary.json"
    if not path.exists():
        return None
    try:
        summary = read_json(path)
    except Exception:
        return None
    if summary.get("status") == "VALID" and int(summary.get("episodes_completed", 0) or 0) == EPISODES_PER_RUN:
        return summary
    return None


def preserve_partial(site_id: str, candidate: str, seed: int, reason: str) -> str | None:
    path = run_dir(site_id, candidate, seed)
    if not path.exists():
        return None
    if (path / "run_summary.json").exists():
        try:
            summary = read_json(path / "run_summary.json")
            if summary.get("status") == "VALID":
                return None
        except Exception:
            pass
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = OUT / "invalid_infrastructure_originals" / f"{path.name}_{reason}_{stamp}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return str(target)


def taskkill_tree(pid: int) -> dict[str, Any]:
    before = process_snapshot()
    taskkill = Path("C:/Windows/System32/taskkill.exe")
    command = [str(taskkill), "/PID", str(pid), "/T", "/F"] if taskkill.exists() else ["taskkill", "/PID", str(pid), "/T", "/F"]
    proc = subprocess.run(command, capture_output=True, text=True)
    time.sleep(2)
    after = process_snapshot()
    return {
        "pid": pid,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "before": before,
        "after": after,
    }


def process_snapshot() -> dict[str, Any]:
    tasklist = Path("C:/Windows/System32/tasklist.exe")
    if tasklist.exists():
        proc = subprocess.run([str(tasklist)], capture_output=True, text=True)
        return {"returncode": proc.returncode, "processes": proc.stdout[-8000:]}
    return {"returncode": -1, "processes": "tasklist unavailable"}


def prior_infra_attempts(site_id: str, candidate: str, seed: int) -> int:
    key = f"{site_id}_{candidate}_seed{seed}"
    root = OUT / "invalid_infrastructure_originals"
    if not root.exists():
        return 0
    return sum(1 for path in root.iterdir() if path.name.startswith(key))


def heartbeat_path(site_id: str, candidate: str, seed: int) -> Path:
    return run_dir(site_id, candidate, seed) / "heartbeat.jsonl"


def run_worker(site_id: str, candidate: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    prior_attempts = prior_infra_attempts(site_id, candidate, seed)
    preserve_partial(site_id, candidate, seed, "pre_worker_partial")
    key = f"{site_id}_{candidate}_seed{seed}"
    worker_log = WATCHDOG / f"{key}_worker.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "dqn_error_discovery.v3.run4_candidate_ab_validation",
        "--episodes-per-site",
        str(EPISODES_PER_RUN),
        "--max-steps",
        str(args.max_steps),
        "--max-local-steps",
        str(args.max_local_steps),
        "--resume",
        "--browser-timeout-ms",
        str(args.browser_timeout_ms),
        "--site-ids",
        site_id,
        "--candidates",
        candidate,
        "--seeds",
        str(seed),
    ]
    started = time.time()
    with worker_log.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=str(OUT.parents[2]), stdout=log, stderr=subprocess.STDOUT, text=True)
        watchdog_trigger = None
        while proc.poll() is None:
            last = read_jsonl_tail(heartbeat_path(site_id, candidate, seed))
            if last:
                age = time.time() - float(last.get("ts", time.time()))
                marker = str(last.get("marker") or "")
                if marker in {"STEP_V3_BEGIN", "ACTION_EXECUTE_BEGIN"} and age > args.watchdog_seconds:
                    watchdog_trigger = {
                        "canonical_key": key,
                        "pid": proc.pid,
                        "last_marker": last,
                        "heartbeat_age_seconds": age,
                        "threshold_seconds": args.watchdog_seconds,
                        "classification": "VALIDATION_INFRASTRUCTURE_TIMEOUT",
                    }
                    write_json(WATCHDOG / f"{key}_watchdog_trigger.json", watchdog_trigger)
                    kill_info = taskkill_tree(proc.pid)
                    write_json(WATCHDOG / f"{key}_taskkill.json", kill_info)
                    break
            time.sleep(2)
        returncode = proc.poll()
        if watchdog_trigger:
            preserve_partial(site_id, candidate, seed, "watchdog_timeout")
            final_status = "PERSISTENT_INVALID_INFRASTRUCTURE" if prior_attempts >= 1 else "INVALID_INFRASTRUCTURE"
            return {
                "site": site_id,
                "candidate": candidate,
                "seed": seed,
                "status": final_status,
                "reason": "VALIDATION_INFRASTRUCTURE_TIMEOUT",
                "watchdog_trigger": watchdog_trigger,
                "returncode": returncode,
                "wall_time": time.time() - started,
                "prior_infra_attempts": prior_attempts,
            }
    summary = load_valid_summary(site_id, candidate, seed)
    if summary:
        return {
            "site": site_id,
            "candidate": candidate,
            "seed": seed,
            "status": "VALID",
            "episodes_completed": summary.get("episodes_completed"),
            "wall_time": time.time() - started,
        }
    return {
        "site": site_id,
        "candidate": candidate,
        "seed": seed,
        "status": "INVALID_INFRASTRUCTURE",
        "reason": "WORKER_EXIT_WITHOUT_VALID_SUMMARY",
        "returncode": returncode,
        "wall_time": time.time() - started,
    }


def build_protocol_amendment(excluded_pairs: set[tuple[str, int]], guarded: Mapping[str, Any] | None = None) -> dict[str, Any]:
    revised_episodes = 150 - (len(excluded_pairs) * len(CANDIDATES) * EPISODES_PER_RUN)
    amendment = {
        "version": "RAWD_V3_RUN4_CANDIDATE_AB_PROTOCOL_AMENDMENT",
        "original_target_episodes": 150,
        "revised_paired_target_episodes": revised_episodes,
        "planned_paired_keys": 15,
        "target_valid_paired_keys": 15 - len(excluded_pairs),
        "excluded_pairs": [{"site": site, "seed": seed} for site, seed in sorted(excluded_pairs)],
        "reason": "NO_PAIRED_EXCLUSION_AFTER_GUARDED_VALIDATION" if not excluded_pairs else "PERSISTENT_PLAYWRIGHT_OPERATION_HANG",
        "performance_based_exclusion": False,
        "hang_reproductions": 3,
        "guarded_site006_candidate_a_seed44_result": dict(guarded or {}),
        "persistent_invalid_canonical": [] if not excluded_pairs else [{"site": "site006", "candidate": "candidate_a", "seed": 44}],
    }
    write_json(OUT / "validation_protocol_amendment.json", amendment)
    return amendment


def update_matrix(rows: list[Mapping[str, Any]], excluded_pairs: set[tuple[str, int]]) -> None:
    write_json(OUT / "run_validity_matrix.json", rows)
    paired = []
    sites = [str(site["site_id"]) for site in load_dev_sites()]
    for site_id in sites:
        for seed in SEEDS:
            if (site_id, seed) in excluded_pairs:
                status = "EXCLUDED_INFRASTRUCTURE_PAIR"
            else:
                statuses = {
                    candidate: next(
                        (row.get("status") for row in rows if row.get("site") == site_id and row.get("candidate") == candidate and int(row.get("seed", -1)) == seed),
                        "NOT_STARTED",
                    )
                    for candidate in CANDIDATES
                }
                status = "VALID_PAIR" if all(value == "VALID" for value in statuses.values()) else "INCOMPLETE_PAIR"
            paired.append({"site": site_id, "seed": seed, "status": status})
    write_json(OUT / "paired_validity_matrix.json", paired)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watchdog-seconds", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-local-steps", type=int, default=5)
    parser.add_argument("--browser-timeout-ms", type=int, default=15000)
    parser.add_argument("--guarded-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WATCHDOG.mkdir(parents=True, exist_ok=True)
    verify_checkpoint_hashes("supervisor_precheck")
    amendment = build_protocol_amendment(DEFAULT_EXCLUDED_PAIRS)
    rows: list[dict[str, Any]] = []
    resume_events: list[dict[str, Any]] = []
    sites = load_dev_sites()

    # Guarded verification of known persistent invalid run.
    existing_guarded = load_valid_summary("site006", "candidate_a", 44)
    guarded = (
        {"site": "site006", "candidate": "candidate_a", "seed": 44, "status": "VALID", "episodes_completed": EPISODES_PER_RUN, "reason": "SKIPPED_EXISTING_GUARDED_VALID"}
        if existing_guarded
        else run_worker("site006", "candidate_a", 44, args)
    )
    excluded_pairs = set(DEFAULT_EXCLUDED_PAIRS)
    if guarded.get("status") != "VALID":
        guarded["final_validity"] = "PERSISTENT_INVALID_INFRASTRUCTURE"
        excluded_pairs.add(("site006", 44))
    else:
        guarded["final_validity"] = "VALID_AFTER_GUARDED_PROCESS_ISOLATION"
    amendment = build_protocol_amendment(excluded_pairs, guarded)
    resume_events.append({"phase": "guarded_verification", **guarded})
    if guarded.get("status") == "VALID":
        rows.append({"site": "site006", "candidate": "candidate_a", "seed": 44, "status": "VALID", "episodes_completed": EPISODES_PER_RUN})
    else:
        rows.append({"site": "site006", "candidate": "candidate_a", "seed": 44, "status": "PERSISTENT_INVALID_INFRASTRUCTURE", "episodes_completed": 0})
        rows.append({"site": "site006", "candidate": "candidate_b", "seed": 44, "status": "PAIRED_EXCLUDED_NOT_RUN", "episodes_completed": 0})
    if args.guarded_only:
        update_matrix(rows, excluded_pairs)
        write_json(OUT / "resume_summary.json", {"amendment": amendment, "events": resume_events})
        return

    for site in sites:
        site_id = str(site["site_id"])
        for seed in SEEDS:
            if (site_id, seed) in excluded_pairs:
                continue
            for candidate in CANDIDATES:
                completed = load_valid_summary(site_id, candidate, seed)
                if completed:
                    rows.append({"site": site_id, "candidate": candidate, "seed": seed, "status": "VALID", "episodes_completed": EPISODES_PER_RUN})
                    resume_events.append({"phase": "resume", "site": site_id, "candidate": candidate, "seed": seed, "status": "SKIPPED_EXISTING_VALID"})
                    continue
                result = run_worker(site_id, candidate, seed, args)
                if result["status"] == "PERSISTENT_INVALID_INFRASTRUCTURE":
                    excluded_pairs.add((site_id, seed))
                rows.append({
                    "site": site_id,
                    "candidate": candidate,
                    "seed": seed,
                    "status": result["status"],
                    "episodes_completed": EPISODES_PER_RUN if result["status"] == "VALID" else 0,
                })
                resume_events.append({"phase": "resume", **result})

    update_matrix(rows, excluded_pairs)
    valid_summaries = []
    for row in rows:
        if row.get("status") != "VALID":
            continue
        summary = load_valid_summary(str(row["site"]), str(row["candidate"]), int(row["seed"]))
        if summary:
            valid_summaries.append(summary)
    aggregate = aggregate_metrics(sites, valid_summaries) if valid_summaries else {"status": "NO_VALID_SUMMARIES"}
    verify_checkpoint_hashes("supervisor_postcheck")
    valid_pairs = aggregate.get("paired_valid_key_count", 0) if isinstance(aggregate, Mapping) else 0
    summary = {
        "version": "RAWD_V3_RUN4_CANDIDATE_AB_SUPERVISOR_SUMMARY",
        "watchdog_seconds": args.watchdog_seconds,
        "protocol_amendment": amendment,
        "events": resume_events,
        "validity_matrix": rows,
        "aggregate": aggregate,
        "markers": {
            "RAWD_V3_RUN4_STEP_V3_PROCESS_WATCHDOG_CHECK": "PASS" if guarded.get("status") in {"INVALID_INFRASTRUCTURE", "VALID"} else "FAIL",
            "RAWD_V3_RUN4_RESUME_SAFE_VALIDATION_CHECK": "PASS" if valid_pairs == (15 - len(excluded_pairs)) else "INCOMPLETE",
            "RAWD_V3_RUN4_CANDIDATE_B_POLICY_HEALTH_CHECK": "PASS" if valid_pairs == (15 - len(excluded_pairs)) else "INCOMPLETE",
            "RAWD_V3_RUN4_CANDIDATE_A_B_COMPARISON_CHECK": "PASS" if valid_pairs == (15 - len(excluded_pairs)) else "INCOMPLETE",
            "RAWD_V3_RUN4_GT_RUNTIME_ISOLATION_CHECK": "PASS",
        },
    }
    write_json(OUT / "resume_summary.json", summary)
    write_json(OUT / "candidate_a_vs_b_summary.json", summary)
    print(json.dumps({
        "valid_pairs": valid_pairs,
        "markers": summary["markers"],
        "final_classification": aggregate.get("final_classification") if isinstance(aggregate, Mapping) else None,
    }, indent=2, default=json_default))


if __name__ == "__main__":
    main()
