"""Build conservative pseudo-catalogs from pass 1 and run deeper fleet training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


EXCLUDED_TYPES = {"action-error", "episode-exception", "low-visibility-target"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass1-summary", default="artifacts/fleet/sites002-090-pass1.json")
    parser.add_argument("--expected-sites", type=int, default=89)
    parser.add_argument("--wait-timeout", type=int, default=21600)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--min-reproductions", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    pass1 = Path(args.pass1_summary)
    _wait_for_pass1(pass1, args.expected_sites, args.wait_timeout)
    report = _build_catalogs(args)
    report_path = Path("artifacts/fleet/phase2_catalog_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[phase2] promoted_catalogs={report['promoted_site_count']}", flush=True)

    command = [
        sys.executable, "-u", "scripts/run_site_fleet_rainbow.py",
        "--start-site", "2", "--end-site", "90",
        "--episodes", str(args.episodes), "--max-steps", str(args.max_steps),
        "--site-timeout", "420", "--output", "artifacts/fleet/sites002-090-phase2.json",
        "--no-resume",
    ]
    return subprocess.run(command).returncode


def _wait_for_pass1(path: Path, expected: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                count = int(payload.get("site_count", 0) or 0)
                print(f"[phase2] waiting pass1 {count}/{expected}", flush=True)
                if count >= expected:
                    return
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(30)
    raise TimeoutError(f"pass 1 did not reach {expected} sites within {timeout}s")


def _build_catalogs(args: argparse.Namespace) -> dict[str, Any]:
    reports = []
    for number in range(2, 91):
        site_id = f"site{number:03d}"
        existing = Path("datasets") / site_id / "bug_catalog.json"
        if existing.exists():
            reports.append({"site_id": site_id, "status": "existing_catalog", "candidate_count": 0})
            continue
        transition_path = Path("artifacts/browsergym") / site_id / "rl_transition_log.jsonl"
        candidates = _catalog_candidates(site_id, transition_path, args)
        if not candidates:
            reports.append({"site_id": site_id, "status": "no_qualified_candidate", "candidate_count": 0})
            continue
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text(
            json.dumps({"site_id": site_id, "generated_by": "fleet-phase2-v1", "bugs": candidates}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        reports.append({"site_id": site_id, "status": "promoted", "candidate_count": len(candidates)})
    return {
        "policy": {
            "min_confidence": args.min_confidence,
            "min_reproductions": args.min_reproductions,
            "max_candidates_per_site": args.max_candidates,
        },
        "promoted_site_count": sum(item["status"] == "promoted" for item in reports),
        "sites": reports,
    }


def _catalog_candidates(site_id: str, path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    groups: dict[tuple[str, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        action = row.get("action", {}) if isinstance(row.get("action"), Mapping) else {}
        for anomaly in row.get("anomalies", []) or []:
            if not isinstance(anomaly, Mapping):
                continue
            anomaly_type = str(anomaly.get("type") or "")
            confidence = float(anomaly.get("confidence", 0.0) or 0.0)
            if not anomaly_type or anomaly_type in EXCLUDED_TYPES or confidence < args.min_confidence:
                continue
            evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
            identity = str(
                evidence.get("clicked_data_bug_id") or evidence.get("data_bug_id")
                or evidence.get("selector_hint") or evidence.get("selector")
                or evidence.get("clicked_bid") or anomaly_type
            )
            groups[(anomaly_type, identity)].append((anomaly, action))

    ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), -max(float(x[0].get("confidence", 0)) for x in item[1])))
    result = []
    used_ids = set()
    for (anomaly_type, identity), samples in ranked:
        if len(samples) < args.min_reproductions:
            continue
        evidence = samples[0][0].get("evidence", {}) or {}
        direct_id = str(evidence.get("clicked_data_bug_id") or evidence.get("data_bug_id") or "")
        bug_id = direct_id if direct_id.startswith(site_id + "-bug") else f"{site_id}-auto-{len(result) + 1:02d}"
        if bug_id in used_ids:
            continue
        used_ids.add(bug_id)
        action_types = sorted({str(action.get("action_type")) for _, action in samples if action.get("action_type")})
        result.append({
            "id": bug_id,
            "bug_id": bug_id,
            "site_id": site_id,
            "type": anomaly_type,
            "selector": str(evidence.get("selector_hint") or evidence.get("selector") or ""),
            "action_hints": action_types,
            "anomaly_evidence_keywords": sorted(str(key) for key in evidence.keys())[:16],
            "discovery_confidence": max(float(sample[0].get("confidence", 0)) for sample in samples),
            "reproduction_count": len(samples),
            "generated": True,
        })
        if len(result) >= args.max_candidates:
            break
    return result


if __name__ == "__main__":
    raise SystemExit(main())
