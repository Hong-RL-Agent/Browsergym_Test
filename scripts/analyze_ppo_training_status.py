#!/usr/bin/env python
"""Analyze BrowserGym raw-observation PPO training/evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


RUN_RE = re.compile(r"^ports_(\d+)_(\d+)(?:_(\d{14}))?$")
FORBIDDEN_KEYWORDS = [
    "has_data_bug_id",
    "data-bug-id",
    "data_bug_id",
    "bug_id",
    "bugId",
    "matched_bug_id",
    "missed_bug_id",
    "known_bug_id",
    "is_known_bug",
]
FORBIDDEN_SITE_BUG_RE = re.compile(r"site\d+-bug\d+", re.IGNORECASE)
POLICY_INPUT_FILES = {
    Path("models/observation_encoder.py"),
    Path("models/action_space.py"),
    Path("scripts/models/observation_encoder.py"),
    Path("scripts/models/action_space.py"),
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    return []


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit and len(rows) >= limit:
                break
    return rows


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def split_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
            return split_values(data)
        except Exception:
            pass
    return [part.strip() for part in re.split(r"[;,|]", text) if part.strip()]


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def ratio(part: int, total: int) -> float:
    return part / total if total else 0.0


def entropy_diversity(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 1 or len(counter) <= 1:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(len(counter))


def top_key(counter: Counter[str]) -> str:
    return counter.most_common(1)[0][0] if counter else ""


def compact_counter(counter: Counter[str]) -> str:
    return ";".join(f"{key}:{value}" for key, value in counter.most_common())


def parse_run_id(path: Path) -> str:
    return path.name


def run_sort_key(path: Path) -> tuple[float, str]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (mtime, path.name)


def command_arg(command: list[Any], name: str) -> str:
    values = [str(item) for item in command]
    try:
        idx = values.index(name)
    except ValueError:
        return ""
    return values[idx + 1] if idx + 1 < len(values) else ""


@dataclass
class RunArtifacts:
    run_id: str
    training_dir: Path | None = None
    evaluation_dir: Path | None = None
    training_summary_paths: list[Path] = field(default_factory=list)
    evaluation_summary_paths: list[Path] = field(default_factory=list)


@dataclass
class SiteAnalysis:
    run_id: str
    site_id: str
    target_url: str = ""
    average_reward: float = 0.0
    detected_bug_count: int = 0
    unique_detected_candidates: int = 0
    total_detected_candidates: int = 0
    anomaly_types: Counter[str] = field(default_factory=Counter)
    action_counts: Counter[str] = field(default_factory=Counter)
    repeated_action_count: int = 0
    repeated_click_target_count: int = 0
    action_diversity_score: float = 0.0
    review_candidate_count: int = 0
    needs_review_count: int = 0
    precision: str = ""
    recall: str = ""
    warnings: list[str] = field(default_factory=list)
    diagnosis: str = ""


@dataclass
class RunAnalysis:
    run_id: str
    port_range: str = ""
    total_sites: int = 0
    active_sites: int = 0
    failed_sites: int = 0
    training_success: str = "unknown"
    evaluation_success: str = "unknown"
    failed_batch_count: int = 0
    started_at: str = ""
    finished_at: str = ""
    average_reward: float = 0.0
    total_detected_candidates: int = 0
    unique_detected_candidates: int = 0
    dominant_action_type: str = ""
    action_diversity_score: float = 0.0
    dominant_anomaly_type: str = ""
    raw_observation_status: str = "unknown"
    label_leakage_status: str = "unknown"
    overall_status: str = "warning"
    diagnosis: str = ""
    reward_components: Counter[str] = field(default_factory=Counter)
    reward_component_sums: dict[str, float] = field(default_factory=dict)
    action_counts: Counter[str] = field(default_factory=Counter)
    anomaly_types: Counter[str] = field(default_factory=Counter)
    raw_observation_keys: Counter[str] = field(default_factory=Counter)
    raw_observation_types: Counter[str] = field(default_factory=Counter)
    sites: dict[str, SiteAnalysis] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def discover_runs(root: Path) -> dict[str, RunArtifacts]:
    runs: dict[str, RunArtifacts] = {}
    for base_name, attr in (("training", "training_dir"), ("evaluations", "evaluation_dir")):
        base = root / "artifacts" / base_name
        if not base.exists():
            continue
        for child in base.iterdir():
            if not child.is_dir() or not RUN_RE.match(child.name):
                continue
            artifact = runs.setdefault(child.name, RunArtifacts(run_id=child.name))
            setattr(artifact, attr, child)
    for summary in (root / "artifacts" / "training").glob("batch-*/training_summary.json"):
        data = read_json(summary)
        run_id = command_arg(data.get("command", []), "--run-id")
        if run_id:
            runs.setdefault(run_id, RunArtifacts(run_id=run_id)).training_summary_paths.append(summary)
    for summary in (root / "artifacts" / "evaluations").glob("ports_*/batch-*-evaluation_summary.json"):
        run_id = summary.parent.name
        runs.setdefault(run_id, RunArtifacts(run_id=run_id)).evaluation_summary_paths.append(summary)
    return runs


def select_runs(runs: dict[str, RunArtifacts], auto: bool, latest: bool, run_id: str) -> list[RunArtifacts]:
    if run_id:
        return [runs.setdefault(run_id, RunArtifacts(run_id=run_id))]
    items = list(runs.values())
    if not items:
        return []
    if latest:
        def key(item: RunArtifacts) -> tuple[float, str]:
            paths = [p for p in (item.training_dir, item.evaluation_dir) if p]
            return max((run_sort_key(path) for path in paths), default=(0.0, item.run_id))
        return [max(items, key=key)]
    if auto:
        return sorted(items, key=lambda item: item.run_id)
    return sorted(items, key=lambda item: item.run_id)


def load_result_sites(directory: Path | None) -> dict[str, dict[str, Any]]:
    sites: dict[str, dict[str, Any]] = {}
    if not directory:
        return sites
    for result_path in sorted(directory.glob("*result.json")):
        data = read_json(result_path)
        for site_id, site in (data.get("sites") or {}).items():
            if isinstance(site, dict):
                sites[str(site_id)] = site
    return sites


def load_training_sites(artifact: RunArtifacts, root: Path) -> dict[str, dict[str, Any]]:
    sites: dict[str, dict[str, Any]] = {}
    if artifact.training_dir:
        for summary in artifact.training_dir.glob("**/site*/training_summary.json"):
            data = read_json(summary)
            site_id = str(data.get("site_id") or summary.parent.name)
            sites[site_id] = data
    for summary in artifact.training_summary_paths:
        multisite = summary.parent / "multisite" / "multisite_training_summary.json"
        data = read_json(multisite)
        for site_id, site in (data.get("sites") or {}).items():
            if isinstance(site, dict):
                sites[str(site_id)] = site
    return sites


def update_site_from_summary(site: SiteAnalysis, data: dict[str, Any]) -> None:
    site.target_url = site.target_url or str(data.get("requested_base_url") or data.get("base_url") or "")
    site.average_reward = to_float(
        data.get("average_reward", data.get("average_episode_reward", data.get("raw_average_reward"))),
        site.average_reward,
    )
    site.detected_bug_count = max(site.detected_bug_count, to_int(data.get("detected_bug_count")))
    site.unique_detected_candidates = max(
        site.unique_detected_candidates,
        to_int(data.get("unique_detected_candidates", data.get("deduped_detected_candidates"))),
    )
    site.total_detected_candidates = max(site.total_detected_candidates, to_int(data.get("total_detected_candidates")))
    for key, count in (data.get("action_counts") or {}).items():
        site.action_counts[str(key)] += to_int(count)
    openended = data.get("openended_anomaly_discovery") or {}
    for key, count in (openended.get("anomaly_types") or {}).items():
        site.anomaly_types[str(key)] += to_int(count)
    for key, count in (data.get("anomaly_types") or {}).items():
        site.anomaly_types[str(key)] += to_int(count)
    site.precision = "" if data.get("precision") is None else str(data.get("precision"))
    site.recall = "" if data.get("recall") is None else str(data.get("recall"))
    for warning in data.get("identity_warnings") or data.get("preflight", {}).get("identity_warnings") or []:
        site.warnings.append(str(warning))


def analyze_csv_rows(analysis: RunAnalysis, rows: list[dict[str, str]], anomaly_rows: list[dict[str, str]]) -> None:
    rewards: list[float] = []
    by_site_rewards: dict[str, list[float]] = defaultdict(list)
    previous_action: dict[str, str] = {}
    previous_click: dict[str, str] = {}
    raw_key_seen = False
    raw_source_seen = False
    derived_source_seen = False

    for row in rows:
        site_id = row.get("site_id") or "unknown"
        site = analysis.sites.setdefault(site_id, SiteAnalysis(run_id=analysis.run_id, site_id=site_id))
        site.target_url = site.target_url or row.get("base_url") or row.get("before_url") or ""
        action = row.get("selected_action_type") or row.get("action_type") or ""
        if action:
            analysis.action_counts[action] += 1
            site.action_counts[action] += 1
            prev_key = f"{site_id}:{row.get('episode_id', '')}"
            if previous_action.get(prev_key) == action:
                site.repeated_action_count += 1
            previous_action[prev_key] = action
        if action == "click_element":
            target = row.get("selected_action_selector") or row.get("selected_action_text") or row.get("selected_action_bbox_center_x") or ""
            prev_key = f"{site_id}:{row.get('episode_id', '')}"
            if target and previous_click.get(prev_key) == target:
                site.repeated_click_target_count += 1
            previous_click[prev_key] = target

        reward = to_float(row.get("reward_total", row.get("reward")), 0.0)
        rewards.append(reward)
        by_site_rewards[site_id].append(reward)
        for field in (
            "reward_bug_found",
            "reward_anomaly_found",
            "reward_penalty_repeat",
            "reward_penalty_invalid_action",
            "reward_penalty_noop",
            "infra_reward_total",
        ):
            if field in row:
                analysis.reward_component_sums[field] = analysis.reward_component_sums.get(field, 0.0) + to_float(row.get(field))

        for anomaly_type in split_values(row.get("detected_anomaly_types") or row.get("infra_anomaly_types")):
            analysis.anomaly_types[anomaly_type] += 1
            site.anomaly_types[anomaly_type] += 1
        site.detected_bug_count += to_int(row.get("detected_anomaly_count"))
        site.total_detected_candidates += len(split_values(row.get("detected_anomaly_ids")))

        raw_type = row.get("raw_observation_type") or ""
        if raw_type:
            analysis.raw_observation_types[raw_type] += 1
        key_count = to_int(row.get("raw_observation_key_count"))
        if key_count > 0:
            raw_key_seen = True
        for key in split_values(row.get("raw_observation_keys")):
            analysis.raw_observation_keys[key] += 1
            if "browsergym_raw_obs" in key:
                raw_source_seen = True
            if "derived_from_raw_obs" in key:
                derived_source_seen = True
        summary = row.get("raw_observation_summary_json") or ""
        if "browsergym_raw_obs" in summary:
            raw_source_seen = True
        if "derived_from_raw_obs" in summary:
            derived_source_seen = True

        candidate_count = max(to_int(row.get("derived_candidate_count")), to_int(row.get("after_candidate_count")), to_int(row.get("before_candidate_count")))
        if candidate_count == 0:
            site.warnings.append("candidate_count is zero in at least one step")

    for site_id, values in by_site_rewards.items():
        analysis.sites[site_id].average_reward = mean(values)
    if rewards:
        analysis.average_reward = mean(rewards)
    for site in analysis.sites.values():
        site.action_diversity_score = entropy_diversity(site.action_counts)

    anomaly_ids_by_site: dict[str, set[str]] = defaultdict(set)
    for row in anomaly_rows:
        site_id = row.get("site_id") or "unknown"
        site = analysis.sites.setdefault(site_id, SiteAnalysis(run_id=analysis.run_id, site_id=site_id))
        anomaly_type = row.get("anomaly_type") or ""
        if anomaly_type:
            analysis.anomaly_types[anomaly_type] += 1
            site.anomaly_types[anomaly_type] += 1
        anomaly_id = row.get("anomaly_id") or ""
        if anomaly_id:
            anomaly_ids_by_site[site_id].add(anomaly_id)
        review_status = (row.get("human_review_status") or "").lower()
        if review_status:
            site.review_candidate_count += 1
            if "needs_review" in review_status:
                site.needs_review_count += 1

    for site_id, ids in anomaly_ids_by_site.items():
        site = analysis.sites.setdefault(site_id, SiteAnalysis(run_id=analysis.run_id, site_id=site_id))
        site.unique_detected_candidates = max(site.unique_detected_candidates, len(ids))
        site.total_detected_candidates = max(site.total_detected_candidates, sum(site.anomaly_types.values()))

    if rows:
        if raw_key_seen and (raw_source_seen or analysis.raw_observation_keys or analysis.raw_observation_types):
            analysis.raw_observation_status = "passed"
        elif raw_key_seen:
            analysis.raw_observation_status = "warning"
            analysis.warnings.append("raw observation keys exist, but source markers were not found")
        else:
            analysis.raw_observation_status = "failed"
            analysis.warnings.append("raw observation keys are empty")
        if derived_source_seen:
            analysis.warnings.append("derived_from_raw_obs marker found; verify encoder uses raw data without label leakage")


def enrich_from_jsonl(analysis: RunAnalysis, artifact: RunArtifacts) -> None:
    paths: list[Path] = []
    if artifact.training_dir:
        paths.extend(artifact.training_dir.glob("**/rl_transition_log.jsonl"))
    for summary in artifact.training_summary_paths:
        paths.extend((summary.parent / "multisite").glob("site*/rl_transition_log.jsonl"))
    for path in paths:
        for row in read_jsonl(path):
            site_id = str(row.get("site_id") or path.parent.name)
            site = analysis.sites.setdefault(site_id, SiteAnalysis(run_id=analysis.run_id, site_id=site_id))
            action = row.get("action_type") or (row.get("action") or {}).get("action_type") or ""
            if action:
                site.action_counts[str(action)] += 1
                analysis.action_counts[str(action)] += 1
            reward = to_float(row.get("scaled_reward", row.get("reward")), 0.0)
            if site.average_reward == 0.0:
                site.average_reward = reward
            state = row.get("state_summary") or {}
            site.target_url = site.target_url or str(state.get("url") or "")
            candidate_count = to_int(state.get("candidate_count"))
            if candidate_count == 0:
                site.warnings.append("candidate_count is zero in transition log")
            for anomaly in row.get("anomalies") or []:
                if isinstance(anomaly, dict):
                    anomaly_type = str(anomaly.get("type") or "")
                    if anomaly_type:
                        analysis.anomaly_types[anomaly_type] += 1
                        site.anomaly_types[anomaly_type] += 1


def analyze_run(artifact: RunArtifacts, root: Path, leakage_status: str) -> RunAnalysis:
    analysis = RunAnalysis(run_id=artifact.run_id, label_leakage_status=leakage_status)
    match = RUN_RE.match(artifact.run_id)
    if match:
        analysis.port_range = f"{match.group(1)}-{match.group(2)}"

    training_rows = read_csv_rows((artifact.training_dir or Path()) / "episode_step_logs.csv") if artifact.training_dir else []
    evaluation_rows = read_csv_rows((artifact.evaluation_dir or Path()) / "episode_step_logs.csv") if artifact.evaluation_dir else []
    training_anomalies = read_csv_rows((artifact.training_dir or Path()) / "anomaly_logs.csv") if artifact.training_dir else []
    evaluation_anomalies = read_csv_rows((artifact.evaluation_dir or Path()) / "anomaly_logs.csv") if artifact.evaluation_dir else []

    for summary in artifact.training_summary_paths:
        data = read_json(summary)
        if data:
            analysis.training_success = str(bool(data.get("success"))).lower()
            analysis.started_at = analysis.started_at or str(data.get("started_at") or "")
            analysis.finished_at = analysis.finished_at or str(data.get("finished_at") or "")
            if not data.get("success"):
                analysis.failed_batch_count += 1
    for summary in artifact.evaluation_summary_paths:
        data = read_json(summary)
        if data:
            analysis.evaluation_success = str(bool(data.get("success"))).lower()
            analysis.started_at = analysis.started_at or str(data.get("started_at") or "")
            analysis.finished_at = str(data.get("finished_at") or analysis.finished_at)
            if not data.get("success"):
                analysis.failed_batch_count += 1

    if artifact.training_dir and analysis.training_success == "unknown":
        analysis.training_success = "true" if training_rows else "warning"
    if artifact.evaluation_dir and analysis.evaluation_success == "unknown":
        analysis.evaluation_success = "true" if evaluation_rows or load_result_sites(artifact.evaluation_dir) else "warning"

    for site_id, data in load_training_sites(artifact, root).items():
        update_site_from_summary(analysis.sites.setdefault(site_id, SiteAnalysis(analysis.run_id, site_id)), data)
    for site_id, data in load_result_sites(artifact.evaluation_dir).items():
        update_site_from_summary(analysis.sites.setdefault(site_id, SiteAnalysis(analysis.run_id, site_id)), data)

    analyze_csv_rows(analysis, training_rows + evaluation_rows, training_anomalies + evaluation_anomalies)
    if not (training_rows or evaluation_rows):
        enrich_from_jsonl(analysis, artifact)

    analysis.total_sites = len(analysis.sites)
    analysis.active_sites = sum(1 for site in analysis.sites.values() if site.target_url or site.action_counts)
    analysis.failed_sites = sum(1 for site in analysis.sites.values() if site.warnings)
    analysis.total_detected_candidates = sum(site.total_detected_candidates for site in analysis.sites.values())
    analysis.unique_detected_candidates = sum(site.unique_detected_candidates for site in analysis.sites.values())
    if analysis.average_reward == 0.0:
        rewards = [site.average_reward for site in analysis.sites.values() if site.average_reward != 0.0]
        analysis.average_reward = mean(rewards)
    analysis.dominant_action_type = top_key(analysis.action_counts)
    analysis.action_diversity_score = entropy_diversity(analysis.action_counts)
    analysis.dominant_anomaly_type = top_key(analysis.anomaly_types)
    if analysis.raw_observation_status == "unknown":
        obs_path = (artifact.training_dir or artifact.evaluation_dir or Path()) / "observation_logs.csv"
        obs_rows = read_csv_rows(obs_path)
        analysis.raw_observation_status = "passed" if obs_rows else "warning"

    for site in analysis.sites.values():
        site.action_diversity_score = entropy_diversity(site.action_counts)
        if not site.anomaly_types:
            site.warnings.append("no detected anomaly types")
        if site.action_counts:
            dominant = top_key(site.action_counts)
            dom_ratio = ratio(site.action_counts[dominant], sum(site.action_counts.values()))
            if dom_ratio >= 0.85:
                site.warnings.append(f"dominant action {dominant} ratio {dom_ratio:.2f}")
            if dominant == "change_viewport_mobile":
                site.warnings.append("change_viewport_mobile dominates action selection")
            if dominant == "click_element" and dom_ratio >= 0.85:
                site.warnings.append("click_element dominates action selection")
        if site.average_reward < 0 and site.unique_detected_candidates == 0:
            site.warnings.append("negative reward without unique detections")
        if top_key(site.anomaly_types) == "layout-overlap" and ratio(site.anomaly_types["layout-overlap"], sum(site.anomaly_types.values())) >= 0.85:
            site.warnings.append("layout-overlap dominates anomaly detections")
        site.diagnosis = "warning: " + "; ".join(sorted(set(site.warnings))[:5]) if site.warnings else "healthy signals"

    warnings = list(analysis.warnings)
    if analysis.failed_batch_count > 0:
        warnings.append("failed_batch_count > 0")
    if analysis.evaluation_success == "false":
        warnings.append("evaluation failed")
    if analysis.raw_observation_status == "failed":
        warnings.append("raw observation missing")
    if analysis.action_counts:
        dominant_ratio = ratio(analysis.action_counts[analysis.dominant_action_type], sum(analysis.action_counts.values()))
        if dominant_ratio >= 0.85:
            warnings.append(f"dominant action {analysis.dominant_action_type} ratio {dominant_ratio:.2f}")
    if analysis.average_reward < 0:
        warnings.append("average reward is negative")
    if analysis.unique_detected_candidates == 0:
        warnings.append("no unique detected candidates")
    if analysis.label_leakage_status == "failed":
        warnings.append("label leakage risk in policy input")

    analysis.warnings = sorted(set(warnings))
    if analysis.warnings:
        analysis.overall_status = "warning" if analysis.label_leakage_status != "failed" else "failed"
        analysis.diagnosis = "; ".join(analysis.warnings[:6])
    else:
        analysis.overall_status = "passed"
        analysis.diagnosis = "reward, action diversity, anomaly detection, raw observation, and batch status look healthy"
    return analysis


def scan_label_leakage(root: Path) -> dict[str, Any]:
    suspicious: list[dict[str, Any]] = []
    search_roots = ["models", "scripts", "services", "adapters", "agents", "runners", "envs"]
    excluded = {Path("scripts/analyze_ppo_training_status.py")}
    for base_name in search_roots:
        base = root / base_name
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(root)
            if rel in excluded:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, start=1):
                matched = [word for word in FORBIDDEN_KEYWORDS if word in line]
                if FORBIDDEN_SITE_BUG_RE.search(line):
                    matched.append("siteXXX-bugXX")
                if matched:
                    suspicious.append(
                        {
                            "file": str(rel),
                            "line": number,
                            "keywords": sorted(set(matched)),
                            "text": line.strip()[:240],
                            "policy_input": rel in POLICY_INPUT_FILES,
                        }
                    )
    policy_hits = [item for item in suspicious if item["policy_input"]]
    if policy_hits:
        status = "failed"
        reason = "forbidden bug-id/data-bug-id signals appear in observation/action policy input code"
    elif suspicious:
        status = "warning"
        reason = "bug-id keywords appear outside direct policy input; verify they are limited to reward/evaluation/report matching"
    else:
        status = "passed"
        reason = "no forbidden label keywords found in scanned Python files"
    return {
        "label_leakage_status": status,
        "suspicious_files": sorted({item["file"] for item in suspicious}),
        "suspicious_lines": suspicious,
        "reason": reason,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def summary_row(analysis: RunAnalysis) -> dict[str, Any]:
    return {
        "run_id": analysis.run_id,
        "port_range": analysis.port_range,
        "total_sites": analysis.total_sites,
        "active_sites": analysis.active_sites,
        "failed_sites": analysis.failed_sites,
        "training_success": analysis.training_success,
        "evaluation_success": analysis.evaluation_success,
        "failed_batch_count": analysis.failed_batch_count,
        "average_reward": f"{analysis.average_reward:.4f}",
        "total_detected_candidates": analysis.total_detected_candidates,
        "unique_detected_candidates": analysis.unique_detected_candidates,
        "dominant_action_type": analysis.dominant_action_type,
        "action_diversity_score": f"{analysis.action_diversity_score:.4f}",
        "dominant_anomaly_type": analysis.dominant_anomaly_type,
        "raw_observation_status": analysis.raw_observation_status,
        "label_leakage_status": analysis.label_leakage_status,
        "overall_status": analysis.overall_status,
        "diagnosis": analysis.diagnosis,
    }


def site_rows(analyses: list[RunAnalysis]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        for site in sorted(analysis.sites.values(), key=lambda item: item.site_id):
            rows.append(
                {
                    "run_id": site.run_id,
                    "site_id": site.site_id,
                    "target_url": site.target_url,
                    "average_reward": f"{site.average_reward:.4f}",
                    "detected_bug_count": site.detected_bug_count,
                    "unique_detected_candidates": site.unique_detected_candidates,
                    "anomaly_types": compact_counter(site.anomaly_types),
                    "action_counts": compact_counter(site.action_counts),
                    "repeated_action_count": site.repeated_action_count,
                    "repeated_click_target_count": site.repeated_click_target_count,
                    "action_diversity_score": f"{site.action_diversity_score:.4f}",
                    "review_candidate_count": site.review_candidate_count,
                    "needs_review_count": site.needs_review_count,
                    "precision": site.precision,
                    "recall": site.recall,
                    "warnings": "; ".join(sorted(set(site.warnings))),
                    "diagnosis": site.diagnosis,
                }
            )
    return rows


def make_markdown(analyses: list[RunAnalysis], leakage: dict[str, Any]) -> str:
    lines = [
        "# PPO Training Status Diagnosis",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Overall Training Status",
    ]
    for analysis in analyses:
        lines.append(
            f"- `{analysis.run_id}`: status={analysis.overall_status}, training={analysis.training_success}, "
            f"evaluation={analysis.evaluation_success}, sites={analysis.total_sites}, "
            f"avg_reward={analysis.average_reward:.4f}, unique_candidates={analysis.unique_detected_candidates}"
        )
    lines.extend(["", "## Evidence That Learning Is Progressing"])
    for analysis in analyses:
        positives = []
        if analysis.average_reward >= 0:
            positives.append("average reward is non-negative")
        if analysis.unique_detected_candidates > 0:
            positives.append("unique detected candidates exist")
        if analysis.action_diversity_score >= 0.35:
            positives.append("action type diversity is present")
        if analysis.raw_observation_status == "passed":
            positives.append("raw observation fields are logged")
        if len(analysis.anomaly_types) > 1:
            positives.append("multiple anomaly types are detected")
        if analysis.failed_batch_count == 0:
            positives.append("no failed batch was found")
        lines.append(f"- `{analysis.run_id}`: " + (", ".join(positives) if positives else "no strong positive signal"))
    lines.extend(["", "## Problem Evidence"])
    for analysis in analyses:
        lines.append(f"- `{analysis.run_id}`: " + (analysis.diagnosis or "no major issue"))
    lines.extend(["", "## Action Repetition"])
    for analysis in analyses:
        total_actions = sum(analysis.action_counts.values())
        lines.append(
            f"- `{analysis.run_id}`: dominant={analysis.dominant_action_type}, "
            f"diversity={analysis.action_diversity_score:.4f}, total_actions={total_actions}, "
            f"counts={compact_counter(analysis.action_counts)}"
        )
    lines.extend(["", "## Reward Design"])
    for analysis in analyses:
        components = ", ".join(f"{k}={v:.4f}" for k, v in sorted(analysis.reward_component_sums.items()))
        lines.append(f"- `{analysis.run_id}`: average_reward={analysis.average_reward:.4f}; {components or 'no component CSV fields found'}")
    lines.extend(["", "## Anomaly Detection"])
    for analysis in analyses:
        lines.append(
            f"- `{analysis.run_id}`: unique={analysis.unique_detected_candidates}, total={analysis.total_detected_candidates}, "
            f"dominant={analysis.dominant_anomaly_type}, types={compact_counter(analysis.anomaly_types)}"
        )
    lines.extend(["", "## Raw Observation Usage"])
    for analysis in analyses:
        keys = ", ".join(key for key, _ in analysis.raw_observation_keys.most_common(20))
        types = compact_counter(analysis.raw_observation_types)
        lines.append(f"- `{analysis.run_id}`: status={analysis.raw_observation_status}, types={types}, keys={keys}")
    lines.extend(["", "## Label Leakage Scan"])
    lines.append(f"- status: {leakage['label_leakage_status']}")
    lines.append(f"- reason: {leakage['reason']}")
    for item in leakage["suspicious_lines"][:80]:
        lines.append(f"- {item['file']}:{item['line']} policy_input={item['policy_input']} keywords={','.join(item['keywords'])}")
    if len(leakage["suspicious_lines"]) > 80:
        lines.append(f"- ... {len(leakage['suspicious_lines']) - 80} more suspicious lines omitted")
    lines.extend(["", "## Next Improvements"])
    lines.extend(
        [
            "- Remove bug-id/data-bug-id derived fields from policy input and action ranking if the leakage scan is failed.",
            "- If click_element or change_viewport_mobile dominates, increase exploration pressure or add action repetition suppression.",
            "- If reward is mostly negative, inspect repeat/noop/infra penalties and reward scaling before longer PPO runs.",
            "- If layout-overlap dominates, tune anomaly deduplication and require stronger evidence for repeated layout findings.",
            "- If raw observation is warning/failed, enable detailed CSV observation logging and confirm browsergym_raw_obs source markers.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_html(analyses: list[RunAnalysis], markdown: str) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(a.run_id)}</td>"
        f"<td>{html.escape(a.overall_status)}</td>"
        f"<td>{a.average_reward:.4f}</td>"
        f"<td>{a.unique_detected_candidates}</td>"
        f"<td>{html.escape(a.dominant_action_type)}</td>"
        f"<td>{a.action_diversity_score:.4f}</td>"
        f"<td>{html.escape(a.raw_observation_status)}</td>"
        f"<td>{html.escape(a.label_leakage_status)}</td>"
        f"<td>{html.escape(a.diagnosis)}</td>"
        "</tr>"
        for a in analyses
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>PPO Training Status Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    pre {{ white-space: pre-wrap; background: #f9fafb; border: 1px solid #e5e7eb; padding: 16px; }}
  </style>
</head>
<body>
  <h1>PPO Training Status Report</h1>
  <table>
    <thead>
      <tr><th>run_id</th><th>status</th><th>avg reward</th><th>unique candidates</th><th>dominant action</th><th>diversity</th><th>raw obs</th><th>label leakage</th><th>diagnosis</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <pre>{html.escape(markdown)}</pre>
</body>
</html>
"""


def print_console_summary(analyses: list[RunAnalysis], leakage: dict[str, Any]) -> None:
    total_runs = len(analyses)
    successful_training = sum(1 for item in analyses if item.training_success == "true")
    successful_eval = sum(1 for item in analyses if item.evaluation_success == "true")
    failed_runs = sum(1 for item in analyses if item.overall_status == "failed" or item.failed_batch_count > 0)
    best = max(analyses, key=lambda item: item.average_reward, default=None)
    worst = min(analyses, key=lambda item: item.average_reward, default=None)
    all_actions = Counter()
    all_anomalies = Counter()
    for item in analyses:
        all_actions.update(item.action_counts)
        all_anomalies.update(item.anomaly_types)
    avg_reward = mean(item.average_reward for item in analyses)
    raw_status = "passed" if analyses and all(item.raw_observation_status == "passed" for item in analyses) else "failed"
    diversity = entropy_diversity(all_actions)
    diversity_status = "passed" if diversity >= 0.35 else "warning"
    unique_total = sum(item.unique_detected_candidates for item in analyses)
    print("[TRAINING STATUS SUMMARY]")
    print(f"total_runs={total_runs}")
    print(f"successful_training_runs={successful_training}")
    print(f"successful_evaluation_runs={successful_eval}")
    print(f"failed_runs={failed_runs}")
    print()
    print("[REWARD]")
    print(f"average_reward={avg_reward:.4f}")
    print(f"best_run={best.run_id if best else ''}")
    print(f"worst_run={worst.run_id if worst else ''}")
    print()
    print("[ACTION]")
    print(f"dominant_action_type={top_key(all_actions)}")
    print(f"action_diversity_status={diversity_status}")
    print()
    print("[ANOMALY]")
    print(f"total_unique_candidates={unique_total}")
    print(f"dominant_anomaly_type={top_key(all_anomalies)}")
    print()
    print("[RAW OBSERVATION]")
    print(f"raw_observation_status={raw_status}")
    print()
    print("[LABEL LEAKAGE]")
    print(f"label_leakage_status={leakage['label_leakage_status']}")
    print()
    print("[OVERALL DIAGNOSIS]")
    if not analyses:
        print("분석할 ports_* run을 찾지 못했습니다.")
    else:
        problem_runs = [item for item in analyses if item.overall_status != "passed"]
        if problem_runs:
            print("현재 모델은 일부 학습 신호가 있으나 경고가 있습니다: " + "; ".join(f"{item.run_id}: {item.diagnosis}" for item in problem_runs[:5]))
        else:
            print("현재 모델은 reward, action 다양성, anomaly 탐지, raw observation 로그 기준으로 정상 학습 신호를 보입니다.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", action="store_true", help="Discover and analyze all ports_* run_ids.")
    parser.add_argument("--latest", action="store_true", help="Analyze only the most recently modified ports_* run.")
    parser.add_argument("--run-id", default="", help="Analyze a specific run_id.")
    parser.add_argument("--output-dir", default="artifacts/analysis", help="Directory for CSV/MD/HTML outputs.")
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = root / args.output_dir
    runs = discover_runs(root)
    selected = select_runs(runs, auto=args.auto or not args.run_id, latest=args.latest, run_id=args.run_id)

    leakage = scan_label_leakage(root)
    analyses = [analyze_run(artifact, root, leakage["label_leakage_status"]) for artifact in selected]

    summary_columns = [
        "run_id",
        "port_range",
        "total_sites",
        "active_sites",
        "failed_sites",
        "training_success",
        "evaluation_success",
        "failed_batch_count",
        "average_reward",
        "total_detected_candidates",
        "unique_detected_candidates",
        "dominant_action_type",
        "action_diversity_score",
        "dominant_anomaly_type",
        "raw_observation_status",
        "label_leakage_status",
        "overall_status",
        "diagnosis",
    ]
    detail_columns = [
        "run_id",
        "site_id",
        "target_url",
        "average_reward",
        "detected_bug_count",
        "unique_detected_candidates",
        "anomaly_types",
        "action_counts",
        "repeated_action_count",
        "repeated_click_target_count",
        "action_diversity_score",
        "review_candidate_count",
        "needs_review_count",
        "precision",
        "recall",
        "warnings",
        "diagnosis",
    ]
    write_csv(output_dir / "ppo_training_status_summary.csv", [summary_row(item) for item in analyses], summary_columns)
    write_csv(output_dir / "ppo_training_site_detail.csv", site_rows(analyses), detail_columns)
    markdown = make_markdown(analyses, leakage)
    (output_dir / "ppo_training_diagnosis_report.md").write_text(markdown, encoding="utf-8")
    (output_dir / "ppo_training_status_report.html").write_text(make_html(analyses, markdown), encoding="utf-8")
    (output_dir / "label_leakage_scan.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")

    print_console_summary(analyses, leakage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
