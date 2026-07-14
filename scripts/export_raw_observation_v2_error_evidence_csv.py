from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUMMARY_COLUMNS = [
    "error_case_id",
    "run_id",
    "site_id",
    "port",
    "base_url",
    "episode",
    "start_step",
    "end_step",
    "error_type",
    "action_sequence",
    "observed_state_sequence",
    "state_combination",
    "final_error_label",
    "final_error_reason",
    "evidence_count",
    "source_run_path",
]

DETAIL_COLUMNS = [
    "error_case_id",
    "evidence_order",
    "run_id",
    "site_id",
    "port",
    "episode",
    "step",
    "action_type",
    "action_target",
    "observation_state",
    "observed_value",
    "reward",
    "evidence_text",
    "supports_error",
    "related_error_type",
    "raw_observation_keys",
    "observation_context",
]

RUN_ID_PATTERNS = [
    re.compile(r"^ports_9000_9000_.*"),
    re.compile(r"^ports_9001_9004_.*"),
    re.compile(r"^ports_9005_9009_.*"),
    re.compile(r"^ports_9010_9014_.*"),
    re.compile(r"^ports_9015_9019_.*"),
    re.compile(r"^ports_9020_9024_.*"),
    re.compile(r"^ports_9025_9029_.*"),
    re.compile(r"^ports_9030_9034_.*"),
    re.compile(r"^ports_9035_9039_.*"),
    re.compile(r"^ports_9040_9044_.*"),
    re.compile(r"^ports_9045_9049_.*"),
    re.compile(r"^ports_9050_9054_.*"),
    re.compile(r"^ports_9055_9059_.*"),
    re.compile(r"^ports_9060_9064_.*"),
    re.compile(r"^ports_9065_9069_.*"),
    re.compile(r"^ports_9070_9074_.*"),
    re.compile(r"^ports_9075_9079_.*"),
    re.compile(r"^ports_9080_9084_.*"),
    re.compile(r"^ports_9085_9089_.*"),
    re.compile(r"^ports_9090_9094_.*"),
    re.compile(r"^ports_9095_9099_.*"),
    re.compile(r"^ports_9100_9100_.*"),
]


@dataclass(frozen=True)
class RunInput:
    run_id: str
    run_path: Path
    episode_steps: Path
    anomaly_logs: Path | None
    active_sites: Path | None
    failed_sites: Path | None


@dataclass(frozen=True)
class EvidenceItem:
    run_id: str
    site_id: str
    port: str
    base_url: str
    episode: str
    step: str
    action_type: str
    action_target: str
    observation_state: str
    observed_value: str
    reward: str
    evidence_text: str
    supports_error: str
    raw_observation_keys: str
    observation_context: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Export raw observation v2 error evidence summary/detail CSVs.")
    parser.add_argument("--auto", action="store_true", help="Scan artifacts/training and artifacts/evaluations.")
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--output-dir", default="artifacts/final")
    parser.add_argument("--run-id", action="append", default=[], help="Specific run_id to export. Can be repeated.")
    args = parser.parse_args()

    if args.sequence_length < 1:
        raise SystemExit("--sequence-length must be >= 1")

    runs = discover_runs(Path("artifacts"), requested_run_ids=set(args.run_id))
    if not runs:
        raise SystemExit("No matching runs found under artifacts/training or artifacts/evaluations.")

    summary_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, str]] = []
    error_case_index = 1

    for run in runs:
        step_rows = read_csv(run.episode_steps)
        if not step_rows:
            warn(f"empty or unreadable episode_step_logs.csv: {run.episode_steps}")
            continue
        anomaly_index = load_anomaly_index(run.anomaly_logs)
        site_status = load_site_status(run.active_sites, run.failed_sites)
        grouped = group_steps(step_rows)

        for (_site_id, _episode), rows in sorted(grouped.items()):
            ordered = sorted(rows, key=lambda row: (int_or_max(cell(row, "step_id", "step")), cell(row, "step_id", "step")))
            for window in sliding_windows(ordered, args.sequence_length):
                evidence_items = [
                    build_evidence_item(run, row, anomaly_index.get(step_key(row), []), site_status)
                    for row in window
                ]
                error_type, final_label, final_reason = classify_chain(evidence_items)
                error_case_id = f"E{error_case_index:06d}"
                summary_rows.append(build_summary_row(error_case_id, run, evidence_items, error_type, final_label, final_reason))
                for order, item in enumerate(evidence_items, start=1):
                    detail_rows.append(build_detail_row(error_case_id, order, item, error_type))
                error_case_index += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "raw_observation_v2_error_evidence_summary.csv"
    detail_path = output_dir / "raw_observation_v2_error_evidence_detail.csv"
    write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    write_csv(detail_path, DETAIL_COLUMNS, detail_rows)
    print(f"generated: {summary_path} rows={len(summary_rows)}")
    print(f"generated: {detail_path} rows={len(detail_rows)}")
    return 0


def discover_runs(artifact_root: Path, requested_run_ids: set[str]) -> list[RunInput]:
    runs: list[RunInput] = []
    for kind in ("training", "evaluations"):
        base = artifact_root / kind
        if not base.exists():
            warn(f"missing directory: {base}")
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            run_id = child.name
            if requested_run_ids:
                if run_id not in requested_run_ids:
                    continue
            elif not matches_run_id_pattern(run_id):
                continue
            episode_steps = child / "episode_step_logs.csv"
            if not episode_steps.exists():
                warn(f"missing episode_step_logs.csv: {child}")
                continue
            preflight_root = artifact_root / "preflight" / run_id
            optional_paths = {
                "observation_logs.csv": child / "observation_logs.csv",
                "action_space_logs.csv": child / "action_space_logs.csv",
                "reward_logs.csv": child / "reward_logs.csv",
                "active_sites.json": preflight_root / "active_sites.json",
                "failed_sites.json": preflight_root / "failed_sites.json",
            }
            for label, path in optional_paths.items():
                if not path.exists():
                    warn(f"missing optional {label}: {path}")
            runs.append(
                RunInput(
                    run_id=run_id,
                    run_path=child,
                    episode_steps=episode_steps,
                    anomaly_logs=existing(child / "anomaly_logs.csv"),
                    active_sites=existing(preflight_root / "active_sites.json"),
                    failed_sites=existing(preflight_root / "failed_sites.json"),
                )
            )
    return runs


def matches_run_id_pattern(run_id: str) -> bool:
    return any(pattern.match(run_id) for pattern in RUN_ID_PATTERNS)


def existing(path: Path) -> Path | None:
    return path if path.exists() else None


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def warn(message: str) -> None:
    print(f"warning: {message}")


def load_anomaly_index(path: Path | None) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    if path is None:
        return index
    for row in read_csv(path):
        index[step_key(row)].append(row)
    return index


def load_site_status(active_path: Path | None, failed_path: Path | None) -> dict[str, str]:
    status: dict[str, str] = {}
    for site in load_json_sites(active_path):
        site_id = str(site.get("site_id") or site.get("id") or site.get("name") or "")
        if site_id:
            status[site_id] = "active"
    for site in load_json_sites(failed_path):
        site_id = str(site.get("site_id") or site.get("id") or site.get("name") or "")
        if site_id:
            status[site_id] = "failed"
    return status


def load_json_sites(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except json.JSONDecodeError:
        warn(f"invalid json: {path}")
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("sites", "active_sites", "failed_sites"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def group_steps(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(cell(row, "site_id"), cell(row, "episode_id", "episode"))].append(row)
    return grouped


def sliding_windows(rows: list[dict[str, str]], sequence_length: int) -> list[list[dict[str, str]]]:
    if not rows:
        return []
    if len(rows) <= sequence_length:
        return [rows]
    return [rows[index : index + sequence_length] for index in range(0, len(rows) - sequence_length + 1)]


def build_evidence_item(
    run: RunInput,
    row: dict[str, str],
    anomaly_rows: list[dict[str, str]],
    site_status: dict[str, str],
) -> EvidenceItem:
    action_type = cell(row, "selected_action_type", "selected_action_name", "action_type", "action_name") or "unknown_action"
    reward = cell(row, "reward_total", "reward")
    observation_state, observed_value = state_for_action(action_type, reward)
    evidence_text = build_evidence_text(row, anomaly_rows, observation_state, observed_value, site_status)
    supports_error = "true" if observed_value not in NORMAL_VALUES and observed_value != "unknown" else "false"
    return EvidenceItem(
        run_id=cell(row, "run_id") or run.run_id,
        site_id=cell(row, "site_id"),
        port=infer_port(row),
        base_url=cell(row, "base_url", "after_url", "before_url", "url"),
        episode=cell(row, "episode_id", "episode"),
        step=cell(row, "step_id", "step"),
        action_type=action_type,
        action_target=cell(row, "selected_action_selector", "target_selector", "selected_action_text", "target_text"),
        observation_state=observation_state,
        observed_value=observed_value,
        reward=reward,
        evidence_text=evidence_text,
        supports_error=supports_error,
        raw_observation_keys=cell(row, "raw_observation_keys", "before_raw_observation_keys", "after_raw_observation_keys"),
        observation_context=observation_context(row),
    )


NORMAL_VALUES = {"normal", "no_error_log", "no_console_error"}


def state_for_action(action_type: str, reward: str) -> tuple[str, str]:
    action = action_type.lower()
    positive = (number(reward) or 0.0) > 0
    mappings = [
        ("inspect_server_health", "server_health", "unhealthy", "normal"),
        ("inspect_latency", "latency", "high", "normal"),
        ("inspect_server_logs", "server_log", "error_log_found", "no_error_log"),
        ("inspect_runtime_metrics", "runtime_metric", "abnormal", "normal"),
        ("inspect_port_status", "port_status", "abnormal", "normal"),
        ("inspect_network_status", "network_status", "network_error", "normal"),
        ("inspect_api_response", "api_response", "failed", "normal"),
        ("inspect_console_errors", "console_error", "console_error_found", "no_console_error"),
        ("inspect_resource_loading", "resource_loading", "resource_failed", "normal"),
    ]
    for action_name, state, positive_value, negative_value in mappings:
        if action_name in action:
            return state, positive_value if positive else negative_value
    return "unknown_state", "unknown"


def build_evidence_text(
    row: dict[str, str],
    anomaly_rows: list[dict[str, str]],
    observation_state: str,
    observed_value: str,
    site_status: dict[str, str],
) -> str:
    evidence: list[str] = []
    if site_status.get(cell(row, "site_id")) == "failed":
        evidence.append("preflight marked this site as failed")
    for key in (
        "detected_anomaly_types",
        "infra_anomaly_types",
        "matched_bug_ids",
        "exploratory_anomaly_ids",
        "console_errors",
        "action_error",
    ):
        value = cell(row, key)
        if value:
            evidence.append(f"{key}={compact(value, 140)}")
    for anomaly in anomaly_rows:
        summary = cell(anomaly, "evidence_summary", "anomaly_type", "infra_error_type")
        if summary:
            evidence.append(f"anomaly_log={compact(summary, 160)}")
    if evidence:
        return "; ".join(dedupe(evidence))
    return f"{observation_state} observed as {observed_value} after action"


def observation_context(row: dict[str, str]) -> str:
    raw_type = cell(row, "raw_observation_type")
    key_count = cell(row, "raw_observation_key_count", "before_raw_observation_key_count", "after_raw_observation_key_count")
    raw_keys = cell(row, "raw_observation_keys", "before_raw_observation_keys", "after_raw_observation_keys")
    candidate_count = cell(row, "derived_candidate_count", "after_candidate_count", "candidate_count")
    parts = []
    if raw_type:
        parts.append(f"raw_observation_type={raw_type}")
    if key_count:
        parts.append(f"raw_observation_key_count={key_count}")
    if raw_keys:
        parts.append(f"raw_keys={compact(raw_keys, 180)}")
    if candidate_count:
        parts.append(f"candidate_count={candidate_count}")
    for label, keys in [
        ("url", ("after_url", "before_url", "url")),
        ("status", ("response_status", "health_status_code")),
        ("latency_ms", ("response_latency_ms", "health_response_time_ms")),
    ]:
        value = cell(row, *keys)
        if value:
            parts.append(f"{label}={compact(value, 80)}")
    return "; ".join(parts)


def classify_chain(items: list[EvidenceItem]) -> tuple[str, str, str]:
    state_values = {item.observation_state: item.observed_value for item in items}
    if (
        state_values.get("server_health") == "unhealthy"
        and state_values.get("latency") == "high"
        and state_values.get("server_log") == "error_log_found"
    ):
        return (
            "server_timeout",
            "true",
            "server health, latency, and server log evidence are all abnormal, so this chain supports server_timeout.",
        )
    if state_values.get("runtime_metric") == "abnormal" and state_values.get("server_log") == "error_log_found":
        return (
            "runtime_error",
            "true",
            "runtime metrics and server logs are both abnormal, so this chain supports runtime_error.",
        )
    if state_values.get("port_status") == "abnormal" and state_values.get("server_health") == "unhealthy":
        return (
            "server_unavailable",
            "true",
            "port status and server health are both abnormal, so this chain supports server_unavailable.",
        )
    if state_values.get("api_response") == "failed" and state_values.get("server_log") == "error_log_found":
        return (
            "api_failure",
            "true",
            "API response failure and server log error are both present, so this chain supports api_failure.",
        )
    if (
        state_values.get("console_error") == "console_error_found"
        and state_values.get("resource_loading") == "resource_failed"
    ):
        return (
            "frontend_runtime_error",
            "true",
            "console errors and resource loading failures are both present, so this chain supports frontend_runtime_error.",
        )
    if all(item.observed_value in NORMAL_VALUES for item in items):
        return (
            "none",
            "false",
            "all observed values are normal, so this chain does not support an error judgment.",
        )
    if any(item.observed_value not in NORMAL_VALUES and item.observed_value != "unknown" for item in items):
        return (
            infer_partial_error_type(state_values),
            "true",
            "one or more observed states contain abnormal evidence, so this chain supports an error judgment.",
        )
    return (
        "none",
        "false",
        "the state combination does not contain enough error evidence, so it is not judged as an error.",
    )


def infer_partial_error_type(state_values: dict[str, str]) -> str:
    if state_values.get("server_health") == "unhealthy" or state_values.get("latency") == "high":
        return "server_timeout"
    if state_values.get("server_log") == "error_log_found":
        return "server_error"
    if state_values.get("runtime_metric") == "abnormal":
        return "runtime_error"
    if state_values.get("port_status") == "abnormal":
        return "server_unavailable"
    if state_values.get("api_response") == "failed":
        return "api_failure"
    if state_values.get("network_status") == "network_error":
        return "network_error"
    if state_values.get("console_error") == "console_error_found" or state_values.get("resource_loading") == "resource_failed":
        return "frontend_runtime_error"
    return "unknown_error"


def build_summary_row(
    error_case_id: str,
    run: RunInput,
    items: list[EvidenceItem],
    error_type: str,
    final_label: str,
    final_reason: str,
) -> dict[str, str]:
    return {
        "error_case_id": error_case_id,
        "run_id": items[0].run_id,
        "site_id": items[0].site_id,
        "port": items[0].port,
        "base_url": items[0].base_url,
        "episode": items[0].episode,
        "start_step": items[0].step,
        "end_step": items[-1].step,
        "error_type": error_type,
        "action_sequence": " -> ".join(item.action_type for item in items),
        "observed_state_sequence": " -> ".join(f"{item.observation_state}={item.observed_value}" for item in items),
        "state_combination": " + ".join(f"{item.observation_state}={item.observed_value}" for item in items),
        "final_error_label": final_label,
        "final_error_reason": final_reason,
        "evidence_count": str(len(items)),
        "source_run_path": str(run.run_path),
    }


def build_detail_row(error_case_id: str, order: int, item: EvidenceItem, error_type: str) -> dict[str, str]:
    return {
        "error_case_id": error_case_id,
        "evidence_order": str(order),
        "run_id": item.run_id,
        "site_id": item.site_id,
        "port": item.port,
        "episode": item.episode,
        "step": item.step,
        "action_type": item.action_type,
        "action_target": item.action_target,
        "observation_state": item.observation_state,
        "observed_value": item.observed_value,
        "reward": item.reward,
        "evidence_text": item.evidence_text,
        "supports_error": item.supports_error,
        "related_error_type": error_type,
        "raw_observation_keys": item.raw_observation_keys,
        "observation_context": item.observation_context,
    }


def step_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        cell(row, "run_id"),
        cell(row, "site_id"),
        cell(row, "episode_id", "episode"),
        cell(row, "step_id", "step"),
    )


def infer_port(row: dict[str, str]) -> str:
    for key in ("port", "base_url", "before_url", "after_url", "url"):
        value = cell(row, key)
        if not value:
            continue
        if value.isdigit():
            return value
        if "localhost:" in value:
            tail = value.split("localhost:", 1)[1]
            return tail.split("/", 1)[0].split(":", 1)[0]
    site_id = cell(row, "site_id")
    return "".join(ch for ch in site_id if ch.isdigit())


def cell(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def number(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def int_or_max(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 10**9


def compact(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
