from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ATOMIC_COLUMNS = [
    "condition_id",
    "run_id",
    "site_id",
    "port",
    "episode",
    "step",
    "observation_state",
    "observation_keys",
    "observation_context",
    "action_type",
    "action_target",
    "reward",
    "condition_result",
    "condition_type",
    "evidence_summary",
    "why_true_or_false",
]

COMBINATION_COLUMNS = [
    "combination_id",
    "run_id",
    "site_id",
    "port",
    "episode",
    "start_step",
    "end_step",
    "condition_ids",
    "observation_action_conditions",
    "action_sequence",
    "true_false_pattern",
    "true_count",
    "false_count",
    "unknown_count",
    "total_reward",
    "predicted_error_type",
    "final_error_label",
    "combination_evidence",
    "rule_explanation",
]

RULE_COLUMNS = [
    "rule_id",
    "error_type",
    "required_condition_pattern",
    "optional_condition_pattern",
    "false_condition_pattern",
    "matched_combination_count",
    "true_combination_count",
    "false_combination_count",
    "confidence",
    "example_site_ids",
    "example_action_sequence",
    "rule_summary",
]

CONDITION_PRIORITY = [
    "server_health_check",
    "latency_check",
    "server_log_check",
    "runtime_metric_check",
    "api_response_check",
    "console_error_check",
    "network_status_check",
    "ui_state_check",
]

DETECTION_ACTION_WORDS = (
    "inspect",
    "check",
    "detect",
    "diagnose",
    "monitor",
    "anomaly",
    "health",
    "latency",
    "metric",
    "log",
    "console",
    "network",
    "api",
    "error",
)


@dataclass(frozen=True)
class RunInputs:
    run_id: str
    root: Path
    kind: str
    episode_steps: Path
    observations: Path | None = None
    action_space: Path | None = None
    anomalies: Path | None = None
    active_sites: Path | None = None
    failed_sites: Path | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export raw observation v2 atomic conditions, rule combinations, and error rules."
    )
    parser.add_argument("--auto", action="store_true", help="Scan artifacts/training, evaluations, and smoke data.")
    parser.add_argument("--run-id", action="append", default=[], help="Specific run_id to export. Can be repeated.")
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--output-dir", default="artifacts/final")
    args = parser.parse_args()

    if args.sequence_length < 1:
        raise SystemExit("--sequence-length must be >= 1")

    artifact_root = Path("artifacts")
    runs = discover_runs(artifact_root, args.run_id, include_smoke=args.auto)
    if not runs:
        raise SystemExit("No input runs found. Expected artifacts/training/{run_id}/episode_step_logs.csv or equivalents.")

    atomic_rows: list[dict[str, str]] = []
    condition_index = 1
    for run in runs:
        step_rows = read_csv(run.episode_steps)
        anomaly_index = load_anomaly_index(run.anomalies)
        observation_index = (
            load_observation_index(run.observations, {step_key(row) for row in step_rows})
            if needs_observation_log(step_rows)
            else {}
        )
        action_index = (
            load_selected_action_index(run.action_space, {step_key(row) for row in step_rows})
            if needs_action_space_log(step_rows)
            else {}
        )
        site_status = load_site_status(run.active_sites, run.failed_sites)
        for step_row in step_rows:
            row = build_atomic_row(
                condition_index=condition_index,
                run=run,
                step_row=step_row,
                anomaly_rows=anomaly_index.get(step_key(step_row), []),
                observation_row=observation_index.get(step_key(step_row)),
                action_row=action_index.get(step_key(step_row)),
                site_status=site_status,
            )
            atomic_rows.append(row)
            condition_index += 1

    combination_rows = build_combinations(atomic_rows, args.sequence_length)
    rule_rows = build_error_rules(combination_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "raw_observation_v2_atomic_condition_dataset.csv", ATOMIC_COLUMNS, atomic_rows)
    write_csv(output_dir / "raw_observation_v2_rule_combination_dataset.csv", COMBINATION_COLUMNS, combination_rows)
    write_csv(output_dir / "raw_observation_v2_error_rule_dataset.csv", RULE_COLUMNS, rule_rows)

    print(f"wrote {len(atomic_rows)} atomic conditions")
    print(f"wrote {len(combination_rows)} rule combinations")
    print(f"wrote {len(rule_rows)} error rules")
    return 0


def discover_runs(artifact_root: Path, requested_run_ids: list[str], include_smoke: bool) -> list[RunInputs]:
    candidates: list[tuple[str, Path, str]] = []
    for kind in ("training", "evaluations"):
        base = artifact_root / kind
        if base.exists():
            candidates.extend((child.name, child, kind) for child in sorted(base.iterdir()) if child.is_dir())

    if include_smoke:
        smoke_base = artifact_root / "csv_smoke"
        if smoke_base.exists():
            candidates.extend((child.name, child, "csv_smoke") for child in sorted(smoke_base.iterdir()) if child.is_dir())

    requested = set(requested_run_ids)
    runs: list[RunInputs] = []
    seen: set[tuple[str, Path]] = set()
    for run_id, root, kind in candidates:
        if requested and run_id not in requested:
            continue
        episode_steps = root / "episode_step_logs.csv"
        if not episode_steps.exists():
            continue
        marker = (run_id, root)
        if marker in seen:
            continue
        seen.add(marker)
        preflight_root = artifact_root / "preflight" / run_id
        runs.append(
            RunInputs(
                run_id=run_id,
                root=root,
                kind=kind,
                episode_steps=episode_steps,
                observations=existing(root / "observation_logs.csv"),
                action_space=existing(root / "action_space_logs.csv"),
                anomalies=existing(root / "anomaly_logs.csv"),
                active_sites=existing(preflight_root / "active_sites.json"),
                failed_sites=existing(preflight_root / "failed_sites.json"),
            )
        )
    return runs


def existing(path: Path) -> Path | None:
    return path if path.exists() else None


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_anomaly_index(path: Path | None) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        index[step_key(row)].append(row)
    return index


def needs_observation_log(step_rows: list[dict[str, str]]) -> bool:
    if not step_rows:
        return False
    return all(not cell(row, "raw_observation_keys", "before_raw_observation_keys") for row in step_rows[:20])


def needs_action_space_log(step_rows: list[dict[str, str]]) -> bool:
    if not step_rows:
        return False
    return all(not cell(row, "selected_action_type", "selected_action_name") for row in step_rows[:20])


def load_observation_index(
    path: Path | None, wanted_keys: set[tuple[str, str, str, str]]
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    index: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in read_csv(path):
        key = step_key(row)
        if key not in wanted_keys:
            continue
        if key not in index or row.get("source") == "derived_from_raw_obs":
            index[key] = row
    return index


def load_selected_action_index(
    path: Path | None, wanted_keys: set[tuple[str, str, str, str]]
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    index: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in read_csv(path):
        key = step_key(row)
        if key in wanted_keys and is_truthy(row.get("selected")):
            index[step_key(row)] = row
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
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("sites", "active_sites", "failed_sites"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def step_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        cell(row, "run_id"),
        cell(row, "site_id"),
        cell(row, "episode_id", "episode"),
        cell(row, "step_id", "step"),
    )


def build_atomic_row(
    condition_index: int,
    run: RunInputs,
    step_row: dict[str, str],
    anomaly_rows: list[dict[str, str]],
    observation_row: dict[str, str] | None,
    action_row: dict[str, str] | None,
    site_status: dict[str, str],
) -> dict[str, str]:
    merged = {**(observation_row or {}), **step_row, **(action_row or {})}
    run_id = cell(merged, "run_id") or run.run_id
    site_id = cell(merged, "site_id")
    episode = cell(merged, "episode_id", "episode")
    step = cell(merged, "step_id", "step")
    action_type = cell(merged, "selected_action_type", "action_type", "action_name") or "unknown_action"
    action_target = cell(merged, "selected_action_selector", "target_selector", "selected_action_text", "target_text")
    observation_keys = cell(merged, "raw_observation_keys", "browsergym_raw_observation_keys", "before_raw_observation_keys")
    reward = cell(merged, "reward_total", "reward")
    condition_type = infer_condition_type(merged, anomaly_rows)
    observation_state = infer_observation_state(condition_type, merged)
    evidence = collect_evidence(merged, anomaly_rows, site_status.get(site_id, ""))
    condition_result, why = condition_result_for(evidence, reward, action_type, action_target)
    return {
        "condition_id": f"A{condition_index:06d}",
        "run_id": run_id,
        "site_id": site_id,
        "port": infer_port(merged),
        "episode": episode,
        "step": step,
        "observation_state": observation_state,
        "observation_keys": observation_keys,
        "observation_context": observation_context(merged),
        "action_type": action_type,
        "action_target": action_target,
        "reward": reward,
        "condition_result": condition_result,
        "condition_type": condition_type,
        "evidence_summary": "; ".join(evidence) if evidence else "no anomaly evidence",
        "why_true_or_false": why,
    }


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
    digits = "".join(ch for ch in site_id if ch.isdigit())
    return digits


def infer_condition_type(row: dict[str, str], anomaly_rows: list[dict[str, str]]) -> str:
    text = " ".join(
        [
            cell(row, "selected_action_type", "selected_action_name", "action_type", "action_name"),
            cell(row, "selected_action_selector", "selected_action_text", "target_selector", "target_text"),
            cell(row, "detected_anomaly_types", "infra_anomaly_types"),
            " ".join(cell(item, "anomaly_type", "infra_error_type") for item in anomaly_rows),
        ]
    ).lower()
    checks = [
        ("server_health_check", ("health", "connection_refused", "port_open", "process_alive")),
        ("latency_check", ("latency", "timeout", "response_time")),
        ("server_log_check", ("log", "exception", "traceback")),
        ("runtime_metric_check", ("cpu", "memory", "runtime", "process")),
        ("api_response_check", ("api", "response_status", "status_code", "4xx", "5xx")),
        ("console_error_check", ("console", "javascript")),
        ("network_status_check", ("network", "request", "connection")),
        ("ui_state_check", ("layout", "ui", "render", "overflow", "overlap", "validation", "loading")),
    ]
    for condition_type, keywords in checks:
        if any(keyword in text for keyword in keywords):
            return condition_type
    if any_positive(row, "health_check_ok", "port_open", "connection_refused", "server_5xx_count"):
        return "server_health_check"
    if any_positive(row, "response_latency_ms", "timeout_occurred"):
        return "latency_check"
    if any_positive(row, "server_log_exception_count"):
        return "server_log_check"
    if any_positive(row, "cpu_usage_percent", "memory_usage_mb", "process_alive"):
        return "runtime_metric_check"
    if any_positive(row, "response_status", "health_status_code", "server_4xx_count"):
        return "api_response_check"
    if any_positive(row, "console_error_count_after", "console_error_delta"):
        return "console_error_check"
    if any_positive(row, "network_error_count_after", "network_error_delta"):
        return "network_status_check"
    return "ui_state_check"


def infer_observation_state(condition_type: str, row: dict[str, str]) -> str:
    base = condition_type.removesuffix("_check")
    view = f"{base}_view"
    title = cell(row, "after_title", "page_title", "before_title")
    url = cell(row, "after_url", "url", "before_url")
    if title:
        return f"{view}:{compact(title, 48)}"
    if url:
        return f"{view}:{compact(url, 48)}"
    return view


def observation_context(row: dict[str, str]) -> str:
    parts = []
    for label, keys in [
        ("url", ("after_url", "url", "before_url")),
        ("title", ("after_title", "page_title", "before_title")),
        ("text_length", ("browsergym_text_length", "dom_text_length")),
        ("status", ("response_status", "health_status_code")),
        ("latency_ms", ("response_latency_ms", "health_response_time_ms")),
    ]:
        value = cell(row, *keys)
        if value:
            parts.append(f"{label}={compact(value, 80)}")
    return "; ".join(parts)


def collect_evidence(row: dict[str, str], anomaly_rows: list[dict[str, str]], site_status: str) -> list[str]:
    evidence: list[str] = []
    if site_status == "failed":
        evidence.append("preflight site status failed")
    for key in (
        "detected_anomaly_types",
        "infra_anomaly_types",
        "matched_bug_ids",
        "missed_bug_ids",
        "exploratory_anomaly_ids",
    ):
        value = cell(row, key)
        if value:
            evidence.append(f"{key}={compact(value, 120)}")
    numeric_evidence = {
        "detected_anomaly_count": 0,
        "infra_anomaly_count": 0,
        "network_error_delta": 0,
        "console_error_delta": 0,
        "layout_overlap_delta": 0,
        "layout_overflow_delta": 0,
        "server_5xx_count": 0,
        "server_4xx_count": 0,
        "server_log_exception_count": 0,
    }
    for key, threshold in numeric_evidence.items():
        value = number(cell(row, key))
        if value is not None and value > threshold:
            evidence.append(f"{key}={format_number(value)}")
    if is_truthy(cell(row, "connection_refused")):
        evidence.append("connection_refused=true")
    if is_truthy(cell(row, "timeout_occurred")):
        evidence.append("timeout_occurred=true")
    for anomaly in anomaly_rows:
        summary = cell(anomaly, "evidence_summary", "anomaly_type", "infra_error_type")
        if summary:
            evidence.append(f"anomaly_log={compact(summary, 160)}")
    return dedupe(evidence)


def condition_result_for(evidence: list[str], reward: str, action_type: str, action_target: str) -> tuple[str, str]:
    reward_value = number(reward)
    is_detection = is_detection_action(action_type, action_target)
    if evidence:
        return "true", "anomaly evidence is present for this observation-action condition"
    if reward_value is not None and reward_value > 0 and is_detection:
        return "true", "reward is positive and the action is an error detection action"
    if reward_value is not None and reward_value <= 0:
        return "false", "no anomaly evidence is present and reward is not positive"
    return "unknown", "insufficient reward or anomaly evidence to judge this condition"


def is_detection_action(action_type: str, action_target: str) -> bool:
    text = f"{action_type} {action_target}".lower()
    return any(word in text for word in DETECTION_ACTION_WORDS)


def build_combinations(atomic_rows: list[dict[str, str]], sequence_length: int) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in atomic_rows:
        grouped[(row["run_id"], row["site_id"], row["episode"])].append(row)

    combinations: list[dict[str, str]] = []
    combination_index = 1
    for (_run_id, _site_id, _episode), rows in sorted(grouped.items()):
        sorted_rows = sorted(rows, key=lambda item: (int_or_max(item["step"]), item["condition_id"]))
        windows = sliding_windows(sorted_rows, sequence_length)
        for window in windows:
            predicted_error_type = predict_error_type(window)
            final_label = final_error_label(window, predicted_error_type)
            counts = Counter(row["condition_result"] for row in window)
            total_reward = sum(number(row["reward"]) or 0.0 for row in window)
            condition_descriptions = [
                f"{row['observation_state']} + {row['action_type']} = {row['condition_result']}"
                for row in window
            ]
            action_sequence = " -> ".join(row["action_type"] for row in window)
            combinations.append(
                {
                    "combination_id": f"C{combination_index:06d}",
                    "run_id": window[0]["run_id"],
                    "site_id": window[0]["site_id"],
                    "port": window[0]["port"],
                    "episode": window[0]["episode"],
                    "start_step": window[0]["step"],
                    "end_step": window[-1]["step"],
                    "condition_ids": ",".join(row["condition_id"] for row in window),
                    "observation_action_conditions": "; ".join(condition_descriptions),
                    "action_sequence": action_sequence,
                    "true_false_pattern": ",".join(row["condition_result"] for row in window),
                    "true_count": str(counts["true"]),
                    "false_count": str(counts["false"]),
                    "unknown_count": str(counts["unknown"]),
                    "total_reward": format_number(total_reward),
                    "predicted_error_type": predicted_error_type,
                    "final_error_label": final_label,
                    "combination_evidence": combination_evidence(window),
                    "rule_explanation": explain_combination(window, predicted_error_type, final_label),
                }
            )
            combination_index += 1
    return combinations


def sliding_windows(rows: list[dict[str, str]], sequence_length: int) -> list[list[dict[str, str]]]:
    if not rows:
        return []
    if len(rows) <= sequence_length:
        return [rows]
    return [rows[index : index + sequence_length] for index in range(0, len(rows) - sequence_length + 1)]


def predict_error_type(window: list[dict[str, str]]) -> str:
    true_types = {row["condition_type"] for row in window if row["condition_result"] == "true"}
    evidence_text = " ".join(row["evidence_summary"] for row in window).lower()
    if {"server_health_check", "latency_check", "server_log_check"} <= true_types:
        return "server_timeout"
    if "timeout" in evidence_text:
        return "server_timeout"
    if "connection_refused" in evidence_text or "port_open=false" in evidence_text:
        return "server_unreachable"
    if "server_log_check" in true_types or "exception" in evidence_text:
        return "server_exception"
    if "api_response_check" in true_types or "5xx" in evidence_text or "4xx" in evidence_text:
        return "api_response_error"
    if "network_status_check" in true_types:
        return "network_error"
    if "console_error_check" in true_types:
        return "console_error"
    if "runtime_metric_check" in true_types:
        return "runtime_metric_error"
    if "ui_state_check" in true_types:
        return "ui_state_error"
    if any(row["condition_result"] == "unknown" for row in window):
        return "unknown"
    return "no_error"


def final_error_label(window: list[dict[str, str]], predicted_error_type: str) -> str:
    if predicted_error_type not in {"no_error", "unknown"} and any(row["condition_result"] == "true" for row in window):
        return "true"
    if all(row["condition_result"] == "false" for row in window):
        return "false"
    return "unknown"


def combination_evidence(window: list[dict[str, str]]) -> str:
    evidence = [row["evidence_summary"] for row in window if row["condition_result"] == "true"]
    return "; ".join(dedupe(item for item in evidence if item and item != "no anomaly evidence")) or "no positive evidence"


def explain_combination(window: list[dict[str, str]], predicted_error_type: str, final_label: str) -> str:
    true_parts = [f"{row['condition_type']}={row['condition_result']}" for row in window]
    pattern = " + ".join(true_parts)
    if final_label == "true":
        return f"{pattern} indicates {predicted_error_type}."
    if final_label == "false":
        return f"{pattern} does not provide enough positive evidence, so the combination is labeled false."
    return f"{pattern} has incomplete evidence, so the combination remains unknown."


def build_error_rules(combinations: list[dict[str, str]]) -> list[dict[str, str]]:
    by_error: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in combinations:
        by_error[row["predicted_error_type"]].append(row)

    rules: list[dict[str, str]] = []
    for index, (error_type, rows) in enumerate(sorted(by_error.items()), start=1):
        true_rows = [row for row in rows if row["final_error_label"] == "true"]
        false_rows = [row for row in rows if row["final_error_label"] == "false"]
        required = condition_pattern(true_rows, wanted="true", require_all=True)
        optional = condition_pattern(true_rows, wanted="true", require_all=False)
        false_pattern = condition_pattern(false_rows, wanted="false", require_all=False)
        confidence = len(true_rows) / len(rows) if rows else 0.0
        examples = dedupe(row["site_id"] for row in rows if row.get("site_id"))
        example_action = next((row["action_sequence"] for row in true_rows), rows[0]["action_sequence"] if rows else "")
        rules.append(
            {
                "rule_id": f"R{index:04d}",
                "error_type": error_type,
                "required_condition_pattern": required,
                "optional_condition_pattern": optional,
                "false_condition_pattern": false_pattern,
                "matched_combination_count": str(len(rows)),
                "true_combination_count": str(len(true_rows)),
                "false_combination_count": str(len(false_rows)),
                "confidence": f"{confidence:.4f}",
                "example_site_ids": ",".join(examples[:10]),
                "example_action_sequence": example_action,
                "rule_summary": summarize_rule(error_type, required, false_pattern),
            }
        )
    return rules


def condition_pattern(rows: list[dict[str, str]], wanted: str, require_all: bool) -> str:
    if not rows:
        return ""
    parsed = [parse_observation_action_conditions(row["observation_action_conditions"]) for row in rows]
    counts: Counter[str] = Counter()
    for conditions in parsed:
        for condition_type, result in conditions:
            if result == wanted:
                counts[condition_type] += 1
    threshold = len(rows) if require_all else 1
    parts = [
        f"{condition_type}={wanted}"
        for condition_type in CONDITION_PRIORITY
        if counts[condition_type] >= threshold
    ]
    return " + ".join(parts)


def parse_observation_action_conditions(value: str) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for part in value.split(";"):
        if "=" not in part:
            continue
        left, result = part.rsplit("=", 1)
        state = left.split("+", 1)[0].strip()
        condition_type = state.split(":", 1)[0].strip().removesuffix("_view") + "_check"
        parsed.append((condition_type, result.strip()))
    return parsed


def summarize_rule(error_type: str, required: str, false_pattern: str) -> str:
    if required:
        return f"When {required}, the case is classified as {error_type}."
    if false_pattern:
        return f"When {false_pattern}, the case is usually not classified as {error_type}."
    return f"Combinations are grouped under {error_type}, but more examples are needed for a stable rule."


def cell(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def any_positive(row: dict[str, str], *keys: str) -> bool:
    return any((number(cell(row, key)) or 0.0) > 0 for key in keys)


def number(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return 1.0 if lowered == "true" else 0.0
    try:
        return float(text)
    except ValueError:
        return None


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def int_or_max(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 10**9


def is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


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
