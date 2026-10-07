"""Diagnose DQN repeated-action/Q-bias signals without changing training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.agents.dqn_agent import QNetwork, Transition, masked_argmax
from dqn_error_discovery.runners.train_dqn_error_detector import (
    calculate_dqn_reward,
    initialize_dqn_candidate_reward_baseline,
    safe_action_mask,
    sanitize_anomalies,
    _load_site_config,
    _build_functional_local_state_signature,
    _build_raw_local_state_signature,
    _build_viewport_context_signature,
)
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies


COMPONENT_KEYS = (
    "new_local_state",
    "new_local_action_target",
    "flow_progress",
    "candidate_anomaly",
    "confirmed_anomaly",
    "confirmation_upgrade",
    "repeat_action_penalty",
    "repeat_state_penalty",
    "repeat_anomaly_penalty",
    "no_effect_penalty",
    "invalid_penalty",
    "action_error_penalty",
    "timeout_penalty",
    "final_reward",
)


def local_state_components(observation: Mapping[str, Any]) -> dict[str, Any]:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    candidates = observation.get("candidate_elements", []) or []
    candidate_summary: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates[:8]:
            if isinstance(candidate, Mapping):
                candidate_summary.append(_target_identity(candidate) or str(candidate.get("role") or candidate.get("tag") or ""))
    local_bits = {}
    for key in (
        "cart_count",
        "form_state",
        "modal_state",
        "validation_state",
        "selected_filter",
        "current_page",
        "quantity",
        "status",
    ):
        value = page_state.get(key)
        if value not in (None, ""):
            local_bits[key] = value
    return {
        "url_path": _url_path(observation),
        "title": str(page_state.get("title") or ""),
        "dom_node_count": str(page_state.get("dom_node_count") or ""),
        "page_text_length": str(page_state.get("page_text_length") or ""),
        "candidate_summary": ",".join(candidate_summary),
        "local_bits": local_bits,
    }


def compare_local_state_components(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, bool]:
    before_parts = local_state_components(before)
    after_parts = local_state_components(after)
    return {
        "url_changed": before_parts["url_path"] != after_parts["url_path"],
        "title_changed": before_parts["title"] != after_parts["title"],
        "dom_count_changed": before_parts["dom_node_count"] != after_parts["dom_node_count"],
        "text_length_changed": before_parts["page_text_length"] != after_parts["page_text_length"],
        "candidate_summary_changed": before_parts["candidate_summary"] != after_parts["candidate_summary"],
        "local_bits_changed": before_parts["local_bits"] != after_parts["local_bits"],
    }


def semantic_local_state_signature(observation: Mapping[str, Any]) -> str:
    return _build_functional_local_state_signature(observation)


def observation_delta(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(after, dtype=np.float32) - np.asarray(before, dtype=np.float32)
    return {
        "l1": float(np.sum(np.abs(delta))),
        "l2": float(np.linalg.norm(delta)),
        "changed_feature_count": int(np.count_nonzero(np.abs(delta) > 1e-6)),
    }


def aggregate_reward_components(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, float]] = {}
    counts: Counter[str] = Counter()
    for record in records:
        action_type = str(record.get("action_type") or "")
        counts[action_type] += 1
        bucket = buckets.setdefault(action_type, {key: 0.0 for key in COMPONENT_KEYS})
        components = record.get("reward_components", {})
        if not isinstance(components, Mapping):
            continue
        for key in COMPONENT_KEYS:
            bucket[key] += float(components.get(key, 0.0) or 0.0)
    return {
        action_type: {
            "count": float(count),
            **values,
            "mean_final_reward": float(values.get("final_reward", 0.0) / count) if count else 0.0,
        }
        for action_type, count in counts.items()
        for values in [buckets[action_type]]
    }


def q_stats_by_action_type(q_values: np.ndarray, mask: np.ndarray, action_space) -> dict[str, dict[str, float]]:
    rows: dict[str, list[float]] = defaultdict(list)
    for action_id in np.flatnonzero(mask > 0):
        decoded = action_space.decode(int(action_id))
        rows[str(decoded["action_type"])].append(float(q_values[int(action_id)]))
    return {
        action_type: {
            "valid_count": float(len(values)),
            "max_q": float(np.max(values)) if values else 0.0,
            "mean_q": float(np.mean(values)) if values else 0.0,
            "std_q": float(np.std(values)) if values else 0.0,
        }
        for action_type, values in rows.items()
    }


def double_dqn_target_samples(
    online: QNetwork,
    target: QNetwork,
    transitions: Sequence[Transition],
    action_space,
    gamma: float,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    rows = []
    for transition in transitions[:limit]:
        state = torch.from_numpy(transition.state).float().unsqueeze(0)
        next_state = torch.from_numpy(transition.next_state).float().unsqueeze(0)
        next_mask = torch.from_numpy(transition.next_mask).bool().unsqueeze(0)
        with torch.no_grad():
            current_q = online(state)[0, int(transition.action)]
            next_online = online(next_state).masked_fill(~next_mask, -torch.inf)
            next_action = int(next_online.argmax(dim=1).item())
            next_target_q = target(next_state)[0, next_action]
            td_target = float(transition.reward) if transition.done else float(transition.reward) + gamma * float(next_target_q)
        rows.append(
            {
                "action": action_space.decode(int(transition.action)),
                "reward": float(transition.reward),
                "done": bool(transition.done),
                "online_next_argmax_action": action_space.decode(next_action),
                "online_next_argmax_valid": bool(transition.next_mask[next_action] > 0),
                "target_network_q": float(next_target_q),
                "td_target": float(td_target),
                "current_q": float(current_q),
                "td_error": float(td_target - float(current_q)),
            }
        )
    return rows


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    site = _load_site_config(args.config, args.site_id, args.base_url)
    env = BrowserGymJAWSEnv(
        site_id=str(site["site_id"]),
        base_url=str(site["base_url"]),
        max_steps=args.max_steps,
        max_candidates=args.max_candidates,
        headless=args.headless,
    )
    action_space = env.action_space
    network = QNetwork(env.observation_encoder.get_obs_dim(), action_space.get_action_dim())
    checkpoint = torch.load(Path(args.model_path), map_location="cpu")
    network.load_state_dict(checkpoint.get("model", checkpoint))
    network.eval()
    target = QNetwork(env.observation_encoder.get_obs_dim(), action_space.get_action_dim())
    target.load_state_dict(network.state_dict())
    target.eval()

    records: list[dict[str, Any]] = []
    transitions: list[Transition] = []
    replay_actions: Counter[str] = Counter()
    q_gap_values: list[float] = []
    q_type_rows: list[dict[str, dict[str, float]]] = []
    obs_delta_by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_signatures_by_semantic: dict[str, set[str]] = defaultdict(set)
    viewport_signatures: set[str] = set()
    functional_signatures: set[str] = set()
    raw_signatures: set[str] = set()
    functional_state_change_count = 0
    viewport_context_change_count = 0
    viewport_only_change_count = 0
    history: dict[str, Any] = {}
    try:
        for episode in range(1, args.episodes + 1):
            raw, _ = env.reset()
            state = env.observation_encoder.encode_observation(raw)
            history = {}
            initialize_dqn_candidate_reward_baseline(
                history,
                sanitize_anomalies(detect_anomalies(raw, raw, {"action": {"action_type": "noop", "candidate_index": 0}})),
            )
            for step in range(1, args.max_steps + 1):
                mask = safe_action_mask(action_space, raw)
                with torch.no_grad():
                    q_values = network(torch.from_numpy(state).float().unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                action_id = masked_argmax(network, state, mask)
                decoded = action_space.decode(action_id)
                action_type = str(decoded.get("action_type") or "")
                valid_values = q_values[np.flatnonzero(mask > 0)]
                ordered = np.sort(valid_values)[::-1] if len(valid_values) else np.asarray([], dtype=np.float32)
                gap = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0
                q_gap_values.append(gap)
                q_type_rows.append(q_stats_by_action_type(q_values, mask, action_space))
                next_raw, _, done, info = env.step(action_id)
                decoded["failed"] = bool(info.get("last_action_error"))
                decoded["timeout"] = bool(info.get("timeout") or info.get("timed_out") or info.get("timeout_occurred"))
                anomalies = sanitize_anomalies(detect_anomalies(raw, next_raw, {**info, "action": decoded}))
                before_signature = _build_functional_local_state_signature(raw)
                after_signature = _build_functional_local_state_signature(next_raw)
                before_raw_signature = _build_raw_local_state_signature(raw)
                after_raw_signature = _build_raw_local_state_signature(next_raw)
                before_viewport_signature = _build_viewport_context_signature(raw)
                after_viewport_signature = _build_viewport_context_signature(next_raw)
                functional_changed = before_signature != after_signature
                viewport_changed = before_viewport_signature != after_viewport_signature
                if functional_changed:
                    functional_state_change_count += 1
                if viewport_changed:
                    viewport_context_change_count += 1
                if viewport_changed and not functional_changed:
                    viewport_only_change_count += 1
                reward, components = calculate_dqn_reward(raw, next_raw, decoded, anomalies, history)
                next_state = env.observation_encoder.encode_observation(next_raw)
                next_mask = safe_action_mask(action_space, next_raw)
                transition = Transition(state, action_id, reward, next_state, done, next_mask)
                transitions.append(transition)
                replay_actions[action_type] += 1
                delta = observation_delta(state, next_state)
                obs_delta_by_action[action_type].append(delta)
                semantic_sig = semantic_local_state_signature(next_raw)
                raw_signatures_by_semantic[semantic_sig].add(after_raw_signature)
                viewport_signatures.add(after_viewport_signature)
                functional_signatures.add(after_signature)
                raw_signatures.add(after_raw_signature)
                records.append(
                    {
                        "episode": episode,
                        "step": step,
                        "action_id": int(action_id),
                        "action_type": action_type,
                        "before_local_state_signature": before_signature,
                        "after_local_state_signature": after_signature,
                        "before_raw_local_state_signature": before_raw_signature,
                        "after_raw_local_state_signature": after_raw_signature,
                        "before_viewport_context_signature": before_viewport_signature,
                        "after_viewport_context_signature": after_viewport_signature,
                        "signature_changed": functional_changed,
                        "raw_signature_changed": before_raw_signature != after_raw_signature,
                        "viewport_context_changed": viewport_changed,
                        "viewport_only_change": viewport_changed and not functional_changed,
                        "signature_component_changes": compare_local_state_components(raw, next_raw),
                        "before_components": local_state_components(raw),
                        "after_components": local_state_components(next_raw),
                        "reward": float(reward),
                        "reward_components": components,
                        "q_gap_top1_second": gap,
                        "q_stats_by_action_type": q_type_rows[-1],
                        "observation_delta": delta,
                        "candidate_signatures": [str(item.get("signature") or "") for item in anomalies if item.get("signature")],
                    }
                )
                raw, state = next_raw, next_state
                if done:
                    break
    finally:
        env.close()

    manifest = _load_manifest(args.model_path)
    report = {
        "model_path": args.model_path,
        "site_id": site["site_id"],
        "url": site["base_url"],
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "local_state_signature_fields": [
            "url_path",
            "title",
            "local_bits(cart_count,form_state,modal_state,validation_state,selected_filter,current_page,quantity,status)",
        ],
        "viewport_context_signature_fields": [
            "url_path",
            "viewport_width",
            "viewport_height",
            "viewport_type",
            "page_text_length_bucket",
            "visible_candidate_summary_first_8_with_bbox",
        ],
        "scroll_signature_changes": _filter_action_records(records, {"scroll_down", "scroll_up"}),
        "local_state_signature_analysis": _state_fragmentation_summary(raw_signatures_by_semantic),
        "functional_state_change_count": functional_state_change_count,
        "viewport_context_change_count": viewport_context_change_count,
        "viewport_only_change_count": viewport_only_change_count,
        "raw_local_state_count": len(raw_signatures),
        "functional_local_state_count": len(functional_signatures),
        "viewport_context_count": len(viewport_signatures),
        "fragmentation_ratio": float(len(raw_signatures) / len(functional_signatures)) if functional_signatures else 0.0,
        "reward_component_by_action_type": aggregate_reward_components(records),
        "replay_action_distribution": dict(replay_actions),
        "sampled_replay_action_distribution": _sampled_action_distribution(transitions, action_space, args.batch_size, args.sample_batches),
        "q_by_action_type_history": _summarize_q_type_rows(q_type_rows),
        "q_gap_summary": _numeric_summary(q_gap_values),
        "td_target_samples": double_dqn_target_samples(network, target, transitions, action_space, args.gamma),
        "next_mask_consistency": _next_mask_consistency(network, transitions),
        "observation_delta_by_action": _observation_delta_summary(obs_delta_by_action),
        "candidate_signature_by_action": _candidate_signature_by_action(records),
        "training_manifest_summary": _manifest_summary(manifest),
        "root_cause_summary": _root_cause_summary(records, manifest),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _target_identity(candidate: Mapping[str, Any] | None) -> str:
    if not candidate:
        return ""
    for key in ("bid", "entity_id", "product_id", "cart_item_id", "item_id", "post_id", "comment_id", "id"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    return ""


def _url_path(observation: Mapping[str, Any]) -> str:
    page_state = observation.get("page_state", {}) if isinstance(observation.get("page_state"), Mapping) else {}
    value = str(page_state.get("url") or "")
    if not value:
        return ""
    from urllib.parse import urlparse

    return urlparse(value).path or value.split("?", 1)[0] or "/"


def _filter_action_records(records: Sequence[Mapping[str, Any]], action_types: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "episode": record.get("episode"),
            "step": record.get("step"),
            "action_type": record.get("action_type"),
            "signature_changed": record.get("signature_changed"),
            "raw_signature_changed": record.get("raw_signature_changed"),
            "viewport_context_changed": record.get("viewport_context_changed"),
            "viewport_only_change": record.get("viewport_only_change"),
            "signature_component_changes": record.get("signature_component_changes"),
            "reward_components": record.get("reward_components"),
            "observation_delta": record.get("observation_delta"),
        }
        for record in records
        if str(record.get("action_type") or "") in action_types
    ][:80]


def _state_fragmentation_summary(raw_by_semantic: Mapping[str, set[str]]) -> dict[str, Any]:
    raw_counts = [len(values) for values in raw_by_semantic.values()]
    fragmented = [count for count in raw_counts if count > 1]
    return {
        "semantic_state_count": len(raw_by_semantic),
        "raw_local_state_count": sum(raw_counts),
        "fragmented_semantic_state_count": len(fragmented),
        "max_raw_signatures_per_semantic_state": max(raw_counts, default=0),
    }


def _sampled_action_distribution(transitions: Sequence[Transition], action_space, batch_size: int, sample_batches: int) -> dict[str, int]:
    if not transitions:
        return {}
    rng = random.Random(123)
    counts: Counter[str] = Counter()
    size = min(batch_size, len(transitions))
    for _ in range(max(1, sample_batches)):
        for transition in rng.sample(list(transitions), size):
            counts[str(action_space.decode(int(transition.action)).get("action_type") or "")] += 1
    return dict(counts)


def _summarize_q_type_rows(rows: Sequence[Mapping[str, Mapping[str, float]]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for action_type, stats in row.items():
            for key in ("valid_count", "max_q", "mean_q", "std_q"):
                values[action_type][key].append(float(stats.get(key, 0.0) or 0.0))
    return {
        action_type: {f"{key}_mean": float(np.mean(items)) if items else 0.0 for key, items in stats.items()}
        for action_type, stats in values.items()
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def _next_mask_consistency(network: QNetwork, transitions: Sequence[Transition]) -> dict[str, Any]:
    invalid = 0
    empty = 0
    for transition in transitions:
        if not np.any(transition.next_mask > 0):
            empty += 1
            continue
        with torch.no_grad():
            values = network(torch.from_numpy(transition.next_state).float().unsqueeze(0))
            mask = torch.from_numpy(transition.next_mask).bool().unsqueeze(0)
            next_action = int(values.masked_fill(~mask, -torch.inf).argmax(dim=1).item())
        if transition.next_mask[next_action] <= 0:
            invalid += 1
    return {"transition_count": len(transitions), "empty_next_mask_count": empty, "invalid_next_argmax_count": invalid}


def _observation_delta_summary(values: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, float]]:
    summary = {}
    for action_type, rows in values.items():
        summary[action_type] = {
            "count": float(len(rows)),
            "l1_mean": float(np.mean([float(row.get("l1", 0.0) or 0.0) for row in rows])) if rows else 0.0,
            "l2_mean": float(np.mean([float(row.get("l2", 0.0) or 0.0) for row in rows])) if rows else 0.0,
            "changed_feature_count_mean": float(np.mean([float(row.get("changed_feature_count", 0.0) or 0.0) for row in rows])) if rows else 0.0,
        }
    return summary


def _candidate_signature_by_action(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    by_action: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for record in records:
        action_type = str(record.get("action_type") or "")
        counts[action_type] += 1
        for signature in record.get("candidate_signatures", []) or []:
            by_action[action_type].add(str(signature))
    return {
        action_type: {"transition_count": float(counts[action_type]), "unique_candidate_signatures": float(len(signatures))}
        for action_type, signatures in by_action.items()
    }


def _load_manifest(model_path: str) -> dict[str, Any]:
    manifest_path = Path(model_path).with_suffix(".manifest.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(Path(model_path), map_location="cpu")
    manifest = checkpoint.get("manifest", {}) if isinstance(checkpoint, Mapping) else {}
    return dict(manifest) if isinstance(manifest, Mapping) else {}


def _manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episodes": manifest.get("episodes"),
        "epsilon_schedule": manifest.get("epsilon_schedule"),
        "epsilon_trajectory": manifest.get("epsilon_trajectory"),
        "action_distribution": manifest.get("action_distribution"),
        "random_action_distribution": manifest.get("random_action_distribution"),
        "greedy_action_distribution": manifest.get("greedy_action_distribution"),
        "replay_reward_distribution": manifest.get("replay_reward_distribution"),
        "action_type_reward_summary": manifest.get("action_type_reward_summary"),
        "q_value_diagnostic_summary": manifest.get("q_value_diagnostic_summary"),
    }


def _root_cause_summary(records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> list[str]:
    causes = []
    scroll_down = [record for record in records if record.get("action_type") == "scroll_down"]
    if scroll_down:
        changed = sum(1 for record in scroll_down if record.get("signature_changed"))
        viewport_only = sum(1 for record in scroll_down if record.get("viewport_only_change"))
        positive = sum(1 for record in scroll_down if float(record.get("reward", 0.0) or 0.0) > 0)
        if viewport_only / len(scroll_down) >= 0.5:
            causes.append("scroll_down_often_changes_viewport_context_without_functional_state_change")
        if changed / len(scroll_down) >= 0.5 and positive / len(scroll_down) >= 0.5:
            causes.append("scroll_down_often_changes_functional_signature_and_gets_positive_reward")
    action_distribution = manifest.get("action_distribution", {})
    if isinstance(action_distribution, Mapping):
        click_count = int(action_distribution.get("click_element", 0) or 0)
        total = sum(int(value or 0) for value in action_distribution.values())
        if total and click_count / total >= 0.35:
            causes.append("training_replay_click_overrepresented")
    q_summary = manifest.get("q_value_diagnostic_summary", {})
    if isinstance(q_summary, Mapping):
        top_counts = q_summary.get("top_q_action_type_counts", {})
        if isinstance(top_counts, Mapping) and top_counts:
            causes.append("greedy_q_top_actions_concentrated_in_recorded_training_diagnostics")
    return causes


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--site-id", default="site001")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model-path", default="artifacts/models/site001_dqn_rewardv2_diversity_sanity.pt")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-batches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="artifacts/sanity/dqn_site001_qbias_diagnostics.json")
    args = parser.parse_args()
    print(json.dumps(_jsonable(run_diagnostics(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
