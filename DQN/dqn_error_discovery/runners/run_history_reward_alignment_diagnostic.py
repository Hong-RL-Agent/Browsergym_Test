"""Run seed/checkpoint history-reward alignment diagnostics without training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.agents.dqn_agent import (
    FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE,
    FactorizedQNetwork,
    factorized_double_dqn_next_actions,
    factorized_q_for_actions,
    factorized_q_value_diagnostics,
    select_factorized_greedy_action,
)
from dqn_error_discovery.agents.confirmation_observation import CONFIRMATION_CONTEXT_DIM, build_dqn_confirmation_context
from dqn_error_discovery.agents.confirmation_reproduction import (
    ACTION_REPRODUCTION,
    ExpectedReproduction,
    is_reproduction_instruction_executable,
)
from dqn_error_discovery.agents.history_observation import (
    HISTORY_CONTEXT_DIM,
    append_history_transition,
    action_identity as history_action_identity,
    augment_dqn_history_observation,
    build_dqn_history_context,
)
from dqn_error_discovery.agents.history_reward_alignment_diagnostics import (
    SCROLL_ACTIONS,
    aggregate_ablation,
    compare_cycle_and_break_q,
    conditional_reward_table,
    confirmation_confounding,
    escape_metric_interpretation,
    feature_ablation_rows,
    greedy_action_id,
    immediate_reward_comparison,
    numeric_summary,
    predictability_classification,
    q_values_for_actions,
    rank_cycle_bias_features,
    reward_timing,
    summarize_predictability,
    zero_history_state,
)
from dqn_error_discovery.runners.run_policy_sanity_check import (
    _candidate_for_decoded_action,
    _dqn_state,
    _load_site_config,
    _sanitize_anomalies,
)
from dqn_error_discovery.runners.train_dqn_error_detector import (
    calculate_dqn_reward,
    initialize_dqn_candidate_reward_baseline,
    safe_action_mask,
)
from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies


def _load_network(model_path: Path, obs_dim: int, action_types: int, max_candidates: int) -> FactorizedQNetwork:
    payload = torch.load(model_path, map_location="cpu")
    manifest = payload.get("manifest", {}) if isinstance(payload, Mapping) else {}
    if manifest.get("dqn_architecture") != FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE:
        raise ValueError("diagnostic requires factorized_v1_confobs_hist checkpoint")
    network = FactorizedQNetwork(obs_dim, action_types, max_candidates)
    network.load_state_dict(payload.get("model", payload))
    network.eval()
    return network


def _history_diag(metadata: Mapping[str, Any], selected_identity: str = "") -> dict[str, Any]:
    history = metadata.get("history_context", {}) if isinstance(metadata.get("history_context"), Mapping) else {}
    recent = history.get("recent_action_identities", []) if isinstance(history, Mapping) else []
    active_cycle = int(history.get("active_cycle_length", 0) or 0) if isinstance(history, Mapping) else 0
    cycle_identity = ""
    if isinstance(recent, list) and active_cycle > 0 and len(recent) >= active_cycle:
        cycle_identity = str(recent[active_cycle - 1] or "")
    previous_unproductive_identity = ""
    outcome = history.get("outcome_flags", []) if isinstance(history, Mapping) else []
    if isinstance(outcome, list) and outcome and isinstance(outcome[0], list) and len(outcome[0]) >= 4 and outcome[0][3]:
        previous_unproductive_identity = str(recent[0] or "") if isinstance(recent, list) and recent else ""
    return {
        "recent_action_identities": list(recent) if isinstance(recent, list) else [],
        "history_valid_mask": list(history.get("history_valid_mask", [])) if isinstance(history, Mapping) else [],
        "history_valid_count": int(sum(int(bool(x)) for x in history.get("history_valid_mask", []))) if isinstance(history, Mapping) else 0,
        "active_cycle_length": active_cycle,
        "same_action_identity_streak": int(history.get("same_action_identity_streak", 0) or 0) if isinstance(history, Mapping) else 0,
        "distinct_action_identity_count": int(history.get("distinct_action_identity_count", 0) or 0) if isinstance(history, Mapping) else 0,
        "outcome_flags": list(history.get("outcome_flags", [])) if isinstance(history, Mapping) else [],
        "cycle_continuation_identity": cycle_identity,
        "previous_unproductive_repeat": bool(previous_unproductive_identity),
        "previous_unproductive_repeat_identity": previous_unproductive_identity,
        "selected_action_identity": selected_identity,
    }


def _zero_greedy_identity(network, state, mask, action_space, observation) -> tuple[int, str]:
    zero_state = zero_history_state(state)
    zero_id = greedy_action_id(network, zero_state, mask, action_space)
    zero_action = action_space.decode(zero_id)
    zero_type = str(zero_action.get("action_type") or "")
    candidate = _candidate_for_decoded_action(observation, zero_action)
    return zero_id, history_action_identity(
        zero_type,
        candidate,
        requires_target=zero_type in {"click_element", "fill_input", "press_enter"},
    )


def _td_decomposition(network, state, action_id, reward, next_state, done, mask, next_mask, action_space, gamma: float) -> dict[str, Any]:
    states = torch.from_numpy(state).float().unsqueeze(0)
    actions = torch.tensor([int(action_id)])
    masks = torch.from_numpy(mask).bool().unsqueeze(0)
    current_q = float(factorized_q_for_actions(network, states, masks, actions, action_space).item())
    if done:
        return {
            "current_q": current_q,
            "reward_t": float(reward),
            "done": True,
            "online_next_greedy_action": "",
            "online_next_q": 0.0,
            "target_next_q": 0.0,
            "gamma_target_q": 0.0,
            "td_target": float(reward),
            "td_error": float(reward) - current_q,
        }
    next_states = torch.from_numpy(next_state).float().unsqueeze(0)
    next_masks = torch.from_numpy(next_mask).bool().unsqueeze(0)
    next_action = factorized_double_dqn_next_actions(network, next_states, next_masks, action_space)
    next_action_id = int(next_action.item())
    target_next_q = float(factorized_q_for_actions(network, next_states, next_masks, next_action.reshape(-1), action_space).item())
    td_target = float(reward) + gamma * target_next_q
    return {
        "current_q": current_q,
        "reward_t": float(reward),
        "done": False,
        "online_next_greedy_action": str(action_space.decode(next_action_id).get("action_type") or ""),
        "online_next_q": target_next_q,
        "target_next_q": target_next_q,
        "gamma_target_q": gamma * target_next_q,
        "td_target": td_target,
        "td_error": td_target - current_q,
    }


def _cycle_window_indexes(rows: list[Mapping[str, Any]]) -> list[int]:
    centers = set()
    for index, row in enumerate(rows):
        repeat = row.get("repeat_diagnostic", {}) if isinstance(row.get("repeat_diagnostic"), Mapping) else {}
        if row.get("action_type") in SCROLL_ACTIONS and bool(repeat.get("cycle_repeat")):
            centers.add(index)
    indexes = set()
    for center in centers:
        for offset in range(-2, 3):
            candidate = center + offset
            if 0 <= candidate < len(rows):
                indexes.add(candidate)
    return sorted(indexes)


def _aggregate_q_margins(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("Q_real_cycle", "Q_real_break", "Q_zero_cycle", "Q_zero_break", "cycle_q_margin_real", "cycle_q_margin_zero", "history_cycle_bias")
    return {key: numeric_summary([float(row["q_margin"][key]) for row in rows if row.get("q_margin")]) for key in keys}


def _td_summary(rows: list[Mapping[str, Any]], *, continue_cycle: bool) -> dict[str, Any]:
    selected = []
    for row in rows:
        history = row.get("history_state_diagnostic", {})
        cycle_identity = str(history.get("cycle_continuation_identity") or "")
        if not cycle_identity:
            continue
        is_continue = str(history.get("selected_action_identity") or "") == cycle_identity
        if is_continue == continue_cycle:
            selected.append(row.get("td_target", {}))
    return {
        "count": len(selected),
        "mean_immediate_reward": numeric_summary([float(row.get("reward_t", 0.0) or 0.0) for row in selected]),
        "mean_bootstrap": numeric_summary([float(row.get("gamma_target_q", 0.0) or 0.0) for row in selected]),
        "mean_td_target": numeric_summary([float(row.get("td_target", 0.0) or 0.0) for row in selected]),
        "bootstrap_scroll_rate": (
            sum(1 for row in selected if str(row.get("online_next_greedy_action") or "") in SCROLL_ACTIONS) / len(selected)
            if selected
            else 0.0
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
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
    obs_dim = env.observation_encoder.get_obs_dim() + CONFIRMATION_CONTEXT_DIM + HISTORY_CONTEXT_DIM
    network = _load_network(Path(args.model_path), obs_dim, len(env.action_space.action_types), env.action_space.max_candidates)
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset()
            reward_history: dict[str, Any] = {}
            initialize_dqn_candidate_reward_baseline(
                reward_history,
                _sanitize_anomalies(detect_anomalies(observation, observation, {"action": {"action_type": "noop", "candidate_index": 0}})),
            )
            for step in range(1, args.max_steps + 1):
                state, metadata = _dqn_state(env, observation, reward_history, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE)
                mask = safe_action_mask(env.action_space, observation)
                action_id = select_factorized_greedy_action(network, state, mask, env.action_space)
                action = env.action_space.decode(action_id)
                action["action_id"] = action_id
                action_type = str(action.get("action_type") or "")
                mode = str(metadata.get("confirmation_mode") or "")
                expected_type = str(metadata.get("expected_action_type") or "")
                if mode == ACTION_REPRODUCTION and expected_type:
                    instruction = ExpectedReproduction(
                        action_type=expected_type,
                        target_identity=str(metadata.get("expected_target_identity") or ""),
                        mode=ACTION_REPRODUCTION,
                        rule_id=str(metadata.get("active_rule_id") or ""),
                    )
                    executable = is_reproduction_instruction_executable(
                        instruction,
                        action_space=env.action_space,
                        observation=observation,
                        mask=mask,
                    )
                    action["confirmation_action_executable"] = bool(executable.get("semantically_executable"))
                    action["confirmation_action_mask_valid"] = bool(executable.get("mask_valid"))
                candidate = _candidate_for_decoded_action(observation, action)
                selected_identity = history_action_identity(
                    action_type,
                    candidate,
                    requires_target=action_type in {"click_element", "fill_input", "press_enter"},
                )
                history_before = _history_diag(metadata, selected_identity)
                zero_id, zero_identity = _zero_greedy_identity(network, state, mask, env.action_space, observation)
                next_observation, _, done, info = env.step(action_id)
                action["failed"] = bool(info.get("last_action_error"))
                action["timeout"] = bool(info.get("timeout") or info.get("timed_out") or info.get("timeout_occurred"))
                anomalies = _sanitize_anomalies(detect_anomalies(observation, next_observation, {**info, "action": action}))
                reward, components = calculate_dqn_reward(observation, next_observation, action, anomalies, reward_history)
                repeat_diagnostic = dict(reward_history.get("last_repeat_diagnostic", {}))
                transition_events = list(reward_history.get("last_confirmation_transition_events", []))
                mode_progress = bool(reward_history.get("last_confirmation_progress") and str(reward_history.get("last_confirmation_mode") or "") == mode)
                if FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE:
                    append_history_transition(
                        reward_history,
                        action_type=action_type,
                        candidate=candidate if isinstance(candidate, Mapping) else None,
                        meaningful_action_effect=not bool(repeat_diagnostic.get("no_effect", False)),
                        productive_transition=bool(repeat_diagnostic.get("productive_transition")),
                        confirmation_progress=mode_progress,
                        unproductive_repeat=bool(repeat_diagnostic.get("unproductive_repeat")),
                    )
                next_state, next_metadata = _dqn_state(env, next_observation, reward_history, FACTORIZED_DQN_CONFOBS_HIST_ARCHITECTURE)
                next_mask = safe_action_mask(env.action_space, next_observation)
                q_diag = factorized_q_value_diagnostics(network, state, mask, env.action_space, action_id)
                td = _td_decomposition(network, state, action_id, reward, next_state, done, mask, next_mask, env.action_space, args.gamma)
                relevant_identity = {
                    str(history_before.get("cycle_continuation_identity") or ""),
                    str(history_before.get("previous_unproductive_repeat_identity") or ""),
                }
                relevant_identity.discard("")
                q_margin = None
                feature_rows = []
                outcome_rows = []
                if action_type in SCROLL_ACTIONS and relevant_identity and selected_identity in relevant_identity:
                    q_margin = compare_cycle_and_break_q(network, state, mask, action_id, zero_id, env.action_space)
                    feature_rows = feature_ablation_rows(network, state, mask, action_id, zero_id, env.action_space)
                    outcome_rows = feature_ablation_rows(network, state, mask, action_id, zero_id, env.action_space, outcome_flags=True)
                rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "global_step": (episode - 1) * args.max_steps + step,
                        "recent_action_identities": history_before["recent_action_identities"],
                        "history_valid_mask": history_before["history_valid_mask"],
                        "active_cycle_length": history_before["active_cycle_length"],
                        "streak": history_before["same_action_identity_streak"],
                        "previous_unproductive_flag": history_before["previous_unproductive_repeat"],
                        "action_type": action_type,
                        "action_identity": selected_identity,
                        "zero_history_action_id": zero_id,
                        "zero_history_action": str(env.action_space.decode(zero_id).get("action_type") or ""),
                        "zero_history_identity": zero_identity,
                        "reward": float(reward),
                        "reward_components": dict(components),
                        "reward_timing": reward_timing(components),
                        "repeat_diagnostic": repeat_diagnostic,
                        "next_state_history": _history_diag(next_metadata),
                        "next_state_cycle": int((_history_diag(next_metadata)).get("active_cycle_length", 0) or 0),
                        "confirmation_event": bool(transition_events),
                        "confirmation_transition_events": transition_events,
                        "confirmation_context": dict(metadata),
                        "history_state_diagnostic": history_before,
                        "predictability": predictability_classification(history_before, repeat_diagnostic),
                        "q_diagnostic": q_diag,
                        "q_margin": q_margin,
                        "type_head": {
                            "real": q_diag.get("type_value_by_action_type", {}),
                            "zero": factorized_q_value_diagnostics(network, zero_history_state(state), mask, env.action_space, zero_id).get("type_value_by_action_type", {}),
                        },
                        "candidate_head_influence": {
                            "selected_centered_advantage": q_diag.get("selected_candidate_centered_advantage"),
                            "scroll_up_valid_count": (q_diag.get("candidate_advantage_stats", {}).get("scroll_up", {}) or {}).get("valid_count", 0),
                            "scroll_down_valid_count": (q_diag.get("candidate_advantage_stats", {}).get("scroll_down", {}) or {}).get("valid_count", 0),
                            "candidate_independent_scroll": True,
                        },
                        "td_target": td,
                        "feature_ablation_rows": feature_rows,
                        "outcome_flag_ablation_rows": outcome_rows,
                    }
                )
                observation = next_observation
                if done:
                    break
    finally:
        env.close()

    cycle_trace = [rows[index] for index in _cycle_window_indexes(rows)]
    q_margin_rows = [row for row in rows if row.get("q_margin")]
    feature_ablation_all = [item for row in q_margin_rows for item in row.get("feature_ablation_rows", [])]
    outcome_ablation_all = [item for row in q_margin_rows for item in row.get("outcome_flag_ablation_rows", [])]
    feature_summary = aggregate_ablation(feature_ablation_all)
    outcome_summary = aggregate_ablation(outcome_ablation_all)
    immediate = immediate_reward_comparison(rows)
    confounding = confirmation_confounding(rows)
    td_cycle = _td_summary(rows, continue_cycle=True)
    td_break = _td_summary(rows, continue_cycle=False)
    hypotheses = _hypotheses(q_margin_rows, immediate, confounding, td_cycle, td_break, feature_summary, outcome_summary)
    result = {
        "checkpoint": args.model_path,
        "site": site["site_id"],
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "cycle_trace": cycle_trace,
        "cycle_trace_count": len(cycle_trace),
        "temporal_alignment": {
            "ordering": [
                "state_t",
                "history through t-1",
                "action_t",
                "execute",
                "detector/tracker",
                "reward_t",
                "append transition_t",
                "state_t+1",
            ],
            "unproductive_repeat_timing": "outcome of transition_t; visible in state_t+1 history, not state_t",
            "productive_transition_timing": "outcome of transition_t; visible in state_t+1 history",
            "confirmation_progress_timing": "tracker/reward event on transition_t; visible in state_t+1 history",
            "active_cycle_timing": "computed from history through t-1; visible before action_t",
            "streak_timing": "computed from history through t-1; visible before action_t",
        },
        "predictability": summarize_predictability([row for row in rows if (row.get("repeat_diagnostic") or {}).get("unproductive_repeat") or (row.get("repeat_diagnostic") or {}).get("cycle_repeat")]),
        "q_margin": {
            "rows": q_margin_rows,
            "summary": _aggregate_q_margins(q_margin_rows),
        },
        "feature_ablation": {
            "rows": feature_ablation_all,
            "summary": feature_summary,
            "cycle_bias_feature_rank": rank_cycle_bias_features(feature_summary),
        },
        "outcome_flag_ablation": {
            "rows": outcome_ablation_all,
            "summary": outcome_summary,
            "cycle_bias_feature_rank": rank_cycle_bias_features(outcome_summary),
        },
        "conditional_reward": conditional_reward_table(rows),
        "confirmation_confounding": confounding,
        "td_target": {
            "cycle_continue_rows": [row["td_target"] for row in rows if (row.get("history_state_diagnostic", {}).get("cycle_continuation_identity") and row.get("history_state_diagnostic", {}).get("selected_action_identity") == row.get("history_state_diagnostic", {}).get("cycle_continuation_identity"))],
            "cycle_break_rows": [row["td_target"] for row in rows if (row.get("history_state_diagnostic", {}).get("cycle_continuation_identity") and row.get("history_state_diagnostic", {}).get("selected_action_identity") != row.get("history_state_diagnostic", {}).get("cycle_continuation_identity"))],
            "target_network_note": "EP30 checkpoint is used as both online and target for offline one-step decomposition.",
        },
        "bootstrap": {
            "cycle_continue": td_cycle,
            "cycle_break": td_break,
        },
        "escape_metrics": escape_metric_interpretation(_basic_metrics(rows)),
        "training_association_diagnostic": {
            "available": False,
            "reason": "training manifest stores aggregate replay/episode summaries, not chronological previous-history-conditioned transitions",
        },
        "hypotheses": hypotheses,
        "root_cause": _root_cause(hypotheses),
        "hard_invariants": {
            "reward_changed": False,
            "observation_changed": False,
            "architecture_changed": False,
            "confirmation_changed": False,
            "ppo_changed": False,
            "ground_truth_used": False,
            "optimizer_called": False,
            "checkpoint_written": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _basic_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    total = len(rows)
    repeat_local = sum(
        1
        for row in rows
        if (row.get("repeat_diagnostic") or {}).get("immediate_repeat")
        or (row.get("repeat_diagnostic") or {}).get("recent_window_repeat")
        or (row.get("repeat_diagnostic") or {}).get("cycle_repeat")
        or (row.get("repeat_diagnostic") or {}).get("unproductive_repeat")
    )
    repeat_escape_follow = sum(1 for row in rows if (row.get("history_state_diagnostic") or {}).get("previous_unproductive_repeat"))
    repeat_escape = sum(
        1
        for row in rows
        if (row.get("history_state_diagnostic") or {}).get("previous_unproductive_repeat")
        and (row.get("history_state_diagnostic") or {}).get("selected_action_identity")
        != (row.get("history_state_diagnostic") or {}).get("previous_unproductive_repeat_identity")
    )
    cycle_active = sum(1 for row in rows if int((row.get("history_state_diagnostic") or {}).get("active_cycle_length", 0) or 0) > 0)
    cycle_break = sum(
        1
        for row in rows
        if int((row.get("history_state_diagnostic") or {}).get("active_cycle_length", 0) or 0) > 0
        and (row.get("history_state_diagnostic") or {}).get("selected_action_identity")
        != (row.get("history_state_diagnostic") or {}).get("cycle_continuation_identity")
    )
    return {
        "repeat_local_action_ratio": repeat_local / total if total else 0.0,
        "repeat_escape_rate": repeat_escape / repeat_escape_follow if repeat_escape_follow else 0.0,
        "cycle_repeat_ratio": sum(1 for row in rows if (row.get("repeat_diagnostic") or {}).get("cycle_repeat")) / total if total else 0.0,
        "cycle_escape_rate": cycle_break / cycle_active if cycle_active else 0.0,
    }


def _hypotheses(q_rows, immediate, confounding, td_cycle, td_break, feature_summary, outcome_summary):
    conf_scroll = confounding["cycle_scroll"]
    conf_break = confounding["cycle_break"]
    cycle_reward = immediate["cycle_continue"]["mean"]
    break_reward = immediate["cycle_break"]["mean"]
    cycle_bootstrap = td_cycle["mean_bootstrap"]["mean"]
    break_bootstrap = td_break["mean_bootstrap"]["mean"]
    cycle_td_target = td_cycle["mean_td_target"]["mean"]
    break_td_target = td_break["mean_td_target"]["mean"]
    outcome_unproductive_delta = abs((outcome_summary.get("unproductive_repeat", {}) or {}).get("mean_abs_cycle_q_margin_delta", 0.0) or 0.0)
    cycle_onehot_delta = abs((feature_summary.get("cycle_onehot", {}) or {}).get("mean_abs_cycle_q_margin_delta", 0.0) or 0.0)
    h4_pass = td_cycle["bootstrap_scroll_rate"] >= 0.7 and cycle_bootstrap > break_bootstrap and cycle_td_target >= break_td_target
    h4_partial = td_cycle["bootstrap_scroll_rate"] >= 0.7
    h5_pass = (outcome_unproductive_delta > 0.25 or cycle_onehot_delta > 0.25) and q_rows
    return {
        "H1_temporal_mismatch": {
            "verdict": "REJECT",
            "reason": "active cycle/streak are visible before action_t; only the current transition's new unproductive flag arrives in state_t+1.",
        },
        "H2_confirmation_reward_confounding": {
            "verdict": "REJECT" if conf_scroll["confirmation_rate"] <= conf_break["confirmation_rate"] else "PARTIAL",
            "reason": "cycle-scroll confirmation rate is not higher than break-action confirmation rate in this trajectory.",
        },
        "H3_penalty_balance": {
            "verdict": "REJECT" if cycle_reward <= break_reward else "PARTIAL",
            "reason": "cycle continuation immediate reward is not better than cycle break reward.",
        },
        "H4_bootstrap_self_reinforcement": {
            "verdict": "PASS" if h4_pass else "PARTIAL" if h4_partial else "REJECT",
            "reason": "scroll bootstrap recurrence is checked separately from whether bootstrap makes cycle TD target favorable.",
        },
        "H5_feature_association_inversion": {
            "verdict": "PASS" if h5_pass else "PARTIAL" if q_rows else "REJECT",
            "reason": "feature ablation identifies whether repeat-state features carry large cycle-margin influence.",
        },
    }


def _root_cause(hypotheses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    causes = []
    if hypotheses["H5_feature_association_inversion"]["verdict"] in {"PASS", "PARTIAL"}:
        causes.append("FEATURE_ASSOCIATION_INVERSION")
    if hypotheses["H4_bootstrap_self_reinforcement"]["verdict"] in {"PASS", "PARTIAL"}:
        causes.append("BOOTSTRAP_SELF_REINFORCEMENT")
    if hypotheses["H2_confirmation_reward_confounding"]["verdict"] == "PARTIAL":
        causes.append("CONFIRMATION_REWARD_CONFOUNDING")
    return {
        "classification": causes or ["OTHER"],
        "primary": causes[0] if causes else "OTHER",
        "next_work": "history feature representation/alignment fix",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--site-id", default="site001")
    parser.add_argument("--base-url")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output": str(Path(args.output)),
        "cycle_trace_count": result["cycle_trace_count"],
        "root_cause": result["root_cause"],
        "H2": result["hypotheses"]["H2_confirmation_reward_confounding"]["verdict"],
        "H4": result["hypotheses"]["H4_bootstrap_self_reinforcement"]["verdict"],
        "H5": result["hypotheses"]["H5_feature_association_inversion"]["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
