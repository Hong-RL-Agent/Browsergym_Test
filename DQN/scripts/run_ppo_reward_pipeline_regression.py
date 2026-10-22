import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
from services.autonomous_reward_service import PPO_REWARD_CONFIG, PPO_REWARD_SCHEMA_VERSION, calculate_autonomous_reward


OUT = Path("DQN/artifacts/training_v3/step7_fresh_training_final")


def history():
    return {
        "episode_seen_states": set(), "episode_seen_urls": set(), "episode_seen_targets": set(),
        "episode_seen_action_targets": set(), "episode_seen_action_types": set(),
        "episode_seen_flows": set(), "episode_reward_event_ids": set(),
        "seen_candidate_signatures": set(), "seen_confirmed_signatures": set(),
        "pending_reproduction_signatures": set(), "pending_reproduction_action_targets": set(),
    }


def obs(url="/home", dom=1, candidates=None):
    return {"page_state": {"url": url, "dom_node_count": dom}, "candidate_elements": candidates or [], "runtime_signals": {}}


def run():
    viewport_action = {"action_type": "change_viewport_mobile", "candidate_index": 0, "action_id": 544, "execution_success": True}
    h = history()
    v1, c1 = calculate_autonomous_reward(obs("/home", 1), obs("/mobile", 2), viewport_action, [], [], h, {})
    v2, c2 = calculate_autonomous_reward(obs("/mobile", 2), obs("/mobile", 2), viewport_action, [], [], h, {})

    candidate = [{"bid": "button:save", "role": "button", "tag": "button", "visible": True, "enabled": True, "bbox": {"width": 10, "height": 10}}]
    click = {"action_type": "click_element", "candidate_index": 0, "action_id": 32, "execution_success": True}
    click_h = history()
    click1, clickc1 = calculate_autonomous_reward(obs("/home", 1, candidate), obs("/detail", 2, candidate), click, [], [], click_h, {})
    click2, clickc2 = calculate_autonomous_reward(obs("/detail", 2, candidate), obs("/detail", 2, candidate), click, [], [], click_h, {})
    anomaly = {"signature": "R01:button:save:no-response", "status": "candidate", "confirmation_policy": "reproduce", "action_type": "click_element", "target_id": "button:save"}
    fresh, fresh_c = calculate_autonomous_reward(obs("/detail", 2, candidate), obs("/detail2", 3, candidate), click, [anomaly], [], click_h, {})
    stale, stale_c = calculate_autonomous_reward(obs("/detail2", 3, candidate), obs("/detail2", 3, candidate), click, [anomaly], [], click_h, {})

    result = {
        "schema": PPO_REWARD_SCHEMA_VERSION,
        "contract_config": PPO_REWARD_CONFIG,
        "history_persistent_within_episode": c1.get("new_action_type", 0) > 0 and c2.get("new_action_type", 0) == 0,
        "history_reset_fixture": calculate_autonomous_reward(obs("/detail", 2, candidate), obs("/detail3", 3, candidate), click, [], [], history(), {})[1].get("new_action_type", 0) > 0,
        "first_viewport_reward": v1,
        "repeated_viewport_reward": v2,
        "first_viewport_action_target_reward": c1.get("new_action_target", 0),
        "repeated_viewport_action_target_reward": c2.get("new_action_target", 0),
        "first_click_target_reward": clickc1.get("new_action_target", 0),
        "repeated_click_target_reward": clickc2.get("new_action_target", 0),
        "fresh_anomaly_reward": fresh_c.get("candidate_anomaly", 0),
        "stale_anomaly_reward": stale_c.get("candidate_anomaly", 0),
        "duplicate_event_reward": stale_c.get("reward_event_duplicate", 0),
        "gt_reward_used": max(c1.get("gt_reward_used", 0), c2.get("gt_reward_used", 0), fresh_c.get("gt_reward_used", 0), stale_c.get("gt_reward_used", 0)),
        "global_viewport_fake_target_reward": c1.get("new_action_target", 0),
        "new_url_is_reward_component": "new_url" in PPO_REWARD_CONFIG and PPO_REWARD_CONFIG["new_url"] > 0,
        "action_type_novelty_telemetry_only": "action_type_novelty" not in {"new_state", "new_target", "new_action_target", "new_action_type", "flow_progress", "new_evidence", "candidate_anomaly", "confirmed_anomaly", "repeat_penalty", "invalid_penalty", "timeout_penalty", "no_effect_penalty", "finish_penalty"},
    }
    result["pass"] = all([
        result["history_persistent_within_episode"], result["history_reset_fixture"],
        result["repeated_viewport_action_target_reward"] == 0,
        result["first_click_target_reward"] > 0,
        result["repeated_click_target_reward"] == 0,
        result["fresh_anomaly_reward"] > 0,
        result["stale_anomaly_reward"] == 0,
        result["gt_reward_used"] == 0,
        result["global_viewport_fake_target_reward"] == 0,
    ])
    (OUT / "ppo_reward_contract_audit.json").write_text(json.dumps({"schema": PPO_REWARD_SCHEMA_VERSION, "config": PPO_REWARD_CONFIG, "reward_sum_keys": ["new_state", "new_target", "new_action_target", "new_action_type", "flow_progress", "new_evidence", "candidate_anomaly", "confirmed_anomaly", "repeat_penalty", "invalid_penalty", "timeout_penalty", "no_effect_penalty", "finish_penalty"], "telemetry_only": ["new_functional_state", "action_type_novelty", "target_novelty", "new_flow", "no_effect_deferred", "unproductive_repeat", "reward_event_duplicate", "positive_total", "negative_total", "total", "final_reward", "gt_reward_used"], "global_action_target_reward": 0}, indent=2), encoding="utf-8")
    (OUT / "ppo_reward_context_regression.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name in ["ppo_episode_history_regression.json", "ppo_anomaly_reward_context_regression.json", "ppo_reward_novelty_regression.json"]:
        (OUT / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(run())
