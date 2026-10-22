"""Deterministic regression for the PPO viewport reward shortcut repair."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))

from models.action_space import ActionSpace
from services.autonomous_reward_service import calculate_autonomous_reward

OUT = ROOT / "DQN" / "artifacts" / "training_v3" / "step7_fresh_training_final"


def history() -> dict:
    return {key: set() for key in (
        "episode_seen_states", "episode_seen_urls", "episode_seen_targets",
        "episode_seen_action_targets", "episode_seen_action_types", "episode_seen_flows",
        "episode_reward_event_ids", "seen_candidate_signatures", "seen_confirmed_signatures",
        "pending_reproduction_signatures", "pending_reproduction_action_targets",
    )}


def obs(url: str, viewport: str, candidates: list | None = None) -> dict:
    return {
        "page_state": {"url": url, "viewport_type": viewport, "dom_node_count": 10},
        "candidate_elements": candidates or [],
        "runtime_signals": {},
    }


def run() -> int:
    viewport = {"action_type": "change_viewport_mobile", "candidate_index": 0, "execution_success": True}
    h = history()
    first, first_components = calculate_autonomous_reward(obs("/home", "desktop"), obs("/home", "mobile"), viewport, [], [], h, {})
    repeat, repeat_components = calculate_autonomous_reward(obs("/home", "mobile"), obs("/home", "mobile"), viewport, [], [], h, {})

    candidate = [{"bid": "button:save", "role": "button", "tag": "button", "visible": True, "enabled": True, "clickable": True, "bbox": {"width": 10, "height": 10}}]
    click = {"action_type": "click_element", "candidate_index": 0, "execution_success": True}
    click_history = history()
    click_first, click_first_components = calculate_autonomous_reward(obs("/home", "desktop", candidate), obs("/detail", "desktop", candidate), click, [], [], click_history, {})
    click_repeat, click_repeat_components = calculate_autonomous_reward(obs("/detail", "desktop", candidate), obs("/detail", "desktop", candidate), click, [], [], click_history, {})

    action_space = ActionSpace(max_candidates=32)
    mobile_mask = action_space.build_action_mask(obs("/home", "mobile", candidate))
    desktop_mask = action_space.build_action_mask(obs("/home", "desktop", candidate))
    mobile_id = action_space.encode("change_viewport_mobile")
    desktop_id = action_space.encode("change_viewport_desktop")
    result = {
        "first_viewport_reward": first,
        "repeated_viewport_reward": repeat,
        "first_viewport_new_state": first_components.get("new_state", 0),
        "repeated_viewport_new_state": repeat_components.get("new_state", 0),
        "first_viewport_new_action_type": first_components.get("new_action_type", 0),
        "repeated_viewport_new_action_type": repeat_components.get("new_action_type", 0),
        "first_click_reward": click_first,
        "repeated_click_reward": click_repeat,
        "first_click_new_action_target": click_first_components.get("new_action_target", 0),
        "repeated_click_new_action_target": click_repeat_components.get("new_action_target", 0),
        "mobile_action_masked_when_already_mobile": bool(mobile_mask[mobile_id] == 0),
        "desktop_action_valid_when_desktop": bool(desktop_mask[desktop_id] == 0),
    }
    result["pass"] = all([
        result["first_viewport_new_state"] == 0,
        result["repeated_viewport_new_state"] == 0,
        result["repeated_viewport_new_action_type"] == 0,
        result["repeated_viewport_reward"] < 0,
        result["first_click_new_action_target"] > 0,
        result["repeated_click_new_action_target"] == 0,
        result["mobile_action_masked_when_already_mobile"],
        result["desktop_action_valid_when_desktop"],
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ppo_reward_shortcut_regression.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(run())
