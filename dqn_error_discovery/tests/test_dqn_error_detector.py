from models.action_space import ActionSpace
from dqn_error_discovery.runners.train_dqn_error_detector import EXCLUDED_ACTIONS, anomaly_reward, safe_action_mask


def test_safe_mask_excludes_infrastructure_actions():
    space = ActionSpace(max_candidates=2)
    mask = safe_action_mask(space, {"page_state": {}, "candidate_elements": []})
    for action_type in EXCLUDED_ACTIONS:
        assert mask[space.encode(action_type, 0)] == 0
    assert mask.any()


def test_anomaly_reward_uses_confidence_only():
    assert anomaly_reward([]) == 0.0
    assert anomaly_reward([{"type": "layout-overlap", "confidence": 0.75}]) == 0.75
