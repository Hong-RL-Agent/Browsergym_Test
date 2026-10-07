from models.action_space import ActionSpace
from adapters.browsergym_observation_adapter import _make_candidate
from dqn_error_discovery.runners.train_dqn_error_detector import (
    EXCLUDED_ACTIONS,
    anomaly_reward,
    safe_action_mask,
    sanitize_anomalies,
)


def test_safe_mask_excludes_infrastructure_actions():
    space = ActionSpace(max_candidates=2)
    mask = safe_action_mask(space, {"page_state": {}, "candidate_elements": []})
    for action_type in EXCLUDED_ACTIONS:
        assert mask[space.encode(action_type, 0)] == 0
    assert mask.any()


def test_anomaly_reward_uses_candidate_confirmed_status_not_confidence():
    assert anomaly_reward([]) == 0.0
    assert anomaly_reward([{"type": "layout-overlap", "confidence": 0.99, "status": "candidate"}]) == 0.0
    assert anomaly_reward(
        [
            {
                "type": "layout-overlap",
                "rule_id": "R05",
                "status": "confirmed",
                "signature": "r05-a",
                "confidence": 0.5,
            }
        ]
    ) > 0.0


def test_sanitize_anomalies_removes_ground_truth_evidence():
    anomalies = sanitize_anomalies(
        [
            {
                "type": "layout-overlap",
                "confidence": 0.9,
                "matched_bug_id": "site001-bug-1",
                "evidence": {
                    "catalog_bug_id_matches": ["site001-bug-1"],
                    "data_bug_id": "site001-bug-1",
                    "selector_hint": '[data-bug-id="site001-bug-1"]',
                    "text": "checkout",
                },
            }
        ]
    )

    assert anomalies == [{"type": "layout-overlap", "confidence": 0.9, "evidence": {"text": "checkout"}}]


def test_candidate_hides_ground_truth_fields_and_hints():
    candidate = _make_candidate(
        bid="1",
        text="checkout",
        name="checkout",
        role="button",
        tag="button",
        bbox=[0, 0, 100, 40],
        visibility=1.0,
        clickable=True,
        enabled=True,
        source="dom",
        page_text="checkout",
        data_bug_id="site001-bug-1",
        site_profile={
            "bugs": [
                {
                    "bug_id": "site001-bug-1",
                    "selector": '[data-bug-id="site001-bug-1"]',
                    "action_hints": ["click"],
                    "target_keywords": ["checkout"],
                }
            ],
            "target_keywords": ["checkout"],
            "interaction_keywords": [],
            "form_keywords": [],
        },
    )

    assert candidate["data_bug_id"] == ""
    assert candidate["has_data_bug_id"] is False
    assert candidate["selector_hint"] == ""
    assert candidate["catalog_bug_id_matches"] == []
    assert candidate["action_hints"] == []


def test_action_mask_does_not_enable_invisible_catalog_candidates():
    space = ActionSpace(max_candidates=2)
    mask = space.build_action_mask(
        {
            "page_state": {},
            "candidate_elements": [
                {"visible": False, "catalog_bug_id_matches": ["site001-bug-1"]},
            ],
        }
    )

    assert mask[space.encode("click_element", 0)] == 0.0
