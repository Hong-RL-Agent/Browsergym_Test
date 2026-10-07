from __future__ import annotations

import numpy as np

from dqn_error_discovery.runners.run_policy_sanity_check import (
    _cycle_reentry_count,
    _histv2_risk_ablation_state,
    _scroll_cycle_summary,
)


def test_scroll_cycle_summary_tracks_duration_and_reentry():
    actions = [
        "noop",
        "scroll_up",
        "scroll_down",
        "scroll_up",
        "noop",
        "scroll_down",
        "scroll_up",
        "finish_episode",
    ]

    summary = _scroll_cycle_summary(actions, window=4)

    assert summary["alternating_scroll_cycle_count"] == 2
    assert summary["longest_alternating_scroll_cycle_length"] == 3
    assert summary["scroll_cycle_duration_max"] == 3
    assert summary["scroll_cycle_reentry_count"] == 1
    assert summary["scroll_cycle_reentry_rate"] == 0.5


def test_cycle_reentry_requires_new_cycle_within_history_window():
    segments = [
        {"cycle_start_step": 2, "cycle_end_step": 4},
        {"cycle_start_step": 9, "cycle_end_step": 11},
        {"cycle_start_step": 13, "cycle_end_step": 14},
    ]

    assert _cycle_reentry_count(segments, window=4) == 1


def test_histv2_full_risk_ablation_zeroes_only_57_dim_risk_span():
    state = np.ones(976, dtype=np.float32)

    ablated = _histv2_risk_ablation_state(state, "prospective_risk_full")

    assert np.all(ablated[:919] == 1.0)
    assert np.all(ablated[919:976] == 0.0)
    assert np.all(state == 1.0)


def test_histv2_cycle_risk_ablation_zeroes_every_third_flag():
    state = np.ones(976, dtype=np.float32)

    ablated = _histv2_risk_ablation_state(state, "cycle_risk")

    for type_index in range(19):
        base = 919 + type_index * 3
        assert ablated[base] == 1.0
        assert ablated[base + 1] == 1.0
        assert ablated[base + 2] == 0.0


def test_histv2_risk_ablation_does_not_use_ground_truth():
    assert "ground_truth" not in _histv2_risk_ablation_state.__code__.co_varnames

