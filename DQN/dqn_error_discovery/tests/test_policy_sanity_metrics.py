from dqn_error_discovery.runners.run_policy_sanity_check import SanityMetricsAggregator


def _candidate(bid="buy", text="Buy", **extra):
    item = {"bid": bid, "text": text, "name": text, "role": "button", "tag": "button", "clickable": True}
    item.update(extra)
    return item


def _obs(url="http://app.test/page", *, text_len=10, candidates=None, **page_extra):
    page = {"url": url, "title": "Page", "dom_node_count": 1, "page_text_length": text_len}
    page.update(page_extra)
    return {"page_state": page, "candidate_elements": candidates or [_candidate()], "runtime_signals": {}}


def _action(action_type="click_element", candidate_index=0, **extra):
    item = {"action_type": action_type, "candidate_index": candidate_index}
    item.update(extra)
    return item


def _anomaly(rule_id="R13", status="candidate", signature="sig-a", **extra):
    item = {"rule_id": rule_id, "status": status, "signature": signature, "evidence": {"rule_id": rule_id}}
    item.update(extra)
    return item


def _record(agg, *, before=None, after=None, action=None, reward_components=None, anomalies=None, step=1):
    agg.record_step(
        step=step,
        before_observation=before or _obs(),
        after_observation=after or _obs(text_len=20),
        action=action or _action(),
        reward=1.0,
        reward_components=reward_components or {},
        anomalies=anomalies or [],
    )


def test_ppo_unique_states():
    agg = SanityMetricsAggregator("ppo")
    _record(agg, after=_obs("/a", text_len=10), step=1)
    _record(agg, after=_obs("/b", text_len=20), step=2)

    assert agg.summary()["metrics"]["unique_states"] == 2


def test_ppo_repeated_state():
    agg = SanityMetricsAggregator("ppo")
    same = _obs("/a", text_len=10)
    _record(agg, after=same, step=1)
    _record(agg, after=same, step=2)

    assert agg.summary()["metrics"]["repeat_state_count"] == 1


def test_ppo_unique_target():
    agg = SanityMetricsAggregator("ppo")
    _record(agg, before=_obs(candidates=[_candidate("a")]), action=_action(), step=1)
    _record(agg, before=_obs(candidates=[_candidate("b")]), action=_action(), step=2)

    assert agg.summary()["metrics"]["unique_targets"] == 2


def test_ppo_action_target_repeat_ratio():
    agg = SanityMetricsAggregator("ppo")
    _record(agg, before=_obs(candidates=[_candidate("a")]), action=_action(), step=1)
    _record(agg, before=_obs(candidates=[_candidate("a")]), action=_action(), step=2)

    assert agg.summary()["metrics"]["repeat_action_target_ratio"] == 0.5


def test_ppo_action_distribution():
    agg = SanityMetricsAggregator("ppo")
    _record(agg, action=_action("click_element"), step=1)
    _record(agg, action=_action("inspect_dom"), step=2)

    assert agg.summary()["action_distribution"] == {"click_element": 1, "inspect_dom": 1}


def test_dqn_local_state_novelty():
    agg = SanityMetricsAggregator("dqn")
    _record(agg, after=_obs(quantity=1), step=1)
    _record(agg, after=_obs(quantity=2), step=2)

    assert agg.summary()["metrics"]["unique_local_states"] == 2


def test_dqn_local_action_target_novelty():
    agg = SanityMetricsAggregator("dqn")
    _record(agg, before=_obs(candidates=[_candidate("plus")]), action=_action(), step=1)
    _record(agg, before=_obs(candidates=[_candidate("minus")]), action=_action(), step=2)

    assert agg.summary()["metrics"]["unique_local_action_targets"] == 2


def test_candidate_signature_dedup():
    agg = SanityMetricsAggregator("ppo")
    _record(agg, anomalies=[_anomaly("R01", "candidate", "same")], step=1)
    _record(agg, anomalies=[_anomaly("R01", "candidate", "same")], step=2)

    assert agg.summary()["metrics"]["unique_candidate_signatures"] == 1


def test_confirmed_signature_dedup():
    agg = SanityMetricsAggregator("ppo")
    _record(agg, anomalies=[_anomaly("R13", "confirmed", "same")], step=1)
    _record(agg, anomalies=[_anomaly("R13", "confirmed", "same")], step=2)

    assert agg.summary()["metrics"]["unique_confirmed_signatures"] == 1


def test_candidate_to_confirmed_count():
    agg = SanityMetricsAggregator("dqn")
    _record(agg, anomalies=[_anomaly("R01", "candidate", "r01")], step=1)
    _record(agg, anomalies=[_anomaly("R01", "confirmed", "r01")], step=2)

    summary = agg.summary()
    assert summary["metrics"]["candidate_to_confirmed_count"] == 1
    assert summary["metrics"]["candidate_confirmation_rate"] == 1.0


def test_strong_confirmed_is_not_candidate_conversion():
    agg = SanityMetricsAggregator("dqn")
    _record(agg, anomalies=[_anomaly("R13", "confirmed", "r13")], step=1)

    assert agg.summary()["metrics"]["candidate_to_confirmed_count"] == 0


def test_repeat_confirmed_anomaly_count():
    agg = SanityMetricsAggregator("dqn")
    _record(agg, anomalies=[_anomaly("R13", "confirmed", "r13")], step=1)
    _record(agg, anomalies=[_anomaly("R13", "confirmed", "r13")], step=2)

    assert agg.summary()["metrics"]["repeat_confirmed_anomaly_count"] == 1


def test_reward_component_aggregation():
    agg = SanityMetricsAggregator("ppo")
    _record(agg, reward_components={"new_state": 0.5, "repeat_penalty": -0.2}, step=1)
    _record(agg, reward_components={"new_state": 0.5}, step=2)

    totals = agg.summary()["reward_component_totals"]
    assert totals["new_state"] == 1.0
    assert totals["repeat_penalty"] == -0.2


def test_top_repeated_actions():
    agg = SanityMetricsAggregator("ppo")
    for step in range(1, 4):
        _record(agg, before=_obs(candidates=[_candidate("a")]), action=_action(), step=step)

    assert agg.summary()["top_repeated_actions"][0]["count"] == 2


def test_ground_truth_metadata_does_not_affect_metrics():
    plain = SanityMetricsAggregator("ppo")
    with_gt = SanityMetricsAggregator("ppo")
    _record(plain, before=_obs(candidates=[_candidate("a")]), step=1)
    _record(with_gt, before=_obs(candidates=[_candidate("a", data_bug_id="BUG", catalog_bug_id_matches=["BUG"])]), step=1)

    assert plain.summary()["metrics"]["unique_targets"] == with_gt.summary()["metrics"]["unique_targets"]


def test_empty_episode_has_no_divide_by_zero():
    summary = SanityMetricsAggregator("ppo").summary()

    assert summary["metrics"]["total_steps"] == 0
    assert summary["metrics"]["state_novelty_ratio"] == 0.0
