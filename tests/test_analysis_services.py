import unittest
from collections import Counter
import numpy as np
from services.coverage_service import compare_coverage, graph_from_transitions
from services.exploration_identifiers import action_id, normalized_url, state_id, transition_id
from services.exploration_service import _mask_repeated_actions
from models.action_space import ActionSpace
from services.risk_scoring_service import score_anomaly
from agents.rainbow_dqn_agent import RainbowDQNAgent
from scripts.run_algorithm_ab import choose_default_policy
from services.api_catalog_service import ApiEndpoint, ApiSite
from services.api_request_mutator import build_fuzz_cases
from services.known_bug_matcher import match_anomalies_to_known_bugs
from services.probe_planner import next_probe_action
from services.anomaly_detection_service import detect_anomalies

class AnalysisServicesTest(unittest.TestCase):
    def test_action_space_normalizes_catalog_action_aliases(self):
        action_space = ActionSpace(max_candidates=4)
        self.assertEqual(action_space.encode("click", 2), action_space.encode("click_element", 2))
        self.assertEqual(action_space.encode("fill", 1), action_space.encode("fill_input", 1))
        self.assertEqual(action_space.encode("enter", 0), action_space.encode("press_enter", 0))

    def test_probe_planner_counts_click_alias_as_click_element(self):
        bugs = [{"bug_id": "site777-bug01", "action_hints": ["click", "inspect_cart"]}]
        history = {"matched_bug_ids": set(), "action_type_counts": {"click_element": 1}}
        self.assertEqual(next_probe_action("site777", {}, history, bugs), "inspect_cart")

    def test_direct_bug_identity_beats_broad_catalog_matches(self):
        anomalies = [{
            "type": "button-no-response",
            "confidence": 0.9,
            "evidence": {
                "clicked_data_bug_id": "site001-bug01",
                "catalog_bug_id_matches": ["site001-bug01", "site001-bug02"],
                "cart_count_before": 1,
                "cart_count_after": 1,
            },
        }]
        bugs = [
            {"bug_id": "site001-bug01", "type": "button-no-response"},
            {"bug_id": "site001-bug02", "type": "button-no-response", "target_keywords": ["site001"]},
        ]
        matches = match_anomalies_to_known_bugs(anomalies, bugs, site_id="site001")
        self.assertEqual(matches[0]["matched_bug_id"], "site001-bug01")

    def test_mobile_layout_probe_recovers_when_a11y_bboxes_are_missing(self):
        before = {"page_state": {"viewport_type": "mobile"}, "candidate_elements": []}
        after = {
            "page_state": {"viewport_type": "mobile"},
            "candidate_elements": [],
            "layout_signals": {
                "layout_overlap_count": 0,
                "layout_overflow_count": 0,
                "layout_overflow_candidates": [{
                    "data_bug_id": "site001-bug03",
                    "selector_hint": "[data-bug-id='site001-bug03']",
                    "catalog_bug_id_matches": ["site001-bug03"],
                    "is_layout_target": True,
                }],
            },
        }
        anomalies = detect_anomalies(before, after, {"action": {"action_type": "inspect_layout"}})
        recovered = [item for item in anomalies if item.get("type") == "layout-overlap"]
        self.assertTrue(recovered)
        self.assertTrue(recovered[0]["evidence"]["bbox_observation_incomplete"])

    def test_stable_identifiers_ignore_query_noise(self):
        self.assertEqual(normalized_url("HTTP://LOCALHOST:3000/a/?ts=1&b=2#x"), "http://localhost:3000/a?b=2")
        obs = {"page_state": {"url": "http://localhost/a", "title": "A"}, "candidate_elements": [{"role": "button", "name": "Save"}]}
        sid = state_id(obs)
        aid = action_id(sid, {"action_type": "click_element"}, obs["candidate_elements"][0])
        self.assertEqual(transition_id(sid, aid, sid), transition_id(sid, aid, sid))

    def test_coverage_against_reference(self):
        graph = graph_from_transitions([{"state_id_before": "a", "state_id_after": "b", "action_id": "x", "transition_id": "t", "url_before": "/a", "url_after": "/b"}])
        metrics = compare_coverage(graph, {"states": ["a", "b", "c"], "actions": ["x"], "transitions": ["t"], "routes": ["/a", "/b"]})
        self.assertAlmostEqual(metrics["state_coverage"], 2 / 3, places=4)
        self.assertEqual(metrics["action_coverage"], 1.0)

    def test_missing_reference_is_not_reported_as_full_coverage(self):
        metrics = compare_coverage({"states": ["a"], "actions": [], "transitions": [], "routes": []})
        self.assertFalse(metrics["reference_available"])
        self.assertIsNone(metrics["coverage_score"])

    def test_risk_increases_with_reproduction(self):
        anomaly = {"type": "button-no-response", "confidence": 0.8, "evidence": {"route_changed": False, "selector": "#save"}}
        low = score_anomaly(anomaly, 5, 1)
        high = score_anomaly(anomaly, 5, 5)
        self.assertGreater(high["score"], low["score"])
        self.assertEqual(high["policy_version"], "risk-v3-general-service")

    def test_security_is_excluded_from_general_service_risk(self):
        result = score_anomaly({"type": "xss", "confidence": 0.9, "evidence": {"selector": "#comment"}})
        self.assertIsNone(result["score"])
        self.assertEqual(result["assessment_status"], "SECURITY_EXCLUDED")

    def test_internal_error_without_metrics_is_not_observable(self):
        result = score_anomaly({"type": "wal-disk-stall", "confidence": 0.8, "evidence": {}})
        self.assertIsNone(result["score"])
        self.assertEqual(result["assessment_status"], "NOT_OBSERVABLE")

    def test_risk_score_is_sum_of_five_components(self):
        result = score_anomaly({
            "type": "data-loss",
            "confidence": 0.9,
            "evidence": {"service_unavailable": True, "scope": "service", "recovery": "impossible"},
        }, 3, 3)
        self.assertEqual(result["score"], sum(result["component_scores"].values()))
        self.assertEqual(result["level"], "CRITICAL")

    def test_repeated_action_is_blocked_within_episode(self):
        action_space = ActionSpace(max_candidates=2)
        observation = {
            "page_state": {"url": "http://localhost/a", "title": "A"},
            "candidate_elements": [{"role": "button", "name": "Save", "visible": True}],
        }
        before_id = state_id(observation)
        encoded = action_space.encode("click_element", 0)
        stable = action_id(before_id, action_space.decode(encoded), observation["candidate_elements"][0])
        mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
        mask[encoded] = 1.0

        filtered = _mask_repeated_actions(action_space, observation, before_id, mask, {stable}, Counter())

        self.assertEqual(filtered[encoded], 0.0)
        self.assertEqual(filtered[action_space.encode("finish_episode", 0)], 1.0)

    def test_action_is_allowed_once_for_cross_episode_reproduction(self):
        action_space = ActionSpace(max_candidates=2)
        observation = {
            "page_state": {"url": "http://localhost/a", "title": "A"},
            "candidate_elements": [{"role": "button", "name": "Save", "visible": True}],
        }
        before_id = state_id(observation)
        encoded = action_space.encode("click_element", 0)
        stable = action_id(before_id, action_space.decode(encoded), observation["candidate_elements"][0])
        mask = np.zeros(action_space.get_action_dim(), dtype=np.float32)
        mask[encoded] = 1.0

        once = _mask_repeated_actions(action_space, observation, before_id, mask, set(), Counter({stable: 1}))
        twice = _mask_repeated_actions(action_space, observation, before_id, mask, set(), Counter({stable: 2}))

        self.assertEqual(once[encoded], 1.0)
        self.assertEqual(twice[encoded], 0.0)

    def test_rainbow_dqn_never_selects_masked_action(self):
        agent = RainbowDQNAgent(obs_dim=4, action_dim=3, atoms=11, warmup_steps=2, batch_size=2)
        mask = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        selected = agent.select_action(np.zeros(4, dtype=np.float32), mask, training=False)
        self.assertEqual(selected["action_id"], 1)

    def test_default_policy_prefers_recall_before_reward(self):
        ppo = {"recall": 0.8, "precision": 0.5, "average_reward": 100, "false_positive_count": 1}
        dqn = {"recall": 1.0, "precision": 0.4, "average_reward": 1, "false_positive_count": 2}
        selected, _ = choose_default_policy(ppo, dqn)
        self.assertEqual(selected, "rainbow-dqn")

    def test_owasp_actions_are_allowlisted_and_mutations_require_opt_in(self):
        endpoint = ApiEndpoint(api_id="create", method="POST", path="/items", body_schema={"name": "string"}, test_safe=True)
        denied_site = ApiSite(site_id="test", base_url="http://localhost:9999")
        self.assertEqual(build_fuzz_cases(denied_site, endpoint, allow_mutating=True), [])
        allowed_site = ApiSite(site_id="test", base_url="http://localhost:9999", allow_mutating_requests=True)
        cases = build_fuzz_cases(allowed_site, endpoint, allow_mutating=True)
        self.assertTrue(cases)
        self.assertTrue(all(case.action["safety"]["allowlisted"] for case in cases))
        self.assertNotIn("repeat_request", {case.action["mutation"] for case in cases})

if __name__ == "__main__": unittest.main()
