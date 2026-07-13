import unittest
from services.coverage_service import compare_coverage, graph_from_transitions
from services.exploration_identifiers import action_id, normalized_url, state_id, transition_id
from services.risk_scoring_service import score_anomaly

class AnalysisServicesTest(unittest.TestCase):
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
        self.assertEqual(high["policy_version"], "risk-v2")

if __name__ == "__main__": unittest.main()
