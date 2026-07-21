from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.action_space import ActionSpace
from services.autonomous_reward_service import calculate_autonomous_reward
from services.extended_policy_oracle_service import detect_extended_policy_findings, make_finding


class ExtendedPolicyOracleTests(unittest.TestCase):
    def test_relational_db_policy_masked_without_hook(self) -> None:
        action_space = ActionSpace(max_candidates=2)
        mask = action_space.build_action_mask(_obs())

        self.assertEqual(0.0, mask[action_space.encode("take_postgres_snapshot_before", 0)])
        self.assertEqual(0.0, mask[action_space.encode("compare_relational_db_state", 0)])

    def test_relational_db_policy_available_with_postgres_hook(self) -> None:
        action_space = ActionSpace(max_candidates=2)
        mask = action_space.build_action_mask(_obs(infra={"db_engine": "postgres", "postgres_hook_available": True, "db_snapshot_available": True}))

        self.assertEqual(1.0, mask[action_space.encode("take_postgres_snapshot_before", 0)])
        self.assertEqual(1.0, mask[action_space.encode("compare_relational_db_state", 0)])
        self.assertEqual(0.0, mask[action_space.encode("take_mysql_snapshot_before", 0)])

    def test_relational_db_policy_available_with_mysql_hook(self) -> None:
        action_space = ActionSpace(max_candidates=2)
        mask = action_space.build_action_mask(_obs(infra={"db_engine": "mysql", "mysql_hook_available": True}))

        self.assertEqual(1.0, mask[action_space.encode("take_mysql_snapshot_before", 0)])
        self.assertEqual(0.0, mask[action_space.encode("take_postgres_snapshot_before", 0)])

    def test_trace_correlation_policy_masked_without_trace_collector(self) -> None:
        action_space = ActionSpace(max_candidates=2)
        mask = action_space.build_action_mask(_obs())

        self.assertEqual(0.0, mask[action_space.encode("capture_trace_id_after_action", 0)])

    def test_api_contract_policy_masked_without_schema(self) -> None:
        action_space = ActionSpace(max_candidates=2)
        mask = action_space.build_action_mask(_obs(runtime={"api_probe_endpoint_available": True}))

        self.assertEqual(0.0, mask[action_space.encode("validate_last_api_schema", 0)])

    def test_login_scenario_policy_masked_without_credentials(self) -> None:
        action_space = ActionSpace(max_candidates=2)
        mask = action_space.build_action_mask(_obs())

        self.assertEqual(0.0, mask[action_space.encode("generate_login_scenario", 0)])

    def test_postgres_hook_generates_db_evidence(self) -> None:
        after = _obs(infra={"db_engine": "postgres", "db_invariant_violation": True})

        findings = detect_extended_policy_findings(_obs(), after, {"action_type": "check_inventory_invariant"})

        self.assertIn("db_invariant_violation", [item["type"] for item in findings])

    def test_mysql_hook_generates_db_evidence(self) -> None:
        after = _obs(infra={"db_engine": "mysql", "partial_commit_detected": True})

        findings = detect_extended_policy_findings(_obs(), after, {"action_type": "check_partial_commit"})

        self.assertIn("transaction_partial_commit", [item["type"] for item in findings])

    def test_trace_correlation_generates_correlation_evidence(self) -> None:
        after = _obs(runtime={"trace_available": True, "trace_id": "trace-1", "server_exception_in_same_trace": True, "correlation_confidence": 0.9})

        findings = detect_extended_policy_findings(_obs(), after, {"action_type": "match_api_response_with_backend_exception"})

        self.assertIn("server_exception_in_same_trace", [item["type"] for item in findings])
        self.assertGreater(findings[0]["correlation_confidence"], 0.0)

    def test_api_contract_generates_schema_violation_evidence(self) -> None:
        after = _obs(
            runtime={
                "schema_available": True,
                "api_probe_status": 200,
                "api_probe_url": "http://local/api/orders",
                "missing_required_fields": ["price"],
                "ui_contract_mismatch": True,
            }
        )

        findings = detect_extended_policy_findings(_obs(), after, {"action_type": "validate_last_api_schema"})

        self.assertIn("api_schema_violation", [item["type"] for item in findings])
        self.assertEqual("verified_browser_signal", findings[0]["classification"])

    def test_login_scenario_generates_auth_evidence(self) -> None:
        after = _obs(runtime={"credential_available": True, "role_based_access_violation": True, "current_role": "user", "protected_resource": "/admin"})

        findings = detect_extended_policy_findings(_obs(), after, {"action_type": "verify_role_based_navigation"})

        self.assertIn("role_based_access_violation", [item["type"] for item in findings])

    def test_reward_does_not_sum_policy_specific_errors(self) -> None:
        anomalies = [
            make_finding(category="api", finding_type="api_failure", severity="high", confidence=1.0, evidence={"endpoint": "/api/orders"}, source_policies=["api_probe_policy"]),
            make_finding(category="backend_db", finding_type="api_failure", severity="high", confidence=1.0, evidence={"endpoint": "/api/orders"}, source_policies=["relational_db_hook_policy"]),
        ]

        _, breakdown = calculate_autonomous_reward(_obs(), _obs(), {"action_type": "click_element"}, anomalies, [], {}, {})

        self.assertLessEqual(breakdown["verified_finding_reward_total"], 8.0)

    def test_reward_given_once_for_unique_verified_finding(self) -> None:
        anomaly = make_finding(category="backend_db", finding_type="order_processing_failure", severity="high", confidence=1.0, evidence={"endpoint": "/api/orders"}, source_policies=["api_probe_policy"])

        _, first = calculate_autonomous_reward(_obs(), _obs(), {"action_type": "click_element"}, [anomaly], [], {}, {})
        _, second = calculate_autonomous_reward(_obs(), _obs(), {"action_type": "click_element"}, [anomaly], [], {"seen_anomaly_keys": {anomaly["signature"]}}, {})

        self.assertEqual(8.0, first["verified_finding_reward_total"])
        self.assertEqual(0.0, second["verified_finding_reward_total"])

    def test_duplicate_finding_gets_no_large_reward(self) -> None:
        anomaly = make_finding(category="api", finding_type="api_failure", severity="high", confidence=1.0, evidence={"endpoint": "/api/orders"}, source_policies=["api_probe_policy"])

        _, breakdown = calculate_autonomous_reward(_obs(), _obs(), {"action_type": "click_element"}, [anomaly], [], {"seen_anomaly_keys": {anomaly["signature"]}}, {})

        self.assertEqual(0.0, breakdown["verified_finding_reward_total"])

    def test_multiple_evidence_sources_do_not_create_multiple_rewards(self) -> None:
        anomaly = make_finding(
            category="backend_db",
            finding_type="order_processing_failure",
            severity="high",
            confidence=1.0,
            evidence={"endpoint": "/api/orders"},
            source_policies=["api_probe_policy", "server_log_probe_policy", "relational_db_hook_policy", "normal_ui_policy"],
        )

        _, breakdown = calculate_autonomous_reward(_obs(), _obs(), {"action_type": "click_element"}, [anomaly], [], {}, {})

        self.assertEqual(8.0, breakdown["verified_finding_reward_total"])
        self.assertLessEqual(breakdown["finding_evidence_reward_total"], 1.0)

    def test_evidence_quality_reward_has_cap(self) -> None:
        anomaly = make_finding(
            category="backend_db",
            finding_type="rich_failure",
            severity="medium",
            confidence=1.0,
            evidence={"endpoint": "/api/a", "status": 500, "method": "POST", "message": "boom", "trace_id": "t", "db_engine": "postgres", "db_invariant": "orders"},
            source_policies=["api_probe_policy", "server_log_probe_policy", "relational_db_hook_policy", "trace_correlation_policy"],
            correlation_confidence=1.0,
        )

        _, breakdown = calculate_autonomous_reward(_obs(), _obs(), {"action_type": "click_element"}, [anomaly], [], {}, {})

        self.assertLessEqual(breakdown["finding_evidence_reward_total"], 1.0)

    def test_reproduction_reward_smaller_than_new_finding_reward(self) -> None:
        anomaly = make_finding(category="api", finding_type="api_failure", severity="high", confidence=1.0, evidence={"endpoint": "/api/orders"}, source_policies=["api_probe_policy"])
        anomaly["reproducible"] = True

        _, breakdown = calculate_autonomous_reward(_obs(), _obs(), {"action_type": "click_element"}, [anomaly], [], {"seen_anomaly_keys": {anomaly["signature"]}}, {})

        self.assertGreater(breakdown["finding_reproduction_reward_total"], 0.0)
        self.assertLess(breakdown["finding_reproduction_reward_total"], 8.0)

    def test_coverage_reward_small_and_non_repeating(self) -> None:
        anomaly = make_finding(category="api", finding_type="new_endpoint_failure", severity="low", confidence=1.0, evidence={"endpoint": "/api/new"}, source_policies=["api_contract_policy"])

        _, breakdown = calculate_autonomous_reward(_obs(), _obs(), {"action_type": "validate_last_api_schema"}, [anomaly], [], {}, {})

        self.assertLessEqual(breakdown["finding_coverage_reward_total"], 0.3)

    def test_schema_candidate_not_large_reward_without_verification(self) -> None:
        anomaly = make_finding(category="api_contract", finding_type="api_schema_violation", severity="medium", confidence=0.7, evidence={"endpoint": "/api/items"}, source_policies=["api_contract_policy"], verified=False)

        _, breakdown = calculate_autonomous_reward(_obs(), _obs(), {"action_type": "validate_last_api_schema"}, [anomaly], [], {}, {})

        self.assertEqual(0.0, breakdown["verified_finding_reward_total"])

    def test_extended_reward_preserves_known_bug_reward_zero(self) -> None:
        anomaly = make_finding(category="auth", finding_type="role_based_access_violation", severity="high", confidence=1.0, evidence={"resource": "/admin"}, source_policies=["login_scenario_policy"])

        _, breakdown = calculate_autonomous_reward(_obs(), _obs(), {"action_type": "verify_role_based_navigation"}, [anomaly], [{"id": "x"}], {}, {"use_known_bug_reward": True})

        self.assertEqual(0.0, breakdown["known_bug_reward_total"])
        self.assertEqual(0.0, breakdown["reward_known_bug_match"])

    def test_extended_policy_does_not_use_bug_id(self) -> None:
        clean = _obs(runtime={"schema_available": True, "missing_required_fields": ["id"], "ui_contract_mismatch": True})
        labeled = _obs(runtime={"schema_available": True, "missing_required_fields": ["id"], "ui_contract_mismatch": True})
        labeled["candidate_elements"][0]["bug_id"] = "site001-bug01"

        self.assertEqual(
            detect_extended_policy_findings(_obs(), clean, {"action_type": "validate_last_api_schema"}),
            detect_extended_policy_findings(_obs(), labeled, {"action_type": "validate_last_api_schema"}),
        )

    def test_extended_policy_does_not_use_data_bug_id_or_catalog_matches(self) -> None:
        clean = _obs(infra={"db_invariant_violation": True})
        labeled = _obs(infra={"db_invariant_violation": True})
        labeled["candidate_elements"][0]["data-bug-id"] = "site001-bug01"
        labeled["candidate_elements"][0]["catalog_bug_id_matches"] = ["site001-bug01"]

        self.assertEqual(
            detect_extended_policy_findings(_obs(), clean, {"action_type": "check_inventory_invariant"}),
            detect_extended_policy_findings(_obs(), labeled, {"action_type": "check_inventory_invariant"}),
        )

    def test_action_signature_deduplicates_extended_findings(self) -> None:
        one = make_finding(category="api", finding_type="api_failure", severity="high", confidence=1.0, evidence={"endpoint": "/api/orders", "action_signature": "click::submit"}, source_policies=["api_probe_policy"])
        two = make_finding(category="api", finding_type="api_failure", severity="high", confidence=1.0, evidence={"endpoint": "/api/orders", "action_signature": "click::submit"}, source_policies=["server_log_probe_policy"])

        self.assertEqual(one["signature"], two["signature"])


def _obs(*, runtime: dict | None = None, infra: dict | None = None) -> dict:
    return {
        "page_state": {"url": "http://local", "page_text_length": 10, "dom_node_count": 3},
        "candidate_elements": [
            {
                "bid": "b1",
                "element_key": "button|submit|main",
                "role": "button",
                "tag": "button",
                "text": "Submit",
                "visible": True,
                "enabled": True,
                "clickable": True,
                "visibility": 1.0,
                "bbox": [0, 0, 100, 40],
            }
        ],
        "runtime_signals": dict(runtime or {}),
        "infra_signals": dict(infra or {}),
        "layout_signals": {},
    }


if __name__ == "__main__":
    unittest.main()
