from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.browsergym_api_server import _is_reportable_anomaly
from services.infra_anomaly_detection_service import detect_infra_anomalies
from services.infra_observation_service import collect_database_hook_observation, collect_infra_observation, collect_server_log_observation
from services.scan_backend_service import start_browsergym_scan


class ServerLogDbHookTests(unittest.TestCase):
    def test_server_log_hook_counts_errors_without_exposing_secret_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "app.log"
            log_path.write_text(
                "\n".join(
                    [
                        "INFO started",
                        "ERROR database error SQLSTATE 23505 duplicate key",
                        "Exception in request handler",
                        "password=super-secret-value",
                    ]
                ),
                encoding="utf-8",
            )

            result = collect_server_log_observation([str(log_path)])

            self.assertTrue(result["server_log_hook_enabled"])
            self.assertEqual(1, result["server_log_files_found"])
            self.assertGreater(result["server_log_error_count"], 0)
            self.assertGreater(result["server_log_exception_count"], 0)
            self.assertGreater(result["server_log_db_error_count"], 0)
            self.assertTrue(any("[redacted" in line for line in result["server_log_tail_sample"]))

    def test_database_hook_reads_sqlite_metadata_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "app.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("create table users(id integer primary key, name text)")
                conn.commit()
            finally:
                conn.close()

            result = collect_database_hook_observation([str(db_path)])

            self.assertTrue(result["db_hook_enabled"])
            self.assertTrue(result["db_hook_read_only"])
            self.assertEqual(1, result["db_connection_ok_count"])
            self.assertEqual(1, result["db_integrity_ok_count"])
            self.assertEqual(1, result["db_table_count"])

    def test_database_hook_missing_path_creates_connection_error(self) -> None:
        result = collect_database_hook_observation(["C:/definitely/missing/app.db"])

        self.assertEqual(1, result["db_connection_error_count"])
        self.assertIn("not found", result["db_error_message"])

    def test_collect_infra_observation_includes_log_and_db_hooks_for_local_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "app.log"
            log_path.write_text("ERROR timeout while querying database", encoding="utf-8")
            db_path = Path(tmp) / "app.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("create table orders(id integer)")
                conn.commit()
            finally:
                conn.close()

            result = collect_infra_observation(
                site_id="local",
                base_url="http://127.0.0.1:9",
                timeout_ms=50,
                log_paths=[str(log_path)],
                database_paths=[str(db_path)],
            )

            self.assertTrue(result["server_log_hook_enabled"])
            self.assertTrue(result["db_hook_enabled"])
            self.assertGreater(result["server_log_error_count"], 0)
            self.assertEqual(1, result["db_connection_ok_count"])

    def test_infra_anomaly_detection_reports_server_log_and_db_hook_errors(self) -> None:
        observation = {
            "page_state": {"url": "http://localhost:9220"},
            "infra_signals": {
                "base_url": "http://localhost:9220",
                "port": 9220,
                "port_open": True,
                "health_check_ok": True,
                "server_log_error_count": 2,
                "server_log_db_error_count": 1,
                "db_hook_enabled": True,
                "db_connection_ok_count": 0,
                "db_connection_error_count": 1,
                "db_error_message": "unable to open database file",
            },
        }

        anomalies = detect_infra_anomalies(observation)
        types = {item["type"] for item in anomalies}

        self.assertIn("server-log-error", types)
        self.assertIn("server-log-database-error", types)
        self.assertIn("database-connection-error", types)
        self.assertTrue(all(item.get("classification") == "verified_browser_signal" for item in anomalies if item["type"].startswith(("server-log", "database-"))))

    def test_non_local_infra_skip_does_not_create_port_or_process_anomalies(self) -> None:
        observation = {
            "page_state": {"url": "https://example.com"},
            "infra_signals": {
                "base_url": "https://example.com",
                "port": 443,
                "port_open": False,
                "process_alive": False,
                "server_error_message": "infra observation skipped for non-local host",
                "infra_observation_skipped": True,
                "non_local_host": True,
            },
        }

        anomalies = detect_infra_anomalies(observation)

        self.assertNotIn("server-port-not-open", [item["type"] for item in anomalies])
        self.assertNotIn("server-process-down", [item["type"] for item in anomalies])

    def test_report_filter_accepts_server_log_and_db_hook_anomalies(self) -> None:
        self.assertTrue(_is_reportable_anomaly({"type": "server-log-error", "confidence": 0.82}))
        self.assertTrue(_is_reportable_anomaly({"type": "database-hook-error", "confidence": 0.8}))

    def test_scan_backend_agent_config_writes_hook_paths_to_scan_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = start_browsergym_scan(
                scan_id="hook-test",
                target_url="http://127.0.0.1:9",
                max_steps=1,
                site_timeout_seconds=30,
                python_executable=sys.executable,
                agent_config={
                    "serverLogPaths": [str(Path(tmp) / "app.log")],
                    "databasePaths": [str(Path(tmp) / "app.db")],
                },
            )
            config = json.loads(Path(result.config_path).read_text(encoding="utf-8"))

            site = config["sites"][0]
            self.assertEqual([str(Path(tmp) / "app.log")], site["server_log_paths"])
            self.assertEqual([str(Path(tmp) / "app.db")], site["database_paths"])

    def test_hooks_do_not_use_bug_labels_or_catalog(self) -> None:
        clean = collect_database_hook_observation([])
        labeled = collect_database_hook_observation([])

        self.assertEqual(clean, labeled)
        self.assertNotIn("bug_id", json.dumps(clean, ensure_ascii=False))
        self.assertNotIn("catalog", json.dumps(clean, ensure_ascii=False).lower())


if __name__ == "__main__":
    unittest.main()
