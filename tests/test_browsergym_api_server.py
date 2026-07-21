from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from runners.browsergym_api_server import (
    ExplorationJob,
    _browsergym_result,
    _filter_report_anomalies,
    _get_job,
    _job_status,
    _looks_like_form_or_login_url,
    _spring_step_events,
)
from services.scan_backend_service import RUNNING_PROCESSES, ScanStartResult, cancel_browsergym_scan, start_browsergym_scan


class BrowserGymApiServerTests(unittest.TestCase):
    def test_step_events_are_spring_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            transition_path = Path(tmp) / "partial_transitions.jsonl"
            transition_path.write_text(
                json.dumps(
                    {
                        "episode": 1,
                        "step": 2,
                        "action": "click_element",
                        "reward": 0.4,
                        "url": "http://localhost:9220/products",
                        "candidate_count": 3,
                        "fillable_count": 1,
                        "clickable_count": 2,
                        "submit_count": 1,
                        "password_input_count": 0,
                        "selected_target_element_key": "button|add",
                        "selected_target_text": "Add",
                        "selected_target_name": "Add",
                        "selected_target_role": "button",
                        "selected_target_type": "submit",
                        "action_success": True,
                        "action_success_reason": "target candidate clicked",
                        "action_mask_enabled_actions": ["click_element", "fill_input"],
                        "detected_anomalies": [
                            {
                                "type": "button-no-response",
                                "confidence": 0.8,
                                "evidence": {"clicked_text": "add to cart"},
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            events = _spring_step_events(job)

        self.assertEqual(1, len(events))
        self.assertEqual("step", events[0]["type"])
        self.assertEqual("click_element", events[0]["action"])
        self.assertEqual(1, events[0]["anomaly_count"])
        self.assertTrue(events[0]["new_state"])
        self.assertEqual(3, events[0]["candidate_count"])
        self.assertEqual(1, events[0]["fillable_count"])
        self.assertEqual("button|add", events[0]["selected_target_element_key"])
        self.assertEqual("target candidate clicked", events[0]["action_success_reason"])
        self.assertIn("fill_input", events[0]["action_mask_enabled_actions"])

    def test_report_does_not_truncate_action_timeline_to_9_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            transition_path = Path(tmp) / "partial_transitions.jsonl"
            rows = [
                json.dumps({"episode": 1, "step": step, "action": "inspect_dom", "candidate_count": step})
                for step in range(1, 13)
            ]
            transition_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            events = _spring_step_events(job)

        self.assertEqual(12, len(events))
        self.assertEqual(12, events[-1]["candidate_count"])

    def test_default_max_steps_at_least_30(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            import services.scan_backend_service as scan_service

            process = Mock()
            process.pid = 4321
            process.communicate.return_value = ("", "")
            process.returncode = 0

            original_root = scan_service.ROOT
            try:
                scan_service.ROOT = Path(tmp_root)
                with patch.object(scan_service, "_preflight_target", return_value=""), patch.object(
                    scan_service.subprocess, "Popen", return_value=process
                ):
                    result = start_browsergym_scan(
                        scan_id="scan-default-steps",
                        target_url="https://example.com/login",
                    )
            finally:
                scan_service.ROOT = original_root
                scan_service.RUNNING_PROCESSES.pop("scan-default-steps", None)

            config = json.loads(Path(result.config_path).read_text(encoding="utf-8"))

        self.assertGreaterEqual(config["max_steps"], 30)

    def test_login_form_min_steps_at_least_30(self) -> None:
        self.assertTrue(_looks_like_form_or_login_url("https://getbootstrap.com/docs/5.3/examples/sign-in/"))

    def test_job_status_completed_from_action_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            (Path(tmp) / "action_logs.jsonl").write_text(
                json.dumps({"event": "scan_completed", "return_code": 0}) + "\n",
                encoding="utf-8",
            )

            status, error = _job_status(job)

        self.assertEqual("completed", status)
        self.assertEqual("", error)

    def test_job_status_running_returns_status_and_error_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)

            status, error = _job_status(job)

        self.assertEqual("running", status)
        self.assertEqual("", error)

    def test_job_status_cancelled_takes_priority_over_failed_process_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            (Path(tmp) / "action_logs.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "scan_cancelled"}),
                        json.dumps({"event": "scan_failed", "message": "process terminated"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            status, error = _job_status(job)

        self.assertEqual("cancelled", status)
        self.assertEqual("", error)

    def test_result_builds_findings_from_partial_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = self._job(tmp)
            (Path(tmp) / "partial_transitions.jsonl").write_text(
                json.dumps(
                    {
                        "episode": 0,
                        "step": 1,
                        "action": "click_element",
                        "reward": 0.2,
                        "url": "http://localhost:9220",
                        "detected_anomalies": [
                            {
                                "type": "functional-no-effect",
                                "confidence": 0.7,
                                "evidence": {"clicked_text": "enroll"},
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = _browsergym_result(job)

        self.assertEqual(1, len(result["findings"]))
        self.assertEqual("functional-no-effect", result["findings"][0]["type"])
        self.assertIn("coverage", result)

    def test_completed_job_can_be_recovered_from_scan_directory(self) -> None:
        # Uses a known repository-local scan directory shape without relying on the
        # in-memory JOBS map. This is what keeps Spring polling alive after bridge restarts.
        with tempfile.TemporaryDirectory() as tmp_root:
            scan_dir = Path(tmp_root) / "artifacts" / "scans" / "job-2"
            scan_dir.mkdir(parents=True)
            (scan_dir / "scan_config.json").write_text(
                json.dumps({"sites": [{"base_url": "http://localhost:9220"}]}),
                encoding="utf-8",
            )

            import runners.browsergym_api_server as api_server

            original_root = api_server.ROOT
            try:
                api_server.ROOT = Path(tmp_root)
                job = _get_job("job-2")
            finally:
                api_server.ROOT = original_root

        self.assertIsNotNone(job)
        self.assertEqual("http://localhost:9220", job.target_url)

    def test_report_filter_suppresses_weak_layout_overlap(self) -> None:
        anomalies = [
            {"type": "layout-overlap", "confidence": 0.4, "evidence": {"layout_overlap_count": 120}},
            {
                "type": "button-no-response",
                "confidence": 0.68,
                "evidence": {"semantic_action_type": "add", "add_no_effect": True},
            },
        ]

        filtered = _filter_report_anomalies(anomalies)

        self.assertEqual(["button-no-response"], [item["type"] for item in filtered])

    def test_scan_launcher_passes_step_timeout_to_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            import services.scan_backend_service as scan_service

            process = Mock()
            process.pid = 4321
            process.communicate.return_value = ("", "")
            process.returncode = 0

            original_root = scan_service.ROOT
            try:
                scan_service.ROOT = Path(tmp_root)
                with patch.object(scan_service, "_preflight_target", return_value=""), patch.object(
                    scan_service.subprocess, "Popen", return_value=process
                ) as popen:
                    result = start_browsergym_scan(
                        scan_id="scan-step-timeout",
                        target_url="http://localhost:9220",
                        max_steps=20,
                        step_timeout_ms=45000,
                    )
            finally:
                scan_service.ROOT = original_root

            command = popen.call_args.args[0]
            config = json.loads(Path(result.config_path).read_text(encoding="utf-8"))

        self.assertIn("--step-timeout-ms", command)
        self.assertEqual("45000", command[command.index("--step-timeout-ms") + 1])
        self.assertEqual(45000, config["step_timeout_ms"])

    def test_cancel_browsergym_scan_terminates_running_process(self) -> None:
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        process.returncode = -15
        RUNNING_PROCESSES["scan-cancel"] = process
        with tempfile.TemporaryDirectory() as tmp:
            action_log = Path(tmp) / "action_logs.jsonl"

            cancelled = cancel_browsergym_scan("scan-cancel", action_log_path=action_log)

            self.assertTrue(cancelled)
            process.terminate.assert_called_once()
            self.assertNotIn("scan-cancel", RUNNING_PROCESSES)
            self.assertIn("subprocess_terminated", action_log.read_text(encoding="utf-8"))

    def test_scan_config_adds_target_url_boundary_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_root:
            import services.scan_backend_service as scan_service

            process = Mock()
            process.pid = 4321
            process.communicate.return_value = ("", "")
            process.returncode = 0

            original_root = scan_service.ROOT
            try:
                scan_service.ROOT = Path(tmp_root)
                with patch.object(scan_service, "_preflight_target", return_value=""), patch.object(
                    scan_service.subprocess, "Popen", return_value=process
                ):
                    result = start_browsergym_scan(
                        scan_id="scan-boundary",
                        target_url="https://www.scrapethissite.com/pages/simple/",
                        max_steps=20,
                    )
            finally:
                scan_service.ROOT = original_root
                scan_service.RUNNING_PROCESSES.pop("scan-boundary", None)

            config = json.loads(Path(result.config_path).read_text(encoding="utf-8"))

        self.assertTrue(config["enforce_target_url_boundary"])
        self.assertEqual(["www.scrapethissite.com"], config["allowed_hosts"])
        self.assertIn("/pages/simple/", config["allowed_path_prefixes"])
        self.assertIn("gumroad.com", config["blocked_url_keywords"])

    def _job(self, scan_dir: str) -> ExplorationJob:
        start = ScanStartResult(
            scan_id="job-1",
            status="running",
            scan_dir=scan_dir,
            config_path=str(Path(scan_dir) / "scan_config.json"),
            errors_log_path=str(Path(scan_dir) / "errors.log"),
            process_pid=123,
        )
        return ExplorationJob(
            job_id="job-1",
            session_id="session-1",
            target_url="http://localhost:9220",
            started_at="2026-07-15T00:00:00+00:00",
            start_result=start,
        )


if __name__ == "__main__":
    unittest.main()
