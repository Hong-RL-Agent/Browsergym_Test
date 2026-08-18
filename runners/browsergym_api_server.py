from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.scan_backend_service import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    ScanStartResult,
    cancel_browsergym_scan,
    start_browsergym_scan,
)


@dataclass
class ExplorationJob:
    job_id: str
    session_id: str
    target_url: str
    started_at: str
    start_result: ScanStartResult


JOBS: dict[str, ExplorationJob] = {}
SERVER_OPTIONS: dict[str, Any] = {}


def main() -> int:
    parser = argparse.ArgumentParser(description="HTTP bridge for the Spring backend BrowserGym integration.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument("--max-steps-cap", type=int, default=60)
    parser.add_argument("--site-timeout-seconds", type=int, default=3600)
    parser.add_argument("--episode-timeout-seconds", type=int, default=0)
    parser.add_argument("--reset-timeout-ms", type=int, default=60000)
    parser.add_argument("--step-timeout-ms", type=int, default=45000)
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()

    SERVER_OPTIONS["model_path"] = _resolve_model_path(args.model_path)
    SERVER_OPTIONS["python_executable"] = args.python_executable or sys.executable
    SERVER_OPTIONS["max_episodes"] = max(1, args.max_episodes)
    SERVER_OPTIONS["min_steps"] = max(0, args.min_steps)
    SERVER_OPTIONS["max_steps_cap"] = max(1, args.max_steps_cap)
    SERVER_OPTIONS["site_timeout_seconds"] = max(30, args.site_timeout_seconds)
    SERVER_OPTIONS["episode_timeout_seconds"] = 0 if args.episode_timeout_seconds <= 0 else max(30, args.episode_timeout_seconds)
    SERVER_OPTIONS["reset_timeout_ms"] = max(10000, args.reset_timeout_ms)
    SERVER_OPTIONS["step_timeout_ms"] = max(10000, args.step_timeout_ms)

    server = QuietThreadingHTTPServer((args.host, args.port), BrowserGymApiHandler)
    print(
        json.dumps(
            {
                "event": "browsergym_api_started",
                "base_url": f"http://{args.host}:{args.port}",
                "model_path": SERVER_OPTIONS["model_path"],
                "max_episodes": SERVER_OPTIONS["max_episodes"],
                "min_steps": SERVER_OPTIONS["min_steps"],
                "max_steps_cap": SERVER_OPTIONS["max_steps_cap"],
                "site_timeout_seconds": SERVER_OPTIONS["site_timeout_seconds"],
                "step_timeout_ms": SERVER_OPTIONS["step_timeout_ms"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class BrowserGymApiHandler(BaseHTTPRequestHandler):
    server_version = "JAWBrowserGymAPI/0.1"
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/explorations/start":
                self._handle_start(parsed)
                return
            parts = _path_parts(parsed.path)
            if len(parts) == 3 and parts[0] == "explorations" and parts[2] == "stop":
                self._handle_stop(parts[1])
                return
            self._json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._handle_uncaught_exception(exc)

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._json({"status": "ok", "service": "browsergym-api"})
                return
            parts = _path_parts(parsed.path)
            if len(parts) == 2 and parts[0] == "explorations":
                self._handle_status(parts[1])
                return
            if len(parts) == 3 and parts[0] == "explorations" and parts[2] == "events":
                self._handle_events(parts[1], parsed)
                return
            if len(parts) == 3 and parts[0] == "explorations" and parts[2] == "result":
                self._handle_result(parts[1])
                return
            if len(parts) == 4 and parts[0] == "explorations" and parts[2] == "screenshots":
                self._handle_screenshot(parts[1], parts[3])
                return
            self._json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._handle_uncaught_exception(exc)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[browsergym-api] {self.address_string()} {format % args}", file=sys.stderr, flush=True)

    def _handle_uncaught_exception(self, exc: Exception) -> None:
        trace = traceback.format_exc()
        print(f"[browsergym-api] handler exception: {exc}\n{trace}", file=sys.stderr, flush=True)
        try:
            self._json(
                {
                    "status": 500,
                    "message": str(exc),
                    "exceptionType": exc.__class__.__name__,
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        except Exception:
            # If the socket is already gone, avoid a second noisy exception in the server loop.
            pass

    def _handle_start(self, parsed: Any) -> None:
        query = parse_qs(parsed.query)
        body = self._read_json_body()
        session_id = _first(query, "sessionId") or str(body.get("sessionId") or uuid.uuid4())
        target_url = _first(query, "targetUrl") or str(body.get("targetUrl") or "")
        requested_episodes = _safe_int(_first(query, "episodes") or body.get("episodes"), 1)
        episodes = min(requested_episodes, int(SERVER_OPTIONS.get("max_episodes") or 1))
        agent_config = body.get("agentConfig") if isinstance(body.get("agentConfig"), dict) else {}
        if not target_url:
            self._json({"status": 400, "message": "targetUrl is required."}, status=HTTPStatus.BAD_REQUEST)
            return
        explicit_max_steps = _first(query, "maxSteps") is not None or body.get("maxSteps") is not None
        requested_max_steps = _safe_int(_first(query, "maxSteps") or body.get("maxSteps"), 0)
        min_steps = int(SERVER_OPTIONS.get("min_steps") or 0)
        is_form_or_login = _looks_like_form_or_login_url(target_url)
        is_api_json = _looks_like_api_json_url(target_url)
        max_steps_cap = int(SERVER_OPTIONS.get("max_steps_cap") or 60)
        max_steps = requested_max_steps if explicit_max_steps and requested_max_steps > 0 else max_steps_cap
        max_steps = min(max_steps, max_steps_cap) if max_steps_cap > 0 else max_steps

        job_id = str(uuid.uuid4())
        try:
            start_result = start_browsergym_scan(
                scan_id=job_id,
                target_url=target_url,
                episodes=episodes,
                max_steps=max_steps,
                site_timeout_seconds=int(SERVER_OPTIONS.get("site_timeout_seconds") or 300),
                episode_timeout_seconds=int(SERVER_OPTIONS.get("episode_timeout_seconds") or 0),
                reset_timeout_ms=int(SERVER_OPTIONS.get("reset_timeout_ms") or 60000),
                step_timeout_ms=int(SERVER_OPTIONS.get("step_timeout_ms") or 45000),
                model_path=str(SERVER_OPTIONS.get("model_path") or DEFAULT_MODEL_PATH),
                python_executable=str(SERVER_OPTIONS.get("python_executable") or sys.executable),
                agent_config=agent_config,
            )
            JOBS[job_id] = ExplorationJob(
                job_id=job_id,
                session_id=session_id,
                target_url=target_url,
                started_at=_utc_now(),
                start_result=start_result,
            )
            status = "failed" if start_result.status == "failed" else "queued"
            self._json(
                {
                    "jobId": job_id,
                    "sessionId": session_id,
                    "status": status,
                    "requestedEpisodes": requested_episodes,
                    "episodes": episodes,
                    "requestedMaxSteps": requested_max_steps,
                    "maxSteps": max_steps,
                    "maxStepsCap": int(SERVER_OPTIONS.get("max_steps_cap") or 0),
                    "actionLoopMode": "observation_driven",
                    "actionCountLimitEnabled": True,
                    "minSteps": min_steps,
                    "pageTypeHint": "api_json_page" if is_api_json else "form_page" if is_form_or_login else "unknown",
                    "siteTimeoutSeconds": int(SERVER_OPTIONS.get("site_timeout_seconds") or 300),
                    "stepTimeoutMs": int(SERVER_OPTIONS.get("step_timeout_ms") or 45000),
                },
                status=HTTPStatus.CREATED,
            )
        except Exception as exc:
            self._json({"status": 500, "message": str(exc), "jobId": job_id}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_status(self, job_id: str) -> None:
        job = _get_job(job_id)
        if job is None:
            self._json({"error": "job not found", "jobId": job_id}, status=HTTPStatus.NOT_FOUND)
            return
        status, error = _job_status(job)
        self._json(
            {
                "jobId": job_id,
                "sessionId": job.session_id,
                "status": status,
                "progress": _job_progress(job, status),
                "error": error or "",
                "targetUrl": job.target_url,
            }
        )

    def _handle_events(self, job_id: str, parsed: Any) -> None:
        job = _get_job(job_id)
        if job is None:
            self._json({"events": [], "next": 0, "error": "job not found"}, status=HTTPStatus.NOT_FOUND)
            return
        query = parse_qs(parsed.query)
        cursor = _safe_int(_first(query, "after"), 0)
        events = _spring_step_events(job)
        cursor = max(0, min(cursor, len(events)))
        self._json({"events": events[cursor:], "next": len(events)})

    def _handle_result(self, job_id: str) -> None:
        job = _get_job(job_id)
        if job is None:
            self._json({"error": "job not found", "jobId": job_id}, status=HTTPStatus.NOT_FOUND)
            return
        result = _browsergym_result(job)
        self._json(result)

    def _handle_stop(self, job_id: str) -> None:
        job = _get_job(job_id)
        terminated = False
        if job is not None:
            _append_bridge_event(job, {"event": "scan_cancel_requested", "timestamp": _utc_now()})
            terminated = cancel_browsergym_scan(job_id, action_log_path=_action_log_path(job))
            _append_bridge_event(job, {"event": "scan_cancelled", "timestamp": _utc_now(), "processTerminated": terminated})
        self._json({"jobId": job_id, "status": "cancelled", "processTerminated": terminated})

    def _handle_screenshot(self, job_id: str, file_name: str) -> None:
        job = _get_job(job_id)
        if job is None or "/" in file_name or "\\" in file_name:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        live_dir = Path(job.start_result.scan_dir) / "live"
        candidate = (live_dir / unquote(file_name)).resolve()
        if not str(candidate).startswith(str(live_dir.resolve())) or not candidate.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = candidate.read_bytes()
        content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _read_json_body(self) -> dict[str, Any]:
        length = _safe_int(self.headers.get("Content-Length"), 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            value = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True


def _resolve_model_path(model_path: str) -> str:
    candidates = [
        model_path,
        "artifacts/models/jaws_ppo_v4_blind_gru_memory.pt",
        "artifacts/models/jaws_ppo_v4_blind_no_memory.pt",
        DEFAULT_MODEL_PATH,
    ]
    for candidate in candidates:
        if candidate and (ROOT / candidate).exists():
            return candidate
    return model_path or DEFAULT_MODEL_PATH


def _get_job(job_id: str) -> ExplorationJob | None:
    job = JOBS.get(job_id)
    if job is not None:
        return job
    scan_dir = ROOT / "artifacts" / "scans" / job_id
    if not scan_dir.exists():
        return None
    config = _read_json(scan_dir / "scan_config.json")
    sites = config.get("sites") if isinstance(config.get("sites"), list) else []
    first_site = sites[0] if sites and isinstance(sites[0], dict) else {}
    target_url = str(first_site.get("base_url") or "")
    job = ExplorationJob(
        job_id=job_id,
        session_id=job_id,
        target_url=target_url,
        started_at=str(config.get("created_at") or ""),
        start_result=ScanStartResult(
            scan_id=job_id,
            status="running",
            scan_dir=str(scan_dir),
            config_path=str(scan_dir / "scan_config.json"),
            errors_log_path=str(scan_dir / "errors.log"),
            process_pid=None,
        ),
    )
    JOBS[job_id] = job
    return job


def _job_status(job: ExplorationJob) -> tuple[str, str]:
    events = _read_jsonl(_action_log_path(job))
    cancelled = any(str(event.get("event") or "") == "scan_cancelled" for event in events)
    completed = any(str(event.get("event") or "") == "scan_completed" for event in events)
    if cancelled and not completed:
        return "cancelled", ""
    terminal = ""
    error = ""
    for event in events:
        name = str(event.get("event") or "")
        if name == "scan_failed":
            terminal = "failed"
            error = str(event.get("message") or event.get("detail") or "BrowserGym scan failed")
        elif name == "scan_completed":
            terminal = "completed"
            error = ""
    if terminal:
        if terminal == "completed":
            raw = _read_json(_evaluation_path(job))
            scan_status = str(raw.get("scan_status") or "")
            if scan_status in {"failed_runner_exception", "failed_no_observation", "failed_no_action_executed"}:
                return "failed", str(raw.get("runner_exception") or raw.get("error_message") or raw.get("zero_action_diagnosis") or raw.get("completed_reason") or "BrowserGym scan failed")
            if scan_status in {"completed_invalid", "failed_no_action_executed", "failed_no_observation"}:
                return scan_status, str(raw.get("zero_action_diagnosis") or raw.get("completed_reason") or "")
        return terminal, error
    if job.start_result.status == "failed":
        return "failed", _last_error(job) or "BrowserGym scan failed"
    return "running", ""


def _job_progress(job: ExplorationJob, status: str) -> int:
    if status == "completed":
        return 100
    if status in {"completed_invalid", "failed_no_action_executed", "failed_no_observation"}:
        return 100
    if status == "failed":
        return 100
    if status == "cancelled":
        return 100
    steps = len(_partial_transitions(job))
    return min(95, 5 + steps * 5)


def _browsergym_result(job: ExplorationJob) -> dict[str, Any]:
    evaluation_path = _evaluation_path(job)
    raw = _read_json(evaluation_path)
    transitions = _partial_transitions(job)
    findings = _filter_report_findings(_findings_from_result(raw))
    if not findings:
        findings = _findings_from_transitions(transitions)
    anomaly_explanation = _anomaly_explanation_counts(transitions, findings)
    coverage = raw.get("coverage") if isinstance(raw.get("coverage"), dict) else {}
    if not coverage:
        actions = {str(row.get("action") or "") for row in transitions if row.get("action")}
        urls = {str(row.get("url") or "") for row in transitions if row.get("url")}
        coverage = {
            "coverage_score": min(1.0, (len(actions) + len(urls)) / 10.0),
            "visited_states": len(urls),
            "visited_actions": len(actions),
            "visited_transitions": len(transitions),
        }
    successful = sum(1 for row in transitions if bool(row.get("action_success")))
    action_rows = [row for row in transitions if _is_action_log_row(row)]
    timeline_successful = sum(1 for row in action_rows if _action_execution_success(row))
    timeline_failed = sum(1 for row in action_rows if _action_execution_failed(row))
    missing_failure_reason = sum(1 for row in action_rows if _action_execution_failed(row) and not _failure_reason_from_row(row))
    raw_event_log_rows = _read_jsonl(_action_log_path(job))
    raw_failed = _summary_int(raw, "failed_action_count")
    runner_action_count = (
        _summary_int(raw, "executed_action_count")
        or _summary_int(raw, "runner_action_count")
        or _summary_int(raw, "result_json_action_count")
        or _summary_int(raw, "actual_steps_executed")
    )
    executed_action_count = max(runner_action_count, len(action_rows))
    successful = _summary_int(raw, "successful_action_count") or timeline_successful
    failed = timeline_failed if action_rows else raw_failed
    mismatch = bool(raw_failed and raw_failed != failed)
    network_failure_count = _summary_int(raw, "network_error_count") or _summary_int(raw, "network_request_failed_count")
    console_error_count = _summary_int(raw, "console_error_count")
    runtime_error_count = _summary_int(raw, "runtime_exception_count")
    warning_issue_count = _summary_int(raw, "warning_issue_count")
    critical_issue_count = _summary_int(raw, "critical_issue_count")
    if not warning_issue_count:
        warning_issue_count = sum(1 for item in findings if str((item.get("risk") or {}).get("severity") if isinstance(item.get("risk"), dict) else item.get("severity") or "").lower() in {"warning", "medium", "low"})
    if not critical_issue_count:
        critical_issue_count = sum(1 for item in findings if str((item.get("risk") or {}).get("severity") if isinstance(item.get("risk"), dict) else item.get("severity") or "").lower() in {"critical", "high"})
    total_problem_signal_count = failed + network_failure_count + console_error_count + runtime_error_count + warning_issue_count + critical_issue_count
    completed_reason = _completed_reason(raw)
    if missing_failure_reason > 0 and not completed_reason.startswith("invalid_completed"):
        completed_reason = "invalid_completed_with_unexplained_action_failure"
    valid_scan_run = bool(raw.get("valid_scan_run", False)) and missing_failure_reason == 0 and not mismatch
    scan_counts = {
        "raw_event_log_count": len(raw_event_log_rows) or len(transitions),
        "action_log_count": len(action_rows) or executed_action_count,
        "state_log_count": sum(1 for row in transitions if _log_type(row) == "state" or (row.get("url") and not _is_action_log_row(row))),
        "network_log_count": sum(1 for row in transitions if _log_type(row) == "network"),
        "issue_log_count": len(findings),
        "executed_action_count": executed_action_count,
        "successful_action_count": successful,
        "failed_action_count": failed,
        "runner_failed_action_count": raw_failed,
        "timeline_failed_action_count": timeline_failed,
        "summary_failed_action_count": failed,
        "exploration_failed_action_count": failed,
        "failed_action_count_mismatch": mismatch,
        "failed_action_count_mismatch_reason": "runner_summary_differs_from_action_timeline" if mismatch else "",
        "valid_report_metrics": not mismatch,
        "report_metric_warning": "failed_action_count_mismatch" if mismatch else "",
        "missing_failure_reason_count": missing_failure_reason,
        "network_failure_count": network_failure_count,
        "console_error_count": console_error_count,
        "runtime_error_count": runtime_error_count,
        "warning_issue_count": warning_issue_count,
        "critical_issue_count": critical_issue_count,
        "total_problem_signal_count": total_problem_signal_count,
        "network_failure_detected": network_failure_count > 0 or bool(raw.get("network_failure_detected", False)),
        "network_verification_required": network_failure_count > 0 or bool(raw.get("network_verification_required", False)),
        "network_verification_performed": bool(raw.get("network_verification_performed", False)),
        "network_verification_success": bool(raw.get("network_verification_success", False)),
        "console_error_detected": console_error_count > 0 or bool(raw.get("console_error_detected", False)),
        "console_verification_required": console_error_count > 0 or bool(raw.get("console_verification_required", False)),
        "console_verification_performed": bool(raw.get("console_verification_performed", False)),
        "console_verification_success": bool(raw.get("console_verification_success", False)),
        "raw_anomaly_candidate_count": _summary_int(raw, "raw_anomaly_candidate_count") or anomaly_explanation["raw_anomaly_candidate_count"],
        "verified_finding_count": _summary_int(raw, "verified_finding_count") or len(findings),
        "filtered_false_positive_count": _summary_int(raw, "filtered_false_positive_count") or anomaly_explanation["filtered_false_positive_count"],
        "duplicate_anomaly_count": _summary_int(raw, "duplicate_anomaly_count") or anomaly_explanation["duplicate_anomaly_count"],
        "anomaly_filter_reason_counts": raw.get("anomaly_filter_reason_counts") if isinstance(raw.get("anomaly_filter_reason_counts"), dict) else anomaly_explanation["anomaly_filter_reason_counts"],
        "finding_promotion_reason": str(raw.get("finding_promotion_reason") or anomaly_explanation["finding_promotion_reason"]),
        "finding_rejection_reason": str(raw.get("finding_rejection_reason") or anomaly_explanation["finding_rejection_reason"]),
        "action_diversity_score": _summary_float(raw, "action_diversity_score"),
        "unique_action_type_count": _summary_int(raw, "unique_action_type_count"),
        "repeated_action_type_count": _summary_int(raw, "repeated_action_type_count"),
        "repeated_strategy_count": _summary_int(raw, "repeated_strategy_count"),
        "consecutive_same_strategy_count": _summary_int(raw, "consecutive_same_strategy_count"),
        "skipped_action_count": sum(1 for row in transitions if row.get("skipped")),
        "invalid_action_count": sum(1 for row in transitions if row.get("invalid_action_reason")),
        "early_stop_triggered": bool(raw.get("early_stop_count") or raw.get("partial")),
        "early_stop_reason": _first_early_stop_reason(raw),
        "max_steps_configured": raw.get("max_steps"),
        "min_steps_configured": raw.get("min_steps"),
        "actual_steps_executed": len(transitions),
        "completed_reason": completed_reason,
        "scan_status": str(raw.get("scan_status") or ""),
        "runner_started": bool(raw.get("runner_started", False)),
        "page_loaded": bool(raw.get("page_loaded", False)),
        "initial_observation_collected": bool(raw.get("initial_observation_collected", False)),
        "action_loop_entered": bool(raw.get("action_loop_entered", False)),
        "action_loop_iteration_count": _summary_int(raw, "action_loop_iteration_count"),
        "first_observation_candidate_count": _summary_int(raw, "first_observation_candidate_count"),
        "initial_candidate_count": _summary_int(raw, "initial_candidate_count"),
        "initial_clickable_count": _summary_int(raw, "initial_clickable_count"),
        "initial_fillable_count": _summary_int(raw, "initial_fillable_count"),
        "initial_link_count": _summary_int(raw, "initial_link_count"),
        "initial_button_count": _summary_int(raw, "initial_button_count"),
        "initial_form_count": _summary_int(raw, "initial_form_count"),
        "initial_network_request_count": _summary_int(raw, "initial_network_request_count"),
        "enabled_opportunity_count": _summary_int(raw, "enabled_opportunity_count"),
        "masked_opportunity_count": _summary_int(raw, "masked_opportunity_count"),
        "finish_only_mask": bool(raw.get("finish_only_mask", False)),
        "finish_allowed_at_step_0": bool(raw.get("finish_allowed_at_step_0", False)),
        "finish_blocked_reason_at_step_0": str(raw.get("finish_blocked_reason_at_step_0") or ""),
        "opportunity_generation_error": str(raw.get("opportunity_generation_error") or ""),
        "action_mask_error": str(raw.get("action_mask_error") or ""),
        "runner_exception": str(raw.get("runner_exception") or ""),
        "runner_exception_traceback": str(raw.get("runner_exception_traceback") or ""),
        "zero_action_diagnosis": str(raw.get("zero_action_diagnosis") or ""),
        "runner_action_count": _summary_int(raw, "runner_action_count"),
        "episode_action_count": _summary_int(raw, "episode_action_count"),
        "result_json_action_count": _summary_int(raw, "result_json_action_count"),
        "report_json_action_count": len(transitions),
        "frontend_rendered_action_count": 0,
        "pdf_rendered_action_count": 0,
        "log_pipeline_mismatch": bool(_summary_int(raw, "result_json_action_count") and _summary_int(raw, "result_json_action_count") != len(transitions)),
        "log_pipeline_mismatch_stage": "result_json_to_report_json" if bool(_summary_int(raw, "result_json_action_count") and _summary_int(raw, "result_json_action_count") != len(transitions)) else "",
        "detected_candidate_count": _summary_int(raw, "detected_candidate_count"),
        "generated_opportunity_count": _summary_int(raw, "generated_opportunity_count"),
        "required_opportunity_count": _summary_int(raw, "required_opportunity_count"),
        "optional_opportunity_count": _summary_int(raw, "optional_opportunity_count"),
        "executed_opportunity_count": _summary_int(raw, "executed_opportunity_count"),
        "verified_opportunity_count": _summary_int(raw, "verified_opportunity_count"),
        "skipped_opportunity_count": _summary_int(raw, "skipped_opportunity_count"),
        "failed_opportunity_count": _summary_int(raw, "failed_opportunity_count"),
        "remaining_opportunity_count": _summary_int(raw, "remaining_opportunity_count"),
        "pending_opportunity_count": _summary_int(raw, "pending_opportunity_count") or _summary_int(raw, "remaining_opportunity_count"),
        "remaining_required_opportunity_count": _summary_int(raw, "remaining_required_opportunity_count"),
        "pending_required_opportunity_count": _summary_int(raw, "pending_required_opportunity_count") or _summary_int(raw, "remaining_required_opportunity_count"),
        "action_opportunity_coverage_rate": _summary_float(raw, "action_opportunity_coverage_rate"),
        "action_coverage_rate": _summary_float(raw, "action_coverage_rate") or _summary_float(raw, "action_opportunity_coverage_rate"),
        "required_opportunity_completion_rate": _summary_float(raw, "required_opportunity_completion_rate"),
        "optional_opportunity_coverage_rate": _summary_float(raw, "optional_opportunity_coverage_rate"),
        "unverified_anomaly_count": _summary_int(raw, "unverified_anomaly_count"),
        "verified_finding_count": _summary_int(raw, "verified_finding_count") or len(findings),
        "anomaly_verification_required": bool(raw.get("anomaly_verification_required", False)),
        "anomaly_verification_completed": bool(raw.get("anomaly_verification_completed", False)),
        "finish_allowed": bool(raw.get("finish_allowed", False)),
        "finish_blocked_reason": str(raw.get("finish_blocked_reason") or ""),
        "opportunity_diagnostics": raw.get("opportunity_diagnostics") if isinstance(raw.get("opportunity_diagnostics"), list) else [],
        "valid_scan_run": valid_scan_run,
        "action_budget_status": str(raw.get("action_budget_status") or ""),
        "page_type": str(raw.get("page_type") or ""),
        "has_login_form": bool(raw.get("has_login_form", False)),
        "has_username_or_email_input": bool(raw.get("has_username_or_email_input", False)),
        "has_password_input": bool(raw.get("has_password_input", False)),
        "has_login_submit": bool(raw.get("has_login_submit", False)),
        "username_or_email_filled": bool(raw.get("username_or_email_filled", False)),
        "password_filled": bool(raw.get("password_filled", False)),
        "login_submit_clicked": bool(raw.get("login_submit_clicked", False)),
        "login_result_checked": bool(raw.get("login_result_checked", False)),
        "login_flow_attempted": bool(raw.get("login_flow_attempted", False)),
        "login_flow_completed": bool(raw.get("login_flow_completed", False)),
        "login_flow_status": str(raw.get("login_flow_status") or ""),
        "repeated_inspect_dom_count": _summary_int(raw, "repeated_inspect_dom_count"),
        "repeated_inspect_dom_penalty_applied": _summary_int(raw, "repeated_inspect_dom_penalty_applied"),
        "repeated_inspect_dom_blocked": _summary_int(raw, "repeated_inspect_dom_blocked"),
        "opportunity_debug": raw.get("opportunity_debug") if isinstance(raw.get("opportunity_debug"), dict) else {},
        "dom_exhaustive_action_mode": bool(raw.get("dom_exhaustive_action_mode", True)),
    }
    opportunity_summary = {
        key: scan_counts[key]
        for key in (
            "generated_opportunity_count",
            "required_opportunity_count",
            "optional_opportunity_count",
            "executed_opportunity_count",
            "verified_opportunity_count",
            "pending_opportunity_count",
            "remaining_required_opportunity_count",
            "pending_required_opportunity_count",
            "action_opportunity_coverage_rate",
            "action_coverage_rate",
            "required_opportunity_completion_rate",
            "unverified_anomaly_count",
            "verified_finding_count",
            "raw_anomaly_candidate_count",
            "filtered_false_positive_count",
            "duplicate_anomaly_count",
            "anomaly_verification_required",
            "anomaly_verification_completed",
            "finish_allowed",
            "finish_blocked_reason",
            "opportunity_diagnostics",
            "completed_reason",
            "valid_scan_run",
            "action_budget_status",
            "valid_report_metrics",
            "report_metric_warning",
            "failed_action_count_mismatch",
            "failed_action_count_mismatch_reason",
            "runner_failed_action_count",
            "timeline_failed_action_count",
            "summary_failed_action_count",
            "exploration_failed_action_count",
            "action_log_count",
            "state_log_count",
            "network_log_count",
            "issue_log_count",
            "total_problem_signal_count",
            "network_failure_count",
            "console_error_count",
            "runtime_error_count",
            "warning_issue_count",
            "critical_issue_count",
            "page_type",
            "console_error_detected",
            "console_verification_required",
            "console_verification_performed",
            "console_verification_success",
            "has_login_form",
            "has_username_or_email_input",
            "has_password_input",
            "has_login_submit",
            "username_or_email_filled",
            "password_filled",
            "login_submit_clicked",
            "login_result_checked",
            "login_flow_attempted",
            "login_flow_completed",
            "login_flow_status",
            "dom_exhaustive_action_mode",
        )
    }
    return {
        "jobId": job.job_id,
        "sessionId": job.session_id,
        "status": _job_status(job)[0],
        "coverage": coverage,
        "scanCounts": scan_counts,
        "scanCountsCamel": _camelize_dict(scan_counts),
        "opportunitySummary": opportunity_summary,
        "opportunitySummaryCamel": _camelize_dict(opportunity_summary),
        "opportunityDiagnostics": scan_counts["opportunity_diagnostics"],
        "scanDecision": {
            "completedReason": scan_counts["completed_reason"],
            "validScanRun": scan_counts["valid_scan_run"],
            "finishAllowed": scan_counts["finish_allowed"],
            "finishBlockedReason": scan_counts["finish_blocked_reason"],
            "actionBudgetStatus": scan_counts["action_budget_status"],
        },
        "findings": findings,
        "rawSummary": raw,
    }


def _spring_step_events(job: ExplorationJob) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    reported_anomaly_fingerprints: set[str] = set()
    reported_anomaly_count = 0
    # Chained per-episode state signature: this step's "before" state is exactly
    # the previous step's "after" state. Signatures are derived purely from
    # observed content (url + candidate/clickable/fillable/submit counts) -- no
    # tick/step index/timestamp is baked in, so a real revisit or self-loop
    # (same content before and after the action) actually hashes equal, unlike
    # falling back to a step-unique placeholder id.
    previous_state_id_by_episode: dict[int, str] = {}
    previous_url_by_episode: dict[int, str] = {}
    for row in _partial_transitions(job):
        anomalies = row.get("detected_anomalies") or []
        if not isinstance(anomalies, list):
            anomalies = []
        report_anomalies: list[dict[str, Any]] = []
        for anomaly in _filter_report_anomalies(anomalies):
            fingerprint = _fingerprint(anomaly)
            if fingerprint in reported_anomaly_fingerprints:
                continue
            if reported_anomaly_count >= _REPORT_FINDING_LIMIT:
                break
            reported_anomaly_fingerprints.add(fingerprint)
            reported_anomaly_count += 1
            report_anomalies.append(anomaly)
        action = str(row.get("action") or "")
        success = bool(row.get("action_success", True))
        episode_num = _safe_int(row.get("episode"), 0)
        row_url = str(row.get("url") or "")
        state_id_after = _semantic_state_signature(row)
        state_id_before = previous_state_id_by_episode.get(episode_num, state_id_after)
        url_before = previous_url_by_episode.get(episode_num, row_url)
        previous_state_id_by_episode[episode_num] = state_id_after
        previous_url_by_episode[episode_num] = row_url
        events.append(
            {
                "type": "step",
                "log_type": _log_type(row),
                "episode": episode_num,
                "step": _safe_int(row.get("step"), 0),
                "step_index": _safe_int(row.get("step_index", row.get("step")), 0),
                "action": action,
                "selected_action_type": str(row.get("selected_action_type") or row.get("selected_action") or action),
                "action_id": f"{action}:{row.get('episode', 0)}:{row.get('step', 0)}",
                "url_before": url_before,
                "url_after": row_url,
                "state_id_before": state_id_before,
                "state_id_after": state_id_after,
                "url": str(row.get("url") or ""),
                "success": success if _is_action_log_row(row) else None,
                "new_state": bool(row.get("reward", 0) or report_anomalies),
                "candidate_count": _safe_int(row.get("candidate_count"), 0),
                "fillable_count": _safe_int(row.get("fillable_count"), 0),
                "clickable_count": _safe_int(row.get("clickable_count"), 0),
                "submit_count": _safe_int(row.get("submit_count"), 0),
                "password_input_count": _safe_int(row.get("password_input_count"), 0),
                "selected_target_element_key": str(row.get("selected_target_element_key") or ""),
                "selected_target_text": str(row.get("selected_target_text") or ""),
                "selected_target_name": str(row.get("selected_target_name") or row.get("selected_target_text") or ""),
                "selected_target_role": str(row.get("selected_target_role") or ""),
                "selected_target_tag": str(row.get("selected_target_tag") or ""),
                "selected_target_type": str(row.get("selected_target_type") or ""),
                "selected_target_semantic_type": str(row.get("selected_target_semantic_type") or ""),
                "action_mask_enabled_actions": row.get("action_mask_enabled_actions") if isinstance(row.get("action_mask_enabled_actions"), list) else [],
                "high_priority_candidate_count": _safe_int(row.get("high_priority_candidate_count"), 0),
                "action_priority_reason": str(row.get("action_priority_reason") or ""),
                "action_success_reason": str(row.get("action_success_reason") or row.get("success_reason") or ""),
                "failure_reason": str(row.get("failure_reason") or ""),
                "invalid_action_reason": str(row.get("invalid_action_reason") or ""),
                "failed_action_type": str(row.get("failed_action_type") or ""),
                "exception_type": str(row.get("exception_type") or ""),
                "exception_message": str(row.get("exception_message") or ""),
                "playwright_error": str(row.get("playwright_error") or ""),
                "network_log_available": bool(row.get("network_log_available", False)),
                "network_capture_enabled": bool(row.get("network_capture_enabled", False)),
                "network_events_count": _safe_int(row.get("network_events_count"), 0),
                "infra_failure_possible": bool(row.get("infra_failure_possible", False)),
                "user_visible_failure_possible": bool(row.get("user_visible_failure_possible", False)),
                "target_element_key": str(row.get("target_element_key") or row.get("selected_target_element_key") or ""),
                "target_selector": str(row.get("target_selector") or ""),
                "target_text": str(row.get("target_text") or row.get("selected_target_text") or ""),
                "target_role": str(row.get("target_role") or row.get("selected_target_role") or ""),
                "target_tag": str(row.get("target_tag") or row.get("selected_target_tag") or ""),
                "target_enabled": bool(row.get("target_enabled", True)),
                "target_visible": bool(row.get("target_visible", True)),
                "target_attached": bool(row.get("target_attached", True)),
                "target_clickable": bool(row.get("target_clickable", False)),
                "retryable": bool(row.get("retryable", False)),
                "alternative_verification_action": str(row.get("alternative_verification_action") or row.get("next_recommended_verification_action") or ""),
                "fallback_applied": bool(row.get("fallback_applied", False)),
                "fallback_reason": str(row.get("fallback_reason") or ""),
                "login_flow_status": str(row.get("login_flow_status") or ""),
                "login_flow_status_after_step": str(row.get("login_flow_status_after_step") or row.get("login_flow_status") or ""),
                "has_login_form": bool(row.get("has_login_form", False)),
                "has_username_or_email_input": bool(row.get("has_username_or_email_input", False)),
                "has_password_input": bool(row.get("has_password_input", False)),
                "has_login_submit": bool(row.get("has_login_submit", False)),
                "username_or_email_filled": bool(row.get("username_or_email_filled", False)),
                "password_filled": bool(row.get("password_filled", False)),
                "login_submit_clicked": bool(row.get("login_submit_clicked", False)),
                "login_result_checked": bool(row.get("login_result_checked", False)),
                "login_flow_attempted": bool(row.get("login_flow_attempted", False)),
                "login_flow_completed": bool(row.get("login_flow_completed", False)),
                "required_actions_remaining": row.get("required_actions_remaining") if isinstance(row.get("required_actions_remaining"), list) else [],
                "page_type": str(row.get("page_type") or ""),
                "required_verifications_remaining": row.get("required_verifications_remaining") if isinstance(row.get("required_verifications_remaining"), list) else [],
                "required_verifications_completed": bool(row.get("required_verifications_completed", False)),
                "api_response_checked": bool(row.get("api_response_checked", False)),
                "json_parse_checked": bool(row.get("json_parse_checked", False)),
                "schema_contract_checked": bool(row.get("schema_contract_checked", False)),
                "network_checked": bool(row.get("network_checked", False)),
                "console_checked": bool(row.get("console_checked", False)),
                "detected_candidate_count": _safe_int(row.get("detected_candidate_count"), 0),
                "generated_opportunity_count": _safe_int(row.get("generated_opportunity_count"), 0),
                "required_opportunity_count": _safe_int(row.get("required_opportunity_count"), 0),
                "optional_opportunity_count": _safe_int(row.get("optional_opportunity_count"), 0),
                "executed_opportunity_count": _safe_int(row.get("executed_opportunity_count"), 0),
                "verified_opportunity_count": _safe_int(row.get("verified_opportunity_count"), 0),
                "skipped_opportunity_count": _safe_int(row.get("skipped_opportunity_count"), 0),
                "failed_opportunity_count": _safe_int(row.get("failed_opportunity_count"), 0),
                "remaining_opportunity_count": _safe_int(row.get("remaining_opportunity_count"), 0),
                "remaining_required_opportunity_count": _safe_int(row.get("remaining_required_opportunity_count"), 0),
                "required_opportunity_remaining": _safe_int(row.get("required_opportunity_remaining"), 0),
                "action_opportunity_coverage_rate": float(row.get("action_opportunity_coverage_rate") or 0.0),
                "action_coverage_rate_after_step": float(row.get("action_coverage_rate_after_step", row.get("action_opportunity_coverage_rate")) or 0.0),
                "required_opportunity_completion_rate": float(row.get("required_opportunity_completion_rate") or 0.0),
                "required_completion_rate_after_step": float(row.get("required_completion_rate_after_step", row.get("required_opportunity_completion_rate")) or 0.0),
                "optional_opportunity_coverage_rate": float(row.get("optional_opportunity_coverage_rate") or 0.0),
                "unverified_anomaly_count": _safe_int(row.get("unverified_anomaly_count"), 0),
                "verified_finding_count": _safe_int(row.get("verified_finding_count"), 0),
                "anomaly_verification_required": bool(row.get("anomaly_verification_required", False)),
                "anomaly_verification_completed": bool(row.get("anomaly_verification_completed", False)),
                "finish_allowed": bool(row.get("finish_allowed", False)),
                "finish_allowed_after_step": bool(row.get("finish_allowed_after_step", row.get("finish_allowed", False))),
                "finish_blocked_reason": str(row.get("finish_blocked_reason") or ""),
                "finish_blocked_reason_after_step": str(row.get("finish_blocked_reason_after_step") or row.get("finish_blocked_reason") or ""),
                "opportunity_diagnostics": row.get("opportunity_diagnostics") if isinstance(row.get("opportunity_diagnostics"), list) else [],
                "selected_opportunity_id": str(row.get("selected_opportunity_id") or ""),
                "selected_opportunity_type": str(row.get("selected_opportunity_type") or ""),
                "opportunity_required": bool(row.get("opportunity_required", False)),
                "opportunity_status_before": str(row.get("opportunity_status_before") or ""),
                "opportunity_status_after_action": str(row.get("opportunity_status_after_action") or ""),
                "opportunity_status_after": str(row.get("opportunity_status_after") or row.get("opportunity_status_after_action") or ""),
                "next_recommended_verification_action": str(row.get("next_recommended_verification_action") or ""),
                "next_recommended_action": str(row.get("next_recommended_action") or row.get("next_recommended_verification_action") or ""),
                "repeated_inspect_dom_count": _safe_int(row.get("repeated_inspect_dom_count"), 0),
                "repeated_inspect_dom_penalty_applied": _safe_int(row.get("repeated_inspect_dom_penalty_applied"), 0),
                "repeated_action_penalty_applied": _safe_int(row.get("repeated_inspect_dom_penalty_applied"), 0),
                "repeated_inspect_dom_blocked": _safe_int(row.get("repeated_inspect_dom_blocked"), 0),
                "attempted_viewport_width": row.get("attempted_viewport_width"),
                "attempted_viewport_height": row.get("attempted_viewport_height"),
                "current_viewport_width": row.get("current_viewport_width"),
                "current_viewport_height": row.get("current_viewport_height"),
                "viewport_change_supported": bool(row.get("viewport_change_supported", False)),
                "anomaly_count": len(report_anomalies),
                "anomalies": report_anomalies,
            }
        )
    if not events:
        for item in _read_jsonl(_action_log_path(job)):
            if item.get("event") != "action":
                continue
            action = str(item.get("action") or "")
            events.append(
                {
                    "type": "step",
                    "episode": _safe_int(item.get("episode"), 0),
                    "step": _safe_int(item.get("step"), 0),
                    "action": action,
                    "action_id": f"{action}:{item.get('episode', 0)}:{item.get('step', 0)}",
                    "url": "",
                    "success": True,
                    "new_state": False,
                    "candidate_count": 0,
                    "anomaly_count": 0,
                    "anomalies": [],
                }
            )
    return events


def _findings_from_result(raw: dict[str, Any]) -> list[dict[str, Any]]:
    findings = raw.get("findings") if isinstance(raw, dict) else None
    return findings if isinstance(findings, list) else []


def _filter_report_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        anomaly = {
            "type": finding.get("type"),
            "confidence": (finding.get("risk") or {}).get("confidence") if isinstance(finding.get("risk"), dict) else finding.get("confidence"),
            "evidence": finding.get("evidence") or {},
        }
        if not _is_reportable_anomaly(anomaly):
            continue
        fingerprint = str(finding.get("fingerprint") or _fingerprint(anomaly))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        filtered.append(finding)
        if len(filtered) >= _REPORT_FINDING_LIMIT:
            break
    return filtered


def _findings_from_transitions(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Count how many times each anomaly signature recurs across the whole scan
    # *before* filtering. A single low-confidence detection is a guess, but the
    # same signature reappearing on independent steps is reproduction -- the
    # exact evidence the confidence threshold is a proxy for -- so it should be
    # reportable even if no single occurrence alone crossed the bar.
    occurrence_counts: Counter[str] = Counter()
    for row in transitions:
        anomalies = row.get("detected_anomalies") or []
        if not isinstance(anomalies, list):
            continue
        for anomaly in anomalies:
            if isinstance(anomaly, dict):
                occurrence_counts[_fingerprint(anomaly)] += 1

    seen: set[str] = set()
    findings: list[dict[str, Any]] = []
    for row in transitions:
        anomalies = row.get("detected_anomalies") or []
        if not isinstance(anomalies, list):
            continue
        for anomaly in anomalies:
            if not isinstance(anomaly, dict):
                continue
            fingerprint = _fingerprint(anomaly)
            reproduced = occurrence_counts.get(fingerprint, 0) >= 2
            if not _is_reportable_anomaly(anomaly) and not (reproduced and float(anomaly.get("confidence", 0.0) or 0.0) >= 0.5):
                continue
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            confidence = float(anomaly.get("confidence", 0.0) or 0.0)
            findings.append(
                {
                    "type": str(anomaly.get("type") or "anomaly"),
                    "fingerprint": fingerprint,
                    "risk": {
                        "score": int(round(confidence * 100)),
                        "level": "high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low",
                        "confidence": confidence,
                    },
                    "evidence": anomaly.get("evidence") or {},
                }
            )
            if len(findings) >= _REPORT_FINDING_LIMIT:
                return findings
    return findings


_REPORT_FINDING_LIMIT = 8


def _filter_report_anomalies(anomalies: list[Any]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for anomaly in anomalies:
        if not isinstance(anomaly, dict) or not _is_reportable_anomaly(anomaly):
            continue
        fingerprint = _fingerprint(anomaly)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        filtered.append(anomaly)
    return filtered


def _anomaly_explanation_counts(transitions: list[dict[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
    raw_count = 0
    filter_reasons: dict[str, int] = {}
    seen: set[str] = set()
    duplicate_count = 0
    promoted_reasons: dict[str, int] = {}
    rejected_reasons: dict[str, int] = {}
    for row in transitions:
        anomalies = row.get("detected_anomalies") or []
        if not isinstance(anomalies, list):
            continue
        for anomaly in anomalies:
            if not isinstance(anomaly, dict):
                continue
            raw_count += 1
            fingerprint = _fingerprint(anomaly)
            if fingerprint in seen:
                duplicate_count += 1
                filter_reasons["duplicate_anomaly"] = filter_reasons.get("duplicate_anomaly", 0) + 1
                continue
            seen.add(fingerprint)
            if _is_reportable_anomaly(anomaly):
                reason = _report_promotion_reason(anomaly)
                promoted_reasons[reason] = promoted_reasons.get(reason, 0) + 1
            else:
                reason = _report_rejection_reason(anomaly)
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
    filtered_count = max(0, raw_count - len(findings))
    return {
        "raw_anomaly_candidate_count": raw_count,
        "filtered_false_positive_count": filtered_count,
        "duplicate_anomaly_count": duplicate_count,
        "anomaly_filter_reason_counts": filter_reasons,
        "finding_promotion_reason": _top_reason(promoted_reasons),
        "finding_rejection_reason": _top_reason(rejected_reasons),
    }


def _report_promotion_reason(anomaly: dict[str, Any]) -> str:
    confidence = float(anomaly.get("confidence", 0.0) or 0.0)
    status = str(anomaly.get("human_review_status") or "")
    if status:
        return f"{status}_reportable"
    if confidence >= 0.8:
        return "high_confidence_observed_signal"
    return "reportable_observed_signal"


def _report_rejection_reason(anomaly: dict[str, Any]) -> str:
    confidence = float(anomaly.get("confidence", 0.0) or 0.0)
    status = str(anomaly.get("human_review_status") or "")
    if status == "likely_false_positive":
        return "human_review_likely_false_positive"
    if confidence < 0.5:
        return "low_confidence_below_report_threshold"
    return "requires_verification_or_not_reportable"


def _top_reason(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-int(item[1] or 0), item[0]))[0][0]


def _camelize_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {_snake_to_camel(str(key)): value for key, value in payload.items()}


def _snake_to_camel(value: str) -> str:
    parts = value.split("_")
    if not parts:
        return value
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _is_reportable_anomaly(anomaly: dict[str, Any]) -> bool:
    anomaly_type = str(anomaly.get("type") or "")
    confidence = float(anomaly.get("confidence", 0.0) or 0.0)
    evidence = anomaly.get("evidence") if isinstance(anomaly.get("evidence"), dict) else {}
    if anomaly_type == "api-forbidden" and not _has_http_forbidden_evidence(evidence):
        return False
    if anomaly_type == "forum-post-detail-not-opened" and not _has_forum_post_detail_evidence(evidence):
        return False
    if anomaly_type == "forum-comment-delete-failed" and not evidence.get("comment_body_evidence"):
        return False
    if anomaly_type == "duplicated-rendering" and str(anomaly.get("ground_truth_match_status") or "") == "requires_component_evidence":
        return False
    if anomaly_type in {
        "api-forbidden",
        "api-4xx",
        "api-5xx",
        "api-ui-mismatch",
        "api-timeout",
        "auth-unauthorized-access",
        "backend-error",
        "database-error",
        "network-error",
        "console-error",
        "cors-error",
        "runtime-exception",
        "server-log-auth-error",
        "server-log-database-error",
        "server-log-error",
        "server-log-exception",
        "server-log-timeout",
        "server-port-not-open",
        "server-process-down",
        "database-connection-error",
        "database-hook-error",
        "database-integrity-failure",
        "db_invariant_violation",
        "transaction_partial_commit",
        "rollback_failed",
        "deadlock_or_lock_timeout",
        "negative_stock_or_balance",
        "server_exception_in_same_trace",
        "db_anomaly_in_same_trace",
        "api_schema_violation",
        "status_code_contract_violation",
        "api_contract_ui_mismatch",
        "login_redirect_mismatch",
        "protected_page_access_after_logout",
        "role_based_access_violation",
        "sensitive_data_visible_after_role_switch",
        "security-token-leak",
        "sensitive-data-exposure",
        "form-no-feedback",
        "functional-no-effect",
        "cart-quantity-mismatch",
        "cart-total-mismatch",
        "login-redirect-mismatch",
        "filter-no-effect",
        "stale-data-rendering",
        "duplicate-submission",
        "weak-password-validation",
        "broken-navigation",
        "product-detail-mismatch",
        "forum-post-detail-not-opened",
        "forum-comment-duplicated",
        "forum-comment-delete-failed",
        "forum-save-feedback-missing",
        "forum-empty-post-validation-missing",
    }:
        return confidence >= 0.5
    if anomaly_type == "layout-overlap" and evidence.get("specific_element_identified") is False:
        return False
    if anomaly_type in {"layout-overlap", "layout-overflow"}:
        return confidence >= 0.75
    if anomaly_type == "button-no-response":
        semantic = str(evidence.get("semantic_action_type") or "")
        target = evidence.get("target") if isinstance(evidence.get("target"), dict) else {}
        high_value = any(
            bool(evidence.get(key))
            for key in (
                "high_value_no_effect",
                "high_value_button_no_response",
                "add_no_effect",
                "cart_no_effect",
                "workout_add_no_effect",
                "enroll_no_effect",
            )
        )
        high_value = high_value or bool(target.get("is_high_value_functional_candidate"))
        return confidence >= 0.75 or (confidence >= 0.6 and bool(semantic or high_value))
    return confidence >= 0.8


def _has_http_forbidden_evidence(evidence: dict[str, Any]) -> bool:
    status = str(evidence.get("network_status") or evidence.get("status") or "")
    if status == "403":
        return True
    if int(evidence.get("api_403_count", 0) or 0) > 0:
        return True
    entries = evidence.get("network_entries")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and str(entry.get("status") or "") == "403":
                return True
    url = str(evidence.get("request_url") or evidence.get("url") or "")
    error = str(evidence.get("error") or "")
    return bool(url and "/api/" in url and ("403" in error or "forbidden" in error.lower()))


def _has_forum_post_detail_evidence(evidence: dict[str, Any]) -> bool:
    target = evidence.get("target") if isinstance(evidence.get("target"), dict) else {}
    role = str(target.get("role") or "").lower()
    tag = str(target.get("tag") or "").lower()
    if bool(target.get("fillable")) or role in {"textbox", "searchbox"} or tag in {"input", "textarea", "select"}:
        return False
    text = " ".join(str(target.get(key) or "") for key in ("name", "text")).strip()
    return len(text) >= 8


def _semantic_state_signature(row: dict[str, Any]) -> str:
    """Content-derived state id for graph/self-loop reporting.

    Deliberately excludes tick/step index, episode number, and timestamps --
    those change on every step by construction and would make every state look
    "new" even when the page content didn't actually change.
    """
    raw = "|".join(
        str(part)
        for part in (
            str(row.get("url") or ""),
            _safe_int(row.get("candidate_count"), 0),
            _safe_int(row.get("clickable_count"), 0),
            _safe_int(row.get("fillable_count"), 0),
            _safe_int(row.get("submit_count"), 0),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _fingerprint(anomaly: dict[str, Any]) -> str:
    evidence = anomaly.get("evidence") if isinstance(anomaly.get("evidence"), dict) else {}
    parts = [
        str(anomaly.get("type") or ""),
        str(evidence.get("before_url") or evidence.get("url") or ""),
        str(evidence.get("clicked_text") or evidence.get("target") or ""),
        str(evidence.get("selector") or ""),
        str(evidence.get("error") or ""),
    ]
    return "|".join(part.lower()[:160] for part in parts)


def _log_type(row: dict[str, Any]) -> str:
    explicit = str(row.get("log_type") or row.get("logCategory") or row.get("log_category") or "").lower()
    if explicit:
        return explicit
    if _is_action_log_row(row):
        return "action"
    if row.get("detected_anomalies") or row.get("anomalies"):
        return "issue"
    if row.get("network_events_count") or row.get("network_request_count"):
        return "network"
    if row.get("url"):
        return "state"
    return "debug"


def _is_action_log_row(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    explicit = str(row.get("log_type") or row.get("logCategory") or row.get("log_category") or "").lower()
    if explicit:
        return explicit == "action"
    action = str(row.get("action") or row.get("selected_action_type") or "").strip()
    return bool(action and action not in {"state", "network", "issue", "debug"})


def _action_execution_success(row: dict[str, Any]) -> bool:
    if not _is_action_log_row(row):
        return False
    return row.get("action_success") is True or row.get("execution_success") is True or row.get("success") is True


def _action_execution_failed(row: dict[str, Any]) -> bool:
    if not _is_action_log_row(row):
        return False
    return row.get("action_success") is False or row.get("execution_success") is False or row.get("success") is False


def _failure_reason_from_row(row: dict[str, Any]) -> str:
    return str(
        row.get("failure_reason")
        or row.get("invalid_action_reason")
        or row.get("exception_message")
        or row.get("playwright_error")
        or ""
    )


def _partial_transitions(job: ExplorationJob) -> list[dict[str, Any]]:
    return _read_jsonl(Path(job.start_result.scan_dir) / "partial_transitions.jsonl")


def _action_log_path(job: ExplorationJob) -> Path:
    return Path(job.start_result.scan_dir) / "action_logs.jsonl"


def _evaluation_path(job: ExplorationJob) -> Path:
    return Path(job.start_result.scan_dir) / "evaluation.json"


def _append_bridge_event(job: ExplorationJob, payload: dict[str, Any]) -> None:
    path = _action_log_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _last_error(job: ExplorationJob) -> str:
    errors = _read_jsonl(Path(job.start_result.errors_log_path))
    if not errors:
        return ""
    last = errors[-1]
    return str(last.get("message") or last)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _path_parts(path: str) -> list[str]:
    return [unquote(part) for part in path.strip("/").split("/") if part]


def _first(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key) or []
    return values[0] if values else ""


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _looks_like_form_or_login_url(target_url: str) -> bool:
    lowered = str(target_url or "").lower()
    return any(token in lowered for token in ("login", "sign-in", "signin", "auth", "form", "signup", "sign-up"))


def _looks_like_api_json_url(target_url: str) -> bool:
    lowered = str(target_url or "").lower()
    if not lowered:
        return False
    return bool(
        "/api/" in lowered
        or "json" in lowered
        or "typicode" in lowered
        or "my-json-server" in lowered
        or any(lowered.rstrip("/").endswith(suffix) for suffix in ("/posts", "/comments", "/profile", ".json"))
    )


def _first_early_stop_reason(raw: dict[str, Any]) -> str:
    reasons = raw.get("early_stop_reasons")
    if isinstance(reasons, dict):
        for key, value in reasons.items():
            if int(value or 0) > 0:
                return str(key)
    sites = raw.get("sites")
    if isinstance(sites, list):
        for site in sites:
            if isinstance(site, dict):
                reason = _first_early_stop_reason(site)
                if reason:
                    return reason
    return ""


def _completed_reason(raw: dict[str, Any]) -> str:
    if raw.get("completed_reason"):
        return str(raw.get("completed_reason"))
    status = str(raw.get("status") or "")
    if status == "failed":
        return "failed"
    reason = _first_early_stop_reason(raw)
    if reason:
        return reason
    if raw.get("partial"):
        return "partial"
    return "completed"


def _summary_int(raw: dict[str, Any], key: str) -> int:
    if raw.get(key) is not None:
        return _safe_int(raw.get(key), 0)
    sites = raw.get("sites")
    if isinstance(sites, list):
        return sum(_safe_int(site.get(key), 0) for site in sites if isinstance(site, dict))
    return 0


def _summary_float(raw: dict[str, Any], key: str) -> float:
    if raw.get(key) is not None:
        return float(raw.get(key) or 0.0)
    sites = raw.get("sites")
    if isinstance(sites, list) and sites:
        values = [float(site.get(key) or 0.0) for site in sites if isinstance(site, dict)]
        return sum(values) / len(values) if values else 0.0
    return 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
