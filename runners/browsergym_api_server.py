from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
import uuid
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
    parser.add_argument("--min-steps", type=int, default=30)
    parser.add_argument("--max-steps-cap", type=int, default=50)
    parser.add_argument("--site-timeout-seconds", type=int, default=120)
    parser.add_argument("--episode-timeout-seconds", type=int, default=60)
    parser.add_argument("--reset-timeout-ms", type=int, default=60000)
    parser.add_argument("--step-timeout-ms", type=int, default=15000)
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()

    SERVER_OPTIONS["model_path"] = _resolve_model_path(args.model_path)
    SERVER_OPTIONS["python_executable"] = args.python_executable or sys.executable
    SERVER_OPTIONS["max_episodes"] = max(1, args.max_episodes)
    SERVER_OPTIONS["min_steps"] = max(1, args.min_steps)
    SERVER_OPTIONS["max_steps_cap"] = max(SERVER_OPTIONS["min_steps"], args.max_steps_cap)
    SERVER_OPTIONS["site_timeout_seconds"] = max(30, args.site_timeout_seconds)
    SERVER_OPTIONS["episode_timeout_seconds"] = max(10, args.episode_timeout_seconds)
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
        requested_max_steps = _safe_int(_first(query, "maxSteps") or body.get("maxSteps"), 30)
        min_steps = int(SERVER_OPTIONS.get("min_steps") or 30)
        if _looks_like_form_or_login_url(target_url):
            min_steps = max(min_steps, 30)
        if explicit_max_steps:
            max_steps = max(requested_max_steps, min_steps if _looks_like_form_or_login_url(target_url) else 1)
        else:
            max_steps = min(
                max(requested_max_steps, min_steps),
                int(SERVER_OPTIONS.get("max_steps_cap") or 50),
            )

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
                    "maxStepsCap": int(SERVER_OPTIONS.get("max_steps_cap") or 50),
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
        return terminal, error
    if job.start_result.status == "failed":
        return "failed", _last_error(job) or "BrowserGym scan failed"
    return "running", ""


def _job_progress(job: ExplorationJob, status: str) -> int:
    if status == "completed":
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
    findings = _filter_report_findings(_findings_from_result(raw))
    if not findings:
        findings = _findings_from_transitions(_partial_transitions(job))
    coverage = raw.get("coverage") if isinstance(raw.get("coverage"), dict) else {}
    if not coverage:
        transitions = _partial_transitions(job)
        actions = {str(row.get("action") or "") for row in transitions if row.get("action")}
        urls = {str(row.get("url") or "") for row in transitions if row.get("url")}
        coverage = {
            "coverage_score": min(1.0, (len(actions) + len(urls)) / 10.0),
            "visited_states": len(urls),
            "visited_actions": len(actions),
            "visited_transitions": len(transitions),
        }
    return {
        "jobId": job.job_id,
        "sessionId": job.session_id,
        "status": _job_status(job)[0],
        "coverage": coverage,
        "findings": findings,
        "rawSummary": raw,
    }


def _spring_step_events(job: ExplorationJob) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    reported_anomaly_fingerprints: set[str] = set()
    reported_anomaly_count = 0
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
        events.append(
            {
                "type": "step",
                "episode": _safe_int(row.get("episode"), 0),
                "step": _safe_int(row.get("step"), 0),
                "action": action,
                "action_id": f"{action}:{row.get('episode', 0)}:{row.get('step', 0)}",
                "url": str(row.get("url") or ""),
                "success": success,
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
                "target_element_key": str(row.get("target_element_key") or row.get("selected_target_element_key") or ""),
                "target_selector": str(row.get("target_selector") or ""),
                "retryable": bool(row.get("retryable", False)),
                "fallback_applied": bool(row.get("fallback_applied", False)),
                "fallback_reason": str(row.get("fallback_reason") or ""),
                "login_flow_status": str(row.get("login_flow_status") or ""),
                "required_actions_remaining": row.get("required_actions_remaining") if isinstance(row.get("required_actions_remaining"), list) else [],
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
    seen: set[str] = set()
    findings: list[dict[str, Any]] = []
    for row in transitions:
        for anomaly in _filter_report_anomalies(row.get("detected_anomalies") or []):
            if not isinstance(anomaly, dict):
                continue
            fingerprint = _fingerprint(anomaly)
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


def _is_reportable_anomaly(anomaly: dict[str, Any]) -> bool:
    anomaly_type = str(anomaly.get("type") or "")
    confidence = float(anomaly.get("confidence", 0.0) or 0.0)
    evidence = anomaly.get("evidence") if isinstance(anomaly.get("evidence"), dict) else {}
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
    }:
        return confidence >= 0.5
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
