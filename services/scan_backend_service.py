from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = "artifacts/models/jaws_browsergym_shared_ppo_v3_policy_safe.pt"
EVALUATOR_PATH = "runners/evaluate_multisite_browsergym_agent.py"
RUNNING_PROCESSES: dict[str, subprocess.Popen[str]] = {}


@dataclass(frozen=True)
class ScanStartResult:
    scan_id: str
    status: str
    scan_dir: str
    config_path: str
    errors_log_path: str
    process_pid: int | None = None


def start_browsergym_scan(
    *,
    scan_id: str,
    target_url: str,
    episodes: int = 1,
    max_steps: int = 30,
    site_timeout_seconds: int = 300,
    episode_timeout_seconds: int = 0,
    reset_timeout_ms: int = 60000,
    step_timeout_ms: int = 45000,
    model_path: str = DEFAULT_MODEL_PATH,
    python_executable: str | None = None,
    agent_config: dict[str, Any] | None = None,
) -> ScanStartResult:
    """Start a BrowserGym scan for exactly one user-provided target URL."""

    normalized_url = _normalize_target_url(target_url)
    scan_dir = ROOT / "artifacts" / "scans" / scan_id
    scan_dir.mkdir(parents=True, exist_ok=True)
    errors_log = scan_dir / "errors.log"
    action_log = scan_dir / "action_logs.jsonl"
    config_path = scan_dir / "scan_config.json"
    output_path = scan_dir / "evaluation.json"

    site_id = _site_id_from_url(normalized_url)
    agent_config = agent_config if isinstance(agent_config, dict) else {}
    observability_config = _observability_config(agent_config)
    config = {
        "run_id": scan_id,
        "eval_episodes": episodes,
        "max_steps": max_steps,
        "site_timeout_seconds": site_timeout_seconds,
        "episode_timeout_seconds": episode_timeout_seconds,
        "reset_timeout_ms": reset_timeout_ms,
        "step_timeout_ms": step_timeout_ms,
        "enforce_target_url_boundary": bool(agent_config.get("enforce_target_url_boundary", True)),
        "allowed_hosts": _allowed_hosts(normalized_url, agent_config),
        "allowed_path_prefixes": _allowed_path_prefixes(normalized_url, agent_config),
        "blocked_url_keywords": _blocked_url_keywords(agent_config),
        "finish_on_external_navigation": bool(agent_config.get("finish_on_external_navigation", True)),
        "reset_retry_count": 1,
        "use_memory_encoder": "gru_memory" in str(model_path).lower(),
        "memory_encoder_type": "gru",
        "memory_hidden_size": 128,
        "use_known_bug_for_evaluation": False,
        "use_known_bug_reward": False,
        "allow_known_bug_reward_for_debug_only": False,
        "policy_uses_bug_labels": False,
        "sites": [
            {
                "site_id": site_id,
                "base_url": normalized_url,
                **observability_config,
            }
        ],
    }
    _write_json(config_path, config)
    _append_action(action_log, event="scan_config_created", site_id=site_id, base_url=normalized_url, config=str(config_path))

    preflight_error = _preflight_target(normalized_url)
    if preflight_error:
        message = f"Target URL unreachable from backend runtime: {normalized_url}"
        _append_error(errors_log, message, detail=preflight_error)
        _append_action(action_log, event="scan_failed", message=message, detail=preflight_error)
        return ScanStartResult(
            scan_id=scan_id,
            status="failed",
            scan_dir=str(scan_dir),
            config_path=str(config_path),
            errors_log_path=str(errors_log),
        )

    command = [
        python_executable or sys.executable,
        EVALUATOR_PATH,
        "--config",
        str(config_path),
        "--model-path",
        model_path,
        "--episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--site-timeout-seconds",
        str(site_timeout_seconds),
        "--episode-timeout-seconds",
        str(episode_timeout_seconds),
        "--reset-timeout-ms",
        str(reset_timeout_ms),
        "--step-timeout-ms",
        str(step_timeout_ms),
        "--output",
        str(output_path),
        "--run-id",
        scan_id,
    ]
    _append_action(action_log, event="subprocess_started", command=command)

    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    RUNNING_PROCESSES[scan_id] = process
    threading.Thread(
        target=_capture_process_output,
        args=(scan_id, process, action_log, errors_log),
        daemon=True,
    ).start()
    return ScanStartResult(
        scan_id=scan_id,
        status="running",
        scan_dir=str(scan_dir),
        config_path=str(config_path),
        errors_log_path=str(errors_log),
        process_pid=process.pid,
    )


def cancel_browsergym_scan(scan_id: str, *, action_log_path: str | Path | None = None) -> bool:
    process = RUNNING_PROCESSES.get(str(scan_id))
    if process is None or process.poll() is not None:
        return False
    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if action_log_path:
            _append_action(Path(action_log_path), event="subprocess_terminated", scan_id=scan_id, return_code=process.returncode)
        return True
    finally:
        RUNNING_PROCESSES.pop(str(scan_id), None)


def _capture_process_output(scan_id: str, process: subprocess.Popen[str], action_log: Path, errors_log: Path) -> None:
    stdout_text, stderr_text = process.communicate()
    RUNNING_PROCESSES.pop(str(scan_id), None)
    for line in stdout_text.splitlines():
        text = line.rstrip()
        if not text:
            continue
        event = _try_json(text) or {"event": "subprocess_stdout", "message": text}
        _append_action(action_log, **event)

    if stderr_text:
        _append_error(errors_log, stderr_text.rstrip())
        _append_action(action_log, event="subprocess_stderr", message=stderr_text.rstrip())
    return_code = process.returncode
    if return_code != 0:
        message = f"BrowserGym evaluator failed with exit code {return_code}"
        _append_error(errors_log, message)
        _append_action(action_log, event="scan_failed", message=message, return_code=return_code)
    else:
        _append_action(action_log, event="scan_completed", return_code=return_code)


def _preflight_target(target_url: str) -> str:
    try:
        request = Request(target_url, method="GET", headers={"User-Agent": "browsergym-scan-preflight/1.0"})
        with urlopen(request, timeout=5) as response:
            if int(response.status) >= 500:
                return f"HTTP {response.status}"
    except HTTPError as exc:
        if int(exc.code) >= 500:
            return f"HTTP {exc.code}"
    except URLError as exc:
        return str(exc.reason)
    except Exception as exc:
        return str(exc)
    return ""


def _normalize_target_url(target_url: str) -> str:
    value = str(target_url or "").strip()
    if not value:
        raise ValueError("targetUrl is required.")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"targetUrl must be an absolute http(s) URL: {target_url}")
    return value


def _site_id_from_url(target_url: str) -> str:
    parsed = urlparse(target_url)
    if parsed.port:
        return f"site{parsed.port}"
    host = "".join(ch for ch in parsed.hostname or "target" if ch.isalnum())
    return f"site{host or 'target'}"


def _allowed_hosts(target_url: str, agent_config: dict[str, Any]) -> list[str]:
    configured = _string_list(agent_config.get("allowed_hosts") or agent_config.get("allowedHosts"))
    if configured:
        return configured
    parsed = urlparse(target_url)
    return [str(parsed.hostname or "").lower()] if parsed.hostname else []


def _allowed_path_prefixes(target_url: str, agent_config: dict[str, Any]) -> list[str]:
    configured = _string_list(agent_config.get("allowed_path_prefixes") or agent_config.get("allowedPathPrefixes"))
    if configured:
        return configured
    parsed = urlparse(target_url)
    path = str(parsed.path or "/")
    if path.endswith("/"):
        return [path]
    parent = path.rsplit("/", 1)[0] + "/"
    return [path, parent]


def _blocked_url_keywords(agent_config: dict[str, Any]) -> list[str]:
    configured = _string_list(agent_config.get("blocked_url_keywords") or agent_config.get("blockedUrlKeywords"))
    defaults = [
        "checkout",
        "payment",
        "billing",
        "gumroad.com",
        "paypal.com",
        "stripe.com",
        "oauth",
        "sign-up",
        "signup",
    ]
    return configured or defaults


def _observability_config(agent_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "server_log_paths": _string_list(agent_config.get("server_log_paths") or agent_config.get("serverLogPaths")),
        "database_paths": _string_list(
            agent_config.get("database_paths")
            or agent_config.get("db_paths")
            or agent_config.get("databasePaths")
            or agent_config.get("dbPaths")
        ),
        "db_engine": str(agent_config.get("db_engine") or agent_config.get("dbEngine") or ""),
        "relational_db_hook_available": bool(agent_config.get("relational_db_hook_available") or agent_config.get("relationalDbHookAvailable")),
        "postgres_hook_available": bool(agent_config.get("postgres_hook_available") or agent_config.get("postgresHookAvailable")),
        "mysql_hook_available": bool(agent_config.get("mysql_hook_available") or agent_config.get("mysqlHookAvailable")),
        "db_invariant_available": bool(agent_config.get("db_invariant_available") or agent_config.get("dbInvariantAvailable")),
        "trace_collector_available": bool(agent_config.get("trace_collector_available") or agent_config.get("traceCollectorAvailable")),
        "api_contract_available": bool(agent_config.get("api_contract_available") or agent_config.get("apiContractAvailable")),
        "schema_available": bool(agent_config.get("schema_available") or agent_config.get("schemaAvailable")),
        "openapi_spec_available": bool(agent_config.get("openapi_spec_available") or agent_config.get("openapiSpecAvailable")),
        "openapi_spec_path": str(agent_config.get("openapi_spec_path") or agent_config.get("openapiSpecPath") or ""),
        "api_contract": agent_config.get("api_contract") if isinstance(agent_config.get("api_contract"), dict) else {},
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_action(path: Path, **payload: Any) -> None:
    payload.setdefault("timestamp", _utc_now())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_error(path: Path, message: str, **payload: Any) -> None:
    entry = {"timestamp": _utc_now(), "message": message, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _try_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
