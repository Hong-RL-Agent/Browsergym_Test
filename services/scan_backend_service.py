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
DEFAULT_MODEL_PATH = "artifacts/models/jaws_browsergym_shared_ppo_v2_browsergym_raw_obs.pt"
EVALUATOR_PATH = "runners/evaluate_multisite_browsergym_agent.py"


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
    episodes: int = 3,
    max_steps: int = 25,
    model_path: str = DEFAULT_MODEL_PATH,
    python_executable: str | None = None,
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
    config = {
        "sites": [
            {
                "site_id": site_id,
                "base_url": normalized_url,
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
    threading.Thread(
        target=_capture_process_output,
        args=(process, action_log, errors_log),
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


def _capture_process_output(process: subprocess.Popen[str], action_log: Path, errors_log: Path) -> None:
    stdout_text, stderr_text = process.communicate()
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
