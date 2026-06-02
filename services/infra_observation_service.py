"""Local server/infrastructure observation helpers for BrowserGym PPO."""

from __future__ import annotations

import socket
import time
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


HEALTH_PATHS = ("/", "/health", "/api/health", "/status")


def collect_infra_observation(
    site_id: str,
    base_url: str,
    timeout_ms: int = 750,
    health_paths: tuple[str, ...] = HEALTH_PATHS,
    log_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Collect safe, non-destructive local server signals for one site."""
    parsed = urlparse(str(base_url or ""))
    hostname = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    timeout_seconds = max(0.05, float(timeout_ms or 750) / 1000.0)
    result: dict[str, Any] = {
        "site_id": site_id,
        "base_url": base_url,
        "port": port,
        "port_open": False,
        "connection_refused": False,
        "health_check_ok": False,
        "health_endpoint": "",
        "health_status_code": "",
        "health_response_time_ms": "",
        "response_status": "",
        "response_latency_ms": "",
        "timeout_occurred": False,
        "timeout_ms": int(timeout_ms or 750),
        "server_5xx_count": 0,
        "server_4xx_count": 0,
        "server_error_message": "",
        "server_log_exception_count": _count_log_exceptions(log_paths or []),
        "process_alive": False,
        "cpu_usage_percent": "",
        "memory_usage_mb": "",
    }
    if not _is_local_host(hostname):
        result["server_error_message"] = "infra observation skipped for non-local host"
        return result

    port_status = _check_port(hostname, port, timeout_seconds)
    result.update(port_status)
    result["process_alive"] = bool(result["port_open"])
    if not result["port_open"]:
        return result

    for path in health_paths:
        health = _check_health(parsed.scheme or "http", hostname, port, path, timeout_seconds)
        if health.get("timeout_occurred"):
            result["timeout_occurred"] = True
        if health.get("server_error_message"):
            result["server_error_message"] = str(health.get("server_error_message") or "")
        status = health.get("response_status")
        if status not in (None, ""):
            result.update(
                {
                    "health_endpoint": path,
                    "health_status_code": status,
                    "health_response_time_ms": health.get("response_latency_ms", ""),
                    "response_status": status,
                    "response_latency_ms": health.get("response_latency_ms", ""),
                    "server_5xx_count": int(500 <= int(status) <= 599),
                    "server_4xx_count": int(400 <= int(status) <= 499),
                    "health_check_ok": 200 <= int(status) <= 399,
                }
            )
            if result["health_check_ok"]:
                break
    return result


def _check_port(hostname: str, port: int, timeout_seconds: float) -> dict[str, Any]:
    try:
        with socket.create_connection((hostname, int(port)), timeout=timeout_seconds):
            return {"port_open": True, "connection_refused": False}
    except ConnectionRefusedError as exc:
        return {"port_open": False, "connection_refused": True, "server_error_message": str(exc)}
    except socket.timeout as exc:
        return {
            "port_open": False,
            "connection_refused": False,
            "timeout_occurred": True,
            "server_error_message": str(exc),
        }
    except OSError as exc:
        return {"port_open": False, "connection_refused": False, "server_error_message": str(exc)}


def _check_health(scheme: str, hostname: str, port: int, path: str, timeout_seconds: float) -> dict[str, Any]:
    connection_cls = HTTPSConnection if scheme == "https" else HTTPConnection
    start = time.perf_counter()
    conn = connection_cls(hostname, port=int(port), timeout=timeout_seconds)
    try:
        conn.request("GET", path, headers={"User-Agent": "jaws-browsergym-infra-observer"})
        response = conn.getresponse()
        response.read(2048)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        return {"response_status": int(response.status), "response_latency_ms": latency_ms}
    except TimeoutError as exc:
        return {"timeout_occurred": True, "server_error_message": str(exc)}
    except socket.timeout as exc:
        return {"timeout_occurred": True, "server_error_message": str(exc)}
    except OSError as exc:
        return {"server_error_message": str(exc)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _count_log_exceptions(log_paths: list[str]) -> int:
    count = 0
    for raw_path in log_paths[:5]:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[-20000:].lower()
        except OSError:
            continue
        count += text.count("traceback") + text.count("exception") + text.count("error:")
    return count


def _is_local_host(hostname: str) -> bool:
    lowered = hostname.lower()
    return lowered in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or lowered.startswith("127.")
