from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.api_anomaly_detection_service import detect_api_anomalies, summarize_api_anomalies
from services.api_catalog_service import ApiEndpoint, ApiSite, load_api_catalog
from services.api_request_mutator import ApiFuzzCase, build_fuzz_cases
from services.api_reward_service import ApiRewardScorer


def main() -> int:
    parser = argparse.ArgumentParser(description="Run J.A.W.S direct API fuzz evaluation.")
    parser.add_argument("--config", default="configs/api_sites.json")
    parser.add_argument("--start-port", type=int, default=None)
    parser.add_argument("--end-port", type=int, default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--max-cases-per-endpoint", type=int, default=16)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    sites = load_api_catalog(args.config, args.start_port, args.end_port)
    run_id = args.run_id or _default_run_id(args.start_port, args.end_port)
    output_path = Path(args.output or f"artifacts/evaluations/{run_id}/result.json")
    scorer = ApiRewardScorer()
    site_results: Dict[str, Any] = {}
    all_anomalies: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []
    total_endpoints = 0
    total_cases = 0
    evaluated_sites = 0

    for site in sites:
        site_result = _evaluate_site(site, args.max_cases_per_endpoint, args.timeout_ms, scorer)
        site_results[site.site_id] = site_result
        all_anomalies.extend(site_result["anomalies"])
        transitions.extend(site_result["transitions"])
        total_endpoints += site_result["total_endpoints"]
        total_cases += site_result["total_cases"]
        if site_result["status"] == "evaluated":
            evaluated_sites += 1

    result = {
        "run_id": run_id,
        "source": "api",
        "evaluation_mode": "api_rule_based_fuzzing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "port_range": {"start_port": args.start_port, "end_port": args.end_port},
        "total_sites": len(sites),
        "evaluated_sites": evaluated_sites,
        "total_endpoints": total_endpoints,
        "total_cases": total_cases,
        "anomalies": all_anomalies,
        "summary": summarize_api_anomalies(all_anomalies),
        "sites": site_results,
        "transitions": transitions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("run_id", "total_sites", "evaluated_sites", "total_endpoints", "total_cases", "summary")}, ensure_ascii=False, indent=2))
    return 0


def _evaluate_site(
    site: ApiSite,
    max_cases_per_endpoint: int,
    timeout_ms: int,
    scorer: ApiRewardScorer,
) -> Dict[str, Any]:
    site_anomalies: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []
    endpoint_summaries: Dict[str, Any] = {}
    status = "evaluated"
    warnings: List[str] = []
    for endpoint in site.apis:
        endpoint_cases = 0
        endpoint_anomalies = 0
        endpoint_reward = 0.0
        try:
            cases = build_fuzz_cases(site, endpoint, max_cases=max_cases_per_endpoint)
            for case in cases:
                responses = _execute_case(case, timeout_ms)
                case_anomalies: List[Dict[str, Any]] = []
                for response in responses:
                    case_anomalies.extend(detect_api_anomalies(case.observation, case.action, response))
                expected_error_handled = _expected_error_handled(case, case_anomalies, responses)
                reward, reward_breakdown = scorer.score(
                    site.site_id,
                    endpoint.api_id,
                    str(case.action.get("mutation") or ""),
                    case_anomalies,
                    expected_error_handled=expected_error_handled,
                )
                endpoint_reward += reward
                endpoint_cases += 1
                endpoint_anomalies += len(case_anomalies)
                site_anomalies.extend(case_anomalies)
                transitions.append(
                    {
                        "source": "api",
                        "site_id": site.site_id,
                        "api_id": endpoint.api_id,
                        "observation": case.observation,
                        "action": case.action,
                        "reward": reward,
                        "reward_breakdown": reward_breakdown,
                        "anomalies": case_anomalies,
                    }
                )
        except Exception as exc:
            warnings.append(f"{endpoint.api_id}: {exc}")
        endpoint_summaries[endpoint.api_id] = {
            "method": endpoint.method,
            "path": endpoint.path,
            "cases": endpoint_cases,
            "anomalies": endpoint_anomalies,
            "reward": endpoint_reward,
        }
    if not site.apis:
        status = "skipped"
        warnings.append("no APIs configured")
    return {
        "source": "api",
        "site_id": site.site_id,
        "base_url": site.base_url,
        "status": status,
        "total_endpoints": len(site.apis),
        "total_cases": sum(item["cases"] for item in endpoint_summaries.values()),
        "anomaly_count": len(site_anomalies),
        "summary": summarize_api_anomalies(site_anomalies),
        "endpoints": endpoint_summaries,
        "anomalies": site_anomalies,
        "transitions": transitions,
        "warnings": warnings,
    }


def _execute_case(case: ApiFuzzCase, timeout_ms: int) -> List[Dict[str, Any]]:
    mutation = str(case.action.get("mutation") or "")
    if mutation == "repeat_request":
        return [_send_request(case.request, timeout_ms), _send_request(case.request, timeout_ms)]
    if mutation == "parallel_request":
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_send_request, case.request, timeout_ms) for _ in range(3)]
            return [future.result() for future in as_completed(futures)]
    return [_send_request(case.request, timeout_ms)]


def _send_request(request_data: Mapping[str, Any], timeout_ms: int) -> Dict[str, Any]:
    method = str(request_data.get("method") or "GET").upper()
    url = str(request_data.get("url") or "")
    query = request_data.get("query", {}) if isinstance(request_data.get("query"), Mapping) else {}
    if query:
        separator = "&" if "?" in url else "?"
        url = url + separator + urlencode(query, doseq=True)
    headers = dict(request_data.get("headers", {})) if isinstance(request_data.get("headers"), Mapping) else {}
    body_data = None
    body = request_data.get("body", {})
    if method in {"POST", "PUT", "PATCH"}:
        if str(headers.get("Content-Type", "")).lower().startswith("application/json"):
            body_data = json.dumps(body).encode("utf-8")
        elif body not in ({}, None):
            body_data = str(body).encode("utf-8")
    started = time.perf_counter()
    req = Request(url, data=body_data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=max(0.001, timeout_ms / 1000.0)) as response:
            raw = response.read(100000)
            latency_ms = (time.perf_counter() - started) * 1000.0
            return _response_payload(request_data, url, response.status, response.headers.get("Content-Type", ""), raw, latency_ms)
    except HTTPError as exc:
        raw = exc.read(100000)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return _response_payload(request_data, url, exc.code, exc.headers.get("Content-Type", ""), raw, latency_ms)
    except (TimeoutError, socket.timeout) as exc:
        return _exception_payload(request_data, url, timeout_ms, exc, timeout=True)
    except URLError as exc:
        reason = exc.reason if hasattr(exc, "reason") else exc
        if isinstance(reason, socket.timeout):
            return _exception_payload(request_data, url, timeout_ms, exc, timeout=True)
        return _exception_payload(request_data, url, timeout_ms, exc, timeout=False)
    except Exception as exc:
        return _exception_payload(request_data, url, timeout_ms, exc, timeout=False)


def _response_payload(
    request_data: Mapping[str, Any],
    url: str,
    status: int,
    content_type: str,
    raw: bytes,
    latency_ms: float,
) -> Dict[str, Any]:
    text = raw.decode("utf-8", errors="replace")
    json_body = None
    if text:
        try:
            json_body = json.loads(text)
        except json.JSONDecodeError:
            json_body = None
    return {
        "request": dict(request_data),
        "url": url,
        "status": int(status),
        "content_type": content_type,
        "body_preview": text[:2000],
        "json_body": json_body,
        "latency_ms": latency_ms,
        "timeout": False,
        "exception": "",
    }


def _exception_payload(
    request_data: Mapping[str, Any],
    url: str,
    timeout_ms: int,
    exc: BaseException,
    timeout: bool,
) -> Dict[str, Any]:
    return {
        "request": dict(request_data),
        "url": url,
        "status": None,
        "content_type": "",
        "body_preview": "",
        "json_body": None,
        "latency_ms": timeout_ms,
        "timeout": timeout,
        "exception": str(exc) or exc.__class__.__name__,
    }


def _expected_error_handled(
    case: ApiFuzzCase,
    anomalies: List[Mapping[str, Any]],
    responses: List[Mapping[str, Any]],
) -> bool:
    if anomalies:
        return False
    mutation = str(case.action.get("mutation") or "")
    if mutation not in {
        "missing_required_param",
        "empty_param",
        "null_param",
        "invalid_type_param",
        "invalid_enum_param",
        "missing_auth",
        "invalid_auth",
        "expired_auth",
        "invalid_content_type",
    }:
        return False
    statuses = [response.get("status") for response in responses]
    return any(isinstance(status, int) and 400 <= status < 500 for status in statuses)


def _default_run_id(start_port: int | None, end_port: int | None) -> str:
    if start_port is not None and end_port is not None:
        return f"api_ports_{start_port}_{end_port}"
    return "api_fuzz_eval"


if __name__ == "__main__":
    raise SystemExit(main())
