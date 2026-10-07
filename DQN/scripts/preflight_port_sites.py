from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.known_bug_matcher import load_known_bugs
from services.site_profile_service import build_site_profile, validate_site_identity


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight generated port sites and keep only runnable sites.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--http-timeout", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _read_json_dict(config_path)
    sites = _validate_sites(config.get("sites", []))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_sites = []
    failed_sites = []
    checks = []

    for site in sites:
        result = _check_site(site, args.http_timeout, args.max_steps, args.max_candidates, args.headless)
        checks.append(result)
        if result.get("status") == "active":
            active_site = dict(site)
            active_site["preflight"] = {
                key: value
                for key, value in result.items()
                if key not in {"status", "reason", "error"}
            }
            active_sites.append(active_site)
            print(f"[active] {site['site_id']} {site['base_url']} candidates={result.get('candidate_count')}")
        else:
            failed = {
                "site_id": site["site_id"],
                "base_url": site["base_url"],
                "port": _port_from_base_url(str(site["base_url"])),
                "reason": str(result.get("reason") or result.get("error") or "preflight_failed"),
            }
            failed_sites.append(failed)
            print(f"[failed] {site['site_id']} {site['base_url']} reason={failed['reason']}")

    active_config = dict(config)
    active_config["sites"] = active_sites
    active_config["source_config"] = str(config_path)

    summary = {
        "source_config": str(config_path),
        "total_sites": len(sites),
        "active_sites": len(active_sites),
        "failed_sites": len(failed_sites),
        "checks": checks,
    }
    (output_dir / "active_sites.json").write_text(
        json.dumps(active_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "failed_sites.json").write_text(
        json.dumps(failed_sites, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "failed_ports.json").write_text(
        json.dumps(failed_sites, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in ("total_sites", "active_sites", "failed_sites")}, indent=2))
    return 0


def _check_site(
    site: Mapping[str, Any],
    http_timeout: float,
    max_steps: int,
    max_candidates: int,
    headless: bool,
) -> Dict[str, Any]:
    site_id = str(site["site_id"])
    base_url = str(site["base_url"])
    result: Dict[str, Any] = {
        "site_id": site_id,
        "base_url": base_url,
        "status": "failed",
        "http_ok": False,
        "browsergym_reset_ok": False,
        "candidate_count": 0,
        "has_bug_catalog": bool(site.get("has_bug_catalog")) and Path(str(site.get("bug_catalog_path", ""))).exists(),
        "bug_catalog_path": str(site.get("bug_catalog_path") or ""),
        "evaluation_mode": "known_bug_catalog" if bool(site.get("has_bug_catalog")) else "openended_anomaly_discovery",
        "login_required": bool(site.get("requires_login")),
        "login_attempted": False,
        "login_success": False,
    }

    http_result = _http_probe(base_url, http_timeout)
    result.update(http_result)
    if not http_result["http_ok"]:
        result["reason"] = http_result["reason"]
        return result

    known_bugs = _load_known_bugs(site)
    site_profile = build_site_profile(
        site_id,
        known_bugs,
        exploration_profile=site.get("exploration_profile"),
    )
    env = BrowserGymJAWSEnv(
        site_id=site_id,
        base_url=base_url,
        max_steps=max_steps,
        max_candidates=max_candidates,
        headless=headless,
        site_profile=site_profile,
        requires_login=bool(site.get("requires_login")),
        login_config=site.get("login") if isinstance(site.get("login"), Mapping) else None,
    )
    try:
        observation, reset_info = env.reset()
        identity = validate_site_identity(site_id, observation)
        candidates = observation.get("candidate_elements", []) or []
        result.update(
            {
                "status": "active",
                "browsergym_reset_ok": True,
                "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
                "observed_url": observation.get("page_state", {}).get("url", ""),
                "observed_title": observation.get("page_state", {}).get("title", ""),
                "login_required": bool(reset_info.get("login_required")),
                "login_attempted": bool(reset_info.get("login_attempted")),
                "login_success": bool(reset_info.get("login_success")),
                "post_login_url": str(reset_info.get("post_login_url") or ""),
                "known_bug_count": len(known_bugs),
                **identity,
            }
        )
    except Exception as exc:
        result["reason"] = _classify_exception(exc)
        result["error"] = str(exc)
    finally:
        env.close()
    return result


def _http_probe(base_url: str, timeout: float) -> Dict[str, Any]:
    request = Request(base_url, headers={"User-Agent": "jaws-browsergym-preflight/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return {
                "http_ok": True,
                "http_status": int(getattr(response, "status", 0) or 0),
                "reason": "",
            }
    except HTTPError as exc:
        return {"http_ok": True, "http_status": int(exc.code), "reason": ""}
    except URLError as exc:
        return {"http_ok": False, "http_status": None, "reason": _classify_exception(exc)}
    except (TimeoutError, socket.timeout) as exc:
        return {"http_ok": False, "http_status": None, "reason": f"TIMEOUT: {exc}"}
    except Exception as exc:
        return {"http_ok": False, "http_status": None, "reason": _classify_exception(exc)}


def _load_known_bugs(site: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not bool(site.get("has_bug_catalog")):
        return []
    path = Path(str(site.get("bug_catalog_path") or ""))
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [dict(item) for item in raw if isinstance(item, Mapping)]
        if isinstance(raw, Mapping):
            bugs = raw.get("bugs") or raw.get("items") or raw.get("bug_catalog") or raw.get("known_bugs") or []
            if isinstance(bugs, list):
                return [dict(item) for item in bugs if isinstance(item, Mapping)]
    return load_known_bugs(str(site.get("site_id") or ""))


def _classify_exception(exc: BaseException) -> str:
    message = str(exc)
    lowered = message.lower()
    if "connection refused" in lowered or "actively refused" in lowered:
        return "ERR_CONNECTION_REFUSED"
    if "timed out" in lowered or "timeout" in lowered:
        return "TIMEOUT"
    if "name or service not known" in lowered:
        return "DNS_ERROR"
    return message or exc.__class__.__name__


def _port_from_base_url(base_url: str) -> int | None:
    try:
        return int(str(base_url).rsplit(":", 1)[1].split("/", 1)[0])
    except (IndexError, ValueError):
        return None


def _validate_sites(raw_sites: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_sites, list):
        raise ValueError("Config requires a sites list.")
    sites = []
    for item in raw_sites:
        if not isinstance(item, Mapping):
            continue
        site_id = str(item.get("site_id") or "")
        base_url = str(item.get("base_url") or "")
        if site_id and base_url:
            sites.append(dict(item))
    return sites


def _read_json_dict(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
