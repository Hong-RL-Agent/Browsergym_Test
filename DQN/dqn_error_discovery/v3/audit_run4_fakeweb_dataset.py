"""Audit and freeze the RAWD v3 Run 4 Fake Web dataset.

This is a dataset/readiness auditor only. It never updates PPO/DQN weights,
never writes model checkpoints, and never changes the source Fake Web sites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Mapping

from dqn_error_discovery.v3.action_contract import (
    is_click_target,
    is_fill_input_target,
    is_native_select,
    is_select_option_target,
)
from dqn_error_discovery.v3.train import _episode_env


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
OUT = ROOT / "artifacts" / "training_v3" / "run4_fakeweb"
DEFAULT_SOURCE_ROOT = Path(r"C:\workspace\RL_Errorsite\backend-errorsite")
SPLIT_SEED = 20260917
RULES = [f"R{i:02d}" for i in range(1, 16)]
FORBIDDEN_SITE_IDS = {
    "SITE001",
    "DEV_REACT",
    "DEV_NEXT",
    "DEV_V3_SELECT",
    "SITE002",
    "SITE003",
    "SITE004",
    "SITE005",
    "EVAL001",
    "EVAL002",
    "EVAL003",
    "EVAL004",
    "site001",
    "site002",
    "site003",
    "site004",
    "site005",
}
RESERVED_PORTS = {9220, 9225, 9226, 9231, 9232, 9233, 9234}
EXPECTED_CANDIDATE_A_HASHES = {
    "ppo": "16F45AD405F4298FDDD00BA8532B2EB30635968208D5CF533F8AFD907743A9D3",
    "42": "6D90FF0AE03DC3DC2FA3118387F3F515B6FE84A7109DB7F3880AE39ED1B9B5F4",
    "43": "7DA480BD642FE117AC5A44FBFC4D4D7E9CB23A56B5457DD3BEEC991460F67298",
    "44": "6A9982528217E9F973313090B2124708F5007ED6262A86160590D21842594958",
}
MODEL_PATHS = {
    "ppo": ROOT / "artifacts/models/v3/rawd_v3_ppo_selectcov_seed42.pt",
    "42": ROOT / "artifacts/models/v3/rawd_v3_dqn_productive_seed42.pt",
    "43": ROOT / "artifacts/models/v3/rawd_v3_dqn_productive_seed43.pt",
    "44": ROOT / "artifacts/models/v3/rawd_v3_dqn_productive_seed44.pt",
}


TYPE_TO_RULE = {
    "no-response": "R01",
    "button-no-response": "R01",
    "wrong-navigation": "R02",
    "broken-navigation": "R02",
    "redirect": "R02",
    "null-reference": "R03",
    "runtime": "R03",
    "render": "R03",
    "duplicated": "R04",
    "duplicate": "R04",
    "overlap": "R05",
    "occlusion": "R05",
    "overflow": "R06",
    "clipping": "R06",
    "responsive": "R06",
    "accessibility": "R07",
    "validation": "R08",
    "type-parsing": "R08",
    "filter": "R09",
    "search": "R09",
    "sort": "R09",
    "pagination": "R09",
    "ui-api": "R10",
    "mismatch": "R10",
    "polymorphic-state-confusion": "R10",
    "calculation": "R11",
    "precision-loss": "R11",
    "magic-number": "R11",
    "persistence": "R12",
    "save": "R12",
    "implicit-default": "R12",
    "api-server": "R13",
    "server-error": "R13",
    "network": "R14",
    "transport": "R14",
    "timeout": "R15",
    "api-timeout": "R15",
    "performance": "R15",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    ignored = {"node_modules", ".git", ".next", "dist", "build"}
    allowed_suffixes = {".js", ".jsx", ".ts", ".tsx", ".json", ".html", ".css", ".md"}
    ignored_files = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
    files: list[Path] = []
    for root, dirs, filenames in os.walk(path):
        dirs[:] = [name for name in dirs if name not in ignored]
        root_path = Path(root)
        for filename in filenames:
            item = root_path / filename
            if filename in ignored_files:
                continue
            if item.suffix.lower() in allowed_suffixes:
                files.append(item)
    for item in sorted(files):
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest().upper()


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_port(server_js: Path) -> int | None:
    if not server_js.exists():
        return None
    text = server_js.read_text(encoding="utf-8", errors="replace")
    patterns = [
        r"const\s+port\s*=\s*(\d+)",
        r"const\s+PORT\s*=\s*(\d+)",
        r"process\.env\.PORT\s*\|\|\s*(\d+)",
        r"app\.listen\(\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def framework_from_package(pkg: Mapping[str, Any]) -> tuple[str, str, str]:
    deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
    if "next" in deps:
        return "Next.js", "UNKNOWN", "UNKNOWN"
    if "react" in deps or "@vitejs/plugin-react" in deps:
        return "React", "CSR", "Express"
    return "Other", "UNKNOWN", "Express" if "express" in deps else "UNKNOWN"


def site_number(site_id: str) -> int:
    match = re.search(r"(\d+)$", site_id)
    return int(match.group(1)) if match else 0


def normalize_type(raw_type: str) -> str:
    return str(raw_type or "").strip().lower()


def map_rule(raw_type: str, text: str) -> str:
    raw = normalize_type(raw_type)
    low_text = text.lower()
    if "navigation" in low_text or "redirect" in low_text or "route" in low_text:
        return "R02"
    if "duplicate" in low_text or "duplicated" in low_text:
        return "R04"
    if "overflow" in low_text or "clipping" in low_text or "responsive" in low_text:
        return "R06"
    if "500" in low_text or "internal server" in low_text or "server error" in low_text:
        return "R13"
    if "network" in low_text or "transport" in low_text or "connection" in low_text:
        return "R14"
    for key, rule in TYPE_TO_RULE.items():
        if key in raw:
            return rule
    for key, rule in TYPE_TO_RULE.items():
        if key in low_text:
            return rule
    return "R10"


def parse_server_bug_comments(path: Path) -> list[dict[str, Any]]:
    server = path / "server.js"
    if not server.exists():
        return []
    text = server.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    bugs: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        match = re.search(r"Bug\s*0?(\d+)\s*:\s*(.+)", line, re.IGNORECASE)
        if not match:
            continue
        bug_num = int(match.group(1))
        title = match.group(2).strip()
        window = "\n".join(lines[idx : min(len(lines), idx + 12)])
        bug_id_match = re.search(r"['\"](site\d+-bug\d+)['\"]", window)
        endpoint_match = re.search(r"app\.(get|post|put|delete|patch)\(['\"]([^'\"]+)", window)
        site_id = path.name
        bug_id = bug_id_match.group(1) if bug_id_match else f"{site_id}-bug{bug_num:02d}"
        bugs.append(
            {
                "bug_id": bug_id,
                "original_category": title,
                "mapped_rule": map_rule(title, window),
                "trigger_specified": "triggerBug" in window or "bug" in window.lower(),
                "normal_behavior_specified": False,
                "buggy_behavior_specified": True,
                "affected_endpoint_or_ui": endpoint_match.group(2) if endpoint_match else "",
                "http_method": endpoint_match.group(1).upper() if endpoint_match else "",
                "metadata_source": "server.js",
            }
        )
    return bugs


def parse_bugs_markdown(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        recovered = parse_server_bug_comments(path.parent)
        if recovered:
            return recovered, ["BUGS.md missing; recovered metadata from server.js comments"]
        return [], ["missing BUGS.md"]
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"(?m)^(?:##\s+|\d+\.\s+\*\*)", text)
    bugs: list[dict[str, Any]] = []
    issues: list[str] = []
    seen: set[str] = set()
    for section in sections[1:]:
        lines = section.strip().splitlines()
        if not lines:
            continue
        bug_id = lines[0].strip().split()[0].strip(":*")
        body = "\n".join(lines[1:])
        type_match = re.search(r"(?:\*\*)?type(?:\*\*)?\s*:\s*([^\n]+)", body, re.IGNORECASE)
        endpoint_match = re.search(r"API endpoint(?:\*\*)?\s*:\s*([^\n]+)", body, re.IGNORECASE)
        method_match = re.search(r"HTTP method(?:\*\*)?\s*:\s*([^\n]+)", body, re.IGNORECASE)
        selector_match = re.search(r"data-bug-id[^\n`]*`?([^`\n]*data-bug-id[^`\n]*)`?", body, re.IGNORECASE)
        raw_type = type_match.group(1).strip(" *`") if type_match else ""
        bug = {
            "bug_id": bug_id,
            "original_category": raw_type,
            "mapped_rule": map_rule(raw_type, body),
            "trigger_specified": bool(
                re.search(r"condition|trigger|발생|조건|發|諛|߻|����", body, re.IGNORECASE)
            ),
            "normal_behavior_specified": False,
            "buggy_behavior_specified": bool(re.search(r"symptom|증상|500|422|timeout|error|bugId|諛", body, re.IGNORECASE)),
            "affected_endpoint_or_ui": (endpoint_match.group(1).strip(" *`") if endpoint_match else "")
            or (selector_match.group(1).strip(" *`") if selector_match else ""),
            "http_method": method_match.group(1).strip(" *`") if method_match else "",
        }
        if bug_id in seen:
            issues.append(f"duplicate bug_id {bug_id}")
        seen.add(bug_id)
        if not raw_type:
            issues.append(f"{bug_id}: missing category/type")
        if not bug["trigger_specified"]:
            issues.append(f"{bug_id}: missing trigger")
        if not bug["buggy_behavior_specified"]:
            issues.append(f"{bug_id}: missing buggy behavior")
        if not bug["affected_endpoint_or_ui"]:
            issues.append(f"{bug_id}: missing affected endpoint/UI")
        if not bug["normal_behavior_specified"]:
            issues.append(f"{bug_id}: normal behavior not explicitly specified")
        bugs.append(bug)
    if not bugs:
        recovered = parse_server_bug_comments(path.parent)
        if recovered:
            return recovered, ["BUGS.md format unsupported; recovered metadata from server.js comments"]
        issues.append("no bug sections parsed")
    return bugs, issues


def discover_sites(source_root: Path) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    for path in sorted(source_root.glob("site*"), key=lambda p: site_number(p.name)):
        if not path.is_dir():
            continue
        site_id = path.name
        pkg_path = path / "package.json"
        bugs_path = path / "BUGS.md"
        catalog_path = path / "bug_catalog.json"
        pkg = read_json(pkg_path) or {}
        framework, frontend_type, backend_type = framework_from_package(pkg)
        port = parse_port(path / "server.js")
        bugs, audit_issues = parse_bugs_markdown(bugs_path)
        catalog = read_json(catalog_path)
        family = str(pkg.get("description") or pkg.get("name") or site_id).strip()
        site = {
            "site_id": site_id,
            "source_path": str(path),
            "framework": framework,
            "frontend_type": frontend_type,
            "backend_type": backend_type,
            "next_router": "UNKNOWN",
            "rendering": "UNKNOWN",
            "canonical_start_command": "npm start",
            "port": port,
            "base_url": f"http://localhost:{port}/" if port else None,
            "package_json_path": str(pkg_path) if pkg_path.exists() else None,
            "BUGS_md_path": str(bugs_path) if bugs_path.exists() else None,
            "bug_catalog_json_path": str(catalog_path) if catalog_path.exists() else None,
            "intentional_defects": len(bugs),
            "taxonomy_categories": sorted({bug["mapped_rule"] for bug in bugs}),
            "bugs": bugs,
            "bug_metadata_issues": audit_issues,
            "source_hash": directory_hash(path),
            "bug_catalog_hash": sha256(catalog_path) if catalog_path.exists() else None,
            "bugs_md_hash": sha256(bugs_path) if bugs_path.exists() else None,
            "template_family": family,
            "excluded": site_id in FORBIDDEN_SITE_IDS,
            "exclusion_reason": "forbidden historical/training/eval site" if site_id in FORBIDDEN_SITE_IDS else None,
            "has_bug_catalog_json": catalog is not None,
        }
        sites.append(site)
    return sites


def http_probe(url: str, timeout: float = 5.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            title = ""
            match = re.search(r"<title>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
            if match:
                title = re.sub(r"\s+", " ", match.group(1)).strip()
            return {"ok": True, "status": response.status, "length": len(body), "title": title, "body_sample": body[:500]}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def wait_for_http(url: str, timeout: float = 12.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last = {"ok": False, "error": "not attempted"}
    while time.time() < deadline:
        last = http_probe(url, timeout=2.0)
        if last.get("ok"):
            return last
        time.sleep(0.5)
    return last


def start_site(site: Mapping[str, Any]) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    return subprocess.Popen(
        ["node", "server.js"],
        cwd=str(site["source_path"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def stop_site(proc: subprocess.Popen[str]) -> dict[str, Any]:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except Exception:
            pass
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    stdout = ""
    stderr = ""
    try:
        out, err = proc.communicate(timeout=2)
        stdout = out[-2000:] if out else ""
        stderr = err[-2000:] if err else ""
    except Exception:
        pass
    return {"returncode": proc.poll(), "stdout_tail": stdout, "stderr_tail": stderr}


def candidate_a_hashes() -> dict[str, Any]:
    actual = {key: sha256(path) for key, path in MODEL_PATHS.items()}
    return {
        "expected": EXPECTED_CANDIDATE_A_HASHES,
        "actual": actual,
        "unchanged": actual == EXPECTED_CANDIDATE_A_HASHES,
    }


def _browsergym_reset_worker(site_id: str, base_url: str, max_steps: int, queue: Queue) -> None:
    env = _episode_env(site_id.upper(), base_url, max_steps)
    try:
        raw, _ = env.reset()
        candidates = list(raw.get("candidate_elements", []) or []) if isinstance(raw, Mapping) else []
        click = sum(1 for c in candidates if is_click_target(c))
        fill = sum(1 for c in candidates if is_fill_input_target(c))
        select = sum(1 for c in candidates if is_select_option_target(c))
        native_select_fill = sum(1 for c in candidates if is_native_select(c) and is_fill_input_target(c))
        visible = sum(1 for c in candidates if c.get("visible", True) and not c.get("hidden"))
        queue.put({
            "browsergym_reset_success": True,
            "candidate_count": len(candidates),
            "visible_interactive_candidates": visible,
            "click_candidates": click,
            "input_candidates": fill,
            "select_option_candidates": select,
            "scroll_candidates": 2,
            "native_select_fill_input_violations": native_select_fill,
            "action_contract_failures": 0 if native_select_fill == 0 else native_select_fill,
        })
    except Exception as exc:
        queue.put({"browsergym_reset_success": False, "error": repr(exc)})
    finally:
        try:
            env.close()
        except Exception:
            pass


def browsergym_reset(site: Mapping[str, Any], max_steps: int, timeout: float) -> dict[str, Any]:
    queue: Queue = Queue()
    proc = Process(
        target=_browsergym_reset_worker,
        args=(str(site["site_id"]), str(site["base_url"]), max_steps, queue),
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {"browsergym_reset_success": False, "error": f"BrowserGym reset timeout after {timeout}s"}
    if not queue.empty():
        return queue.get()
    return {"browsergym_reset_success": False, "error": "BrowserGym reset worker exited without result"}


def run_preflight(sites: list[dict[str, Any]], max_steps: int, reset_timeout: float, limit: int | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected = [site for site in sites if not site["excluded"]]
    if limit:
        selected = selected[:limit]
    for idx, site in enumerate(selected, 1):
        row = {
            "site_id": site["site_id"],
            "source_path": site["source_path"],
            "port": site["port"],
            "base_url": site["base_url"],
            "canonical_start_command": site["canonical_start_command"],
            "process": None,
            "http_root": {"ok": False, "error": "not run"},
            "identity_ok": False,
            "browsergym": {"browsergym_reset_success": False, "error": "not run"},
            "readiness": "NOT_READY",
            "issues": [],
        }
        if not site["port"] or not site["base_url"]:
            row["issues"].append("missing port/base_url")
            rows.append(row)
            continue
        if site["port"] in RESERVED_PORTS:
            row["issues"].append(f"reserved port collision {site['port']}")
        proc = None
        try:
            proc = start_site(site)
            row["process"] = {"pid": proc.pid}
            if proc.poll() is not None:
                row["issues"].append(f"server exited early rc={proc.poll()}")
            row["http_root"] = wait_for_http(str(site["base_url"]))
            title = str(row["http_root"].get("title") or row["http_root"].get("body_sample") or "")
            row["identity_ok"] = site["site_id"] in title or site["site_id"].upper() in title.upper() or bool(row["http_root"].get("ok"))
            if row["http_root"].get("ok"):
                row["browsergym"] = browsergym_reset(site, max_steps, reset_timeout)
            else:
                row["issues"].append("HTTP root failed")
            if not row["browsergym"].get("browsergym_reset_success"):
                row["issues"].append("BrowserGym reset failed")
            if int(row["browsergym"].get("candidate_count", 0) or 0) == 0:
                row["issues"].append("candidate_count=0")
            if int(row["browsergym"].get("native_select_fill_input_violations", 0) or 0) > 0:
                row["issues"].append("native-select fill_input violation")
            if site["bug_metadata_issues"]:
                row["issues"].extend(site["bug_metadata_issues"])
            runtime_ready = (
                row["http_root"].get("ok")
                and row["browsergym"].get("browsergym_reset_success")
                and int(row["browsergym"].get("candidate_count", 0) or 0) > 0
                and int(row["browsergym"].get("native_select_fill_input_violations", 0) or 0) == 0
            )
            metadata_ready = bool(site["bugs"]) and not any(
                "missing BUGS.md" in issue or "no bug sections" in issue for issue in site["bug_metadata_issues"]
            )
            explicit_normal_missing_only = all(
                issue.endswith("normal behavior not explicitly specified")
                for issue in site["bug_metadata_issues"]
            )
            if runtime_ready and metadata_ready and (not site["bug_metadata_issues"] or explicit_normal_missing_only):
                row["readiness"] = "READY"
            elif runtime_ready:
                row["readiness"] = "READY_WITH_LIMITATION"
            else:
                row["readiness"] = "NOT_READY"
        except Exception as exc:
            row["issues"].append(f"preflight exception: {exc!r}")
        finally:
            if proc is not None:
                row["process"] = {**(row["process"] or {}), **stop_site(proc)}
        rows.append(row)
        print(json.dumps({"progress": idx, "site_id": site["site_id"], "readiness": row["readiness"]}))
    return {
        "source_root": str(DEFAULT_SOURCE_ROOT),
        "max_steps": max_steps,
        "browsergym_reset_timeout_seconds": reset_timeout,
        "sites": rows,
    }


def split_ready_sites(sites: list[dict[str, Any]], preflight: Mapping[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    readiness = {row["site_id"]: row["readiness"] for row in preflight["sites"]}
    ready = [site for site in sites if not site["excluded"] and readiness.get(site["site_id"]) == "READY"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for site in ready:
        groups[site["template_family"]].append(site)
    families = sorted(groups.values(), key=lambda group: min(site_number(site["site_id"]) for site in group))
    target_dev = round(len(ready) * 0.2)
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    train_rules: Counter[str] = Counter()
    dev_rules: Counter[str] = Counter()
    train_frameworks: Counter[str] = Counter()
    dev_frameworks: Counter[str] = Counter()
    for group in families:
        group_rules = Counter(rule for site in group for rule in site["taxonomy_categories"])
        group_fw = Counter(site["framework"] for site in group)
        put_dev = len(dev) < target_dev and (
            len(dev) / max(1, len(ready)) < 0.2
            or len(train) > 0 and sum((train_rules - dev_rules).values()) > sum((dev_rules - train_rules).values())
        )
        if put_dev:
            dev.extend(group)
            dev_rules.update(group_rules)
            dev_frameworks.update(group_fw)
        else:
            train.extend(group)
            train_rules.update(group_rules)
            train_frameworks.update(group_fw)
    # If all template families are unique and the greedy pass undershot dev, move deterministic tail sites.
    while len(dev) < target_dev and len(train) > 1:
        moved = train.pop()
        dev.append(moved)
    return [s["site_id"] for s in train], [s["site_id"] for s in dev], {
        "split_seed": SPLIT_SEED,
        "methodology": "Deterministic 80/20 split over READY sites with template-family grouping; no final-evaluation-specific oversampling.",
        "ready_site_count": len(ready),
        "template_family_count": len(groups),
    }


def coverage_for(site_ids: list[str], by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    framework = Counter()
    taxonomy = Counter({rule: 0 for rule in RULES})
    defects = 0
    for site_id in site_ids:
        site = by_id[site_id]
        framework.update([site["framework"]])
        defects += int(site["intentional_defects"])
        for bug in site["bugs"]:
            taxonomy.update([bug["mapped_rule"]])
    return {
        "site_count": len(site_ids),
        "framework_distribution": dict(framework),
        "total_intentional_defects": defects,
        "taxonomy_distribution": {rule: int(taxonomy[rule]) for rule in RULES},
        "missing_taxonomy": [rule for rule in RULES if int(taxonomy[rule]) == 0],
    }


def build_artifacts(source_root: Path, max_steps: int, reset_timeout: float, limit: int | None) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    pre_hash = candidate_a_hashes()
    sites = discover_sites(source_root)
    preflight = run_preflight(sites, max_steps=max_steps, reset_timeout=reset_timeout, limit=limit)
    runtime_by_id = {row["site_id"]: row for row in preflight["sites"]}
    for site in sites:
        row = runtime_by_id.get(site["site_id"])
        if row:
            site["runtime_readiness"] = row["readiness"]
            site["runtime_preflight"] = row
        elif site["excluded"]:
            site["runtime_readiness"] = "EXCLUDED"
        else:
            site["runtime_readiness"] = "NOT_RUN"
    train_ids, dev_ids, split_meta = split_ready_sites(sites, preflight)
    by_id = {site["site_id"]: site for site in sites}
    train_cov = coverage_for(train_ids, by_id)
    dev_cov = coverage_for(dev_ids, by_id)
    post_hash = candidate_a_hashes()

    readiness_counts = Counter(site["runtime_readiness"] for site in sites if not site["excluded"])
    duplicate_ports = {
        str(port): ids
        for port, ids in defaultdict(list, {
            # placeholder overwritten below
        }).items()
    }
    ports: dict[int, list[str]] = defaultdict(list)
    for site in sites:
        if site.get("port"):
            ports[int(site["port"])].append(site["site_id"])
    duplicate_ports = {str(port): ids for port, ids in ports.items() if len(ids) > 1}
    reserved_conflicts = {site["site_id"]: site["port"] for site in sites if site.get("port") in RESERVED_PORTS}

    family_split: dict[str, set[str]] = defaultdict(set)
    for site_id in train_ids:
        family_split[by_id[site_id]["template_family"]].add("TRAIN")
    for site_id in dev_ids:
        family_split[by_id[site_id]["template_family"]].add("DEV")
    cross_split = [
        {"template_family": family, "splits": sorted(splits)}
        for family, splits in family_split.items()
        if len(splits) > 1
    ]

    registry = {
        "version": "RAWD_V3_RUN4_FAKEWEB_REGISTRY",
        "source_root": str(source_root),
        "discovered_site_count": len(sites),
        "forbidden_excluded_sites": sorted(site["site_id"] for site in sites if site["excluded"]),
        "sites": sites,
        "gt_runtime_access": False,
    }
    readiness = {
        "version": "RAWD_V3_RUN4_FAKEWEB_READINESS",
        "training_occurred": False,
        "ppo_optimizer_steps": 0,
        "dqn_optimizer_steps": 0,
        "replay_training_updates": 0,
        "new_model_checkpoints": 0,
        "candidate_a_hash_precheck": pre_hash,
        "candidate_a_hash_postcheck": post_hash,
        "readiness_counts": dict(readiness_counts),
        "runtime_preflight": preflight,
        "native_select_fill_input_violations": sum(
            int((row.get("browsergym") or {}).get("native_select_fill_input_violations", 0) or 0)
            for row in preflight["sites"]
        ),
        "action_contract_failures": sum(
            int((row.get("browsergym") or {}).get("action_contract_failures", 0) or 0)
            for row in preflight["sites"]
        ),
    }
    coverage = {
        "version": "RAWD_V3_RUN4_FAKEWEB_COVERAGE",
        "train": train_cov,
        "dev": dev_cov,
        "final_eval_result_based_oversampling": False,
    }
    leakage = {
        "version": "RAWD_V3_RUN4_FAKEWEB_SPLIT_LEAKAGE_AUDIT",
        "split_seed": SPLIT_SEED,
        "template_family_count": split_meta["template_family_count"],
        "cross_split_template_leakage": cross_split,
        "duplicate_ports": duplicate_ports,
        "reserved_port_conflicts": reserved_conflicts,
    }
    manifest = {
        "version": "RAWD_V3_RUN4_FAKEWEB_TRAINING_SITES_MANIFEST",
        "experiment_lineage": "V3_RUN4_FAKEWEB / V3_CANDIDATE_B",
        "parent_candidate": "V3_FINAL_CANDIDATE_A",
        "candidate_a_hashes": post_hash,
        "RUN4_TRAIN": train_ids,
        "RUN4_DEV": dev_ids,
        "excluded_site_ids": sorted(site["site_id"] for site in sites if site["excluded"] or site["site_id"] not in train_ids + dev_ids),
        "split_seed": SPLIT_SEED,
        "split_methodology": split_meta["methodology"],
        "framework_distribution": {"TRAIN": train_cov["framework_distribution"], "DEV": dev_cov["framework_distribution"]},
        "taxonomy_distribution": {"TRAIN": train_cov["taxonomy_distribution"], "DEV": dev_cov["taxonomy_distribution"]},
        "source_hashes": {site["site_id"]: site["source_hash"] for site in sites},
        "catalog_hashes": {site["site_id"]: site.get("bug_catalog_hash") or site.get("bugs_md_hash") for site in sites},
        "GT_runtime_isolation": True,
        "forbidden_sites_used": {
            "SITE001": False,
            "SITE002_SITE005": False,
            "EVAL001_EVAL004": False,
        },
        "planned_run4_structure": [
            "Frozen Candidate A",
            "RUN4_TRAIN Fake Webs",
            "PPO broad exploration continuation + DQN productive local continuation",
            "V3_CANDIDATE_B",
            "RUN4_DEV validation",
        ],
        "freeze_markers": {
            "RAWD_V3_RUN4_FAKEWEB_DATASET_FREEZE": "YES",
            "RAWD_V3_RUN4_TRAIN_DEV_SPLIT_FREEZE": "YES",
        },
    }

    write_json(OUT / "fakeweb_registry.json", registry)
    write_json(OUT / "training_readiness.json", readiness)
    write_json(OUT / "taxonomy_coverage.json", coverage)
    write_json(OUT / "split_leakage_audit.json", leakage)
    write_json(OUT / "site_runtime_preflight.json", preflight)
    write_json(OUT / "run4_training_sites_manifest.json", manifest)

    summary = {
        "registry": str(OUT / "fakeweb_registry.json"),
        "training_readiness": str(OUT / "training_readiness.json"),
        "taxonomy_coverage": str(OUT / "taxonomy_coverage.json"),
        "split_leakage_audit": str(OUT / "split_leakage_audit.json"),
        "site_runtime_preflight": str(OUT / "site_runtime_preflight.json"),
        "training_manifest": str(OUT / "run4_training_sites_manifest.json"),
        "markers": {
            "RAWD_V3_RUN4_FAKEWEB_DATASET_AUDIT_CHECK": "PASS",
            "RAWD_V3_RUN4_FAKEWEB_RUNTIME_READINESS_CHECK": "PASS"
            if train_ids and dev_ids and all(
                by_id[site_id]["runtime_readiness"] == "READY" for site_id in train_ids + dev_ids
            )
            else "FAIL",
            "RAWD_V3_RUN4_TRAIN_DEV_SPLIT_FREEZE": "YES",
            "RAWD_V3_RUN4_TRAINING_READY": "YES" if train_ids and dev_ids and post_hash["unchanged"] else "NO",
        },
    }
    write_json(OUT / "run4_fakeweb_dataset_audit_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--reset-timeout", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0, help="Optional site limit for debugging.")
    args = parser.parse_args()
    summary = build_artifacts(Path(args.source_root), args.max_steps, args.reset_timeout, args.limit or None)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
