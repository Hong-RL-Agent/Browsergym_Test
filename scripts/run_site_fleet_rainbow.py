"""Train every configured site sequentially with Rainbow DQN and isolated artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generated_local_sites.json")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/fleet/rainbow_summary.json")
    parser.add_argument("--sites-root", default="../RL_Errorsite-frontend-errorsite")
    parser.add_argument("--start-site", type=int, default=2)
    parser.add_argument("--end-site", type=int, default=90)
    parser.add_argument("--site-timeout", type=int, default=900)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(output.read_text(encoding="utf-8")) if args.resume and output.exists() else {}
    results_by_site = {item["site_id"]: item for item in previous.get("results", [])}
    for site in config.get("sites", []):
        site_id = str(site.get("site_id") or "")
        try:
            site_number = int(site_id.removeprefix("site"))
        except ValueError:
            continue
        if not args.start_site <= site_number <= args.end_site:
            continue
        if args.resume and results_by_site.get(site_id, {}).get("returncode") == 0:
            print(f"[fleet] {site_id} already completed; skipping", flush=True)
            continue
        base_url = str(site.get("base_url") or "")
        model_path = Path("artifacts/models/fleet") / f"{site_id}_browsergym_rainbow_dqn.pt"
        command = [
            sys.executable, "runners/train_browsergym_agent.py",
            "--site-id", site_id, "--base-url", base_url,
            "--episodes", str(args.episodes), "--max-steps", str(args.max_steps),
            "--seed", str(args.seed), "--headless", "true",
            "--algorithm", "rainbow-dqn", "--save-model", str(model_path),
        ]
        if model_path.exists():
            command.extend(["--load-model", str(model_path)])
        site_dirs = sorted(Path(args.sites_root).glob(f"{site_id}*"))
        server = None
        completed = None
        if site_dirs and (site_dirs[0] / "server.js").exists():
            env = os.environ.copy()
            dependency_roots = sorted(Path(args.sites_root).glob("site*/node_modules"))
            if dependency_roots:
                env["NODE_PATH"] = str(dependency_roots[0].resolve())
            server = subprocess.Popen(
                ["node", "server.js"], cwd=site_dirs[0], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        try:
            if not _wait_until_ready(base_url, server):
                raise RuntimeError(f"site server did not become ready: {base_url}")
            completed = subprocess.run(command, text=True, capture_output=True, timeout=args.site_timeout)
        except Exception as exc:
            completed = subprocess.CompletedProcess(command, 124, "", str(exc))
        finally:
            if server and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
        summary_path = Path("artifacts/browsergym") / site_id / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        results_by_site[site_id] = {
            "site_id": site_id,
            "base_url": base_url,
            "returncode": completed.returncode,
            "recall": summary.get("recall"),
            "precision": summary.get("precision"),
            "failure_type_counts": summary.get("failure_type_counts", {}),
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
            "model_path": str(model_path),
        }
        results = [results_by_site[key] for key in sorted(results_by_site)]
        output.write_text(json.dumps({"site_count": len(results), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[fleet] {site_id} returncode={completed.returncode} recall={summary.get('recall')}", flush=True)
    results = [results_by_site[key] for key in sorted(results_by_site)]
    output.write_text(json.dumps({"site_count": len(results), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"fleet_summary={output}")
    return 0 if all(item["returncode"] == 0 for item in results) else 1


def _wait_until_ready(base_url: str, server: subprocess.Popen, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    health_url = base_url.rstrip("/") + "/api/health"
    while time.monotonic() < deadline:
        if server and server.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status < 500:
                    return True
        except Exception:
            try:
                with urllib.request.urlopen(base_url, timeout=1.0) as response:
                    if response.status < 500:
                        return True
            except Exception:
                time.sleep(0.5)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
