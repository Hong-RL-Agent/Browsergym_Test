"""Evaluate site-specific Rainbow checkpoints without heuristic action overrides."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_site_fleet_rainbow import _wait_until_ready


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generated_local_sites.json")
    parser.add_argument("--sites-root", default="../RL_Errorsite-frontend-errorsite")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--output", default="artifacts/fleet/unguided_evaluation.json")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(output.read_text(encoding="utf-8")) if args.resume and output.exists() else {}
    results = {item["site_id"]: item for item in previous.get("results", [])}
    dependency_root = Path(args.sites_root) / "node_modules"

    for site in config.get("sites", []):
        site_id = str(site.get("site_id") or "")
        if site_id == "site001" or not site_id.startswith("site"):
            continue
        if args.resume and results.get(site_id, {}).get("returncode") == 0:
            continue
        model = Path("artifacts/models/fleet") / f"{site_id}_browsergym_rainbow_dqn.pt"
        if not model.exists():
            results[site_id] = {"site_id": site_id, "returncode": 2, "error": "missing_model"}
            _write(output, results)
            continue
        site_dirs = sorted(Path(args.sites_root).glob(f"{site_id}*"))
        server = None
        completed = None
        result_path = Path("artifacts/evaluations/fleet") / f"{site_id}.json"
        try:
            env = os.environ.copy()
            if dependency_root.exists():
                env["NODE_PATH"] = str(dependency_root.resolve())
            server = subprocess.Popen(
                ["node", "server.js"], cwd=site_dirs[0], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if not _wait_until_ready(str(site["base_url"]), server):
                raise RuntimeError("server_not_ready")
            command = [
                sys.executable, "runners/evaluate_browsergym_agent.py",
                "--site-id", site_id, "--base-url", str(site["base_url"]),
                "--model-path", str(model), "--algorithm", "rainbow-dqn",
                "--episodes", str(args.episodes), "--max-steps", str(args.max_steps),
                "--seed", str(args.seed), "--guided-actions", "false",
                "--output", str(result_path),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, timeout=420)
            payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            results[site_id] = {"site_id": site_id, "returncode": completed.returncode, **payload,
                                "stderr_tail": completed.stderr[-800:]}
        except Exception as exc:
            results[site_id] = {"site_id": site_id, "returncode": 124, "error": str(exc)}
        finally:
            if server and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
        _write(output, results)
        print(f"[eval-fleet] {site_id} returncode={results[site_id]['returncode']} recall={results[site_id].get('recall')}", flush=True)
    return 0


def _write(path: Path, results: dict[str, dict]) -> None:
    ordered = [results[key] for key in sorted(results)]
    path.write_text(json.dumps({"site_count": len(ordered), "results": ordered}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
