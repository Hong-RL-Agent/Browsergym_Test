from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description="Run J.A.W.S API fuzzing over active localhost port batches.")
    parser.add_argument("--start-port", type=int, default=9110)
    parser.add_argument("--end-port", type=int, default=9219)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-cases-per-endpoint", type=int, default=16)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--base-artifact-dir", default="artifacts")
    parser.add_argument("--template", default="")
    args = parser.parse_args()

    failed_ports = []
    batch_results = []
    for batch_start in range(args.start_port, args.end_port + 1, args.batch_size):
        batch_end = min(args.end_port, batch_start + args.batch_size - 1)
        active_ports = [port for port in range(batch_start, batch_end + 1) if _is_active(port, args.timeout_ms)]
        inactive_ports = [port for port in range(batch_start, batch_end + 1) if port not in set(active_ports)]
        failed_ports.extend({"port": port, "reason": "inactive"} for port in inactive_ports)
        if not active_ports:
            continue
        config_path = Path("configs/generated") / f"api_port_sites_{batch_start}_{batch_end}.json"
        run_id = f"api_ports_{batch_start}_{batch_end}"
        output_path = Path(args.base_artifact_dir) / "evaluations" / run_id / "result.json"
        gen_cmd = [
            sys.executable,
            "scripts/generate_api_port_sites_config.py",
            "--start-port",
            str(min(active_ports)),
            "--end-port",
            str(max(active_ports)),
            "--output",
            str(config_path),
        ]
        if args.template:
            gen_cmd.extend(["--template", args.template])
        eval_cmd = [
            sys.executable,
            "runners/run_api_fuzz_eval.py",
            "--config",
            str(config_path),
            "--start-port",
            str(min(active_ports)),
            "--end-port",
            str(max(active_ports)),
            "--output",
            str(output_path),
            "--max-cases-per-endpoint",
            str(args.max_cases_per_endpoint),
            "--timeout-ms",
            str(args.timeout_ms),
            "--run-id",
            run_id,
        ]
        for command in (gen_cmd, eval_cmd):
            completed = _run(command)
            if completed.returncode != 0:
                failed_ports.extend({"port": port, "reason": f"command_failed:{completed.returncode}"} for port in active_ports)
                break
        if output_path.exists():
            batch_results.append(str(output_path))

    final_dir = Path(args.base_artifact_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    failed_ports_path = final_dir / "api_failed_ports.json"
    failed_ports_path.write_text(json.dumps(failed_ports, ensure_ascii=False, indent=2), encoding="utf-8")
    merged_path = final_dir / f"api_ports_{args.start_port}_{args.end_port}_summary.json"
    merge_cmd = [
        sys.executable,
        "scripts/merge_api_results.py",
        "--input-dir",
        str(Path(args.base_artifact_dir) / "evaluations"),
        "--output",
        str(merged_path),
    ]
    merge_result = _run(merge_cmd)
    summary = {
        "run_id": f"api_ports_{args.start_port}_{args.end_port}",
        "batch_result_count": len(batch_results),
        "failed_ports": str(failed_ports_path),
        "merged_summary": str(merged_path),
        "merge_returncode": merge_result.returncode,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return merge_result.returncode


def _is_active(port: int, timeout_ms: int) -> bool:
    request = Request(f"http://localhost:{port}/", headers={"User-Agent": "jaws-api-preflight/1.0"})
    try:
        with urlopen(request, timeout=max(0.001, timeout_ms / 1000.0)):
            return True
    except HTTPError:
        return True
    except (URLError, TimeoutError, socket.timeout):
        return False
    except Exception:
        return False


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"[run] {' '.join(command)}", flush=True)
    return subprocess.run(command, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
