"""Run equal-budget baseline/fault-discovery jobs and compare outcomes."""
from __future__ import annotations
import argparse, json, time
from urllib.request import Request, urlopen

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=25)
    args = parser.parse_args()
    results = {}
    for label, mode in (("baseline", "coverage"), ("candidate", "fault-discovery")):
        job = _json(args.api_url + "/explorations", "POST", {"targetUrl": args.target_url, "mode": mode, "episodes": args.episodes, "maxSteps": args.max_steps})
        job_id = job["jobId"]
        while True:
            status = _json(f"{args.api_url}/explorations/{job_id}")
            if status["status"] == "completed": break
            if status["status"] in {"failed", "cancelled"}: raise RuntimeError(str(status))
            time.sleep(1)
        results[label] = _json(f"{args.api_url}/explorations/{job_id}/result")
    base, candidate = results["baseline"], results["candidate"]
    comparison = {"target_url": args.target_url, "equal_budget": {"episodes": args.episodes, "max_steps": args.max_steps}, "baseline": _metrics(base), "candidate": _metrics(candidate)}
    comparison["delta"] = {key: round(comparison["candidate"][key] - comparison["baseline"][key], 4) for key in comparison["baseline"]}
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0

def _metrics(result):
    return {"coverage_score": float(result["coverage"]["coverage_score"]), "finding_count": float(len(result["findings"])), "error_yield": float(result["metrics"]["error_yield"]), "exploration_efficiency": float(result["metrics"]["exploration_efficiency"])}

def _json(url, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response: return json.loads(response.read())

if __name__ == "__main__": raise SystemExit(main())

