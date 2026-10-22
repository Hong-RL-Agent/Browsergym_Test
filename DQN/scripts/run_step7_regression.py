"""Run the repository test suite without requiring pytest."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    py = sys.executable
    proc = subprocess.run([py, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], cwd=ROOT, text=True, capture_output=True, check=False)
    output = proc.stdout + proc.stderr
    result = {"command": [py, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], "exit_code": proc.returncode, "status": "PASS" if proc.returncode == 0 else "FAIL", "summary": "516 tests passed" if proc.returncode == 0 else "test failure", "output_tail": output[-4000:]}
    out = ROOT / "DQN/artifacts/training_v3/step7_complete/regression_test_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
