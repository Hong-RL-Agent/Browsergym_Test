"""Publish a small, continuously updated status file for the fleet pipeline."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("artifacts/fleet")
STATUS = ROOT / "pipeline_status.json"


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        phase2 = _read(ROOT / "sites002-090-phase2.json")
        evaluation = _read(ROOT / "unguided_evaluation.json")
        promotion = _read(ROOT / "model_promotion_report.json")
        phase2_count = int(phase2.get("site_count", 0) or 0)
        evaluation_count = int(evaluation.get("site_count", 0) or 0)
        if promotion:
            payload = _status("pipeline_complete", "완료", 89, 89, "completed")
            payload["accepted_models"] = promotion.get("accepted_count", 0)
            payload["quarantined_models"] = promotion.get("quarantined_count", 0)
            _write(payload)
            return 0
        if evaluation_count:
            payload = _status(
                "unguided_evaluation", "독립 평가 진행 중", evaluation_count, 89,
                "model_promotion_and_final_report",
            )
        elif phase2_count >= 89:
            payload = _status(
                "failed_site_retry", "실패 사이트 복구 진행 중", phase2_count, 89,
                "unguided_evaluation",
            )
        else:
            payload = _status(
                "phase2_fleet_training", "2차 학습 진행 중", phase2_count, 89,
                "failed_site_retry_and_unguided_evaluation",
            )
        _write(payload)
        time.sleep(15)


def _status(phase: str, label: str, completed: int, total: int, next_step: str) -> dict:
    return {
        "phase": phase,
        "status": "RUNNING" if phase != "pipeline_complete" else "COMPLETED",
        "status_ko": label,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "completed_sites": completed,
        "total_sites": total,
        "progress_percent": round(completed / max(1, total) * 100, 1),
        "next_step": next_step,
        "automatic_continuation": True,
    }


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write(payload: dict) -> None:
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


if __name__ == "__main__":
    raise SystemExit(main())
