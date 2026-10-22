from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"


def main() -> int:
    progress_path = OUT / "gru_seed42_progress.jsonl"
    events = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    updates = [e for e in events if e.get("event") == "UPDATE_COMPLETE"]
    episodes = [e for e in events if e.get("event") == "EPISODE_COMPLETE"]
    heartbeats = [e for e in events if e.get("event") == "PPO_HEARTBEAT"]
    checkpoints = sorted((OUT / "gru_seed42_checkpoints").glob("update_*.pt"))
    numbers = [int(re.search(r"update_(\d+)", p.name).group(1)) for p in checkpoints]
    final = OUT / "gru_seed42_checkpoints/update_024.pt"
    payload = torch.load(final, map_location="cpu", weights_only=False)
    meta = payload.get("metadata", {})
    progress_rows = [json.loads(line) for line in (OUT / "gru_training.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    telemetry_fields = {key for row in progress_rows for key in row}
    required_telemetry = {"hidden_norm_mean", "hidden_norm_min", "hidden_norm_max", "hidden_reset_count", "padding_loss_contribution", "sequence_order_violation", "future_leakage"}
    missing_telemetry = sorted(required_telemetry - telemetry_fields)
    result = {
        "status": "FAIL" if missing_telemetry else "PASS",
        "run_id": "gru_42",
        "raw": {
            "update_complete": len(updates),
            "completed_updates": sorted(int(e["update"]) for e in updates),
            "episode_complete": len(episodes),
            "heartbeat": len(heartbeats),
            "environment_steps": sum(int(e.get("steps_completed", e.get("steps", 0))) for e in episodes),
            "optimizer_steps": sum(int(e.get("optimizer_steps", 0)) for e in updates),
            "interruptions": sum(e.get("event") in {"RUN_INTERRUPTED", "EPISODE_FAILED"} for e in events),
        },
        "checkpoint": {
            "count": len(checkpoints),
            "sequence": numbers,
            "expected_sequence": list(range(1, 25)),
            "architecture": meta.get("architecture"),
            "seed": meta.get("seed"),
            "completed_update": meta.get("completed_update"),
            "file_sha256": hashlib.sha256(final.read_bytes()).hexdigest().upper(),
            "clean_reload": bool(payload.get("model_state_dict")) and bool(payload.get("optimizer_state_dict")),
        },
        "recurrent_integrity": {
            "required_telemetry_present": not missing_telemetry,
            "missing_telemetry_fields": missing_telemetry,
            "sequence_order_violation": "UNRECORDED",
            "future_leakage": "UNRECORDED",
            "hidden_reset_failure": "UNRECORDED",
            "hidden_propagation_failure": "UNRECORDED",
            "padding_loss_contribution": "UNRECORDED",
        },
        "blocking_failure": "GRU_SEQUENCE_HIDDEN_TELEMETRY_UNRECORDED" if missing_telemetry else None,
    }
    (OUT / "gru_seed42_sequence_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "gru_seed42_hidden_state_audit.json").write_text(json.dumps(result["recurrent_integrity"], indent=2), encoding="utf-8")
    (OUT / "gru_seed42_raw_recomputation.json").write_text(json.dumps(result["raw"], indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
