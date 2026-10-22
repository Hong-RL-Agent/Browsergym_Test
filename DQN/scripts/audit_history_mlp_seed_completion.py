from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"


def audit(seed: int) -> dict:
    prefix = f"history_mlp_seed{seed}"
    progress_path = OUT / f"{prefix}_progress.jsonl"
    checkpoint_dir = OUT / f"{prefix}_checkpoints"
    metadata_path = OUT / f"{prefix}_metadata.json"
    events = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    updates = [e for e in events if e.get("event") == "UPDATE_COMPLETE"]
    episodes = [e for e in events if e.get("event") == "EPISODE_COMPLETE"]
    heartbeats = [e for e in events if e.get("event") == "PPO_HEARTBEAT"]
    interruptions = [e for e in events if e.get("event") in {"RUN_INTERRUPTED", "EPISODE_FAILED"}]
    checkpoints = sorted(checkpoint_dir.glob("update_*.pt"))
    numbers = [int(re.search(r"update_(\d+)", p.name).group(1)) for p in checkpoints]
    final = checkpoint_dir / "update_024.pt"
    payload = torch.load(final, map_location="cpu", weights_only=False)
    checkpoint_meta = payload.get("metadata", {})
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    final_sha = hashlib.sha256(final.read_bytes()).hexdigest().upper()
    clean_reload = (
        checkpoint_meta.get("completed_update") == 24
        and checkpoint_meta.get("architecture") == "history_mlp"
        and checkpoint_meta.get("seed") == seed
        and bool(payload.get("model_state_dict"))
        and bool(payload.get("optimizer_state_dict"))
    )
    update_numbers = sorted(int(e["update"]) for e in updates)
    raw = {
        "run_start": sum(e.get("event") == "RUN_START" for e in events),
        "update_start": sum(e.get("event") == "UPDATE_START" for e in events),
        "update_complete": len(updates),
        "episode_complete": len(episodes),
        "heartbeat": len(heartbeats),
        "interruptions": len(interruptions),
        "environment_steps": sum(int(e.get("steps_completed", e.get("steps", 0))) for e in episodes),
        "optimizer_steps": sum(int(e.get("optimizer_steps", 0)) for e in updates),
        "completed_updates": update_numbers,
        "checkpoint_numbers": numbers,
    }
    expected = list(range(1, 25))
    summary_match = (
        raw["episode_complete"] == metadata.get("episodes")
        and raw["environment_steps"] == metadata.get("steps")
        and raw["optimizer_steps"] == metadata.get("optimizer_steps")
    )
    result = {
        "status": "PASS" if raw["update_complete"] == 24 and update_numbers == expected and numbers == expected and clean_reload and summary_match else "FAIL",
        "run_id": f"history_mlp_{seed}",
        "architecture": "history_mlp",
        "seed": seed,
        "raw": raw,
        "summary": {k: metadata.get(k) for k in ["episodes", "steps", "optimizer_steps", "runtime_errors", "gt_runtime_access", "legacy_initialization", "smoke_only"]},
        "mismatches": {
            "raw_summary": 0 if summary_match else 1,
            "checkpoint_sequence": 0 if numbers == expected and update_numbers == expected else 1,
            "checkpoint_metadata": 0 if clean_reload else 1,
        },
        "heartbeat": {
            "timestamps_monotonic": all(heartbeats[i]["timestamp"] <= heartbeats[i + 1]["timestamp"] for i in range(len(heartbeats) - 1)),
            "constant_run": len({e.get("run_id") for e in heartbeats}) == 1,
            "last_completed_non_decreasing": all(int(heartbeats[i].get("last_completed_update", 0)) <= int(heartbeats[i + 1].get("last_completed_update", 0)) for i in range(len(heartbeats) - 1)),
        },
        "safety": {
            "gt_runtime_access": metadata.get("gt_runtime_access"),
            "legacy_initialization": metadata.get("legacy_initialization"),
            "smoke_only": metadata.get("smoke_only"),
            "observation_dim": metadata.get("observation_schema", {}).get("schema_dim"),
            "action_dim": metadata.get("action_dim"),
        },
        "final_checkpoint": str(final),
        "final_sha256": final_sha,
        "clean_reload": clean_reload,
        "production_selected": False,
    }
    (OUT / f"{prefix}_raw_recomputation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / f"{prefix}_checkpoint_audit.json").write_text(json.dumps({"status": result["status"], "count": len(checkpoints), "sequence": numbers, "final_sha256": final_sha, "clean_reload": clean_reload}, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    results = [audit(seed) for seed in args.seeds]
    aggregate = {"status": "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL", "runs": results}
    (OUT / "history_mlp_all_seeds_audit.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return 0 if aggregate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
