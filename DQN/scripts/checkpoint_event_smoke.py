"""Isolated checkpoint/event contract smoke test for Step 7."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from run_ppo_full_training_v1 import append_event, save_checkpoint


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rawd_checkpoint_smoke_") as tmp:
        root = Path(tmp)
        checkpoint = root / "update_001.pt"
        events = root / "progress.jsonl"
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        payload = {
            "metadata": {
                "run_id": "checkpoint_event_smoke",
                "architecture": "gru",
                "seed": 43,
                "completed_update": 1,
                "training_contract_version": "rawd_ppo_full_training_v1",
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        hashes = save_checkpoint(payload, checkpoint)
        append_event(events, {
            "event": "CHECKPOINT_SAVED",
            "run_id": "checkpoint_event_smoke",
            "architecture": "gru",
            "seed": 43,
            "completed_update": 1,
            "checkpoint_path": str(checkpoint),
            **hashes,
            "atomic_save_success": True,
        })
        rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
        reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result = {
            "status": "PASS" if (
                checkpoint.exists()
                and len(rows) == 1
                and rows[0]["event"] == "CHECKPOINT_SAVED"
                and rows[0]["checkpoint_path"] == str(checkpoint)
                and rows[0]["checkpoint_file_sha256"] == hashes["checkpoint_file_sha256"]
                and rows[0]["checkpoint_metadata_sha256"] == hashes["checkpoint_metadata_sha256"]
                and reloaded["metadata"]["checkpoint_metadata_sha256"] == hashes["checkpoint_metadata_sha256"]
            ) else "FAIL",
            "checkpoint_exists": checkpoint.exists(),
            "checkpoint_saved_event_count": sum(row.get("event") == "CHECKPOINT_SAVED" for row in rows),
            "path_match": rows[0].get("checkpoint_path") == str(checkpoint),
            "file_sha_match": rows[0].get("checkpoint_file_sha256") == hashes["checkpoint_file_sha256"],
            "metadata_sha_match": rows[0].get("checkpoint_metadata_sha256") == hashes["checkpoint_metadata_sha256"],
            "clean_reload": bool(reloaded.get("model_state_dict")) and bool(reloaded.get("optimizer_state_dict")),
        }
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
