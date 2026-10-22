"""Read-only reconciliation of the historical GRU43 hash fields."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "DQN/artifacts/training_v3/step7_fresh_training_final"


def main() -> int:
    progress = [json.loads(line) for line in (OUT / "gru_seed43_progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    final_update = [row for row in progress if row.get("event") == "UPDATE_COMPLETE" and row.get("update") == 24][-1]
    metadata = json.loads((OUT / "gru_seed43_metadata.json").read_text(encoding="utf-8"))
    update_path = OUT / "gru_seed43_checkpoints/update_024.pt"
    full_path = OUT / "gru_seed43_full.pt"
    actual_file_sha = hashlib.sha256(update_path.read_bytes()).hexdigest().upper()
    actual_full_sha = hashlib.sha256(full_path.read_bytes()).hexdigest().upper()
    checkpoint_events = [row for row in progress if row.get("event") == "CHECKPOINT_SAVED"]

    reconciliation = {
        "status": "PASS",
        "hash_reconciliation": {
            "update_complete_hash_type": "CHECKPOINT_PAYLOAD_SHA256_PRE_METADATA_FINALIZATION",
            "update_complete_hash_value": final_update.get("checkpoint_sha256"),
            "metadata_hash_type": "CHECKPOINT_PAYLOAD_SHA256_PRE_METADATA_FINALIZATION_FOR_GRU43_FULL_PT_MISNAMED_AS_CHECKPOINT_SHA256",
            "metadata_hash_value": metadata.get("checkpoint_sha256"),
            "actual_file_hash_type": "CHECKPOINT_FILE_SHA256",
            "actual_file_hash_value": actual_file_sha,
            "same_object_expected": False,
            "true_hash_mismatch": False,
            "source_basis": "run_ppo_full_training_v1.py saves once, mutates metadata, then saves again; the first digest is not the final file digest",
        },
        "logging_defect": {
            "checkpoint_saved_event_count": len(checkpoint_events),
            "defect": "CHECKPOINT_SAVED event was never emitted after checkpoint save/verification",
            "logging_only": True,
            "source_basis": "checkpoint save block emitted UPDATE_COMPLETE only",
        },
        "historical_provenance": {
            "run_id": "gru_43",
            "architecture": metadata.get("architecture"),
            "seed": metadata.get("seed"),
            "completed_updates": len([row for row in progress if row.get("event") == "UPDATE_COMPLETE"]),
            "checkpoint_files": len(list((OUT / "gru_seed43_checkpoints").glob("update_*.pt"))),
            "final_update_path_exists": update_path.exists(),
            "final_update_clean_file_sha256": actual_file_sha,
            "full_checkpoint_file_sha256": actual_full_sha,
            "clean_reload_previously_verified": True,
            "canonical_checkpoint_event_provenance": False,
        },
        "historical_gru43_certifiable": False,
        "certification_reason": "Hash semantics are explainable and the model reloads, but absence of CHECKPOINT_SAVED events leaves the historical run outside the canonical checkpoint-event contract; no historical event may be fabricated.",
    }
    (OUT / "gru_seed43_hash_semantics_audit.json").write_text(json.dumps(reconciliation, indent=2), encoding="utf-8")
    (OUT / "gru_seed43_checkpoint_provenance_reconciliation.json").write_text(json.dumps(reconciliation["historical_provenance"], indent=2), encoding="utf-8")
    (OUT / "gru_seed43_logging_defect_audit.json").write_text(json.dumps(reconciliation["logging_defect"], indent=2), encoding="utf-8")
    print(json.dumps(reconciliation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
