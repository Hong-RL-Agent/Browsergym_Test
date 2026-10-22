"""Prepare and audit the Step-7 site-level split without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SITE_CONFIG = ROOT.parent / "configs/training_sites.json"
DEFAULT_SPLIT = ROOT.parent / "configs/step7_site_split.json"
DEFAULT_OUT = ROOT / "artifacts/training_v3/step7_fresh_training"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def port_available(base_url: str, timeout: float = 0.5) -> bool:
    parsed = urlparse(base_url)
    if not parsed.hostname or not parsed.port:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-config", default=str(DEFAULT_SITE_CONFIG))
    parser.add_argument("--split-config", default=str(DEFAULT_SPLIT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    site_config = load(Path(args.site_config))
    split = load(Path(args.split_config))
    sites = {str(item["site_id"]): item for item in site_config.get("sites", [])}
    groups = {key: list(split.get(key, [])) for key in ("train", "validation", "test")}
    all_ids = [site_id for values in groups.values() for site_id in values]
    duplicate_ids = sorted({site_id for site_id in all_ids if all_ids.count(site_id) > 1})
    missing_ids = sorted(set(all_ids) - set(sites))
    inventory = []
    for site_id in sorted(set(all_ids)):
        spec = sites.get(site_id, {})
        catalog_path = ROOT / str(spec.get("bug_catalog_path", ""))
        inventory.append({
            "site_id": site_id,
            "base_url": spec.get("base_url"),
            "configured": site_id in sites,
            "available": port_available(str(spec.get("base_url", ""))),
            "frontend_backend_start_command": spec.get("start_command"),
            "bug_catalog_available": catalog_path.is_file(),
            "training_permitted": spec.get("training_permitted", True),
            "evaluation_permitted": spec.get("evaluation_permitted", True),
            "split_membership": [key for key, values in groups.items() if site_id in values],
        })
    overlap = {
        "train_validation": len(set(groups["train"]) & set(groups["validation"])),
        "train_test": len(set(groups["train"]) & set(groups["test"])),
        "validation_test": len(set(groups["validation"]) & set(groups["test"])),
    }
    nonempty = all(bool(groups[key]) for key in ("train", "validation", "test"))
    structural_pass = nonempty and not duplicate_ids and not missing_ids and all(value == 0 for value in overlap.values())
    runtime_ready = structural_pass and all(item["available"] for item in inventory if item["site_id"] in set(all_ids))
    result = {
        "split_version": split.get("split_version"),
        "split_type": split.get("split_type"),
        "source_config": str(Path(args.site_config).resolve()),
        "train": groups["train"],
        "validation": groups["validation"],
        "test": groups["test"],
        "overlap": overlap,
        "duplicate_site_ids": duplicate_ids,
        "missing_site_ids": missing_ids,
        "nonempty_site_level_split": structural_pass,
        "runtime_sites_available": runtime_ready,
        "inventory": inventory,
        "training_started": False,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "site_inventory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "split_manifest.json").write_text(json.dumps({
        "split_version": split.get("split_version"),
        "split_type": split.get("split_type"),
        "train": groups["train"],
        "validation": groups["validation"],
        "test": groups["test"],
        "overlap": overlap,
    }, indent=2), encoding="utf-8")
    (out / "split_leakage_audit.json").write_text(json.dumps({
        "overlap": overlap,
        "duplicate_site_ids": duplicate_ids,
        "missing_site_ids": missing_ids,
        "nonempty_site_level_split": structural_pass,
        "runtime_sites_available": runtime_ready,
    }, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if structural_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
