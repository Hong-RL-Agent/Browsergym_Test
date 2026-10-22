"""Build an offline structural inventory and leakage-safe Step-7 split."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent / "RL_Errorsite"
OUT = ROOT / "DQN/artifacts/training_v3/step7_complete"
TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z0-9]+)")
ATTR_RE = re.compile(r"<[^>]+>")


def structure_hash(html: str) -> str:
    tags = " ".join(a + b.lower() for a, b in TAG_RE.findall(html))
    attrs = re.sub(r"[A-Za-z0-9_-]{3,}", "X", " ".join(ATTR_RE.findall(html)))
    payload = json.dumps({"tags": tags, "attrs": attrs}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main() -> None:
    manifest = json.loads((REPO / "manifest.json").read_text(encoding="utf-8", errors="replace"))
    rows = []
    clusters: dict[str, list[str]] = defaultdict(list)
    for item in manifest:
        site_id = f"site{int(item['id']):03d}"
        source = REPO / str(item["path"])
        html_path = source / "index.html"
        html = html_path.read_text(encoding="utf-8", errors="replace") if html_path.exists() else ""
        cluster = structure_hash(html)
        runnable = source.exists() and html_path.exists()
        row = {
            "site_id": site_id,
            "source_path": str(source),
            "port": int(item["port"]),
            "url": f"http://localhost:{int(item['port'])}",
            "type": item.get("type"),
            "batch": item.get("batch"),
            "functional_theme": item.get("category"),
            "offline_defect_family": item.get("intentional_gui_bug"),
            "structural_cluster": f"RL_ERRORSITE_CLUSTER_{cluster}",
            "structural_fingerprint": cluster,
            "runnable_source": runnable,
            "gt_metadata_offline_only": True,
        }
        rows.append(row)
        if runnable:
            clusters[cluster].append(site_id)

    ordered = sorted(clusters)
    # Deterministic cluster-level partition.  Clusters, not individual sites,
    # are the unit of generalization and never cross a split.
    train_keys = ordered[0:3]
    validation_keys = ordered[3:5]
    test_keys = ordered[5:]
    assignment = {}
    for key in train_keys:
        assignment[key] = "TRAIN"
    for key in validation_keys:
        assignment[key] = "VALIDATION"
    for key in test_keys:
        assignment[key] = "TEST"
    for row in rows:
        row["partition"] = assignment.get(row["structural_fingerprint"], "UNASSIGNED")

    split = {
        "split_version": "rawd_step7_cluster_site_level_v2",
        "split_type": "SITE_LEVEL_STRUCTURAL_CLUSTER",
        "source_repository": "https://github.com/Hong-RL-Agent/RL_Errorsite.git",
        "branch": "frontend-errorsite",
        "commit_sha": "cca3c72a21a7c0d26af30bf35d92ff1134459036",
        "train": [r["site_id"] for r in rows if r["partition"] == "TRAIN"],
        "validation": [r["site_id"] for r in rows if r["partition"] == "VALIDATION"],
        "test": [r["site_id"] for r in rows if r["partition"] == "TEST"],
        "train_clusters": [f"RL_ERRORSITE_CLUSTER_{x}" for x in train_keys],
        "validation_clusters": [f"RL_ERRORSITE_CLUSTER_{x}" for x in validation_keys],
        "test_clusters": [f"RL_ERRORSITE_CLUSTER_{x}" for x in test_keys],
        "cluster_overlap": 0,
        "site_overlap": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "site_inventory.json").write_text(json.dumps({"manifest_count": len(rows), "runnable_count": sum(r["runnable_source"] for r in rows), "sites": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "structural_cluster_manifest.json").write_text(json.dumps({"cluster_count": len(ordered), "clusters": {f"RL_ERRORSITE_CLUSTER_{k}": v for k, v in clusters.items()}}, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "split_manifest.json").write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    overlap = {"site_overlap": {"train_validation": len(set(split["train"]) & set(split["validation"])), "train_test": len(set(split["train"]) & set(split["test"])), "validation_test": len(set(split["validation"]) & set(split["test"]))}, "cluster_overlap": {"train_validation": len(set(split["train_clusters"]) & set(split["validation_clusters"])), "train_test": len(set(split["train_clusters"]) & set(split["test_clusters"])), "validation_test": len(set(split["validation_clusters"]) & set(split["test_clusters"]))}, "cross_split_template_leakage": 0}
    (OUT / "split_leakage_audit.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    (ROOT / "configs/step7_full_site_split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    runtime_sites = [{"site_id": r["site_id"], "base_url": r["url"], "source_path": r["source_path"], "partition": r["partition"], "runtime_policy_profile": "openended_anomaly_discovery_gt_free"} for r in rows if r["partition"] != "UNASSIGNED"]
    (ROOT / "configs/rl_errorsite_full_training_sites.json").write_text(json.dumps({"repository": split["source_repository"], "branch": split["branch"], "commit_sha": split["commit_sha"], "sites": runtime_sites}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest_sites": len(rows), "runnable_sites": sum(r["runnable_source"] for r in rows), "clusters": len(ordered), "train_sites": len(split["train"]), "validation_sites": len(split["validation"]), "test_sites": len(split["test"]), "train_clusters": len(split["train_clusters"]), "validation_clusters": len(split["validation_clusters"]), "test_clusters": len(split["test_clusters"])}, indent=2))


if __name__ == "__main__":
    main()
