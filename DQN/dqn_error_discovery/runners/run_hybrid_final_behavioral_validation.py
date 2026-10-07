"""Build final Hybrid v1 behavioral validation and freeze artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_error_discovery.hybrid.final_behavioral_metrics import (
    build_final_behavioral_summary,
    final_freeze_manifest,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    hybrid = {seed: _load_json(artifact_dir / args.hybrid_template.format(seed=seed)) for seed in seeds}
    ppo = {seed: _load_json(artifact_dir / args.ppo_template.format(seed=seed)) for seed in seeds}
    summary = build_final_behavioral_summary(hybrid, ppo, controller_version=args.controller_version)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if summary.get("HYBRID_CONTROLLER_FREEZE") == "YES" and args.manifest_output:
        manifest = final_freeze_manifest(summary)
        manifest_path = Path(args.manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["final_manifest"] = str(manifest_path)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/sanity")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--hybrid-template", default="hybrid_final_behavior_seed{seed}.json")
    parser.add_argument("--ppo-template", default="ppo_only_final_seed{seed}.json")
    parser.add_argument("--output", default="artifacts/sanity/hybrid_final_behavior_multiseed.json")
    parser.add_argument("--manifest-output", default="artifacts/models/hybrid_v1_controller_final_manifest.json")
    parser.add_argument("--controller-version", default="hybrid_controller_v1_stable_pending")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
