from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Split active BrowserGym sites into batch config files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-output-root", default="artifacts/training")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    input_path = Path(args.input)
    config = _read_active_config(input_path)
    sites = _validate_sites(config.get("sites", []))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_paths = []
    for index, start in enumerate(range(0, len(sites), args.batch_size), start=1):
        batch_name = f"batch-{index:03d}"
        batch_config = dict(config)
        batch_config["sites"] = sites[start : start + args.batch_size]
        batch_config["batch_id"] = batch_name
        batch_config["source_active_sites"] = str(input_path)
        batch_config.setdefault("shared_model_path", "artifacts/models/jaws_browsergym_shared_ppo_v2_browsergym_raw_obs.pt")
        batch_config["output_dir"] = str(Path(args.training_output_root) / batch_name / "multisite")
        batch_path = output_dir / f"{batch_name}.json"
        batch_path.write_text(json.dumps(batch_config, ensure_ascii=False, indent=2), encoding="utf-8")
        batch_paths.append(str(batch_path))

    manifest = {
        "input": str(input_path),
        "batch_size": args.batch_size,
        "total_sites": len(sites),
        "total_batches": len(batch_paths),
        "batches": batch_paths,
    }
    (output_dir / "batches_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _read_active_config(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"sites": data}
    raise ValueError(f"Expected JSON object or list: {path}")


def _validate_sites(raw_sites: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_sites, list):
        raise ValueError("Input requires a sites list.")
    sites = [dict(item) for item in raw_sites if isinstance(item, Mapping)]
    return sites


if __name__ == "__main__":
    raise SystemExit(main())
