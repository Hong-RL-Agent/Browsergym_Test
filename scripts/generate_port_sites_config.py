from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BrowserGym multi-site config from localhost port range.")
    parser.add_argument("--start-port", type=int, required=True)
    parser.add_argument("--end-port", type=int, required=True)
    parser.add_argument("--site-map", default="", help="Optional JSON map of port string to site_id.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shared-model-path", default="artifacts/models/jaws_browsergym_shared_ppo_v2_browsergym_raw_obs.pt")
    parser.add_argument("--output-dir", default="artifacts/multisite")
    args = parser.parse_args()

    if args.start_port > args.end_port:
        raise ValueError("--start-port must be less than or equal to --end-port")

    site_map = _read_site_map(Path(args.site_map)) if args.site_map else {}
    sites = []
    for port in range(args.start_port, args.end_port + 1):
        site_id = str(site_map.get(str(port)) or f"site{port}")
        catalog_path = Path("datasets") / site_id / "bug_catalog.json"
        has_catalog = catalog_path.exists()
        site: Dict[str, Any] = {
            "site_id": site_id,
            "base_url": f"http://localhost:{port}",
            "has_bug_catalog": has_catalog,
            "bug_catalog_path": catalog_path.as_posix(),
            "reward_scale": 1.0,
        }
        if not has_catalog:
            site["evaluation_mode"] = "openended_anomaly_discovery"
        sites.append(site)

    output = {
        "shared_model_path": args.shared_model_path,
        "output_dir": args.output_dir,
        "port_range": {"start_port": args.start_port, "end_port": args.end_port},
        "sites": sites,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated_sites={len(sites)}")
    print(f"output={output_path}")
    return 0


def _read_site_map(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object site map: {path}")
    return {str(port): str(site_id) for port, site_id in data.items()}


if __name__ == "__main__":
    raise SystemExit(main())
