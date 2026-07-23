"""Generate a multisite config for every checked-in error-site directory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites-root", default="../RL_Errorsite-frontend-errorsite")
    parser.add_argument("--start-port", type=int, default=9220)
    parser.add_argument("--output", default="configs/generated_local_sites.json")
    args = parser.parse_args()
    root = Path(args.sites_root)
    sites = []
    for index, directory in enumerate(sorted(root.glob("site*"))):
        match = re.match(r"(site\d+)", directory.name)
        if not match or not (directory / "server.js").exists():
            continue
        site_id = match.group(1)
        catalog = Path("datasets") / site_id / "bug_catalog.json"
        sites.append({
            "site_id": site_id,
            "base_url": f"http://localhost:{args.start_port + index}",
            "has_bug_catalog": catalog.exists(),
            "bug_catalog_path": catalog.as_posix(),
            "evaluation_mode": "catalog" if catalog.exists() else "openended_anomaly_discovery",
            "reward_scale": 1.0,
        })
    output = {"shared_model_path": "artifacts/models/jaws_browsergym_shared_rainbow_dqn.pt", "sites": sites}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated_sites={len(sites)}")
    print(f"output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
