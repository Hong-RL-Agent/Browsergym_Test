from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.browsergym_training_service import BrowserGymTrainingService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_sites.json")
    parser.add_argument("--site-id", default="site001")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--headless", type=_parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--load-model", default="")
    parser.add_argument("--save-model", default="")
    parser.add_argument("--disable-guidance", action="store_true")
    parser.add_argument("--disable-ground-truth-metadata", action="store_true")
    args = parser.parse_args()
    site = _load_site_config(args.config, args.site_id, args.base_url)

    service = BrowserGymTrainingService(
        site_id=str(site["site_id"]),
        base_url=str(site["base_url"]),
        episodes=args.episodes,
        max_steps=args.max_steps,
        headless=args.headless,
        seed=args.seed,
        entropy_coef=args.entropy_coef,
        load_model_path=args.load_model or None,
        model_output_path=args.save_model or None,
        use_guided_actions=not args.disable_guidance,
        use_ground_truth_metadata=not args.disable_ground_truth_metadata,
    )
    summary = service.train()
    print("training_summary_path:", service.summary_path)
    print("model_path:", summary["model_path"])
    return 0


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _load_site_config(config_path: str, site_id: str | None, base_url: str | None) -> dict[str, Any]:
    if base_url:
        return {"site_id": site_id or "site001", "base_url": base_url}
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    sites = config.get("sites", []) if isinstance(config, Mapping) else []
    for site in sites:
        if isinstance(site, Mapping) and str(site.get("site_id") or "") == str(site_id or ""):
            return {"site_id": site.get("site_id"), "base_url": site.get("base_url")}
    raise ValueError(f"site_id {site_id!r} not found in {config_path}")


if __name__ == "__main__":
    raise SystemExit(main())
