from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_APIS = [
    {
        "api_id": "products-search",
        "method": "GET",
        "path": "/api/products",
        "query_schema": {"q": "string", "category": "string", "sort": "string"},
        "expected_status": [200],
        "auth_required": False
    },
    {
        "api_id": "cart-create",
        "method": "POST",
        "path": "/api/cart",
        "body_schema": {"productId": "string", "quantity": "number"},
        "expected_status": [200, 201],
        "auth_required": False
    },
    {
        "api_id": "item-detail",
        "method": "GET",
        "path": "/api/products/{id}",
        "path_schema": {"id": "string"},
        "expected_status": [200, 404],
        "auth_required": False
    }
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate J.A.W.S API catalog config from localhost ports.")
    parser.add_argument("--start-port", type=int, required=True)
    parser.add_argument("--end-port", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--site-map", default="")
    parser.add_argument("--template", default="", help="Optional endpoint template JSON with an apis list.")
    args = parser.parse_args()

    if args.start_port > args.end_port:
        raise ValueError("--start-port must be less than or equal to --end-port")
    site_map = _read_site_map(Path(args.site_map)) if args.site_map else {}
    apis = _read_template(Path(args.template)) if args.template else DEFAULT_APIS
    sites = []
    for port in range(args.start_port, args.end_port + 1):
        site_id = str(site_map.get(str(port)) or f"site{port}")
        sites.append(
            {
                "site_id": site_id,
                "base_url": f"http://localhost:{port}",
                "health_check": "/",
                "apis": apis,
            }
        )
    output = {
        "source": "api",
        "port_range": {"start_port": args.start_port, "end_port": args.end_port},
        "sites": sites,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"generated_sites": len(sites), "output": str(output_path)}, ensure_ascii=False))
    return 0


def _read_site_map(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object site map: {path}")
    return {str(port): str(site_id) for port, site_id in data.items()}


def _read_template(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    apis = data.get("apis", data) if isinstance(data, dict) else data
    if not isinstance(apis, list):
        raise ValueError(f"Expected an apis list in template: {path}")
    return [dict(item) for item in apis if isinstance(item, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
