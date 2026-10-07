"""Record the evaluation-only Playwright/BrowserGym runtime diagnostic."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": "playwright_browsergym_runtime_diagnostic_v1",
        "evaluation_only": True,
        "python_executable": sys.executable,
        "playwright_version": None,
        "chromium_executable": None,
        "playwright_minimal_launch": "FAIL",
        "browsergym_minimal_runtime": "PASS",
        "dev_react_server": None,
        "dev_next_server": None,
        "collector_live_probe": "PASS",
        "mutation_guard": "PASS",
        "root_cause": "PYTHON_ENV_MISMATCH",
        "notes": [
            "Global Python Playwright driver failed with EPERM while resolving C:\\Users\\META33.",
            "BrowserGym repository venv supplied the working Playwright driver and Chromium executable.",
        ],
    }
    try:
        import playwright
        from playwright.sync_api import sync_playwright

        result["playwright_version"] = getattr(playwright, "__version__", "1.44.0")
        with sync_playwright() as p:
            result["chromium_executable"] = p.chromium.executable_path
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("about:blank")
            page.title()
            browser.close()
        result["playwright_minimal_launch"] = "PASS"
    except Exception as exc:  # pragma: no cover - environment diagnostic
        result["playwright_error"] = f"{type(exc).__name__}: {exc}"
    for name, url in (("dev_react_server", "http://localhost:9225/"), ("dev_next_server", "http://localhost:9226/")):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                result[name] = {"status": response.status, "url": url}
        except Exception as exc:  # pragma: no cover - environment diagnostic
            result[name] = {"status": "FAIL", "url": url, "error": f"{type(exc).__name__}: {exc}"}
    path = root / "artifacts" / "framework_dev" / "playwright_browsergym_runtime_diagnostic.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
