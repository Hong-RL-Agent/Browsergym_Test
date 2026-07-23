"""Loads metadata for allow-listed, non-destructive OWASP-aligned tests."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "owasp_safe_test_catalog.json"


def load_safe_test_catalog(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("default_mode") != "non-destructive" or not isinstance(data.get("tests"), dict):
        raise ValueError("OWASP safe catalog must declare non-destructive mode and tests")
    return data


def safe_test_metadata(mutation: str, path: str | Path = DEFAULT_PATH) -> dict[str, Any] | None:
    item = load_safe_test_catalog(path)["tests"].get(mutation)
    return dict(item) if isinstance(item, dict) else None
