"""Generate J.A.W.S web application error report PDFs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from exporters.pdf_report_exporter import export_pdf_report


def generate_pdf_report(report_data: Mapping[str, Any], output_dir: str | Path = "artifacts/reports") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = str(report_data.get("mode") or "")
    site_id = str(report_data.get("site_id") or "site")
    if mode == "BrowserGym Autonomous PPO":
        timestamp = _timestamp_from_report_id(str(report_data.get("report_id") or "")) or datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"error_report_browsergym_{site_id}_{timestamp}.pdf"
    else:
        report_id = str(report_data.get("report_id") or f"{site_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        filename = f"error_report_{report_id}.pdf"
    return export_pdf_report(report_data, output_dir / filename)


def _timestamp_from_report_id(report_id: str) -> str:
    parts = report_id.split("-")
    if len(parts) >= 4 and parts[-2].isdigit():
        return f"{parts[-2]}-{parts[-1]}"
    return ""
