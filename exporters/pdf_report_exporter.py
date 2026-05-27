"""HTML-to-PDF exporter for J.A.W.S error reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, select_autoescape


def export_pdf_report(
    report_data: Mapping[str, Any],
    output_path: str | Path,
    template_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template_path = Path(template_path or Path("reports") / "templates" / "error_report_template.html")
    html = render_report_html(report_data, template_path)
    html_path = output_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    _html_to_pdf(html_path, output_path)
    return output_path


def render_report_html(report_data: Mapping[str, Any], template_path: str | Path) -> str:
    template_path = Path(template_path)
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["fmt"] = _fmt
    template = env.get_template(template_path.name)
    return template.render(report=report_data)


def _html_to_pdf(html_path: Path, output_path: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("PDF export requires Playwright in the active Python environment.") from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1600})
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(output_path),
            format="A4",
            print_background=True,
            margin={"top": "14mm", "right": "12mm", "bottom": "14mm", "left": "12mm"},
        )
        browser.close()


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)
