from __future__ import annotations
import json
import traceback
from playwright.sync_api import sync_playwright

result = {"pass": False, "primary_exception": None, "cleanup_exception": None}
pw = browser = context = page = None
try:
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.set_content("<html><body>ok</body></html>")
    result.update({"pass": True, "url": page.url, "executable": pw.chromium.executable_path})
except Exception as exc:
    result["primary_exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
finally:
    for name, obj in (("page", page), ("context", context), ("browser", browser)):
        if obj is not None:
            try:
                obj.close()
            except Exception as exc:
                result["cleanup_exception"] = {"component": name, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    if pw is not None:
        try:
            pw.stop()
        except Exception as exc:
            result["cleanup_exception"] = {"component": "playwright", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
print(json.dumps(result, ensure_ascii=False))
