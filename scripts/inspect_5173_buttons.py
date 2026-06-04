from playwright.sync_api import sync_playwright

URL = "http://localhost:5173"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(2000)

    buttons = page.locator("button")
    count = buttons.count()

    print(f"Total buttons: {count}")

    for i in range(count):
        button = buttons.nth(i)

        try:
            text = button.inner_text(timeout=1000).strip()
        except Exception:
            text = ""

        try:
            attrs = button.evaluate(
                """el => ({
                    id: el.id || "",
                    class: el.className || "",
                    ariaLabel: el.getAttribute("aria-label") || "",
                    dataTrigger: el.getAttribute("data-trigger") || "",
                    dataTriggerId: el.getAttribute("data-trigger-id") || "",
                    dataTestId: el.getAttribute("data-testid") || ""
                })"""
            )
        except Exception:
            attrs = {}

        print("----")
        print(f"index: {i}")
        print(f"text: {text}")
        print(f"attrs: {attrs}")

    browser.close()