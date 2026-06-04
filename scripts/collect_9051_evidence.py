import csv
import re
from pathlib import Path
from playwright.sync_api import sync_playwright


URL = "http://localhost:5173"
OUT = Path("artifacts/logs/site9051_multi_evidence.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)


TRIGGERS = [
    {
        "button_text": "interrupt storm",
        "action_label": "Interrupt Storm",
        "trigger_id": "interrupt-storm",
        "expected_alert": "Virtualization Driver Interrupt Storm",
    },
    {
        "button_text": "kernel lockup",
        "action_label": "Kernel Lockup",
        "trigger_id": "kernel-lockup",
        "expected_alert": "Kernel High Lockup",
    },
    {
        "button_text": "cache bloat",
        "action_label": "Cache Bloat",
        "trigger_id": "cache-bloat",
        "expected_alert": "cgroup Memory Limit & Cache Bloat",
    },
    {
        "button_text": "numa paradox",
        "action_label": "Numa Paradox",
        "trigger_id": "numa-paradox",
        "expected_alert": "NUMA Auto-balancing Paradox",
    },
]


def extract_number(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None

    value = match.group(1).replace(",", "").replace("%", "").replace("ms", "").strip()

    try:
        return float(value)
    except ValueError:
        return None


def collect_state(page):
    text = page.locator("body").inner_text(timeout=5000)

    return {
        "body_text": text,
        "active_alerts": extract_number(text, r"ACTIVE\s*ALERTS\s*([0-9]+)"),
        "cpu_load": extract_number(text, r"CPU\s*Load\s*([0-9.]+)\s*%"),
        "cpu_steal": extract_number(text, r"CPU\s*Steal\s*([0-9.]+)\s*%"),
        "memory_pressure": extract_number(text, r"Memory\s*Pressure\s*([0-9.]+)\s*%"),
        "io_wait": extract_number(text, r"I/O\s*Wait\s*([0-9.]+)\s*%"),
        "p99_latency": extract_number(text, r"P99\s*LATENCY\s*([0-9.]+)\s*ms"),
    }


def detect_alert(text, expected_alert):
    lower_text = text.lower()

    if expected_alert.lower() in lower_text:
        return expected_alert

    fallback_keywords = [
        "interrupt storm",
        "kernel lockup",
        "cache bloat",
        "numa",
        "pid limit",
        "journal",
        "gpu",
        "bandwidth",
        "pcie",
        "compaction",
        "thundering herd",
    ]

    for keyword in fallback_keywords:
        if keyword in lower_text:
            return keyword

    return ""


def click_trigger(page, button_text):
    """
    실제 오류 트리거 버튼을 클릭한다.
    같은 이름이 여러 개 잡힐 수 있으므로 control-button.compact를 우선 사용한다.
    """

    pattern = re.compile(rf"^{re.escape(button_text)}$", re.IGNORECASE)

    try:
        page.locator("button.control-button.compact").filter(
            has_text=pattern
        ).first.click(timeout=5000)
        return True
    except Exception:
        pass

    try:
        page.get_by_role(
            "button",
            name=pattern,
            exact=True,
        ).click(timeout=5000)
        return True
    except Exception:
        pass

    buttons = page.locator("button")
    count = buttons.count()

    for i in range(count):
        button = buttons.nth(i)
        try:
            text = button.inner_text(timeout=1000).strip().lower()
        except Exception:
            continue

        if text == button_text.lower():
            button.click(timeout=5000)
            return True

    return False


def reset_lab(page):
    """
    다음 오류를 독립적으로 측정하기 위해 Reset Lab 버튼을 누른다.
    """
    try:
        page.get_by_role("button", name=re.compile(r"reset lab", re.IGNORECASE)).click(timeout=3000)
        page.wait_for_timeout(1500)
    except Exception:
        pass


def write_row(row):
    fieldnames = [
        "step",
        "url",
        "action",
        "trigger_id",
        "expected_alert",
        "observation_alert",
        "active_alerts_before",
        "active_alerts_after",
        "cpu_load_before",
        "cpu_load_after",
        "cpu_steal_before",
        "cpu_steal_after",
        "memory_pressure_before",
        "memory_pressure_after",
        "io_wait_before",
        "io_wait_after",
        "p99_latency_before",
        "p99_latency_after",
        "reward",
        "anomaly_detected",
    ]

    exists = OUT.exists()

    with OUT.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerow(row)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1600, "height": 1000})

    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(1500)

    for step, trigger in enumerate(TRIGGERS, start=1):
        print(f"\n=== Step {step}: {trigger['action_label']} ===")

        reset_lab(page)
        before = collect_state(page)

        clicked = click_trigger(page, trigger["button_text"])

        if not clicked:
            print(f"[FAILED] Button not found: {trigger['button_text']}")
            continue

        page.wait_for_timeout(3000)

        after = collect_state(page)
        alert = detect_alert(after["body_text"], trigger["expected_alert"])

        anomaly_detected = bool(alert) or (
            before["active_alerts"] is not None
            and after["active_alerts"] is not None
            and after["active_alerts"] > before["active_alerts"]
        )

        reward = 1.0 if anomaly_detected else 0.0

        row = {
            "step": step,
            "url": URL,
            "action": f"click(button='{trigger['action_label']}')",
            "trigger_id": trigger["trigger_id"],
            "expected_alert": trigger["expected_alert"],
            "observation_alert": alert,
            "active_alerts_before": before["active_alerts"],
            "active_alerts_after": after["active_alerts"],
            "cpu_load_before": before["cpu_load"],
            "cpu_load_after": after["cpu_load"],
            "cpu_steal_before": before["cpu_steal"],
            "cpu_steal_after": after["cpu_steal"],
            "memory_pressure_before": before["memory_pressure"],
            "memory_pressure_after": after["memory_pressure"],
            "io_wait_before": before["io_wait"],
            "io_wait_after": after["io_wait"],
            "p99_latency_before": before["p99_latency"],
            "p99_latency_after": after["p99_latency"],
            "reward": reward,
            "anomaly_detected": anomaly_detected,
        }

        write_row(row)

        print("Action:", row["action"])
        print("Observation alert:", alert)
        print("Active alerts:", before["active_alerts"], "->", after["active_alerts"])
        print("CPU load:", before["cpu_load"], "->", after["cpu_load"])
        print("CPU steal:", before["cpu_steal"], "->", after["cpu_steal"])
        print("Memory pressure:", before["memory_pressure"], "->", after["memory_pressure"])
        print("I/O wait:", before["io_wait"], "->", after["io_wait"])
        print("P99 latency:", before["p99_latency"], "->", after["p99_latency"])
        print("Reward:", reward)
        print("Anomaly detected:", anomaly_detected)

    print("\nSaved multi evidence log:", OUT)

    browser.close()