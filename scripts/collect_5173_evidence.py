import csv
import re
from pathlib import Path
from playwright.sync_api import sync_playwright


URL = "http://localhost:5173"
OUT = Path("artifacts/logs/site5173_multi_evidence.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)


TRIGGERS = [
    # 서버/인프라 오류 11개
    {
        "category": "server-infra",
        "trigger_id": "interrupt-storm",
        "button_test_id": "trigger-interrupt-storm",
        "button_text": "Virtualization Driver Interrupt Storm",
    },
    {
        "category": "server-infra",
        "trigger_id": "kernel-lockup",
        "button_test_id": "trigger-kernel-lockup",
        "button_text": "Kernel High Lockup",
    },
    {
        "category": "server-infra",
        "trigger_id": "cache-bloat",
        "button_test_id": "trigger-cache-bloat",
        "button_text": "cgroup Memory Limit & Cache Bloat",
    },
    {
        "category": "server-infra",
        "trigger_id": "numa-paradox",
        "button_test_id": "trigger-numa-paradox",
        "button_text": "NUMA Auto-balancing Paradox",
    },
    {
        "category": "server-infra",
        "trigger_id": "pid-limit",
        "button_test_id": "trigger-pid-limit",
        "button_text": "Container PID Limit Fork Failure",
    },
    {
        "category": "server-infra",
        "trigger_id": "journal-delay",
        "button_test_id": "trigger-journal-delay",
        "button_text": "File System Journaling Delay",
    },
    {
        "category": "server-infra",
        "trigger_id": "gpu-launch-delay",
        "button_test_id": "trigger-gpu-launch-delay",
        "button_text": "GPU Kernel Launch Delay",
    },
    {
        "category": "server-infra",
        "trigger_id": "bandwidth-saturation",
        "button_test_id": "trigger-bandwidth-saturation",
        "button_text": "Memory Bandwidth Saturation",
    },
    {
        "category": "server-infra",
        "trigger_id": "pcie-p2p",
        "button_test_id": "trigger-pcie-p2p",
        "button_text": "GPU PCIe P2P Topology Mismatch",
    },
    {
        "category": "server-infra",
        "trigger_id": "compaction-storm",
        "button_test_id": "trigger-compaction-storm",
        "button_text": "Memory Compaction Storm",
    },
    {
        "category": "server-infra",
        "trigger_id": "thundering-herd",
        "button_test_id": "trigger-thundering-herd",
        "button_text": "Thundering Herd Problem",
    },

    # 프론트엔드 오류 5개
    {
        "category": "frontend",
        "trigger_id": "seat-map-render-crash",
        "button_test_id": "trigger-seat-map-render-crash",
        "button_text": "Seat Map Rendering Component Crash",
    },
    {
        "category": "frontend",
        "trigger_id": "queue-ui-freeze",
        "button_test_id": "trigger-queue-ui-freeze",
        "button_text": "Queue Panel UI Freeze",
    },
    {
        "category": "frontend",
        "trigger_id": "chart-overflow",
        "button_test_id": "trigger-chart-overflow",
        "button_text": "Dashboard Chart Overflow",
    },
    {
        "category": "frontend",
        "trigger_id": "stale-seat-status",
        "button_test_id": "trigger-stale-seat-status",
        "button_text": "Stale Seat Status Badge",
    },
    {
        "category": "frontend",
        "trigger_id": "modal-state-leak",
        "button_test_id": "trigger-modal-state-leak",
        "button_text": "Booking Modal State Leak",
    },

    # 백엔드/API 오류 5개
    {
        "category": "backend-api",
        "trigger_id": "api-timeout",
        "button_test_id": "trigger-api-timeout",
        "button_text": "Seat Lock API Timeout",
    },
    {
        "category": "backend-api",
        "trigger_id": "route-500",
        "button_test_id": "trigger-route-500",
        "button_text": "Seat Allocation 500 Error",
    },
    {
        "category": "backend-api",
        "trigger_id": "auth-token-expired",
        "button_test_id": "trigger-auth-token-expired",
        "button_text": "Expired Booking Token",
    },
    {
        "category": "backend-api",
        "trigger_id": "duplicate-booking",
        "button_test_id": "trigger-duplicate-booking",
        "button_text": "Duplicate Seat Booking",
    },
    {
        "category": "backend-api",
        "trigger_id": "ticket-schema-mismatch",
        "button_test_id": "trigger-ticket-schema-mismatch",
        "button_text": "Ticket API Schema Mismatch",
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
        "active_alerts": extract_number(text, r"Active\s*Alerts\s*([0-9]+)"),
        "cpu_load": extract_number(text, r"CPU\s*Load\s*([0-9.]+)\s*%"),
        "cpu_steal": extract_number(text, r"CPU\s*Steal\s*([0-9.]+)\s*%"),
        "memory_pressure": extract_number(text, r"Memory\s*Pressure\s*([0-9.]+)\s*%"),
        "io_wait": extract_number(text, r"I/O\s*Wait\s*([0-9.]+)\s*%"),
        "p99_latency": extract_number(text, r"P99\s*Latency\s*([0-9.]+)\s*ms"),
        "queue_length": extract_number(text, r"Queue\s*Length\s*([0-9]+)"),
        "active_users": extract_number(text, r"Active\s*Users\s*([0-9]+)"),
    }


def reset_site(page):
    try:
        page.get_by_test_id("reset-button").click(timeout=3000)
        page.wait_for_timeout(1000)
    except Exception:
        try:
            page.get_by_role("button", name=re.compile(r"reset", re.IGNORECASE)).first.click(timeout=3000)
            page.wait_for_timeout(1000)
        except Exception:
            pass


def detect_observation(page, trigger):
    text = page.locator("body").inner_text(timeout=5000)
    lower_text = text.lower()

    trigger_id = trigger["trigger_id"]
    button_text = trigger["button_text"]

    detected_keywords = [
        trigger_id.replace("-", " "),
        button_text.lower(),
        "error",
        "failed",
        "failure",
        "timeout",
        "warning",
        "critical",
        "alert",
        "regression",
        "crash",
        "freeze",
        "stale",
        "duplicate",
        "schema",
        "expired",
    ]

    for keyword in detected_keywords:
        if keyword and keyword.lower() in lower_text:
            return button_text

    return ""


def write_row(row):
    fieldnames = [
        "step",
        "url",
        "category",
        "trigger_id",
        "action",
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
        "queue_length_before",
        "queue_length_after",
        "active_users_before",
        "active_users_after",
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

    # 테스트 컨트롤이 숨겨져 있으면 열기
    try:
        page.get_by_role("button", name=re.compile(r"show test controls", re.IGNORECASE)).click(timeout=3000)
        page.wait_for_timeout(1000)
    except Exception:
        pass

    for step, trigger in enumerate(TRIGGERS, start=1):
        print(f"\n=== Step {step}: {trigger['button_text']} ===")

        reset_site(page)
        page.wait_for_timeout(1000)

        # reset 후 test controls가 다시 숨겨질 수 있어서 다시 열기
        try:
            page.get_by_role("button", name=re.compile(r"show test controls", re.IGNORECASE)).click(timeout=2000)
            page.wait_for_timeout(500)
        except Exception:
            pass

        before = collect_state(page)

        try:
            page.get_by_test_id(trigger["button_test_id"]).click(timeout=5000)
            clicked = True
        except Exception as e:
            print(f"[FAILED] Button not found or not clickable: {trigger['button_test_id']}")
            print("Reason:", e)
            clicked = False

        if not clicked:
            continue

        page.wait_for_timeout(3000)

        after = collect_state(page)
        observation_alert = detect_observation(page, trigger)

        metric_changed = any(
            [
                before["active_alerts"] is not None
                and after["active_alerts"] is not None
                and after["active_alerts"] > before["active_alerts"],

                before["cpu_load"] is not None
                and after["cpu_load"] is not None
                and after["cpu_load"] != before["cpu_load"],

                before["cpu_steal"] is not None
                and after["cpu_steal"] is not None
                and after["cpu_steal"] != before["cpu_steal"],

                before["memory_pressure"] is not None
                and after["memory_pressure"] is not None
                and after["memory_pressure"] != before["memory_pressure"],

                before["io_wait"] is not None
                and after["io_wait"] is not None
                and after["io_wait"] != before["io_wait"],

                before["p99_latency"] is not None
                and after["p99_latency"] is not None
                and after["p99_latency"] != before["p99_latency"],
            ]
        )

        anomaly_detected = bool(observation_alert) or metric_changed
        reward = 1.0 if anomaly_detected else 0.0

        row = {
            "step": step,
            "url": URL,
            "category": trigger["category"],
            "trigger_id": trigger["trigger_id"],
            "action": f"click(testid='{trigger['button_test_id']}')",
            "observation_alert": observation_alert,
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
            "queue_length_before": before["queue_length"],
            "queue_length_after": after["queue_length"],
            "active_users_before": before["active_users"],
            "active_users_after": after["active_users"],
            "reward": reward,
            "anomaly_detected": anomaly_detected,
        }

        write_row(row)

        print("Category:", row["category"])
        print("Trigger ID:", row["trigger_id"])
        print("Action:", row["action"])
        print("Observation alert:", row["observation_alert"])
        print("Active alerts:", before["active_alerts"], "->", after["active_alerts"])
        print("CPU load:", before["cpu_load"], "->", after["cpu_load"])
        print("CPU steal:", before["cpu_steal"], "->", after["cpu_steal"])
        print("Memory pressure:", before["memory_pressure"], "->", after["memory_pressure"])
        print("I/O wait:", before["io_wait"], "->", after["io_wait"])
        print("P99 latency:", before["p99_latency"], "->", after["p99_latency"])
        print("Reward:", reward)
        print("Anomaly detected:", anomaly_detected)

    print("\nSaved evidence log:", OUT)
    browser.close()