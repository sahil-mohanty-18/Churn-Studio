"""Optional Playwright browser checks; not required to run the application."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
shots = root / "reports/screenshots"
shots.mkdir(exist_ok=True)
log = (root / "reports/browser-server.log").open("w")
server = subprocess.Popen(
    [sys.executable, "run.py", "serve", "--port", "5058"],
    cwd=root,
    stdout=log,
    stderr=log,
)
base = "http://127.0.0.1:5058"
errors = []
results = []
try:
    for _ in range(60):
        try:
            urlopen(base + "/api/health", timeout=1).close()
            break
        except OSError:
            time.sleep(0.5)
    with sync_playwright() as p:
        kwargs = {"headless": True, "args": ["--no-sandbox"]}
        if os.environ.get("CHROMIUM_PATH"):
            kwargs["executable_path"] = os.environ["CHROMIUM_PATH"]
        browser = p.chromium.launch(**kwargs)
        page = browser.new_page(
            viewport={"width": 1440, "height": 1080}, device_scale_factor=1
        )
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base, wait_until="networkidle")
        page.wait_for_selector("#kpi-0")
        page.screenshot(path=str(shots / "dashboard.png"), full_page=True)
        results.append("Dashboard loaded")
        assert page.locator("#kpi-0").inner_text() == "7,043"
        page.locator("#theme-toggle").click()
        assert page.locator("html").get_attribute("data-theme") == "dark"
        page.screenshot(path=str(shots / "dashboard-dark.png"), full_page=True)
        page.locator("#theme-toggle").click()
        page.goto(base + "/predict", wait_until="networkidle")
        page.locator("#prediction-form button[type=submit]").click()
        page.wait_for_selector(".risk-gauge")
        assert page.locator(".reason").count() == 5
        page.screenshot(path=str(shots / "predict.png"), full_page=True)
        results.append("Form prediction and five SHAP reasons rendered")
        page.goto(base + "/batch", wait_until="networkidle")
        page.locator("#csv-file").set_input_files(
            str(root / "static/sample_customers.csv")
        )
        page.locator("#batch-form button").click()
        page.wait_for_selector("#batch-result table")
        assert page.locator("#batch-result tbody tr").count() == 4
        results.append("Batch upload and results rendered")
        page.goto(base + "/customers", wait_until="networkidle")
        page.wait_for_selector("[data-customer]")
        cid = page.locator("[data-customer]").first.get_attribute("data-customer")
        page.locator("[data-customer]").first.click()
        page.wait_for_selector("dialog .risk-gauge")
        page.locator("#close-dialog").click()
        page.locator("#search").fill(cid)
        page.wait_for_timeout(600)
        assert page.locator("[data-customer]").count() == 1
        results.append("Customer drawer and search work")
        page.goto(base + "/insights", wait_until="networkidle")
        page.wait_for_selector(".matrix-cell")
        assert page.locator("tbody tr").count() == 8
        page.screenshot(path=str(shots / "model-insights.png"), full_page=True)
        results.append("All eight model comparison rows rendered")
        page.goto(base + "/simulator", wait_until="networkidle")
        page.wait_for_selector(".risk-gauge")
        before = page.locator(".risk-gauge strong").inner_text()
        page.locator("[name=Contract]").select_option("Two year")
        page.wait_for_timeout(1800)
        after = page.locator(".risk-gauge strong").inner_text()
        assert before != after
        results.append("What-if contract change updated probability")
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(base, wait_until="networkidle")
        page.wait_for_selector("#kpi-0")
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )
        page.screenshot(path=str(shots / "mobile.png"), full_page=True)
        page.locator("#menu-toggle").click()
        assert page.locator("#sidebar").get_attribute("class") == "sidebar open"
        results.append("Mobile layout and menu work without horizontal overflow")
        offline = browser.new_page(viewport={"width": 1440, "height": 1000})
        offline.route("https://**/*", lambda route: route.abort())
        offline.goto(base, wait_until="networkidle")
        offline.wait_for_selector(".chart-fallback")
        assert offline.locator("#kpi-0").count() == 1
        results.append("CDN-blocked chart data fallback works")
        offline.close()
        browser.close()
    assert not errors, errors
    print(json.dumps({"passed": results, "javascript_errors": errors}, indent=2))
finally:
    server.terminate()
    server.wait(timeout=10)
    log.close()
