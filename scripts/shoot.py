"""Local visual QA helper: screenshot Streamlit routes with installed Chrome.

Usage:
  python3 scripts/shoot.py [--url URL] [--out DIR] [--width W] [--height H] [--wait MS] [route ...]
Each route is passed as ?view=<route>.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default="/tmp/icu_shots")
    ap.add_argument("--width", type=int, default=1680)
    ap.add_argument("--height", type=int, default=1050)
    ap.add_argument("--wait", type=int, default=2500)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("routes", nargs="*", default=[""])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
        errors: list[str] = []
        page.on("console", lambda m: errors.append(f"[{m.type}] {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
        for route in args.routes:
            target = args.url + (f"/?view={route}" if route else "/")
            page.goto(target, wait_until="networkidle", timeout=60000)
            try:
                page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=30000)
                page.wait_for_timeout(args.wait)
                page.wait_for_function(
                    "() => !document.querySelector('[data-testid=\"stSkeleton\"]')", timeout=15000
                )
            except Exception:
                pass
            name = route or "root"
            path = out / f"{name}.png"
            page.screenshot(path=str(path), full_page=args.full)
            print(f"saved {path}")
        if errors:
            print("--- console errors ---")
            for e in errors[:20]:
                print(e)
        browser.close()


if __name__ == "__main__":
    main()
