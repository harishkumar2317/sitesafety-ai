#!/usr/bin/env python
"""Smoke test: renders every website route through Flask's test client.

    python tools/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "website"))

from app import app  # noqa: E402
from safety import database as db  # noqa: E402


def main() -> int:
    client = app.test_client()
    months = [m["month"] for m in db.month_list()]
    worst = db.list_violations(limit=1)
    first_day = worst[0]["day"] if worst else None

    routes = [
        ("GET", "/"),
        ("GET", "/api/dashboard"),
        ("GET", "/api/series?limit=30"),
        ("GET", "/violations"),
        ("GET", "/violations?kind=helmet"),
        ("GET", "/violations?kind=vest"),
        ("GET", "/api/violations?limit=5"),
        ("GET", "/day"),
        ("GET", "/history"),
        ("GET", "/members"),
        ("GET", "/upload"),
        ("GET", "/export.csv"),
    ]
    if first_day:
        routes += [("GET", f"/day/{first_day}"), ("GET", f"/violations?day={first_day}")]
    for m in months[:3]:
        routes.append(("GET", f"/history/{m}"))
    if worst:
        routes.append(("GET", f"/violations/{worst[0]['id']}"))
        routes.append(("GET", f"/photo/{worst[0]['id']}"))

    failures = 0
    for method, url in routes:
        try:
            rv = client.get(url)
            ok = rv.status_code == 200
            if not ok:
                failures += 1
            size = len(rv.data)
            flag = "OK  " if ok else "FAIL"
            print(f"{flag} {rv.status_code} {url:<38} {size:>8} bytes")
            if not ok:
                print("      " + rv.data[:400].decode("utf-8", "replace").replace("\n", " "))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  ERROR {url}: {exc}")

    print(f"\n{len(routes) - failures}/{len(routes)} routes OK")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
