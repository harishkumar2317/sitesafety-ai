#!/usr/bin/env python
"""Optional: fill the database with plausible history so the website demos well.

Real detections are written by Phase 2. This script only *supplements* them
with older days/months (marked note='demo seed') so the Day Report and Month
History screens have something to show on day one.

    python tools/seed_demo_data.py            # add 60 days of history
    python tools/seed_demo_data.py --days 30
    python tools/seed_demo_data.py --reset    # wipe the database first
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from safety import database as db  # noqa: E402
from safety.config import DB_PATH, VIOLATION_DIR, ensure_dirs  # noqa: E402

SOURCES = ["camera:gate-north", "camera:zone-b", "camera:scaffold-03", "site-walkthrough"]
KINDS = [("helmet", "missing", "worn"),
         ("vest", "worn", "missing"),
         ("helmet+vest", "missing", "missing")]


def reset() -> None:
    ensure_dirs()
    for p in DB_PATH.parent.glob(DB_PATH.name + "*"):
        p.unlink(missing_ok=True)
    for p in VIOLATION_DIR.glob("*.jpg"):
        p.unlink(missing_ok=True)
    print("[seed] database and photos cleared")


def seed(days: int) -> None:
    ensure_dirs()
    db.init_db()

    photos = sorted(VIOLATION_DIR.glob("*.jpg"))
    if not photos:
        print("[seed] no captured photos yet - run Phase 2 first so records can "
              "point at real evidence images.")
        return

    today = datetime.now().date()
    created = 0
    for back in range(days, 0, -1):
        day = today - timedelta(days=back)
        if day.weekday() >= 5 and random.random() < 0.75:
            continue  # quiet weekends
        n = random.choice([0, 1, 1, 2, 2, 3, 4, 6])
        for _ in range(n):
            kind, helmet, vest = random.choice(KINDS)
            when = datetime.combine(
                day, datetime.min.time()
            ) + timedelta(hours=random.randint(7, 17),
                          minutes=random.randint(0, 59),
                          seconds=random.randint(0, 59))
            photo = random.choice(photos)
            db.add_violation(
                kind=kind, helmet=helmet, vest=vest,
                source=random.choice(SOURCES),
                image_path=str(photo),
                track_id=random.randint(1, 9),
                helmet_conf=round(random.uniform(0.4, 0.95), 3),
                vest_conf=round(random.uniform(0.4, 0.95), 3),
                person_conf=round(random.uniform(0.4, 0.95), 3),
                box=(random.randint(50, 400), random.randint(30, 200),
                     random.randint(500, 1100), random.randint(400, 700)),
                note="demo seed",
                when=when,
            )
            created += 1
    print(f"[seed] inserted {created} historical records over {days} days")


def main() -> None:
    p = argparse.ArgumentParser(description="Seed demo history for the website")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--reset", action="store_true", help="delete the database first")
    args = p.parse_args()

    if args.reset:
        reset()
    seed(args.days)

    print("[seed] open the website with:  python website/app.py")


if __name__ == "__main__":
    main()
