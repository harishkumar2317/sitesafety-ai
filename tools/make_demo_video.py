#!/usr/bin/env python
"""Builds a synthetic construction-site camera feed from still photos.

Useful for demonstrating Phase 2 tracking without a real webcam:

    python tools/make_demo_video.py
    python phase2_monitor.py --source data/uploads/demo_site_feed.mp4

The camera slowly pans/zooms over each photo so the same worker stays in view
across many frames - which is what exercises the tracker and de-duplication.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from safety.config import ASSETS_DIR, DATA_DIR, ensure_dirs  # noqa: E402

W, H = 1280, 720
FPS = 25
SECONDS_PER_CLIP = 2

# put the frames that are known to contain PPE breaches first so a short
# run still produces violations
PRIORITY = ("image1276", "image140", "site1")


def load_images(folder: Path, limit: int = 6):
    files = []
    for pat in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(sorted(folder.glob(pat)))
    files = [f for f in files if "preview" not in f.name][:limit]
    imgs = []
    for f in files:
        im = cv2.imread(str(f))
        if im is not None:
            imgs.append((f.stem, im))
    return imgs


def pan_zoom(img: np.ndarray, t: float) -> np.ndarray:
    """t in [0,1] -> a slow diagonal drift + gentle zoom."""
    ih, iw = img.shape[:2]
    zoom = 1.0 + 0.16 * t
    cw, ch = iw / zoom, ih / zoom
    x0 = (iw - cw) * (0.15 + 0.7 * t)
    y0 = (ih - ch) * (0.75 - 0.6 * t)
    x1, y1 = x0 + cw, y0 + ch
    if x1 > iw:
        x0, x1 = iw - cw, iw
    if y1 > ih:
        y0, y1 = ih - ch, ih
    if y0 < 0:
        y0, y1 = 0, ch
    crop = img[int(y0):int(y1), int(x0):int(x1)]
    if crop.size == 0:
        crop = img
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)


def main() -> None:
    ensure_dirs()
    out = DATA_DIR / "uploads" / "demo_site_feed.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    sources = []
    for folder in (ASSETS_DIR, ASSETS_DIR / "testset"):
        if folder.exists():
            sources.extend(load_images(folder))
    if not sources:
        sys.exit("[demo] no source photos found in assets/")

    def rank(item):
        name = item[0]
        return (PRIORITY.index(name) if name in PRIORITY else len(PRIORITY), name)

    sources.sort(key=rank)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, FPS, (W, H))
    if not writer.isOpened():
        sys.exit("[demo] cannot create video writer")

    total = 0
    per = FPS * SECONDS_PER_CLIP
    for name, img in sources:
        for i in range(per):
            t = i / max(per - 1, 1)
            writer.write(pan_zoom(img, t))
            total += 1
        print(f"[demo] + {name}: {per} frames")
    writer.release()

    print(f"[demo] wrote {total} frames ({total/FPS:.0f}s) -> {out}")
    print("[demo] run:")
    print(f"  python phase2_monitor.py --source {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
