"""Saves the evidence image for every detected violation."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import VIOLATION_DIR, ensure_dirs
from . import geometry as g
from .detector import Worker
from .detector import Analysis


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in text)
    return keep.strip("_") or "src"


def crop_worker(frame: np.ndarray, w: Worker, pad: float = 0.18) -> np.ndarray:
    box = g.clip(g.expand(w.box, pad), frame.shape)
    x1, y1, x2, y2 = g.to_int(box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame
    return crop


def build_evidence(
    frame: np.ndarray,
    analysis: Analysis,
    w: Worker,
    label: str = "",
) -> np.ndarray:
    """Annotated full frame with the offending worker boxed in red."""
    img = frame.copy()
    color = (0, 0, 255)
    x1, y1, x2, y2 = g.to_int(w.box)
    for _ in range(3):
        cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), color, 2)

    head = g.to_int(g.sub_box(w.box, 0.0, 0.34, 0.10, 0.90))
    cv2.rectangle(img, (head[0], head[1]), (head[2], head[3]), color, 2)

    banner = f"VIOLATION: {label or w.violation_kind.replace('+', ' + ').upper()}"
    cv2.rectangle(img, (0, img.shape[0] - 54), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, banner, (14, img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, stamp, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    return img


def save_violation_image(
    frame: np.ndarray,
    analysis: Analysis,
    w: Worker,
    source: str,
    vid: int,
    out_dir: Path = None,
) -> Optional[Path]:
    ensure_dirs()
    out_dir = Path(out_dir or VIOLATION_DIR)
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.md5(
        f"{source}:{vid}:{w.box}:{stamp}".encode()
    ).hexdigest()[:6]
    name = f"{stamp}_{_slug(Path(source).stem)}_v{vid}_{digest}.jpg"
    path = out_dir / name

    evidence = build_evidence(frame, analysis, w)
    crop = crop_worker(frame, w)
    h, wdt = evidence.shape[:2]
    ch, cw = crop.shape[:2]
    scale = min(1.0, h / max(ch, 1))
    crop_r = cv2.resize(crop, (int(cw * scale), int(ch * scale)))

    canvas = np.zeros((max(h, crop_r.shape[0]), wdt + crop_r.shape[1], 3), dtype=np.uint8)
    canvas[:h, :wdt] = evidence
    canvas[:crop_r.shape[0], wdt:wdt + crop_r.shape[1]] = crop_r
    cv2.putText(canvas, "WORKER CROP", (wdt + 10, crop_r.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path
