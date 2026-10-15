"""OpenCV drawing helpers for phase 1 / phase 2 overlays."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .config import (
    COLOR_BAD,
    COLOR_OK,
    COLOR_PERSON,
    COLOR_TEXT,
    COLOR_WARN,
    HELMET_UNKNOWN,
)
from . import geometry as g
from .detector import Analysis, Worker

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, org, color, scale=0.5, thickness=1, bg=None):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_person_boxes(frame: np.ndarray, analysis: Analysis) -> np.ndarray:
    """Phase 1: humans only."""
    img = frame.copy()
    for w in analysis.workers:
        x1, y1, x2, y2 = g.to_int(w.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PERSON, 2)
        _label(img, f"Person {int(w.person_conf*100)}%", (x1, y1), COLOR_PERSON, 0.5, 1,
               bg=(60, 40, 0))
    hud = f"Persons detected: {analysis.persons}"
    cv2.rectangle(img, (0, 0), (330, 34), (0, 0, 0), -1)
    _label(img, hud, (8, 25), (0, 255, 255), 0.65, 2)
    return img


def _status_color(w: Worker) -> tuple:
    if w.is_violation:
        return COLOR_BAD
    if HELMET_UNKNOWN in (w.helmet, w.vest):
        return COLOR_WARN
    return COLOR_OK


def draw_safety_boxes(frame: np.ndarray, analysis: Analysis) -> np.ndarray:
    """Phase 2: colour coded PPE status per worker."""
    img = frame.copy()

    for w in analysis.workers:
        color = _status_color(w)
        x1, y1, x2, y2 = g.to_int(w.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        head = g.to_int(g.sub_box(w.box, 0.0, 0.34, 0.10, 0.90))
        if w.is_violation:
            cv2.rectangle(img, (head[0], head[1]), (head[2], head[3]), color, 2)

        tag = "OK" if w.status == "SAFE" else w.status
        tid = f"#{w.track_id}" if w.track_id > 0 else ""
        _label(img, f"{tid} {tag}", (x1, y1), color, 0.55, 2, bg=(0, 0, 0))

        hx = _ppe_tag("Helmet", w.helmet)
        vx = _ppe_tag("Vest", w.vest)
        yy = y2 + 16
        _label(img, hx, (x1, yy), color, 0.45, 1, bg=(0, 0, 0))
        _label(img, vx, (x1, yy + 17), color, 0.45, 1, bg=(0, 0, 0))

        if w.is_violation:
            if w.helmet_box:
                hb = g.to_int(w.helmet_box)
                cv2.rectangle(img, (hb[0], hb[1]), (hb[2], hb[3]), color, 2)
            if w.vest_box:
                vb = g.to_int(w.vest_box)
                cv2.rectangle(img, (vb[0], vb[1]), (vb[2], vb[3]), color, 2)

    _draw_hud(img, analysis)
    return img


def _ppe_tag(name: str, state: str) -> str:
    mark = {"worn": "[+]", "missing": "[!]", "unknown": "[?]"}[state]
    return f"{name} {mark} {state.upper()}"


def _draw_hud(img: np.ndarray, analysis: Analysis, fps: Optional[float] = None) -> None:
    viol = len(analysis.violations)
    helmet = analysis.helmets_missing
    vest = analysis.vests_missing
    panel_w = 430
    cv2.rectangle(img, (0, 0), (panel_w, 62), (10, 10, 10), -1)
    _label(img, f"Workers: {analysis.persons}    Violations: {viol}",
           (8, 22), (0, 255, 255), 0.6, 2)
    _label(img, f"No helmet: {helmet}    No vest: {vest}", (8, 46),
           COLOR_BAD if (helmet or vest) else COLOR_OK, 0.55, 2)
    if fps:
        _label(img, f"{fps:.1f} FPS", (panel_w + 8, 22), COLOR_TEXT, 0.55, 1)


def side_panel(analysis: Analysis, width: int = 340) -> np.ndarray:
    """Right-hand list of worker states (handy for demos / screenshots)."""
    rows = max(len(analysis.workers), 1)
    h = 60 + rows * 34
    panel = np.full((h, width, 3), (24, 24, 24), dtype=np.uint8)
    _label(panel, "WORKER   HELMET   VEST", (10, 26), (0, 255, 255), 0.6, 2)
    y = 58
    for w in analysis.workers:
        color = _status_color(w)
        name = f"#{w.track_id if w.track_id > 0 else '-'}"
        _label(panel, f"{name:<8}", (10, y), color, 0.5, 1)
        _label(panel, f"{w.helmet:<9}", (95, y), color, 0.5, 1)
        _label(panel, f"{w.vest:<9}", (195, y), color, 0.5, 1)
        y += 34
    if not analysis.workers:
        _label(panel, "no worker detected", (10, y), COLOR_WARN, 0.5, 1)
    return panel


def hconcat_pad(images: List[np.ndarray]) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    padded = []
    for i in images:
        if i.shape[0] < h:
            pad = np.zeros((h - i.shape[0], i.shape[1], 3), dtype=i.dtype)
            i = np.vstack([i, pad])
        padded.append(i)
    return np.hstack(padded)
