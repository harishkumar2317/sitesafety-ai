"""Phase 2 pipeline: detect -> track -> decide -> persist."""

from __future__ import annotations

import atexit
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .capture import save_violation_image
from .config import DB_PATH, VIOLATION_COOLDOWN_SEC, ensure_dirs
from .database import add_violation, finish_session, init_db, start_session
from .detector import Analysis, SafetyDetector
from .tracker import IOUTracker


class SafetyMonitor:
    """Turns raw frames into tracked, de-duplicated, stored violations."""

    def __init__(
        self,
        detector: Optional[SafetyDetector] = None,
        source: str = "webcam",
        cooldown: float = VIOLATION_COOLDOWN_SEC,
        save_images: bool = True,
        track: bool = True,
        label: str = "",
        db_path: Path = None,
        out_dir: Path = None,
        with_session: bool = True,
        independent_frames: bool = False,
    ):
        ensure_dirs()
        init_db()
        self.detector = detector or SafetyDetector()
        self.source = source
        self.cooldown = cooldown
        self.save_images = save_images
        self.track = track
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.out_dir = Path(out_dir) if out_dir else None
        self.independent_frames = independent_frames
        self.tracker = IOUTracker()
        self._last: Dict[int, tuple] = {}
        self.frames = 0
        self.persons_total = 0
        self.violation_total = 0
        self.session_id = start_session(source, label, self.db_path) if with_session else None
        self._closed = False
        if self.session_id:
            atexit.register(self.close)

    # ------------------------------------------------------------------ #
    def step(self, frame: np.ndarray) -> tuple:
        """Returns (analysis, list_of_newly_recorded_violation_rows)."""
        self.frames += 1
        analysis = self.detector.analyze(frame)

        if self.track and analysis.workers:
            boxes = [w.box for w in analysis.workers]
            tracks = self.tracker.update(boxes, analysis.workers)
            id_by_box = {}
            for tr in tracks:
                id_by_box[round(tr.box[0]), round(tr.box[1])] = tr.track_id
            for w in analysis.workers:
                key = (round(w.box[0]), round(w.box[1]))
                w.track_id = id_by_box.get(key, -1)
            self._sync_track_ids(tracks, analysis.workers)
        elif not analysis.workers:
            if self.track:
                self.tracker.update([])

        self.persons_total += analysis.persons
        new_rows = self._persist(analysis, frame)
        if self.independent_frames:
            # unrelated stills: keep id numbering but stop cross-image matching
            self.tracker.reset(keep_ids=True)
            self._last.clear()
        return analysis, new_rows

    @staticmethod
    def _sync_track_ids(tracks, workers) -> None:
        """Match workers to tracks by best IoU (bbox rounding is not reliable)."""
        from . import geometry as g

        for w in workers:
            best_id, best = -1, 0.0
            for tr in tracks:
                s = g.iou(w.box, tr.box)
                if s > best:
                    best, best_id = s, tr.track_id
            if best_id > 0:
                w.track_id = best_id

    # ------------------------------------------------------------------ #
    def _persist(self, analysis: Analysis, frame: np.ndarray) -> List[dict]:
        rows = []
        now = datetime.now()
        epoch = time.time()
        for w in analysis.violations:
            key = w.track_id if w.track_id > 0 else -id(w)
            kind = w.violation_kind
            prev = self._last.get(key)
            if prev is not None:
                prev_kind, prev_time = prev
                if prev_kind == kind and (epoch - prev_time) < self.cooldown:
                    continue
            image_path = None
            if self.save_images:
                path = save_violation_image(frame, analysis, w, self.source, self.violation_total + 1, self.out_dir)
                image_path = str(path) if path else None
            vid = add_violation(
                kind=kind,
                helmet=w.helmet,
                vest=w.vest,
                source=self.source,
                image_path=image_path,
                track_id=w.track_id,
                helmet_conf=w.helmet_conf,
                vest_conf=w.vest_conf,
                person_conf=w.person_conf,
                box=w.box,
                session_id=self.session_id,
                when=now,
                path=self.db_path,
            )
            self._last[key] = (kind, epoch)
            self.violation_total += 1
            rows.append({"id": vid, "kind": kind, "track_id": w.track_id, "image": image_path})
        return rows

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self._closed or not self.session_id:
            return
        self._closed = True
        try:
            finish_session(self.session_id, self.frames, self.persons_total,
                           self.violation_total, self.db_path)
        except Exception:
            pass

    def reset_tracking(self) -> None:
        self.tracker.reset()
        self._last.clear()
