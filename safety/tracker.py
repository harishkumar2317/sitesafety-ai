"""Lightweight IoU tracker so one worker is counted once, frame after frame."""

from __future__ import annotations

from typing import List, Optional

from . import geometry as g
from .config import TRACK_MAX_AGE


class Track:
    __slots__ = ("track_id", "box", "hits", "missed", "worker")

    def __init__(self, track_id: int, box, worker=None):
        self.track_id = track_id
        self.box = box
        self.hits = 1
        self.missed = 0
        self.worker = worker

    @property
    def centroid(self):
        return g.center(self.box)


class IOUTracker:
    """Assigns a stable integer id to each person seen across frames."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = TRACK_MAX_AGE):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[Track] = []
        self._next_id = 1

    @property
    def ids(self) -> set:
        return {t.track_id for t in self.tracks}

    def update(self, boxes: List[tuple], workers: Optional[List] = None) -> List[Track]:
        if workers is None:
            workers = [None] * len(boxes)

        pairs = []
        for ti, tr in enumerate(self.tracks):
            for bi, box in enumerate(boxes):
                score = g.iou(tr.box, box)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, bi))
        pairs.sort(key=lambda p: -p[0])

        matched_tracks: set = set()
        matched_boxes: set = set()
        for _score, ti, bi in pairs:
            if ti in matched_tracks or bi in matched_boxes:
                continue
            tr = self.tracks[ti]
            tr.box = boxes[bi]
            tr.worker = workers[bi]
            tr.hits += 1
            tr.missed = 0
            matched_tracks.add(ti)
            matched_boxes.add(bi)

        survivors: List[Track] = []
        for ti, tr in enumerate(self.tracks):
            if ti in matched_tracks:
                survivors.append(tr)
            else:
                tr.missed += 1
                if tr.missed <= self.max_age:
                    survivors.append(tr)

        for bi, box in enumerate(boxes):
            if bi in matched_boxes:
                continue
            survivors.append(Track(self._next_id, box, workers[bi]))
            self._next_id += 1

        self.tracks = survivors
        return list(self.tracks)

    def reset(self, keep_ids: bool = False) -> None:
        self.tracks = []
        if not keep_ids:
            self._next_id = 1
