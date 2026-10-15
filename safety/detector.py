"""Phase 1 + Phase 2 core: YOLOv8 person detection and PPE (helmet/vest) analysis.

Two models are used:

* ``yolov8n.pt`` (COCO)  -> reliable full-body **person** boxes   (Phase 1)
* ``ppe_yolov8s.pt``     -> Hardhat / NO-Hardhat / Safety Vest /
                            NO-Safety Vest boxes                   (Phase 2)

The PPE boxes are then attached to the person whose body they sit on, which
gives an accurate "worker -> helmet / vest" decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import (
    CONF_THRESHOLD,
    HARDHAT,
    HELMET_MISSING,
    HELMET_UNKNOWN,
    HELMET_WORN,
    IOU_THRESHOLD,
    NO_HARDHAT,
    NO_VEST,
    PERSON,
    VEST,
    DEFAULT_WEIGHTS,
    WEIGHTS_DIR,
)
from . import geometry as g

PERSON_WEIGHTS = WEIGHTS_DIR / "yolov8n.pt"
COCO_PERSON_ID = 0

HELMET_CLASSES = (HARDHAT, NO_HARDHAT)
VEST_CLASSES = (VEST, NO_VEST)


@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: tuple

    @property
    def box(self) -> tuple:
        return self.xyxy


@dataclass
class Worker:
    """One detected human together with the PPE state derived for them."""

    box: tuple
    person_conf: float
    helmet: str = HELMET_UNKNOWN
    vest: str = HELMET_UNKNOWN
    helmet_conf: float = 0.0
    vest_conf: float = 0.0
    helmet_box: Optional[tuple] = None
    vest_box: Optional[tuple] = None
    track_id: int = -1

    @property
    def helmet_missing(self) -> bool:
        return self.helmet == HELMET_MISSING

    @property
    def vest_missing(self) -> bool:
        return self.vest == HELMET_MISSING

    @property
    def violation_kinds(self) -> List[str]:
        kinds = []
        if self.helmet_missing:
            kinds.append("helmet")
        if self.vest_missing:
            kinds.append("vest")
        return kinds

    @property
    def is_violation(self) -> bool:
        return bool(self.violation_kinds)

    @property
    def violation_kind(self) -> str:
        kinds = self.violation_kinds
        return "+".join(kinds) if kinds else ""

    @property
    def status(self) -> str:
        if self.is_violation:
            return "VIOLATION"
        if HELMET_UNKNOWN in (self.helmet, self.vest):
            return "CHECK"
        return "SAFE"


@dataclass
class Analysis:
    workers: List[Worker] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    frame_shape: tuple = (0, 0, 3)

    @property
    def persons(self) -> int:
        return len(self.workers)

    @property
    def violations(self) -> List[Worker]:
        return [w for w in self.workers if w.is_violation]

    @property
    def helmets_missing(self) -> int:
        return sum(1 for w in self.workers if w.helmet_missing)

    @property
    def vests_missing(self) -> int:
        return sum(1 for w in self.workers if w.vest_missing)


class SafetyDetector:
    """Wraps the YOLOv8 person + PPE models and turns boxes into worker states."""

    def __init__(
        self,
        weights: str = None,
        person_weights: str = None,
        conf: float = CONF_THRESHOLD,
        iou: float = IOU_THRESHOLD,
        strict: bool = False,
        device: str = None,
        person_only: bool = False,
        use_person_model: bool = True,
        imgsz: int = 512,
        max_det: int = 100,
    ):
        from ultralytics import YOLO

        self.conf = conf
        self.iou = iou
        self.strict = strict
        self.device = device
        self.imgsz = imgsz
        self.max_det = max_det
        self.person_only = person_only
        self.weights_path = str(weights or DEFAULT_WEIGHTS)
        self.ppe_model = None if person_only else YOLO(self.weights_path)

        self.person_weights = str(person_weights or PERSON_WEIGHTS)
        self.person_model = None
        if use_person_model:
            try:
                self.person_model = YOLO(self.person_weights)
            except Exception:
                self.person_model = None

        if self.ppe_model is not None:
            self.names = self.ppe_model.names
        else:
            self.names = self.person_model.names

    # ------------------------------------------------------------------ #
    # raw inference
    # ------------------------------------------------------------------ #
    def _run(self, model, frame: np.ndarray, classes=None) -> List[Detection]:
        results = model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
            classes=classes,
            imgsz=self.imgsz,
            max_det=self.max_det,
        )
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        names = model.names
        for xyxy, conf, c in zip(boxes, confs, clss):
            out.append(
                Detection(
                    cls=names.get(int(c), str(c)),
                    conf=float(conf),
                    xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                )
            )
        return out

    def infer_ppe(self, frame: np.ndarray) -> List[Detection]:
        if self.ppe_model is None:
            return []
        return self._run(self.ppe_model, frame)

    def infer_persons(self, frame: np.ndarray) -> List[Detection]:
        if self.person_model is not None:
            return self._run(self.person_model, frame, classes=[COCO_PERSON_ID])
        if self.ppe_model is None:
            return []
        return [d for d in self._run(self.ppe_model, frame) if d.cls.lower() == "person"]

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Everything the models see (person boxes first)."""
        return self.infer_persons(frame) + self.infer_ppe(frame)

    # ------------------------------------------------------------------ #
    # PPE association
    # ------------------------------------------------------------------ #
    @staticmethod
    def _best_inside(dets, region) -> Optional[Detection]:
        best = None
        for d in dets:
            if not g.center_inside(d.xyxy, region):
                continue
            if best is None or d.conf > best.conf:
                best = d
        return best

    def _state(self, person: Detection, dets: List[Detection]):
        head = g.sub_box(person.xyxy, 0.0, 0.34, 0.10, 0.90)
        torso = g.sub_box(person.xyxy, 0.22, 0.80, 0.0, 1.0)

        pos_hit = self._best_inside([d for d in dets if d.cls == HARDHAT], head) \
            or self._best_inside([d for d in dets if d.cls == HARDHAT], person.xyxy)
        neg_hit = self._best_inside([d for d in dets if d.cls == NO_HARDHAT], head) \
            or self._best_inside([d for d in dets if d.cls == NO_HARDHAT], person.xyxy)

        if neg_hit and (not pos_hit or neg_hit.conf >= pos_hit.conf):
            helmet, hconf, hbox = HELMET_MISSING, neg_hit.conf, neg_hit.xyxy
        elif pos_hit:
            helmet, hconf, hbox = HELMET_WORN, pos_hit.conf, pos_hit.xyxy
        else:
            helmet = HELMET_MISSING if self.strict else HELMET_UNKNOWN
            hconf, hbox = 0.0, None

        vpos = self._best_inside([d for d in dets if d.cls == VEST], torso) \
            or self._best_inside([d for d in dets if d.cls == VEST], person.xyxy)
        vneg = self._best_inside([d for d in dets if d.cls == NO_VEST], torso) \
            or self._best_inside([d for d in dets if d.cls == NO_VEST], person.xyxy)

        if vneg and (not vpos or vneg.conf >= vpos.conf):
            vest, vconf, vbox = HELMET_MISSING, vneg.conf, vneg.xyxy
        elif vpos:
            vest, vconf, vbox = HELMET_WORN, vpos.conf, vpos.xyxy
        else:
            vest = HELMET_MISSING if self.strict else HELMET_UNKNOWN
            vconf, vbox = 0.0, None

        return Worker(
            box=person.xyxy,
            person_conf=person.conf,
            helmet=helmet,
            vest=vest,
            helmet_conf=hconf,
            vest_conf=vconf,
            helmet_box=hbox,
            vest_box=vbox,
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def analyze(self, frame: np.ndarray) -> Analysis:
        if self.person_only:
            persons = self.infer_persons(frame)
            workers = [Worker(box=p.xyxy, person_conf=p.conf) for p in persons]
            workers.sort(key=lambda w: -w.person_conf)
            return Analysis(workers=workers, detections=persons, frame_shape=frame.shape)

        person_dets = self.infer_persons(frame)
        ppe_dets = self.infer_ppe(frame)
        dets = person_dets + ppe_dets

        persons = [d for d in person_dets if d.cls.lower() == "person"]
        if not persons:
            persons = [d for d in ppe_dets if d.cls.lower() == "person"]
        if not persons:
            persons = self._merge_ppe_into_persons(ppe_dets)
        else:
            persons = persons + self._uncovered_persons(ppe_dets, persons)

        workers = [self._state(p, dets) for p in persons]
        workers.sort(key=lambda w: -w.person_conf)
        return Analysis(workers=workers, detections=dets, frame_shape=frame.shape)

    @classmethod
    def _uncovered_persons(cls, ppe: List[Detection], persons: List[Detection]) -> List[Detection]:
        """Person boxes rebuilt around PPE that sits outside every known person."""
        loose = [d for d in ppe
                 if not any(g.center_inside(d.xyxy, p.xyxy) for p in persons)]
        out = []
        for cand in cls._merge_ppe_into_persons(loose):
            if any(g.iou(cand.xyxy, p.xyxy) > 0.35 for p in persons):
                continue
            out.append(cand)
        return out

    @staticmethod
    def _merge_ppe_into_persons(ppe: List[Detection]) -> List[Detection]:
        """Fallback: rebuild a rough person box around clustered PPE boxes."""
        groups: List[List[Detection]] = []
        for d in sorted(ppe, key=lambda x: -x.conf):
            placed = False
            for grp in groups:
                anchor = grp[0]
                if g.iou(d.xyxy, anchor.xyxy) > 0.12 or g.center_inside(d.xyxy, anchor.xyxy) \
                        or g.center_inside(anchor.xyxy, d.xyxy):
                    grp.append(d)
                    placed = True
                    break
            if not placed:
                groups.append([d])

        persons = []
        for grp in groups:
            x1 = min(d.xyxy[0] for d in grp)
            y1 = min(d.xyxy[1] for d in grp)
            x2 = max(d.xyxy[2] for d in grp)
            y2 = max(d.xyxy[3] for d in grp)
            h = max(y2 - y1, 1.0)
            box = (max(0, x1 - h * 0.25), max(0, y1 - h * 0.45),
                   x2 + h * 0.25, y2 + h * 1.05)
            conf = max(d.conf for d in grp)
            persons.append(Detection(cls=PERSON, conf=conf, xyxy=box))
        return persons
