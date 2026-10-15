#!/usr/bin/env python
"""PHASE 1 - Human (person) detection on a construction site.

OpenCV captures the frames, YOLOv8 finds every worker and draws a box + label.

    python phase1_detect.py --source 0                 # live webcam
    python phase1_detect.py --source assets/site.mp4   # recorded video
    python phase1_detect.py --source assets/site.mp4 --save out.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from safety.config import CONF_THRESHOLD, DEFAULT_WEIGHTS, ensure_dirs
from safety.detector import SafetyDetector
from safety.visualize import draw_person_boxes

WANTED = ["Person"]


def gui_available() -> bool:
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except Exception:
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 - person detection")
    p.add_argument("--source", default="0",
                   help="camera index (0) or path to a video/image")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    p.add_argument("--save", default=None, help="write annotated video to this file")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = forever)")
    p.add_argument("--no-show", action="store_true", help="do not open a preview window")
    p.add_argument("--camera-width", type=int, default=1280)
    p.add_argument("--camera-height", type=int, default=720)
    p.add_argument("--imgsz", type=int, default=512,
                   help="inference size - lower is faster")
    return p.parse_args()


def open_capture(source: str, width: int, height: int):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap, f"camera:{source}"
    if not Path(source).exists():
        sys.exit(f"[phase1] source not found: {source}")
    return cv2.VideoCapture(source), source


def main() -> None:
    args = parse_args()
    ensure_dirs()

    detector = SafetyDetector(weights=args.weights, conf=args.conf, person_only=True,
                              imgsz=args.imgsz)
    print(f"[phase1] person model loaded -> {detector.person_weights}")

    cap, label = open_capture(args.source, args.camera_width, args.camera_height)
    if not cap.isOpened():
        sys.exit(f"[phase1] cannot open {label}")

    show = (not args.no_show) and gui_available()

    writer = None
    fps = 0.0
    prev = time.time()
    total_persons = 0
    frames = 0
    window = "Phase 1 - Person Detection (OpenCV + YOLOv8)"

    print(f"[phase1] reading from {label}  (press q to quit)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            analysis = detector.analyze(frame)
            total_persons += analysis.persons
            out = draw_person_boxes(frame, analysis)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6)) if fps else 1.0 / max(now - prev, 1e-6)
            prev = now

            if args.save:
                if writer is None:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, 25.0,
                                             (out.shape[1], out.shape[0]))
                writer.write(out)

            if show:
                cv2.imshow(window, out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if frames % 30 == 0:
                print(f"[phase1] frame {frames} | persons={analysis.persons} | fps={fps:.1f}")
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"[phase1] done. frames={frames} persons_seen={total_persons}")
    if args.save:
        print(f"[phase1] annotated video -> {args.save}")


if __name__ == "__main__":
    main()
