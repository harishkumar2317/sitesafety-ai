#!/usr/bin/env python
"""PHASE 2 - Safety feature detection + violation data collection.

For every worker the system decides:
    helmet  = worn / missing / unknown
    vest    = worn / missing / unknown
A violation is photographed and written to SQLite (data/safety.db) so that the
Phase 3 website can show who broke the rule, their picture and the daily total.

    python phase2_monitor.py --source 0                       # live webcam
    python phase2_monitor.py --source assets/site.mp4 --save out.mp4
    python phase2_monitor.py --source assets/photos --strict   # folder of images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from safety.config import CONF_THRESHOLD, DEFAULT_WEIGHTS, ensure_dirs
from safety.detector import SafetyDetector
from safety.pipeline import SafetyMonitor
from safety.visualize import draw_safety_boxes, hconcat_pad, side_panel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def gui_available() -> bool:
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except Exception:
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 - PPE / safety monitoring")
    p.add_argument("--source", default="0",
                   help="camera index (0), video file, or a folder of images")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    p.add_argument("--strict", action="store_true",
                   help="treat 'not detected' helmet/vest as a violation too")
    p.add_argument("--cooldown", type=float, default=6.0,
                   help="seconds before the same worker can be logged again")
    p.add_argument("--no-save-images", action="store_true")
    p.add_argument("--save", default=None, help="write annotated video here")
    p.add_argument("--label", default="", help="free-text label stored with the session")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--camera-width", type=int, default=1280)
    p.add_argument("--camera-height", type=int, default=720)
    p.add_argument("--imgsz", type=int, default=512,
                   help="inference size - lower is faster")
    return p.parse_args()


def iter_sources(source: str):
    """Yield (frame_provider_kind, payload, label)."""
    path = Path(source)
    if source.isdigit():
        yield "camera", int(source), f"camera:{source}"
    elif path.is_dir():
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit(f"[phase2] no images in {source}")
        yield "images", files, path.name
    elif path.exists() and path.suffix.lower() in IMAGE_EXTS:
        yield "images", [path], path.stem
    elif path.exists() and path.suffix.lower() in VIDEO_EXTS:
        yield "video", str(path), path.name
    else:
        sys.exit(f"[phase2] unsupported source: {source}")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    detector = SafetyDetector(weights=args.weights, conf=args.conf, strict=args.strict,
                              imgsz=args.imgsz)
    print(f"[phase2] model -> {args.weights} | strict={args.strict}")

    kind, payload, label = next(iter_sources(args.source))

    monitor = SafetyMonitor(
        detector=detector,
        source=str(args.source) if not str(args.source).isdigit() else f"camera:{args.source}",
        cooldown=args.cooldown,
        save_images=not args.no_save_images,
        label=args.label,
        independent_frames=(kind == "images"),
    )
    print(f"[phase2] session #{monitor.session_id} source={label}")

    show = not args.no_show and gui_available()
    writer_holder = [None]
    frames = 0
    window = "Phase 2 - Safety Monitoring (OpenCV + YOLOv8)"

    def emit(frame, analysis, new_rows):
        out = draw_safety_boxes(frame, analysis)
        for row in new_rows:
            print(f"[phase2] VIOLATION track#{row['track_id']} kind={row['kind']} "
                  f"-> id={row['id']} image={row['image']}")
        if kind == "images":
            out = hconcat_pad([out, side_panel(analysis)])
        if args.save:
            if writer_holder[0] is None:
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_holder[0] = cv2.VideoWriter(
                    args.save, fourcc, 25.0, (out.shape[1], out.shape[0]))
            writer_holder[0].write(out)
        if show:
            cv2.imshow(window, out)
            delay = 0 if kind == "images" else 1
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                return False
            if kind == "images" and key in (ord("n"), 32):
                return False
        return True

    try:
        if kind == "camera":
            cap = cv2.VideoCapture(payload)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
            if not cap.isOpened():
                sys.exit("[phase2] cannot open camera")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                analysis, new_rows = monitor.step(frame)
                if not emit(frame, analysis, new_rows):
                    break
                if frames % 30 == 0:
                    _progress(frames, monitor)
                if args.max_frames and frames >= args.max_frames:
                    break
            cap.release()

        elif kind == "video":
            cap = cv2.VideoCapture(payload)
            if not cap.isOpened():
                sys.exit(f"[phase2] cannot open {payload}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                analysis, new_rows = monitor.step(frame)
                if not emit(frame, analysis, new_rows):
                    break
                if frames % 30 == 0:
                    _progress(frames, monitor)
                if args.max_frames and frames >= args.max_frames:
                    break
            cap.release()

        else:
            for file in payload:
                frame = cv2.imread(str(file))
                if frame is None:
                    continue
                frames += 1
                analysis, new_rows = monitor.step(frame)
                if not emit(frame, analysis, new_rows):
                    break
                if args.max_frames and frames >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        monitor.close()
        if writer_holder[0] is not None:
            writer_holder[0].release()
        if show:
            cv2.destroyAllWindows()

    print("\n[phase2] ===== SESSION SUMMARY =====")
    print(f"[phase2] frames processed   : {monitor.frames}")
    print(f"[phase2] worker detections  : {monitor.persons_total}")
    print(f"[phase2] violations stored  : {monitor.violation_total}")
    print("[phase2] database           : data/safety.db")
    print("[phase2] photos             : data/violations/")
    print("[phase2] run 'python website/app.py' to open the central website")


def _progress(frames: int, monitor: SafetyMonitor) -> None:
    print(f"[phase2] frame={frames} workers_total={monitor.persons_total} "
          f"violations={monitor.violation_total}")


if __name__ == "__main__":
    main()
