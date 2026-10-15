#!/usr/bin/env python
"""Downloads the two YOLOv8 models the project needs (once).

    python tools/fetch_weights.py

* weights/ppe_yolov8s.pt  - 13-class construction PPE model
                            (Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest,
                             Person, gloves, goggles, mask, harness, fall)
* weights/yolov8n.pt      - COCO person detector used for Phase 1
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from safety.config import DEFAULT_WEIGHTS, WEIGHTS_DIR, ensure_dirs  # noqa: E402

PPE_REPO = "ayushgupta7777/safetyvision-yolov8"
PPE_FILE = "v2/best.pt"
COCO_URL = ("https://github.com/ultralytics/assets/releases/download/v8.4.0/"
            "yolov8n.pt")


def fetch_ppe() -> Path:
    if DEFAULT_WEIGHTS.exists():
        print(f"[weights] PPE model already present -> {DEFAULT_WEIGHTS}")
        return DEFAULT_WEIGHTS

    print(f"[weights] downloading {PPE_REPO}:{PPE_FILE} ...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("[weights] huggingface_hub is required: pip install huggingface_hub")

    src = hf_hub_download(repo_id=PPE_REPO, filename=PPE_FILE)
    shutil.copy(src, DEFAULT_WEIGHTS)
    print(f"[weights] saved -> {DEFAULT_WEIGHTS}")
    return DEFAULT_WEIGHTS


def fetch_coco() -> Path:
    target = WEIGHTS_DIR / "yolov8n.pt"
    if target.exists():
        print(f"[weights] person model already present -> {target}")
        return target

    print("[weights] downloading yolov8n.pt (COCO person detector) ...")
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(COCO_URL, target)
    except Exception as exc:  # noqa: BLE001
        # ultralytics will download it automatically on first use
        print(f"[weights] direct download failed ({exc}) - ultralytics will fetch it")
        return target
    print(f"[weights] saved -> {target}")
    return target


def main() -> None:
    ensure_dirs()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    fetch_ppe()
    fetch_coco()

    from ultralytics import YOLO

    for path in (DEFAULT_WEIGHTS, WEIGHTS_DIR / "yolov8n.pt"):
        if path.exists():
            names = YOLO(str(path)).names
            print(f"[weights] {path.name}: {len(names)} classes")


if __name__ == "__main__":
    main()
