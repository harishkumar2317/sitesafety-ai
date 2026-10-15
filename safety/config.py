from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WEIGHTS_DIR = ROOT / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "ppe_yolov8s.pt"

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "safety.db"
VIOLATION_DIR = DATA_DIR / "violations"
ASSETS_DIR = ROOT / "assets"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"

MODEL_CLASSES = {
    0: "Fall-Detected",
    1: "Gloves",
    2: "Goggles",
    3: "Hardhat",
    4: "Mask",
    5: "NO-Gloves",
    6: "NO-Goggles",
    7: "NO-Hardhat",
    8: "NO-Mask",
    9: "NO-Safety Vest",
    10: "No_Harness",
    11: "Person",
    12: "Safety Vest",
}

PERSON = "Person"
HARDHAT = "Hardhat"
NO_HARDHAT = "NO-Hardhat"
VEST = "Safety Vest"
NO_VEST = "NO-Safety Vest"

# YOLO confidence / NMS defaults
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5

# A worker record is only re-created for the same track after this many seconds
VIOLATION_COOLDOWN_SEC = 6
# How long a lost track is kept alive (frames)
TRACK_MAX_AGE = 45

HELMET_WORN = "worn"
HELMET_MISSING = "missing"
HELMET_UNKNOWN = "unknown"

VIOLATION_KINDS = {
    "helmet": "Helmet missing",
    "vest": "Safety vest missing",
    "helmet+vest": "Helmet & safety vest missing",
}

COLOR_OK = (60, 200, 70)
COLOR_BAD = (50, 50, 235)
COLOR_WARN = (0, 170, 255)
COLOR_PERSON = (255, 200, 0)
COLOR_TEXT = (255, 255, 255)


def ensure_dirs() -> None:
    for path in (DATA_DIR, VIOLATION_DIR, UPLOAD_DIR, RESULT_DIR, ASSETS_DIR):
        path.mkdir(parents=True, exist_ok=True)
