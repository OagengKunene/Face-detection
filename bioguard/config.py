"""Application defaults; runtime data lives outside the source repository."""

import os
from pathlib import Path

MODEL_ID = "opencv-sface-2021dec-v1"
THRESHOLD = 0.45
ENROLMENT_SAMPLES = 8
MATCH_FRAMES = 5
SAMPLE_INTERVAL = 0.35
CAPTURE_TIMEOUT = 25.0


def data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "BioGuard"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "bioguard"


def model_dir() -> Path:
    return data_dir() / "models"
