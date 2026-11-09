"""Download immutable, checksum-verified ONNX models from OpenCV Zoo."""

import hashlib
import os
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
BASE_URL = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models"


@dataclass(frozen=True)
class Model:
    folder: str
    filename: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.folder}/{self.filename}"


MODELS = (
    Model(
        "face_detection_yunet",
        "face_detection_yunet_2023mar.onnx",
        232589,
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    Model(
        "face_recognition_sface",
        "face_recognition_sface_2021dec.onnx",
        38696353,
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
)


def valid_model(path: Path, spec: Model) -> bool:
    if not path.is_file() or path.stat().st_size != spec.size:
        return False
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest == spec.sha256


def ensure_models(directory: Path, progress: Callable[[str], None] = print) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in MODELS:
        target = directory / spec.filename
        if not valid_model(target, spec):
            progress(f"Downloading {spec.filename} ({spec.size / 1_000_000:.1f} MB)...")
            request = urllib.request.Request(spec.url, headers={"User-Agent": "Face-detection/1.0"})
            temporary: Path | None = None
            try:
                with (
                    urllib.request.urlopen(request, timeout=45) as response,
                    tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as out,
                ):
                    temporary = Path(out.name)
                    received = 0
                    while chunk := response.read(65536):
                        received += len(chunk)
                        if received > spec.size:
                            raise ValueError(f"Unexpected model size: {spec.filename}")
                        out.write(chunk)
                if not valid_model(temporary, spec):
                    raise ValueError(f"Model checksum failed: {spec.filename}")
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        progress(f"Verified {spec.filename}")
        paths.append(target)
    return paths[0], paths[1]


def checked_paths(directory: Path) -> tuple[Path, Path]:
    paths = tuple(directory / spec.filename for spec in MODELS)
    if not all(valid_model(path, spec) for path, spec in zip(paths, MODELS, strict=True)):
        raise FileNotFoundError("Models are missing or damaged. Run: python run.py models")
    return paths[0], paths[1]
