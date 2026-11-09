"""YuNet detection, image-quality gates, alignment and SFace embeddings."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bioguard.decisions import unit_vector
from bioguard.models import checked_paths


@dataclass(frozen=True)
class Analysis:
    faces: np.ndarray
    message: str
    accepted: bool
    blur: float = 0.0
    brightness: float = 0.0


def assess_quality(frame: np.ndarray, faces: np.ndarray) -> Analysis:
    if len(faces) == 0:
        return Analysis(faces, "No face detected. Look towards the camera.", False)
    if len(faces) != 1:
        return Analysis(faces, "Multiple faces detected. Only one person may be in view.", False)
    face = faces[0]
    if face.shape != (15,) or not np.isfinite(face).all():
        return Analysis(faces, "Invalid face detection. Try again.", False)
    x, y, w, h = face[:4]
    height, width = frame.shape[:2]
    if min(w, h) < 100:
        return Analysis(faces, "Move closer: your face is too small.", False)
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return Analysis(faces, "Keep your whole face inside the image.", False)
    crop = frame[int(y) : int(y + h), int(x) : int(x + w)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    if not 45 <= brightness <= 210:
        return Analysis(
            faces, "Adjust the lighting: avoid darkness or strong glare.", False, blur, brightness
        )
    if blur < 45:
        return Analysis(faces, "Hold still: the face image is blurred.", False, blur, brightness)
    return Analysis(
        faces, "Face ready. Look forward with a relaxed expression.", True, blur, brightness
    )


class FaceEngine:
    def __init__(self, directory: Path):
        detector_path, recognizer_path = checked_paths(directory)
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (640, 480), 0.9, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def analyse(self, frame: np.ndarray) -> Analysis:
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) < 32
        ):
            raise ValueError("Expected a nonempty BGR uint8 camera frame.")
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        if faces is None:
            faces = np.empty((0, 15), dtype=np.float32)
        return assess_quality(frame, faces)

    def embedding(self, frame: np.ndarray, analysis: Analysis) -> np.ndarray:
        if not analysis.accepted or len(analysis.faces) != 1:
            raise ValueError("A single, good-quality face is required.")
        aligned = self.recognizer.alignCrop(frame, analysis.faces[0])
        return unit_vector(self.recognizer.feature(aligned))
