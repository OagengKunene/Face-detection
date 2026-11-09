"""Pure numerical validation and temporal verification decisions."""

import math
import re
import time
from dataclasses import dataclass

import numpy as np

from bioguard.config import MATCH_FRAMES, THRESHOLD


def validate_id(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,31}", value):
        raise ValueError("Person ID: use 3-32 letters, digits, hyphens or underscores.")
    return value


def unit_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (128,) or not np.isfinite(vector).all():
        raise ValueError("Expected a finite SFace embedding with 128 values.")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("The face embedding is empty.")
    return (vector.astype(np.float64) / norm).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(unit_vector(a), unit_vector(b)), -1.0, 1.0))


def make_template(samples: list[np.ndarray], minimum: int = 8) -> np.ndarray:
    if not minimum <= len(samples) <= 30:
        raise ValueError(f"Enrolment requires {minimum}-30 accepted face samples.")
    vectors = np.stack([unit_vector(sample) for sample in samples])
    centre = unit_vector(vectors.mean(axis=0))
    if float(np.min(vectors @ centre)) < 0.65:
        raise ValueError("Face samples are inconsistent. Enrol one person in steady lighting.")
    if float(np.min(vectors @ vectors.T)) < 0.45:
        raise ValueError("Face samples are inconsistent. Enrol one person in steady lighting.")
    return centre


@dataclass
class MatchWindow:
    """A fail-closed streak of spaced frames; scores are similarities, not probabilities."""

    threshold: float = THRESHOLD
    required: int = MATCH_FRAMES
    interval: float = 0.20
    timeout: float = 15.0
    started: float | None = None
    matches: int = 0
    last_sample: float | None = None
    result: str = "waiting"

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("Threshold must be finite and between zero and one.")
        if self.required < 2 or self.interval <= 0 or self.timeout <= 0:
            raise ValueError("Invalid verification timing settings.")

    def observe(self, score: float | None, now: float | None = None) -> str:
        if self.result != "waiting":
            return self.result
        now = time.monotonic() if now is None else now
        if self.started is None:
            self.started = now
        if now - self.started >= self.timeout:
            self.result = "failed"
            return self.result
        if score is None or not math.isfinite(score) or not self.threshold <= score <= 1.00001:
            self.matches = 0
            self.last_sample = None
            return self.result
        if self.last_sample is None or now - self.last_sample >= self.interval:
            self.matches += 1
            self.last_sample = now
        if self.matches >= self.required:
            self.result = "passed"
        return self.result
