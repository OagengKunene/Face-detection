"""Opt-in real OpenCV inference. Fixtures stay local and are not distributed."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from bioguard.decisions import MatchWindow, cosine, make_template
from bioguard.storage import Vault
from bioguard.vision import FaceEngine

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def assets():
    paths = [
        os.environ.get(key) for key in ("BIOGUARD_MODELS", "BIOGUARD_FACE_A", "BIOGUARD_FACE_B")
    ]
    if not all(paths):
        pytest.skip(
            "Set BIOGUARD_MODELS, BIOGUARD_FACE_A and BIOGUARD_FACE_B for real-model tests."
        )
    engine = FaceEngine(Path(paths[0]))
    frames = [cv2.imread(path) for path in paths[1:]]
    assert all(frame is not None for frame in frames)
    cv2.setNumThreads(2)
    return engine, frames


def test_real_enrol_verify_nonmatch_and_delete(assets, tmp_path):
    engine, (a, b) = assets
    # The original public sample has a face smaller than our 100px quality gate.
    # Resize only this fixture; the production quality gate stays unchanged.
    a = cv2.resize(a, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    augmentations = [
        a,
        cv2.convertScaleAbs(a, alpha=0.95, beta=3),
        cv2.convertScaleAbs(a, alpha=1.03, beta=1),
        cv2.GaussianBlur(a, (3, 3), 0.3),
    ]
    samples = []
    for frame in augmentations * 2:
        analysis = engine.analyse(frame)
        assert analysis.accepted, analysis.message
        samples.append(engine.embedding(frame, analysis))
    template = make_template(samples)
    other = engine.analyse(b)
    assert other.accepted, other.message
    impostor = cosine(template, engine.embedding(b, other))
    genuine = cosine(template, samples[2])
    assert genuine >= 0.45
    assert impostor < 0.45
    with Vault(tmp_path / "real-test.sqlite3", "temporary test passphrase", create=True) as vault:
        vault.add_profile("FIXTURE01", "Temporary test profile", template, 8, consent=True)
        for similarity, expected in [(genuine, "passed"), (impostor, "failed")]:
            window = MatchWindow(started=0, timeout=2)
            for step in range(10):
                window.observe(similarity, step * 0.25)
            assert window.result == expected
            vault.record_verification(
                "FIXTURE01", passed=expected == "passed", score=similarity, threshold=0.45
            )
        assert len(vault.recent_events()) == 3
        vault.delete_profile("FIXTURE01")
        assert not vault.profiles()
    print(json.dumps({"genuine_similarity": genuine, "nonmatch_similarity": impostor}))


def test_real_blank_and_two_face_frames_are_rejected(assets):
    engine, (a, b) = assets
    blank = engine.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(blank.faces) == 0 and not blank.accepted
    a = cv2.resize(a, (512, 512))
    b = cv2.resize(b, (512, 512))
    result = engine.analyse(np.hstack([a, b]))
    assert len(result.faces) == 2
    assert not result.accepted and "Multiple" in result.message
