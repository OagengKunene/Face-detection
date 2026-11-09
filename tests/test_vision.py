import numpy as np
import pytest

from bioguard.vision import FaceEngine, assess_quality


def scene():
    rng = np.random.default_rng(42)
    frame = rng.integers(70, 190, (480, 640, 3), dtype=np.uint8)
    face = np.array(
        [[100, 100, 200, 200, 160, 160, 240, 160, 200, 210, 160, 260, 240, 260, 0.99]],
        dtype=np.float32,
    )
    return frame, face


def test_quality_accepts_single_sharp_face():
    frame, face = scene()
    result = assess_quality(frame, face)
    assert result.accepted
    assert result.blur > 45


def test_quality_rejects_no_face():
    frame, _ = scene()
    assert not assess_quality(frame, np.empty((0, 15))).accepted


def test_quality_rejects_multiple_faces():
    frame, face = scene()
    result = assess_quality(frame, np.vstack([face, face]))
    assert not result.accepted and "Multiple" in result.message


@pytest.mark.parametrize(
    "box", [(100, 100, 50, 50), (-1, 100, 200, 200), (500, 100, 200, 200), (100, 400, 200, 200)]
)
def test_quality_rejects_small_or_clipped_face(box):
    frame, face = scene()
    face[0, :4] = box
    assert not assess_quality(frame, face).accepted


@pytest.mark.parametrize("brightness", [0, 20, 240, 255])
def test_quality_rejects_bad_lighting(brightness):
    frame, face = scene()
    frame[:] = brightness
    assert not assess_quality(frame, face).accepted


def test_quality_rejects_blurred_face():
    frame, face = scene()
    frame[:] = 128
    result = assess_quality(frame, face)
    assert not result.accepted and "blurred" in result.message


def test_quality_rejects_nonfinite_detection():
    frame, face = scene()
    face[0, 0] = np.nan
    assert not assess_quality(frame, face).accepted


@pytest.mark.parametrize(
    "frame",
    [None, np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((480, 640)), np.zeros((480, 640, 4))],
)
def test_engine_rejects_invalid_frame_before_inference(frame):
    engine = object.__new__(FaceEngine)
    with pytest.raises(ValueError):
        engine.analyse(frame)


def test_embedding_requires_quality_approval():
    frame, face = scene()
    rejected = assess_quality(frame, np.vstack([face, face]))
    with pytest.raises(ValueError):
        object.__new__(FaceEngine).embedding(frame, rejected)
