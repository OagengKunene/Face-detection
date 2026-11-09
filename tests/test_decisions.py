import numpy as np
import pytest

from bioguard.decisions import MatchWindow, cosine, make_template, unit_vector, validate_id


@pytest.mark.parametrize("value", ["ab", "../admin", "=FORMULA", "id space", "x" * 33, ""])
def test_invalid_ids(value):
    with pytest.raises(ValueError):
        validate_id(value)


def test_valid_id_is_trimmed():
    assert validate_id(" KUNENE_01-a ") == "KUNENE_01-a"


@pytest.mark.parametrize(
    "value", [np.zeros(128), np.ones(127), np.full(128, np.nan), np.full(128, np.inf)]
)
def test_invalid_embeddings(value):
    with pytest.raises(ValueError):
        unit_vector(value)


def test_normalization_handles_large_finite_values():
    value = unit_vector(np.full(128, 1e30))
    assert np.isfinite(value).all()
    assert np.linalg.norm(value) == pytest.approx(1)


def test_similarity_and_aggregation(vector):
    other = np.eye(128, dtype=np.float32)[1]
    template = make_template([vector] * 8)
    assert cosine(template, vector) == pytest.approx(1)
    assert cosine(template, other) == pytest.approx(0)
    assert cosine(template, -vector) == pytest.approx(-1)


def test_mixed_identities_are_rejected(vector):
    other = np.eye(128, dtype=np.float32)[1]
    with pytest.raises(ValueError, match="inconsistent"):
        make_template([vector] * 4 + [other] * 4)


def test_incomplete_enrolment_is_rejected(vector):
    with pytest.raises(ValueError):
        make_template([vector] * 7)


def test_five_spaced_frames_are_required():
    window = MatchWindow(started=0)
    for step in range(4):
        assert window.observe(0.8, step * 0.25) == "waiting"
    assert window.observe(0.8, 1.0) == "passed"
    assert window.observe(None, 100) == "passed"  # final decisions cannot be changed


def test_repeated_fast_frames_do_not_count():
    window = MatchWindow(started=0)
    for step in range(10):
        window.observe(0.9, step * 0.01)
    assert window.matches == 1


@pytest.mark.parametrize("score", [None, 0.1, float("nan"), float("inf"), 1.5, -0.2])
def test_bad_or_missing_scores_reset_streak(score):
    window = MatchWindow(started=0)
    window.observe(0.9, 0)
    window.observe(0.9, 0.25)
    assert window.matches == 2
    assert window.observe(score, 0.5) == "waiting"
    assert window.matches == 0


def test_timeout_is_checked_before_acceptance():
    window = MatchWindow(started=0, timeout=1)
    for now in [0, 0.25, 0.5, 0.75]:
        window.observe(0.8, now)
    assert window.observe(0.8, 1.0) == "failed"


@pytest.mark.parametrize("threshold", [-1, 0, 1, 1.1, float("nan")])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        MatchWindow(threshold=threshold)
