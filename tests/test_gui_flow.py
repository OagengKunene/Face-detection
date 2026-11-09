"""Exercise real controller methods and vault writes without requiring a display."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from bioguard.config import ENROLMENT_SAMPLES
from bioguard.gui import Application


@pytest.fixture
def controller(vault, vector):
    app = object.__new__(Application)
    app.vault = vault
    app.camera = MagicMock()
    app.engine = MagicMock()
    app.engine.embedding.return_value = vector
    app.status = MagicMock()
    app.progress = MagicMock()
    app.enrol_button = MagicMock()
    app.verify_button = MagicMock()
    app.person_id = MagicMock()
    app.person_id.get.return_value = "TEST01"
    app.name = MagicMock()
    app.name.get.return_value = "Synthetic profile"
    app.selected = MagicMock()
    app.selected.get.return_value = "TEST01"
    app.consent = MagicMock()
    app.consent.get.return_value = True
    app.refresh = MagicMock()
    app.mode = None
    app.downloading = False
    app.samples = []
    app.window = None
    app.target = None
    return app


def test_complete_enrolment_saves_one_encrypted_profile(controller):
    app = controller
    app.begin_enrol()
    started = app.started
    for step in range(ENROLMENT_SAMPLES):
        app._process(None, SimpleNamespace(accepted=True, message="ready"), started + step * 0.5)
    assert app.mode is None
    assert len(app.vault.profiles()) == 1
    assert app.vault.get_profile("TEST01").samples == 8
    app.consent.set.assert_called_with(False)


def test_cancel_and_timeout_do_not_save_partial_profiles(controller):
    app = controller
    app.begin_enrol()
    app._process(None, SimpleNamespace(accepted=True, message="ready"), app.started)
    app.cancel()
    assert not app.vault.profiles() and not app.samples
    app.begin_enrol()
    app._process(None, SimpleNamespace(accepted=True, message="ready"), app.started + 30)
    assert app.mode is None and not app.vault.profiles()


def test_withdrawing_consent_discards_samples(controller):
    app = controller
    app.begin_enrol()
    app.consent.get.return_value = False
    app._process(None, SimpleNamespace(accepted=True, message="ready"), app.started)
    assert app.mode is None and not app.vault.profiles()


def test_enrolment_requires_consent_before_capture(controller):
    app = controller
    app.consent.get.return_value = False
    app.begin_enrol()
    assert app.mode is None
    assert "consent" in app.status.set.call_args.args[0]


def test_verification_pass_is_logged(controller, vector):
    app = controller
    app.vault.add_profile("TEST01", "Test", vector, 8, consent=True)
    app.begin_verify()
    started = app.started
    for step in range(5):
        app._process(None, SimpleNamespace(accepted=True, message="ready"), started + step * 0.3)
    assert app.mode is None
    assert app.vault.recent_events()[0]["action"] == "verification_passed"
    assert app.status.set.call_args.args[0].startswith("VERIFIED")


def test_verification_nonmatch_fails_and_is_logged(controller, vector):
    app = controller
    app.vault.add_profile("TEST01", "Test", vector, 8, consent=True)
    app.engine.embedding.return_value = np.eye(128, dtype=np.float32)[1]
    app.begin_verify()
    app._process(None, SimpleNamespace(accepted=True, message="ready"), app.started + 16)
    assert app.mode is None
    assert app.vault.recent_events()[0]["action"] == "verification_failed"


def test_missing_or_multiple_faces_reset_verification(controller, vector):
    app = controller
    app.vault.add_profile("TEST01", "Test", vector, 8, consent=True)
    app.begin_verify()
    started = app.started
    app._process(None, SimpleNamespace(accepted=True, message="ready"), started)
    assert app.window.matches == 1
    app._process(None, SimpleNamespace(accepted=False, message="Multiple faces"), started + 0.3)
    assert app.window.matches == 0
    app._process(None, SimpleNamespace(accepted=False, message="No face"), started + 16)
    assert app.vault.recent_events()[0]["action"] == "verification_failed"


def test_camera_is_required_to_begin_capture(controller):
    controller.camera = None
    controller.begin_enrol()
    assert controller.mode is None
    controller.status.set.assert_called_with("Start the camera first.")
