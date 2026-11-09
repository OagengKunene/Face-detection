"""Reproducible PC diagnostics and an offline self-test; no faces are stored."""

import importlib
import platform
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from bioguard import __version__
from bioguard.camera import Camera
from bioguard.decisions import MatchWindow, cosine, make_template
from bioguard.models import MODELS, valid_model
from bioguard.storage import Vault, VaultError
from bioguard.vision import FaceEngine


def doctor(directory: Path, camera_index: int | None = None) -> dict:
    checks = []

    def check(name, fn):
        try:
            detail = fn()
            checks.append({"check": name, "status": "PASS", "detail": detail})
        except Exception as exc:
            checks.append({"check": name, "status": "FAIL", "detail": str(exc)})

    def python_version():
        if not (3, 12) <= sys.version_info[:2] < (3, 15):
            raise RuntimeError("Use 64-bit Python 3.12, 3.13 or 3.14.")
        if platform.architecture()[0] != "64bit":
            raise RuntimeError("A 64-bit Python installation is required.")
        return platform.python_version()

    check("python", python_version)
    check("tkinter_import", lambda: importlib.import_module("tkinter").TkVersion)
    check("opencv", lambda: cv2.__version__)
    for spec in MODELS:

        def verify(spec=spec):
            if not valid_model(directory / spec.filename, spec):
                raise RuntimeError("Missing or damaged. Run: python run.py models")
            return "Size and SHA-256 match the pinned OpenCV model."

        check(spec.filename, verify)

    def inference():
        engine = FaceEngine(directory)
        result = engine.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        if result.accepted or len(result.faces):
            raise RuntimeError("Blank image was unexpectedly detected as a face.")
        return "YuNet inference ran; a blank frame was correctly rejected."

    check("real_dnn_inference", inference)
    if camera_index is not None:

        def camera():
            with Camera(camera_index) as device:
                frame = device.read()
                return {"index": camera_index, "width": frame.shape[1], "height": frame.shape[0]}

        check("local_webcam", camera)
    else:
        checks.append(
            {
                "check": "local_webcam",
                "status": "SKIP",
                "detail": "Use --camera 0 to test the webcam on this PC.",
            }
        )
    return {
        "application": f"Face-detection {__version__}",
        "platform": platform.platform(),
        "checks": checks,
        "ok": all(c["status"] != "FAIL" for c in checks),
    }


def self_test() -> list[str]:
    """Exercise storage, encryption and decisions using synthetic vectors, never real identities."""
    passed = []
    a, b = np.eye(128, dtype=np.float32)[:2]
    centre = make_template([a.copy() for _ in range(8)])
    assert cosine(centre, a) > 0.99 and abs(cosine(centre, b)) < 1e-6
    passed.append("Template aggregation and cosine matching")
    window = MatchWindow(interval=0.2, started=0)
    for step in range(5):
        window.observe(0.9, now=step * 0.25)
    assert window.result == "passed"
    passed.append("Five spaced matching frames are accepted")
    rejected = MatchWindow(started=0)
    assert rejected.observe(0.1, now=0) == "waiting"
    assert rejected.observe(None, now=16) == "failed"
    passed.append("Nonmatch and missing-face verification fail closed")
    with tempfile.TemporaryDirectory(prefix="bioguard-self-test-") as temporary:
        path = Path(temporary) / "test.sqlite3"
        phrase = "temporary self test passphrase"
        with Vault(path, phrase, create=True) as vault:
            vault.add_profile("TEST01", "Synthetic example", centre, 8, consent=True)
            vault.record_verification("TEST01", passed=True, score=0.9, threshold=0.45)
            assert len(vault.profiles()) == 1 and len(vault.recent_events()) == 2
        assert b"Synthetic example" not in path.read_bytes()
        passed.append("Encrypted enrolment and activity persistence")
        try:
            Vault(path, "the wrong passphrase")
        except VaultError:
            passed.append("Wrong vault passphrase is rejected")
        else:
            raise AssertionError("Wrong passphrase accepted")
        with Vault(path, phrase) as vault:
            assert cosine(vault.get_profile("TEST01").embedding, a) > 0.99
            vault.delete_profile("TEST01")
            assert not vault.profiles()
            assert all(e["person_id"] is None for e in vault.recent_events())
        passed.append("Vault reopen, profile deletion and related-event removal")
    return passed
