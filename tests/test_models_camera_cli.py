import hashlib
import io
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from bioguard import camera as camera_module
from bioguard import models as model_module
from bioguard.camera import Camera, CameraError
from bioguard.cli import main
from bioguard.models import Model, checked_paths, ensure_models, valid_model


def test_checksum_validation(tmp_path):
    raw = b"example model"
    spec = Model("folder", "file.onnx", len(raw), hashlib.sha256(raw).hexdigest())
    path = tmp_path / spec.filename
    assert not valid_model(path, spec)
    path.write_bytes(raw)
    assert valid_model(path, spec)
    path.write_bytes(b"wrong content")
    assert not valid_model(path, spec)


def test_missing_models_are_not_loaded(tmp_path):
    with pytest.raises(FileNotFoundError):
        checked_paths(tmp_path)


def test_model_download_is_verified_and_existing_files_are_reused(tmp_path, monkeypatch):
    raw = b"test onnx"
    spec = Model("folder", "test.onnx", len(raw), hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(model_module, "MODELS", (spec, spec))
    downloader = MagicMock(side_effect=lambda *a, **kw: io.BytesIO(raw))
    monkeypatch.setattr(model_module.urllib.request, "urlopen", downloader)
    ensure_models(tmp_path, lambda _: None)
    assert downloader.call_count == 1
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("raw", [b"bad", b"x" * 20])
def test_invalid_download_never_replaces_valid_destination(tmp_path, monkeypatch, raw):
    spec = Model("folder", "test.onnx", 8, hashlib.sha256(b"expected").hexdigest())
    monkeypatch.setattr(model_module, "MODELS", (spec, spec))
    monkeypatch.setattr(model_module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(raw))
    with pytest.raises(ValueError):
        ensure_models(tmp_path, lambda _: None)
    assert not (tmp_path / "test.onnx").exists()
    assert not list(tmp_path.glob("*.part"))


def test_camera_releases_device_after_success(monkeypatch):
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.ones((480, 640, 3), dtype=np.uint8))
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *a: capture)
    with Camera(0) as device:
        assert device.read().shape == (480, 640, 3)
    capture.release.assert_called_once()


def test_camera_releases_failed_device(monkeypatch):
    capture = MagicMock()
    capture.isOpened.return_value = False
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *a: capture)
    with pytest.raises(CameraError, match="Cannot read camera"):
        Camera(0).open()
    assert capture.release.called


def test_camera_disconnection_fails(monkeypatch):
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.side_effect = [(True, np.ones((480, 640, 3), dtype=np.uint8)), (False, None)]
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *a: capture)
    with Camera(0) as device:
        device.read()
        with pytest.raises(CameraError, match="interrupted"):
            device.read()


@pytest.mark.parametrize("index", [-1, 10, "0"])
def test_invalid_camera_index(index):
    with pytest.raises(ValueError):
        Camera(index)


def test_doctor_returns_failure_and_saves_report_for_missing_models(tmp_path):
    output = tmp_path / "test_report.json"
    assert main(["doctor", "--model-dir", str(tmp_path), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["ok"] is False
    assert any(c["status"] == "SKIP" for c in report["checks"])


def test_offline_self_test():
    assert main(["self-test"]) == 0
