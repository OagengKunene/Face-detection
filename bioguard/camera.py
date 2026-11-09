"""Explicit camera lifecycle, with a Windows DirectShow fallback."""

import os

import cv2


class CameraError(Exception):
    pass


class Camera:
    def __init__(self, index: int = 0):
        if not isinstance(index, int) or not 0 <= index <= 9:
            raise ValueError("Camera index must be 0-9.")
        self.index = index
        self.capture = None
        self.first_frame = None

    def open(self):
        self.close()
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for backend in backends:
            capture = cv2.VideoCapture(self.index, backend)
            try:
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    for _ in range(12):
                        ok, frame = capture.read()
                        if ok and frame is not None and frame.size:
                            self.capture = capture
                            self.first_frame = frame
                            return self
            finally:
                if self.capture is not capture:
                    capture.release()
        raise CameraError(
            f"Cannot read camera {self.index}. Close Teams/Zoom, enable camera access "
            "for desktop apps in Windows Settings, or try camera 1."
        )

    def read(self):
        if self.capture is None:
            raise CameraError("Start the camera first.")
        if self.first_frame is not None:
            frame, self.first_frame = self.first_frame, None
            return frame
        ok, frame = self.capture.read()
        if not ok or frame is None or not frame.size:
            raise CameraError("Camera feed was interrupted. Reconnect and start it again.")
        return frame

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.first_frame = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()
