"""
camera.py — Capture a single frame from the Raspberry Pi Camera Module 3.

Returns a numpy array (224, 224, 3) uint8 RGB ready for Hailo inference.
The camera stays open between captures so the sensor is always warmed up.
"""

import numpy as np
from picamera2 import Picamera2


# Native capture resolution before downscaling — larger gives the ISP more
# to work with for auto-exposure and white balance before we crop to 224x224.
CAPTURE_SIZE  = (672, 672)
INFER_SIZE    = (224, 224)


class Camera:
    """
    Wrapper around picamera2 that keeps the camera warm and vends frames.

    Usage:
        with Camera() as cam:
            frame = cam.capture()   # np.ndarray (224, 224, 3) uint8 RGB
    """

    def __init__(self):
        self._cam: Picamera2 | None = None

    def __enter__(self):
        self._cam = Picamera2()
        config = self._cam.create_still_configuration(
            main={"size": CAPTURE_SIZE, "format": "RGB888"},
            # lores stream at model input size for live viewfinder (optional)
            lores={"size": INFER_SIZE,  "format": "YUV420"},
        )
        self._cam.configure(config)
        self._cam.start()
        return self

    def __exit__(self, *_):
        if self._cam:
            self._cam.stop()
            self._cam.close()
            self._cam = None

    def capture(self) -> np.ndarray:
        """
        Capture one frame and resize to (224, 224, 3) uint8 RGB.
        Uses the main high-res stream, then downscales with OpenCV.
        """
        import cv2
        frame = self._cam.capture_array("main")          # (672, 672, 3) RGB
        resized = cv2.resize(frame, INFER_SIZE, interpolation=cv2.INTER_AREA)
        return resized  # (224, 224, 3) uint8 RGB
