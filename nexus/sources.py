"""Frame sources: video files, USB webcams and RTSP/HTTP streams, with reconnect support."""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional, Union

import cv2
import numpy as np

log = logging.getLogger("nexus.sources")

STREAM_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "udp://", "tcp://")


class SourceError(RuntimeError):
    """The source could not be opened or delivered no usable frames."""


class FrameSource:
    """Uniform frame reader.

    * file    : ends at EOF (``finished`` becomes True) or loops when ``loop=True``
                (``consume_wrap()`` then tells the caller to reset tracking).
    * webcam  : tries DirectShow/MSMF/default on Windows and rejects black frames.
    * stream  : RTSP over TCP by default; ``read()`` returns None on a dropout and the
                caller decides when to ``reopen()``.
    """

    def __init__(self, source: Union[str, int], loop: bool = False, max_failed_reads: int = 30) -> None:
        text = str(source)
        self.source: Union[str, int] = int(text) if text.isdigit() else text
        self.is_webcam = isinstance(self.source, int)
        self.is_stream = isinstance(self.source, str) and self.source.lower().startswith(STREAM_PREFIXES)
        self.is_file = not (self.is_webcam or self.is_stream)
        self.loop = loop
        self.max_failed_reads = max_failed_reads
        self.finished = False
        self._wrapped = False
        self._failed = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_index = 0
        if self.is_stream:
            # TCP is far more reliable than UDP for IP cameras behind NAT/Wi-Fi.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    @property
    def is_live(self) -> bool:
        return not self.is_file

    @property
    def fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap is not None else 0.0
        return float(fps) if fps and fps > 1 else 30.0

    # -- open / close -------------------------------------------------------- #
    def open(self) -> "FrameSource":
        self.release()
        if self.is_file:
            if not os.path.isfile(str(self.source)):
                raise SourceError(f"video file not found: {self.source!r}")
            self.cap = cv2.VideoCapture(str(self.source))
        elif self.is_stream:
            self.cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
        else:
            self.cap = self._open_webcam(int(self.source))
        if self.cap is None or not self.cap.isOpened():
            raise SourceError(f"cannot open source {self.source!r}")
        self._failed = 0
        self.finished = False
        return self

    def reopen(self) -> bool:
        """Try to re-open after a dropout. Returns True on success."""
        try:
            self.open()
            log.info("source %r re-opened", self.source)
            return True
        except SourceError as exc:
            log.warning("re-open failed: %s", exc)
            return False

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @staticmethod
    def _open_webcam(index: int, warmup_frames: int = 40, min_brightness: float = 8.0) -> cv2.VideoCapture:
        """Open a webcam, accepting only a backend that delivers real (non-black) frames."""
        if sys.platform == "win32":
            backends = [("DirectShow", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("default", cv2.CAP_ANY)]
        else:
            backends = [("default", cv2.CAP_ANY)]
        for name, backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                for _ in range(warmup_frames):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.mean() > min_brightness:
                        log.info("webcam %d opened (%s backend)", index, name)
                        return cap
                    time.sleep(0.03)
            cap.release()
        raise SourceError(
            f"webcam {index} gave no usable frames: close other apps using the camera, check the "
            "privacy shutter / OS camera permissions, or try another index")

    # -- reading ------------------------------------------------------------- #
    def read(self) -> Optional[np.ndarray]:
        """Next frame, or None (EOF for files when not looping, a dropout for live sources)."""
        if self.cap is None:
            self.open()
        assert self.cap is not None
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._failed = 0
            self.frame_index += 1
            return frame
        if self.is_file:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._wrapped = True
                ok, frame = self.cap.read()
                return frame if ok else None
            self.finished = True
            return None
        self._failed += 1
        if self._failed < self.max_failed_reads:
            time.sleep(0.01)
            return self.read()      # brief dropout: retry before giving up
        return None

    def consume_wrap(self) -> bool:
        """True once after a looped file wrapped to the start (reset tracking then)."""
        wrapped, self._wrapped = self._wrapped, False
        return wrapped
