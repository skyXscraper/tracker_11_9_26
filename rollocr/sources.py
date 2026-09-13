"""Frame sources: live USB cameras on the Pi, or recorded video for bench tests.

Both expose the same ``read()``.  The difference that matters is staleness: a
live camera must always hand back the *newest* frame (a backlog turns into
visible lag and, worse, OCR on a roll that has already left), while a recording
must hand back *every* frame so offline runs are reproducible.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    index: int
    timestamp: float          # seconds; wall clock live, media time for files


class FrameSource:
    name: str

    def read(self) -> Frame | None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    @property
    def size(self) -> tuple[int, int]:
        raise NotImplementedError


class CameraSource(FrameSource):
    """USB camera, grabbed on its own thread, keeping only the latest frame."""

    def __init__(self, device: str | int, name: str, cfg) -> None:
        self.name = name
        self.device = device
        self._cfg = cfg
        self._cap = self._open(device, cfg)
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"cap-{name}", daemon=True)
        self._thread.start()

    @staticmethod
    def _open(device, cfg):
        # V4L2 is the right backend on the Pi; let OpenCV choose on Windows.
        index = int(device) if str(device).isdigit() else device
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {device!r}")
        # MJPG first: two uncompressed streams will not fit through one USB bus.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)
        except cv2.error:
            pass
        CameraSource._apply_exposure(cap, cfg)
        return cap

    @staticmethod
    def _apply_exposure(cap, cfg) -> None:
        """Lower exposure at the sensor, before a bright roll clips to white.

        This is the only overexposure fix that works on a blown-out wrap: a
        highlight that has clipped has lost its detail, and no processing after
        capture recovers it. Each setting is applied only when configured, so an
        unconfigured camera keeps whatever it was doing.

        V4L2 encodes auto-exposure oddly -- through OpenCV, 1 selects manual and
        3 selects aperture-priority auto -- and a manual exposure value is ignored
        until auto is switched off, so the mode is set first.
        """
        if cfg.auto_exposure is not None:
            mode = 3 if cfg.auto_exposure else 1
            if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, mode):
                # Some drivers use 0.75 / 0.25 for the same two modes.
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if cfg.auto_exposure else 0.25)

        for name, prop in (("exposure", cv2.CAP_PROP_EXPOSURE),
                           ("brightness", cv2.CAP_PROP_BRIGHTNESS),
                           ("gain", cv2.CAP_PROP_GAIN)):
            value = getattr(cfg, name, None)
            if value is None:
                continue
            accepted = cap.set(prop, float(value))
            actual = cap.get(prop)
            note = "" if accepted else "  (driver refused it)"
            print(f"[camera] {name} -> requested {value}, now {actual}{note}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, img = self._cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            self._count += 1
            frame = Frame(img, self._count, time.time())
            with self._lock:
                self._latest = frame       # deliberately drops anything unconsumed

    def read(self) -> Frame | None:
        with self._lock:
            frame = self._latest
            self._latest = None
        return frame

    @property
    def size(self) -> tuple[int, int]:
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()


class VideoFileSource(FrameSource):
    """Recorded file, read in order.  Used to test the whole pipeline offline."""

    def __init__(self, path: str, name: str, realtime: bool = False, start_s: float = 0.0) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        self.name = name
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video {path!r}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._realtime = realtime
        self._count = 0
        self._t0 = time.time()
        if start_s > 0:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

    def read(self) -> Frame | None:
        ok, img = self._cap.read()
        if not ok:
            return None
        self._count += 1
        media_t = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if self._realtime:
            # Pace playback so display and per-track OCR throttles behave as they
            # would on a live camera.
            target = self._t0 + media_t
            delay = target - time.time()
            if delay > 0:
                time.sleep(min(delay, 0.25))
        return Frame(img, self._count, media_t)

    def read_at(self, media_t: float) -> Frame | None:
        """The first frame at or after ``media_t`` seconds, skipping earlier ones.

        Lets a 30 fps recording be run at a steady 15 fps: frames that fall
        between ticks are grabbed without being decoded into an image, which is
        far cheaper than reading and discarding them.
        """
        while True:
            if not self._cap.grab():
                return None
            self._count += 1
            timestamp = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp + 1e-6 >= media_t:
                ok, img = self._cap.retrieve()
                return Frame(img, self._count, timestamp) if ok else None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self) -> tuple[int, int]:
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self) -> None:
        self._cap.release()


def open_source(spec: str, name: str, cfg, realtime: bool = False) -> FrameSource:
    """``spec`` is a device index, a V4L2 path, or a video file path."""
    if str(spec).isdigit() or str(spec).startswith("/dev/video"):
        return CameraSource(spec, name, cfg)
    return VideoFileSource(spec, name, realtime=realtime)
