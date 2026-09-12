"""Per-camera pipeline, and the policy that decides when to spend OCR time.

On a 2 GB Pi 5 the OCR call is by far the most expensive thing in the system -
a few hundred milliseconds against a few milliseconds for detection - so the
gating here is what keeps the whole thing real-time. A crop is only sent when
it is actually worth reading:

  * the track has been seen enough times to be real, not a one-frame flicker;
  * the roll is moving, which rules out the rolls stacked on background racks;
  * the frame is sharp enough that the recogniser has a chance;
  * the writing is tall enough in pixels to resolve;
  * the track is not already read, and is not already waiting on a result;
  * and no more often than once every ``min_interval_s`` per track.

Detection itself runs on a downscaled copy every Nth frame. Tracking carries
the boxes through the frames in between.
"""

from __future__ import annotations

import time

import cv2

from .detect import RollDetector
from .ocr import OcrRequest
from .track import Tracker


def writing_crop(frame, writing_bbox, pad_x_frac: float = 0.15, pad_y_frac: float = 0.4):
    """Crop the marking with enough margin to keep both written lines intact."""
    x0, y0, x1, y1 = writing_bbox
    w, h = x1 - x0, y1 - y0
    pad_x = int(w * pad_x_frac) + 10
    pad_y = int(h * pad_y_frac) + 10
    cx0 = max(0, x0 - pad_x)
    cy0 = max(0, y0 - pad_y)
    cx1 = min(frame.shape[1], x1 + pad_x)
    cy1 = min(frame.shape[0], y1 + pad_y)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return frame[cy0:cy1, cx0:cx1].copy()


class CameraPipeline:
    def __init__(self, name: str, source, cfg) -> None:
        self.name = name
        self.source = source
        self.cfg = cfg
        self.frame_size = source.size
        self.detector = RollDetector(cfg.detect, self.frame_size)
        self.tracker = Tracker(name, cfg.track, self.frame_size)

        self.frame_index = 0
        self.last_frame = None
        self.last_detections: list = []
        self.submitted = 0
        self.fps = 0.0
        self._fps_t0 = time.time()
        self._fps_n = 0

    def read(self):
        frame = self.source.read()
        if frame is None:
            return None
        self.frame_index = frame.index
        self.last_frame = frame.image
        self._tick_fps()
        return frame

    def _tick_fps(self) -> None:
        self._fps_n += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 1.0:
            self.fps = self._fps_n / elapsed
            self._fps_n = 0
            self._fps_t0 = time.time()

    def step(self, frame, now: float):
        """Detect (periodically) and track. Returns (active tracks, dropped tracks)."""
        if self.frame_index % max(1, self.cfg.detect.detect_every) == 0:
            self.last_detections = self.detector.detect(frame.image)
        dropped = self.tracker.update(self.last_detections, now)
        return self.tracker.active(), dropped

    def ocr_candidates(self, frame, now: float) -> list[OcrRequest]:
        """Tracks worth spending an OCR call on, this frame, best first.

        OCR capacity is the scarce resource, so when several tracks qualify the
        big, sharp marking is read before a small blurred one -- otherwise a
        stray red mark on the floor can starve the roll that actually matters.
        """
        cfg = self.cfg.ocr
        requests = []
        ranked = sorted(
            self.tracker.active(),
            key=lambda t: -((t.writing_bbox[3] - t.writing_bbox[1]) * max(t.sharpness, 1.0)),
        )
        for track in ranked:
            if len(requests) >= cfg.max_per_frame:
                break
            if track.fully_read or track.pending_ocr:
                continue
            if track.ocr_attempts >= cfg.max_attempts:
                continue
            if now - track.last_ocr_at < cfg.min_interval_s:
                continue
            if track.sharpness < cfg.min_sharpness:
                continue
            height = track.writing_bbox[3] - track.writing_bbox[1]
            if height < self.cfg.detect.min_text_height_px:
                continue
            if self.cfg.detect.require_motion:
                moved = track.displacement(self.cfg.detect.motion_window_s, now)
                seen_for = now - track.first_seen
                # Only enforce once there is enough history to judge by.
                if seen_for > self.cfg.detect.motion_window_s and moved < self.cfg.detect.motion_px:
                    continue

            crop = writing_crop(frame.image, track.writing_bbox)
            if crop is None:
                continue
            track.pending_ocr = True
            track.last_crop = crop      # kept so the read can be shown later
            track.last_ocr_at = now
            track.ocr_attempts += 1
            self.submitted += 1
            requests.append(OcrRequest(
                camera=self.name,
                track_id=track.id,
                crop=crop,
                writing_bbox=track.writing_bbox,
                timestamp=now,
            ))
        return requests

    def track_by_id(self, track_id: int):
        return self.tracker.tracks.get(track_id)

    def release(self) -> None:
        self.source.release()
