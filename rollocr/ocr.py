"""OCR: PP-OCR models running on ONNX Runtime.

Why ONNX rather than the PaddlePaddle runtime: the recognition and detection
weights are the same PP-OCR ones, but the session holds a few hundred MB
instead of well over a gigabyte, which is what makes two camera pipelines fit
on a 2 GB Pi 5 at all.

Preprocessing is doing a lot of the work here.  Measured on the test footage:

  * feeding raw crops beats hard-binarising the ink -- thresholding throws away
    the stroke shape the recogniser depends on;
  * darkening the strokes *in proportion to their redness* keeps that shape and
    was the only variant that read the harder, edge-clipped frames;
  * the crop must be scaled up so the text stands ~80 px tall.

One engine instance serves both cameras from a single worker thread.  That is a
deliberate memory cap: two sessions would double the largest allocation in the
process for no throughput gain on four cores.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .detect import ink_mask


@dataclass
class OcrRequest:
    camera: str
    track_id: int
    crop: np.ndarray
    writing_bbox: tuple[int, int, int, int]
    timestamp: float


@dataclass
class OcrResult:
    camera: str
    track_id: int
    lines: list[tuple[str, float, float]]   # (text, y-centre, confidence)
    elapsed_ms: float
    timestamp: float


def soft_ink(bgr: np.ndarray, strength: float = 0.75) -> np.ndarray:
    """Darken red strokes proportionally to redness, preserving stroke shape.

    A hard mask turns anti-aliased handwriting into ragged blobs; this keeps the
    gradient and simply deepens the contrast against the pale wrap.
    """
    blue, green, red = (bgr[:, :, i].astype(np.float32) for i in range(3))
    excess = np.clip((red - np.maximum(green, blue)) / 45.0, 0.0, 1.0)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out = gray * (1.0 - strength * excess)
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def deskew_angle(crop: np.ndarray, cfg_detect) -> float:
    """Angle of the writing, from the ink's minimum-area rectangle."""
    mask = ink_mask(crop, cfg_detect)
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 30:
        return 0.0
    (_, _), (w, h), angle = cv2.minAreaRect(points)
    if w < h:
        angle += 90.0
    return angle if abs(angle) < 35.0 else 0.0


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1.0:
        return img
    h, w = img.shape[:2]
    centre = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2.0 - centre[0]
    matrix[1, 2] += new_h / 2.0 - centre[1]
    return cv2.warpAffine(img, matrix, (new_w, new_h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def scale_for_ocr(img: np.ndarray, reference: np.ndarray, cfg, cfg_detect) -> np.ndarray:
    """Scale so a text line stands about ``target_text_height`` pixels tall."""
    mask = ink_mask(reference, cfg_detect)
    ys, _ = np.where(mask > 0)
    # The marking is two stacked lines, so one line is about half its height.
    line_h = max((ys.max() - ys.min()) / 2.0, 8.0) if len(ys) else 25.0
    factor = float(np.clip(cfg.target_text_height / line_h, 1.0, 6.0))
    scaled = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    if max(scaled.shape[:2]) > cfg.max_side:
        shrink = cfg.max_side / max(scaled.shape[:2])
        scaled = cv2.resize(scaled, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    return scaled


class OcrEngine:
    """Thin wrapper over RapidOCR, kept to one shared instance."""

    def __init__(self, cfg, cfg_detect) -> None:
        self.cfg = cfg
        self.cfg_detect = cfg_detect
        self._engine = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:            # pragma: no cover - deployment guard
                raise RuntimeError(
                    "rapidocr-onnxruntime is not installed; run: pip install -r requirements.txt"
                ) from exc
            self._engine = RapidOCR(
                intra_op_num_threads=self.cfg.num_threads,
                inter_op_num_threads=1,
            )
        return self._engine

    def read(self, crop: np.ndarray) -> list[tuple[str, float, float]]:
        """Return (text, y-centre, confidence) for each line found in the crop."""
        if crop is None or crop.size == 0:
            return []
        engine = self._ensure()

        prepared = soft_ink(crop)
        if self.cfg.deskew:
            angle = deskew_angle(crop, self.cfg_detect)
            prepared = rotate(prepared, angle)
        prepared = scale_for_ocr(prepared, crop, self.cfg, self.cfg_detect)

        with self._lock:
            result, _ = engine(prepared)
        if not result:
            return []

        lines: list[tuple[str, float, float]] = []
        for box, text, confidence in result:
            ys = [point[1] for point in box]
            lines.append((text, float(sum(ys) / len(ys)), float(confidence)))
        return lines


class OcrWorker:
    """Single background thread draining a bounded queue.

    The bound matters more than it looks: if OCR falls behind on the Pi we want
    new requests dropped, not queued into memory behind a roll that has already
    left the frame.
    """

    def __init__(self, engine: OcrEngine, cfg) -> None:
        self.engine = engine
        self.requests: queue.Queue[OcrRequest] = queue.Queue(maxsize=cfg.queue_size)
        self.results: queue.Queue[OcrResult] = queue.Queue()
        self.dropped = 0
        self.processed = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="ocr", daemon=True)
        self._thread.start()

    def submit(self, request: OcrRequest) -> bool:
        try:
            self.requests.put_nowait(request)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                request = self.requests.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.time()
            try:
                lines = self.engine.read(request.crop)
            except Exception as exc:                # keep the pipeline alive
                print(f"[ocr] error on {request.camera}#{request.track_id}: {exc}")
                lines = []
            self.processed += 1
            self.results.put(OcrResult(
                camera=request.camera,
                track_id=request.track_id,
                lines=lines,
                elapsed_ms=(time.time() - started) * 1000.0,
                timestamp=request.timestamp,
            ))

    def drain(self) -> list[OcrResult]:
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class SyncOcrRunner:
    """Inline OCR, used when processing recorded files.

    Offline runs exist to measure the pipeline, so they must not silently drop
    work when OCR falls behind the reader -- that turns a tuning run into
    noise. This mirrors OcrWorker's interface but does the read in the calling
    thread, so every submitted crop is processed and a given video always
    produces the same result.
    """

    def __init__(self, engine: OcrEngine, cfg) -> None:
        self.engine = engine
        self.cfg = cfg
        self._results: list[OcrResult] = []
        self.dropped = 0
        self.processed = 0

    def submit(self, request: OcrRequest) -> bool:
        started = time.time()
        try:
            lines = self.engine.read(request.crop)
        except Exception as exc:
            print(f"[ocr] error on {request.camera}#{request.track_id}: {exc}")
            lines = []
        self.processed += 1
        self._results.append(OcrResult(
            camera=request.camera,
            track_id=request.track_id,
            lines=lines,
            elapsed_ms=(time.time() - started) * 1000.0,
            timestamp=request.timestamp,
        ))
        return True

    def drain(self) -> list[OcrResult]:
        out, self._results = self._results, []
        return out

    def stop(self) -> None:
        return
