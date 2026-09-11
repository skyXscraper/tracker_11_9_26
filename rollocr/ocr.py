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

import os
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
        self._flavour = None
        self._hailo = None
        self._lock = threading.Lock()
        if cfg.backend == "hailo":
            self._hailo = self._try_hailo()

    def _try_hailo(self):
        """Bring up the NPU recogniser, or fall back to the CPU path.

        A missing accelerator should degrade to a slower system, not a dead
        one -- the same code has to run on a laptop with no Hailo in it.
        """
        from .hailo_ocr import HailoRecognizer
        try:
            recognizer = HailoRecognizer(self.cfg, self.cfg_detect)
            print("[ocr] backend: hailo (PP-OCRv5 recognition on NPU)")
            return recognizer
        except Exception as exc:
            if not self.cfg.hailo_fallback:
                raise
            print(f"[ocr] hailo backend unavailable ({exc}); falling back to rapidocr")
            return None

    def _ensure(self):
        """Load whichever RapidOCR generation is installed.

        The package split in two and the halves disagree about Python support:
        ``rapidocr-onnxruntime`` (1.x) caps out below Python 3.13, while its
        successor ``rapidocr`` (3.x) runs on it. Raspberry Pi OS Trixie ships
        Python 3.13, so the Pi can only have the newer one -- and the two
        return results in different shapes. Both are accepted here so the same
        tree runs on a 3.11 laptop and a 3.13 Pi.
        """
        if self._engine is not None:
            return self._engine

        # ONNX Runtime reads this at session creation; 3.x exposes no thread
        # argument, so this is the portable way to keep it off every core.
        os.environ.setdefault("OMP_NUM_THREADS", str(self.cfg.num_threads))

        try:
            from rapidocr import RapidOCR
            self._flavour = "rapidocr3"
            self._engine = RapidOCR()
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:        # pragma: no cover - deployment guard
                raise RuntimeError(
                    "No RapidOCR package found. Install one of:\n"
                    "  pip install rapidocr           # Python 3.8+, needed on 3.13\n"
                    "  pip install rapidocr-onnxruntime  # Python < 3.13\n"
                    "Not required when ocr.backend is 'hailo'."
                ) from exc
            self._flavour = "rapidocr1"
            self._engine = RapidOCR(
                intra_op_num_threads=self.cfg.num_threads,
                inter_op_num_threads=1,
            )
        return self._engine

    def _run(self, image) -> list[tuple[str, float, float]]:
        """Normalise the two packages' very different return shapes."""
        engine = self._ensure()
        with self._lock:
            raw = engine(image)

        lines: list[tuple[str, float, float]] = []
        if self._flavour == "rapidocr1":
            result = raw[0] if isinstance(raw, tuple) else raw
            for box, text, confidence in result or []:
                ys = [point[1] for point in box]
                lines.append((text, float(sum(ys) / len(ys)), float(confidence)))
            return lines

        # 3.x returns an object with parallel boxes/txts/scores arrays.
        boxes = getattr(raw, "boxes", None)
        texts = getattr(raw, "txts", None)
        scores = getattr(raw, "scores", None)
        if boxes is None or texts is None:
            return lines
        for index, text in enumerate(texts):
            box = boxes[index]
            ys = [float(point[1]) for point in box]
            score = float(scores[index]) if scores is not None else 0.0
            lines.append((text, sum(ys) / len(ys), score))
        return lines

    def read(self, crop: np.ndarray) -> list[tuple[str, float, float]]:
        """Return (text, y-centre, confidence) for each line found in the crop."""
        if crop is None or crop.size == 0:
            return []

        if self._hailo is not None:
            # The NPU model is a single-line recogniser and does its own line
            # splitting, so it takes the crop as-is rather than the enhanced,
            # rescaled image the CPU detector needs.
            return self._hailo.read(crop)

        prepared = soft_ink(crop)
        if self.cfg.deskew:
            angle = deskew_angle(crop, self.cfg_detect)
            prepared = rotate(prepared, angle)
        prepared = scale_for_ocr(prepared, crop, self.cfg, self.cfg_detect)
        return self._run(prepared)


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
