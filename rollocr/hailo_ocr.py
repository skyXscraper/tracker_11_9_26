"""PP-OCRv5 recognition on the Hailo-8L NPU.

Drop-in replacement for the recognition half of the CPU pipeline, behind the
same ``read(crop) -> [(text, y_centre, confidence)]`` contract as OcrEngine.

Three things about the compiled model shape the code:

* **Input is fixed at 48x320x3 UINT8.** Crops are resized to 48 tall keeping
  aspect, then padded out to 320 wide.  The pad value is 128, not 0: PP-OCR
  normalises with ``(x/255 - 0.5)/0.5``, that normalisation is baked into the
  HEF, and "zero padding" in the trained model means zero *after* normalising,
  which is mid-grey going in.
* **Output is 1x40x18385, already softmaxed on-device.** 40 CTC timesteps over
  the PP-OCRv5 charset, decoded greedily here.
* **It recognises one line at a time.** A roll carries two written lines, so
  the crop is split on the ink mask's horizontal projection before inference.
  That is much cheaper than running the 544x960 detection HEF to find two lines
  we can already locate, and it preserves each line's vertical position, which
  is what tells ply from lengths downstream.

Written against the tensor layout reported by ``hailortcli parse-hef`` on a
HAILO8L with firmware 4.23.0. It has not been executed on the device from here
-- run ``tools/test_hailo.py`` first to confirm before trusting a live run.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .detect import ink_mask

REC_HEIGHT = 48
REC_WIDTH = 320
PAD_VALUE = 128          # mid-grey: "zero" once PP-OCR normalisation is applied


def split_text_lines(crop: np.ndarray, cfg_detect, max_lines: int = 3) -> list[tuple[int, int]]:
    """Row bands containing ink, one per written line, top to bottom.

    The ply number sits above the lengths with a clear gap, so a horizontal
    projection of the ink mask separates them for almost no cost. The work is
    in not over-splitting: handwriting is sparse, so a raw projection breaks a
    single line into several fragments wherever the pen lifted. Closing the
    mask horizontally, smoothing the projection, and then merging bands that
    sit closer together than a line is tall keeps one line as one band.
    """
    mask = ink_mask(crop, cfg_detect)
    if mask is None or not mask.any():
        return []

    height, width = mask.shape[:2]
    # Join strokes within a line before projecting.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, width // 10), 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    rows = (mask > 0).sum(axis=1).astype(np.float32)
    window = max(3, height // 30)
    rows = np.convolve(rows, np.ones(window, dtype=np.float32) / window, mode="same")
    if rows.max() <= 0:
        return []

    active = rows >= max(1.0, rows.max() * 0.15)
    bands: list[list[int]] = []
    start = None
    for y, on in enumerate(active):
        if on and start is None:
            start = y
        elif not on and start is not None:
            bands.append([start, y - 1])
            start = None
    if start is not None:
        bands.append([start, height - 1])
    if not bands:
        return []

    # Merge fragments of the same line: a gap smaller than half a typical line
    # height is a lifted pen, not a new line.
    typical = float(np.median([b[1] - b[0] + 1 for b in bands]))
    tolerance = max(3.0, typical * 0.5)
    merged = [bands[0]]
    for top, bottom in bands[1:]:
        if top - merged[-1][1] <= tolerance:
            merged[-1][1] = bottom
        else:
            merged.append([top, bottom])

    # Drop faint scraps -- partial markings clipped in from a neighbouring roll.
    mass = [(mask[t:b + 1] > 0).sum() for t, b in merged]
    strongest = max(mass) if mass else 0
    kept = [(t, b) for (t, b), m in zip(merged, mass)
            if m >= strongest * 0.15 and (b - t) >= 4]

    kept.sort(key=lambda band: -(mask[band[0]:band[1] + 1] > 0).sum())
    return sorted(kept[:max_lines])


def preprocess_line(line: np.ndarray) -> np.ndarray:
    """Resize to 48 tall keeping aspect, pad to 320 wide, return RGB UINT8."""
    h, w = line.shape[:2]
    if h == 0 or w == 0:
        return np.full((REC_HEIGHT, REC_WIDTH, 3), PAD_VALUE, dtype=np.uint8)

    scale = REC_HEIGHT / float(h)
    new_w = max(1, min(REC_WIDTH, int(round(w * scale))))
    resized = cv2.resize(line, (new_w, REC_HEIGHT), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((REC_HEIGHT, REC_WIDTH, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:, :new_w] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def load_charset(path: str) -> list[str]:
    """PP-OCRv5 dictionary, with index 0 reserved for the CTC blank."""
    lines = Path(os.path.expanduser(path)).read_text(encoding="utf-8").splitlines()
    # PaddleOCR maps class 0 to blank and appends a space class at the end.
    return ["<blank>"] + lines + [" "]


def ctc_greedy_decode(logits: np.ndarray, charset: list[str]) -> tuple[str, float]:
    """Collapse repeats, drop blanks, average the winning probabilities."""
    probs = logits.reshape(-1, logits.shape[-1])
    indices = probs.argmax(axis=1)
    scores = probs.max(axis=1)

    text: list[str] = []
    kept: list[float] = []
    previous = -1
    for index, score in zip(indices, scores):
        if index != previous and index != 0:
            if index < len(charset):
                text.append(charset[index])
                kept.append(float(score))
        previous = int(index)

    confidence = float(np.mean(kept)) if kept else 0.0
    return "".join(text), confidence


class HailoRecognizer:
    """One VDevice, one activated network group, serialised by a lock.

    The Hailo device is a single shared resource and both cameras feed the same
    OCR worker, so inference is serialised here exactly as the CPU engine is.
    """

    def __init__(self, cfg, cfg_detect) -> None:
        self.cfg = cfg
        self.cfg_detect = cfg_detect
        self.charset = load_charset(cfg.hailo_charset)
        self._lock = threading.Lock()
        self._ready = False
        self._setup()

    def _setup(self) -> None:
        try:
            from hailo_platform import (HEF, ConfigureParams, FormatType,
                                        HailoStreamInterface, InferVStreams,
                                        InputVStreamParams, OutputVStreamParams,
                                        VDevice)
        except ImportError as exc:  # pragma: no cover - device-only path
            raise RuntimeError(
                "hailo_platform is not available; install the Hailo stack "
                "(sudo apt install hailo-all) or set ocr.backend to 'rapidocr'"
            ) from exc

        self._InferVStreams = InferVStreams
        self._hef = HEF(os.path.expanduser(self.cfg.hailo_rec_hef))
        self._device = VDevice()

        params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe)
        self._network_group = self._device.configure(self._hef, params)[0]
        self._group_params = self._network_group.create_params()

        self._in_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.UINT8)
        # FLOAT32 so HailoRT dequantises the UINT8 softmax for us.
        self._out_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32)

        self._input_name = self._hef.get_input_vstream_infos()[0].name
        self._output_name = self._hef.get_output_vstream_infos()[0].name
        self._ready = True

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        with self._lock:
            with self._network_group.activate(self._group_params):
                with self._InferVStreams(self._network_group, self._in_params,
                                         self._out_params) as pipeline:
                    result = pipeline.infer({self._input_name: batch})
        return np.asarray(result[self._output_name])

    def read(self, crop: np.ndarray) -> list[tuple[str, float, float]]:
        """Recognise every written line in the crop, top to bottom."""
        if crop is None or crop.size == 0 or not self._ready:
            return []

        bands = split_text_lines(crop, self.cfg_detect)
        if not bands:
            bands = [(0, crop.shape[0] - 1)]

        prepared, centres = [], []
        for top, bottom in bands:
            pad = max(2, int((bottom - top) * 0.2))
            y0 = max(0, top - pad)
            y1 = min(crop.shape[0], bottom + pad + 1)
            if y1 - y0 < 4:
                continue
            prepared.append(preprocess_line(crop[y0:y1]))
            centres.append((y0 + y1) / 2.0)

        if not prepared:
            return []

        outputs = self._infer(np.stack(prepared).astype(np.uint8))

        lines: list[tuple[str, float, float]] = []
        for index, centre in enumerate(centres):
            text, confidence = ctc_greedy_decode(outputs[index], self.charset)
            if text.strip():
                lines.append((text, centre, confidence))
        return lines
