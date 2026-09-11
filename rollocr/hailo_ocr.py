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

    # Group the individual strokes, rather than thresholding a row profile.
    # A row profile cannot tell "two lines close together" from "one tall
    # line", and the two written lines here sit only a few pixels apart -- any
    # tolerance loose enough to join a lifted pen also welds ply to lengths,
    # which hands a single-line recogniser a two-line image and it returns
    # nothing at all.
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    strokes = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 20 or h < 5:
            continue
        strokes.append({"x0": int(x), "x1": int(x + w - 1),
                        "top": int(y), "bottom": int(y + h - 1),
                        "height": int(h), "area": int(area),
                        "cy": float(centroids[i][1])})
    if not strokes:
        return []

    # Size the tolerance from the *digits*, not from every speck. The median is
    # dragged down by punctuation and noise -- on a real crop it gave 9 px
    # against a 30 px digit, which split "99" in half. The upper quartile
    # tracks the character height that actually matters.
    typical = float(np.percentile([s["height"] for s in strokes], 75))
    tolerance = max(6.0, typical * 0.9)

    strokes.sort(key=lambda s: s["cy"])
    groups: list[list[dict]] = [[strokes[0]]]
    for stroke in strokes[1:]:
        if stroke["cy"] - groups[-1][-1]["cy"] <= tolerance:
            groups[-1].append(stroke)
        else:
            groups.append([stroke])

    bands = []
    for group in groups:
        # A written line is several glyphs side by side. Strokes stacked on
        # the same column are one glyph (or one stain), not a line of text.
        if _distinct_glyphs(group) < 2:
            continue
        top = min(s["top"] for s in group)
        bottom = max(s["bottom"] for s in group)
        if (bottom - top) < 5:
            continue
        bands.append({"top": top, "bottom": bottom,
                      "ink": sum(s["area"] for s in group)})
    if not bands:
        return []

    strongest = max(b["ink"] for b in bands)
    kept = [b for b in bands if b["ink"] >= strongest * 0.15]
    kept.sort(key=lambda b: -b["ink"])
    return sorted((b["top"], b["bottom"]) for b in kept[:max_lines])


def _distinct_glyphs(group: list[dict]) -> int:
    """How many separate characters a group of strokes spans horizontally."""
    spans = sorted((s["x0"], s["x1"]) for s in group)
    glyphs = 0
    reach = -1
    for x0, x1 in spans:
        if x0 > reach:
            glyphs += 1
            reach = x1
        else:
            reach = max(reach, x1)
    return glyphs


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

        # Imported here rather than at module scope: ocr.py loads this module
        # lazily, and importing it back at load time would be circular.
        from .ocr import deskew_angle, rotate, soft_ink

        work = crop
        if self.cfg.deskew:
            work = rotate(crop, deskew_angle(crop, self.cfg_detect))

        # Line bands must come from the colour image -- the ink mask keys on
        # the red hue, which enhancement deliberately flattens away.
        bands = split_text_lines(work, self.cfg_detect)
        if not bands:
            bands = [(0, work.shape[0] - 1)]

        # The recogniser was trained on printed text: dark glyphs on a light
        # ground. Red pen on a glossy wrap is neither, and feeding it raw is
        # why the network returned blank for every timestep. Darkening the
        # strokes in proportion to their redness is the same enhancement the
        # CPU path uses, and it is what makes the crop legible to the model.
        source = soft_ink(work) if self.cfg.hailo_enhance else work

        prepared, centres = [], []
        for top, bottom in bands:
            pad = max(2, int((bottom - top) * 0.2))
            y0 = max(0, top - pad)
            y1 = min(source.shape[0], bottom + pad + 1)
            if y1 - y0 < 4:
                continue
            prepared.append(preprocess_line(source[y0:y1]))
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
