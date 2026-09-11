#!/usr/bin/env python3
"""Prove the Hailo recognition path before trusting it in a live run.

Runs the NPU recogniser over a known crop and prints what it read, how the
crop was split into lines, and how long inference took. Where RapidOCR is also
installed it runs both and shows them side by side, which is the only way to
tell whether moving to PP-OCRv5 on the NPU actually helps on this handwriting
or merely makes the same mistakes faster.

    # against the committed reference crops (ground truth is in their names)
    python tools/test_hailo.py --image tests/fixtures/roll_ply99.jpg

    # against a live camera frame
    python tools/test_hailo.py --camera /dev/video0

    # detection + recognition over a whole frame
    python tools/test_hailo.py --image frame.jpg --detect
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollocr.config import Config              # noqa: E402
from rollocr.detect import RollDetector        # noqa: E402
from rollocr.parse import parse_lines          # noqa: E402
from rollocr.pipeline import writing_crop      # noqa: E402


def grab(args):
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"cannot read {args.image}")
        return image
    source = cv2.VideoCapture(int(args.camera) if str(args.camera).isdigit() else args.camera)
    if not source.isOpened():
        raise SystemExit(f"cannot open {args.camera}")
    for _ in range(10):        # let exposure settle
        ok, image = source.read()
    source.release()
    if not ok:
        raise SystemExit("no frame captured")
    return image


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image")
    group.add_argument("--camera")
    parser.add_argument("--detect", action="store_true",
                        help="treat the input as a full frame and find the roll first")
    parser.add_argument("--config")
    parser.add_argument("--runs", type=int, default=5, help="timed repetitions")
    parser.add_argument("--save-lines", help="write the split line images here")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    image = grab(args)
    print(f"[test] input {image.shape[1]}x{image.shape[0]}")

    crop = image
    if args.detect:
        detector = RollDetector(cfg.detect, (image.shape[1], image.shape[0]))
        found = detector.detect(image)
        print(f"[test] detections: {len(found)}")
        if not found:
            return 1
        crop = writing_crop(image, found[0].writing_bbox)
        print(f"[test] crop {crop.shape[1]}x{crop.shape[0]} "
              f"from writing bbox {found[0].writing_bbox}")

    # -- line splitting ---------------------------------------------------
    from rollocr.hailo_ocr import preprocess_line, split_text_lines

    bands = split_text_lines(crop, cfg.detect)
    print(f"[test] text lines found: {len(bands)} -> {bands}")
    if args.save_lines:
        out = Path(args.save_lines)
        out.mkdir(parents=True, exist_ok=True)
        for i, (top, bottom) in enumerate(bands):
            cv2.imwrite(str(out / f"line{i}.png"), preprocess_line(crop[top:bottom + 1]))
        print(f"[test] wrote {len(bands)} preprocessed lines to {out}")

    # -- Hailo ------------------------------------------------------------
    print("\n[hailo] loading recogniser...")
    try:
        from rollocr.hailo_ocr import HailoRecognizer
        cfg.ocr.hailo_fallback = False
        recognizer = HailoRecognizer(cfg.ocr, cfg.detect)
    except Exception as exc:
        print(f"[hailo] UNAVAILABLE: {exc}")
        recognizer = None

    if recognizer is not None:
        print(f"[hailo] charset classes: {len(recognizer.charset)} "
              f"(model outputs 18385 -- these should agree)")
        lines = recognizer.read(crop)
        elapsed = []
        for _ in range(args.runs):
            started = time.perf_counter()
            recognizer.read(crop)
            elapsed.append((time.perf_counter() - started) * 1000.0)
        print(f"[hailo] read: {[(t, round(c, 2)) for t, _, c in lines]}")
        print(f"[hailo] {np.median(elapsed):.1f} ms median over {args.runs} runs")
        ply, ranges = parse_lines(lines, cfg.values)
        print(f"[hailo] parsed -> ply {ply}  "
              f"{[(r.start, r.end) for r in ranges[:3]]}")

    # -- CPU, for comparison ----------------------------------------------
    print("\n[cpu] loading rapidocr...")
    try:
        from rollocr.ocr import OcrEngine
        cfg.ocr.backend = "rapidocr"
        engine = OcrEngine(cfg.ocr, cfg.detect)
        lines = engine.read(crop)
        started = time.perf_counter()
        engine.read(crop)
        cpu_ms = (time.perf_counter() - started) * 1000.0
        print(f"[cpu] read: {[(t, round(c, 2)) for t, _, c in lines]}")
        print(f"[cpu] {cpu_ms:.1f} ms")
        ply, ranges = parse_lines(lines, cfg.values)
        print(f"[cpu] parsed -> ply {ply}  {[(r.start, r.end) for r in ranges[:3]]}")
    except Exception as exc:
        print(f"[cpu] unavailable: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
