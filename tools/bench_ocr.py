#!/usr/bin/env python3
"""Measure what the Pi can actually sustain, before trusting the pipeline on it.

Reports, per stage, on real frames from a recording:

  * detection time -- runs on every camera, every Nth frame, so it sets the
    floor on frame rate;
  * OCR time and resident memory -- the expensive stage, and the one that
    decides whether a 2 GB Pi copes;
  * an estimate of the sustainable two-camera frame rate.

Usage:

    python tools/bench_ocr.py --video videos/cam0_test2.mp4 --frames 120
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollocr.config import Config             # noqa: E402
from rollocr.detect import RollDetector       # noqa: E402
from rollocr.ocr import OcrEngine             # noqa: E402
from rollocr.pipeline import writing_crop     # noqa: E402


def resident_mb() -> float | None:
    """Resident set size, without depending on psutil."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def summarise(name: str, samples: list[float]) -> None:
    if not samples:
        print(f"  {name:<12} no samples")
        return
    data = np.array(samples)
    print(f"  {name:<12} n={len(data):<4} mean={data.mean():7.1f} ms   "
          f"median={np.median(data):7.1f}   p95={np.percentile(data, 95):7.1f}   "
          f"max={data.max():7.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--config")
    parser.add_argument("--threads", type=int, help="override ocr.num_threads")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.threads:
        cfg.ocr.num_threads = args.threads

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = RollDetector(cfg.detect, (width, height), cfg.exposure)
    engine = OcrEngine(cfg.ocr, cfg.detect, cfg.exposure)

    print(f"[bench] {args.video}  {width}x{height}  ocr threads={cfg.ocr.num_threads}")
    baseline = resident_mb()
    if baseline:
        print(f"[bench] resident before OCR model load: {baseline:.0f} MB")

    detect_ms: list[float] = []
    ocr_ms: list[float] = []
    detections = 0

    for _ in range(args.frames):
        ok, frame = capture.read()
        if not ok:
            break

        started = time.perf_counter()
        found = detector.detect(frame)
        detect_ms.append((time.perf_counter() - started) * 1000.0)

        if found:
            detections += 1
            crop = writing_crop(frame, found[0].writing_bbox)
            if crop is not None and crop.size:
                started = time.perf_counter()
                engine.read(crop)
                ocr_ms.append((time.perf_counter() - started) * 1000.0)
    capture.release()

    print(f"\n[bench] frames with a detection: {detections}/{len(detect_ms)}")
    summarise("detect", detect_ms)
    summarise("ocr", ocr_ms)

    peak = resident_mb()
    if peak:
        print(f"\n[bench] resident after {len(ocr_ms)} OCR calls: {peak:.0f} MB")
        print("[bench] two pipelines share one OCR engine, so this is close to "
              "the whole system's footprint")

    if detect_ms:
        per_frame = np.median(detect_ms)
        # Both cameras detect on the same core budget; OCR runs on its own
        # thread and is throttled per track, so it does not gate frame rate.
        budget = 1000.0 / (per_frame * 2 / max(1, cfg.detect.detect_every))
        print(f"\n[bench] sustainable two-camera rate from detection alone: "
              f"~{budget:.0f} fps (detect_every={cfg.detect.detect_every})")
    if ocr_ms:
        print(f"[bench] at {np.median(ocr_ms):.0f} ms per read and "
              f"{cfg.ocr.min_interval_s}s per-track throttle, one OCR thread "
              f"serves ~{1.0 / (np.median(ocr_ms) / 1000.0):.1f} reads/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
