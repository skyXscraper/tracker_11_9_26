#!/usr/bin/env python3
"""Pick the work zone and the two cameras' overlap zones, by hand, on a frame.

Both zones pay for themselves on a 2 GB Pi:

  * the work zone stops the detector from ever considering the rolls stacked on
    background racks, so no OCR time is spent on them;
  * the overlap zones are what let a roll be linked across the two cameras
    before its writing has been read, which is the difference between an ID
    that is stable from the first frame and one that appears late.

Usage:

    python tools/calibrate_roi.py --cam1 videos/cam0_test1.mp4 \
                                  --cam2 videos/cam2_test1.mp4 \
                                  --out config.json

Drag a box, press ENTER to accept, c to clear, n for the next zone, q to quit.
The fractions it prints can also just be pasted into a config file by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollocr.config import CaptureConfig      # noqa: E402
from rollocr.sources import open_source       # noqa: E402


def grab_frame(spec: str, name: str, seek: float):
    source = open_source(spec, name, CaptureConfig())
    frame = None
    # Seek by reading: works the same for a file and a live camera warming up.
    target = max(1, int(seek))
    for _ in range(target):
        got = source.read()
        if got is not None:
            frame = got.image
    source.release()
    if frame is None:
        raise SystemExit(f"could not read a frame from {spec}")
    return frame


def pick(frame, title: str) -> tuple[float, float, float, float] | None:
    height, width = frame.shape[:2]
    scale = min(1.0, 900.0 / max(height, 1))
    view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()
    cv2.putText(view, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)

    box = cv2.selectROI(title, view, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    return (round(x / view.shape[1], 4), round(y / view.shape[0], 4),
            round((x + w) / view.shape[1], 4), round((y + h) / view.shape[0], 4))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam1", required=True)
    parser.add_argument("--cam2", required=True)
    parser.add_argument("--name1", default="cam1")
    parser.add_argument("--name2", default="cam2")
    parser.add_argument("--seek", type=float, default=1,
                        help="frames to skip before grabbing (find one with a roll in it)")
    parser.add_argument("--out", help="write a config JSON with the zones filled in")
    args = parser.parse_args()

    frames = {
        args.name1: grab_frame(args.cam1, args.name1, args.seek),
        args.name2: grab_frame(args.cam2, args.name2, args.seek),
    }

    overlap = {}
    for name, frame in frames.items():
        zone = pick(frame, f"{name}: OVERLAP zone (where both cameras see the roll)")
        if zone:
            overlap[name] = zone

    work = pick(frames[args.name1], f"{args.name1}: WORK zone (ignore everything outside)")

    config = {"identity": {"overlap_roi": overlap}}
    if work:
        config["detect"] = {"roi": list(work)}

    text = json.dumps(config, indent=2)
    print("\n" + text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten to {args.out} -- pass it with: python run.py --config {args.out} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
