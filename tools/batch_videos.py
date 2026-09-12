#!/usr/bin/env python3
"""Run the pipeline over a folder of single-camera recordings, in batch.

`run.py` handles one clip at a time; this walks a dataset of them and writes a
per-clip result plus one summary across the lot. Use it to sweep a shift's
recordings, or to compare a config change across a whole set.

    python tools/batch_videos.py --dataset path/to/dataset --out path/to/results

It looks for `<folder>/cam0.mp4` under the dataset, falling back to any `.mp4`
it can find. For each clip it writes, under `--out/<clip>/`:

  * `annotated.mp4`   - the source video with detections, tracks and reads drawn
  * `rolls.csv`       - one row per roll read in that clip
  * `sightings.jsonl` - every OCR read, in order, including the failures

and a `summary.csv` across the whole dataset at the top level.

Like the rest of the pipeline it reports what OCR read -- the ply number and
the lengths as start-end -- and consults no packing list.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollocr import annotate                        # noqa: E402
from rollocr.config import Config                   # noqa: E402
from rollocr.identity import RollRegistry           # noqa: E402
from rollocr.logio import ResultLog                 # noqa: E402
from rollocr.ocr import OcrEngine, SyncOcrRunner    # noqa: E402
from rollocr.parse import parse_lines               # noqa: E402
from rollocr.pipeline import CameraPipeline         # noqa: E402
from rollocr.sources import VideoFileSource         # noqa: E402

CAMERA = "cam0"


def process(video: Path, out_dir: Path, cfg: Config, engine: OcrEngine,
            quiet: bool = False) -> dict:
    """Run one clip end to end. Returns a summary row."""
    out_dir.mkdir(parents=True, exist_ok=True)

    source = VideoFileSource(str(video), CAMERA)
    pipeline = CameraPipeline(CAMERA, source, cfg)
    registry = RollRegistry(cfg.identity, {CAMERA: pipeline.frame_size})
    worker = SyncOcrRunner(engine, cfg.ocr)

    cfg.output.dir = str(out_dir)
    log = ResultLog(cfg.output)

    width, height = pipeline.frame_size
    fps = source.fps or 25.0
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    started = time.time()
    frames = 0
    reads = 0
    while True:
        frame = pipeline.read()
        if frame is None:
            break
        frames += 1
        now = frame.timestamp          # media clock: reproducible run to run

        tracks, dropped = pipeline.step(frame, now)
        for track in tracks:
            if track.global_id is None:
                track.global_id = registry.assign_provisional(CAMERA, track, now)
            registry.observe(CAMERA, track, now)
        for request in pipeline.ocr_candidates(frame, now):
            worker.submit(request)
        for track in dropped:
            registry.release(CAMERA, track.id)

        reads += fold_results(worker, pipeline, registry, cfg, log, now, quiet)
        registry.cleanup(now)

        canvas = frame.image.copy()
        annotate.draw_roi(canvas, cfg.detect.roi, label="work zone")
        annotate.draw_tracks(canvas, pipeline.tracker.active(), registry, CAMERA)
        annotate.draw_status(canvas, CAMERA, fps, {
            "tracks": len(pipeline.tracker.active()),
            "ocr": worker.processed,
            "dropped": worker.dropped,
        })
        annotate.draw_header(canvas, frames / max(time.time() - started, 1e-6),
                             {CAMERA: fps},
                             {"rolls": sum(1 for r in registry.rolls.values()
                                           if not r.provisional),
                              "ocr": worker.processed, "dropped": worker.dropped})
        writer.write(canvas)

    writer.release()
    pipeline.release()
    log.close()

    read_rolls = [r for r in registry.rolls.values() if not r.provisional]
    return {
        "clip": video.parent.name,
        "frames": frames,
        "seconds": round(frames / fps, 1),
        "ocr_calls": worker.processed,
        "text_reads": reads,
        "rolls": len(read_rolls),
        "with_ply": sum(1 for r in read_rolls if r.read_ply),
        "detail": "; ".join(
            f"{r.global_id} ply {r.read_ply or '?'} {r.range_text}"
            for r in sorted(read_rolls, key=lambda r: r.first_seen)) or "-",
        "runtime_s": round(time.time() - started, 1),
    }


def fold_results(worker, pipeline, registry, cfg, log, now, quiet) -> int:
    """The same bookkeeping run.py does, for a single camera."""
    reads = 0
    for result in worker.drain():
        track = pipeline.track_by_id(result.track_id)
        if track is None:
            continue
        track.pending_ocr = False
        if not result.lines:
            continue
        reads += 1

        track.raw_reads.append([text for text, _, _ in result.lines])
        ply, readings = parse_lines(result.lines, cfg.values)
        chosen = readings[0] if readings else None
        before = (track.confirmed_ply, track.confirmed_range)
        track.add_reading(ply, chosen, cfg.ocr)

        log.event("read", camera=CAMERA, track=track.id,
                  raw=[text for text, _, _ in result.lines], ply=ply,
                  start=chosen.start if chosen else None,
                  end=chosen.end if chosen else None,
                  ms=round(result.elapsed_ms))

        if track.complete and (track.confirmed_ply, track.confirmed_range) != before:
            roll = registry.confirm(CAMERA, track, now)
            track.global_id = roll.global_id
            first = not track.reported
            track.reported = True
            log.event("confirmed", camera=CAMERA, track=track.id,
                      global_id=roll.global_id, ply=roll.read_ply,
                      start=roll.read_start, end=roll.read_end,
                      range=roll.range_text, status=roll.status,
                      confidence=roll.confidence)
            if track.logged_id and track.logged_id != roll.global_id:
                log.drop(track.logged_id)
            track.logged_id = roll.global_id
            log.roll(roll)
            if first and not quiet:
                print(f"    [roll] {roll.global_id}   ply {roll.read_ply or '?'}"
                      f"   {roll.range_text}   [{roll.status}]")
    return reads


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(dataset.glob("*/cam0.mp4"))
    if not videos:
        videos = sorted(dataset.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"no videos found under {dataset}")

    cfg = Config.load(args.config)
    engine = OcrEngine(cfg.ocr, cfg.detect)   # one engine shared by every clip

    print(f"[batch] {len(videos)} clips, OCR-only (no packing list)")
    rows = []
    for index, video in enumerate(videos, 1):
        name = video.parent.name
        print(f"\n[{index}/{len(videos)}] {name}")
        row = process(video, out_root / name, cfg, engine, args.quiet)
        rows.append(row)
        print(f"    frames={row['frames']} ocr={row['ocr_calls']} "
              f"reads={row['text_reads']} rolls={row['rolls']} "
              f"({row['runtime_s']}s)")

    fields = ["clip", "frames", "seconds", "ocr_calls", "text_reads",
              "rolls", "with_ply", "runtime_s", "detail"]
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[batch] done. {sum(r['rolls'] for r in rows)} rolls read across "
          f"{len(rows)} clips -> {out_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
