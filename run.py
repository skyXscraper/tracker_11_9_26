#!/usr/bin/env python3
"""Sheet-roll tracking and handwriting OCR. One script, any source.

Give it one or two sources -- a still image, recorded video, or a live camera:

    python run.py photo.jpg                             # a single image
    python run.py videos/cam0_test3.mp4                 # one recording
    python run.py videos/cam0_test3.mp4 videos/cam2_test3.mp4
    python run.py /dev/video0                           # one live camera
    python run.py /dev/video0 /dev/video2               # both cameras

A source is a camera when it is a device index or a /dev/videoN path, an image
when it has an image extension, and a recording otherwise. Two sources are
tracked as two views of the same work area, so a roll carried from one into
the other keeps a single ID.

Output is what OCR read, in the format the marking is written in: the ply
number, then the lengths as start-end. Nothing is checked against or corrected
towards a packing list.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from pathlib import Path

from rollocr import annotate
from rollocr.config import Config
from rollocr.detect import RollDetector
from rollocr.logio import ResultLog
from rollocr.ocr import OcrEngine, OcrWorker, SyncOcrRunner
from rollocr.parse import parse_lines
from rollocr.pipeline import CameraPipeline, writing_crop
from rollocr.identity import RollRegistry
from rollocr.sources import open_source


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def run_directory(output_cfg, sources) -> str:
    """A folder of its own for this run, named for when and what.

    Without this, every run overwrites the last one's annotated video and CSV,
    and comparing a change against the run before it means having remembered to
    copy the results out first.
    """
    if not output_cfg.per_run_dir:
        return output_cfg.dir
    labels = []
    for spec in sources:
        stem = Path(spec).stem if not str(spec).isdigit() else f"cam{spec}"
        labels.append("".join(ch for ch in stem if ch.isalnum() or ch in "-_")[:24])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return str(Path(output_cfg.dir) / f"{stamp}_{'+'.join(labels) or 'run'}")


def is_image(spec: str) -> bool:
    return Path(spec).suffix.lower() in IMAGE_SUFFIXES


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+",
                        help="one or two of: image file, video file, device index, /dev/videoN")
    parser.add_argument("--names", nargs="+", default=None,
                        help="names for the sources (default cam1, cam2)")
    parser.add_argument("--config", help="JSON file overriding any config field")
    parser.add_argument("--output", help="output directory")
    parser.add_argument("--no-display", action="store_true", help="headless (use on the Pi)")
    parser.add_argument("--no-record", action="store_true",
                        help="skip the annotated video (a little faster on the Pi)")
    parser.add_argument("--no-run-dir", action="store_true",
                        help="write straight into --output instead of a timestamped subfolder")
    parser.add_argument("--realtime", action="store_true",
                        help="pace video files at their own frame rate")
    parser.add_argument("--max-seconds", type=float, help="stop after this long")
    parser.add_argument("--quiet", action="store_true", help="only print confirmed rolls")
    return parser.parse_args()


def main() -> int:
    # PP-OCR carries a CJK charset and can emit non-Latin characters from noise;
    # a default-encoded Windows console dies on them.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_args()
    if len(args.sources) > 2:
        raise SystemExit("at most two sources; got {}".format(len(args.sources)))
    cfg = Config.load(args.config)
    if args.output:
        cfg.output.dir = args.output
    if args.no_display:
        cfg.output.display = False
    if args.no_record:
        cfg.output.record = False
    if args.no_run_dir:
        cfg.output.per_run_dir = False
    cfg.output.dir = run_directory(cfg.output, args.sources)

    names = args.names or ["cam1", "cam2"]
    if len(names) < len(args.sources):
        names = names + [f"cam{i}" for i in range(len(names) + 1, len(args.sources) + 1)]

    # A still image has no motion to track and no frames to vote over, so it
    # takes a direct read rather than the streaming pipeline.
    if any(is_image(spec) for spec in args.sources):
        if len(args.sources) > 1:
            raise SystemExit("give one image at a time")
        return run_image(args.sources[0], cfg)

    # No packing list is loaded: the pipeline reports what OCR read and
    # nothing else. Compare a finished run against the list afterwards with
    # tools/compare_to_master.py, where both columns stay visible.
    pipelines = {}
    for spec, name in zip(args.sources, names):
        source = open_source(spec, name, cfg.capture, realtime=args.realtime)
        pipelines[name] = CameraPipeline(name, source, cfg)
        print(f"[init] {name}: {spec} -> {source.size[0]}x{source.size[1]}")

    frame_sizes = {name: pipe.frame_size for name, pipe in pipelines.items()}
    registry = RollRegistry(cfg.identity, frame_sizes)
    log = ResultLog(cfg.output)

    # Video files carry their own clock; live cameras use the wall clock.
    offline = all(hasattr(p.source, "fps") for p in pipelines.values()) and not args.realtime
    use_media_clock = offline

    engine = OcrEngine(cfg.ocr, cfg.detect, cfg.exposure)
    # Offline: read every crop inline so a tuning run is complete and repeatable.
    # Live: a background thread with a bounded queue, so a slow read never
    # stalls capture and never grows memory behind a roll that has left.
    worker = SyncOcrRunner(engine, cfg.ocr) if offline else OcrWorker(engine, cfg.ocr)
    print("[init] ocr mode: {}".format("sync (offline)" if offline else "threaded (live)"))

    writer = None
    meter = FpsMeter()
    started = time.time()
    finished = set()
    confirmed_count = 0

    print(f"[init] results -> {cfg.output.dir}"
          f"{'  (annotated video on)' if cfg.output.record else ''}")
    print("[run] press q in the window, or ctrl-c, to stop")
    try:
        while len(finished) < len(pipelines):
            frames = {}
            for name, pipe in pipelines.items():
                if name in finished:
                    continue
                frame = pipe.read()
                if frame is None:
                    if hasattr(pipe.source, "fps"):
                        finished.add(name)      # file exhausted
                    continue
                frames[name] = frame

            if not frames:
                if len(finished) >= len(pipelines):
                    break
                time.sleep(0.005)
                continue

            now = (max(f.timestamp for f in frames.values())
                   if use_media_clock else time.time())

            for name, frame in frames.items():
                pipe = pipelines[name]
                tracks, dropped = pipe.step(frame, now)

                for track in tracks:
                    if track.global_id is None:
                        track.global_id = registry.assign_provisional(name, track, now)
                    registry.observe(name, track, now)

                for request in pipe.ocr_candidates(frame, now):
                    worker.submit(request)

                for track in dropped:
                    registry.release(name, track.id)

            confirmed_count += handle_ocr(worker, pipelines, registry, cfg, log, now,
                                          quiet=args.quiet)
            registry.cleanup(now)

            meter.tick()
            if cfg.output.display or cfg.output.record:
                canvas = render(frames, pipelines, registry, cfg, worker, meter.value)
                if canvas is not None:
                    if cfg.output.record:
                        writer = ensure_writer(writer, canvas, cfg)
                        writer.write(canvas)
                    if cfg.output.display:
                        cv2.imshow("rollocr - two cameras", canvas)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

            if args.max_seconds and (time.time() - started) > args.max_seconds:
                break
    except KeyboardInterrupt:
        print("\n[run] interrupted")
    finally:
        worker.stop()
        for pipe in pipelines.values():
            pipe.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        log.close()

    summarise(registry, log, worker, started, confirmed_count)
    return 0


def run_image(path: str, cfg) -> int:
    """Read one still image: detect the markings, OCR each, print what it says.

    No tracking and no multi-frame voting -- there is only one frame, so the
    reading stands on its own. Useful for checking the OCR end of the pipeline
    without a camera or a recording.
    """
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"cannot read image: {path}")
    height, width = image.shape[:2]
    print(f"[image] {path} -> {width}x{height}")

    detector = RollDetector(cfg.detect, (width, height), cfg.exposure)
    engine = OcrEngine(cfg.ocr, cfg.detect, cfg.exposure)
    detections = detector.detect(image)
    print(f"[image] markings found: {len(detections)}")

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rolls_dir = out_dir / "rolls"

    canvas = image.copy()
    found = 0
    for index, detection in enumerate(detections, 1):
        crop = writing_crop(image, detection.writing_bbox)
        if crop is None or crop.size == 0:
            continue
        lines = engine.read(crop)
        ply, readings = parse_lines(lines, cfg.values)
        raw = [text for text, _, _ in lines]
        if not lines:
            print(f"  [{index}] {detection.writing_bbox}  nothing read")
            continue
        found += 1
        values = f"{readings[0].start}-{readings[0].end}" if readings else "?"
        print(f"  [{index}] ply {ply or '?':<6} {values:<18} raw={raw}")

        if cfg.output.save_roll_images:
            rolls_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(rolls_dir / f"marking{index}_ply{ply or 'unknown'}.jpg"), crop)

        x0, y0, x1, y1 = detection.writing_bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 200, 90), 2)
        cv2.putText(canvas, f"ply {ply or '?'}  {values}", (x0, max(18, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 90), 2, cv2.LINE_AA)

    out_path = out_dir / (Path(path).stem + "_annotated.jpg")
    cv2.imwrite(str(out_path), canvas)
    print(f"[image] {found} marking(s) read")
    print(f"[image] results -> {out_dir}")
    return 0


def handle_ocr(worker, pipelines, registry, cfg, log, now, quiet: bool) -> int:
    """Fold finished OCR results back into their tracks, and confirm identities."""
    confirmed = 0
    for result in worker.drain():
        pipe = pipelines.get(result.camera)
        if pipe is None:
            continue
        track = pipe.track_by_id(result.track_id)
        if track is None:
            continue                      # the roll left before the read came back
        track.pending_ocr = False
        if not result.lines:
            continue

        track.raw_reads.append([text for text, _, _ in result.lines])
        ply, readings = parse_lines(result.lines, cfg.values)
        # Best reading by the parser's own constraints. Nothing is consulted to
        # decide which roll this "must" be.
        chosen = readings[0] if readings else None
        before = (track.confirmed_ply, track.confirmed_range)
        track.add_reading(ply, chosen, cfg.ocr)
        after = (track.confirmed_ply, track.confirmed_range)

        log.event("read", camera=result.camera, track=track.id,
                  raw=[text for text, _, _ in result.lines],
                  ply=ply, start=chosen.start if chosen else None,
                  end=chosen.end if chosen else None, ms=round(result.elapsed_ms))

        # Re-confirm on any change, not just the first: the ply line often
        # arrives several frames after the lengths, once the roll turns.
        if track.complete and after != before:
            roll = registry.confirm(result.camera, track, now)
            track.global_id = roll.global_id
            first_report = not track.reported
            track.reported = True
            confirmed += 1 if first_report else 0
            log.event("confirmed", camera=result.camera, track=track.id,
                      global_id=roll.global_id, ply=roll.read_ply,
                      start=roll.read_start, end=roll.read_end,
                      range=roll.range_text, status=roll.status,
                      confidence=roll.confidence)
            # The roll may have just been re-keyed from its placeholder id;
            # retire the row written under the old one.
            if track.logged_id and track.logged_id != roll.global_id:
                log.drop(track.logged_id)
            track.logged_id = roll.global_id
            log.roll(roll)
            # Re-saved on every confirmation, not just the first: a roll is
            # re-keyed when its ply is finally read, and a snapshot still
            # named after the placeholder ID cannot be matched to its CSV row.
            if cfg.output.save_roll_images:
                save_roll_image(cfg, roll, track)
            if first_report or not quiet:
                print("[roll] " + describe(roll, result.camera))
    return confirmed


class FpsMeter:
    """Frame rate of the whole loop, averaged over a short sliding window.

    Per-camera rates say how fast frames arrive; this says how fast the
    pipeline is actually getting through them, which is the number that tells
    you whether the Pi is keeping up.
    """

    def __init__(self, window_s: float = 1.0) -> None:
        self._window = window_s
        self._count = 0
        self._since = time.time()
        self.value = 0.0

    def tick(self) -> float:
        self._count += 1
        elapsed = time.time() - self._since
        if elapsed >= self._window:
            self.value = self._count / elapsed
            self._count = 0
            self._since = time.time()
        return self.value


def render(frames, pipelines, registry, cfg, worker, pipeline_fps=0.0):
    views = []
    for name, pipe in pipelines.items():
        frame = frames.get(name)
        image = (frame.image if frame is not None else pipe.last_frame)
        if image is None:
            continue
        canvas = image.copy()
        annotate.draw_roi(canvas, cfg.detect.roi, label="work zone")
        annotate.draw_roi(canvas, cfg.identity.overlap_roi.get(name),
                          colour=(200, 140, 60), label="overlap")
        annotate.draw_tracks(canvas, pipe.tracker.active(), registry, name)
        annotate.draw_status(canvas, name, pipe.fps, {
            "tracks": len(pipe.tracker.active()),
            "ocr": worker.processed,
            "dropped": worker.dropped,
        })
        views.append(canvas)

    canvas = annotate.stack(views, cfg.output.display_scale)
    if canvas is None:
        return None
    annotate.draw_header(
        canvas, pipeline_fps,
        {name: pipe.fps for name, pipe in pipelines.items()},
        {"rolls": sum(1 for r in registry.rolls.values() if not r.provisional),
         "ocr": worker.processed, "dropped": worker.dropped},
    )
    return canvas


def ensure_writer(writer, canvas, cfg):
    if writer is not None:
        return writer
    Path(cfg.output.dir).mkdir(parents=True, exist_ok=True)
    path = f"{cfg.output.dir}/annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 15.0, (canvas.shape[1], canvas.shape[0]))
    print(f"[run] recording annotated view to {path}")
    return writer


def save_roll_image(cfg, roll, track) -> None:
    """Save the crop this roll was read from, captioned with what it said.

    The annotated video shows the whole pass; this is the single frame that
    produced the reading, which is what you want when a number looks wrong.
    """
    crop = getattr(track, "last_crop", None)
    if crop is None or getattr(crop, "size", 0) == 0:
        return
    caption = "ply {}   {}".format(roll.read_ply or "?", roll.range_text)
    # Big enough to judge the handwriting against the reading by eye, which is
    # the whole reason for saving it.
    scale = min(6.0, max(1.0, 360.0 / max(crop.shape[0], 1)))
    canvas = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    banner = 40
    canvas = cv2.copyMakeBorder(canvas, banner, 0, 0, 0,
                                cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(canvas, caption, (10, banner - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (90, 220, 110), 2, cv2.LINE_AA)

    rolls_dir = Path(cfg.output.dir) / "rolls"
    rolls_dir.mkdir(parents=True, exist_ok=True)
    path = rolls_dir / f"{roll.global_id}.jpg"

    previous = getattr(track, "snapshot_path", None)
    if previous is not None and previous != path:
        try:
            previous.unlink()          # drop the placeholder-named copy
        except OSError:
            pass
    cv2.imwrite(str(path), canvas)
    track.snapshot_path = path


def describe(roll, camera: str | None = None) -> str:
    """One line in the format the marking is written in: ply, then start-end."""
    where = " ({})".format(camera) if camera else " [{}]".format(
        "+".join(roll.camera_names) or "-")
    tail = "  (ply not read)" if roll.status == "partial" else ""
    return "{}   ply {:<6} {:<18} conf {}{}{}".format(
        roll.global_id, roll.read_ply or "?", roll.range_text,
        roll.confidence, where, tail)


def summarise(registry, log, worker, started, confirmed_count) -> None:
    elapsed = time.time() - started
    print(f"\n[done] {elapsed:.1f}s  ocr_calls={worker.processed}  "
          f"dropped={worker.dropped}")
    rolls = [r for r in registry.rolls.values() if not r.provisional]
    print(f"[done] rolls read: {len(rolls)}  (confirmations: {confirmed_count})")
    for roll in sorted(rolls, key=lambda r: r.first_seen):
        print("       " + describe(roll))
    print(f"[done] results -> {Path(log.csv_path).parent}")
    print(f"       {Path(log.csv_path).name}, {Path(log.jsonl_path).name}"
          f"{', annotated.mp4' if worker and Path(log.csv_path).with_name('annotated.mp4').exists() else ''}")
    rolls_dir = Path(log.csv_path).parent / "rolls"
    if rolls_dir.exists():
        print(f"       rolls/ ({len(list(rolls_dir.glob('*.jpg')))} snapshots)")


if __name__ == "__main__":
    raise SystemExit(main())
