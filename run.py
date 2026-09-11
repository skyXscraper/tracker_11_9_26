#!/usr/bin/env python3
"""Two-camera sheet-roll tracking and handwriting OCR.

Live on the Pi:

    python run.py --cam1 /dev/video0 --cam2 /dev/video2

Offline against the recordings, which is how the pipeline is meant to be
tuned before it goes near the plant:

    python run.py --cam1 videos/cam0_test1.mp4 --cam2 videos/cam2_test1.mp4

The same code path serves both; only the frame source differs.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from rollocr import annotate
from rollocr.config import Config
from rollocr.logio import ResultLog
from rollocr.master import MasterList
from rollocr.ocr import OcrEngine, OcrWorker, SyncOcrRunner
from rollocr.parse import parse_lines
from rollocr.pipeline import CameraPipeline
from rollocr.identity import RollRegistry
from rollocr.sources import open_source


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam1", required=True, help="device index, /dev/videoN, or a video file")
    parser.add_argument("--cam2", required=True, help="device index, /dev/videoN, or a video file")
    parser.add_argument("--name1", default="cam1")
    parser.add_argument("--name2", default="cam2")
    parser.add_argument("--config", help="JSON file overriding any config field")
    parser.add_argument("--master", help="path to master_list.csv")
    parser.add_argument("--output", help="output directory")
    parser.add_argument("--no-display", action="store_true", help="headless (use on the Pi)")
    parser.add_argument("--record", action="store_true", help="write an annotated video")
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
    cfg = Config.load(args.config)
    if args.master:
        cfg.master_csv = args.master
    if args.output:
        cfg.output.dir = args.output
    if args.no_display:
        cfg.output.display = False
    if args.record:
        cfg.output.record = True

    master = MasterList.load(cfg.master_csv)
    print(f"[init] master list: {len(master.rows)} rolls from {cfg.master_csv}")

    pipelines = {}
    for spec, name in ((args.cam1, args.name1), (args.cam2, args.name2)):
        source = open_source(spec, name, cfg.capture, realtime=args.realtime)
        pipelines[name] = CameraPipeline(name, source, cfg)
        print(f"[init] {name}: {spec} -> {source.size[0]}x{source.size[1]}")

    frame_sizes = {name: pipe.frame_size for name, pipe in pipelines.items()}
    registry = RollRegistry(cfg.identity, frame_sizes)
    log = ResultLog(cfg.output)

    # Video files carry their own clock; live cameras use the wall clock.
    offline = all(hasattr(p.source, "fps") for p in pipelines.values()) and not args.realtime
    use_media_clock = offline

    engine = OcrEngine(cfg.ocr, cfg.detect)
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

            confirmed_count += handle_ocr(worker, pipelines, registry, master, cfg, log, now,
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


def handle_ocr(worker, pipelines, registry, master, cfg, log, now, quiet: bool) -> int:
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
        chosen, _status = master.choose_reading(ply, readings, cfg.values)
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
            roll = registry.confirm(result.camera, track, master, cfg.values, now)
            track.global_id = roll.global_id
            first_report = not track.reported
            track.reported = True
            confirmed += 1 if first_report else 0
            log.event("confirmed", camera=result.camera, track=track.id,
                      global_id=roll.global_id, read_ply=roll.read_ply,
                      read_start=roll.read_start, read_end=roll.read_end,
                      matched_ply=roll.matched_ply, ply_source=roll.ply_source,
                      match_score=roll.match_score, status=roll.status,
                      expected_start=roll.expected_start,
                      expected_end=roll.expected_end,
                      confidence=roll.confidence)
            log.roll(roll)
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
    path = f"{cfg.output.dir}/annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 15.0, (canvas.shape[1], canvas.shape[0]))
    print(f"[run] recording annotated view to {path}")
    return writer


def describe(roll, camera: str | None = None) -> str:
    """One line stating what was read and, separately, what it matched.

    The two are never merged: an operator reading this must be able to tell a
    number the camera saw from a number the master list supplied.
    """
    read = "read " + roll.read_range_text
    if roll.read_ply:
        read = "read ply {}  {}".format(roll.read_ply, roll.read_range_text)
    where = " ({})".format(camera) if camera else " [{}]".format(
        "+".join(roll.camera_names) or "-")

    if roll.status == "unidentified":
        return "{}  {}  -- no master row resembles this{}".format(
            roll.global_id, read, where)

    tail = ""
    if roll.status == "value_mismatch":
        tail = "  -- list says {} - {} (match {} via {})".format(
            roll.expected_start, roll.expected_end, roll.match_score, roll.ply_source)
    return "{}  ply {}  {}{}  conf {}{}".format(
        roll.global_id, roll.matched_ply, read, tail, roll.confidence, where)


def summarise(registry, log, worker, started, confirmed_count) -> None:
    elapsed = time.time() - started
    print(f"\n[done] {elapsed:.1f}s  ocr_calls={worker.processed}  "
          f"dropped={worker.dropped}")
    rolls = [r for r in registry.rolls.values() if not r.provisional]
    print(f"[done] rolls identified: {len(rolls)}  (confirmations: {confirmed_count})")
    for roll in sorted(rolls, key=lambda r: r.first_seen):
        print("       " + describe(roll))
    print(f"[done] results: {log.csv_path}  events: {log.jsonl_path}")


if __name__ == "__main__":
    raise SystemExit(main())
