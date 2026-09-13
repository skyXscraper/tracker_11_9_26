"""Tests for running at a steady frame rate.

Two failures motivated this. OCR ran inline on recordings, so every 1-2 s read
froze the loop and the frame rate collapsed. And the recorder was hard-coded to
15 fps while writing every frame of a 30 fps source, so recordings played at
half speed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import Pacer                              # noqa: E402
from rollocr.sources import VideoFileSource        # noqa: E402


def test_each_tick_writes_one_recording_frame_when_on_time():
    pacer = Pacer(15)
    assert [pacer.frames_owed(t) for t in range(5)] == [1, 1, 1, 1, 1]


def test_a_late_tick_repeats_frames_so_the_recording_keeps_true_speed():
    """If ticks 1-3 are skipped because processing ran long, the recording must
    still gain 4 frames by tick 4, or the video would play back too fast."""
    pacer = Pacer(15)
    pacer.frames_owed(0)
    assert pacer.frames_owed(4) == 4


def test_a_slow_iteration_skips_ahead_instead_of_building_a_backlog():
    pacer = Pacer(50)                   # 20 ms slots, so the test stays quick
    pacer.begin()
    assert pacer.due_tick() == 0
    time.sleep(0.105)                   # overrun by several slots
    tick = pacer.due_tick()
    assert tick >= 4, "must jump to the tick that is due now"
    assert pacer.dropped >= 3


def test_ticks_never_go_backwards_or_repeat():
    pacer = Pacer(1000)
    pacer.begin()
    ticks = [pacer.due_tick() for _ in range(20)]
    assert ticks == sorted(set(ticks)), "each iteration serves a new tick"


def test_waiting_holds_the_loop_to_the_target_rate():
    pacer = Pacer(40)                   # 25 ms slots
    pacer.begin()
    started = time.perf_counter()
    for _ in range(8):
        pacer.wait_for_next(pacer.due_tick())
    elapsed = time.perf_counter() - started
    assert 0.17 <= elapsed <= 0.40, f"8 ticks at 40 fps took {elapsed:.3f}s"


def test_a_30fps_recording_can_be_sampled_at_15fps(tmp_path):
    """read_at must hand back the frame due at each 15 fps tick, skipping the
    in-between frames, so a 30 fps file runs at real speed at 15 fps."""
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for index in range(60):                         # 2 s at 30 fps
        frame = np.full((48, 64, 3), index * 4 % 256, np.uint8)
        writer.write(frame)
    writer.release()

    source = VideoFileSource(path, "cam")
    served = []
    tick = 0
    while True:
        frame = source.read_at(tick / 15.0)
        if frame is None:
            break
        served.append(frame.timestamp)
        tick += 1
    source.release()

    assert 28 <= len(served) <= 31, f"expected ~30 frames for 2 s at 15 fps, got {len(served)}"
    gaps = np.diff(served)
    assert np.all(gaps > 0), "timestamps must advance"
    assert abs(float(np.median(gaps)) - 1 / 15.0) < 0.02
