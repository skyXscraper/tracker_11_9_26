"""Configuration for the roll OCR pipeline.

Defaults are tuned for a Raspberry Pi 5 (2 GB) running two USB cameras, and
are deliberately conservative about CPU and RAM.  Any field can be overridden
from a JSON file passed with ``--config``; only the keys you list are changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path


@dataclass
class CaptureConfig:
    width: int = 960                 # matches the supplied test recordings
    height: int = 1080
    fps: int = 25
    fourcc: str = "MJPG"             # essential: two raw YUYV streams saturate USB
    buffer_size: int = 1             # keep latency down; we always want the newest frame


@dataclass
class DetectConfig:
    """Red handwriting is the detection signal -- it is the one thing on a roll
    that no desk, chair or floor stain reliably imitates."""

    scale: float = 0.4               # detect on a downscaled copy, report full-res boxes
    detect_every: int = 2            # run detection every Nth frame

    # HSV gates for red ink.  Hue wraps, so red needs two bands.
    # Tuned on the supplied footage: through the glossy wrap the ink reads as
    # pale magenta (H~170, S 40-80), and skin -- the one dangerous false
    # positive -- sits on the orange side with HIGHER saturation.  So the
    # magenta band does the work and the orange band stays narrow and strict.
    sat_min: int = 32
    val_min: int = 100
    hue_hi_min: int = 158
    hue_lo_max: int = 6          # set to 0 to disable the orange-side band
    red_excess_min: int = 12     # R - max(G, B); ink barely reaches 45 here

    # Stroke / text-line grouping (all in downscaled pixels).
    close_kernel: tuple[int, int] = (9, 3)
    min_stroke_area: int = 12
    group_gap_x: int = 30        # ink blobs closer than this (downscaled px) join up
    group_gap_y: int = 20        # kept smaller: only the two written lines should merge
    min_strokes: int = 2             # a real marking has several strokes
    min_group_area: int = 120

    # Full-resolution sanity gates.
    min_text_height_px: int = 14     # below this PP-OCR recognition collapses
    roll_pad_frac: float = 0.45      # grow the writing box to take in the roll body

    # Reject the rolls stacked on racks in the background: they never move.
    require_motion: bool = True
    motion_px: float = 6.0           # centroid travel over motion_window_s to count as live
    motion_window_s: float = 3.0

    # Optional work zone, as fractions of the frame (x0, y0, x1, y1).
    roi: tuple[float, float, float, float] | None = None


@dataclass
class TrackConfig:
    iou_weight: float = 0.7
    max_center_dist_frac: float = 0.18   # of frame diagonal
    min_iou: float = 0.05
    max_misses: int = 18                 # frames a track survives unseen (~0.7 s at 25 fps)
    min_hits: int = 3                    # detections before a track is trusted


@dataclass
class OcrConfig:
    backend: str = "rapidocr"        # PP-OCR models on ONNX Runtime
    num_threads: int = 2             # leave cores for capture on the Pi's 4
    target_text_height: int = 80     # measured better than 48 on this footage
    max_side: int = 960              # cap the crop we hand to detection
    queue_size: int = 4              # bounded: back-pressure instead of RAM growth

    min_interval_s: float = 0.45     # per track, throttles OCR attempts
    min_sharpness: float = 28.0      # variance of Laplacian; skips motion-blurred frames
    max_attempts: int = 40           # give up on a track after this many reads

    # Multi-frame voting before a value is treated as read.
    max_per_frame: int = 2       # cap OCR submissions per camera per frame
    min_votes: int = 2
    vote_margin: int = 1             # winner must lead the runner-up by this much

    deskew: bool = True
    save_debug_crops: bool = False


@dataclass
class ValueConfig:
    """Constraints that make the dotted handwriting parseable.

    On the roll the operator writes ply on top and ``start`` / ``end`` below,
    but the separator is inconsistent -- sometimes a dash, often just another
    dot, e.g. ``48.3.65.433`` meaning 48.3 and 65.433.  These bounds resolve
    that split: on the supplied master list they leave 53 of 54 rows with
    exactly one legal reading.
    """

    ply_min_digits: int = 2
    ply_max_digits: int = 4
    value_max: float = 100.0         # master list tops out at 65.433 m
    value_min: float = 0.0
    prefer_end_gt_start: bool = True   # a preference, not a filter: clipped
                                      # rolls lose digits and still matter
    max_decimals: int = 3
    length_tolerance_m: float = 0.06  # end - start must equal length_m in the master list


@dataclass
class IdentityConfig:
    """Cross-camera identity.  The cameras overlap, so a roll can be linked by
    being in both views at once, then confirmed by the ply number it carries."""

    sync_window_s: float = 1.5       # co-occurrence window for a provisional link
    twin_similarity: float = 0.8     # how alike two unidentified reads must be
                                     # to be treated as the same roll
    # Separate from sync_window_s on purpose.  That one is about being in both
    # views at the same instant; this one is about the same roll reaching the
    # other camera at all.  In the test recordings the operator presented the
    # roll to one camera and then walked to the other, 14 s apart, so a window
    # sized for simultaneity split one roll into two records.
    twin_window_s: float = 30.0
    overlap_roi: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    forget_after_s: float = 180.0    # drop a roll from the registry after this idle time
    id_prefix: str = "ROLL"


@dataclass
class OutputConfig:
    dir: str = "output"
    jsonl_name: str = "sightings.jsonl"
    csv_name: str = "rolls.csv"
    display: bool = True
    display_scale: float = 0.5
    record: bool = False
    annotate_fps: bool = True


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    values: ValueConfig = field(default_factory=ValueConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    master_csv: str = "data/master_list.csv"

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        if not path:
            return cfg
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for section, values in raw.items():
            if not hasattr(cfg, section):
                raise KeyError(f"unknown config section: {section}")
            current = getattr(cfg, section)
            if is_dataclass(current) and isinstance(values, dict):
                for key, value in values.items():
                    if not hasattr(current, key):
                        raise KeyError(f"unknown config key: {section}.{key}")
                    # JSON has no tuples; restore them where the default is one.
                    if isinstance(getattr(current, key), tuple) and isinstance(value, list):
                        value = tuple(value)
                    setattr(current, key, value)
            else:
                setattr(cfg, section, values)
        return cfg

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=list)
