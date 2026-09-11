"""Per-camera multi-object tracking plus multi-frame read voting.

The tracker is intentionally plain -- greedy IoU/centroid association, no deep
re-identification model.  With at most a handful of rolls in view, a learned
re-ID network would cost more CPU and RAM on the Pi than it could possibly buy
back in accuracy.

Voting is the part that carries the accuracy.  Any single frame can be blurred,
clipped by the frame edge or angled away from the lens, and individual reads
fail often.  Requiring the same value from several independent frames turns a
mediocre per-frame read rate into a reliable per-pass one.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter, deque
from dataclasses import dataclass, field

from .detect import Detection


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    camera: str
    bbox: tuple[int, int, int, int]
    writing_bbox: tuple[int, int, int, int]
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    sharpness: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=120))

    # Read state, accumulated over the pass.
    ply_votes: Counter = field(default_factory=Counter)
    range_votes: Counter = field(default_factory=Counter)
    raw_reads: list = field(default_factory=list)
    ocr_attempts: int = 0
    last_ocr_at: float = 0.0
    pending_ocr: bool = False

    confirmed_ply: str | None = None
    confirmed_range: tuple[str, str] | None = None
    global_id: str | None = None
    reported: bool = False

    @property
    def centroid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def complete(self) -> bool:
        """Enough to report: the two lengths agree across frames.

        The ply line sits above the lengths and is the first thing lost when the
        roll is clipped by the frame edge, so it cannot be a precondition for
        reporting -- the lengths alone identify a roll in the master list.
        """
        return self.confirmed_range is not None

    @property
    def fully_read(self) -> bool:
        return self.confirmed_range is not None and self.confirmed_ply is not None

    def displacement(self, window_s: float, now: float) -> float:
        """How far the roll has travelled recently.

        Rolls stacked on background racks never move; live ones carried by an
        operator always do.  That is the cheapest way to tell them apart.
        """
        recent = [(t, c) for t, c in self.history if now - t <= window_s]
        if len(recent) < 2:
            return 0.0
        xs = [c[0] for _, c in recent]
        ys = [c[1] for _, c in recent]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    def update(self, detection: Detection, now: float) -> None:
        self.bbox = detection.bbox
        self.writing_bbox = detection.writing_bbox
        self.sharpness = detection.sharpness
        self.last_seen = now
        self.hits += 1
        self.misses = 0
        self.history.append((now, self.centroid))

    def add_reading(self, ply: str | None, chosen, cfg) -> None:
        if ply:
            self.ply_votes[ply] += 1
        if chosen is not None:
            self.range_votes[(chosen.start, chosen.end)] += 1
        self._confirm(cfg)

    def _confirm(self, cfg) -> None:
        self.confirmed_ply = _winner(self.ply_votes, cfg.min_votes, cfg.vote_margin)
        self.confirmed_range = _winner(self.range_votes, cfg.min_votes, cfg.vote_margin)

    def confidence(self) -> float:
        """Share of the vote held by the winners -- a rough agreement measure."""
        parts = []
        for votes, winner in ((self.ply_votes, self.confirmed_ply),
                              (self.range_votes, self.confirmed_range)):
            total = sum(votes.values())
            if total and winner is not None:
                parts.append(votes[winner] / total)
        return round(sum(parts) / len(parts), 3) if parts else 0.0


def _winner(votes: Counter, min_votes: int, margin: int):
    if not votes:
        return None
    ranked = votes.most_common(2)
    top, top_count = ranked[0]
    if top_count < min_votes:
        return None
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    return top if (top_count - runner_up) >= margin else None


class Tracker:
    def __init__(self, camera: str, cfg, frame_size: tuple[int, int]) -> None:
        self.camera = camera
        self.cfg = cfg
        self.tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)
        w, h = frame_size
        self._max_dist = (w ** 2 + h ** 2) ** 0.5 * cfg.max_center_dist_frac

    def update(self, detections: list[Detection], now: float | None = None) -> list[Track]:
        now = time.time() if now is None else now

        pairs = []
        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                overlap = iou(track.bbox, detection.bbox)
                tx, ty = track.centroid
                dx, dy = detection.centroid
                distance = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5
                if overlap < self.cfg.min_iou and distance > self._max_dist:
                    continue
                score = (self.cfg.iou_weight * overlap
                         + (1 - self.cfg.iou_weight) * (1 - min(distance / self._max_dist, 1.0)))
                pairs.append((score, track_id, index))

        pairs.sort(reverse=True)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for score, track_id, index in pairs:
            if track_id in used_tracks or index in used_dets:
                continue
            self.tracks[track_id].update(detections[index], now)
            used_tracks.add(track_id)
            used_dets.add(index)

        for index, detection in enumerate(detections):
            if index in used_dets:
                continue
            track_id = next(self._ids)
            track = Track(
                id=track_id, camera=self.camera,
                bbox=detection.bbox, writing_bbox=detection.writing_bbox,
                first_seen=now, last_seen=now, sharpness=detection.sharpness,
            )
            track.history.append((now, track.centroid))
            self.tracks[track_id] = track

        expired = []
        for track_id, track in self.tracks.items():
            if track_id in used_tracks:
                continue
            track.misses += 1
            if track.misses > self.cfg.max_misses:
                expired.append(track_id)

        dropped = [self.tracks.pop(track_id) for track_id in expired]
        return dropped

    def active(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.hits >= self.cfg.min_hits]
