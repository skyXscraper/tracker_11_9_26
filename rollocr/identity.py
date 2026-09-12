"""One identity per physical roll, across both cameras, from OCR alone.

What a roll reports is what the camera read off it. The master list is not
consulted here -- not to supply values, and not to decide which roll this is.
An earlier version scored the OCR'd digits against all 54 master rows and
adopted the best-matching row's ply number, so a read of "19" / "4.7-135" was
reported as ply 99. That flatters the numbers: the station exists to verify
what is handwritten on each roll, and a pipeline that resolves its answer
against the expected answer verifies nothing.

So identity is resolved from the reading and from tracking:

1.  The ply number read off the roll. Two tracks that read the same ply are
    the same roll, in either camera.
2.  Co-occurrence in the cameras' overlap zone, before anything has been read,
    so the roll carries one ID from its first frame.
3.  A twin reading -- an as-yet-unidentified roll the other camera read the
    same way moments ago.

An ID moves in one direction only: unidentified, then identified by ply, then
never again. Without that rule a single roll became three rows in the results
as its reading drifted between frames.
"""

from __future__ import annotations

import difflib
import itertools
import re
from dataclasses import dataclass, field


def _digits(text) -> str:
    return re.sub(r"\D", "", text or "")


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


STATUS_PENDING = "pending"        # still reading
STATUS_READ = "read"              # ply and lengths both read
STATUS_PARTIAL = "partial"        # lengths read, ply not legible


@dataclass
class GlobalRoll:
    """Everything here was read from the roll by OCR."""

    global_id: str
    read_ply: str | None = None
    read_start: str | None = None
    read_end: str | None = None
    status: str = STATUS_PENDING
    confidence: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    cameras: dict = field(default_factory=dict)
    provisional: bool = True

    def seen_by(self, camera: str, track_id: int, bbox, now: float) -> None:
        self.cameras[camera] = {"track_id": track_id, "bbox": list(bbox), "last_seen": now}
        self.last_seen = now
        if not self.first_seen:
            self.first_seen = now

    @property
    def camera_names(self) -> list[str]:
        return sorted(self.cameras)

    @property
    def range_text(self) -> str:
        """The lengths as written on the roll: ``start-end``."""
        if not self.read_start or not self.read_end:
            return "?"
        return "{}-{}".format(self.read_start, self.read_end)

    @property
    def text(self) -> str:
        """The whole marking, in the format it is written in: ply then range."""
        return "ply {}   {}".format(self.read_ply or "?", self.range_text)

    @property
    def ply(self) -> str | None:
        return self.read_ply


class RollRegistry:
    def __init__(self, cfg, frame_sizes: dict[str, tuple[int, int]]) -> None:
        self.cfg = cfg
        self.frame_sizes = frame_sizes
        self.rolls: dict[str, GlobalRoll] = {}
        self.by_ply: dict[str, str] = {}
        self._assigned: dict[tuple[str, int], str] = {}
        self._temp_ids = itertools.count(1)

    # -- geometry ---------------------------------------------------------

    def in_overlap(self, camera: str, bbox) -> bool:
        """Is this track inside the camera's configured hand-off zone?

        With no zone configured this is False, not True. Treating the whole
        frame as an overlap zone links any two tracks that happen to coexist,
        and a link is later swept onto the real roll's identity, so desks and
        chair legs end up labelled as the roll.
        """
        roi = self.cfg.overlap_roi.get(camera)
        if not roi:
            return False
        width, height = self.frame_sizes.get(camera, (1, 1))
        rx0, ry0 = roi[0] * width, roi[1] * height
        rx1, ry1 = roi[2] * width, roi[3] * height
        x0, y0, x1, y1 = bbox
        return not (x1 < rx0 or x0 > rx1 or y1 < ry0 or y0 > ry1)

    # -- identity ---------------------------------------------------------

    def _new_roll(self, global_id: str, provisional: bool, now: float) -> GlobalRoll:
        roll = GlobalRoll(global_id=global_id, provisional=provisional,
                          first_seen=now, last_seen=now)
        self.rolls[global_id] = roll
        return roll

    def assign_provisional(self, camera: str, track, now: float) -> str:
        """Give an as-yet-unread track a temporary identity, shared with a track
        in the other camera when both sit in the overlap zone."""
        key = (camera, track.id)
        if key in self._assigned:
            return self._assigned[key]

        if self.in_overlap(camera, track.bbox):
            for other_key, global_id in self._assigned.items():
                if other_key[0] == camera:
                    continue
                roll = self.rolls.get(global_id)
                if roll is None or not roll.provisional or roll.read_ply:
                    continue
                if now - roll.last_seen > self.cfg.sync_window_s:
                    continue
                other = roll.cameras.get(other_key[0])
                if other and self.in_overlap(other_key[0], other["bbox"]):
                    self._assigned[key] = global_id
                    return global_id

        global_id = "{}-U{}".format(self.cfg.id_prefix, next(self._temp_ids))
        self._new_roll(global_id, provisional=True, now=now)
        self._assigned[key] = global_id
        return global_id

    def confirm(self, camera: str, track, now: float) -> GlobalRoll:
        """Attach a track to its identity, from what OCR read off the roll."""
        key = (camera, track.id)
        previous_id = self._assigned.get(key)
        start, end = track.confirmed_range or (None, None)
        ply = track.confirmed_ply

        target_id = self._target_id(ply, previous_id, camera, start, end, now)
        if previous_id and previous_id != target_id:
            self._migrate(previous_id, target_id, now)

        roll = self.rolls.get(target_id) or self._new_roll(target_id, False, now)
        roll.provisional = False
        roll.read_ply, roll.read_start, roll.read_end = ply, start, end
        roll.confidence = track.confidence()
        roll.status = STATUS_READ if ply else STATUS_PARTIAL

        if ply:
            self.by_ply[ply] = target_id
        self._assigned[key] = target_id
        roll.seen_by(camera, track.id, track.bbox, now)
        return roll

    def _target_id(self, ply: str | None, previous_id: str | None,
                   camera: str, start, end, now: float) -> str:
        """Which identity a freshly read track belongs to.

        Stability is the point. An ID that has settled is never swapped for
        another because a later frame read the lengths slightly differently --
        that is what turns one roll into three rows in the results.
        """
        if ply:
            return self.by_ply.get(ply) or "{}-{}".format(self.cfg.id_prefix, ply)

        previous = self.rolls.get(previous_id) if previous_id else None
        if previous is not None and not previous.provisional:
            return previous_id

        twin = self._recent_twin(camera, start, end, now)
        if twin is not None:
            return twin
        if previous_id:
            return previous_id
        return "{}-U{}".format(self.cfg.id_prefix, next(self._temp_ids))

    def _recent_twin(self, camera: str, start, end, now: float) -> str | None:
        """An unidentified roll the other camera just read the same way.

        With overlapping views, the same writing read in the other camera
        moments ago is the same roll: two rolls presented at once with
        near-identical markings is not a situation that arises.
        """
        if not start or not end:
            return None
        digits = _digits(start) + _digits(end)
        for global_id, roll in self.rolls.items():
            if roll.provisional or roll.read_ply:
                continue
            if camera in roll.cameras:
                continue
            if now - roll.last_seen > self.cfg.twin_window_s:
                continue
            other = _digits(roll.read_start) + _digits(roll.read_end)
            if _similarity(digits, other) >= self.cfg.twin_similarity:
                return global_id
        return None

    def _migrate(self, previous_id: str, target_id: str, now: float) -> None:
        """Move a record and every track pointing at it onto its real identity."""
        previous = self.rolls.pop(previous_id, None)
        target = self.rolls.get(target_id)
        if target is None:
            target = self._new_roll(target_id, provisional=False, now=now)
        if previous is not None:
            target.first_seen = min(target.first_seen or previous.first_seen,
                                    previous.first_seen or now)
            for camera, seen in previous.cameras.items():
                target.cameras.setdefault(camera, seen)
            if previous.read_ply:
                self.by_ply.pop(previous.read_ply, None)
        for member in list(self._assigned):
            if self._assigned[member] == previous_id:
                self._assigned[member] = target_id

    def observe(self, camera: str, track, now: float) -> GlobalRoll | None:
        global_id = self._assigned.get((camera, track.id))
        if global_id is None:
            return None
        roll = self.rolls.get(global_id)
        if roll is not None:
            roll.seen_by(camera, track.id, track.bbox, now)
        return roll

    def assigned_id(self, camera: str, track_id: int) -> str | None:
        return self._assigned.get((camera, track_id))

    def release(self, camera: str, track_id: int) -> None:
        self._assigned.pop((camera, track_id), None)

    def cleanup(self, now: float) -> None:
        stale = [gid for gid, roll in self.rolls.items()
                 if now - roll.last_seen > self.cfg.forget_after_s]
        for global_id in stale:
            roll = self.rolls.pop(global_id)
            if roll.read_ply:
                self.by_ply.pop(roll.read_ply, None)
            for key in list(self._assigned):
                if self._assigned[key] == global_id:
                    self._assigned.pop(key, None)
