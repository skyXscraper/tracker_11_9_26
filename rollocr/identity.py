"""One identity per physical roll, across both cameras.

Two things are kept strictly apart in this module, because conflating them is
how an OCR system quietly starts reporting fiction:

  * what was **read** off the roll -- ``read_ply``, ``read_start``, ``read_end``.
    These come from the camera and are never overwritten from the master list.
  * what that reading was **matched** to -- ``matched_ply`` and the expected
    values from the list. This is identity, and it is allowed to be inferred.

So a roll whose writing came back as "4.7-135" is reported as having been read
as 4.7-135, matched to ply 99 whose list values are 4.7/17.5, with a status
saying the two disagree. The operator sees both numbers and can tell which is
which. Nothing silently becomes the "right" answer.

Identity across the two cameras is resolved in order of evidence quality:

1.  An exact ply number read off the roll.
2.  An exact pair of lengths -- unique per row in the list, so this recovers a
    roll whose ply line was clipped by the frame edge.
3.  A best fuzzy match against the list, which handles the recogniser's
    systematic handwriting errors.
4.  Failing all three, co-occurrence in the cameras' overlap zone, so the roll
    still carries one consistent ID through both views.
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

# How a roll's identity was established, best first.
SOURCE_OCR_EXACT = "ocr_exact"
SOURCE_VALUES_EXACT = "values_exact"
SOURCE_FUZZY = "fuzzy"

STATUS_MATCH = "match"                 # read values equal the master row
STATUS_VALUE_MISMATCH = "value_mismatch"   # identified, but the read disagrees
STATUS_UNIDENTIFIED = "unidentified"   # nothing in the list resembles this


@dataclass
class GlobalRoll:
    global_id: str

    # Read off the roll by OCR. Never sourced from the master list.
    read_ply: str | None = None
    read_start: str | None = None
    read_end: str | None = None

    # Resolved against the master list.
    matched_ply: str | None = None
    match_score: float = 0.0
    ply_source: str | None = None
    status: str = "pending"
    expected_start: str | None = None
    expected_end: str | None = None
    expected_length_m: str | None = None
    item_number: str | None = None
    item_description: str | None = None

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
    def read_range_text(self) -> str:
        return "{} - {}".format(self.read_start or "?", self.read_end or "?")

    @property
    def ply(self) -> str | None:
        """The ply this roll is believed to be, however that was established."""
        return self.matched_ply or self.read_ply


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
        and since a link is later swept onto the real roll's identity, a desk
        and a chair leg end up labelled as the roll. Un-calibrated cameras fall
        back to linking on what was read instead, which needs no geometry.
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
        """Give an as-yet-unread track a temporary identity.

        It is shared with a track in the other camera when both are sitting in
        the overlap zone, so the roll carries one ID from the first frame rather
        than only once its writing has been read.
        """
        key = (camera, track.id)
        if key in self._assigned:
            return self._assigned[key]

        if self.in_overlap(camera, track.bbox):
            for other_key, global_id in self._assigned.items():
                if other_key[0] == camera:
                    continue
                roll = self.rolls.get(global_id)
                if roll is None or not roll.provisional or roll.matched_ply:
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

    def _resolve(self, master, read_ply, start, end):
        """Find this roll in the master list. Returns (row, source, score)."""
        row = master.lookup(read_ply)
        if row is not None:
            return row, SOURCE_OCR_EXACT, 1.0

        row = master.lookup_by_values(start, end)
        if row is not None:
            return row, SOURCE_VALUES_EXACT, 1.0

        row, score = master.best_match(read_ply, start, end)
        if row is not None:
            return row, SOURCE_FUZZY, score
        return None, None, score

    def confirm(self, camera: str, track, master, values_cfg, now: float) -> GlobalRoll:
        """Attach a track to its identity, now that its values have been read."""
        key = (camera, track.id)
        previous_id = self._assigned.get(key)
        start, end = track.confirmed_range or (None, None)
        read_ply = track.confirmed_ply

        row, source, score = self._resolve(master, read_ply, start, end)
        matched_ply = row.ply_no if row is not None else None
        target_id = self._target_id(matched_ply, previous_id, camera, start, end, now)

        if previous_id and previous_id != target_id:
            self._migrate(previous_id, target_id, now)

        roll = self.rolls.get(target_id) or self._new_roll(target_id, False, now)
        roll.provisional = False
        roll.read_ply, roll.read_start, roll.read_end = read_ply, start, end
        roll.matched_ply = matched_ply
        roll.ply_source = source
        roll.match_score = score
        roll.confidence = track.confidence()

        if row is None:
            roll.status = STATUS_UNIDENTIFIED
            roll.expected_start = roll.expected_end = roll.expected_length_m = None
            roll.item_number = roll.item_description = None
        else:
            roll.expected_start, roll.expected_end = row.start, row.end
            roll.expected_length_m = row.length_m
            roll.item_number = row.item_number or None
            roll.item_description = row.item_description or None
            exact = (start, end) == (row.start, row.end)
            roll.status = STATUS_MATCH if exact else STATUS_VALUE_MISMATCH
            self.by_ply[matched_ply] = target_id

        # Remember where this track points, so the next read on the same track
        # updates this record instead of minting a second one for the same roll.
        self._assigned[key] = target_id
        roll.seen_by(camera, track.id, track.bbox, now)
        return roll

    def _target_id(self, matched_ply: str | None, previous_id: str | None,
                   camera: str, start, end, now: float) -> str:
        """Decide which identity a freshly read track belongs to.

        Stability is the point of this method. An ID that has already settled is
        never swapped for another settled one because a later frame read the
        lengths slightly differently -- that is what turns one roll into three
        rows in the results. It moves in one direction only: unidentified, then
        identified, then never again.
        """
        if matched_ply:
            return self.by_ply.get(matched_ply) or "{}-{}".format(
                self.cfg.id_prefix, matched_ply)

        previous = self.rolls.get(previous_id) if previous_id else None
        if previous is not None and not previous.provisional:
            return previous_id      # already settled: never re-key it

        # Still only a placeholder, so a roll the other camera just read the
        # same way is better evidence than the ID we happened to mint.
        twin = self._recent_twin(camera, start, end, now)
        if twin is not None:
            return twin
        if previous_id:
            return previous_id
        return "{}-U{}".format(self.cfg.id_prefix, next(self._temp_ids))

    def _recent_twin(self, camera: str, start, end, now: float) -> str | None:
        """An unidentified roll the other camera just read the same way.

        A roll that matches no master row still has to carry one ID through both
        views. With overlapping fields of view, the same writing read in the
        other camera moments ago is the same roll -- two different rolls being
        presented simultaneously with near-identical markings is not a situation
        that arises.
        """
        if not start or not end:
            return None
        digits = _digits(start) + _digits(end)
        for global_id, roll in self.rolls.items():
            if roll.provisional or roll.matched_ply:
                continue
            if camera in roll.cameras:
                continue                       # already this camera's own record
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
            if previous.matched_ply:
                self.by_ply.pop(previous.matched_ply, None)
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
            if roll.matched_ply:
                self.by_ply.pop(roll.matched_ply, None)
            for key in list(self._assigned):
                if self._assigned[key] == global_id:
                    self._assigned.pop(key, None)
