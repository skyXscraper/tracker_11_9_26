"""Tests for cross-camera identity, from OCR alone.

Two requirements are guarded here. The first is the one a plant operator would
notice immediately: a roll carried from camera 1 into camera 2 must keep one ID,
and must not acquire a second one because a later frame read its lengths
slightly differently.

The second matters more. Identity comes from what the camera read and from
tracking -- never from the packing list. An earlier version scored the OCR'd
digits against the master rows and adopted the best-matching row's ply number,
so a read of "19" / "4.7-135" was reported as ply 99. That makes the station
self-confirming: it resolves its answer against the expected answer and so
verifies nothing. These tests pin the reading through unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollocr.config import Config          # noqa: E402
from rollocr.identity import RollRegistry  # noqa: E402

FRAME_SIZES = {"cam1": (960, 1080), "cam2": (960, 1080)}


class FakeTrack:
    """Just the surface RollRegistry touches."""

    def __init__(self, track_id, bbox, ply=None, values=None, conf=1.0):
        self.id = track_id
        self.bbox = bbox
        self.confirmed_ply = ply
        self.confirmed_range = values
        self._conf = conf

    def confidence(self):
        return self._conf


@pytest.fixture
def registry():
    cfg = Config().identity
    # Both cameras catch the roll at the right-hand edge of frame, which is
    # where the two views actually overlap in the supplied footage.
    cfg.overlap_roi = {"cam1": (0.55, 0.0, 1.0, 1.0), "cam2": (0.55, 0.0, 1.0, 1.0)}
    return RollRegistry(cfg, FRAME_SIZES)


# -- one roll, one ID --------------------------------------------------------

def test_same_roll_keeps_one_id_across_both_cameras(registry):
    values = ("48.3", "65.433")

    first = registry.confirm("cam1", FakeTrack(1, (700, 400, 900, 560), "88", values), 10.0)
    # The same physical roll, now seen by the second camera as a new track.
    second = registry.confirm("cam2", FakeTrack(7, (720, 380, 930, 550), "88", values), 10.4)

    assert first.global_id == second.global_id == "ROLL-88"
    assert second.camera_names == ["cam1", "cam2"]


def test_overlap_link_shares_an_id_before_anything_is_read(registry):
    a, b = FakeTrack(1, (700, 400, 900, 560)), FakeTrack(2, (740, 390, 940, 570))

    first = registry.assign_provisional("cam1", a, now=5.0)
    registry.observe("cam1", a, now=5.0)
    second = registry.assign_provisional("cam2", b, now=5.3)

    assert first == second, "tracks co-occurring in the overlap zone should link"


def test_tracks_outside_the_overlap_zone_are_not_linked(registry):
    a, b = FakeTrack(1, (10, 400, 120, 560)), FakeTrack(2, (20, 390, 130, 570))

    first = registry.assign_provisional("cam1", a, now=5.0)
    registry.observe("cam1", a, now=5.0)
    second = registry.assign_provisional("cam2", b, now=5.3)

    assert first != second


def test_link_lapses_once_the_sync_window_has_passed(registry):
    a, b = FakeTrack(1, (700, 400, 900, 560)), FakeTrack(2, (740, 390, 940, 570))

    first = registry.assign_provisional("cam1", a, now=5.0)
    registry.observe("cam1", a, now=5.0)
    late = registry.assign_provisional("cam2", b, now=99.0)

    assert first != late


def test_two_different_plys_keep_separate_ids(registry):
    a = registry.confirm("cam1", FakeTrack(1, (700, 400, 900, 560), "88", ("48.3", "65.433")), 10.0)
    b = registry.confirm("cam1", FakeTrack(2, (100, 100, 300, 260), "93", ("28.4", "48.5")), 10.1)

    assert a.global_id != b.global_id
    assert {a.global_id, b.global_id} == {"ROLL-88", "ROLL-93"}


def test_a_roll_seen_again_later_returns_to_its_original_id(registry):
    values = ("30.8", "42.6")
    first = registry.confirm("cam1", FakeTrack(1, (700, 400, 900, 560), "7841", values), 10.0)
    again = registry.confirm("cam2", FakeTrack(42, (700, 400, 900, 560), "7841", values), 60.0)

    assert first.global_id == again.global_id


# -- stability ---------------------------------------------------------------

def test_id_does_not_churn_when_a_later_read_refines_the_values(registry):
    """Regression: a roll clipped at the frame edge read 54.7-9.2, then
    54.7-9.21 a few frames later, and used to end up as three separate rolls."""
    track = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.2"))
    first = registry.confirm("cam1", track, 10.0)

    track.confirmed_range = ("54.7", "9.21")
    second = registry.confirm("cam1", track, 10.5)

    assert first.global_id == second.global_id
    assert len([r for r in registry.rolls.values() if not r.provisional]) == 1
    assert second.read_end == "9.21", "the record should hold the latest reading"


def test_a_roll_is_re_keyed_once_its_ply_is_read(registry):
    """Only one re-key is allowed, and only on a gain of information:
    unidentified, then identified by the ply read off the roll."""
    track = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.2"))
    before = registry.confirm("cam1", track, 10.0)
    assert before.status == "partial"
    assert "-U" in before.global_id

    track.confirmed_ply = "88"
    after = registry.confirm("cam1", track, 10.6)

    assert after.global_id == "ROLL-88"
    assert after.status == "read"
    assert before.global_id not in registry.rolls, "no duplicate left behind"
    assert len([r for r in registry.rolls.values() if not r.provisional]) == 1


def test_cleanup_forgets_only_stale_rolls(registry):
    roll = registry.confirm("cam1", FakeTrack(1, (700, 400, 900, 560), "88",
                                              ("48.3", "65.433")), 10.0)
    forget = Config().identity.forget_after_s

    registry.cleanup(now=10.0 + forget - 1)
    assert roll.global_id in registry.rolls

    registry.cleanup(now=10.0 + forget + 1)
    assert roll.global_id not in registry.rolls


# -- the reading is reported as read -----------------------------------------

def test_values_are_reported_exactly_as_read(registry):
    """Ply 88's packing-list row says 48.3-65.433. A roll read as 11.1-22.2
    must be reported as 11.1-22.2, with no correction toward the list."""
    roll = registry.confirm("cam1", FakeTrack(1, (700, 400, 900, 560), "88",
                                              ("11.1", "22.2")), 10.0)

    assert (roll.read_ply, roll.read_start, roll.read_end) == ("88", "11.1", "22.2")
    assert roll.range_text == "11.1-22.2"
    assert roll.status == "read"
    assert not hasattr(roll, "expected_start"), "no expected columns exist any more"


def test_a_misread_ply_is_kept_not_corrected(registry):
    """The real failure from the footage: ply 99 / 4.7-17.5 comes back as "19"
    and "4.7-135". Both are reported as read. An earlier version silently
    relabelled this as ply 99 by matching the packing list."""
    roll = registry.confirm("cam1", FakeTrack(1, (700, 400, 900, 560), "19",
                                              ("4.7", "135")), 10.0)

    assert roll.read_ply == "19"
    assert roll.range_text == "4.7-135"
    assert roll.global_id == "ROLL-19"


def test_a_roll_with_no_legible_ply_is_marked_partial(registry):
    roll = registry.confirm("cam1", FakeTrack(1, (700, 400, 900, 560),
                                              values=("54.7", "9.2")), 10.0)

    assert roll.read_ply is None
    assert roll.status == "partial"
    assert roll.range_text == "54.7-9.2"


def test_roll_text_is_formatted_as_ply_then_range(registry):
    roll = registry.confirm("cam1", FakeTrack(1, (700, 400, 900, 560), "88",
                                              ("48.3", "65.433")), 10.0)
    assert roll.text == "ply 88   48.3-65.433"


# -- linking unidentified rolls across the two cameras ------------------------

def test_unidentified_roll_merges_across_cameras(registry):
    """A roll whose ply never read still needs one ID in both views.

    Regression: the cam1/cam2 footage produced ROLL-U5 and ROLL-U11 for a
    single roll, because an unidentified record had nothing to key on.
    """
    first = registry.confirm("cam2", FakeTrack(1, (700, 400, 900, 560),
                                               values=("54.7", "9.2")), 10.0)
    second = registry.confirm("cam1", FakeTrack(9, (720, 380, 930, 550),
                                                values=("54.7", "9.21")), 10.5)

    assert first.global_id == second.global_id
    assert second.camera_names == ["cam1", "cam2"]
    assert len([r for r in registry.rolls.values() if not r.provisional]) == 1


def test_unlike_unidentified_rolls_are_not_merged(registry):
    a = registry.confirm("cam2", FakeTrack(1, (700, 400, 900, 560),
                                           values=("54.7", "9.2")), 10.0)
    b = registry.confirm("cam1", FakeTrack(9, (100, 100, 300, 260),
                                           values=("81.2", "3.4")), 10.5)
    assert a.global_id != b.global_id


def test_twin_merge_spans_a_walk_between_the_cameras(registry):
    """In the test footage the roll reached the second camera 14 s after the
    first, so the window has to cover a walk, not just simultaneity."""
    a = registry.confirm("cam2", FakeTrack(1, (700, 400, 900, 560),
                                           values=("54.7", "9.2")), 10.0)
    b = registry.confirm("cam1", FakeTrack(9, (720, 380, 930, 550),
                                           values=("54.7", "9.2")), 24.0)
    assert a.global_id == b.global_id


def test_twin_merge_lapses_eventually(registry):
    window = Config().identity.twin_window_s
    a = registry.confirm("cam2", FakeTrack(1, (700, 400, 900, 560),
                                           values=("54.7", "9.2")), 10.0)
    b = registry.confirm("cam1", FakeTrack(9, (720, 380, 930, 550),
                                           values=("54.7", "9.2")), 10.0 + window + 5)
    assert a.global_id != b.global_id


def test_twin_merge_works_without_calibrated_overlap_zones(registry):
    """With no overlap ROI each camera mints its own placeholder first, and
    keeping that placeholder used to pre-empt the merge."""
    registry.cfg.overlap_roi = {}

    cam2 = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.21"))
    registry.assign_provisional("cam2", cam2, now=10.0)
    first = registry.confirm("cam2", cam2, 10.0)

    cam1 = FakeTrack(9, (720, 380, 930, 550), values=("54.7", "9.21"))
    registry.assign_provisional("cam1", cam1, now=10.4)
    second = registry.confirm("cam1", cam1, 10.4)

    assert first.global_id == second.global_id


def test_a_settled_id_survives_a_twin_that_looks_similar(registry):
    track = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.2"))
    first = registry.confirm("cam1", track, 10.0)

    registry.confirm("cam2", FakeTrack(2, (100, 100, 300, 260),
                                       values=("54.7", "9.21")), 10.3)

    track.confirmed_range = ("54.7", "9.22")
    assert registry.confirm("cam1", track, 10.6).global_id == first.global_id
