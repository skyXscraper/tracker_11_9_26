"""Tests for cross-camera identity.

Two requirements are guarded here. The first is the one a plant operator would
notice immediately: a roll carried from camera 1 into camera 2 must keep one ID,
and must not acquire a second one because a later frame read its lengths
slightly differently.

The second is quieter but matters more: the values reported for a roll are the
ones OCR read, never the ones the master list would have preferred. The list
decides *which roll this is*; it never decides *what is written on it*.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollocr.config import Config          # noqa: E402
from rollocr.identity import RollRegistry  # noqa: E402
from rollocr.master import MasterList      # noqa: E402

MASTER_PATH = Path(__file__).resolve().parents[1] / "data" / "master_list.csv"
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
def master():
    return MasterList.load(str(MASTER_PATH))


@pytest.fixture
def registry():
    cfg = Config().identity
    # Both cameras catch the roll at the right-hand edge of frame, which is
    # where the two views actually overlap in the supplied footage.
    cfg.overlap_roi = {"cam1": (0.55, 0.0, 1.0, 1.0), "cam2": (0.55, 0.0, 1.0, 1.0)}
    return RollRegistry(cfg, FRAME_SIZES)


def confirm(registry, master, camera, track, now):
    return registry.confirm(camera, track, master, Config().values, now)


# -- one roll, one ID --------------------------------------------------------

def test_same_roll_keeps_one_id_across_both_cameras(registry, master):
    row = master.rows[0]
    values = (row.start, row.end)

    cam1_track = FakeTrack(1, (700, 400, 900, 560), ply=row.ply_no, values=values)
    first = confirm(registry, master, "cam1", cam1_track, now=10.0)

    # The same physical roll, now seen by the second camera as a new track.
    cam2_track = FakeTrack(7, (720, 380, 930, 550), ply=row.ply_no, values=values)
    second = confirm(registry, master, "cam2", cam2_track, now=10.4)

    assert first.global_id == second.global_id
    assert second.camera_names == ["cam1", "cam2"]


def test_overlap_link_shares_an_id_before_anything_is_read(registry):
    cam1_track = FakeTrack(1, (700, 400, 900, 560))
    cam2_track = FakeTrack(2, (740, 390, 940, 570))

    first = registry.assign_provisional("cam1", cam1_track, now=5.0)
    registry.observe("cam1", cam1_track, now=5.0)
    second = registry.assign_provisional("cam2", cam2_track, now=5.3)

    assert first == second, "tracks co-occurring in the overlap zone should link"


def test_tracks_outside_the_overlap_zone_are_not_linked(registry):
    cam1_track = FakeTrack(1, (10, 400, 120, 560))     # far left, not in overlap
    cam2_track = FakeTrack(2, (20, 390, 130, 570))

    first = registry.assign_provisional("cam1", cam1_track, now=5.0)
    registry.observe("cam1", cam1_track, now=5.0)
    second = registry.assign_provisional("cam2", cam2_track, now=5.3)

    assert first != second


def test_link_lapses_once_the_sync_window_has_passed(registry):
    cam1_track = FakeTrack(1, (700, 400, 900, 560))
    cam2_track = FakeTrack(2, (740, 390, 940, 570))

    first = registry.assign_provisional("cam1", cam1_track, now=5.0)
    registry.observe("cam1", cam1_track, now=5.0)
    late = registry.assign_provisional("cam2", cam2_track, now=99.0)

    assert first != late


def test_two_different_rolls_keep_separate_ids(registry, master):
    first_row, second_row = master.rows[0], master.rows[1]
    a = FakeTrack(1, (700, 400, 900, 560), ply=first_row.ply_no,
                  values=(first_row.start, first_row.end))
    b = FakeTrack(2, (100, 100, 300, 260), ply=second_row.ply_no,
                  values=(second_row.start, second_row.end))

    assert (confirm(registry, master, "cam1", a, now=10.0).global_id
            != confirm(registry, master, "cam1", b, now=10.1).global_id)


def test_a_roll_seen_again_later_returns_to_its_original_id(registry, master):
    row = master.rows[2]
    values = (row.start, row.end)

    first = confirm(registry, master, "cam1",
                    FakeTrack(1, (700, 400, 900, 560), ply=row.ply_no, values=values), now=10.0)
    again = confirm(registry, master, "cam2",
                    FakeTrack(42, (700, 400, 900, 560), ply=row.ply_no, values=values), now=60.0)

    assert first.global_id == again.global_id


# -- stability ---------------------------------------------------------------

def test_id_does_not_churn_when_a_later_read_refines_the_values(registry, master):
    """Regression: a roll clipped at the frame edge read 54.7-9.2, then
    54.7-9.21 a few frames later, and used to end up as three separate rolls."""
    track = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.2"))
    first = confirm(registry, master, "cam1", track, now=10.0)

    track.confirmed_range = ("54.7", "9.21")
    second = confirm(registry, master, "cam1", track, now=10.5)

    assert first.global_id == second.global_id
    assert len([r for r in registry.rolls.values() if not r.provisional]) == 1
    assert second.read_end == "9.21", "the record should hold the latest reading"


def test_unidentified_roll_is_upgraded_once_the_ply_is_read(registry, master):
    """Lengths matching no master row leave the roll unidentified; reading the
    ply later is the one piece of evidence allowed to re-key it."""
    track = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.2"))
    before = confirm(registry, master, "cam1", track, now=10.0)
    assert before.status == "unidentified"
    assert "-U" in before.global_id

    track.confirmed_ply = "88"
    after = confirm(registry, master, "cam1", track, now=10.6)

    assert after.global_id == "ROLL-88"
    assert after.ply_source == "ocr_exact"
    assert before.global_id not in registry.rolls, "no duplicate left behind"
    assert len([r for r in registry.rolls.values() if not r.provisional]) == 1


def test_cleanup_forgets_only_stale_rolls(registry, master):
    row = master.rows[4]
    roll = confirm(registry, master, "cam1",
                   FakeTrack(1, (700, 400, 900, 560), ply=row.ply_no,
                             values=(row.start, row.end)), now=10.0)

    registry.cleanup(now=10.0 + Config().identity.forget_after_s - 1)
    assert roll.global_id in registry.rolls

    registry.cleanup(now=10.0 + Config().identity.forget_after_s + 1)
    assert roll.global_id not in registry.rolls


# -- the list identifies; it never supplies values ---------------------------

def test_values_are_reported_as_read_even_when_they_contradict_the_master(registry, master):
    row = master.lookup("88")
    track = FakeTrack(1, (700, 400, 900, 560), ply="88", values=("11.1", "22.2"))

    roll = confirm(registry, master, "cam1", track, now=10.0)

    assert (roll.read_start, roll.read_end) == ("11.1", "22.2"), "never overwritten"
    assert roll.status == "value_mismatch"
    assert (roll.expected_start, roll.expected_end) == (row.start, row.end)


def test_ply_recovered_from_lengths_is_labelled_as_such(registry, master):
    """When the ply line is clipped away, the lengths still identify the roll -
    but the result must not claim the ply was read off the roll."""
    row = master.rows[5]
    track = FakeTrack(1, (700, 400, 900, 560), values=(row.start, row.end))

    roll = confirm(registry, master, "cam1", track, now=10.0)

    assert roll.matched_ply == row.ply_no
    assert roll.read_ply is None
    assert roll.ply_source == "values_exact"
    assert (roll.read_start, roll.read_end) == (row.start, row.end)


def test_systematic_handwriting_misread_still_identifies_the_roll(registry, master):
    """The real failure from the footage: ply 99 / 4.7-17.5 comes back from
    PP-OCR as "19" and "4.7-135". The roll must still be identified, with both
    the reading and the list values visible and clearly distinguished."""
    track = FakeTrack(1, (700, 400, 900, 560), ply="19", values=("4.7", "135"))

    roll = confirm(registry, master, "cam1", track, now=10.0)

    assert roll.matched_ply == "99"
    assert roll.ply_source == "fuzzy"
    assert roll.status == "value_mismatch"
    assert (roll.read_ply, roll.read_start, roll.read_end) == ("19", "4.7", "135")
    assert (roll.expected_start, roll.expected_end) == ("4.7", "17.5")


def test_a_roll_absent_from_the_list_is_not_forced_onto_a_row(registry, master):
    """Fuzzy matching must not turn every stray reading into some roll.

    The test roll in the cam1/cam2 footage reads 54.7-9.2, which is in no row;
    inventing a match for it would be worse than admitting ignorance.
    """
    track = FakeTrack(1, (700, 400, 900, 560), ply="547", values=("54.7", "9.2"))

    roll = confirm(registry, master, "cam1", track, now=10.0)

    assert roll.matched_ply is None
    assert roll.status == "unidentified"
    assert (roll.read_start, roll.read_end) == ("54.7", "9.2")


def test_fuzzy_match_prefers_the_lengths_over_a_badly_misread_ply(master):
    """The lengths carry more digits, so they are stronger evidence; a mangled
    ply must not veto them."""
    row, score = master.best_match("14", "4.7", "135")
    assert row is not None and row.ply_no == "99"
    assert score > 0.6


def test_fuzzy_match_needs_real_evidence(master):
    assert master.best_match(None, None, None)[0] is None
    assert master.best_match("1", "1", "2")[0] is None


def test_unidentified_roll_merges_across_cameras(registry, master):
    """A roll matching no master row still needs one ID in both views.

    Regression: the cam1/cam2 test footage produced ROLL-U5 and ROLL-U11 for a
    single roll, because an unidentified record had nothing to key on.
    """
    cam2 = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.2"))
    first = confirm(registry, master, "cam2", cam2, now=10.0)

    cam1 = FakeTrack(9, (720, 380, 930, 550), values=("54.7", "9.21"))
    second = confirm(registry, master, "cam1", cam1, now=10.5)

    assert first.global_id == second.global_id
    assert second.camera_names == ["cam1", "cam2"]
    assert len([r for r in registry.rolls.values() if not r.provisional]) == 1


def test_unlike_unidentified_rolls_are_not_merged(registry, master):
    a = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.2"))
    b = FakeTrack(9, (100, 100, 300, 260), values=("81.2", "3.4"))

    first = confirm(registry, master, "cam2", a, now=10.0)
    second = confirm(registry, master, "cam1", b, now=10.5)

    assert first.global_id != second.global_id


def test_twin_merge_spans_a_walk_between_the_cameras(registry, master):
    """In the test footage the roll reached the second camera 14 s after the
    first, so the merge window has to cover a walk, not just simultaneity."""
    a = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.2"))
    b = FakeTrack(9, (720, 380, 930, 550), values=("54.7", "9.2"))

    first = confirm(registry, master, "cam2", a, now=10.0)
    second = confirm(registry, master, "cam1", b, now=24.0)

    assert first.global_id == second.global_id


def test_twin_merge_lapses_eventually(registry, master):
    a = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.2"))
    b = FakeTrack(9, (720, 380, 930, 550), values=("54.7", "9.2"))

    first = confirm(registry, master, "cam2", a, now=10.0)
    second = confirm(registry, master, "cam1", b, now=10.0 + Config().identity.twin_window_s + 5)

    assert first.global_id != second.global_id


def test_twin_merge_works_without_calibrated_overlap_zones(registry, master):
    """Regression: with no overlap ROI configured each camera assigns its own
    placeholder ID first, and keeping that placeholder used to pre-empt the
    twin merge, leaving one roll as ROLL-U7 and ROLL-U13."""
    registry.cfg.overlap_roi = {}

    cam2 = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.21"))
    registry.assign_provisional("cam2", cam2, now=10.0)
    first = confirm(registry, master, "cam2", cam2, now=10.0)

    cam1 = FakeTrack(9, (720, 380, 930, 550), values=("54.7", "9.21"))
    registry.assign_provisional("cam1", cam1, now=10.4)
    second = confirm(registry, master, "cam1", cam1, now=10.4)

    assert first.global_id == second.global_id
    assert second.camera_names == ["cam1", "cam2"]


def test_a_settled_id_survives_a_twin_that_looks_similar(registry, master):
    """Stability still wins once a record has settled in this camera."""
    track = FakeTrack(1, (700, 400, 900, 560), values=("54.7", "9.2"))
    first = confirm(registry, master, "cam1", track, now=10.0)

    other = FakeTrack(2, (100, 100, 300, 260), values=("54.7", "9.21"))
    confirm(registry, master, "cam2", other, now=10.3)

    track.confirmed_range = ("54.7", "9.22")
    again = confirm(registry, master, "cam1", track, now=10.6)
    assert again.global_id == first.global_id
