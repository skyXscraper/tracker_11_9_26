"""Tests for the results a run leaves behind.

The summary is what someone reads afterwards, so a roll appearing twice in it
is worse than a missing row -- it silently inflates the count and there is
nothing in the file to say which of the two is current.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollocr.config import Config          # noqa: E402
from rollocr.identity import RollRegistry  # noqa: E402
from rollocr.logio import ResultLog        # noqa: E402
from tests.test_identity import FakeTrack  # noqa: E402


@pytest.fixture
def log(tmp_path):
    cfg = Config().output
    cfg.dir = str(tmp_path)
    handle = ResultLog(cfg)
    yield handle
    handle.close()


def rows_of(log):
    with Path(log.csv_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def registry():
    cfg = Config().identity
    cfg.overlap_roi = {}
    return RollRegistry(cfg, {"cam1": (960, 1080)})


def test_a_re_keyed_roll_leaves_one_row_not_two(log, registry):
    """A roll starts as a placeholder and becomes ROLL-<ply> once its ply is
    read. Both rows used to survive, so one physical roll appeared twice."""
    track = FakeTrack(1, (700, 400, 900, 560), values=("4.7", "17.5"))

    placeholder = registry.confirm("cam1", track, 10.0)
    log.roll(placeholder)
    assert len(rows_of(log)) == 1

    track.confirmed_ply = "99"
    final = registry.confirm("cam1", track, 10.6)
    assert final.global_id != placeholder.global_id, "this test needs a re-key"

    log.drop(placeholder.global_id)
    log.roll(final)

    rows = rows_of(log)
    assert len(rows) == 1
    assert rows[0]["global_id"] == "ROLL-99"
    assert rows[0]["ply_no"] == "99"
    assert rows[0]["range"] == "4.7-17.5"


def test_dropping_an_unknown_id_is_harmless(log, registry):
    roll = registry.confirm("cam1", FakeTrack(1, (700, 400, 900, 560), "88",
                                              ("48.3", "65.433")), 10.0)
    log.roll(roll)
    log.drop("ROLL-does-not-exist")
    assert len(rows_of(log)) == 1


def test_rows_carry_the_reading_in_written_form(log, registry):
    roll = registry.confirm("cam1", FakeTrack(1, (700, 400, 900, 560), "88",
                                              ("48.3", "65.433")), 10.0)
    log.roll(roll)
    row = rows_of(log)[0]

    assert row["ply_no"] == "88"
    assert row["range"] == "48.3-65.433"
    assert row["start"] == "48.3" and row["end"] == "65.433"
    assert row["status"] == "read"
    # Nothing from a packing list may appear in the results.
    for column in row:
        assert not column.startswith("expected"), f"unexpected column {column}"
        assert "match" not in column, f"unexpected column {column}"


def test_refreshing_a_roll_updates_in_place(log, registry):
    track = FakeTrack(1, (700, 400, 900, 560), "88", ("48.3", "65.4"))
    log.roll(registry.confirm("cam1", track, 10.0))

    track.confirmed_range = ("48.3", "65.433")
    log.roll(registry.confirm("cam1", track, 10.5))

    rows = rows_of(log)
    assert len(rows) == 1
    assert rows[0]["range"] == "48.3-65.433", "the latest reading wins"


def test_events_are_appended_and_utf8_safe(log, registry):
    """PP-OCR carries a CJK charset and emits stray glyphs from noise; the log
    must take them rather than dying on a default-encoded writer."""
    log.event("read", camera="cam1", raw=["厂", "4.7-17.5"])
    log.event("read", camera="cam1", raw=["88"])

    lines = Path(log.jsonl_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "厂" in lines[0]
