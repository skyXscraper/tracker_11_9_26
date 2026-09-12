"""Tests for the value parser -- the part where a wrong answer is silent.

A misdetected roll is obvious on screen. A mis-split length pair looks
perfectly plausible and quietly records the wrong roll, so the split logic is
pinned down here against the real master list.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollocr.config import Config            # noqa: E402
from rollocr.master import MasterList        # noqa: E402
from rollocr.parse import (                  # noqa: E402
    normalise, parse_lines, parse_ply, parse_range,
)

CFG = Config().values
MASTER_PATH = Path(__file__).resolve().parents[1] / "data" / "master_list.csv"


@pytest.fixture(scope="module")
def master():
    return MasterList.load(str(MASTER_PATH))


# -- the dotted-separator problem -------------------------------------------

@pytest.mark.parametrize("written,expected", [
    ("48.3.65.433", ("48.3", "65.433")),   # the roll in the reference photo
    ("28.4.48.5", ("28.4", "48.5")),
    ("30.8.42.6", ("30.8", "42.6")),
    ("45.65.433", ("45", "65.433")),       # integer start, decimal end
    ("4.3.17.8", ("4.3", "17.8")),
])
def test_dot_separated_split(written, expected):
    readings = parse_range(written, CFG)
    assert readings, f"no reading for {written}"
    assert (readings[0].start, readings[0].end) == expected


@pytest.mark.parametrize("written,expected", [
    ("54.7-9.2", ("54.7", "9.2")),         # as seen in the cam2 footage
    ("14.7-66.56", ("14.7", "66.56")),
    ("9.123-22.12", ("9.123", "22.12")),
])
def test_dash_separator_is_authoritative(written, expected):
    readings = parse_range(written, CFG)
    assert (readings[0].start, readings[0].end) == expected


def test_out_of_order_reading_is_kept_not_discarded():
    """A roll clipped by the frame edge loses digits.

    The result can read as end < start, which is not a reason to throw away the
    only evidence there is -- it is demoted, never dropped.
    """
    readings = parse_range("54.7-9.2", CFG)
    assert readings
    assert (readings[0].start, readings[0].end) == ("54.7", "9.2")


def test_the_parser_alone_resolves_almost_every_real_value(master):
    """The worst case: the operator wrote dots throughout, so the digit string
    alone must be split correctly.

    The packing list supplies realistic value pairs to test against, but is
    never consulted to make the decision -- the parser gets only the digits, as
    it does at runtime. 53 of 54 resolve on the constraints alone; the last
    ("5.9"/"16", equally readable as "5"/"9.16") is genuinely ambiguous.
    """
    wrong = []
    for row in master.rows:
        readings = parse_range(f"{row.start}.{row.end}", CFG)
        assert readings, f"no reading for {row.start}/{row.end}"
        if (readings[0].start, readings[0].end) != (row.start, row.end):
            wrong.append((row.ply_no, row.start, row.end,
                          readings[0].start, readings[0].end))
    assert len(wrong) <= 1, f"mis-split rows: {wrong}"


def test_the_true_reading_is_never_discarded(master):
    """Even where the split is ambiguous, the correct pair must survive as a
    candidate -- ranking may demote it, but nothing may drop it."""
    for row in master.rows:
        readings = parse_range(f"{row.start}.{row.end}", CFG)
        pairs = [(r.start, r.end) for r in readings]
        assert (row.start, row.end) in pairs, f"lost {row.start}/{row.end}"


# -- ply number --------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("88", "88"), ("7841", "7841"), ("102", "102"),
    ("8", None),            # too short to be a ply
    ("12345", None),        # too long
    ("", None),
])
def test_parse_ply(text, expected):
    assert parse_ply(text, CFG) == expected


def test_ply_is_never_scraped_off_the_length_line():
    """Regression: "54.7-9.2" must not yield ply 54.

    OCR often returns the lengths as the only line. Taking its leading digits
    as a ply number invents an identity that was never written on the roll.
    """
    ply, readings = parse_lines([("54.7-9.2", 100.0, 0.9)], CFG)
    assert ply is None
    assert (readings[0].start, readings[0].end) == ("54.7", "9.2")


def test_two_lines_assign_roles_by_vertical_position():
    ply, readings = parse_lines([
        ("48.3.65.433", 140.0, 0.9),   # lower line: the lengths
        ("88", 60.0, 0.95),            # upper line: the ply
    ], CFG)
    assert ply == "88"
    assert (readings[0].start, readings[0].end) == ("48.3", "65.433")


def test_glued_lines_still_split():
    ply, readings = parse_lines([("88", 40.0, 0.9), ("48.3.65.433", 90.0, 0.9)], CFG)
    assert ply == "88"
    assert (readings[0].start, readings[0].end) == ("48.3", "65.433")


# -- OCR character confusions ------------------------------------------------

def test_normalise_folds_letter_digit_confusions():
    assert normalise("4S.3") == "45.3"
    assert normalise("1O.2") == "10.2"
    assert normalise("48·3·65·433") == "48.3.65.433"


def test_noise_produces_no_reading():
    assert parse_range("", CFG) == []
    assert parse_range("abc", CFG) == []
    assert parse_range("7", CFG) == []          # a single group cannot be a pair


# -- the packing list is not part of the runtime path ------------------------

def test_master_module_exposes_no_identification_helpers(master):
    """Guards the rule directly: nothing may resolve a reading against the list.

    These helpers existed and were removed. If one comes back, a reading can
    once again be relabelled to whatever the list expected, and the station
    stops verifying anything.
    """
    for gone in ("best_match", "lookup_by_values", "choose_reading", "validate"):
        assert not hasattr(master, gone), f"{gone} must not be reintroduced"


def test_a_reading_matching_no_row_is_still_a_valid_reading(master):
    """54.7-9.2 appears in no row. The parser must still return it."""
    readings = parse_range("54.7-9.2", CFG)
    assert (readings[0].start, readings[0].end) == ("54.7", "9.2")
    assert master.lookup("54") is None


def test_span_matches_length_column_for_every_row(master):
    """end - start == length_m throughout the list, which is what makes the
    span check a usable tie-breaker."""
    for row in master.rows:
        assert abs(row.span - float(row.length_m)) <= CFG.length_tolerance_m
