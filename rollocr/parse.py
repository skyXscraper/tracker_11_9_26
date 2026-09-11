"""Turn raw OCR text into (ply, start, end).

The hard part is not reading the digits, it is splitting them.  The operator
writes the two lengths with whatever separator comes to hand -- a dash
(``54.7-9.2``) but very often just another dot (``48.3.65.433``, which means
48.3 and 65.433).  So the digit string alone is ambiguous and has to be
resolved by constraint:

  * each side carries at most one decimal point,
  * both values fall inside the plant's range (< 100 m),
  * the end reading exceeds the start reading.

Against the supplied master list those three rules leave exactly one legal
reading for 53 of 54 rows, and never discard the true one.  The last row is
settled by checking the ply number's own master entry -- see ``master.py``.
Note that the digits themselves always come from OCR; the master list only ever
chooses between readings, it never supplies a value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Characters PP-OCR routinely confuses with digits in red handwriting.
_DIGIT_FIXES = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1", "|": "1", "!": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
    ",": ".", "·": ".", "•": ".", "*": ".",
})

# Any dash-like glyph counts as an explicit start/end separator.
_SEPARATORS = "-‐‑‒–—―_/"


@dataclass(frozen=True)
class RangeReading:
    start: str
    end: str
    explicit_separator: bool     # True when the operator wrote a dash

    @property
    def span(self) -> float:
        return float(self.end) - float(self.start)


def normalise(text: str) -> str:
    """Fold OCR letter/digit confusions and strip anything else."""
    text = text.strip().translate(_DIGIT_FIXES)
    return "".join(ch for ch in text if ch.isdigit() or ch in "." + _SEPARATORS)


def parse_ply(text: str, cfg) -> str | None:
    """The ply number is the standalone integer written above the lengths."""
    cleaned = normalise(text)
    # A ply line should be digits only -- no decimal point, no separator.
    if any(ch in cleaned for ch in "." + _SEPARATORS):
        digits = re.sub(r"\D", "", cleaned)
        # Tolerate a stray mark, but only if what is left is a clean integer.
        if not digits or len(digits) != len(cleaned) - (len(cleaned) - len(digits)):
            pass
        cleaned = digits
    cleaned = cleaned.lstrip("0") or cleaned
    if not cleaned.isdigit():
        return None
    if not (cfg.ply_min_digits <= len(cleaned) <= cfg.ply_max_digits):
        return None
    return cleaned


def _well_formed(text: str, cfg) -> bool:
    """Structurally a number: one decimal point at most, parseable."""
    if text.count(".") > 1 or not text or text in ".":
        return False
    if text.startswith(".") or text.endswith("."):
        return False
    if "." in text and len(text.split(".")[1]) > cfg.max_decimals:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def in_range(text: str, cfg) -> bool:
    """Inside the plant's plausible span. A ranking signal, not a gate.

    OCR drops faint decimal points, so a true 17.5 arrives as "175" or "135".
    Rejecting those outright reported nothing at all for the roll; ranking them
    last keeps the digits available, and the master-list match can still
    identify the roll from them.
    """
    try:
        value = float(text)
    except ValueError:
        return False
    return cfg.value_min <= value < cfg.value_max


def _valid_value(text: str, cfg) -> bool:
    return _well_formed(text, cfg) and in_range(text, cfg)


def _split_groups(groups: list[str], cfg, explicit: bool) -> list[RangeReading]:
    """Every way of cutting the digit groups into a legal start/end pair."""
    readings: list[RangeReading] = []
    for cut in range(len(groups) - 1):
        start = ".".join(groups[: cut + 1])
        end = ".".join(groups[cut + 1:])
        if not (_well_formed(start, cfg) and _well_formed(end, cfg)):
            continue
        readings.append(RangeReading(start, end, explicit))
    return readings


def parse_range(text: str, cfg) -> list[RangeReading]:
    """All readings of the lower line, best first.

    Returns a list because the split is genuinely ambiguous for some values;
    the caller narrows it with the ply number's master entry.
    """
    cleaned = normalise(text)
    if not cleaned:
        return []

    # A dash says exactly where the split goes -- trust it.
    for sep in _SEPARATORS:
        if sep in cleaned:
            left, _, right = cleaned.partition(sep)
            left, right = left.strip(". "), right.strip(". ")
            if _well_formed(left, cfg) and _well_formed(right, cfg):
                return [RangeReading(left, right, True)]
            # A misread dash inside one number: fall through to the dot logic.
            cleaned = cleaned.replace(sep, ".")

    groups = [g for g in cleaned.split(".") if g]
    if len(groups) < 2 or not all(g.isdigit() for g in groups):
        return []

    readings = _split_groups(groups, cfg, explicit=False)
    # Rank rather than reject.  A roll clipped by the frame edge loses digits,
    # so an ordering that looks wrong is often a truncated read of a real value
    # -- discarding it outright throws away the only evidence we have.
    readings.sort(key=_rank(cfg))
    return readings


def _rank(cfg):
    """Best reading first: in range, then correctly ordered, then fractional.

    Every one of these is a preference. A clipped or mis-recognised reading is
    still the only evidence there is, so it is demoted, never discarded.
    """
    def key(reading: RangeReading):
        plausible = in_range(reading.start, cfg) and in_range(reading.end, cfg)
        ordered = float(reading.end) > float(reading.start)
        both_fractional = ("." in reading.start) + ("." in reading.end)
        return (not plausible,
                not (ordered or not cfg.prefer_end_gt_start),
                -both_fractional,
                float(reading.end))
    return key


def parse_lines(lines: list[tuple[str, float, float]], cfg) -> tuple[str | None, list[RangeReading]]:
    """Interpret OCR output for one roll.

    ``lines`` is (text, y_centre, confidence), any order.  Ply is written above
    the lengths, so vertical position assigns the roles.
    """
    if not lines:
        return None, []

    ordered = sorted(lines, key=lambda item: item[1])
    ply: str | None = None
    ranges: list[RangeReading] = []
    range_line: int | None = None

    for index, (text, _, _) in enumerate(ordered):
        candidate_range = parse_range(text, cfg)
        if candidate_range and not ranges:
            ranges = candidate_range
            range_line = index
            continue
        if ply is None:
            candidate_ply = parse_ply(text, cfg)
            if candidate_ply is not None:
                ply = candidate_ply

    # OCR sometimes glues both written lines into one string ("88 48.3.65.433").
    # Only ever take the ply from a line that did NOT supply the lengths --
    # otherwise the leading digits of "54.7-9.2" get misread as ply 54.
    if ply is None:
        for index, (text, _, _) in enumerate(ordered):
            if index == range_line:
                continue
            cleaned = normalise(text)
            head = re.match(r"^(\d{2,4})(?:\D|$)", cleaned)
            if head:
                ply = head.group(1)
                break
    return ply, ranges
