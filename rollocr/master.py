"""The master list: identity and validation, never a source of values.

The CSV tells us which roll a ply number belongs to and what the plant expects
its lengths to be.  It is used to *choose between* readings the OCR produced
and to flag disagreements -- it never fills in a value that was not read off
the roll.  A roll whose ply is absent from the list is still reported, marked
``unknown_ply``; a roll whose lengths disagree with the list is reported with
the values as read, marked ``mismatch``.
"""

from __future__ import annotations

import csv
import difflib
import re
from dataclasses import dataclass
from pathlib import Path


def _digits(text: str | None) -> str:
    return re.sub(r"\D", "", text or "")


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


@dataclass(frozen=True)
class MasterRow:
    ply_no: str
    start: str
    end: str
    length_m: str
    no_of_ply: str
    item_number: str
    item_description: str
    packing_list: str

    @property
    def span(self) -> float:
        return float(self.end) - float(self.start)


class MasterList:
    def __init__(self, rows: list[MasterRow]) -> None:
        self.rows = rows
        self.by_ply: dict[str, MasterRow] = {}
        self.by_values: dict[tuple[str, str], MasterRow] = {}
        for row in rows:
            # Ply numbers are unique in the supplied list; keep the first if not.
            self.by_ply.setdefault(row.ply_no.strip().lstrip("0") or row.ply_no, row)
            # So are the (start, end) pairs, which gives a second way in when the
            # ply line is clipped by the frame edge and never gets read.
            self.by_values.setdefault((row.start, row.end), row)

    @classmethod
    def load(cls, path: str) -> "MasterList":
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"master list not found: {path}")
        rows = []
        with file.open(newline="", encoding="utf-8-sig") as handle:
            for record in csv.DictReader(handle):
                rows.append(MasterRow(
                    ply_no=(record.get("ply_no") or "").strip(),
                    start=(record.get("start") or "").strip(),
                    end=(record.get("end") or "").strip(),
                    length_m=(record.get("length_m") or "").strip(),
                    no_of_ply=(record.get("no_of_ply") or "").strip(),
                    item_number=(record.get("item_number") or "").strip(),
                    item_description=(record.get("item_description") or "").strip(),
                    packing_list=(record.get("packing_list") or "").strip(),
                ))
        return cls([r for r in rows if r.ply_no])

    def best_match(self, ply: str | None, start: str | None, end: str | None,
                   min_score: float = 0.62) -> tuple[MasterRow | None, float]:
        """Rank rows by how closely they resemble what was actually read.

        PP-OCR's recogniser is trained on printed text and makes consistent
        mistakes on this handwriting -- a crossed 7 reads as 3, a looped 9 as
        1, and faint decimal points vanish, so ply 99 / 4.7-17.5 comes back as
        "19" and "4.7-135" no matter how the crop is scaled or enhanced.

        The saving grace is that the master list is a closed set of 54 rolls.
        Scoring the digits that were read against that set turns an open-ended
        handwriting problem into a 54-way choice, which is far more robust.

        This decides *identity only*. The values reported for a roll stay the
        ones OCR actually produced -- the caller records this match beside the
        reading, never in place of it, so a bad match is visible rather than
        laundered into a plausible-looking number.
        """
        read_ply = _digits(ply)
        read_values = _digits(start) + _digits(end)
        if not read_ply and not read_values:
            return None, 0.0

        best_row, best_score = None, 0.0
        for row in self.rows:
            value_score = _similarity(read_values, _digits(row.start) + _digits(row.end))
            if read_ply:
                ply_score = _similarity(read_ply, _digits(row.ply_no))
                # The lengths carry more digits than the ply, so a coincidental
                # agreement there is far less likely: weight them higher.  Take
                # the better of the combined and the lengths-alone score, so a
                # badly misread ply cannot veto strong evidence from the values.
                score = max(0.35 * ply_score + 0.65 * value_score, 0.95 * value_score)
            else:
                score = value_score
            if score > best_score:
                best_row, best_score = row, score

        if best_score < min_score:
            return None, best_score
        return best_row, round(best_score, 3)

    def lookup_by_values(self, start: str | None, end: str | None) -> MasterRow | None:
        """Find the roll from its lengths alone.

        Every (start, end) pair in the supplied list is unique, so a roll whose
        ply number was never legible can still be identified from the two
        numbers that were.
        """
        if not start or not end:
            return None
        return self.by_values.get((start, end))

    def lookup(self, ply: str | None) -> MasterRow | None:
        if not ply:
            return None
        return self.by_ply.get(ply.strip().lstrip("0") or ply)

    def choose_reading(self, ply: str | None, readings: list, cfg):
        """Pick the reading consistent with this ply's master row, if any.

        This is disambiguation, not substitution: every candidate handed in was
        built from digits the OCR actually read.  Returns (reading, status).
        """
        if not readings:
            return None, "no_reading"

        row = self.lookup(ply)
        if row is None:
            return readings[0], ("unknown_ply" if ply else "no_ply")

        for reading in readings:
            if reading.start == row.start and reading.end == row.end:
                return reading, "match"

        # No exact textual match.  Fall back to the numeric span the row states:
        # end - start equals length_m for every row in the supplied list.
        if row.length_m:
            try:
                expected = float(row.length_m)
                for reading in readings:
                    if abs(reading.span - expected) <= cfg.length_tolerance_m:
                        return reading, "span_match"
            except ValueError:
                pass

        return readings[0], "mismatch"

    def validate(self, ply: str | None, reading, cfg) -> dict:
        """Describe how a completed read compares with the master list."""
        row = self.lookup(ply)
        result = {
            "status": "unknown_ply" if row is None else "match",
            "expected_start": row.start if row else None,
            "expected_end": row.end if row else None,
            "expected_length_m": row.length_m if row else None,
            "item_number": row.item_number if row else None,
            "item_description": row.item_description if row else None,
        }
        if row is None or reading is None:
            if reading is None:
                result["status"] = "no_reading"
            return result
        if reading.start != row.start or reading.end != row.end:
            result["status"] = "mismatch"
        return result
