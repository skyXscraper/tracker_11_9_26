"""The packing list, as a reference for people -- not an input to the pipeline.

Nothing in the runtime path consults this. The station exists to verify what is
handwritten on each roll, so the pipeline reports what OCR read and nothing
else: if the list were allowed to supply values, or to pick which roll a
reading "must" have been, the comparison would be circular and would verify
nothing. A mislabelled roll would quietly display the expected numbers and
pass.

It is kept because comparing a finished run against the list is a useful thing
for an operator to do *afterwards*, with both columns visible and the
difference obvious. `tools/compare_to_master.py` does exactly that, on the CSV
a run produced, well after every reading was fixed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


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

    @property
    def range_text(self) -> str:
        return "{}-{}".format(self.start, self.end)


class MasterList:
    def __init__(self, rows: list[MasterRow]) -> None:
        self.rows = rows
        self.by_ply: dict[str, MasterRow] = {}
        for row in rows:
            self.by_ply.setdefault(row.ply_no.strip().lstrip("0") or row.ply_no, row)

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

    def lookup(self, ply: str | None) -> MasterRow | None:
        """Find a row by ply number. For reporting after the fact only."""
        if not ply:
            return None
        return self.by_ply.get(ply.strip().lstrip("0") or ply)
