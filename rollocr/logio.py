"""Result output: an append-only event log plus a one-row-per-roll summary.

Two files because they answer different questions. The JSONL is the audit
trail - every sighting and every confirmed read, in order, including reads that
disagreed with the master list. The CSV is the operational view: the current
state of each roll seen in this run, rewritten as results firm up.

Everything is written UTF-8. The PP-OCR recognition model carries a CJK
charset and will occasionally emit a non-Latin character from noise, which
crashes a default-encoded writer on Windows.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

# read_* is what the camera saw; matched_/expected_* is what the master list
# says. Keeping both in every row is the point -- a reader can always tell an
# OCR reading from a list lookup.
CSV_FIELDS = [
    "global_id", "read_ply", "read_start", "read_end", "read_span_m",
    "matched_ply", "ply_source", "match_score", "status",
    "expected_start", "expected_end", "expected_length_m",
    "item_number", "item_description",
    "confidence", "cameras", "first_seen", "last_seen",
]


class ResultLog:
    def __init__(self, cfg) -> None:
        self.dir = Path(cfg.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / cfg.jsonl_name
        self.csv_path = self.dir / cfg.csv_name
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self._rows: dict[str, dict] = {}

    def event(self, kind: str, **payload) -> None:
        record = {"t": round(time.time(), 3), "event": kind}
        record.update(payload)
        self._jsonl.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._jsonl.flush()

    def roll(self, roll) -> None:
        """Record or refresh the summary row for a confirmed roll."""
        span = ""
        if roll.read_start and roll.read_end:
            try:
                span = round(float(roll.read_end) - float(roll.read_start), 3)
            except ValueError:
                span = ""
        self._rows[roll.global_id] = {
            "global_id": roll.global_id,
            "read_ply": roll.read_ply or "",
            "read_start": roll.read_start or "",
            "read_end": roll.read_end or "",
            "read_span_m": span,
            "matched_ply": roll.matched_ply or "",
            "ply_source": roll.ply_source or "",
            "match_score": roll.match_score,
            "status": roll.status,
            "expected_start": roll.expected_start or "",
            "expected_end": roll.expected_end or "",
            "expected_length_m": roll.expected_length_m or "",
            "item_number": roll.item_number or "",
            "item_description": roll.item_description or "",
            "confidence": roll.confidence,
            "cameras": "|".join(roll.camera_names),
            "first_seen": round(roll.first_seen, 3),
            "last_seen": round(roll.last_seen, 3),
        }
        self._flush_csv()

    def _flush_csv(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in self._rows.values():
                writer.writerow(row)

    @property
    def rows(self) -> dict[str, dict]:
        return self._rows

    def close(self) -> None:
        self._flush_csv()
        self._jsonl.close()
