"""Result output: an append-only event log plus a one-row-per-roll summary.

Every column here is something OCR read off a roll. There are no "expected"
columns: the pipeline does not consult the packing list, so it has nothing to
compare against and says only what it saw.

Everything is written UTF-8. The PP-OCR recognition model carries a CJK charset
and will occasionally emit a non-Latin character from noise, which crashes a
default-encoded writer on Windows.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

CSV_FIELDS = [
    "global_id", "ply_no", "range", "start", "end", "span_m",
    "status", "confidence", "cameras", "first_seen", "last_seen",
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
        """Record or refresh the summary row for a roll that has been read."""
        span = ""
        if roll.read_start and roll.read_end:
            try:
                span = round(float(roll.read_end) - float(roll.read_start), 3)
            except ValueError:
                span = ""
        self._rows[roll.global_id] = {
            "global_id": roll.global_id,
            "ply_no": roll.read_ply or "",
            "range": roll.range_text,
            "start": roll.read_start or "",
            "end": roll.read_end or "",
            "span_m": span,
            "status": roll.status,
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
