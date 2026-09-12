#!/usr/bin/env python3
"""Compare a finished run against the packing list -- after the fact, opt-in.

This is deliberately a separate tool rather than part of the pipeline. The
station exists to verify what is handwritten on each roll, so the run itself
must not consult the list: if a reading could be resolved against the expected
answer, the comparison would be circular and a mislabelled roll would quietly
display the expected numbers and pass.

So the run reports what it read, and this compares that finished output with
the list side by side, changing nothing. Every row shows both columns and the
verdict between them.

    python tools/compare_to_master.py output/rolls.csv
    python tools/compare_to_master.py output/rolls.csv --master data/master_list.csv --csv report.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollocr.master import MasterList        # noqa: E402

VERDICTS = {
    "exact": "read matches the list",
    "values_differ": "ply found, lengths read differently",
    "ply_absent": "ply read is not in the list",
    "no_ply": "ply line was never read",
}


def compare(row: dict, master: MasterList) -> dict:
    """One result row against the list. Neither side is altered."""
    ply = (row.get("ply_no") or "").strip()
    read_range = (row.get("range") or "").strip()

    if not ply:
        verdict, listed = "no_ply", ""
    else:
        entry = master.lookup(ply)
        if entry is None:
            verdict, listed = "ply_absent", ""
        else:
            listed = entry.range_text
            verdict = "exact" if read_range == listed else "values_differ"

    return {
        "global_id": row.get("global_id", ""),
        "read_ply": ply,
        "read_range": read_range,
        "listed_range": listed,
        "verdict": verdict,
        "cameras": row.get("cameras", ""),
        "confidence": row.get("confidence", ""),
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="a rolls.csv produced by a run")
    parser.add_argument("--master", default="data/master_list.csv")
    parser.add_argument("--csv", help="write the comparison to this file")
    args = parser.parse_args()

    results = Path(args.results)
    if not results.exists():
        raise SystemExit(f"no such results file: {results}")

    master = MasterList.load(args.master)
    with results.open(newline="", encoding="utf-8") as handle:
        rows = [compare(row, master) for row in csv.DictReader(handle)]

    if not rows:
        print("[compare] the run recorded no rolls")
        return 0

    print(f"[compare] {len(rows)} rolls against {len(master.rows)} list entries\n")
    print(f"{'id':<14} {'read ply':<9} {'read range':<16} {'list range':<16} verdict")
    print("-" * 78)
    for row in rows:
        print(f"{row['global_id']:<14} {row['read_ply'] or '?':<9} "
              f"{row['read_range']:<16} {row['listed_range'] or '-':<16} "
              f"{row['verdict']}")

    print()
    for verdict, description in VERDICTS.items():
        count = sum(1 for row in rows if row["verdict"] == verdict)
        if count:
            print(f"  {count:3d}  {verdict:<14} {description}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[compare] written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
