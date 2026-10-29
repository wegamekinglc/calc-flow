"""Supplement the sealed collector audit with schema and window-end checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from fixtures import CASES, validate


def verify(root):
    rows = 0
    samples = json.loads((root / "samples.json").read_text())
    for sample in samples:
        started, finished = sample["round_trip_ns"]
        if sample["round_trip_ms"] != (finished - started) / 1e6:
            raise ValueError("round-trip summary differs from raw parent clock")
        case = next(case for case in CASES if case.name == sample["case"])
        directory = root / f"{case.name}-{sample['pair']}-{sample['arm']}"
        table = pa.ipc.open_stream(directory / "output.arrow").read_all()
        validate(case, table)
        names = (
            {"window_start", "window_end", "k", "total", "cnt"}
            if case.window
            else {"id", "y"}
        )
        if set(table.column_names) != names:
            raise ValueError("unexpected Arrow output columns")
        if case.window:
            start = pc.cast(table["window_start"], pa.int64())
            end = pc.cast(table["window_end"], pa.int64())
            if not pc.subtract(end, start).equals(
                pa.chunked_array([pa.array([4000000] * table.num_rows)])
            ):
                raise ValueError("window end differs from exact four-second boundary")
        rows += table.num_rows
    return {
        "schema_and_all_values": "passed",
        "samples": len(samples),
        "output_rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(verify(parser.parse_args().root)))
