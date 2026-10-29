"""One real open-source/window trace probe; no performance ranking samples."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from extension_transport import cf_sample, native_sample
from fixtures import CASES, encode, fixture, save, validate


def canonical(table):
    return pa.table(
        {
            "window_start": pc.cast(
                table["window_start"], pa.timestamp("us", tz="UTC")
            ),
            "window_end": pc.cast(table["window_end"], pa.timestamp("us", tz="UTC")),
            "k": table["k"],
            "total": table["total"],
            "cnt": pc.cast(table["cnt"], pa.int64()),
        }
    )


def probe(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    budget_file = root.parent / "measurement-budget.json"
    budget = json.loads(budget_file.read_text())
    started = time.monotonic()
    try:
        case = CASES[3]
        _, batches = fixture(case)
        payload = encode(batches)
        (root / "input.arrow").write_bytes(payload)
        daemon = json.loads(
            subprocess.check_output(
                ["multica", "daemon", "status", "--output", "json"], text=True
            )
        )["pid"]
        cf_dir, native_dir = root / "cf", root / "arroyo"
        cf_dir.mkdir()
        native_dir.mkdir()
        cf_table, cf_clock = cf_sample(case, cf_dir, payload, daemon)
        native_table, native_clock = native_sample(
            case, native_dir, payload, (args.binary.resolve(), daemon)
        )
        cf_table, native_table = canonical(cf_table), canonical(native_table)
        validate(case, cf_table)
        validate(case, native_table)
        for table in (cf_table, native_table):
            widths = pc.subtract(
                pc.cast(table["window_end"], pa.int64()),
                pc.cast(table["window_start"], pa.int64()),
            )
            if not pc.all(pc.equal(widths, 4000000)).as_py():
                raise ValueError("window-end boundary mismatch")
        cf_meta = json.loads((cf_dir / "worker.json").read_text())
        native_meta = json.loads((native_dir / "worker.json").read_text())
        native_wm = [
            int(line)
            for line in (native_dir / "watermarks.txt").read_text().splitlines()
        ]
        cf_wm = cf_meta["returned_watermarks_us"]
        result = {
            "kind": "single semantic probe, not ranked timing",
            "rows_each": cf_table.num_rows,
            "all_values_match": True,
            "row_order_equal": all(
                cf_table[name].equals(native_table[name])
                for name in cf_table.column_names
            ),
            "watermark_traces_equal": cf_wm == native_wm,
            "cf_watermarks_us": cf_wm,
            "arroyo_watermarks_us": native_wm,
            "cf_consumed_before_eof": cf_clock["observer_ns"][1]
            < cf_meta["source_eof_ns"],
            "arroyo_consumed_before_eof": not native_meta["source_eof_at_consume"],
            "arroyo_emitted_before_eof": not native_meta["source_eof_before_output"],
            "clock_boundaries": {"cf": cf_clock, "arroyo": native_clock},
        }
        save(root / "alignment.json", result)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key
                    not in {
                        "cf_watermarks_us",
                        "arroyo_watermarks_us",
                        "clock_boundaries",
                    }
                }
            )
        )
    finally:
        elapsed = time.monotonic() - started
        budget["spent_s"] += elapsed
        budget["entries"].append(
            {"kind": "window alignment probe", "elapsed_s": elapsed}
        )
        save(budget_file, budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    probe(parser.parse_args())
