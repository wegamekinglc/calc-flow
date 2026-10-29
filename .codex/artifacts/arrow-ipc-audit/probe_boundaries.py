"""Exercise the maintained engine harness with a deterministic audit clock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import benchmarks.engine_comparison as engine  # noqa: E402


def probe(root):
    events = ["fixture, declaration and compile start"]
    case = engine.EngineCase(
        {
            "family": "engines",
            "backend": "calc-flow-sql",
            "scenario": "projection",
            "rows": 1024,
        },
        root,
    )
    events.append("fixture, declaration and compile finished")
    calculate, validate = case.calculate, case.validate
    clock_values = iter((1000000000, 1000001000))

    def clock():
        value = next(clock_values)
        events.append("clock start" if value == 1000000000 else "clock stop")
        return value

    def measured():
        events.append("real compiled engine execute start")
        table = calculate()
        events.append(f"materialized Arrow result: {table.num_rows} rows")
        return table

    def checked(table):
        events.append("full correctness check start")
        result = validate(table)
        events.append("full correctness check finished")
        return result

    case.calculate, case.validate = measured, checked
    original_clock = engine.time
    try:
        engine.time = SimpleNamespace(perf_counter_ns=clock)
        result = case.sample()
    finally:
        engine.time = original_clock
        case.close()
    wire = json.dumps(result)
    events.append("JSON metadata response encoded after clock stop")
    if result["seconds"] != 1e-6 or not result["correctness"]["passed"]:
        raise ValueError("maintained timer/validation boundary differs from source")
    return {
        "kind": "deterministic clock probe, not a timing observation",
        "events": events,
        "response_fields": sorted(result),
        "metadata_bytes": len(wire),
        "data_rows_sent_in_json": 0,
        "correctness": result["correctness"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(probe(args.output.parent), indent=2) + "\n")
