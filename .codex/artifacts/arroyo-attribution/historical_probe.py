"""Identify the original engine/graph using the retained sealed historical wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path

EXPECTED_NATIVE = "b9e63aef83465ba675ff96c449729f3d474abeac7abbec65af6f21ab12c8f267"


def main():
    import pyarrow as pa

    import calc_flow as cf
    import calc_flow._native as native

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checksum = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    if checksum != EXPECTED_NATIVE:
        raise RuntimeError("historical wheel identity does not match #494 evidence")
    args.output.mkdir(parents=True, exist_ok=False)
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("k", pa.int64()),
            ("ts", pa.timestamp("us", tz="UTC")),
            ("x", pa.int64()),
        ]
    )
    source = cf.table_input("events", schema=schema)
    accepted = source.filter(source["id"] >= 0)
    selected = accepted.filter(accepted["k"] < 8)
    expression = selected.select("id", y=selected["x"] * 2 + 1)
    window = cf.window.tumbling(
        accepted,
        event_time="ts",
        size_micros=4_000_000,
        group_by=("k",),
        aggregates=(
            cf.window.sum("x", output="total"),
            cf.window.count("id", output="cnt"),
        ),
    )
    for label, output in (("stateless", expression), ("window", window)):
        program = cf.Program("stream", engine="streaming", outputs={"result": output})
        (args.output / f"{label}-graph.json").write_text(
            json.dumps(program.to_project().model_dump(), indent=2) + "\n"
        )
    module = importlib.import_module("calc_flow.stream")
    (args.output / "table-stream-entry.py.txt").write_text(
        inspect.getsource(cf.TableExpr.stream)
    )
    (args.output / "internal-stream-entry.py.txt").write_text(
        inspect.getsource(module._stream_table)
    )
    (args.output / "identity.json").write_text(
        json.dumps(
            {
                "native_sha256": checksum,
                "package_version": importlib.metadata.version("calc-flow-python"),
                "source_ref_from_original_build": (
                    "dab98a559314bd46d866ed7a6489abcdd3810afe"
                ),
                "operation": (
                    "declaration/graph inspection only; no performance resampling"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print("Historical native seal, TableExpr.stream entry and graph captured")


if __name__ == "__main__":
    main()
