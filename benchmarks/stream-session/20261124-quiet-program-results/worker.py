from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter_ns

site = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(site))
import calc_flow as cf
import pyarrow as pa
from calc_flow import _native
from calc_flow.symbolic import Field, Program, table_input

native = Path(_native.__file__).resolve()
if native.parent.parent != site or not gc.isenabled():
    raise ValueError("unexpected installed package or GC mode")
source = table_input("events", schema=[Field("value", "int64")])
positive = source.filter(source["value"] > 0)
names = [f"signal{i:02d}" for i in range(32)]
program = Program(
    "quiet", mode="stream", inputs=[source], outputs=[(n, positive) for n in names]
)
schemas = program.collect_schema()
one = pa.table({"value": pa.array([7], type=pa.int64())})
inputs = {"events": cf.Batch.from_pyarrow(one)}
origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
marks = [{"events": origin + timedelta(microseconds=i)} for i in range(66)]


def check(tables, count):
    if list(tables) != names:
        raise ValueError("declared output order changed")
    for name, table in tables.items():
        if (
            not table.schema.equals(schemas[name], check_metadata=True)
            or table.num_rows != count
        ):
            raise ValueError("incorrect schema or length")
        if count and not table.equals(one, check_metadata=True):
            raise ValueError("incorrect output values")


def independent(first, second, rows):
    if first is second:
        raise ValueError("session reused the public output dictionary")
    check(first, rows)
    first.clear()
    check(second, rows)


with program.session(watermarks={"events": cf.SourceProvidedWatermarks()}) as session:
    independent(session.push(inputs), session.push(inputs), 1)
    independent(session.advance(marks[0]), session.advance(marks[1]), 0)
    quiet, data = [], []
    for watermark in marks[2:]:
        start = perf_counter_ns()
        tables = session.advance(watermark)
        quiet.append(perf_counter_ns() - start)
        check(tables, 0)
        del tables
        start = perf_counter_ns()
        tables = session.push(inputs)
        data.append(perf_counter_ns() - start)
        check(tables, 1)
        del tables
    check(session.finish(), 0)

with native.open("rb") as f:
    digest = hashlib.file_digest(f, "sha256").hexdigest()
print(
    json.dumps(
        {
            "quiet": quiet,
            "data": data,
            "outputs": len(names),
            "gc": gc.isenabled(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "tokio": os.environ.get("TOKIO_WORKER_THREADS"),
            "native": digest,
            "python": sys.version,
            "arrow": pa.__version__,
        }
    )
)
