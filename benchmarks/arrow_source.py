"""Two fixed full-lifecycle controls for materialized Arrow streaming inputs.

Worker invocations use a sealed release wheel and are paired by
scripts.benchmark_suite.release_pairs. This is a focused diagnostic, not a
replacement for the scheduled suite. Run from a fresh process on Linux.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

from benchmarks.native_release_identity import file_hash, release_identity
from scripts.toolkit import fingerprint_json, write_json


def _projection_workload():
    import pyarrow as pa

    import calc_flow as cf

    data = pa.table({"value": list(range(512))})
    source = cf.table_input("events", schema=data.schema)
    program = cf.Program(
        "arrow-projection",
        engine="streaming",
        outputs={"result": source.select(doubled=source["value"] * 2)},
    )
    expected = pa.table({"doubled": [2 * value for value in range(512)]})
    return program, {"events": data}, expected


def _join_workload():
    from datetime import timedelta

    import pyarrow as pa

    import calc_flow as cf

    schema = pa.schema(
        [
            pa.field("key", pa.int64(), nullable=False),
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("value", pa.int64()),
        ]
    )
    inputs = {
        name: pa.table(
            {
                "key": list(range(64)),
                "ts": pa.array(list(range(64)), type=pa.timestamp("us", tz="UTC")),
                "value": [index + offset for index in range(64)],
            },
            schema=schema,
        )
        for name, offset in (("left", 0), ("right", 100))
    }
    left, right = (
        cf.table_input(
            name,
            schema=data.schema,
            entity_by=("key",),
            event_time="ts",
            sequence_by=("ts",),
        )
        for name, data in inputs.items()
    )
    joined = cf.table.stream_join(
        left,
        right,
        left_keys=["key"],
        right_keys=["key"],
        left_event_time="ts",
        right_event_time="ts",
        bounds=cf.JoinTimeBounds(timedelta(0), timedelta(0)),
        limits=cf.JoinStateLimits(1024, 16 << 20, 4096),
        left_prefix="l",
        right_prefix="r",
    )
    program = cf.Program(
        "arrow-join",
        engine="streaming",
        outputs={
            "result": joined.select(
                key=joined["l__key"],
                left=joined["l__value"],
                right=joined["r__value"],
            )
        },
    )
    expected = pa.table(
        {
            "key": list(range(64)),
            "left": list(range(64)),
            "right": list(range(100, 164)),
        }
    )
    expected = expected.cast(
        pa.schema(
            [
                field.with_nullable(False) if field.name == "key" else field
                for field in expected.schema
            ]
        )
    )
    return program, inputs, expected


def workload(case: str):
    """Return a graph, immutable multi-record inputs and an independent oracle."""
    import pyarrow as pa

    program, inputs, expected = (
        _projection_workload() if case == "projection" else _join_workload()
    )
    inputs = {
        name: pa.Table.from_batches(data.to_batches(max_chunksize=8))
        for name, data in inputs.items()
    }
    return program, inputs, expected


async def _feed(data):
    for record in data.to_batches():
        yield record


async def _collect_tables(program, inputs):
    import pyarrow as pa

    import calc_flow as cf

    config = cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(8, 1 << 20))
    async with program.stream(
        inputs,
        config=config,
        watermarks={name: cf.DisabledWatermarks() for name in inputs},
    ) as results:
        tables = [event.table async for event in results]
    return pa.concat_tables(tables), results.job.status()


def _validate_result(actual, expected, inputs, before, status):
    if not actual.equals(expected, check_metadata=True):
        raise AssertionError("full output oracle differs")
    if any(
        (data.to_pydict(), data.schema) != before[name] for name, data in inputs.items()
    ):
        raise AssertionError("caller input changed")
    if status["state"] != "completed" or status["task_count"] != 0:
        raise AssertionError("native lifecycle did not settle")


def _expected_source_callbacks(inputs, side):
    if side == "candidate":
        return 0
    return sum(len(data.to_batches()) + 1 for data in inputs.values())


async def collect(case: str, side: str) -> dict:
    from calc_flow._stream_inputs import _IterableSource

    program, inputs, expected = workload(case)
    before = {name: (data.to_pydict(), data.schema) for name, data in inputs.items()}
    calls = 0
    original = _IterableSource.next

    async def counted(self):
        nonlocal calls
        calls += 1
        return await original(self)

    _IterableSource.next = counted
    bound = (
        inputs
        if side == "candidate"
        else {name: _feed(data) for name, data in inputs.items()}
    )
    try:
        actual, status = await _collect_tables(program, bound)
        if case == "join":
            actual = actual.sort_by("key")
        _validate_result(actual, expected, inputs, before, status)
        expected_calls = _expected_source_callbacks(inputs, side)
        if calls != expected_calls:
            raise AssertionError(f"source callback count {calls} != {expected_calls}")
        return {
            "source_callbacks": calls,
            "rows": actual.num_rows,
            "task_count": status["task_count"],
        }
    finally:
        _IterableSource.next = original


def worker(case: str, side: str, build: Path) -> dict:
    import resource

    proof = release_identity(build)
    import pyarrow as pa

    before = pa.total_allocated_bytes()
    started = time.perf_counter()
    facts = asyncio.run(collect(case, side))
    gc.collect()
    elapsed = time.perf_counter() - started
    after = pa.total_allocated_bytes()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    machine = {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    dependencies = {
        "python": platform.python_version(),
        "pyarrow": version("pyarrow"),
        "calc-flow-python": version("calc-flow-python"),
    }
    workload_identity = {
        "case": case,
        "batch_rows": 8,
        "scope": "fixture-through-cleanup",
        "instrument_sha256": file_hash(Path(__file__)),
    }
    identities = {
        "machine": machine,
        "dependency": dependencies,
        "workload": workload_identity,
    }
    metadata = {
        key: value
        for name, identity in identities.items()
        for key, value in (
            (f"{name}_identity", identity),
            (f"{name}_fingerprint", fingerprint_json(identity)),
        )
    }
    if after != before:
        raise AssertionError(
            f"Arrow allocator did not return to baseline: {before} -> {after}"
        )
    return {
        "binary_sha256": proof["native_sha256"],
        "correctness": True,
        "lifecycle_seconds": elapsed,
        "metadata": metadata,
        "facts": facts,
        "peak_rss_kib": usage.ru_maxrss,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "arrow_bytes_before": before,
        "arrow_bytes_after": after,
        "release": proof,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("projection", "join"), required=True)
    parser.add_argument("--side", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, worker(args.case, args.side, args.build))


if __name__ == "__main__":
    main()
