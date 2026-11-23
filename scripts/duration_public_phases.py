"""Time the existing public stream boundaries on the released single-SUM wheel."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch


class Phases(ExitStack):
    def __init__(self):
        super().__init__()
        self.calls = Counter()
        self.nanoseconds = Counter()

    def wrap_async(self, owner, attr, label):
        original = getattr(owner, attr)

        async def wrapped(*args, **kwargs):
            started = perf_counter_ns()
            try:
                return await original(*args, **kwargs)
            finally:
                self.calls[label] += 1
                self.nanoseconds[label] += perf_counter_ns() - started

        self.enter_context(patch.object(owner, attr, wrapped))

    def wrap(self, owner, attr, label):
        original = getattr(owner, attr)

        def wrapped(*args, **kwargs):
            started = perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self.calls[label] += 1
                self.nanoseconds[label] += perf_counter_ns() - started

        self.enter_context(patch.object(owner, attr, wrapped))


async def profiled(batches):
    import pyarrow as pa

    import calc_flow as cf
    from benchmarks.public_duration_sum import declaration

    inputs = importlib.import_module("calc_flow._stream_inputs")
    stream = importlib.import_module("calc_flow.stream")

    async def source():
        for batch in batches:
            yield batch

    with Phases() as phases:
        phases.wrap_async(inputs._IterableSource, "next", "source_callback_inclusive")
        phases.wrap(inputs._IterableSource, "_validate_structure", "source_structure")
        phases.wrap(inputs._IterableSource, "_validate_time", "source_timestamp")
        phases.wrap(inputs, "_table_batch", "source_import")
        phases.wrap(stream, "_compile_stream", "compile_stream_project")
        phases.wrap_async(stream._QueueSink, "write", "sink_callback_inclusive")
        phases.wrap(pa, "table", "pyarrow_table_factory")
        started = perf_counter_ns()
        expr = declaration()
        result = expr.stream(
            source(), config=cf.StreamRuntimeConfig(checkpointing=False)
        )
        declared = perf_counter_ns()
        outputs = []
        async with result:
            entered = perf_counter_ns()
            if result._native_queue is not None:
                raise ValueError("expected async iterable to select Python result sink")
            phases.wrap_async(result._ready, "wait", "output_ready_wait")
            first = None
            last = entered
            async for table in result:
                outputs.append(table)
                last = perf_counter_ns()
                if first is None:
                    first = last
            exhausted = perf_counter_ns()
        closed = perf_counter_ns()
    if first is None:
        raise ValueError("public stream had no output")
    return (
        closed - started,
        outputs,
        {
            "disjoint_ns": {
                "declare": declared - started,
                "enter": entered - declared,
                "first_output": first - entered,
                "remaining_outputs": last - first,
                "terminal_next": exhausted - last,
                "context_exit": closed - exhausted,
            },
            "inclusive_ns": dict(phases.nanoseconds),
            "calls": dict(phases.calls),
        },
    )


async def exercise(output: Path, fixture: Path, site: Path):
    import pyarrow as pa

    import calc_flow as cf
    from benchmarks.public_duration_sum import (
        declaration,
        expected,
        load,
        observe,
        plan,
        validate,
    )
    from calc_flow import _native
    from scripts.toolkit import write_json

    if {
        Path(cf.__file__).resolve().parent,
        Path(_native.__file__).resolve().parent,
    } != {site.resolve()}:
        raise ValueError("unexpected worker package source")
    batches = load(fixture)
    oracle = expected(pa.concat_tables(batches))
    physical = plan(declaration())
    rows = []
    for index, kind in enumerate(
        ("warmup", "plain", "profile", "profile", "plain", "plain", "profile")
    ):
        if kind == "profile":
            elapsed, returned, details = await profiled(batches)
        else:
            elapsed, returned = await observe(batches)
            details = {}
        counts = validate(returned, oracle)
        if counts != [100000, 124]:
            raise ValueError("unexpected complete output")
        rows.append(
            {
                "index": index,
                "kind": kind,
                "elapsed_ns": elapsed,
                "details": details,
                "emitted_tables": len(returned),
                "expected_counts": counts,
                "cpu": sorted(os.sched_getaffinity(0)),
            }
        )
        write_json(output, {"physical": physical, "observations": rows})


if __name__ == "__main__":
    asyncio.run(exercise(*map(Path, sys.argv[1:])))
