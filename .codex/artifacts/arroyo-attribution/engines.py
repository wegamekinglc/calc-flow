"""Selected engine paths with the same declaration graph and Arrow values."""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

from common import check, declaration, legacy, normalize, prepared, save, schema
from profiling import Profile, bridge_diagnostic, callback_trace
from websockets.asyncio.client import connect


def diagnostic_collect_hooks(profile):
    if not profile.enabled:
        return
    compute = importlib.import_module("calc_flow.compute")

    for name, label in (
        ("_prepare_collect_batches", "sql.lower_compile"),
        ("_input_batch", "sql.input_bridge"),
        ("_tables", "sql.output_bridge"),
    ):
        original = getattr(compute, name)

        def measured(*args, _original=original, _label=label, **kwargs):
            return profile.call(_label, _original, *args, **kwargs)

        setattr(compute, name, measured)


def finite_plan(program, runtime, profile):
    from calc_flow.compute import _lowered_collect

    lowered = profile.call("sql.lower_once", _lowered_collect, program, runtime)
    plan = profile.call(
        "sql.compile_once", runtime.compile_batch_project, lowered.project_json
    )
    save(profile.directory / "compiled-project.json", json.loads(lowered.project_json))
    return (
        plan,
        dict(lowered.input_names)["events"],
        dict(lowered.output_names)["result"],
    )


async def finite_output(program, runtime, table, compiled, profile):
    import calc_flow as cf

    if compiled is None:
        outputs = await profile.await_call(
            "sql.collect_async",
            program.collect_async,
            {"events": table},
            runtime=runtime,
        )
        return outputs["result"]
    plan, inputs, output = compiled
    batch = profile.call("sql.input_bridge", cf.Batch.from_pyarrow, table)
    result = await profile.await_call(
        "sql.plan_execute_async", plan.execute_async, {name: batch for name in inputs}
    )
    if profile.enabled:
        profile.details.setdefault("datafusion_metrics", []).extend(
            result.datafusion_metrics
        )
        profile.details.setdefault("node_timings", []).append(result.node_timings)
    return profile.call("sql.output_bridge", result.outputs[output].to_pyarrow)


def write_output(table, window, profile):
    table = normalize(table, window)
    rows = profile.call("output.to_pylist", table.to_pylist)
    payload = profile.call(
        "output.json_encode",
        lambda: "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
    )
    profile.call("output.pipe_write", sys.stdout.write, payload)
    profile.call("output.pipe_flush", sys.stdout.flush)


async def live_tables(endpoint, window, profile):
    import pyarrow.json as paj

    import calc_flow as cf

    table_schema = schema()
    async with connect(endpoint, compression=None, max_size=8 * 1024 * 1024) as ws:
        async for frame in ws:
            table = profile.call(
                "input.json_decode",
                paj.read_json,
                io.BytesIO(frame),
                read_options=paj.ReadOptions(use_threads=False),
                parse_options=paj.ParseOptions(explicit_schema=table_schema),
            )
            yield table
            if window:
                yield cf.Watermark(table["ts"][0].as_py() - timedelta(microseconds=1))


async def run_live(case, arm, endpoint, directory, profile):
    import calc_flow as cf

    runtime = cf.Runtime()
    program = declaration("streaming" if arm == "streaming" else "sql", case.window)
    save(directory / "program.json", program.to_project(runtime).model_dump())
    source = live_tables(endpoint, case.window, profile)
    if arm != "streaming":
        diagnostic_collect_hooks(profile)
        async for table in source:
            output = await finite_output(program, runtime, table, None, profile)
            await profile.await_call(
                "output.thread_await",
                asyncio.to_thread,
                write_output,
                output,
                False,
                profile,
            )
        return
    context = program.stream(
        {"events": source},
        runtime=runtime,
        config=cf.StreamRuntimeConfig(checkpointing=False),
        watermarks=cf.SourceProvidedWatermarks(),
    )
    job = None
    try:
        async with context as results:
            job = results.job
            if profile.enabled:
                job._inner._enable_callback_profiling()
                (directory / "profile-ready").touch()
            async for event in results:
                await profile.await_call(
                    "output.thread_await",
                    asyncio.to_thread,
                    write_output,
                    event.table,
                    case.window,
                    profile,
                )
    finally:
        if job is not None:
            if profile.enabled:
                profile.details["stream_status"] = job.status()
            callback_trace(job, profile)


def arrow_tables(case):
    import pyarrow.json as paj

    frames, _, expected = prepared(case)
    tables = [
        paj.read_json(
            io.BytesIO(frame),
            read_options=paj.ReadOptions(use_threads=False, block_size=8 * 1024 * 1024),
            parse_options=paj.ParseOptions(explicit_schema=schema()),
        )
        for frame in frames[:-1]
    ]
    return tables, expected


async def arrow_stream(program, runtime, tables, expected, profile):
    import calc_flow as cf

    gate = asyncio.Event()
    received = []
    times = []

    async def source():
        await gate.wait()
        for table in tables:
            yield table
        await asyncio.Event().wait()

    async with program.stream(
        {"events": source()},
        runtime=runtime,
        config=cf.StreamRuntimeConfig(checkpointing=False),
    ) as results:
        job = results.job
        if profile.enabled:
            job._inner._enable_callback_profiling()
        start = time.perf_counter()
        gate.set()
        async for event in results:
            received.append(event.table)
            times.append(time.perf_counter() - start)
            if sum(table.num_rows for table in received) >= len(expected):
                break
        elapsed = time.perf_counter() - start
        if profile.enabled:
            profile.details["stream_status"] = job.status()
    callback_trace(job, profile)
    return received, elapsed, times


async def arrow_finite(program, runtime, tables, arm, profile):
    compiled = finite_plan(program, runtime, profile) if arm == "sql_plan" else None
    diagnostic_collect_hooks(profile)
    received = []
    times = []
    start = time.perf_counter()
    for table in tables:
        received.append(await finite_output(program, runtime, table, compiled, profile))
        times.append(time.perf_counter() - start)
    return received, time.perf_counter() - start, times


async def run_arrow(case, arm, directory, profile):
    import calc_flow as cf

    tables, expected = arrow_tables(case)
    runtime = cf.Runtime()
    program = declaration("streaming" if arm == "streaming" else "sql")
    save(directory / "program.json", program.to_project(runtime).model_dump())
    if arm == "streaming":
        received, elapsed, times = await arrow_stream(
            program, runtime, tables, expected, profile
        )
    else:
        received, elapsed, times = await arrow_finite(
            program, runtime, tables, arm, profile
        )
    actual = [row for table in received for row in table.to_pylist()]
    check(actual, expected)
    bridge_diagnostic(tables, profile)
    save(directory / "arrow-outputs.json", actual)
    save(
        directory / "worker-result.json",
        {
            "elapsed_s": elapsed,
            "rows_per_second": 32768 / elapsed,
            "output_rows": len(actual),
            "input_batches": len(tables),
            "input_record_batches": [
                [batch.num_rows for batch in table.to_batches()] for table in tables
            ],
            "completion_times_s": times,
            "correctness": "exact_all_rows",
            "source_open_until_result": arm == "streaming",
        },
    )


async def worker(case, args):
    import calc_flow._native as native

    actual = legacy.digest(Path(native.__file__))
    if actual != args.native_sha:
        raise RuntimeError("loaded native hash differs from the current sealed wheel")
    directory = args.directory
    save(directory / "loaded-native.json", {"sha256": actual, "path": native.__file__})
    profile = Profile(directory, args.profile)
    try:
        if case.transport == "live":
            await run_live(case, args.arm, args.endpoint, directory, profile)
        else:
            await run_arrow(case, args.arm, directory, profile)
    finally:
        profile.dump()
