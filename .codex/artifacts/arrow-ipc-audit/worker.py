"""One real calc-flow execution; stdin/stdout carry Arrow IPC data only."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from fixtures import CASES, NATIVE, ROWS, declaration, digest, save


def receive():
    started = time.perf_counter_ns()
    with pa.ipc.open_stream(sys.stdin.buffer) as reader:
        batches = list(reader)
    return batches, [started, time.perf_counter_ns()]


def emit(table):
    started = time.perf_counter_ns()
    with pa.ipc.new_stream(sys.stdout.buffer, table.schema) as writer:
        writer.write_table(table)
    sys.stdout.buffer.flush()
    return [started, time.perf_counter_ns()]


async def inputs(case, directory, metadata):
    (directory / "ready").touch()
    batches, metadata["input_ipc_ns"] = await asyncio.to_thread(receive)
    started = time.perf_counter_ns()
    tables = [pa.Table.from_batches([batch]) for batch in batches]
    events = []
    import calc_flow as cf

    for table in tables:
        events.append(table)
        if case.window:
            events.append(
                cf.Watermark(pc.max(table["ts"]).as_py() - timedelta(microseconds=1))
            )
    metadata["input_binding_ns"] = [started, time.perf_counter_ns()]
    return tables, events


def compiled(program, runtime):
    from calc_flow.compute import _lowered_collect

    lowered = _lowered_collect(program, runtime)
    return (
        runtime.compile_batch_project(lowered.project_json),
        dict(lowered.input_names)["events"],
        dict(lowered.output_names)["result"],
    )


async def finite(case, arm, directory, execution, metadata):
    import calc_flow as cf

    program, runtime = execution
    start = time.perf_counter_ns()
    prepared = compiled(program, runtime) if arm == "sql_plan" else None
    metadata["compile_ns"] = [start, time.perf_counter_ns()]
    tables, _ = await inputs(case, directory, metadata)
    outputs, times, last_result = [], [], None
    start = time.perf_counter_ns()
    for table in tables:
        if prepared is None:
            output = (await program.collect_async({"events": table}, runtime=runtime))[
                "result"
            ]
        else:
            plan, names, name = prepared
            batch = cf.Batch.from_pyarrow(table)
            result = await plan.execute_async({key: batch for key in names})
            output = result.outputs[name].to_pyarrow()
            last_result = result
        outputs.append(output)
        times.append(time.perf_counter_ns())
    output = pa.concat_tables(outputs)
    metadata["core_ns"] = [start, time.perf_counter_ns()]
    metadata["batch_completed_ns"] = times
    metadata["output_ipc_ns"] = await asyncio.to_thread(emit, output)
    metadata["last_datafusion_metrics"] = (
        last_result.datafusion_metrics if last_result is not None else []
    )


async def continuous(case, directory, program, runtime, metadata):
    import calc_flow as cf

    gate, polled = asyncio.Event(), asyncio.Event()
    events = []

    async def source():
        polled.set()
        await gate.wait()
        for event in events:
            yield event
        await asyncio.Event().wait()

    started = time.perf_counter_ns()
    async with program.stream(
        {"events": source()},
        runtime=runtime,
        config=cf.StreamRuntimeConfig(checkpointing=False),
        watermarks=cf.SourceProvidedWatermarks(),
    ) as results:
        await asyncio.wait_for(polled.wait(), 10)
        metadata["compile_job_ready_ns"] = [started, time.perf_counter_ns()]
        _, prepared = await inputs(case, directory, metadata)
        events.extend(prepared)
        outputs, times, count = [], [], 0
        expected_rows = 512 if case.window else ROWS // 2
        start = time.perf_counter_ns()
        gate.set()
        async for event in results:
            outputs.append(event.table)
            count += event.table.num_rows
            times.append(time.perf_counter_ns())
            if count >= expected_rows:
                break
        output = pa.concat_tables(outputs)
        metadata["core_ns"] = [start, time.perf_counter_ns()]
        metadata["batch_completed_ns"] = times
        metadata["output_ipc_ns"] = await asyncio.to_thread(emit, output)
        metadata["status_before_close"] = results.job.status()


async def main(args):
    import calc_flow as cf
    import calc_flow._native as native

    directory = args.directory
    if digest(native.__file__) != args.native_sha:
        raise ValueError("worker native differs from the sealed release")
    case = next(case for case in CASES if case.name == args.case)
    start = time.perf_counter_ns()
    runtime, program = cf.Runtime(), declaration(case, args.arm)
    metadata = {
        "native_sha256": args.native_sha,
        "declaration_ns": [start, time.perf_counter_ns()],
    }
    save(directory / "program.json", program.to_project(runtime).model_dump())
    if args.arm == "streaming":
        await continuous(case, directory, program, runtime, metadata)
    else:
        await finite(case, args.arm, directory, (program, runtime), metadata)
    save(directory / "worker.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--native-sha", default=NATIVE)
    asyncio.run(main(parser.parse_args()))
