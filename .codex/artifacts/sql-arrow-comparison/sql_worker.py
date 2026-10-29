"""One CF SQL invocation sequence, with file IPC outside normal API timing."""

from __future__ import annotations

import argparse
import asyncio
import functools
import importlib
import time
from pathlib import Path

import pyarrow as pa
from common import CASES, NATIVE, digest, save
from fixtures import declaration
from worker import compiled


class PhaseRecorder:
    """Own nested observation hooks for this isolated worker's collect calls."""

    def __init__(self):
        self.module = importlib.import_module("calc_flow.compute")
        self.names = ("_collect_batches", "_prepare_collect_batches", "_tables")
        self.originals = {name: getattr(self.module, name) for name in self.names}
        self.spans = {name: [] for name in self.names}

    def wrap(self, name):
        original = self.originals[name]

        @functools.wraps(original)
        def observed(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self.spans[name].append([start, time.perf_counter_ns()])

        return observed

    def __enter__(self):
        for name in self.names:
            setattr(self.module, name, self.wrap(name))
        return self

    def __exit__(self, *_):
        for name, original in self.originals.items():
            setattr(self.module, name, original)


def inputs(directory, metadata):
    started = time.perf_counter_ns()
    with pa.ipc.open_stream(directory / "input.arrow") as reader:
        batches = list(reader)
    metadata["input_ipc_ns"] = [started, time.perf_counter_ns()]
    started = time.perf_counter_ns()
    tables = [pa.Table.from_batches([batch]) for batch in batches]
    metadata["caller_table_views_ns"] = [started, time.perf_counter_ns()]
    metadata["input_batches"] = len(tables)
    metadata["input_rows"] = sum(table.num_rows for table in tables)
    return tables


async def execute_plan(prepared, table, phases):
    import calc_flow as cf

    plan, names, name = prepared
    started = time.perf_counter_ns()
    batch = cf.Batch.from_pyarrow(table)
    bound = {key: batch for key in names}
    phases["normal_input_binding_ns"].append([started, time.perf_counter_ns()])
    started = time.perf_counter_ns()
    result = await plan.execute_async(bound)
    phases["native_execute_ns"].append([started, time.perf_counter_ns()])
    started = time.perf_counter_ns()
    output = result.outputs[name].to_pyarrow()
    phases["result_packaging_ns"].append([started, time.perf_counter_ns()])
    return output, result


async def api_calls(execution, tables, metadata):
    program, runtime, prepared = execution
    outputs, finished, last = [], [], None
    phases = {
        name: []
        for name in (
            "normal_input_binding_ns",
            "native_execute_ns",
            "result_packaging_ns",
        )
    }
    start = time.perf_counter_ns()
    for table in tables:
        if prepared is None:
            output = (await program.collect_async({"events": table}, runtime=runtime))[
                "result"
            ]
        else:
            output, last = await execute_plan(prepared, table, phases)
        outputs.append(output)
        finished.append(time.perf_counter_ns())
    concat_start = time.perf_counter_ns()
    output = pa.concat_tables(outputs)
    metadata["final_concat_ns"] = [concat_start, time.perf_counter_ns()]
    metadata["api_ready_ns"] = [start, time.perf_counter_ns()]
    metadata["plan_nested_ns"] = phases
    metadata["batch_completed_ns"] = finished
    return output, last


def emit(table, directory):
    started = time.perf_counter_ns()
    temporary = directory / "output.arrow.partial"
    with (
        temporary.open("wb") as stream,
        pa.ipc.new_stream(stream, table.schema) as writer,
    ):
        writer.write_table(table)
    temporary.rename(directory / "output.arrow")
    return [started, time.perf_counter_ns()]


async def main(args):
    import calc_flow as cf
    import calc_flow._native as native

    if digest(native.__file__) != NATIVE:
        raise ValueError("incompatible CF native seal")
    directory = args.directory
    case = next(case for case in CASES[:2] if case.name == args.case)
    started = time.perf_counter_ns()
    runtime, program = cf.Runtime(), declaration(case, "sql")
    metadata = {
        "native_sha256": NATIVE,
        "arm": args.arm,
        "declaration_ns": [started, time.perf_counter_ns()],
    }
    started = time.perf_counter_ns()
    prepared = compiled(program, runtime) if args.arm == "sql_plan" else None
    metadata["precompile_ns"] = [started, time.perf_counter_ns()]
    tables = inputs(directory, metadata)
    with PhaseRecorder() as recorder:
        save(directory / "input.arrow.ready.json", metadata)
        while not (directory / "input.arrow.start").exists():
            await asyncio.sleep(0.001)
        table, last = await api_calls((program, runtime, prepared), tables, metadata)
        metadata["output_ipc_ns"] = await asyncio.to_thread(emit, table, directory)
    metadata["normal_collect_nested_ns"] = recorder.spans
    metadata["last_datafusion_metrics"] = (
        [] if last is None else last.datafusion_metrics
    )
    save(directory / "program.json", program.to_project(runtime).model_dump())
    save(directory / "worker.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--arm", choices=("sql", "sql_plan"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
