"""Ready native CF Batch-to-Batch boundary for the new adjacent comparison."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fixtures import CASES, NATIVE, ROWS, declaration, digest, save  # noqa: E402
from worker import emit, receive  # noqa: E402

from benchmarks.warm_stream import _InteractiveSource  # noqa: E402


class ReadySource(_InteractiveSource):
    def __init__(self):
        super().__init__(max_batch_rows=ROWS + 512)
        self.polled = asyncio.Event()
        self.watermarks = []

    async def next(self):
        self.polled.set()
        event = await super().next()
        from calc_flow import Watermark

        if isinstance(event, Watermark):
            self.watermarks.append(
                (event.at - datetime(1970, 1, 1, tzinfo=UTC))
                // timedelta(microseconds=1)
            )
        return event


class NativeSink:
    def __init__(self, expected):
        self.expected = expected
        self.rows = 0
        self.batches = []
        self.opened = asyncio.Event()
        self.complete = asyncio.Event()

    async def open(self):
        self.opened.set()

    async def write(self, batch):
        self.batches.append(batch)
        self.rows += batch.num_rows
        if self.rows >= self.expected:
            self.complete.set()

    async def close(self):
        return None


async def bind_inputs(case, directory, metadata):
    import calc_flow as cf

    (directory / "ready").touch()
    batches, metadata["input_ipc_ns"] = await asyncio.to_thread(receive)
    started = time.perf_counter_ns()
    events, watermarks, offset = [], [], 0
    for batch in batches:
        offset += batch.num_rows
        cursor = cf.Cursor(offset.to_bytes(8, "big"), {"offset": offset})
        events.append(cf.Data(cf.Batch.from_pyarrow(batch), cursor))
        if case.window:
            maximum = pc.max(batch.column(batch.schema.get_field_index("ts")))
            timestamp = maximum.as_py()
            timestamp -= timedelta(microseconds=1)
            events.append(cf.Watermark(timestamp))
            watermarks.append(maximum.cast(pa.int64()).as_py() - 1)
    metadata["input_binding_ns"] = [started, time.perf_counter_ns()]
    metadata["watermarks_us"] = watermarks
    return events


async def execute(case, directory, metadata):
    import calc_flow as cf

    started = time.perf_counter_ns()
    program = declaration(case, "streaming")
    plan = program.compile_stream(cf.Runtime())
    source = ReadySource()
    sink = NativeSink(512 if case.window else ROWS // 2)
    if len(plan.source_binding_ids) != 1 or len(plan.sink_binding_ids) != 1:
        raise ValueError("unexpected physical binding inventory")
    job = await cf.StreamingRunner(
        plan,
        {
            plan.source_binding_ids[0]: cf.SourceBinding(
                source, watermark_policy=cf.SourceProvidedWatermarks()
            )
        },
        {plan.sink_binding_ids[0]: [cf.SinkBinding.ordinary("dal417", sink)]},
        config=cf.StreamRuntimeConfig(
            checkpointing=False,
            edge_budget=cf.EdgeBudget(max_rows=ROWS + 512, max_bytes=64 << 20),
        ),
    ).start_async()
    try:
        await asyncio.wait_for(
            asyncio.gather(source.polled.wait(), sink.opened.wait()), 10
        )
        metadata["compile_job_ready_ns"] = [started, time.perf_counter_ns()]
        save(directory / "program.json", program.to_project().model_dump())
        events = await bind_inputs(case, directory, metadata)
        started = time.perf_counter_ns()
        for event in events:
            await source.push(event)
        await asyncio.wait_for(sink.complete.wait(), 20)
        metadata["core_ns"] = [started, time.perf_counter_ns()]
        if sink.rows != sink.expected:
            raise ValueError("native sink output count mismatch")
        started = time.perf_counter_ns()
        table = pa.concat_tables([batch.to_pyarrow() for batch in sink.batches])
        metadata["output_binding_ns"] = [started, time.perf_counter_ns()]
        metadata["output_ipc_ns"] = await asyncio.to_thread(emit, table)
        metadata["status_before_eof"] = job.status()
        metadata["returned_watermarks_us"] = source.watermarks
        while not (directory / "finish").exists():
            await asyncio.sleep(0.001)
        metadata["source_eof_ns"] = time.perf_counter_ns()
        await source.push(None)
        outcome = await asyncio.wait_for(job.wait_async(), 20)
        if outcome.state != "completed" or sink.rows != sink.expected:
            raise ValueError("native result changed at terminal completion")
    finally:
        await job.cancel_async()


async def main(args):
    import calc_flow._native as native

    if digest(native.__file__) != NATIVE:
        raise ValueError("incompatible CF native seal")
    case = next(case for case in CASES if case.name == args.case)
    metadata = {
        "native_sha256": NATIVE,
        "boundary": "ready Batch-to-Batch; bindings and IPC separate",
    }
    await execute(case, args.directory, metadata)
    save(args.directory / "worker.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
