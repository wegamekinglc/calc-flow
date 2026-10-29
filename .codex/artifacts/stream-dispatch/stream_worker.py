"""Existing Arrow streaming scenario, with full output consumption and seals."""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import sys
import time
from pathlib import Path

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".codex/artifacts/sql-arrow-comparison"))
from common import CASES, ROWS, digest, save  # noqa: E402
from fixtures import declaration  # noqa: E402
from sql_worker import emit, inputs  # noqa: E402


async def execute(args, case, tables, metadata):
    import calc_flow as cf

    gate = asyncio.Event()

    async def events():
        yield tables[0]
        await gate.wait()
        for table in tables:
            yield table
        await asyncio.Event().wait()

    started = time.perf_counter_ns()
    program, runtime = declaration(case, "streaming"), cf.Runtime()
    metadata["declaration_ns"] = [started, time.perf_counter_ns()]
    started = time.perf_counter_ns()
    async with program.stream(
        {"events": events()},
        runtime=runtime,
        config=cf.StreamRuntimeConfig(checkpointing=False),
    ) as results:
        metadata["start_ns"] = [started, time.perf_counter_ns()]
        started = time.perf_counter_ns()
        warm = await results.__anext__()
        metadata["warmup_ns"] = [started, time.perf_counter_ns()]
        metadata["warmup_rows"] = warm.table.num_rows
        if args.diagnostic:
            results.job._inner._enable_callback_profiling()
        save(args.directory / "input.arrow.ready.json", metadata)
        while not (args.directory / "input.arrow.start").exists():
            await asyncio.sleep(0.001)
        profile = cProfile.Profile() if args.diagnostic else None
        if profile:
            profile.enable()
        received, rows, completed = [], 0, []
        started = time.perf_counter_ns()
        gate.set()
        async for event in results:
            received.append(event.table)
            rows += event.table.num_rows
            completed.append(time.perf_counter_ns())
            if rows >= ROWS // 2:
                break
        metadata["execute_ns"] = [started, time.perf_counter_ns()]
        started = time.perf_counter_ns()
        table = pa.concat_tables(received)
        metadata["concat_ns"] = [started, time.perf_counter_ns()]
        metadata["batch_completed_ns"] = completed
        metadata["output_batch_rows"] = [t.num_rows for t in received]
        metadata["output_ipc_ns"] = await asyncio.to_thread(emit, table, args.directory)
        if profile:
            profile.disable()
            profile.dump_stats(str(args.directory / "python.pstats"))
            metadata["callbacks"] = json.loads(
                results.job._inner._take_callback_profile()
            )
            metadata["status"] = results.job.status()
        started = time.perf_counter_ns()
    metadata["close_ns"] = [started, time.perf_counter_ns()]


async def main(args):
    import calc_flow._native as native

    if digest(native.__file__) != args.native_sha:
        raise ValueError("loaded native differs from sealed wheel")
    case = next(case for case in CASES[:2] if case.name == args.case)
    metadata = {"native_sha256": args.native_sha}
    tables = inputs(args.directory, metadata)
    await execute(args, case, tables, metadata)
    save(args.directory / "worker.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--native-sha", required=True)
    parser.add_argument("--diagnostic", action="store_true")
    asyncio.run(main(parser.parse_args()))
