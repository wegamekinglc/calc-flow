"""Reuse reviewed SQL API timing; read final-query metrics after output IPC."""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".codex/artifacts/sql-arrow-comparison"))

import sql_worker as reviewed  # noqa: E402
from common import CASES, digest, save  # noqa: E402
from fixtures import declaration  # noqa: E402
from worker import compiled  # noqa: E402


async def main(args):
    import calc_flow as cf
    import calc_flow._native as native

    if digest(native.__file__) != args.native_sha:
        raise ValueError("loaded native differs from sealed wheel")
    case = next(case for case in CASES[:2] if case.name == args.case)
    directory, metadata = args.directory, {"native_sha256": args.native_sha}
    started = time.perf_counter_ns()
    runtime, program = cf.Runtime(), declaration(case, "sql")
    prepared = compiled(program, runtime)
    metadata["declaration_and_precompile_ns"] = [started, time.perf_counter_ns()]
    tables = reviewed.inputs(directory, metadata)
    original = reviewed.execute_plan
    started = time.perf_counter_ns()
    phases = {
        key: []
        for key in (
            "normal_input_binding_ns",
            "native_execute_ns",
            "result_packaging_ns",
        )
    }
    warm_output, warm = await original(prepared, tables[0], phases)
    metadata["warmup_ns"] = [started, time.perf_counter_ns()]
    metadata["warmup_metrics"] = warm.datafusion_metrics
    del warm_output, warm

    save(directory / "input.arrow.ready.json", metadata)
    while not (directory / "input.arrow.start").exists():
        await asyncio.sleep(0.001)
    profile = cProfile.Profile() if args.diagnostic else None
    if profile:
        profile.enable()
    table, last = await reviewed.api_calls(
        (program, runtime, prepared), tables, metadata
    )
    if profile:
        profile.disable()
        profile.dump_stats(str(directory / "python.pstats"))
    metadata["output_ipc_ns"] = await asyncio.to_thread(reviewed.emit, table, directory)
    metadata["last_datafusion_metrics"] = last.datafusion_metrics
    save(directory / "worker.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--native-sha", required=True)
    parser.add_argument("--diagnostic", action="store_true")
    asyncio.run(main(parser.parse_args()))
