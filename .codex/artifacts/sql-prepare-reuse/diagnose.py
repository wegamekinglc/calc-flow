"""Public sync/async calls on the two fixed SQL workloads; no speedup verdict."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".codex/artifacts/sql-arrow-comparison"))
from common import CASES, check_rows, digest, fixture, save  # noqa: E402
from fixtures import declaration  # noqa: E402
from worker import compiled  # noqa: E402


def unpack(result, name):
    return result.outputs[name].to_pyarrow()


def sync_calls(plan, bound, name):
    outputs, metrics = [], []
    started = time.perf_counter_ns()
    for inputs in bound:
        result = plan.execute(inputs)
        outputs.append(unpack(result, name))
        metrics.extend(result.datafusion_metrics)
    return outputs, metrics, time.perf_counter_ns() - started


async def async_calls(plan, bound, name):
    outputs, metrics = [], []
    started = time.perf_counter_ns()
    for inputs in bound:
        result = await plan.execute_async(inputs)
        outputs.append(unpack(result, name))
        metrics.extend(result.datafusion_metrics)
    return outputs, metrics, time.perf_counter_ns() - started


def main(args):
    import pyarrow as pa

    import calc_flow as cf
    import calc_flow._native as native

    if digest(native.__file__) != args.native_sha:
        raise ValueError("diagnostic native seal differs")
    results = []
    for case in CASES[:2]:
        tables = [pa.Table.from_batches([batch]) for batch in fixture(case)[1]]
        for mode in ("sync", "async"):
            runtime, program = cf.Runtime(), declaration(case, "sql")
            plan, names, name = compiled(program, runtime)
            bound = [
                {key: cf.Batch.from_pyarrow(table) for key in names} for table in tables
            ]
            started = time.perf_counter_ns()
            cold = plan.execute(bound[0])
            cold_ns = time.perf_counter_ns() - started
            if mode == "sync":
                outputs, metrics, elapsed = sync_calls(plan, bound, name)
            else:
                outputs, metrics, elapsed = asyncio.run(async_calls(plan, bound, name))
            check_rows(case, pa.concat_tables(outputs))
            results.append(
                {
                    "case": case.name,
                    "mode": mode,
                    "api_and_packaging_ns": elapsed,
                    "cold_ns": cold_ns,
                    "cold_metrics": cold.datafusion_metrics,
                    "metrics": metrics,
                }
            )
    save(args.output, {"native_sha256": args.native_sha, "samples": results})
    print("four diagnostic sequences passed the full Arrow oracle", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
