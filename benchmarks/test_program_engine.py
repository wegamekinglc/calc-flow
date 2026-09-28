"""Public Program.execute timing with one declaration and two selected engines."""

from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

from benchmarks.support import (
    BenchmarkFixture,
    benchmark_group,
    record_comparable_identity,
    selected_scale,
)
from calc_flow import Program, table_input

BATCH_ROWS = 640
MAX_ROWS = 20_000
SCENARIOS = {
    "projection": "SELECT x + 1 AS value FROM input",
    "aggregate": "SELECT SUM(x) AS value FROM input",
}


def _workload() -> tuple[pa.Table, tuple[pa.Table, ...]]:
    rows = min(selected_scale().table_rows, MAX_ROWS)
    table = pa.table({"x": pa.array([index % 97 for index in range(rows)])})
    parts = tuple(
        pa.Table.from_batches([batch])
        for batch in table.to_batches(max_chunksize=BATCH_ROWS)
    )
    return table, parts


def _program(scenario: str, engine: str) -> Program:
    program = Program("benchmark-program-engine", engine=engine)
    source = table_input("events", schema=pa.schema([("x", pa.int64())]))
    return program.output("result", source.sql(SCENARIOS[scenario]))


def _expected(
    scenario: str, engine: str, table: pa.Table, parts: tuple[pa.Table, ...]
) -> list[int]:
    if scenario == "projection":
        return [value + 1 for value in table["x"].to_pylist()]
    if engine == "sql":
        return [sum(table["x"].to_pylist())]
    cumulative = 0
    snapshots = []
    for part in parts:
        cumulative += sum(part["x"].to_pylist())
        snapshots.append(cumulative)
    return snapshots


async def _stream(program: Program, parts: tuple[pa.Table, ...]) -> pa.Table:
    async def source():
        for part in parts:
            yield part

    tables = []
    async with program.execute({"events": source()}) as results:
        async for output in results:
            if output.name != "result":
                raise RuntimeError(f"unexpected output {output.name!r}")
            tables.append(output.table)
    return pa.concat_tables(tables)


@pytest.mark.benchmark(
    group=benchmark_group("program-engine-execute"), min_rounds=3, max_time=1.0
)
@pytest.mark.parametrize("scenario", tuple(SCENARIOS))
@pytest.mark.parametrize("engine", ("sql", "streaming"))
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_program_engine_execute(
    benchmark: BenchmarkFixture, scenario: str, engine: str, _scale: str
) -> None:
    table, parts = _workload()
    program = _program(scenario, engine)
    expected = _expected(scenario, engine, table, parts)

    def run() -> pa.Table:
        if engine == "sql":
            return program.execute({"events": table})["result"]
        return asyncio.run(_stream(program, parts))

    assert run()["value"].to_pylist() == expected
    scope = "program-execute-to-arrow"
    benchmark.extra_info = {
        **benchmark.extra_info,
        "scenario": scenario,
        "scope": scope,
        "scale": selected_scale().name,
        "backend": f"calc-flow-program-{engine}",
        "input_rows": table.num_rows,
        "output_rows": len(expected),
        "stream_batch_rows": BATCH_ROWS if engine == "streaming" else None,
    }
    record_comparable_identity(
        benchmark,
        workload_identity={
            "scenario": scenario,
            "scope": scope,
            "engine": engine,
            "scale": selected_scale().name,
            "input_rows": table.num_rows,
            "output_rows": len(expected),
            "stream_batch_rows": BATCH_ROWS if engine == "streaming" else None,
            "fixture": "bounded-int64-modulo-97",
        },
        dependency_packages=("pyarrow", "pytest", "pytest-benchmark"),
    )
    assert benchmark(run)["value"].to_pylist() == expected
