"""End-to-end examples for the principal native streaming operator families.

The measured scope starts with a declared Program and prepared immutable Arrow
batches. It includes compilation, a fresh native job, source delivery,
watermarks, output conversion, and completion. The scheduled Python shard
collects every case at overhead, small, and standard scales.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pyarrow as pa
import pytest

from benchmarks.support import (
    BenchmarkFixture,
    benchmark_group,
    record_comparable_identity,
    selected_scale,
)
from benchmarks.warm_stream import SCHEMA, _segment
from calc_flow import AsofStateLimits, JoinStateLimits, JoinTimeBounds
from calc_flow.symbolic import (
    Field,
    Program,
    cs,
    exact_time,
    rows,
    table,
    table_input,
    ts,
    window,
)

ENTITIES = 16
BATCH_ROWS = 640
WINDOW_TICKS = 10
MAX_ROWS = 20_000
JOIN_MAX_ROWS = 3_200
ASOF_MAX_ROWS = 1_920
SCENARIOS = (
    "expression",
    "sql",
    "rolling_state",
    "rolling_scan_64",
    "rolling_scan_256",
    "cross_section",
    "window_aggregate",
    "stream_join",
    "stream_asof_join",
)
NONNULL_OUTPUTS = {
    "rolling_state": ("mean", "average", "ewma"),
    "cross_section": ("mean", "residual", "top", "bottom"),
    "window_aggregate": ("value",),
    "stream_join": ("right__price",),
    "stream_asof_join": ("right__price",),
}
ROLLING_SCAN_NONNULL = ("argmax", "argmin", "rank", "unique_count", "decay")


def _source(name: str):
    return table_input(
        name,
        schema=(
            Field("event_time", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("price", "float64", nullable=False),
        ),
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )


def _scan_output(source, size: int):
    price = source["price"]
    frame = rows(size)
    return source.select(
        "sequence",
        argmax=ts.argmax(price, window=frame),
        argmin=ts.argmin(price, window=frame),
        rank=ts.rank(price, window=frame),
        quantile=ts.quantile(price, window=frame),
        unique_count=ts.unique_count(price, window=frame),
        decay=ts.decay(price, window=frame),
    )


def _joined_output(scenario: str, source, reference, input_rows: int):
    if scenario == "stream_join":
        joined = table.stream_join(
            source,
            reference,
            left_keys=("symbol",),
            right_keys=("symbol",),
            left_event_time="event_time",
            right_event_time="event_time",
            bounds=JoinTimeBounds(timedelta(), timedelta()),
            limits=JoinStateLimits(2 * input_rows, 64 << 20, 2 * input_rows),
        )
    else:
        joined = table.stream_asof_join(
            source,
            reference,
            tolerance=timedelta(),
            limits=AsofStateLimits(4 * input_rows, 256 << 20),
        )
    return joined.select("left__sequence", "right__price")


def _program(scenario: str, input_rows: int) -> Program:
    source = _source("quotes")
    price = source["price"]
    inputs = (source,)
    if scenario == "expression":
        output = source.select("sequence", value=price * 2.0 + 1.0)
    elif scenario == "sql":
        output = source.sql("SELECT sequence, price * 2 + 1 AS value FROM input")
    elif scenario == "rolling_state":
        output = source.select(
            "sequence",
            mean=ts.mean(price, window=rows(20)),
            average=ts.average(price),
            ewma=ts.ewma(price, span=20),
        )
    elif scenario.startswith("rolling_scan_"):
        output = _scan_output(source, int(scenario.rsplit("_", 1)[1]))
    elif scenario == "cross_section":
        group = exact_time(source["event_time"])
        output = source.select(
            "sequence",
            mean=cs.mean(price, group=group),
            residual=cs.residual(price, source["sequence"], group=group),
            top=cs.top_quantile(price, group=group, fraction=0.25),
            bottom=cs.bottom_quantile(price, group=group, fraction=0.25),
        )
    elif scenario == "window_aggregate":
        output = window.tumbling(
            source,
            event_time="event_time",
            size_micros=WINDOW_TICKS * 1_000_000,
            group_by=("symbol",),
            aggregates=(window.sum("price", output="value"),),
        )
    elif scenario in ("stream_join", "stream_asof_join"):
        reference = _source("reference")
        inputs = (source, reference)
        output = _joined_output(scenario, source, reference, input_rows)
    else:
        raise ValueError(f"unsupported stream operator benchmark: {scenario}")
    return Program(
        f"benchmark-{scenario}", inputs=inputs, outputs=(("result", output),)
    )


def _workload(scenario: str) -> tuple[pa.Table, tuple[pa.Table, ...]]:
    scale = selected_scale()
    size = ENTITIES * WINDOW_TICKS
    cap = {
        "stream_join": JOIN_MAX_ROWS,
        "stream_asof_join": ASOF_MAX_ROWS,
    }.get(scenario, MAX_ROWS)
    input_rows = min(scale.table_rows, cap) // size * size
    source = _segment(0, input_rows, ENTITIES)
    assert source.schema.equals(SCHEMA)
    parts = tuple(
        pa.Table.from_batches([batch])
        for batch in source.to_batches(max_chunksize=BATCH_ROWS)
    )
    return source, parts


async def _feed(parts: tuple[pa.Table, ...]):
    for part in parts:
        yield part


async def _run(program: Program, parts: tuple[pa.Table, ...], joined: bool) -> pa.Table:
    inputs = {"quotes": _feed(parts)}
    if joined:
        inputs["reference"] = _feed(parts)
    outputs = []
    async with program.stream(inputs) as results:
        async for event in results:
            if event.name != "result":
                raise RuntimeError(f"unexpected stream output: {event.name}")
            outputs.append(event.table)
    return pa.concat_tables(outputs)


def _validate(scenario: str, output: pa.Table, source: pa.Table) -> None:
    expected_rows = (
        source.num_rows // WINDOW_TICKS
        if scenario == "window_aggregate"
        else source.num_rows
    )
    assert output.num_rows == expected_rows
    if scenario in ("expression", "sql"):
        ordered = output.sort_by("sequence")
        assert ordered["value"][0].as_py() == source["price"][0].as_py() * 2 + 1
        return
    if scenario.startswith("rolling_scan_"):
        names = ROLLING_SCAN_NONNULL
        assert 0 < output["quantile"].null_count < output.num_rows
    else:
        names = NONNULL_OUTPUTS[scenario]
    for name in names:
        assert output[name].null_count == 0


@pytest.mark.benchmark(
    group=benchmark_group("stream-operators"), min_rounds=3, max_time=1.0
)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_stream_operator_lifecycle(
    benchmark: BenchmarkFixture, scenario: str, _scale: str
) -> None:
    source, parts = _workload(scenario)
    program = _program(scenario, source.num_rows)
    joined = scenario in ("stream_join", "stream_asof_join")

    def run() -> pa.Table:
        return asyncio.run(_run(program, parts, joined))

    _validate(scenario, run(), source)
    benchmark.extra_info = {
        **benchmark.extra_info,
        "scenario": scenario,
        "scope": "program-stream-lifecycle-to-arrow",
        "scale": selected_scale().name,
        "backend": "calc-flow-stream",
        "input_rows": source.num_rows * (2 if joined else 1),
        "output_rows": (
            source.num_rows // WINDOW_TICKS
            if scenario == "window_aggregate"
            else source.num_rows
        ),
        "stream_batch_rows": BATCH_ROWS,
        "stream_entities": ENTITIES,
    }
    record_comparable_identity(
        benchmark,
        workload_identity={
            "scenario": scenario,
            "scope": "program-stream-lifecycle-to-arrow",
            "backend": "calc-flow-stream",
            "scale": selected_scale().name,
            "rows_per_input": source.num_rows,
            "batch_rows": BATCH_ROWS,
            "entities": ENTITIES,
            "window_ticks": WINDOW_TICKS,
            "fixture": "warm_stream._segment",
        },
        dependency_packages=("numpy", "pyarrow", "pytest", "pytest-benchmark"),
    )
    _validate(scenario, benchmark(run), source)
