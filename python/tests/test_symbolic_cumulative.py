"""Cumulative statistics share the entity clock across native batches."""

from __future__ import annotations

import asyncio
import math

import pyarrow as pa
import pytest

import calc_flow as cf


def _input(data_type: pa.DataType | None = None) -> tuple[cf.TableExpr, pa.Schema]:
    data_type = pa.float64() if data_type is None else data_type
    schema = pa.schema(
        [
            pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("entity", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("value", data_type),
        ]
    )
    return cf.table_input(
        "events",
        schema=schema,
        entity_by=("entity",),
        event_time="time",
        sequence_by=("seq",),
    ), schema


def _data(schema: pa.Schema, values: list[object]) -> pa.Table:
    return pa.table(
        {
            "time": list(range(1, len(values) + 1)),
            "entity": ["a"] * len(values),
            "seq": list(range(1, len(values) + 1)),
            "value": values,
        },
        schema=schema,
    )


def test_cumulative_statistics_skip_missing_and_preserve_input() -> None:
    source, schema = _input()
    value = source["value"]
    output = source.with_columns(
        low=cf.ts.cumulative_min(value),
        high=cf.ts.cumulative_max(value),
        total=cf.ts.cumulative_sum(value, min_periods=2),
        sample=cf.ts.cumulative_variance(value),
        population=cf.ts.cumulative_variance(value, ddof=0),
    )
    data = _data(schema, [None, math.nan, 4.0, None, 2.0, 6.0])
    original = data.to_batches()[0].serialize().to_pybytes()
    result = output.collect({"events": data})
    assert result["low"].to_pylist() == [None, None, 4.0, 4.0, 2.0, 2.0]
    assert result["high"].to_pylist() == [None, None, 4.0, 4.0, 4.0, 6.0]
    assert result["total"].to_pylist() == [None, None, None, None, 6.0, 12.0]
    assert result["sample"].to_pylist() == [None, None, None, None, 2.0, 4.0]
    assert result["population"].to_pylist() == [None, None, 0.0, 0.0, 1.0, 8 / 3]
    assert data.to_batches()[0].serialize().to_pybytes() == original
    assert output.collect({"events": _data(schema, [])}).num_rows == 0


@pytest.mark.parametrize("data_type", [pa.int8(), pa.uint16(), pa.float32()])
def test_cumulative_output_types(data_type: pa.DataType) -> None:
    source, schema = _input(data_type)
    value = source["value"]
    result = source.select(
        low=cf.ts.cumulative_min(value),
        high=cf.ts.cumulative_max(value),
        total=cf.ts.cumulative_sum(value),
    ).collect({"events": _data(schema, [3, None, 1, 5])})
    assert result.schema.field("low").type == data_type
    assert result.schema.field("high").type == data_type
    total_type = (
        pa.uint64()
        if pa.types.is_unsigned_integer(data_type)
        else pa.int64()
        if pa.types.is_integer(data_type)
        else pa.float64()
    )
    assert result.schema.field("total").type == total_type
    assert result["total"].to_pylist() == [3, 3, 4, 9]


def test_cumulative_integer_sum_checks_overflow_at_the_readout_gate() -> None:
    source, schema = _input(pa.int64())
    maximum = 2**63 - 1
    data = _data(schema, [maximum, 1, -1])
    result = source.select(
        total=cf.ts.cumulative_sum(source["value"], min_periods=3)
    ).collect({"events": data})
    assert result["total"].to_pylist() == [None, None, maximum]
    with pytest.raises(cf.ExecutionError, match="overflow"):
        source.select(total=cf.ts.cumulative_sum(source["value"])).collect(
            {"events": data}
        )


def test_cumulative_variance_is_stable_for_large_offsets() -> None:
    source, schema = _input()
    result = source.select(variance=cf.ts.cumulative_variance(source["value"])).collect(
        {"events": _data(schema, [1e12 + i for i in range(5)])}
    )
    assert result["variance"].to_pylist() == [None, 0.5, 1.0, 5 / 3, 2.5]


def test_cumulative_variance_retains_small_spreads_at_large_offsets() -> None:
    source, schema = _input()
    result = source.select(variance=cf.ts.cumulative_variance(source["value"])).collect(
        {"events": _data(schema, [1e16, 1e16 + 2, 1e16 + 4])}
    )
    assert result["variance"].to_pylist() == [None, 2.0, 4.0]


def test_cumulative_stream_isolates_entities_across_batches() -> None:
    source, schema = _input()
    value = source["value"]
    output = source.with_columns(
        low=cf.ts.cumulative_min(value),
        high=cf.ts.cumulative_max(value),
        total=cf.ts.cumulative_sum(value),
        variance=cf.ts.cumulative_variance(value),
    )
    data = pa.table(
        {
            "time": [1, 1, 2, 2, 3, 3],
            "entity": ["a", "b"] * 3,
            "seq": [1, 1, 2, 2, 3, 3],
            "value": [1.0, 10.0, None, 20.0, 3.0, 30.0],
        },
        schema=schema,
    )

    async def batches():
        for part in (
            data.slice(0, 2),
            data.slice(2, 0),
            data.slice(2, 2),
            data.slice(4),
        ):
            yield part

    async def run() -> pa.Table:
        async with output.stream({"events": batches()}) as results:
            return pa.concat_tables([part async for part in results])

    result = asyncio.run(run()).sort_by(
        [("time", "ascending"), ("entity", "ascending")]
    )
    assert result["total"].to_pylist() == [1.0, 10.0, 1.0, 30.0, 4.0, 60.0]
    assert result["variance"].to_pylist() == [None, None, None, 50.0, 2.0, 100.0]
    assert result["low"].to_pylist() == [1.0, 10.0, 1.0, 10.0, 1.0, 10.0]
    assert result["high"].to_pylist() == [1.0, 10.0, 1.0, 20.0, 3.0, 30.0]


def test_cumulative_arguments_are_validated() -> None:
    source, _ = _input()
    for operation in (
        cf.ts.cumulative_min,
        cf.ts.cumulative_max,
        cf.ts.cumulative_sum,
        cf.ts.cumulative_variance,
    ):
        with pytest.raises(ValueError):
            operation(source["value"], min_periods=0)
        with pytest.raises(TypeError):
            operation(source["value"], min_periods=True)
    with pytest.raises(ValueError):
        cf.ts.cumulative_variance(source["value"], ddof=2)
    with pytest.raises(TypeError):
        cf.ts.cumulative_variance(source["value"], ddof=True)
