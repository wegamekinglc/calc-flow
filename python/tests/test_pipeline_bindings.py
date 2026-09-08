"""Logical bindings across existing native state and static-provider boundaries."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest

import calc_flow as cf


class _NativeFeed:
    def __init__(self, table: pa.Table) -> None:
        self.messages = iter(
            [
                cf.Data(cf.Batch.from_pyarrow(table), cf.Cursor(b"1", {})),
                cf.Watermark(datetime(2030, 1, 1, tzinfo=UTC)),
            ]
        )
        self.opened = 0
        self.closed = 0

    def capabilities(self) -> cf.SourceCapabilities:
        return cf.SourceCapabilities(
            cf.ReplayPositioning.UNSUPPORTED,
            cf.SourceDeliveryCapability.LOSSY,
            max_batch_rows=100,
            max_batch_bytes=65536,
            native_watermarks=cf.NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor) -> None:
        self.opened += 1

    async def next(self):
        return next(self.messages, None)

    async def close(self) -> None:
        self.closed += 1


def _data(value: float) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
        ]
    )
    return pa.table({"symbol": ["A"], "ts": [1], "value": [value]}, schema=schema)


def _binding(feed: _NativeFeed) -> cf.SourceBinding:
    return cf.SourceBinding(feed, watermark_policy=cf.SourceProvidedWatermarks())


def _ordered_source(name: str, data: pa.Table) -> cf.TableExpr:
    return cf.table_input(
        name,
        schema=data.schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )


def _collect_named_stream(program, data):
    feeds = {name: _NativeFeed(table) for name, table in data.items()}

    async def run():
        collected = {}
        inputs = {name: _binding(feed) for name, feed in feeds.items()}
        async with program.stream(inputs) as results:
            async for item in results:
                collected.setdefault(item.name, []).append(item.table)
        assert results.job.status()["task_count"] == 0
        return {name: pa.concat_tables(tables) for name, tables in collected.items()}

    result = asyncio.run(run())
    assert [(feed.opened, feed.closed) for feed in feeds.values()] == [(1, 1), (1, 1)]
    return result


@pytest.mark.parametrize("streaming", [False, True], ids=["batch", "stream"])
@pytest.mark.parametrize("project_marker", [False, True], ids=["names", "projection"])
def test_stateful_fanout_keeps_logical_names_and_projection_schemas(
    streaming, project_marker
):
    left_data, right_data = _data(10.0), _data(3.0)
    left_name, right_name = (
        ("left_input", "right_input") if project_marker else ("left", "right")
    )
    left = _ordered_source(left_name, left_data)
    right = _ordered_source(right_name, right_data)
    outputs = {
        "left": left.select(mean=cf.ts.mean(left["value"], window=cf.rows(1))),
        "right": right.select(mean=cf.ts.mean(right["value"], window=cf.rows(1))),
    }
    if project_marker:
        outputs["marker"] = right.select("ts")
    program = cf.Program("stateful-fanout", outputs=outputs)
    inputs = {right_name: right_data, left_name: left_data}
    result = (
        _collect_named_stream(program, inputs) if streaming else program.collect(inputs)
    )
    assert result["left"].to_pydict() == {"mean": [10.0]}
    assert result["right"].to_pydict() == {"mean": [3.0]}
    if project_marker:
        assert result["marker"].equals(right_data.select(["ts"]))


@pytest.mark.parametrize("with_sql", [False, True])
def test_stream_window_preserves_source_watermarks_and_logical_outputs(with_sql):
    data = _data(7.0)
    source = cf.table_input("trades", schema=data.schema)
    output = cf.window.tumbling(
        source,
        event_time="ts",
        size_micros=10,
        group_by=["symbol"],
        aggregates=[cf.window.sum("value", output="total")],
    )
    if with_sql:
        output = output.sql("SELECT total FROM input")
    feed = _NativeFeed(data)

    async def run():
        async with output.stream(_binding(feed)) as results:
            return [value async for value in results]

    tables = asyncio.run(run())
    assert pa.concat_tables(tables)["total"].to_pylist() == [7.0]
    assert (feed.opened, feed.closed) == (1, 1)


def test_stream_join_keeps_distinct_same_schema_logical_sources():
    left_data, right_data = _data(10.0), _data(3.0)
    left = cf.table_input(
        "orders",
        schema=left_data.schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    right = cf.table_input(
        "fees",
        schema=right_data.schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    joined = cf.table.stream_join(
        left,
        right,
        left_keys=["symbol"],
        right_keys=["symbol"],
        left_event_time="ts",
        right_event_time="ts",
        bounds=cf.JoinTimeBounds(timedelta(seconds=1), timedelta(seconds=1)),
        limits=cf.JoinStateLimits(100, 65536, 100),
    ).sql("SELECT left__value - right__value AS net FROM input")
    orders, fees = _NativeFeed(left_data), _NativeFeed(right_data)

    async def run():
        async with joined.stream(
            {"fees": _binding(fees), "orders": _binding(orders)}
        ) as results:
            return [value async for value in results]

    tables = asyncio.run(run())
    assert pa.concat_tables(tables).to_pydict() == {"net": [7.0]}
    assert (orders.opened, fees.opened, orders.closed, fees.closed) == (1, 1, 1, 1)


def test_stream_sql_matrix_keeps_static_weights_across_batches():
    data = pa.table({"x": [1.0, 3.0]})
    source = cf.table_input("values", schema=data.schema).sql("SELECT x FROM input")
    weights = cf.parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(1, 1)
    )
    matrix = cf.linalg.from_columns(source, columns=["x"], backend="numpy")
    output = cf.table.attach_columns(
        source, cf.linalg.matmul(matrix, weights), names=["weighted"]
    )
    runtime = cf.Runtime()
    cf.register_numpy(runtime)

    async def feed():
        yield data.slice(0, 1)
        yield data.slice(1)

    async def run():
        inputs = {
            "values": feed(),
            "weights": cf.Batch.from_array(np.array([[2.0]]), backend="numpy"),
        }
        async with output.stream(inputs, runtime=runtime) as results:
            return [value async for value in results]

    tables = asyncio.run(run())
    assert pa.concat_tables(tables)["weighted"].to_pylist() == [2.0, 6.0]
