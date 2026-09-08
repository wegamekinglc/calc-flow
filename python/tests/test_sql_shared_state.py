"""SQL boundaries preserve the native expression planner's state sharing."""

from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

import calc_flow as cf


def _quotes() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("price", pa.float64(), nullable=False),
        ]
    )
    return pa.table(
        {"symbol": ["a"] * 4, "ts": [1, 2, 3, 4], "price": [10.0, 12.0, 15.0, 14.0]},
        schema=schema,
    )


def _source() -> cf.TableExpr:
    return cf.table_input(
        "quotes",
        schema=_quotes().schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )


class _Feed:
    def __init__(self) -> None:
        self.opened = 0
        self.closed = 0
        self.index = 0

    def __aiter__(self):
        self.opened += 1
        return self

    async def __anext__(self):
        if self.index == 2:
            raise StopAsyncIteration
        result = _quotes().slice(self.index * 2, 2)
        self.index += 1
        return result

    async def aclose(self) -> None:
        self.closed += 1


def _run(program: cf.Program) -> dict[str, list[float | None]]:
    feed = _Feed()

    async def run():
        values = {name: [] for name, _ in program.outputs}
        async with program.stream({"quotes": feed}) as results:
            async for output in results:
                values[output.name].extend(output.table.column(0).to_pylist())
        assert results.job.status()["task_count"] == 0
        return values

    result = asyncio.run(run())
    assert (feed.opened, feed.closed) == (1, 1)
    return result


def test_sql_branches_share_a_column_expression_state_once():
    source = _source()
    delta = cf.ts.delta(source["price"])
    first = source.select(delta=delta).sql("SELECT delta FROM input")
    second = source.select(twice=delta * 2.0).sql("SELECT twice FROM input")
    program = cf.Program("shared-column", outputs={"delta": first, "twice": second})
    document = program.to_project(mode="stream").root
    rolling = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    ]
    assert len(rolling) == 1
    actual = _run(program)
    assert actual["delta"] == pytest.approx([None, 2.0, 3.0, -1.0])
    assert actual["twice"] == pytest.approx([None, 4.0, 6.0, -2.0])


def test_sql_branches_keep_distinct_filter_admission_for_shared_columns():
    source = _source()
    delta = cf.ts.delta(source["price"])
    first = source.filter(source["price"] >= 12.0).select(delta=delta)
    second = source.filter(source["price"] != 15.0).select(delta=delta)
    program = cf.Program(
        "filtered-columns",
        outputs={
            "high": first.sql("SELECT delta FROM input"),
            "other": second.sql("SELECT delta FROM input"),
        },
    )
    actual = _run(program)
    assert actual["high"] == pytest.approx([None, 3.0, -1.0])
    assert actual["other"] == pytest.approx([None, 2.0, 2.0])
