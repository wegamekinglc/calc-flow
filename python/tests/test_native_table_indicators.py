from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pyarrow as pa
import pytest

import calc_flow as cf


def _prices() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("entity", pa.string(), nullable=False),
            pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("high", pa.float32()),
            pa.field("low", pa.float64()),
            pa.field("close", pa.float64()),
        ],
        metadata={b"fixture": b"native-table-indicators"},
    )
    return pa.table(
        {
            "entity": ["a"] * 5,
            "time": [1, 2, 3, 4, 5],
            "seq": [1, 2, 3, 4, 5],
            "high": [11.0, 12.0, 13.0, 12.0, 12.0],
            "low": [9.0, 10.0, 11.0, 10.0, 10.0],
            "close": [10.0, 11.0, 12.0, 11.0, 11.0],
        },
        schema=schema,
    )


def _source(data: pa.Table) -> cf.TableExpr:
    return cf.table_input(
        "quotes",
        schema=data.schema,
        entity_by=("entity",),
        event_time="time",
        sequence_by=("seq",),
    )


def _indicators(source: cf.TableExpr, *, period: int = 2) -> cf.TableExpr:
    return source.with_columns(
        rsi=cf.ts.rsi(source["close"], period=period),
        atr=cf.ts.atr(source["high"], source["low"], source["close"], period=period),
    )


def test_installed_table_indicators_execute_mixed_floats_without_mutating_input() -> (
    None
):
    data = _prices()
    before = data.to_pydict()
    schema = data.schema
    output = _indicators(_source(data)).collect(data)

    assert output["rsi"].to_pylist() == [None, None, 100.0, 50.0, 50.0]
    assert output["atr"].to_pylist() == [None, None, 2.0, 2.0, 2.0]
    assert output.schema.field("rsi") == pa.field("rsi", pa.float64())
    assert output.schema.field("atr") == pa.field("atr", pa.float64())
    assert output.schema.field("high").type == pa.float32()
    assert data.to_pydict() == before
    assert data.schema.equals(schema, check_metadata=True)

    source = _source(data)
    rsi32 = source.select(rsi32=cf.ts.rsi(source["high"], period=2)).collect(data)
    assert rsi32["rsi32"].to_pylist() == [None, None, 100.0, 50.0, 50.0]
    assert rsi32.schema.field("rsi32") == pa.field("rsi32", pa.float64())


def _price_rows(
    close: list[float | None],
    high: list[float | None],
    low: list[float | None],
) -> pa.Table:
    count = len(close)
    return pa.table(
        {
            "entity": ["a"] * count,
            "time": list(range(1, count + 1)),
            "seq": list(range(1, count + 1)),
            "high": high,
            "low": low,
            "close": close,
        },
        schema=_prices().schema,
    )


def test_installed_table_indicators_reset_after_null_and_nan_samples() -> None:
    data = _price_rows(
        [1.0, 2.0, None, 4.0, 5.0, float("nan"), 7.0, 8.0],
        [2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0],
        [0.0, 1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )
    before = [buffer.to_pybytes() for buffer in data["close"].chunk(0).buffers()]
    output = _indicators(_source(data), period=1).collect(data)

    assert output["rsi"].to_pylist() == [
        None,
        100.0,
        None,
        None,
        100.0,
        None,
        None,
        100.0,
    ]
    assert output["atr"].to_pylist() == [None, 2.0, None, None, 2.0, None, None, 2.0]
    assert [
        buffer.to_pybytes() for buffer in data["close"].chunk(0).buffers()
    ] == before


def test_installed_table_rsi_preserves_the_subnormal_ieee_operation_order() -> None:
    smallest = float.fromhex("0x0.0000000000001p-1022")
    data = _price_rows([0.0, smallest, 0.0], [1.0] * 3, [-1.0] * 3)
    output = _indicators(_source(data)).collect(data)
    assert output["rsi"].to_pylist() == [None, None, 100.0]
    assert output["atr"].to_pylist() == [None, None, 2.0]


async def _feed(
    chunks: tuple[pa.Table, ...], closed: asyncio.Event
) -> AsyncIterator[pa.Table]:
    try:
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0)
    finally:
        closed.set()


def test_installed_table_owned_stream_keeps_indicator_state_across_batches() -> None:
    data = _prices()
    output = _indicators(_source(data))

    async def run() -> pa.Table:
        closed = asyncio.Event()
        chunks = (data.slice(0, 2), data.slice(2, 1), data.slice(3))
        results = output.stream(_feed(chunks, closed))
        async with results:
            tables = [table async for table in results]
        assert closed.is_set()
        assert results.job.status()["task_count"] == 0
        return pa.concat_tables(tables)

    actual = asyncio.run(asyncio.wait_for(run(), 10))
    assert actual["seq"].to_pylist() == [1, 2, 3, 4, 5]
    assert actual["rsi"].to_pylist() == [None, None, 100.0, 50.0, 50.0]
    assert actual["atr"].to_pylist() == [None, None, 2.0, 2.0, 2.0]


def test_installed_named_program_shares_state_for_subset_indicator_outputs() -> None:
    data = _prices()
    source = _source(data)
    rsi = cf.ts.rsi(source["close"], period=2)
    atr = cf.ts.atr(source["high"], source["low"], source["close"], period=2)
    program = cf.Program(
        "shared-indicators",
        engine="streaming",
        outputs={
            "momentum": source.select(rsi=rsi),
            "range": source.select(atr=atr),
            "scaled": source.select(twice=rsi * 2.0),
        },
    )
    rolling = _rolling_specs(program)
    assert len(rolling) == 1
    assert sorted(item["kind"] for item in rolling[0]["outputs"]) == ["atr", "rsi"]

    async def run() -> dict[str, pa.Table]:
        closed = asyncio.Event()
        chunks = (data.slice(0, 2), data.slice(2))
        results = program.stream({"quotes": _feed(chunks, closed)})
        tables: dict[str, list[pa.Table]] = {name: [] for name, _ in program.outputs}
        async with results:
            async for event in results:
                tables[event.name].append(event.table)
        assert closed.is_set()
        assert results.job.status()["task_count"] == 0
        return {name: pa.concat_tables(values) for name, values in tables.items()}

    actual = asyncio.run(asyncio.wait_for(run(), 10))
    assert actual["momentum"].to_pydict() == {"rsi": [None, None, 100.0, 50.0, 50.0]}
    assert actual["range"].to_pydict() == {"atr": [None, None, 2.0, 2.0, 2.0]}
    assert actual["scaled"].to_pydict() == {"twice": [None, None, 200.0, 100.0, 100.0]}


def _invalid_range() -> tuple[pa.Table, cf.TableExpr]:
    data = _price_rows([None], [1.0], [2.0])
    source = _source(data)
    return data, source.select(
        atr=cf.ts.atr(source["high"], source["low"], source["close"], period=2)
    )


def test_installed_table_domain_error_keeps_the_direct_batch_projection() -> None:
    data, output = _invalid_range()
    with pytest.raises(
        cf.ExecutionError,
        match="node .*rolling failed: high: .*greater than or equal.*low",
    ):
        output.collect(data)
    assert data["high"].to_pylist() == [1.0]
    assert data["close"].to_pylist() == [None]


def test_installed_table_domain_error_keeps_the_owned_stream_projection() -> None:
    data, output = _invalid_range()

    async def run() -> None:
        closed = asyncio.Event()
        results = output.stream(_feed((data,), closed))
        with pytest.raises(cf.StreamingRuntimeError) as captured:
            async with results:
                async for _ in results:
                    pytest.fail("invalid indicator callback emitted output")
        error = captured.value
        assert error.category == "operator"
        assert error.component_kind == "operator"
        assert error.component_id.endswith("rolling")
        assert "high" in str(error)
        assert "input row 0" in str(error)
        assert "ordered position not assigned" in str(error)
        assert "greater than or equal" in str(error)
        assert closed.is_set()
        assert results.job.status()["task_count"] == 0

    asyncio.run(asyncio.wait_for(run(), 10))


def _rolling_specs(program: cf.Program) -> list[dict[str, object]]:
    document = program.to_project().root
    assert document["format_version"] == 3
    return [
        node["operator"]["spec"]
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    ]


def test_installed_layout_five_inventory_and_native_project_export() -> None:
    runtime = cf.Runtime()
    rolling = next(
        item for item in runtime.capabilities().operators if item.kind == "rolling"
    )
    assert rolling.state_layouts == (1, 2, 3, 4, 5)
    assert rolling.state_version == 1
    data = _prices()
    program = cf.Program(
        "exported", engine="sql", outputs={"signals": _indicators(_source(data))}
    )
    specs = _rolling_specs(program)
    assert len(specs) == 1
    assert specs[0]["state_layout_version"] == 5
    assert specs[0]["outputs"] == [
        {
            "kind": "rsi",
            "primitive_version": 1,
            "input": "close",
            "output": "rsi",
            "period": 2,
        },
        {
            "kind": "atr",
            "primitive_version": 1,
            "high": "high",
            "low": "low",
            "close": "close",
            "output": "atr",
            "period": 2,
        },
    ]
    exported = program.to_project(runtime).model_dump_json()
    recovered = cf.ProjectDocument.model_validate_json(exported)
    assert recovered == program.to_project(runtime)
    runtime.compile_batch_project(exported)
