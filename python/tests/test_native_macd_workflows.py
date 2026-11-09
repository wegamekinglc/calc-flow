from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pyarrow as pa

import calc_flow as cf
import calc_flow.array as cfa


def _quotes() -> pa.Table:
    return pa.table(
        {
            "entity": ["a", "b"] * 4,
            "time": pa.array(
                [1, 1, 2, 2, 3, 3, 4, 4], type=pa.timestamp("us", tz="UTC")
            ),
            "seq": pa.array(range(8), type=pa.uint64()),
            "close": [1.0, 10.0, 2.0, 8.0, None, None, 3.0, 12.0],
        },
        schema=pa.schema(
            [
                pa.field("entity", pa.string(), nullable=False),
                pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("close", pa.float64()),
            ]
        ),
    )


def _source(data: pa.Table) -> cf.TableExpr:
    return cf.table_input(
        "quotes",
        schema=data.schema,
        entity_by=("entity",),
        event_time="time",
        sequence_by=("seq",),
    )


def _legacy(source: cf.TableExpr, *, offset: float = 0.0) -> cf.TableExpr:
    line = cf.ts.macd(source["close"] + offset, fast_span=2, slow_span=4)
    signal = cf.ts.ema(line, span=2, min_periods=2)
    return source.select("seq", line=line, signal=signal, histogram=line - signal)


def _owners(
    program: cf.Program, runtime: cf.Runtime | None = None
) -> list[dict[str, object]]:
    return [
        node
        for node in program.to_project(runtime).root["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
        and any(
            output["kind"] == "macd_component"
            for output in node["operator"]["spec"]["outputs"]
        )
    ]


async def _feed(
    chunks: tuple[pa.Table, ...], closed: asyncio.Event
) -> AsyncIterator[pa.Table]:
    try:
        for chunk in chunks:
            yield chunk
    finally:
        closed.set()


def test_installed_macd_stream_shares_subset_outputs_across_entities_and_batches() -> (
    None
):
    data = _quotes()
    prices = data["close"].to_pylist()
    prices[5] = float("nan")
    data = data.set_column(
        data.schema.get_field_index("close"), "close", pa.array(prices)
    )
    source = _source(data)
    parts = cf.ts.macd_components(
        source["close"], fast_span=2, slow_span=4, signal_span=2, signal_min_periods=2
    )
    program = cf.Program(
        "stream-subsets",
        engine="streaming",
        outputs={
            "signal": source.select("seq", signal=parts.signal),
            "histogram": source.select("seq", histogram=parts.histogram),
        },
    )
    owners = _owners(program)
    assert len(owners) == 1
    assert owners[0]["operator"]["spec"]["state_layout_version"] == 6
    assert {item["component"] for item in owners[0]["operator"]["spec"]["outputs"]} == {
        "signal",
        "histogram",
    }
    expected = _legacy(source).collect(data)

    async def run() -> dict[str, pa.Table]:
        closed = asyncio.Event()
        chunks = (data.slice(0, 3), data.slice(3, 0), data.slice(3, 2), data.slice(5))
        results = program.stream({"quotes": _feed(chunks, closed)})
        tables: dict[str, list[pa.Table]] = {"signal": [], "histogram": []}
        async with results:
            async for event in results:
                tables[event.name].append(event.table)
        assert closed.is_set()
        assert results.job.status()["task_count"] == 0
        return {name: pa.concat_tables(values) for name, values in tables.items()}

    actual = asyncio.run(asyncio.wait_for(run(), 10))
    for name, table in actual.items():
        assert table.column_names == ["seq", name]
        assert table["seq"].to_pylist() == expected["seq"].to_pylist()
        assert table[name].to_pylist() == expected[name].to_pylist()


def _attach(source: cf.TableExpr, column: str, name: str) -> cf.TableExpr:
    weights = cf.parameter(
        "weights", kind="array", backend="native", dtype="float64", shape=(1, 1)
    )
    values = cf.linalg.from_columns(source, columns=(column,), backend="native")
    return cf.table.attach_columns(
        source, cf.linalg.matmul(values, weights), names=(name,)
    )


def test_installed_macd_owner_survives_two_native_attachments_and_sql() -> None:
    data = _quotes().drop_null()
    source = _source(data)
    parts = cf.ts.macd_components(
        source["close"] + 1.0,
        fast_span=2,
        slow_span=4,
        signal_span=2,
        signal_min_periods=2,
    )
    first = source.with_columns(line=parts.line)
    once = _attach(first, "close", "copy1")
    twice = _attach(once, "copy1", "copy2")
    program = cf.Program(
        "native-bridges",
        engine="sql",
        outputs={
            "before": first.select("line"),
            "after": twice.select(copy=twice["copy2"], histogram=parts.histogram),
            "sql": twice.sql("SELECT * FROM input"),
        },
    )
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    assert len(_owners(program, runtime)) == 1
    actual = program.collect(
        {"quotes": data, "weights": cfa.from_arrow(pa.array([2.0]), shape=(1, 1))},
        runtime=runtime,
    )
    expected = _legacy(source, offset=1.0).collect(data)
    assert actual["before"].column_names == ["line"]
    assert actual["before"]["line"].to_pylist() == expected["line"].to_pylist()
    assert actual["after"].column_names == ["copy", "histogram"]
    assert actual["after"]["copy"].to_pylist() == [
        None if price is None else price * 4 for price in data["close"].to_pylist()
    ]
    assert actual["after"]["histogram"].to_pylist() == expected["histogram"].to_pylist()
    assert actual["sql"].column_names == [*data.column_names, "line", "copy1", "copy2"]


def test_installed_macd_shared_node_avoids_a_valid_user_output_collision() -> None:
    data = _quotes()
    source = _source(data)
    parts = cf.ts.macd_components(source["close"], fast_span=2, slow_span=4)
    outputs = {
        "line": source.select(line=parts.line),
        "histogram": source.select(histogram=parts.histogram),
    }
    baseline = cf.Program("shared-node", engine="sql", outputs=outputs)
    reserved = _owners(baseline)[0]["id"]
    program = cf.Program(
        "shared-node",
        engine="sql",
        outputs={**outputs, reserved: source.select("close")},
    )
    owners = _owners(program)
    assert len(owners) == 1
    assert owners[0]["id"] != reserved
    actual = program.collect({"quotes": data})
    assert actual[reserved].to_pydict() == {"close": data["close"].to_pylist()}
