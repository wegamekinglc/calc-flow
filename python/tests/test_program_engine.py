from __future__ import annotations

import asyncio
import dataclasses
import inspect

import pyarrow as pa
import pytest

import calc_flow as cf


def _program(engine: str) -> cf.Program:
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    return cf.Program(
        "totals",
        engine=engine,
        outputs={"totals": source.sql("SELECT SUM(x) AS total FROM input")},
    )


def test_program_requires_immutable_engine_at_construction() -> None:
    engine = inspect.signature(cf.Program).parameters["engine"]
    assert engine.kind is inspect.Parameter.KEYWORD_ONLY
    assert engine.default is inspect.Parameter.empty

    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    program = cf.Program("values", engine="streaming")
    complete = program.with_input(source).output("values", source.select("x"))

    assert program.engine == complete.engine == "streaming"
    with pytest.raises(dataclasses.FrozenInstanceError):
        complete._engine = "sql"  # type: ignore[misc]
    assert not hasattr(complete, "set_engine")


def test_output_added_after_engine_selection_discovers_input() -> None:
    program = cf.Program("values", engine="sql")
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    complete = program.output("values", source.select("x"))

    assert len(complete.inputs) == 1
    assert complete.execute({"events": pa.table({"x": [1]})})["values"].to_pydict() == {
        "x": [1]
    }


def test_engine_specific_compilation_and_project_export_are_fixed() -> None:
    sql_program = _program("sql")
    streaming_program = _program("streaming")

    with pytest.raises(RuntimeError, match="engine.*sql.*streaming"):
        sql_program.compile_stream()
    with pytest.raises(RuntimeError, match="engine.*streaming.*sql"):
        streaming_program.compile_batch()
    with pytest.raises(ValueError, match="requires mode 'batch'"):
        sql_program.to_project(mode="stream")
    with pytest.raises(ValueError, match="requires mode 'stream'"):
        streaming_program.to_project(mode="batch")


def test_sql_engine_rejects_stream_only_output_on_immutable_copy() -> None:
    source = cf.table_input(
        "events",
        schema=pa.schema([("ts", pa.timestamp("us", tz="UTC")), ("x", pa.int64())]),
        event_time="ts",
    )
    program = cf.Program("windows", engine="sql").with_input(source)
    value = cf.window.tumbling(source, event_time="ts", size_micros=60_000_000)

    with pytest.raises(ValueError, match="window_tumbling requires the streaming"):
        program.output("windows", value)
    assert program.outputs == ()


def test_streaming_program_rejects_multi_alias_sql_at_declaration() -> None:
    left = cf.table_input("left", schema=pa.schema([("x", pa.int64())]))
    right = cf.table_input("right", schema=pa.schema([("y", pa.int64())]))
    joined = cf.sql("SELECT x, y FROM l JOIN r ON true", l=left, r=right)

    with pytest.raises(ValueError, match="streaming.*one.*alias"):
        cf.Program("joined", engine="streaming", outputs={"joined": joined})


def test_engine_must_be_selected_once_and_keeps_declaration_fingerprint() -> None:
    sql_program = _program("sql")
    streaming_program = _program("streaming")
    assert sql_program.fingerprint == streaming_program.fingerprint
    assert sql_program.engine == "sql"
    assert streaming_program.engine == "streaming"
    with pytest.raises(dataclasses.FrozenInstanceError):
        sql_program._name = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="Program.engine"):
        _program("unknown")


def test_sql_execute_returns_named_arrow_tables() -> None:
    program = _program("sql")

    result = program.execute({"events": pa.table({"x": [1, 2, 3]})})

    assert result["totals"].to_pydict() == {"total": [6]}


def test_streaming_execute_returns_owned_cumulative_snapshots() -> None:
    program = _program("streaming")

    async def source():
        yield pa.table({"x": [1]})
        yield pa.table({"x": [2, 3]})

    async def collect() -> list[dict[str, list[int]]]:
        snapshots = []
        async with program.execute({"events": source()}) as results:
            async for output in results:
                assert output.name == "totals"
                snapshots.append(output.table.to_pydict())
        return snapshots

    assert asyncio.run(collect()) == [{"total": [1]}, {"total": [6]}]


def test_execute_rejects_options_for_the_other_engine() -> None:
    sql_program = _program("sql")
    with pytest.raises(ValueError, match="config.*streaming"):
        sql_program.execute({"events": pa.table({"x": [1]})}, config=object())

    stream_program = _program("streaming")
    with pytest.raises(ValueError, match="options.*sql"):
        stream_program.execute({"events": pa.table({"x": [1]})}, options=object())


def test_selected_engine_rejects_opposite_convenience_execution() -> None:
    sql_program = _program("sql")
    with pytest.raises(RuntimeError, match="engine.*sql.*streaming"):
        sql_program.stream({"events": pa.table({"x": [1]})})

    streaming_program = _program("streaming")
    with pytest.raises(RuntimeError, match="engine.*streaming.*sql"):
        streaming_program.collect({"events": pa.table({"x": [1]})})
    with pytest.raises(RuntimeError, match="engine.*streaming.*sql"):
        streaming_program.collect_async({"events": pa.table({"x": [1]})})


def test_same_row_local_declaration_executes_in_both_engines() -> None:
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))

    def make_program(engine: str) -> cf.Program:
        return (
            cf.Program("values", engine=engine)
            .with_input(source)
            .output("values", source.select("x"))
        )

    sql_program = make_program("sql")
    sql_rows = sql_program.execute({"events": pa.table({"x": [1, 2, 3]})})[
        "values"
    ].to_pydict()

    stream_program = make_program("streaming")

    async def source_batches():
        yield pa.table({"x": [1]})
        yield pa.table({"x": [2, 3]})

    async def collect_stream() -> dict[str, list[int]]:
        async with stream_program.execute({"events": source_batches()}) as results:
            tables = [output.table async for output in results]
        return pa.concat_tables(tables).to_pydict()

    assert sql_rows == asyncio.run(collect_stream()) == {"x": [1, 2, 3]}
