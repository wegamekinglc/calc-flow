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
    mode = inspect.signature(cf.Program).parameters["mode"]
    assert mode.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="provide exactly one"):
        cf.Program("missing")
    with pytest.raises(ValueError, match="cannot both be supplied"):
        cf.Program("conflict", engine="sql", mode="batch")

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


@pytest.mark.parametrize("engine", ["sql", "streaming"])
def test_legacy_window_copy_preserves_engine_and_rejects_execution(engine) -> None:
    source = cf.table_input(
        "events",
        schema=pa.schema(
            [
                pa.field("ts", pa.timestamp("us", tz="UTC"), False),
                pa.field("entity", pa.string(), False),
                pa.field("seq", pa.uint64(), False),
                pa.field("x", pa.int64()),
            ]
        ),
        event_time="ts",
        entity_by=["entity"],
        sequence_by=["seq"],
    )
    program = cf.Program("windows", engine=engine).with_input(source)
    original = program.fingerprint
    value = cf.window.tumbling(source, event_time="ts", size_micros=60_000_000)
    identity = (value.digest, value._node.node_bytes)
    complete = program.output("windows", value)

    assert complete is not program
    assert program.outputs == ()
    assert program.fingerprint == original
    assert program.engine == complete.engine == engine
    assert complete.outputs == (("windows", value),)
    assert value._node.op.version == 1
    assert value._node.attrs.get("aggregates") is None
    compile_plan = (
        complete.compile_batch if engine == "sql" else complete.compile_stream
    )
    with pytest.raises(cf.CompileError, match="unknown_primitive_version"):
        compile_plan()
    assert (value.digest, value._node.node_bytes) == identity


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
