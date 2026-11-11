from __future__ import annotations

import pyarrow as pa
import pytest

import calc_flow as cf


def test_mode_alias_copies_only_declarations() -> None:
    batch = cf.Program("values", mode="batch")
    stream = batch.for_mode("stream")
    assert batch.mode == "batch" and batch.engine == "sql"
    assert stream.mode == "stream" and stream.engine == "streaming"
    assert stream is not batch
    assert stream.fingerprint == batch.fingerprint
    assert batch.for_mode("batch") is not batch


@pytest.mark.parametrize("selection", [{}, {"mode": None}, {"engine": None}])
def test_mode_requires_one_valid_selection(selection: dict[str, object]) -> None:
    with pytest.raises(TypeError, match="Program.(mode|engine)"):
        cf.Program("invalid", **selection)


def test_mode_and_engine_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="cannot both be supplied"):
        cf.Program("invalid", engine="sql", mode="batch")


def test_schema_plans_both_frontends_without_rows() -> None:
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    scaled = source.with_columns(y=source["x"] + cf.lit(1))
    first = cf.sql("SELECT x, y FROM s WHERE y > 1", s=scaled)
    selected = first.select(z=first["y"] * cf.lit(2))
    result = selected.sql("SELECT z FROM input")
    program = cf.Program(
        "mixed", mode="batch", outputs={"scaled": scaled, "result": result}
    )
    assert program.collect_schema() == {
        "scaled": pa.schema([("x", pa.int64()), ("y", pa.int64())]),
        "result": pa.schema([("z", pa.int64())]),
    }
    assert result.collect_schema() == pa.schema([("z", pa.int64())])
    assert program.for_mode("stream").collect_schema() == program.collect_schema()


@pytest.mark.parametrize("mode,engine", [("batch", "sql"), ("stream", "streaming")])
def test_mode_preserves_legacy_identity_and_inference(mode: str, engine: str) -> None:
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    original = cf.Program("values", engine=engine, outputs={"out": source})
    alias = cf.Program("values", mode=mode, outputs={"out": source})
    assert alias.fingerprint == original.fingerprint
    assert alias.to_project().model_dump() == original.to_project().model_dump()
    added = cf.Program("values", mode=mode).for_mode(mode).output("out", source)
    assert len(added.inputs) == len(alias.inputs) == 1
    assert added.inputs[0].identical(alias.inputs[0])
    assert added.fingerprint == alias.fingerprint
    assert (
        cf.CompileMode
        is __import__("calc_flow.symbolic", fromlist=["CompileMode"]).CompileMode
    )


@pytest.mark.parametrize(
    "selection", [{"mode": "streaming"}, {"engine": "batch"}, {"mode": ""}]
)
def test_invalid_selection_is_located(selection: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="Program.*invalid_literal"):
        cf.Program("invalid", **selection)


def test_for_mode_rechecks_structure_without_mutating_original() -> None:
    source = cf.table_input("left", schema=pa.schema([("x", pa.int64())]))
    right = cf.table_input("right", schema=pa.schema([("y", pa.int64())]))
    program = cf.Program(
        "join",
        mode="batch",
        outputs={"out": cf.sql("SELECT x, y FROM l CROSS JOIN r", l=source, r=right)},
    )
    with pytest.raises(
        ValueError, match="outputs.out.*unsupported_mode.*one table alias"
    ):
        program.for_mode("stream")
    assert program.mode == "batch"


def _shape(source: cf.TableExpr, shape: str, values: list[int | None]):
    if shape == "dataframe":
        scaled = source.with_columns(y=source["x"] + cf.lit(1))
        result = scaled.filter(scaled["y"] > cf.lit(1)).select(
            z=scaled["y"] * cf.lit(2)
        )
        expected = [2 * (x + 1) for x in values if x is not None and x > 0]
    elif shape == "sql":
        result = source.sql("SELECT x + 1 AS z FROM input")
        expected = [None if x is None else x + 1 for x in values]
    elif shape == "df_sql_df_sql":
        scaled = source.with_columns(y=source["x"] + cf.lit(1))
        first = cf.sql("SELECT x, y FROM s WHERE y > 1", s=scaled)
        selected = first.select(z=first["y"] * cf.lit(2))
        result = selected.sql("SELECT z FROM input")
        expected = [2 * (x + 1) for x in values if x is not None and x > 0]
    else:
        first = source.sql("SELECT x FROM input")
        selected = first.select(z=first["x"] + cf.lit(1))
        result = selected.sql("SELECT z FROM input")
        expected = [None if x is None else x + 1 for x in values]
    return result, expected


@pytest.mark.parametrize("shape", ["dataframe", "sql", "df_sql_df_sql", "sql_df_sql"])
@pytest.mark.parametrize("values", [[1, None, 3, 1], [None, None], []])
def test_both_frontends_named_dag_async_capture_and_owned_stream(
    shape: str, values: list[int | None]
) -> None:
    import asyncio

    data = pa.table({"x": pa.array(values, type=pa.int64())})
    source = cf.table_input("events", schema=data.schema)
    result, expected = _shape(source, shape, values)
    batch = cf.Program(
        "mixed", mode="batch", outputs={"source": source, "result": result}
    )
    stream = batch.for_mode("stream")
    request = cf.PortabilityRequest(
        comparison="per_batch", history="current_batch", cutoff="batch_boundary"
    )
    assert (
        batch.analyze(options=cf.AnalysisOptions(portability=request)).query.portability
        == "supported"
    )
    assert batch.collect_schema()["result"] == pa.schema([("z", pa.int64())])

    async def run() -> None:
        mapping = {"events": data}
        pending = batch.collect_async(mapping, portability=request)
        mapping.clear()
        assert (await pending)["result"].to_pydict() == {"z": expected}
        seen = []
        opened = 0
        closed = False

        async def batches():
            nonlocal opened, closed
            opened += 1
            try:
                yield data
            finally:
                closed = True

        context = stream.stream({"events": batches()}, portability=request)
        assert opened == 0
        async with context as events:
            async for event in events:
                if event.name == "result":
                    seen.extend(event.table.column("z").to_pylist())
        assert opened == 1 and closed
        assert seen == expected

    asyncio.run(run())
    exported = batch.to_project().model_dump()
    assert all(source["data"] == [] for source in exported["data_sources"])


def test_schema_rejects_event_loop_and_array_outputs() -> None:
    import asyncio

    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))

    async def run() -> None:
        with pytest.raises(RuntimeError, match="collect_schema.*event loop"):
            source.collect_schema()

    asyncio.run(run())
    array = cf.array_input("a", dtype="float64", shape=(2,))
    with pytest.raises(cf.CompileError, match="outputs.out.schema.*unsupported_type"):
        cf.Program("array", mode="batch", outputs={"out": array}).collect_schema()


@pytest.mark.parametrize("name", ["result", "a.b", 'a"b', "input"])
def test_schema_keeps_logical_output_names_and_nullability(name: str) -> None:
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    filled = source.sql("SELECT coalesce(x, 0) AS x FROM input")
    program = cf.Program("schema", mode="batch", outputs={name: filled})
    assert program.collect_schema() == {
        name: pa.schema([pa.field("x", pa.int64(), nullable=False)])
    }


@pytest.mark.parametrize("kind", ["missing", "filter", "lineage"])
def test_schema_errors_keep_existing_logical_diagnostics(kind: str) -> None:
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    if kind == "missing":
        result = source.select(value=source["missing"])
    elif kind == "filter":
        result = source.filter(source["x"])
    else:
        other = cf.table_input("other", schema=source.collect_schema())
        result = source.select(value=source["x"] + other["x"])
    with pytest.raises(
        cf.CompileError,
        match=r"outputs.out.*(schema_mismatch|unresolved_type|unsupported_type)",
    ):
        cf.Program("invalid", mode="batch", outputs={"out": result}).collect_schema()


def test_exported_batch_plan_exposes_physical_names_for_public_reload() -> None:
    data = pa.table({"x": [1, 3]})
    source = cf.table_input("events", schema=data.schema)
    result = source.sql("SELECT x + 1 AS z FROM input")
    program = cf.Program("export", mode="batch", outputs={"result": result})
    document = cf.ProjectDocument.model_validate_json(
        program.to_project().model_dump_json()
    )
    plan = cf.Runtime().compile_batch_project(document.model_dump_json())
    assert type(plan.external_inputs) is tuple
    assert type(plan.external_outputs) is tuple
    assert len(plan.external_inputs) == len(plan.external_outputs) == 1
    reloaded = plan.execute({plan.external_inputs[0]: cf.Batch.from_pyarrow(data)})
    assert reloaded.outputs[plan.external_outputs[0]].to_pyarrow().to_pydict() == {
        "z": [2, 4]
    }
