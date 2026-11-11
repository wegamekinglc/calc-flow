from __future__ import annotations

import pyarrow as pa
import pytest

import calc_flow as cf


def test_replacement_analysis_edges_preserve_declared_input_aliases():
    from calc_flow.symbolic.query_analysis import _query_edges

    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    for value, alias in [
        (source.aggregate({"total": source["x"].sum()}), "input"),
        (cf.sql("SELECT SUM(x) AS total FROM incoming", incoming=source), "incoming"),
    ]:
        node = value._node
        edges = _query_edges(
            {node.digest: node},
            {node.digest: "outputs.result", source._node.digest: "inputs.events"},
        )
        assert len(edges) == 1
        assert edges[0].get("input_alias") == alias
        assert edges[0]["source_path"] == "inputs.events"
        assert edges[0]["target_path"] == "outputs.result"


def test_aggregate_schema_quotes_sql_syntax_inside_identifiers():
    column = 'x"), COUNT(*) AS injected FROM input --'
    key = 'g" GROUP BY injected --'
    name = 'total" FROM input; SELECT 1 --'
    source = cf.table_input(
        "events", schema=pa.schema([(key, pa.string()), (column, pa.int64())])
    )
    result = source.aggregate({name: source[column].sum()}, group_by=[key])
    program = cf.Program("quoted", mode="batch", outputs={"result": result})
    assert program.collect_schema()["result"] == pa.schema(
        [(key, pa.string()), (name, pa.int64())]
    )


@pytest.mark.parametrize(
    "values,total,count,rows",
    [([], None, 0, 0), ([None, None], None, 0, 2), ([1, None, 3], 4, 2, 3)],
)
def test_dataframe_aggregate_nulls(values, total, count, rows):
    source = cf.table_input("events", schema=pa.schema([("x", pa.int32())]))
    result = source.aggregate(
        {
            "total": source["x"].sum(),
            "count": source["x"].count(),
            "rows": cf.count_rows(),
        }
    )
    program = cf.Program("aggregate", mode="batch", outputs={"result": result})
    actual = program.collect(
        {"events": pa.table({"x": pa.array(values, type=pa.int32())})}
    )["result"]
    assert actual.to_pydict() == {"total": [total], "count": [count], "rows": [rows]}
    assert actual.schema == pa.schema(
        [
            pa.field("total", pa.int64(), True),
            pa.field("count", pa.int64(), False),
            pa.field("rows", pa.int64(), False),
        ]
    )


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (pa.int32(), pa.int64()),
        (pa.uint32(), pa.uint64()),
        (pa.float32(), pa.float64()),
    ],
)
def test_dtype_reducers_and_null_group(dtype, expected):
    source = cf.table_input(
        "events", schema=pa.schema([("g", pa.string()), ("x", dtype)])
    )
    result = source.aggregate(
        {
            "total": source["x"].sum(),
            "lo": source["x"].min(),
            "hi": source["x"].max(),
            "mean": source["x"].mean(),
            "rows": cf.count_rows(),
        },
        group_by=["g"],
    )
    sql = cf.sql(
        "SELECT g, SUM(x) AS total, MIN(x) AS lo, MAX(x) AS hi, "
        "AVG(x) AS mean, COUNT(*) AS rows FROM input GROUP BY g",
        input=source,
    )
    inputs = {
        "events": pa.table(
            {"g": [None, None, "a"], "x": pa.array([1, 3, None], type=dtype)}
        )
    }
    program = cf.Program("dtype", mode="batch", outputs={"df": result, "sql": sql})
    actual = program.collect(inputs)
    schema = pa.schema(
        [
            pa.field("g", pa.string()),
            pa.field("total", expected),
            pa.field("lo", dtype),
            pa.field("hi", dtype),
            pa.field("mean", pa.float64()),
            pa.field("rows", pa.int64(), False),
        ]
    )
    for table in actual.values():
        assert table.schema == schema
        assert sorted(table.to_pylist(), key=lambda row: str(row["g"])) == sorted(
            [
                {"g": None, "total": 4, "lo": 1, "hi": 3, "mean": 2.0, "rows": 2},
                {
                    "g": "a",
                    "total": None,
                    "lo": None,
                    "hi": None,
                    "mean": None,
                    "rows": 1,
                },
            ],
            key=lambda row: str(row["g"]),
        )


def _chain(semantics, dataframe=True):
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    if dataframe == "sql_df":
        source = source.sql("SELECT x FROM input")
        dataframe = True
    first = (
        source.aggregate({"total": source["x"].sum()})
        if dataframe
        else source.sql("SELECT SUM(x) AS total FROM input")
    )
    visible = first.select("total")
    visible = visible.filter(visible["total"] >= cf.lit(0))
    result = cf.sql("SELECT SUM(total) AS value FROM s", s=visible)
    return cf.Program(
        "replacement", mode="stream", outputs={"result": result, "shared": first}
    ).with_update_semantics(semantics)


def _observe(program, data, config=None):
    import asyncio

    async def feed():
        for table in data:
            yield table

    async def run():
        async with program.stream({"events": feed()}, config=config) as events:
            seen = {"result": [], "shared": []}
            async for event in events:
                seen[event.name].append(event.table.to_pydict())
            return seen

    return asyncio.run(run())


@pytest.mark.parametrize("dataframe", [True, False, "sql_df"])
@pytest.mark.parametrize(
    "semantics,last", [("latest_relation", 6), ("legacy_append", 7)]
)
def test_mixed_chain_preserves_updates_and_shared_output(dataframe, semantics, last):
    program = _chain(semantics, dataframe)
    config = cf.StreamRuntimeConfig(
        sql_replacement_state_budget=cf.StateBudget(100, 1 << 24)
    )
    data = [pa.table({"x": [1]}), pa.table({"x": [2, 3]})]
    assert _observe(program, data, config) == {
        "result": [{"value": [1]}, {"value": [last]}],
        "shared": [{"total": [1]}, {"total": [6]}],
    }
    finite = program.for_mode("batch").compile_batch()
    for values, expected in [([1], 1), ([2, 3], 5), ([1, 2, 3], 6)]:
        # The same warmed native plan executes twice without a reset.
        result = finite.execute(
            {finite.external_inputs[0]: cf.Batch.from_pyarrow(pa.table({"x": values}))}
        )
        tables = [batch.to_pyarrow() for batch in result.outputs.values()]
        actual = {tuple(table.column_names): table.to_pydict() for table in tables}
        assert actual == {
            ("value",): {"value": [expected]},
            ("total",): {"total": [expected]},
        }
    assert _observe(program, [], config) == {"result": [], "shared": []}


@pytest.mark.parametrize("compact", [False, True])
def test_budget_preflight_precedes_iterator_bindings_and_job(monkeypatch, compact):
    import asyncio

    import calc_flow.stream as stream_module

    program = _chain("latest_relation")
    runtime = cf.Runtime()
    program.compile_stream(runtime)

    class Unread:
        def __aiter__(self):
            raise AssertionError("iterator acquired")

    def forbid(*args, **kwargs):
        raise AssertionError("bindings factory called")

    monkeypatch.setattr(stream_module.StreamResults, "_bindings", forbid)
    config = cf.StreamRuntimeConfig(
        sql_replacement_state_budget=cf.StateBudget(10, 1 << 20) if compact else None,
        sql_compact_state_budget=cf.StateBudget(10, 1 << 20) if compact else None,
    )

    async def run():
        async with program.stream({"events": Unread()}, config=config, runtime=runtime):
            raise AssertionError("job started")

    with pytest.raises(
        (ValueError, cf.ConfigError),
        match="sql_compact_state_budget" if compact else "sql_replacement_state_budget",
    ):
        asyncio.run(run())


@pytest.mark.parametrize(
    "query",
    [
        "SELECT SUM(DISTINCT total) AS n FROM s",
        "SELECT SUM(total) AS n FROM s HAVING SUM(total)>0",
        "SELECT SUM(total) AS n FROM s ORDER BY 1",
        "SELECT SUM(total) AS n FROM s LIMIT 1",
        "SELECT SUM(total)+1 AS n FROM s",
        "SELECT SUM(total) AS n FROM s WHERE total>0",
        "SELECT SUM(ABS(total)) AS n FROM s",
        "SELECT COUNT(1) AS n FROM s",
        "WITH c AS (SELECT * FROM s) SELECT SUM(total) AS n FROM c",
        "SELECT SUM(a.total) AS n FROM s a JOIN s b ON a.total=b.total",
    ],
)
def test_replacement_allowlist_rejects_unsupported_queries(query):
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    first = source.aggregate({"total": source["x"].sum()})
    program = cf.Program(
        "reject", mode="stream", outputs={"result": cf.sql(query, s=first)}
    ).with_update_semantics("latest_relation")
    with pytest.raises(
        cf.CompileError,
        match="update_interpretation_mismatch|update_semantics_unproved",
    ) as caught:
        program.compile_stream()
    expected = next(
        reason
        for marker, reason in [
            ("DISTINCT", "DISTINCT/ALL"),
            ("HAVING", "HAVING"),
            ("ORDER BY", "ORDER BY"),
            ("LIMIT", "LIMIT/OFFSET/FETCH"),
            ("+1", "projection"),
            ("WHERE", "WHERE"),
            ("ABS", "direct column"),
            ("COUNT(1)", "direct column"),
            ("WITH c", "CTE"),
            ("JOIN", "JOIN"),
        ]
        if marker in query
    )
    root = next(
        node
        for node in caught.value.query_analysis.nodes
        if node.path == "outputs.result"
    )
    assert expected in root.stream.reason
    program.for_mode("batch").compile_batch()


def test_aggregate_copies_names_and_rejects_invalid_domains():
    source = cf.table_input(
        "events", schema=pa.schema([("x", pa.int64()), ("g", pa.string())])
    )
    mapping = {"total": source["x"].sum()}
    groups = ["g"]
    result = source.aggregate(mapping, group_by=groups)
    digest = result.digest
    mapping.clear()
    groups.clear()
    assert result.digest == digest
    assert result.collect_schema().names == ["g", "total"]
    with pytest.raises(ValueError, match="direct column"):
        (source["x"] + cf.lit(1)).sum()
    with pytest.raises(TypeError):
        source.aggregate({"bad": source["x"]})
    with pytest.raises(ValueError, match="duplicate_name"):
        source.aggregate({"g": cf.count_rows()}, group_by=["g"])
    bad = source.aggregate({"bad": source["g"].sum()})
    with pytest.raises(cf.CompileError, match="aggregate.aggregates.bad.value"):
        bad.collect_schema()


def test_native_budget_pairs_and_project_reload_selection():
    program = _chain("latest_relation")
    runtime = cf.Runtime()
    plan = program.compile_stream(runtime)
    config = cf.StreamRuntimeConfig(
        sql_replacement_state_budget=cf.StateBudget(100, 1 << 24)
    )._native()
    plan._inner.validate_runtime_config(config)
    for key in ("sql_replacement_state_max_rows", "sql_replacement_state_max_bytes"):
        incomplete = dict(config)
        del incomplete[key]
        with pytest.raises((ValueError, TypeError), match="supplied together"):
            plan._inner.validate_runtime_config(incomplete)
    from calc_flow.store import import_project_json

    project = import_project_json(program.to_project().model_dump_json())
    reloaded = runtime._compile_stream_graph_project(project.model_dump_json())
    assert reloaded.fingerprint == plan.fingerprint
    reloaded._inner.validate_runtime_config(config)
    with pytest.raises(TypeError, match="AggregateExpr"):
        cf.AggregateExpr()
