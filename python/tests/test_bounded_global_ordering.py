from __future__ import annotations

import asyncio
import importlib

import pyarrow as pa
import pytest

import calc_flow as cf


def test_dataframe_and_sql_top_k_export_same_native_node():
    schema = pa.schema(
        [pa.field("x", pa.int64()), pa.field("seq", pa.uint64(), nullable=False)]
    )
    t = cf.table_input("events", schema=schema)
    spec = cf.SortSpec.bounded_source(
        cf.StateBudget(10, 8 << 20), offset=1, sequence_by=["seq"]
    )
    df = t.top_k(1, [cf.SortKey("x")], spec=spec)
    sql = cf.sql(
        "SELECT * FROM CF_ORDER_BOUNDED('a',10,8388608,ARRAY['seq']) "
        "ORDER BY x ASC NULLS LAST LIMIT 1 OFFSET 1",
        a=t,
    )
    data = pa.Table.from_pydict({"x": [1, 1, 2], "seq": [20, 10, 40]}, schema)
    configs = []
    for out in (df, sql):
        program = cf.Program("order", mode="batch", outputs={"out": out})
        result = program.collect({"events": data})["out"]
        assert result.schema == schema
        assert result.to_pydict() == {"x": [1], "seq": [20]}
        nodes = program.to_project().model_dump()["graph"]["nodes"]
        configs.append(
            next(
                n["operator"] for n in nodes if n["operator"]["kind"] == "bounded_order"
            )
        )
    assert configs[0] == configs[1]


def test_unknown_source_rejected_before_iterator_and_temporary_root(monkeypatch):
    stream_module = importlib.import_module("calc_flow.stream")
    roots = []
    original = stream_module.tempfile.mkdtemp

    def traced_root(*args, **kwargs):
        roots.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(stream_module.tempfile, "mkdtemp", traced_root)
    touched = []

    class Unknown:
        def __aiter__(self):
            touched.append("iterator")
            raise AssertionError("source iterator acquired")

    t = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    out = t.top_k(
        1,
        [cf.SortKey("x")],
        spec=cf.SortSpec.bounded_source(cf.StateBudget(2, 8 << 20)),
    )
    program = cf.Program("preflight", mode="stream", outputs={"out": out})

    async def run():
        with pytest.raises(
            cf.StreamingRuntimeError, match="source_boundedness_unproved"
        ) as rejected:
            async with program.stream({"events": Unknown()}):
                pass
        assert rejected.value.component_id == "events"

    asyncio.run(run())
    assert touched == []
    assert roots == []


@pytest.mark.parametrize("syntax", ["df", "sql"])
def test_stream_global_top_one_preserves_two_legacy_batch_results(syntax):
    t = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    out = (
        t.top_k(
            1,
            [cf.SortKey("x")],
            spec=cf.SortSpec.bounded_source(cf.StateBudget(3, 8 << 20)),
        )
        if syntax == "df"
        else cf.sql(
            "SELECT * FROM CF_ORDER_BOUNDED('a',3,8388608) ORDER BY x LIMIT 1", a=t
        )
    )
    legacy = cf.sql("SELECT x FROM a ORDER BY x LIMIT 1", a=t)
    program = cf.Program(
        "two_batches", mode="stream", outputs={"out": out, "legacy": legacy}
    )

    async def run():
        source = cf.SourceBinding.from_arrow(
            pa.table({"x": [3, 1, 2]}), budget=cf.EdgeBudget(max_rows=2, max_bytes=1024)
        )
        events = []
        async with program.stream(
            {"events": source}, config=cf.StreamRuntimeConfig(checkpointing=False)
        ) as results:
            async for event in results:
                events.append((event.name, event.table.to_pydict()))
        assert [data for name, data in events if name == "legacy"] == [
            {"x": [1]},
            {"x": [2]},
        ]
        assert [data for name, data in events if name == "out"] == [{"x": [1]}]

    asyncio.run(run())


def test_limits_schema_and_macro_errors_are_declaration_failures():
    budget = cf.StateBudget(10, 8 << 20)
    for value in (-1, True, 1.5, 1 << 32):
        with pytest.raises((TypeError, ValueError), match="4294967295"):
            cf.SortSpec.bounded_source(budget, offset=value)
    with pytest.raises(ValueError, match="4294967295"):
        cf.SortSpec.bounded_source(cf.StateBudget(1 << 32, 8 << 20))
    t = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    spec = cf.SortSpec.bounded_source(budget)
    for value in (-1, True, 1.5, 1 << 32):
        with pytest.raises((TypeError, ValueError), match="4294967295"):
            t.top_k(value, [cf.SortKey("x")], spec=spec)
    with pytest.raises(ValueError, match="4294967295"):
        t.top_k(
            1,
            [cf.SortKey("x")],
            spec=cf.SortSpec.bounded_source(budget, offset=(1 << 32) - 1),
        )
    for query in (
        "SELECT * FROM CF_ORDER_WRONG('a',10,8388608) ORDER BY x",
        "SELECT * FROM CF_ORDER_BOUNDED('a',10,8388608) ORDER BY x+1",
        "SELECT * FROM CF_ORDER_BOUNDED('a',10,8388608) WHERE x>0 ORDER BY x",
    ):
        with pytest.raises(cf.CompileError):
            cf.Program(
                "bad_macro", mode="batch", outputs={"out": cf.sql(query, a=t)}
            ).to_project()


def test_analysis_r10_stable_final_order_and_unproved_selection():
    schema = pa.schema(
        [pa.field("x", pa.int64()), pa.field("seq", pa.uint64(), nullable=False)]
    )
    t = cf.table_input("events", schema=schema)
    request = cf.PortabilityRequest(
        comparison="final_relation",
        history="accepted_history",
        cutoff="all_sources_eof",
        ordering="declared",
    )
    for sequence, expected in ((["seq"], "supported"), ([], "unknown")):
        out = t.top_k(
            1,
            [cf.SortKey("x")],
            spec=cf.SortSpec.bounded_source(
                cf.StateBudget(10, 8 << 20), sequence_by=sequence
            ),
        )
        program = cf.Program("facts", mode="batch", outputs={"out": out})
        report = program.analyze(options=cf.AnalysisOptions(portability=request)).query
        assert report.analysis_revision == 10
        assert report.portability == expected
        if expected == "supported":
            program.compile_batch(portability=request)
        else:
            with pytest.raises(cf.CompileError):
                program.compile_batch(portability=request)


def test_sort_local_late_policy_rejects_unused_global_override():
    schema = pa.schema([pa.field("ts", pa.timestamp("us")), pa.field("x", pa.int64())])
    t = cf.table_input("events", schema=schema)
    out = t.sort(
        [cf.SortKey("x")],
        spec=cf.SortSpec.closed_window(
            cf.SortWindow("ts", start_micros=0, end_micros=10),
            state_budget=cf.StateBudget(2, 8 << 20),
            late_policy="drop",
        ),
    )
    program = cf.Program("local", mode="stream", outputs={"out": out})
    document = program.to_project().model_dump()
    order = next(
        n["operator"]
        for n in document["graph"]["nodes"]
        if n["operator"]["kind"] == "bounded_order"
    )
    assert order["spec"]["late_policy"] == {"kind": "drop", "metrics_version": 1}
    with pytest.raises(cf.CompileError, match="SortSpec"):
        program.to_project(late_policy="drop")


def test_order_schema_expresses_uint32_limits_and_retains_nulls():
    schema = cf.ProjectDocument.model_json_schema()

    def resolve(value):
        return (
            schema["$defs"][value["$ref"].split("/")[-1]] if "$ref" in value else value
        )

    spec = schema["$defs"]["SortSpec"]["properties"]
    assert spec["history"]["$ref"] == "#/$defs/SortHistory"
    assert resolve(spec["offset"])["maximum"] == (1 << 32) - 1
    budget = resolve(spec["state_budget"])["properties"]
    assert budget["max_rows"]["maximum"] == (1 << 32) - 1
    nulls = resolve(schema["$defs"]["SortKey"]["properties"]["nulls"])
    assert nulls["enum"] == ["first", "last"]
    variants = schema["$defs"]["OperatorSpec"]["oneOf"]
    order = next(
        v for v in variants if v["properties"]["kind"].get("const") == "bounded_order"
    )
    assert "fetch" in order["required"]
    assert order["properties"]["by"]["minItems"] == 1
    assert (
        schema["$defs"]["DistinctSpec"]["properties"]["ingress_policy"]["$ref"]
        == "#/$defs/IngressPolicy"
    )


def test_sort_schema_and_sequence_origin_errors_keep_logical_paths():
    unsupported = cf.table_input(
        "events",
        schema=pa.schema(
            [pa.field("x", pa.int64()), pa.field("payload", pa.float64())]
        ),
    )
    out = unsupported.sort(
        [cf.SortKey("x")], spec=cf.SortSpec.bounded_source(cf.StateBudget(10, 8 << 20))
    )
    report = cf.Program("types", mode="batch", outputs={"out": out}).analyze()
    assert any(
        issue.path.endswith(".sort.input.schema[1]")
        and issue.code == "unsupported_type"
        for issue in report.issues
    )
    schema = pa.schema(
        [pa.field("x", pa.int64()), pa.field("seq", pa.uint64(), nullable=False)]
    )
    t = cf.table_input("events", schema=schema).sql("SELECT x, seq+1 AS seq FROM input")
    out = t.sort(
        [cf.SortKey("x")],
        spec=cf.SortSpec.bounded_source(
            cf.StateBudget(10, 8 << 20), sequence_by=["seq"]
        ),
    )
    report = cf.Program("origin", mode="batch", outputs={"out": out}).analyze()
    assert any(
        issue.path.endswith(".sort.spec.ties.columns[0]")
        and issue.code == "capability_mismatch"
        for issue in report.issues
    )


def test_declared_order_survives_literal_dataframe_projection():
    schema = pa.schema(
        [pa.field("x", pa.int64()), pa.field("seq", pa.uint64(), nullable=False)]
    )
    t = cf.table_input("events", schema=schema)
    sorted_values = t.sort(
        [cf.SortKey("x")],
        spec=cf.SortSpec.bounded_source(
            cf.StateBudget(10, 8 << 20), sequence_by=["seq"]
        ),
    )
    request = cf.PortabilityRequest(
        comparison="final_relation",
        history="accepted_history",
        cutoff="all_sources_eof",
        ordering="declared",
    )
    for out, expected in [
        (sorted_values.select("x", "seq"), "supported"),
        (sorted_values.select("x"), "unknown"),
    ]:
        p = cf.Program("project_order", mode="batch", outputs={"out": out})
        assert (
            p.analyze(options=cf.AnalysisOptions(portability=request)).query.portability
            == expected
        )


@pytest.mark.parametrize(
    ("name", "projection", "portable"),
    [
        ("x", 'SELECT "x", "seq" FROM input', True),
        (
            'x" FROM input; DROP TABLE input; --',
            'SELECT "x"" FROM input; DROP TABLE input; --", "seq" FROM input',
            False,
        ),
        ("select", 'SELECT "select", "seq" FROM input', True),
    ],
)
def test_order_projection_treats_quoted_column_contents_as_identifiers(
    name, projection, portable
):
    from calc_flow.symbolic.order import projection_query

    schema = pa.schema(
        [pa.field(name, pa.int64()), pa.field("seq", pa.uint64(), nullable=False)]
    )
    events = cf.table_input("events", schema=schema)
    ordered = events.top_k(
        1,
        [cf.SortKey(name)],
        spec=cf.SortSpec.bounded_source(
            cf.StateBudget(10, 8 << 20), sequence_by=["seq"]
        ),
    ).select(name, "seq")
    assert projection_query(ordered._node) == projection
    program = cf.Program("quoted_order", mode="batch", outputs={"out": ordered})
    data = pa.record_batch(
        [
            pa.array([3, 1, 2], type=pa.int64()),
            pa.array([10, 11, 12], type=pa.uint64()),
        ],
        schema=schema,
    )
    if portable:
        assert program.collect({"events": data})["out"].to_pydict() == {
            name: [1],
            "seq": [11],
        }
    else:
        with pytest.raises(
            cf.ConfigError, match="Arrow field name must be a portable SQL identifier"
        ):
            program.collect({"events": data})
