from __future__ import annotations

import pyarrow as pa
import pytest

import calc_flow as cf


def fixture():
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC")),
            pa.field("g", pa.string()),
            pa.field("q", pa.int8()),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array([-1, 0, 9, 10, None, 2, 3], type=schema.field("ts").type),
            pa.array(["A", "A", "A", "A", "A", "B", None]),
            pa.array([2, 3, None, 5, 99, None, 4], type=pa.int8()),
        ],
        schema=schema,
    )
    return cf.table_input("events", schema=schema), table


def declared(source, sql=False, hop=False):
    if sql:
        queries = {
            False: (
                "SELECT window_start, window_end, g, COUNT(q) AS n, SUM(q) AS s, "
                "MIN(q) AS lo, MAX(q) AS hi, AVG(q) AS mean FROM "
                "CF_TUMBLE('a', 'ts', 10) GROUP BY window_start, window_end, g"
            ),
            True: (
                "SELECT window_start, window_end, g, COUNT(q) AS n, SUM(q) AS s, "
                "MIN(q) AS lo, MAX(q) AS hi, AVG(q) AS mean FROM "
                "CF_HOP('a', 'ts', 10, 5) GROUP BY window_start, window_end, g"
            ),
        }
        return cf.sql(queries[hop], a=source)
    kwargs = dict(
        event_time="ts",
        size_micros=10,
        group_by=["g"],
        aggregates=[
            cf.window.count("q", output="n"),
            cf.window.sum("q", output="s"),
            cf.window.min("q", output="lo"),
            cf.window.max("q", output="hi"),
            cf.window.avg("q", output="mean"),
        ],
    )
    return (
        cf.window.hopping(source, slide_micros=5, **kwargs)
        if hop
        else cf.window.tumbling(source, **kwargs)
    )


def rows(table):
    columns = [
        table.column(i).cast(pa.int64()).to_pylist()
        if i < 2
        else table.column(i).to_pylist()
        for i in range(table.num_columns)
    ]
    return sorted(zip(*columns), key=repr)


def execute_table(plan: cf.BatchExecutionPlan, table: pa.Table) -> cf.RunResult:
    batch = cf.Batch.from_pyarrow(table)
    bindings = {plan.external_inputs[0]: batch}
    return plan.execute(inputs=bindings)


@pytest.mark.parametrize("sql", [False, True])
def test_finite_window_public_fixture_and_schema(sql):
    source, table = fixture()
    value = declared(source, sql)
    program = cf.Program("finite", mode="batch", outputs={"bars": value})
    out = program.collect({"events": table})["bars"]
    assert rows(out) == sorted(
        [
            (-10, 0, "A", 1, 2, 2, 2, 2.0),
            (0, 10, "A", 1, 3, 3, 3, 3.0),
            (0, 10, "B", 0, None, None, None, None),
            (0, 10, None, 1, 4, 4, 4, 4.0),
            (10, 20, "A", 1, 5, 5, 5, 5.0),
        ],
        key=repr,
    )
    assert out.schema == pa.schema(
        [
            pa.field("window_start", pa.timestamp("us", tz="UTC"), False),
            pa.field("window_end", pa.timestamp("us", tz="UTC"), False),
            pa.field("g", pa.string()),
            pa.field("n", pa.uint64(), False),
            pa.field("s", pa.int64()),
            pa.field("lo", pa.int8()),
            pa.field("hi", pa.int8()),
            pa.field("mean", pa.float64()),
        ]
    )
    document = program.to_project().model_dump()
    assert (
        sum(node["operator"]["kind"] == "window" for node in document["graph"]["nodes"])
        == 1
    )


def test_window_analysis_proves_conditional_eof_and_leaves_prefix_unknown():
    source, _ = fixture()
    program = cf.Program("proof", mode="stream", outputs={"bars": declared(source)})
    request = cf.PortabilityRequest(
        comparison="final_relation",
        history="accepted_history",
        cutoff="all_sources_eof",
        ordering="unordered",
    )
    report = program.analyze(options=cf.AnalysisOptions(portability=request)).query
    assert (report.schema_version, report.analysis_revision) == (2, 5)
    assert report.portability == "supported", report.diagnostics
    request = cf.PortabilityRequest(
        comparison="final_relation",
        history="event_window",
        cutoff="closed_window",
        ordering="unordered",
    )
    report = program.analyze(options=cf.AnalysisOptions(portability=request)).query
    assert report.portability == "unknown"
    assert any(d.code == "finality_unproven" for d in report.diagnostics)


@pytest.mark.parametrize("sql", [False, True])
@pytest.mark.parametrize("hop", [False, True])
def test_window_batch_stream_eof_and_compiled_plan_isolation(sql, hop):
    import asyncio

    source, table = fixture()
    program = cf.Program(
        "reuse", mode="batch", outputs={"bars": declared(source, sql, hop)}
    )
    runtime = cf.Runtime()
    plan = program.compile_batch(runtime)
    assert program.compile_batch(runtime) is plan
    expected = program.collect({"events": table})["bars"]
    single = pa.Table.from_arrays(
        [
            pa.array([1], type=table.schema.field(0).type),
            pa.array(["A"]),
            pa.array([8], type=pa.int8()),
        ],
        schema=table.schema,
    )
    for current, count in [
        (table, expected.num_rows),
        (pa.concat_tables([table.slice(0, 3), table.slice(3)]), expected.num_rows),
        (single, 2 if hop else 1),
        (
            table.set_column(
                0,
                table.schema.field(0),
                pa.nulls(table.num_rows, type=table.schema.field(0).type),
            ),
            0,
        ),
        (table.slice(0, 0), 0),
    ]:
        out = execute_table(plan, current)
        actual = next(iter(out.outputs.values())).to_pyarrow()
        assert actual.num_rows == count
        if current is single:
            assert actual["s"].to_pylist() == [8] * count
            assert actual["n"].to_pylist() == [1] * count

    async def exercise():
        for chunks in [
            [table],
            [table.slice(i, 1) for i in range(table.num_rows)],
            [table.slice(0, 3), table.slice(0, 0), table.slice(3)],
        ]:
            observed = []
            async with program.for_mode("stream").stream(
                {"events": chunks}, watermarks=cf.DisabledWatermarks()
            ) as events:
                async for event in events:
                    observed.extend(rows(event.table))
            assert sorted(observed, key=repr) == rows(expected)

    asyncio.run(exercise())


@pytest.mark.parametrize("sql", [False, True])
def test_window_mixed_graph_retains_shared_and_independent_branches(sql):
    source, table = fixture()
    renamed = source.select("ts", "g", "q").sql(
        "SELECT ts AS time, g, q FROM input WHERE q IS NULL OR q >= 0"
    )
    if sql:
        bars = cf.sql(
            "SELECT window_start, window_end, g, SUM(q) AS s "
            "FROM CF_TUMBLE('a', 'time', 10) "
            "GROUP BY window_start, window_end, g",
            a=renamed,
        )
    else:
        bars = cf.window.tumbling(
            renamed,
            event_time="time",
            size_micros=10,
            group_by=["g"],
            aggregates=[cf.window.sum("q", output="s")],
        )
    post = bars.select("g", "s").sql("SELECT * FROM input")
    independent = source.select("q").union_all(source.select("q"))
    program = cf.Program(
        "mixed",
        mode="batch",
        outputs={
            "one": bars,
            "two": post,
            "outside": independent,
            "total": source.aggregate({"total": source["q"].sum()}),
        },
    )
    result = program.collect({"events": table})
    assert result["outside"].num_rows == 14
    assert result["total"]["total"].to_pylist() == [113]
    assert result["one"].num_rows == result["two"].num_rows == 5
    doc = program.to_project().model_dump()
    assert sum(n["operator"]["kind"] == "window" for n in doc["graph"]["nodes"]) == 1
    assert sum(n["operator"]["kind"] == "union" for n in doc["graph"]["nodes"]) == 1
    reloaded = cf.Runtime().compile_batch_project(
        cf.ProjectDocument.model_validate(doc).model_dump_json()
    )
    assert len(reloaded.external_inputs) == 1
    restored = execute_table(reloaded, table)
    actual = {
        tuple(batch.to_pyarrow().column_names): sorted(
            batch.to_pyarrow().to_pylist(), key=repr
        )
        for batch in restored.outputs.values()
    }
    expected = {
        tuple(value.column_names): sorted(value.to_pylist(), key=repr)
        for value in result.values()
    }
    assert actual == expected


def test_window_sql_time_arithmetic_is_rejected():
    source, _ = fixture()
    changed = source.sql("SELECT ts + INTERVAL '1 microsecond' AS ts, g, q FROM input")
    program = cf.Program("time", mode="batch", outputs={"bars": declared(changed)})
    with pytest.raises(cf.CompileError, match="unchanged input column or pure rename"):
        program.compile_batch()


@pytest.mark.parametrize(
    "fragment",
    [
        "COUNT(*) AS n",
        "SUM(q + 1) AS n",
        "COUNT(DISTINCT q) AS n",
        "SUM(q) FILTER (WHERE q > 0) AS n",
        "ROW_NUMBER() OVER () AS n",
    ],
)
def test_window_sql_rejects_unsupported_aggregates(fragment):
    source, _ = fixture()
    value = cf.sql(
        "SELECT window_start, window_end, "
        + fragment
        + " FROM CF_TUMBLE('a', 'ts', 10) GROUP BY window_start, window_end",
        a=source,
    )
    with pytest.raises(cf.CompileError, match="sql.window"):
        cf.Program("reject", mode="batch", outputs={"bars": value}).compile_batch()


@pytest.mark.parametrize("hop", [False, True])
def test_window_memberships_have_independent_negative_and_boundary_oracle(hop):
    timestamps = [-11, -10, -1, 0, 4, 5, 9, 10, None]
    schema = pa.schema([("ts", pa.timestamp("us")), ("id", pa.int64())])
    source = cf.table_input("input", schema=schema)
    args = dict(
        event_time="ts",
        size_micros=10,
        group_by=["id"],
        aggregates=[cf.window.count("id", output="n")],
    )
    value = (
        cf.window.hopping(source, slide_micros=5, **args)
        if hop
        else cf.window.tumbling(source, **args)
    )
    table = pa.Table.from_arrays(
        [pa.array(timestamps, type=pa.timestamp("us")), pa.array(range(9))],
        schema=schema,
    )
    result = cf.Program("memberships", mode="batch", outputs={"out": value}).collect(
        {"input": table}
    )["out"]
    actual = sorted(
        zip(
            result["id"].to_pylist(),
            result["window_start"].cast(pa.int64()).to_pylist(),
        )
    )
    expected = (
        [[-20, -15], [-15, -10], [-10, -5], [-5, 0], [-5, 0], [0, 5], [0, 5], [5, 10]]
        if hop
        else [[-20], [-10], [-10], [0], [0], [0], [0], [10]]
    )
    assert actual == [
        (index, start) for index, starts in enumerate(expected) for start in starts
    ]
    assert result["n"].to_pylist() == [1] * result.num_rows


@pytest.mark.parametrize(
    "dtype,output",
    [(pa.int8(), pa.int64()), (pa.uint8(), pa.uint64()), (pa.float32(), pa.float64())],
)
def test_window_sum_promotes_and_empty_has_no_global_row(dtype, output):
    schema = pa.schema([("ts", pa.timestamp("us")), ("q", dtype)])
    source = cf.table_input("input", schema=schema)
    value = cf.window.tumbling(
        source,
        event_time="ts",
        size_micros=10,
        aggregates=[
            cf.window.sum("q", output="s"),
            cf.window.count("q", output="n"),
            cf.window.avg("q", output="mean"),
        ],
    )
    program = cf.Program("types", mode="batch", outputs={"out": value})
    data = pa.Table.from_arrays(
        [
            pa.array([0, 1, 2], type=pa.timestamp("us")),
            pa.array([1, None, 2], type=dtype),
        ],
        schema=schema,
    )
    result = program.collect({"input": data})["out"]
    assert result["s"].type == output
    assert result.select(["s", "n", "mean"]).to_pydict() == {
        "s": [3],
        "n": [2],
        "mean": [1.5],
    }
    assert program.collect({"input": data.slice(0, 0)})["out"].num_rows == 0


@pytest.mark.parametrize("intermediate", [False, True])
@pytest.mark.parametrize("mode", ["batch", "stream"])
def test_low_level_latest_snapshot_window_is_rejected_before_execution(
    mode, intermediate
):
    document = _snapshot_window_document(mode, intermediate)
    import json

    runtime = cf.Runtime()
    compile_plan = (
        runtime.compile_batch_project
        if mode == "batch"
        else runtime.compile_stream_project
    )
    for _ in range(2):
        with pytest.raises(cf.CompileError, match="update_interpretation_mismatch"):
            compile_plan(json.dumps(document))


def _snapshot_window_document(mode, intermediate):
    source, _ = fixture()
    program = cf.Program("latest", mode=mode, outputs={"out": declared(source)})
    document = (
        program.with_update_semantics("latest_relation").to_project().model_dump()
    )
    graph = document["graph"]
    window = next(n for n in graph["nodes"] if n["operator"]["kind"] == "window")
    edge = next(e for e in graph["edges"] if e["target_node"] == window["id"])
    producer = next(n for n in graph["nodes"] if n["id"] == edge["source_node"])
    producer["operator"] = dict(
        kind="sql",
        query="SELECT MIN(ts) AS ts, MIN(g) AS g, MIN(q) AS q FROM input",
        aliases=["input"],
        udfs=[],
    )
    if intermediate:
        wrapper = {
            **producer,
            "id": "passthrough",
            "operator": dict(
                kind="sql", query="SELECT * FROM input", aliases=["input"], udfs=[]
            ),
        }
        graph["nodes"].append(wrapper)
        graph["edges"].append(
            dict(
                source_node=producer["id"],
                source_port="output",
                target_node="passthrough",
                target_port="input",
            )
        )
        edge["source_node"] = "passthrough"
    graph["update_semantics"] = "latest_relation"
    return document


@pytest.mark.parametrize("sql", [False, True])
def test_latest_append_window_is_admitted_and_empty_stays_empty(sql):
    import asyncio

    source, table = fixture()
    program = cf.Program("append", mode="batch", outputs={"out": declared(source, sql)})
    latest = program.with_update_semantics("latest_relation")
    assert latest.collect({"events": table})["out"].num_rows == 5
    runtime = cf.Runtime()
    latest.compile_batch(runtime)
    latest.compile_batch(runtime)
    latest.for_mode("stream").compile_stream(runtime)

    async def exercise():
        async with latest.for_mode("stream").stream(
            {"events": [table.slice(0, 0)]},
            runtime=runtime,
            watermarks=cf.DisabledWatermarks(),
        ) as events:
            assert [event async for event in events] == []

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "call",
    [
        "CF_TUMBLE('a', 'ts', 10, 0)",
        "CF_HOP('a', 'ts', 10, 5, 0)",
        "CF_HOP('a', 'ts', 10, 3)",
        "CF_HOP('a', 'ts', 1025, 1)",
        "CF_TUMBLE('a', 'ts', 0)",
        "CF_TUMBLE('a', 'ts', -1)",
        "CF_TUMBLE('a', 'ts', 1.5)",
        "CF_TUMBLE('a', 'ts', 18446744073709551616)",
        "CF_TUMBLE('missing', 'ts', 10)",
    ],
)
def test_window_sql_rejects_geometry_and_alias_outside_contract(call):
    source, _ = fixture()
    value = cf.sql(
        "SELECT window_start, window_end, COUNT(q) AS n FROM "
        + call
        + " GROUP BY window_start, window_end",
        a=source,
    )
    with pytest.raises(cf.CompileError):
        cf.Program("geometry", mode="batch", outputs={"out": value}).compile_batch()


def test_window_sql_geometry_diagnostic_keeps_native_field_and_code():
    source, _ = fixture()
    value = cf.sql(
        "SELECT window_start, window_end, COUNT(q) AS n "
        "FROM CF_TUMBLE('a', 'ts', 10, 0) "
        "GROUP BY window_start, window_end",
        a=source,
    )
    report = cf.Program("diagnostic", mode="batch", outputs={"out": value}).analyze()
    assert any(
        issue.path == "outputs.out.sql.window.geometry"
        and issue.code == "invalid_literal"
        for issue in report.issues
    )


def test_closed_window_request_does_not_prove_a_graph_without_windows():
    source, _ = fixture()
    program = cf.Program("plain", mode="batch", outputs={"out": source})
    request = cf.PortabilityRequest(
        comparison="final_relation",
        history="event_window",
        cutoff="closed_window",
        ordering="unordered",
    )
    report = program.analyze(options=cf.AnalysisOptions(portability=request)).query
    assert report.portability == "unsupported"
    assert any(issue.code == "history_scope_mismatch" for issue in report.diagnostics)


def test_independent_sql_window_relations_cannot_align_columns():
    source, _ = fixture()
    query = (
        "SELECT window_start, window_end, COUNT(q) AS n "
        "FROM CF_TUMBLE('a', 'ts', 10) GROUP BY window_start, window_end"
    )
    first = cf.sql(query, a=source)
    second = cf.sql(query + " ", a=source)
    mixed = first.with_columns(bad=first["n"] + second["n"])
    program = cf.Program("origins", mode="batch", outputs={"out": mixed})
    assert any(issue.code == "schema_mismatch" for issue in program.analyze().issues)


@pytest.mark.parametrize(
    "dtype,maximum", [(pa.int64(), 2**63 - 1), (pa.uint64(), 2**64 - 1)]
)
def test_sum_overflow_and_schema_failure_leave_the_warm_plan_fresh(dtype, maximum):
    schema = pa.schema([("ts", pa.timestamp("us")), ("q", dtype)])
    source = cf.table_input("input", schema=schema)
    value = cf.window.tumbling(
        source,
        event_time="ts",
        size_micros=10,
        aggregates=[cf.window.sum("q", output="s")],
    )
    runtime = cf.Runtime()
    plan = cf.Program("failures", mode="batch", outputs={"out": value}).compile_batch(
        runtime
    )

    def execute(table):
        bindings = {plan.external_inputs[0]: cf.Batch.from_pyarrow(table)}
        output = plan.execute(inputs=bindings)
        return next(iter(output.outputs.values())).to_pyarrow()

    bad = pa.Table.from_arrays(
        [pa.array([0, 1], type=pa.timestamp("us")), pa.array([maximum, 1], type=dtype)],
        schema=schema,
    )
    good = pa.Table.from_arrays(
        [pa.array([1], type=pa.timestamp("us")), pa.array([8], type=dtype)],
        schema=schema,
    )
    with pytest.raises(cf.ExecutionError, match="overflow"):
        execute(bad)
    assert execute(good)["s"].to_pylist() == [8]
    with pytest.raises(cf.CompileError, match="schema"):
        execute(good.set_column(1, "q", pa.array([100], type=pa.int8())))
    assert execute(good)["s"].to_pylist() == [8]


def test_hopping_overlap_limit_and_equal_slide_are_admitted():
    schema = pa.schema([("ts", pa.timestamp("us")), ("q", pa.int8())])
    source = cf.table_input("input", schema=schema)
    table = pa.Table.from_arrays(
        [pa.array([0], type=pa.timestamp("us")), pa.array([1], type=pa.int8())],
        schema=schema,
    )
    for size, slide, expected in [(1024, 1, list(range(-1023, 1))), (10, 10, [0])]:
        value = cf.window.hopping(
            source,
            event_time="ts",
            size_micros=size,
            slide_micros=slide,
            aggregates=[cf.window.count("q", output="n")],
        )
        result = cf.Program("limit", mode="batch", outputs={"out": value}).collect(
            {"input": table}
        )["out"]
        assert result["window_start"].cast(pa.int64()).to_pylist() == expected
        assert result["n"].to_pylist() == [1] * len(expected)


@pytest.mark.parametrize(
    "query,time",
    [
        ("SELECT input.ts AS time, g, q FROM input", "time"),
        ("SELECT e.ts AS time, e.g, e.q FROM input AS e", "time"),
        ("SELECT e.* FROM input AS e", "ts"),
    ],
)
def test_sql_window_time_proof_accepts_qualified_passthrough(query, time):
    source, table = fixture()
    before = source.sql(query)
    bars = cf.window.tumbling(
        before,
        event_time=time,
        size_micros=10,
        aggregates=[cf.window.sum("q", output="s")],
    )
    result = cf.Program("qualified", mode="batch", outputs={"out": bars}).collect(
        {"events": table}
    )["out"]
    assert result["s"].to_pylist() == [2, 7, 5]
