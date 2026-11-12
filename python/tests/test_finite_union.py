from __future__ import annotations

import pyarrow as pa
import pytest

import calc_flow as cf


def schema():
    return pa.schema([pa.field("x", pa.int64(), nullable=True)])


def source(name):
    return cf.table_input(name, schema=schema())


def test_union_declaration_keeps_order_and_multiplicity():
    a, b = source("a"), source("b")
    value = a.union_all(b)
    assert value.digest != b.union_all(a).digest
    assert a.union_all(a)._node.args == (a._node, a._node)
    with pytest.raises(TypeError, match="union_all.other requires TableExpr"):
        a.union_all(1)


def table(values):
    return pa.Table.from_arrays([pa.array(values, type=pa.int64())], schema=schema())


def program(kind="df"):
    a, b = source("a"), source("b")
    value = (
        a.union_all(b)
        if kind == "df"
        else cf.sql("SELECT x FROM a UNION ALL SELECT x FROM b", a=a, b=b)
    )
    return cf.Program("union", mode="batch", outputs={"out": value})


@pytest.mark.parametrize("kind", ["df", "sql"])
def test_public_union_both_modes_keep_nulls_duplicates_and_empty(kind):
    import asyncio

    asyncio.run(_public_union_both_modes(kind))


async def _public_union_both_modes(kind):
    import asyncio
    from collections import Counter

    batch = program(kind)
    stream = batch.for_mode("stream")
    assert batch.fingerprint == stream.fingerprint
    assert (await asyncio.to_thread(batch.collect_schema))["out"] == schema()
    for left, right in [([1, None, 1], [1, 2]), ([9], []), ([], [])]:
        expected = Counter(left + right)
        result = await batch.collect_async({"a": table(left), "b": table(right)})
        assert Counter(result["out"]["x"].to_pylist()) == expected
        seen = []
        async with stream.stream({"a": [table(left)], "b": [table(right)]}) as events:
            async for event in events:
                seen.extend(event.table["x"].to_pylist())
        assert Counter(seen) == expected
    document = batch.to_project().model_dump()
    assert any(
        node["operator"]["kind"] == "union" for node in document["graph"]["nodes"]
    )


def request(history="accepted_history", ordering="unordered"):
    return cf.PortabilityRequest(
        comparison="final_relation",
        history=history,
        cutoff="all_sources_eof",
        ordering=ordering,
    )


@pytest.mark.parametrize("kind", ["df", "sql"])
def test_union_reports_append_history_and_no_global_order(kind):
    batch = program(kind)
    report = batch.analyze(options=cf.AnalysisOptions(portability=request())).query
    assert (report.schema_version, report.analysis_revision) == (2, 7)
    assert report.portability == "supported", report.diagnostics
    report = batch.analyze(
        options=cf.AnalysisOptions(portability=request(ordering="declared"))
    ).query
    assert report.portability == "unknown"
    assert any(issue.code == "row_order_unproven" for issue in report.diagnostics)
    report = batch.analyze(
        options=cf.AnalysisOptions(portability=request("latest_relation"))
    ).query
    assert any(issue.code == "history_scope_mismatch" for issue in report.diagnostics)
    per_batch = cf.PortabilityRequest(
        comparison="per_batch", history="current_batch", cutoff="batch_boundary"
    )
    report = batch.analyze(options=cf.AnalysisOptions(portability=per_batch)).query
    assert report.portability == "unknown"


@pytest.mark.parametrize(
    "fields,suffix",
    [
        ([pa.field("y", pa.int64())], ".schema[0].name"),
        ([pa.field("x", pa.int32())], ".schema[0].data_type"),
        ([pa.field("x", pa.int64(), False)], ".schema[0].nullable"),
        ([pa.field("x", pa.int64()), pa.field("y", pa.int64())], ".schema"),
    ],
)
def test_dataframe_union_rejects_inexact_schema(fields, suffix):
    a = source("a")
    b = cf.table_input("b", schema=pa.schema(fields))
    value = cf.Program("schema", mode="batch", outputs={"out": a.union_all(b)})
    report = value.analyze()
    assert any(
        issue.code == "schema_mismatch"
        and issue.path == "outputs.out.union_all.inputs[1]" + suffix
        for issue in report.issues
    )
    with pytest.raises(cf.CompileError, match="exact schema required"):
        value.collect_schema()


def test_sql_union_coercion_stays_legacy_batch_only():
    from collections import Counter

    a = source("a")
    b = cf.table_input("b", schema=pa.schema([("x", pa.int32())]))
    value = cf.sql("SELECT x FROM a UNION ALL SELECT x FROM b", a=a, b=b)
    batch = cf.Program("coercion", mode="batch", outputs={"out": value})
    actual = batch.collect(
        {
            "a": table([1, None, 1]),
            "b": pa.table({"x": pa.array([2, 1, None], type=pa.int32())}),
        }
    )
    assert actual["out"].schema == schema()
    assert Counter(actual["out"]["x"].to_pylist()) == Counter({1: 3, None: 2, 2: 1})
    assert not any(
        node["operator"]["kind"] == "union"
        for node in batch.to_project().model_dump()["graph"]["nodes"]
    )
    report = batch.analyze().query
    assert any(
        issue.code == "schema_mismatch" and "sql.tables.b" in issue.path
        for issue in report.diagnostics
    )
    with pytest.raises(cf.CompileError, match="schema_mismatch"):
        batch.for_mode("stream").compile_stream()


def snapshot_union(kind):
    a, b = source("a"), source("b")
    total = a.sql("SELECT SUM(x) AS x FROM input")
    value = (
        total.union_all(b)
        if kind == "df"
        else cf.sql("SELECT x FROM a UNION ALL SELECT x FROM b", a=total, b=b)
    )
    return cf.Program("snapshots", mode="stream", outputs={"out": value})


@pytest.mark.parametrize("kind", ["df", "sql"])
def test_latest_snapshot_union_rejects_before_all_source_resources(kind, monkeypatch):
    import asyncio
    import importlib

    stream_module = importlib.import_module("calc_flow.stream")
    runtime = cf.Runtime()
    legacy = snapshot_union(kind)
    legacy.compile_stream(runtime)
    latest = legacy.with_update_semantics("latest_relation")
    calls = []

    def forbid(*args, **kwargs):
        calls.append("binding")
        raise AssertionError("binding acquired before gate")

    async def forbid_async(*args, **kwargs):
        calls.append("root/job")
        raise AssertionError("root or job acquired before gate")

    class Unread:
        def __aiter__(self):
            calls.append("source")
            raise AssertionError("source acquired before gate")

    monkeypatch.setattr(stream_module, "_source_binding", forbid)
    monkeypatch.setattr(stream_module.StreamResults, "_create_root", forbid_async)
    monkeypatch.setattr(cf.StreamingRunner, "start_async", forbid_async)
    edge_path = (
        "outputs.out.union_all.inputs[0]"
        if kind == "df"
        else "outputs.out.sql.tables.a"
    )
    for _ in range(2):
        with pytest.raises(
            cf.CompileError, match="update_interpretation_mismatch"
        ) as error:
            latest.compile_stream(runtime)
        edge = next(
            edge for edge in error.value.query_analysis.edges if edge.path == edge_path
        )
        assert edge.stream.producer_update == "complete_snapshot"
        assert edge.stream.consumer_interpretation == "append"
        assert edge.stream.support == "unsupported"
        with pytest.raises(cf.CompileError, match="update_interpretation_mismatch"):
            asyncio.run(enter(latest, runtime, {"a": Unread(), "b": Unread()}))
    assert calls == []
    report = legacy.analyze(
        runtime, options=cf.AnalysisOptions(portability=request())
    ).query
    assert any(
        issue.path == edge_path and issue.code == "update_interpretation_mismatch"
        for issue in report.diagnostics
    )
    latest.for_mode("batch").compile_batch(runtime)


async def enter(value, runtime, inputs):
    async with value.stream(inputs, runtime=runtime):
        raise AssertionError("stream started")


@pytest.mark.parametrize("kind", ["df", "sql"])
def test_stream_union_fifo_shared_upstream_and_silent_source(kind):
    import asyncio

    asyncio.run(stream_fifo(kind))


async def stream_fifo(kind):
    a, b = source("a"), source("b")
    value = (
        a.union_all(b)
        if kind == "df"
        else cf.sql("SELECT x FROM a UNION ALL SELECT x FROM b", a=a, b=b)
    )
    repeated = (
        a.union_all(a)
        if kind == "df"
        else cf.sql(
            "SELECT x FROM left_side UNION ALL SELECT x FROM right_side",
            left_side=a,
            right_side=a,
        )
    )
    value = value.select("x").sql("SELECT x FROM input WHERE x >= 0")
    stream = cf.Program(
        "fifo", mode="stream", outputs={"out": value, "twice": repeated, "shared": a}
    )
    calls = []

    async def ingress():
        for values in ([10, 11], [12]):
            calls.append(values)
            yield table(values)

    seen = {"out": [], "twice": [], "shared": []}
    async with stream.stream(
        {"a": ingress(), "b": [table([20]), table([21, 22])]}
    ) as events:
        async for event in events:
            seen[event.name].extend(event.table["x"].to_pylist())
    assert calls == [[10, 11], [12]]
    assert_fifo(seen)
    empty = []

    async def silent():
        for values in ():
            yield table(values)

    async with stream.stream({"a": [table([9])], "b": silent()}) as events:
        async for event in events:
            if event.name == "out":
                empty.extend(event.table["x"].to_pylist())
    assert empty == [9]


def assert_fifo(seen):
    assert [x for x in seen["out"] if x < 20] == [10, 11, 12]
    assert [x for x in seen["out"] if x >= 20] == [20, 21, 22]
    assert seen["shared"] == [10, 11, 12]
    assert sorted(seen["twice"]) == [10, 10, 11, 11, 12, 12]


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM a UNION ALL (SELECT x FROM b WHERE x > 0)",
        "SELECT a.x AS x FROM a UNION ALL SELECT b.x AS x FROM b",
    ],
)
def test_sql_union_row_local_arms(query):
    value = cf.sql(query, a=source("a"), b=source("b"))
    batch = cf.Program("arms", mode="batch", outputs={"out": value})
    assert sorted(
        batch.collect({"a": table([1]), "b": table([2])})["out"]["x"].to_pylist()
    ) == [1, 2]
    batch.for_mode("stream").compile_stream()


@pytest.mark.parametrize(
    "query",
    [
        "SELECT x FROM a UNION SELECT x FROM b",
        "SELECT DISTINCT x FROM a UNION ALL SELECT x FROM b",
        "SELECT x FROM a UNION ALL SELECT x FROM b ORDER BY x",
        "SELECT SUM(x) AS x FROM a UNION ALL SELECT x FROM b",
        "SELECT x FROM a UNION ALL SELECT x FROM a",
    ],
)
def test_wider_union_sql_keeps_legacy_scope(query):
    batch = cf.Program(
        "wide",
        mode="batch",
        outputs={"out": cf.sql(query, a=source("a"), b=source("b"))},
    )
    batch.compile_batch()
    with pytest.raises(ValueError, match="unsupported_mode"):
        batch.for_mode("stream")
    report = batch.analyze(options=cf.AnalysisOptions(portability=request())).query
    assert report.portability != "supported"


@pytest.mark.parametrize("kind", ["df", "sql"])
def test_union_project_roundtrip_and_new_cache_policy(kind):
    from calc_flow.symbolic.union import cache_policy

    runtime = cf.Runtime()
    batch = program(kind)
    document = batch.to_project(runtime)
    reloaded = cf.ProjectDocument.model_validate_json(document.model_dump_json())
    assert reloaded.model_dump_json() == document.model_dump_json()
    runtime.compile_batch_project(reloaded.model_dump_json())
    stream = batch.for_mode("stream").to_project(runtime)
    runtime._compile_stream_graph_project(stream.model_dump_json())
    assert cache_policy(batch, runtime) == (
        ("finite_union_v1",) if kind == "sql" else ()
    )


def test_single_alias_set_sql_retains_original_topology():
    a = source("a")
    value = a.sql("SELECT x FROM input UNION ALL SELECT x FROM input")
    batch = cf.Program("legacy", mode="batch", outputs={"out": value})
    kinds = [
        node["operator"]["kind"]
        for node in batch.to_project().model_dump()["graph"]["nodes"]
    ]
    assert "sql" in kinds and "union" not in kinds
    assert batch.collect({"a": table([1])})["out"]["x"].to_pylist() == [1, 1]
    batch.for_mode("stream").compile_stream()


def test_sql_union_arm_errors_are_data_only_diagnostics():
    value = cf.sql(
        "SELECT x FROM a UNION ALL SELECT missing FROM b", a=source("a"), b=source("b")
    )
    batch = cf.Program("arm-error", mode="batch", outputs={"out": value})
    report = batch.analyze()
    assert any(
        issue.code == "schema_mismatch" and issue.path == "outputs.out.sql.arms[1]"
        for issue in report.issues
    )
    with pytest.raises(cf.CompileError, match=r"sql.arms\[1\].*schema_mismatch"):
        batch.compile_batch()


@pytest.mark.parametrize("kind", ["df", "sql"])
def test_legacy_snapshot_union_keeps_seven_and_batch_six(kind):
    import asyncio

    union = dict(snapshot_union(kind).outputs)["out"]
    total = union.sql("SELECT SUM(x) AS x FROM input")
    stream = cf.Program("legacy-union", mode="stream", outputs={"out": total})
    batch = stream.for_mode("batch")
    assert batch.collect({"a": table([1, 2, 3]), "b": table([])})["out"][
        "x"
    ].to_pylist() == [6]
    assert asyncio.run(snapshot_history(stream)) == [1, 7]


async def snapshot_history(stream):
    async def silent():
        for values in ():
            yield table(values)

    result = []
    async with stream.stream(
        {"a": [table([1]), table([2, 3])], "b": silent()}
    ) as events:
        async for event in events:
            result.extend(event.table["x"].to_pylist())
    return result
