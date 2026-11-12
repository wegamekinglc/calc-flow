from __future__ import annotations

import json
from datetime import timedelta

import pyarrow as pa
import pytest

import calc_flow as cf

JOIN_QUERY = """SELECT * FROM CF_INNER_JOIN('a', 'b', '{
    "spec": {
        "join_type": "inner", "left_keys": ["k"], "right_keys": ["k"],
        "left_event_time": "ts", "right_event_time": "ts",
        "bounds": {"before_micros": 2, "after_micros": 3},
        "limits": {"max_state_rows_per_side": 100,
            "max_state_bytes_per_side": 1048576, "max_matches_per_input_batch": 1000}
    }
}')"""


ORDERED_JOIN_QUERY = """SELECT * FROM CF_INNER_JOIN('a', 'b', '{
    "spec": {
        "join_type": "inner", "left_keys": ["k"], "right_keys": ["k"],
        "left_event_time": "ts", "right_event_time": "ts",
        "bounds": {"before_micros": 2, "after_micros": 3},
        "limits": {"max_state_rows_per_side": 100,
            "max_state_bytes_per_side": 1048576, "max_matches_per_input_batch": 1000}
    },
    "output_ordering": {"entity_by": ["left__k"], "event_time": "left__ts",
        "sequence_by": ["left__seq", "right__seq"]}
}')"""


NESTED_JOIN_QUERY = """SELECT * FROM CF_INNER_JOIN('a', 'b', '{
    "spec": {
        "join_type": "inner", "left_keys": ["left__k"], "right_keys": ["k"],
        "left_event_time": "left__ts", "right_event_time": "ts",
        "bounds": {"before_micros": 0, "after_micros": 0},
        "limits": {"max_state_rows_per_side": 100,
            "max_state_bytes_per_side": 1048576, "max_matches_per_input_batch": 1000}
    }
}')"""


def source(name):
    return cf.table_input(
        name,
        schema=[
            cf.Field("k", "int64", nullable=False),
            cf.Field("ts", "timestamp[us, UTC]", nullable=False),
            cf.Field("seq", "uint64", nullable=False),
        ],
        entity_by=["k"],
        event_time="ts",
        sequence_by=["seq"],
    )


def values(times):
    schema = pa.schema(
        [
            pa.field("k", pa.int64(), False),
            pa.field("ts", pa.timestamp("us", tz="UTC"), False),
            pa.field("seq", pa.uint64(), False),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array([1] * len(times), type=pa.int64()),
            pa.array(times, type=schema.field("ts").type),
            pa.array(range(len(times)), type=pa.uint64()),
        ],
        schema=schema,
    )


def join(a, b, sql=False, ordering=False):
    order = dict(
        entity_by=["left__k"],
        event_time="left__ts",
        sequence_by=["left__seq", "right__seq"],
    )
    if sql:
        query = ORDERED_JOIN_QUERY if ordering else JOIN_QUERY
        return cf.sql(query, a=a, b=b)
    return cf.table.stream_join(
        a,
        b,
        left_keys=["k"],
        right_keys=["k"],
        left_event_time="ts",
        right_event_time="ts",
        bounds=cf.JoinTimeBounds(timedelta(microseconds=2), timedelta(microseconds=3)),
        limits=cf.JoinStateLimits(100, 1048576, 1000),
        **(
            dict(
                output_entity_by=order["entity_by"],
                output_event_time=order["event_time"],
                output_sequence_by=order["sequence_by"],
            )
            if ordering
            else {}
        ),
    )


@pytest.mark.parametrize("sql", [False, True])
def test_four_surface_join_fixture(sql):
    a, b = source("a"), source("b")
    value = join(a, b, sql)
    program = cf.Program("finite", mode="batch", outputs={"pairs": value})
    out = program.collect({"a": values([10, 10]), "b": values([7, 8, 8, 13, 14])})[
        "pairs"
    ]
    assert sorted(zip(out["left__seq"].to_pylist(), out["right__seq"].to_pylist())) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
    ]
    assert out.schema.names == [
        "left__k",
        "left__ts",
        "left__seq",
        "right__k",
        "right__ts",
        "right__seq",
    ]
    for mode in ("batch", "stream"):
        p = program.for_mode(mode)
        document = p.to_project().model_dump()
        assert (
            sum(
                n["operator"]["kind"] == "stream_join"
                for n in document["graph"]["nodes"]
            )
            == 1
        )
        assert p.analyze().query.analysis_revision == 6


def test_inner_join_conditional_eof_analysis():
    p = cf.Program(
        "proof", mode="stream", outputs={"pairs": join(source("a"), source("b"))}
    )
    request = cf.PortabilityRequest(
        comparison="final_relation",
        history="accepted_history",
        cutoff="all_sources_eof",
    )
    report = p.analyze(options=cf.AnalysisOptions(portability=request)).query
    assert report.analysis_revision == 6
    assert report.portability == "supported", report.diagnostics


@pytest.mark.parametrize("sql", [False, True])
def test_same_finite_plan_and_runtime_cache_are_isolated(sql):
    p = cf.Program(
        "reuse", mode="batch", outputs={"pairs": join(source("a"), source("b"), sql)}
    )
    runtime = cf.Runtime()
    plan = p.compile_batch(runtime)

    def execute(left, right):
        assert plan.external_inputs == ("a.input", "b.input")
        result = plan.execute(
            inputs={
                "a.input": cf.Batch.from_pyarrow(left),
                "b.input": cf.Batch.from_pyarrow(right),
            }
        )
        return next(iter(result.outputs.values())).to_pyarrow()

    for left, right, expected in [
        ([10, 10], [8, 8, 13], 6),
        ([], [8, 8, 13], 0),
        ([10], [], 0),
        ([], [], 0),
        ([100], [100], 1),
    ]:
        assert execute(values(left), values(right)).num_rows == expected
    assert p.compile_batch(runtime) is plan


@pytest.mark.parametrize("sql", [False, True])
def test_mixed_join_sharing_and_independent_branches(sql):
    a, b = source("a"), source("b")
    first = join(a.sql("SELECT * FROM input"), b, sql, ordering=True)
    post = first.select(
        k=first["left__k"], ts=first["left__ts"], seq=first["left__seq"]
    ).sql("SELECT * FROM input")
    p = cf.Program(
        "mixed",
        mode="batch",
        outputs={
            "first": first,
            "post": post,
            "outside": a.union_all(b),
            "count": a.sql("SELECT COUNT(*) AS n FROM input"),
        },
    )
    result = p.collect({"a": values([10, 10]), "b": values([8, 8, 13])})
    assert {name: table.num_rows for name, table in result.items()} == dict(
        first=6, post=6, outside=5, count=1
    )
    doc = p.to_project().model_dump()
    assert (
        sum(n["operator"]["kind"] == "stream_join" for n in doc["graph"]["nodes"]) == 1
    )
    assert sum(n["operator"]["kind"] == "union" for n in doc["graph"]["nodes"]) == 1

    reloaded = cf.Runtime().compile_batch_project(
        cf.ProjectDocument.model_validate(doc).model_dump_json()
    )
    assert reloaded.external_inputs == ("a.input", "b.input")
    restored = reloaded.execute(
        inputs={
            "a.input": cf.Batch.from_pyarrow(values([10, 10])),
            "b.input": cf.Batch.from_pyarrow(values([8, 8, 13])),
        }
    )
    for name, output in restored.outputs.items():
        expected = result[name.split(".")[0]]
        actual = output.to_pyarrow()
        assert actual.schema == expected.schema
        order = [(column, "ascending") for column in actual.column_names]
        assert actual.sort_by(order).equals(expected.sort_by(order))


@pytest.mark.parametrize("sql", [False, True])
def test_three_source_nested_join(sql):
    a, b, c = source("a"), source("b"), source("c")
    first = join(a, b, sql, ordering=True).sql("SELECT * FROM input")
    if sql:
        second = cf.sql(NESTED_JOIN_QUERY, a=first, b=c)
    else:
        second = cf.table.stream_join(
            first,
            c,
            left_keys=["left__k"],
            right_keys=["k"],
            left_event_time="left__ts",
            right_event_time="ts",
            bounds=cf.JoinTimeBounds(timedelta(0), timedelta(0)),
            limits=cf.JoinStateLimits(100, 1048576, 1000),
        )
    p = cf.Program("nested", mode="batch", outputs={"first": first, "second": second})
    out = p.collect({"a": values([10, 10]), "b": values([8, 8, 13]), "c": values([10])})
    assert out["first"].num_rows == out["second"].num_rows == 6
    doc = p.to_project().model_dump()
    assert (
        sum(n["operator"]["kind"] == "stream_join" for n in doc["graph"]["nodes"]) == 2
    )
    p.for_mode("stream").compile_stream()


@pytest.mark.parametrize("sql", [False, True])
def test_stream_terminal_multiset_and_new_job_is_fresh(sql):
    import asyncio

    async def exercise():
        p = cf.Program(
            "stream",
            mode="stream",
            outputs={"pairs": join(source("a"), source("b"), sql)},
        )
        result = []
        async with p.stream(
            {
                "a": [values([10]), values([]), values([10])],
                "b": [values([7, 8]), values([8, 13, 14])],
            },
            watermarks={name: cf.DisabledWatermarks() for name in ("a", "b")},
        ) as stream:
            async for event in stream:
                result.extend(
                    zip(
                        event.table["left__ts"].cast(pa.int64()).to_pylist(),
                        event.table["right__ts"].cast(pa.int64()).to_pylist(),
                    )
                )
        assert sorted(result) == [(10, 8)] * 4 + [(10, 13)] * 2

        async def silent():
            for table in ():
                yield table

        async with p.stream(
            {"a": silent(), "b": [values([8])]},
            watermarks={name: cf.DisabledWatermarks() for name in ("a", "b")},
        ) as stream:
            assert [event async for event in stream] == []

    asyncio.run(exercise())


@pytest.mark.parametrize("sql", [False, True])
def test_public_nullable_time_rejected(sql):
    a = cf.table_input(
        "a",
        schema=[
            cf.Field("k", "int64", nullable=False),
            cf.Field("ts", "timestamp[us, UTC]"),
            cf.Field("seq", "uint64", nullable=False),
        ],
        entity_by=["k"],
        event_time="ts",
        sequence_by=["seq"],
    )
    p = cf.Program(
        "nullable", mode="batch", outputs={"pairs": join(a, source("b"), sql)}
    )
    expected = (
        "outputs.pairs.sql.inner_join.left_event_time"
        if sql
        else "outputs.pairs.stream_join.left_event_time"
    )
    assert any(issue.path == expected for issue in p.analyze().issues)
    with pytest.raises(cf.CompileError, match="non-null"):
        p.compile_batch()


@pytest.mark.parametrize("sql", [False, True])
def test_join_time_rewrite_is_not_a_temporal_passthrough(sql):
    a = source("a").sql("SELECT k, ts + INTERVAL '1 microsecond' AS ts, seq FROM input")
    p = cf.Program(
        "rewrite", mode="batch", outputs={"pairs": join(a, source("b"), sql)}
    )
    with pytest.raises(cf.CompileError, match="unchanged|ordering_required"):
        p.compile_batch()


@pytest.mark.parametrize("fragment", [" WHERE true", " LIMIT 1", " ORDER BY left__k"])
def test_sql_join_rejects_extra_clauses_at_macro(fragment):
    a, b = source("a"), source("b")
    original = join(a, b, sql=True)
    query = original._node.attr("query").value + fragment
    p = cf.Program("shape", mode="batch", outputs={"pairs": cf.sql(query, a=a, b=b)})
    with pytest.raises(cf.CompileError, match="sql.inner_join"):
        p.compile_batch()


@pytest.mark.parametrize(
    "field,value",
    [("before_micros", -1), ("before_micros", True), ("after_micros", 0.5)],
)
def test_sql_join_configuration_errors_keep_subfield(field, value):
    original = join(source("a"), source("b"), sql=True)
    query = original._node.attr("query").value
    old = '"' + field + '": ' + ("2" if field == "before_micros" else "3")
    query = query.replace(old, '"' + field + '": ' + json.dumps(value))
    p = cf.Program(
        "invalid",
        mode="batch",
        outputs={"pairs": cf.sql(query, a=source("a"), b=source("b"))},
    )
    with pytest.raises(cf.CompileError, match="spec.bounds." + field):
        p.compile_batch()


@pytest.mark.parametrize("sql", [False, True])
def test_latest_snapshot_join_is_rejected_before_source_iteration(sql):
    import asyncio

    a, b = source("a"), source("b")
    snapshot = a.sql("SELECT k, MAX(ts) AS ts, MAX(seq) AS seq FROM input GROUP BY k")
    p = cf.Program(
        "latest", mode="stream", outputs={"pairs": join(snapshot, b, sql)}
    ).with_update_semantics("latest_relation")
    opened = []

    async def ingress():
        opened.append(True)
        yield values([10])

    async def run():
        with pytest.raises(cf.CompileError, match="update_interpretation_mismatch"):
            async with p.stream({"a": ingress(), "b": ingress()}):
                pass

    asyncio.run(run())
    assert opened == []


@pytest.mark.parametrize("sql", [False, True])
def test_public_key_domain_is_proved_before_native_compile(sql):
    def keyed(name):
        return cf.table_input(
            name,
            schema=[
                cf.Field("k", "float64", nullable=False),
                cf.Field("ts", "timestamp[us, UTC]", nullable=False),
                cf.Field("seq", "uint64", nullable=False),
            ],
            entity_by=["seq"],
            event_time="ts",
            sequence_by=["seq"],
        )

    report = cf.Program(
        "keys", mode="batch", outputs={"pairs": join(keyed("a"), keyed("b"), sql)}
    ).analyze()
    assert any(
        issue.code == "unsupported_type" and "left_keys[0]" in issue.path
        for issue in report.issues
    )
