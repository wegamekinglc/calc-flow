from __future__ import annotations

import json
from datetime import timedelta

import pyarrow as pa
import pytest

import calc_flow as cf


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


def values(times, sequences=None):
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
            pa.array(
                range(len(times)) if sequences is None else sequences, type=pa.uint64()
            ),
        ],
        schema=schema,
    )


def query():
    side = dict(keys=["k"], event_time="ts", sequence_by=["seq"])
    spec = dict(
        left=dict(side, prefix="left"),
        right=dict(side, prefix="right"),
        tolerance_micros=2,
        limits=dict(max_state_rows=100, max_state_bytes=1048576),
        late_policy="error",
    )
    literal = json.dumps(dict(spec=spec)).replace("'", "''")
    return f"SELECT * FROM CF_ASOF_JOIN('a', 'b', '{literal}')"


def joined(a, b, sql=False):
    if sql:
        return cf.sql(query(), a=a, b=b)
    return a.stream_asof_join(
        b,
        tolerance=timedelta(microseconds=2),
        limits=cf.AsofStateLimits(100, 1048576),
    )


@pytest.mark.parametrize("sql", [False, True])
def test_finite_asof_four_cells(sql):
    relation = joined(source("a"), source("b"), sql)
    result = relation.collect(
        {"a": values([10, 10]), "b": values([8, 9, 9, 11], [1, 2, 10, 1])}
    )
    assert result["right__seq"].to_pylist() == [10, 10]
    assert result["left__seq"].to_pylist() == [0, 1]
    assert all(field.nullable for field in list(result.schema)[3:])


def test_asof_conditional_final_relation_proof_revision():
    p = cf.Program(
        "asof", mode="stream", outputs={"matched": joined(source("a"), source("b"))}
    )
    request = cf.PortabilityRequest(
        comparison="final_relation",
        history="accepted_history",
        cutoff="all_sources_eof",
    )
    report = p.analyze(options=cf.AnalysisOptions(portability=request)).query
    assert report.analysis_revision == 7
    assert report.portability == "supported", report.diagnostics
    result = next(node for node in report.nodes if node.path == "outputs.matched")
    assert result.stream.finality == "group_final"
    assert "snapshot" not in result.stream.reason


@pytest.mark.parametrize("sql", [False, True])
def test_same_finite_asof_plan_empty_inputs_and_failure_retry(sql):
    p = cf.Program(
        "reuse", mode="batch", outputs={"match": joined(source("a"), source("b"), sql)}
    )
    runtime = cf.Runtime()
    plan = p.compile_batch(runtime)

    def execute(left, right):
        result = plan.execute(
            inputs={
                "a.input": cf.Batch.from_pyarrow(left),
                "b.input": cf.Batch.from_pyarrow(right),
            }
        )
        return next(iter(result.outputs.values())).to_pyarrow()

    for left, right, expected in [
        ([10], [8], [0]),
        ([10], [], [None]),
        ([], [8], []),
        ([], [], []),
        ([100], [100], [0]),
    ]:
        assert (
            execute(values(left), values(right))["right__seq"].to_pylist() == expected
        )
    duplicate = pa.concat_tables([values([8]), values([8])])
    with pytest.raises(cf.ExecutionError, match="duplicate"):
        execute(values([10]), duplicate)
    assert execute(values([10]), values([9], [10]))["right__seq"].to_pylist() == [10]
    assert p.compile_batch(runtime) is plan


@pytest.mark.parametrize("sql", [False, True])
def test_asof_mixed_sharing_nested_export(sql):
    a, b, c = source("a"), source("b"), source("c")
    first = joined(a.sql("SELECT * FROM input WHERE k = 1"), b, sql)
    post = first.select(
        k=first["left__k"], ts=first["left__ts"], seq=first["left__seq"]
    ).sql("SELECT * FROM input")
    second = joined(post, c, sql)
    p = cf.Program(
        "mixed",
        mode="batch",
        outputs={
            "first": first,
            "post": post,
            "second": second,
            "outside": a.union_all(b),
            "count": a.sql("SELECT COUNT(*) AS n FROM input"),
        },
    )
    inputs = {"a": values([10, 12]), "b": values([8, 11]), "c": values([9, 11])}
    result = p.collect(inputs)
    assert result["first"]["right__seq"].to_pylist() == [0, 1]
    assert result["second"]["right__seq"].to_pylist() == [0, 1]
    assert result["outside"].num_rows == 4
    assert result["count"]["n"].to_pylist() == [2]
    doc = p.to_project().model_dump()
    assert (
        sum(
            node["operator"]["kind"] == "stream_asof_join"
            for node in doc["graph"]["nodes"]
        )
        == 2
    )
    plan = cf.Runtime().compile_batch_project(
        cf.ProjectDocument.model_validate(doc).model_dump_json()
    )
    out = plan.execute(
        inputs={
            f"{name}.input": cf.Batch.from_pyarrow(value)
            for name, value in inputs.items()
        }
    )
    for name, value in out.outputs.items():
        assert value.to_pyarrow().equals(result[name.split(".")[0]])
    p.for_mode("stream").compile_stream()


@pytest.mark.parametrize("sql", [False, True])
def test_asof_real_stream_terminal_matches_independent_oracle(sql):
    import asyncio

    relation = joined(source("a"), source("b"), sql)

    async def feed(parts):
        for part in parts:
            yield part

    async def run():
        async with relation.stream(
            {
                "a": feed([values([10], [1]), values([10], [2])]),
                "b": feed([values([8, 9, 9, 11], [1, 2, 10, 1])]),
            }
        ) as stream:
            return pa.concat_tables([table async for table in stream])

    result = asyncio.run(run())
    assert result["right__seq"].to_pylist() == [10, 10]
    assert result["left__seq"].to_pylist() == [1, 2]


def test_strict_rejects_pre_r7_report_before_cache(monkeypatch):
    from dataclasses import replace

    p = cf.Program("strict", mode="batch", outputs={"rows": source("a")})
    runtime = cf.Runtime()
    report = p.analyze(runtime)
    stale = replace(
        report,
        query=replace(report.query, analysis_revision=6, portability="supported"),
    )
    monkeypatch.setattr(cf.Program, "analyze", lambda *args, **kwargs: stale)
    request = cf.PortabilityRequest(
        comparison="final_relation",
        history="accepted_history",
        cutoff="all_sources_eof",
    )
    with pytest.raises(cf.CompileError, match="matching.*analysis"):
        p.compile_batch(runtime, portability=request)


@pytest.mark.parametrize(
    "field", ["tolerance_micros", "limits.max_state_rows", "limits.max_state_bytes"]
)
@pytest.mark.parametrize("bad", [True, 1.5, -1, 9007199254740992])
def test_sql_asof_raw_numeric_paths(field, bad):
    from calc_flow import _native

    description = json.loads(_native._asof_join_query(query(), ["a", "b"]))
    spec = description["spec"]
    target = spec
    path = field.split(".")
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = bad
    config = json.dumps(dict(spec=spec))
    report = json.loads(
        _native._asof_join_query(
            f"SELECT * FROM CF_ASOF_JOIN('a', 'b', '{config}')", ["a", "b"]
        )
    )
    assert report["error"]["path"] == f"sql.asof_join.spec.{field}"
    assert report["error"]["code"] in {"invalid_type", "out_of_range"}


@pytest.mark.parametrize(
    "suffix", [" WHERE true", " LIMIT 1", " ORDER BY left__seq", " AS pairs"]
)
def test_sql_asof_unsupported_shape_does_not_fallback(suffix):
    p = cf.Program(
        "shape",
        mode="batch",
        outputs={"rows": cf.sql(query() + suffix, a=source("a"), b=source("b"))},
    )
    with pytest.raises(cf.CompileError, match="sql.asof_join"):
        p.compile_batch()


def test_asof_nullable_payload_empty_schema_and_self_ingress():
    schema = values([]).schema.append(pa.field("payload", pa.int64(), True))
    a, b = [
        cf.table_input(
            name, schema=schema, entity_by=["k"], event_time="ts", sequence_by=["seq"]
        )
        for name in ("a", "b")
    ]
    left = values([10]).append_column("payload", pa.array([None], type=pa.int64()))
    right = values([8]).append_column("payload", pa.array([None], type=pa.int64()))
    relation = joined(a, b)
    out = relation.collect({"a": left, "b": right})
    assert out["left__payload"].to_pylist() == [None]
    assert out["right__payload"].to_pylist() == [None]
    assert out["right__ts"].null_count == 0
    empty = relation.collect({"a": left.slice(0, 0), "b": right})
    assert empty.num_rows == 0 and empty.schema == out.schema
    shared = joined(source("a"), source("a"))
    assert shared.collect({"a": values([10])})["right__seq"].to_pylist() == [0]


@pytest.mark.parametrize("sql", [False, True])
@pytest.mark.parametrize("side", [0, 1])
def test_asof_latest_snapshot_preflight_reads_no_inputs(sql, side):
    import asyncio

    a, b = source("a"), source("b")
    inputs = [a, b]
    inputs[side] = inputs[side].sql(
        "SELECT k, MAX(ts) AS ts, MAX(seq) AS seq FROM input GROUP BY k"
    )
    if sql:
        relation = joined(*inputs, sql=True)
    else:
        from calc_flow.symbolic.expr import TableExpr
        from calc_flow.symbolic.nodes import build

        original = joined(a, b)._node
        relation = TableExpr(
            build(
                "stream_asof_join",
                tuple(value._node for value in inputs),
                dict(original.attrs.entries),
            )
        )
    p = cf.Program(
        "latest", mode="stream", outputs={"match": relation}
    ).with_update_semantics("latest_relation")
    opened = []

    async def feed():
        opened.append(True)
        yield values([10])

    async def run():
        with pytest.raises(cf.CompileError, match="update_interpretation_mismatch"):
            async with p.stream({"a": feed(), "b": feed()}):
                pass

    asyncio.run(run())
    assert opened == []


@pytest.mark.parametrize(
    "kind,sequences,expected",
    [(pa.int64(), [-1, 2, 10], 10), (pa.string(), ["a", "é", "中"], "中")],
)
def test_asof_finite_ties_use_each_sides_typed_sequence(kind, sequences, expected):
    schema = pa.schema(
        [
            pa.field("k", pa.int64(), False),
            pa.field("ts", pa.timestamp("us", tz="UTC"), False),
            pa.field("seq", kind, False),
        ]
    )
    right = cf.table_input(
        "b", schema=schema, entity_by=["k"], event_time="ts", sequence_by=["seq"]
    )
    data = pa.Table.from_arrays(
        [
            pa.array([1, 1, 1]),
            pa.array([9, 9, 9], type=schema.field("ts").type),
            pa.array(sequences, type=kind),
        ],
        schema=schema,
    )
    result = joined(source("a"), right).collect({"a": values([10]), "b": data})
    assert result["right__seq"].to_pylist() == [expected]


def test_asof_finite_composite_tie_and_nonnull_identity_contract():
    schema = pa.schema(
        [
            pa.field("k", pa.int64(), False),
            pa.field("ts", pa.timestamp("us", tz="UTC"), False),
            pa.field("seq", pa.int64(), False),
            pa.field("tag", pa.string(), False),
        ]
    )
    right = cf.table_input(
        "b", schema=schema, entity_by=["k"], event_time="ts", sequence_by=["seq", "tag"]
    )
    data = pa.Table.from_arrays(
        [
            pa.array([1] * 3),
            pa.array([9] * 3, type=schema.field("ts").type),
            pa.array([-1, 2, 2]),
            pa.array(["中", "é", "中"]),
        ],
        schema=schema,
    )
    result = joined(source("a"), right).collect({"a": values([10]), "b": data})
    assert result["right__seq"].to_pylist() == [2]
    assert result["right__tag"].to_pylist() == ["中"]
    for name in ("k", "ts", "seq"):
        nullable = values([]).schema
        index = nullable.get_field_index(name)
        nullable = nullable.set(index, nullable.field(index).with_nullable(True))
        bad = cf.table_input(
            "a", schema=nullable, entity_by=["k"], event_time="ts", sequence_by=["seq"]
        )
        with pytest.raises(cf.CompileError, match="non-null"):
            joined(bad, source("b")).collect({"a": values([10]), "b": values([9])})


def test_asof_sql_missing_alias_has_alias_path():
    from calc_flow import _native

    report = json.loads(_native._asof_join_query(query(), ["a", "unused"]))
    assert report["error"]["path"] == "sql.asof_join.tables.b"


def test_legacy_asof_declaration_and_native_topology_identity():
    import hashlib

    a, b = source("a"), source("b")
    relation = joined(a, b)
    p = cf.Program(
        "identity",
        mode="stream",
        outputs={"one": relation, "two": relation.select("left__k")},
    )
    assert (
        relation.digest
        == "662069e2734627ef0c359c58c6c3512f8652c5a6ee9039b3539a44e6bcc5b2ca"
    )
    assert (
        p.fingerprint
        == "cb3fdec2d27688912555f8318d542ef27d741973a19e593888d7e54dc653fc68"
    )
    document = json.dumps(p.to_project().model_dump(mode="json"), sort_keys=True)
    assert (
        hashlib.sha256(document.encode()).hexdigest()
        == "3819fa0efc674ea82b4f523843465b3e3f154131c1e78bba2605672efeda8099"
    )


def test_null_identity_is_rejected_by_arrow_binding_before_execution():
    p = cf.Program(
        "invalid_rows",
        mode="batch",
        outputs={"match": joined(source("a"), source("b"))},
    )
    plan = p.compile_batch()
    key_null = values([10]).set_column(
        0, values([]).schema.field("k"), pa.array([None], type=pa.int64())
    )
    for bad in [key_null, values([None]), values([10], [None])]:
        with pytest.raises(TypeError, match="non-nullable.*null values"):
            plan.execute(
                inputs={
                    "a.input": cf.Batch.from_pyarrow(bad),
                    "b.input": cf.Batch.from_pyarrow(values([9])),
                }
            )
    result = plan.execute(
        inputs={
            "a.input": cf.Batch.from_pyarrow(values([10])),
            "b.input": cf.Batch.from_pyarrow(values([])),
        }
    )
    assert next(iter(result.outputs.values())).to_pyarrow()[
        "right__seq"
    ].to_pylist() == [None]


def test_finite_inner_and_asof_compose_in_both_directions():
    a, b, c = source("a"), source("b"), source("c")

    def inner(left, right, key, time, ordering=False):
        return cf.table.stream_join(
            left,
            right,
            left_keys=[key],
            right_keys=["k"],
            left_event_time=time,
            right_event_time="ts",
            bounds=cf.JoinTimeBounds(timedelta(microseconds=2), timedelta(0)),
            limits=cf.JoinStateLimits(100, 1048576, 100),
            **(
                dict(
                    output_entity_by=["left__k"],
                    output_event_time="left__ts",
                    output_sequence_by=["left__seq", "right__seq"],
                )
                if ordering
                else {}
            ),
        )

    asof_first = joined(a, b)
    inner_first = inner(a, b, "k", "ts", True)
    asof_after = inner_first.stream_asof_join(
        c, tolerance=timedelta(microseconds=2), limits=cf.AsofStateLimits(100, 1048576)
    )
    p = cf.Program(
        "mixed_kernels",
        mode="batch",
        outputs={
            "asof_inner": inner(asof_first, c, "left__k", "left__ts"),
            "inner_asof": asof_after,
        },
    )
    out = p.collect(
        {"a": values([10], [1]), "b": values([9], [2]), "c": values([8], [3])}
    )
    assert {name: table["right__seq"].to_pylist() for name, table in out.items()} == {
        "asof_inner": [3],
        "inner_asof": [3],
    }
    bad = asof_first.stream_asof_join(
        c,
        keys=(["right__k"], ["k"]),
        tolerance=timedelta(microseconds=2),
        limits=cf.AsofStateLimits(100, 1048576),
    )
    with pytest.raises(cf.CompileError, match="non-null"):
        bad.collect({"a": values([10]), "b": values([9]), "c": values([8])})
