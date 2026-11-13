from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pyarrow as pa
import pytest

import calc_flow as cf


def over_order():
    return cf.OverOrder([], [cf.SortKey("v")], sequence_by=["seq"])


def over_spec():
    return cf.OverSpec.bounded_source(
        over_order(), state_budget=cf.StateBudget(4, 8 << 20)
    )


def over_input():
    return cf.table_input(
        "events",
        schema=pa.schema(
            [
                pa.field("ts", pa.timestamp("us")),
                pa.field("v", pa.int64()),
                pa.field("seq", pa.uint64(), nullable=False),
            ]
        ),
    )


@pytest.mark.parametrize(
    "parts,keys,sequence,error,path",
    [
        ("p", [cf.SortKey("v")], ["seq"], TypeError, "partition_by"),
        ([], [cf.SortKey("v")], [], ValueError, "sequence_by"),
        ([], [cf.SortKey("v")], [False], TypeError, r"sequence_by\[0\]"),
        ([], [cf.SortKey("v")], [""], ValueError, r"sequence_by\[0\]"),
        (["p", "p"], [cf.SortKey("v")], ["seq"], ValueError, "partition_by"),
        ([], [], ["seq"], ValueError, "order_by"),
        ([], [cf.SortKey("v"), cf.SortKey("v")], ["seq"], ValueError, "order_by"),
        (["v"], [cf.SortKey("v")], ["seq"], ValueError, "disjoint"),
        ([], [cf.SortKey("v")], ["v"], ValueError, "disjoint"),
    ],
)
def test_over_order_rejects_ambiguous_or_malformed_column_roles(
    parts, keys, sequence, error, path
):
    with pytest.raises(error, match=path):
        cf.OverOrder(parts, keys, sequence_by=sequence)


@pytest.mark.parametrize(
    "factory,message",
    [
        (cf.OverSpec, "OverSpec.bounded_source or OverSpec.closed_window"),
        (cf.RowsFrame, "RowsFrame.cumulative or RowsFrame.preceding"),
        (cf.AnalyticExpr, "AnalyticExpr.row_number, lag or sum"),
    ],
)
def test_over_requires_explicit_history_frame_and_function_factories(factory, message):
    with pytest.raises(TypeError, match=message):
        factory()


@pytest.mark.parametrize(
    "order,budget,error,path",
    [
        (None, cf.StateBudget(4, 8 << 20), TypeError, "over.spec.order"),
        (over_order(), None, TypeError, "over.spec.state_budget"),
        (
            over_order(),
            cf.StateBudget(1 << 32, 8 << 20),
            ValueError,
            "over.spec.state_budget.max_rows",
        ),
    ],
)
def test_over_history_requires_typed_order_and_uint32_row_budget(
    order, budget, error, path
):
    with pytest.raises(error, match=path):
        cf.OverSpec.bounded_source(order, state_budget=budget)


def test_over_closed_window_requires_its_own_window_type():
    with pytest.raises(TypeError, match="over.spec.history requires an OverWindow"):
        cf.OverSpec.closed_window(
            cf.SortWindow("ts", start_micros=0, end_micros=10),
            over_order(),
            state_budget=cf.StateBudget(4, 8 << 20),
        )


@pytest.mark.parametrize(
    "rows,error", [(True, TypeError), (-1, ValueError), (1 << 32, ValueError)]
)
def test_over_rows_frame_rejects_boolean_negative_and_overflow_bounds(rows, error):
    with pytest.raises(error, match="over.frame.preceding"):
        cf.RowsFrame.preceding(rows)


def test_over_sum_requires_an_explicit_rows_frame():
    with pytest.raises(TypeError, match="over.expressions.frame requires a RowsFrame"):
        cf.AnalyticExpr.sum("v", frame=0, name="total")


@pytest.mark.parametrize("expressions", [None, "rn", b"rn"])
def test_over_rejects_scalar_expression_containers(expressions):
    with pytest.raises(TypeError, match="over.expressions requires a sequence"):
        over_input().over(expressions, spec=over_spec())


def test_over_requires_nonempty_distinct_analytic_output_names():
    source = over_input()
    spec = over_spec()
    with pytest.raises(ValueError, match="over.expressions requires nonempty"):
        source.over([], spec=spec)
    with pytest.raises(
        ValueError, match="over.expressions requires unique output names"
    ):
        source.over(
            [cf.AnalyticExpr.row_number("rn"), cf.AnalyticExpr.lag("v", name="rn")],
            spec=spec,
        )
    with pytest.raises(TypeError, match="over.spec requires an OverSpec"):
        source.over([cf.AnalyticExpr.row_number("rn")], spec=None)


def test_over_declaration_captures_lists_and_retains_input_and_frame_boundaries():
    parts, keys, sequence = [], [cf.SortKey("v", descending=True)], ["seq"]
    order = cf.OverOrder(parts, keys, sequence_by=sequence)
    spec = cf.OverSpec.bounded_source(order, state_budget=cf.StateBudget(4, 8 << 20))
    current = cf.RowsFrame.preceding(0)
    expressions = [
        cf.AnalyticExpr.row_number("rn"),
        cf.AnalyticExpr.lag("v", name="prev"),
        cf.AnalyticExpr.sum("v", frame=current, name="current"),
    ]
    source = over_input()
    out = source.over(expressions, spec=spec)
    assert parts == []
    assert keys == [cf.SortKey("v", descending=True)]
    assert sequence == ["seq"]
    assert len(expressions) == 3
    before = out.digest
    parts.append("missing")
    keys.clear()
    sequence[0] = "missing"
    expressions.clear()
    for value, field, replacement in (
        (order, "partition_by", ("missing",)),
        (spec, "late_policy", "drop"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, replacement)
    assert out.digest == before
    assert order.partition_by == ()
    assert order.sequence_by == ("seq",)
    assert spec.late_policy == "error"
    data = pa.Table.from_pydict(
        {"ts": [0, 1, 2, 3], "v": [2, None, 2, 1], "seq": [20, 40, 10, 30]},
        schema=pa.schema(
            [
                pa.field("ts", pa.timestamp("us")),
                pa.field("v", pa.int64()),
                pa.field("seq", pa.uint64(), nullable=False),
            ]
        ),
    )
    original = data.to_pydict()
    program = cf.Program(
        "captured", mode="batch", outputs={"out": out, "input": source}
    )
    result = program.collect({"events": data})
    assert result["out"].select(["seq", "rn", "prev", "current"]).to_pydict() == {
        "seq": [40, 10, 20, 30],
        "rn": [1, 2, 3, 4],
        "prev": [None, None, 2, 2],
        "current": [None, 2, 2, 1],
    }
    assert result["input"].to_pydict() == original
    assert data.to_pydict() == original


@pytest.mark.parametrize("syntax", ["df", "sql"])
def test_over_derived_sequence_is_unsupported_before_compilation(syntax):
    changed = over_input().sql("SELECT ts,v,CAST(seq AS BIGINT)+1 AS seq FROM input")
    out = (
        changed.over([cf.AnalyticExpr.row_number("rn")], spec=over_spec())
        if syntax == "df"
        else cf.sql(
            "SELECT *, ROW_NUMBER() OVER (ORDER BY v,seq) AS rn "
            "FROM CF_OVER_BOUNDED('a',4,8388608,ARRAY['seq'])",
            a=changed,
        )
    )
    program = cf.Program("derived", mode="batch", outputs={"out": out})
    report = program.analyze().query
    role = "over" if syntax == "df" else "sql.over"
    assert {
        (issue.path, issue.code, issue.status)
        for issue in report.diagnostics
        if "sequence_by[0]" in issue.path
    } == {
        (
            f"outputs.out.{mode}.{role}.spec.order.sequence_by[0]",
            "capability_mismatch",
            "unsupported",
        )
        for mode in ("batch", "stream")
    }
    with pytest.raises(cf.CompileError, match=r"spec.order.sequence_by\[0\]"):
        program.compile_batch()


@pytest.mark.parametrize("syntax", ["df", "sql"])
def test_over_pure_sequence_rename_and_projection_preserve_proved_total_order(syntax):
    renamed = over_input().sql("SELECT ts,v,seq AS replay FROM input")
    out = (
        renamed.over(
            [cf.AnalyticExpr.row_number("rn")],
            spec=cf.OverSpec.bounded_source(
                cf.OverOrder([], [cf.SortKey("v")], sequence_by=["replay"]),
                state_budget=cf.StateBudget(4, 8 << 20),
            ),
        )
        if syntax == "df"
        else cf.sql(
            "SELECT *, ROW_NUMBER() OVER (ORDER BY v,replay) AS rn "
            "FROM CF_OVER_BOUNDED('a',4,8388608,ARRAY['replay'])",
            a=renamed,
        )
    )
    program = cf.Program(
        "renamed", mode="batch", outputs={"out": out.select("v", "replay", "rn")}
    )
    request = cf.PortabilityRequest(
        comparison="final_relation",
        history="accepted_history",
        cutoff="all_sources_eof",
        ordering="declared",
    )
    report = program.analyze(options=cf.AnalysisOptions(portability=request)).query
    assert report.portability == "supported"
    output = next(node for node in report.nodes if node.path == "outputs.out")
    for facts in (output.batch, output.stream):
        assert facts.support == "supported"
        assert facts.ordering.kind == "ordered"
        assert facts.ordering.ties == "total"
        assert [
            (key.field, key.direction, key.nulls) for key in facts.ordering.keys
        ] == [
            ("v", "ascending", "last"),
            ("replay", "ascending", "last"),
        ]
    program.compile_batch(portability=request)
    program.for_mode("stream").compile_stream(portability=request)
    data = pa.table(
        {
            "ts": pa.array([1, 2, 3], pa.timestamp("us")),
            "v": pa.array([2, 1, 2], pa.int64()),
            "seq": pa.array([30, 20, 10], pa.uint64()),
        },
        schema=pa.schema(
            [
                pa.field("ts", pa.timestamp("us")),
                pa.field("v", pa.int64()),
                pa.field("seq", pa.uint64(), nullable=False),
            ]
        ),
    )
    assert program.collect({"events": data}, portability=request)[
        "out"
    ].to_pydict() == {
        "v": [1, 2, 2],
        "replay": [20, 10, 30],
        "rn": [1, 2, 3],
    }


def test_over_native_schema_errors_keep_analytic_field_path_and_code():
    out = over_input().over(
        [cf.AnalyticExpr.lag("missing", name="prev")], spec=over_spec()
    )
    program = cf.Program("missing", mode="batch", outputs={"out": out})
    report = program.analyze().query
    assert {(issue.path, issue.code, issue.status) for issue in report.diagnostics} == {
        (
            f"outputs.out.{mode}.over.expressions[0].column",
            "unresolved_type",
            "unsupported",
        )
        for mode in ("batch", "stream")
    }
    with pytest.raises(
        cf.CompileError, match=r"over.expressions\[0\].column: unresolved_type"
    ):
        program.compile_batch()


@pytest.mark.parametrize("offered", ["absent", "layout1"])
def test_over_drop_requires_the_advertised_policy_layout(monkeypatch, offered):
    runtime = cf.Runtime()
    base = runtime.capabilities()
    operators = tuple(o for o in base.operators if o.kind != "bounded_over")
    if offered == "layout1":
        over = next(o for o in base.operators if o.kind == "bounded_over")
        operators += (replace(over, state_layouts=(1,)),)
    monkeypatch.setattr(
        cf.Runtime, "capabilities", lambda _self: replace(base, operators=operators)
    )
    out = over_input().over(
        [cf.AnalyticExpr.row_number("rn")],
        spec=cf.OverSpec.closed_window(
            cf.OverWindow("ts", start_micros=0, end_micros=10),
            over_order(),
            state_budget=cf.StateBudget(4, 8 << 20),
            late_policy="drop",
        ),
    )
    program = cf.Program("capability", mode="stream", outputs={"out": out})
    with pytest.raises(
        cf.CompileError, match="requires bounded_over@1 and the selected policy layout"
    ):
        program.compile_stream(runtime)
    if offered == "layout1":
        error = over_input().over([cf.AnalyticExpr.row_number("rn")], spec=over_spec())
        cf.Program("error", mode="stream", outputs={"out": error}).compile_stream(
            runtime
        )


def test_over_sql_drop_explain_export_and_cache_preserve_local_policy():
    source = over_input()
    out = cf.sql(
        "SELECT *, ROW_NUMBER() OVER (ORDER BY v,seq) AS rn "
        "FROM CF_OVER_WINDOW('a','ts',0,10,4,8388608,ARRAY['seq'],'drop')",
        a=source,
    )
    text = out.explain()
    assert "complete accepted H" in text
    assert "local OVER late_policy=drop layout=2" in text
    assert "local OVER" not in cf.sql("SELECT * FROM a", a=source).explain()
    assert (
        "local OVER"
        not in cf.sql(
            "SELECT *, RANK() OVER (ORDER BY v,seq) AS rn "
            "FROM CF_OVER_BOUNDED('a',4,8388608,ARRAY['seq'])",
            a=source,
        ).explain()
    )
    program = cf.Program("local", mode="stream", outputs={"out": out})
    runtime = cf.Runtime()
    plan = program.compile_stream(runtime)
    assert program.compile_stream(runtime).fingerprint == plan.fingerprint
    nodes = program.to_project(runtime).model_dump()["graph"]["nodes"]
    (analytic,) = [
        node["operator"] for node in nodes if node["operator"]["kind"] == "bounded_over"
    ]
    assert analytic["spec"]["late_policy"] == {"kind": "drop", "metrics_version": 1}
    for option in ({"late_policy": "drop"}, {"allowed_lateness_micros": 1}):
        with pytest.raises(
            cf.CompileError, match="select the local policy in OverSpec.closed_window"
        ):
            program.compile_stream(runtime, **option)
    assert program.compile_stream(runtime).fingerprint == plan.fingerprint
