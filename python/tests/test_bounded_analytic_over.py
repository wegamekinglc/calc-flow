from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import calc_flow as cf


@pytest.mark.parametrize(
    "query,suffix",
    [
        (
            "SELECT *, RANK() OVER (ORDER BY v,seq) AS rn "
            "FROM CF_OVER_BOUNDED('a',10,8388608,ARRAY['seq'])",
            ".expressions[0]",
        ),
        (
            "SELECT *, SUM(v) OVER (ORDER BY v,seq) AS total "
            "FROM CF_OVER_BOUNDED('a',10,8388608,ARRAY['seq'])",
            ".expressions[0].frame",
        ),
        (
            "SELECT *, ROW_NUMBER() OVER (ORDER BY v,seq) AS rn, "
            "LAG(v,1) OVER (ORDER BY v DESC,seq) AS prev "
            "FROM CF_OVER_BOUNDED('a',10,8388608,ARRAY['seq'])",
            ".expressions[1].order",
        ),
    ],
    ids=["rank", "missing_rows", "mixed_order"],
)
def test_over_known_rejections_have_unsupported_public_analysis(
    query, suffix, monkeypatch
):
    import pyarrow as pa

    schema = pa.schema(
        [pa.field("v", pa.int64()), pa.field("seq", pa.uint64(), nullable=False)]
    )
    source = cf.table_input("a", schema=schema)
    program = cf.Program(
        "rejected", mode="batch", outputs={"out": cf.sql(query, a=source)}
    )
    report = program.analyze().query
    diagnostics = [issue for issue in report.diagnostics if ".sql.over" in issue.path]
    assert {issue.path for issue in diagnostics} == {
        f"outputs.out.batch.sql.over{suffix}",
        f"outputs.out.stream.sql.over{suffix}",
    }
    assert {issue.code for issue in diagnostics} == {"capability_mismatch"}
    assert {issue.status for issue in diagnostics} == {"unsupported"}
    output = next(node for node in report.nodes if node.path == "outputs.out")
    assert (output.batch.support, output.stream.support) == (
        "unsupported",
        "unsupported",
    )
    _assert_over_rejected_before_rows(program, monkeypatch)


def _assert_over_rejected_before_rows(program, monkeypatch):
    import asyncio
    import importlib

    compute = importlib.import_module("calc_flow.compute")

    streaming = program.for_mode("stream")
    for compile_plan in (program.compile_batch, streaming.compile_stream):
        with pytest.raises(cf.CompileError):
            compile_plan()

    reads = []

    def forbid_rows(*args, **kwargs):
        reads.append("batch")
        raise AssertionError("input wrapped before strict rejection")

    class Unread:
        def __aiter__(self):
            reads.append("stream")
            raise AssertionError("source iterated before strict rejection")

    request = cf.PortabilityRequest(
        comparison="final_relation",
        history="accepted_history",
        cutoff="all_sources_eof",
    )
    monkeypatch.setattr(compute, "_collect_batches", forbid_rows)
    with pytest.raises(cf.CompileError):
        program.collect({"a": object()}, portability=request)

    async def run():
        with pytest.raises(cf.CompileError):
            async with streaming.stream({"a": Unread()}, portability=request):
                raise AssertionError("strict stream started")

    asyncio.run(run())
    assert reads == []


def test_example_rejects_incorrect_batch_even_with_optimized_python():
    script = """
import asyncio
import runpy
import sys
from unittest.mock import patch
import pyarrow as pa
import calc_flow as cf

example = runpy.run_path(sys.argv[1])
collect = cf.Program.collect_async
async def incorrect_batch(self, *args, **kwargs):
    outputs = await collect(self, *args, **kwargs)
    table = outputs['out']
    return {'out': table.set_column(table.schema.get_field_index('total'),
                                   'total', pa.array([-1, -1, -1]))}
with patch.object(cf.Program, 'collect_async', incorrect_batch):
    asyncio.run(example['main']())
"""
    example = Path(__file__).resolve().parents[2] / "examples/bounded_analytic_over.py"
    result = subprocess.run(
        ["python", "-O", "-c", script, str(example)],
        executable=sys.executable,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "RuntimeError: batch analytic result differs" in result.stderr


def test_over_declaration_is_immutable_and_strict():
    parts = ["p"]
    order = cf.OverOrder(parts, [cf.SortKey("score")], sequence_by=["seq"])
    parts.append("other")
    assert order.partition_by == ("p",)
    spec = cf.OverSpec.bounded_source(order, state_budget=cf.StateBudget(9, 8 << 20))
    assert spec.order == order
    with pytest.raises(TypeError):
        cf.RowsFrame.preceding(True)
    with pytest.raises(ValueError):
        cf.RowsFrame.preceding(-1)
    with pytest.raises(ValueError):
        cf.OverOrder(["score"], [cf.SortKey("score")], sequence_by=["seq"])
    with pytest.raises(ValueError):
        cf.AnalyticExpr.lag("v", name="")


def test_over_appends_columns_and_collects_checked_rows():
    import pyarrow as pa

    schema = pa.schema(
        [pa.field("v", pa.int64()), pa.field("seq", pa.uint64(), nullable=False)]
    )
    events = cf.table_input("events", schema=schema)
    spec = cf.OverSpec.bounded_source(
        cf.OverOrder([], [cf.SortKey("v")], sequence_by=["seq"]),
        state_budget=cf.StateBudget(9, 8 << 20),
    )
    out = events.over(
        [
            cf.AnalyticExpr.row_number("rn"),
            cf.AnalyticExpr.lag("v", name="prev"),
            cf.AnalyticExpr.sum("v", frame=cf.RowsFrame.cumulative(), name="total"),
        ],
        spec=spec,
    )
    program = cf.Program("over", mode="batch", outputs={"out": out})
    data = pa.Table.from_pydict({"v": [3, 1, 2], "seq": [30, 10, 20]}, schema)
    result = program.collect({"events": data})["out"]
    assert result.to_pydict() == dict(
        v=[1, 2, 3], seq=[10, 20, 30], rn=[1, 2, 3], prev=[None, 1, 2], total=[1, 3, 6]
    )


@pytest.mark.parametrize("factory", [cf.AnalyticExpr.lag, cf.AnalyticExpr.sum])
def test_over_operand_type_is_rejected_at_construction(factory):
    options = {"name": "out"}
    if factory == cf.AnalyticExpr.sum:
        options["frame"] = cf.RowsFrame.cumulative()
    with pytest.raises(TypeError, match="column"):
        factory(None, **options)


def nine_fixture():
    import pyarrow as pa

    schema = pa.schema(
        [
            pa.field("p", pa.string()),
            pa.field("ts", pa.timestamp("us")),
            pa.field("score", pa.int64()),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("v", pa.int64()),
        ]
    )
    data = pa.Table.from_pydict(
        dict(
            p=["A", "A", "B", "A", "A", "B", "C", None, None, "A", "A"],
            ts=[0, 1, 9, 9, 8, 2, 5, 3, 4, 10, None],
            score=[2, 1, None, 1, 3, 0, 0, 1, 1, 0, 0],
            seq=[12, 11, 22, 10, 13, 21, 40, 30, 31, 99, 100],
            v=[None, 2, None, 3, 5, 7, None, 4, None, 100, 100],
        ),
        schema,
    )
    return schema, data


def analytic_features(events, syntax):
    if syntax == "sql":
        return cf.sql(
            "SELECT *, ROW_NUMBER() OVER (PARTITION BY p ORDER BY score,seq) AS rn, "
            "LAG(v,1) OVER (PARTITION BY p ORDER BY score,seq) AS prev, "
            "SUM(v) OVER (PARTITION BY p ORDER BY score,seq "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total, "
            "SUM(v) OVER (PARTITION BY p ORDER BY score,seq "
            "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS last_two "
            "FROM CF_OVER_WINDOW('a','ts',0,10,9,8388608,ARRAY['seq'])",
            a=events,
        )
    return events.over(
        [
            cf.AnalyticExpr.row_number("rn"),
            cf.AnalyticExpr.lag("v", name="prev"),
            cf.AnalyticExpr.sum("v", frame=cf.RowsFrame.cumulative(), name="total"),
            cf.AnalyticExpr.sum("v", frame=cf.RowsFrame.preceding(1), name="last_two"),
        ],
        spec=cf.OverSpec.closed_window(
            cf.OverWindow("ts", start_micros=0, end_micros=10),
            cf.OverOrder(["p"], [cf.SortKey("score")], sequence_by=["seq"]),
            state_budget=cf.StateBudget(9, 8 << 20),
        ),
    )


def assert_nine(table):
    expected = dict(
        seq=[10, 11, 12, 13, 21, 22, 40, 30, 31],
        rn=[1, 2, 3, 4, 1, 2, 1, 1, 2],
        prev=[None, 3, 2, None, None, 7, None, None, 4],
        total=[3, 5, 5, 10, 7, 7, None, 4, 4],
        last_two=[3, 5, 2, 5, 7, 7, None, 4, 4],
    )
    assert table.select(list(expected)).to_pydict() == expected
    assert table.schema.field("rn").nullable is False


@pytest.mark.parametrize("syntax", ["df", "sql"])
def test_over_nine_public_frontends_and_real_native_watermarks(syntax):
    import asyncio
    from datetime import datetime, timedelta, timezone

    schema, data = nine_fixture()
    events = cf.table_input("events", schema=schema)
    out = analytic_features(events, syntax)
    program = cf.Program("nine", mode="batch", outputs={"out": out})
    assert_nine(program.collect({"events": data})["out"])
    assert program.analyze().query.analysis_revision == 11
    nodes = program.to_project().model_dump()["graph"]["nodes"]
    assert sum(n["operator"]["kind"] == "bounded_over" for n in nodes) == 1
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    empty = data.slice(0, 0)
    records = [
        cf.Data(cf.Batch.from_pyarrow(data.slice(0, 3)), cf.Cursor(b"1", {})),
        cf.Data(cf.Batch.from_pyarrow(empty), cf.Cursor(b"2", {})),
        cf.Watermark(epoch + timedelta(microseconds=9)),
        cf.Data(cf.Batch.from_pyarrow(data.slice(3)), cf.Cursor(b"3", {})),
        cf.Watermark(epoch + timedelta(microseconds=10)),
    ]

    class Source:
        def __init__(self):
            self.records = iter(records)

        def capabilities(self):
            return cf.SourceCapabilities(
                cf.ReplayPositioning.UNSUPPORTED,
                cf.SourceDeliveryCapability.LOSSLESS,
                11,
                8192,
                schema,
                cf.NativeWatermarkCapability.EMITS_NATIVE,
                cf.SourceBoundedness.UNBOUNDED,
            )

        async def open(self, cursor):
            assert cursor is None

        async def next(self):
            return next(self.records, None)

        async def close(self):
            pass

    async def run():
        finals = []
        async with program.for_mode("stream").stream(
            {"events": cf.SourceBinding(Source())},
            config=cf.StreamRuntimeConfig(checkpointing=False),
        ) as results:
            async for event in results:
                finals.append(event.table)
        assert len(finals) == 1
        assert_nine(finals[0])

    asyncio.run(run())


def test_over_unknown_source_is_rejected_before_iterator_and_root(monkeypatch):
    import asyncio
    import importlib

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
            touched.append(True)
            raise AssertionError("iterator acquired before source proof")

    schema, _ = nine_fixture()
    events = cf.table_input("events", schema=schema)
    order = cf.OverOrder(["p"], [cf.SortKey("score")], sequence_by=["seq"])
    out = events.over(
        [cf.AnalyticExpr.row_number("rn")],
        spec=cf.OverSpec.bounded_source(
            order,
            state_budget=cf.StateBudget(9, 8 << 20),
        ),
    )
    program = cf.Program("unknown", mode="stream", outputs={"out": out})

    async def run():
        with pytest.raises(
            cf.StreamingRuntimeError, match="source_boundedness_unproved"
        ):
            async with program.stream({"events": Unknown()}):
                pass

    asyncio.run(run())
    assert touched == []
    assert roots == []


def test_over_mixed_dag_fanout_reuses_one_owner_and_batch_state():
    schema, data = nine_fixture()
    events = cf.table_input("events", schema=schema)
    renamed = events.sql("SELECT p,ts,score,seq AS replay,v FROM input")
    out = renamed.over(
        [
            cf.AnalyticExpr.row_number("rn"),
            cf.AnalyticExpr.sum("v", frame=cf.RowsFrame.cumulative(), name="total"),
        ],
        spec=cf.OverSpec.closed_window(
            cf.OverWindow("ts", start_micros=0, end_micros=10),
            cf.OverOrder(["p"], [cf.SortKey("score")], sequence_by=["replay"]),
            state_budget=cf.StateBudget(9, 8 << 20),
        ),
    )
    projected = out.select("p", "score", "replay", "rn", "total")
    program = cf.Program("mixed", mode="batch", outputs={"all": out, "post": projected})
    runtime = cf.Runtime()
    plan = program.compile_batch(runtime)
    assert program.compile_batch(runtime) is plan
    document = program.to_project(runtime).model_dump()
    assert (
        sum(n["operator"]["kind"] == "bounded_over" for n in document["graph"]["nodes"])
        == 1
    )
    for table, expected_rows in [
        (data, 9),
        (data.slice(1, 1), 1),
        (data.slice(0, 0), 0),
    ]:
        result = program.collect({"events": table}, runtime=runtime)["all"]
        assert result.num_rows == expected_rows
        if expected_rows == 1:
            assert result["rn"].to_pylist() == [1]
            assert result["total"].to_pylist() == [2]


def test_over_expression_explain_states_history_checked_sum_and_owned_peak():
    schema, _ = nine_fixture()
    events = cf.table_input("events", schema=schema)
    text = analytic_features(events, "df").explain()
    assert "complete accepted H" in text
    assert "checked integer SUM" in text
    assert "owned participating peak" in text


def test_over_bounded_final_keeps_plain_sql_over_per_batch():
    import asyncio

    import pyarrow as pa

    schema = pa.schema(
        [pa.field("v", pa.int64()), pa.field("seq", pa.uint64(), nullable=False)]
    )
    events = cf.table_input("events", schema=schema)
    bounded = events.over(
        [
            cf.AnalyticExpr.row_number("rn"),
            cf.AnalyticExpr.sum("v", frame=cf.RowsFrame.cumulative(), name="total"),
        ],
        spec=cf.OverSpec.bounded_source(
            cf.OverOrder([], [cf.SortKey("v")], sequence_by=["seq"]),
            state_budget=cf.StateBudget(3, 8 << 20),
        ),
    )
    legacy = cf.sql("SELECT v,ROW_NUMBER() OVER (ORDER BY v) AS rn FROM a", a=events)
    program = cf.Program(
        "legacy", mode="stream", outputs={"bounded": bounded, "legacy": legacy}
    )
    data = pa.Table.from_pydict(dict(v=[3, 1, 2], seq=[30, 10, 20]), schema)

    async def run():
        source = cf.SourceBinding.from_arrow(
            data, budget=cf.EdgeBudget(max_rows=2, max_bytes=4096)
        )
        collected = {"bounded": [], "legacy": []}
        async with program.stream(
            {"events": source}, config=cf.StreamRuntimeConfig(checkpointing=False)
        ) as results:
            async for event in results:
                collected[event.name].append(event.table.to_pydict())
        assert collected["legacy"] == [dict(v=[1, 3], rn=[1, 2]), dict(v=[2], rn=[1])]
        assert collected["bounded"] == [
            dict(v=[1, 2, 3], seq=[10, 20, 30], rn=[1, 2, 3], total=[1, 3, 6])
        ]

    asyncio.run(run())


@pytest.mark.parametrize("bad", [False, "score", [True], ["score"]])
def test_over_order_requires_sortkey_sequence_with_its_own_path(bad):
    with pytest.raises(TypeError, match="order_by"):
        cf.OverOrder([], bad, sequence_by=["seq"])


def test_over_expressions_types_are_rejected_before_schema():
    schema, _ = nine_fixture()
    events = cf.table_input("events", schema=schema)
    spec = cf.OverSpec.bounded_source(
        cf.OverOrder(["p"], [cf.SortKey("score")], sequence_by=["seq"]),
        state_budget=cf.StateBudget(9, 8 << 20),
    )
    with pytest.raises(TypeError, match=r"expressions\[0\]"):
        events.over([True], spec=spec)
