from __future__ import annotations

from dataclasses import replace

import pytest

from calc_flow import Runtime
from calc_flow.capabilities import ProviderPort
from calc_flow.errors import CompileError
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    TableExpr,
    linalg,
    row,
    rows,
    table,
    table_input,
    ts,
    window,
)
from calc_flow.symbolic.analyzer import _contains_event_window, _run
from calc_flow.symbolic.nodes import Node


def _trades(*, timestamp: str = "timestamp[us, UTC]"):
    return table_input(
        "trades",
        schema=[
            Field("ts", timestamp),
            Field("symbol", "string"),
            Field("price", "float64"),
            Field("quantity", "int16"),
            Field("unsigned", "uint8", nullable=False),
        ],
    )


def _minute(trades, *, event_time="ts", group_by=(), aggregates=None):
    return window.tumbling(
        trades,
        event_time=event_time,
        size_micros=60_000_000,
        group_by=group_by,
        aggregates=aggregates or [window.sum("quantity", output="volume")],
    )


def _program(trades, result):
    return Program("event-window", inputs=[trades], outputs=[("minute", result)])


def test_native_window_capability_is_precise_and_sorted() -> None:
    capabilities = Runtime().capabilities()
    window = next(item for item in capabilities.operators if item.kind == "window")

    assert capabilities.schema_version == 3
    assert window.version == "1"
    assert window.modes == ("stream",)
    assert window.input_ports == (ProviderPort("input", "table", required=True),)
    assert window.output_ports == (ProviderPort("output", "table", required=True),)
    assert window.finality == "group_final_append_only"
    assert window.checkpoint_support == "checkpointed_stateful"
    assert window.state_version == 1
    assert window.state_layouts == (1,)
    assert window.requires_datafusion is False
    assert window.stateful and window.requires_watermark
    assert window.deterministic and window.replay_safe and window.microbatch_invariant
    keys = [(item.kind, item.version) for item in capabilities.operators]
    assert keys == sorted(keys)


@pytest.mark.parametrize(
    "timestamp", ["timestamp[ms]", "timestamp[us]", "timestamp[us, UTC]"]
)
def test_window_schema_and_nullable_time_do_not_require_rolling_order(
    timestamp,
) -> None:
    trades = _trades(timestamp=timestamp)
    minute = _minute(
        trades,
        group_by=["symbol"],
        aggregates=[
            window.count("ts", output="count"),
            window.sum("quantity", output="volume"),
            window.sum("unsigned", output="unsigned_sum"),
            window.min("price", output="low"),
            window.max("symbol", output="last_symbol"),
            window.avg("quantity", output="average"),
        ],
    )
    analyzer, _ = _run(_program(trades, minute), Runtime(), "stream")
    facts = analyzer.table(minute._node, "outputs.minute")

    assert analyzer.issues == ()
    assert facts.schema == (
        Field("window_start", "timestamp[us, UTC]", nullable=False),
        Field("window_end", "timestamp[us, UTC]", nullable=False),
        Field("symbol", "string"),
        Field("count", "uint64", nullable=False),
        Field("volume", "int64"),
        Field("unsigned_sum", "uint64"),
        Field("low", "float64"),
        Field("last_symbol", "string"),
        Field("average", "float64"),
    )
    assert facts.event_time is None
    assert facts.entity_by == facts.sequence_by == ()
    assert facts.state == frozenset({"window"})
    assert str(facts.lineage) == f"window:{minute._node.digest}"
    assert facts.lineage != str(facts.lineage)
    assert analyzer.temporal_lineages == frozenset()


@pytest.mark.parametrize(
    ("column", "data_type", "function", "expected", "nullable"),
    [
        ("value", "int8", "sum", "int64", True),
        ("value", "uint32", "sum", "uint64", True),
        ("value", "float32", "sum", "float64", True),
        ("value", "uint64", "avg", "float64", True),
        ("value", "bool", "min", "bool", True),
        ("value", "large_string", "max", "large_string", True),
        ("value", "date32", "min", "date32", True),
        ("value", "date64", "max", "date64", True),
        ("value", "timestamp[us]", "min", "timestamp[us]", True),
        ("value", "timestamp[us, UTC]", "max", "timestamp[us, UTC]", True),
        ("value", "time32[s]", "count", "uint64", False),
        ("value", "time64[us]", "count", "uint64", False),
    ],
)
def test_aggregate_type_matrix(column, data_type, function, expected, nullable) -> None:
    trades = table_input(
        "trades",
        schema=[Field("ts", "timestamp[us]"), Field(column, data_type, nullable=False)],
    )
    minute = _minute(
        trades, aggregates=[getattr(window, function)(column, output="value")]
    )
    analyzer, _ = _run(_program(trades, minute), Runtime(), "stream")

    assert analyzer.issues == ()
    assert analyzer.table(minute._node, "outputs.minute").schema[-1] == Field(
        "value", expected, nullable
    )


@pytest.mark.parametrize(
    ("event_time", "group_by", "aggregate", "suffix", "code"),
    [
        ("missing", (), ("sum", "quantity"), "event_time", "unresolved_type"),
        ("price", (), ("sum", "quantity"), "event_time.dtype", "unsupported_type"),
        ("ts", ("missing",), ("sum", "quantity"), "group_by[0]", "unresolved_type"),
        ("ts", (), ("sum", "missing"), "aggregates[0].column", "unresolved_type"),
        ("ts", (), ("avg", "symbol"), "aggregates[0].column.dtype", "unsupported_type"),
        ("ts", (), ("sum", "symbol"), "aggregates[0].column.dtype", "unsupported_type"),
    ],
)
def test_window_schema_diagnostics_have_stable_field_paths(
    event_time, group_by, aggregate, suffix, code
) -> None:
    trades = _trades()
    function, column = aggregate
    minute = _minute(
        trades,
        event_time=event_time,
        group_by=group_by,
        aggregates=[getattr(window, function)(column, output="value")],
    )
    program = _program(trades, minute)
    result = program.analyze(Runtime(), mode="stream")

    assert [(issue.path, issue.code) for issue in result.issues] == [
        (f"outputs.minute.window_tumbling.{suffix}", code)
    ]
    with pytest.raises(CompileError, match=code):
        program.compile_stream(Runtime())


@pytest.mark.parametrize("dtype", ["timestamp[ms]", "time32[s]", "time64[us]"])
@pytest.mark.parametrize("use", ["group", "min", "max"])
def test_window_rejects_non_native_ordered_types(dtype, use) -> None:
    trades = table_input(
        "trades", schema=[Field("ts", "timestamp[us]"), Field("value", dtype)]
    )
    minute = _minute(
        trades,
        group_by=["value"] if use == "group" else (),
        aggregates=[
            getattr(window, "count" if use == "group" else use)(
                "value", output="result"
            )
        ],
    )

    issues = _program(trades, minute).analyze(Runtime(), mode="stream").issues

    suffix = "group_by[0].dtype" if use == "group" else "aggregates[0].column.dtype"
    assert [(issue.path, issue.code) for issue in issues] == [
        (f"outputs.minute.window_tumbling.{suffix}", "unsupported_type")
    ]


def test_window_is_stream_only_in_analysis_and_compile() -> None:
    trades = _trades()
    program = _program(trades, _minute(trades))

    assert [
        (issue.path, issue.code)
        for issue in program.analyze(Runtime(), mode="batch").issues
    ] == [("outputs.minute.window_tumbling", "unsupported_mode")]
    with pytest.raises(CompileError, match="unsupported_mode"):
        program.compile_batch(Runtime())


@pytest.mark.parametrize(
    "change",
    [
        {"version": "2"},
        {"modes": ("batch",)},
        {"finality": "unproven"},
        {"state_version": 2, "state_layouts": (2,)},
        {"microbatch_invariant": False},
        {"requires_watermark": False},
        {"deterministic": False},
        {"replay_safe": False},
        {"input_ports": (ProviderPort("input", "table", required=False),)},
        {"output_ports": (ProviderPort("output", "array", required=True),)},
        {"checkpoint_support": "unproven", "state_version": None, "state_layouts": ()},
        None,
    ],
)
def test_window_capability_mismatch_is_shared_by_analysis_and_compile(
    monkeypatch, change
) -> None:
    trades = _trades()
    program = _program(trades, _minute(trades))
    runtime = Runtime()
    snapshot = runtime.capabilities()
    operators = tuple(
        replace(item, **change)
        if item.kind == "window" and change is not None
        else item
        for item in snapshot.operators
        if change is not None or item.kind != "window"
    )
    changed = replace(snapshot, operators=operators)
    monkeypatch.setattr(Runtime, "capabilities", lambda self: changed)

    assert [
        (issue.path, issue.code)
        for issue in program.analyze(runtime, mode="stream").issues
    ] == [("outputs.minute.window_tumbling", "capability_mismatch")]
    with pytest.raises(
        CompileError, match="outputs.minute.window_tumbling: capability_mismatch"
    ):
        program.compile_stream(runtime)


@pytest.mark.parametrize("kind", ["direct", "arithmetic", "spoofed_name"])
def test_window_columns_cannot_align_with_source_columns(kind) -> None:
    trades = _trades()
    minute = _minute(trades)
    foreign = (
        trades
        if kind != "spoofed_name"
        else table_input(
            f"window:{minute._node.digest}", schema=[Field("volume", "int64")]
        )
    )
    value = foreign["quantity"] if kind != "spoofed_name" else foreign["volume"]
    expression = (
        minute["volume"] + row.cast(value, "int64") if kind == "arithmetic" else value
    )
    output = minute.with_columns(FeatureSet([("raw", expression)]))
    inputs = [trades] if kind != "spoofed_name" else [trades, foreign]

    issues = (
        Program("mixed", inputs=inputs, outputs=[("minute", output)])
        .analyze(Runtime(), mode="stream")
        .issues
    )

    assert any(
        issue.code == "schema_mismatch" and issue.path.endswith(".lineage")
        for issue in issues
    )


def test_window_time_coordinate_accepts_rename_but_rejects_cast() -> None:
    trades = _trades()
    renamed = trades.with_columns(FeatureSet([("renamed", trades["ts"])]))
    assert (
        _program(trades, _minute(renamed, event_time="renamed"))
        .analyze(Runtime(), mode="stream")
        .issues
        == ()
    )
    cast = trades.with_columns(
        FeatureSet([("cast_time", row.cast(trades["ts"], "timestamp[us]"))])
    )
    issues = (
        _program(trades, _minute(cast, event_time="cast_time"))
        .analyze(Runtime(), mode="stream")
        .issues
    )
    assert [(issue.path, issue.code) for issue in issues] == [
        ("outputs.minute.window_tumbling.event_time", "capability_mismatch")
    ]


@pytest.mark.parametrize("stage", ["before", "after", "predicate", "nested"])
def test_window_paths_reject_other_stateful_stages(stage) -> None:
    trades = _trades()
    if stage == "before":
        prepared = trades.with_columns(
            FeatureSet([("rolling", ts.sum(trades["quantity"], window=rows(2)))])
        )
        minute = _minute(table.project(prepared, ["ts", "quantity"]))
    elif stage == "nested":
        first = _minute(trades)
        minute = _minute(
            first,
            event_time="window_end",
            aggregates=[window.sum("volume", output="total")],
        )
    elif stage == "predicate":
        prepared = table.filter(trades, ts.sum(trades["quantity"], window=rows(2)) > 0)
        minute = _minute(prepared)
    else:
        first = _minute(trades)
        minute = first.with_columns(
            FeatureSet([("rolling", ts.sum(first["volume"], window=rows(2)))])
        )

    issues = _program(trades, minute).analyze(Runtime(), mode="stream").issues

    assert any(
        issue.code == "capability_mismatch" and "window" in issue.message
        for issue in issues
    )


@pytest.mark.parametrize("stage", ["before", "after", "wrapped", "array_output"])
def test_window_paths_reject_array_boundaries_before_shape_inference(stage) -> None:
    trades = _trades()
    first = _minute(trades)
    if stage == "before":
        array = linalg.from_columns(trades, columns=["quantity"], backend="numpy")
        attached = table.attach_columns(trades, array, names=["copy"])
        result = _minute(table.project(attached, ["ts", "quantity"]))
    elif stage == "array_output":
        result = linalg.from_columns(first, columns=["volume"], backend="numpy")
    else:
        target = (
            table.project(first, ["window_start", "volume"])
            if stage == "wrapped"
            else first
        )
        array = linalg.from_columns(target, columns=["volume"], backend="numpy")
        result = table.attach_columns(target, array, names=["copy"])

    program = _program(trades, result)
    analyzer, _ = _run(program, Runtime(), "stream")

    assert any(
        issue.code == "capability_mismatch" and "window" in issue.message
        for issue in analyzer.issues
    )
    if stage == "array_output":
        assert all(
            type(dimension) in (str, int)
            for dimension in analyzer.array(result._node, "outputs.minute").shape
        )


@pytest.mark.parametrize("version", [0, 3])
def test_window_analysis_rejects_unknown_primitive_versions(version) -> None:
    trades = _trades()
    minute = _minute(trades)
    unknown = TableExpr(
        replace(minute._node, op=replace(minute._node.op, version=version))
    )

    issues = _program(trades, unknown).analyze(Runtime(), mode="stream").issues

    assert [(issue.path, issue.code) for issue in issues] == [
        ("outputs.minute.window_tumbling", "unknown_primitive_version")
    ]


def test_independent_rolling_branch_keeps_its_input_ordering_requirements() -> None:
    trades = _trades()
    rolling = trades.with_columns(
        FeatureSet([("rolling", ts.sum(trades["quantity"], window=rows(2)))])
    )
    program = Program(
        "mixed",
        inputs=[trades],
        outputs=[("minute", _minute(trades)), ("rolling", rolling)],
    )

    issues = program.analyze(Runtime(), mode="stream").issues

    assert {issue.path for issue in issues} == {
        "inputs.trades.event_time",
        "inputs.trades.entity_by",
        "inputs.trades.sequence_by",
    }
    assert {issue.code for issue in issues} == {"ordering_required"}


def test_window_explain_displays_typed_row_origin() -> None:
    trades = _trades()
    minute = _minute(trades)

    explanation = _program(trades, minute).explain(Runtime(), mode="stream")

    assert f"lineage window:{minute._node.digest}" in explanation


def test_window_detection_visits_shared_declaration_nodes_once(monkeypatch):
    trades = _trades()
    expression = trades["price"]
    for _ in range(12):
        expression = expression + expression
    args_descriptor = Node.args
    visits: dict[str, int] = {}

    def observed_args(node):
        visits[node.digest] = visits.get(node.digest, 0) + 1
        return args_descriptor.__get__(node, Node)

    monkeypatch.setattr(Node, "args", property(observed_args))
    assert not _contains_event_window(expression._node)
    assert len(visits) == 14
    assert max(visits.values()) == 1


def test_unlowerable_window_input_cast_returns_a_stable_analysis_issue():
    trades = _trades()
    prepared = trades.with_columns(
        FeatureSet([("text", row.cast(trades["price"], "string"))])
    )
    program = _program(
        trades, _minute(prepared, aggregates=[window.count("text", output="count")])
    )
    runtime = Runtime()
    issues = program.analyze(runtime, mode="stream").issues
    assert len(issues) == 1
    assert issues[0].path == "outputs.minute.window_tumbling.value.schema"
    assert issues[0].code == "unsupported_type"
    with pytest.raises(CompileError, match=r"window_tumbling\.value\.schema"):
        program.compile_stream(runtime)
