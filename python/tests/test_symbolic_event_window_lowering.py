"""Native event-window lowering, state ownership, and finality boundaries."""

from __future__ import annotations

import pyarrow as pa
import pytest

from calc_flow import JoinStateLimits, JoinTimeBounds, Runtime, register_numpy
from calc_flow.errors import CompileError, ConfigError, ExecutionError
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    linalg,
    parameter,
    rows,
    table,
    table_input,
    ts,
    window,
)
from calc_flow.symbolic.lower import lower_program_document


def _input(name: str = "trades"):
    return table_input(
        name,
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("quantity", "int64"),
            Field("price", "float64"),
        ],
        event_time="ts",
        entity_by=["symbol"],
        sequence_by=["seq"],
    )


def _window(value, *, hopping: bool = False):
    arguments = {
        "event_time": "ts",
        "size_micros": 60_000_000,
        "group_by": ["symbol"],
        "aggregates": [
            window.count("seq", output="count"),
            window.sum("quantity", output="volume"),
            window.avg("price", output="mean"),
        ],
    }
    if hopping:
        return window.hopping(value, slide_micros=30_000_000, **arguments)
    return window.tumbling(value, **arguments)


def _nodes(document, kind: str):
    return [
        node for node in document["graph"]["nodes"] if node["operator"]["kind"] == kind
    ]


@pytest.mark.parametrize("hopping", [False, True])
def test_window_lowers_to_existing_native_spec(hopping: bool) -> None:
    trades = _input()
    result = _window(trades, hopping=hopping)
    program = Program("native-window", inputs=[trades], outputs=[("minute", result)])
    runtime = Runtime()

    document = lower_program_document(program, runtime, "stream")
    native = _nodes(document, "window")

    assert len(native) == 1
    assert native[0]["operator"]["spec"] == {
        "event_time_column": "ts",
        "group_by": ["symbol"],
        "geometry": {
            "kind": "hopping" if hopping else "tumbling",
            "size_micros": 60_000_000,
            **({"slide_micros": 30_000_000} if hopping else {}),
        },
        "aggregates": [
            {"function": "count", "column": "seq", "output": "count"},
            {"function": "sum", "column": "quantity", "output": "volume"},
            {"function": "avg", "column": "price", "output": "mean"},
        ],
    }
    assert native[0]["input_ports"][0]["schema"]
    assert len(program.compile_stream(runtime).source_binding_ids) == 1


def test_shared_window_preserves_filters_on_each_side() -> None:
    trades = _input()
    selected = table.filter(trades, trades["quantity"] > 0)
    minute = _window(selected)
    selected_minute = table.filter(minute, minute["volume"] > 10)
    program = Program(
        "shared-window",
        inputs=[trades],
        outputs=[("all", minute), ("selected", selected_minute)],
    )
    document = lower_program_document(program, Runtime(), "stream")
    native = _nodes(document, "window")
    assert len(native) == 1
    edges = document["graph"]["edges"]
    nodes = {node["id"]: node for node in document["graph"]["nodes"]}
    window_id = native[0]["id"]
    incoming = [
        edge["source_node"] for edge in edges if edge["target_node"] == window_id
    ]
    assert len(incoming) == 1
    assert '"quantity"' in nodes[incoming[0]]["operator"]["filter"]
    assert nodes["selected"]["operator"]["filter"] == '("volume" > 10)'
    assert nodes["all"]["operator"]["filter"] is None
    program.compile_stream(Runtime())


def test_distinct_window_inputs_do_not_share_state() -> None:
    first, second = _input("first"), _input("second")
    a, b = _window(first), _window(second)
    program = Program(
        "independent-windows", inputs=[first, second], outputs=[("a", a), ("b", b)]
    )
    document = lower_program_document(program, Runtime(), "stream")
    assert len(_nodes(document, "window")) == 2
    assert len(program.compile_stream(Runtime()).source_binding_ids) == 2


def test_window_and_independent_rolling_share_one_source() -> None:
    trades = _input()
    minute = _window(trades)
    rolling = trades.with_columns(
        FeatureSet([("average", ts.mean(trades["price"], window=rows(2)))])
    )
    program = Program(
        "mixed-windows",
        inputs=[trades],
        outputs=[("minute", minute), ("rolling", rolling)],
    )
    document = lower_program_document(
        program, Runtime(), "stream", allowed_lateness_micros=500, late_policy="drop"
    )
    assert len(_nodes(document, "window")) == 1
    assert len(_nodes(document, "rolling")) == 1
    assert (
        "allowed_lateness_micros"
        not in _nodes(document, "window")[0]["operator"]["spec"]
    )
    assert (
        len(
            program.compile_stream(
                Runtime(), allowed_lateness_micros=500, late_policy="drop"
            ).source_binding_ids
        )
        == 1
    )


def test_window_and_independent_join_preserve_bindings() -> None:
    from datetime import timedelta

    left, right = _input("left"), _input("right")
    joined = table.stream_join(
        left,
        right,
        left_keys=["symbol"],
        right_keys=["symbol"],
        left_event_time="ts",
        right_event_time="ts",
        bounds=JoinTimeBounds(before=timedelta(seconds=1), after=timedelta(seconds=1)),
        limits=JoinStateLimits(
            max_state_rows_per_side=100,
            max_state_bytes_per_side=65536,
            max_matches_per_input_batch=100,
        ),
    )
    program = Program(
        "window-join",
        inputs=[left, right],
        outputs=[("minute", _window(left)), ("joined", joined)],
    )
    document = lower_program_document(program, Runtime(), "stream")
    assert len(_nodes(document, "window")) == 1
    assert len(_nodes(document, "stream_join")) == 1
    assert len(program.compile_stream(Runtime()).source_binding_ids) == 2


def test_window_and_independent_matrix_preserve_static_bindings() -> None:
    trades = _input()
    weights = parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(1, 1)
    )
    matrix = linalg.from_columns(trades, columns=["price"], backend="numpy")
    attached = table.attach_columns(
        trades, linalg.matmul(matrix, weights), names=["weighted"]
    )
    program = Program(
        "window-matrix",
        inputs=[trades, weights],
        outputs=[("minute", _window(trades)), ("matrix", attached)],
    )
    runtime = Runtime()
    register_numpy(runtime)
    document = lower_program_document(program, runtime, "stream")
    assert len(_nodes(document, "window")) == 1
    assert len(_nodes(document, "external")) == 1
    assert [item["name"] for item in document["static_inputs"]] == ["weights"]
    plan = program.compile_stream(runtime)
    assert len(plan.source_binding_ids) == 1


def test_window_does_not_expose_an_unused_declared_source() -> None:
    used, unused = _input("used"), _input("unused")
    program = Program(
        "unused-source", inputs=[used, unused], outputs=[("minute", _window(used))]
    )
    assert len(program.compile_stream(Runtime()).source_binding_ids) == 1


def test_failed_native_stream_compile_does_not_populate_cache() -> None:
    trades = _input()
    program = Program(
        "invalid-node", inputs=[trades], outputs=[("invalid/output", _window(trades))]
    )
    runtime = Runtime()
    with pytest.raises(ConfigError, match="invalid_id"):
        program.compile_stream(runtime)
    assert runtime._symbolic_compile_cache == {}


@pytest.mark.parametrize(
    "options", [{"allowed_lateness_micros": 1}, {"late_policy": "drop"}]
)
def test_window_only_rejects_unused_lateness_options(options) -> None:
    trades = _input()
    program = Program(
        "window-options", inputs=[trades], outputs=[("minute", _window(trades))]
    )
    with pytest.raises(
        CompileError, match=r"window-options\.compile_stream\..*: capability_mismatch"
    ):
        program.compile_stream(Runtime(), **options)


def test_window_node_ids_handle_output_and_source_name_collisions() -> None:
    trades = _input("minute")
    minute = _window(trades)
    collision = f"cf_window_{minute.digest[:24]}"
    program = Program(
        "collisions", inputs=[trades], outputs=[("minute", minute), (collision, minute)]
    )
    document = lower_program_document(program, Runtime(), "stream")
    ids = [node["id"] for node in document["graph"]["nodes"]]
    assert len(ids) == len(set(ids))
    assert len(_nodes(document, "window")) == 1
    assert len(program.compile_stream(Runtime()).source_binding_ids) == 1


def test_window_explain_reports_native_finality_and_state_sharing() -> None:
    trades = _input()
    minute = _window(trades)
    program = Program(
        "explain-window", inputs=[trades], outputs=[("a", minute), ("b", minute)]
    )
    explained = program.explain(Runtime(), mode="stream")
    assert "window state_stages 1 shared_outputs 2" in explained
    assert "group_final_append_only" in explained
    assert "late_assignments=drop" in explained
    assert "default compile options" in explained


def test_rebuilt_window_graph_has_stable_fingerprint_and_cache_identity() -> None:
    runtime = Runtime()
    first = _input()
    second = _input()
    a = Program("cache-window", inputs=[first], outputs=[("minute", _window(first))])
    b = Program("cache-window", inputs=[second], outputs=[("minute", _window(second))])
    assert lower_program_document(a, runtime, "stream") == lower_program_document(
        b, runtime, "stream"
    )
    first_plan, second_plan = a.compile_stream(runtime), b.compile_stream(runtime)
    assert first_plan is not second_plan
    assert first_plan.fingerprint == second_plan.fingerprint
    changed = Program(
        "cache-window",
        inputs=[first],
        outputs=[("minute", _window(first, hopping=True))],
    )
    assert first_plan.fingerprint != changed.compile_stream(runtime).fingerprint


def test_native_schema_conversion_preserves_every_symbolic_field_type():
    from calc_flow.symbolic.lower.schema import _arrow_schema, _fields
    from calc_flow.symbolic.types import TABLE_FIELD_TYPES

    fields = tuple(
        Field(f"value_{index}", data_type, nullable=index % 2 == 0)
        for index, data_type in enumerate(sorted(TABLE_FIELD_TYPES))
    )
    assert _fields(_arrow_schema(fields)) == fields


def test_native_schema_cache_is_bounded_and_registration_invalidates(monkeypatch):
    monkeypatch.setattr("calc_flow.pipeline._SYMBOLIC_COMPILE_CACHE_MAX_ENTRIES", 2)
    runtime = Runtime()
    source = pa.schema([pa.field("value", pa.int64())])
    select = ["value"]
    first = runtime._infer_symbolic_expression_schema(select, None, source)
    assert runtime._infer_symbolic_expression_schema(select, None, source) is first
    select[0] = "value AS renamed"
    renamed = runtime._infer_symbolic_expression_schema(select, None, source)
    assert renamed.names == ["renamed"]
    runtime._infer_symbolic_expression_schema(["value AS third"], None, source)
    assert len(runtime._symbolic_schema_cache) == 2
    assert (
        runtime._infer_symbolic_expression_schema(["value"], None, source) is not first
    )
    runtime.register_provider("test", "schema-cache", "1", lambda _inputs, _options: {})
    assert runtime._symbolic_schema_cache == {}


def test_native_schema_cache_never_retains_failed_planning():
    runtime = Runtime()
    source = pa.schema([pa.field("value", pa.int64())])
    with pytest.raises(ExecutionError):
        runtime._infer_symbolic_expression_schema(["missing + 1"], None, source)
    assert runtime._symbolic_schema_cache == {}
    expected = pa.schema([pa.field("value", pa.int64())])
    assert (
        runtime._infer_symbolic_expression_schema(["value"], None, source) == expected
    )
