"""DataFusion planning proves exact schemas without running source data."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import pyarrow as pa
import pytest
from test_symbolic_event_window_recovery import BASE, MINUTE, _complete, _ScriptedSource

from calc_flow import Batch, PipelineBuilder, Runtime, StreamingRunner
from calc_flow.pipeline import BatchExecutionPlan
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    row,
    table,
    table_input,
    window,
)
from calc_flow.symbolic.lower import lower_program_document

_CASES = (
    ("where_guard", "CASE WHEN x > 0.0 THEN x ELSE 0.0 END", "float64", False),
    (
        "where_true",
        "CASE WHEN CAST(TRUE AS BOOLEAN) THEN 1.0 ELSE x END",
        "float64",
        False,
    ),
    ("and_false", "condition AND FALSE", "bool", False),
    ("or_true", "condition OR TRUE", "bool", False),
    ("sqrt_nonnull", "SQRT(x)", "float64", True),
)


def _source(operation="where_guard"):
    return table_input(
        "trades",
        schema=[
            Field("ts", "timestamp[us, UTC]"),
            Field("x", "float64", nullable=operation != "sqrt_nonnull"),
            Field("condition", "bool"),
        ],
    )


def _source_schema(operation="where_guard"):
    return pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC")),
            pa.field("x", pa.float64(), nullable=operation != "sqrt_nonnull"),
            pa.field("condition", pa.bool_()),
        ]
    )


def _data(operation="where_guard"):
    return pa.table(
        {
            "ts": [BASE + timedelta(seconds=value) for value in (5, 35, 65)],
            "x": [1.0, 4.0, 9.0] if operation == "sqrt_nonnull" else [1.0, None, -2.0],
            "condition": [True, None, False],
        },
        schema=_source_schema(operation),
    )


def _value(operation, source):
    if operation == "where_guard":
        return row.where(source["x"] > 0.0, source["x"], 0.0)
    if operation == "where_true":
        return row.where(row.cast(True, "bool"), 1.0, source["x"])
    if operation == "and_false":
        return source["condition"] & False
    if operation == "or_true":
        return source["condition"] | True
    return row.sqrt(source["x"])


def _declaration(operation, placement):
    source = _source(operation)
    if placement == "before":
        prepared = source.with_columns(FeatureSet([("key", _value(operation, source))]))
        projected = table.project(prepared, ["ts", "key"])
        result = window.tumbling(
            projected,
            event_time="ts",
            size_micros=MINUTE,
            group_by=["key"],
            aggregates=[window.count("key", output="count")],
        )
    else:
        minute = window.tumbling(
            source,
            event_time="ts",
            size_micros=MINUTE,
            aggregates=[
                window.min("x", output="x"),
                window.max("condition", output="condition"),
            ],
        )
        result = table.project(
            minute.with_columns(FeatureSet([("value", _value(operation, minute))])),
            ["window_start", "window_end", "value"],
        )
    return Program(
        "planned-window-schema", inputs=[source], outputs=[("result", result)]
    )


def _field(name, dtype, nullable=True):
    return {"name": name, "data_type": dtype, "nullable": nullable}


def _port(schema):
    return {"name": "input", "kind": "table", "required": True, "schema": schema}


def _expression_node(select, schema, output):
    return {
        "id": "transform",
        "input_ports": [_port(schema)],
        "output_ports": [{**_port(output), "name": "output"}],
        "operator": {
            "kind": "expression",
            "expression": "",
            "select": select,
            "filter": None,
            "udfs": [],
        },
    }


def _window_node(schema, aggregates, groups=(), time="ts"):
    return {
        "id": "window",
        "input_ports": [_port(schema)],
        "output_ports": [],
        "operator": {
            "kind": "window",
            "spec": {
                "event_time_column": time,
                "group_by": list(groups),
                "geometry": {"kind": "tumbling", "size_micros": MINUTE},
                "aggregates": aggregates,
            },
        },
    }


def _manual_plan(nodes):
    edges = [
        {
            "source_node": left["id"],
            "source_port": "output",
            "target_node": right["id"],
            "target_port": "input",
        }
        for left, right in zip(nodes[:-1], nodes[1:], strict=True)
    ]
    document = {
        "format_version": 3,
        "id": "native-schema-oracle",
        "name": "native-schema-oracle",
        "runtime": {"mode": "stream", "options": {}},
        "data_sources": [],
        "graph": {"name": "native-schema-oracle", "nodes": nodes, "edges": edges},
    }
    return PipelineBuilder._from_json(json.dumps(document)).compile_stream(
        runtime=Runtime()
    )


def _native(case, placement):
    operation, sql, dtype, nullable = case
    source = [
        _field("ts", "timestamp[us, UTC]"),
        _field("x", "float64", operation != "sqrt_nonnull"),
        _field("condition", "bool"),
    ]
    if placement == "before":
        prepared = [_field("ts", "timestamp[us, UTC]"), _field("key", dtype, nullable)]
        return _manual_plan(
            [
                _expression_node(["ts", f"{sql} AS key"], source, prepared),
                _window_node(
                    prepared,
                    [{"function": "count", "column": "key", "output": "count"}],
                    ["key"],
                ),
            ]
        )
    minute = [
        _field("window_start", "timestamp[us, UTC]", False),
        _field("window_end", "timestamp[us, UTC]", False),
        _field("x", "float64"),
        _field("condition", "bool"),
    ]
    return _manual_plan(
        [
            _window_node(
                source,
                [
                    {"function": "min", "column": "x", "output": "x"},
                    {"function": "max", "column": "condition", "output": "condition"},
                ],
            ),
            _expression_node(
                ["window_start", "window_end", f"{sql} AS value"],
                minute,
                [*minute[:2], _field("value", dtype, nullable)],
            ),
        ]
    )


@pytest.mark.parametrize("case", _CASES, ids=[case[0] for case in _CASES])
@pytest.mark.parametrize("placement", ["before", "after"])
def test_window_schema_planning_matches_manual_native(case, placement, tmp_path):
    operation, _sql, dtype, nullable = case
    runtime = Runtime()
    program = _declaration(operation, placement)
    assert program.analyze(runtime, mode="stream").issues == ()
    explanation = program.explain(runtime, mode="stream")
    field = "key" if placement == "before" else "value"
    assert f"field {field} {dtype} nullable={str(nullable).lower()}" in explanation
    events = [("data", _data(operation))]
    expected, _ = asyncio.run(
        _complete(_native(case, placement), events, tmp_path / "native")
    )
    actual, _ = asyncio.run(
        _complete(program.compile_stream(runtime), events, tmp_path / "symbolic")
    )
    assert actual.schema == expected.schema
    assert actual.equals(expected), (actual.to_pylist(), expected.to_pylist())


def test_window_planning_handles_rename_projection_and_multiple_cse_stages(tmp_path):
    source = _source()
    aliased = table.project(
        source.with_columns(
            FeatureSet([("time", source["ts"]), ("value", source["x"])])
        ),
        ["time", "value"],
    )
    positive = row.where(aliased["value"] > 0.0, aliased["value"], 0.0)
    incremented = positive + 1.0
    prepared = table.project(
        aliased.with_columns(
            FeatureSet(
                [("key", incremented + positive), ("copy", incremented - positive)]
            )
        ),
        ["time", "key", "copy"],
    )
    result = window.tumbling(
        prepared,
        event_time="time",
        size_micros=MINUTE,
        group_by=["key"],
        aggregates=[window.count("copy", output="count")],
    )
    program = Program(
        "planned-cse-window", inputs=[source], outputs=[("result", result)]
    )
    runtime = Runtime()
    document = lower_program_document(program, runtime, "stream")
    materializations = [
        node
        for node in document["graph"]["nodes"]
        if any(' AS "__cf_cse_' in item for item in node["operator"].get("select", []))
    ]
    assert len(materializations) >= 2
    assert "field key float64 nullable=false" in program.explain(runtime, mode="stream")
    positive_sql = "(CASE WHEN x > 0.0 THEN x ELSE 0.0 END)"
    source_fields = [
        _field("ts", "timestamp[us, UTC]"),
        _field("x", "float64"),
        _field("condition", "bool"),
    ]
    prepared_fields = [
        _field("time", "timestamp[us, UTC]"),
        _field("key", "float64", False),
        _field("copy", "float64", False),
    ]
    native = _manual_plan(
        [
            _expression_node(
                [
                    "ts AS time",
                    f"({positive_sql} + 1.0 + {positive_sql}) AS key",
                    f"({positive_sql} + 1.0 - {positive_sql}) AS copy",
                ],
                source_fields,
                prepared_fields,
            ),
            _window_node(
                prepared_fields,
                [{"function": "count", "column": "copy", "output": "count"}],
                ["key"],
                "time",
            ),
        ]
    )
    events = [("data", _data())]
    expected, _ = asyncio.run(_complete(native, events, tmp_path / "native"))
    actual, _ = asyncio.run(
        _complete(program.compile_stream(runtime), events, tmp_path / "symbolic")
    )
    assert actual.schema == expected.schema
    assert actual.equals(expected)


def test_schema_planning_is_data_free_inside_an_existing_event_loop(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("schema planning must not execute batches or open a stream source")

    monkeypatch.setattr(BatchExecutionPlan, "execute", forbidden)
    monkeypatch.setattr(BatchExecutionPlan, "execute_async", forbidden)
    monkeypatch.setattr(StreamingRunner, "start_async", forbidden)
    monkeypatch.setattr(_ScriptedSource, "open", forbidden)

    async def plan():
        runtime = Runtime()
        program = _declaration("where_guard", "before")
        assert program.analyze(runtime, mode="stream").issues == ()
        assert "field key float64 nullable=false" in program.explain(
            runtime, mode="stream"
        )
        assert program.compile_stream(runtime).source_binding_ids

    asyncio.run(plan())
    assert calls == []


def test_private_native_schema_projection_preserves_arrow_metadata():
    runtime = Runtime()
    schema = pa.schema(
        [
            pa.field("x", pa.float64(), nullable=False, metadata={"unit": "USD"}),
            pa.field("unused", pa.bool_()),
        ],
        metadata={"origin": "declared"},
    )
    planned = runtime._inner._infer_expression_schema(["x"], None, schema)
    assert planned.equals(
        pa.schema([schema.field("x")], metadata=schema.metadata), check_metadata=True
    )
    assert schema.names == ["x", "unused"]


def test_row_only_projection_keeps_declared_field_nullability():
    source = _source("sqrt_nonnull")
    program = Program(
        "plain-projection",
        inputs=[source],
        outputs=[("result", table.project(source, ["x", "condition"]))],
    )
    runtime = Runtime()
    assert program.analyze(runtime, mode="batch").issues == ()
    explanation = program.explain(runtime, mode="batch")
    assert "field x float64 nullable=false" in explanation
    assert "field condition bool nullable=true" in explanation
    output = (
        program.compile_batch(runtime)
        .execute({"input": Batch.from_pyarrow(_data("sqrt_nonnull"))})
        .outputs["output"]
        .to_pyarrow()
    )
    assert output.schema == pa.schema(
        [_source_schema("sqrt_nonnull").field("x"), _source_schema().field("condition")]
    )
