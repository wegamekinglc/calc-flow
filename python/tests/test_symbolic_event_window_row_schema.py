"""Row transforms keep their analyzed schema across native window boundaries."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import pyarrow as pa
import pytest
from test_symbolic_event_window_recovery import BASE, MINUTE, _complete

from calc_flow import Batch, PipelineBuilder, Runtime
from calc_flow.errors import CompileError
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

_CLIP_SQL = (
    "CAST((CASE WHEN {value} < -1.1 THEN -1.1"
    " WHEN {value} > 1.1 THEN 1.1 ELSE {value} END) AS REAL)"
)
_ROW_CASES = (
    ("sqrt", "SQRT(CAST({value} AS DOUBLE))", "float64"),
    ("exp", "EXP(CAST({value} AS DOUBLE))", "float64"),
    ("log", "LN(CAST({value} AS DOUBLE))", "float64"),
    ("clip", _CLIP_SQL, "float32"),
    ("clip_abs", _CLIP_SQL.replace("{value}", "ABS({value})"), "float32"),
    ("clip_clip", _CLIP_SQL.replace("{value}", f"({_CLIP_SQL})"), "float32"),
    (
        "clip_coalesce",
        _CLIP_SQL.replace("{value}", "COALESCE({value}, NULL)"),
        "float32",
    ),
    (
        "clip_where",
        _CLIP_SQL.replace(
            "{value}", "(CASE WHEN {value} > 0 THEN {value} ELSE NULL END)"
        ),
        "float32",
    ),
    ("clip_cast_null", _CLIP_SQL.replace("{value}", "CAST(NULL AS REAL)"), "float32"),
    ("abs", "ABS({value})", "float32"),
)


def _row_expression(name, value):
    if name == "clip_abs":
        value = row.abs(value)
    elif name == "clip_clip":
        value = row.clip(value, lower=-1.1, upper=1.1)
    elif name == "clip_coalesce":
        value = row.coalesce(value, None)
    elif name == "clip_where":
        value = row.where(value > row.cast(0.0, "float32"), value, None)
    elif name == "clip_cast_null":
        value = row.cast(None, "float32")
    if name.startswith("clip"):
        return row.clip(value, lower=-1.1, upper=1.1)
    return getattr(row, name)(value)


def _input(dtype="float32", nullable=True):
    return table_input(
        "trades",
        schema=[
            Field("ts", "timestamp[us, UTC]"),
            Field("x", dtype, nullable=nullable),
        ],
    )


def _window(value, aggregates):
    return window.tumbling(
        value, event_time="ts", size_micros=MINUTE, aggregates=aggregates
    )


def _program(operation, placement, nullable):
    trades = _input(nullable=nullable)
    if placement == "before":
        expression = _row_expression(operation, trades["x"])
        prepared = trades.with_columns(
            FeatureSet([("value", expression), ("copy", expression)])
        )
        prepared = table.project(prepared, ["ts", "value", "copy"])
        result = _window(
            prepared,
            [window.min("value", output="result"), window.max("copy", output="copy")],
        )
    else:
        minute = _window(trades, [window.min("x", output="value")])
        expression = _row_expression(operation, minute["value"])
        result = table.project(
            minute.with_columns(
                FeatureSet([("result", expression), ("copy", expression)])
            ),
            ["window_start", "window_end", "result", "copy"],
        )
    return Program("row-window-schema", inputs=[trades], outputs=[("result", result)])


def _field(name, dtype, nullable=True):
    return {"name": name, "data_type": dtype, "nullable": nullable}


def _port(fields, name="input"):
    return {"name": name, "kind": "table", "required": True, "schema": fields}


def _native_plan(case, placement, nullable):
    operation, sql, dtype = case
    source_fields = [
        _field("ts", "timestamp[us, UTC]"),
        _field("x", "float32", nullable),
    ]
    bounds = [
        _field("window_start", "timestamp[us, UTC]", False),
        _field("window_end", "timestamp[us, UTC]", False),
    ]
    output_fields = [*bounds, _field("result", dtype), _field("copy", dtype)]
    if placement == "before":
        result_nullable = nullable or operation in {
            "sqrt",
            "exp",
            "log",
            "abs",
            "clip_abs",
            "clip_where",
            "clip_cast_null",
        }
        projected_fields = [
            source_fields[0],
            _field("value", dtype, result_nullable),
            _field("copy", dtype, result_nullable),
        ]
        select = [
            "ts",
            f"{sql.format(value='x')} AS value",
            f"{sql.format(value='x')} AS copy",
        ]
        aggregate_inputs = projected_fields
        aggregate_outputs = output_fields
        aggregates = [
            {"function": "min", "column": "value", "output": "result"},
            {"function": "max", "column": "copy", "output": "copy"},
        ]
        projection_inputs, projection_outputs = source_fields, projected_fields
        source_node, target_node = "transform", "window"
    else:
        aggregate_inputs = source_fields
        aggregate_outputs = [*bounds, _field("value", "float32")]
        aggregates = [{"function": "min", "column": "x", "output": "value"}]
        select = [
            "window_start",
            "window_end",
            f"{sql.format(value='value')} AS result",
            f"{sql.format(value='value')} AS copy",
        ]
        projection_inputs, projection_outputs = aggregate_outputs, output_fields
        source_node, target_node = "window", "transform"
    nodes = [
        {
            "id": "transform",
            "input_ports": [_port(projection_inputs)],
            "output_ports": [_port(projection_outputs, "output")],
            "operator": {
                "kind": "expression",
                "expression": "",
                "select": select,
                "filter": None,
                "udfs": [],
            },
        },
        {
            "id": "window",
            "input_ports": [_port(aggregate_inputs)],
            "output_ports": [_port(aggregate_outputs, "output")],
            "operator": {
                "kind": "window",
                "spec": {
                    "event_time_column": "ts",
                    "group_by": [],
                    "geometry": {"kind": "tumbling", "size_micros": MINUTE},
                    "aggregates": aggregates,
                },
            },
        },
    ]
    document = {
        "format_version": 3,
        "id": "manual-row-window",
        "name": "manual-row-window",
        "runtime": {"mode": "stream", "options": {}},
        "data_sources": [],
        "graph": {
            "name": "manual-row-window",
            "nodes": nodes,
            "edges": [
                {
                    "source_node": source_node,
                    "source_port": "output",
                    "target_node": target_node,
                    "target_port": "input",
                }
            ],
        },
    }
    return PipelineBuilder._from_json(json.dumps(document)).compile_stream(
        runtime=Runtime()
    )


@pytest.mark.parametrize("placement", ["before", "after"])
@pytest.mark.parametrize("nullable", [True, False])
@pytest.mark.parametrize("case", _ROW_CASES, ids=[case[0] for case in _ROW_CASES])
def test_window_row_transform_schema_and_values_match_manual_native(
    case, placement, nullable, tmp_path
):
    operation, _sql, dtype = case
    program = _program(operation, placement, nullable)
    runtime = Runtime()
    analysis = program.analyze(runtime, mode="stream")
    assert analysis.issues == ()
    data = pa.table(
        {
            "ts": pa.array(
                [BASE + timedelta(seconds=seconds) for seconds in (5, 35, 65, 125)],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "x": pa.array(
                [1.0, None, 2.0, None] if nullable else [1.0, 2.0, 3.0, 4.0],
                type=pa.float32(),
            ),
        },
        schema=pa.schema(
            [
                pa.field("ts", pa.timestamp("us", tz="UTC")),
                pa.field("x", pa.float32(), nullable=nullable),
            ]
        ),
    )
    events = [("data", data)]
    expected, _ = asyncio.run(
        _complete(_native_plan(case, placement, nullable), events, tmp_path / "native")
    )
    actual, _ = asyncio.run(
        _complete(program.compile_stream(runtime), events, tmp_path / "symbolic")
    )
    assert actual.schema == expected.schema
    assert actual.equals(expected), (actual.to_pylist(), expected.to_pylist())
    assert actual["result"].null_count == expected["result"].null_count
    if nullable or operation == "clip_cast_null":
        assert actual["result"][-1].as_py() is None
    assert f"field result {dtype} nullable=true" in program.explain(
        runtime, mode="stream"
    )


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64"])
def test_unsigned_negation_is_rejected_before_window_source_open(dtype):
    trades = _input(dtype)
    prepared = trades.with_columns(FeatureSet([("negative", -trades["x"])]))
    result = _window(prepared, [window.min("negative", output="result")])
    program = Program("unsigned-window", inputs=[trades], outputs=[("result", result)])
    path = "outputs.result.window_tumbling.value.negative.neg.value.dtype"
    issues = program.analyze(Runtime(), mode="stream").issues
    assert any(
        issue.path == path and issue.code == "unsupported_type" for issue in issues
    )
    with pytest.raises(CompileError, match=rf"{path}: unsupported_type"):
        program.compile_stream(Runtime())


def test_unsigned_negation_allows_explicit_signed_cast():
    source = _input("uint8")
    result = source.with_columns(
        FeatureSet([("negative", -row.cast(source["x"], "int16"))])
    )
    program = Program("signed-negation", inputs=[source], outputs=[("result", result)])
    data = pa.table(
        {
            "ts": pa.array([BASE, BASE], type=pa.timestamp("us", tz="UTC")),
            "x": pa.array([200, None], type=pa.uint8()),
        }
    )
    output = (
        program.compile_batch(Runtime())
        .execute({"input": Batch.from_pyarrow(data)})
        .outputs["output"]
        .to_pyarrow()
    )
    assert output["negative"].type == pa.int16()
    assert output["negative"].to_pylist() == [-200, None]


@pytest.mark.parametrize(
    ("operation", "sql"),
    [("sqrt", 'sqrt("x")'), ("exp", 'exp("x")'), ("log", 'ln("x")')],
)
def test_float64_row_rendering_keeps_existing_sql(operation, sql):
    source = _input("float64")
    result = source.with_columns(
        FeatureSet([("result", getattr(row, operation)(source["x"]))])
    )
    program = Program("unchanged-double", inputs=[source], outputs=[("result", result)])
    document = lower_program_document(program, Runtime(), "batch")
    assert (
        document["graph"]["nodes"][0]["operator"]["select"][-1] == f'{sql} AS "result"'
    )
