from __future__ import annotations

import pyarrow as pa
import pytest

from calc_flow import Batch, Runtime
from calc_flow.errors import CompileError
from calc_flow.pipeline import BatchExecutionPlan, StreamExecutionPlan
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    TableExpr,
    table,
    table_input,
    ts,
)
from calc_flow.symbolic.lower import lower_program_document


def _ordered() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64", nullable=False),
            Field("v", "int64"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


def _program(features: list[tuple[str, object]]) -> Program:
    quotes = _ordered()
    signals = quotes.with_columns(FeatureSet(features))
    return Program("p", inputs=[quotes], outputs=[("signals", signals)])


def _rolling_nodes(document: dict[str, object]) -> list[dict[str, object]]:
    graph = document["graph"]
    return [
        node
        for node in graph["nodes"]  # type: ignore[index]
        if node["operator"]["kind"] == "rolling"  # type: ignore[index]
    ]


def _quotes_batch() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("x", pa.float64(), nullable=False),
            pa.field("v", pa.int64(), nullable=True),
        ]
    )
    rows = [
        (20, "a", 2, 2.0, 20),
        (10, "a", 1, 1.0, 10),
        (10, "b", 1, 5.0, 50),
        (11, "a", 3, 3.0, 30),
    ]
    return pa.table(
        {
            "ts": pa.array([row[0] for row in rows], type=pa.timestamp("us", tz="UTC")),
            "symbol": pa.array([row[1] for row in rows]),
            "seq": pa.array([row[2] for row in rows], type=pa.uint64()),
            "x": pa.array([row[3] for row in rows], type=pa.float64()),
            "v": pa.array([row[4] for row in rows], type=pa.int64()),
        },
        schema=schema,
    )


def test_lag_delta_lower_to_one_rolling_node_with_the_frozen_shape() -> None:
    quotes = _ordered()
    program = _program(
        [("prev", ts.lag(quotes["x"])), ("change", ts.delta(quotes["v"]))]
    )

    document = lower_program_document(program, Runtime(), "batch")

    rolling = _rolling_nodes(document)
    assert len(rolling) == 1
    node = rolling[0]
    assert node["id"] == "signals__cf_rolling"
    assert node["operator"] == {
        "kind": "rolling",
        "spec": {
            "configuration_version": 1,
            "state_layout_version": 1,
            "partition_by": ["symbol"],
            "event_time": "ts",
            "sequence_by": ["seq"],
            "outputs": [
                {
                    "kind": "lag",
                    "primitive_version": 1,
                    "input": "x",
                    "output": "prev",
                    "periods": 1,
                },
                {
                    "kind": "delta",
                    "primitive_version": 1,
                    "input": "v",
                    "output": "change",
                    "periods": 1,
                },
            ],
            "allowed_lateness_micros": 0,
            "late_policy": {"kind": "error", "scope": "envelope"},
            "value_policy": "stateful_numeric_v1",
        },
    }
    input_schema = [
        {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": False},
        {"name": "symbol", "data_type": "string", "nullable": False},
        {"name": "seq", "data_type": "uint64", "nullable": False},
        {"name": "x", "data_type": "float64", "nullable": False},
        {"name": "v", "data_type": "int64", "nullable": True},
    ]
    assert node["input_ports"] == [
        {"name": "input", "kind": "table", "required": True, "schema": input_schema}
    ]
    assert node["output_ports"] == [
        {
            "name": "output",
            "kind": "table",
            "required": True,
            "schema": [
                *input_schema,
                {"name": "prev", "data_type": "float64", "nullable": True},
                {"name": "change", "data_type": "int64", "nullable": True},
            ],
        }
    ]
    graph = document["graph"]
    assert (
        {
            "source_node": "signals__cf_rolling",
            "source_port": "output",
            "target_node": "signals",
            "target_port": "input",
        }
    ) in graph["edges"]  # type: ignore[index]


def test_stream_lowering_carries_the_lateness_arguments() -> None:
    quotes = _ordered()
    program = _program([("prev", ts.lag(quotes["x"]))])

    document = lower_program_document(
        program,
        Runtime(),
        "stream",
        allowed_lateness_micros=7,
        late_policy="drop",
    )

    rolling = _rolling_nodes(document)
    assert len(rolling) == 1
    spec = rolling[0]["operator"]["spec"]  # type: ignore[index]
    assert spec["allowed_lateness_micros"] == 7
    assert spec["late_policy"] == {"kind": "drop", "metrics_version": 1}


def test_nested_rolling_arguments_lower_through_a_materialized_column() -> None:
    quotes = _ordered()
    program = _program([("momentum", ts.delta(quotes["x"]) + 1.0)])

    document = lower_program_document(program, Runtime(), "batch")

    rolling = _rolling_nodes(document)
    assert len(rolling) == 1
    outputs = rolling[0]["operator"]["spec"]["outputs"]  # type: ignore[index]
    assert len(outputs) == 1
    assert outputs[0]["kind"] == "delta"
    assert outputs[0]["input"] == "x"
    materialized = outputs[0]["output"]
    assert materialized != "momentum"
    final = next(
        node
        for node in document["graph"]["nodes"]  # type: ignore[index]
        if node["id"] == "signals"
    )
    select = final["operator"]["select"]  # type: ignore[index]
    assert any(materialized in item and "momentum" in item for item in select)


def test_rolling_features_mix_with_row_local_expressions() -> None:
    quotes = _ordered()
    program = _program([("double", quotes["x"] * 2.0), ("prev", ts.lag(quotes["x"]))])

    document = lower_program_document(program, Runtime(), "batch")

    rolling = _rolling_nodes(document)
    assert len(rolling) == 1
    outputs = rolling[0]["operator"]["spec"]["outputs"]  # type: ignore[index]
    assert [output["output"] for output in outputs] == ["prev"]
    final = next(
        node
        for node in document["graph"]["nodes"]  # type: ignore[index]
        if node["id"] == "signals"
    )
    select = final["operator"]["select"]  # type: ignore[index]
    assert any("double" in item for item in select)
    assert '"prev"' in select


def test_composed_lag_argument_is_rejected_loudly() -> None:
    quotes = _ordered()
    program = _program([("prev", ts.lag(quotes["x"] + 1.0))])

    with pytest.raises(CompileError) as excinfo:
        lower_program_document(program, Runtime(), "batch")

    message = str(excinfo.value)
    assert "outputs.prev" in message or "outputs.signals" in message


def test_lag_requires_declared_ordering() -> None:
    quotes = table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("x", "float64", nullable=False),
        ],
    )
    signals = quotes.with_columns(FeatureSet([("prev", ts.lag(quotes["x"]))]))
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    with pytest.raises(CompileError, match="ordering_required"):
        lower_program_document(program, Runtime(), "batch")


def test_batch_execution_produces_lag_delta_rows() -> None:
    quotes = _ordered()
    program = _program(
        [("prev", ts.lag(quotes["x"])), ("change", ts.delta(quotes["v"]))]
    )

    plan = program.compile_batch(Runtime())

    assert isinstance(plan, BatchExecutionPlan)
    result = plan.execute({"input": Batch.from_pyarrow(_quotes_batch())})
    output = result.outputs["output"].to_pyarrow()
    assert output.schema.names == [
        "ts",
        "symbol",
        "seq",
        "x",
        "v",
        "prev",
        "change",
    ]
    values = output.drop_columns(["ts"]).to_pydict()
    assert values["symbol"] == ["a", "b", "a", "a"]
    assert values["seq"] == [1, 1, 3, 2]
    assert values["prev"] == [None, None, 1.0, 3.0]
    assert values["change"] == [None, None, 20, -10]


def test_stream_compile_lowers_a_rolling_program() -> None:
    quotes = _ordered()
    program = _program([("prev", ts.lag(quotes["x"]))])

    plan = program.compile_stream(
        Runtime(), allowed_lateness_micros=5, late_policy="drop"
    )

    assert isinstance(plan, StreamExecutionPlan)


def test_filter_below_rolling_feeds_a_prefilter_stage() -> None:
    quotes = _ordered()
    filtered = table.filter(quotes, quotes["x"] > 1.0)
    signals = filtered.with_columns(FeatureSet([("prev", ts.lag(quotes["x"]))]))
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")

    ids = [node["id"] for node in document["graph"]["nodes"]]  # type: ignore[index]
    assert ids == ["signals__cf_prefilter", "signals__cf_rolling", "signals"]
    prefilter = document["graph"]["nodes"][0]  # type: ignore[index]
    assert prefilter["operator"]["filter"] is not None

    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_quotes_batch())})
    values = result.outputs["output"].to_pyarrow().drop_columns(["ts"]).to_pydict()
    assert values["symbol"] == ["b", "a", "a"]
    assert values["prev"] == [None, None, 3.0]


def test_filter_above_rolling_applies_after_the_rolling_stage() -> None:
    quotes = _ordered()
    featured = quotes.with_columns(FeatureSet([("prev", ts.lag(quotes["x"]))]))
    signals = table.filter(featured, featured["prev"] > 1.0)
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")

    ids = [node["id"] for node in document["graph"]["nodes"]]  # type: ignore[index]
    assert ids == ["signals__cf_rolling", "signals"]

    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_quotes_batch())})
    values = result.outputs["output"].to_pyarrow().drop_columns(["ts"]).to_pydict()
    assert values["symbol"] == ["a"]
    assert values["seq"] == [2]
    assert values["prev"] == [3.0]


def test_rolling_capability_is_advertised_with_frozen_lifecycle_facts() -> None:
    capabilities = Runtime().capabilities()

    rolling = [
        operator for operator in capabilities.operators if operator.kind == "rolling"
    ]
    assert len(rolling) == 1
    operator = rolling[0]
    assert operator.version == "1"
    assert operator.modes == ("batch", "stream")
    assert operator.finality == "per_row_final"
    assert operator.stateful is True
    assert operator.microbatch_invariant is True
    assert operator.requires_watermark is True
    assert operator.checkpoint_support == "checkpointed_stateful"
    assert operator.state_version == 1
    assert operator.deterministic is True
    assert operator.replay_safe is True
