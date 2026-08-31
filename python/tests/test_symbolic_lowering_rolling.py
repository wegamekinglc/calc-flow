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
    duration,
    row,
    rows,
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


def test_row_local_rolling_operand_materializes_before_state() -> None:
    quotes = _ordered()
    program = _program([("previous_log", ts.lag(row.log(quotes["x"])))])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = document["graph"]["nodes"]  # type: ignore[index]
    assert [node["id"] for node in nodes] == [
        "signals__cf_rolling_input",
        "signals__cf_rolling",
        "signals",
    ]
    materialized = nodes[0]
    select = materialized["operator"]["select"]  # type: ignore[index]
    assert select[-1] == 'ln("x") AS "signals__cf_roll_input_0"'
    rolling = nodes[1]
    outputs = rolling["operator"]["spec"]["outputs"]  # type: ignore[index]
    assert outputs == [
        {
            "kind": "lag",
            "primitive_version": 1,
            "input": "signals__cf_roll_input_0",
            "output": "previous_log",
            "periods": 1,
        }
    ]

    result = program.compile_batch(Runtime()).execute(
        {"input": Batch.from_pyarrow(_quotes_batch())}
    )
    values = result.outputs["output"].to_pyarrow().drop_columns(["ts"]).to_pydict()
    assert values["symbol"] == ["a", "b", "a", "a"]
    assert values["previous_log"] == pytest.approx(
        [None, None, 0.0, 1.0986122886681098], nan_ok=True
    )


def test_identical_row_local_rolling_operands_materialize_once() -> None:
    quotes = _ordered()
    logged = row.log(quotes["x"])
    program = _program(
        [
            ("previous_log", ts.lag(logged)),
            ("mean_log", ts.mean(logged, window=rows(2))),
        ]
    )

    document = lower_program_document(program, Runtime(), "batch")

    nodes = document["graph"]["nodes"]  # type: ignore[index]
    select = nodes[0]["operator"]["select"]  # type: ignore[index]
    assert select.count('ln("x") AS "signals__cf_roll_input_0"') == 1
    outputs = nodes[1]["operator"]["spec"]["outputs"]  # type: ignore[index]
    assert [output["input"] for output in outputs] == [
        "signals__cf_roll_input_0",
        "signals__cf_roll_input_0",
    ]


def test_nested_rolling_operand_builds_two_state_stages_and_executes() -> None:
    quotes = _ordered()
    mean_change = ts.mean(ts.delta(quotes["x"]), window=rows(2))
    program = _program([("mean_change", mean_change)])

    document = lower_program_document(program, Runtime(), "batch")

    rolling = _rolling_nodes(document)
    assert [node["id"] for node in rolling] == [
        "signals__cf_rolling_1",
        "signals__cf_rolling_2",
    ]
    first_output = rolling[0]["operator"]["spec"]["outputs"][0]  # type: ignore[index]
    second_output = rolling[1]["operator"]["spec"]["outputs"][0]  # type: ignore[index]
    assert first_output["kind"] == "delta"
    assert second_output == {
        "kind": "mean",
        "primitive_version": 1,
        "input": first_output["output"],
        "output": "mean_change",
        "frame": {"kind": "rows", "size": 2},
        "min_periods": 1,
    }

    result = program.compile_batch(Runtime()).execute(
        {"input": Batch.from_pyarrow(_quotes_batch())}
    )
    values = result.outputs["output"].to_pyarrow().drop_columns(["ts"]).to_pydict()
    assert values["mean_change"] == [None, None, 2.0, 0.5]


def test_row_local_bridge_materializes_between_rolling_stages() -> None:
    quotes = _ordered()
    change = ts.delta(quotes["x"])
    gain = row.where(change > 0.0, change, 0.0)
    program = _program([("average_gain", ts.mean(gain, window=rows(2)))])

    document = lower_program_document(program, Runtime(), "stream")

    ids = [
        node["id"]
        for node in document["graph"]["nodes"]  # type: ignore[index]
    ]
    assert ids == [
        "signals__cf_rolling_1",
        "signals__cf_rolling_input_2",
        "signals__cf_rolling_2",
        "signals",
    ]
    program.compile_stream(Runtime())


def test_row_local_rolling_operands_execute_across_scalar_shapes() -> None:
    quotes = _ordered()
    conditional = row.where(quotes["x"] > 1.0, quotes["x"], 0.0)
    filled = row.coalesce(row.cast(quotes["v"], "float64"), 0.0)
    program = _program(
        [
            ("previous_conditional", ts.lag(conditional)),
            ("previous_filled", ts.lag(filled)),
        ]
    )

    result = program.compile_batch(Runtime()).execute(
        {"input": Batch.from_pyarrow(_quotes_batch())}
    )

    values = result.outputs["output"].to_pyarrow().drop_columns(["ts"]).to_pydict()
    assert values["previous_conditional"] == [None, None, 0.0, 3.0]
    assert values["previous_filled"] == [None, None, 10.0, 30.0]


def test_materialized_rolling_input_collision_is_rejected() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("signals__cf_roll_input_0", quotes["x"]),
            ("previous_adjusted", ts.lag(quotes["x"] + 1.0)),
        ]
    )

    with pytest.raises(CompileError, match="duplicate_name"):
        lower_program_document(program, Runtime(), "batch")


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


def test_materialized_rolling_column_collision_is_rejected() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("signals__cf_roll_0", quotes["x"]),
            ("momentum", ts.delta(quotes["x"]) + 1.0),
        ]
    )

    with pytest.raises(CompileError, match="duplicate_name"):
        lower_program_document(program, Runtime(), "batch")


def test_composed_lag_argument_is_materialized() -> None:
    quotes = _ordered()
    program = _program([("prev", ts.lag(quotes["x"] + 1.0))])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = document["graph"]["nodes"]  # type: ignore[index]
    assert [node["id"] for node in nodes[:2]] == [
        "signals__cf_rolling_input",
        "signals__cf_rolling",
    ]
    program.compile_batch(Runtime())


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


def test_filter_below_materialized_rolling_preserves_stage_order() -> None:
    quotes = _ordered()
    filtered = table.filter(quotes, quotes["x"] > 1.0)
    signals = filtered.with_columns(
        FeatureSet((("previous_log", ts.lag(row.log(quotes["x"]))),))
    )
    program = Program("p", inputs=(quotes,), outputs=(("signals", signals),))

    document = lower_program_document(program, Runtime(), "batch")

    ids = [
        node["id"]
        for node in document["graph"]["nodes"]  # type: ignore[index]
    ]
    assert ids == [
        "signals__cf_prefilter",
        "signals__cf_rolling_input",
        "signals__cf_rolling",
        "signals",
    ]


def test_materialized_rolling_outputs_connect_through_input_fanout() -> None:
    quotes = _ordered()
    first = quotes.with_columns(
        FeatureSet((("previous_log", ts.lag(row.log(quotes["x"]))),))
    )
    second = quotes.with_columns(
        FeatureSet((("previous_adjusted", ts.lag(quotes["x"] + 1.0)),))
    )
    program = Program(
        "p",
        inputs=(quotes,),
        outputs=(("first", first), ("second", second)),
    )

    document = lower_program_document(program, Runtime(), "batch")

    edges = document["graph"]["edges"]  # type: ignore[index]
    targets = {edge["target_node"] for edge in edges if edge["source_node"] == "quotes"}
    assert targets == {"first__cf_rolling_input", "second__cf_rolling_input"}


def test_identical_multi_stage_rolling_pipelines_share_physical_state() -> None:
    quotes = _ordered()
    first_value = ts.mean(ts.delta(quotes["x"]), window=rows(2))
    second_value = ts.mean(ts.delta(quotes["x"]), window=rows(2))
    first = quotes.with_columns(FeatureSet((("first_mean", first_value),)))
    second = quotes.with_columns(FeatureSet((("second_mean", second_value),)))
    program = Program(
        "p",
        inputs=(quotes,),
        outputs=(("first", first), ("second", second)),
    )

    document = lower_program_document(program, Runtime(), "batch")

    rolling = _rolling_nodes(document)
    assert [node["id"] for node in rolling] == [
        "first__cf_shared_rolling_1",
        "first__cf_shared_rolling_2",
    ]
    targets = {
        edge["target_node"]
        for edge in document["graph"]["edges"]  # type: ignore[index]
        if edge["source_node"] == "first__cf_shared_rolling_2"
    }
    assert targets == {"first", "second"}

    result = program.compile_batch(Runtime()).execute(
        {"input": Batch.from_pyarrow(_quotes_batch())}
    )
    assert result.outputs["first.output"].to_pyarrow()["first_mean"].to_pylist() == [
        None,
        None,
        2.0,
        0.5,
    ]
    assert result.outputs["second.output"].to_pyarrow()["second_mean"].to_pylist() == [
        None,
        None,
        2.0,
        0.5,
    ]


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


def test_lower_program_document_validates_lateness_when_rolling_present() -> None:
    quotes = _ordered()
    program = _program([("prev", ts.lag(quotes["x"]))])

    with pytest.raises(ValueError, match="non-negative"):
        lower_program_document(program, Runtime(), "stream", allowed_lateness_micros=-1)
    with pytest.raises(ValueError, match="unsigned"):
        lower_program_document(
            program, Runtime(), "stream", allowed_lateness_micros=1 << 64
        )
    with pytest.raises(TypeError):
        lower_program_document(
            program, Runtime(), "stream", allowed_lateness_micros=True
        )
    with pytest.raises(ValueError, match="late_policy"):
        lower_program_document(program, Runtime(), "stream", late_policy="retry")
    with pytest.raises(TypeError):
        lower_program_document(program, Runtime(), "stream", late_policy=1)


def test_row_local_program_ignores_lateness_arguments() -> None:
    quotes = _ordered()
    signals = quotes.with_columns(FeatureSet([("double", quotes["x"] * 2.0)]))
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(
        program, Runtime(), "stream", allowed_lateness_micros=-1, late_policy="bogus"
    )

    assert document["graph"]["nodes"]


def test_compile_stream_rejects_out_of_range_lateness() -> None:
    quotes = _ordered()
    program = _program([("prev", ts.lag(quotes["x"]))])

    with pytest.raises(ValueError, match="unsigned"):
        program.compile_stream(Runtime(), allowed_lateness_micros=1 << 64)


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


def test_aggregates_lower_to_one_rolling_node_with_the_frozen_shape() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("n", ts.count(quotes["x"], window=rows(2))),
            ("total", ts.sum(quotes["v"], window=rows(2), min_periods=2)),
            ("avg", ts.mean(quotes["x"], window=rows(2))),
            ("var", ts.variance(quotes["x"], window=rows(2), ddof=1)),
            ("std", ts.stddev(quotes["x"], window=rows(2), ddof=0)),
        ]
    )

    document = lower_program_document(program, Runtime(), "batch")

    rolling = _rolling_nodes(document)
    assert len(rolling) == 1
    spec = rolling[0]["operator"]["spec"]  # type: ignore[index]
    assert spec["outputs"] == [
        {
            "kind": "count",
            "primitive_version": 1,
            "input": "x",
            "output": "n",
            "frame": {"kind": "rows", "size": 2},
            "min_periods": 1,
        },
        {
            "kind": "sum",
            "primitive_version": 1,
            "input": "v",
            "output": "total",
            "frame": {"kind": "rows", "size": 2},
            "min_periods": 2,
        },
        {
            "kind": "mean",
            "primitive_version": 1,
            "input": "x",
            "output": "avg",
            "frame": {"kind": "rows", "size": 2},
            "min_periods": 1,
        },
        {
            "kind": "variance",
            "primitive_version": 1,
            "input": "x",
            "output": "var",
            "frame": {"kind": "rows", "size": 2},
            "min_periods": 1,
            "ddof": 1,
        },
        {
            "kind": "stddev",
            "primitive_version": 1,
            "input": "x",
            "output": "std",
            "frame": {"kind": "rows", "size": 2},
            "min_periods": 1,
            "ddof": 0,
        },
    ]
    output_schema = rolling[0]["output_ports"][0]["schema"]  # type: ignore[index]
    derived = output_schema[len(input_schema_fields()) :]
    assert derived == [
        {"name": "n", "data_type": "uint64", "nullable": True},
        {"name": "total", "data_type": "int64", "nullable": True},
        {"name": "avg", "data_type": "float64", "nullable": True},
        {"name": "var", "data_type": "float64", "nullable": True},
        {"name": "std", "data_type": "float64", "nullable": True},
    ]


def input_schema_fields() -> list[dict[str, object]]:
    return [
        {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": False},
        {"name": "symbol", "data_type": "string", "nullable": False},
        {"name": "seq", "data_type": "uint64", "nullable": False},
        {"name": "x", "data_type": "float64", "nullable": False},
        {"name": "v", "data_type": "int64", "nullable": True},
    ]


def test_duration_and_pair_lower_to_the_frozen_rolling_shape() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("m", ts.mean(quotes["x"], window=duration(60_000_000))),
            ("peak", ts.max(quotes["x"], window=duration(60_000_000))),
            ("corr", ts.correlation(quotes["x"], quotes["v"], window=rows(4))),
        ]
    )

    document = lower_program_document(program, Runtime(), "batch")

    rolling = _rolling_nodes(document)
    assert len(rolling) == 1
    spec = rolling[0]["operator"]["spec"]  # type: ignore[index]
    assert spec["outputs"] == [
        {
            "kind": "mean",
            "primitive_version": 1,
            "input": "x",
            "output": "m",
            "frame": {"kind": "duration", "micros": 60_000_000},
            "min_periods": 1,
        },
        {
            "kind": "max",
            "primitive_version": 1,
            "input": "x",
            "output": "peak",
            "frame": {"kind": "duration", "micros": 60_000_000},
            "min_periods": 1,
        },
        {
            "kind": "correlation",
            "primitive_version": 1,
            "left": "x",
            "right": "v",
            "output": "corr",
            "frame": {"kind": "rows", "size": 4},
            "min_periods": 1,
            "ddof": 1,
        },
    ]
    output_schema = rolling[0]["output_ports"][0]["schema"]  # type: ignore[index]
    derived = output_schema[len(input_schema_fields()) :]
    assert derived == [
        {"name": "m", "data_type": "float64", "nullable": True},
        {"name": "peak", "data_type": "float64", "nullable": True},
        {"name": "corr", "data_type": "float64", "nullable": True},
    ]


def test_batch_execution_produces_aggregate_rows() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("n", ts.count(quotes["x"], window=rows(2))),
            ("avg", ts.mean(quotes["x"], window=rows(2))),
            ("var", ts.variance(quotes["x"], window=rows(2), ddof=1)),
            ("total", ts.sum(quotes["v"], window=rows(2))),
        ]
    )

    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_quotes_batch())})
    output = result.outputs["output"].to_pyarrow()
    assert output.schema.names == [
        "ts",
        "symbol",
        "seq",
        "x",
        "v",
        "n",
        "avg",
        "var",
        "total",
    ]
    values = output.drop_columns(["ts"]).to_pydict()
    assert values["symbol"] == ["a", "b", "a", "a"]
    assert values["seq"] == [1, 1, 3, 2]
    assert values["n"] == [1, 1, 2, 2]
    assert values["avg"] == [1.0, 5.0, 2.0, 2.5]
    assert values["var"] == [None, None, 2.0, 0.5]
    assert values["total"] == [10, 50, 40, 50]


def test_stream_compile_lowers_an_aggregate_program() -> None:
    quotes = _ordered()
    program = _program([("avg", ts.mean(quotes["x"], window=rows(3)))])

    plan = program.compile_stream(
        Runtime(), allowed_lateness_micros=5, late_policy="drop"
    )

    assert isinstance(plan, StreamExecutionPlan)
