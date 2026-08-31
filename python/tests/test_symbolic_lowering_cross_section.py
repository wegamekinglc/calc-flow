from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

from calc_flow import Batch, Runtime
from calc_flow.capabilities import RuntimeCapabilities
from calc_flow.errors import CompileError
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    TableExpr,
    cs,
    event_time_bucket,
    exact_time,
    row,
    rows,
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
            Field("industry", "string", nullable=True),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64", nullable=True),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


def _group(table: TableExpr) -> object:
    return exact_time(table["ts"], partition_by=[table["industry"]])


def _program(features: list[tuple[str, object]]) -> Program:
    quotes = _ordered()
    signals = quotes.with_columns(FeatureSet(features))
    return Program("p", inputs=[quotes], outputs=[("signals", signals)])


def _cross_section_nodes(document: dict[str, object]) -> list[dict[str, object]]:
    graph = document["graph"]
    return [
        node
        for node in graph["nodes"]  # type: ignore[index]
        if node["operator"]["kind"] == "cross_section"  # type: ignore[index]
    ]


def _runtime_with_cross_section_capability(
    *, drop: bool = False, **changes: object
) -> Runtime:
    class CapabilityRuntime(Runtime):
        def capabilities(self) -> RuntimeCapabilities:
            base = super().capabilities()
            operators = tuple(
                dataclasses.replace(operator, **changes)
                if operator.kind == "cross_section" and not drop
                else operator
                for operator in base.operators
                if operator.kind != "cross_section" or not drop
            )
            return dataclasses.replace(base, operators=operators)

    return CapabilityRuntime()


def test_rank_zscore_lower_to_one_cross_section_node_with_the_frozen_shape() -> None:
    quotes = _ordered()
    program = _program(
        [
            (
                "rank",
                cs.rank(quotes["x"], group=_group(quotes), tie_method="min"),
            ),
            ("alpha", cs.zscore(quotes["x"], group=_group(quotes), ddof=0)),
        ]
    )

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _cross_section_nodes(document)
    assert len(nodes) == 1
    node = nodes[0]
    assert node["id"] == "signals__cf_cross_section"
    assert node["operator"] == {
        "kind": "cross_section",
        "spec": {
            "configuration_version": 1,
            "state_layout_version": 1,
            "event_time": "ts",
            "entity_by": ["symbol"],
            "partition_by": ["industry"],
            "sequence_by": ["seq"],
            "grouping": {"kind": "exact_time"},
            "outputs": [
                {
                    "kind": "rank",
                    "primitive_version": 1,
                    "input": "x",
                    "output": "rank",
                    "direction": "ascending",
                    "tie_method": "min",
                    "null_placement": "exclude",
                    "min_samples": 1,
                },
                {
                    "kind": "zscore",
                    "primitive_version": 1,
                    "input": "x",
                    "output": "alpha",
                    "min_samples": 1,
                    "ddof": 0,
                },
            ],
            "allowed_lateness_micros": 0,
            "late_policy": {"kind": "error", "scope": "envelope"},
            "value_policy": "nan_exclude_preserve_v1",
        },
    }
    input_schema = [
        {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": False},
        {"name": "symbol", "data_type": "string", "nullable": False},
        {"name": "industry", "data_type": "string", "nullable": True},
        {"name": "seq", "data_type": "uint64", "nullable": False},
        {"name": "x", "data_type": "float64", "nullable": True},
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
                {"name": "rank", "data_type": "float64", "nullable": True},
                {"name": "alpha", "data_type": "float64", "nullable": True},
            ],
        }
    ]
    graph = document["graph"]
    assert (
        {
            "source_node": "signals__cf_cross_section",
            "source_port": "output",
            "target_node": "signals",
            "target_port": "input",
        }
        in graph["edges"]  # type: ignore[index]
    )


def test_nested_cross_section_materializes_before_final_rewrite() -> None:
    quotes = _ordered()
    program = _program(
        [("adjusted_rank", cs.rank(quotes["x"], group=_group(quotes)) + 1.0)]
    )

    document = lower_program_document(program, Runtime(), "batch")

    (node,) = _cross_section_nodes(document)
    outputs = node["operator"]["spec"]["outputs"]  # type: ignore[index]
    assert outputs[0]["kind"] == "rank"
    assert outputs[0]["output"] == "signals__cf_cs_0"
    final = next(
        graph_node
        for graph_node in document["graph"]["nodes"]  # type: ignore[index]
        if graph_node["id"] == "signals"
    )
    assert (
        '("signals__cf_cs_0" + 1.0) AS "adjusted_rank"' in final["operator"]["select"]  # type: ignore[index]
    )


def test_row_local_cross_section_operand_materializes_before_grouping() -> None:
    quotes = _ordered()
    adjusted = row.coalesce(quotes["x"], 0.0) + 1.0
    program = _program([("adjusted_rank", cs.rank(adjusted, group=_group(quotes)))])

    document = lower_program_document(program, Runtime(), "stream")

    nodes = document["graph"]["nodes"]  # type: ignore[index]
    assert [node["id"] for node in nodes] == [
        "signals__cf_cross_section_input",
        "signals__cf_cross_section",
        "signals",
    ]
    select = nodes[0]["operator"]["select"]  # type: ignore[index]
    assert select[-1].endswith('AS "signals__cf_cs_input_0"')
    output = nodes[1]["operator"]["spec"]["outputs"][0]  # type: ignore[index]
    assert output["input"] == "signals__cf_cs_input_0"
    program.compile_stream(Runtime())


def test_multi_stage_rolling_result_materializes_before_cross_section() -> None:
    quotes = _ordered()
    momentum = ts.mean(ts.delta(quotes["x"]), window=rows(2))
    adjusted = row.coalesce(momentum, 0.0) + 1.0
    program = _program([("momentum_rank", cs.rank(adjusted, group=_group(quotes)))])

    document = lower_program_document(program, Runtime(), "stream")

    nodes = document["graph"]["nodes"]  # type: ignore[index]
    assert [node["id"] for node in nodes] == [
        "signals__cf_rolling_1",
        "signals__cf_rolling_2",
        "signals__cf_cross_section_input",
        "signals__cf_cross_section",
        "signals",
    ]
    edges = document["graph"]["edges"]  # type: ignore[index]
    assert [(edge["source_node"], edge["target_node"]) for edge in edges] == [
        ("signals__cf_rolling_1", "signals__cf_rolling_2"),
        ("signals__cf_rolling_2", "signals__cf_cross_section_input"),
        ("signals__cf_cross_section_input", "signals__cf_cross_section"),
        ("signals__cf_cross_section", "signals"),
    ]
    program.compile_stream(Runtime())


def test_identical_row_local_cross_section_operands_materialize_once() -> None:
    quotes = _ordered()
    adjusted = row.coalesce(quotes["x"], 0.0) + 1.0
    program = _program(
        [
            ("adjusted_rank", cs.rank(adjusted, group=_group(quotes))),
            ("adjusted_zscore", cs.zscore(adjusted, group=_group(quotes))),
        ]
    )

    document = lower_program_document(program, Runtime(), "batch")

    nodes = document["graph"]["nodes"]  # type: ignore[index]
    select = nodes[0]["operator"]["select"]  # type: ignore[index]
    assert len([item for item in select if "signals__cf_cs_input_0" in item]) == 1
    outputs = nodes[1]["operator"]["spec"]["outputs"]  # type: ignore[index]
    assert [output["input"] for output in outputs] == [
        "signals__cf_cs_input_0",
        "signals__cf_cs_input_0",
    ]


def test_row_local_cross_section_branches_share_materialization_and_state() -> None:
    quotes = _ordered()
    first = quotes.with_columns(
        FeatureSet((("rank", cs.rank(quotes["x"] + 1.0, group=_group(quotes))),))
    )
    second = quotes.with_columns(
        FeatureSet(
            (
                (
                    "zscore",
                    cs.zscore(
                        row.coalesce(quotes["x"], 0.0),
                        group=_group(quotes),
                    ),
                ),
            )
        )
    )
    program = Program(
        "p",
        inputs=(quotes,),
        outputs=(("first", first), ("second", second)),
    )

    document = lower_program_document(program, Runtime(), "batch")

    nodes = document["graph"]["nodes"]  # type: ignore[index]
    shared_input = next(
        node for node in nodes if node["id"] == "first__cf_shared_cross_section_input"
    )
    assert len(shared_input["operator"]["select"]) == 7  # type: ignore[index]
    cross = _cross_section_nodes(document)
    assert [node["id"] for node in cross] == ["first__cf_shared_cross_section"]
    assert len(cross[0]["operator"]["spec"]["outputs"]) == 2  # type: ignore[index]
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("industry", pa.string(), nullable=True),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("x", pa.float64(), nullable=True),
        ]
    )
    table = pa.table(
        {
            "ts": [100, 100, 100],
            "symbol": ["a", "b", "c"],
            "industry": ["tech", "tech", "tech"],
            "seq": [1, 2, 3],
            "x": [1.0, 2.0, None],
        },
        schema=schema,
    )
    result = program.compile_batch(Runtime()).execute(
        {"input": Batch.from_pyarrow(table)}
    )
    assert result.outputs["first.output"].to_pyarrow().num_rows == 3
    assert result.outputs["second.output"].to_pyarrow().num_rows == 3


def test_bucketed_grouping_writes_the_fixed_width_shape() -> None:
    quotes = _ordered()
    bucketed = event_time_bucket(
        quotes["ts"], width_micros=60000000, partition_by=[quotes["industry"]]
    )
    program = _program([("rank", cs.rank(quotes["x"], group=bucketed))])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _cross_section_nodes(document)
    assert len(nodes) == 1
    assert nodes[0]["operator"]["spec"]["grouping"] == {  # type: ignore[index]
        "kind": "fixed_bucket",
        "width_micros": 60000000,
    }


def test_percentile_and_demean_carry_their_own_options() -> None:
    quotes = _ordered()
    program = _program(
        [
            (
                "pct",
                cs.percentile(
                    quotes["x"],
                    group=_group(quotes),
                    direction="descending",
                    tie_method="max",
                    null_placement="last",
                ),
            ),
            ("resid", cs.demean(quotes["x"], group=_group(quotes), min_samples=2)),
        ]
    )

    document = lower_program_document(program, Runtime(), "batch")

    outputs = _cross_section_nodes(document)[0]["operator"]["spec"]["outputs"]  # type: ignore[index]
    assert outputs[0] == {
        "kind": "percentile",
        "primitive_version": 1,
        "input": "x",
        "output": "pct",
        "direction": "descending",
        "tie_method": "max",
        "null_placement": "last",
        "min_samples": 1,
    }
    assert outputs[1] == {
        "kind": "demean",
        "primitive_version": 1,
        "input": "x",
        "output": "resid",
        "min_samples": 2,
    }


def test_winsorize_lowers_with_bounds_and_preserves_the_input_type() -> None:
    quotes = _ordered()
    program = _program(
        [("w", cs.winsorize(quotes["x"], group=_group(quotes), lower=0.1, upper=0.9))]
    )

    document = lower_program_document(program, Runtime(), "batch")

    (node,) = _cross_section_nodes(document)
    assert node["operator"]["spec"]["outputs"] == [  # type: ignore[index]
        {
            "kind": "winsorize",
            "primitive_version": 1,
            "input": "x",
            "output": "w",
            "min_samples": 1,
            "lower": 0.1,
            "upper": 0.9,
        }
    ]
    assert node["output_ports"][0]["schema"][-1] == {  # type: ignore[index]
        "name": "w",
        "data_type": "float64",
        "nullable": True,
    }


def test_top_bottom_and_mean_fill_lower_to_one_shared_group_stage() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("top", cs.top(quotes["x"], group=_group(quotes), count=3)),
            (
                "bottom",
                cs.bottom(
                    quotes["x"],
                    group=_group(quotes),
                    count=2,
                    include_ties=False,
                    min_samples=4,
                ),
            ),
            ("filled", cs.mean_fill(quotes["x"], group=_group(quotes))),
        ]
    )

    document = lower_program_document(program, Runtime(), "batch")

    (node,) = _cross_section_nodes(document)
    assert node["operator"]["spec"]["outputs"] == [  # type: ignore[index]
        {
            "kind": "top",
            "primitive_version": 1,
            "input": "x",
            "output": "top",
            "count": 3,
            "include_ties": True,
            "min_samples": 1,
        },
        {
            "kind": "bottom",
            "primitive_version": 1,
            "input": "x",
            "output": "bottom",
            "count": 2,
            "include_ties": False,
            "min_samples": 4,
        },
        {
            "kind": "mean_fill",
            "primitive_version": 1,
            "input": "x",
            "output": "filled",
            "min_samples": 1,
        },
    ]
    assert node["output_ports"][0]["schema"][-3:] == [  # type: ignore[index]
        {"name": "top", "data_type": "bool", "nullable": True},
        {"name": "bottom", "data_type": "bool", "nullable": True},
        {"name": "filled", "data_type": "float64", "nullable": True},
    ]


def test_missing_ordering_keys_are_rejected() -> None:
    unordered = table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("industry", "string", nullable=True),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64", nullable=True),
        ],
    )
    signals = unordered.with_columns(
        FeatureSet([("rank", cs.rank(unordered["x"], group=_group(unordered)))])
    )
    program = Program("p", inputs=[unordered], outputs=[("signals", signals)])

    with pytest.raises(CompileError) as error:
        lower_program_document(program, Runtime(), "batch")
    assert "ordering" in str(error.value).lower()


def test_derived_row_local_column_argument_is_materialized() -> None:
    quotes = _ordered()
    derived = quotes.with_columns(FeatureSet([("y", quotes["x"] + 1.0)]))
    group = exact_time(derived["ts"], partition_by=[derived["industry"]])
    signals = derived.with_columns(
        FeatureSet([("rank", cs.rank(derived["y"], group=group))])
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = document["graph"]["nodes"]  # type: ignore[index]
    assert [node["id"] for node in nodes] == [
        "signals__cf_cross_section_input",
        "signals__cf_cross_section",
        "signals",
    ]
    select = nodes[0]["operator"]["select"]  # type: ignore[index]
    assert select[-1] == '("x" + 1.0) AS "signals__cf_cs_input_0"'
    program.compile_batch(Runtime())


def test_non_input_group_expression_is_rejected() -> None:
    quotes = _ordered()
    group = exact_time(quotes["ts"], partition_by=[quotes["x"] + 1.0])
    signals = quotes.with_columns(
        FeatureSet([("rank", cs.rank(quotes["x"], group=group))])
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    with pytest.raises(CompileError, match="group columns must be input columns"):
        lower_program_document(program, Runtime(), "batch")


def test_materialized_cross_section_input_collision_is_rejected() -> None:
    quotes = _ordered()
    program = _program(
        [
            ("signals__cf_cs_input_0", quotes["x"]),
            (
                "rank",
                cs.rank(quotes["x"] + 1.0, group=_group(quotes)),
            ),
        ]
    )

    with pytest.raises(CompileError, match="duplicate_name"):
        lower_program_document(program, Runtime(), "batch")


def test_mixed_grouping_declarations_are_rejected() -> None:
    quotes = _ordered()
    bucketed = event_time_bucket(quotes["ts"], width_micros=1000)
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("rank", cs.rank(quotes["x"], group=_group(quotes))),
                ("other", cs.zscore(quotes["x"], group=bucketed)),
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    with pytest.raises(CompileError) as error:
        lower_program_document(program, Runtime(), "batch")
    assert "one grouping declaration" in str(error.value)


def test_mixed_grouping_kinds_with_equal_partitions_are_rejected() -> None:
    # The grouping kind is part of the one grouping declaration an output may
    # carry: silently reusing the first occurrence's kind would replace the
    # user-declared grouping of the later primitives.
    quotes = _ordered()
    bucketed = event_time_bucket(
        quotes["ts"], width_micros=1000, partition_by=[quotes["industry"]]
    )
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("rank", cs.rank(quotes["x"], group=_group(quotes))),
                ("other", cs.zscore(quotes["x"], group=bucketed)),
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    with pytest.raises(CompileError) as error:
        lower_program_document(program, Runtime(), "batch")
    assert "one grouping declaration" in str(error.value)


def test_mixed_bucket_widths_with_equal_partitions_are_rejected() -> None:
    quotes = _ordered()
    narrow = event_time_bucket(
        quotes["ts"], width_micros=1000, partition_by=[quotes["industry"]]
    )
    wide = event_time_bucket(
        quotes["ts"], width_micros=60_000_000, partition_by=[quotes["industry"]]
    )
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("rank", cs.rank(quotes["x"], group=narrow)),
                ("other", cs.zscore(quotes["x"], group=wide)),
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    with pytest.raises(CompileError) as error:
        lower_program_document(program, Runtime(), "batch")
    assert "one grouping declaration" in str(error.value)


def test_cross_section_capability_requires_the_operator_and_selected_mode() -> None:
    quotes = _ordered()
    program = _program([("rank", cs.rank(quotes["x"], group=_group(quotes)))])

    with pytest.raises(CompileError, match="does not offer the cross-section"):
        lower_program_document(
            program, _runtime_with_cross_section_capability(drop=True), "stream"
        )
    with pytest.raises(CompileError, match="does not support stream mode"):
        lower_program_document(
            program,
            _runtime_with_cross_section_capability(modes=("batch",)),
            "stream",
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"finality": "unproven"},
        {"stateful": False},
        {"microbatch_invariant": False},
        {"checkpoint_support": "unproven", "state_version": None},
        {"deterministic": False},
        {"replay_safe": False},
    ],
)
def test_cross_section_stream_capability_requires_every_lifecycle_fact(
    changes: dict[str, object],
) -> None:
    quotes = _ordered()
    program = _program([("rank", cs.rank(quotes["x"], group=_group(quotes)))])

    with pytest.raises(CompileError, match="does not prove stream lifecycle facts"):
        lower_program_document(
            program, _runtime_with_cross_section_capability(**changes), "stream"
        )


def test_cross_section_lowering_compiles_and_executes_in_batch_mode() -> None:
    quotes = _ordered()
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("rank", cs.rank(quotes["x"], group=_group(quotes))),
                ("resid", cs.demean(quotes["x"], group=_group(quotes))),
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    runtime = Runtime()
    plan = program.compile_batch(runtime)
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("industry", pa.string(), nullable=True),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("x", pa.float64(), nullable=True),
        ]
    )
    table = pa.table(
        {
            "ts": pa.array([100, 100, 100, 200], type=pa.timestamp("us", tz="UTC")),
            "symbol": pa.array(["a", "b", "c", "a"]),
            "industry": pa.array(["tech", "tech", "tech", "tech"]),
            "seq": pa.array([1, 2, 3, 1], type=pa.uint64()),
            "x": pa.array([2.0, 2.0, 1.0, 5.0], type=pa.float64()),
        },
        schema=schema,
    )
    outputs = plan.execute({"input": Batch.from_pyarrow(table)})
    result = outputs.outputs["output"].to_pyarrow()
    assert result.column("rank").to_pylist() == [2.5, 2.5, 1.0, 1.0]
    demean = result.column("resid").to_pylist()
    assert demean[3] == 0.0
    assert abs(demean[0] - (1.0 / 3.0)) < 1e-10
    assert abs(demean[2] - (-2.0 / 3.0)) < 1e-10


def test_grouped_features_execute_with_partition_ties_and_missing_values() -> None:
    quotes = _ordered()
    group = _group(quotes)
    signals = quotes.with_columns(
        FeatureSet(
            [
                (
                    "winsorized",
                    cs.winsorize(quotes["x"], group=group, lower=0.25, upper=0.75),
                ),
                ("top", cs.top(quotes["x"], group=group, count=2)),
                (
                    "bottom",
                    cs.bottom(quotes["x"], group=group, count=2, include_ties=False),
                ),
                ("filled", cs.mean_fill(quotes["x"], group=group)),
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("industry", pa.string(), nullable=True),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("x", pa.float64(), nullable=True),
        ]
    )
    table = pa.table(
        {
            "ts": pa.array([100] * 7, type=pa.timestamp("us", tz="UTC")),
            "symbol": pa.array(["a", "b", "c", "d", "e", "f", "g"]),
            "industry": pa.array(["tech"] * 6 + ["fin"]),
            "seq": pa.array(range(1, 8), type=pa.uint64()),
            "x": pa.array([0.0, 10.0, 20.0, 20.0, None, float("nan"), 100.0]),
        },
        schema=schema,
    )

    outputs = program.compile_batch(Runtime()).execute(
        {"input": Batch.from_pyarrow(table)}
    )
    result = outputs.outputs["output"].to_pyarrow()

    assert result.column("symbol").to_pylist() == ["g", "a", "b", "c", "d", "e", "f"]
    assert result.column("top").to_pylist() == [
        True,
        False,
        False,
        True,
        True,
        None,
        None,
    ]
    assert result.column("bottom").to_pylist() == [
        True,
        True,
        True,
        False,
        False,
        None,
        None,
    ]
    assert result.column("filled").to_pylist()[:6] == [
        100.0,
        0.0,
        10.0,
        20.0,
        20.0,
        12.5,
    ]
    assert (
        result.column("filled").to_pylist()[6] != result.column("filled").to_pylist()[6]
    )


def test_winsorize_and_mean_fill_lower_as_float32_for_float32_inputs() -> None:
    quotes = table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float32", nullable=True),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )
    group = exact_time(quotes["ts"])
    signals = quotes.with_columns(
        FeatureSet(
            [
                (
                    "winsorized",
                    cs.winsorize(quotes["x"], group=group, lower=0.1, upper=0.9),
                ),
                ("filled", cs.mean_fill(quotes["x"], group=group)),
            ]
        )
    )
    document = lower_program_document(
        Program("p", inputs=[quotes], outputs=[("signals", signals)]),
        Runtime(),
        "batch",
    )

    (node,) = _cross_section_nodes(document)
    assert node["output_ports"][0]["schema"][-2:] == [  # type: ignore[index]
        {"name": "winsorized", "data_type": "float32", "nullable": True},
        {"name": "filled", "data_type": "float32", "nullable": True},
    ]


def test_rolling_outputs_feed_the_cross_section_stage() -> None:
    quotes = _ordered()
    momentum = ts.mean(quotes["x"], window=rows(2))
    group = _group(quotes)
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("momentum", momentum),
                ("rank", cs.rank(quotes["x"], group=group)),
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")
    graph = document["graph"]
    kinds = [
        node["operator"]["kind"]  # type: ignore[index]
        for node in graph["nodes"]  # type: ignore[index]
    ]
    assert "rolling" in kinds and "cross_section" in kinds
    assert (
        {
            "source_node": "signals__cf_rolling",
            "source_port": "output",
            "target_node": "signals__cf_cross_section",
            "target_port": "input",
        }
        in graph["edges"]  # type: ignore[index]
    )
    cross = _cross_section_nodes(document)[0]
    input_names = [
        field["name"]
        for field in cross["input_ports"][0]["schema"]  # type: ignore[index]
    ]
    assert input_names == ["ts", "symbol", "industry", "seq", "x", "momentum"]
