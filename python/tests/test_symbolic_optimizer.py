from __future__ import annotations

import dataclasses
from copy import deepcopy

import pyarrow as pa
import pytest

from calc_flow import Batch, Runtime, register_numpy
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    TableExpr,
    cs,
    event_time_bucket,
    exact_time,
    linalg,
    parameter,
    rows,
    table,
    table_input,
    ts,
)
from calc_flow.symbolic.lower import _compile_cache_key, lower_program_document
from calc_flow.symbolic.optimizer import (
    _rolling_group_key,
    _rolling_input_columns,
    _rolling_kernel_fallback,
    _rolling_kernel_line,
    _rolling_leaf_outputs,
    _rolling_spec_outputs,
)


def _ordered() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("industry", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64", nullable=True),
            Field("y", "float64", nullable=True),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


def _operator_nodes(document: dict[str, object], kind: str) -> list[dict[str, object]]:
    graph = document["graph"]
    return [
        node
        for node in graph["nodes"]  # type: ignore[index]
        if node["operator"]["kind"] == kind  # type: ignore[index]
    ]


def _ordered_batch() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("industry", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("x", pa.float64()),
            pa.field("y", pa.float64()),
        ]
    )
    return pa.table(
        {
            "ts": pa.array([1, 1, 2, 2], type=pa.timestamp("us", tz="UTC")),
            "symbol": ["a", "b", "a", "b"],
            "industry": ["i", "i", "i", "i"],
            "seq": pa.array([1, 1, 2, 2], type=pa.uint64()),
            "x": [1.0, 3.0, 5.0, 7.0],
            "y": [2.0, 4.0, 6.0, 8.0],
        },
        schema=schema,
    )


def test_program_wide_cse_shares_a_common_row_local_materialization() -> None:
    quotes = _ordered()
    common = quotes.with_columns(FeatureSet((("product", quotes["x"] * quotes["y"]),)))
    first = common.with_columns(FeatureSet((("first_score", common["product"] + 1.0),)))
    second = common.with_columns(
        FeatureSet((("second_score", common["product"] + 2.0),))
    )
    program = Program(
        "program-wide-cse",
        inputs=(quotes,),
        outputs=(("first", first), ("second", second)),
    )

    document = lower_program_document(program, Runtime(), "batch")

    materializations = [
        node
        for node in _operator_nodes(document, "expression")
        if any(" * " in item for item in node["operator"]["select"])  # type: ignore[index]
        and node["id"] not in {"first", "second"}
    ]
    assert len(materializations) == 1
    shared_id = materializations[0]["id"]
    outgoing = [
        edge
        for edge in document["graph"]["edges"]  # type: ignore[index]
        if edge["source_node"] == shared_id
    ]
    assert [edge["target_node"] for edge in outgoing] == ["first", "second"]
    program.compile_batch(Runtime())


def test_compatible_rolling_outputs_share_one_state_stage() -> None:
    quotes = _ordered()
    short = quotes.with_columns(
        FeatureSet((("short_mean", ts.mean(quotes["x"], window=rows(3))),))
    )
    long = quotes.with_columns(
        FeatureSet((("long_max", ts.max(quotes["x"], window=rows(5))),))
    )
    program = Program(
        "rolling-sharing",
        inputs=(quotes,),
        outputs=(("short", short), ("long", long)),
    )

    document = lower_program_document(program, Runtime(), "batch")

    (rolling,) = _operator_nodes(document, "rolling")
    outputs = rolling["operator"]["spec"]["outputs"]
    assert [item["kind"] for item in outputs] == ["mean", "max"]
    outgoing = [
        edge
        for edge in document["graph"]["edges"]  # type: ignore[index]
        if edge["source_node"] == rolling["id"]
    ]
    assert [edge["target_node"] for edge in outgoing] == ["short", "long"]
    result = program.compile_batch(Runtime()).execute(
        {"input": Batch.from_pyarrow(_ordered_batch())}
    )
    assert result.outputs["short.output"].to_pyarrow()["short_mean"].to_pylist() == [
        1.0,
        3.0,
        3.0,
        5.0,
    ]
    assert result.outputs["long.output"].to_pyarrow()["long_max"].to_pylist() == [
        1.0,
        3.0,
        5.0,
        7.0,
    ]
    program.compile_stream(Runtime())


def test_rolling_kernel_explain_helpers_fail_closed_on_unknown_shapes() -> None:
    field_types = {"x": "float64", "label": "string"}

    assert _rolling_leaf_outputs(None) == ()
    pair = {
        "kind": "correlation",
        "left": "x",
        "right": "label",
        "frame": {"kind": "rows", "size": 3},
    }
    assert _rolling_group_key(pair) == (
        "pair",
        "x",
        "label",
        "rows",
        3,
    )
    assert _rolling_input_columns(pair, "pair") == ("x", "label")
    assert _rolling_kernel_fallback(pair, field_types) == (
        "primitive_correlation_requires_numeric_column_label"
    )
    assert _rolling_kernel_fallback({"kind": "unknown"}, field_types) == (
        "primitive_unknown_missing_from_census"
    )
    assert _rolling_kernel_fallback({"kind": "lag", "input": "x"}, field_types) == (
        "primitive_lag_has_no_typed_transition"
    )
    assert (
        _rolling_kernel_fallback(
            {
                "kind": "difference",
                "left": {"kind": "mean", "input": "x"},
                "right": {"kind": "unknown", "input": "x"},
            },
            field_types,
        )
        == "primitive_unknown_missing_from_census"
    )

    malformed = {"id": "rolling", "operator": {"kind": "rolling", "spec": {}}}
    assert _rolling_spec_outputs(malformed) is None
    assert _rolling_kernel_line(malformed) is None


def test_filter_is_not_moved_across_a_rolling_finality_boundary() -> None:
    quotes = _ordered()
    filtered = table.filter(quotes, quotes["x"] > 0.0)
    filtered_output = filtered.with_columns(
        FeatureSet((("mean", ts.mean(quotes["x"], window=rows(3))),))
    )
    unfiltered_output = quotes.with_columns(
        FeatureSet((("mean", ts.mean(quotes["x"], window=rows(3))),))
    )
    program = Program(
        "unsafe-filter",
        inputs=(quotes,),
        outputs=(("filtered", filtered_output), ("unfiltered", unfiltered_output)),
    )

    document = lower_program_document(program, Runtime(), "batch")

    assert len(_operator_nodes(document, "rolling")) == 2
    assert (
        len(
            [
                node
                for node in _operator_nodes(document, "expression")
                if node["operator"]["filter"] is not None  # type: ignore[index]
            ]
        )
        == 1
    )


def test_compatible_cross_sections_share_grouping_and_sorting() -> None:
    quotes = _ordered()
    group = exact_time(quotes["ts"], partition_by=(quotes["industry"],))
    ranked = quotes.with_columns(
        FeatureSet((("rank", cs.rank(quotes["x"], group=group)),))
    )
    normalized = quotes.with_columns(
        FeatureSet((("zscore", cs.zscore(quotes["y"], group=group)),))
    )
    program = Program(
        "cross-section-sharing",
        inputs=(quotes,),
        outputs=(("ranked", ranked), ("normalized", normalized)),
    )

    document = lower_program_document(program, Runtime(), "batch")

    (cross_section,) = _operator_nodes(document, "cross_section")
    outputs = cross_section["operator"]["spec"]["outputs"]
    assert [item["kind"] for item in outputs] == ["rank", "zscore"]
    result = program.compile_batch(Runtime()).execute(
        {"input": Batch.from_pyarrow(_ordered_batch())}
    )
    ranked_values = result.outputs["ranked.output"].to_pyarrow()["rank"].to_pylist()
    normalized_values = (
        result.outputs["normalized.output"].to_pyarrow()["zscore"].to_pylist()
    )
    assert ranked_values == [1.0, 2.0, 1.0, 2.0]
    assert normalized_values == [-1.0, 1.0, -1.0, 1.0]
    program.compile_stream(Runtime())


def test_incompatible_cross_section_finality_is_not_shared() -> None:
    quotes = _ordered()
    exact = exact_time(quotes["ts"], partition_by=(quotes["industry"],))
    bucket = event_time_bucket(
        quotes["ts"], width_micros=60, partition_by=(quotes["industry"],)
    )
    exact_output = quotes.with_columns(
        FeatureSet((("rank", cs.rank(quotes["x"], group=exact)),))
    )
    bucket_output = quotes.with_columns(
        FeatureSet((("rank", cs.rank(quotes["x"], group=bucket)),))
    )
    program = Program(
        "cross-section-finality",
        inputs=(quotes,),
        outputs=(("exact", exact_output), ("bucket", bucket_output)),
    )

    document = lower_program_document(program, Runtime(), "batch")

    assert len(_operator_nodes(document, "cross_section")) == 2


def test_compile_cache_reuses_immutable_values_and_revision_invalidates() -> None:
    quotes = _ordered()
    output = quotes.with_columns(FeatureSet((("score", quotes["x"] + 1.0),)))
    program = Program("compile-cache", inputs=(quotes,), outputs=(("output", output),))
    runtime = Runtime()

    first = program.compile_batch(runtime)
    second = program.compile_batch(runtime)

    assert first is second
    with pytest.raises(dataclasses.FrozenInstanceError):
        first._inner = object()  # type: ignore[misc]

    runtime.register_provider("test", "unused", "2", lambda _inputs, _options: {})
    after_revision = program.compile_batch(runtime)
    assert after_revision is not first


def test_compile_cache_key_captures_every_frozen_identity_dimension() -> None:
    quotes = _ordered()
    output = quotes.with_columns(FeatureSet((("score", quotes["x"] + 1.0),)))
    program = Program("cache-key", inputs=(quotes,), outputs=(("output", output),))
    runtime = Runtime()
    document = lower_program_document(program, runtime, "batch")
    augmented = deepcopy(document)
    nodes = augmented["graph"]["nodes"]  # type: ignore[index]
    nodes[0]["operator"]["udfs"] = [  # type: ignore[index]
        {"provider": "trusted", "name": "score", "version": "7"}
    ]
    nodes.append(
        {
            "id": "provider",
            "operator": {
                "kind": "external",
                "name": "kernel",
                "options": {},
                "provider": "numpy",
                "version": "3",
            },
        }
    )
    capabilities = runtime.capabilities()

    key = _compile_cache_key(program, "batch", augmented, capabilities, 0, "error")

    assert key.program_fingerprint == program.fingerprint
    assert key.mode == "batch"
    assert key.input_declarations == (quotes._node.node_bytes.hex(),)
    assert key.capability_schema_version == capabilities.schema_version
    assert key.capability_session_id == capabilities.scope.session_id
    assert key.capability_revision == capabilities.scope.revision
    assert ("expression", "1") in key.operator_versions
    assert key.provider_versions == (("numpy", "kernel", "3"),)
    assert key.udf_versions == (("trusted", "score", "7"),)


def test_explain_reports_deterministic_optimization_and_cost_facts() -> None:
    quotes = _ordered()
    shared = quotes["x"] * quotes["y"]
    output = quotes.with_columns(
        FeatureSet(
            (
                ("mean", ts.mean(quotes["x"], window=rows(5))),
                ("left", shared + 1.0),
                ("right", shared + 2.0),
            )
        )
    )
    program = Program("explain-costs", inputs=(quotes,), outputs=(("output", output),))
    runtime = Runtime()

    explanation = program.explain(runtime, mode="batch")

    assert explanation == program.explain(runtime, mode="batch")
    assert "  optimization" in explanation
    assert "    cse materializations 1" in explanation
    assert "    rolling state_stages 1 shared_outputs 1" in explanation
    assert (
        "    rolling kernel output__cf_rolling selected=ordered_primitive"
        " profile=stable_v1 complexity=amortized_constant"
        " order=ts,symbol,seq shared_state_groups=1 fallback=none" in explanation
    )
    assert "  costs" in explanation
    assert "    state output__cf_rolling rows=5" in explanation
    assert "    copies none" in explanation
    assert "    providers none" in explanation


def test_explain_reports_fused_array_provider_and_copy_boundaries() -> None:
    quotes = _ordered()
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(2, 1),
    )
    dense = linalg.from_columns(quotes, columns=("x", "y"), backend="numpy")
    scores = linalg.matmul((dense * 2.0) + 1.0, weights)
    output = table.attach_columns(quotes, scores, names=("score",))
    program = Program(
        "array-fusion",
        inputs=(quotes, weights),
        outputs=(("output", output),),
    )
    runtime = Runtime()
    register_numpy(runtime)

    explanation = program.explain(runtime, mode="stream")

    assert "    array fused_stages 1 provider_calls_per_microbatch 1" in explanation
    assert "    copies output table_to_dense columns=2" in explanation
    assert "host_to_device=no" in explanation
    assert (
        "    providers output numpy:symbolic_matrix@1 calls_per_microbatch=1"
        in explanation
    )
