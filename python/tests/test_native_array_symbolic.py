from __future__ import annotations

import asyncio
import builtins
import json
import math
import runpy
from pathlib import Path

import pyarrow as pa
import pytest

import calc_flow as cf
import calc_flow.array as cfa
from calc_flow import _native
from calc_flow.symbolic.nodes import CInt, CNull, CSeq


def test_array_input_is_an_immutable_native_descriptor() -> None:
    shape = [None, 2]
    value = cf.array_input("values", dtype="float32", shape=shape)
    shape[1] = 8
    assert value._node.op.name == "array_input"
    assert value._node.attr("shape") == CSeq((CNull(), CInt(2)))
    assert value.identical(cf.array_input("values", dtype="float32", shape=(None, 2)))


def test_program_discovers_dynamic_array_inputs_and_rejects_computed_bindings() -> None:
    value = cf.array_input("x", dtype="float64", shape=(None,))
    program = (
        cf.Program("arrays", engine="sql").with_input(value).output("out", value + 1.0)
    )
    assert program.inputs == (value,)
    discovered = cf.Program("arrays", engine="sql", outputs={"out": value + 1.0})
    assert discovered.inputs[0].identical(value)
    with pytest.raises(ValueError, match="declared.*array_input"):
        cf.Program("invalid", engine="sql").with_input(value + 1.0)


def test_array_shape_and_reduction_methods_build_data_only_nodes() -> None:
    value = cf.array_input("x", dtype="float32", shape=(4,))
    result = value.reshape((2, 2)).transpose().mean(axis=1, keepdims=True)
    assert result._node.op.name == "array_reduce"
    assert result._node.args[0].op.name == "array_transpose"
    assert result._node.args[0].args[0].op.name == "array_reshape"
    assert value.sum()._node.attr("axis") == CNull()
    with pytest.raises(TypeError, match="axis"):
        value.mean(axis=True)
    with pytest.raises(ValueError, match="shape"):
        value.reshape((2, -1))


def test_real_rust_planning_query_infers_axis_reduction_without_rows() -> None:
    program = {
        "ir_version": 2,
        "inputs": [{"name": "x", "dtype": "float32", "shape": [None, 2]}],
        "nodes": [
            {"id": "x", "op": "input", "name": "x"},
            {"id": "avg", "op": "reduce", "function": "mean", "input": "x", "axis": 1},
        ],
        "outputs": [{"name": "output", "node": "avg"}],
    }
    inferred = json.loads(_native._infer_native_array_program(json.dumps(program)))
    assert inferred["outputs"] == [
        {"name": "output", "node": "avg", "dtype": "float64", "shape": [None]}
    ]


def test_native_query_translation_preserves_declarations_without_python_inference() -> (
    None
):
    from calc_flow.symbolic.native_arrays import build_native_program

    left = cf.array_input("x", dtype="float32", shape=(None, 2))
    right = cf.parameter(
        "coefficients", kind="array", backend="native", dtype="float32", shape=(2, 1)
    )
    value = cf.linalg.matmul(left, right).mean(axis=1)
    program, bindings = build_native_program(value._node)
    assert program["ir_version"] == 2
    assert {tuple(item["shape"]) for item in program["inputs"]} == {(None, 2), (2, 1)}
    assert [node["op"] for node in program["nodes"]] == [
        "input",
        "input",
        "matmul",
        "reduce",
    ]
    assert program["nodes"][-1]["axis"] == 1
    assert set(program["outputs"][0]) == {"name", "node"}
    assert {node.attr("name").value for node in bindings.values()} == {
        "x",
        "coefficients",
    }


def test_real_native_array_analysis_uses_rust_dtype_and_shape() -> None:
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    value = cf.array_input("x", dtype="float32", shape=(None, 2)).mean(axis=1)
    program = cf.Program("analysis", engine="sql", outputs={"output": value})
    result = program.analyze(runtime)
    assert result.issues == ()


@pytest.mark.parametrize("dtype,shape", [("int64", (2,)), ("float64", (1, 2, 3))])
def test_native_static_parameters_reject_unsupported_descriptors(
    dtype: str, shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="static_inputs.beta.(dtype|shape)"):
        cf.parameter("beta", kind="array", backend="native", dtype=dtype, shape=shape)


@pytest.mark.parametrize("entry", ["collect", "collect_async"])
def test_program_table_collectors_explicitly_reject_array_outputs(entry: str) -> None:
    value = cf.array_input("x", dtype="float64", shape=(None,))
    program = cf.Program("array", engine="sql", outputs={"out": value})
    with pytest.raises(TypeError, match="collect_array"):
        getattr(program, entry)({})


def test_native_query_cache_uses_existing_registration_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def infer(program: str) -> str:
        calls.append(program)
        return '{"outputs":[]}'

    monkeypatch.setattr(_native, "_infer_native_array_program", infer, raising=False)
    runtime = cf.Runtime()
    assert runtime._infer_native_array_program("{}") == '{"outputs":[]}'
    assert runtime._infer_native_array_program("{}") == '{"outputs":[]}'
    assert calls == ["{}"]
    runtime.register_provider("acme", "example", "1", lambda batch, _: batch)
    runtime._infer_native_array_program("{}")
    assert calls == ["{}", "{}"]


def test_table_array_query_translation_cuts_at_the_existing_table_boundary() -> None:
    from calc_flow.symbolic.analyzer import _run
    from calc_flow.symbolic.native_arrays import build_native_program

    table = cf.table_input("prices", schema=(cf.Field("price", "float32"),))
    expression = cf.linalg.from_columns(table, columns=("price",), backend="native")
    analyzer, _ = _run(
        cf.Program("table", engine="sql", outputs={"table": table}),
        cf.Runtime(),
        "batch",
    )
    program, bindings = build_native_program(expression._node, analyzer=analyzer)
    assert program["inputs"][0]["dtype"] == "float32"
    assert program["inputs"][0]["shape"] == [None, 1]
    assert len(program["nodes"]) == 1
    assert next(iter(bindings.values())).args[0] is table._node


def test_native_vector_attachment_has_one_column_independent_of_row_extent() -> None:
    from calc_flow.symbolic.analyzer import ArrayFacts, _Analyzer

    analyzer = _Analyzer("batch", frozenset(), frozenset(), True)
    vector = ArrayFacts("native", "float64", (None,), None, frozenset())
    analyzer._attach_width_check(vector, ("score",), "outputs.table")
    assert analyzer.issues == ()
    analyzer._attach_width_check(vector, ("left", "right"), "outputs.table")
    assert analyzer.issues[0].code == "schema_mismatch"


@pytest.mark.parametrize("origin", [None, "same_table"])
def test_high_level_native_array_outputs_are_closed_in_stream_mode(
    origin: str | None,
) -> None:
    from calc_flow.symbolic.analyzer import ArrayFacts, _Analyzer

    analyzer = _Analyzer("stream", frozenset(), frozenset(), True)
    analyzer.check_unbounded_output(
        "out", ArrayFacts("native", "float64", (None,), origin, frozenset())
    )
    assert len(analyzer.issues) == 1
    assert analyzer.issues[0].code == "unsupported_mode"


@pytest.mark.parametrize("number", [2**53 + 1, -(2**53 + 1), 2**64 + 1, -(2**64 + 1)])
def test_native_python_integer_literals_are_bounded_before_json_encoding(
    number: int,
) -> None:
    values = cf.array_input("x", dtype="float64", shape=(None,))
    with pytest.raises(ValueError, match=r"native.*literal.*2\^53"):
        values + number
    with pytest.raises(ValueError, match=r"native.*literal.*2\^53"):
        number - values


def test_native_literal_frontend_preserves_full_finite_float_and_separate_bool() -> (
    None
):
    from calc_flow.symbolic.native_arrays import build_native_program

    values = cf.array_input("x", dtype="float64", shape=(None,))
    program, _ = build_native_program((values + 1e308)._node)
    literal = next(node for node in program["nodes"] if node["op"] == "literal")
    assert literal["dtype"] == "float64"
    assert literal["value"] == 1e308
    flags = cf.array_input("flags", dtype="bool", shape=(None,))
    program, _ = build_native_program((flags & True)._node)
    literal = next(node for node in program["nodes"] if node["op"] == "literal")
    assert literal["dtype"] == "bool"
    assert literal["value"] is True


@pytest.mark.parametrize("number", [math.inf, -math.inf, math.nan])
def test_native_non_finite_python_literals_are_rejected(number: float) -> None:
    values = cf.array_input("x", dtype="float64", shape=(None,))
    with pytest.raises(ValueError, match="finite"):
        values + number


def test_native_inference_rejects_foreign_operands_before_any_jax_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calc_flow.symbolic.analyzer import _run
    from calc_flow.symbolic.native_arrays import _inference_operand

    table = cf.table_input("prices", schema=(cf.Field("price", "float32"),))
    foreign = cf.linalg.from_columns(table, columns=("price",), backend="jax") + 1.0
    analyzer, _ = _run(
        cf.Program("input", engine="sql", outputs={"table": table}),
        cf.Runtime(),
        "batch",
    )
    original_import = builtins.__import__

    def checked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "jax" or name.startswith("jax."):
            raise AssertionError("native inference must reject before a JAX probe")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    with pytest.raises(TypeError, match="native operands"):
        _inference_operand(foreign._node, analyzer, "outputs.result")


def test_native_logical_bindings_capture_arrays_without_mutating_the_mapping() -> None:
    from calc_flow.compute import _collect_batches

    values = cf.array_input("values", dtype="float64", shape=(None,))
    coefficient = cf.parameter(
        "beta", kind="array", backend="native", dtype="float64", shape=()
    )
    expression = values + coefficient
    program = cf.Program("bindings", engine="sql", outputs={"out": expression})
    array = cfa.from_arrow(pa.array([1.0, 3.0]))
    scalar = cfa.from_arrow(pa.array([2.0]), shape=())
    supplied = {"values": array, "beta": scalar}
    bound = _collect_batches(program, supplied)
    supplied.clear()
    assert set(bound) == {"values", "beta"}
    assert all(batch.backend == "native" for batch in bound.values())
    assert bound["values"].array.to_arrow().to_pylist() == [1.0, 3.0]
    assert bound["beta"].array.shape == ()


@pytest.mark.parametrize("failure", ["dtype", "shape", "backend"])
def test_native_dynamic_bindings_reject_wrong_descriptors_early(failure: str) -> None:
    from calc_flow.compute import _collect_batches

    values = cf.array_input("values", dtype="float64", shape=(2,))
    program = cf.Program("bindings", engine="sql", outputs={"out": values})
    if failure == "dtype":
        supplied = cfa.from_arrow(pa.array([1.0, 2.0], type=pa.float32()))
    elif failure == "shape":
        supplied = cfa.from_arrow(pa.array([1.0, 2.0, 3.0]))
    else:
        supplied = pa.table({"x": [1.0, 2.0]})
    with pytest.raises((TypeError, ValueError), match=f"inputs.values.{failure}"):
        _collect_batches(program, {"values": supplied})


def test_program_array_expr_input_must_be_the_dynamic_declaration_kind() -> None:
    parameter = cf.parameter(
        "beta", kind="array", backend="native", dtype="float64", shape=()
    )
    with pytest.raises(ValueError, match="array_input"):
        cf.Program("invalid", engine="sql").with_input(cf.ArrayExpr(parameter._node))


def test_collect_array_rejects_a_running_event_loop_before_planning() -> None:
    value = cf.array_input("x", dtype="float64", shape=(None,))
    runtime = cf.Runtime()

    async def collect() -> None:
        value.collect_array({}, runtime=runtime)

    with pytest.raises(
        RuntimeError, match="collect_array.*cannot run inside an event loop"
    ):
        asyncio.run(collect())


def test_real_collect_array_uses_logical_inputs_and_resource_specific_templates() -> (
    None
):
    value = (cf.array_input("x", dtype="float64", shape=(None,)) + 1.0).mean()
    collect = value.collect_array
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    first = collect({"x": cfa.from_arrow(pa.array([1.0, 3.0]))}, runtime=runtime)
    assert isinstance(first, cfa.NativeArray)
    assert first.shape == ()
    assert first.to_arrow().to_pylist() == [3.0]
    second = collect({"x": cfa.from_arrow(pa.array([5.0, 7.0]))}, runtime=runtime)
    assert second.to_arrow().to_pylist() == [7.0]
    with pytest.raises(cf.CalcFlowError, match="elements"):
        collect(
            {"x": cfa.from_arrow(pa.array([1.0, 3.0]))},
            runtime=runtime,
            array_options=cfa.ArrayExecutionOptions(max_elements=0),
        )


@pytest.mark.parametrize("reduce_rows", [False, True])
def test_real_table_rooted_collect_array_after_sql_and_batch_boundary(
    reduce_rows: bool,
) -> None:
    prices = cf.table_input("prices", schema=(cf.Field("price", "float64"),))
    projected = cf.sql("SELECT price * 2 AS px FROM source", source=prices)
    value = cf.linalg.from_columns(projected, columns=("px",), backend="native")
    if reduce_rows:
        value = value.mean(axis=0)
    collect = value.collect_array
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    result = collect({"prices": pa.table({"price": [1.0, 3.0]})}, runtime=runtime)
    assert result.shape == ((1,) if reduce_rows else (2, 1))
    assert result.to_arrow().to_pylist() == ([4.0] if reduce_rows else [2.0, 6.0])


def test_native_array_segment_walk_stops_at_an_upstream_table_stage() -> None:
    from calc_flow.symbolic.lower.native_arrays import _bridge_leaves

    prices = cf.table_input("prices", schema=(cf.Field("price", "float64"),))
    first = cf.linalg.from_columns(prices, columns=("price",), backend="native")
    attached = cf.table.attach_columns(prices, first, names=("copy",))
    second = cf.linalg.from_columns(attached, columns=("copy",), backend="native")
    assert _bridge_leaves(second._node) == (second._node,)


@pytest.mark.parametrize("mode", ["batch", "stream"])
def test_native_graph_document_keeps_static_declarations_from_existing_segments(
    mode: str,
) -> None:
    from calc_flow.symbolic.analyzer import _Analyzer
    from calc_flow.symbolic.lower.native_arrays import _NativeGraph

    graph = _NativeGraph(
        cf.Program("static", engine="sql"),
        _Analyzer("batch", frozenset(), frozenset(), True),
        0,
        "error",
        cfa.ArrayExecutionOptions(),
    )
    declaration = {
        "name": "legacy",
        "kind": "array",
        "backend": "numpy",
        "dtype": "float64",
        "shape": [1],
        "mutability": "static",
    }
    graph.graph.static_inputs.append(declaration)
    graph.graph.nodes.append(
        {
            "id": "legacy",
            "input_ports": [
                {"name": "legacy", "kind": "array", "required": True, "schema": []}
            ],
            "operator": {
                "kind": "external",
                "provider": "numpy",
                "name": "expression",
                "version": "1",
                "options": {"expression": "x"},
            },
            "output_ports": [
                {"name": "output", "kind": "array", "required": True, "schema": []}
            ],
        }
    )
    document = graph.document(mode)
    assert document["static_inputs"] == [declaration]
    assert {source["input"] for source in document["data_sources"]} == (
        {"legacy"} if mode == "batch" else set()
    )


def _mixed_legacy_and_native_expression() -> cf.ArrayExpr:
    source = cf.table_input(
        "prices", schema=(cf.Field("x", "float64"), cf.Field("y", "float64"))
    )
    legacy_weights = cf.parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(2, 1)
    )
    legacy_matrix = cf.linalg.from_columns(source, columns=("x", "y"), backend="numpy")
    attached = cf.table.attach_columns(
        source, cf.linalg.matmul(legacy_matrix, legacy_weights), names=("score",)
    )
    scores = cf.linalg.from_columns(attached, columns=("score",), backend="native")
    independent = cf.array_input("native_values", dtype="float64", shape=(None, 2))
    coefficients = cf.parameter(
        "native_coefficients",
        kind="array",
        backend="native",
        dtype="float64",
        shape=(2, 1),
    )
    return scores.mean(axis=0) + cf.linalg.matmul(independent, coefficients).mean(
        axis=0
    )


def test_native_batch_document_preserves_legacy_static_source_placeholders() -> None:
    from calc_flow.symbolic.lower import lower_program_document
    from calc_flow.symbolic.lower.bindings import _BatchBindings

    runtime = cf.Runtime()
    cf.register_numpy(runtime)
    cfa.register_native(runtime)
    program = cf.Program(
        "mixed_backends",
        engine="sql",
        outputs={"out": _mixed_legacy_and_native_expression()},
    )
    bindings = _BatchBindings()
    document = lower_program_document(program, runtime, "batch", _bindings=bindings)
    physical, _ = bindings.names()
    assert {source["input"] for source in document["data_sources"]} == {
        name
        for logical in ("prices", "weights", "native_values")
        for name in physical[logical]
    }
    assert {spec["backend"] for spec in document["static_inputs"]} == {"native"}
    assert {spec["name"] for spec in document["static_inputs"]} == set(
        physical["native_coefficients"]
    )


def test_real_mixed_numpy_static_table_and_native_independent_root() -> None:
    import numpy as np

    runtime = cf.Runtime()
    cf.register_numpy(runtime)
    cfa.register_native(runtime)
    weights = np.array([[2.0], [1.0]], dtype=np.float64)
    result = _mixed_legacy_and_native_expression().collect_array(
        {
            "prices": pa.table({"x": [1.0, 2.0], "y": [3.0, 4.0]}),
            "weights": cf.Batch.from_array(weights, backend="numpy"),
            "native_values": cfa.from_arrow(
                pa.array([1.0, 3.0, 2.0, 4.0]), shape=(2, 2)
            ),
            "native_coefficients": cfa.from_arrow(pa.array([1.0, 1.0]), shape=(2, 1)),
        },
        runtime=runtime,
    )
    assert result.shape == (1,)
    assert result.to_arrow().to_pylist() == [11.5]
    assert weights.tolist() == [[2.0], [1.0]]


def test_native_graph_document_rejects_unconsumed_explicit_inputs() -> None:
    from calc_flow.symbolic.analyzer import _Analyzer
    from calc_flow.symbolic.lower.native_arrays import _NativeGraph

    values = cf.array_input("values", dtype="float64", shape=(None,))
    unused = cf.array_input("unused", dtype="float64", shape=(None,))
    program = cf.Program(
        "unused", engine="sql", inputs=(values, unused), outputs={"out": values}
    )
    graph = _NativeGraph(
        program,
        _Analyzer("batch", frozenset(), frozenset(), True),
        0,
        "error",
        cfa.ArrayExecutionOptions(),
    )
    graph.graph.bindings.inputs["values"] = {("source", "input")}
    with pytest.raises(cf.CalcFlowError, match="inputs.unused.*unconsumed"):
        graph.document("batch")


def test_real_native_sql_attachment_and_following_sql_keep_table_collection() -> None:
    source = cf.table_input(
        "prices",
        schema=(cf.Field("feature_x", "float64"), cf.Field("feature_y", "float64")),
    )
    prepared = cf.sql(
        "SELECT feature_x + 1 AS feature_x, feature_y FROM base", base=source
    )
    weights = cf.parameter(
        "coefficients", kind="array", backend="native", dtype="float64", shape=(2, 1)
    )
    matrix = cf.linalg.from_columns(
        prepared, columns=("feature_x", "feature_y"), backend="native"
    )
    attached = cf.table.attach_columns(
        prepared, cf.linalg.matmul(matrix, weights), names=("score",)
    )
    final = cf.sql("SELECT score + 1 AS score FROM enriched", enriched=attached)
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    result = final.collect(
        {
            "prices": pa.table({"feature_x": [1.0, 2.0], "feature_y": [3.0, 4.0]}),
            "coefficients": cfa.from_arrow(pa.array([2.0, 1.0]), shape=(2, 1)),
        },
        runtime=runtime,
    )
    assert isinstance(result, pa.Table)
    assert result.to_pydict() == {"score": [8.0, 11.0]}


def test_real_native_vector_attachment_can_be_filtered_and_projected_directly() -> None:
    source = cf.table_input(
        "prices",
        schema=(cf.Field("feature_x", "float64"), cf.Field("feature_y", "float64")),
    )
    values = cf.linalg.from_columns(
        source, columns=("feature_x", "feature_y"), backend="native"
    ).mean(axis=1)
    attached = cf.table.attach_columns(source, values, names=("average",))
    final = attached.filter(attached["average"] > 2.0).select("average")
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    result = final.collect(
        {"prices": pa.table({"feature_x": [1.0, 2.0], "feature_y": [3.0, 4.0]})},
        runtime=runtime,
    )
    assert result.to_pydict() == {"average": [3.0]}


@pytest.mark.parametrize(
    "transform", ["independent", "transpose_back", "axis0", "full"]
)
def test_real_native_attachment_rejects_unproved_origin_at_equal_lengths(
    transform: str,
) -> None:
    source = cf.table_input(
        "prices",
        schema=(cf.Field("feature_x", "float64"), cf.Field("feature_y", "float64")),
    )
    matrix = cf.linalg.from_columns(
        source, columns=("feature_x", "feature_y"), backend="native"
    )
    names = ("value",)
    if transform == "independent":
        values = cf.array_input("other", dtype="float64", shape=(2,))
    elif transform == "transpose_back":
        values = matrix.transpose().transpose()
        names = ("a", "b")
    elif transform == "axis0":
        values = matrix.mean(axis=0)
    else:
        values = matrix.mean().reshape((1,))
    output = cf.table.attach_columns(source, values, names=names)
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    rows = [1.0] if transform == "full" else [1.0, 2.0]
    supplied = {"prices": pa.table({"feature_x": rows, "feature_y": rows})}
    if transform == "independent":
        supplied["other"] = cfa.from_arrow(pa.array([4.0, 5.0]))
    with pytest.raises(cf.CalcFlowError, match="lineage"):
        output.collect(supplied, runtime=runtime)


def test_real_native_two_outputs_share_an_arbitrarily_named_static_parameter() -> None:
    source = cf.table_input(
        "prices",
        schema=(cf.Field("feature_x", "float64"), cf.Field("feature_y", "float64")),
    )
    matrix = cf.linalg.from_columns(
        source, columns=("feature_x", "feature_y"), backend="native"
    )
    beta = cf.parameter(
        "beta", kind="array", backend="native", dtype="float64", shape=(2, 1)
    )
    first = cf.table.attach_columns(
        source, cf.linalg.matmul(matrix, beta), names=("a",)
    )
    second = cf.table.attach_columns(
        source, cf.linalg.matmul(matrix + 1.0, beta), names=("b",)
    )
    program = cf.Program(
        "shared", engine="sql", outputs={"first": first, "second": second}
    )
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    supplied = {
        "prices": pa.table({"feature_x": [1.0, 2.0], "feature_y": [3.0, 4.0]}),
        "beta": cfa.from_arrow(pa.array([2.0, 1.0]), shape=(2, 1)),
    }
    result = program.collect(supplied, runtime=runtime)
    assert result["first"]["a"].to_pylist() == [5.0, 8.0]
    assert result["second"]["b"].to_pylist() == [8.0, 11.0]
    supplied["beta"] = cfa.from_arrow(pa.array([3.0, 1.0]), shape=(2, 1))
    assert program.collect(supplied, runtime=runtime)["first"]["a"].to_pylist() == [
        6.0,
        10.0,
    ]


def test_real_independent_collect_chained_matmul_executes_one_matmul_per_segment() -> (
    None
):
    source = cf.array_input("x", dtype="float64", shape=(None, 2))
    first = cf.parameter(
        "first_matrix", kind="array", backend="native", dtype="float64", shape=(2, 2)
    )
    second = cf.parameter(
        "coefficients", kind="array", backend="native", dtype="float64", shape=(2, 1)
    )
    expression = cf.linalg.matmul(cf.linalg.matmul(source, first), second)
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    result = expression.collect_array(
        {
            "x": cfa.from_arrow(pa.array([1.0, 3.0, 2.0, 4.0]), shape=(2, 2)),
            "first_matrix": cfa.from_arrow(
                pa.array([1.0, 0.0, 0.0, 1.0]), shape=(2, 2)
            ),
            "coefficients": cfa.from_arrow(pa.array([2.0, 1.0]), shape=(2, 1)),
        },
        runtime=runtime,
    )
    assert result.shape == (2, 1)
    assert result.to_arrow().to_pylist() == [5.0, 8.0]


def test_native_chained_matmul_document_has_one_matmul_per_segment() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = cf.array_input("x", dtype="float64", shape=(None, 2))
    first = cf.parameter(
        "first_matrix", kind="array", backend="native", dtype="float64", shape=(2, 2)
    )
    second = cf.parameter(
        "coefficients", kind="array", backend="native", dtype="float64", shape=(2, 1)
    )
    expression = cf.linalg.matmul(cf.linalg.matmul(source, first), second)
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    document = lower_program_document(
        cf.Program("chain", engine="sql", outputs={"out": expression}), runtime, "batch"
    )
    counts = [
        sum(item["op"] == "matmul" for item in node["operator"]["options"]["nodes"])
        for node in document["graph"]["nodes"]
        if node["operator"].get("provider") == "native"
    ]
    assert max(counts) == 1
    assert sum(counts) == 2
    assert len(document["static_inputs"]) == 2


def _assert_native_static_array_ports(nodes: list[dict[str, object]]) -> None:
    static_ports = [
        declaration
        for node in nodes
        for declaration in node["operator"]["options"]["inputs"]
        if declaration["kind"] == "array"
    ]
    assert len(static_ports) == 2
    assert all(declaration["binding"] == "static" for declaration in static_ports)
    assert all("mutability" not in declaration for declaration in static_ports)


def test_native_parameter_document_shares_logical_binding_across_physical_aliases() -> (
    None
):
    from calc_flow.symbolic.lower import lower_program_document
    from calc_flow.symbolic.lower.bindings import _BatchBindings

    source = cf.table_input(
        "prices", schema=(cf.Field("x", "float64"), cf.Field("y", "float64"))
    )
    matrix = cf.linalg.from_columns(source, columns=("x", "y"), backend="native")
    beta = cf.parameter(
        "arbitrary_beta", kind="array", backend="native", dtype="float64", shape=(2, 1)
    )
    program = cf.Program(
        "aliases",
        engine="sql",
        outputs={
            "first": cf.table.attach_columns(
                source, cf.linalg.matmul(matrix, beta), names=("a",)
            ),
            "second": cf.table.attach_columns(
                source, cf.linalg.matmul(matrix + 1.0, beta), names=("b",)
            ),
        },
    )
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    bindings = _BatchBindings()
    document = lower_program_document(program, runtime, "batch", _bindings=bindings)
    physical, _ = bindings.names()
    aliases = physical["arbitrary_beta"]
    assert len(aliases) == 2
    table_nodes = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"].get("name") == "table_array"
    ]
    assert len(table_nodes) == 2
    _assert_native_static_array_ports(table_nodes)
    assert {entry["name"] for entry in document["static_inputs"]} == set(aliases)
    assert all(entry["input"] not in aliases for entry in document["data_sources"])


def test_collect_array_rejects_external_backends_before_binding_or_planning() -> None:
    source = cf.table_input("prices", schema=(cf.Field("price", "float64"),))
    external = cf.linalg.from_columns(source, columns=("price",), backend="numpy")
    with pytest.raises(TypeError, match="collect_array.*native"):
        external.collect_array({}, runtime=cf.Runtime())


def test_real_table_origin_chained_matmul_attachment_requires_explicit_stages() -> None:
    source = cf.table_input(
        "prices", schema=(cf.Field("x", "float64"), cf.Field("y", "float64"))
    )
    first = cf.parameter(
        "first_matrix", kind="array", backend="native", dtype="float64", shape=(2, 2)
    )
    second = cf.parameter(
        "coefficients", kind="array", backend="native", dtype="float64", shape=(2, 1)
    )
    matrix = cf.linalg.from_columns(source, columns=("x", "y"), backend="native")
    expression = cf.linalg.matmul(cf.linalg.matmul(matrix, first), second)
    attached = cf.table.attach_columns(source, expression, names=("score",))
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    with pytest.raises(cf.CompileError, match="at most one matmul.*found 2"):
        attached.collect(
            {
                "prices": pa.table({"x": [1.0, 2.0], "y": [3.0, 4.0]}),
                "first_matrix": cfa.from_arrow(
                    pa.array([1.0, 0.0, 0.0, 1.0]), shape=(2, 2)
                ),
                "coefficients": cfa.from_arrow(pa.array([2.0, 1.0]), shape=(2, 1)),
            },
            runtime=runtime,
        )


def test_table_origin_chained_matmul_document_requires_explicit_stages() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = cf.table_input(
        "prices", schema=(cf.Field("x", "float64"), cf.Field("y", "float64"))
    )
    first = cf.parameter(
        "first_matrix", kind="array", backend="native", dtype="float64", shape=(2, 2)
    )
    second = cf.parameter(
        "coefficients", kind="array", backend="native", dtype="float64", shape=(2, 1)
    )
    matrix = cf.linalg.from_columns(source, columns=("x", "y"), backend="native")
    expression = cf.linalg.matmul(cf.linalg.matmul(matrix, first), second)
    attached = cf.table.attach_columns(source, expression, names=("score",))
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    with pytest.raises(cf.CompileError, match="at most one matmul.*found 2"):
        lower_program_document(
            cf.Program("chain", engine="sql", outputs={"out": attached}),
            runtime,
            "batch",
        )


def test_stream_static_binding_captures_once_and_shares_physical_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import calc_flow.stream as stream_module

    parameter = cf.parameter(
        "beta", kind="array", backend="native", dtype="float64", shape=(1,)
    )
    array = cfa.from_arrow(pa.array([2.0]))
    supplied = {"beta": array}
    request = stream_module._StreamRequest(
        cf.Program("bindings", engine="streaming").with_input(parameter),
        supplied,
        None,
        None,
        None,
    )
    result = stream_module.StreamResults(request, table_output=False)
    calls = []
    original = stream_module._input_batch

    def capture(*args: object) -> cf.Batch:
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(stream_module, "_input_batch", capture)
    sources, static = result._bindings(
        {"beta": parameter},
        {"beta": ("first.rhs", "second.rhs")},
        cf.EdgeBudget(),
        {},
        frozenset(),
    )
    assert sources == {}
    assert set(static) == {"first.rhs", "second.rhs"}
    assert static["first.rhs"] is static["second.rhs"]
    assert static["first.rhs"].array.to_arrow().to_pylist() == [2.0]
    assert len(calls) == 1
    assert supplied == {"beta": array}


def test_stream_numpy_static_binding_preserves_its_single_physical_name() -> None:
    import numpy as np

    import calc_flow.stream as stream_module

    parameter = cf.parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(1,)
    )
    batch = cf.Batch.from_array(np.array([2.0]), backend="numpy")
    request = stream_module._StreamRequest(
        cf.Program("bindings", engine="streaming").with_input(parameter),
        {"weights": batch},
        None,
        None,
        None,
    )
    result = stream_module.StreamResults(request, table_output=False)
    sources, static = result._bindings(
        {"weights": parameter},
        {"weights": ("weights",)},
        cf.EdgeBudget(),
        {},
        frozenset(),
    )
    assert sources == {}
    assert static == {"weights": batch}


def test_real_native_table_stream_shares_arbitrarily_named_static_parameter() -> None:
    source = cf.table_input(
        "prices", schema=(cf.Field("x", "float64"), cf.Field("y", "float64"))
    )
    matrix = cf.linalg.from_columns(source, columns=("x", "y"), backend="native")
    beta = cf.parameter(
        "arbitrary_beta", kind="array", backend="native", dtype="float64", shape=(2, 1)
    )
    program = cf.Program(
        "native-stream",
        engine="streaming",
        outputs={
            "first": cf.table.attach_columns(
                source, cf.linalg.matmul(matrix, beta), names=("a",)
            ),
            "second": cf.table.attach_columns(
                source, cf.linalg.matmul(matrix + 1.0, beta), names=("b",)
            ),
        },
    )
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    beta_value = cfa.from_arrow(pa.array([2.0, 1.0]), shape=(2, 1))
    data = pa.table({"x": [1.0, 2.0], "y": [3.0, 4.0]})

    async def collect() -> dict[str, list[dict[str, list[float]]]]:
        collected: dict[str, list[dict[str, list[float]]]] = {"first": [], "second": []}
        async with program.stream(
            {"prices": [data], "arbitrary_beta": beta_value},
            runtime=runtime,
            config=cf.StreamRuntimeConfig(checkpointing=False),
        ) as results:
            async for output in results:
                collected[output.name].append(output.table.to_pydict())
        return collected

    assert asyncio.run(collect()) == {
        "first": [{"x": [1.0, 2.0], "y": [3.0, 4.0], "a": [5.0, 8.0]}],
        "second": [{"x": [1.0, 2.0], "y": [3.0, 4.0], "b": [8.0, 11.0]}],
    }


def _explicit_table_matmul_stages() -> cf.TableExpr:
    source = cf.table_input(
        "prices", schema=(cf.Field("x", "float64"), cf.Field("y", "float64"))
    )
    first = cf.parameter(
        "first_matrix", kind="array", backend="native", dtype="float64", shape=(2, 2)
    )
    second = cf.parameter(
        "coefficients", kind="array", backend="native", dtype="float64", shape=(2, 1)
    )
    matrix = cf.linalg.from_columns(source, columns=("x", "y"), backend="native")
    stage = cf.table.attach_columns(
        source,
        cf.linalg.matmul(matrix, first),
        names=("intermediate_x", "intermediate_y"),
    )
    projected = cf.linalg.from_columns(
        stage, columns=("intermediate_x", "intermediate_y"), backend="native"
    )
    return cf.table.attach_columns(
        stage, cf.linalg.matmul(projected, second), names=("score",)
    ).select("x", "y", "score")


def test_native_explicit_table_stages_have_one_matmul_each() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    runtime = cf.Runtime()
    cfa.register_native(runtime)
    document = lower_program_document(
        cf.Program(
            "stages", engine="sql", outputs={"out": _explicit_table_matmul_stages()}
        ),
        runtime,
        "batch",
    )
    programs = [
        node["operator"]["options"]
        for node in document["graph"]["nodes"]
        if node["operator"].get("provider") == "native"
    ]
    assert [
        sum(node["op"] == "matmul" for node in item["nodes"]) for item in programs
    ] == [1, 1]
    output = document["graph"]["nodes"][-1]
    assert [field["name"] for field in output["input_ports"][0]["schema"]] == [
        "x",
        "y",
        "score",
    ]


def test_real_explicit_table_stages_execute_separate_matmuls() -> None:
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    result = _explicit_table_matmul_stages().collect(
        {
            "prices": pa.table({"x": [1.0, 2.0], "y": [3.0, 4.0]}),
            "first_matrix": cfa.from_arrow(
                pa.array([1.0, 0.0, 0.0, 1.0]), shape=(2, 2)
            ),
            "coefficients": cfa.from_arrow(pa.array([2.0, 1.0]), shape=(2, 1)),
        },
        runtime=runtime,
    )
    assert result.to_pydict() == {"x": [1.0, 2.0], "y": [3.0, 4.0], "score": [5.0, 8.0]}


@pytest.mark.parametrize("program_json", ["[]", "{"])
def test_real_native_planning_binding_rejects_invalid_json(program_json: str) -> None:
    with pytest.raises(cf.CalcFlowError, match="program_json"):
        _native._infer_native_array_program(program_json)


def test_real_native_planning_binding_limits_utf8_bytes_before_json_parsing() -> None:
    with pytest.raises(cf.CalcFlowError, match="program_json.*64 KiB"):
        _native._infer_native_array_program("é" * 32_769)


def test_real_documented_native_array_collection_example(tmp_path: Path) -> None:
    document = (Path(__file__).resolve().parents[2] / "docs/python-api.md").read_text()
    section = document.split("### Native ArrayExpr declarations and collection", 1)[1]
    snippet = section.split("```python\n", 1)[1].split("```", 1)[0]
    script = tmp_path / "native_array_collection.py"
    script.write_text(snippet, encoding="utf-8")
    runpy.run_path(str(script))


def test_real_collect_array_combines_table_and_dynamic_array_roots() -> None:
    source = cf.table_input(
        "prices", schema=(cf.Field("x", "float64"), cf.Field("y", "float64"))
    )
    right = cf.array_input("dynamic_rhs", dtype="float64", shape=(2, 1))
    matrix = cf.linalg.from_columns(source, columns=("x", "y"), backend="native")
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    output = cf.linalg.matmul(matrix, right).collect_array(
        {
            "prices": pa.table({"x": [1.0, 2.0], "y": [3.0, 4.0]}),
            "dynamic_rhs": cfa.from_arrow(pa.array([2.0, 1.0]), shape=(2, 1)),
        },
        runtime=runtime,
    )
    assert output.shape == (2, 1)
    assert output.to_arrow().to_pylist() == [5.0, 8.0]


def test_real_native_negation_uses_rust_query_and_executor() -> None:
    values = cf.array_input("values", dtype="float32", shape=(None,))
    supplied = cfa.from_arrow(pa.array([1.0, None, -3.0], type=pa.float32()))
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    result = (-values).collect_array({"values": supplied}, runtime=runtime)
    assert result.dtype == "float32"
    assert result.shape == (3,)
    assert result.to_arrow().to_pylist() == [-1.0, None, 3.0]
    assert supplied.to_arrow().to_pylist() == [1.0, None, -3.0]


def _distinct_native_attachment_target(transform: str) -> cf.TableExpr:
    source = cf.table_input(
        "prices", schema=(cf.Field("x", "float64"), cf.Field("y", "float64"))
    )
    if transform == "filter":
        target = source.filter(source["x"] > 0.0)
    elif transform == "with_columns":
        target = source.with_columns(z=source["x"] * 10.0)
    else:
        target = source.select("x", "y")
    values = cf.linalg.from_columns(source, columns=("x",), backend="native")
    return cf.table.attach_columns(target, values, names=("score",))


@pytest.mark.parametrize("transform", ["filter", "with_columns", "project"])
def test_native_attachment_document_rejects_a_distinct_target_table_node(
    transform: str,
) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    runtime = cf.Runtime()
    cfa.register_native(runtime)
    output = _distinct_native_attachment_target(transform)
    program = cf.Program("source-proof", engine="sql", outputs={"out": output})
    with pytest.raises(cf.CompileError, match="same TableExpr node"):
        lower_program_document(program, runtime, "batch")
    assert any(
        issue.code == "schema_mismatch"
        and issue.path.endswith("attach_columns.array.source")
        for issue in program.analyze(runtime).issues
    )


@pytest.mark.parametrize("transform", ["filter", "with_columns", "project"])
def test_real_native_attachment_rejects_a_distinct_target_table_node(
    transform: str,
) -> None:
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    with pytest.raises(cf.CompileError, match="same TableExpr node"):
        _distinct_native_attachment_target(transform).collect(
            {"prices": pa.table({"x": [-1.0, 2.0], "y": [10.0, 20.0]})},
            runtime=runtime,
        )


def _matching_filtered_native_attachment() -> cf.TableExpr:
    source = cf.table_input(
        "prices", schema=(cf.Field("x", "float64"), cf.Field("y", "float64"))
    )
    filtered = source.filter(source["x"] > 0.0)
    values = cf.linalg.from_columns(filtered, columns=("x",), backend="native")
    return cf.table.attach_columns(filtered, values + 5.0, names=("score",))


def test_native_attachment_document_keeps_the_matching_filter_stage() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    runtime = cf.Runtime()
    cfa.register_native(runtime)
    document = lower_program_document(
        cf.Program(
            "matching-filter",
            engine="sql",
            outputs={"out": _matching_filtered_native_attachment()},
        ),
        runtime,
        "batch",
    )
    assert any(
        node["operator"].get("filter") is not None
        for node in document["graph"]["nodes"]
    )
    output = document["graph"]["nodes"][-1]
    assert [field["name"] for field in output["input_ports"][0]["schema"]] == [
        "x",
        "y",
        "score",
    ]


def test_real_native_attachment_matching_filter_preserves_rows_and_schema() -> None:
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    result = _matching_filtered_native_attachment().collect(
        {"prices": pa.table({"x": [-1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})},
        runtime=runtime,
    )
    assert result.to_pydict() == {
        "x": [2.0, 3.0],
        "y": [20.0, 30.0],
        "score": [7.0, 8.0],
    }


def test_native_bridge_rejects_a_distinct_target_without_analysis_gate() -> None:
    from calc_flow.symbolic.analyzer import _run
    from calc_flow.symbolic.lower.native_arrays import _NativeGraph

    output = _distinct_native_attachment_target("filter")
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    program = cf.Program("source-proof", engine="sql", outputs={"out": output})
    analyzer, _ = _run(program, runtime, "batch")
    graph = _NativeGraph(program, analyzer, 0, "error", cfa.ArrayExecutionOptions())
    with pytest.raises(cf.CompileError, match="same TableExpr node"):
        graph._bridge(output._node.args[1], output._node)


def test_real_static_parameter_fans_out_to_table_and_standalone_consumers() -> None:
    source = cf.table_input(
        "prices", schema=(cf.Field("x", "float64"), cf.Field("y", "float64"))
    )
    matrix = cf.linalg.from_columns(source, columns=("x", "y"), backend="native")
    native = cf.array_input("native_values", dtype="float64", shape=(None, 2))
    beta = cf.parameter(
        "shared_coefficients",
        kind="array",
        backend="native",
        dtype="float64",
        shape=(2, 1),
    )
    first = cf.table.attach_columns(
        source, cf.linalg.matmul(matrix, beta), names=("a",)
    )
    second = cf.table.attach_columns(
        source, cf.linalg.matmul(matrix + 1.0, beta), names=("b",)
    )
    first_mean = cf.linalg.from_columns(first, columns=("a",), backend="native").mean(
        axis=0
    )
    second_mean = cf.linalg.from_columns(second, columns=("b",), backend="native").mean(
        axis=0
    )
    standalone_mean = cf.linalg.matmul(native, beta).mean(axis=0)
    output = first_mean + second_mean + standalone_mean
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    inputs = {
        "prices": pa.table({"x": [1.0, 2.0], "y": [3.0, 4.0]}),
        "native_values": cfa.from_arrow(pa.array([1.0, 3.0, 2.0, 4.0]), shape=(2, 2)),
        "shared_coefficients": cfa.from_arrow(pa.array([2.0, 1.0]), shape=(2, 1)),
    }
    first_result = output.collect_array(inputs, runtime=runtime)
    assert first_result.shape == (1,)
    assert first_result.to_arrow().to_pylist() == [22.5]
    next_inputs = {
        **inputs,
        "shared_coefficients": cfa.from_arrow(pa.array([1.0, 1.0]), shape=(2, 1)),
    }
    assert output.collect_array(
        next_inputs, runtime=runtime
    ).to_arrow().to_pylist() == [17.0]
    assert inputs["shared_coefficients"].to_arrow().to_pylist() == [2.0, 1.0]
