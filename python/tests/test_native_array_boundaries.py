from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa
import pytest

import calc_flow as cf
import calc_flow.array as cfa


def _runtime() -> cf.Runtime:
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    return runtime


def _values() -> cfa.NativeArray:
    return cfa.from_arrow(pa.array([2.0, None, 4.0]))


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda value: 3.0 * value, [6.0, None, 12.0]),
        (lambda value: 8.0 / value, [4.0, None, 2.0]),
        (lambda value: value.min(skip_nulls=True), [2.0]),
        (lambda value: value.max(skip_nulls=True), [4.0]),
    ],
)
def test_native_reflected_arithmetic_and_extrema_preserve_nulls(
    operation: Callable[[cf.ArrayExpr], cf.ArrayExpr],
    expected: list[float | None],
) -> None:
    source = cf.array_input("values", dtype="float64", shape=(None,))
    array = _values()
    result = operation(source).collect_array({"values": array}, runtime=_runtime())
    assert result.to_arrow().to_pylist() == expected
    assert array.to_arrow().to_pylist() == [2.0, None, 4.0]


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda value: False & value, [False, False, False]),
        (lambda value: True | value, [True, True, True]),
    ],
)
def test_reflected_native_logic_uses_kleene_missing_value_rules(
    operation: Callable[[cf.ArrayExpr], cf.ArrayExpr],
    expected: list[bool],
) -> None:
    source = cf.array_input("flags", dtype="bool", shape=(None,))
    flags = cfa.from_arrow(pa.array([True, None, False]))
    result = operation(source).collect_array({"flags": flags}, runtime=_runtime())
    assert result.to_arrow().to_pylist() == expected
    assert flags.to_arrow().to_pylist() == [True, None, False]


@pytest.mark.parametrize(
    ("shape", "error"), [("3", TypeError), ((1, 1, 3), ValueError)]
)
def test_native_reshape_rejects_invalid_shape_structure(
    shape: object, error: type[Exception]
) -> None:
    source = cf.array_input("values", dtype="float64", shape=(3,))
    with pytest.raises(error, match="ArrayExpr.reshape.shape"):
        source.reshape(shape)


@pytest.mark.parametrize("keyword", ["keepdims", "skip_nulls"])
def test_native_reduction_flags_require_exact_booleans(keyword: str) -> None:
    source = cf.array_input("values", dtype="float64", shape=(3,))
    with pytest.raises(TypeError, match=f"ArrayExpr.min.{keyword}"):
        source.min(**{keyword: 1})


@pytest.mark.parametrize(
    ("arguments", "error", "path"),
    [
        ({"backend": "numpy", "shape": (3,)}, ValueError, "array_input.backend"),
        ({"shape": "3"}, TypeError, "array_input.shape"),
    ],
)
def test_native_input_rejects_implicit_backends_and_nonsequence_shapes(
    arguments: dict[str, object], error: type[Exception], path: str
) -> None:
    with pytest.raises(error, match=path):
        cf.array_input("values", dtype="float64", **arguments)


@pytest.mark.parametrize("kind, error", [(False, TypeError), ("tensor", ValueError)])
def test_static_array_kind_rejects_nonliteral_or_unknown_values(
    kind: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match="static_inputs.weights.kind"):
        cf.parameter("weights", kind=kind)


def test_native_collection_accepts_an_owned_batch_and_validates_its_rank() -> None:
    source = cf.array_input("values", dtype="float64", shape=(None,))
    batch = cf.Batch.from_array(_values(), backend="native")
    result = (source + 1.0).collect_array({"values": batch}, runtime=_runtime())
    assert result.to_arrow().to_pylist() == [3.0, None, 5.0]
    assert batch.array.to_arrow().to_pylist() == [2.0, None, 4.0]
    matrix = cfa.from_arrow(pa.array([1.0, 2.0, 3.0, 4.0]), shape=(2, 2))
    with pytest.raises(ValueError, match="inputs.values.shape: native rank"):
        source.collect_array({"values": matrix}, runtime=_runtime())


def test_native_collection_rejects_table_batches_and_nonmapping_inputs() -> None:
    source = cf.array_input("values", dtype="float64", shape=(None,))
    runtime = _runtime()
    table = cf.Batch.from_pyarrow(pa.table({"values": [1.0]}))
    with pytest.raises(TypeError, match="inputs.values.backend: expected a native"):
        source.collect_array({"values": table}, runtime=runtime)
    with pytest.raises(TypeError, match="collect.inputs: expected a mapping"):
        source.collect_array([], runtime=runtime)


def test_native_collection_rejects_invalid_runtime_and_execution_limits() -> None:
    source = cf.array_input("values", dtype="float64", shape=(None,))
    with pytest.raises(TypeError, match="explicit calc_flow Runtime"):
        source.collect_array({}, runtime=None)
    with pytest.raises(TypeError, match="array_options must be ArrayExecutionOptions"):
        source.collect_array({}, runtime=_runtime(), array_options={})


def test_native_collection_reports_missing_static_parameter_by_logical_name() -> None:
    source = cf.array_input("values", dtype="float64", shape=(None,))
    weights = cf.parameter(
        "weights", kind="array", backend="native", dtype="float64", shape=(3,)
    )
    with pytest.raises(ValueError, match="inputs.weights: missing static parameter"):
        (source + weights).collect_array({"values": _values()}, runtime=_runtime())


def test_repeated_native_table_operand_keeps_one_row_lineage_and_correct_values() -> (
    None
):
    source = cf.table_input("prices", schema=(cf.Field("price", "float64"),))
    matrix = cf.linalg.from_columns(source, columns=("price",), backend="native")
    attached = cf.table.attach_columns(
        source, matrix * matrix + matrix, names=("score",)
    )
    result = attached.collect(
        {"prices": pa.table({"price": [1.0, None, 3.0]})}, runtime=_runtime()
    )
    assert result.to_pydict() == {
        "price": [1.0, None, 3.0],
        "score": [2.0, None, 12.0],
    }
