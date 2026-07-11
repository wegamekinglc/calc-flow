from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from calc_flow import (
    ArrayExpressionOperator,
    Batch,
    DataFusionExecutionError,
    DuplicateUdfError,
    ExpressionOperator,
    Pipeline,
    SqlOperator,
    StatelessOperator,
    UdfExecutionError,
    UdfReference,
    UdfRegistry,
    UdfVersionConflictError,
    UnknownUdfError,
)
from calc_flow.engine import DataFusionEngine, JaxEngine, NumpyEngine


def _registry() -> UdfRegistry:
    registry = UdfRegistry()

    @registry.datafusion_scalar(
        name="double_value",
        version="1",
        input_fields=[pa.field("value", pa.int64())],
        return_field=pa.field("result", pa.int64()),
        volatility="immutable",
        description="Double an integer column.",
    )
    def double_value(values):
        return pc.multiply(values, 2)

    @registry.array(
        name="triple",
        version="1",
        argument_count=1,
        description="Triple an array.",
    )
    def triple(values):
        xp = values.__array_namespace__()
        return xp.multiply(values, 3)

    return registry


def test_udf_reference_round_trips_serializable_identity() -> None:
    reference = UdfReference("double_value", "1")

    assert UdfReference.from_dict(reference.to_dict()) == reference
    with pytest.raises(ValueError, match="exactly"):
        UdfReference.from_dict({"name": "double_value"})
    with pytest.raises(TypeError, match="strings"):
        UdfReference.from_dict({"name": "double_value", "version": 1})


def test_registry_catalog_exposes_metadata_without_executable_code() -> None:
    catalog = _registry().catalog()
    encoded = json.dumps(catalog)

    assert [entry["kind"] for entry in catalog] == [
        "array",
        "datafusion_scalar",
    ]
    scalar = catalog[1]
    assert scalar["name"] == "double_value"
    assert scalar["version"] == "1"
    assert scalar["parameters"] == [
        {"name": "value", "type": "int64", "nullable": True}
    ]
    assert scalar["return"]["type"] == "int64"
    assert "implementation" not in encoded
    assert "__main__" not in encoded


def test_registry_snapshot_does_not_observe_later_registration() -> None:
    registry = _registry()
    snapshot = registry.snapshot()

    @registry.array(name="later", version="1", argument_count=1)
    def later(values):
        return values

    with pytest.raises(UnknownUdfError, match="later@1"):
        snapshot.resolve_array(UdfReference("later", "1"))
    assert registry.snapshot().resolve_array(UdfReference("later", "1"))
    with pytest.raises(AttributeError, match="immutable"):
        snapshot._array = {}  # type: ignore[attr-defined]


def test_registry_rejects_duplicate_identity() -> None:
    registry = _registry()

    with pytest.raises(DuplicateUdfError, match="already registered"):

        @registry.array(name="triple", version="1", argument_count=1)
        def duplicate(values):
            return values


@pytest.mark.parametrize(
    ("name", "version"),
    [("NotLower", "1"), ("has-dash", "1"), ("valid_name", "")],
)
def test_registry_rejects_invalid_identity(name: str, version: str) -> None:
    registry = UdfRegistry()

    with pytest.raises(ValueError):
        registry.array(name=name, version=version, argument_count=1)


def test_registry_rejects_invalid_datafusion_signature() -> None:
    registry = UdfRegistry()

    with pytest.raises(ValueError, match="at least one"):
        registry.datafusion_scalar(
            name="empty",
            version="1",
            input_fields=[],
            return_field=pa.int64(),
        )
    with pytest.raises(ValueError, match="unique"):
        registry.datafusion_scalar(
            name="duplicate_fields",
            version="1",
            input_fields=[pa.field("x", pa.int64()), pa.field("x", pa.int64())],
            return_field=pa.int64(),
        )
    with pytest.raises(ValueError, match="volatility"):
        registry.datafusion_scalar(
            name="bad_volatility",
            version="1",
            input_fields=[pa.int64()],
            return_field=pa.int64(),
            volatility="sometimes",
        )


def test_registry_rejects_invalid_array_signature() -> None:
    with pytest.raises(ValueError, match="argument_count"):
        UdfRegistry().array(name="bad", version="1", argument_count=0)


def test_datafusion_udf_validates_input_nullability_and_output_length() -> None:
    registry = UdfRegistry()

    @registry.datafusion_scalar(
        name="short_result",
        version="1",
        input_fields=[pa.field("value", pa.int64(), nullable=False)],
        return_field=pa.int64(),
    )
    def short_result(values):
        return values.slice(0, max(0, len(values) - 1))

    specification = registry.snapshot().resolve_datafusion(
        UdfReference("short_result", "1")
    )

    with pytest.raises(UdfExecutionError, match="non-nullable input"):
        specification.invoke(pa.array([1, None], type=pa.int64()))
    with pytest.raises(UdfExecutionError, match="values for 2 input rows"):
        specification.invoke(pa.array([1, 2], type=pa.int64()))


def test_datafusion_engine_executes_selected_registered_udf() -> None:
    registry = _registry()
    engine = DataFusionEngine(
        udf_registry=registry,
        udfs=[UdfReference("double_value", "1")],
    )

    result = engine.evaluate(
        "doubled = double_value(value)",
        Batch.table(pa.table({"value": [1, 2, None]})),
    )

    assert result.table_payload.to_pylist() == [
        {"value": 1, "doubled": 2},
        {"value": 2, "doubled": 4},
        {"value": None, "doubled": None},
    ]


def test_registered_udf_executes_in_expression_and_sql_operators() -> None:
    registry = _registry()
    reference = UdfReference("double_value", "1")
    pipeline = (
        Pipeline("udfs", udf_registry=registry)
        .then(
            ExpressionOperator(
                "first", "doubled = double_value(value)", udfs=[reference]
            )
        )
        .then(
            SqlOperator(
                "second",
                "select value, doubled, double_value(doubled) as quadrupled "
                "from input_table",
                inputs=("input_table",),
                udfs=[reference],
            )
        )
    )

    result = pipeline.compile().execute(
        {"input": Batch.table(pa.table({"value": [2, 4]}))}
    )

    assert result.output.table_payload.to_pylist() == [
        {"value": 2, "doubled": 4, "quadrupled": 8},
        {"value": 4, "doubled": 8, "quadrupled": 16},
    ]
    assert len(result.datafusion_metrics) == 2


def test_pipeline_registers_only_explicit_udf_references() -> None:
    pipeline = Pipeline("undeclared", udf_registry=_registry()).then(
        ExpressionOperator("calculate", "result = double_value(value)")
    )

    with pytest.raises(ValueError, match="double_value"):
        pipeline.compile().execute({"input": Batch.table(pa.table({"value": [1]}))})


def test_compile_rejects_unknown_udf_version_before_execution() -> None:
    calls: list[bool] = []
    pipeline = Pipeline("unknown", udf_registry=_registry()).then(
        ExpressionOperator(
            "calculate",
            "result = double_value(value)",
            udfs=[UdfReference("double_value", "99")],
        )
    )
    pipeline.then(
        StatelessOperator(
            "downstream",
            lambda inputs, context: calls.append(True) or {"output": inputs["input"]},
        )
    )

    with pytest.raises(UnknownUdfError, match="double_value@99"):
        pipeline.compile()
    assert calls == []


def test_compile_rejects_two_versions_of_same_datafusion_name() -> None:
    registry = _registry()

    @registry.datafusion_scalar(
        name="double_value",
        version="2",
        input_fields=[pa.int64()],
        return_field=pa.int64(),
    )
    def double_v2(values):
        return pc.multiply(values, 2)

    pipeline = (
        Pipeline("conflict", udf_registry=registry)
        .then(
            ExpressionOperator(
                "one",
                "first = double_value(value)",
                udfs=[UdfReference("double_value", "1")],
            )
        )
        .then(
            ExpressionOperator(
                "two",
                "second = double_value(first)",
                udfs=[UdfReference("double_value", "2")],
            )
        )
    )

    with pytest.raises(UdfVersionConflictError, match="both"):
        pipeline.compile()


def test_udf_output_type_violation_stops_before_downstream() -> None:
    registry = UdfRegistry()

    @registry.datafusion_scalar(
        name="wrong_type",
        version="1",
        input_fields=[pa.int64()],
        return_field=pa.int64(),
    )
    def wrong_type(values):
        return pa.array(["wrong"] * len(values))

    downstream_calls: list[bool] = []
    pipeline = (
        Pipeline("wrong", udf_registry=registry)
        .then(
            ExpressionOperator(
                "udf",
                "result = wrong_type(value)",
                udfs=[UdfReference("wrong_type", "1")],
            )
        )
        .then(
            StatelessOperator(
                "downstream",
                lambda inputs, context: (
                    downstream_calls.append(True) or {"output": inputs["input"]}
                ),
            )
        )
    )

    with pytest.raises(DataFusionExecutionError, match="expected int64"):
        pipeline.compile().execute({"input": Batch.table(pa.table({"value": [1]}))})
    assert downstream_calls == []


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_array_expression_operator_executes_registered_udf(backend: str) -> None:
    if backend == "jax":
        pytest.importorskip("jax")
        import jax.numpy as xp
    else:
        import numpy as xp

    pipeline = Pipeline("array", udf_registry=_registry()).then(
        ArrayExpressionOperator(
            "calculate",
            "triple(x) + 1",
            backend=backend,
            udfs=[UdfReference("triple", "1")],
        )
    )

    result = pipeline.compile().execute({"input": Batch.array(xp.asarray([1, 2, 3]))})

    assert result.output.array_payload.tolist() == [4, 7, 10]


@pytest.mark.parametrize("engine_type", [NumpyEngine, JaxEngine])
def test_array_engine_executes_configured_udf(engine_type) -> None:
    if engine_type is JaxEngine:
        pytest.importorskip("jax")
    engine = engine_type(udf_registry=_registry(), udfs=[UdfReference("triple", "1")])
    batch = Batch.array(engine.xp.asarray([2, 4]))

    result = engine.evaluate("triple(x)", batch)

    assert result.array_payload.tolist() == [6, 12]


def test_array_expression_rejects_undeclared_function() -> None:
    import numpy as np

    engine = NumpyEngine()

    with pytest.raises(ValueError, match="approved xp"):
        engine.evaluate("triple(x)", Batch.array(np.asarray([1])))


def test_array_udf_validates_argument_count_and_return_contract() -> None:
    import numpy as np

    registry = UdfRegistry()

    @registry.array(name="broken", version="1", argument_count=1)
    def broken(values):
        return [1, 2, 3]

    engine = NumpyEngine(udf_registry=registry, udfs=[UdfReference("broken", "1")])
    batch = Batch.array(np.asarray([1, 2, 3]))

    with pytest.raises(UdfExecutionError, match="Array API"):
        engine.evaluate("broken(x)", batch)
    with pytest.raises(UdfExecutionError, match="expects 1 arguments"):
        engine.evaluate("broken(x, x)", batch)


def test_array_udf_cannot_change_backend() -> None:
    import numpy as np

    jnp = pytest.importorskip("jax.numpy")
    registry = UdfRegistry()

    @registry.array(name="move_backend", version="1", argument_count=1)
    def move_backend(values):
        return jnp.asarray(values)

    engine = NumpyEngine(
        udf_registry=registry, udfs=[UdfReference("move_backend", "1")]
    )

    with pytest.raises(UdfExecutionError, match="changed array backend"):
        engine.evaluate("move_backend(x)", Batch.array(np.asarray([1, 2, 3])))


def test_array_operator_rejects_unknown_version_during_compile() -> None:
    pipeline = Pipeline("array", udf_registry=_registry()).then(
        ArrayExpressionOperator(
            "calculate",
            "triple(x)",
            backend="numpy",
            udfs=[UdfReference("triple", "2")],
        )
    )

    with pytest.raises(UnknownUdfError, match="triple@2"):
        pipeline.compile()


def test_deserialized_udf_reference_has_identical_pipeline_fingerprint() -> None:
    registry = _registry()
    direct = UdfReference("double_value", "1")
    restored = UdfReference.from_dict({"name": "double_value", "version": "1"})

    first = Pipeline("same", udf_registry=registry).then(
        ExpressionOperator("calculate", "x = double_value(value)", udfs=[direct])
    )
    second = Pipeline("same", udf_registry=registry).then(
        ExpressionOperator("calculate", "x = double_value(value)", udfs=[restored])
    )

    assert first.compile().fingerprint == second.compile().fingerprint
