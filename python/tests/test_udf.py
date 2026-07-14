from __future__ import annotations

import asyncio
import gc
import weakref

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from calc_flow import Batch, ConfigError, ExecutionError, PipelineBuilder, Runtime


def _register(
    runtime: Runtime,
    function: object,
    *,
    provider: str = "python",
    name: str = "transform",
    version: str = "1",
    input_types: tuple[str, ...] = ("int64",),
    return_type: str = "int64",
    volatility: str = "immutable",
) -> None:
    runtime.register_scalar_udf(
        provider=provider,
        name=name,
        version=version,
        input_types=input_types,
        return_type=return_type,
        volatility=volatility,
        function=function,
    )


def _plan(runtime: Runtime, expression: str = "result = transform(value)"):
    return (
        PipelineBuilder("udf")
        .expression("calc", expression, udfs=(("python", "transform", "1"),))
        .compile(runtime)
    )


def _execute(plan, values: pa.Array | list[object]) -> list[object]:
    table = pa.table({"value": values})
    return (
        plan.execute({"input": Batch.from_pyarrow(table)})
        .outputs["output"]
        .to_pyarrow()["result"]
        .to_pylist()
    )


def test_python_scalar_udf_executes_arrays_and_preserves_nulls() -> None:
    runtime = Runtime()
    seen: list[pa.Array] = []

    def double(value: pa.Array) -> pa.Array:
        seen.append(value)
        return pc.multiply(value, 2)

    _register(runtime, double)
    assert _execute(_plan(runtime), [2, None, 4]) == [4, None, 8]
    assert isinstance(seen[0], pa.Array)


def test_python_scalar_udf_requires_exact_input_types_before_callback() -> None:
    runtime = Runtime()
    calls = 0

    def identity(value: pa.Array) -> pa.Array:
        nonlocal calls
        calls += 1
        return value

    _register(runtime, identity)
    plan = _plan(runtime)
    with pytest.raises(
        ExecutionError,
        match=r"python:transform@1.*exact Arrow input types.*int64.*int32",
    ):
        _execute(plan, pa.array([1, 2], type=pa.int32()))
    assert calls == 0

    assert _execute(plan, pa.array([1, 2], type=pa.int64())) == [1, 2]
    assert calls == 1


@pytest.mark.parametrize(
    ("expression", "values", "actual"),
    [
        ("result = transform(value)", pa.array([1.0], type=pa.float64()), "float64"),
        ("result = transform()", pa.array([1], type=pa.int64()), "zero arguments"),
        (
            "result = transform(value, value)",
            pa.array([1], type=pa.int64()),
            "2 arguments",
        ),
    ],
)
def test_python_scalar_udf_planner_errors_include_identity_and_exact_contract(
    expression: str, values: pa.Array, actual: str
) -> None:
    runtime = Runtime()
    calls = 0

    def identity(value: pa.Array) -> pa.Array:
        nonlocal calls
        calls += 1
        return value

    _register(runtime, identity)
    with pytest.raises(
        ExecutionError,
        match=rf"python:transform@1.*exact Arrow input types.*{actual}",
    ):
        _execute(_plan(runtime, expression), values)
    assert calls == 0


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (lambda value: 7, [7, 7, 7]),
        (lambda value: pa.scalar(8, type=pa.int64()), [8, 8, 8]),
        (lambda value: pa.array([9, 10, 11], type=pa.int64()), [9, 10, 11]),
    ],
)
def test_python_scalar_udf_accepts_scalar_and_array_results(
    function: object, expected: list[int]
) -> None:
    runtime = Runtime()
    _register(runtime, function)
    assert _execute(_plan(runtime), [1, 2, 3]) == expected


def test_zero_argument_and_all_scalar_udfs_keep_scalar_mode() -> None:
    runtime = Runtime()
    _register(runtime, lambda: pa.scalar(5, type=pa.int64()), input_types=())
    plan = _plan(runtime, "result = transform()")
    assert _execute(plan, [1, 2, 3]) == [5, 5, 5]

    runtime = Runtime()
    _register(runtime, lambda value: value, input_types=("int64",))
    plan = _plan(runtime, "result = transform(CAST(6 AS BIGINT))")
    assert _execute(plan, [1, 2, 3]) == [6, 6, 6]


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (lambda value: pa.array([1], type=pa.int64()), "output length"),
        (lambda value: pa.array([1, 2], type=pa.float64()), "output Arrow type"),
        (lambda value: pa.scalar(1.0, type=pa.float64()), "output Arrow type"),
        (lambda value: pa.chunked_array([[1], [2]]), "PyArrow array or scalar"),
        (lambda value: pa.table({"value": [1, 2]}), "PyArrow array or scalar"),
    ],
)
def test_python_scalar_udf_rejects_invalid_outputs_and_plan_is_reusable(
    function: object, message: str
) -> None:
    runtime = Runtime()
    _register(runtime, function)
    plan = _plan(runtime)
    with pytest.raises(ExecutionError, match=message):
        _execute(plan, [1, 2])
    with pytest.raises(ExecutionError, match="python:transform@1"):
        _execute(plan, [1, 2])


def test_python_scalar_udf_wraps_callback_errors_with_exact_identity() -> None:
    runtime = Runtime()

    def fail(value: pa.Array) -> pa.Array:
        raise RuntimeError("kaboom")

    _register(runtime, fail)
    plan = _plan(runtime)
    with pytest.raises(ExecutionError, match=r"python:transform@1.*kaboom"):
        _execute(plan, [1])


def test_scalar_udf_registration_validates_before_mutating_catalog() -> None:
    runtime = Runtime()
    assert runtime.catalog() == []
    invalid = (
        {"input_types": ("object",)},
        {"return_type": "object"},
        {"volatility": "sometimes"},
        {"provider": "not valid!"},
        {"function": object()},
    )
    for update in invalid:
        options = {
            "provider": "python",
            "name": "transform",
            "version": "1",
            "input_types": ("int64",),
            "return_type": "int64",
            "volatility": "immutable",
            "function": lambda value: value,
            **update,
        }
        with pytest.raises((TypeError, ConfigError, ValueError)):
            runtime.register_scalar_udf(**options)
        assert runtime.catalog() == []


def test_catalog_is_sorted_defensive_and_redacted() -> None:
    runtime = Runtime()
    secret = "do-not-leak"
    _register(
        runtime,
        lambda value: (value, secret)[0],
        name="zeta",
        version="2",
        volatility="volatile",
    )
    _register(
        runtime,
        lambda value: value,
        name="alpha",
        input_types=("string",),
        return_type="string",
    )

    catalog = runtime.catalog()
    assert [entry["name"] for entry in catalog] == ["alpha", "zeta"]
    assert catalog[0] == {
        "kind": "data_fusion_scalar",
        "name": "alpha",
        "provider": "python",
        "signature": {"input_types": ["string"], "return_type": "string"},
        "version": "1",
        "volatility": "immutable",
    }
    assert secret not in str(catalog)
    assert "function" not in str(catalog)
    catalog.clear()
    assert len(runtime.catalog()) == 2


def test_duplicate_registration_leaves_catalog_and_roots_unchanged() -> None:
    runtime = Runtime()
    _register(runtime, lambda value: value)
    before = runtime.catalog()
    with pytest.raises(ConfigError, match="duplicate"):
        _register(runtime, lambda value: value)
    assert runtime.catalog() == before


def test_unknown_version_and_sql_name_collision_fail_compilation() -> None:
    runtime = Runtime()
    _register(runtime, lambda value: value)
    with pytest.raises(ConfigError, match="python:transform@2"):
        PipelineBuilder("unknown").expression(
            "calc", "result = transform(value)", udfs=(("python", "transform", "2"),)
        ).compile(runtime)

    _register(runtime, lambda value: value, provider="plugin")
    with pytest.raises(ConfigError, match="conflicting DataFusion SQL names"):
        PipelineBuilder("collision").expression(
            "calc",
            "result = transform(value)",
            udfs=(("python", "transform", "1"), ("plugin", "transform", "1")),
        ).compile(runtime)


def test_runtime_callback_cycle_is_collected() -> None:
    class Callback:
        owner: Runtime | None = None

        def __call__(self, value: pa.Array) -> pa.Array:
            return value

    callback = Callback()
    reference = weakref.ref(callback)
    runtime = Runtime()
    callback.owner = runtime
    _register(runtime, callback)
    del callback, runtime
    gc.collect()
    assert reference() is None


def test_async_plan_retains_udf_callback_after_runtime_references_drop() -> None:
    async def exercise() -> None:
        runtime = Runtime()

        def callback(value: pa.Array) -> pa.Array:
            return pc.add(value, 1)

        _register(runtime, callback)
        plan = _plan(runtime)
        del callback, runtime
        gc.collect()

        result = await plan.execute_async(
            {"input": Batch.from_pyarrow(pa.table({"value": [1, 2]}))}
        )
        assert result.outputs["output"].to_pyarrow()["result"].to_pylist() == [
            2,
            3,
        ]
        await asyncio.sleep(0)

    asyncio.run(exercise())
