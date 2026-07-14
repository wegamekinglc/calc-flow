from __future__ import annotations

import gc
import weakref
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from calc_flow import (
    Batch,
    PipelineBuilder,
    ProviderError,
    Runtime,
    register_jax,
    register_numpy,
)


def _external(
    name: str,
    provider: str,
    expression: str,
    *,
    options: dict[str, object] | None = None,
) -> PipelineBuilder:
    return PipelineBuilder(name).external(
        "calc",
        provider,
        "expression",
        "1",
        {"expression": expression, **(options or {})},
    )


def test_numpy_provider_owns_read_only_arrays() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    source = np.array([1.0, 2.0])
    batch = Batch.from_array(source, backend="numpy")
    source[0] = 99
    plan = _external("arrays", "numpy", "x * 2").compile(runtime)

    output = plan.execute({"input": batch}).outputs["output"].array

    assert output.tolist() == [2.0, 4.0]
    assert not output.flags.writeable
    with pytest.raises(ValueError):
        output[0] = 10
    with pytest.raises(ValueError):
        output.setflags(write=True)


def test_array_batch_copies_metadata_and_reports_shape_length() -> None:
    metadata = {"nested": {"value": 1}}
    batch = Batch.from_array(np.zeros((3, 2)), backend="numpy", metadata=metadata)
    metadata["nested"]["value"] = 2

    assert batch.kind == "array"
    assert batch.num_rows == 3
    assert batch.metadata == {"nested": {"value": 1}}
    assert not batch.array.flags.writeable
    with pytest.raises(TypeError, match="table batches do not contain an array"):
        _ = Batch.from_pyarrow(pa.table({"value": [1]})).array


def test_missing_python_provider_fails_during_compile() -> None:
    with pytest.raises(Exception, match="provider numpy:expression@1 is unavailable"):
        _external("arrays", "numpy", "x + 1").compile(Runtime())


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('whoami')",
        "x.real",
        "(lambda value: value)(x)",
        "[value for value in x]",
        "unknown",
        "unknown(x)",
        "sum(x, axis=0)",
        "sum(x, **{})",
        "True",
        "1e309",
    ],
)
def test_array_expression_rejects_unsafe_syntax(expression: str) -> None:
    with pytest.raises(ValueError, match="array expression"):
        _external("unsafe", "numpy", expression).compile()


def test_array_expression_enforces_node_and_depth_limits() -> None:
    too_many_nodes = "+".join("x" for _ in range(200))
    too_deep = "-" * 40 + "x"

    with pytest.raises(ValueError, match="node limit"):
        _external("large", "numpy", too_many_nodes).compile()
    with pytest.raises(ValueError, match="depth limit"):
        _external("deep", "numpy", too_deep).compile()


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("sum()", "sum expects 1 argument"),
        ("transpose(x, x)", "transpose expects 1 argument"),
        ("reshape(x)", "reshape expects 2 arguments"),
        ("reshape(x, (2, True))", "reshape dimensions must be integers"),
        ("reshape(x, (2, -1, -1))", "at most one -1"),
        ("reshape(x, (2, -2))", "reshape dimensions must be non-negative or -1"),
        ("reshape(x, (1000001,))", "reshape dimension limit"),
    ],
)
def test_array_expression_validates_function_arguments(
    expression: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _external("invalid", "numpy", expression).compile()


def test_numpy_provider_supports_bounded_array_operations() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    source = np.arange(6).reshape(2, 3)

    reshaped = _external("reshape", "numpy", "reshape(transpose(x), (6,))").compile(
        runtime
    )
    output = reshaped.execute(
        {"input": Batch.from_array(source, backend="numpy")}
    ).outputs["output"]

    assert output.array.tolist() == [0, 3, 1, 4, 2, 5]
    assert output.num_rows == 6
    assert not output.array.flags.writeable


def test_numpy_provider_preserves_metadata() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    plan = _external("metadata", "numpy", "mean(x)").compile(runtime)

    output = plan.execute(
        {
            "input": Batch.from_array(
                np.array([1.0, 3.0]),
                backend="numpy",
                metadata={"request": "demo"},
            )
        }
    ).outputs["output"]

    assert output.metadata == {"request": "demo"}
    assert output.num_rows == 1
    assert output.array.item() == 2.0


def test_provider_rejects_backend_mismatches_and_table_batches() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    plan = _external("backend", "numpy", "x + 1").compile(runtime)

    with pytest.raises(ProviderError, match="requires backend numpy"):
        plan.execute({"input": Batch._from_external(object(), "other", 1, {})})
    with pytest.raises(Exception, match="expects a Array batch"):
        plan.execute({"input": Batch.from_pyarrow(pa.table({"value": [1]}))})


def test_custom_array_udf_references_are_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match="custom array UDFs are unavailable"):
        _external(
            "udf",
            "numpy",
            "custom(x)",
            options={"udfs": [{"name": "custom", "version": "1"}]},
        ).compile()


def test_callback_failure_leaves_plan_reusable() -> None:
    class FailOnce:
        calls = 0

        def validate(self, options: dict[str, object]) -> None:
            if options != {"increment": 1}:
                raise ValueError("unexpected options")

        def __call__(self, batch: Batch, options: dict[str, object]) -> Batch:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("intentional callback failure")
            return Batch.from_array(
                batch.array + options["increment"],
                backend="numpy",
                metadata=batch.metadata,
            )

    runtime = Runtime()
    callback = FailOnce()
    runtime.register_provider("test", "increment", "1", callback)
    plan = (
        PipelineBuilder("reuse")
        .external("calc", "test", "increment", "1", {"increment": 1})
        .compile(runtime)
    )
    batch = Batch.from_array(np.array([1]), backend="numpy")

    with pytest.raises(ProviderError, match="intentional callback failure"):
        plan.execute({"input": batch})

    assert plan.execute({"input": batch}).outputs["output"].array.tolist() == [2]


def test_provider_registration_cycles_are_collected() -> None:
    class Callback:
        runtime: Runtime | None = None

        def validate(self, _options: dict[str, object]) -> None:
            pass

        def __call__(self, batch: Batch, _options: dict[str, object]) -> Batch:
            return batch

    runtime = Runtime()
    callback = Callback()
    callback.runtime = runtime
    callback_ref = weakref.ref(callback)
    runtime.register_provider("cycle", "identity", "1", callback)

    del callback
    del runtime
    gc.collect()

    assert callback_ref() is None


def test_jax_provider_retains_jax_arrays() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    runtime = Runtime()
    register_jax(runtime)
    source = jnp.array([1.0, 2.0])
    plan = _external("jax", "jax", "x ** 2").compile(runtime)

    output = (
        plan.execute({"input": Batch.from_array(source, backend="jax")})
        .outputs["output"]
        .array
    )

    assert isinstance(output, jax.Array)
    assert output.tolist() == [1.0, 4.0]


def test_jax_provider_keeps_constant_results_on_jax() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    runtime = Runtime()
    register_jax(runtime)
    plan = _external("jax_constant", "jax", "2").compile(runtime)

    output = (
        plan.execute({"input": Batch.from_array(jnp.array([1.0]), backend="jax")})
        .outputs["output"]
        .array
    )

    assert isinstance(output, jax.Array)
    assert output.item() == 2


def test_provider_registration_validates_public_arguments() -> None:
    runtime = Runtime()

    with pytest.raises(TypeError, match="callable"):
        runtime.register_provider("test", "identity", "1", object())
    with pytest.raises(Exception, match="duplicate provider"):
        runtime.register_provider("test", "identity", "1", lambda *_: None)
        runtime.register_provider("test", "identity", "1", lambda *_: None)


def test_registration_does_not_mutate_options() -> None:
    class Callback:
        def validate(self, options: dict[str, Any]) -> None:
            options["mutated"] = True

        def __call__(self, batch: Batch, _options: dict[str, Any]) -> Batch:
            return batch

    runtime = Runtime()
    runtime.register_provider("copy", "identity", "1", Callback())
    options: dict[str, object] = {"value": 1}

    PipelineBuilder("copy").external("calc", "copy", "identity", "1", options).compile(
        runtime
    )

    assert options == {"value": 1}
