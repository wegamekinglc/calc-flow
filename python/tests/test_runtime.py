from __future__ import annotations

import asyncio
import json

import numpy as np
import pyarrow as pa
import pytest

from calc_flow import (
    Batch,
    ConfigError,
    PipelineBuilder,
    Runtime,
)


def _batch(value: int) -> Batch:
    return Batch.from_pyarrow(pa.table({"value": [value]}))


def _plan(name: str = "stream"):
    return PipelineBuilder(name).expression("calc", "result = value + 1").compile()


def test_registration_snapshot_is_success_only_and_defensive() -> None:
    runtime = Runtime()

    def provider(batch: Batch, _options: dict[str, object]) -> Batch:
        return batch

    def identity(value: pa.Array) -> pa.Array:
        return value

    runtime.register_provider("test", "identity", "1", provider)
    runtime.register_scalar_udf(
        provider="python",
        name="identity",
        version="1",
        input_types=("int64",),
        return_type="int64",
        volatility="immutable",
        function=identity,
    )
    snapshot = runtime._registration_snapshot()
    assert [(record["kind"], record["name"]) for record in snapshot] == [
        ("provider", "identity"),
        ("scalar_udf", "identity"),
    ]
    assert snapshot[0] == {
        "kind": "provider",
        "provider": "test",
        "name": "identity",
        "version": "1",
        "callback": provider,
        "options_schema": None,
    }
    snapshot[0]["name"] = "mutated"
    assert runtime._registration_snapshot()[0]["name"] == "identity"

    with pytest.raises(ConfigError, match="duplicate"):
        runtime.register_provider("test", "identity", "1", provider)
    with pytest.raises(ConfigError, match="duplicate"):
        runtime.register_scalar_udf(
            provider="python",
            name="identity",
            version="1",
            input_types=("int64",),
            return_type="int64",
            volatility="immutable",
            function=identity,
        )
    assert len(runtime._registration_snapshot()) == 2


def test_mapping_provider_executes_mixed_named_inputs() -> None:
    runtime = Runtime()

    def callback(
        inputs: dict[str, Batch],
        options: dict[str, object],
    ) -> dict[str, Batch]:
        assert sorted(inputs) == ["table", "weights"]
        assert options == {"columns": ["value"]}
        assert (
            inputs["table"].to_pyarrow()["value"].chunk(0).buffers()[1].address
            == table.to_pyarrow()["value"].chunk(0).buffers()[1].address
        )
        assert inputs["weights"].array is weights.array
        return {"output": inputs["weights"]}

    runtime._register_mapping_provider(
        "test",
        "table_matmul",
        "1",
        callback,
        input_ports=(("table", "table"), ("weights", "array")),
        output_ports=(("output", "array"),),
    )
    project = (
        PipelineBuilder("mapping")
        .table_matmul("multiply", backend="numpy", columns=("value",))
        .project
    )
    project["pipeline"]["nodes"][0]["operator"]["provider"] = "test"
    plan = runtime.compile_project(json.dumps(project))
    table = _batch(3)
    weights = Batch.from_array(np.array([[2.0]]), backend="numpy")

    result = plan.execute({"table": table, "weights": weights})

    assert result.outputs["output"].array.tolist() == [[2.0]]


def test_mapping_provider_registration_validates_private_arguments() -> None:
    runtime = Runtime()

    with pytest.raises(TypeError, match="provider callback must be callable"):
        runtime._register_mapping_provider(
            "test",
            "mapping",
            "1",
            object(),
            input_ports=(("input", "array"),),
            output_ports=(("output", "array"),),
        )
    with pytest.raises(
        ValueError,
        match="mapping provider port kind must be 'table' or 'array'",
    ):
        runtime._register_mapping_provider(
            "test",
            "mapping",
            "1",
            lambda inputs, options: inputs,
            input_ports=(("input", "scalar"),),
            output_ports=(("output", "array"),),
        )


def test_mapping_registration_snapshot_preserves_copied_contracts() -> None:
    runtime = Runtime()
    input_ports = [("table", "table"), ("weights", "array")]
    output_ports = [("output", "array")]

    runtime._register_mapping_provider(
        "test",
        "mapping",
        "1",
        lambda inputs, options: {"output": inputs["weights"]},
        input_ports=input_ports,
        output_ports=output_ports,
    )
    input_ports.clear()
    output_ports.clear()

    snapshot = runtime._registration_snapshot()

    assert snapshot[0]["provider_mode"] == "mapping"
    assert snapshot[0]["input_ports"] == (
        ("table", "table"),
        ("weights", "array"),
    )
    assert snapshot[0]["output_ports"] == (("output", "array"),)
    snapshot[0]["input_ports"] = ()
    snapshot[0]["output_ports"] = ()
    assert runtime._registration_snapshot()[0]["input_ports"] == (
        ("table", "table"),
        ("weights", "array"),
    )
    assert runtime._registration_snapshot()[0]["output_ports"] == (("output", "array"),)


def test_execution_plan_lifecycle_is_async_defensive_and_guarded() -> None:
    plan = _plan("lifecycle")

    async def exercise() -> None:
        state = await plan.snapshot_async()
        state["calc"] = {"changed": True}
        assert await plan.snapshot_async() == {"calc": None}
        await plan.restore_async({"calc": None})
        await plan.reset_async()
        with pytest.raises(RuntimeError, match="await snapshot_async"):
            plan.snapshot()

    asyncio.run(exercise())
    assert plan.snapshot() == {"calc": None}
    plan.restore({"calc": None})
    plan.reset()
