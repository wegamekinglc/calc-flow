from __future__ import annotations

import builtins
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pyarrow as pa
import pytest

import calc_flow.array as array_module
from calc_flow import (
    Batch,
    CalcFlowError,
    CapabilityRule,
    PipelineBuilder,
    ProviderCapability,
    ProviderOptionsSchema,
    ProviderPort,
    Runtime,
    _native,
)
from calc_flow.capabilities import runtime_capabilities


def _capability_arguments() -> dict[str, object]:
    return {
        "provider": "native",
        "name": "array_expr",
        "version": "1",
        "input_ports": (),
        "output_ports": (),
        "options_schema": None,
        "modes": ("batch",),
        "finality": "unproven",
        "stateful": False,
        "microbatch_invariant": False,
        "requires_watermark": False,
        "checkpoint_support": "unproven",
        "state_version": None,
        "deterministic": True,
        "replay_safe": True,
        "supports_static_inputs": False,
        "partition_contract": "none",
        "array_rules": None,
    }


def test_options_bound_capability_uses_immutable_json_schema() -> None:
    schema = '{"additionalProperties":false,"properties":{},"type":"object"}'
    capability = ProviderCapability(
        **_capability_arguments(),
        port_binding="options",
        options_json_schema=schema,
    )

    assert capability.port_binding == "options"
    assert capability.input_ports == capability.output_ports == ()
    assert capability.options_json_schema == schema
    with pytest.raises(FrozenInstanceError):
        capability.options_json_schema = "{}"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "error", "field"),
    [
        ({"port_binding": "dynamic"}, ValueError, "port_binding"),
        ({"port_binding": None}, TypeError, "port_binding"),
        ({"port_binding": "options"}, ValueError, "options_json_schema"),
        ({"options_json_schema": "{}"}, ValueError, "options_json_schema"),
        (
            {"port_binding": "options", "options_json_schema": {}},
            TypeError,
            "options_json_schema",
        ),
        (
            {
                "port_binding": "options",
                "options_json_schema": "{}",
                "input_ports": (ProviderPort("x", "array", True),),
            },
            ValueError,
            "input_ports",
        ),
        (
            {
                "port_binding": "options",
                "options_json_schema": "{}",
                "options_schema": ProviderOptionsSchema(),
            },
            ValueError,
            "options_schema",
        ),
    ],
)
def test_provider_port_binding_rejects_ambiguous_or_non_strict_contracts(
    overrides: dict[str, object], error: type[Exception], field: str
) -> None:
    with pytest.raises(error, match=field):
        ProviderCapability(**{**_capability_arguments(), **overrides})


@pytest.mark.parametrize(
    "schema",
    [
        "not json",
        "[]",
        "{}",
        '{"type":"object","additionalProperties":true}',
        '{"type":"object","additionalProperties":false,"limit":NaN}',
        '{"type":"object","additionalProperties":false,"type":"object"}',
    ],
)
def test_options_json_schema_rejects_non_strict_json(schema: str) -> None:
    with pytest.raises(ValueError, match="options_json_schema"):
        ProviderCapability(
            **_capability_arguments(),
            port_binding="options",
            options_json_schema=schema,
        )


def test_capability_schema_five_preserves_fixed_port_defaults() -> None:
    assert Runtime().capabilities().schema_version == 5
    capability = ProviderCapability(**_capability_arguments())
    assert capability.port_binding == "fixed"
    assert capability.options_json_schema is None


@pytest.mark.parametrize(
    "name",
    (
        "native_float_bool_dtype",
        "native_scalar_vector_broadcast",
        "native_packed_rank2",
        "native_matmul",
        "native_axis_reduction",
    ),
)
def test_native_capability_rules_use_closed_versioned_identities(name: str) -> None:
    assert CapabilityRule(name, "1").name == name
    with pytest.raises(ValueError, match="unknown capability rule"):
        CapabilityRule(name, "2")


def _descriptor() -> dict[str, object]:
    return {
        **_capability_arguments(),
        "input_ports": [],
        "output_ports": [],
        "modes": ["batch"],
        "supports_static_inputs": True,
        "state_layouts": [],
        "port_binding": "options",
        "options_json_schema": (
            '{"additionalProperties":false,"properties":{},"type":"object"}'
        ),
        "array_rules": {
            "supported_dtypes": ["bool", "float32", "float64"],
            "safe_dtype_rule": {"name": "native_float_bool_dtype", "version": "1"},
            "shape_rules": [{"name": "native_scalar_vector_broadcast", "version": "1"}],
        },
    }


def _descriptors(first: dict[str, object] | None = None) -> list[dict[str, object]]:
    table = {
        **_descriptor(),
        "name": "table_array",
        "modes": ["batch", "stream"],
        "checkpoint_support": "stateless",
        "finality": "per_row_final",
        "microbatch_invariant": True,
        "partition_contract": "row_axis_independent",
    }
    fixture = (
        Path(__file__).with_name("fixtures")
        / "native_array_capabilities"
        / "rust_sequence_bundle_ir1.json"
    )
    sequence = next(
        descriptor
        for descriptor in json.loads(fixture.read_text())
        if descriptor["name"] == "array_sequence"
    )
    return [_descriptor() if first is None else first, sequence, table]


class _NativeRegistration:
    def __init__(self) -> None:
        self.calls = 0
        self.error: Exception | None = None

    def _register_native_array(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


def test_native_registration_commits_one_frozen_capability_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor()
    monkeypatch.setattr(
        _native,
        "_native_array_capabilities",
        lambda: _descriptors(descriptor),
        raising=False,
    )
    inner = _NativeRegistration()
    runtime = Runtime(_inner=inner)  # type: ignore[arg-type]
    runtime._symbolic_compile_cache["plan"] = object()
    runtime._symbolic_schema_cache["schema"] = object()

    array_module.register_native(runtime)

    snapshot, records = runtime.capability_registration_snapshot()
    capability = snapshot.providers[0]
    assert snapshot.scope.revision == 3
    assert len(records) == 3
    assert inner.calls == 1
    assert runtime._symbolic_cache_generation == 1
    assert runtime._symbolic_compile_cache == runtime._symbolic_schema_cache == {}
    assert records[0]["capability"] is capability
    assert records[0]["kind"] == "native_provider"
    assert "callback" not in records[0]
    assert capability.modes == ("batch",)
    assert capability.checkpoint_support == capability.finality == "unproven"
    assert capability.stateful is capability.microbatch_invariant is False
    assert capability.partition_contract == "none"
    assert capability.deterministic is capability.replay_safe is True
    assert capability.array_rules.supported_dtypes == ("bool", "float32", "float64")
    descriptor["modes"].append("stream")
    descriptor["array_rules"]["supported_dtypes"].append("int64")
    records[0]["provider"] = "changed"
    assert runtime.capabilities() == snapshot
    assert runtime.registration_snapshot()[0]["provider"] == "native"


def _registration_state(runtime: Runtime) -> tuple[object, ...]:
    return (
        runtime.capability_registration_snapshot(),
        runtime._symbolic_cache_generation,
        dict(runtime._symbolic_compile_cache),
        dict(runtime._symbolic_schema_cache),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"name": 42},
        {"provider": "other"},
        {"modes": ("batch",)},
        {"input_ports": ()},
        {"extra": True},
        {"array_rules": {"extra": True}},
    ],
)
def test_invalid_descriptor_is_rejected_before_native_insertion(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, object]
) -> None:
    descriptor = {**_descriptor(), **change}
    monkeypatch.setattr(
        _native,
        "_native_array_capabilities",
        lambda: _descriptors(descriptor),
        raising=False,
    )
    inner = _NativeRegistration()
    runtime = Runtime(_inner=inner)  # type: ignore[arg-type]
    runtime._symbolic_compile_cache["plan"] = object()
    runtime._symbolic_schema_cache["schema"] = object()
    before = _registration_state(runtime)

    with pytest.raises((TypeError, ValueError)):
        array_module.register_native(runtime)

    assert inner.calls == 0
    assert _registration_state(runtime) == before


def test_native_registration_error_keeps_snapshots_and_both_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _native, "_native_array_capabilities", _descriptors, raising=False
    )
    inner = _NativeRegistration()
    runtime = Runtime(_inner=inner)  # type: ignore[arg-type]
    array_module.register_native(runtime)
    runtime._symbolic_compile_cache["plan"] = object()
    runtime._symbolic_schema_cache["schema"] = object()
    before = _registration_state(runtime)
    inner.error = RuntimeError("native rejected registration")

    with pytest.raises(RuntimeError, match="native rejected registration"):
        array_module.register_native(runtime)

    assert inner.calls == 2
    assert _registration_state(runtime) == before
    inner.error = None
    runtime._cached_symbolic_compile("new", lambda: "retried")
    assert runtime._symbolic_compile_cache["new"] == "retried"


def test_real_native_registration_and_descriptor_queries_are_atomic() -> None:
    runtime = Runtime()
    descriptor = _native._native_array_capability()
    descriptor["modes"].append("stream")
    assert _native._native_array_capability()["modes"] == ["batch"]
    assert runtime.capabilities().scope.revision == 0

    array_module.register_native(runtime)

    capability = runtime.capabilities().providers[0]
    assert (capability.provider, capability.name, capability.version) == (
        "native",
        "array_expr",
        "1",
    )
    assert capability.port_binding == "options"
    assert capability.modes == ("batch",)
    assert capability.checkpoint_support == "unproven"
    assert (
        capability.options_json_schema
        == _native._native_array_capability()["options_json_schema"]
    )
    runtime._symbolic_compile_cache["plan"] = object()
    runtime._symbolic_schema_cache["schema"] = object()
    before = _registration_state(runtime)
    with pytest.raises(CalcFlowError, match="duplicate provider/name/version"):
        array_module.register_native(runtime)
    assert _registration_state(runtime) == before


def test_unavailable_native_symbolic_backend_never_imports_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def checked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "jax" or name.startswith("jax."):
            raise AssertionError("an unsupported native backend must not import JAX")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    with pytest.raises(TypeError, match="unsupported symbolic array backend 'native'"):
        array_module._symbolic_dtype_namespace("native")
    with pytest.raises(TypeError, match="unsupported symbolic array backend 'native'"):
        array_module._symbolic_result_dtype("native", "add", 1.0, 2.0)


def test_registration_during_cache_factory_keeps_the_new_generation_uncached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _native, "_native_array_capabilities", _descriptors, raising=False
    )
    inner = _NativeRegistration()
    runtime = Runtime(_inner=inner)  # type: ignore[arg-type]

    def compile_after_registration() -> object:
        array_module.register_native(runtime)
        return object()

    result = runtime._cached_symbolic_compile("pending", compile_after_registration)
    assert inner.calls == 1
    assert "pending" not in runtime._symbolic_compile_cache
    assert runtime._symbolic_cache_generation == 1
    assert runtime._cached_symbolic_compile("pending", lambda: result) is result


def test_cleared_native_registration_leaves_the_empty_runtime_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _native, "_native_array_capabilities", _descriptors, raising=False
    )
    inner = _NativeRegistration()
    inner.error = RuntimeError("Runtime has been cleared by garbage collection")
    runtime = Runtime(_inner=inner)  # type: ignore[arg-type]
    runtime._symbolic_compile_cache["plan"] = object()
    runtime._symbolic_schema_cache["schema"] = object()
    before = _registration_state(runtime)
    with pytest.raises(RuntimeError, match="Runtime has been cleared"):
        array_module.register_native(runtime)
    assert inner.calls == 1
    assert _registration_state(runtime) == before


def test_python_provider_records_keep_their_existing_conservative_semantics() -> None:
    runtime = Runtime()
    runtime.register_provider("acme", "transform", "1", lambda batch, _: batch)
    record = runtime.registration_snapshot()[0]
    record["capability"] = ProviderCapability(**_capability_arguments())
    snapshot = runtime_capabilities(
        session_id="test-session",
        revision=1,
        package_version="test-version",
        registrations=(record,),
    )

    capability = snapshot.providers[0]
    assert (capability.provider, capability.name) == ("acme", "transform")
    assert capability.checkpoint_support == "stateless"
    assert capability.deterministic is capability.replay_safe is False
    assert capability.port_binding == "fixed"
    assert capability.options_json_schema is None


def test_real_native_program_runs_in_batch_and_rejects_stream_compilation() -> None:
    runtime = Runtime()
    array_module.register_native(runtime)
    program = {
        "ir_version": 1,
        "inputs": [{"name": "input", "dtype": "float64", "shape": [None]}],
        "nodes": [
            {"id": "x", "op": "input", "name": "input"},
            {"id": "avg", "op": "rolling_mean", "input": "x", "window": 2},
        ],
        "outputs": [
            {"name": "output", "node": "avg", "dtype": "float64", "shape": [None]}
        ],
    }
    builder = PipelineBuilder("native-rolling").external(
        "calc", "native", "array_expr", "1", program
    )
    plan = builder.compile_batch(runtime)
    values = array_module.from_arrow(pa.array([1.0, None, 3.0]))
    batch = Batch.from_array(values, backend="native")
    result = plan.execute({"input": batch})
    output = result.outputs["output"]
    assert output.kind == "array"
    assert output.backend == "native"
    assert output.array.to_arrow().to_pylist() == [1.0, 1.0, 3.0]
    assert values.to_arrow().to_pylist() == [1.0, None, 3.0]
    assert result.datafusion_metrics == []
    next_batch = Batch.from_array(
        array_module.from_arrow(pa.array([5.0])), backend="native"
    )
    assert plan.execute({"input": next_batch}).outputs[
        "output"
    ].array.to_arrow().to_pylist() == [5.0]
    with pytest.raises(
        CalcFlowError,
        match=r"graph\.nodes\[0\]\.operator \[missing_provider\]: "
        r"provider native:array_expr@1 is unavailable",
    ):
        builder.compile_stream(runtime=runtime)


def test_real_ir2_program_combines_matmul_and_axis_reduction() -> None:
    runtime = Runtime()
    array_module.register_native(runtime)
    program = {
        "ir_version": 2,
        "inputs": [
            {"name": "input", "dtype": "float32", "shape": [None, 2]},
        ],
        "nodes": [
            {"id": "x", "op": "input", "name": "input"},
            {"id": "w", "op": "transpose", "input": "x"},
            {"id": "product", "op": "matmul", "left": "x", "right": "w"},
            {
                "id": "rows",
                "op": "reduce",
                "function": "mean",
                "input": "product",
                "axis": 1,
            },
        ],
        "outputs": [
            {"name": "output", "node": "rows", "dtype": "float64", "shape": [None]}
        ],
    }
    plan = (
        PipelineBuilder("native-matrix")
        .external("calc", "native", "array_expr", "1", program)
        .compile_batch(runtime)
    )
    values = array_module.from_arrow(
        pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float32()), shape=(2, 2)
    )
    result = plan.execute({"input": Batch.from_array(values, backend="native")})
    output = result.outputs["output"].array
    assert output.shape == (2,)
    assert output.dtype == "float64"
    assert output.to_arrow().to_pylist() == [8.0, 18.0]
    assert values.to_arrow().to_pylist() == [1.0, 2.0, 3.0, 4.0]
    assert result.datafusion_metrics == []
