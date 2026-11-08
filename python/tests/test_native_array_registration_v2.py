from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import calc_flow
import calc_flow.array as cfa
from calc_flow import Batch, CapabilityRule, Runtime, _native


def _bundle() -> list[dict[str, object]]:
    array = {
        **_native._native_array_capability(),
        "state_layouts": [],
    }
    table = {
        **array,
        "name": "table_array",
        "modes": ["batch", "stream"],
        "finality": "per_row_final",
        "microbatch_invariant": True,
        "checkpoint_support": "stateless",
        "supports_static_inputs": True,
        "partition_contract": "row_axis_independent",
    }
    sequence = {
        **array,
        "name": "array_sequence",
        "input_ports": [{"name": "input", "kind": "array", "required": True}],
        "output_ports": [{"name": "output", "kind": "array", "required": True}],
        "modes": ["stream"],
        "stateful": True,
        "microbatch_invariant": True,
        "finality": "per_row_final",
        "checkpoint_support": "checkpointed_stateful",
        "state_version": 1,
        "state_layouts": [1],
        "supports_static_inputs": False,
        "partition_contract": "single_ordered_sequence",
        "port_binding": "fixed",
        "array_rules": {
            "supported_dtypes": ["float32", "float64"],
            "safe_dtype_rule": {"name": "native_float_dtype", "version": "1"},
            "shape_rules": [{"name": "native_sequence_vector", "version": "1"}],
        },
    }
    return [array, sequence, table]


class _NativeInsertion:
    def __init__(self) -> None:
        self.calls = 0

    def _register_native_array(self) -> None:
        self.calls += 1


def test_native_registration_bundle_publishes_three_records_and_one_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "_native_array_capabilities", _bundle, raising=False)
    inner = _NativeInsertion()
    runtime = Runtime(_inner=inner)  # type: ignore[arg-type]
    runtime._symbolic_compile_cache["plan"] = object()
    runtime._symbolic_schema_cache["schema"] = object()

    cfa.register_native(runtime)

    snapshot, records = runtime.capability_registration_snapshot()
    assert snapshot.scope.revision == 3
    assert tuple(item.name for item in snapshot.providers) == (
        "array_expr",
        "array_sequence",
        "table_array",
    )
    assert snapshot.providers[2].modes == ("batch", "stream")
    assert snapshot.providers[2].checkpoint_support == "stateless"
    assert snapshot.providers[1].state_layouts == (1,)
    assert all(record["kind"] == "native_provider" for record in records)
    assert all("callback" not in record and "root" not in record for record in records)
    assert inner.calls == runtime._symbolic_cache_generation == 1
    assert runtime._symbolic_compile_cache == runtime._symbolic_schema_cache == {}


@pytest.mark.parametrize("failure", ["missing", "duplicate", "bad_second"])
def test_invalid_native_bundle_fails_before_any_factory_insertion(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    descriptors = _bundle()
    if failure == "missing":
        descriptors.pop()
    elif failure == "duplicate":
        descriptors[1] = descriptors[0]
    else:
        descriptors[1] = {**descriptors[1], "name": "wrong"}
    monkeypatch.setattr(
        _native, "_native_array_capabilities", lambda: descriptors, raising=False
    )
    inner = _NativeInsertion()
    runtime = Runtime(_inner=inner)  # type: ignore[arg-type]
    runtime._symbolic_compile_cache["plan"] = object()
    runtime._symbolic_schema_cache["schema"] = object()
    before = (
        runtime.capability_registration_snapshot(),
        runtime._symbolic_cache_generation,
        dict(runtime._symbolic_compile_cache),
        dict(runtime._symbolic_schema_cache),
    )

    with pytest.raises(ValueError):
        cfa.register_native(runtime)

    assert inner.calls == 0
    assert before == (
        runtime.capability_registration_snapshot(),
        runtime._symbolic_cache_generation,
        runtime._symbolic_compile_cache,
        runtime._symbolic_schema_cache,
    )


@pytest.mark.parametrize(
    "name",
    [
        "native_packed_rank2",
        "native_matmul",
        "native_axis_reduction",
        "native_table_static_rhs",
    ],
)
def test_m2_native_rules_are_explicit_new_versioned_identities(name: str) -> None:
    assert CapabilityRule(name, "1").name == name
    with pytest.raises(ValueError, match="unknown capability rule"):
        CapabilityRule(name, "2")


@pytest.mark.parametrize("incomplete", [False, True])
def test_documented_registration_replay_installs_the_native_bundle_once(
    monkeypatch: pytest.MonkeyPatch,
    incomplete: bool,
    tmp_path: Path,
) -> None:
    class Probe:
        def __init__(self) -> None:
            self.inner = _native.Runtime()
            self.calls = 0

        def _register_native_array(self) -> None:
            if self.calls:
                raise RuntimeError("duplicate native bundle insertion")
            self.calls += 1

        def __getattr__(self, name: str) -> object:
            return getattr(self.inner, name)

    probes: list[Probe] = []

    def runtime() -> Runtime:
        probe = Probe()
        probes.append(probe)
        return Runtime(_inner=probe)  # type: ignore[arg-type]

    monkeypatch.setattr(calc_flow, "Runtime", runtime)
    monkeypatch.setattr(_native, "_native_array_capabilities", _bundle, raising=False)
    if incomplete:
        original_snapshot = Runtime.capability_registration_snapshot

        def without_sequence(self: Runtime) -> object:
            snapshot, records = original_snapshot(self)
            return snapshot, tuple(
                record for record in records if record["name"] != "array_sequence"
            )

        monkeypatch.setattr(
            Runtime, "capability_registration_snapshot", without_sequence
        )
    document = (Path(__file__).resolve().parents[2] / "docs/python-api.md").read_text()
    section = document.split("The example registers a mapping provider", 1)[1]
    snippet = section.split("```python\n", 1)[1].split("```", 1)[0]
    script = tmp_path / "registration_replay.py"
    script.write_text(snippet, encoding="utf-8")
    if incomplete:
        with pytest.raises(ValueError, match="native capability bundle"):
            runpy.run_path(str(script))
        assert [probe.calls for probe in probes] == [1]
    else:
        runpy.run_path(str(script))
        assert [probe.calls for probe in probes] == [1, 1]


def test_real_native_bundle_descriptors_preserve_lifecycle_contracts() -> None:
    descriptors = _native._native_array_capabilities()
    assert [descriptor["name"] for descriptor in descriptors] == [
        "array_expr",
        "array_sequence",
        "table_array",
    ]
    array, sequence, table = descriptors
    assert sequence["modes"] == ["stream"]
    assert sequence["checkpoint_support"] == "checkpointed_stateful"
    assert sequence["state_layouts"] == [1]
    assert sequence["partition_contract"] == "single_ordered_sequence"
    assert sequence["supports_static_inputs"] is False
    assert array == _native._native_array_capability()
    assert array["modes"] == ["batch"]
    assert array["checkpoint_support"] == "unproven"
    assert array["supports_static_inputs"] is True
    assert table["modes"] == ["batch", "stream"]
    assert table["checkpoint_support"] == "stateless"
    assert table["partition_contract"] == "row_axis_independent"
    assert table["supports_static_inputs"] is True
    assert array["stateful"] is table["stateful"] is False

    runtime = Runtime()
    cfa.register_native(runtime)
    snapshot, records = runtime.capability_registration_snapshot()
    assert snapshot.scope.revision == len(records) == 3
    assert runtime._symbolic_cache_generation == 1
    from calc_flow import CalcFlowError, PipelineBuilder

    with pytest.raises(CalcFlowError, match=r"\[missing_provider\]"):
        PipelineBuilder("sequence-batch").external(
            "calc", "native", "array_sequence", "1", {}
        ).compile_batch(runtime)


class _CollisionProvider:
    def __call__(self, batch: Batch, _options: object) -> Batch:
        return batch

    def validate_stream(self, _options: object) -> None:
        return None


_NATIVE_REGISTRY_KEYS = (
    ("array_expr", "batch"),
    ("table_array", "batch"),
    ("table_array", "stream"),
    ("array_sequence", "stream"),
)


@pytest.mark.parametrize(
    "name,mode,absent",
    [
        (name, mode, tuple(key for key in _NATIVE_REGISTRY_KEYS if key != (name, mode)))
        for name, mode in _NATIVE_REGISTRY_KEYS
    ],
)
def test_real_native_bundle_collision_preserves_all_factories_and_python_state(
    name: str, mode: str, absent: tuple[tuple[str, str], ...]
) -> None:
    from calc_flow import CalcFlowError, PipelineBuilder

    runtime = Runtime()
    collision = _CollisionProvider()
    if mode == "batch":
        runtime.register_provider("native", name, "1", collision)
    else:
        runtime._inner._register_stateless_stream_provider(
            "native",
            name,
            "1",
            collision,
            microbatch_invariant=True,
            deterministic=True,
            replay_safe=True,
        )
    runtime._symbolic_compile_cache["plan"] = object()
    runtime._symbolic_schema_cache["schema"] = object()
    before = (
        runtime.capability_registration_snapshot(),
        runtime._symbolic_cache_generation,
        dict(runtime._symbolic_compile_cache),
        dict(runtime._symbolic_schema_cache),
    )

    with pytest.raises(CalcFlowError, match="(already|duplicate)"):
        cfa.register_native(runtime)

    assert before == (
        runtime.capability_registration_snapshot(),
        runtime._symbolic_cache_generation,
        runtime._symbolic_compile_cache,
        runtime._symbolic_schema_cache,
    )
    existing = PipelineBuilder("preserved-native").external(
        "calc", "native", name, "1", {}
    )
    if mode == "batch":
        existing.compile_batch(runtime)
    else:
        existing.compile_stream(runtime=runtime)
    for absent_name, absent_mode in absent:
        builder = PipelineBuilder("atomic-native").external(
            "calc", "native", absent_name, "1", {}
        )
        with pytest.raises(CalcFlowError, match=r"\[missing_provider\]"):
            if absent_mode == "batch":
                builder.compile_batch(runtime)
            else:
                builder.compile_stream(runtime=runtime)
