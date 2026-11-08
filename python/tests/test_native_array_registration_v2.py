from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import calc_flow
import calc_flow.array as cfa
from calc_flow import Batch, CapabilityRule, Runtime, _native


def _bundle() -> list[dict[str, object]]:
    array = _native._native_array_capability()
    table = {
        **_native._native_array_capability(),
        "name": "table_array",
        "modes": ["batch", "stream"],
        "finality": "per_row_final",
        "microbatch_invariant": True,
        "checkpoint_support": "stateless",
        "supports_static_inputs": True,
        "partition_contract": "row_axis_independent",
    }
    return [array, table]


class _NativeInsertion:
    def __init__(self) -> None:
        self.calls = 0

    def _register_native_array(self) -> None:
        self.calls += 1


def test_native_registration_bundle_publishes_two_records_and_one_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "_native_array_capabilities", _bundle, raising=False)
    inner = _NativeInsertion()
    runtime = Runtime(_inner=inner)  # type: ignore[arg-type]
    runtime._symbolic_compile_cache["plan"] = object()
    runtime._symbolic_schema_cache["schema"] = object()

    cfa.register_native(runtime)

    snapshot, records = runtime.capability_registration_snapshot()
    assert snapshot.scope.revision == 2
    assert tuple(item.name for item in snapshot.providers) == (
        "array_expr",
        "table_array",
    )
    assert snapshot.providers[1].modes == ("batch", "stream")
    assert snapshot.providers[1].checkpoint_support == "stateless"
    assert all(record["kind"] == "native_provider" for record in records)
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


def test_documented_registration_replay_installs_the_native_bundle_once(
    monkeypatch: pytest.MonkeyPatch,
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
    document = (Path(__file__).resolve().parents[2] / "docs/python-api.md").read_text()
    section = document.split("The example registers a mapping provider", 1)[1]
    snippet = section.split("```python\n", 1)[1].split("```", 1)[0]
    script = tmp_path / "registration_replay.py"
    script.write_text(snippet, encoding="utf-8")
    runpy.run_path(str(script))
    assert [probe.calls for probe in probes] == [1, 1]


def test_real_native_bundle_descriptors_preserve_lifecycle_contracts() -> None:
    descriptors = _native._native_array_capabilities()
    assert [descriptor["name"] for descriptor in descriptors] == [
        "array_expr",
        "table_array",
    ]
    array, table = descriptors
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
    assert snapshot.scope.revision == len(records) == 2
    assert runtime._symbolic_cache_generation == 1


class _CollisionProvider:
    def __call__(self, batch: Batch, _options: object) -> Batch:
        return batch

    def validate_stream(self, _options: object) -> None:
        return None


@pytest.mark.parametrize(
    "name,mode,absent",
    [
        ("array_expr", "batch", (("table_array", "batch"), ("table_array", "stream"))),
        ("table_array", "batch", (("array_expr", "batch"), ("table_array", "stream"))),
        ("table_array", "stream", (("array_expr", "batch"), ("table_array", "batch"))),
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
