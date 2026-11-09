from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from calc_flow import CapabilityRule
from calc_flow.capabilities import _native_array_provider_capability

_FIXTURES = Path(__file__).with_name("fixtures") / "native_array_capabilities"


def _sequence_descriptor(version: int) -> dict[str, object]:
    if version == 2:
        return json.loads((_FIXTURES / "rust_sequence_ir2.json").read_text())
    bundle = json.loads((_FIXTURES / "rust_sequence_bundle_ir1.json").read_text())
    return next(
        descriptor for descriptor in bundle if descriptor["name"] == "array_sequence"
    )


def test_actual_rust_ir2_sequence_descriptor_parses_without_rewriting_evidence() -> (
    None
):
    descriptor = _sequence_descriptor(2)
    capability = _native_array_provider_capability(descriptor)
    assert capability.state_layouts == (1, 2)
    assert capability.options_json_schema == descriptor["options_json_schema"]
    assert descriptor["array_rules"]["shape_rules"] == [
        {"name": "native_sequence_vector", "version": "1"},
        {"name": "native_sequence_indicators", "version": "2"},
    ]
    assert capability.array_rules.shape_rules == (
        CapabilityRule("native_sequence_indicators", "2"),
        CapabilityRule("native_sequence_vector", "1"),
    )


def test_actual_legacy_rust_ir1_sequence_evidence_stays_unchanged() -> None:
    descriptor = _sequence_descriptor(1)
    capability = _native_array_provider_capability(descriptor)
    assert capability.state_layouts == (1,)
    assert capability.array_rules.shape_rules == (
        CapabilityRule("native_sequence_vector", "1"),
    )
    assert capability.options_json_schema == descriptor["options_json_schema"]


def test_native_sequence_indicator_rule_is_version_two_only() -> None:
    assert CapabilityRule("native_sequence_indicators", "2").version == "2"
    for version in ("1", "3"):
        with pytest.raises(ValueError, match="unknown capability rule"):
            CapabilityRule("native_sequence_indicators", version)


@pytest.mark.parametrize(
    "layouts,rules,schema",
    [(1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1), (2, 1, 2), (2, 2, 1)],
)
def test_native_sequence_profiles_reject_crossed_actual_producer_evidence(
    layouts: int, rules: int, schema: int
) -> None:
    descriptor = {
        **_sequence_descriptor(2),
        "state_layouts": _sequence_descriptor(layouts)["state_layouts"],
        "array_rules": _sequence_descriptor(rules)["array_rules"],
        "options_json_schema": _sequence_descriptor(schema)["options_json_schema"],
    }
    with pytest.raises(ValueError, match="(capability|schema)"):
        _native_array_provider_capability(descriptor)


@pytest.mark.parametrize("layouts", [[2], [1, 3], [1, 2, 3]])
def test_native_sequence_profiles_reject_unimplemented_layout_inventory(
    layouts: list[int],
) -> None:
    descriptor = {**_sequence_descriptor(2), "state_layouts": layouts}
    with pytest.raises(ValueError, match="state_layouts"):
        _native_array_provider_capability(descriptor)


@pytest.mark.parametrize(
    "path,value",
    [
        (("oneOf", 0, "additionalProperties"), True),
        (("oneOf", 0, "properties", "ir_version", "const"), 3),
        (("oneOf", 0, "properties", "ir_version", "const"), True),
        (("properties", "ir_version", "enum"), [True, 2]),
        (("properties", "ir_version", "enum"), [1, 3]),
        (("oneOf", 1, "properties", "kernel", "oneOf", 2, "required"), ["function"]),
        (
            (
                "oneOf",
                1,
                "properties",
                "kernel",
                "oneOf",
                3,
                "properties",
                "function",
                "const",
            ),
            "macd_components",
        ),
        (
            ("oneOf", 0, "properties", "kernel", "oneOf", 0, "additionalProperties"),
            True,
        ),
        (("properties", "kernel", "oneOf", 2, "properties", "period", "minimum"), 0),
        (("properties", "kernel", "oneOf", 2, "properties", "period", "maximum"), True),
        (("properties", "execution", "additionalProperties"), True),
        (("oneOf", 1, "properties", "state", "additionalProperties"), True),
        (("oneOf", 1, "required"), ["ir_version", "kernel"]),
    ],
)
def test_native_sequence_schema_profiles_reject_open_or_contradictory_alternatives(
    path: tuple[str | int, ...], value: object
) -> None:
    descriptor = _sequence_descriptor(2)
    schema = json.loads(descriptor["options_json_schema"])
    cursor = schema
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    descriptor["options_json_schema"] = json.dumps(
        schema, sort_keys=True, separators=(",", ":")
    )
    with pytest.raises(ValueError, match="(capability|schema)"):
        _native_array_provider_capability(descriptor)


@pytest.mark.parametrize("missing", ["oneOf", "kernel"])
def test_native_sequence_schema_profiles_require_all_version_and_kernel_alternatives(
    missing: str,
) -> None:
    descriptor = _sequence_descriptor(2)
    schema = json.loads(descriptor["options_json_schema"])
    if missing == "oneOf":
        schema.pop("oneOf")
    else:
        schema["oneOf"][1]["properties"]["kernel"]["oneOf"].pop()
    descriptor["options_json_schema"] = json.dumps(
        schema, sort_keys=True, separators=(",", ":")
    )
    with pytest.raises(ValueError, match="(capability|schema)"):
        _native_array_provider_capability(descriptor)


def _captured_bundle(version: int) -> list[dict[str, object]]:
    bundle = json.loads((_FIXTURES / "rust_sequence_bundle_ir1.json").read_text())
    return [
        _sequence_descriptor(version) if value["name"] == "array_sequence" else value
        for value in bundle
    ]


def _records(version: int) -> list[dict[str, object]]:
    from calc_flow.capabilities import _native_array_provider_capabilities

    return [
        {
            "kind": "native_provider",
            "provider": capability.provider,
            "name": capability.name,
            "version": capability.version,
            "capability": capability,
        }
        for capability in _native_array_provider_capabilities(_captured_bundle(version))
    ]


@pytest.mark.parametrize("version", [1, 2])
def test_actual_native_profiles_keep_three_frozen_snapshot_records(
    version: int,
) -> None:
    from calc_flow.capabilities import runtime_capabilities

    records = _records(version)
    original = records[1]["capability"]
    snapshot = runtime_capabilities(
        session_id="source-evidence",
        revision=3,
        package_version="test",
        registrations=records,
    )
    assert snapshot.schema_version == 5
    assert snapshot.scope.revision == len(records) == 3
    assert tuple(value.name for value in snapshot.providers) == (
        "array_expr",
        "array_sequence",
        "table_array",
    )
    assert snapshot.providers[1] is original
    assert original.state_layouts == ((1,) if version == 1 else (1, 2))
    assert (
        original.options_json_schema
        == _sequence_descriptor(version)["options_json_schema"]
    )


def test_snapshot_revalidates_mismatched_frozen_native_profile_evidence() -> None:
    from calc_flow.capabilities import runtime_capabilities

    records = _records(2)
    records[1] = {
        **records[1],
        "capability": replace(records[1]["capability"], state_layouts=(1,)),
    }
    with pytest.raises(ValueError, match="native array_sequence capability"):
        runtime_capabilities(
            session_id="source-evidence",
            revision=3,
            package_version="test",
            registrations=records,
        )


def test_real_native_bundle_keeps_one_generation_and_replays_once() -> None:
    import calc_flow as cf
    import calc_flow.array as cfa
    from calc_flow import _native

    actual = next(
        value
        for value in _native._native_array_capabilities()
        if value["name"] == "array_sequence"
    )
    parent, worker = cf.Runtime(), cf.Runtime()
    cfa.register_native(parent)
    snapshot, records = parent.capability_registration_snapshot()
    original = records[1]["capability"]
    cfa.register_native(worker)
    assert snapshot.providers == worker.capabilities().providers
    assert snapshot.scope.revision == len(records) == 3
    assert parent._symbolic_cache_generation == worker._symbolic_cache_generation == 1
    assert original.state_layouts == tuple(actual["state_layouts"])
    assert parent.registration_snapshot()[1]["capability"] is original


def test_real_documented_replay_uses_the_same_actual_native_profile(
    tmp_path: Path,
) -> None:
    document = (Path(__file__).resolve().parents[2] / "docs/python-api.md").read_text()
    section = document.split("The example registers a mapping provider", 1)[1]
    snippet = section.split("```python\n", 1)[1].split("```", 1)[0]
    script = tmp_path / "native_replay.py"
    script.write_text(snippet, encoding="utf-8")
    namespace = runpy.run_path(str(script))
    parent, worker = namespace["parent"], namespace["worker"]
    assert (
        len(parent.registration_snapshot()) == len(worker.registration_snapshot()) == 4
    )
    assert parent.capabilities().providers == worker.capabilities().providers
