from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from calc_flow import ProviderCapability, ProviderOptionsSchema, ProviderPort, Runtime

_SCHEMA = '{"additionalProperties":false,"properties":{},"type":"object"}'


def _arguments() -> dict[str, object]:
    return {
        "provider": "example",
        "name": "sequence",
        "version": "1",
        "input_ports": (ProviderPort("input", "array", True),),
        "output_ports": (ProviderPort("output", "array", True),),
        "options_schema": None,
        "modes": ("stream",),
        "finality": "per_row_final",
        "stateful": True,
        "microbatch_invariant": True,
        "requires_watermark": False,
        "checkpoint_support": "checkpointed_stateful",
        "state_version": 1,
        "deterministic": True,
        "replay_safe": True,
        "supports_static_inputs": False,
        "partition_contract": "none",
        "array_rules": None,
    }


def test_provider_layout_inventory_is_frozen_and_schema_five_is_explicit() -> None:
    capability = ProviderCapability(**_arguments(), state_layouts=(2, 3))
    assert capability.state_layouts == (2, 3)
    assert Runtime().capabilities().schema_version == 5
    with pytest.raises(FrozenInstanceError):
        capability.state_layouts = (2,)  # type: ignore[misc]


def test_fixed_ports_may_pair_with_strict_json_schema_without_field_schema() -> None:
    capability = ProviderCapability(
        **_arguments(), state_layouts=(1,), options_json_schema=_SCHEMA
    )
    assert capability.port_binding == "fixed"
    assert capability.input_ports == (ProviderPort("input", "array", True),)
    assert capability.options_json_schema == _SCHEMA
    with pytest.raises(ValueError, match="exclusive"):
        ProviderCapability(
            **{**_arguments(), "options_schema": ProviderOptionsSchema()},
            state_layouts=(1,),
            options_json_schema=_SCHEMA,
        )


@pytest.mark.parametrize(
    "layouts,error",
    [
        ([1], TypeError),
        ((True,), TypeError),
        ((0,), ValueError),
        ((2, 1), ValueError),
        ((1, 1), ValueError),
        ((), ValueError),
    ],
)
def test_provider_layout_inventory_uses_existing_strict_contract(
    layouts: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match="(state.layout|layout)"):
        ProviderCapability(**_arguments(), state_layouts=layouts)


def test_legacy_provider_defaults_keep_empty_layouts() -> None:
    arguments = {
        **_arguments(),
        "checkpoint_support": "stateless",
        "stateful": False,
        "state_version": None,
    }
    assert ProviderCapability(**arguments).state_layouts == ()
    with pytest.raises(ValueError, match="empty unless checkpointed_stateful"):
        ProviderCapability(**arguments, state_layouts=(1,))
    runtime = Runtime()
    runtime.register_provider("example", "plain", "1", lambda batch, _: batch)
    plain = runtime.capabilities().providers[0]
    assert plain.state_layouts == ()
    assert plain.modes == ("batch",)
    assert plain.checkpoint_support == "stateless"
    assert plain.partition_contract == "none"


def test_sequence_partition_contract_is_explicit() -> None:
    capability = ProviderCapability(
        **{**_arguments(), "partition_contract": "single_ordered_sequence"},
        state_layouts=(1,),
    )
    assert capability.partition_contract == "single_ordered_sequence"
    with pytest.raises(ValueError, match="partition_contract"):
        ProviderCapability(
            **{**_arguments(), "partition_contract": "ordered_partitions"},
            state_layouts=(1,),
        )


@pytest.mark.parametrize("name", ["native_float_dtype", "native_sequence_vector"])
def test_sequence_rule_identities_are_closed_and_versioned(name: str) -> None:
    from calc_flow import CapabilityRule

    assert CapabilityRule(name, "1").name == name
    with pytest.raises(ValueError, match="unknown capability rule"):
        CapabilityRule(name, "2")


def _sequence_descriptor() -> dict[str, object]:
    return {
        **_arguments(),
        "provider": "native",
        "name": "array_sequence",
        "input_ports": [{"name": "input", "kind": "array", "required": True}],
        "output_ports": [{"name": "output", "kind": "array", "required": True}],
        "modes": ["stream"],
        "state_layouts": [1],
        "partition_contract": "single_ordered_sequence",
        "port_binding": "fixed",
        "options_json_schema": _SCHEMA,
        "array_rules": {
            "supported_dtypes": ["float32", "float64"],
            "safe_dtype_rule": {"name": "native_float_dtype", "version": "1"},
            "shape_rules": [{"name": "native_sequence_vector", "version": "1"}],
        },
    }


def test_sequence_descriptor_parses_fixed_json_ports_and_state_layouts() -> None:
    from calc_flow.capabilities import _native_array_provider_capability

    descriptor = _sequence_descriptor()
    capability = _native_array_provider_capability(descriptor)
    assert capability.name == "array_sequence"
    assert capability.input_ports == (ProviderPort("input", "array", True),)
    assert capability.output_ports == (ProviderPort("output", "array", True),)
    assert capability.state_layouts == (1,)
    assert capability.checkpoint_support == "checkpointed_stateful"
    layouts = descriptor["state_layouts"]
    assert isinstance(layouts, list)
    layouts.append(2)
    assert capability.state_layouts == (1,)


def _bundle_descriptors() -> list[dict[str, object]]:
    array = {
        **_sequence_descriptor(),
        "name": "array_expr",
        "input_ports": [],
        "output_ports": [],
        "modes": ["batch"],
        "finality": "unproven",
        "stateful": False,
        "microbatch_invariant": False,
        "checkpoint_support": "unproven",
        "state_version": None,
        "state_layouts": [],
        "supports_static_inputs": True,
        "partition_contract": "none",
        "port_binding": "options",
        "array_rules": {
            "supported_dtypes": ["bool", "float32", "float64"],
            "safe_dtype_rule": {"name": "native_float_bool_dtype", "version": "1"},
            "shape_rules": [{"name": "native_scalar_vector_broadcast", "version": "1"}],
        },
    }
    table = {
        **array,
        "name": "table_array",
        "modes": ["batch", "stream"],
        "finality": "per_row_final",
        "microbatch_invariant": True,
        "checkpoint_support": "stateless",
        "partition_contract": "row_axis_independent",
    }
    return [array, _sequence_descriptor(), table]


def test_native_bundle_accepts_three_canonical_frozen_records() -> None:
    from calc_flow.capabilities import _native_array_provider_capabilities

    bundle = _native_array_provider_capabilities(list(reversed(_bundle_descriptors())))
    assert [capability.name for capability in bundle] == [
        "array_expr",
        "array_sequence",
        "table_array",
    ]
    assert [capability.state_layouts for capability in bundle] == [(), (1,), ()]


@pytest.mark.parametrize(
    "overrides",
    [
        {"modes": ["batch", "stream"]},
        {"state_layouts": [2]},
        {"partition_contract": "none"},
        {"checkpoint_support": "unproven", "state_version": None, "state_layouts": []},
        {"supports_static_inputs": True},
        {"input_ports": [{"name": "input", "kind": "array", "required": False}]},
        {"options_json_schema": None},
    ],
)
def test_native_bundle_rejects_contradictory_sequence_evidence(
    overrides: dict[str, object],
) -> None:
    from calc_flow.capabilities import _native_array_provider_capabilities

    bundle = _bundle_descriptors()
    bundle[1] = {**bundle[1], **overrides}
    with pytest.raises(ValueError, match="array_sequence.*capability"):
        _native_array_provider_capabilities(bundle)


def _records() -> list[dict[str, object]]:
    from calc_flow.capabilities import _native_array_provider_capabilities

    return [
        {
            "kind": "native_provider",
            "provider": capability.provider,
            "name": capability.name,
            "version": capability.version,
            "capability": capability,
        }
        for capability in _native_array_provider_capabilities(_bundle_descriptors())
    ]


@pytest.mark.parametrize("failure", ["incomplete", "identity", "mutable_capability"])
def test_snapshot_rejects_incomplete_or_contradictory_native_record_evidence(
    failure: str,
) -> None:
    from calc_flow.capabilities import runtime_capabilities

    records = _records()
    error: type[Exception] = ValueError
    if failure == "incomplete":
        records.pop()
    elif failure == "identity":
        records[1] = {**records[1], "name": "different"}
    else:
        records[1] = {**records[1], "capability": _sequence_descriptor()}
        error = TypeError
    with pytest.raises(error, match="native.*(bundle|record|capability)"):
        runtime_capabilities(
            session_id="unit-session",
            revision=len(records),
            package_version="test",
            registrations=records,
        )


@pytest.mark.parametrize(
    "failure,error",
    [
        ("missing_layouts", ValueError),
        ("layout_tuple", TypeError),
        ("port_tuple", TypeError),
        ("integer_required", TypeError),
        ("unknown_port_key", ValueError),
    ],
)
def test_native_descriptor_requires_schema_five_strict_json_fields(
    failure: str, error: type[Exception]
) -> None:
    from calc_flow.capabilities import _native_array_provider_capability

    descriptor = _sequence_descriptor()
    if failure == "missing_layouts":
        descriptor.pop("state_layouts")
    elif failure == "layout_tuple":
        descriptor["state_layouts"] = (1,)
    elif failure == "port_tuple":
        descriptor["input_ports"] = (
            {"name": "input", "kind": "array", "required": True},
        )
    elif failure == "integer_required":
        descriptor["input_ports"] = [{"name": "input", "kind": "array", "required": 1}]
    else:
        descriptor["input_ports"] = [
            {"name": "input", "kind": "array", "required": True, "schema": []}
        ]
    with pytest.raises(error):
        _native_array_provider_capability(descriptor)


def test_schema_five_json_snapshot_contains_provider_layouts() -> None:
    from calc_flow.capabilities import runtime_capabilities

    snapshot = runtime_capabilities(
        session_id="unit-session",
        revision=3,
        package_version="test",
        registrations=_records(),
    )
    decoded = json.loads(json.dumps(asdict(snapshot), sort_keys=True, allow_nan=False))
    assert decoded["schema_version"] == 5
    assert [provider["state_layouts"] for provider in decoded["providers"]] == [
        [],
        [1],
        [],
    ]
    assert decoded["providers"][1]["input_ports"] == [
        {"name": "input", "kind": "array", "required": True}
    ]


@pytest.mark.parametrize(
    "index,overrides",
    [
        (0, {"modes": ["stream"]}),
        (0, {"stateful": True}),
        (2, {"partition_contract": "none"}),
    ],
)
def test_native_bundle_retains_batch_and_stateless_table_contracts(
    index: int,
    overrides: dict[str, object],
) -> None:
    from calc_flow.capabilities import _native_array_provider_capabilities

    bundle = _bundle_descriptors()
    bundle[index] = {**bundle[index], **overrides}
    with pytest.raises(ValueError, match=f"native {bundle[index]['name']} capability"):
        _native_array_provider_capabilities(bundle)
