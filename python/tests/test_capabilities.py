from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import calc_flow
from calc_flow import (
    CalcFlowError,
    OperatorCapability,
    ProviderCapability,
    ProviderOption,
    ProviderOptionsSchema,
    ProviderPort,
    Runtime,
    RuntimeCapabilities,
    UdfCapability,
    register_jax,
    register_numpy,
)
from calc_flow.capabilities import runtime_capabilities


def _secret_rejected_callable() -> None:
    raise AssertionError("the rejected callable must not be invoked")


class _RejectedValueWithHostileFormatting:
    def __str__(self) -> str:
        raise AssertionError("the rejected value must not be stringified")

    def __repr__(self) -> str:
        raise AssertionError("the rejected value must not be represented")


def test_empty_runtime_capabilities_are_frozen_and_session_scoped() -> None:
    runtime = Runtime()

    snapshot = runtime.capabilities()
    repeated = runtime.capabilities()

    assert isinstance(snapshot, RuntimeCapabilities)
    assert snapshot.schema_version == 2
    assert snapshot.scope.kind == "runtime_session"
    assert snapshot.scope.revision == 0
    assert snapshot.scope.session_id == repeated.scope.session_id
    assert snapshot.package_version == "4.0.0"
    assert snapshot.project_format_versions == (3,)
    assert snapshot.batch_kinds == ("array", "table")
    assert snapshot.operators == (
        OperatorCapability(
            kind="cross_section",
            version="1",
            input_ports=(ProviderPort("input", "table", required=True),),
            output_ports=(ProviderPort("output", "table", required=True),),
            modes=("batch", "stream"),
            finality="group_final_append_only",
            requires_datafusion=False,
            stateful=True,
            microbatch_invariant=True,
            requires_watermark=True,
            checkpoint_support="checkpointed_stateful",
            state_version=1,
            deterministic=True,
            replay_safe=True,
        ),
        OperatorCapability(
            kind="expression",
            version="1",
            input_ports=(ProviderPort("input", "table", required=True),),
            output_ports=(ProviderPort("output", "table", required=True),),
            modes=("batch", "stream"),
            finality="per_row_final",
            requires_datafusion=True,
            stateful=False,
            microbatch_invariant=True,
            requires_watermark=False,
            checkpoint_support="stateless",
            state_version=None,
            deterministic=True,
            replay_safe=True,
        ),
        OperatorCapability(
            kind="rolling",
            version="1",
            input_ports=(ProviderPort("input", "table", required=True),),
            output_ports=(ProviderPort("output", "table", required=True),),
            modes=("batch", "stream"),
            finality="per_row_final",
            requires_datafusion=False,
            stateful=True,
            microbatch_invariant=True,
            requires_watermark=True,
            checkpoint_support="checkpointed_stateful",
            state_version=1,
            deterministic=True,
            replay_safe=True,
        ),
        OperatorCapability(
            kind="sql",
            version="1",
            input_ports=(ProviderPort("input", "table", required=True),),
            output_ports=(ProviderPort("output", "table", required=True),),
            modes=("batch", "stream"),
            finality="unproven",
            requires_datafusion=True,
            stateful=False,
            microbatch_invariant=False,
            requires_watermark=False,
            checkpoint_support="stateless",
            state_version=None,
            deterministic=True,
            replay_safe=True,
        ),
        OperatorCapability(
            kind="stream_join",
            version="1",
            input_ports=(
                ProviderPort("left", "table", required=True),
                ProviderPort("right", "table", required=True),
            ),
            output_ports=(ProviderPort("output", "table", required=True),),
            modes=("stream",),
            finality="unproven",
            requires_datafusion=True,
            stateful=True,
            microbatch_invariant=False,
            requires_watermark=True,
            checkpoint_support="checkpointed_stateful",
            state_version=1,
            deterministic=True,
            replay_safe=True,
        ),
    )
    assert snapshot.udfs == ()
    assert snapshot.providers == ()
    assert tuple(connector.name for connector in snapshot.connectors) == ("file",)
    with pytest.raises(FrozenInstanceError):
        snapshot.scope.revision = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.operators[0].finality = "unproven"  # type: ignore[misc]


def test_successful_provider_registration_advances_revision_once() -> None:
    runtime = Runtime()
    schema = ProviderOptionsSchema(
        fields=(ProviderOption("expression", "string", required=True),)
    )

    def callback(batch: object, _options: object) -> object:
        return batch

    runtime.register_provider(
        "numpy",
        "expression",
        "1",
        callback,
        options_schema=schema,
    )

    assert runtime.capabilities().scope.revision == 1
    assert runtime.capabilities().providers == (
        ProviderCapability(
            provider="numpy",
            name="expression",
            version="1",
            input_ports=(ProviderPort("input", "array", required=True),),
            output_ports=(ProviderPort("output", "array", required=True),),
            options_schema=schema,
            modes=("batch",),
            finality="unproven",
            stateful=False,
            microbatch_invariant=False,
            requires_watermark=False,
            checkpoint_support="stateless",
            state_version=None,
            deterministic=False,
            replay_safe=False,
            supports_static_inputs=False,
            partition_contract="none",
            array_rules=None,
        ),
    )
    with pytest.raises(CalcFlowError):
        runtime.register_provider(
            "numpy",
            "expression",
            "1",
            callback,
            options_schema=schema,
        )
    assert runtime.capabilities().scope.revision == 1


def test_provider_options_schema_rejects_non_data_and_unsupported_shapes() -> None:
    runtime = Runtime()

    with pytest.raises(
        TypeError,
        match="options_schema must be a ProviderOptionsSchema or None; found dict",
    ):
        runtime.register_provider(
            "test",
            "identity",
            "1",
            lambda batch, _options: batch,
            options_schema={},  # type: ignore[arg-type]
        )
    assert runtime.capabilities().scope.revision == 0

    with pytest.raises(ValueError, match="must be string, integer, number, or boolean"):
        ProviderOption("nested", "object")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate field 'expression'"):
        ProviderOptionsSchema(
            fields=(
                ProviderOption("expression", "string"),
                ProviderOption("expression", "boolean"),
            )
        )


@pytest.mark.parametrize(
    ("name", "type_name", "secret"),
    [
        pytest.param(
            _secret_rejected_callable,
            "function",
            "_secret_rejected_callable",
            id="callable",
        ),
        pytest.param(
            Path("/secret/provider-option-name"),
            type(Path()).__name__,
            "/secret/provider-option-name",
            id="path",
        ),
        pytest.param(
            {"outer": ["secret-name-value"]},
            "dict",
            "secret-name-value",
            id="container",
        ),
        pytest.param(
            _RejectedValueWithHostileFormatting(),
            "_RejectedValueWithHostileFormatting",
            None,
            id="hostile-formatting-hooks",
        ),
    ],
)
def test_later_provider_option_rejects_non_string_name_without_position_or_value(
    name: object,
    type_name: str,
    secret: str | None,
) -> None:
    runtime = Runtime()
    revision = runtime.capabilities().scope.revision

    with pytest.raises(TypeError) as raised:
        ProviderOptionsSchema(
            fields=(
                ProviderOption("first", "string"),
                ProviderOption(name, "string"),  # type: ignore[arg-type]
            )
        )

    message = str(raised.value)
    assert message == (
        "provider options_schema field name must contain strict data; "
        f"found {type_name}"
    )
    assert "fields[" not in message
    if secret is not None:
        assert secret not in message
    assert runtime.capabilities().scope.revision == revision


@pytest.mark.parametrize(
    ("value_type", "type_name", "secret"),
    [
        pytest.param(
            _secret_rejected_callable,
            "function",
            "_secret_rejected_callable",
            id="callable",
        ),
        pytest.param(
            Path("/secret/provider-option-value-type"),
            type(Path()).__name__,
            "/secret/provider-option-value-type",
            id="path",
        ),
        pytest.param(
            {"outer": ["secret-value-type"]},
            "dict",
            "secret-value-type",
            id="container",
        ),
        pytest.param(
            _RejectedValueWithHostileFormatting(),
            "_RejectedValueWithHostileFormatting",
            None,
            id="hostile-formatting-hooks",
        ),
    ],
)
def test_later_provider_option_rejects_non_string_value_type_without_position_or_value(
    value_type: object,
    type_name: str,
    secret: str | None,
) -> None:
    runtime = Runtime()
    revision = runtime.capabilities().scope.revision

    with pytest.raises(TypeError) as raised:
        ProviderOptionsSchema(
            fields=(
                ProviderOption("first", "string"),
                ProviderOption("second", value_type),  # type: ignore[arg-type]
            )
        )

    message = str(raised.value)
    assert message == (
        "provider options_schema field 'second'.value_type must contain strict "
        f"data; found {type_name}"
    )
    assert "fields[" not in message
    if secret is not None:
        assert secret not in message
    assert runtime.capabilities().scope.revision == revision


def test_provider_option_validates_name_before_value_type() -> None:
    with pytest.raises(TypeError) as raised:
        ProviderOption([], {})  # type: ignore[arg-type]

    assert str(raised.value) == (
        "provider options_schema field name must contain strict data; found list"
    )


def test_provider_option_validates_value_type_before_required() -> None:
    with pytest.raises(TypeError) as raised:
        ProviderOption("ordered", [], required="yes")  # type: ignore[arg-type]

    assert str(raised.value) == (
        "provider options_schema field 'ordered'.value_type must contain strict "
        "data; found list"
    )


def test_provider_option_preserves_strict_required_validation() -> None:
    with pytest.raises(
        TypeError,
        match="provider options_schema at 'required'\\.required must contain "
        "strict data; found int",
    ):
        ProviderOption("required", "string", required=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param([ProviderOption("field", "string")], id="list"),
        pytest.param((object(),), id="non-provider-option"),
    ],
)
def test_provider_options_schema_preserves_strict_field_container_validation(
    fields: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="fields must be a tuple of ProviderOption values",
    ):
        ProviderOptionsSchema(fields=fields)  # type: ignore[arg-type]


def test_provider_options_schema_preserves_sorted_fields() -> None:
    schema = ProviderOptionsSchema(
        fields=(
            ProviderOption("zebra", "boolean"),
            ProviderOption("alpha", "string"),
        )
    )

    assert tuple(field.name for field in schema.fields) == ("alpha", "zebra")


@pytest.mark.parametrize(
    ("backend", "register"),
    [("numpy", register_numpy), ("jax", register_jax)],
)
def test_array_helpers_expose_expression_and_mapped_provider_contracts(
    backend: str,
    register: Callable[[Runtime], None],
) -> None:
    runtime = Runtime()

    register(runtime)

    snapshot = runtime.capabilities()
    assert snapshot.scope.revision == 2
    identities = tuple(
        (item.provider, item.name, item.version) for item in snapshot.providers
    )
    assert identities == (
        (backend, "expression", "1"),
        (backend, "table_matmul", "1"),
    )
    assert snapshot.providers[0].input_ports == (
        ProviderPort("input", "array", required=True),
    )
    assert snapshot.providers[0].options_schema == ProviderOptionsSchema(
        fields=(ProviderOption("expression", "string", required=True),)
    )
    assert snapshot.providers[1].input_ports == (
        ProviderPort("table", "table", required=True),
        ProviderPort("weights", "array", required=True),
    )
    assert snapshot.providers[1].output_ports == (
        ProviderPort("output", "array", required=True),
    )
    assert snapshot.providers[1].options_schema is None
    assert all(
        provider.modes == ("batch",)
        and provider.finality == "unproven"
        and provider.stateful is False
        and provider.microbatch_invariant is False
        and provider.requires_watermark is False
        and provider.checkpoint_support == "stateless"
        and provider.state_version is None
        and provider.deterministic is False
        and provider.replay_safe is False
        and provider.supports_static_inputs is False
        and provider.partition_contract == "none"
        and provider.array_rules is None
        for provider in snapshot.providers
    )


def test_compound_registration_exposes_a_real_partial_success() -> None:
    runtime = Runtime()
    runtime._register_mapping_provider(
        "numpy",
        "table_matmul",
        "1",
        lambda _inputs, _options: {},
        input_ports=(("table", "table"), ("weights", "array")),
        output_ports=(("output", "array"),),
    )

    with pytest.raises(CalcFlowError):
        register_numpy(runtime)

    snapshot = runtime.capabilities()
    assert snapshot.scope.revision == 2
    assert tuple(item.name for item in snapshot.providers) == (
        "expression",
        "table_matmul",
    )


def test_capability_values_are_top_level_public_exports() -> None:
    expected = {
        "CapabilityRule",
        "OperatorCapability",
        "ProviderArrayRules",
        "ProviderCapability",
        "ProviderOption",
        "ProviderOptionsSchema",
        "ProviderPort",
        "RuntimeCapabilities",
        "RuntimeSessionScope",
        "UdfCapability",
    }

    assert expected <= set(calc_flow.__all__)
    assert all(
        getattr(calc_flow, name).__module__ == "calc_flow.capabilities"
        for name in expected
    )


def test_scalar_udf_capability_uses_the_successful_registration_snapshot() -> None:
    runtime = Runtime()
    before = runtime.capabilities()

    runtime.register_scalar_udf(
        provider="python",
        name="identity",
        version="1",
        input_types=("int64",),
        return_type="int64",
        volatility="immutable",
        function=lambda value: value,
    )

    after = runtime.capabilities()
    assert before.scope.revision == 0
    assert before.udfs == ()
    assert after.scope.session_id == before.scope.session_id
    assert after.scope.revision == 1
    assert after.udfs == (
        UdfCapability(
            provider="python",
            name="identity",
            version="1",
            kind="data_fusion_scalar",
            input_types=("int64",),
            return_type="int64",
            volatility="immutable",
        ),
    )


def test_capability_snapshot_does_not_reflect_on_hostile_callbacks() -> None:
    class HostileCallback:
        def __call__(self, batch: object, _options: object) -> object:
            return batch

        def __getattribute__(self, name: str) -> object:
            if name == "__call__":
                return object.__getattribute__(self, name)
            raise AssertionError(
                f"capability discovery inspected callback attribute {name}"
            )

    runtime = Runtime()
    runtime.register_provider("test", "hostile", "1", HostileCallback())

    provider = runtime.capabilities().providers[0]

    assert (provider.provider, provider.name, provider.version) == (
        "test",
        "hostile",
        "1",
    )


def test_registration_records_cannot_upgrade_lifecycle_by_smuggling() -> None:
    runtime = Runtime()
    runtime.register_provider("numpy", "smuggled", "1", lambda batch, _: batch)

    forged = runtime_capabilities(
        session_id="session",
        revision=9,
        package_version="4.0.0",
        registrations=(
            {
                "kind": "provider",
                "provider_mode": "mapping",
                "provider": "hostile",
                "name": "upgrade",
                "version": "1",
                "callback": lambda batch, _: batch,
                "input_ports": (("input", "array"),),
                "output_ports": (("output", "array"),),
                "options_schema": None,
                "modes": ("batch", "stream"),
                "finality": "per_row_final",
                "stateful": False,
                "microbatch_invariant": True,
                "requires_watermark": False,
                "checkpoint_support": "stateless",
                "state_version": 7,
                "deterministic": True,
                "replay_safe": True,
                "supports_static_inputs": True,
                "partition_contract": "row_axis_independent",
                "array_rules": {
                    "supported_dtypes": ("float64",),
                    "safe_dtype_rule": ("array_api_safe_dtype", "1"),
                },
            },
        ),
    )

    hostile = forged.providers[0]
    assert (hostile.provider, hostile.name) == ("hostile", "upgrade")
    assert hostile.modes == ("batch",)
    assert hostile.finality == "unproven"
    assert hostile.checkpoint_support == "stateless"
    assert hostile.state_version is None
    assert hostile.deterministic is False
    assert hostile.replay_safe is False
    assert hostile.supports_static_inputs is False
    assert hostile.partition_contract == "none"
    assert hostile.array_rules is None
    assert runtime.capabilities().providers[0].modes == ("batch",)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param(
            {
                "kind": "expression",
                "version": "1",
                "input_ports": (ProviderPort("input", "table", required=True),),
                "output_ports": (ProviderPort("output", "table", required=True),),
                "modes": ("batch",),
                "finality": "per_row_final",
                "requires_datafusion": True,
                "stateful": True,
                "microbatch_invariant": False,
                "requires_watermark": False,
                "checkpoint_support": "stateless",
                "state_version": None,
                "deterministic": True,
                "replay_safe": True,
            },
            id="stateless-with-stateful",
        ),
        pytest.param(
            {
                "kind": "expression",
                "version": "1",
                "input_ports": (ProviderPort("input", "table", required=True),),
                "output_ports": (ProviderPort("output", "table", required=True),),
                "modes": ("batch", "stream"),
                "finality": "per_row_final",
                "requires_datafusion": True,
                "stateful": True,
                "microbatch_invariant": False,
                "requires_watermark": False,
                "checkpoint_support": "checkpointed_stateful",
                "state_version": None,
                "deterministic": True,
                "replay_safe": True,
            },
            id="checkpointed-without-state-version",
        ),
        pytest.param(
            {
                "kind": "expression",
                "version": "1",
                "input_ports": (ProviderPort("input", "table", required=True),),
                "output_ports": (ProviderPort("output", "table", required=True),),
                "modes": ("batch", "stream"),
                "finality": "per_row_final",
                "requires_datafusion": True,
                "stateful": False,
                "microbatch_invariant": False,
                "requires_watermark": False,
                "checkpoint_support": "checkpointed_stateful",
                "state_version": 0,
                "deterministic": True,
                "replay_safe": True,
            },
            id="checkpointed-with-zero-state-version",
        ),
        pytest.param(
            {
                "kind": "expression",
                "version": "1",
                "input_ports": (ProviderPort("input", "table", required=True),),
                "output_ports": (ProviderPort("output", "table", required=True),),
                "modes": ("batch",),
                "finality": "per_row_final",
                "requires_datafusion": True,
                "stateful": False,
                "microbatch_invariant": False,
                "requires_watermark": False,
                "checkpoint_support": "stateless",
                "state_version": 2,
                "deterministic": True,
                "replay_safe": True,
            },
            id="stateless-with-state-version",
        ),
        pytest.param(
            {
                "kind": "expression",
                "version": "1",
                "input_ports": (ProviderPort("input", "table", required=True),),
                "output_ports": (ProviderPort("output", "table", required=True),),
                "modes": (),
                "finality": "per_row_final",
                "requires_datafusion": True,
                "stateful": False,
                "microbatch_invariant": False,
                "requires_watermark": False,
                "checkpoint_support": "stateless",
                "state_version": None,
                "deterministic": True,
                "replay_safe": True,
            },
            id="empty-modes",
        ),
        pytest.param(
            {
                "kind": "expression",
                "version": "1",
                "input_ports": (ProviderPort("input", "table", required=True),),
                "output_ports": (ProviderPort("output", "table", required=True),),
                "modes": ("batch", "continuous"),
                "finality": "per_row_final",
                "requires_datafusion": True,
                "stateful": False,
                "microbatch_invariant": False,
                "requires_watermark": False,
                "checkpoint_support": "stateless",
                "state_version": None,
                "deterministic": True,
                "replay_safe": True,
            },
            id="unknown-mode",
        ),
        pytest.param(
            {
                "kind": "expression",
                "version": "1",
                "input_ports": (ProviderPort("input", "table", required=True),),
                "output_ports": (ProviderPort("output", "table", required=True),),
                "modes": ("batch",),
                "finality": "append_only",
                "requires_datafusion": True,
                "stateful": False,
                "microbatch_invariant": False,
                "requires_watermark": False,
                "checkpoint_support": "stateless",
                "state_version": None,
                "deterministic": True,
                "replay_safe": True,
            },
            id="unknown-finality",
        ),
        pytest.param(
            {
                "kind": "expression",
                "version": "1",
                "input_ports": (ProviderPort("input", "table", required=True),),
                "output_ports": (ProviderPort("output", "table", required=True),),
                "modes": ("batch",),
                "finality": "per_row_final",
                "requires_datafusion": True,
                "stateful": False,
                "microbatch_invariant": False,
                "requires_watermark": False,
                "checkpoint_support": "durable",
                "state_version": None,
                "deterministic": True,
                "replay_safe": True,
            },
            id="unknown-checkpoint-support",
        ),
    ],
)
def test_operator_capability_rejects_unprovable_lifecycle_combinations(
    fields: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        OperatorCapability(**fields)  # type: ignore[arg-type]


def test_operator_capability_requires_strict_lifecycle_data() -> None:
    base = {
        "kind": "expression",
        "version": "1",
        "input_ports": (ProviderPort("input", "table", required=True),),
        "output_ports": (ProviderPort("output", "table", required=True),),
        "modes": ("batch", "stream"),
        "finality": "per_row_final",
        "requires_datafusion": True,
        "stateful": False,
        "microbatch_invariant": True,
        "requires_watermark": False,
        "checkpoint_support": "stateless",
        "state_version": None,
        "deterministic": True,
        "replay_safe": True,
    }

    with pytest.raises(TypeError, match="modes must be a tuple of execution modes"):
        OperatorCapability(**{**base, "modes": ["batch"]})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="stateful must be an exact bool"):
        OperatorCapability(**{**base, "stateful": 0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state_version must be"):
        OperatorCapability(
            **{
                **base,
                "checkpoint_support": "checkpointed_stateful",
                "state_version": True,
            }  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="finality must be an exact str"):
        OperatorCapability(**{**base, "finality": b"per_row_final"})  # type: ignore[arg-type]


def test_provider_capability_rejects_unprovable_lifecycle_combinations() -> None:
    def capability(**overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "provider": "numpy",
            "name": "table_matmul",
            "version": "1",
            "input_ports": (ProviderPort("table", "table", required=True),),
            "output_ports": (ProviderPort("output", "array", required=True),),
            "options_schema": None,
            "modes": ("batch", "stream"),
            "finality": "per_row_final",
            "stateful": False,
            "microbatch_invariant": True,
            "requires_watermark": False,
            "checkpoint_support": "stateless",
            "state_version": None,
            "deterministic": True,
            "replay_safe": True,
            "supports_static_inputs": True,
            "partition_contract": "row_axis_independent",
            "array_rules": None,
        }
        return {**base, **overrides}

    with pytest.raises(ValueError, match="stateless .* stateful=False"):
        ProviderCapability(**capability(stateful=True))
    with pytest.raises(ValueError, match="checkpointed_stateful .* positive"):
        ProviderCapability(
            **capability(
                checkpoint_support="checkpointed_stateful",
                stateful=True,
                state_version=None,
            )
        )
    with pytest.raises(ValueError, match="partition_contract"):
        ProviderCapability(**capability(partition_contract="shared_partitions"))
    with pytest.raises(ValueError, match="finality"):
        ProviderCapability(**capability(finality="eventual"))


def test_capability_rules_accept_only_the_closed_versioned_identities() -> None:
    from calc_flow import CapabilityRule, ProviderArrayRules

    rules = ProviderArrayRules(
        supported_dtypes=("float64", "float32"),
        safe_dtype_rule=CapabilityRule("array_api_safe_dtype", "1"),
        shape_rules=(
            CapabilityRule("table_matmul_static_rhs", "1"),
            CapabilityRule("elementwise_broadcast", "1"),
        ),
    )

    assert rules.supported_dtypes == ("float32", "float64")
    assert rules.shape_rules == (
        CapabilityRule("elementwise_broadcast", "1"),
        CapabilityRule("table_matmul_static_rhs", "1"),
    )
    with pytest.raises(ValueError, match="unknown capability rule"):
        ProviderArrayRules(
            supported_dtypes=("float64",),
            safe_dtype_rule=CapabilityRule("array_api_safe_dtype", "2"),
            shape_rules=(),
        )
    with pytest.raises(ValueError, match="unknown capability rule"):
        ProviderArrayRules(
            supported_dtypes=("float64",),
            safe_dtype_rule=CapabilityRule("array_api_safe_dtype", "1"),
            shape_rules=(CapabilityRule("reduce_along_any_axis", "1"),),
        )
    with pytest.raises(TypeError, match="supported_dtypes must be a tuple of str"):
        ProviderArrayRules(
            supported_dtypes=["float64"],  # type: ignore[arg-type]
            safe_dtype_rule=CapabilityRule("array_api_safe_dtype", "1"),
            shape_rules=(),
        )
    with pytest.raises(
        TypeError,
        match="capability rule name and version must be exact strings",
    ):
        CapabilityRule(b"array_api_safe_dtype", "1")  # type: ignore[arg-type]
    with pytest.raises(
        TypeError,
        match="capability rule name and version must be exact strings",
    ):
        CapabilityRule("array_api_safe_dtype", 1)  # type: ignore[arg-type]


def test_capability_snapshot_ignores_later_mutation_of_caller_owned_sequences() -> None:
    runtime = Runtime()
    input_ports = [("table", "table"), ("weights", "array")]
    output_ports = [("output", "array")]
    input_types = ["int64", "float64"]

    runtime._register_mapping_provider(
        "numpy",
        "table_matmul",
        "1",
        lambda _inputs, _options: {},
        input_ports=input_ports,
        output_ports=output_ports,
    )
    runtime.register_scalar_udf(
        provider="python",
        name="scaled",
        version="1",
        input_types=input_types,
        return_type="float64",
        volatility="immutable",
        function=lambda left, right: left,
    )
    snapshot = runtime.capabilities()

    input_ports.append(("hijacked", "table"))
    output_ports[0] = ("hijacked", "array")
    input_types[0] = "hijacked"

    assert runtime.capabilities() == snapshot
    assert snapshot.providers[0].input_ports == (
        ProviderPort("table", "table", required=True),
        ProviderPort("weights", "array", required=True),
    )
    assert snapshot.providers[0].output_ports == (
        ProviderPort("output", "array", required=True),
    )
    assert snapshot.udfs[0].input_types == ("int64", "float64")


def test_capability_snapshot_stays_immutable_across_later_registrations() -> None:
    runtime = Runtime()

    before = runtime.capabilities()
    runtime.register_provider("zeta", "transform", "1", lambda batch, _: batch)
    runtime.register_provider("alpha", "transform", "1", lambda batch, _: batch)

    after = runtime.capabilities()

    assert before.scope.revision == 0
    assert before.providers == ()
    assert after.scope.session_id == before.scope.session_id
    assert after.scope.revision == 2
    assert tuple(provider.provider for provider in after.providers) == (
        "alpha",
        "zeta",
    )
    assert runtime.capabilities() == after
