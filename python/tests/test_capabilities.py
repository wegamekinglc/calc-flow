from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import calc_flow
from calc_flow import (
    CalcFlowError,
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
    assert snapshot.schema_version == 1
    assert snapshot.scope.kind == "runtime_session"
    assert snapshot.scope.revision == 0
    assert snapshot.scope.session_id == repeated.scope.session_id
    assert snapshot.package_version == "3.0.0"
    assert snapshot.project_format_versions == (3,)
    assert snapshot.batch_kinds == ("array", "table")
    assert tuple(operator.kind for operator in snapshot.operators) == (
        "expression",
        "sql",
    )
    assert all(operator.input_kinds == ("table",) for operator in snapshot.operators)
    assert all(operator.output_kinds == ("table",) for operator in snapshot.operators)
    assert all(operator.requires_datafusion for operator in snapshot.operators)
    assert snapshot.udfs == ()
    assert snapshot.providers == ()
    assert tuple(connector.name for connector in snapshot.connectors) == ("file",)
    with pytest.raises(FrozenInstanceError):
        snapshot.scope.revision = 1  # type: ignore[misc]


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
        "OperatorCapability",
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
