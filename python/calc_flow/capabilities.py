from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

type BatchKind = Literal["table", "array"]
type OptionValueType = Literal["string", "integer", "number", "boolean"]

CAPABILITY_SCHEMA_VERSION = 1

PORTABLE_ARROW_TYPES = (
    "bool",
    "date32",
    "date64",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "large_string",
    "string",
    "time32[s]",
    "time64[us]",
    "timestamp[ms]",
    "timestamp[us]",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)


@dataclass(frozen=True, slots=True)
class ProviderOption:
    name: str
    value_type: OptionValueType
    required: bool = False

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError(
                "provider options_schema field name must contain strict data; "
                f"found {type(self.name).__name__}"
            )
        if type(self.value_type) is not str:
            raise TypeError(
                f"provider options_schema field {self.name!r}.value_type must "
                "contain strict data; "
                f"found {type(self.value_type).__name__}"
            )
        if self.value_type not in {"string", "integer", "number", "boolean"}:
            raise ValueError(
                f"provider options_schema field {self.name!r}.value_type must be "
                "string, integer, number, or boolean; "
                f"found {self.value_type}"
            )
        if type(self.required) is not bool:
            raise TypeError(
                f"provider options_schema at {self.name!r}.required must contain "
                f"strict data; found {type(self.required).__name__}"
            )


@dataclass(frozen=True, slots=True)
class ProviderOptionsSchema:
    fields: tuple[ProviderOption, ...] = ()
    additional_properties: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.fields) is not tuple or any(
            not isinstance(field, ProviderOption) for field in self.fields
        ):
            raise TypeError(
                "provider options_schema fields must be a tuple of "
                "ProviderOption values"
            )
        if self.additional_properties is not False:
            raise ValueError(
                "provider options_schema additional_properties must be False"
            )
        names = tuple(field.name for field in self.fields)
        duplicate = next(
            (name for index, name in enumerate(names) if name in names[:index]),
            None,
        )
        if duplicate is not None:
            raise ValueError(
                f"provider options_schema contains duplicate field {duplicate!r}"
            )
        object.__setattr__(
            self,
            "fields",
            tuple(sorted(self.fields, key=lambda field: field.name)),
        )


@dataclass(frozen=True, slots=True)
class RuntimeSessionScope:
    kind: Literal["runtime_session"]
    session_id: str
    revision: int


@dataclass(frozen=True, slots=True)
class OperatorCapability:
    kind: str
    input_kinds: tuple[BatchKind, ...]
    output_kinds: tuple[BatchKind, ...]
    requires_datafusion: bool


@dataclass(frozen=True, slots=True)
class UdfCapability:
    provider: str
    name: str
    version: str
    kind: Literal["data_fusion_scalar"]
    input_types: tuple[str, ...]
    return_type: str
    volatility: str


@dataclass(frozen=True, slots=True)
class ProviderPort:
    name: str
    kind: BatchKind
    required: bool


@dataclass(frozen=True, slots=True)
class ProviderCapability:
    provider: str
    name: str
    version: str
    input_ports: tuple[ProviderPort, ...]
    output_ports: tuple[ProviderPort, ...]
    options_schema: ProviderOptionsSchema | None


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    schema_version: Literal[1]
    scope: RuntimeSessionScope
    package_version: str
    project_format_versions: tuple[int, ...]
    batch_kinds: tuple[BatchKind, ...]
    portable_arrow_types: tuple[str, ...]
    operators: tuple[OperatorCapability, ...]
    udfs: tuple[UdfCapability, ...]
    providers: tuple[ProviderCapability, ...]
    connectors: tuple[ConnectorCapability, ...]


def runtime_capabilities(
    *,
    session_id: str,
    revision: int,
    package_version: str,
    registrations: Sequence[Mapping[str, Any]],
) -> RuntimeCapabilities:
    from calc_flow import _native

    udfs = tuple(
        sorted(
            (
                UdfCapability(
                    provider=str(registration["provider"]),
                    name=str(registration["name"]),
                    version=str(registration["version"]),
                    kind="data_fusion_scalar",
                    input_types=tuple(registration["input_types"]),
                    return_type=str(registration["return_type"]),
                    volatility=str(registration["volatility"]),
                )
                for registration in registrations
                if registration["kind"] == "scalar_udf"
            ),
            key=lambda item: (item.provider, item.name, item.version),
        )
    )
    providers = tuple(
        sorted(
            (
                ProviderCapability(
                    provider=str(registration["provider"]),
                    name=str(registration["name"]),
                    version=str(registration["version"]),
                    input_ports=tuple(
                        ProviderPort(str(name), kind, required=True)
                        for name, kind in registration.get(
                            "input_ports", (("input", "array"),)
                        )
                    ),
                    output_ports=tuple(
                        ProviderPort(str(name), kind, required=True)
                        for name, kind in registration.get(
                            "output_ports", (("output", "array"),)
                        )
                    ),
                    options_schema=registration.get("options_schema"),
                )
                for registration in registrations
                if registration["kind"] == "provider"
            ),
            key=lambda item: (item.provider, item.name, item.version),
        )
    )
    return RuntimeCapabilities(
        schema_version=CAPABILITY_SCHEMA_VERSION,
        scope=RuntimeSessionScope(
            kind="runtime_session",
            session_id=session_id,
            revision=revision,
        ),
        package_version=package_version,
        project_format_versions=(3,),
        batch_kinds=("array", "table"),
        portable_arrow_types=PORTABLE_ARROW_TYPES,
        operators=(
            OperatorCapability(
                kind="expression",
                input_kinds=("table",),
                output_kinds=("table",),
                requires_datafusion=True,
            ),
            OperatorCapability(
                kind="sql",
                input_kinds=("table",),
                output_kinds=("table",),
                requires_datafusion=True,
            ),
            OperatorCapability(
                kind="stream_join",
                input_kinds=("table", "table"),
                output_kinds=("table",),
                requires_datafusion=True,
            ),
        ),
        udfs=udfs,
        providers=providers,
        connectors=connector_capabilities(_native.registered_connectors()),
    )


# ---------------------------------------------------------------------------
# Connector capability surface (M6-08)
# ---------------------------------------------------------------------------

type DeliveryCapabilityKind = Literal["best_effort", "at_least_once", "exactly_once"]
type ReplayCapabilityKind = Literal["replayable_exact", "unreplayable"]
type WatermarkSupportKind = Literal["native", "generated_only"]
type TransactionSupportKind = Literal[
    "none",
    "pre_commit_commit",
    "ledger_idempotent",
    "retry_deduplicated",
]


@dataclass(frozen=True, slots=True)
class ConnectorCapabilities:
    delivery: DeliveryCapabilityKind
    replay: ReplayCapabilityKind
    watermark: WatermarkSupportKind
    transaction: TransactionSupportKind
    snapshot: bool
    polling: bool
    cdc: bool
    lookup: bool


@dataclass(frozen=True, slots=True)
class ConnectorCapability:
    provider: str
    name: str
    version: str
    kind: Literal["source", "sink", "both"]
    capabilities: ConnectorCapabilities
    formats: tuple[str, ...]
    options_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.provider) is not str or not self.provider:
            raise ValueError("connector capability provider must be a non-empty string")
        if type(self.name) is not str or not self.name:
            raise ValueError("connector capability name must be a non-empty string")
        if type(self.version) is not str or not self.version:
            raise ValueError("connector capability version must be a non-empty string")
        if self.kind not in {"source", "sink", "both"}:
            raise ValueError(
                f"connector capability kind must be source, sink, or both; "
                f"found {self.kind}"
            )
        if not isinstance(self.capabilities, ConnectorCapabilities):
            raise TypeError(
                "connector capability capabilities must be a ConnectorCapabilities; "
                f"found {type(self.capabilities).__name__}"
            )
        if not isinstance(self.formats, tuple):
            raise TypeError(
                "connector capability formats must be a tuple of strings; "
                f"found {type(self.formats).__name__}"
            )
        if not isinstance(self.options_schema, Mapping):
            raise TypeError(
                "connector capability options_schema must be a Mapping; "
                f"found {type(self.options_schema).__name__}"
            )


def _parse_options_schema(raw: object) -> dict[str, object]:
    """Parses the JSON string the native layer serializes into a dict."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return dict(json.loads(raw))
    return {}


def connector_capabilities(
    registrations: Sequence[Mapping[str, Any]],
) -> tuple[ConnectorCapability, ...]:
    """Builds the sorted tuple of connector capabilities from native data."""
    parsed: list[ConnectorCapability] = []
    for registration in registrations:
        caps_data = registration.get("capabilities", {})
        caps = ConnectorCapabilities(
            delivery=str(caps_data.get("delivery", "at_least_once")),
            replay=str(caps_data.get("replay", "unreplayable")),
            watermark=str(caps_data.get("watermark", "generated_only")),
            transaction=str(caps_data.get("transaction", "none")),
            snapshot=bool(caps_data.get("snapshot", False)),
            polling=bool(caps_data.get("polling", False)),
            cdc=bool(caps_data.get("cdc", False)),
            lookup=bool(caps_data.get("lookup", False)),
        )
        parsed.append(
            ConnectorCapability(
                provider=str(registration["provider"]),
                name=str(registration["name"]),
                version=str(registration["version"]),
                kind=str(registration.get("kind", "both")),
                capabilities=caps,
                formats=tuple(str(f) for f in registration.get("formats", ())),
                options_schema=_parse_options_schema(
                    registration.get("options_schema", "{}")
                ),
            )
        )
    return tuple(sorted(parsed, key=lambda c: (c.provider, c.name, c.version)))
