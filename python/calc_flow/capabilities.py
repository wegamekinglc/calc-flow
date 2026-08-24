from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

type BatchKind = Literal["table", "array"]
type OptionValueType = Literal["string", "integer", "number", "boolean"]
type ExecutionMode = Literal["batch", "stream"]
type OutputFinality = Literal["per_row_final", "group_final_append_only", "unproven"]
type CheckpointSupport = Literal["stateless", "checkpointed_stateful", "unproven"]
type PartitionContract = Literal["none", "row_axis_independent"]

CAPABILITY_SCHEMA_VERSION = 2

_EXECUTION_MODES = frozenset(("batch", "stream"))
_OUTPUT_FINALITIES = frozenset(("per_row_final", "group_final_append_only", "unproven"))
_CHECKPOINT_SUPPORTS = frozenset(("stateless", "checkpointed_stateful", "unproven"))
_PARTITION_CONTRACTS = frozenset(("none", "row_axis_independent"))
_CAPABILITY_RULES = frozenset(
    (
        ("array_api_safe_dtype", "1"),
        ("elementwise_broadcast", "1"),
        ("feature_axis_reduction", "1"),
        ("table_matmul_static_rhs", "1"),
    )
)

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


def _require_exact_str(owner: str, field: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(
            f"{owner} {field} must be an exact str; found {type(value).__name__}"
        )


def _require_exact_bool(owner: str, field: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(
            f"{owner} {field} must be an exact bool; found {type(value).__name__}"
        )


def _require_port_tuple(owner: str, field: str, value: object) -> None:
    if type(value) is not tuple or any(
        not isinstance(port, ProviderPort) for port in value
    ):
        raise TypeError(
            f"{owner} {field} must be a tuple of ProviderPort values; "
            f"found {type(value).__name__}"
        )


def _require_modes(owner: str, value: object) -> tuple[ExecutionMode, ...]:
    if type(value) is not tuple or any(type(mode) is not str for mode in value):
        raise TypeError(f"{owner} modes must be a tuple of execution modes")
    if not value:
        raise ValueError(f"{owner} modes must declare at least one execution mode")
    unknown = next((mode for mode in value if mode not in _EXECUTION_MODES), None)
    if unknown is not None:
        raise ValueError(
            f"{owner} execution modes are batch and stream; found {unknown!r}"
        )
    return value  # type: ignore[return-value]


def _require_finality(owner: str, value: object) -> OutputFinality:
    if type(value) is not str:
        raise TypeError(
            f"{owner} finality must be an exact str; found {type(value).__name__}"
        )
    if value not in _OUTPUT_FINALITIES:
        raise ValueError(
            "output finality must be per_row_final, group_final_append_only, or "
            f"unproven; found {value!r}"
        )
    return value  # type: ignore[return-value]


def _require_checkpoint_support(
    owner: str, support: object, state_version: object, stateful: bool
) -> None:
    if type(support) is not str:
        raise TypeError(
            f"{owner} checkpoint_support must be an exact str; "
            f"found {type(support).__name__}"
        )
    if support not in _CHECKPOINT_SUPPORTS:
        raise ValueError(
            "checkpoint support must be stateless, checkpointed_stateful, or "
            f"unproven; found {support!r}"
        )
    if support == "checkpointed_stateful":
        if state_version is None or (type(state_version) is int and state_version <= 0):
            raise ValueError(
                f"{owner} checkpointed_stateful requires a positive state_version"
            )
        if type(state_version) is not int:
            raise TypeError(
                f"{owner} state_version must be an exact int when "
                f"checkpointed_stateful; found {type(state_version).__name__}"
            )
    elif state_version is not None:
        raise ValueError(
            f"{owner} state_version must be None unless checkpointed_stateful"
        )
    if support == "stateless" and stateful:
        raise ValueError(f"{owner} stateless capability must set stateful=False")


@dataclass(frozen=True, slots=True)
class CapabilityRule:
    name: str
    version: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or type(self.version) is not str:
            raise TypeError(
                "capability rule name and version must be exact strings; found "
                f"{type(self.name).__name__} and {type(self.version).__name__}"
            )
        if (self.name, self.version) not in _CAPABILITY_RULES:
            raise ValueError(f"unknown capability rule {self.name}@{self.version}")


@dataclass(frozen=True, slots=True)
class ProviderArrayRules:
    supported_dtypes: tuple[str, ...]
    safe_dtype_rule: CapabilityRule
    shape_rules: tuple[CapabilityRule, ...]

    def __post_init__(self) -> None:
        if type(self.supported_dtypes) is not tuple or any(
            type(dtype) is not str for dtype in self.supported_dtypes
        ):
            raise TypeError(
                "provider array rules supported_dtypes must be a tuple of str; "
                f"found {type(self.supported_dtypes).__name__}"
            )
        if not isinstance(self.safe_dtype_rule, CapabilityRule):
            raise TypeError(
                "provider array rules safe_dtype_rule must be a CapabilityRule; "
                f"found {type(self.safe_dtype_rule).__name__}"
            )
        if type(self.shape_rules) is not tuple or any(
            not isinstance(rule, CapabilityRule) for rule in self.shape_rules
        ):
            raise TypeError(
                "provider array rules shape_rules must be a tuple of "
                f"CapabilityRule values; found {type(self.shape_rules).__name__}"
            )
        object.__setattr__(
            self, "supported_dtypes", tuple(sorted(self.supported_dtypes))
        )
        object.__setattr__(
            self,
            "shape_rules",
            tuple(
                sorted(
                    self.shape_rules,
                    key=lambda rule: (rule.name, rule.version),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class OperatorCapability:
    kind: str
    version: str
    input_ports: tuple[ProviderPort, ...]
    output_ports: tuple[ProviderPort, ...]
    modes: tuple[ExecutionMode, ...]
    finality: OutputFinality
    requires_datafusion: bool
    stateful: bool
    microbatch_invariant: bool
    requires_watermark: bool
    checkpoint_support: CheckpointSupport
    state_version: int | None
    deterministic: bool
    replay_safe: bool

    def __post_init__(self) -> None:
        _require_exact_str("operator capability", "kind", self.kind)
        _require_exact_str("operator capability", "version", self.version)
        _require_port_tuple("operator capability", "input_ports", self.input_ports)
        _require_port_tuple("operator capability", "output_ports", self.output_ports)
        _require_modes("operator capability", self.modes)
        _require_finality("operator capability", self.finality)
        _require_exact_bool(
            "operator capability", "requires_datafusion", self.requires_datafusion
        )
        _require_exact_bool("operator capability", "stateful", self.stateful)
        _require_exact_bool(
            "operator capability", "microbatch_invariant", self.microbatch_invariant
        )
        _require_exact_bool(
            "operator capability", "requires_watermark", self.requires_watermark
        )
        _require_exact_bool("operator capability", "deterministic", self.deterministic)
        _require_exact_bool("operator capability", "replay_safe", self.replay_safe)
        _require_checkpoint_support(
            "operator capability",
            self.checkpoint_support,
            self.state_version,
            self.stateful,
        )


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
    modes: tuple[ExecutionMode, ...]
    finality: OutputFinality
    stateful: bool
    microbatch_invariant: bool
    requires_watermark: bool
    checkpoint_support: CheckpointSupport
    state_version: int | None
    deterministic: bool
    replay_safe: bool
    supports_static_inputs: bool
    partition_contract: PartitionContract
    array_rules: ProviderArrayRules | None

    def __post_init__(self) -> None:
        _require_exact_str("provider capability", "provider", self.provider)
        _require_exact_str("provider capability", "version", self.version)
        _require_port_tuple("provider capability", "input_ports", self.input_ports)
        _require_port_tuple("provider capability", "output_ports", self.output_ports)
        _require_modes("provider capability", self.modes)
        _require_finality("provider capability", self.finality)
        _require_exact_bool("provider capability", "stateful", self.stateful)
        _require_exact_bool(
            "provider capability", "microbatch_invariant", self.microbatch_invariant
        )
        _require_exact_bool(
            "provider capability", "requires_watermark", self.requires_watermark
        )
        _require_exact_bool("provider capability", "deterministic", self.deterministic)
        _require_exact_bool("provider capability", "replay_safe", self.replay_safe)
        _require_exact_bool(
            "provider capability", "supports_static_inputs", self.supports_static_inputs
        )
        if type(self.partition_contract) is not str:
            raise TypeError(
                "provider capability partition_contract must be an exact str; "
                f"found {type(self.partition_contract).__name__}"
            )
        if self.partition_contract not in _PARTITION_CONTRACTS:
            raise ValueError(
                "provider capability partition_contract must be none or "
                f"row_axis_independent; found {self.partition_contract!r}"
            )
        if self.array_rules is not None and not isinstance(
            self.array_rules, ProviderArrayRules
        ):
            raise TypeError(
                "provider capability array_rules must be a ProviderArrayRules or "
                f"None; found {type(self.array_rules).__name__}"
            )
        _require_checkpoint_support(
            "provider capability",
            self.checkpoint_support,
            self.state_version,
            self.stateful,
        )


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    schema_version: Literal[2]
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
