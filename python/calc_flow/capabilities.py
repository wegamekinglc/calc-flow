from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any, Literal

from calc_flow._compat import TypeAliasType, dataclass

if sys.version_info < (3, 10):
    from calc_flow._compat import zip as zip

BatchKind = TypeAliasType("BatchKind", Literal["table", "array"])
OptionValueType = TypeAliasType(
    "OptionValueType", Literal["string", "integer", "number", "boolean"]
)
ExecutionMode = TypeAliasType("ExecutionMode", Literal["batch", "stream"])
OutputFinality = TypeAliasType(
    "OutputFinality", Literal["per_row_final", "group_final_append_only", "unproven"]
)
CheckpointSupport = TypeAliasType(
    "CheckpointSupport", Literal["stateless", "checkpointed_stateful", "unproven"]
)
PartitionContract = TypeAliasType(
    "PartitionContract",
    Literal["none", "row_axis_independent", "single_ordered_sequence"],
)

CAPABILITY_SCHEMA_VERSION = 5

_EXECUTION_MODES = frozenset(("batch", "stream"))
_OUTPUT_FINALITIES = frozenset(("per_row_final", "group_final_append_only", "unproven"))
_CHECKPOINT_SUPPORTS = frozenset(("stateless", "checkpointed_stateful", "unproven"))
_PARTITION_CONTRACTS = frozenset(
    ("none", "row_axis_independent", "single_ordered_sequence")
)
_CAPABILITY_RULES = frozenset(
    (
        ("array_api_safe_dtype", "1"),
        ("elementwise_broadcast", "1"),
        ("feature_axis_reduction", "1"),
        ("table_matmul_static_rhs", "1"),
        ("native_float_bool_dtype", "1"),
        ("native_scalar_vector_broadcast", "1"),
        ("native_packed_rank2", "1"),
        ("native_matmul", "1"),
        ("native_axis_reduction", "1"),
        ("native_table_static_rhs", "1"),
        ("native_float_dtype", "1"),
        ("native_sequence_vector", "1"),
        ("native_sequence_indicators", "2"),
        ("native_sequence_indicators", "3"),
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


def _require_state_version(owner: str, support: str, state_version: object) -> None:
    if support != "checkpointed_stateful":
        if state_version is not None:
            raise ValueError(
                f"{owner} state_version must be None unless checkpointed_stateful"
            )
        return
    if state_version is None or (type(state_version) is int and state_version <= 0):
        raise ValueError(
            f"{owner} checkpointed_stateful requires a positive state_version"
        )
    if type(state_version) is not int:
        raise TypeError(
            f"{owner} state_version must be an exact int when "
            f"checkpointed_stateful; found {type(state_version).__name__}"
        )


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
    _require_state_version(owner, support, state_version)
    if support == "stateless" and stateful:
        raise ValueError(f"{owner} stateless capability must set stateful=False")


def _require_state_layouts(owner: str, support: str, state_layouts: object) -> None:
    """Validate the durable layout inventory behind a state contract.

    ``state_layouts`` enumerates every checkpoint layout the operator
    version may install; a single ``state_version`` cannot express a
    per-declaration layout split such as rolling's EWMA layout v2.
    """

    if type(state_layouts) is not tuple:
        raise TypeError(
            f"{owner} state_layouts must be an exact tuple; "
            f"found {type(state_layouts).__name__}"
        )
    if support != "checkpointed_stateful":
        if state_layouts:
            raise ValueError(
                f"{owner} state_layouts must be empty unless checkpointed_stateful"
            )
        return
    _require_stateful_layout_inventory(owner, state_layouts)


def _require_stateful_layout_inventory(
    owner: str, state_layouts: tuple[int, ...]
) -> None:
    if not state_layouts:
        raise ValueError(
            f"{owner} checkpointed_stateful requires at least one state layout"
        )
    for layout in state_layouts:
        if type(layout) is not int:
            raise TypeError(
                f"{owner} state layouts must be exact positive integers; "
                f"found {type(layout).__name__}"
            )
        if layout <= 0:
            raise ValueError(f"{owner} state layouts must be positive integers")
    _require_ascending_layouts(owner, state_layouts)


def _require_ascending_layouts(owner: str, state_layouts: tuple[int, ...]) -> None:
    if any(
        left >= right
        for left, right in zip(state_layouts[:-1], state_layouts[1:], strict=True)
    ):
        raise ValueError(
            f"{owner} state_layouts must be strictly ascending without duplicates"
        )


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
class _StatelessProviderLifecycle:
    """Trusted process-local proof attached by the native stream seam."""

    deterministic: bool
    replay_safe: bool
    supports_static_inputs: bool
    array_rules: ProviderArrayRules


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
    state_layouts: tuple[int, ...] = ()

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
        _require_state_layouts(
            "operator capability",
            self.checkpoint_support,
            self.state_layouts,
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


def _unique_schema_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("provider options_json_schema contains duplicate keys")
    return result


def _reject_schema_constant(_value: str) -> None:
    raise ValueError("provider options_json_schema must contain finite JSON data")


def _require_options_json_schema(schema: object) -> None:
    _require_exact_str("provider capability", "options_json_schema", schema)
    try:
        decoded = json.loads(
            schema,
            object_pairs_hook=_unique_schema_object,
            parse_constant=_reject_schema_constant,
        )
    except (ValueError, RecursionError) as error:
        raise ValueError("provider options_json_schema must be strict JSON") from error
    if (
        type(decoded) is not dict
        or decoded.get("type") != "object"
        or decoded.get("additionalProperties") is not False
    ):
        raise ValueError(
            "provider options_json_schema must describe a strict object schema"
        )


def _require_provider_port_binding(capability: ProviderCapability) -> None:
    _require_exact_str("provider capability", "port_binding", capability.port_binding)
    if capability.port_binding not in {"fixed", "options"}:
        raise ValueError("provider port_binding must be fixed or options")
    if capability.options_schema is not None and not isinstance(
        capability.options_schema, ProviderOptionsSchema
    ):
        raise TypeError(
            "provider options_schema must be a ProviderOptionsSchema or None"
        )
    if capability.port_binding == "fixed":
        if capability.options_json_schema is not None:
            _require_json_options_schema(capability)
        return
    _require_options_bound_ports(capability)


def _require_options_bound_ports(capability: ProviderCapability) -> None:
    if capability.input_ports or capability.output_ports:
        raise ValueError(
            "provider options port_binding requires empty input_ports/output_ports"
        )
    _require_json_options_schema(capability)


def _require_json_options_schema(capability: ProviderCapability) -> None:
    if capability.options_schema is not None:
        raise ValueError(
            "provider options_schema and options_json_schema are exclusive"
        )
    if capability.options_json_schema is None:
        raise ValueError("provider options port_binding requires options_json_schema")
    _require_options_json_schema(capability.options_json_schema)


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
    port_binding: Literal["fixed", "options"] = "fixed"
    options_json_schema: str | None = None
    state_layouts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_exact_str("provider capability", "provider", self.provider)
        _require_exact_str("provider capability", "name", self.name)
        _require_exact_str("provider capability", "version", self.version)
        _require_port_tuple("provider capability", "input_ports", self.input_ports)
        _require_port_tuple("provider capability", "output_ports", self.output_ports)
        _require_provider_port_binding(self)
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
                "provider capability partition_contract must be none, "
                "row_axis_independent or single_ordered_sequence; "
                f"found {self.partition_contract!r}"
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

        _require_state_layouts(
            "provider capability", self.checkpoint_support, self.state_layouts
        )


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    schema_version: Literal[5]
    scope: RuntimeSessionScope
    package_version: str
    project_format_versions: tuple[int, ...]
    batch_kinds: tuple[BatchKind, ...]
    portable_arrow_types: tuple[str, ...]
    operators: tuple[OperatorCapability, ...]
    udfs: tuple[UdfCapability, ...]
    providers: tuple[ProviderCapability, ...]
    connectors: tuple[ConnectorCapability, ...]


def _descriptor_object(raw: object, expected_fields: frozenset[str]) -> dict[str, Any]:
    if type(raw) is not dict or any(type(key) is not str for key in raw):
        raise TypeError("native array capability must contain exact JSON objects")
    if raw.keys() != expected_fields:
        raise ValueError("native array capability contains missing or unknown fields")
    return raw


def _descriptor_list(raw: object) -> list[Any]:
    if type(raw) is not list:
        raise TypeError("native array capability must contain exact JSON lists")
    return raw


def _descriptor_port(raw: object) -> ProviderPort:
    port = _descriptor_object(raw, frozenset(("name", "kind", "required")))
    _require_exact_str("native capability port", "name", port["name"])
    _require_exact_str("native capability port", "kind", port["kind"])
    _require_exact_bool("native capability port", "required", port["required"])
    if port["kind"] not in {"table", "array"}:
        raise ValueError("native capability port kind must be table or array")
    return ProviderPort(**port)


def _descriptor_rule(raw: object) -> CapabilityRule:
    rule = _descriptor_object(raw, frozenset(("name", "version")))
    return CapabilityRule(**rule)


def _descriptor_array_rules(raw: object) -> ProviderArrayRules:
    rules = _descriptor_object(
        raw, frozenset(("supported_dtypes", "safe_dtype_rule", "shape_rules"))
    )
    return ProviderArrayRules(
        supported_dtypes=tuple(_descriptor_list(rules["supported_dtypes"])),
        safe_dtype_rule=_descriptor_rule(rules["safe_dtype_rule"]),
        shape_rules=tuple(
            _descriptor_rule(rule) for rule in _descriptor_list(rules["shape_rules"])
        ),
    )


def _require_native_capability_fields(
    capability: ProviderCapability, expected: Mapping[str, object]
) -> None:
    for field, required in expected.items():
        if getattr(capability, field) != required:
            raise ValueError(
                f"native {capability.name} capability {field} contradicts its contract"
            )


def _require_native_regular_capability(capability: ProviderCapability) -> None:
    modes, support, partition = {
        "array_expr": (("batch",), "unproven", "none"),
        "table_array": (("batch", "stream"), "stateless", "row_axis_independent"),
    }[capability.name]
    _require_native_capability_fields(
        capability,
        {
            "modes": modes,
            "checkpoint_support": support,
            "stateful": False,
            "state_layouts": (),
            "partition_contract": partition,
            "port_binding": "options",
        },
    )


def _require_native_sequence_capability(capability: ProviderCapability) -> None:
    if capability.state_layouts not in {(1,), (1, 2), (1, 2, 3)}:
        raise ValueError(
            "native array_sequence capability has unsupported state_layouts"
        )
    shape_rules = (CapabilityRule("native_sequence_vector", "1"),) + tuple(
        CapabilityRule("native_sequence_indicators", str(layout))
        for layout in capability.state_layouts
        if layout >= 2
    )
    expected = {
        "input_ports": (ProviderPort("input", "array", True),),
        "output_ports": (ProviderPort("output", "array", True),),
        "options_schema": None,
        "port_binding": "fixed",
        "modes": ("stream",),
        "stateful": True,
        "microbatch_invariant": True,
        "requires_watermark": False,
        "finality": "per_row_final",
        "checkpoint_support": "checkpointed_stateful",
        "state_version": 1,
        "state_layouts": capability.state_layouts,
        "deterministic": True,
        "replay_safe": True,
        "supports_static_inputs": False,
        "partition_contract": "single_ordered_sequence",
        "array_rules": ProviderArrayRules(
            supported_dtypes=("float32", "float64"),
            safe_dtype_rule=CapabilityRule("native_float_dtype", "1"),
            shape_rules=shape_rules,
        ),
    }
    _require_native_capability_fields(capability, expected)
    if capability.options_json_schema is None:
        raise ValueError(
            "native array_sequence capability requires options_json_schema"
        )
    _require_native_sequence_schema(
        capability.options_json_schema, capability.state_layouts
    )


def _sequence_schema_error(field: str) -> ValueError:
    return ValueError(
        f"native array_sequence capability has incompatible options_json_schema.{field}"
    )


def _sequence_schema_object(
    value: object,
    properties: tuple[str, ...],
    required: tuple[str, ...],
    alternatives: bool = False,
) -> dict[str, Any]:
    keys = {"type", "additionalProperties", "properties", "required"}
    if alternatives:
        keys.add("oneOf")
    schema = _descriptor_object(value, frozenset(keys))
    if schema["type"] != "object" or schema["additionalProperties"] is not False:
        raise _sequence_schema_error("object")
    fields = _descriptor_object(schema["properties"], frozenset(properties))
    if schema["required"] != list(required):
        raise _sequence_schema_error("required")
    return fields


def _sequence_schema_count(
    value: object, minimum: int, default: int | None = None
) -> None:
    expected = {"type": "integer", "minimum": minimum, "maximum": 2**64 - 1}
    if default is not None:
        expected["default"] = default
    schema = _descriptor_object(value, frozenset(expected))
    for name, required in expected.items():
        if type(schema[name]) is not type(required) or schema[name] != required:
            raise _sequence_schema_error(name)


def _sequence_schema_common_options(properties: dict[str, Any]) -> None:
    if properties["input_dtype"] != {"enum": ["float32", "float64"]}:
        raise _sequence_schema_error("input_dtype")
    for field, defaults in (
        (
            "execution",
            {"max_elements": 10_000_000, "max_allocation_bytes": 256 * 1024 * 1024},
        ),
        ("state", {"max_state_bytes": 256 * 1024 * 1024}),
    ):
        options = _sequence_schema_object(properties[field], tuple(defaults), ())
        for name, default in defaults.items():
            _sequence_schema_count(options[name], 0, default)


def _sequence_schema_version(value: object, versions: tuple[int, ...]) -> None:
    field = "const" if len(versions) == 1 else "enum"
    schema = _descriptor_object(value, frozenset({field}))
    observed = [schema[field]] if field == "const" else schema[field]
    if (
        type(observed) is not list
        or any(type(version) is not int for version in observed)
        or tuple(observed) != versions
    ):
        raise _sequence_schema_error("ir_version")


def _sequence_schema_kernel_defaults(properties: dict[str, Any]) -> None:
    if "min_periods" in properties:
        _sequence_schema_count(properties["min_periods"], 1, 1)
    if "numerical_profile" in properties and properties["numerical_profile"] != {
        "enum": ["stable_v1", "stable_v2"],
        "default": "stable_v1",
    }:
        raise _sequence_schema_error("numerical_profile")


def _sequence_schema_kernel_branch(value: object, function: str) -> None:
    if function == "macd_components":
        _sequence_schema_macd(value)
        return
    count, optional = {
        "rolling_mean": ("window", ("min_periods", "numerical_profile")),
        "ewma": ("span", ("min_periods",)),
        "rsi": ("period", ()),
        "atr": ("period", ()),
    }[function]
    properties = _sequence_schema_object(
        value, ("function", count, *optional), ("function", count)
    )
    if properties["function"] != {"const": function}:
        raise _sequence_schema_error("function")
    _sequence_schema_count(properties[count], 1)
    _sequence_schema_kernel_defaults(properties)


def _sequence_schema_macd(value: object) -> None:
    defaults = {
        "fast_span": 12,
        "slow_span": 26,
        "signal_span": 9,
        "min_periods": 1,
        "signal_min_periods": 1,
    }
    properties = _sequence_schema_object(value, ("function", *defaults), ("function",))
    if properties["function"] != {"const": "macd_components"}:
        raise _sequence_schema_error("function")
    for name, default in defaults.items():
        _sequence_schema_count(properties[name], 1, default)


def _sequence_schema_kernels(value: object, functions: tuple[str, ...]) -> None:
    schema = _descriptor_object(value, frozenset({"oneOf"}))
    branches = _descriptor_list(schema["oneOf"])
    if len(branches) != len(functions):
        raise _sequence_schema_error("kernel.oneOf")
    for branch, function in zip(branches, functions, strict=True):
        _sequence_schema_kernel_branch(branch, function)


def _sequence_schema_branch(
    value: object, versions: tuple[int, ...], alternatives: bool = False
) -> None:
    properties = _sequence_schema_object(
        value,
        ("ir_version", "input_dtype", "kernel", "execution", "state"),
        ("ir_version", "input_dtype", "kernel"),
        alternatives,
    )
    _sequence_schema_version(properties["ir_version"], versions)
    _sequence_schema_common_options(properties)
    functions = ("rolling_mean", "ewma")
    if versions[-1] >= 2:
        functions += ("rsi", "atr")
    if versions[-1] >= 3:
        functions += ("macd_components",)
    _sequence_schema_kernels(properties["kernel"], functions)


def _require_native_sequence_schema(text: str, versions: tuple[int, ...]) -> None:
    """Validate the advertised Rust schema against known metadata profiles."""
    schema = json.loads(text)
    _sequence_schema_branch(schema, versions, alternatives=len(versions) > 1)
    if len(versions) > 1:
        branches = _descriptor_list(schema["oneOf"])
        if len(branches) != len(versions):
            raise _sequence_schema_error("oneOf")
        for version, branch in enumerate(branches, 1):
            _sequence_schema_branch(branch, (version,))


def _require_native_factory_capability(capability: ProviderCapability) -> None:
    if (capability.provider, capability.version) != (
        "native",
        "1",
    ) or capability.name not in {"array_expr", "array_sequence", "table_array"}:
        raise ValueError("native array capability must describe a native array factory")
    if capability.name == "array_sequence":
        _require_native_sequence_capability(capability)
    else:
        _require_native_regular_capability(capability)


def _native_array_provider_capability(value: object) -> ProviderCapability:
    raw = _descriptor_object(
        value, frozenset(field.name for field in fields(ProviderCapability))
    )
    capability = ProviderCapability(
        **{
            **raw,
            "input_ports": tuple(
                _descriptor_port(port) for port in _descriptor_list(raw["input_ports"])
            ),
            "output_ports": tuple(
                _descriptor_port(port) for port in _descriptor_list(raw["output_ports"])
            ),
            "state_layouts": tuple(_descriptor_list(raw["state_layouts"])),
            "modes": tuple(_descriptor_list(raw["modes"])),
            "array_rules": _descriptor_array_rules(raw["array_rules"]),
        }
    )
    _require_native_factory_capability(capability)
    return capability


def _native_array_provider_capabilities(
    value: object,
) -> tuple[ProviderCapability, ...]:
    capabilities = tuple(
        _native_array_provider_capability(raw) for raw in _descriptor_list(value)
    )
    return _native_capability_bundle(capabilities)


def _native_capability_bundle(
    capabilities: tuple[ProviderCapability, ...],
) -> tuple[ProviderCapability, ...]:
    if sorted(capability.name for capability in capabilities) != [
        "array_expr",
        "array_sequence",
        "table_array",
    ]:
        raise ValueError(
            "native capability bundle requires array_expr, array_sequence "
            "and table_array exactly once"
        )
    return tuple(sorted(capabilities, key=lambda capability: capability.name))


def _native_record_capability(record: Mapping[str, Any]) -> ProviderCapability:
    capability = record.get("capability")
    if type(capability) is not ProviderCapability:
        raise TypeError("native record requires a frozen ProviderCapability")
    for field in ("provider", "name", "version"):
        _require_exact_str("native record", field, record.get(field))
        if record[field] != getattr(capability, field):
            raise ValueError(f"native record {field} contradicts its capability")
    _require_native_factory_capability(capability)
    return capability


def _require_native_registration_bundle(
    registrations: Sequence[Mapping[str, Any]],
) -> None:
    capabilities = tuple(
        _native_record_capability(record)
        for record in registrations
        if record["kind"] == "native_provider"
    )
    if capabilities:
        _native_capability_bundle(capabilities)


def runtime_capabilities(
    *,
    session_id: str,
    revision: int,
    package_version: str,
    registrations: Sequence[Mapping[str, Any]],
) -> RuntimeCapabilities:
    from calc_flow import _native

    _require_native_registration_bundle(registrations)

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

    def provider_capability(registration: Mapping[str, Any]) -> ProviderCapability:
        if registration["kind"] == "native_provider":
            return registration["capability"]
        lifecycle = registration.get("stream_lifecycle")
        if isinstance(lifecycle, _StatelessProviderLifecycle):
            modes: tuple[ExecutionMode, ...] = ("batch", "stream")
            finality: OutputFinality = "per_row_final"
            microbatch_invariant = True
            deterministic = lifecycle.deterministic
            replay_safe = lifecycle.replay_safe
            supports_static_inputs = lifecycle.supports_static_inputs
            partition_contract: PartitionContract = "row_axis_independent"
            array_rules: ProviderArrayRules | None = lifecycle.array_rules
        else:
            modes = ("batch",)
            finality = "unproven"
            microbatch_invariant = False
            deterministic = False
            replay_safe = False
            supports_static_inputs = False
            partition_contract = "none"
            array_rules = None
        return ProviderCapability(
            provider=str(registration["provider"]),
            name=str(registration["name"]),
            version=str(registration["version"]),
            input_ports=tuple(
                ProviderPort(str(name), kind, required=True)
                for name, kind in registration.get("input_ports", (("input", "array"),))
            ),
            output_ports=tuple(
                ProviderPort(str(name), kind, required=True)
                for name, kind in registration.get(
                    "output_ports", (("output", "array"),)
                )
            ),
            options_schema=registration.get("options_schema"),
            modes=modes,
            finality=finality,
            stateful=False,
            microbatch_invariant=microbatch_invariant,
            requires_watermark=False,
            checkpoint_support="stateless",
            state_version=None,
            deterministic=deterministic,
            replay_safe=replay_safe,
            supports_static_inputs=supports_static_inputs,
            partition_contract=partition_contract,
            array_rules=array_rules,
        )

    providers = tuple(
        sorted(
            (
                provider_capability(registration)
                for registration in registrations
                if registration["kind"] in {"provider", "native_provider"}
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
                kind="bounded_order",
                version="1",
                input_ports=(ProviderPort("input", "table", required=True),),
                output_ports=(ProviderPort("output", "table", required=True),),
                modes=("batch", "stream"),
                finality="group_final_append_only",
                requires_datafusion=True,
                stateful=True,
                microbatch_invariant=False,
                requires_watermark=False,
                checkpoint_support="checkpointed_stateful",
                state_version=1,
                state_layouts=(1, 2),
                deterministic=False,
                replay_safe=False,
            ),
            OperatorCapability(
                kind="bounded_over",
                version="1",
                input_ports=(ProviderPort("input", "table", required=True),),
                output_ports=(ProviderPort("output", "table", required=True),),
                modes=("batch", "stream"),
                finality="group_final_append_only",
                requires_datafusion=True,
                stateful=True,
                microbatch_invariant=True,
                requires_watermark=False,
                checkpoint_support="checkpointed_stateful",
                state_version=1,
                state_layouts=(1, 2),
                deterministic=True,
                replay_safe=True,
            ),
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
                state_layouts=(1,),
                deterministic=True,
                replay_safe=True,
            ),
            OperatorCapability(
                kind="distinct",
                version="1",
                input_ports=(ProviderPort("input", "table", required=True),),
                output_ports=(ProviderPort("output", "table", required=True),),
                modes=("batch", "stream"),
                finality="group_final_append_only",
                requires_datafusion=True,
                stateful=True,
                microbatch_invariant=True,
                requires_watermark=False,
                checkpoint_support="checkpointed_stateful",
                state_version=1,
                state_layouts=(2, 3),
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
                state_layouts=(1, 2, 3, 4, 5, 6),
                deterministic=True,
                replay_safe=True,
            ),
            OperatorCapability(
                kind="snapshot_sql",
                version="1",
                input_ports=(ProviderPort("input", "table", required=True),),
                output_ports=(ProviderPort("output", "table", required=True),),
                modes=("stream",),
                finality="unproven",
                requires_datafusion=True,
                stateful=True,
                microbatch_invariant=False,
                requires_watermark=False,
                checkpoint_support="checkpointed_stateful",
                state_version=1,
                state_layouts=(1,),
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
                stateful=True,
                microbatch_invariant=False,
                requires_watermark=False,
                checkpoint_support="checkpointed_stateful",
                state_version=2,
                state_layouts=(3, 4, 5, 6),
                deterministic=True,
                replay_safe=True,
            ),
            OperatorCapability(
                kind="stream_asof_join",
                version="1",
                input_ports=(
                    ProviderPort("left", "table", required=True),
                    ProviderPort("right", "table", required=True),
                ),
                output_ports=(ProviderPort("output", "table", required=True),),
                modes=("batch", "stream"),
                finality="group_final_append_only",
                requires_datafusion=True,
                stateful=True,
                microbatch_invariant=True,
                requires_watermark=True,
                checkpoint_support="checkpointed_stateful",
                state_version=3,
                state_layouts=(10,),
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
                modes=("batch", "stream"),
                finality="unproven",
                requires_datafusion=True,
                stateful=True,
                microbatch_invariant=False,
                requires_watermark=True,
                checkpoint_support="checkpointed_stateful",
                state_version=1,
                state_layouts=(1, 2, 3),
                deterministic=True,
                replay_safe=True,
            ),
            OperatorCapability(
                kind="window",
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
                state_layouts=(1, 2),
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

DeliveryCapabilityKind = TypeAliasType(
    "DeliveryCapabilityKind", Literal["best_effort", "at_least_once", "exactly_once"]
)
ReplayCapabilityKind = TypeAliasType(
    "ReplayCapabilityKind", Literal["replayable_exact", "unreplayable"]
)
WatermarkSupportKind = TypeAliasType(
    "WatermarkSupportKind", Literal["native", "generated_only"]
)
TransactionSupportKind = TypeAliasType(
    "TransactionSupportKind",
    Literal["none", "pre_commit_commit", "ledger_idempotent", "retry_deduplicated"],
)


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
