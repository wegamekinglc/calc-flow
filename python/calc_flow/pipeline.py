from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

from calc_flow import _native
from calc_flow.capabilities import (
    ProviderArrayRules,
    ProviderOptionsSchema,
    RuntimeCapabilities,
    _StatelessProviderLifecycle,
    runtime_capabilities,
)
from calc_flow.join_spec import (
    JoinSideWire,
    JoinStateLimits,
    JoinTimeBounds,
    bounds_wire,
    join_wire_spec,
    limits_wire,
    require_distinct_prefixes,
    require_equal_key_counts,
    require_event_time_columns,
    require_join_bounds,
    require_join_limits,
    timedelta_micros,
)
from calc_flow.store import _copy_json_value, _run_blocking

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
UdfReference = tuple[str, str, str]
_SYMBOLIC_COMPILE_CACHE_MAX_ENTRIES = 128


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _udf_documents(references: Sequence[UdfReference]) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for reference in references:
        if len(reference) != 3:
            raise ValueError("UDF references must be (provider, name, version) tuples")
        provider, name, version = reference
        if not all(isinstance(value, str) for value in reference):
            raise TypeError("UDF reference values must be strings")
        documents.append(
            {
                "kind": "data_fusion_scalar",
                "name": name,
                "provider": provider,
                "version": version,
            }
        )
    return documents


def _node_inputs(node: Mapping[str, Any]) -> tuple[str, ...]:
    configured = node.get("input_ports", [])
    if configured:
        return tuple(port["name"] for port in configured)
    operator = node["operator"]
    if operator["kind"] == "expression":
        return ("input",)
    if operator["kind"] == "sql":
        return tuple(operator["aliases"])
    return ()


def _data_sources(project: Mapping[str, Any]) -> list[dict[str, JSONValue]]:
    graph = project["graph"]
    connected = {
        (edge["target_node"], edge.get("target_port", "input"))
        for edge in graph["edges"]
    }
    endpoints = sorted(
        (node["id"], port)
        for node in graph["nodes"]
        for port in _node_inputs(node)
        if (node["id"], port) not in connected
    )
    counts: dict[str, int] = {}
    for _, port in endpoints:
        counts[port] = counts.get(port, 0) + 1
    names = sorted(
        port if counts[port] == 1 else f"{node_id}.{port}"
        for node_id, port in endpoints
    )
    return [
        {"data": [], "format": "inline_json", "id": f"source_{index}", "input": name}
        for index, name in enumerate(names, start=1)
    ]


def _updated_project(project_json: str, update: Any) -> str:
    project = json.loads(project_json)
    update(project)
    project["data_sources"] = _data_sources(project)
    return _canonical(project)


def _validate_stateless_lifecycle_values(values: Mapping[str, object]) -> None:
    for field_name, value in values.items():
        if type(value) is not bool:
            raise TypeError(
                f"{field_name} must be an exact bool; found {type(value).__name__}"
            )
    if not values["microbatch_invariant"]:
        raise ValueError("stateless stream providers must be micro-batch invariant")


def _registered_batch_provider(
    registrations: Sequence[dict[str, Any]],
    provider: str,
    name: str,
    version: str,
) -> dict[str, Any]:
    matches = [
        registration
        for registration in registrations
        if registration["kind"] == "provider"
        and registration["provider"] == provider
        and registration["name"] == name
        and registration["version"] == version
    ]
    if len(matches) != 1:
        raise ValueError(
            "stateless stream registration requires one existing batch "
            f"provider {provider}:{name}@{version}"
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class Runtime:
    _inner: _native.Runtime = field(default_factory=_native.Runtime, repr=False)
    _registration_lock: RLock = field(default_factory=RLock, repr=False, compare=False)
    _session_id: str = field(
        default_factory=lambda: str(uuid4()), repr=False, compare=False
    )
    _registrations: list[dict[str, Any]] = field(
        default_factory=list, repr=False, compare=False
    )
    _symbolic_compile_cache: dict[object, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    def _cached_symbolic_compile(
        self, key: object, factory: Callable[[], object], /
    ) -> object:
        """Return one immutable compiled plan for a deterministic cache key."""

        with self._registration_lock:
            cached = self._symbolic_compile_cache.get(key)
            if cached is not None:
                return cached
            compiled = factory()
            if len(self._symbolic_compile_cache) >= _SYMBOLIC_COMPILE_CACHE_MAX_ENTRIES:
                oldest = next(iter(self._symbolic_compile_cache))
                del self._symbolic_compile_cache[oldest]
            self._symbolic_compile_cache[key] = compiled
            return compiled

    def _invalidate_symbolic_compile_cache(self) -> None:
        self._symbolic_compile_cache.clear()

    def register_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: Any,
        *,
        options_schema: ProviderOptionsSchema | None = None,
        accepts_context: bool = False,
    ) -> None:
        if type(accepts_context) is not bool:
            raise TypeError(
                "accepts_context must be an exact bool; "
                f"found {type(accepts_context).__name__}"
            )
        if options_schema is not None and not isinstance(
            options_schema, ProviderOptionsSchema
        ):
            raise TypeError(
                "options_schema must be a ProviderOptionsSchema or None; "
                f"found {type(options_schema).__name__}"
            )
        with self._registration_lock:
            self._inner.register_provider(
                provider,
                name,
                version,
                callback,
                accepts_context=accepts_context,
            )
            registration = {
                "kind": "provider",
                "provider": provider,
                "name": name,
                "version": version,
                "callback": callback,
                "options_schema": options_schema,
            }
            if accepts_context:
                registration["accepts_context"] = True
            self._registrations.append(registration)
            self._invalidate_symbolic_compile_cache()

    def _register_mapping_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: Any,
        *,
        input_ports: Sequence[tuple[str, str]],
        output_ports: Sequence[tuple[str, str]],
        options_schema: ProviderOptionsSchema | None = None,
        accepts_context: bool = False,
    ) -> None:
        if type(accepts_context) is not bool:
            raise TypeError(
                "accepts_context must be an exact bool; "
                f"found {type(accepts_context).__name__}"
            )
        if options_schema is not None and not isinstance(
            options_schema, ProviderOptionsSchema
        ):
            raise TypeError(
                "options_schema must be a ProviderOptionsSchema or None; "
                f"found {type(options_schema).__name__}"
            )
        copied_inputs = tuple((port, kind) for port, kind in input_ports)
        copied_outputs = tuple((port, kind) for port, kind in output_ports)
        with self._registration_lock:
            self._inner._register_mapping_provider(
                provider,
                name,
                version,
                callback,
                input_ports=copied_inputs,
                output_ports=copied_outputs,
                accepts_context=accepts_context,
            )
            registration = {
                "kind": "provider",
                "provider_mode": "mapping",
                "provider": provider,
                "name": name,
                "version": version,
                "callback": callback,
                "input_ports": copied_inputs,
                "output_ports": copied_outputs,
                "options_schema": options_schema,
            }
            if accepts_context:
                registration["accepts_context"] = True
            self._registrations.append(registration)
            self._invalidate_symbolic_compile_cache()

    def _register_stateless_stream_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: Any,
        *,
        microbatch_invariant: bool,
        deterministic: bool,
        replay_safe: bool,
        supports_static_inputs: bool,
        array_rules: ProviderArrayRules,
    ) -> None:
        _validate_stateless_lifecycle_values(
            {
                "microbatch_invariant": microbatch_invariant,
                "deterministic": deterministic,
                "replay_safe": replay_safe,
                "supports_static_inputs": supports_static_inputs,
            }
        )
        if not isinstance(array_rules, ProviderArrayRules):
            raise TypeError(
                "array_rules must be a ProviderArrayRules; "
                f"found {type(array_rules).__name__}"
            )
        with self._registration_lock:
            registration = _registered_batch_provider(
                self._registrations, provider, name, version
            )
            self._inner._register_stateless_stream_provider(
                provider,
                name,
                version,
                callback,
                microbatch_invariant=microbatch_invariant,
                deterministic=deterministic,
                replay_safe=replay_safe,
            )
            registration["stream_lifecycle"] = _StatelessProviderLifecycle(
                deterministic=deterministic,
                replay_safe=replay_safe,
                supports_static_inputs=supports_static_inputs,
                array_rules=array_rules,
            )
            self._invalidate_symbolic_compile_cache()

    def _register_stateless_stream_mapping_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: Any,
        *,
        microbatch_invariant: bool,
        deterministic: bool,
        replay_safe: bool,
        supports_static_inputs: bool,
        array_rules: ProviderArrayRules,
    ) -> None:
        _validate_stateless_lifecycle_values(
            {
                "microbatch_invariant": microbatch_invariant,
                "deterministic": deterministic,
                "replay_safe": replay_safe,
                "supports_static_inputs": supports_static_inputs,
            }
        )
        if not isinstance(array_rules, ProviderArrayRules):
            raise TypeError(
                "array_rules must be a ProviderArrayRules; "
                f"found {type(array_rules).__name__}"
            )
        with self._registration_lock:
            registration = _registered_batch_provider(
                self._registrations, provider, name, version
            )
            if registration.get("provider_mode") != "mapping":
                raise ValueError(
                    "stateless stream mapping registration requires an existing "
                    f"mapping provider {provider}:{name}@{version}"
                )
            self._inner._register_stateless_stream_mapping_provider(
                provider,
                name,
                version,
                callback,
                input_ports=registration["input_ports"],
                output_ports=registration["output_ports"],
                microbatch_invariant=microbatch_invariant,
                deterministic=deterministic,
                replay_safe=replay_safe,
            )
            registration["stream_lifecycle"] = _StatelessProviderLifecycle(
                deterministic=deterministic,
                replay_safe=replay_safe,
                supports_static_inputs=supports_static_inputs,
                array_rules=array_rules,
            )
            self._invalidate_symbolic_compile_cache()

    def register_scalar_udf(
        self,
        *,
        provider: str,
        name: str,
        version: str,
        input_types: Sequence[str],
        return_type: str,
        volatility: str,
        function: Any,
    ) -> None:
        from calc_flow.udf import _validate_scalar_udf_registration

        copied_types = _validate_scalar_udf_registration(input_types, function)
        with self._registration_lock:
            self._inner.register_scalar_udf(
                provider=provider,
                name=name,
                version=version,
                input_types=copied_types,
                return_type=return_type,
                volatility=volatility,
                function=function,
            )
            self._registrations.append(
                {
                    "kind": "scalar_udf",
                    "provider": provider,
                    "name": name,
                    "version": version,
                    "input_types": tuple(copied_types),
                    "return_type": return_type,
                    "volatility": volatility,
                    "function": function,
                }
            )
            self._invalidate_symbolic_compile_cache()

    def _copied_registrations(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                **registration,
                **(
                    {"input_types": tuple(registration["input_types"])}
                    if registration["kind"] == "scalar_udf"
                    else (
                        {
                            "input_ports": tuple(registration["input_ports"]),
                            "output_ports": tuple(registration["output_ports"]),
                        }
                        if registration.get("provider_mode") == "mapping"
                        else {}
                    )
                ),
            }
            for registration in self._registrations
        )

    def _registration_snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return successful trusted registrations as defensive plain records."""
        with self._registration_lock:
            return self._copied_registrations()

    def catalog(self) -> list[dict[str, Any]]:
        return self._inner.catalog()

    def capabilities(self) -> RuntimeCapabilities:
        snapshot, _ = self._capability_registration_snapshot()
        return snapshot

    def _capability_registration_snapshot(
        self,
    ) -> tuple[RuntimeCapabilities, tuple[dict[str, Any], ...]]:
        """Capture safe metadata and private worker records at one revision."""
        with self._registration_lock:
            registrations = self._copied_registrations()
            snapshot = runtime_capabilities(
                session_id=self._session_id,
                revision=len(registrations),
                package_version=_native.version(),
                registrations=registrations,
            )
            return snapshot, registrations

    def validation_report(self, project_json: str) -> dict[str, Any]:
        if not isinstance(project_json, str):
            raise TypeError("project_json must be a string")
        return self._inner.validation_report(project_json)

    def compile_project(self, project_json: str) -> BatchExecutionPlan:
        if not isinstance(project_json, str):
            raise TypeError("project_json must be a string")
        return BatchExecutionPlan(self._inner.compile_project(project_json))

    def compile_batch_project(self, project_json: str) -> BatchExecutionPlan:
        """Compile one canonical project-v3 batch document."""
        return self.compile_project(project_json)

    def compile_stream_project(
        self,
        project_json: str,
        *,
        requirements: StreamRequirements | None = None,
    ) -> StreamExecutionPlan:
        """Compile project JSON into a continuous plan owned by this runtime."""
        if not isinstance(project_json, str):
            raise TypeError("project_json must be a string")
        selected = StreamRequirements() if requirements is None else requirements
        if not isinstance(selected, StreamRequirements):
            raise TypeError(
                "requirements must be a calc_flow.StreamRequirements or None"
            )
        delivery = {
            output: guarantee.value for output, guarantee in selected.delivery.items()
        }
        canonical = _native.validate_project_json(project_json)
        project = json.loads(canonical)
        runtime_options = project["runtime"]["options"]
        state = project["state"]
        return StreamExecutionPlan(
            self._inner.compile_stream_project(canonical, delivery),
            _ProjectStreamSettings(
                state_root=state["root"],
                retained_epochs=state["retention"],
                checkpoint_interval_ms=runtime_options["checkpoint_interval_ms"],
                max_batch_rows=runtime_options["max_batch_rows"],
                max_batch_bytes=runtime_options["max_batch_bytes"],
            ),
        )

    def _compile_stream_graph_project(
        self,
        project_json: str,
        *,
        requirements: StreamRequirements | None = None,
    ) -> StreamExecutionPlan:
        selected = StreamRequirements() if requirements is None else requirements
        delivery = {
            output: guarantee.value for output, guarantee in selected.delivery.items()
        }
        return StreamExecutionPlan(
            self._inner._compile_stream_graph_project(project_json, delivery)
        )


@dataclass(frozen=True, slots=True)
class BatchExecutionPlan:
    _inner: _native.ExecutionPlan = field(repr=False)

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    def execute(
        self,
        inputs: Mapping[str, _native.Batch],
        *,
        options: _native.ExecutionOptions | None = None,
    ) -> _native.RunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "execute() cannot run inside an event loop; use execute_async()"
            )
        if options is not None and type(options) is not _native.ExecutionOptions:
            raise TypeError("options must be a calc_flow.ExecutionOptions or None")
        copied = dict(inputs)
        return self._inner.execute(copied, options=options)

    def execute_async(
        self,
        inputs: Mapping[str, _native.Batch],
        *,
        options: _native.ExecutionOptions | None = None,
    ) -> Awaitable[_native.RunResult]:
        if options is not None and type(options) is not _native.ExecutionOptions:
            raise TypeError("options must be a calc_flow.ExecutionOptions or None")
        copied = dict(inputs)

        async def execute() -> _native.RunResult:
            native, cancellation = self._inner._execute_async_cancellable(
                copied, options=options
            )
            try:
                return await asyncio.shield(native)
            except asyncio.CancelledError as cancelled:
                native_completed = native.done()
                if native_completed:
                    return native.result()
                cancellation.cancel()
                # Cancellation precedence: once the handler passes the
                # terminal-at-entry check, the caller's CancelledError wins
                # over any native outcome observed during the drain. A native
                # failure landing mid-drain is retrieved below (so asyncio
                # never reports it as unretrieved) and then discarded.
                while not native.done():
                    try:
                        await asyncio.shield(native)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if native.done() and not native.cancelled():
                    native.exception()
                raise cancelled

        return execute()

    async def snapshot_async(self) -> dict[str, Any]:
        state = await self._inner.snapshot_async()
        return _copy_json_value(state, root_mapping=True, label="plan state")

    def restore_async(self, state: Mapping[str, object]) -> Awaitable[None]:
        copied = _copy_json_value(dict(state), root_mapping=True, label="plan state")
        encoded = json.dumps(copied, separators=(",", ":"), sort_keys=True)

        async def restore() -> None:
            await self._inner.restore_async(encoded)

        return restore()

    async def reset_async(self) -> None:
        await self._inner.reset_async()

    def snapshot(self) -> dict[str, Any]:
        return _run_blocking(self.snapshot_async, "snapshot_async")

    def restore(self, state: Mapping[str, object]) -> None:
        return _run_blocking(lambda: self.restore_async(state), "restore_async")

    def reset(self) -> None:
        return _run_blocking(self.reset_async, "reset_async")


ExecutionPlan = BatchExecutionPlan


class DeliveryGuarantee(StrEnum):
    """Delivery guarantee requested for one external stream output."""

    BEST_EFFORT = "best_effort"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass(frozen=True, slots=True)
class StreamRequirements:
    """Immutable per-output delivery requirements for stream compilation."""

    delivery: Mapping[str, DeliveryGuarantee] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.delivery, Mapping):
            raise TypeError(
                "delivery must be a mapping of output names to DeliveryGuarantee values"
            )
        copied = dict(self.delivery)
        for output, guarantee in copied.items():
            if not isinstance(output, str) or not output:
                raise TypeError("delivery output names must be non-empty strings")
            if not isinstance(guarantee, DeliveryGuarantee):
                raise TypeError(
                    "delivery guarantees must be calc_flow.DeliveryGuarantee values"
                )
        object.__setattr__(self, "delivery", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class ArrowFieldSpec:
    """One exact project-v3 Arrow field."""

    name: str
    data_type: str
    nullable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("name must be a non-empty string")
        if not isinstance(self.data_type, str) or not self.data_type:
            raise TypeError("data_type must be a non-empty string")
        if type(self.nullable) is not bool:
            raise TypeError("nullable must be an exact bool")


def _arrow_fields(
    values: Sequence[ArrowFieldSpec], field_name: str
) -> list[dict[str, object]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of ArrowFieldSpec values")
    copied = tuple(values)
    for index, value in enumerate(copied):
        if not isinstance(value, ArrowFieldSpec):
            raise TypeError(
                f"{field_name} must contain only calc_flow.ArrowFieldSpec values; "
                f"found {type(value).__name__} at index {index}"
            )
    if not copied:
        raise ValueError(f"{field_name} must contain at least one field")
    return [
        {
            "name": value.name,
            "data_type": value.data_type,
            "nullable": value.nullable,
        }
        for value in copied
    ]


def _join_keys(values: Sequence[str], field_name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of column names")
    copied = list(values)
    if not copied or not all(isinstance(value, str) and value for value in copied):
        raise ValueError(f"{field_name} must contain non-empty column names")
    if len(set(copied)) != len(copied):
        raise ValueError(f"{field_name} must contain unique column names")
    return copied


@dataclass(frozen=True, slots=True)
class _ProjectStreamSettings:
    state_root: str
    retained_epochs: int
    checkpoint_interval_ms: int
    max_batch_rows: int
    max_batch_bytes: int


@dataclass(frozen=True, slots=True)
class StreamExecutionPlan:
    """Compiled immutable continuous plan consumed by ``StreamingRunner``."""

    _inner: _native.StreamExecutionPlan = field(repr=False)
    _project_settings: _ProjectStreamSettings | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    @property
    def requirements(self) -> StreamRequirements:
        return StreamRequirements(
            {
                output: DeliveryGuarantee(value)
                for output, value in self._inner.requirements.items()
            }
        )

    @property
    def source_binding_ids(self) -> tuple[str, ...]:
        return self._inner.source_binding_ids

    @property
    def static_input_ids(self) -> tuple[str, ...]:
        return self._inner.static_input_ids

    @property
    def sink_binding_ids(self) -> tuple[str, ...]:
        return self._inner.sink_binding_ids


@dataclass(frozen=True, slots=True, init=False)
class PipelineBuilder:
    _project_json: str = field(repr=False)

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("pipeline name must be a string")
        project = {
            "data_sources": [],
            "format_version": 3,
            "id": name,
            "name": name,
            "runtime": {"mode": "batch", "options": {}},
            "graph": {"edges": [], "name": name, "nodes": []},
        }
        object.__setattr__(self, "_project_json", _canonical(project))

    @classmethod
    def _from_json(cls, project_json: str) -> PipelineBuilder:
        builder = object.__new__(cls)
        object.__setattr__(builder, "_project_json", project_json)
        return builder

    @property
    def project(self) -> dict[str, Any]:
        return json.loads(self._project_json)

    def with_datafusion_config(
        self,
        *,
        batch_size: int = 8_192,
        target_partitions: int = 1,
        parallelism_mode: Literal["fixed", "auto"] = "fixed",
        max_partitions: int = 32,
        min_rows_per_partition: int = 65_536,
        small_rows_threshold: int = 10_001,
        enable_rolling_rewrite: bool = True,
        collect_diagnostics: bool = True,
    ) -> PipelineBuilder:
        """Return a builder with an immutable run-scoped DataFusion policy.

        ``auto`` is opt-in. It uses trusted
        ``calc_flow.datafusion.active_entities`` batch metadata and safely
        falls back to one partition when that statistic is absent or invalid.
        """
        integers = {
            "batch_size": batch_size,
            "target_partitions": target_partitions,
            "max_partitions": max_partitions,
            "min_rows_per_partition": min_rows_per_partition,
            "small_rows_threshold": small_rows_threshold,
        }
        for field_name, value in integers.items():
            if type(value) is not int:
                raise TypeError(f"{field_name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if type(parallelism_mode) is not str:
            raise TypeError("parallelism_mode must be fixed or auto")
        if parallelism_mode not in {"fixed", "auto"}:
            raise ValueError("parallelism_mode must be fixed or auto")
        for field_name, value in {
            "enable_rolling_rewrite": enable_rolling_rewrite,
            "collect_diagnostics": collect_diagnostics,
        }.items():
            if type(value) is not bool:
                raise TypeError(f"{field_name} must be an exact bool")

        def update(project: dict[str, Any]) -> None:
            project["graph"]["datafusion"] = {
                **integers,
                "parallelism_mode": parallelism_mode,
                "enable_rolling_rewrite": enable_rolling_rewrite,
                "collect_diagnostics": collect_diagnostics,
            }

        return self._from_json(_updated_project(self._project_json, update))

    def expression(
        self,
        name: str,
        expression: str,
        *,
        select: Sequence[str] = (),
        filter: str | None = None,
        udfs: Sequence[UdfReference] = (),
    ) -> PipelineBuilder:
        def add(project: dict[str, Any]) -> None:
            project["graph"]["nodes"].append(
                {
                    "id": name,
                    "operator": {
                        "expression": expression,
                        "filter": filter,
                        "kind": "expression",
                        "select": list(select),
                        "udfs": _udf_documents(tuple(udfs)),
                    },
                }
            )

        return self._from_json(_updated_project(self._project_json, add))

    def sql(
        self,
        name: str,
        query: str,
        *,
        aliases: Sequence[str] = ("input",),
        udfs: Sequence[UdfReference] = (),
    ) -> PipelineBuilder:
        def add(project: dict[str, Any]) -> None:
            project["graph"]["nodes"].append(
                {
                    "id": name,
                    "operator": {
                        "aliases": list(aliases),
                        "kind": "sql",
                        "query": query,
                        "udfs": _udf_documents(tuple(udfs)),
                    },
                }
            )

        return self._from_json(_updated_project(self._project_json, add))

    def external(
        self,
        node_id: str,
        provider: str,
        name: str,
        version: str,
        options: Mapping[str, object],
    ) -> PipelineBuilder:
        copied_options = dict(options)

        def add(project: dict[str, Any]) -> None:
            project["graph"]["nodes"].append(
                {
                    "id": node_id,
                    "input_ports": [
                        {
                            "kind": "array",
                            "name": "input",
                            "required": True,
                            "schema": [],
                        }
                    ],
                    "operator": {
                        "kind": "external",
                        "name": name,
                        "options": copied_options,
                        "provider": provider,
                        "version": version,
                    },
                    "output_ports": [
                        {
                            "kind": "array",
                            "name": "output",
                            "required": True,
                            "schema": [],
                        }
                    ],
                }
            )

        return self._from_json(_updated_project(self._project_json, add))

    def table_matmul(
        self,
        node_id: str,
        *,
        backend: Literal["numpy", "jax"],
        columns: Sequence[str],
    ) -> PipelineBuilder:
        if backend not in {"numpy", "jax"}:
            raise ValueError("backend must be 'numpy' or 'jax'")
        if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
            raise TypeError("columns must be a sequence of column names")
        copied_columns = list(columns)
        if not copied_columns:
            raise ValueError("columns must contain at least one column name")
        if not all(isinstance(column, str) and column for column in copied_columns):
            raise TypeError("columns must contain non-empty strings")
        if len(set(copied_columns)) != len(copied_columns):
            raise ValueError("columns must be unique")

        def add(project: dict[str, Any]) -> None:
            project["graph"]["nodes"].append(
                {
                    "id": node_id,
                    "input_ports": [
                        {
                            "kind": "table",
                            "name": "table",
                            "required": True,
                            "schema": [],
                        },
                        {
                            "kind": "array",
                            "name": "weights",
                            "required": True,
                            "schema": [],
                        },
                    ],
                    "operator": {
                        "kind": "external",
                        "name": "table_matmul",
                        "options": {"columns": copied_columns},
                        "provider": backend,
                        "version": "1",
                    },
                    "output_ports": [
                        {
                            "kind": "array",
                            "name": "output",
                            "required": True,
                            "schema": [],
                        }
                    ],
                }
            )

        return self._from_json(_updated_project(self._project_json, add))

    def stream_join(
        self,
        name: str,
        *,
        left_schema: Sequence[ArrowFieldSpec],
        right_schema: Sequence[ArrowFieldSpec],
        left_keys: Sequence[str],
        right_keys: Sequence[str],
        left_event_time: str,
        right_event_time: str,
        bounds: JoinTimeBounds,
        limits: JoinStateLimits,
        left_prefix: str = "left",
        right_prefix: str = "right",
    ) -> PipelineBuilder:
        """Return a new builder containing one bounded inner stream Join."""
        if not isinstance(name, str) or not name:
            raise TypeError("name must be a non-empty string")
        copied_left_schema = _arrow_fields(left_schema, "left_schema")
        copied_right_schema = _arrow_fields(right_schema, "right_schema")
        copied_left_keys = _join_keys(left_keys, "left_keys")
        copied_right_keys = _join_keys(right_keys, "right_keys")
        require_equal_key_counts(copied_left_keys, copied_right_keys)
        require_event_time_columns(left_event_time, right_event_time)
        require_join_bounds(bounds)
        require_join_limits(limits)
        require_distinct_prefixes(left_prefix, right_prefix)

        def add(project: dict[str, Any]) -> None:
            project["graph"]["nodes"].append(
                {
                    "id": name,
                    "input_ports": [
                        {
                            "kind": "table",
                            "name": "left",
                            "required": True,
                            "schema": copied_left_schema,
                        },
                        {
                            "kind": "table",
                            "name": "right",
                            "required": True,
                            "schema": copied_right_schema,
                        },
                    ],
                    "operator": {
                        "kind": "stream_join",
                        "spec": join_wire_spec(
                            JoinSideWire(
                                keys=tuple(copied_left_keys),
                                event_time=left_event_time,
                                prefix=left_prefix,
                            ),
                            JoinSideWire(
                                keys=tuple(copied_right_keys),
                                event_time=right_event_time,
                                prefix=right_prefix,
                            ),
                            bounds_wire(
                                timedelta_micros(bounds.before, "before"),
                                timedelta_micros(bounds.after, "after"),
                            ),
                            limits_wire(
                                limits.max_state_rows_per_side,
                                limits.max_state_bytes_per_side,
                                limits.max_matches_per_input_batch,
                            ),
                        ),
                    },
                    "output_ports": [],
                }
            )

        return self._from_json(_updated_project(self._project_json, add))

    def connect(
        self,
        source_node: str,
        target_node: str,
        *,
        source_port: str = "output",
        target_port: str = "input",
    ) -> PipelineBuilder:
        def add(project: dict[str, Any]) -> None:
            project["graph"]["edges"].append(
                {
                    "source_node": source_node,
                    "source_port": source_port,
                    "target_node": target_node,
                    "target_port": target_port,
                }
            )

        return self._from_json(_updated_project(self._project_json, add))

    def compile_batch(self, runtime: Runtime | None = None) -> BatchExecutionPlan:
        from calc_flow.array import _validate_provider_options

        for node in self.project["graph"]["nodes"]:
            operator = node["operator"]
            if operator["kind"] == "external":
                _validate_provider_options(
                    operator["provider"],
                    operator["name"],
                    operator["version"],
                    operator["options"],
                )
        selected = Runtime() if runtime is None else runtime
        if not isinstance(selected, Runtime):
            raise TypeError("runtime must be a calc_flow.Runtime")
        return selected.compile_batch_project(self._project_json)

    def compile_stream(
        self,
        *,
        requirements: StreamRequirements | None = None,
        runtime: Runtime | None = None,
    ) -> StreamExecutionPlan:
        """Compile a continuous plan with explicit delivery requirements."""
        selected = Runtime() if runtime is None else runtime
        if not isinstance(selected, Runtime):
            raise TypeError("runtime must be a calc_flow.Runtime")
        project = self.project
        project["runtime"] = {"mode": "stream", "options": {}}
        project["data_sources"] = []
        return selected._compile_stream_graph_project(
            _canonical(project), requirements=requirements
        )


def compile_stream_project(
    project: Mapping[str, object],
    *,
    requirements: StreamRequirements | None = None,
    runtime: Runtime | None = None,
) -> StreamExecutionPlan:
    """Defensively compile one connector-backed project-v3 document."""
    from calc_flow.config import ProjectDocument

    if not isinstance(project, Mapping):
        raise TypeError("project must be a mapping")
    document = ProjectDocument.model_validate(dict(project))
    selected = Runtime() if runtime is None else runtime
    if not isinstance(selected, Runtime):
        raise TypeError("runtime must be a calc_flow.Runtime")
    return selected.compile_stream_project(
        document.canonical_json(), requirements=requirements
    )


def project_json_schema() -> str:
    return _native.project_json_schema()


def validate_project_json(project_json: str) -> str:
    return _native.validate_project_json(project_json)
