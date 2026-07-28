from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from calc_flow import _native
from calc_flow.capabilities import (
    ProviderOptionsSchema,
    RuntimeCapabilities,
    runtime_capabilities,
)
from calc_flow.store import _copy_json_value, _run_blocking

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
UdfReference = tuple[str, str, str]


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
    pipeline = project["pipeline"]
    connected = {
        (edge["target_node"], edge.get("target_port", "input"))
        for edge in pipeline["edges"]
    }
    endpoints = sorted(
        (node["id"], port)
        for node in pipeline["nodes"]
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

    def compile_project(self, project_json: str) -> ExecutionPlan:
        if not isinstance(project_json, str):
            raise TypeError("project_json must be a string")
        return ExecutionPlan(self._inner.compile_project(project_json))


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
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
                while not native.done():
                    try:
                        await asyncio.shield(native)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
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


@dataclass(frozen=True, slots=True, init=False)
class PipelineBuilder:
    _project_json: str = field(repr=False)

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("pipeline name must be a string")
        project = {
            "data_sources": [],
            "format_version": 2,
            "id": name,
            "name": name,
            "pipeline": {"edges": [], "name": name, "nodes": []},
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
            project["pipeline"]["nodes"].append(
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
            project["pipeline"]["nodes"].append(
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
            project["pipeline"]["nodes"].append(
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
            project["pipeline"]["nodes"].append(
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

    def connect(
        self,
        source_node: str,
        target_node: str,
        *,
        source_port: str = "output",
        target_port: str = "input",
    ) -> PipelineBuilder:
        def add(project: dict[str, Any]) -> None:
            project["pipeline"]["edges"].append(
                {
                    "source_node": source_node,
                    "source_port": source_port,
                    "target_node": target_node,
                    "target_port": target_port,
                }
            )

        return self._from_json(_updated_project(self._project_json, add))

    def compile(self, runtime: Runtime | None = None) -> ExecutionPlan:
        from calc_flow.array import _validate_provider_options

        for node in self.project["pipeline"]["nodes"]:
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
        return selected.compile_project(self._project_json)


def project_json_schema() -> str:
    return _native.project_json_schema()


def validate_project_json(project_json: str) -> str:
    return _native.validate_project_json(project_json)
