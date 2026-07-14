from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from calc_flow import _native

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

    def register_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: Any,
    ) -> None:
        self._inner.register_provider(provider, name, version, callback)

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
        self._inner.register_scalar_udf(
            provider=provider,
            name=name,
            version=version,
            input_types=copied_types,
            return_type=return_type,
            volatility=volatility,
            function=function,
        )

    def catalog(self) -> list[dict[str, Any]]:
        return self._inner.catalog()

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

    def execute(self, inputs: Mapping[str, _native.Batch]) -> _native.RunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "execute() cannot run inside an event loop; use execute_async()"
            )
        return self._inner.execute(dict(inputs))

    def execute_async(
        self, inputs: Mapping[str, _native.Batch]
    ) -> Awaitable[_native.RunResult]:
        copied = dict(inputs)

        async def execute() -> _native.RunResult:
            return await self._inner.execute_async(copied)

        return execute()


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
