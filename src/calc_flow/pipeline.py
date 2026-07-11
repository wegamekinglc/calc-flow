from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any

from calc_flow.batch import Batch
from calc_flow.context import CancellationToken, RunContext
from calc_flow.engine.datafusion import (
    DataFusionConfig,
    DataFusionQueryMetrics,
    DataFusionRuntime,
)
from calc_flow.operator import Operator, Port
from calc_flow.udf import UdfRegistry, UdfRegistrySnapshot


@dataclass(frozen=True, slots=True)
class PortEndpoint:
    node_id: str
    port: str

    def __str__(self) -> str:
        return f"{self.node_id}.{self.port}"


@dataclass(frozen=True, slots=True)
class Edge:
    source: PortEndpoint
    target: PortEndpoint


@dataclass(frozen=True, slots=True)
class CompiledNode:
    node_id: str
    operator: Operator


@dataclass(frozen=True, slots=True)
class NodeTiming:
    node_id: str
    duration_ns: int
    input_rows: Mapping[str, int]
    output_rows: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    pipeline_name: str
    pipeline_fingerprint: str
    started_at: datetime
    finished_at: datetime


@dataclass(frozen=True, slots=True)
class RunResult:
    """Named graph outputs and observations from one execution run."""

    outputs: Mapping[str, Batch]
    warnings: tuple[str, ...]
    node_timings: Mapping[str, NodeTiming]
    datafusion_metrics: tuple[DataFusionQueryMetrics, ...]
    metadata: RunMetadata

    @property
    def output(self) -> Batch:
        """Return the only graph output, rejecting ambiguous multi-output runs."""
        if len(self.outputs) != 1:
            msg = "RunResult.output requires exactly one graph output"
            raise ValueError(msg)
        return next(iter(self.outputs.values()))


def _port_map(ports: tuple[Port, ...]) -> dict[str, Port]:
    return {port.name: port for port in ports}


def _external_names(endpoints: list[PortEndpoint]) -> dict[str, PortEndpoint]:
    counts = Counter(endpoint.port for endpoint in endpoints)
    return {
        endpoint.port if counts[endpoint.port] == 1 else str(endpoint): endpoint
        for endpoint in endpoints
    }


def _pipeline_fingerprint(
    name: str,
    nodes: Mapping[str, Operator],
    edges: tuple[Edge, ...],
    datafusion_config: DataFusionConfig,
    udfs: UdfRegistrySnapshot,
) -> str:
    definition = {
        "name": name,
        "nodes": [
            {"node_id": node_id, "operator": nodes[node_id].fingerprint_data()}
            for node_id in sorted(nodes)
        ],
        "edges": [
            {
                "source": str(edge.source),
                "target": str(edge.target),
            }
            for edge in sorted(
                edges,
                key=lambda edge: (
                    edge.source.node_id,
                    edge.source.port,
                    edge.target.node_id,
                    edge.target.port,
                ),
            )
        ],
        "datafusion": asdict(datafusion_config),
        "udfs": udfs.catalog(),
    }
    try:
        encoded = json.dumps(
            definition, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        msg = "operator fingerprint configuration must be JSON-compatible"
        raise TypeError(msg) from error
    return hashlib.sha256(encoded).hexdigest()


class ExecutionPlan:
    """An immutable, validated graph ready for repeated execution."""

    __slots__ = (
        "_frozen",
        "_incoming",
        "_input_names",
        "_node_map",
        "datafusion_config",
        "edges",
        "fingerprint",
        "graph_inputs",
        "graph_outputs",
        "name",
        "nodes",
        "udfs",
    )

    def __init__(
        self,
        *,
        name: str,
        nodes: tuple[CompiledNode, ...],
        edges: tuple[Edge, ...],
        graph_inputs: Mapping[str, PortEndpoint],
        graph_outputs: Mapping[str, PortEndpoint],
        datafusion_config: DataFusionConfig,
        udfs: UdfRegistrySnapshot,
        fingerprint: str,
    ) -> None:
        object.__setattr__(self, "_frozen", False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "graph_inputs", MappingProxyType(dict(graph_inputs)))
        object.__setattr__(self, "graph_outputs", MappingProxyType(dict(graph_outputs)))
        object.__setattr__(self, "datafusion_config", datafusion_config)
        object.__setattr__(self, "udfs", udfs)
        object.__setattr__(self, "fingerprint", fingerprint)

        object.__setattr__(
            self,
            "_node_map",
            MappingProxyType({node.node_id: node.operator for node in nodes}),
        )
        incoming: dict[PortEndpoint, PortEndpoint] = {}
        for edge in edges:
            incoming[edge.target] = edge.source
        object.__setattr__(self, "_incoming", MappingProxyType(incoming))
        object.__setattr__(
            self,
            "_input_names",
            MappingProxyType(
                {endpoint: name for name, endpoint in graph_inputs.items()}
            ),
        )
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            msg = "ExecutionPlan is immutable"
            raise AttributeError(msg)
        object.__setattr__(self, name, value)

    def execute(
        self,
        inputs: Mapping[str, Batch],
        *,
        cancellation: CancellationToken | None = None,
        deadline: datetime | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> RunResult:
        """Execute the graph once with formed batches at its external ports."""
        if deadline is not None and deadline.tzinfo is None:
            msg = "deadline must include timezone information"
            raise ValueError(msg)
        unknown = set(inputs) - set(self.graph_inputs)
        if unknown:
            msg = f"unknown graph inputs: {sorted(unknown)}"
            raise ValueError(msg)

        values: dict[PortEndpoint, Batch] = {}
        for name, endpoint in self.graph_inputs.items():
            operator = self._node_map[endpoint.node_id]
            port = _port_map(operator.input_ports)[endpoint.port]
            if name not in inputs:
                if port.required:
                    msg = f"missing required graph input {name!r}"
                    raise ValueError(msg)
                continue
            batch = inputs[name]
            port.validate(batch, endpoint=f"graph input {name!r}")
            values[endpoint] = batch

        state_before = self.snapshot()
        started_at = datetime.now(UTC)
        runtime = DataFusionRuntime(
            self.datafusion_config, udfs=self.udfs.datafusion_specs
        )
        context = RunContext.create(
            runtime,
            udfs=self.udfs,
            cancellation=cancellation,
            deadline=deadline,
            settings=dict(settings or {}),
        )
        timings: dict[str, NodeTiming] = {}

        try:
            context.check_cancelled()
            for node in self.nodes:
                node_context = context.for_node(node.node_id)
                node_context.check_cancelled()
                operator_inputs: dict[str, Batch] = {}
                for port in node.operator.input_ports:
                    target = PortEndpoint(node.node_id, port.name)
                    source = self._incoming.get(target)
                    if source is not None:
                        if source not in values:
                            if port.required:
                                msg = (
                                    f"required input {target} did not receive "
                                    f"optional output {source}"
                                )
                                raise ValueError(msg)
                            continue
                        batch = values[source]
                    elif target in values:
                        batch = values[target]
                    elif port.required:
                        name = self._input_names.get(target, str(target))
                        msg = f"missing required graph input {name!r}"
                        raise ValueError(msg)
                    else:
                        continue
                    port.validate(batch, endpoint=f"input {target}")
                    operator_inputs[port.name] = batch

                node_started = perf_counter_ns()
                operator_outputs = node.operator.process(operator_inputs, node_context)
                duration_ns = perf_counter_ns() - node_started
                if not isinstance(operator_outputs, Mapping):
                    msg = f"operator {node.node_id!r} must return a mapping"
                    raise TypeError(msg)

                output_ports = _port_map(node.operator.output_ports)
                unknown_outputs = set(operator_outputs) - set(output_ports)
                if unknown_outputs:
                    msg = (
                        f"operator {node.node_id!r} returned unknown outputs: "
                        f"{sorted(unknown_outputs)}"
                    )
                    raise ValueError(msg)
                missing_outputs = {
                    port.name
                    for port in node.operator.output_ports
                    if port.required and port.name not in operator_outputs
                }
                if missing_outputs:
                    msg = (
                        f"operator {node.node_id!r} omitted required outputs: "
                        f"{sorted(missing_outputs)}"
                    )
                    raise ValueError(msg)

                for port_name, batch in operator_outputs.items():
                    endpoint = PortEndpoint(node.node_id, port_name)
                    output_ports[port_name].validate(
                        batch, endpoint=f"output {endpoint}"
                    )
                    values[endpoint] = batch

                timings[node.node_id] = NodeTiming(
                    node_id=node.node_id,
                    duration_ns=duration_ns,
                    input_rows=MappingProxyType(
                        {
                            name: batch.num_rows
                            for name, batch in operator_inputs.items()
                        }
                    ),
                    output_rows=MappingProxyType(
                        {
                            name: batch.num_rows
                            for name, batch in operator_outputs.items()
                        }
                    ),
                )
                node_context.check_cancelled()

            outputs = {
                name: values[endpoint]
                for name, endpoint in self.graph_outputs.items()
                if endpoint in values
            }
            finished_at = datetime.now(UTC)
            return RunResult(
                outputs=MappingProxyType(outputs),
                warnings=(),
                node_timings=MappingProxyType(timings),
                datafusion_metrics=runtime.metrics,
                metadata=RunMetadata(
                    run_id=context.run_id,
                    pipeline_name=self.name,
                    pipeline_fingerprint=self.fingerprint,
                    started_at=started_at,
                    finished_at=finished_at,
                ),
            )
        except BaseException:
            self.restore(state_before)
            raise
        finally:
            runtime.close()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            node.node_id: deepcopy(state)
            for node in self.nodes
            if (state := node.operator.snapshot())
        }

    def restore(self, checkpoint: Mapping[str, dict[str, Any]]) -> None:
        unknown = set(checkpoint) - set(self._node_map)
        if unknown:
            msg = f"checkpoint contains unknown nodes: {sorted(unknown)}"
            raise ValueError(msg)
        for node in self.nodes:
            if node.node_id in checkpoint:
                node.operator.restore(deepcopy(dict(checkpoint[node.node_id])))
            else:
                node.operator.reset()

    def reset(self) -> None:
        for node in self.nodes:
            node.operator.reset()


class Pipeline:
    """Mutable graph builder that compiles into an immutable execution plan."""

    def __init__(
        self,
        name: str = "pipeline",
        *,
        datafusion_config: DataFusionConfig | None = None,
        udf_registry: UdfRegistry | UdfRegistrySnapshot | None = None,
    ) -> None:
        if not name:
            msg = "pipeline name must not be empty"
            raise ValueError(msg)
        self.name = name
        self.datafusion_config = datafusion_config or DataFusionConfig()
        self.udf_registry = udf_registry or UdfRegistrySnapshot()
        self._nodes: dict[str, Operator] = {}
        self._edges: list[Edge] = []
        self._linear_tail: str | None = None

    def add_node(self, node_id: str, operator: Operator) -> Pipeline:
        if not node_id:
            msg = "node ID must not be empty"
            raise ValueError(msg)
        if node_id in self._nodes:
            msg = f"node ID {node_id!r} is already in pipeline {self.name!r}"
            raise ValueError(msg)
        self._nodes[node_id] = operator
        self._linear_tail = node_id
        return self

    def connect(
        self,
        source_node: str,
        target_node: str,
        *,
        source_port: str = "output",
        target_port: str = "input",
    ) -> Pipeline:
        self._edges.append(
            Edge(
                PortEndpoint(source_node, source_port),
                PortEndpoint(target_node, target_port),
            )
        )
        return self

    def then(self, operator: Operator, *, node_id: str | None = None) -> Pipeline:
        previous = self._linear_tail
        resolved_id = node_id or operator.name
        self.add_node(resolved_id, operator)
        if previous is not None:
            source = self._nodes[previous]
            if len(source.output_ports) != 1 or len(operator.input_ports) != 1:
                msg = "linear sugar requires one input and one output port"
                self._nodes.pop(resolved_id)
                self._linear_tail = previous
                raise ValueError(msg)
            self.connect(
                previous,
                resolved_id,
                source_port=source.output_ports[0].name,
                target_port=operator.input_ports[0].name,
            )
        return self

    def add(self, operator: Operator) -> Pipeline:
        """Add an operator using its name as node ID and connect it linearly."""
        return self.then(operator)

    def compile(self) -> ExecutionPlan:
        if not self._nodes:
            msg = "cannot compile an empty pipeline"
            raise ValueError(msg)

        incoming: dict[PortEndpoint, PortEndpoint] = {}
        outgoing: dict[PortEndpoint, list[PortEndpoint]] = defaultdict(list)
        adjacency: dict[str, list[str]] = defaultdict(list)
        indegree = {node_id: 0 for node_id in self._nodes}

        for edge in self._edges:
            if edge.source.node_id not in self._nodes:
                msg = f"edge references unknown source node {edge.source.node_id!r}"
                raise ValueError(msg)
            if edge.target.node_id not in self._nodes:
                msg = f"edge references unknown target node {edge.target.node_id!r}"
                raise ValueError(msg)

            source_operator = self._nodes[edge.source.node_id]
            target_operator = self._nodes[edge.target.node_id]
            source_ports = _port_map(source_operator.output_ports)
            target_ports = _port_map(target_operator.input_ports)
            if edge.source.port not in source_ports:
                msg = f"edge references unknown output port {edge.source}"
                raise ValueError(msg)
            if edge.target.port not in target_ports:
                msg = f"edge references unknown input port {edge.target}"
                raise ValueError(msg)
            if edge.target in incoming:
                msg = f"input port {edge.target} has more than one connection"
                raise ValueError(msg)

            source_port = source_ports[edge.source.port]
            target_port = target_ports[edge.target.port]
            if source_port.kind is not target_port.kind:
                msg = (
                    f"incompatible edge {edge.source} -> {edge.target}: "
                    f"{source_port.kind.value} cannot feed {target_port.kind.value}"
                )
                raise TypeError(msg)
            if (
                source_port.schema is not None
                and target_port.schema is not None
                and not source_port.schema.equals(
                    target_port.schema, check_metadata=True
                )
            ):
                msg = (
                    f"incompatible Arrow schemas on edge {edge.source} -> {edge.target}"
                )
                raise TypeError(msg)

            incoming[edge.target] = edge.source
            outgoing[edge.source].append(edge.target)
            adjacency[edge.source.node_id].append(edge.target.node_id)
            indegree[edge.target.node_id] += 1

        ready = deque(node_id for node_id in self._nodes if indegree[node_id] == 0)
        topology: list[str] = []
        while ready:
            node_id = ready.popleft()
            topology.append(node_id)
            for target in adjacency[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if len(topology) != len(self._nodes):
            msg = "pipeline graph contains a cycle"
            raise ValueError(msg)

        external_inputs = [
            PortEndpoint(node_id, port.name)
            for node_id in topology
            for port in self._nodes[node_id].input_ports
            if PortEndpoint(node_id, port.name) not in incoming
        ]
        external_outputs = [
            PortEndpoint(node_id, port.name)
            for node_id in topology
            for port in self._nodes[node_id].output_ports
            if PortEndpoint(node_id, port.name) not in outgoing
        ]
        if not external_outputs:
            msg = "pipeline graph has no reachable outputs"
            raise ValueError(msg)

        edges = tuple(self._edges)
        registry = (
            self.udf_registry.snapshot()
            if isinstance(self.udf_registry, UdfRegistry)
            else self.udf_registry
        )
        selected_udfs = registry.select(
            datafusion=(
                reference
                for operator in self._nodes.values()
                for reference in operator.datafusion_udfs()
            ),
            array=(
                reference
                for operator in self._nodes.values()
                for reference in operator.array_udfs()
            ),
        )
        return ExecutionPlan(
            name=self.name,
            nodes=tuple(
                CompiledNode(node_id, self._nodes[node_id]) for node_id in topology
            ),
            edges=edges,
            graph_inputs=_external_names(external_inputs),
            graph_outputs=_external_names(external_outputs),
            datafusion_config=self.datafusion_config,
            udfs=selected_udfs,
            fingerprint=_pipeline_fingerprint(
                self.name,
                self._nodes,
                edges,
                self.datafusion_config,
                selected_udfs,
            ),
        )

    def execute(self, inputs: Mapping[str, Batch], **options: Any) -> RunResult:
        return self.compile().execute(inputs, **options)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return self.compile().snapshot()

    def restore(self, checkpoint: Mapping[str, dict[str, Any]]) -> None:
        self.compile().restore(checkpoint)

    def reset(self) -> None:
        for operator in self._nodes.values():
            operator.reset()

    def __iter__(self) -> Iterator[Operator]:
        return iter(self._nodes.values())

    def __repr__(self) -> str:
        nodes = ", ".join(self._nodes)
        return f"Pipeline(name={self.name!r}, nodes=[{nodes}])"
