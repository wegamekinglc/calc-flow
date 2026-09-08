"""Private declaration-to-endpoint metadata kept beside native project documents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from calc_flow.pipeline import _node_inputs

type Endpoint = tuple[str, str]


@dataclass(slots=True)
class _BatchBindings:
    inputs: dict[str, set[Endpoint]] = field(default_factory=dict)
    outputs: dict[str, Endpoint] = field(default_factory=dict)

    def add_table(
        self, logical: str, output_id: str, document: dict[str, object]
    ) -> None:
        graph = document["graph"]
        nodes = {node["id"]: node for node in graph["nodes"]}
        incoming = {
            (edge["target_node"], edge["target_port"]): (
                edge["source_node"],
                edge["source_port"],
            )
            for edge in graph["edges"]
        }
        visited: set[str] = set()
        endpoints: set[Endpoint] = set()

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            for port in _node_inputs(nodes[node_id]):
                endpoint = (node_id, port)
                parent = incoming.get(endpoint)
                if parent is None:
                    endpoints.add(endpoint)
                else:
                    visit(parent[0])

        visit(output_id)
        self.inputs.setdefault(logical, set()).update(endpoints)
        self.outputs[output_id] = (output_id, "output")

    def names(self) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
        inputs = _endpoint_names(
            {endpoint for endpoints in self.inputs.values() for endpoint in endpoints}
        )
        outputs = _endpoint_names(set(self.outputs.values()))
        return (
            {
                logical: tuple(inputs[endpoint] for endpoint in sorted(endpoints))
                for logical, endpoints in self.inputs.items()
            },
            {logical: outputs[endpoint] for logical, endpoint in self.outputs.items()},
        )


def _endpoint_names(endpoints: set[Endpoint]) -> dict[Endpoint, str]:
    counts = Counter(port for _, port in endpoints)
    return {
        (node, port): port if counts[port] == 1 else f"{node}.{port}"
        for node, port in endpoints
    }
