"""Common subexpression extraction over resolved row-local forests.

Structurally identical, non-trivial subtrees referenced at least twice are
materialized once as ``__cf_cse_N`` columns in tiered expression nodes so the
final fused node computes every shared subexpression exactly once. Tiers are
emitted deepest first; discovery order, naming, and passthrough order are all
deterministic functions of the declaration.
"""

from __future__ import annotations

from dataclasses import dataclass

from calc_flow.symbolic.nodes import CStr, Node, build

_TRIVIAL = frozenset({"column_ref", "literal"})
_PREDICATE_KEY = ("predicate",)
_FIXED_TYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "float32": 4,
    "int32": 4,
    "uint32": 4,
    "float64": 8,
    "int64": 8,
    "uint64": 8,
    "timestamp[us]": 8,
    "timestamp[us, UTC]": 8,
}


@dataclass(frozen=True, slots=True)
class FusedSegment:
    """The extraction outcome: emission-ordered tiers, final selects, predicate."""

    tiers: tuple[tuple[tuple[str, Node], ...], ...]
    selects: tuple[tuple[str, Node], ...]
    predicate: Node | None


def expression_refs(tree: Node, /) -> frozenset[str]:
    """Collect every column reference name inside one resolved tree."""

    names: set[str] = set()

    def walk(node: Node) -> None:
        if node.op.name == "column_ref":
            value = node.attr("name")
            if isinstance(value, CStr):
                names.add(value.value)
            return
        for argument in node.args:
            walk(argument)

    walk(tree)
    return frozenset(names)


def _subtree_counts(forest: list[tuple[tuple[str, ...], Node]], /) -> dict[str, int]:
    counts: dict[str, int] = {}

    def walk(node: Node) -> None:
        counts[node.digest] = counts.get(node.digest, 0) + 1
        for argument in node.args:
            walk(argument)

    for _, tree in forest:
        walk(tree)
    return counts


def _maximal_candidates(
    forest: list[tuple[tuple[str, ...], Node]],
    counts: dict[str, int],
    /,
) -> list[tuple[str, Node]]:
    """Shared non-trivial subtrees not contained in any shared subtree.

    A subtree that only occurs inside other shared subtrees is deferred: once
    the enclosing candidates are rewritten to references, the deferred subtree
    is rediscovered in a deeper (earlier-emitted) tier, so no materialized
    column ever aliases another.
    """

    contained: set[str] = set()

    def mark_descendants(node: Node) -> None:
        for argument in node.args:
            contained.add(argument.digest)
            mark_descendants(argument)

    chosen: list[tuple[str, Node]] = []
    seen: set[str] = set()

    def walk(node: Node) -> None:
        if counts.get(node.digest, 0) >= 2 and node.op.name not in _TRIVIAL:
            if node.digest not in seen:
                seen.add(node.digest)
                chosen.append((node.digest, node))
                mark_descendants(node)
            return
        for argument in node.args:
            walk(argument)

    for _, tree in forest:
        walk(tree)
    return [(digest, tree) for digest, tree in chosen if digest not in contained]


def _rewrite(tree: Node, replacements: dict[str, str], /) -> Node:
    replacement = replacements.get(tree.digest)
    if replacement is not None:
        return build("column_ref", (), {"name": CStr(replacement)})
    if not tree.args:
        return tree
    return build(
        tree.op.name,
        tuple(_rewrite(argument, replacements) for argument in tree.args),
        dict(tree.attrs.entries),
        version=tree.op.version,
    )


def extract_common(
    selects: tuple[tuple[str, Node], ...],
    predicate: Node | None,
    reserved: frozenset[str],
    /,
) -> FusedSegment:
    """Extract shared subexpressions into emission-ordered materialization tiers.

    ``reserved`` names (declared fields) are never used for materialized
    columns. Discovery-order tiers are reversed for emission so deeper shared
    subexpressions are computed before the tiers that reference them.
    """

    forest: list[tuple[tuple[str, ...], Node]] = [
        (("select", name), tree) for name, tree in selects
    ]
    if predicate is not None:
        forest.append((_PREDICATE_KEY, predicate))
    iterations: list[tuple[str, ...]] = []
    counter = 0

    def next_name() -> str:
        nonlocal counter
        while f"__cf_cse_{counter}" in reserved:
            counter += 1
        name = f"__cf_cse_{counter}"
        counter += 1
        return name

    while True:
        counts = _subtree_counts(forest)
        candidates = _maximal_candidates(forest, counts)
        if not candidates:
            break
        replacements: dict[str, str] = {}
        names: list[str] = []
        defs: list[tuple[tuple[str, ...], Node]] = []
        for digest, tree in candidates:
            name = next_name()
            replacements[digest] = name
            names.append(name)
            defs.append((("cse", name), tree))
        iterations.append(tuple(names))
        forest = [(key, _rewrite(tree, replacements)) for key, tree in forest]
        forest.extend(defs)

    by_key = {key: tree for key, tree in forest}
    tiers = tuple(
        tuple((name, by_key[("cse", name)]) for name in names)
        for names in reversed(iterations)
    )
    final_selects = tuple((name, by_key[("select", name)]) for name, _ in selects)
    final_predicate = by_key.get(_PREDICATE_KEY)
    return FusedSegment(tiers, final_selects, final_predicate)


def _document_nodes(document: dict[str, object], /) -> list[dict[str, object]]:
    graph = document.get("graph")
    if not isinstance(graph, dict):
        return []
    nodes = graph.get("nodes")
    return nodes if isinstance(nodes, list) else []  # type: ignore[return-value]


def _state_layout(node: dict[str, object], /) -> str:
    input_ports = node.get("input_ports")
    schema = (
        input_ports[0].get("schema")
        if isinstance(input_ports, list)
        and input_ports
        and isinstance(input_ports[0], dict)
        else None
    )
    fields = schema if isinstance(schema, list) else []
    fixed_bytes = sum(
        _FIXED_TYPE_BYTES.get(field.get("data_type"), 0)
        for field in fields
        if isinstance(field, dict)
    )
    variable_columns = sum(
        field.get("data_type") not in _FIXED_TYPE_BYTES
        for field in fields
        if isinstance(field, dict)
    )
    layout = (
        f"retained_columns={len(fields)} fixed_bytes_per_row={fixed_bytes}"
        f" variable_columns={variable_columns}"
    )
    return layout


def _rolling_boundary(outputs: list[object], /) -> str:
    row_bounds: list[int] = []
    duration_bounds: list[int] = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        periods = output.get("periods")
        if isinstance(periods, int):
            row_bounds.append(periods + 1)
        frame = output.get("frame")
        if not isinstance(frame, dict):
            continue
        if frame.get("kind") == "rows" and isinstance(frame.get("size"), int):
            row_bounds.append(frame["size"])
        if frame.get("kind") == "duration" and isinstance(frame.get("micros"), int):
            duration_bounds.append(frame["micros"])
    bounds = []
    if row_bounds:
        bounds.append(f"rows={max(row_bounds)}")
    if duration_bounds:
        bounds.append(f"duration_micros={max(duration_bounds)}")
    return " ".join(bounds) or "constant"


def _cross_section_boundary(spec: dict[str, object], /) -> str:
    grouping = spec.get("grouping")
    if isinstance(grouping, dict) and grouping.get("kind") == "fixed_bucket":
        return f"bucket_width_micros={grouping.get('width_micros')}"
    return "exact_time_groups"


def _state_cost(node: dict[str, object], /) -> str | None:
    operator = node.get("operator")
    spec = operator.get("spec") if isinstance(operator, dict) else None
    outputs = spec.get("outputs") if isinstance(spec, dict) else None
    if not isinstance(outputs, list):
        return None
    layout = _state_layout(node)
    kind = operator.get("kind")
    if kind == "rolling":
        return f"    state {node['id']} {_rolling_boundary(outputs)} {layout}"
    if kind == "cross_section":
        boundary = _cross_section_boundary(spec)
        return f"    state {node['id']} {boundary} active_groups=runtime {layout}"
    return None


def _static_weight_bytes(document: dict[str, object], /) -> int | None:
    static_inputs = document.get("static_inputs")
    if not isinstance(static_inputs, list):
        return None
    for declaration in static_inputs:
        if not isinstance(declaration, dict) or declaration.get("kind") != "array":
            continue
        shape = declaration.get("shape")
        width = _FIXED_TYPE_BYTES.get(declaration.get("dtype"))
        if not isinstance(shape, list) or width is None:
            continue
        elements = 1
        for dimension in shape:
            if not isinstance(dimension, int):
                return None
            elements *= dimension
        return elements * width
    return None


def _copy_cost(
    node: dict[str, object], static_weight_bytes: int | None, /
) -> str | None:
    operator = node.get("operator")
    if not isinstance(operator, dict) or operator.get("kind") != "external":
        return None
    options = operator.get("options")
    columns = options.get("columns") if isinstance(options, dict) else None
    column_count = len(columns) if isinstance(columns, list) else 0
    backend = operator.get("provider")
    device_copy = "yes" if backend == "jax" else "no"
    weights = "runtime" if static_weight_bytes is None else str(static_weight_bytes)
    return (
        f"    copies {node['id']} table_to_dense columns={column_count}"
        f" rows=runtime host_to_device={device_copy} static_weights_bytes={weights}"
    )


def _provider_cost(node: dict[str, object], /) -> str | None:
    operator = node.get("operator")
    if not isinstance(operator, dict) or operator.get("kind") != "external":
        return None
    return (
        f"    providers {node['id']} {operator.get('provider')}:"
        f"{operator.get('name')}@{operator.get('version')} calls_per_microbatch=1"
    )


def _nodes_of_kind(
    nodes: list[dict[str, object]], kind: str, /
) -> list[dict[str, object]]:
    return [
        node
        for node in nodes
        if isinstance(node.get("operator"), dict)
        and node["operator"].get("kind") == kind  # type: ignore[union-attr]
    ]


def _state_output_count(items: list[dict[str, object]], /) -> int:
    count = 0
    for item in items:
        operator = item["operator"]
        spec = operator.get("spec") if isinstance(operator, dict) else None
        outputs = spec.get("outputs") if isinstance(spec, dict) else None
        count += len(outputs) if isinstance(outputs, list) else 0
    return count


def explain_optimization(document: dict[str, object], /) -> tuple[str, ...]:
    """Render deterministic physical sharing and bounded cost facts."""

    nodes = _document_nodes(document)
    cse_count = sum("__cf_cse_" in str(node.get("id")) for node in nodes)
    rolling = _nodes_of_kind(nodes, "rolling")
    cross_section = _nodes_of_kind(nodes, "cross_section")
    external = _nodes_of_kind(nodes, "external")

    lines = (
        "  optimization",
        f"    cse materializations {cse_count}",
        "    rolling state_stages"
        f" {len(rolling)} shared_outputs {_state_output_count(rolling)}",
        "    cross_section grouping_stages"
        f" {len(cross_section)} shared_outputs {_state_output_count(cross_section)}",
        f"    array fused_stages {len(external)} provider_calls_per_microbatch"
        f" {len(external)}",
        "  costs",
    )
    state = tuple(cost for node in nodes if (cost := _state_cost(node)) is not None)
    static_weight_bytes = _static_weight_bytes(document)
    copies = tuple(
        cost
        for node in nodes
        if (cost := _copy_cost(node, static_weight_bytes)) is not None
    )
    providers = tuple(
        cost for node in nodes if (cost := _provider_cost(node)) is not None
    )
    return (
        *lines,
        *(state or ("    state none",)),
        *(copies or ("    copies none",)),
        *(providers or ("    providers none",)),
    )
