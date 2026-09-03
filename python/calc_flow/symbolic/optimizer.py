"""Common subexpression extraction over resolved row-local forests.

Structurally identical, non-trivial subtrees referenced at least twice are
materialized once as ``__cf_cse_N`` columns in tiered expression nodes so the
final fused node computes every shared subexpression exactly once. Tiers are
emitted deepest first; discovery order, naming, and passthrough order are all
deterministic functions of the declaration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from calc_flow.symbolic._generated_rolling_kernels import (
    ROLLING_KERNEL_CAPABILITIES,
)
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
_PRIMITIVE_NUMERIC_TYPES = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    }
)


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


def _input_schema_fields(
    node: dict[str, object], index: int = 0, /
) -> list[dict[str, object]]:
    input_ports = node.get("input_ports")
    if not isinstance(input_ports, list):
        return []
    if not input_ports:
        return []
    if index >= len(input_ports):
        return []
    port = input_ports[index]
    if not isinstance(port, dict):
        return []
    schema = port.get("schema")
    if not isinstance(schema, list):
        return []
    return [field for field in schema if isinstance(field, dict)]


def _state_layout(node: dict[str, object], index: int = 0, /) -> str:
    fields = _input_schema_fields(node, index)
    fixed_bytes = sum(
        _FIXED_TYPE_BYTES.get(field.get("data_type"), 0) for field in fields
    )
    variable_columns = sum(
        field.get("data_type") not in _FIXED_TYPE_BYTES for field in fields
    )
    layout = (
        f"retained_columns={len(fields)} fixed_bytes_per_row={fixed_bytes}"
        f" variable_columns={variable_columns}"
    )
    return layout


def _frame_bounds(frame: object, /) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(frame, dict):
        return (), ()
    kind = frame.get("kind")
    if kind == "rows":
        size = frame.get("size")
        return ((size,) if isinstance(size, int) else ()), ()
    if kind == "duration":
        micros = frame.get("micros")
        return (), ((micros,) if isinstance(micros, int) else ())
    return (), ()


def _rolling_output_bounds(
    output: object, /
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(output, dict):
        return (), ()
    if output.get("kind") == "difference":
        left_rows, left_durations = _rolling_output_bounds(output.get("left"))
        right_rows, right_durations = _rolling_output_bounds(output.get("right"))
        return (*left_rows, *right_rows), (*left_durations, *right_durations)
    periods = output.get("periods")
    row_bounds = (periods + 1,) if isinstance(periods, int) else ()
    frame_rows, duration_bounds = _frame_bounds(output.get("frame"))
    return (*row_bounds, *frame_rows), duration_bounds


def _rolling_boundary(outputs: list[object], /) -> str:
    row_bounds: list[int] = []
    duration_bounds: list[int] = []
    for output in outputs:
        output_rows, output_durations = _rolling_output_bounds(output)
        row_bounds.extend(output_rows)
        duration_bounds.extend(output_durations)
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
    kind = operator.get("kind") if isinstance(operator, dict) else None
    if kind == "stream_join" and isinstance(spec, dict):
        limits = spec.get("limits")
        if not isinstance(limits, dict):
            return None
        return (
            f"    state {node['id']}"
            f" max_state_rows_per_side={limits.get('max_state_rows_per_side')}"
            f" max_state_bytes_per_side={limits.get('max_state_bytes_per_side')}"
            " max_matches_per_input_batch="
            f"{limits.get('max_matches_per_input_batch')}"
            f" left_{_state_layout(node, 0)} right_{_state_layout(node, 1)}"
        )
    outputs = spec.get("outputs") if isinstance(spec, dict) else None
    if not isinstance(outputs, list):
        return None
    layout = _state_layout(node)
    if kind == "rolling":
        return f"    state {node['id']} {_rolling_boundary(outputs)} {layout}"
    if kind == "cross_section":
        boundary = _cross_section_boundary(spec)
        return f"    state {node['id']} {boundary} active_groups=runtime {layout}"
    return None


def _static_array_weight(declaration: object, /) -> tuple[bool, int | None]:
    if not isinstance(declaration, dict):
        return False, None
    if declaration.get("kind") != "array":
        return False, None
    shape = declaration.get("shape")
    width = _FIXED_TYPE_BYTES.get(declaration.get("dtype"))
    if not isinstance(shape, list):
        return False, None
    if width is None:
        return False, None
    elements = 1
    for dimension in shape:
        if not isinstance(dimension, int):
            return True, None
        elements *= dimension
    return True, elements * width


def _static_weight_bytes(document: dict[str, object], /) -> int | None:
    static_inputs = document.get("static_inputs")
    if not isinstance(static_inputs, list):
        return None
    for declaration in static_inputs:
        found, weight = _static_array_weight(declaration)
        if found:
            return weight
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


def _rolling_fusion_count(items: list[dict[str, object]], /) -> int:
    count = 0
    for item in items:
        operator = item["operator"]
        spec = operator.get("spec") if isinstance(operator, dict) else None
        outputs = spec.get("outputs") if isinstance(spec, dict) else None
        if isinstance(outputs, list):
            count += sum(
                isinstance(output, dict) and output.get("kind") == "difference"
                for output in outputs
            )
    return count


def _rolling_leaf_outputs(output: object, /) -> tuple[dict[str, object], ...]:
    if not isinstance(output, dict):
        return ()
    if output.get("kind") != "difference":
        return (output,)
    return (
        *_rolling_leaf_outputs(output.get("left")),
        *_rolling_leaf_outputs(output.get("right")),
    )


def _rolling_frame_key(output: dict[str, object], /) -> tuple[object, object]:
    frame = output.get("frame")
    if not isinstance(frame, dict):
        return (None, None)
    kind = frame.get("kind")
    coordinate = frame.get("size") if kind == "rows" else frame.get("micros")
    return kind, coordinate


def _rolling_group_key(output: dict[str, object], /) -> tuple[object, ...] | None:
    kind = output.get("kind")
    frame = _rolling_frame_key(output)
    if kind in {"count", "sum", "mean", "variance", "stddev"}:
        return "numeric", output.get("input"), *frame
    if kind in {"min", "max"}:
        return "extrema", kind, output.get("input"), *frame
    if kind in {"covariance", "correlation"}:
        return "pair", output.get("left"), output.get("right"), *frame
    if kind == "ewma":
        return "ewma", output.get("input"), output.get("span")
    return None


def _rolling_kernel_fallback(
    output: dict[str, object], field_types: dict[str, object], /
) -> str | None:
    kind = output.get("kind")
    capability = ROLLING_KERNEL_CAPABILITIES.get(kind)
    if capability is None:
        return f"primitive_{kind}_missing_from_census"
    transition = capability[0]
    if transition is None:
        return f"primitive_{kind}_has_no_typed_transition"
    if kind == "difference":
        for leaf in _rolling_leaf_outputs(output):
            fallback = _rolling_kernel_fallback(leaf, field_types)
            if fallback is not None:
                return fallback
        return None
    columns = (
        (output.get("left"), output.get("right"))
        if transition == "pair"
        else (output.get("input"),)
    )
    for column in columns:
        data_type = field_types.get(column) if isinstance(column, str) else None
        if data_type not in _PRIMITIVE_NUMERIC_TYPES:
            return f"primitive_{kind}_requires_numeric_column_{column}"
    return None


def _rolling_kernel_line(node: dict[str, object], /) -> str | None:
    operator = node.get("operator")
    spec = operator.get("spec") if isinstance(operator, dict) else None
    outputs = spec.get("outputs") if isinstance(spec, dict) else None
    if not isinstance(spec, dict) or not isinstance(outputs, list):
        return None
    field_types = {
        str(field.get("name")): field.get("data_type")
        for field in _input_schema_fields(node)
    }
    groups = {
        key
        for output in outputs
        for leaf in _rolling_leaf_outputs(output)
        if (key := _rolling_group_key(leaf)) is not None
    }
    fallback = next(
        (
            reason
            for output in outputs
            if isinstance(output, dict)
            if (reason := _rolling_kernel_fallback(output, field_types)) is not None
        ),
        None,
    )
    selected = "ordered_primitive" if fallback is None and groups else "general"
    complexity = "amortized_constant" if selected == "ordered_primitive" else "general"
    order = ",".join(
        str(value)
        for value in (
            spec.get("event_time"),
            *(spec.get("partition_by") or []),
            *(spec.get("sequence_by") or []),
        )
    )
    profile = spec.get("numerical_profile", "stable_v1")
    return (
        f"    rolling kernel {node['id']} selected={selected}"
        f" profile={profile} complexity={complexity} order={order}"
        f" shared_state_groups={len(groups)} fallback={fallback or 'none'}"
    )


def _cost_lines(
    nodes: list[dict[str, object]],
    renderer: Callable[[dict[str, object]], str | None],
    /,
) -> tuple[str, ...]:
    lines: list[str] = []
    for node in nodes:
        line = renderer(node)
        if line is not None:
            lines.append(line)
    return tuple(lines)


def explain_optimization(document: dict[str, object], /) -> tuple[str, ...]:
    """Render deterministic physical sharing and bounded cost facts."""

    nodes = _document_nodes(document)
    cse_count = sum("__cf_cse_" in str(node.get("id")) for node in nodes)
    rolling = _nodes_of_kind(nodes, "rolling")
    cross_section = _nodes_of_kind(nodes, "cross_section")
    stream_join = _nodes_of_kind(nodes, "stream_join")
    external = _nodes_of_kind(nodes, "external")

    lines = (
        "  optimization",
        f"    cse materializations {cse_count}",
        "    rolling state_stages"
        f" {len(rolling)} shared_outputs {_state_output_count(rolling)}",
        "    rolling fused_outputs"
        f" {_rolling_fusion_count(rolling)} hidden_materializations 0",
        "    cross_section grouping_stages"
        f" {len(cross_section)} shared_outputs {_state_output_count(cross_section)}",
        f"    stream_join state_stages {len(stream_join)}",
        f"    array fused_stages {len(external)} provider_calls_per_microbatch"
        f" {len(external)}",
    )
    kernels = _cost_lines(rolling, _rolling_kernel_line)
    state = _cost_lines(nodes, _state_cost)
    static_weight_bytes = _static_weight_bytes(document)
    copies = _cost_lines(nodes, lambda node: _copy_cost(node, static_weight_bytes))
    providers = _cost_lines(nodes, _provider_cost)
    return (
        *lines,
        *(kernels or ("    rolling kernels none",)),
        "  costs",
        *(state or ("    state none",)),
        *(copies or ("    copies none",)),
        *(providers or ("    providers none",)),
    )
