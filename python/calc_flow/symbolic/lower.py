"""Deterministic lowering of symbolic programs to strict project-v3.

The lowerer resolves each declared table output into one fused row-local
segment, renders the segment as DataFusion SQL inside strict project-v3
``expression`` nodes, and hands the document to the existing Rust graph
compiler for final port, schema, topology, and fingerprint validation. No data
object, source, sink, or runner is accepted here, and no symbolic Python runs
while a compiled plan executes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Never

from calc_flow.capabilities import RuntimeCapabilities
from calc_flow.pipeline import (
    Runtime,
    StreamRequirements,
    _canonical,
    _data_sources,
)
from calc_flow.symbolic import errors
from calc_flow.symbolic.analyzer import (
    _ROW_LOCAL_PRIMITIVES,
    TableFacts,
    _Analyzer,
    _literal_dtype,
    _require_mode,
    _rolling_output_type,
    _run,
    _schema_fields,
)
from calc_flow.symbolic.domains import type_name
from calc_flow.symbolic.nodes import (
    CBool,
    CDType,
    CEnum,
    CFloat,
    CInt,
    CMap,
    CNull,
    CSeq,
    CStr,
    CValue,
    Node,
    build,
)
from calc_flow.symbolic.optimizer import expression_refs, extract_common
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    from calc_flow.pipeline import BatchExecutionPlan, StreamExecutionPlan
    from calc_flow.symbolic.program import Program


@dataclass(frozen=True, slots=True)
class _CompileCacheKey:
    """Deterministic, runtime-scoped identity for one symbolic compilation."""

    program_fingerprint: str
    mode: str
    input_declarations: tuple[str, ...]
    capability_schema_version: int
    capability_session_id: str
    capability_revision: int
    operator_versions: tuple[tuple[str, str], ...]
    provider_versions: tuple[tuple[str, str, str], ...]
    udf_versions: tuple[tuple[str, str, str], ...]
    allowed_lateness_micros: int
    late_policy: str


_TABLE_OUTPUT_PRIMITIVES: Final = frozenset(
    {"table_input", "project", "filter", "with_columns"}
)

_MATRIX_PRIMITIVES: Final = frozenset(
    {
        "add",
        "and",
        "eq",
        "ge",
        "gt",
        "le",
        "lt",
        "matmul",
        "mul",
        "ne",
        "neg",
        "not",
        "or",
        "sub",
        "truediv",
    }
)

_ROLLING_PRIMITIVES: Final = frozenset(
    {
        "lag",
        "delta",
        "ewma",
        "count",
        "sum",
        "mean",
        "min",
        "max",
        "variance",
        "stddev",
        "covariance",
        "correlation",
    }
)

_ROLLING_DDOF_PRIMITIVES: Final = frozenset(
    {"variance", "stddev", "covariance", "correlation"}
)

_ROLLING_PAIR_PRIMITIVES: Final = frozenset({"covariance", "correlation"})

_CROSS_SECTION_PRIMITIVES: Final = frozenset(
    {
        "rank",
        "percentile",
        "demean",
        "zscore",
        "winsorize",
        "top",
        "bottom",
        "mean_fill",
    }
)

_CROSS_SECTION_ORDERING: Final = ("rank", "percentile")

_CROSS_SECTION_DDOF: Final = "zscore"

_U64_MAX: Final = (1 << 64) - 1

_BINARY_SQL: Final = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "eq": "=",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "and": "AND",
    "or": "OR",
}

_FUNCTION_SQL: Final = {"log": "ln", "exp": "exp", "sqrt": "sqrt", "abs": "abs"}

_CAST_TYPES: Final = {
    "bool": "BOOLEAN",
    "int8": "TINYINT",
    "int16": "SMALLINT",
    "int32": "INT",
    "int64": "BIGINT",
    "uint8": "TINYINT UNSIGNED",
    "uint16": "SMALLINT UNSIGNED",
    "uint32": "INT UNSIGNED",
    "uint64": "BIGINT UNSIGNED",
    "float32": "REAL",
    "float64": "DOUBLE",
}


@dataclass(frozen=True, slots=True)
class _Segment:
    """One fused row-local table resolution over one table input lineage.

    ``predicate`` filters declared *below* every rolling feature (they feed
    the rolling stage); ``post_predicate`` filters declared *above* them
    (they apply after). Without rolling primitives both fuse at the final
    stage, preserving the historical behavior.
    """

    input_node: Node
    fields: tuple[str, ...]
    env: tuple[tuple[str, Node], ...]
    predicate: Node | None
    post_predicate: Node | None = None


def _cstr(value: CValue | None, /) -> str:
    return value.value if isinstance(value, CStr) else ""


def _cstr_seq(value: CValue | None, /) -> tuple[str, ...]:
    from calc_flow.symbolic.nodes import CSeq

    if isinstance(value, CSeq):
        return tuple(item.value for item in value.items if isinstance(item, CStr))
    return ()


def _base_ref(name: str, /) -> Node:
    return build("column_ref", (), {"name": CStr(name)})


def _reject_primitive(path: str, node: Node, /) -> None:
    errors.raise_compile(
        path,
        errors.UNKNOWN_PRIMITIVE_VERSION,
        f"primitive {node.op.name!r} is not supported by the row-local lowerer",
    )


def _resolve_table(node: Node, path: str, /) -> _Segment:
    name = node.op.name
    if name == "table_input":
        fields = _schema_fields(node.attr("schema"))
        return _Segment(
            node,
            tuple(field.name for field in fields),
            tuple((field.name, _base_ref(field.name)) for field in fields),
            None,
        )
    if name == "project":
        child = _resolve_table(node.args[0], f"{path}.project.value")
        env = dict(child.env)
        columns = _cstr_seq(node.attr("columns"))
        return _Segment(
            child.input_node,
            columns,
            tuple((field, env[field]) for field in columns),
            child.predicate,
            child.post_predicate,
        )
    if name == "filter":
        child = _resolve_table(node.args[0], f"{path}.filter.value")
        predicate = _inline(node.args[1], dict(child.env), f"{path}.filter.predicate")
        if (
            _segment_has_rolling(child)
            or any(True for _ in _find_rolling(predicate))
            or _segment_has_cross_section(child)
            or any(True for _ in _find_cross_section(predicate))
        ):
            combined = (
                predicate
                if child.post_predicate is None
                else build("and", (child.post_predicate, predicate), {})
            )
            return _Segment(
                child.input_node, child.fields, child.env, child.predicate, combined
            )
        combined = (
            predicate
            if child.predicate is None
            else build("and", (child.predicate, predicate), {})
        )
        return _Segment(
            child.input_node, child.fields, child.env, combined, child.post_predicate
        )
    if name == "with_columns":
        child = _resolve_table(node.args[0], f"{path}.with_columns.value")
        env = dict(child.env)
        names = _cstr_seq(node.attr("names"))
        for index, feature in enumerate(names):
            env[feature] = _inline(node.args[index + 1], env, f"{path}.{feature}")
        return _Segment(
            child.input_node,
            (*child.fields, *names),
            tuple(env.items()),
            child.predicate,
            child.post_predicate,
        )
    _reject_primitive(path, node)


def _segment_has_rolling(segment: _Segment, /) -> bool:
    return any(True for _, tree in segment.env for _ in _find_rolling(tree)) or any(
        True
        for tree in (segment.predicate, segment.post_predicate)
        if tree is not None
        for _ in _find_rolling(tree)
    )


def _segment_has_cross_section(segment: _Segment, /) -> bool:
    return any(
        True for _, tree in segment.env for _ in _find_cross_section(tree)
    ) or any(
        True
        for tree in (segment.predicate, segment.post_predicate)
        if tree is not None
        for _ in _find_cross_section(tree)
    )


def _inline(node: Node, env: dict[str, Node], path: str, /) -> Node:
    name = node.op.name
    if name == "column_ref":
        return env[_cstr(node.attr("name"))]
    if name == "literal":
        return node
    if (
        name not in _ROW_LOCAL_PRIMITIVES
        and name not in _ROLLING_PRIMITIVES
        and name not in _CROSS_SECTION_PRIMITIVES
    ):
        _reject_primitive(path, node)
    if name == "cast":
        _cast_target(node, path)
    return build(
        name,
        tuple(_inline(argument, env, path) for argument in node.args),
        dict(node.attrs.entries),
        version=node.op.version,
    )


def _find_primitives(node: Node, primitives: frozenset[str], /):
    """Yield matching subtrees in deterministic first-appearance order."""

    if node.op.name in primitives:
        yield node
    for argument in node.args:
        yield from _find_primitives(argument, primitives)


def _find_rolling(node: Node, /):
    """Yield every rolling temporal subtree in first-appearance order."""

    yield from _find_primitives(node, _ROLLING_PRIMITIVES)


def _find_cross_section(node: Node, /):
    """Yield every cross-section subtree in first-appearance order."""

    yield from _find_primitives(node, _CROSS_SECTION_PRIMITIVES)


def _cast_target(node: Node, path: str, /) -> str:
    raw = node.attr("data_type")
    declared = _cstr(raw) or (raw.name if isinstance(raw, CDType) else "")
    target = _CAST_TYPES.get(declared)
    if target is None:
        errors.raise_compile(
            f"{path}.cast.data_type",
            errors.UNSUPPORTED_TYPE,
            f"cast target {declared!r} is not portable in the row-local lowerer",
        )
    return target


def _sql(node: Node, /) -> str:
    name = node.op.name
    if name == "column_ref":
        return _quote_identifier(_cstr(node.attr("name")))
    if name == "literal":
        return _sql_literal(node.attr("value"))
    if name in _BINARY_SQL:
        return f"({_sql(node.args[0])} {_BINARY_SQL[name]} {_sql(node.args[1])})"
    if name == "neg":
        return f"(-{_sql(node.args[0])})"
    if name == "not":
        return f"(NOT {_sql(node.args[0])})"
    if name == "where":
        return (
            f"(CASE WHEN {_sql(node.args[0])} THEN {_sql(node.args[1])}"
            f" ELSE {_sql(node.args[2])} END)"
        )
    if name == "coalesce":
        return "COALESCE(" + ", ".join(_sql(argument) for argument in node.args) + ")"
    if name in _FUNCTION_SQL:
        return f"{_FUNCTION_SQL[name]}({_sql(node.args[0])})"
    if name == "clip":
        value = _sql(node.args[0])
        lower = _sql_literal(node.attr("lower"))
        upper = _sql_literal(node.attr("upper"))
        return (
            f"(CASE WHEN {value} < {lower} THEN {lower}"
            f" WHEN {value} > {upper} THEN {upper} ELSE {value} END)"
        )
    if name == "cast":
        return f"CAST({_sql(node.args[0])} AS {_CAST_TYPES[_cast_type_name(node)]})"
    raise AssertionError(f"unlowerable primitive reached SQL rendering: {name}")


def _cast_type_name(node: Node, /) -> str:
    raw = node.attr("data_type")
    return _cstr(raw) or (raw.name if isinstance(raw, CDType) else "")


def _sql_literal(value: CValue | None, /) -> str:
    if isinstance(value, CNull) or value is None:
        return "NULL"
    if isinstance(value, CBool):
        return "TRUE" if value.value else "FALSE"
    if isinstance(value, CInt):
        return str(value.value)
    if isinstance(value, CFloat):
        return repr(value.value)
    if isinstance(value, CStr):
        return "'" + value.value.replace("'", "''") + "'"
    raise AssertionError(f"unsupported literal reached SQL rendering: {value!r}")


def _quote_identifier(name: str, /) -> str:
    return '"' + name.replace('"', '""') + '"'


def _select_item(name: str, tree: Node, /) -> str:
    if tree.op.name == "column_ref" and tree.attr("name") == CStr(name):
        return _quote_identifier(name)
    return f"{_sql(tree)} AS {_quote_identifier(name)}"


def _expression_node(
    node_id: str,
    select: list[str],
    filter_sql: str | None,
    input_schema: tuple[Field, ...] | None,
    output_schema: tuple[Field, ...] | None = None,
    /,
) -> dict[str, object]:
    node: dict[str, object] = {
        "id": node_id,
        "operator": {
            "kind": "expression",
            "expression": "",
            "select": select,
            "filter": filter_sql,
            "udfs": [],
        },
    }
    if input_schema is not None:
        node["input_ports"] = [
            {
                "name": "input",
                "kind": "table",
                "required": True,
                "schema": [_field_json(field) for field in input_schema],
            }
        ]
    if output_schema is not None:
        node["output_ports"] = [
            {
                "name": "output",
                "kind": "table",
                "required": True,
                "schema": [_field_json(field) for field in output_schema],
            }
        ]
    return node


def _field_json(field: Field, /) -> dict[str, object]:
    return {
        "name": field.name,
        "data_type": field.data_type,
        "nullable": field.nullable,
    }


def _cint(value: CValue | None, /) -> int | None:
    return value.value if isinstance(value, CInt) else None


def _cnumber(value: CValue | None, /) -> int | float | None:
    return value.value if isinstance(value, (CInt, CFloat)) else None


def _cbool(value: CValue | None, /) -> bool | None:
    return value.value if isinstance(value, CBool) else None


def _replace_materialized(node: Node, replacements: dict[str, str], /) -> Node:
    replacement = replacements.get(node.digest)
    if replacement is not None:
        return _base_ref(replacement)
    return build(
        node.op.name,
        tuple(_replace_materialized(argument, replacements) for argument in node.args),
        dict(node.attrs.entries),
        version=node.op.version,
    )


@dataclass(frozen=True, slots=True)
class _RollingPlan:
    """One lowered rolling stage: the project node plus the rewritten
    row-local environment that references its output columns."""

    node_id: str
    node: dict[str, object]
    materialization_node_id: str | None
    materialization_node: dict[str, object] | None
    env: tuple[tuple[str, Node], ...]
    post_predicate: Node | None
    input_field_names: tuple[str, ...]
    output_fields: tuple[Field, ...]
    replacements: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _RollingPipeline:
    """Ordered rolling stages and their final rewritten environment."""

    stages: tuple[_RollingPlan, ...]
    env: tuple[tuple[str, Node], ...]
    post_predicate: Node | None
    input_field_names: tuple[str, ...]
    output_fields: tuple[Field, ...]

    @property
    def node_id(self) -> str:
        """Return the final state stage identifier."""

        return self.stages[-1].node_id


@dataclass(frozen=True, slots=True)
class _StatefulInputPlan:
    """One deterministic row-local materialization before native state."""

    names: tuple[tuple[str, str], ...]
    fields: tuple[Field, ...]
    node_id: str | None
    node: dict[str, object] | None
    input_fields: tuple[Field, ...]
    used_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class _StatefulInputRequest:
    output_name: str
    path: str
    column_stem: str
    node_id: str
    domain: str


def _required_stateful_input(
    request: _StatefulInputRequest,
    primitive: str,
    argument: Node,
    index: int,
    input_types: dict[str, Field],
    reserved: set[str],
    /,
) -> tuple[str, Field, str]:
    if not _rolling_argument_is_row_local(argument):
        errors.raise_compile(
            request.path,
            errors.UNSUPPORTED_TYPE,
            f"{request.domain} {primitive} argument must be an input column"
            " or row-local expression after earlier state staging",
        )
    name = f"{request.output_name}__cf_{request.column_stem}_{index}"
    if name in reserved:
        errors.raise_compile(
            f"{request.path}.{name}",
            errors.DUPLICATE_NAME,
            f"materialized {request.domain} input {name!r} collides"
            " with a declared field",
        )
    return (
        name,
        _row_local_field(argument, name, input_types),
        _select_item(name, argument),
    )


def _plan_stateful_inputs(
    request: _StatefulInputRequest,
    input_fields: tuple[Field, ...],
    used_names: set[str],
    arguments: tuple[tuple[str, Node], ...],
    /,
) -> _StatefulInputPlan:
    input_types = {field.name: field for field in input_fields}
    reserved = set(used_names)
    names: dict[str, str] = {}
    fields: list[Field] = []
    selects = [_quote_identifier(field.name) for field in input_fields]
    for primitive, argument in arguments:
        if argument.op.name == "column_ref" or argument.digest in names:
            continue
        name, field, select = _required_stateful_input(
            request,
            primitive,
            argument,
            len(names),
            input_types,
            reserved,
        )
        reserved.add(name)
        names[argument.digest] = name
        fields.append(field)
        selects.append(select)
    state_input_fields = (*input_fields, *fields)
    materialization_id = request.node_id if fields else None
    materialization = (
        _expression_node(
            request.node_id,
            selects,
            None,
            input_fields,
            state_input_fields,
        )
        if fields
        else None
    )
    return _StatefulInputPlan(
        tuple(names.items()),
        tuple(fields),
        materialization_id,
        materialization,
        state_input_fields,
        frozenset(reserved),
    )


def _row_local_field(
    node: Node,
    name: str,
    input_types: dict[str, Field],
    /,
) -> Field:
    """Infer a validated row-local expression field for stateful staging."""

    leaf = _row_local_leaf_field(node, name, input_types)
    if leaf is not None:
        return leaf
    children = [_row_local_field(argument, name, input_types) for argument in node.args]
    return _row_local_composite_field(node, name, children)


def _row_local_leaf_field(
    node: Node,
    name: str,
    input_types: dict[str, Field],
    /,
) -> Field | None:
    operation = node.op.name
    if operation == "column_ref":
        source = input_types[_cstr(node.attr("name"))]
        return Field(name, source.data_type, nullable=source.nullable)
    if operation != "literal":
        return None
    value = node.attr("value")
    data_type = None if value is None else _literal_dtype(value)
    if data_type is None:
        raise RuntimeError("validated rolling literal has no data type")
    return Field(name, data_type, nullable=isinstance(value, CNull))


def _row_local_composite_field(
    node: Node,
    name: str,
    children: list[Field],
    /,
) -> Field:
    operation = node.op.name
    nullable = _any_nullable(children)
    if operation in {"eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not"}:
        return Field(name, "bool", nullable=nullable)
    if operation in _FUNCTION_SQL:
        return Field(name, "float64", nullable=True)
    if operation == "cast":
        return Field(name, _cast_type_name(node), nullable=nullable)
    return _row_local_conditional_field(node, name, children, nullable)


def _any_nullable(fields: list[Field], /) -> bool:
    return any(field.nullable for field in fields)


def _row_local_conditional_field(
    node: Node,
    name: str,
    children: list[Field],
    nullable: bool,
    /,
) -> Field:
    operation = node.op.name
    if operation == "where":
        return Field(
            name,
            children[1].data_type,
            nullable=_where_result_is_nullable(node, children),
        )
    if operation == "coalesce":
        return Field(
            name,
            children[0].data_type,
            nullable=all(field.nullable for field in children),
        )
    return Field(name, children[0].data_type, nullable=nullable)


def _where_result_is_nullable(node: Node, children: list[Field], /) -> bool:
    """Mirror DataFusion's non-null proof for a directly guarded column."""

    if children[2].nullable:
        return True
    selected = node.args[1]
    condition = node.args[0]
    if selected.op.name == "column_ref" and any(
        argument.digest == selected.digest for argument in condition.args
    ):
        return False
    return children[1].nullable


def _rolling_argument_is_row_local(node: Node, /) -> bool:
    return node.op.name in _ROW_LOCAL_PRIMITIVES and all(
        _rolling_argument_is_row_local(argument) for argument in node.args
    )


def _find_ready_rolling(node: Node, /):
    """Yield innermost rolling subtrees ready for one physical stage."""

    if node.op.name in _ROLLING_PRIMITIVES:
        nested = any(True for argument in node.args for _ in _find_rolling(argument))
        if not nested:
            yield node
            return
    for argument in node.args:
        yield from _find_ready_rolling(argument)


def _rolling_frame(subtree: Node, path: str, kind: str, /) -> dict[str, object]:
    """Render the frozen frame JSON: row-count or duration (SCE-08)."""

    frame = subtree.attr("frame")
    variant = None
    if isinstance(frame, CMap):
        tag = frame.get("frame")
        if isinstance(tag, CEnum):
            variant = tag.variant
    if variant == "duration":
        micros = _cint(frame.get("micros")) if isinstance(frame, CMap) else None
        return {"kind": "duration", "micros": 1 if micros is None else micros}
    if variant != "rows":
        errors.raise_compile(
            path,
            errors.UNSUPPORTED_TYPE,
            f"rolling {kind} requires a rows or duration frame",
        )
    size = _cint(frame.get("size"))
    return {"kind": "rows", "size": 1 if size is None else size}


_FUSED_FLOAT_ROLLING_LEAVES: Final = frozenset({"mean", "variance", "stddev", "ewma"})


def _fused_difference_outputs(
    segment: _Segment, occurrences: tuple[Node, ...], /
) -> tuple[tuple[str, Node], ...]:
    """Return final ``left - right`` expressions safe for one state stage."""

    ready = {node.digest for node in occurrences}
    return tuple(
        (name, tree)
        for name, tree in segment.env
        if tree.op.name == "sub"
        and len(tree.args) == 2
        and all(
            argument.op.name in _FUSED_FLOAT_ROLLING_LEAVES and argument.digest in ready
            for argument in tree.args
        )
    )


def _rolling_input_name(
    argument: Node, materializations: dict[str, str], path: str, /
) -> str:
    if argument.op.name == "column_ref":
        return _cstr(argument.attr("name"))
    name = materializations.get(argument.digest)
    if name is None:
        errors.raise_compile(
            path,
            errors.SCHEMA_MISMATCH,
            "fused rolling input was not materialized for the state stage",
        )
    return name


def _fused_float_leaf(
    subtree: Node,
    materializations: dict[str, str],
    input_types: dict[str, Field],
    path: str,
    /,
) -> dict[str, object]:
    kind = subtree.op.name
    input_name = _rolling_input_name(subtree.args[0], materializations, path)
    if input_name not in input_types:
        errors.raise_compile(
            path,
            errors.SCHEMA_MISMATCH,
            f"rolling {kind} argument column {input_name!r} is not in the input schema",
        )
    if kind == "ewma":
        return {
            "kind": kind,
            "primitive_version": 1,
            "input": input_name,
            "span": _cint(subtree.attr("span")),
            "min_periods": _cint(subtree.attr("min_periods")) or 1,
        }
    declaration: dict[str, object] = {
        "kind": kind,
        "primitive_version": 1,
        "input": input_name,
        "frame": _rolling_frame(subtree, path, kind),
        "min_periods": _cint(subtree.attr("min_periods")) or 1,
    }
    if kind in _ROLLING_DDOF_PRIMITIVES:
        ddof = _cint(subtree.attr("ddof"))
        declaration["ddof"] = 1 if ddof is None else ddof
    return declaration


def _rolling_declaration_requires_ewma(declaration: dict[str, object], /) -> bool:
    if declaration["kind"] == "ewma":
        return True
    if declaration["kind"] != "difference":
        return False
    return any(
        isinstance(leaf, dict) and leaf.get("kind") == "ewma"
        for leaf in (declaration["left"], declaration["right"])
    )


# Rolling planning validates every lag/delta/aggregate occurrence with
# stable, declaration-ordered error paths before emitting the frozen node
# shape.
def _plan_rolling_stage(
    output_name: str,
    segment: _Segment,
    path: str,
    allowed_lateness_micros: int,
    late_policy: str,
    occurrences: tuple[Node, ...],
    input_fields: tuple[Field, ...],
    stage_number: int,
    stage_count: int,
    /,
) -> _RollingPlan:
    # #lizard forgives
    input_types = {field.name: field for field in input_fields}
    entity_by = _cstr_seq(segment.input_node.attr("entity_by"))
    sequence_by = _cstr_seq(segment.input_node.attr("sequence_by"))
    event_time = _cstr(segment.input_node.attr("event_time"))
    if not entity_by or not sequence_by or not event_time:
        errors.raise_compile(
            path,
            errors.ORDERING_REQUIRED,
            "rolling temporal primitives require declared entity_by,"
            " event_time, and sequence_by ordering keys on the input table",
        )

    whole_feature = {
        tree.digest: name
        for name, tree in segment.env
        if tree.op.name in _ROLLING_PRIMITIVES
    }
    fused_differences = _fused_difference_outputs(segment, occurrences)
    fused_leaf_digests = {
        argument.digest for _, tree in fused_differences for argument in tree.args
    }
    fused_roots = {tree.digest: name for name, tree in fused_differences}
    unfused_leaf_digests = {
        subtree.digest
        for _, tree in segment.env
        for subtree in _find_rolling(_replace_materialized(tree, fused_roots))
    }
    hidden_fused_leaf_digests = fused_leaf_digests - unfused_leaf_digests
    used_names = set(input_types) | {name for name, _ in segment.env}
    stage_fragment = "" if stage_count == 1 else f"{stage_number}_"
    stage_suffix = "" if stage_count == 1 else f"_{stage_number}"
    materialization = _plan_stateful_inputs(
        _StatefulInputRequest(
            output_name,
            path,
            f"roll_input_{stage_fragment}".removesuffix("_"),
            f"{output_name}__cf_rolling_input{stage_suffix}",
            "rolling",
        ),
        input_fields,
        used_names,
        tuple(
            (subtree.op.name, argument)
            for subtree in occurrences
            for argument in subtree.args
        ),
    )
    materializations = dict(materialization.names)
    state_input_fields = materialization.input_fields
    input_types = {field.name: field for field in state_input_fields}
    used_names = set(materialization.used_names)
    replacements: dict[str, str] = {}
    declarations: list[dict[str, object]] = []
    derived_fields: list[Field] = []
    for index, subtree in enumerate(occurrences):
        if subtree.digest in hidden_fused_leaf_digests:
            continue
        kind = subtree.op.name
        name = whole_feature.get(subtree.digest)
        if name is None:
            name = f"{output_name}__cf_roll_{stage_fragment}{index}"
            if name in used_names:
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.DUPLICATE_NAME,
                    f"materialized rolling column {name!r} collides with a"
                    " declared field",
                )
            used_names.add(name)
        replacements[subtree.digest] = name
        operands: list[tuple[str, str, Field]] = []
        for role, argument in zip(
            ("input", "left", "right"), subtree.args, strict=False
        ):
            input_name = (
                _cstr(argument.attr("name"))
                if argument.op.name == "column_ref"
                else materializations[argument.digest]
            )
            field = input_types.get(input_name)
            if field is None:
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.SCHEMA_MISMATCH,
                    f"rolling {kind} argument column {input_name!r} is not in the"
                    " input schema",
                )
            operands.append((role, input_name, field))
        periods = _cint(subtree.attr("periods"))
        if periods is not None:
            declarations.append(
                {
                    "kind": kind,
                    "primitive_version": 1,
                    "input": operands[0][1],
                    "output": name,
                    "periods": periods,
                }
            )
            derived_fields.append(Field(name, operands[0][2].data_type, nullable=True))
            continue
        if kind == "ewma":
            declarations.append(
                {
                    "kind": kind,
                    "primitive_version": 1,
                    "input": operands[0][1],
                    "output": name,
                    "span": _cint(subtree.attr("span")),
                    "min_periods": _cint(subtree.attr("min_periods")) or 1,
                }
            )
            derived_fields.append(Field(name, "float64", nullable=True))
            continue
        frame = _rolling_frame(subtree, f"{path}.{name}", kind)
        declaration: dict[str, object] = {
            "kind": kind,
            "primitive_version": 1,
            "output": name,
            "frame": frame,
            "min_periods": _cint(subtree.attr("min_periods")) or 1,
        }
        if kind in _ROLLING_PAIR_PRIMITIVES:
            declaration["left"] = operands[0][1]
            declaration["right"] = operands[1][1]
        else:
            declaration["input"] = operands[0][1]
        if kind in _ROLLING_DDOF_PRIMITIVES:
            ddof = _cint(subtree.attr("ddof"))
            declaration["ddof"] = 1 if ddof is None else ddof
        declarations.append(declaration)
        derived_fields.append(
            Field(
                name,
                _rolling_output_type(kind, operands[0][2].data_type) or "float64",
                nullable=True,
            )
        )

    for name, tree in fused_differences:
        replacements[tree.digest] = name
        declarations.append(
            {
                "kind": "difference",
                "primitive_version": 1,
                "left": _fused_float_leaf(
                    tree.args[0],
                    materializations,
                    input_types,
                    f"{path}.{name}.left",
                ),
                "right": _fused_float_leaf(
                    tree.args[1],
                    materializations,
                    input_types,
                    f"{path}.{name}.right",
                ),
                "output": name,
            }
        )
        derived_fields.append(Field(name, "float64", nullable=True))

    node_id = f"{output_name}__cf_rolling{stage_suffix}"
    node: dict[str, object] = {
        "id": node_id,
        "operator": {
            "kind": "rolling",
            "spec": {
                "configuration_version": 1,
                "state_layout_version": (
                    2
                    if any(
                        _rolling_declaration_requires_ewma(item)
                        for item in declarations
                    )
                    else 1
                ),
                "partition_by": list(entity_by),
                "event_time": event_time,
                "sequence_by": list(sequence_by),
                "outputs": declarations,
                "allowed_lateness_micros": allowed_lateness_micros,
                "late_policy": (
                    {"kind": "error", "scope": "envelope"}
                    if late_policy == "error"
                    else {"kind": "drop", "metrics_version": 1}
                ),
                "value_policy": "stateful_numeric_v1",
            },
        },
        "input_ports": [
            {
                "name": "input",
                "kind": "table",
                "required": True,
                "schema": [_field_json(field) for field in state_input_fields],
            }
        ],
        "output_ports": [
            {
                "name": "output",
                "kind": "table",
                "required": True,
                "schema": [
                    *(_field_json(field) for field in state_input_fields),
                    *(_field_json(field) for field in derived_fields),
                ],
            }
        ],
    }
    env = tuple(
        (name, _replace_materialized(tree, replacements)) for name, tree in segment.env
    )
    post_predicate = (
        None
        if segment.post_predicate is None
        else _replace_materialized(segment.post_predicate, replacements)
    )
    return _RollingPlan(
        node_id=node_id,
        node=node,
        materialization_node_id=materialization.node_id,
        materialization_node=materialization.node,
        env=env,
        post_predicate=post_predicate,
        input_field_names=(*input_types, *(field.name for field in derived_fields)),
        output_fields=(*state_input_fields, *derived_fields),
        replacements=tuple(replacements.items()),
    )


def _ready_rolling_occurrences(segment: _Segment, /) -> tuple[Node, ...]:
    ordered_trees = [tree for _, tree in segment.env]
    ordered_trees += [
        tree for tree in (segment.predicate, segment.post_predicate) if tree is not None
    ]
    occurrences: list[Node] = []
    seen: set[str] = set()
    for tree in ordered_trees:
        for subtree in _find_ready_rolling(tree):
            if subtree.digest not in seen:
                seen.add(subtree.digest)
                occurrences.append(subtree)
    return tuple(occurrences)


def _rolling_depth(node: Node, /) -> int:
    child_depth = max((_rolling_depth(argument) for argument in node.args), default=0)
    return child_depth + 1 if node.op.name in _ROLLING_PRIMITIVES else child_depth


def _rolling_stage_count(segment: _Segment, /) -> int:
    """Count the deterministic innermost-first rolling layers."""

    trees = [tree for _, tree in segment.env]
    trees += [
        tree for tree in (segment.predicate, segment.post_predicate) if tree is not None
    ]
    return max((_rolling_depth(tree) for tree in trees), default=0)


def _plan_rolling(
    output_name: str,
    segment: _Segment,
    path: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> _RollingPipeline | None:
    """Plan every innermost-first rolling layer for one output branch."""

    stage_count = _rolling_stage_count(segment)
    if stage_count == 0:
        return None
    stages: list[_RollingPlan] = []
    current = segment
    input_fields = _schema_fields(segment.input_node.attr("schema"))
    for stage_number in range(1, stage_count + 1):
        occurrences = _ready_rolling_occurrences(current)
        if not occurrences:
            raise RuntimeError("rolling stage count diverged during lowering")
        stage = _plan_rolling_stage(
            output_name,
            current,
            path,
            allowed_lateness_micros,
            late_policy,
            occurrences,
            input_fields,
            stage_number,
            stage_count,
        )
        stages.append(stage)
        current = replace(
            current,
            env=stage.env,
            post_predicate=stage.post_predicate,
        )
        input_fields = stage.output_fields
    final = stages[-1]
    return _RollingPipeline(
        tuple(stages),
        final.env,
        final.post_predicate,
        final.input_field_names,
        final.output_fields,
    )


@dataclass(frozen=True, slots=True)
class _CrossSectionPlan:
    """One lowered cross-section stage: the project node plus the rewritten
    row-local environment that references its output columns."""

    node_id: str
    node: dict[str, object]
    materialization_node_id: str | None
    materialization_node: dict[str, object] | None
    env: tuple[tuple[str, Node], ...]
    post_predicate: Node | None
    input_field_names: tuple[str, ...]
    output_fields: tuple[Field, ...]
    materializations: tuple[tuple[str, str], ...]
    replacements: tuple[tuple[str, str], ...]


def _cross_section_grouping(subtree: Node, path: str, /) -> dict[str, object]:
    """Render the frozen exact-time or fixed-bucket grouping JSON."""

    grouping = subtree.attr("grouping")
    if isinstance(grouping, CEnum) and grouping.variant == "exact_time":
        return {"kind": "exact_time"}
    if isinstance(grouping, CMap):
        tag = grouping.get("grouping")
        width = _cint(grouping.get("width_micros"))
        if isinstance(tag, CEnum) and tag.variant == "fixed_bucket" and width:
            return {"kind": "fixed_bucket", "width_micros": width}
    errors.raise_compile(
        path,
        errors.UNSUPPORTED_TYPE,
        "cross-section grouping is neither exact_time nor a fixed bucket",
    )
    raise AssertionError("unreachable")


def _enum_attr(subtree: Node, name: str, /) -> str:
    value = subtree.attr(name)
    return value.variant if isinstance(value, CEnum) else ""


def _grouping_shape(subtree: Node, /) -> tuple[str, int] | None:
    """Comparable grouping identity: the kind and, for buckets, the width."""

    grouping = subtree.attr("grouping")
    if isinstance(grouping, CEnum) and grouping.variant == "exact_time":
        return ("exact_time", 0)
    if isinstance(grouping, CMap):
        tag = grouping.get("grouping")
        width = _cint(grouping.get("width_micros"))
        if isinstance(tag, CEnum) and tag.variant == "fixed_bucket" and width:
            return ("fixed_bucket", width)
    return None


# Cross-section planning validates every occurrence with stable,
# declaration-ordered error paths before emitting the frozen node shape.
# #lizard forgives
def _plan_cross_section(
    output_name: str,
    segment: _Segment,
    path: str,
    allowed_lateness_micros: int,
    late_policy: str,
    input_fields_override: tuple[Field, ...] | None,
    /,
) -> _CrossSectionPlan | None:
    # #lizard forgives
    occurrences: list[Node] = []
    seen: set[str] = set()
    ordered_trees = [tree for _, tree in segment.env]
    ordered_trees += [
        tree for tree in (segment.predicate, segment.post_predicate) if tree is not None
    ]
    for tree in ordered_trees:
        for subtree in _find_cross_section(tree):
            if subtree.digest not in seen:
                seen.add(subtree.digest)
                occurrences.append(subtree)
    if not occurrences:
        return None

    input_fields = (
        _schema_fields(segment.input_node.attr("schema"))
        if input_fields_override is None
        else input_fields_override
    )
    input_types = {field.name: field for field in input_fields}
    entity_by = _cstr_seq(segment.input_node.attr("entity_by"))
    sequence_by = _cstr_seq(segment.input_node.attr("sequence_by"))
    event_time = _cstr(segment.input_node.attr("event_time"))
    if not entity_by or not sequence_by or not event_time:
        errors.raise_compile(
            path,
            errors.ORDERING_REQUIRED,
            "cross-section primitives require declared entity_by,"
            " event_time, and sequence_by ordering keys on the input table",
        )

    whole_feature = {
        tree.digest: name
        for name, tree in segment.env
        if tree.op.name in _CROSS_SECTION_PRIMITIVES
    }
    used_names = set(input_types) | {name for name, _ in segment.env}
    materialization = _plan_stateful_inputs(
        _StatefulInputRequest(
            output_name,
            path,
            "cs_input",
            f"{output_name}__cf_cross_section_input",
            "cross-section",
        ),
        input_fields,
        used_names,
        tuple((subtree.op.name, subtree.args[0]) for subtree in occurrences),
    )
    materializations = dict(materialization.names)
    state_input_fields = materialization.input_fields
    input_types = {field.name: field for field in state_input_fields}
    used_names = set(materialization.used_names)
    replacements: dict[str, str] = {}
    declarations: list[dict[str, object]] = []
    partition_columns: list[str] = []
    derived_fields: list[Field] = []
    for index, subtree in enumerate(occurrences):
        kind = subtree.op.name
        name = whole_feature.get(subtree.digest)
        if name is None:
            name = f"{output_name}__cf_cs_{index}"
            if name in used_names:
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.DUPLICATE_NAME,
                    f"materialized cross-section column {name!r} collides with a"
                    " declared field",
                )
            used_names.add(name)
        replacements[subtree.digest] = name
        argument = subtree.args[0]
        input_name = (
            _cstr(argument.attr("name"))
            if argument.op.name == "column_ref"
            else materializations[argument.digest]
        )
        if input_name not in input_types:
            errors.raise_compile(
                f"{path}.{name}",
                errors.SCHEMA_MISMATCH,
                f"cross-section {kind} argument column {input_name!r} is not in"
                " the input schema",
            )
        event_time_argument = subtree.args[1]
        if event_time_argument.op.name != "column_ref":
            errors.raise_compile(
                f"{path}.{name}",
                errors.UNSUPPORTED_TYPE,
                "cross-section grouping event time must be an input column in"
                " this release",
            )
        event_time_name = _cstr(event_time_argument.attr("name"))
        if event_time_name != event_time:
            errors.raise_compile(
                f"{path}.{name}",
                errors.SCHEMA_MISMATCH,
                f"cross-section grouping event time {event_time_name!r} does not"
                " match the declared input event time",
            )
        group_partitions: list[str] = []
        for group_argument in subtree.args[2:]:
            if group_argument.op.name != "column_ref":
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.UNSUPPORTED_TYPE,
                    "cross-section group columns must be input columns in this release",
                )
            group_name = _cstr(group_argument.attr("name"))
            if group_name not in input_types:
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.SCHEMA_MISMATCH,
                    f"cross-section group column {group_name!r} is not in the"
                    " input schema",
                )
            group_partitions.append(group_name)
        grouping_shape = _grouping_shape(subtree)
        if index == 0:
            partition_columns = group_partitions
            declared_grouping = grouping_shape
        elif (
            partition_columns != group_partitions or declared_grouping != grouping_shape
        ):
            errors.raise_compile(
                path,
                errors.SCHEMA_MISMATCH,
                "cross-section primitives in one output must share one"
                " grouping declaration",
            )
        declaration: dict[str, object] = {
            "kind": kind,
            "primitive_version": 1,
            "input": input_name,
            "output": name,
        }
        if kind in _CROSS_SECTION_ORDERING:
            declaration["direction"] = _enum_attr(subtree, "direction") or "ascending"
            declaration["tie_method"] = _enum_attr(subtree, "tie_method") or "average"
            declaration["null_placement"] = (
                _enum_attr(subtree, "null_placement") or "exclude"
            )
        declaration["min_samples"] = _cint(subtree.attr("min_samples")) or 1
        if kind == _CROSS_SECTION_DDOF:
            declaration["ddof"] = _cint(subtree.attr("ddof")) or 0
        if kind == "winsorize":
            declaration["lower"] = _cnumber(subtree.attr("lower"))
            declaration["upper"] = _cnumber(subtree.attr("upper"))
        if kind in ("top", "bottom"):
            declaration["count"] = _cint(subtree.attr("count"))
            declaration["include_ties"] = _cbool(subtree.attr("include_ties"))
        declarations.append(declaration)
        output_type = (
            "bool"
            if kind in ("top", "bottom")
            else input_types[input_name].data_type
            if kind in ("winsorize", "mean_fill")
            else "float64"
        )
        derived_fields.append(Field(name, output_type, nullable=True))

    node_id = f"{output_name}__cf_cross_section"
    node: dict[str, object] = {
        "id": node_id,
        "operator": {
            "kind": "cross_section",
            "spec": {
                "configuration_version": 1,
                "state_layout_version": 1,
                "event_time": event_time,
                "entity_by": list(entity_by),
                "partition_by": list(partition_columns),
                "sequence_by": list(sequence_by),
                "grouping": _cross_section_grouping(
                    occurrences[0], f"{path}.{output_name}"
                ),
                "outputs": declarations,
                "allowed_lateness_micros": allowed_lateness_micros,
                "late_policy": (
                    {"kind": "error", "scope": "envelope"}
                    if late_policy == "error"
                    else {"kind": "drop", "metrics_version": 1}
                ),
                "value_policy": "nan_exclude_preserve_v1",
            },
        },
        "input_ports": [
            {
                "name": "input",
                "kind": "table",
                "required": True,
                "schema": [_field_json(field) for field in state_input_fields],
            }
        ],
        "output_ports": [
            {
                "name": "output",
                "kind": "table",
                "required": True,
                "schema": [
                    *(_field_json(field) for field in state_input_fields),
                    *(_field_json(field) for field in derived_fields),
                ],
            }
        ],
    }
    env = tuple(
        (name, _replace_materialized(tree, replacements)) for name, tree in segment.env
    )
    post_predicate = (
        None
        if segment.post_predicate is None
        else _replace_materialized(segment.post_predicate, replacements)
    )
    return _CrossSectionPlan(
        node_id,
        node,
        materialization.node_id,
        materialization.node,
        env,
        post_predicate,
        (*input_types, *(field.name for field in derived_fields)),
        (*state_input_fields, *derived_fields),
        materialization.names,
        tuple(replacements.items()),
    )


def _shared_materialized_name(
    prefix: str, counter: int, reserved: set[str], /
) -> tuple[str, int]:
    while True:
        name = f"__cf_shared_{prefix}_{counter}"
        counter += 1
        if name not in reserved:
            reserved.add(name)
            return name, counter


def _plan_outputs(plan: _RollingPlan | _CrossSectionPlan, /) -> list[dict[str, object]]:
    return plan.node["operator"]["spec"]["outputs"]  # type: ignore[index,return-value]


def _required_state_plan[StatePlanT: (_RollingPlan, _CrossSectionPlan)](
    plans: dict[str, StatePlanT | None], output_name: str, /
) -> StatePlanT:
    plan = plans[output_name]
    if plan is None:
        raise RuntimeError(f"missing shared-state plan for output {output_name!r}")
    return plan


def _merge_state_outputs[StatePlanT: (_RollingPlan, _CrossSectionPlan)](
    members: list[str],
    plans: dict[str, StatePlanT | None],
    prefix: str,
    /,
) -> tuple[StatePlanT, dict[str, str], list[dict[str, object]], tuple[Field, ...]]:
    first = _required_state_plan(plans, members[0])
    base_count = len(first.output_fields) - len(first.replacements)
    base_fields = first.output_fields[:base_count]
    reserved = {
        field.name
        for output_name in members
        for field in _required_state_plan(plans, output_name).output_fields
    }
    replacements: dict[str, str] = {}
    declarations: list[dict[str, object]] = []
    derived_fields: list[Field] = []
    counter = 0
    for output_name in members:
        plan = _required_state_plan(plans, output_name)
        declarations_by_name = {item["output"]: item for item in _plan_outputs(plan)}
        fields_by_name = {field.name: field for field in plan.output_fields}
        for digest, old_name in plan.replacements:
            if digest in replacements:
                continue
            new_name, counter = _shared_materialized_name(prefix, counter, reserved)
            replacements[digest] = new_name
            declarations.append({**declarations_by_name[old_name], "output": new_name})
            field = fields_by_name[old_name]
            derived_fields.append(
                Field(new_name, field.data_type, nullable=field.nullable)
            )
    return first, replacements, declarations, (*base_fields, *derived_fields)


def _shared_state_node(
    plan: _RollingPlan | _CrossSectionPlan,
    node_id: str,
    declarations: list[dict[str, object]],
    output_fields: tuple[Field, ...],
    /,
) -> dict[str, object]:
    operator = plan.node["operator"]
    spec = operator["spec"]  # type: ignore[index]
    return {
        **plan.node,
        "id": node_id,
        "operator": {
            **operator,  # type: ignore[arg-type]
            "spec": {**spec, "outputs": declarations},  # type: ignore[arg-type]
        },
        "output_ports": [
            {
                "name": "output",
                "kind": "table",
                "required": True,
                "schema": [_field_json(field) for field in output_fields],
            }
        ],
    }


def _unique_shared_node_id(stem: str, reserved: set[str], /) -> str:
    candidate = stem
    counter = 1
    while candidate in reserved:
        candidate = f"{stem}_{counter}"
        counter += 1
    reserved.add(candidate)
    return candidate


def _reserved_state_node_ids[StatePlanT: (_RollingPlan, _CrossSectionPlan)](
    segments: list[tuple[str, _Segment]],
    plans: dict[str, StatePlanT | None],
    /,
) -> set[str]:
    reserved = {output_name for output_name, _ in segments}
    reserved.update(_cstr(segment.input_node.attr("name")) for _, segment in segments)
    reserved.update(plan.node_id for plan in plans.values() if plan is not None)
    return reserved


def _shared_group_plans[StatePlanT: (_RollingPlan, _CrossSectionPlan)](
    members: list[str],
    by_name: dict[str, _Segment],
    first: StatePlanT,
    replacements: dict[str, str],
    node_id: str,
    node: dict[str, object],
    output_fields: tuple[Field, ...],
    /,
) -> dict[str, StatePlanT]:
    shared: dict[str, StatePlanT] = {}
    for output_name in members:
        segment = by_name[output_name]
        post_predicate = segment.post_predicate
        if post_predicate is not None:
            post_predicate = _replace_materialized(post_predicate, replacements)
        shared[output_name] = replace(
            first,
            node_id=node_id,
            node=node,
            env=tuple(
                (name, _replace_materialized(tree, replacements))
                for name, tree in segment.env
            ),
            post_predicate=post_predicate,
            input_field_names=tuple(field.name for field in output_fields),
            output_fields=output_fields,
            replacements=tuple(replacements.items()),
        )
    return shared


def _share_state_groups[StatePlanT: (_RollingPlan, _CrossSectionPlan)](
    groups: list[list[str]],
    segments: list[tuple[str, _Segment]],
    plans: dict[str, StatePlanT | None],
    prefix: str,
    suffix: str,
    /,
) -> dict[str, StatePlanT | None]:
    shared = dict(plans)
    by_name = dict(segments)
    reserved_ids = _reserved_state_node_ids(segments, plans)
    for members in groups:
        if len(members) < 2:
            continue
        first, replacements, declarations, output_fields = _merge_state_outputs(
            members, plans, prefix
        )
        node_id = _unique_shared_node_id(
            f"{members[0]}__cf_shared_{suffix}", reserved_ids
        )
        node = _shared_state_node(first, node_id, declarations, output_fields)
        shared.update(
            _shared_group_plans(
                members,
                by_name,
                first,
                replacements,
                node_id,
                node,
                output_fields,
            )
        )
    return shared


def _share_single_rolling_stages(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPlan | None],
    /,
) -> dict[str, _RollingPlan | None]:
    groups: dict[tuple[str, str | None, str | None], list[str]] = {}
    for output_name, segment in segments:
        plan = plans[output_name]
        if plan is not None:
            predicate = None if segment.predicate is None else segment.predicate.digest
            materialization = (
                None
                if plan.materialization_node is None
                else _canonical(plan.materialization_node)
            )
            groups.setdefault(
                (segment.input_node.digest, predicate, materialization), []
            ).append(output_name)
    return _share_state_groups(
        list(groups.values()), segments, plans, "roll", "rolling"
    )


def _share_rolling_plans(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPipeline | None],
    /,
) -> dict[str, _RollingPipeline | None]:
    """Preserve existing cross-output sharing for one-stage pipelines."""

    stages = {
        output_name: (
            None
            if pipeline is None or len(pipeline.stages) != 1
            else pipeline.stages[0]
        )
        for output_name, pipeline in plans.items()
    }
    shared_stages = _share_single_rolling_stages(segments, stages)
    shared = dict(plans)
    for output_name, stage in shared_stages.items():
        pipeline = plans[output_name]
        if pipeline is None or stage is None:
            continue
        shared[output_name] = _RollingPipeline(
            (stage,),
            stage.env,
            stage.post_predicate,
            stage.input_field_names,
            stage.output_fields,
        )
    return _share_identical_multi_stage_pipelines(segments, shared)


def _rolling_pipeline_identity(
    segment: _Segment, /
) -> tuple[str, str | None, tuple[str, ...]]:
    predicate = None if segment.predicate is None else segment.predicate.digest
    seen: set[str] = set()
    digests: list[str] = []
    trees = [tree for _, tree in segment.env]
    trees += [tree for tree in (segment.post_predicate,) if tree is not None]
    for tree in trees:
        for subtree in _find_rolling(tree):
            if subtree.digest not in seen:
                seen.add(subtree.digest)
                digests.append(subtree.digest)
    return segment.input_node.digest, predicate, tuple(digests)


def _shared_pipeline_stages(
    first_name: str,
    pipeline: _RollingPipeline,
    reserved_ids: set[str],
    /,
) -> tuple[_RollingPlan, ...]:
    stages: list[_RollingPlan] = []
    for index, stage in enumerate(pipeline.stages, start=1):
        node_id = _unique_shared_node_id(
            f"{first_name}__cf_shared_rolling_{index}", reserved_ids
        )
        materialization_id = None
        materialization = None
        if stage.materialization_node is not None:
            materialization_id = _unique_shared_node_id(
                f"{first_name}__cf_shared_rolling_input_{index}", reserved_ids
            )
            materialization = {
                **stage.materialization_node,
                "id": materialization_id,
            }
        stages.append(
            replace(
                stage,
                node_id=node_id,
                node={**stage.node, "id": node_id},
                materialization_node_id=materialization_id,
                materialization_node=materialization,
            )
        )
    return tuple(stages)


def _rewrite_pipeline_environment(
    segment: _Segment,
    stages: tuple[_RollingPlan, ...],
    /,
) -> tuple[tuple[tuple[str, Node], ...], Node | None]:
    env = segment.env
    post_predicate = segment.post_predicate
    for stage in stages:
        replacements = dict(stage.replacements)
        env = tuple(
            (name, _replace_materialized(tree, replacements)) for name, tree in env
        )
        if post_predicate is not None:
            post_predicate = _replace_materialized(post_predicate, replacements)
    return env, post_predicate


def _multi_stage_pipeline_groups(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPipeline | None],
    /,
) -> tuple[tuple[str, ...], ...]:
    groups: dict[tuple[str, str | None, tuple[str, ...]], list[str]] = {}
    for output_name, segment in segments:
        pipeline = plans[output_name]
        if pipeline is not None and len(pipeline.stages) > 1:
            groups.setdefault(_rolling_pipeline_identity(segment), []).append(
                output_name
            )
    return tuple(tuple(members) for members in groups.values() if len(members) > 1)


def _reserved_rolling_pipeline_ids(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPipeline | None],
    /,
) -> set[str]:
    reserved_ids = {output_name for output_name, _ in segments}
    for pipeline in plans.values():
        if pipeline is None:
            continue
        for stage in pipeline.stages:
            reserved_ids.add(stage.node_id)
            if stage.materialization_node_id is not None:
                reserved_ids.add(stage.materialization_node_id)
    return reserved_ids


def _shared_multi_stage_group(
    members: tuple[str, ...],
    segments: dict[str, _Segment],
    plans: dict[str, _RollingPipeline | None],
    reserved_ids: set[str],
    /,
) -> dict[str, _RollingPipeline]:
    first = plans[members[0]]
    if first is None:
        raise RuntimeError("missing multi-stage rolling pipeline")
    stages = _shared_pipeline_stages(members[0], first, reserved_ids)
    shared: dict[str, _RollingPipeline] = {}
    for output_name in members:
        env, post_predicate = _rewrite_pipeline_environment(
            segments[output_name], stages
        )
        shared[output_name] = _RollingPipeline(
            stages,
            env,
            post_predicate,
            first.input_field_names,
            first.output_fields,
        )
    return shared


def _share_identical_multi_stage_pipelines(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPipeline | None],
    /,
) -> dict[str, _RollingPipeline | None]:
    groups = _multi_stage_pipeline_groups(segments, plans)
    reserved_ids = _reserved_rolling_pipeline_ids(segments, plans)
    segments_by_name = dict(segments)
    shared = dict(plans)
    for members in groups:
        shared.update(
            _shared_multi_stage_group(
                members,
                segments_by_name,
                plans,
                reserved_ids,
            )
        )
    return shared


def _cross_section_group_identity(plan: _CrossSectionPlan, /) -> str:
    spec = plan.node["operator"]["spec"]  # type: ignore[index]
    return _canonical({key: value for key, value in spec.items() if key != "outputs"})


def _materialized_select_expression(plan: _CrossSectionPlan, name: str, /) -> str:
    node = plan.materialization_node
    if node is None:
        raise RuntimeError(f"missing cross-section materialization for {name!r}")
    suffix = f" AS {_quote_identifier(name)}"
    selects = node["operator"]["select"]  # type: ignore[index]
    for item in selects:
        if item.endswith(suffix):
            return item[: -len(suffix)]
    raise RuntimeError(f"missing materialized select for {name!r}")


@dataclass(frozen=True, slots=True)
class _SharedCrossInputs:
    base_fields: tuple[Field, ...]
    fields: tuple[Field, ...]
    expressions: tuple[str, ...]
    names: tuple[tuple[str, str], ...]

    @property
    def input_fields(self) -> tuple[Field, ...]:
        return (*self.base_fields, *self.fields)


def _collect_shared_cross_inputs(
    members: list[str],
    plans: dict[str, _CrossSectionPlan | None],
    /,
) -> _SharedCrossInputs:
    first = _required_state_plan(plans, members[0])
    base_count = (
        len(first.output_fields) - len(first.replacements) - len(first.materializations)
    )
    reserved = {
        field.name
        for output_name in members
        for field in _required_state_plan(plans, output_name).output_fields
    }
    names: dict[str, str] = {}
    fields: list[Field] = []
    expressions: list[str] = []
    counter = 0
    for output_name in members:
        plan = _required_state_plan(plans, output_name)
        fields_by_name = {field.name: field for field in plan.output_fields}
        for digest, old_name in plan.materializations:
            if digest in names:
                continue
            name, counter = _shared_materialized_name("cs_input", counter, reserved)
            names[digest] = name
            field = fields_by_name[old_name]
            fields.append(Field(name, field.data_type, nullable=field.nullable))
            expressions.append(_materialized_select_expression(plan, old_name))
    return _SharedCrossInputs(
        first.output_fields[:base_count],
        tuple(fields),
        tuple(expressions),
        tuple(names.items()),
    )


def _shared_cross_materialization(
    shared: _SharedCrossInputs, node_id: str, /
) -> dict[str, object] | None:
    if not shared.fields:
        return None
    return _expression_node(
        node_id,
        [
            *(_quote_identifier(field.name) for field in shared.base_fields),
            *(
                f"{expression} AS {_quote_identifier(field.name)}"
                for expression, field in zip(
                    shared.expressions, shared.fields, strict=True
                )
            ),
        ],
        None,
        shared.base_fields,
        shared.input_fields,
    )


def _aligned_cross_section_plan(
    plan: _CrossSectionPlan,
    shared: _SharedCrossInputs,
    node_id: str,
    materialization: dict[str, object] | None,
    /,
) -> _CrossSectionPlan:
    names = dict(shared.names)
    old_to_new = {old_name: names[digest] for digest, old_name in plan.materializations}
    outputs = [
        {**item, "input": old_to_new.get(item["input"], item["input"])}
        for item in _plan_outputs(plan)
    ]
    derived_fields = plan.output_fields[-len(plan.replacements) :]
    output_fields = (*shared.input_fields, *derived_fields)
    return replace(
        plan,
        node={
            **plan.node,
            "operator": {
                **plan.node["operator"],  # type: ignore[dict-item]
                "spec": {
                    **plan.node["operator"]["spec"],  # type: ignore[index]
                    "outputs": outputs,
                },
            },
            "input_ports": [
                {
                    "name": "input",
                    "kind": "table",
                    "required": True,
                    "schema": [_field_json(field) for field in shared.input_fields],
                }
            ],
            "output_ports": [
                {
                    "name": "output",
                    "kind": "table",
                    "required": True,
                    "schema": [_field_json(field) for field in output_fields],
                }
            ],
        },
        materialization_node_id=node_id if shared.fields else None,
        materialization_node=materialization,
        input_field_names=tuple(field.name for field in output_fields),
        output_fields=output_fields,
        materializations=shared.names,
    )


def _combined_cross_section_inputs(
    members: list[str],
    plans: dict[str, _CrossSectionPlan | None],
    node_id: str,
    /,
) -> dict[str, _CrossSectionPlan]:
    shared = _collect_shared_cross_inputs(members, plans)
    materialization = _shared_cross_materialization(shared, node_id)
    return {
        output_name: _aligned_cross_section_plan(
            _required_state_plan(plans, output_name),
            shared,
            node_id,
            materialization,
        )
        for output_name in members
    }


def _share_cross_section_plans(
    segments: list[tuple[str, _Segment]],
    upstream_ids: dict[str, str | None],
    plans: dict[str, _CrossSectionPlan | None],
    /,
) -> dict[str, _CrossSectionPlan | None]:
    groups: dict[tuple[str, str | None, str], list[str]] = {}
    for output_name, segment in segments:
        plan = plans[output_name]
        if plan is not None:
            upstream = upstream_ids[output_name] or segment.input_node.digest
            predicate = None if segment.predicate is None else segment.predicate.digest
            groups.setdefault(
                (upstream, predicate, _cross_section_group_identity(plan)), []
            ).append(output_name)
    aligned = dict(plans)
    reserved_ids = _reserved_state_node_ids(segments, plans)
    for group in groups.values():
        if len(group) < 2:
            continue
        materialization_id = _unique_shared_node_id(
            f"{group[0]}__cf_shared_cross_section_input", reserved_ids
        )
        aligned.update(_combined_cross_section_inputs(group, plans, materialization_id))
    return _share_state_groups(
        list(groups.values()), segments, aligned, "cs", "cross_section"
    )


def _check_declared_inputs(program: Program, /) -> None:
    for value in program.inputs:
        node = value._node
        if node.op.name == "parameter":
            errors.raise_compile(
                f"static_inputs.{_cstr(node.attr('name'))}",
                errors.UNKNOWN_PRIMITIVE_VERSION,
                "static parameters are not supported by the row-local lowerer",
            )


@dataclass(frozen=True, slots=True)
class _MatrixExpression:
    backend: str
    columns: tuple[str, ...]
    source_digests: frozenset[str]
    parameter: Node | None
    weights_count: int
    matmul_count: int
    matmul_rhs_is_weights: bool
    tree: dict[str, object]


@dataclass(frozen=True, slots=True)
class _LoweringValue:
    _node: Node


@dataclass(frozen=True, slots=True)
class _LoweringProgram:
    name: str
    inputs: tuple[_LoweringValue, ...]
    outputs: tuple[tuple[str, _LoweringValue], ...]


def _matrix_literal(node: Node, path: str, /) -> bool | int | float:
    value = node.attr("value")
    if isinstance(value, (CBool, CInt, CFloat)):
        return value.value
    errors.raise_compile(
        path,
        errors.UNSUPPORTED_TYPE,
        "symbolic matrix literals must be finite bool, int, or float values",
    )


def _matrix_backend(
    left: _MatrixExpression,
    right: _MatrixExpression,
    path: str,
    /,
) -> str:
    if left.backend and right.backend and left.backend != right.backend:
        errors.raise_compile(
            path,
            errors.CAPABILITY_MISMATCH,
            "symbolic matrix operands must use one provider backend",
        )
    return left.backend or right.backend


def _matrix_columns(
    left: _MatrixExpression,
    right: _MatrixExpression,
    path: str,
    /,
) -> tuple[str, ...]:
    if left.columns and right.columns and left.columns != right.columns:
        errors.raise_compile(
            path,
            errors.SCHEMA_MISMATCH,
            "symbolic matrix operands must use one ordered column selection",
        )
    return left.columns or right.columns


def _matrix_parameter(
    left: _MatrixExpression,
    right: _MatrixExpression,
    path: str,
    /,
) -> Node | None:
    if (
        left.parameter is not None
        and right.parameter is not None
        and left.parameter.digest != right.parameter.digest
    ):
        errors.raise_compile(
            path,
            errors.CAPABILITY_MISMATCH,
            "one symbolic matrix output supports exactly one static parameter",
        )
    return left.parameter or right.parameter


def _merge_matrix_expression(
    left: _MatrixExpression,
    right: _MatrixExpression,
    path: str,
    /,
) -> tuple[str, tuple[str, ...], Node | None]:
    return (
        _matrix_backend(left, right, path),
        _matrix_columns(left, right, path),
        _matrix_parameter(left, right, path),
    )


def _matrix_leaf_expression(
    node: Node,
    path: str,
    operation: str,
    /,
) -> _MatrixExpression | None:
    if operation == "from_columns":
        return _MatrixExpression(
            _cstr(node.attr("backend")),
            _cstr_seq(node.attr("columns")),
            frozenset({node.args[0].digest}),
            None,
            0,
            0,
            True,
            {"op": "input"},
        )
    if operation == "parameter":
        name = _cstr(node.attr("name"))
        if name != "weights":
            errors.raise_compile(
                f"static_inputs.{name}",
                errors.CAPABILITY_MISMATCH,
                "symbolic matrix lowering currently requires the static array"
                " parameter name 'weights'",
            )
        return _MatrixExpression(
            _cstr(node.attr("backend")),
            (),
            frozenset(),
            node,
            1,
            0,
            True,
            {"op": "weights"},
        )
    if operation == "literal":
        return _MatrixExpression(
            "",
            (),
            frozenset(),
            None,
            0,
            0,
            True,
            {"op": "literal", "value": _matrix_literal(node, path)},
        )
    return None


def _matrix_unary_expression(
    node: Node,
    path: str,
    operation: str,
    /,
) -> _MatrixExpression:
    value = _matrix_expression(node.args[0], f"{path}.{operation}.value")
    return _MatrixExpression(
        value.backend,
        value.columns,
        value.source_digests,
        value.parameter,
        value.weights_count,
        value.matmul_count,
        value.matmul_rhs_is_weights,
        {"op": operation, "value": value.tree},
    )


def _matrix_rhs_is_weights(node: Node, operation: str, /) -> bool:
    return operation != "matmul" or (
        node.args[1].op.name == "parameter"
        and _cstr(node.args[1].attr("name")) == "weights"
    )


def _matrix_binary_expression(
    node: Node,
    path: str,
    operation: str,
    /,
) -> _MatrixExpression:
    left = _matrix_expression(node.args[0], f"{path}.{operation}.left")
    right = _matrix_expression(node.args[1], f"{path}.{operation}.right")
    backend, columns, parameter = _merge_matrix_expression(left, right, path)
    return _MatrixExpression(
        backend,
        columns,
        left.source_digests | right.source_digests,
        parameter,
        left.weights_count + right.weights_count,
        left.matmul_count + right.matmul_count + (operation == "matmul"),
        left.matmul_rhs_is_weights
        and right.matmul_rhs_is_weights
        and _matrix_rhs_is_weights(node, operation),
        {"left": left.tree, "op": operation, "right": right.tree},
    )


def _matrix_expression(node: Node, path: str, /) -> _MatrixExpression:
    operation = node.op.name
    leaf = _matrix_leaf_expression(node, path, operation)
    if leaf is not None:
        return leaf
    if operation not in _MATRIX_PRIMITIVES:
        _reject_primitive(path, node)
    if operation in ("neg", "not"):
        return _matrix_unary_expression(node, path, operation)
    return _matrix_binary_expression(node, path, operation)


def _static_array_declaration(node: Node, /) -> dict[str, object]:
    dtype = node.attr("dtype")
    shape = node.attr("shape")
    return {
        "backend": _cstr(node.attr("backend")),
        "dtype": dtype.name if isinstance(dtype, CDType) else "",
        "kind": "array",
        "mutability": "static",
        "name": _cstr(node.attr("name")),
        "shape": [
            dimension.value
            for dimension in (shape.items if isinstance(shape, CSeq) else ())
            if isinstance(dimension, CInt)
        ],
    }


def _matrix_program_output(
    program: Program | _LoweringProgram,
    /,
) -> tuple[str, Node] | None:
    if len(program.outputs) != 1:
        return None
    output_name, value = program.outputs[0]
    if value._node.op.name != "attach_columns":
        return None
    return output_name, value._node


def _required_matrix_expression(node: Node, output_name: str, /) -> _MatrixExpression:
    path = f"outputs.{output_name}.array"
    matrix = _matrix_expression(node.args[1], path)
    if not matrix.backend or not matrix.columns:
        errors.raise_compile(
            path,
            errors.UNRESOLVED_TYPE,
            "symbolic matrix output requires linalg.from_columns",
        )
    return matrix


def _required_matrix_parameter(
    matrix: _MatrixExpression,
    output_name: str,
    /,
) -> Node:
    parameter = matrix.parameter
    if parameter is None:
        errors.raise_compile(
            f"outputs.{output_name}.array",
            errors.UNRESOLVED_TYPE,
            "symbolic matrix output requires one static array parameter",
        )
    if _cstr(parameter.attr("name")) != "weights":
        errors.raise_compile(
            f"outputs.{output_name}.array",
            errors.UNRESOLVED_TYPE,
            "symbolic matrix output requires one static array parameter",
        )
    return parameter


def _require_frozen_matrix_shape(
    matrix: _MatrixExpression,
    output_name: str,
    /,
) -> None:
    if (
        matrix.weights_count != 1
        or matrix.matmul_count != 1
        or not matrix.matmul_rhs_is_weights
    ):
        errors.raise_compile(
            f"outputs.{output_name}.array",
            errors.CAPABILITY_MISMATCH,
            "symbolic matrix output requires exactly one matmul whose right"
            " operand is the static 'weights' parameter",
        )


def _attached_matrix_input(
    node: Node,
    matrix: _MatrixExpression,
    output_name: str,
    /,
) -> Node:
    attached_input = node.args[0]
    if matrix.source_digests != frozenset({attached_input.digest}):
        errors.raise_compile(
            f"outputs.{output_name}.value",
            errors.SCHEMA_MISMATCH,
            "attached matrix columns must come from the attached table",
        )
    return attached_input


def _matrix_external_node(
    output_name: str,
    node: Node,
    matrix: _MatrixExpression,
    /,
) -> dict[str, object]:
    return {
        "id": output_name,
        "input_ports": [
            {"kind": "table", "name": "input", "required": True},
            {"kind": "array", "name": "weights", "required": True},
        ],
        "operator": {
            "kind": "external",
            "name": "symbolic_matrix",
            "options": {
                "columns": list(matrix.columns),
                "expression": matrix.tree,
                "names": list(_cstr_seq(node.attr("names"))),
            },
            "provider": matrix.backend,
            "version": "1",
        },
        "output_ports": [{"kind": "table", "name": "output", "required": True}],
    }


def _matrix_upstream_project(
    program: Program | _LoweringProgram,
    attached_input: Node,
    upstream_id: str,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object]:
    upstream = _LoweringProgram(
        program.name,
        tuple(
            _LoweringValue(item._node)
            for item in program.inputs
            if item._node.op.name == "table_input"
        ),
        ((upstream_id, _LoweringValue(attached_input)),),
    )
    return _lower_program(
        upstream,
        mode,
        allowed_lateness_micros,
        late_policy,
    )


def _raise_lowering_invariant(message: str, /) -> Never:
    raise RuntimeError(f"symbolic lowering invariant violated: {message}")


def _project_graph_lists(
    project: dict[str, object],
    /,
) -> tuple[list[object], list[object]]:
    graph = project.get("graph")
    if not isinstance(graph, dict):
        _raise_lowering_invariant("project.graph must be a mapping")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        _raise_lowering_invariant("project.graph.nodes must be a list")
    edges = graph.get("edges")
    if not isinstance(edges, list):
        _raise_lowering_invariant("project.graph.edges must be a list")
    return nodes, edges


def _project_document(
    name: str,
    mode: str,
    nodes: list[object],
    edges: list[object],
    /,
) -> dict[str, object]:
    project: dict[str, object] = {
        "data_sources": [],
        "format_version": 3,
        "id": name,
        "name": name,
        "runtime": {"mode": mode, "options": {}},
        "graph": {"edges": edges, "name": name, "nodes": nodes},
    }
    project["data_sources"] = [] if mode == "stream" else _data_sources(project)
    return project


def _wire_matrix_node(
    project: dict[str, object],
    external: dict[str, object],
    upstream_id: str,
    output_name: str,
    /,
) -> None:
    nodes, edges = _project_graph_lists(project)
    nodes.append(external)
    edges.append(
        {
            "source_node": upstream_id,
            "source_port": "output",
            "target_node": output_name,
            "target_port": "input",
        }
    )


def _lower_matrix_program(
    program: Program | _LoweringProgram,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object] | None:
    output = _matrix_program_output(program)
    if output is None:
        return None
    output_name, node = output
    matrix = _required_matrix_expression(node, output_name)
    parameter = _required_matrix_parameter(matrix, output_name)
    _require_frozen_matrix_shape(matrix, output_name)
    attached_input = _attached_matrix_input(node, matrix, output_name)
    external = _matrix_external_node(output_name, node, matrix)
    upstream_id = f"{output_name}__cf_matrix_input"
    project = _matrix_upstream_project(
        program,
        attached_input,
        upstream_id,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    _wire_matrix_node(project, external, upstream_id, output_name)
    if mode == "stream":
        project["static_inputs"] = [_static_array_declaration(parameter)]
    else:
        project["data_sources"] = _data_sources(project)
    return project


def _stream_join_nodes(program: Program, /) -> tuple[Node, ...]:
    joins: dict[str, Node] = {}
    for _, value in program.outputs:
        for node in _walk_nodes(value._node):
            if node.op.name == "stream_join":
                joins[node.digest] = node
    return tuple(joins[digest] for digest in sorted(joins))


def _contains_node(root: Node, digest: str, /) -> bool:
    return any(node.digest == digest for node in _walk_nodes(root))


def _contains_primitive(root: Node, primitive: str, /) -> bool:
    return any(node.op.name == primitive for node in _walk_nodes(root))


def _replace_node(root: Node, digest: str, replacement: Node, /) -> Node:
    if root.digest == digest:
        return replacement
    return build(
        root.op.name,
        tuple(_replace_node(child, digest, replacement) for child in root.args),
        dict(root.attrs.entries),
        version=root.op.version,
    )


def _stream_join_node(
    node: Node,
    node_id: str,
    left_schema: tuple[Field, ...],
    right_schema: tuple[Field, ...],
    /,
) -> dict[str, object]:
    return {
        "id": node_id,
        "input_ports": [
            {
                "kind": "table",
                "name": "left",
                "required": True,
                "schema": [_field_json(field) for field in left_schema],
            },
            {
                "kind": "table",
                "name": "right",
                "required": True,
                "schema": [_field_json(field) for field in right_schema],
            },
        ],
        "operator": {
            "kind": "stream_join",
            "spec": {
                "join_type": "inner",
                "left_keys": list(_cstr_seq(node.attr("left_keys"))),
                "right_keys": list(_cstr_seq(node.attr("right_keys"))),
                "left_event_time": _cstr(node.attr("left_event_time")),
                "right_event_time": _cstr(node.attr("right_event_time")),
                "bounds": {
                    "before_micros": _cint(node.attr("before_micros")),
                    "after_micros": _cint(node.attr("after_micros")),
                },
                "limits": {
                    "max_state_rows_per_side": _cint(
                        node.attr("max_state_rows_per_side")
                    ),
                    "max_state_bytes_per_side": _cint(
                        node.attr("max_state_bytes_per_side")
                    ),
                    "max_matches_per_input_batch": _cint(
                        node.attr("max_matches_per_input_batch")
                    ),
                },
                "left_prefix": _cstr(node.attr("left_prefix")),
                "right_prefix": _cstr(node.attr("right_prefix")),
            },
        },
        "output_ports": [],
    }


@dataclass(frozen=True, slots=True)
class _StreamJoinSide:
    port: str
    node_id: str
    node: Node
    schema: tuple[Field, ...]


@dataclass(frozen=True, slots=True)
class _StreamJoinPlan:
    node: Node
    node_id: str
    sides: tuple[_StreamJoinSide, _StreamJoinSide]
    output_schema: tuple[Field, ...]


@dataclass(frozen=True, slots=True)
class _RelationalFragment:
    boundary: Node
    project: dict[str, object] | None


def _connected_input_endpoints(edges: list[object], /) -> set[tuple[str, str]]:
    connected: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        connected.add((str(edge["target_node"]), str(edge.get("target_port", "input"))))
    return connected


def _required_input_port_names(node: dict[str, object], /) -> tuple[str, ...]:
    ports = node.get("input_ports")
    if not isinstance(ports, list):
        return ("input",)
    if not ports:
        return ("input",)
    names: list[str] = []
    for port in ports:
        if not isinstance(port, dict):
            continue
        if port.get("required") is True:
            names.append(str(port["name"]))
    return tuple(names)


def _graph_node(node: object, /) -> dict[str, object]:
    if not isinstance(node, dict):
        _raise_lowering_invariant("graph node must be a mapping")
    return node


def _downstream_input_endpoints(
    nodes: list[object], edges: list[object], /
) -> tuple[tuple[str, str], ...]:
    connected = _connected_input_endpoints(edges)
    endpoints: list[tuple[str, str]] = []
    for raw_node in nodes:
        node = _graph_node(raw_node)
        node_id = str(node["id"])
        for name in _required_input_port_names(node):
            endpoint = (node_id, name)
            if endpoint not in connected:
                endpoints.append(endpoint)
    return tuple(sorted(endpoints))


def _pin_table_output(
    nodes: list[object], node_id: str, schema: tuple[Field, ...], /
) -> None:
    for node in nodes:
        if isinstance(node, dict) and node.get("id") == node_id:
            node["output_ports"] = [
                {
                    "kind": "table",
                    "name": "output",
                    "required": True,
                    "schema": [_field_json(field) for field in schema],
                }
            ]
            return
    _raise_lowering_invariant(f"missing stream join input stage {node_id!r}")


def _required_stream_join(program: Program, joins: tuple[Node, ...], /) -> Node:
    if len(joins) != 1:
        errors.raise_compile(
            program.name,
            errors.CAPABILITY_MISMATCH,
            "SCE-17 supports exactly one unique symbolic stream join per program",
        )
    return joins[0]


def _check_stream_join_outputs(program: Program, join: Node, /) -> None:
    for output_name, value in program.outputs:
        if _contains_node(value._node, join.digest):
            continue
        errors.raise_compile(
            f"outputs.{output_name}",
            errors.CAPABILITY_MISMATCH,
            "every output in a symbolic stream-join program must descend"
            " from its one shared join",
        )


def _check_stream_join_inputs(program: Program, join: Node, /) -> None:
    for side_name, side in zip(("left", "right"), join.args, strict=True):
        if not _contains_primitive(side, "attach_columns"):
            continue
        errors.raise_compile(
            f"{program.name}.stream_join.{side_name}",
            errors.CAPABILITY_MISMATCH,
            "matrix attachment around a symbolic stream join is not supported",
        )


def _stream_join_plan(
    program: Program, analyzer: _Analyzer, join: Node, /
) -> _StreamJoinPlan:
    left_facts = analyzer.table(join.args[0], f"{program.name}.stream_join.left")
    right_facts = analyzer.table(join.args[1], f"{program.name}.stream_join.right")
    join_facts = analyzer.table(join, f"{program.name}.stream_join")
    join_id = f"cf_stream_join_{join.digest[:16]}"
    return _StreamJoinPlan(
        join,
        join_id,
        (
            _StreamJoinSide(
                "left",
                f"{join_id}__left",
                join.args[0],
                left_facts.schema,
            ),
            _StreamJoinSide(
                "right",
                f"{join_id}__right",
                join.args[1],
                right_facts.schema,
            ),
        ),
        join_facts.schema,
    )


def _stream_join_upstream_outputs(
    plan: _StreamJoinPlan, /
) -> tuple[tuple[str, _LoweringValue], ...]:
    return tuple(
        (side.node_id, _LoweringValue(side.node))
        for side in plan.sides
        if side.node.op.name != "table_input"
    )


def _input_reaches_join_side(
    input_node: Node, upstream_sides: tuple[Node, ...], /
) -> bool:
    return any(_contains_node(side, input_node.digest) for side in upstream_sides)


def _stream_join_upstream_inputs(
    program: Program,
    upstream_outputs: tuple[tuple[str, _LoweringValue], ...],
    /,
) -> tuple[_LoweringValue, ...]:
    upstream_sides = tuple(value._node for _, value in upstream_outputs)
    inputs: list[_LoweringValue] = []
    for value in program.inputs:
        if value._node.op.name != "table_input":
            continue
        if _input_reaches_join_side(value._node, upstream_sides):
            inputs.append(_LoweringValue(value._node))
    return tuple(inputs)


def _stream_join_upstream_project(
    program: Program,
    plan: _StreamJoinPlan,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object]:
    outputs = _stream_join_upstream_outputs(plan)
    if not outputs:
        return _project_document(program.name, mode, [], [])
    inputs = _stream_join_upstream_inputs(program, outputs)
    return _lower_program(
        _LoweringProgram(program.name, inputs, outputs),
        mode,
        allowed_lateness_micros,
        late_policy,
    )


def _wire_stream_join_inputs(
    nodes: list[object], edges: list[object], plan: _StreamJoinPlan, /
) -> None:
    for side in plan.sides:
        if side.node.op.name == "table_input":
            continue
        _pin_table_output(nodes, side.node_id, side.schema)
        edges.append(
            {
                "source_node": side.node_id,
                "source_port": "output",
                "target_node": plan.node_id,
                "target_port": side.port,
            }
        )


def _direct_stream_join_output(program: Program, plan: _StreamJoinPlan, /) -> bool:
    if len(program.outputs) != 1:
        return False
    return program.outputs[0][1]._node.digest == plan.node.digest


def _wire_stream_join_downstream(
    program: Program,
    plan: _StreamJoinPlan,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    upstream_nodes: list[object],
    upstream_edges: list[object],
    /,
) -> None:
    from calc_flow.symbolic.expr import table_input

    virtual_id = f"{plan.node_id}__output"
    virtual = table_input(virtual_id, schema=plan.output_schema)
    downstream = _LoweringProgram(
        program.name,
        (_LoweringValue(virtual._node),),
        tuple(
            (
                output_name,
                _LoweringValue(
                    _replace_node(value._node, plan.node.digest, virtual._node)
                ),
            )
            for output_name, value in program.outputs
        ),
    )
    downstream_project = _lower_program(
        downstream,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    downstream_nodes, downstream_edges = _project_graph_lists(downstream_project)
    endpoints = _downstream_input_endpoints(downstream_nodes, downstream_edges)
    if not endpoints:
        _raise_lowering_invariant("stream join downstream has no input endpoint")
    upstream_nodes.extend(downstream_nodes)
    upstream_edges.extend(downstream_edges)
    upstream_edges.extend(
        {
            "source_node": plan.node_id,
            "source_port": "output",
            "target_node": target_node,
            "target_port": target_port,
        }
        for target_node, target_port in endpoints
    )


def _lower_stream_join_program(
    program: Program,
    analyzer: _Analyzer,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object] | None:
    joins = _stream_join_nodes(program)
    if not joins:
        return None
    if _requires_relational_dag_lowering(program, joins):
        return _lower_relational_dag_program(
            program,
            analyzer,
            mode,
            allowed_lateness_micros,
            late_policy,
            joins,
        )
    join = _required_stream_join(program, joins)
    _check_stream_join_outputs(program, join)
    _check_stream_join_inputs(program, join)
    plan = _stream_join_plan(program, analyzer, join)
    upstream_project = _stream_join_upstream_project(
        program,
        plan,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    upstream_nodes, upstream_edges = _project_graph_lists(upstream_project)
    _wire_stream_join_inputs(upstream_nodes, upstream_edges, plan)
    upstream_nodes.append(
        _stream_join_node(
            plan.node,
            plan.node_id,
            plan.sides[0].schema,
            plan.sides[1].schema,
        )
    )
    if _direct_stream_join_output(program, plan):
        return upstream_project
    _wire_stream_join_downstream(
        program,
        plan,
        mode,
        allowed_lateness_micros,
        late_policy,
        upstream_nodes,
        upstream_edges,
    )
    return upstream_project


def _requires_relational_dag_lowering(
    program: Program, joins: tuple[Node, ...], /
) -> bool:
    if len(joins) != 1 or joins[0].op.version >= 2:
        return True
    join = joins[0]
    return any(
        not _contains_node(value._node, join.digest) for _, value in program.outputs
    )


def _relational_boundary(node: Node, path: str, /) -> Node:
    current = node
    while current.op.name in ("project", "filter", "with_columns"):
        current = current.args[0]
    if current.op.name in ("table_input", "stream_join"):
        return current
    if _contains_primitive(current, "attach_columns"):
        errors.raise_compile(
            path,
            errors.CAPABILITY_MISMATCH,
            "matrix attachment around a symbolic stream join is not supported",
        )
    _reject_primitive(path, current)


def _virtual_relational_input(node_id: str, facts: TableFacts, /) -> Node:
    from calc_flow.symbolic.expr import table_input

    return table_input(
        node_id,
        schema=facts.schema,
        entity_by=facts.entity_by,
        event_time=facts.event_time,
        sequence_by=facts.sequence_by,
    )._node


def _reachable_relational_sources(program: Program, /) -> frozenset[str]:
    return frozenset(
        node.digest
        for _, value in program.outputs
        for node in _walk_nodes(value._node)
        if node.op.name == "table_input"
    )


def _relational_source_name(node: Node, reserved_ids: frozenset[str], /) -> str:
    declared_name = _cstr(node.attr("name"))
    if declared_name is None:
        _raise_lowering_invariant("relational source is missing its declared name")
    if declared_name in reserved_ids:
        return f"cf_source_{node.digest[:16]}"
    return declared_name


def _relational_source_nodes(
    program: Program, reserved_ids: frozenset[str], /
) -> tuple[dict[str, str], list[dict[str, object]]]:
    reachable = _reachable_relational_sources(program)
    by_digest: dict[str, str] = {}
    nodes: list[dict[str, object]] = []
    for value in program.inputs:
        node = value._node
        if node.op.name != "table_input" or node.digest not in reachable:
            continue
        name = _relational_source_name(node, reserved_ids)
        schema = _schema_fields(node.attr("schema"))
        by_digest[node.digest] = name
        nodes.append(
            _expression_node(
                name,
                [_quote_identifier(field.name) for field in schema],
                None,
                schema,
                schema,
            )
        )
    return by_digest, nodes


def _relational_upstream_id(
    boundary: Node,
    sources: dict[str, str],
    joins: dict[str, _StreamJoinPlan],
    path: str,
    /,
) -> str:
    if boundary.op.name == "table_input":
        source_id = sources.get(boundary.digest)
        if source_id is None:
            _raise_lowering_invariant(
                f"missing declared source for relational boundary at {path}"
            )
        return source_id
    plan = joins.get(boundary.digest)
    if plan is None:
        _raise_lowering_invariant(
            f"missing physical join for relational boundary at {path}"
        )
    return plan.node_id


def _relational_fragment(
    program: Program,
    analyzer: _Analyzer,
    expression: Node,
    boundary: Node,
    output_id: str,
    path: str,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object]:
    if boundary.op.name == "stream_join":
        facts = analyzer.table(boundary, f"{path}.boundary")
        declared = _virtual_relational_input(
            f"cf_join_output_{boundary.digest[:16]}", facts
        )
        lowered = _replace_node(expression, boundary.digest, declared)
    else:
        declared = boundary
        lowered = expression
    return _lower_program(
        _LoweringProgram(
            program.name,
            (_LoweringValue(declared),),
            ((output_id, _LoweringValue(lowered)),),
        ),
        mode,
        allowed_lateness_micros,
        late_policy,
    )


def _relational_output_fragments(
    program: Program,
    analyzer: _Analyzer,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, _RelationalFragment]:
    fragments: dict[str, _RelationalFragment] = {}
    for output_name, value in program.outputs:
        path = f"outputs.{output_name}"
        boundary = _relational_boundary(value._node, path)
        fragments[output_name] = _RelationalFragment(
            boundary,
            _relational_fragment(
                program,
                analyzer,
                value._node,
                boundary,
                output_name,
                path,
                mode,
                allowed_lateness_micros,
                late_policy,
            ),
        )
    return fragments


def _relational_join_fragments(
    program: Program,
    analyzer: _Analyzer,
    plans: dict[str, _StreamJoinPlan],
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[tuple[str, str], _RelationalFragment]:
    fragments: dict[tuple[str, str], _RelationalFragment] = {}
    for plan in plans.values():
        for side in plan.sides:
            path = f"{program.name}.{plan.node_id}.{side.port}"
            boundary = _relational_boundary(side.node, path)
            project = None
            if side.node.digest != boundary.digest:
                project = _relational_fragment(
                    program,
                    analyzer,
                    side.node,
                    boundary,
                    side.node_id,
                    path,
                    mode,
                    allowed_lateness_micros,
                    late_policy,
                )
            fragments[(plan.node.digest, side.port)] = _RelationalFragment(
                boundary,
                project,
            )
    return fragments


def _wire_relational_fragment(
    nodes: list[object],
    edges: list[object],
    fragment: dict[str, object],
    upstream_id: str,
    output_id: str,
    output_schema: tuple[Field, ...] | None,
    /,
) -> None:
    fragment_nodes, fragment_edges = _project_graph_lists(fragment)
    endpoints = _downstream_input_endpoints(fragment_nodes, fragment_edges)
    if len(endpoints) != 1:
        _raise_lowering_invariant(
            f"relational fragment {output_id!r} has {len(endpoints)} input endpoints"
        )
    if output_schema is not None:
        _pin_table_output(fragment_nodes, output_id, output_schema)
    nodes.extend(fragment_nodes)
    edges.extend(fragment_edges)
    target_node, target_port = endpoints[0]
    edges.append(
        {
            "source_node": upstream_id,
            "source_port": "output",
            "target_node": target_node,
            "target_port": target_port,
        }
    )


def _wire_relational_join_side(
    program: Program,
    plan: _StreamJoinPlan,
    side: _StreamJoinSide,
    fragment: _RelationalFragment,
    sources: dict[str, str],
    joins: dict[str, _StreamJoinPlan],
    nodes: list[object],
    edges: list[object],
    /,
) -> None:
    path = f"{program.name}.{plan.node_id}.{side.port}"
    upstream_id = _relational_upstream_id(fragment.boundary, sources, joins, path)
    if fragment.project is None:
        edges.append(
            {
                "source_node": upstream_id,
                "source_port": "output",
                "target_node": plan.node_id,
                "target_port": side.port,
            }
        )
        return
    _wire_relational_fragment(
        nodes,
        edges,
        fragment.project,
        upstream_id,
        side.node_id,
        side.schema,
    )
    edges.append(
        {
            "source_node": side.node_id,
            "source_port": "output",
            "target_node": plan.node_id,
            "target_port": side.port,
        }
    )


def _wire_relational_output(
    output_name: str,
    fragment: _RelationalFragment,
    sources: dict[str, str],
    joins: dict[str, _StreamJoinPlan],
    nodes: list[object],
    edges: list[object],
    /,
) -> None:
    path = f"outputs.{output_name}"
    upstream_id = _relational_upstream_id(fragment.boundary, sources, joins, path)
    if fragment.project is None:
        _raise_lowering_invariant(f"relational output {output_name!r} has no fragment")
    _wire_relational_fragment(
        nodes,
        edges,
        fragment.project,
        upstream_id,
        output_name,
        None,
    )


def _relational_join_plans(
    program: Program,
    analyzer: _Analyzer,
    join_nodes: tuple[Node, ...],
    /,
) -> dict[str, _StreamJoinPlan]:
    for join in join_nodes:
        _check_stream_join_inputs(program, join)
    return {
        join.digest: _stream_join_plan(program, analyzer, join) for join in join_nodes
    }


def _relational_reserved_ids(
    program: Program,
    plans: dict[str, _StreamJoinPlan],
    join_fragments: dict[tuple[str, str], _RelationalFragment],
    output_fragments: dict[str, _RelationalFragment],
    /,
) -> frozenset[str]:
    reserved = {
        *(output_name for output_name, _ in program.outputs),
        *(plan.node_id for plan in plans.values()),
        *(side.node_id for plan in plans.values() for side in plan.sides),
    }
    for fragment in (*join_fragments.values(), *output_fragments.values()):
        if fragment.project is None:
            continue
        nodes, _ = _project_graph_lists(fragment.project)
        reserved.update(str(_graph_node(node)["id"]) for node in nodes)
    return frozenset(reserved)


def _append_relational_joins(
    program: Program,
    join_nodes: tuple[Node, ...],
    sources: dict[str, str],
    plans: dict[str, _StreamJoinPlan],
    fragments: dict[tuple[str, str], _RelationalFragment],
    nodes: list[object],
    edges: list[object],
    /,
) -> None:
    for join in join_nodes:
        plan = plans[join.digest]
        nodes.append(
            _stream_join_node(
                plan.node,
                plan.node_id,
                plan.sides[0].schema,
                plan.sides[1].schema,
            )
        )
        for side in plan.sides:
            _wire_relational_join_side(
                program,
                plan,
                side,
                fragments[(plan.node.digest, side.port)],
                sources,
                plans,
                nodes,
                edges,
            )


def _append_relational_outputs(
    program: Program,
    sources: dict[str, str],
    plans: dict[str, _StreamJoinPlan],
    fragments: dict[str, _RelationalFragment],
    nodes: list[object],
    edges: list[object],
    /,
) -> None:
    for output_name, _ in program.outputs:
        _wire_relational_output(
            output_name,
            fragments[output_name],
            sources,
            plans,
            nodes,
            edges,
        )


def _lower_relational_dag_program(
    program: Program,
    analyzer: _Analyzer,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    join_nodes: tuple[Node, ...],
    /,
) -> dict[str, object]:
    plans = _relational_join_plans(program, analyzer, join_nodes)
    join_fragments = _relational_join_fragments(
        program,
        analyzer,
        plans,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    output_fragments = _relational_output_fragments(
        program,
        analyzer,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    reserved_ids = _relational_reserved_ids(
        program,
        plans,
        join_fragments,
        output_fragments,
    )
    sources, source_nodes = _relational_source_nodes(program, reserved_ids)
    nodes: list[object] = list(source_nodes)
    edges: list[object] = []
    _append_relational_joins(
        program,
        join_nodes,
        sources,
        plans,
        join_fragments,
        nodes,
        edges,
    )
    _append_relational_outputs(
        program,
        sources,
        plans,
        output_fragments,
        nodes,
        edges,
    )
    typed_nodes = [_graph_node(node) for node in nodes]
    typed_edges = [_graph_node(edge) for edge in edges]
    typed_nodes, typed_edges = _deduplicate_node_ids(typed_nodes, typed_edges)
    return _project_document(program.name, mode, typed_nodes, typed_edges)


def _required_segment_state_plan(
    rolling: _RollingPipeline | None, cross: _CrossSectionPlan | None, /
) -> _RollingPipeline | _CrossSectionPlan:
    plan = rolling if rolling is not None else cross
    if plan is None:
        raise RuntimeError("missing state plan for a stateful symbolic segment")
    return plan


def _walk_nodes(root: Node, /) -> tuple[Node, ...]:
    nodes: list[Node] = []

    def visit(node: Node) -> None:
        nodes.append(node)
        for child in node.args:
            visit(child)

    visit(root)
    return tuple(nodes)


def _deduplicated_expression_signature(
    node: dict[str, object],
    incoming: dict[str, list[dict[str, object]]],
    aliases: dict[str, str],
    output_ids: frozenset[str],
    /,
) -> str | None:
    node_id = str(node["id"])
    operator = node["operator"]
    if not isinstance(operator, dict):
        return None
    if operator.get("kind") != "expression" or node_id in output_ids:
        return None
    node_incoming = incoming.get(node_id, [])
    if len(node_incoming) != 1:
        return None
    edge = node_incoming[0]
    source_node = str(edge["source_node"])
    source = aliases.get(source_node, source_node)
    return _canonical(
        {
            "input_ports": node.get("input_ports", []),
            "operator": operator,
            "output_ports": node.get("output_ports", []),
            "source_node": source,
            "source_port": edge.get("source_port", "output"),
            "target_port": edge.get("target_port", "input"),
        }
    )


def _resolve_node_alias(node_id: str, aliases: dict[str, str], /) -> str:
    while node_id in aliases:
        node_id = aliases[node_id]
    return node_id


def _rewrite_deduplicated_edges(
    edges: list[dict[str, object]], aliases: dict[str, str], /
) -> list[dict[str, object]]:
    rewritten: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for edge in edges:
        target = str(edge["target_node"])
        if target in aliases:
            continue
        source = _resolve_node_alias(str(edge["source_node"]), aliases)
        normalized = {**edge, "source_node": source, "target_node": target}
        identity = (
            source,
            str(normalized.get("source_port", "output")),
            target,
            str(normalized.get("target_port", "input")),
        )
        if identity in seen:
            continue
        seen.add(identity)
        rewritten.append(normalized)
    return rewritten


def _deduplicate_pure_expression_nodes(
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
    output_ids: frozenset[str],
    /,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Share identical connected expression stages across output branches."""

    incoming: dict[str, list[dict[str, object]]] = {}
    for edge in edges:
        incoming.setdefault(str(edge["target_node"]), []).append(edge)
    aliases: dict[str, str] = {}
    signatures: dict[str, str] = {}
    kept: list[dict[str, object]] = []
    for node in nodes:
        node_id = str(node["id"])
        signature = _deduplicated_expression_signature(
            node, incoming, aliases, output_ids
        )
        if signature is None:
            kept.append(node)
            continue
        canonical = signatures.get(signature)
        if canonical is None:
            signatures[signature] = node_id
            kept.append(node)
        else:
            aliases[node_id] = canonical
    return kept, _rewrite_deduplicated_edges(edges, aliases)


def _deduplicate_node_ids(
    nodes: list[dict[str, object]], edges: list[dict[str, object]], /
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collapse repeated references to one already-shared physical stage."""

    by_id: dict[str, str] = {}
    unique_nodes: list[dict[str, object]] = []
    for node in nodes:
        node_id = str(node["id"])
        canonical = _canonical(node)
        existing = by_id.get(node_id)
        if existing is None:
            by_id[node_id] = canonical
            unique_nodes.append(node)
        elif existing != canonical:
            raise RuntimeError(
                f"symbolic optimizer emitted conflicting node id {node_id!r}"
            )
    unique_edges: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for edge in edges:
        identity = (
            str(edge["source_node"]),
            str(edge.get("source_port", "output")),
            str(edge["target_node"]),
            str(edge.get("target_port", "input")),
        )
        if identity not in seen:
            seen.add(identity)
            unique_edges.append(edge)
    return unique_nodes, unique_edges


# The lowerer keeps per-output segment staging in one deterministic pass:
# stage order, edge wiring, and id assignment are semantic, so the rolling,
# prefilter, and CSE stages stay in one place.
def _lower_program(
    program: Program | _LoweringProgram,
    mode: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> dict[str, object]:
    # #lizard forgives
    matrix_project = _lower_matrix_program(
        program,
        mode,
        allowed_lateness_micros,
        late_policy,
    )
    if matrix_project is not None:
        return matrix_project
    _check_declared_inputs(program)
    segments = []
    for output_name, value in program.outputs:
        node = value._node
        path = f"outputs.{output_name}"
        if node.op.name not in _TABLE_OUTPUT_PRIMITIVES:
            _reject_primitive(path, node)
        segments.append((output_name, _resolve_table(node, path)))
    consumed = {segment.input_node.digest for _, segment in segments}
    multi_output_lineages = {
        digest
        for digest in consumed
        if sum(1 for _, segment in segments if segment.input_node.digest == digest) > 1
    }
    fanout = len(program.inputs) > 1 or bool(multi_output_lineages)
    plans = {
        output_name: _plan_rolling(
            output_name,
            segment,
            f"outputs.{output_name}",
            allowed_lateness_micros,
            late_policy,
        )
        for output_name, segment in segments
    }
    plans = _share_rolling_plans(segments, plans)
    rolling_digests = {
        segment.input_node.digest
        for (output_name, segment), plan in zip(segments, plans.values(), strict=True)
        if plan is not None
    }
    cross_plans: dict[str, _CrossSectionPlan | None] = {}
    cross_segments: list[tuple[str, _Segment]] = []
    for (output_name, segment), rolling in zip(segments, plans.values(), strict=True):
        if rolling is not None:
            # Cross-section planning runs over the rolling-rewritten
            # environment so a measured value may be a materialized rolling
            # output column.
            after_rolling = _Segment(
                segment.input_node,
                segment.fields,
                rolling.env,
                None,
                rolling.post_predicate,
            )
            cross_segments.append((output_name, after_rolling))
            cross_plans[output_name] = _plan_cross_section(
                output_name,
                after_rolling,
                f"outputs.{output_name}",
                allowed_lateness_micros,
                late_policy,
                rolling.output_fields,
            )
        else:
            cross_segments.append((output_name, segment))
            cross_plans[output_name] = _plan_cross_section(
                output_name,
                segment,
                f"outputs.{output_name}",
                allowed_lateness_micros,
                late_policy,
                None,
            )
    cross_plans = _share_cross_section_plans(
        cross_segments,
        {
            output_name: None
            if plans[output_name] is None
            else plans[output_name].node_id
            for output_name, _ in segments
        },
        cross_plans,
    )
    direct_cross_section_digests = {
        segment.input_node.digest
        for (output_name, segment), rolling, cross in zip(
            segments, plans.values(), cross_plans.values(), strict=True
        )
        if rolling is None and cross is not None
    }
    shared_plan_counts: dict[str, int] = {}
    for plan in (*plans.values(), *cross_plans.values()):
        if plan is not None:
            shared_plan_counts[plan.node_id] = (
                shared_plan_counts.get(plan.node_id, 0) + 1
            )
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    fanout_ids: dict[str, str] = {}
    if fanout:
        for value in program.inputs:
            input_node = value._node
            if input_node.digest not in consumed:
                continue
            input_name = _cstr(input_node.attr("name"))
            schema = _schema_fields(input_node.attr("schema"))
            pinned = (
                input_node.digest in rolling_digests
                or input_node.digest in direct_cross_section_digests
            )
            nodes.append(
                _expression_node(
                    input_name,
                    [_quote_identifier(field.name) for field in schema],
                    None,
                    schema,
                    schema if pinned else None,
                )
            )
            fanout_ids[input_node.digest] = input_name
    for output_name, segment in segments:
        rolling = plans[output_name]
        cross = cross_plans[output_name]
        env = dict(segment.env)
        input_field_names = [
            field.name for field in _schema_fields(segment.input_node.attr("schema"))
        ]
        upstream_id: str | None = None
        final_predicate = segment.predicate
        if (rolling is not None or cross is not None) and segment.predicate is not None:
            state_plan = _required_segment_state_plan(rolling, cross)
            prefilter_id = (
                f"{state_plan.node_id}__prefilter"
                if shared_plan_counts[state_plan.node_id] > 1
                else f"{output_name}__cf_prefilter"
            )
            input_fields = _schema_fields(segment.input_node.attr("schema"))
            nodes.append(
                _expression_node(
                    prefilter_id,
                    [_quote_identifier(name) for name in input_field_names],
                    _sql(segment.predicate),
                    input_fields,
                    input_fields,
                )
            )
            if fanout:
                edges.append(
                    {
                        "source_node": fanout_ids[segment.input_node.digest],
                        "source_port": "output",
                        "target_node": prefilter_id,
                        "target_port": "input",
                    }
                )
            upstream_id = prefilter_id
        if rolling is not None:
            for stage in rolling.stages:
                if stage.materialization_node is not None:
                    nodes.append(stage.materialization_node)
                    materialization_id = stage.materialization_node_id
                    if materialization_id is None:
                        raise RuntimeError("rolling materialization node has no id")
                    if upstream_id is not None:
                        edges.append(
                            {
                                "source_node": upstream_id,
                                "source_port": "output",
                                "target_node": materialization_id,
                                "target_port": "input",
                            }
                        )
                    elif fanout:
                        edges.append(
                            {
                                "source_node": fanout_ids[segment.input_node.digest],
                                "source_port": "output",
                                "target_node": materialization_id,
                                "target_port": "input",
                            }
                        )
                    upstream_id = materialization_id
                nodes.append(stage.node)
                if upstream_id is not None:
                    edges.append(
                        {
                            "source_node": upstream_id,
                            "source_port": "output",
                            "target_node": stage.node_id,
                            "target_port": "input",
                        }
                    )
                elif fanout:
                    edges.append(
                        {
                            "source_node": fanout_ids[segment.input_node.digest],
                            "source_port": "output",
                            "target_node": stage.node_id,
                            "target_port": "input",
                        }
                    )
                upstream_id = stage.node_id
            env = dict(rolling.env)
            input_field_names = list(rolling.input_field_names)
            final_predicate = rolling.post_predicate
        if cross is not None:
            if cross.materialization_node is not None:
                nodes.append(cross.materialization_node)
                materialization_id = cross.materialization_node_id
                if materialization_id is None:
                    raise RuntimeError("cross-section materialization node has no id")
                if upstream_id is not None:
                    edges.append(
                        {
                            "source_node": upstream_id,
                            "source_port": "output",
                            "target_node": materialization_id,
                            "target_port": "input",
                        }
                    )
                elif fanout:
                    edges.append(
                        {
                            "source_node": fanout_ids[segment.input_node.digest],
                            "source_port": "output",
                            "target_node": materialization_id,
                            "target_port": "input",
                        }
                    )
                upstream_id = materialization_id
            nodes.append(cross.node)
            if upstream_id is not None:
                edges.append(
                    {
                        "source_node": upstream_id,
                        "source_port": "output",
                        "target_node": cross.node_id,
                        "target_port": "input",
                    }
                )
            elif fanout:
                edges.append(
                    {
                        "source_node": fanout_ids[segment.input_node.digest],
                        "source_port": "output",
                        "target_node": cross.node_id,
                        "target_port": "input",
                    }
                )
            upstream_id = cross.node_id
            env = dict(cross.env)
            input_field_names = list(cross.input_field_names)
            final_predicate = cross.post_predicate
        elif rolling is None and segment.post_predicate is not None:
            final_predicate = (
                segment.post_predicate
                if segment.predicate is None
                else build("and", (segment.predicate, segment.post_predicate), {})
            )
        reserved = frozenset(env)
        fused = extract_common(
            tuple((field, env[field]) for field in segment.fields),
            final_predicate,
            reserved,
        )
        cse_order = [name for tier in fused.tiers for name, _ in tier]
        needed: set[str] = set()
        for _, tree in fused.selects:
            needed |= expression_refs(tree)
        if fused.predicate is not None:
            needed |= expression_refs(fused.predicate)
        tier_items: list[list[str]] = []
        for index in range(len(fused.tiers) - 1, -1, -1):
            tier = fused.tiers[index]
            defined = {name for name, _ in tier}
            passthrough = needed - defined
            items = [
                _quote_identifier(field)
                for field in input_field_names
                if field in passthrough
            ]
            items += [
                _quote_identifier(name) for name in cse_order if name in passthrough
            ]
            items += [
                f"{_sql(tree)} AS {_quote_identifier(name)}" for name, tree in tier
            ]
            tier_items.append(items)
            needed = set(passthrough)
            for _, tree in tier:
                needed |= expression_refs(tree)
        tier_items.reverse()
        stage_ids = [
            f"{output_name}__cf_cse_{index}" for index in range(1, len(fused.tiers) + 1)
        ] + [output_name]
        stage_selects = [
            *tier_items,
            [_select_item(field, tree) for field, tree in fused.selects],
        ]
        for position, (node_id, select) in enumerate(
            zip(stage_ids, stage_selects, strict=True)
        ):
            filter_sql = None
            if position == len(stage_ids) - 1 and fused.predicate is not None:
                filter_sql = _sql(fused.predicate)
            if position == 0:
                if upstream_id is not None:
                    edges.append(
                        {
                            "source_node": upstream_id,
                            "source_port": "output",
                            "target_node": node_id,
                            "target_port": "input",
                        }
                    )
                    input_schema = (
                        cross.output_fields
                        if cross is not None
                        else rolling.output_fields
                        if rolling is not None
                        else None
                    )
                elif fanout:
                    edges.append(
                        {
                            "source_node": fanout_ids[segment.input_node.digest],
                            "source_port": "output",
                            "target_node": node_id,
                            "target_port": "input",
                        }
                    )
                    input_schema = None
                else:
                    input_schema = _schema_fields(segment.input_node.attr("schema"))
            else:
                edges.append(
                    {
                        "source_node": stage_ids[position - 1],
                        "source_port": "output",
                        "target_node": node_id,
                        "target_port": "input",
                    }
                )
                input_schema = None
            nodes.append(_expression_node(node_id, select, filter_sql, input_schema))
    nodes, edges = _deduplicate_node_ids(nodes, edges)
    nodes, edges = _deduplicate_pure_expression_nodes(
        nodes,
        edges,
        frozenset(output_name for output_name, _ in program.outputs),
    )
    return _project_document(program.name, mode, nodes, edges)


def _require_runtime(runtime: object, entry: str, /) -> Runtime:
    if not isinstance(runtime, Runtime):
        raise TypeError(
            f"{entry} requires an explicit calc_flow Runtime; got {type_name(runtime)}"
        )
    return runtime


def _check_expression_capability(
    program: Program,
    runtime: Runtime,
    mode: str,
    /,
) -> tuple[_Analyzer, RuntimeCapabilities]:
    analyzer, capabilities = _run(program, runtime, mode)
    issues = analyzer.issues
    if issues:
        first = issues[0]
        errors.raise_compile(first.path, first.code, first.message)
    for operator in capabilities.operators:
        if operator.kind != "expression":
            continue
        if mode not in operator.modes:
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                f"the expression operator does not support {mode} mode in the"
                " selected capability snapshot",
            )
        if mode == "stream" and (
            operator.finality == "unproven"
            or not operator.microbatch_invariant
            or not operator.deterministic
            or not operator.replay_safe
        ):
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                "the expression operator does not prove stream lifecycle facts"
                " in the selected capability snapshot",
            )
        return analyzer, capabilities
    errors.raise_compile(
        program.name,
        errors.CAPABILITY_MISMATCH,
        "the capability snapshot does not offer the expression operator",
    )


def _program_needs_rolling(program: Program, /) -> bool:
    return any(True for _, value in program.outputs for _ in _find_rolling(value._node))


def _program_needs_cross_section(program: Program, /) -> bool:
    return any(
        True for _, value in program.outputs for _ in _find_cross_section(value._node)
    )


def _program_needs_stream_join(program: Program, /) -> bool:
    return bool(_stream_join_nodes(program))


def _stream_join_ports(operator: object, /) -> bool:
    inputs = tuple(
        (port.name, port.kind, port.required) for port in operator.input_ports
    )
    outputs = tuple(
        (port.name, port.kind, port.required) for port in operator.output_ports
    )
    return inputs == (("left", "table", True), ("right", "table", True)) and (
        outputs == (("output", "table", True),)
    )


def _positive_state_version(value: object, /) -> bool:
    return isinstance(value, int) and value > 0


def _stream_join_capability_facts(operator: object, mode: str, /) -> tuple[bool, ...]:
    return (
        operator.version == "1",
        mode in operator.modes,
        _stream_join_ports(operator),
        operator.requires_watermark,
        operator.stateful,
        operator.checkpoint_support == "checkpointed_stateful",
        _positive_state_version(operator.state_version),
        operator.deterministic,
        operator.replay_safe,
    )


def _check_stream_join_capability(
    program: Program,
    capabilities: RuntimeCapabilities,
    mode: str,
    /,
) -> None:
    for operator in capabilities.operators:
        if operator.kind != "stream_join":
            continue
        if not all(_stream_join_capability_facts(operator, mode)):
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                "the stream_join operator does not prove the required stream"
                " ports, watermark, checkpoint, determinism, and replay facts",
            )
        return
    errors.raise_compile(
        program.name,
        errors.CAPABILITY_MISMATCH,
        "the capability snapshot does not offer stream_join@1",
    )


# The cross-section gate mirrors the rolling gate over the frozen
# group-final stream lifecycle facts.
def _cross_section_checkpoint_capability(operator: object, /) -> bool:
    return (
        operator.stateful
        and operator.checkpoint_support == "checkpointed_stateful"
        and isinstance(operator.state_version, int)
        and operator.state_version > 0
    )


def _cross_section_stream_capability(operator: object, /) -> bool:
    return (
        operator.finality != "unproven"
        and operator.microbatch_invariant
        and _cross_section_checkpoint_capability(operator)
        and operator.deterministic
        and operator.replay_safe
    )


def _check_cross_section_capability(
    program: Program,
    capabilities: object,
    mode: str,
    /,
) -> None:
    for operator in capabilities.operators:
        if operator.kind != "cross_section":
            continue
        if mode not in operator.modes:
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                f"the cross-section operator does not support {mode} mode in"
                " the selected capability snapshot",
            )
        if mode == "stream" and not _cross_section_stream_capability(operator):
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                "the cross-section operator does not prove stream lifecycle"
                " facts in the selected capability snapshot",
            )
        return
    errors.raise_compile(
        program.name,
        errors.CAPABILITY_MISMATCH,
        "the capability snapshot does not offer the cross-section operator",
    )


# The capability gate conjoins the frozen stream lifecycle facts; every
# fact fails with the same stable capability_mismatch code.
def _check_rolling_capability(
    program: Program,
    capabilities: object,
    mode: str,
    /,
) -> None:
    # #lizard forgives
    for operator in capabilities.operators:
        if operator.kind != "rolling":
            continue
        if mode not in operator.modes:
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                f"the rolling operator does not support {mode} mode in the"
                " selected capability snapshot",
            )
        if mode == "stream" and (
            operator.finality == "unproven"
            or not operator.stateful
            or not operator.microbatch_invariant
            or operator.checkpoint_support != "checkpointed_stateful"
            or not isinstance(operator.state_version, int)
            or operator.state_version <= 0
            or not operator.deterministic
            or not operator.replay_safe
        ):
            errors.raise_compile(
                program.name,
                errors.CAPABILITY_MISMATCH,
                "the rolling operator does not prove stream lifecycle facts"
                " in the selected capability snapshot",
            )
        return
    errors.raise_compile(
        program.name,
        errors.CAPABILITY_MISMATCH,
        "the capability snapshot does not offer the rolling operator",
    )


def lower_program_document(
    program: Program,
    runtime: Runtime,
    mode: str,
    /,
    *,
    allowed_lateness_micros: int = 0,
    late_policy: str = "error",
) -> dict[str, object]:
    """Analyze and lower one program to its strict project-v3 document.

    The lateness arguments are validated whenever the program contains
    rolling or cross-section primitives; row-local programs do not consume
    them.
    """

    selected = _require_runtime(runtime, "lower_program_document")
    mode_value = _require_mode(mode)
    analyzer, capabilities = _check_expression_capability(program, selected, mode_value)
    if _program_needs_stream_join(program):
        _check_stream_join_capability(program, capabilities, mode_value)
    if _program_needs_rolling(program) or _program_needs_cross_section(program):
        _validate_lateness(allowed_lateness_micros, late_policy)
        if _program_needs_rolling(program):
            _check_rolling_capability(program, capabilities, mode_value)
        if _program_needs_cross_section(program):
            _check_cross_section_capability(program, capabilities, mode_value)
    join_project = _lower_stream_join_program(
        program,
        analyzer,
        mode_value,
        allowed_lateness_micros,
        late_policy,
    )
    if join_project is not None:
        return join_project
    return _lower_program(program, mode_value, allowed_lateness_micros, late_policy)


def _cache_graph_nodes(document: dict[str, object], /) -> list[dict[str, object]]:
    graph = document["graph"]
    return graph["nodes"]  # type: ignore[index,return-value]


def _cache_operator_versions(
    nodes: list[dict[str, object]], capabilities: RuntimeCapabilities, /
) -> tuple[tuple[str, str], ...]:
    operator_kinds = {
        node["operator"]["kind"]  # type: ignore[index]
        for node in nodes
        if node["operator"]["kind"] != "external"  # type: ignore[index]
    }
    return tuple(
        sorted(
            (operator.kind, operator.version)
            for operator in capabilities.operators
            if operator.kind in operator_kinds
        )
    )


def _cache_provider_versions(
    nodes: list[dict[str, object]], /
) -> tuple[tuple[str, str, str], ...]:
    versions: set[tuple[str, str, str]] = set()
    for node in nodes:
        operator = node["operator"]
        if operator["kind"] == "external":  # type: ignore[index]
            versions.add(  # type: ignore[arg-type]
                (operator["provider"], operator["name"], operator["version"])  # type: ignore[index]
            )
    return tuple(sorted(versions))


def _cache_udf_versions(
    nodes: list[dict[str, object]], /
) -> tuple[tuple[str, str, str], ...]:
    versions: set[tuple[str, str, str]] = set()
    for node in nodes:
        operator = node["operator"]
        for udf in operator.get("udfs", ()):  # type: ignore[union-attr]
            if isinstance(udf, dict):
                versions.add((udf["provider"], udf["name"], udf["version"]))  # type: ignore[arg-type]
    return tuple(sorted(versions))


def _compile_cache_key(
    program: Program,
    mode: str,
    document: dict[str, object],
    capabilities: RuntimeCapabilities,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> _CompileCacheKey:
    nodes = _cache_graph_nodes(document)
    return _CompileCacheKey(
        program_fingerprint=program.fingerprint,
        mode=mode,
        input_declarations=tuple(
            value._node.node_bytes.hex() for value in program.inputs
        ),
        capability_schema_version=capabilities.schema_version,
        capability_session_id=capabilities.scope.session_id,
        capability_revision=capabilities.scope.revision,
        operator_versions=_cache_operator_versions(nodes, capabilities),
        provider_versions=_cache_provider_versions(nodes),
        udf_versions=_cache_udf_versions(nodes),
        allowed_lateness_micros=allowed_lateness_micros,
        late_policy=late_policy,
    )


def compile_program_batch(program: Program, runtime: object, /) -> BatchExecutionPlan:
    """Lower one program to a strict project-v3 batch plan."""

    selected = _require_runtime(runtime, "compile_batch")
    document = lower_program_document(program, selected, "batch")
    capabilities = selected.capabilities()
    key = _compile_cache_key(program, "batch", document, capabilities, 0, "error")
    return selected._cached_symbolic_compile(
        key, lambda: selected.compile_batch_project(_canonical(document))
    )  # type: ignore[return-value]


def compile_program_stream(
    program: Program,
    runtime: object,
    allowed_lateness_micros: object,
    late_policy: object,
    /,
) -> StreamExecutionPlan:
    """Lower one program to a strict project-v3 continuous plan.

    The validated lateness arguments are written into every lowered rolling
    node; row-local programs are unaffected by them.
    """

    selected = _require_runtime(runtime, "compile_stream")
    _validate_lateness(allowed_lateness_micros, late_policy)
    document = lower_program_document(
        program,
        selected,
        "stream",
        allowed_lateness_micros=allowed_lateness_micros,
        late_policy=late_policy,
    )
    capabilities = selected.capabilities()
    key = _compile_cache_key(
        program,
        "stream",
        document,
        capabilities,
        allowed_lateness_micros,
        late_policy,
    )
    return selected._cached_symbolic_compile(
        key,
        lambda: selected._compile_stream_graph_project(
            _canonical(document), requirements=StreamRequirements()
        ),
    )  # type: ignore[return-value]


def _validate_lateness(allowed_lateness_micros: object, late_policy: object, /) -> None:
    if type(allowed_lateness_micros) is not int:
        raise TypeError(
            "allowed_lateness_micros must be an exact int; got"
            f" {type_name(allowed_lateness_micros)}"
        )
    if allowed_lateness_micros < 0:
        raise ValueError(
            "allowed_lateness_micros: invalid_literal: must be non-negative"
        )
    if allowed_lateness_micros > _U64_MAX:
        raise ValueError(
            "allowed_lateness_micros: invalid_literal: must fit the unsigned"
            " 64-bit microsecond range"
        )
    if type(late_policy) is not str:
        raise TypeError(f"late_policy must be a string; got {type_name(late_policy)}")
    if late_policy not in ("error", "drop"):
        raise ValueError(
            "late_policy: invalid_literal: must be 'error' or"
            f" 'drop'; got {late_policy!r}"
        )
