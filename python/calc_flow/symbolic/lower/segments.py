"""Segment constants, cache identity, and SQL rendering for the lowerer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from calc_flow.symbolic import errors
from calc_flow.symbolic.analyzer import (
    _ROW_LOCAL_PRIMITIVES,
    _literal_dtype,
    _schema_fields,
)
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
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    pass


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
