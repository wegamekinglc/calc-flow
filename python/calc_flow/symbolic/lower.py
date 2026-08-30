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


_COLUMN_PRIMITIVES: Final = frozenset(
    {
        "column_ref",
        "literal",
        "add",
        "sub",
        "mul",
        "truediv",
        "neg",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and",
        "or",
        "not",
        "where",
        "coalesce",
        "log",
        "exp",
        "sqrt",
        "abs",
        "clip",
        "cast",
    }
)

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
        name not in _COLUMN_PRIMITIVES
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


def _find_rolling(node: Node, /):
    """Yield every rolling temporal subtree in first-appearance order."""
    if node.op.name in _ROLLING_PRIMITIVES:
        yield node
    for argument in node.args:
        yield from _find_rolling(argument)


def _find_cross_section(node: Node, /):
    """Yield every cross-section subtree in first-appearance order."""
    if node.op.name in _CROSS_SECTION_PRIMITIVES:
        yield node
    for argument in node.args:
        yield from _find_cross_section(argument)


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


def _replace_rolling(node: Node, replacements: dict[str, str], /) -> Node:
    replacement = replacements.get(node.digest)
    if replacement is not None:
        return _base_ref(replacement)
    return build(
        node.op.name,
        tuple(_replace_rolling(argument, replacements) for argument in node.args),
        dict(node.attrs.entries),
        version=node.op.version,
    )


@dataclass(frozen=True, slots=True)
class _RollingPlan:
    """One lowered rolling stage: the project node plus the rewritten
    row-local environment that references its output columns."""

    node_id: str
    node: dict[str, object]
    env: tuple[tuple[str, Node], ...]
    post_predicate: Node | None
    input_field_names: tuple[str, ...]
    output_fields: tuple[Field, ...]
    replacements: tuple[tuple[str, str], ...]


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


# Rolling planning validates every lag/delta/aggregate occurrence with
# stable, declaration-ordered error paths before emitting the frozen node
# shape.
def _plan_rolling(
    output_name: str,
    segment: _Segment,
    path: str,
    allowed_lateness_micros: int,
    late_policy: str,
    /,
) -> _RollingPlan | None:
    # #lizard forgives
    occurrences: list[Node] = []
    seen: set[str] = set()
    ordered_trees = [tree for _, tree in segment.env]
    ordered_trees += [
        tree for tree in (segment.predicate, segment.post_predicate) if tree is not None
    ]
    for tree in ordered_trees:
        for subtree in _find_rolling(tree):
            if subtree.digest not in seen:
                seen.add(subtree.digest)
                occurrences.append(subtree)
    if not occurrences:
        return None

    input_fields = _schema_fields(segment.input_node.attr("schema"))
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
    used_names = set(input_types) | {name for name, _ in segment.env}
    replacements: dict[str, str] = {}
    declarations: list[dict[str, object]] = []
    derived_fields: list[Field] = []
    for index, subtree in enumerate(occurrences):
        kind = subtree.op.name
        name = whole_feature.get(subtree.digest)
        if name is None:
            name = f"{output_name}__cf_roll_{index}"
            if name in used_names:
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.DUPLICATE_NAME,
                    f"materialized rolling column {name!r} collides with a"
                    " declared field",
                )
            used_names.add(name)
        replacements[subtree.digest] = name
        operands: list[tuple[str, str]] = []
        for role, argument in zip(
            ("input", "left", "right"), subtree.args, strict=False
        ):
            if argument.op.name != "column_ref":
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.UNSUPPORTED_TYPE,
                    f"rolling {kind} argument must be an input column in this release",
                )
            input_name = _cstr(argument.attr("name"))
            field = input_types.get(input_name)
            if field is None:
                errors.raise_compile(
                    f"{path}.{name}",
                    errors.SCHEMA_MISMATCH,
                    f"rolling {kind} argument column {input_name!r} is not in the"
                    " input schema",
                )
            operands.append((role, input_name))
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
            derived_fields.append(Field(name, field.data_type, nullable=True))
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
                _rolling_output_type(kind, field.data_type) or "float64",
                nullable=True,
            )
        )

    node_id = f"{output_name}__cf_rolling"
    node: dict[str, object] = {
        "id": node_id,
        "operator": {
            "kind": "rolling",
            "spec": {
                "configuration_version": 1,
                "state_layout_version": 1,
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
                "schema": [_field_json(field) for field in input_fields],
            }
        ],
        "output_ports": [
            {
                "name": "output",
                "kind": "table",
                "required": True,
                "schema": [
                    *(_field_json(field) for field in input_fields),
                    *(_field_json(field) for field in derived_fields),
                ],
            }
        ],
    }
    env = tuple(
        (name, _replace_rolling(tree, replacements)) for name, tree in segment.env
    )
    post_predicate = (
        None
        if segment.post_predicate is None
        else _replace_rolling(segment.post_predicate, replacements)
    )
    return _RollingPlan(
        node_id,
        node,
        env,
        post_predicate,
        (*input_types, *(field.name for field in derived_fields)),
        (*input_fields, *derived_fields),
        tuple(replacements.items()),
    )


@dataclass(frozen=True, slots=True)
class _CrossSectionPlan:
    """One lowered cross-section stage: the project node plus the rewritten
    row-local environment that references its output columns."""

    node_id: str
    node: dict[str, object]
    env: tuple[tuple[str, Node], ...]
    post_predicate: Node | None
    input_field_names: tuple[str, ...]
    output_fields: tuple[Field, ...]
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
        if argument.op.name != "column_ref":
            errors.raise_compile(
                f"{path}.{name}",
                errors.UNSUPPORTED_TYPE,
                f"cross-section {kind} argument must be an input column in this"
                " release",
            )
        input_name = _cstr(argument.attr("name"))
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
                "schema": [_field_json(field) for field in input_fields],
            }
        ],
        "output_ports": [
            {
                "name": "output",
                "kind": "table",
                "required": True,
                "schema": [
                    *(_field_json(field) for field in input_fields),
                    *(_field_json(field) for field in derived_fields),
                ],
            }
        ],
    }
    env = tuple(
        (name, _replace_rolling(tree, replacements)) for name, tree in segment.env
    )
    post_predicate = (
        None
        if segment.post_predicate is None
        else _replace_rolling(segment.post_predicate, replacements)
    )
    return _CrossSectionPlan(
        node_id,
        node,
        env,
        post_predicate,
        (*input_types, *(field.name for field in derived_fields)),
        (*input_fields, *derived_fields),
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
            post_predicate = _replace_rolling(post_predicate, replacements)
        shared[output_name] = replace(
            first,
            node_id=node_id,
            node=node,
            env=tuple(
                (name, _replace_rolling(tree, replacements))
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


def _share_rolling_plans(
    segments: list[tuple[str, _Segment]],
    plans: dict[str, _RollingPlan | None],
    /,
) -> dict[str, _RollingPlan | None]:
    groups: dict[tuple[str, str | None], list[str]] = {}
    for output_name, segment in segments:
        if plans[output_name] is not None:
            predicate = None if segment.predicate is None else segment.predicate.digest
            groups.setdefault((segment.input_node.digest, predicate), []).append(
                output_name
            )
    return _share_state_groups(
        list(groups.values()), segments, plans, "roll", "rolling"
    )


def _cross_section_identity(plan: _CrossSectionPlan, /) -> str:
    spec = plan.node["operator"]["spec"]  # type: ignore[index]
    return _canonical(
        {
            "input_ports": plan.node["input_ports"],
            "spec": {key: value for key, value in spec.items() if key != "outputs"},
        }
    )


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
                (upstream, predicate, _cross_section_identity(plan)), []
            ).append(output_name)
    return _share_state_groups(
        list(groups.values()), segments, plans, "cs", "cross_section"
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


def _raise_matrix_invariant(message: str, /) -> Never:
    raise RuntimeError(f"symbolic matrix lowering invariant violated: {message}")


def _matrix_graph_lists(
    project: dict[str, object],
    /,
) -> tuple[list[object], list[object]]:
    graph = project.get("graph")
    if not isinstance(graph, dict):
        _raise_matrix_invariant("project.graph must be a mapping")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        _raise_matrix_invariant("project.graph.nodes must be a list")
    edges = graph.get("edges")
    if not isinstance(edges, list):
        _raise_matrix_invariant("project.graph.edges must be a list")
    return nodes, edges


def _wire_matrix_node(
    project: dict[str, object],
    external: dict[str, object],
    upstream_id: str,
    output_name: str,
    /,
) -> None:
    nodes, edges = _matrix_graph_lists(project)
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


def _required_segment_state_plan(
    rolling: _RollingPlan | None, cross: _CrossSectionPlan | None, /
) -> _RollingPlan | _CrossSectionPlan:
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
            nodes.append(rolling.node)
            if upstream_id is not None:
                edges.append(
                    {
                        "source_node": upstream_id,
                        "source_port": "output",
                        "target_node": rolling.node_id,
                        "target_port": "input",
                    }
                )
            elif fanout:
                edges.append(
                    {
                        "source_node": fanout_ids[segment.input_node.digest],
                        "source_port": "output",
                        "target_node": rolling.node_id,
                        "target_port": "input",
                    }
                )
            upstream_id = rolling.node_id
            env = dict(rolling.env)
            input_field_names = list(rolling.input_field_names)
            final_predicate = rolling.post_predicate
        if cross is not None:
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
    project: dict[str, object] = {
        "data_sources": [],
        "format_version": 3,
        "id": program.name,
        "name": program.name,
        "runtime": {"mode": mode, "options": {}},
        "graph": {"edges": edges, "name": program.name, "nodes": nodes},
    }
    project["data_sources"] = [] if mode == "stream" else _data_sources(project)
    return project


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
) -> None:
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
        return
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
    _check_expression_capability(program, selected, mode_value)
    if _program_needs_rolling(program) or _program_needs_cross_section(program):
        _validate_lateness(allowed_lateness_micros, late_policy)
        _, capabilities = _run(program, selected, mode_value)
        if _program_needs_rolling(program):
            _check_rolling_capability(program, capabilities, mode_value)
        if _program_needs_cross_section(program):
            _check_cross_section_capability(program, capabilities, mode_value)
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
