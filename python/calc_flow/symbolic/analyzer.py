"""Static analysis of symbolic programs.

The analyzer infers value types, domains, lineages, state requirements, and
stream safety from the declaration graph alone — no data object, source, sink,
or runner is accepted — and reports every finding as an immutable
``AnalysisIssue`` with a stable path rooted at a named program output or input.

Path grammar (frozen by this stage): a program output roots ``outputs.<name>``
and a declared input roots ``inputs.<name>``/``static_inputs.<name>``. A
derived field of a table node at path ``P`` is defined by the expression at
``P.<field_name>``; an argument of a node with primitive ``prim`` at path ``P``
sits at ``P.<prim>.<role>``; the failing aspect of an operand appends
``.dtype``, ``.lineage``, ``.shape[<i>]``, and so on.

Value-type inference proves only what the capability snapshot proves: identical
operand types, the frozen rolling/cross-section output-type table, and exact
field resolution. Cross-type arithmetic needs an explicit ``row.cast``; no
competing Python promotion table exists here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from calc_flow.capabilities import RuntimeCapabilities
from calc_flow.pipeline import Runtime
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
)
from calc_flow.symbolic.types import CompileMode, Field

_MODES: Final[tuple[str, ...]] = ("batch", "stream")
_EVENT_TIME_TYPE: Final = "timestamp[us, UTC]"
_FLOATING_TYPES: Final = ("float32", "float64")
_SIGNED_INT_TYPES: Final = ("int8", "int16", "int32", "int64")
_UNSIGNED_INT_TYPES: Final = ("uint8", "uint16", "uint32", "uint64")
_NUMERIC_TYPES: Final = frozenset(
    (*_SIGNED_INT_TYPES, *_UNSIGNED_INT_TYPES, *_FLOATING_TYPES)
)


def _is_array_parameter(node: Node, /) -> bool:
    kind = node.attr("kind") if node.op.name == "parameter" else None
    return isinstance(kind, CEnum) and kind.variant == "array"


def _sum_output_type(input_type: str | None, /) -> str:
    if input_type in _SIGNED_INT_TYPES:
        return "int64"
    if input_type in _UNSIGNED_INT_TYPES:
        return "uint64"
    return "float64"


def _rolling_output_type(primitive: str, input_type: str | None, /) -> str | None:
    if primitive == "count":
        return "uint64"
    if primitive == "sum":
        return _sum_output_type(input_type)
    if primitive in ("min", "max"):
        return input_type
    return "float64"


def _entity_field_is_valid(field: Field | None, /) -> bool:
    return field is not None


def _entity_field_message(field_name: str, /) -> str:
    return f"entity field {field_name!r} is not in the input schema"


def _sequence_field_is_valid(field: Field | None, /) -> bool:
    return (
        field is not None
        and not field.nullable
        and field.data_type not in _FLOATING_TYPES
    )


def _sequence_field_message(_field_name: str, /) -> str:
    return (
        "sequence fields must be non-null with a portable total order;"
        " floating sequence fields are forbidden"
    )


_FRAME_ATTRIBUTES: Final[dict[str, str]] = {"rows": "size", "duration": "micros"}
_ARITHMETIC: Final = frozenset({"add", "sub", "mul", "truediv"})
_COMPARISONS: Final = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
_BOOLEANS: Final = frozenset({"and", "or"})
_UNARY_NUMERIC: Final = frozenset({"log", "exp", "sqrt"})
_ROW_LOCAL_PRIMITIVES: Final = frozenset(
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
_ROLLING_AGGREGATES: Final = frozenset(
    {"count", "sum", "mean", "min", "max", "variance", "stddev"}
)
_ROLLING_DDOF: Final = frozenset({"variance", "stddev"})
_ROLLING_PRIMITIVES: Final = _ROLLING_AGGREGATES | frozenset(
    {"lag", "delta", "covariance", "correlation"}
)
_CROSS_SECTION: Final = frozenset(
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


@dataclass(frozen=True, slots=True)
class AnalysisIssue:
    """One analysis finding with its stable path, code, and message."""

    path: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """The immutable outcome of analyzing one program in one mode."""

    mode: CompileMode
    program_fingerprint: str
    capability_session_id: str
    capability_revision: int
    issues: tuple[AnalysisIssue, ...]


@dataclass(frozen=True, slots=True)
class TableFacts:
    """The inferred schema, lineage, ordering, and state of one table value."""

    schema: tuple[Field, ...]
    lineage: str | None
    state: frozenset[str]
    event_time: str | None
    entity_by: tuple[str, ...]
    sequence_by: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ColumnFacts:
    """The inferred value type and lineage of one column expression."""

    data_type: str | None
    nullable: bool
    lineage: str | None
    state: frozenset[str]


@dataclass(frozen=True, slots=True)
class ArrayFacts:
    """The inferred backend, dtype, shape, and row lineage of one array."""

    backend: str | None
    dtype: str | None
    shape: tuple[int | str, ...]
    lineage: str | None
    state: frozenset[str]


def _cstr(value: CValue, /) -> str | None:
    return value.value if isinstance(value, CStr) else None


def _ctype_str(value: CValue, /) -> str | None:
    if isinstance(value, CStr):
        return value.value
    if isinstance(value, CDType):
        return value.name
    return None


def _cint(value: CValue, /) -> int | None:
    return value.value if isinstance(value, CInt) else None


def _cstr_seq(value: CValue, /) -> tuple[str, ...]:
    if isinstance(value, CSeq):
        return tuple(item.value for item in value.items if isinstance(item, CStr))
    return ()


def _declared_type_name(data_type: CValue, /) -> str | None:
    if isinstance(data_type, CStr):
        return data_type.value
    if isinstance(data_type, CDType):
        return data_type.name
    return None


def _declared_field(item: CValue, /) -> Field | None:
    if not isinstance(item, CMap):
        return None
    name = item.get("name")
    nullable = item.get("nullable")
    if not isinstance(name, CStr) or not isinstance(nullable, CBool):
        return None
    type_name_value = _declared_type_name(item.get("data_type"))
    if type_name_value is None:
        return None
    return Field(name.value, type_name_value, nullable.value)


def _schema_fields(value: CValue, /) -> tuple[Field, ...]:
    if not isinstance(value, CSeq):
        return ()
    fields = [
        field for item in value.items if (field := _declared_field(item)) is not None
    ]
    return tuple(fields)


def _resolves_to_input_column(node: Node, /) -> bool:
    """Whether the row-local lowerer resolves ``node`` to one source column."""

    if node.op.name != "column_ref" or not node.args:
        return False
    return _table_field_resolves_to_input(
        node.args[0],
        _cstr(node.attr("name")) or "",
    )


def _table_field_resolves_to_input(table: Node, field_name: str, /) -> bool:
    operation = table.op.name
    if operation == "table_input":
        return True
    if operation in ("project", "filter"):
        return _table_field_resolves_to_input(table.args[0], field_name)
    if operation != "with_columns":
        return False
    names = _cstr_seq(table.attr("names"))
    if field_name not in names:
        return _table_field_resolves_to_input(table.args[0], field_name)
    index = names.index(field_name)
    return _resolves_to_input_column(table.args[index + 1])


def _stateful_operand_is_stageable(node: Node, /) -> bool:
    """Whether an operand can be scheduled before its stateful consumer."""

    operation = node.op.name
    if operation == "column_ref":
        if not node.args:
            return True
        return _table_field_is_stageable(
            node.args[0],
            _cstr(node.attr("name")) or "",
        )
    if operation == "literal":
        return True
    if operation not in _ROW_LOCAL_PRIMITIVES and operation not in _ROLLING_PRIMITIVES:
        return False
    return all(_stateful_operand_is_stageable(argument) for argument in node.args)


def _table_field_is_stageable(table: Node, field_name: str, /) -> bool:
    operation = table.op.name
    if operation == "table_input":
        return True
    if operation in ("project", "filter"):
        return _table_field_is_stageable(table.args[0], field_name)
    if operation != "with_columns":
        return False
    names = _cstr_seq(table.attr("names"))
    if field_name not in names:
        return _table_field_is_stageable(table.args[0], field_name)
    index = names.index(field_name)
    return _stateful_operand_is_stageable(table.args[index + 1])


class _Analyzer:
    """One analysis pass over one program, mode, and capability snapshot."""

    def __init__(
        self,
        mode: CompileMode,
        declared: frozenset[str],
        portable_types: frozenset[str],
        supports_array_kind: bool,
    ) -> None:
        self._mode = mode
        self._declared = declared
        self._portable_types = portable_types
        self._supports_array_kind = supports_array_kind
        self._issues: list[AnalysisIssue] = []
        self._table_cache: dict[str, TableFacts] = {}
        self._column_cache: dict[str, ColumnFacts] = {}
        self._array_cache: dict[str, ArrayFacts] = {}
        self._undeclared_reported: set[str] = set()
        self._temporal_lineages: set[str] = set()

    def issue(self, path: str, code: str, message: str, /) -> None:
        self._issues.append(AnalysisIssue(path, code, message))

    # -- declaration-level checks ------------------------------------------

    def check_input_declaration(self, node: Node, root: str, name: str, /) -> None:
        if _is_array_parameter(node) and not self._supports_array_kind:
            self.issue(
                f"{root}.{name}",
                "capability_mismatch",
                "the capability snapshot does not support the array batch kind",
            )
        for index, field in enumerate(_schema_fields(node.attr("schema"))):
            if field.data_type not in self._portable_types:
                self.issue(
                    f"{root}.{name}.schema[{index}].data_type",
                    "capability_mismatch",
                    f"type {field.data_type!r} is not portable in the selected"
                    " runtime capability snapshot",
                )

    def check_array_output_kind(self, output_name: str, /) -> None:
        if not self._supports_array_kind:
            self.issue(
                f"outputs.{output_name}",
                "capability_mismatch",
                "the capability snapshot does not support the array batch kind",
            )

    def check_stream_ordering(self, node: Node, root: str, name: str, /) -> None:
        schema = {field.name: field for field in _schema_fields(node.attr("schema"))}
        base = f"{root}.{name}"
        self._ordering_event_time(node, schema, base)
        self._ordering_key_fields(
            node.attr("entity_by"),
            schema,
            f"{base}.entity_by",
            "temporal work in stream mode requires a non-empty entity key",
            _entity_field_is_valid,
            _entity_field_message,
        )
        self._ordering_key_fields(
            node.attr("sequence_by"),
            schema,
            f"{base}.sequence_by",
            "temporal work in stream mode requires a non-empty sequence key",
            _sequence_field_is_valid,
            _sequence_field_message,
        )

    def _ordering_event_time(
        self, node: Node, schema: dict[str, Field], base: str, /
    ) -> None:
        event_time = _cstr(node.attr("event_time"))
        field = None if event_time is None else schema.get(event_time)
        if field is None or field.data_type != _EVENT_TIME_TYPE or field.nullable:
            message = (
                "temporal work in stream mode requires a declared event-time column"
                if event_time is None
                else "the event-time column must be a non-null timestamp[us, UTC] field"
            )
            self.issue(f"{base}.event_time", "ordering_required", message)

    def _ordering_key_fields(
        self,
        declared: CValue,
        schema: dict[str, Field],
        base: str,
        empty_message: str,
        field_is_valid,
        field_message,
        /,
    ) -> None:
        names = _cstr_seq(declared)
        if not names:
            self.issue(base, "ordering_required", empty_message)
            return
        for index, field_name in enumerate(names):
            if not field_is_valid(schema.get(field_name)):
                self.issue(
                    f"{base}[{index}]",
                    "ordering_required",
                    field_message(field_name),
                )

    def check_unbounded_output(self, output_name: str, facts: ArrayFacts, /) -> None:
        if self._mode == "stream" and facts.lineage is not None:
            self.issue(
                f"outputs.{output_name}",
                "unbounded_state",
                "an array output with row-axis lineage over a stream table has"
                " unbounded state; attach it to its table or express it through"
                " an explicit window",
            )

    @property
    def issues(self) -> tuple[AnalysisIssue, ...]:
        return tuple(
            sorted(
                self._issues,
                key=lambda issue: (issue.path, issue.code, issue.message),
            )
        )

    @property
    def temporal_lineages(self) -> frozenset[str]:
        return frozenset(self._temporal_lineages)

    # -- table analysis ------------------------------------------------------

    def table(self, node: Node, path: str, /) -> TableFacts:
        cached = self._table_cache.get(node.digest)
        if cached is not None:
            return cached
        facts = self._analyze_table(node, path)
        self._table_cache[node.digest] = facts
        return facts

    def _analyze_table(self, node: Node, path: str, /) -> TableFacts:
        name = node.op.name
        if name in ("table_input", "parameter"):
            return self._table_declaration(node, path)
        if name == "project":
            return self._project_table(node, path)
        if name == "filter":
            return self._filter_table(node, path)
        if name == "with_columns":
            return self._with_columns_table(node, path)
        if name == "attach_columns":
            return self._attach_columns_table(node, path)
        if name in ("window_tumbling", "window_hopping"):
            return self._window_table(node, path)
        self.issue(
            path,
            "unknown_primitive_version",
            f"primitive {name!r} does not produce a table value",
        )
        return TableFacts((), None, frozenset(), None, (), ())

    def _table_declaration(self, node: Node, path: str, /) -> TableFacts:
        name = _cstr(node.attr("name"))
        is_parameter = node.op.name == "parameter"
        root = "static_inputs" if is_parameter else "inputs"
        if (
            name is not None
            and node.digest not in self._declared
            and name not in self._undeclared_reported
        ):
            self._undeclared_reported.add(name)
            self.issue(
                f"{root}.{name}",
                "unresolved_type",
                f"input {name!r} is referenced by program outputs but not"
                " declared in Program inputs",
            )
        return TableFacts(
            schema=_schema_fields(node.attr("schema")),
            lineage=name,
            state=frozenset({"static"}) if is_parameter else frozenset(),
            event_time=_cstr(node.attr("event_time")),
            entity_by=_cstr_seq(node.attr("entity_by")),
            sequence_by=_cstr_seq(node.attr("sequence_by")),
        )

    def _project_table(self, node: Node, path: str, /) -> TableFacts:
        child = self.table(node.args[0], f"{path}.project.value")
        columns = _cstr_seq(node.attr("columns"))
        by_name = {field.name: field for field in child.schema}
        projected: list[Field] = []
        for index, column in enumerate(columns):
            field = by_name.get(column)
            if field is None:
                self.issue(
                    f"{path}.project.columns[{index}]",
                    "unresolved_type",
                    f"unknown field {column!r} in the projected table schema",
                )
            else:
                projected.append(field)
        return TableFacts(
            tuple(projected),
            child.lineage,
            child.state,
            child.event_time,
            child.entity_by,
            child.sequence_by,
        )

    def _filter_table(self, node: Node, path: str, /) -> TableFacts:
        child = self.table(node.args[0], f"{path}.filter.value")
        predicate = self.column(node.args[1], f"{path}.filter.predicate")
        if predicate.lineage is not None and predicate.lineage != child.lineage:
            self.issue(
                f"{path}.filter.predicate.lineage",
                "schema_mismatch",
                f"the filter predicate mixes table lineage {child.lineage!r}"
                f" with columns from {predicate.lineage!r}",
            )
        if predicate.data_type is not None and predicate.data_type != "bool":
            self.issue(
                f"{path}.filter.predicate.dtype",
                "unsupported_type",
                "a filter predicate must be a boolean column expression",
            )
        return child

    def _with_columns_table(self, node: Node, path: str, /) -> TableFacts:
        child = self.table(node.args[0], f"{path}.with_columns.value")
        names = _cstr_seq(node.attr("names"))
        fields = list(child.schema)
        existing = {field.name for field in child.schema}
        states: set[str] = set(child.state)
        for index, name in enumerate(names):
            expression = node.args[index + 1]
            facts = self.column(expression, f"{path}.{name}")
            states |= facts.state
            if facts.lineage is not None and facts.lineage != child.lineage:
                self.issue(
                    f"{path}.{name}.lineage",
                    "schema_mismatch",
                    f"feature {name!r} mixes table lineage {child.lineage!r}"
                    f" with columns from {facts.lineage!r}",
                )
            if name in existing:
                self.issue(
                    f"{path}.{name}",
                    "duplicate_name",
                    f"derived field {name!r} collides with an existing schema field",
                )
            elif facts.data_type is not None:
                fields.append(Field(name, facts.data_type, facts.nullable))
                existing.add(name)
        return TableFacts(
            tuple(fields),
            child.lineage,
            frozenset(states),
            child.event_time,
            child.entity_by,
            child.sequence_by,
        )

    def _attach_columns_table(self, node: Node, path: str, /) -> TableFacts:
        child = self.table(node.args[0], f"{path}.attach_columns.value")
        array = self.array(node.args[1], f"{path}.attach_columns.array")
        names = _cstr_seq(node.attr("names"))
        self._attach_lineage_check(array, child, path)
        self._attach_width_check(array, names, path)
        fields = self._attached_fields(child, array, names, path)
        return TableFacts(
            tuple(fields),
            child.lineage,
            child.state | array.state,
            child.event_time,
            child.entity_by,
            child.sequence_by,
        )

    def _attach_lineage_check(
        self, array: ArrayFacts, child: TableFacts, path: str, /
    ) -> None:
        if array.lineage is None or array.lineage != child.lineage:
            self.issue(
                f"{path}.attach_columns.array.lineage",
                "schema_mismatch",
                "an attached array must carry the row-axis lineage of the"
                f" target table {child.lineage!r}",
            )

    def _attach_width_check(
        self, array: ArrayFacts, names: tuple[str, ...], path: str, /
    ) -> None:
        if not array.shape:
            return
        width_path = f"{path}.attach_columns.array.shape[{len(array.shape) - 1}]"
        width = array.shape[-1]
        if isinstance(width, str):
            self.issue(
                width_path,
                "unresolved_type",
                "the attached array width is symbolic and cannot be proved"
                " to match the declared names",
            )
        elif width != len(names):
            self.issue(
                width_path,
                "schema_mismatch",
                f"array width {width} does not match {len(names)} declared names",
            )

    def _attached_fields(
        self,
        child: TableFacts,
        array: ArrayFacts,
        names: tuple[str, ...],
        path: str,
        /,
    ) -> list[Field]:
        fields = list(child.schema)
        existing = {field.name for field in child.schema}
        for name in names:
            if name in existing:
                self.issue(
                    f"{path}.{name}",
                    "duplicate_name",
                    f"attached field {name!r} collides with an existing schema field",
                )
            elif array.dtype is not None:
                fields.append(Field(name, array.dtype, nullable=True))
                existing.add(name)
        return fields

    def _window_table(self, node: Node, path: str, /) -> TableFacts:
        child = self.table(node.args[0], f"{path}.{node.op.name}.value")
        if child.lineage is not None:
            self._temporal_lineages.add(child.lineage)
        by_name = {field.name: field for field in child.schema}
        event_time = _cstr(node.attr("event_time"))
        if event_time is not None and event_time not in by_name:
            self.issue(
                f"{path}.{node.op.name}.event_time",
                "unresolved_type",
                f"unknown event-time field {event_time!r} in the windowed table schema",
            )
        fields = [
            Field("window_start", _EVENT_TIME_TYPE, nullable=False),
            Field("window_end", _EVENT_TIME_TYPE, nullable=False),
        ]
        for index, name in enumerate(_cstr_seq(node.attr("group_by"))):
            field = by_name.get(name)
            if field is None:
                self.issue(
                    f"{path}.{node.op.name}.group_by[{index}]",
                    "unresolved_type",
                    f"unknown group field {name!r} in the windowed table schema",
                )
            else:
                fields.append(field)
        return TableFacts(
            tuple(fields),
            None,
            child.state | frozenset({"window"}),
            child.event_time,
            child.entity_by,
            child.sequence_by,
        )

    # -- column analysis -----------------------------------------------------

    def column(self, node: Node, path: str, /) -> ColumnFacts:
        cached = self._column_cache.get(node.digest)
        if cached is not None:
            return cached
        facts = self._analyze_column(node, path)
        self._column_cache[node.digest] = facts
        return facts

    def _analyze_column(self, node: Node, path: str, /) -> ColumnFacts:
        handler = _COLUMN_HANDLERS.get(node.op.name)
        if handler is None:
            self.issue(
                path,
                "unknown_primitive_version",
                f"primitive {node.op.name!r} does not produce a column value",
            )
            return ColumnFacts(None, True, None, frozenset())
        return handler(self, node, path)

    def _column_ref(self, node: Node, path: str, /) -> ColumnFacts:
        table = self.table(node.args[0], f"{path}.column_ref.value")
        field_name = _cstr(node.attr("name"))
        for field in table.schema:
            if field.name == field_name:
                return ColumnFacts(
                    field.data_type,
                    field.nullable,
                    table.lineage,
                    table.state,
                )
        self.issue(
            path,
            "unresolved_type",
            f"unknown field {field_name!r} in the schema of input {table.lineage!r}",
        )
        return ColumnFacts(None, True, table.lineage, table.state)

    def _anchored_operands(
        self,
        nodes: tuple[Node, ...],
        paths: tuple[str, ...],
        /,
        initial_anchor: str | None = None,
    ) -> tuple[ColumnFacts, ...]:
        """Analyze operands left to right against the first resolved lineage."""

        operands: list[ColumnFacts] = []
        anchor = initial_anchor
        for child, operand_path in zip(nodes, paths, strict=True):
            facts = self._operand(child, operand_path, anchor)
            operands.append(facts)
            if anchor is None and facts.lineage is not None:
                anchor = facts.lineage
        return tuple(operands)

    @staticmethod
    def _running_anchor(operands: tuple[ColumnFacts, ...], /) -> str | None:
        return next(
            (facts.lineage for facts in operands if facts.lineage is not None),
            None,
        )

    def _operand(self, node: Node, path: str, anchor: str | None, /) -> ColumnFacts:
        if node.op.name == "literal":
            facts = _literal_facts(node)
        else:
            facts = self.column(node, path)
        if anchor is not None and facts.lineage is not None and facts.lineage != anchor:
            self.issue(
                f"{path}.lineage",
                "schema_mismatch",
                f"column operands span inputs {anchor!r} and {facts.lineage!r}",
            )
        return facts

    def _pair(self, node: Node, path: str, /) -> tuple[ColumnFacts, ColumnFacts] | None:
        left = self._operand(node.args[0], f"{path}.{node.op.name}.left", None)
        anchor = left.lineage
        right = self._operand(node.args[1], f"{path}.{node.op.name}.right", anchor)
        if (
            left.data_type is not None
            and right.data_type is not None
            and left.data_type != right.data_type
        ):
            self.issue(
                f"{path}.{node.op.name}.right.dtype",
                "unsupported_type",
                f"no provable common type for {left.data_type!r} and"
                f" {right.data_type!r}; use row.cast for an explicit conversion",
            )
            return None
        return left, right

    def _arithmetic(self, node: Node, path: str, /) -> ColumnFacts:
        pair = self._pair(node, path)
        if pair is None:
            return ColumnFacts(None, True, None, frozenset())
        left, right = pair
        data_type = left.data_type
        if (
            node.op.name == "truediv"
            and data_type is not None
            and data_type not in _FLOATING_TYPES
        ):
            self.issue(
                f"{path}.{node.op.name}.right.dtype",
                "unsupported_type",
                "division is only provable for floating columns; cast operands"
                " explicitly",
            )
            data_type = None
        return ColumnFacts(
            data_type,
            left.nullable or right.nullable,
            left.lineage,
            left.state | right.state,
        )

    def _comparison(self, node: Node, path: str, /) -> ColumnFacts:
        pair = self._pair(node, path)
        if pair is None:
            return ColumnFacts(None, True, None, frozenset())
        left, right = pair
        return ColumnFacts(
            "bool",
            left.nullable or right.nullable,
            left.lineage,
            left.state | right.state,
        )

    def _boolean_pair(self, node: Node, path: str, /) -> ColumnFacts:
        pair = self._pair(node, path)
        if pair is None:
            return ColumnFacts(None, True, None, frozenset())
        left, right = pair
        if left.data_type is not None and left.data_type != "bool":
            self.issue(
                f"{path}.{node.op.name}.left.dtype",
                "unsupported_type",
                "boolean composition requires boolean column operands",
            )
            return ColumnFacts(None, True, left.lineage, left.state | right.state)
        return ColumnFacts(
            "bool",
            left.nullable or right.nullable,
            left.lineage,
            left.state | right.state,
        )

    def _unary(self, node: Node, path: str, /) -> ColumnFacts:
        operand = self._operand(node.args[0], f"{path}.{node.op.name}.value", None)
        if node.op.name == "not":
            if operand.data_type is not None and operand.data_type != "bool":
                self.issue(
                    f"{path}.not.value.dtype",
                    "unsupported_type",
                    "boolean negation requires a boolean column operand",
                )
                return ColumnFacts(None, True, operand.lineage, operand.state)
            return ColumnFacts("bool", operand.nullable, operand.lineage, operand.state)
        if operand.data_type is not None and operand.data_type not in (
            *_SIGNED_INT_TYPES,
            *_UNSIGNED_INT_TYPES,
            *_FLOATING_TYPES,
        ):
            self.issue(
                f"{path}.neg.value.dtype",
                "unsupported_type",
                "negation requires a numeric column operand",
            )
            return ColumnFacts(None, True, operand.lineage, operand.state)
        return ColumnFacts(
            operand.data_type, operand.nullable, operand.lineage, operand.state
        )

    def _where(self, node: Node, path: str, /) -> ColumnFacts:
        role = f"{path}.where"
        operand_paths = (f"{role}.condition", f"{role}.when_true", f"{role}.when_false")
        operands = self._anchored_operands(node.args, operand_paths)
        condition, left, right = operands
        if condition.data_type is not None and condition.data_type != "bool":
            self.issue(
                f"{role}.condition.dtype",
                "unsupported_type",
                "a conditional requires a boolean condition",
            )
        anchor = self._running_anchor(operands)
        states = _column_state_union(operands)
        if (
            left.data_type is not None
            and right.data_type is not None
            and left.data_type != right.data_type
        ):
            self.issue(
                f"{role}.when_false.dtype",
                "unsupported_type",
                f"no provable common type for {left.data_type!r} and"
                f" {right.data_type!r}; use row.cast for an explicit conversion",
            )
            return ColumnFacts(None, True, anchor, states)
        return ColumnFacts(
            left.data_type,
            left.nullable or right.nullable,
            anchor,
            states,
        )

    def _coalesce(self, node: Node, path: str, /) -> ColumnFacts:
        role = f"{path}.coalesce"
        operands = self._anchored_operands(
            node.args,
            tuple(f"{role}.values[{index}]" for index in range(len(node.args))),
        )
        first = operands[0]
        for index, facts in enumerate(operands[1:], start=1):
            if (
                first.data_type is not None
                and facts.data_type is not None
                and facts.data_type != first.data_type
            ):
                self.issue(
                    f"{role}.values[{index}].dtype",
                    "unsupported_type",
                    "coalesce requires operands of one provable type",
                )
                return ColumnFacts(
                    None,
                    True,
                    self._running_anchor(operands),
                    _column_state_union(operands),
                )
        return ColumnFacts(
            first.data_type,
            all(facts.nullable for facts in operands),
            self._running_anchor(operands),
            _column_state_union(operands),
        )

    def _scalar_function(self, node: Node, path: str, /) -> ColumnFacts:
        operand = self._operand(node.args[0], f"{path}.{node.op.name}.value", None)
        if operand.data_type is not None and operand.data_type not in (
            "float32",
            "float64",
        ):
            self.issue(
                f"{path}.{node.op.name}.value.dtype",
                "unsupported_type",
                f"{node.op.name} is only provable for floating columns; use"
                " row.cast for an explicit conversion",
            )
            return ColumnFacts(None, True, operand.lineage, operand.state)
        return ColumnFacts("float64", operand.nullable, operand.lineage, operand.state)

    def _arithmetic_like_unary(self, node: Node, path: str, /) -> ColumnFacts:
        operand = self._operand(node.args[0], f"{path}.{node.op.name}.value", None)
        if operand.data_type is not None and operand.data_type not in (
            *_SIGNED_INT_TYPES,
            *_UNSIGNED_INT_TYPES,
            *_FLOATING_TYPES,
        ):
            self.issue(
                f"{path}.{node.op.name}.value.dtype",
                "unsupported_type",
                f"{node.op.name} requires a numeric column operand",
            )
            return ColumnFacts(None, True, operand.lineage, operand.state)
        return ColumnFacts(
            operand.data_type, operand.nullable, operand.lineage, operand.state
        )

    def _clip(self, node: Node, path: str, /) -> ColumnFacts:
        operand = self._operand(node.args[0], f"{path}.clip.value", None)
        if operand.data_type is not None and operand.data_type not in (
            "float32",
            "float64",
        ):
            self.issue(
                f"{path}.clip.value.dtype",
                "unsupported_type",
                "clipping is only provable for floating columns",
            )
            return ColumnFacts(None, True, operand.lineage, operand.state)
        return ColumnFacts(
            operand.data_type, operand.nullable, operand.lineage, operand.state
        )

    def _cast(self, node: Node, path: str, /) -> ColumnFacts:
        operand = self._operand(node.args[0], f"{path}.cast.value", None)
        target = _ctype_str(node.attr("data_type"))
        return ColumnFacts(target, operand.nullable, operand.lineage, operand.state)

    def _lag_like(self, node: Node, path: str, /) -> ColumnFacts:
        role = f"{path}.{node.op.name}"
        operand = self._operand(node.args[0], f"{role}.value", None)
        self._require_rolling_operand(
            node.args[0],
            f"{role}.value",
            f"rolling {node.op.name} argument must be an input column or"
            " row-local expression in this release",
        )
        if operand.lineage is not None:
            self._temporal_lineages.add(operand.lineage)
        periods = _cint(node.attr("periods")) or 1
        states = operand.state | frozenset({f"rows({periods})"})
        if node.op.name == "delta" and not self._numeric_or_issue(
            operand.data_type, f"{role}.value", node.op.name
        ):
            return ColumnFacts(None, True, operand.lineage, states)
        return ColumnFacts(operand.data_type, True, operand.lineage, states)

    def _numeric_or_issue(
        self, data_type: str | None, operand_path: str, primitive: str, /
    ) -> bool:
        """Require a provably numeric input; unknown types stay unresolved."""

        if data_type is None or data_type in _NUMERIC_TYPES:
            return True
        self.issue(
            f"{operand_path}.dtype",
            "unsupported_type",
            f"{primitive} is only defined for numeric columns; use row.cast"
            " for an explicit conversion",
        )
        return False

    def _frame_state(self, node: Node, /) -> str | None:
        frame = node.attr("frame")
        if not isinstance(frame, CMap):
            return None
        kind = frame.get("frame")
        attribute = (
            _FRAME_ATTRIBUTES.get(kind.variant) if isinstance(kind, CEnum) else None
        )
        if attribute is None:
            return None
        value = _cint(frame.get(attribute))
        return None if value is None else f"{kind.variant}({value})"

    def _rolling_aggregate(self, node: Node, path: str, /) -> ColumnFacts:
        operand = self._operand(node.args[0], f"{path}.{node.op.name}.value", None)
        self._require_rolling_operand(
            node.args[0],
            f"{path}.{node.op.name}.value",
            f"rolling {node.op.name} argument must be an input column or"
            " row-local expression in this release",
        )
        if operand.lineage is not None:
            self._temporal_lineages.add(operand.lineage)
        state = self._frame_state(node)
        states = operand.state | (
            frozenset({state}) if state is not None else frozenset()
        )
        primitive = node.op.name
        input_type = operand.data_type
        if primitive not in ("count", "min", "max") and not self._numeric_or_issue(
            input_type, f"{path}.{primitive}.value", primitive
        ):
            return ColumnFacts(None, True, operand.lineage, states)
        return ColumnFacts(
            _rolling_output_type(primitive, input_type),
            True,
            operand.lineage,
            states,
        )

    def _rolling_pair(self, node: Node, path: str, /) -> ColumnFacts:
        role = f"{path}.{node.op.name}"
        left = self._operand(node.args[0], f"{role}.left", None)
        right = self._operand(node.args[1], f"{role}.right", left.lineage)
        message = (
            f"rolling {node.op.name} argument must be an input column or"
            " row-local expression in this release"
        )
        self._require_rolling_operand(node.args[0], f"{role}.left", message)
        self._require_rolling_operand(node.args[1], f"{role}.right", message)
        if left.lineage is not None:
            self._temporal_lineages.add(left.lineage)
        state = self._frame_state(node)
        states = (
            left.state
            | right.state
            | (frozenset({state}) if state is not None else frozenset())
        )
        numeric = self._numeric_or_issue(
            left.data_type, f"{role}.left", node.op.name
        ) & self._numeric_or_issue(right.data_type, f"{role}.right", node.op.name)
        if not numeric:
            return ColumnFacts(None, True, left.lineage, states)
        return ColumnFacts("float64", True, left.lineage, states)

    def _cross_section(self, node: Node, path: str, /) -> ColumnFacts:
        role = f"{path}.{node.op.name}"
        operand = self._operand(node.args[0], f"{role}.value", None)
        self._require_stageable_stateful_operand(
            node.args[0],
            f"{role}.value",
            f"cross-section {node.op.name} argument must be an input column,"
            " row-local expression, or rolling result in this release",
        )
        if operand.lineage is not None:
            self._temporal_lineages.add(operand.lineage)
        group_paths = (
            f"{role}.event_time",
            *(f"{role}.partition_by[{index}]" for index in range(len(node.args) - 2)),
        )
        group = self._anchored_operands(node.args[1:], group_paths, operand.lineage)
        grouping_messages = (
            "cross-section grouping event time must be an input column in this release",
            *(
                "cross-section group columns must be input columns in this release"
                for _ in node.args[2:]
            ),
        )
        for group_node, group_path, message in zip(
            node.args[1:], group_paths, grouping_messages, strict=True
        ):
            self._require_stateful_input_column(
                group_node,
                group_path,
                message,
            )
        states = (
            operand.state | _column_state_union(group) | frozenset({"cross_section"})
        )
        return self._cross_section_output(node.op.name, operand, role, states)

    def _require_stateful_input_column(
        self,
        node: Node,
        path: str,
        message: str,
        /,
    ) -> None:
        if not _resolves_to_input_column(node):
            self.issue(
                path,
                "unsupported_type",
                message,
            )

    def _require_rolling_operand(
        self,
        node: Node,
        path: str,
        message: str,
        /,
    ) -> None:
        self._require_stageable_stateful_operand(node, path, message)

    def _require_stageable_stateful_operand(
        self,
        node: Node,
        path: str,
        message: str,
        /,
    ) -> None:
        if not _stateful_operand_is_stageable(node):
            self.issue(path, "unsupported_type", message)

    def _cross_section_output(
        self,
        primitive: str,
        operand: ColumnFacts,
        role: str,
        states: frozenset[str],
        /,
    ) -> ColumnFacts:
        unresolved = ColumnFacts(None, True, operand.lineage, states)
        if primitive in ("winsorize", "mean_fill"):
            if operand.data_type is not None and operand.data_type not in (
                "float32",
                "float64",
            ):
                self.issue(
                    f"{role}.value.dtype",
                    "unsupported_type",
                    f"{primitive} is only supported for floating columns",
                )
                return unresolved
            return ColumnFacts(operand.data_type, True, operand.lineage, states)
        if primitive in ("top", "bottom"):
            if not self._numeric_or_issue(
                operand.data_type, f"{role}.value", primitive
            ):
                return unresolved
            return ColumnFacts("bool", True, operand.lineage, states)
        if primitive in ("zscore", "demean") and not self._numeric_or_issue(
            operand.data_type, f"{role}.value", primitive
        ):
            return unresolved
        return ColumnFacts("float64", True, operand.lineage, states)

    # -- array analysis ------------------------------------------------------

    def array(self, node: Node, path: str, /) -> ArrayFacts:
        cached = self._array_cache.get(node.digest)
        if cached is not None:
            return cached
        facts = self._analyze_array(node, path)
        self._array_cache[node.digest] = facts
        return facts

    def _analyze_array(self, node: Node, path: str, /) -> ArrayFacts:
        handler = _ARRAY_HANDLERS.get(node.op.name)
        if handler is None:
            self.issue(
                path,
                "unknown_primitive_version",
                f"primitive {node.op.name!r} does not produce an array value",
            )
            return ArrayFacts(None, None, (), None, frozenset())
        return handler(self, node, path)

    def _array_parameter(self, node: Node, path: str, /) -> ArrayFacts:
        name = _cstr(node.attr("name"))
        if (
            name is not None
            and node.digest not in self._declared
            and name not in self._undeclared_reported
        ):
            self._undeclared_reported.add(name)
            self.issue(
                f"static_inputs.{name}",
                "unresolved_type",
                f"input {name!r} is referenced by program outputs but not"
                " declared in Program inputs",
            )
        shape = tuple(
            dimension.value
            for dimension in (
                node.attr("shape").items if isinstance(node.attr("shape"), CSeq) else ()
            )
            if isinstance(dimension, CInt)
        )
        return ArrayFacts(
            _cstr(node.attr("backend")),
            _ctype_str(node.attr("dtype")),
            shape,
            None,
            frozenset({"static"}),
        )

    def _from_columns(self, node: Node, path: str, /) -> ArrayFacts:
        role = f"{path}.from_columns"
        table = self.table(node.args[0], f"{role}.value")
        columns = _cstr_seq(node.attr("columns"))
        data_type = self._from_columns_dtype(columns, table.schema, role)
        rows: int | str = table.lineage if table.lineage is not None else "rows"
        return ArrayFacts(
            _cstr(node.attr("backend")),
            data_type,
            (rows, len(columns)),
            table.lineage,
            table.state,
        )

    def _from_columns_dtype(
        self,
        columns: tuple[str, ...],
        schema: tuple[Field, ...],
        role: str,
        /,
    ) -> str | None:
        data_type: str | None = None
        for index, column in enumerate(columns):
            field = next((item for item in schema if item.name == column), None)
            if field is None:
                self.issue(
                    f"{role}.columns[{index}]",
                    "unresolved_type",
                    f"unknown field {column!r} in the source table schema",
                )
                continue
            if data_type is None:
                data_type = field.data_type
            elif field.data_type != data_type:
                self.issue(
                    f"{role}.columns[{index}].dtype",
                    "unsupported_type",
                    f"from_columns requires one dtype; found {data_type!r} and"
                    f" {field.data_type!r}",
                )
                return None
        return data_type

    def _array_pair_compat(
        self,
        left: ArrayFacts,
        right: ArrayFacts,
        role: str,
        lineage_message: str,
        /,
    ) -> str | None:
        """Report dtype, backend, and row-lineage incompatibilities."""

        dtype = self._safe_array_dtype(
            left.backend or right.backend,
            left.dtype,
            right.dtype,
            role,
        )
        self._array_pair_aspects(left, right, role, lineage_message)
        return dtype

    def _array_pair_aspects(
        self,
        left: ArrayFacts,
        right: ArrayFacts,
        role: str,
        lineage_message: str,
        /,
    ) -> None:
        self._aspect_mismatch(
            left.backend,
            right.backend,
            f"{role}.right.backend",
            "implicit cross-backend conversion between {!r} and {!r} is rejected",
        )
        if (
            left.lineage is not None
            and right.lineage is not None
            and left.lineage != right.lineage
        ):
            self.issue(f"{role}.right.lineage", "schema_mismatch", lineage_message)

    def _safe_array_dtype(
        self,
        backend: str | None,
        left: str | None,
        right: str | None,
        role: str,
        /,
    ) -> str | None:
        if left is None:
            return right
        if right is None:
            return left
        if backend is not None:
            from calc_flow.array import _symbolic_result_dtype

            try:
                return _symbolic_result_dtype(backend, "matmul", left, right)
            except (KeyError, TypeError, ValueError):
                pass
        elif left == right:
            return left
        self.issue(
            f"{role}.right.dtype",
            "unsupported_type",
            f"no provable result dtype for {left!r} and {right!r}",
        )
        return None

    def _aspect_mismatch(
        self, left: object, right: object, path: str, template: str, /
    ) -> None:
        if left is not None and right is not None and left != right:
            self.issue(path, "unsupported_type", template.format(left, right))

    def _matmul_inner_dims(
        self, left: ArrayFacts, right: ArrayFacts, role: str, /
    ) -> tuple[int | str, int | str] | None:
        """Check the inner-dimension contract; None marks a known mismatch."""

        inner_left = left.shape[1]
        inner_right = right.shape[0]
        if inner_left == inner_right:
            return left.shape[0], right.shape[1]
        if isinstance(inner_left, int) and isinstance(inner_right, int):
            self.issue(
                f"{role}.right.shape[0]",
                "schema_mismatch",
                f"matmul inner dimensions {inner_left} and {inner_right} do not match",
            )
            return None
        self.issue(
            f"{role}.right.shape[0]",
            "unresolved_type",
            f"matmul inner dimensions {inner_left!r} and {inner_right!r}"
            " cannot be proved equal",
        )
        return left.shape[0], right.shape[1]

    def _matmul(self, node: Node, path: str, /) -> ArrayFacts:
        role = f"{path}.matmul"
        left = self.array(node.args[0], f"{role}.left")
        right = self.array(node.args[1], f"{role}.right")
        states = left.state | right.state
        if len(left.shape) != 2 or len(right.shape) != 2:
            side = "left" if len(left.shape) != 2 else "right"
            rank = len(left.shape) if side == "left" else len(right.shape)
            self.issue(
                f"{role}.{side}.shape",
                "schema_mismatch",
                f"matmul requires rank-2 operands; got rank {rank}",
            )
            return ArrayFacts(None, None, (), None, states)
        domains_valid = self._array_numeric_domain(
            left.dtype,
            f"{role}.left",
            "matmul",
        ) and self._array_numeric_domain(
            right.dtype,
            f"{role}.right",
            "matmul",
        )
        dtype = self._array_pair_compat(
            left, right, role, "matmul operands carry different row-axis lineages"
        )
        if not domains_valid:
            dtype = None
        output_dims = self._matmul_inner_dims(left, right, role)
        shape = () if output_dims is None else output_dims
        return ArrayFacts(
            left.backend,
            dtype,
            shape,
            left.lineage,
            states,
        )

    def _array_operand(self, node: Node, operand_path: str, /) -> ArrayFacts:
        if node.op.name == "literal":
            return _array_literal_facts(node)
        return self.array(node, operand_path)

    def _array_numeric_domain(
        self,
        data_type: str | None,
        operand_path: str,
        primitive: str,
        /,
    ) -> bool:
        if data_type is None or data_type in _NUMERIC_TYPES:
            return True
        self.issue(
            f"{operand_path}.dtype",
            "unsupported_type",
            f"{primitive} is only defined for numeric arrays",
        )
        return False

    def _array_boolean_domain(
        self,
        data_type: str | None,
        operand_path: str,
        primitive: str,
        /,
    ) -> bool:
        if data_type is None or data_type == "bool":
            return True
        self.issue(
            f"{operand_path}.dtype",
            "unsupported_type",
            f"{primitive} is only defined for boolean arrays",
        )
        return False

    @staticmethod
    def _array_dtype_operand(
        node: Node,
        facts: ArrayFacts,
        /,
    ) -> str | bool | int | float | None:
        if node.op.name != "literal":
            return facts.dtype
        value = node.attr("value")
        if isinstance(value, (CBool, CInt, CFloat)):
            return value.value
        return None

    def _array_binary_domain(
        self,
        primitive: str,
        left: ArrayFacts,
        right: ArrayFacts,
        role: str,
        /,
    ) -> bool:
        if primitive in _BOOLEANS:
            left_valid = self._array_boolean_domain(
                left.dtype,
                f"{role}.left",
                primitive,
            )
            right_valid = self._array_boolean_domain(
                right.dtype,
                f"{role}.right",
                primitive,
            )
            return left_valid and right_valid
        if primitive in _ARITHMETIC or primitive in {"lt", "le", "gt", "ge"}:
            left_valid = self._array_numeric_domain(
                left.dtype,
                f"{role}.left",
                primitive,
            )
            right_valid = self._array_numeric_domain(
                right.dtype,
                f"{role}.right",
                primitive,
            )
            return left_valid and right_valid
        return True

    def _elementwise(self, node: Node, path: str, /) -> ArrayFacts:
        role = f"{path}.{node.op.name}"
        primitive = node.op.name
        left = self._array_operand(node.args[0], f"{role}.left")
        if len(node.args) == 1:
            if primitive == "not":
                domain_valid = self._array_boolean_domain(
                    left.dtype,
                    f"{role}.value",
                    primitive,
                )
            else:
                domain_valid = self._array_numeric_domain(
                    left.dtype,
                    f"{role}.value",
                    primitive,
                )
            dtype: str | None = None
            operand = self._array_dtype_operand(node.args[0], left)
            if primitive == "not" and domain_valid and left.dtype == "bool":
                dtype = "bool"
            elif domain_valid and left.backend is not None and operand is not None:
                from calc_flow.array import _symbolic_result_dtype

                try:
                    dtype = _symbolic_result_dtype(left.backend, primitive, operand)
                except (KeyError, TypeError, ValueError):
                    self.issue(
                        f"{role}.value.dtype",
                        "unsupported_type",
                        f"the {left.backend!r} provider cannot prove {primitive} dtype",
                    )
            return ArrayFacts(left.backend, dtype, left.shape, left.lineage, left.state)
        right = self._array_operand(node.args[1], f"{role}.right")
        self._array_pair_aspects(
            left,
            right,
            role,
            "array operands carry different row-axis lineages",
        )
        domain_valid = self._array_binary_domain(primitive, left, right, role)
        shape = _broadcast_shapes(left.shape, right.shape, role, self)
        backend = left.backend or right.backend
        dtype: str | None = None
        left_operand = self._array_dtype_operand(node.args[0], left)
        right_operand = self._array_dtype_operand(node.args[1], right)
        if (
            domain_valid
            and backend is not None
            and left_operand is not None
            and right_operand is not None
        ):
            from calc_flow.array import _symbolic_result_dtype

            try:
                dtype = _symbolic_result_dtype(
                    backend,
                    primitive,
                    left_operand,
                    right_operand,
                )
            except (KeyError, TypeError, ValueError):
                self.issue(
                    f"{role}.right.dtype",
                    "unsupported_type",
                    "the selected provider cannot prove a safe result dtype for"
                    f" {left.dtype!r} and {right.dtype!r}",
                )
        return ArrayFacts(
            backend,
            dtype,
            shape,
            left.lineage if left.lineage is not None else right.lineage,
            left.state | right.state,
        )


def _literal_column(node: Node, _path: str, /) -> ColumnFacts:
    return _literal_facts(node)


_ELEMENTWISE_PRIMITIVES: Final[frozenset[str]] = (
    _ARITHMETIC | _COMPARISONS | _BOOLEANS | frozenset({"neg", "not"})
)

_ARRAY_HANDLERS: Final[dict[str, Callable[[_Analyzer, Node, str], ArrayFacts]]] = {
    "parameter": _Analyzer._array_parameter,
    "from_columns": _Analyzer._from_columns,
    "matmul": _Analyzer._matmul,
    **dict.fromkeys(_ELEMENTWISE_PRIMITIVES, _Analyzer._elementwise),
}

_COLUMN_HANDLERS: Final[dict[str, Callable[[_Analyzer, Node, str], ColumnFacts]]] = {
    "column_ref": _Analyzer._column_ref,
    "literal": _literal_column,
    **dict.fromkeys(("add", "sub", "mul", "truediv"), _Analyzer._arithmetic),
    **dict.fromkeys(_COMPARISONS, _Analyzer._comparison),
    **dict.fromkeys(_BOOLEANS, _Analyzer._boolean_pair),
    **dict.fromkeys(("neg", "not"), _Analyzer._unary),
    "where": _Analyzer._where,
    "coalesce": _Analyzer._coalesce,
    **dict.fromkeys(_UNARY_NUMERIC, _Analyzer._scalar_function),
    "abs": _Analyzer._arithmetic_like_unary,
    "clip": _Analyzer._clip,
    "cast": _Analyzer._cast,
    **dict.fromkeys(("lag", "delta"), _Analyzer._lag_like),
    **dict.fromkeys(_ROLLING_AGGREGATES, _Analyzer._rolling_aggregate),
    **dict.fromkeys(("covariance", "correlation"), _Analyzer._rolling_pair),
    **dict.fromkeys(_CROSS_SECTION, _Analyzer._cross_section),
}


def _column_state_union(
    operands: tuple[ColumnFacts, ...] | list[ColumnFacts], /
) -> frozenset[str]:
    combined: set[str] = set()
    for facts in operands:
        combined |= facts.state
    return frozenset(combined)


def _array_literal_facts(node: Node, /) -> ArrayFacts:
    value = node.attr("value")
    dtype = _literal_dtype(value)
    return ArrayFacts(None, dtype, (), None, frozenset())


def _literal_dtype(value: CValue, /) -> str | None:
    if isinstance(value, CNull):
        return None
    if isinstance(value, CBool):
        return "bool"
    if isinstance(value, CInt):
        if -(2**63) <= value.value <= 2**63 - 1:
            return "int64"
        return "uint64"
    if isinstance(value, CFloat):
        return "float64"
    if isinstance(value, CStr):
        return "string"
    return None


def _literal_facts(node: Node, /) -> ColumnFacts:
    value = node.attr("value")
    return ColumnFacts(
        _literal_dtype(value),
        isinstance(value, CNull),
        None,
        frozenset(),
    )


def _broadcast_shapes(
    left: tuple[int | str, ...],
    right: tuple[int | str, ...],
    role: str,
    analyzer: _Analyzer,
    /,
) -> tuple[int | str, ...]:
    rank = max(len(left), len(right))
    result: list[int | str] = []
    for index in range(rank):
        first = left[len(left) - 1 - index] if index < len(left) else 1
        second = right[len(right) - 1 - index] if index < len(right) else 1
        result.append(
            _broadcast_dimension(first, second, role, rank - 1 - index, analyzer)
        )
    result.reverse()
    return tuple(result)


def _broadcast_dimension(
    first: int | str,
    second: int | str,
    role: str,
    index: int,
    analyzer: _Analyzer,
    /,
) -> int | str:
    if first == second:
        return first
    if first == 1:
        return second
    if second == 1:
        return first
    if isinstance(first, int) and isinstance(second, int):
        analyzer.issue(
            f"{role}.right.shape[{index}]",
            "schema_mismatch",
            f"array shapes {first} and {second} are not broadcast-compatible",
        )
        return first
    analyzer.issue(
        f"{role}.right.shape[{index}]",
        "unresolved_type",
        f"array dimensions {first!r} and {second!r} cannot be proved"
        " broadcast-compatible",
    )
    return first


def _require_runtime(runtime: object, /) -> Runtime:
    if not isinstance(runtime, Runtime):
        raise TypeError(
            f"analyze requires an explicit calc_flow Runtime; got {type_name(runtime)}"
        )
    return runtime


def _require_mode(mode: object, /) -> CompileMode:
    if type(mode) is not str or mode not in _MODES:
        raise TypeError(
            f"mode must be 'batch' or 'stream'; got {type_name(mode)}"
            if type(mode) is not str
            else f"mode must be 'batch' or 'stream'; got {mode!r}"
        )
    return mode  # type: ignore[return-value]


def _run(
    program: object, runtime: Runtime, mode: CompileMode, /
) -> tuple[_Analyzer, RuntimeCapabilities]:
    capabilities = runtime.capabilities()
    declared = frozenset(value._node.digest for value in program.inputs)
    portable = frozenset((*capabilities.portable_arrow_types, _EVENT_TIME_TYPE))
    analyzer = _Analyzer(
        mode,
        declared,
        portable,
        "array" in capabilities.batch_kinds,
    )
    for value in program.inputs:
        node = value._node
        root = _declaration_root(node)
        name = _cstr(node.attr("name")) or ""
        analyzer.check_input_declaration(node, root, name)
    _analyze_outputs(program, analyzer)
    if mode == "stream":
        _check_stream_ordering_for_inputs(program, analyzer)
    return analyzer, capabilities


def _analyze_outputs(program: object, analyzer: _Analyzer, /) -> None:
    from calc_flow.symbolic.expr import ArrayExpr, TableExpr

    for output_name, value in program.outputs:
        path = f"outputs.{output_name}"
        if isinstance(value, TableExpr):
            analyzer.table(value._node, path)
        elif isinstance(value, ArrayExpr):
            facts = analyzer.array(value._node, path)
            analyzer.check_array_output_kind(output_name)
            analyzer.check_unbounded_output(output_name, facts)


def _declaration_root(node: Node, /) -> str:
    return "static_inputs" if node.op.name == "parameter" else "inputs"


def _check_stream_ordering_for_inputs(program: object, analyzer: _Analyzer, /) -> None:
    for value in program.inputs:
        node = value._node
        name = _cstr(node.attr("name")) or ""
        if name in analyzer.temporal_lineages:
            analyzer.check_stream_ordering(node, _declaration_root(node), name)


def analyze_program(
    program: object, runtime: object, mode: object, /
) -> AnalysisResult:
    """Analyze one program against one immutable capability snapshot."""

    runtime_value = _require_runtime(runtime)
    mode_value = _require_mode(mode)
    analyzer, capabilities = _run(program, runtime_value, mode_value)
    return AnalysisResult(
        mode=mode_value,
        program_fingerprint=program.fingerprint,
        capability_session_id=capabilities.scope.session_id,
        capability_revision=capabilities.scope.revision,
        issues=analyzer.issues,
    )


def explain_program(program: object, runtime: object, mode: object, /) -> str:
    """Render deterministic analysis facts for one program."""

    runtime_value = _require_runtime(runtime)
    mode_value = _require_mode(mode)
    analyzer, capabilities = _run(program, runtime_value, mode_value)
    lines = _explain_header(program, mode_value, capabilities)
    if program.inputs:
        lines.append("  inputs")
        lines.extend(_explain_input(value) for value in program.inputs)
    lines.append("  outputs")
    for output_name, value in program.outputs:
        lines.extend(_explain_output(output_name, value, analyzer))
    issues = analyzer.issues
    if not issues:
        from calc_flow.errors import CompileError
        from calc_flow.symbolic.lower import lower_program_document
        from calc_flow.symbolic.optimizer import explain_optimization

        try:
            document = lower_program_document(program, runtime_value, mode_value)
        except CompileError:
            pass
        else:
            lines.extend(explain_optimization(document))
    if issues:
        lines.append("  issues")
        lines.extend(
            f"    {issue.path}: {issue.code}: {issue.message}" for issue in issues
        )
    return "\n".join(lines)


def _explain_header(
    program: object, mode: CompileMode, capabilities: RuntimeCapabilities, /
) -> list[str]:
    return [
        f"program {program.name}",
        f"  mode {mode}",
        f"  fingerprint {program.fingerprint}",
        "capability session"
        f" {capabilities.scope.session_id} revision"
        f" {capabilities.scope.revision}",
    ]


def _explain_output(
    output_name: str, value: object, analyzer: _Analyzer, /
) -> list[str]:
    from calc_flow.symbolic.expr import ArrayExpr, TableExpr

    path = f"outputs.{output_name}"
    if isinstance(value, TableExpr):
        return _explain_table_output(output_name, analyzer.table(value._node, path))
    if isinstance(value, ArrayExpr):
        return _explain_array_output(output_name, analyzer.array(value._node, path))
    return []


def _explain_input(value: object, /) -> str:
    node = value._node
    name = _cstr(node.attr("name")) or ""
    if node.op.name != "parameter":
        return (
            f"    input {name} event_time"
            f" {_cstr(node.attr('event_time')) or 'none'} entity_by"
            f" {_render_names(_cstr_seq(node.attr('entity_by')))} sequence_by"
            f" {_render_names(_cstr_seq(node.attr('sequence_by')))}"
        )
    kind = node.attr("kind")
    if isinstance(kind, CEnum) and kind.variant == "array":
        return (
            f"    static_input {name} array backend"
            f" {_cstr(node.attr('backend'))} dtype"
            f" {_ctype_str(node.attr('dtype'))} shape"
            f" {_render_shape(_parameter_shape(node))}"
        )
    return (
        f"    static_input {name} table fields"
        f" {len(_schema_fields(node.attr('schema')))}"
    )


def _explain_table_output(output_name: str, facts: TableFacts, /) -> list[str]:
    lines = [f"    output {output_name} table"]
    lines.extend(
        f"      field {field.name} {field.data_type}"
        f" nullable={'true' if field.nullable else 'false'}"
        for field in facts.schema
    )
    lines.append(f"      state {_render_state(facts.state)}")
    return lines


def _explain_array_output(output_name: str, facts: ArrayFacts, /) -> list[str]:
    lineage = facts.lineage if facts.lineage is not None else "none"
    return [
        f"    output {output_name} array backend {facts.backend}"
        f" dtype {facts.dtype} shape {_render_shape(facts.shape)}"
        f" lineage {lineage}",
        f"      state {_render_state(facts.state)}",
    ]


def _parameter_shape(node: Node, /) -> tuple[int | str, ...]:
    shape = node.attr("shape")
    if not isinstance(shape, CSeq):
        return ()
    return tuple(item.value for item in shape.items if isinstance(item, CInt))


def _render_shape(shape: tuple[int | str, ...], /) -> str:
    return "(" + ", ".join(str(dimension) for dimension in shape) + ")"


def _render_names(names: tuple[str, ...], /) -> str:
    return "[" + ", ".join(names) + "]"


def _render_state(state: frozenset[str], /) -> str:
    if not state:
        return "stateless"
    return ", ".join(sorted(state))
