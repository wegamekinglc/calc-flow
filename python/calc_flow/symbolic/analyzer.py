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
_ARITHMETIC: Final = frozenset({"add", "sub", "mul", "truediv"})
_COMPARISONS: Final = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
_BOOLEANS: Final = frozenset({"and", "or"})
_UNARY_NUMERIC: Final = frozenset({"log", "exp", "sqrt"})
_ROLLING_AGGREGATES: Final = frozenset(
    {"count", "sum", "mean", "min", "max", "variance", "stddev"}
)
_ROLLING_DDOF: Final = frozenset({"variance", "stddev"})
_CROSS_SECTION: Final = frozenset(
    {"rank", "percentile", "demean", "zscore", "winsorize"}
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


def _schema_fields(value: CValue, /) -> tuple[Field, ...]:
    if not isinstance(value, CSeq):
        return ()
    fields: list[Field] = []
    for item in value.items:
        if not isinstance(item, CMap):
            continue
        name = item.get("name")
        data_type = item.get("data_type")
        nullable = item.get("nullable")
        if isinstance(name, CStr) and isinstance(nullable, CBool):
            type_name_value = (
                data_type.value
                if isinstance(data_type, CStr)
                else data_type.name
                if isinstance(data_type, CDType)
                else None
            )
            if type_name_value is not None:
                fields.append(Field(name.value, type_name_value, nullable.value))
    return tuple(fields)


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
        kind = (
            "array"
            if node.op.name == "parameter"
            and isinstance(node.attr("kind"), CEnum)
            and node.attr("kind").variant == "array"
            else "table"
        )
        if kind == "array" and not self._supports_array_kind:
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
        event_time = _cstr(node.attr("event_time"))
        if event_time is None:
            self.issue(
                f"{root}.{name}.event_time",
                "ordering_required",
                "temporal work in stream mode requires a declared event-time column",
            )
        else:
            field = schema.get(event_time)
            if field is None or field.data_type != _EVENT_TIME_TYPE or field.nullable:
                self.issue(
                    f"{root}.{name}.event_time",
                    "ordering_required",
                    "the event-time column must be a non-null timestamp[us, UTC] field",
                )
        entity_by = _cstr_seq(node.attr("entity_by"))
        if not entity_by:
            self.issue(
                f"{root}.{name}.entity_by",
                "ordering_required",
                "temporal work in stream mode requires a non-empty entity key",
            )
        for index, field_name in enumerate(entity_by):
            if field_name not in schema:
                self.issue(
                    f"{root}.{name}.entity_by[{index}]",
                    "ordering_required",
                    f"entity field {field_name!r} is not in the input schema",
                )
        sequence_by = _cstr_seq(node.attr("sequence_by"))
        if not sequence_by:
            self.issue(
                f"{root}.{name}.sequence_by",
                "ordering_required",
                "temporal work in stream mode requires a non-empty sequence key",
            )
        for index, field_name in enumerate(sequence_by):
            field = schema.get(field_name)
            if field is None or field.nullable or field.data_type in _FLOATING_TYPES:
                self.issue(
                    f"{root}.{name}.sequence_by[{index}]",
                    "ordering_required",
                    "sequence fields must be non-null with a portable total"
                    " order; floating sequence fields are forbidden",
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
        if array.lineage is None or array.lineage != child.lineage:
            self.issue(
                f"{path}.attach_columns.array.lineage",
                "schema_mismatch",
                "an attached array must carry the row-axis lineage of the"
                f" target table {child.lineage!r}",
            )
        elif array.shape:
            width = array.shape[-1]
            if isinstance(width, str):
                self.issue(
                    f"{path}.attach_columns.array.shape[{len(array.shape) - 1}]",
                    "unresolved_type",
                    "the attached array width is symbolic and cannot be proved"
                    " to match the declared names",
                )
            elif width != len(names):
                self.issue(
                    f"{path}.attach_columns.array.shape[{len(array.shape) - 1}]",
                    "schema_mismatch",
                    f"array width {width} does not match {len(names)} declared names",
                )
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
        return TableFacts(
            tuple(fields),
            child.lineage,
            child.state | array.state,
            child.event_time,
            child.entity_by,
            child.sequence_by,
        )

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
        name = node.op.name
        if name == "column_ref":
            return self._column_ref(node, path)
        if name == "literal":
            return _literal_facts(node)
        if name in ("add", "sub", "mul", "truediv"):
            return self._arithmetic(node, path)
        if name in _COMPARISONS:
            return self._comparison(node, path)
        if name in _BOOLEANS:
            return self._boolean_pair(node, path)
        if name in ("neg", "not"):
            return self._unary(node, path)
        if name == "where":
            return self._where(node, path)
        if name == "coalesce":
            return self._coalesce(node, path)
        if name in _UNARY_NUMERIC:
            return self._scalar_function(node, path)
        if name == "abs":
            return self._arithmetic_like_unary(node, path)
        if name == "clip":
            return self._clip(node, path)
        if name == "cast":
            return self._cast(node, path)
        if name in ("lag", "delta"):
            return self._lag_like(node, path)
        if name in _ROLLING_AGGREGATES:
            return self._rolling_aggregate(node, path)
        if name in ("covariance", "correlation"):
            return self._rolling_pair(node, path)
        if name in _CROSS_SECTION:
            return self._cross_section(node, path)
        self.issue(
            path,
            "unknown_primitive_version",
            f"primitive {name!r} does not produce a column value",
        )
        return ColumnFacts(None, True, None, frozenset())

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
        if isinstance(kind, CEnum) and kind.variant == "rows":
            size = _cint(frame.get("size"))
            return f"rows({size})" if size is not None else None
        if isinstance(kind, CEnum) and kind.variant == "duration":
            micros = _cint(frame.get("micros"))
            return f"duration({micros})" if micros is not None else None
        return None

    def _rolling_aggregate(self, node: Node, path: str, /) -> ColumnFacts:
        operand = self._operand(node.args[0], f"{path}.{node.op.name}.value", None)
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
        if primitive == "count":
            output_type: str | None = "uint64"
        elif primitive == "sum":
            if input_type in _SIGNED_INT_TYPES:
                output_type = "int64"
            elif input_type in _UNSIGNED_INT_TYPES:
                output_type = "uint64"
            else:
                output_type = "float64"
        elif primitive in ("min", "max"):
            output_type = input_type
        else:
            output_type = "float64"
        return ColumnFacts(output_type, True, operand.lineage, states)

    def _rolling_pair(self, node: Node, path: str, /) -> ColumnFacts:
        role = f"{path}.{node.op.name}"
        left = self._operand(node.args[0], f"{role}.left", None)
        right = self._operand(node.args[1], f"{role}.right", left.lineage)
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
        if operand.lineage is not None:
            self._temporal_lineages.add(operand.lineage)
        group_paths = (
            f"{role}.event_time",
            *(f"{role}.partition_by[{index}]" for index in range(len(node.args) - 2)),
        )
        group = self._anchored_operands(node.args[1:], group_paths, operand.lineage)
        states = (
            operand.state | _column_state_union(group) | frozenset({"cross_section"})
        )
        if node.op.name == "winsorize":
            if operand.data_type is not None and operand.data_type not in (
                "float32",
                "float64",
            ):
                self.issue(
                    f"{role}.value.dtype",
                    "unsupported_type",
                    "winsorization is only provable for floating columns",
                )
                return ColumnFacts(None, True, operand.lineage, states)
            return ColumnFacts(operand.data_type, True, operand.lineage, states)
        if node.op.name in ("zscore", "demean") and not self._numeric_or_issue(
            operand.data_type, f"{role}.value", node.op.name
        ):
            return ColumnFacts(None, True, operand.lineage, states)
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
        name = node.op.name
        if name == "parameter":
            return self._array_parameter(node, path)
        if name == "from_columns":
            return self._from_columns(node, path)
        if name == "matmul":
            return self._matmul(node, path)
        if (
            name in _ARITHMETIC
            or name in _COMPARISONS
            or name in _BOOLEANS
            or name in ("neg", "not")
        ):
            return self._elementwise(node, path)
        self.issue(
            path,
            "unknown_primitive_version",
            f"primitive {name!r} does not produce an array value",
        )
        return ArrayFacts(None, None, (), None, frozenset())

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
        data_type: str | None = None
        for index, column in enumerate(columns):
            field = next((item for item in table.schema if item.name == column), None)
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
                data_type = None
                break
        rows: int | str = table.lineage if table.lineage is not None else "rows"
        return ArrayFacts(
            _cstr(node.attr("backend")),
            data_type,
            (rows, len(columns)),
            table.lineage,
            table.state,
        )

    def _matmul(self, node: Node, path: str, /) -> ArrayFacts:
        role = f"{path}.matmul"
        left = self.array(node.args[0], f"{role}.left")
        right = self.array(node.args[1], f"{role}.right")
        if len(left.shape) != 2 or len(right.shape) != 2:
            side = "left" if len(left.shape) != 2 else "right"
            self.issue(
                f"{role}.{side}.shape",
                "schema_mismatch",
                f"matmul requires rank-2 operands; got rank"
                f" {len(left.shape) if side == 'left' else len(right.shape)}",
            )
            return ArrayFacts(None, None, (), None, left.state | right.state)
        if (
            right.dtype is not None
            and left.dtype is not None
            and left.dtype != right.dtype
        ):
            self.issue(
                f"{role}.right.dtype",
                "unsupported_type",
                f"no provable result dtype for {left.dtype!r} and {right.dtype!r}",
            )
        if (
            right.backend is not None
            and left.backend is not None
            and left.backend != right.backend
        ):
            self.issue(
                f"{role}.right.backend",
                "unsupported_type",
                f"implicit cross-backend conversion between {left.backend!r}"
                f" and {right.backend!r} is rejected",
            )
        if (
            right.lineage is not None
            and left.lineage is not None
            and left.lineage != right.lineage
        ):
            self.issue(
                f"{role}.right.lineage",
                "schema_mismatch",
                "matmul operands carry different row-axis lineages",
            )
        inner_left = left.shape[1]
        inner_right = right.shape[0]
        if isinstance(inner_left, int) and isinstance(inner_right, int):
            if inner_left != inner_right:
                self.issue(
                    f"{role}.right.shape[0]",
                    "schema_mismatch",
                    f"matmul inner dimensions {inner_left} and {inner_right} do"
                    " not match",
                )
                return ArrayFacts(
                    left.backend,
                    left.dtype,
                    (),
                    left.lineage,
                    left.state | right.state,
                )
        elif inner_left != inner_right:
            self.issue(
                f"{role}.right.shape[0]",
                "unresolved_type",
                f"matmul inner dimensions {inner_left!r} and {inner_right!r}"
                " cannot be proved equal",
            )
        return ArrayFacts(
            left.backend,
            left.dtype,
            (left.shape[0], right.shape[1]),
            left.lineage,
            left.state | right.state,
        )

    def _elementwise(self, node: Node, path: str, /) -> ArrayFacts:
        role = f"{path}.{node.op.name}"
        primitive = node.op.name
        left_node = node.args[0]
        left = (
            _array_literal_facts(left_node)
            if left_node.op.name == "literal"
            else self.array(left_node, f"{role}.left")
        )
        if len(node.args) == 1:
            dtype = "bool" if primitive == "not" else left.dtype
            return ArrayFacts(left.backend, dtype, left.shape, left.lineage, left.state)
        right_node = node.args[1]
        right = (
            _array_literal_facts(right_node)
            if right_node.op.name == "literal"
            else self.array(right_node, f"{role}.right")
        )
        if (
            left.dtype is not None
            and right.dtype is not None
            and left.dtype != right.dtype
        ):
            self.issue(
                f"{role}.right.dtype",
                "unsupported_type",
                f"no provable result dtype for {left.dtype!r} and {right.dtype!r}",
            )
        if (
            right.lineage is not None
            and left.lineage is not None
            and left.lineage != right.lineage
        ):
            self.issue(
                f"{role}.right.lineage",
                "schema_mismatch",
                "array operands carry different row-axis lineages",
            )
        if (
            right.backend is not None
            and left.backend is not None
            and left.backend != right.backend
        ):
            self.issue(
                f"{role}.right.backend",
                "unsupported_type",
                f"implicit cross-backend conversion between {left.backend!r}"
                f" and {right.backend!r} is rejected",
            )
        shape = _broadcast_shapes(
            left.shape,
            right.shape,
            role,
            self,
        )
        dtype: str | None
        if primitive in _COMPARISONS or primitive in _BOOLEANS:
            dtype = "bool"
        else:
            dtype = left.dtype
        return ArrayFacts(
            left.backend,
            dtype,
            shape,
            left.lineage if left.lineage is not None else right.lineage,
            left.state | right.state,
        )


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
    from calc_flow.symbolic.expr import ArrayExpr, TableExpr

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
        root = "static_inputs" if node.op.name == "parameter" else "inputs"
        name = _cstr(node.attr("name")) or ""
        analyzer.check_input_declaration(node, root, name)
    for output_name, value in program.outputs:
        if isinstance(value, TableExpr):
            analyzer.table(value._node, f"outputs.{output_name}")
        elif isinstance(value, ArrayExpr):
            facts = analyzer.array(value._node, f"outputs.{output_name}")
            analyzer.check_array_output_kind(output_name)
            analyzer.check_unbounded_output(output_name, facts)
    if mode == "stream":
        for value in program.inputs:
            node = value._node
            root = "static_inputs" if node.op.name == "parameter" else "inputs"
            name = _cstr(node.attr("name")) or ""
            if name in analyzer.temporal_lineages:
                analyzer.check_stream_ordering(node, root, name)
    return analyzer, capabilities


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

    from calc_flow.symbolic.expr import ArrayExpr, TableExpr

    runtime_value = _require_runtime(runtime)
    mode_value = _require_mode(mode)
    analyzer, capabilities = _run(program, runtime_value, mode_value)
    lines = [
        f"program {program.name}",
        f"  mode {mode_value}",
        f"  fingerprint {program.fingerprint}",
        "capability session"
        f" {capabilities.scope.session_id} revision"
        f" {capabilities.scope.revision}",
    ]
    if program.inputs:
        lines.append("  inputs")
        for value in program.inputs:
            node = value._node
            name = _cstr(node.attr("name")) or ""
            if node.op.name == "parameter":
                kind = node.attr("kind")
                if isinstance(kind, CEnum) and kind.variant == "array":
                    lines.append(
                        f"    static_input {name} array backend"
                        f" {_cstr(node.attr('backend'))} dtype"
                        f" {_ctype_str(node.attr('dtype'))} shape"
                        f" {_render_shape(_parameter_shape(node))}"
                    )
                else:
                    lines.append(
                        f"    static_input {name} table fields"
                        f" {len(_schema_fields(node.attr('schema')))}"
                    )
            else:
                lines.append(
                    f"    input {name} event_time"
                    f" {_cstr(node.attr('event_time')) or 'none'} entity_by"
                    f" {_render_names(_cstr_seq(node.attr('entity_by')))}"
                    f" sequence_by"
                    f" {_render_names(_cstr_seq(node.attr('sequence_by')))}"
                )
    lines.append("  outputs")
    for output_name, value in program.outputs:
        if isinstance(value, TableExpr):
            facts = analyzer.table(value._node, f"outputs.{output_name}")
            lines.append(f"    output {output_name} table")
            for field in facts.schema:
                lines.append(
                    f"      field {field.name} {field.data_type}"
                    f" nullable={'true' if field.nullable else 'false'}"
                )
            lines.append(f"      state {_render_state(facts.state)}")
        elif isinstance(value, ArrayExpr):
            facts = analyzer.array(value._node, f"outputs.{output_name}")
            lines.append(
                f"    output {output_name} array backend {facts.backend}"
                f" dtype {facts.dtype} shape {_render_shape(facts.shape)}"
                f" lineage {facts.lineage if facts.lineage is not None else 'none'}"
            )
            lines.append(f"      state {_render_state(facts.state)}")
    issues = analyzer.issues
    if issues:
        lines.append("  issues")
        for issue in issues:
            lines.append(f"    {issue.path}: {issue.code}: {issue.message}")
    return "\n".join(lines)


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
