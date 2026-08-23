"""Immutable symbolic expressions for tables, columns, arrays, parameters.

Expression objects are declaration-only values: they compose into a node
graph, expose a canonical digest and a deterministic explanation, and offer
no data-execution path. Operator dunders build new expressions; ``==`` and
friends never produce Python booleans, and structural identity is available
only through ``identical()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from calc_flow.symbolic.domains import (
    array_operator_error,
    bool_error,
    column_operator_error,
    is_strict_scalar_type,
    table_operator_error,
    type_name,
)
from calc_flow.symbolic.nodes import (
    CBool,
    CDType,
    CEnum,
    CInt,
    CMap,
    CNull,
    CSeq,
    CStr,
    Node,
    build,
    explain_node,
    literal_value,
)
from calc_flow.symbolic.types import (
    BatchKind,
    Field,
    ScalarLiteral,
    check_array_dtype,
    check_table_field_type,
    require_non_empty_str,
    require_non_negative_int,
)

type ColumnOperand = ColumnExpr | ScalarLiteral
type ArrayOperand = ArrayExpr | Parameter[ArrayExpr] | ScalarLiteral


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Expr[T]:
    """Base class of every immutable symbolic expression."""

    _node: Node

    @property
    def digest(self) -> str:
        """The lowercase hexadecimal v1 node digest of this declaration."""

        return self._node.digest

    def identical(self, other: object, /) -> bool:
        """Compare the complete normalized structure of two expressions."""

        if not isinstance(other, Expr):
            return False
        return (
            self._node.digest == other._node.digest
            and self._node.node_bytes == other._node.node_bytes
        )

    def explain(self) -> str:
        """Render the declaration tree deterministically."""

        return explain_node(self._node)

    def __bool__(self) -> bool:
        raise bool_error()

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}(digest={self._node.digest})"

    __hash__ = None


def _literal_node(value: ScalarLiteral, /) -> Node:
    return build("literal", (), {"value": literal_value(value)})


def _column_operand(operand: object, operator: str, /) -> Node:
    if isinstance(operand, ColumnExpr):
        return operand._node
    if is_strict_scalar_type(operand):
        return _literal_node(operand)  # type: ignore[arg-type]
    raise column_operator_error(operator, operand)


def _array_operand(operand: object, operator: str, /) -> Node:
    if isinstance(operand, ArrayExpr):
        return operand._node
    if isinstance(operand, Parameter) and operand.kind == "array":
        return operand._node
    if is_strict_scalar_type(operand):
        return _literal_node(operand)  # type: ignore[arg-type]
    raise array_operator_error(operator, operand)


def _column_binary(
    left: ColumnExpr, right: object, primitive: str, operator: str, /
) -> ColumnExpr:
    return ColumnExpr(
        build(
            primitive,
            (
                _column_operand(left, operator),
                _column_operand(right, operator),
            ),
            {},
        )
    )


def _array_binary(
    left: ArrayExpr, right: object, primitive: str, operator: str, /
) -> ArrayExpr:
    return ArrayExpr(
        build(
            primitive,
            (_array_operand(left, operator), _array_operand(right, operator)),
            {},
        )
    )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ColumnExpr(Expr[object]):
    """A row-local column expression over one table lineage."""

    def __eq__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "eq", "==")

    def __ne__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "ne", "!=")

    def __lt__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "lt", "<")

    def __le__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "le", "<=")

    def __gt__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "gt", ">")

    def __ge__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "ge", ">=")

    def __add__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "add", "+")

    def __radd__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "add", "+")

    def __sub__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "sub", "-")

    def __rsub__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _reflected_column(self, other, "sub", "-")

    def __mul__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "mul", "*")

    def __rmul__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "mul", "*")

    def __truediv__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "truediv", "/")

    def __rtruediv__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _reflected_column(self, other, "truediv", "/")

    def __neg__(self, /) -> ColumnExpr:
        return ColumnExpr(build("neg", (self._node,), {}))

    def __and__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "and", "&")

    def __rand__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "and", "&")

    def __or__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "or", "|")

    def __ror__(self, other: ColumnOperand, /) -> ColumnExpr:
        return _column_binary(self, other, "or", "|")

    def __invert__(self, /) -> ColumnExpr:
        return ColumnExpr(build("not", (self._node,), {}))


def _reflected_column(
    left: ColumnExpr, right: object, primitive: str, operator: str, /
) -> ColumnExpr:
    """Build the reflected non-commutative operators ``right <op> left``."""

    return ColumnExpr(
        build(
            primitive,
            (
                _column_operand(right, operator),
                _column_operand(left, operator),
            ),
            {},
        )
    )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ArrayExpr(Expr[object]):
    """An Array API-backed expression with backend, dtype, and lineage."""

    def __eq__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "eq", "==")

    def __ne__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "ne", "!=")

    def __lt__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "lt", "<")

    def __le__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "le", "<=")

    def __gt__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "gt", ">")

    def __ge__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "ge", ">=")

    def __add__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "add", "+")

    def __radd__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "add", "+")

    def __sub__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "sub", "-")

    def __rsub__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _reflected_array(self, other, "sub", "-")

    def __mul__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "mul", "*")

    def __rmul__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "mul", "*")

    def __truediv__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "truediv", "/")

    def __rtruediv__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _reflected_array(self, other, "truediv", "/")

    def __neg__(self, /) -> ArrayExpr:
        return ArrayExpr(build("neg", (self._node,), {}))

    def __and__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "and", "&")

    def __rand__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "and", "&")

    def __or__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "or", "|")

    def __ror__(self, other: ArrayOperand, /) -> ArrayExpr:
        return _array_binary(self, other, "or", "|")

    def __invert__(self, /) -> ArrayExpr:
        return ArrayExpr(build("not", (self._node,), {}))


def _reflected_array(
    left: ArrayExpr, right: object, primitive: str, operator: str, /
) -> ArrayExpr:
    """Build the reflected non-commutative operators ``right <op> left``."""

    return ArrayExpr(
        build(
            primitive,
            (
                _array_operand(right, operator),
                _array_operand(left, operator),
            ),
            {},
        )
    )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class TableExpr(Expr[object]):
    """An Arrow-backed table expression."""

    def __eq__(self, other: object, /) -> bool:
        raise table_operator_error("==")

    def __ne__(self, other: object, /) -> bool:
        raise table_operator_error("!=")

    def __lt__(self, other: object, /) -> bool:
        raise table_operator_error("<")

    def __le__(self, other: object, /) -> bool:
        raise table_operator_error("<=")

    def __gt__(self, other: object, /) -> bool:
        raise table_operator_error(">")

    def __ge__(self, other: object, /) -> bool:
        raise table_operator_error(">=")

    def __add__(self, other: object, /) -> bool:
        raise table_operator_error("+")

    def __radd__(self, other: object, /) -> bool:
        raise table_operator_error("+")

    def __sub__(self, other: object, /) -> bool:
        raise table_operator_error("-")

    def __rsub__(self, other: object, /) -> bool:
        raise table_operator_error("-")

    def __mul__(self, other: object, /) -> bool:
        raise table_operator_error("*")

    def __rmul__(self, other: object, /) -> bool:
        raise table_operator_error("*")

    def __truediv__(self, other: object, /) -> bool:
        raise table_operator_error("/")

    def __rtruediv__(self, other: object, /) -> bool:
        raise table_operator_error("/")

    def __neg__(self, /) -> bool:
        raise table_operator_error("unary -")

    def __and__(self, other: object, /) -> bool:
        raise table_operator_error("&")

    def __rand__(self, other: object, /) -> bool:
        raise table_operator_error("&")

    def __or__(self, other: object, /) -> bool:
        raise table_operator_error("|")

    def __ror__(self, other: object, /) -> bool:
        raise table_operator_error("|")

    def __invert__(self, /) -> bool:
        raise table_operator_error("~")

    def __getitem__(self, field: str, /) -> ColumnExpr:
        if type(field) is not str:
            raise TypeError(
                "TableExpr field selection requires a string field name;"
                f" got {type_name(field)}"
            )
        return ColumnExpr(build("column_ref", (self._node,), {"name": CStr(field)}))


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Parameter[T](Expr[T]):
    """A named external static input of table or array kind.

    Parameters define no scalar operator dunders; an array-kind parameter
    participates in expressions only where a signature names it.
    """

    @property
    def name(self) -> str:
        value = self._node.attr("name")
        if isinstance(value, CStr):
            return value.value
        raise TypeError("parameter node is missing its name attribute")

    @property
    def kind(self) -> BatchKind:
        value = self._node.attr("kind")
        if isinstance(value, CEnum):
            return value.variant  # type: ignore[return-value]
        raise TypeError("parameter node is missing its kind attribute")


def _validate_schema(schema: Sequence[Field], path: str, /) -> tuple[Field, ...]:
    fields: list[Field] = []
    seen: set[str] = set()
    for index, field in enumerate(schema):
        if not isinstance(field, Field):
            raise TypeError(
                f"{path}[{index}]: schema entries must be Field values; got"
                f" {type_name(field)}"
            )
        check_table_field_type(field.data_type, f"{path}[{index}].data_type")
        if field.name in seen:
            raise ValueError(
                f"{path}[{index}].name: duplicate_name: duplicate field"
                f" name {field.name!r}"
            )
        seen.add(field.name)
        fields.append(field)
    return tuple(fields)


def _validate_name_sequence(values: object, path: str, /) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(
            f"{path} must be a sequence of strings; got {type_name(values)}"
        )
    names: list[str] = []
    for index, value in enumerate(values):
        names.append(require_non_empty_str(value, f"{path}[{index}]"))
    return tuple(names)


def _cv_schema(fields: Sequence[Field], /) -> CSeq:
    return CSeq(
        tuple(
            CMap.from_mapping(
                {
                    "name": CStr(field.name),
                    "data_type": CDType(field.data_type),
                    "nullable": CBool(field.nullable),
                }
            )
            for field in fields
        )
    )


def table_input(
    name: str,
    /,
    *,
    schema: Sequence[Field],
    entity_by: Sequence[str] = (),
    event_time: str | None = None,
    sequence_by: Sequence[str] = (),
) -> TableExpr:
    """Declare one named table input with its exact schema and ordering."""

    if type(name) is not str:
        raise TypeError(
            f"inputs.table_input.name must be a non-empty string; got {type_name(name)}"
        )
    if not name:
        raise ValueError(
            "inputs.table_input.name: invalid_literal: must be a non-empty string"
        )
    base = f"inputs.{name}"
    fields = _validate_schema(schema, f"{base}.schema")
    entity = _validate_name_sequence(entity_by, f"{base}.entity_by")
    sequence = _validate_name_sequence(sequence_by, f"{base}.sequence_by")
    if event_time is not None:
        require_non_empty_str(event_time, f"{base}.event_time")
    node = build(
        "table_input",
        (),
        {
            "name": CStr(name),
            "schema": _cv_schema(fields),
            "entity_by": CSeq(tuple(CStr(value) for value in entity)),
            "event_time": CNull() if event_time is None else CStr(event_time),
            "sequence_by": CSeq(tuple(CStr(value) for value in sequence)),
        },
    )
    return TableExpr(node)


def _is_absent_or_empty(value: object, /) -> bool:
    """Type-safe absence check that never triggers element-wise equality.

    Only ``None`` and a genuinely empty ``list``/``tuple`` count as absent;
    array-like values (for example NumPy arrays) fail without invoking
    their overloaded ``==`` so the stable rejection path is reached.
    """

    return value is None or (isinstance(value, (list, tuple)) and len(value) == 0)


@overload
def parameter(
    name: str,
    /,
    *,
    kind: Literal["table"],
    schema: Sequence[Field],
    backend: None = None,
    dtype: None = None,
    shape: Sequence[int] = (),
    mutability: Literal["static"] = "static",
) -> Parameter[TableExpr]: ...


@overload
def parameter(
    name: str,
    /,
    *,
    kind: Literal["array"],
    schema: Sequence[Field] = (),
    backend: str,
    dtype: str,
    shape: Sequence[int],
    mutability: Literal["static"] = "static",
) -> Parameter[ArrayExpr]: ...


def parameter(
    name: str,
    /,
    *,
    kind: object,
    schema: object = (),
    backend: object = None,
    dtype: object = None,
    shape: object = (),
    mutability: object = "static",
) -> Parameter[object]:
    """Declare one named static external parameter of table or array kind."""

    if type(name) is not str or not name:
        if type(name) is not str:
            raise TypeError(
                "static_inputs.parameter.name must be a non-empty string;"
                f" got {type_name(name)}"
            )
        raise ValueError(
            "static_inputs.parameter.name: invalid_literal: must be a non-empty string"
        )
    base = f"static_inputs.{name}"
    if type(kind) is not str or kind not in ("table", "array"):
        if type(kind) is not str:
            raise TypeError(f"{base}.kind must be a string; got {type_name(kind)}")
        raise ValueError(
            f"{base}.kind: invalid_literal: kind must be 'table' or 'array'"
        )
    if type(mutability) is not str:
        raise TypeError(
            f"{base}.mutability must be a string; got {type_name(mutability)}"
        )
    if mutability != "static":
        raise ValueError(
            f"{base}.mutability: invalid_literal: the only accepted"
            " mutability is 'static'"
        )
    if kind == "table":
        for field_name, value in (("backend", backend), ("dtype", dtype)):
            if value is not None:
                raise ValueError(
                    f"{base}.{field_name}: invalid_literal: table"
                    f" parameters do not accept {field_name}"
                )
        if not _is_absent_or_empty(shape):
            raise ValueError(
                f"{base}.shape: invalid_literal: table parameters do not accept shape"
            )
        fields = _validate_schema(schema, f"{base}.schema")  # type: ignore[arg-type]
        node = build(
            "parameter",
            (),
            {
                "name": CStr(name),
                "kind": CEnum("batch_kind", "table"),
                "mutability": CEnum("mutability", "static"),
                "schema": _cv_schema(fields),
            },
        )
        return Parameter(node)
    if not _is_absent_or_empty(schema):
        raise ValueError(
            f"{base}.schema: invalid_literal: array parameters do not accept schema"
        )
    backend_value = require_non_empty_str(backend, f"{base}.backend")
    dtype_value = require_non_empty_str(dtype, f"{base}.dtype")
    check_array_dtype(dtype_value, f"{base}.dtype")
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise TypeError(
            f"{base}.shape must be a sequence of integers; got {type_name(shape)}"
        )
    dims = tuple(
        require_non_negative_int(value, f"{base}.shape[{index}]")
        for index, value in enumerate(shape)
    )
    node = build(
        "parameter",
        (),
        {
            "name": CStr(name),
            "kind": CEnum("batch_kind", "array"),
            "mutability": CEnum("mutability", "static"),
            "backend": CStr(backend_value),
            "dtype": CDType(dtype_value),
            "shape": CSeq(tuple(CInt(value) for value in dims)),
        },
    )
    return Parameter(node)
