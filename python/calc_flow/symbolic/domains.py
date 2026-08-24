"""Domain admission templates for symbolic declarations.

The exact message templates are frozen by
``.codex/artifacts/api-notes/symbolic-computation-engine.md`` section 2.3.
"""

from __future__ import annotations

COLUMN_OPERATOR_ERROR = (
    "symbolic column operator {operator} requires ColumnExpr or a strict"
    " scalar literal; got {type}"
)
ARRAY_OPERATOR_ERROR = (
    "symbolic array operator {operator} requires ArrayExpr,"
    " Parameter[ArrayExpr], or a strict scalar literal; got {type}"
)
TABLE_OPERATOR_ERROR = (
    "symbolic table expressions do not support scalar operator {operator};"
    " select a column or use table/window operations"
)
NAMESPACE_ERROR = (
    "calc_flow.symbolic.{function}.{parameter}: expected {expected}; got {type}"
)
BOOL_ERROR = (
    "symbolic expressions have no truth value; use &, |, and ~ for symbolic"
    " boolean composition, or identical() for structural identity"
)


def type_name(value: object, /) -> str:
    """Return the module-qualified type name without a representation."""

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def column_operator_error(operator: str, operand: object, /) -> TypeError:
    return TypeError(
        COLUMN_OPERATOR_ERROR.format(operator=operator, type=type_name(operand))
    )


def array_operator_error(operator: str, operand: object, /) -> TypeError:
    return TypeError(
        ARRAY_OPERATOR_ERROR.format(operator=operator, type=type_name(operand))
    )


def table_operator_error(operator: str, /) -> TypeError:
    return TypeError(TABLE_OPERATOR_ERROR.format(operator=operator))


def namespace_error(
    function: str,
    parameter: str,
    expected: str,
    operand: object,
    /,
) -> TypeError:
    return TypeError(
        NAMESPACE_ERROR.format(
            function=function,
            parameter=parameter,
            expected=expected,
            type=type_name(operand),
        )
    )


def is_strict_scalar_type(value: object, /) -> bool:
    """Admit exactly the strict JSON scalar host types by exact type."""

    return value is None or type(value) in (bool, int, float, str)


def bool_error() -> TypeError:
    return TypeError(BOOL_ERROR)
