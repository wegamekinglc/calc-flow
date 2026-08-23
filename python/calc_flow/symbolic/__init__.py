"""The public symbolic declaration surface of ``calc_flow``.

``calc_flow.symbolic`` is the only public declaration module. It builds
immutable expression IR with canonical v1 digests and offers no data
execution path: there is no ``eval``, ``push``, ``value``, ``transform``,
preview evaluator, or formula parser. Semantics are frozen by
``.codex/artifacts/specs/symbolic-computation-contract.md`` and the exact
signatures by ``.codex/artifacts/api-notes/symbolic-computation-engine.md``.
"""

from __future__ import annotations

from calc_flow.symbolic.expr import (
    ArrayExpr,
    ColumnExpr,
    Expr,
    Parameter,
    TableExpr,
    parameter,
    table_input,
)
from calc_flow.symbolic.ops import cs, linalg, row, table, ts, window
from calc_flow.symbolic.types import Field
from calc_flow.symbolic.windows import (
    CrossSectionGroup,
    DurationFrame,
    EventTimeBucket,
    RowFrame,
    duration,
    event_time_bucket,
    exact_time,
    rows,
)

__all__ = [
    "ArrayExpr",
    "ColumnExpr",
    "CrossSectionGroup",
    "DurationFrame",
    "EventTimeBucket",
    "Expr",
    "Field",
    "Parameter",
    "RowFrame",
    "TableExpr",
    "cs",
    "duration",
    "event_time_bucket",
    "exact_time",
    "linalg",
    "parameter",
    "row",
    "rows",
    "table",
    "table_input",
    "ts",
    "window",
]
