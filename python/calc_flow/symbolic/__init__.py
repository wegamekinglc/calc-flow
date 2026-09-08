"""Compatibility imports for the expression API also exported by ``calc_flow``.

Expressions and programs retain canonical declaration identities. Execution
uses the Rust runtime; Python builders never become serialized project data.
"""

from __future__ import annotations

from calc_flow.symbolic.analyzer import AnalysisIssue, AnalysisResult
from calc_flow.symbolic.expr import (
    ArrayExpr,
    ColumnExpr,
    Expr,
    Parameter,
    TableExpr,
    lit,
    parameter,
    table_input,
)
from calc_flow.symbolic.ops import cs, linalg, row, table, ts, window
from calc_flow.symbolic.program import FeatureSet, Program
from calc_flow.symbolic.types import Field
from calc_flow.symbolic.windows import (
    CrossSectionGroup,
    DurationFrame,
    EventTimeBucket,
    RowFrame,
    WindowAggregate,
    duration,
    event_time_bucket,
    exact_time,
    rows,
)

__all__ = [
    "AnalysisIssue",
    "AnalysisResult",
    "ArrayExpr",
    "ColumnExpr",
    "CrossSectionGroup",
    "DurationFrame",
    "EventTimeBucket",
    "Expr",
    "FeatureSet",
    "Field",
    "Parameter",
    "Program",
    "RowFrame",
    "TableExpr",
    "WindowAggregate",
    "cs",
    "duration",
    "event_time_bucket",
    "exact_time",
    "linalg",
    "lit",
    "parameter",
    "row",
    "rows",
    "table",
    "table_input",
    "ts",
    "window",
]
