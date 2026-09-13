"""Paired declarations for a native stateful stage and its rejected rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from calc_flow.symbolic.expr import TableExpr, table_input
from calc_flow.symbolic.nodes import CInt, Node, build
from calc_flow.symbolic.types import Field

if TYPE_CHECKING:
    from calc_flow.symbolic.analyzer import TableFacts, _Analyzer
    from calc_flow.symbolic.program import Program


@dataclass(frozen=True, slots=True)
class LateOutputs:
    """Normal and late table references sharing one native stateful stage."""

    output: TableExpr
    late: TableExpr


def with_late_output(
    value: TableExpr, /, *, allowed_lateness_micros: int = 0
) -> LateOutputs:
    """Declare both outputs of a stateful stage for explicit Program consumption."""
    if not isinstance(value, TableExpr):
        raise TypeError("with_late_output.value: expected TableExpr")
    if type(allowed_lateness_micros) is not int:
        raise TypeError("with_late_output.allowed_lateness_micros: expected int")
    if not 0 <= allowed_lateness_micros < 1 << 64:
        raise ValueError(
            "with_late_output.allowed_lateness_micros: expected unsigned 64-bit integer"
        )
    owner = build(
        "late_output",
        (value._node,),
        {
            "allowed_lateness_micros": CInt(allowed_lateness_micros),
            "schema_version": CInt(1),
            "metrics_version": CInt(1),
        },
    )
    return LateOutputs(TableExpr(owner), TableExpr(build("late_rows", (owner,), {})))


def check_consumption(program: Program, analyzer: _Analyzer) -> bool:
    uses: dict[str, set[str]] = {}
    paths: dict[str, str] = {}
    visited: set[str] = set()

    def visit(node: Node, path: str) -> None:
        if node.digest in visited:
            return
        visited.add(node.digest)
        if node.op.name in {"late_output", "late_rows"}:
            owner = node if node.op.name == "late_output" else node.args[0]
            uses.setdefault(owner.digest, set()).add(node.op.name)
            paths.setdefault(owner.digest, path)
            if not _is_single_stage(owner.args[0]):
                analyzer.issue(
                    path,
                    "ambiguous_late_stage",
                    "with_late_output requires one current rolling or cross-section"
                    " stage with explicit existing-column operands;"
                    " split the stage declaration",
                )
            visit(owner.args[0], path)
            return
        for child in node.args:
            visit(child, path)

    for name, value in program.outputs:
        visit(value._node, f"outputs.{name}")
    for digest, branches in uses.items():
        if branches != {"late_output", "late_rows"}:
            analyzer.issue(
                paths[digest],
                "unconsumed_output",
                "consume both output and late explicitly in a Program",
            )
        if analyzer._mode != "stream":
            analyzer.issue(
                paths[digest], "unsupported_mode", "late outputs require stream mode"
            )

    return bool(uses)


_DIAGNOSTICS = (
    ("_cf_late_node", "string"),
    ("_cf_late_input_port", "string"),
    ("_cf_late_event_time_micros", "int64"),
    ("_cf_late_closing_time_micros", "int64"),
    ("_cf_late_watermark_micros", "int64"),
    ("_cf_late_reason", "string"),
    ("_cf_late_source", "string"),
    ("_cf_late_sequence", "uint64"),
    ("_cf_late_row_index", "uint64"),
)


def late_schema(fields: tuple[Field, ...]) -> tuple[Field, ...]:
    return fields + tuple(
        Field(name, dtype, nullable=False) for name, dtype in _DIAGNOSTICS
    )


def analyze_late_table(analyzer: _Analyzer, node: Node, path: str) -> TableFacts:
    from calc_flow.symbolic.analyzer import TableFacts, _LateRowOrigin
    from calc_flow.symbolic.lower.event_windows import _rewrite_nodes

    owner = node if node.op.name == "late_output" else node.args[0]
    value = owner.args[0]
    if not _is_single_stage(value):
        return TableFacts((), None, frozenset(), None, (), ())
    source = value.args[0]
    facts = analyzer.table(source, f"{path}.late_output.input")
    if node.op.name == "late_rows":
        analyzer.table(owner, f"{path}.late.owner")
        return TableFacts(
            late_schema(facts.schema),
            _LateRowOrigin(owner.digest),
            facts.state | {"late_diagnostic"},
            None,
            (),
            (),
        )
    boundary = _analysis_boundary(analyzer, owner, facts)
    # Named stage inputs are materialized boundaries, including computed group columns.
    analyzer._table_cache[boundary.digest] = facts
    rewritten = _rewrite_nodes(value, {source.digest: boundary._node})
    return analyzer.table(rewritten, f"{path}.late_output.value")


def _analysis_boundary(
    analyzer: _Analyzer, owner: Node, facts: TableFacts
) -> TableExpr:
    prefix = f"cf_late_stage_{owner.digest[:24]}"
    ordinal = 0
    while True:
        name = prefix if ordinal == 0 else f"{prefix}_{ordinal}"
        boundary = table_input(
            name,
            schema=facts.schema,
            entity_by=facts.entity_by,
            event_time=facts.event_time,
            sequence_by=facts.sequence_by,
        )
        if (
            boundary.digest not in analyzer._declared
            and boundary.digest not in analyzer._table_cache
        ):
            return boundary
        ordinal += 1


def _is_single_stage(value: Node) -> bool:
    from calc_flow.symbolic.lower.segments import (
        _CROSS_SECTION_PRIMITIVES,
        _ROLLING_PRIMITIVES,
    )

    if value.op.name not in {"with_columns", "filter"}:
        return False
    states: dict[str, Node] = {}

    def visit(node: Node) -> None:
        if node.op.name == "column_ref":
            return
        if node.op.name in _ROLLING_PRIMITIVES | _CROSS_SECTION_PRIMITIVES:
            states[node.digest] = node
        for child in node.args:
            visit(child)

    for expression in value.args[1:]:
        visit(expression)
    families = {node.op.name in _ROLLING_PRIMITIVES for node in states.values()}
    if len(families) != 1:
        return False
    if _has_non_column_operands(tuple(states.values()), value.args[0]):
        return False
    return _same_cross_section_group(tuple(states.values()))


def _has_non_column_operands(states: tuple[Node, ...], source: Node) -> bool:
    return any(
        operand.op.name != "column_ref" or operand.args[0].digest != source.digest
        for node in states
        for operand in node.args
    )


def _same_cross_section_group(states: tuple[Node, ...]) -> bool:
    from calc_flow.symbolic.lower.segments import _CROSS_SECTION_PRIMITIVES

    groups = {
        (node.attr("grouping"), tuple(child.digest for child in node.args[1:]))
        for node in states
        if node.op.name in _CROSS_SECTION_PRIMITIVES
    }
    return len(groups) <= 1


def check_late_successors(program: Program, analyzer: _Analyzer) -> None:
    from calc_flow.symbolic.analyzer import _ROW_LOCAL_PRIMITIVES

    allowed = _ROW_LOCAL_PRIMITIVES | {"project", "filter", "with_columns", "sql"}
    derived: dict[str, bool] = {}

    def visit(node: Node, path: str) -> bool:
        if node.digest in derived:
            return derived[node.digest]
        if node.op.name == "late_rows":
            visit(node.args[0].args[0], path)
            derived[node.digest] = True
            return True
        parents = [visit(child, path) for child in node.args]
        is_late = any(parents)
        if is_late and (
            node.op.name not in allowed
            or (node.op.name == "sql" and len(node.args) != 1)
        ):
            analyzer.issue(
                path,
                "unsupported_mode",
                f"late output cannot enter {node.op.name}; use single-input"
                " native expressions or SQL before a sink",
            )
        derived[node.digest] = is_late
        return is_late

    for name, value in program.outputs:
        visit(value._node, f"outputs.{name}")
