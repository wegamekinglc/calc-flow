from __future__ import annotations

from typing import TYPE_CHECKING

from calc_flow.capabilities import RuntimeCapabilities
from calc_flow.symbolic.nodes import CEnum, CInt, CSeq, Node
from calc_flow.symbolic.query_analysis import QueryAnalysis
from calc_flow.symbolic.types import CompileMode

if TYPE_CHECKING:
    from calc_flow.symbolic.analyzer import ArrayFacts, TableFacts, _Analyzer


def _explain_query(query: QueryAnalysis) -> list[str]:
    lines = []
    lines.append(
        f"  query schema_version {query.schema_version}"
        f" analysis_revision {query.analysis_revision}"
    )
    lines.append(
        "    comparison assumes the same accepted inputs and legal order; "
        "batch receives H or the same batch; actual EOF/closed-window cutoff; "
        "execution budgets and overflow still apply"
    )
    lines.append(f"    portability {query.portability}")
    for node in query.nodes:
        for name in ("batch", "stream"):
            facts = getattr(node, name)
            lines.append(
                f"    {node.path}.{name}: {facts.support} {facts.history}"
                f" {facts.update} {facts.finality}"
            )
    for edge in query.edges:
        lines.append(
            f"    {edge.path}: {edge.source_path} -> {edge.target_path}; "
            f"stream {edge.stream.producer_update}/"
            f"{edge.stream.consumer_interpretation}"
        )
    lines.extend(
        f"    {issue.path}: {issue.code}: {issue.status}: {issue.message}"
        for issue in query.diagnostics
    )
    return lines


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
    from calc_flow.symbolic.analyzer import _cstr_or_none, _cstr_seq, _ctype_str

    node = value._node
    name = _cstr_or_none(node.attr("name")) or ""
    if node.op.name != "parameter":
        return (
            f"    input {name} event_time"
            f" {_cstr_or_none(node.attr('event_time')) or 'none'} entity_by"
            f" {_render_names(_cstr_seq(node.attr('entity_by')))} sequence_by"
            f" {_render_names(_cstr_seq(node.attr('sequence_by')))}"
        )
    kind = node.attr("kind")
    if isinstance(kind, CEnum) and kind.variant == "array":
        return (
            f"    static_input {name} array backend"
            f" {_cstr_or_none(node.attr('backend'))} dtype"
            f" {_ctype_str(node.attr('dtype'))} shape"
            f" {_render_shape(_parameter_shape(node))}"
        )
    from calc_flow.symbolic.analyzer import _schema_fields

    return (
        f"    static_input {name} table fields"
        f" {len(_schema_fields(node.attr('schema')))}"
    )


def _explain_table_output(output_name: str, facts: TableFacts, /) -> list[str]:
    from calc_flow.symbolic.analyzer import _WindowRowOrigin

    lines = [f"    output {output_name} table"]
    if isinstance(facts.lineage, _WindowRowOrigin):
        lines.append(f"      lineage {facts.lineage}")
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
