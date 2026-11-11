"""Immutable query facts projected from the single Rust analysis rule set."""

from __future__ import annotations

import json
from typing import Literal, get_args

from calc_flow._compat import dataclass
from calc_flow.symbolic.types import Field

SupportStatus = Literal["supported", "unsupported", "unknown"]
UpdateSemantics = Literal["legacy_append", "latest_relation"]
HistoryScope = Literal[
    "current_batch", "accepted_history", "event_window", "latest_relation", "unknown"
]
UpdateKind = Literal["append", "complete_snapshot", "unknown"]
InputInterpretation = Literal["append", "complete_replacement", "unknown"]
QueryFinality = Literal[
    "per_batch", "per_row_final", "group_final", "revisable_snapshot", "unknown"
]
Comparison = Literal["per_batch", "final_relation"]
ComparisonCutoff = Literal["batch_boundary", "all_sources_eof", "closed_window"]
ComparisonOrdering = Literal["unordered", "declared"]
QueryRowOrder = Literal["unspecified", "ordered", "unknown"]
QueryTieOrder = Literal["unspecified", "total", "unknown"]


def _literal(value: str, choices: tuple[str, ...], field: str) -> None:
    if type(value) is not str:
        raise TypeError(f"options.portability.{field}: expected a string")
    if value not in choices:
        raise ValueError(f"options.portability.{field}: invalid_literal: unknown value")


@dataclass(frozen=True, slots=True, init=False)
class PortabilityRequest:
    """Compare the same declaration over matching accepted inputs and cutoff."""

    comparison: Comparison
    history: HistoryScope
    cutoff: ComparisonCutoff
    ordering: ComparisonOrdering

    def __init__(
        self,
        *,
        comparison: Comparison,
        history: HistoryScope,
        cutoff: ComparisonCutoff,
        ordering: ComparisonOrdering = "unordered",
    ) -> None:
        _literal(comparison, ("per_batch", "final_relation"), "comparison")
        _literal(
            history,
            ("current_batch", "accepted_history", "event_window", "latest_relation"),
            "history",
        )
        _literal(
            cutoff, ("batch_boundary", "all_sources_eof", "closed_window"), "cutoff"
        )
        _literal(ordering, ("unordered", "declared"), "ordering")
        if (comparison, history, cutoff) not in (
            ("per_batch", "current_batch", "batch_boundary"),
            ("final_relation", "accepted_history", "all_sources_eof"),
            ("final_relation", "latest_relation", "all_sources_eof"),
            ("final_relation", "event_window", "closed_window"),
        ):
            raise ValueError(
                "options.portability.cutoff: invalid_literal: "
                "inconsistent comparison, history and cutoff"
            )
        for name, value in (
            ("comparison", comparison),
            ("history", history),
            ("cutoff", cutoff),
            ("ordering", ordering),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, init=False)
class AnalysisOptions:
    """Optional explicit comparison obligations; omitted means facts only."""

    portability: PortabilityRequest | None

    def __init__(self, *, portability: PortabilityRequest | None = None) -> None:
        if portability is not None and type(portability) is not PortabilityRequest:
            raise TypeError("options.portability: expected PortabilityRequest or None")
        object.__setattr__(self, "portability", portability)


@dataclass(frozen=True, slots=True)
class QuerySortKey:
    """A sort key that resolves to an output field."""

    field: str
    direction: Literal["ascending", "descending"]
    nulls: Literal["exclude", "first", "last"]


@dataclass(frozen=True, slots=True)
class QueryTemporalIdentity:
    """A declaration's temporal identity, separate from SQL result order."""

    entity_by: tuple[str, ...]
    event_time: str
    sequence_by: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_by", tuple(self.entity_by))
        object.__setattr__(self, "sequence_by", tuple(self.sequence_by))


@dataclass(frozen=True, slots=True)
class QueryOrderingFacts:
    """Proved output ordering and temporal identity."""

    kind: QueryRowOrder
    keys: tuple[QuerySortKey, ...]
    ties: QueryTieOrder
    temporal_identity: QueryTemporalIdentity | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))


@dataclass(frozen=True, slots=True)
class QueryModeFacts:
    """History, updates, finality and admission for one lifecycle."""

    support: SupportStatus
    history: HistoryScope
    update: UpdateKind
    finality: QueryFinality
    ordering: QueryOrderingFacts
    reason: str


@dataclass(frozen=True, slots=True)
class QueryNodeFacts:
    """One uniquely addressed logical declaration."""

    path: str
    schema: tuple[Field, ...] | None
    batch: QueryModeFacts
    stream: QueryModeFacts

    def __post_init__(self) -> None:
        if self.schema is not None:
            object.__setattr__(self, "schema", tuple(self.schema))


@dataclass(frozen=True, slots=True)
class QueryEdgeModeFacts:
    """Update production and consumption at an actual input edge."""

    support: SupportStatus
    producer_update: UpdateKind
    consumer_interpretation: InputInterpretation
    reason: str


@dataclass(frozen=True, slots=True)
class QueryEdgeFacts:
    """A consumption edge referencing canonical logical node addresses."""

    path: str
    source_path: str
    target_path: str
    batch: QueryEdgeModeFacts
    stream: QueryEdgeModeFacts


@dataclass(frozen=True, slots=True)
class QueryDiagnostic:
    """A stable path/code/status and explanatory message."""

    path: str
    code: str
    status: SupportStatus
    message: str


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    """Versioned facts; this report is not an execution permit."""

    schema_version: int
    analysis_revision: int
    request: PortabilityRequest | None
    portability: SupportStatus | None
    nodes: tuple[QueryNodeFacts, ...]
    edges: tuple[QueryEdgeFacts, ...]
    diagnostics: tuple[QueryDiagnostic, ...]
    update_semantics: UpdateSemantics = "legacy_append"

    def __post_init__(self) -> None:
        for name in ("nodes", "edges", "diagnostics"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


def _mode(value: dict) -> QueryModeFacts:
    _check_wire_literals(
        value,
        {
            "support": SupportStatus,
            "history": HistoryScope,
            "update": UpdateKind,
            "finality": QueryFinality,
        },
    )
    ordering = value["ordering"]
    identity = ordering["temporal_identity"]
    return QueryModeFacts(
        value["support"],
        value["history"],
        value["update"],
        value["finality"],
        QueryOrderingFacts(
            ordering["kind"],
            tuple(QuerySortKey(**key) for key in ordering["keys"]),
            ordering["ties"],
            None
            if identity is None
            else QueryTemporalIdentity(
                tuple(identity["entity_by"]),
                identity["event_time"],
                tuple(identity["sequence_by"]),
            ),
        ),
        value["reason"],
    )


def _node_from_dict(node):
    return QueryNodeFacts(
        node["path"],
        None
        if node["schema"] is None
        else tuple(Field(**field) for field in node["schema"]),
        _mode(node["batch"]),
        _mode(node["stream"]),
    )


def _wire_semantics(value: dict) -> UpdateSemantics:
    version = (value["schema_version"], value["analysis_revision"])
    if version not in ((1, 1), (2, 2), (2, 3)):
        raise ValueError("query_analysis: unsupported revision")
    semantics = (
        value.get("update_semantics", "legacy_append")
        if version == (1, 1)
        else value["update_semantics"]
    )
    if semantics not in ("legacy_append", "latest_relation") or (
        version == (1, 1) and semantics != "legacy_append"
    ):
        raise ValueError("query_analysis.update_semantics: invalid_literal")
    return semantics


def _from_dict(value: dict) -> QueryAnalysis:
    semantics = _wire_semantics(value)
    return QueryAnalysis(
        value["schema_version"],
        value["analysis_revision"],
        None if value["request"] is None else PortabilityRequest(**value["request"]),
        value["portability"],
        tuple(_node_from_dict(node) for node in value["nodes"]),
        tuple(
            QueryEdgeFacts(
                edge["path"],
                edge["source_path"],
                edge["target_path"],
                _edge_mode(edge["batch"]),
                _edge_mode(edge["stream"]),
            )
            for edge in value["edges"]
        ),
        tuple(QueryDiagnostic(**issue) for issue in value["diagnostics"]),
        semantics,
    )


def _check_wire_literals(value, fields):
    for field, vocabulary in fields.items():
        if value[field] not in get_args(vocabulary):
            raise ValueError(f"query_analysis.{field}: invalid_literal")


def _edge_mode(value):
    _check_wire_literals(
        value,
        {
            "support": SupportStatus,
            "producer_update": UpdateKind,
            "consumer_interpretation": InputInterpretation,
        },
    )
    return QueryEdgeModeFacts(**value)


def _request_dict(request: PortabilityRequest | None) -> dict | None:
    return (
        None
        if request is None
        else {
            field: getattr(request, field)
            for field in ("comparison", "history", "cutoff", "ordering")
        }
    )


def _segment(root: str, name: str) -> str:
    return (
        f"{root}.{name}"
        if name.isidentifier()
        else f"{root}[{json.dumps(name, ensure_ascii=False)}]"
    )


_DIRECT_ROLES = {
    **dict.fromkeys(
        (
            "add",
            "sub",
            "mul",
            "truediv",
            "power",
            "gt",
            "ge",
            "lt",
            "le",
            "eq",
            "ne",
            "and",
            "or",
            "stream_join",
            "stream_asof_join",
            "covariance",
            "correlation",
        ),
        ("left", "right"),
    ),
    **dict.fromkeys(
        (
            "column_ref",
            "neg",
            "not",
            "cast",
            "abs",
            "sqrt",
            "log",
            "exp",
            "isnan",
            "is_null",
            "clip",
            "window_tumbling",
            "window_hopping",
            "project",
        ),
        ("value",),
    ),
    "filter": ("value", "predicate"),
    "attach_columns": ("value", "array"),
    "where": ("predicate", "when_true", "when_false"),
}


def _named_roles(node):
    from calc_flow.symbolic.analyzer import _cstr_seq

    name = node.op.name
    if name == "aggregate":
        return (".aggregate.value",)
    if name in ("sql", "snapshot_sql"):
        return tuple(
            _segment(f".{name}.tables", alias)
            for alias in _cstr_seq(node.attr("aliases"))
        )
    if name == "with_columns":
        return (
            f".{name}.value",
            *(_segment("", field) for field in _cstr_seq(node.attr("names"))),
        )
    if name == "coalesce":
        return tuple(f".coalesce.values[{index}]" for index in range(len(node.args)))
    return tuple(f".{name}.{role}" for role in _DIRECT_ROLES.get(name, ()))


def _roles(node):
    named = _named_roles(node)
    return tuple(
        named[index] if index < len(named) else f".{node.op.name}.args[{index}]"
        for index in range(len(node.args))
    )


def _legacy_role(role):
    if role.startswith('["') and role.endswith("]"):
        return "." + json.loads(role[1:-1])
    return role


def _declaration_paths(program):
    from calc_flow.symbolic.analyzer import _cstr_or_none

    paths, nodes, legacy_paths = {}, {}, {}

    def visit(node, path, legacy):
        key = node.digest
        if node.op.name in ("table_input", "parameter"):
            root = "static_inputs" if node.op.name == "parameter" else "inputs"
            name = _cstr_or_none(node.attr("name"))
            path, legacy = _segment(root, name), f"{root}.{name}"
        legacy_paths[legacy] = key
        if key in paths and paths[key].encode() <= path.encode():
            return
        paths[key], nodes[key] = path, node
        for child, role in zip(node.args, _roles(node)):
            visit(child, path + role, legacy + _legacy_role(role))

    for name, value in sorted(program.outputs, key=lambda item: item[0].encode()):
        visit(value._node, _segment("outputs", name), f"outputs.{name}")
    return paths, nodes, legacy_paths


def _contains_issue(path, owner):
    return path == owner or path.startswith((owner + ".", owner + "["))


def _issue_diagnostic(issue, mode, paths, legacy_paths):
    owner = max(
        (path for path in legacy_paths if _contains_issue(issue.path, path)),
        key=len,
        default=None,
    )
    status = "unknown" if issue.code == "capability_mismatch" else "unsupported"
    key = None
    if owner is None:
        path = f"{issue.path}.{mode}.support"
    else:
        key = legacy_paths[owner]
        path = f"{paths[key]}.{mode}{issue.path[len(owner) :] or '.support'}"
    return key, dict(
        path=path,
        code=issue.code,
        status=status,
        message=f"declaration analysis reported {issue.code}; "
        "see existing analysis issues for field/type details",
    )


def _query_diagnostics(analyzers, paths, legacy_paths):
    diagnostics, statuses = [], {}
    for analyzer in analyzers:
        for issue in analyzer.issues:
            key, diagnostic = _issue_diagnostic(
                issue, analyzer._mode, paths, legacy_paths
            )
            if key is not None:
                owner = (key, analyzer._mode)
                status = diagnostic["status"]
                previous = statuses.get(owner)
                statuses[owner] = (
                    "unsupported" if "unsupported" in (status, previous) else status
                )
            diagnostics.append(diagnostic)
    return diagnostics, statuses


def _table_facts(key, analyzers, statuses):
    return next(
        (
            analyzer._table_cache[key]
            for analyzer in analyzers
            if key in analyzer._table_cache
            and statuses.get((key, analyzer._mode)) != "unsupported"
        ),
        None,
    )


def _query_schema(facts):
    return (
        None
        if facts is None
        else [
            dict(name=f.name, data_type=f.data_type, nullable=f.nullable)
            for f in facts.schema
        ]
    )


def _query_identity(facts):
    if facts is None or facts.event_time is None:
        return None
    return dict(
        entity_by=list(facts.entity_by),
        event_time=facts.event_time,
        sequence_by=list(facts.sequence_by),
    )


def _declaration(node, path, facts, statuses):
    from calc_flow.symbolic.nodes import CStr

    query = node.attr("query")
    return dict(
        path=path,
        kind=node.op.name,
        query=query.value if isinstance(query, CStr) else None,
        schema=_query_schema(facts),
        ordering=dict(
            kind="unspecified",
            keys=[],
            ties="unspecified",
            temporal_identity=_query_identity(facts),
        ),
        batch_support=statuses.get((node.digest, "batch")),
        stream_support=statuses.get((node.digest, "stream")),
    )


def _input_alias(node, index):
    from calc_flow.symbolic.analyzer import _cstr_seq

    if node.op.name not in ("sql", "snapshot_sql", "aggregate"):
        return None
    aliases = _cstr_seq(node.attr("aliases"))
    return aliases[index] if index < len(aliases) else None


def _query_edges(nodes, paths):
    return [
        dict(
            path=paths[key] + role,
            source_path=paths[child.digest],
            target_path=paths[key],
            input_alias=_input_alias(node, index),
        )
        for key, node in nodes.items()
        for index, (child, role) in enumerate(zip(node.args, _roles(node)))
    ]


def _query(program, runtime, options, analyzers) -> QueryAnalysis:
    paths, nodes, legacy_paths = _declaration_paths(program)
    diagnostics, statuses = _query_diagnostics(analyzers, paths, legacy_paths)
    graph = dict(
        nodes=[
            _declaration(
                node, paths[key], _table_facts(key, analyzers, statuses), statuses
            )
            for key, node in nodes.items()
        ],
        edges=_query_edges(nodes, paths),
        options=dict(portability=_request_dict(options.portability)),
        diagnostics=diagnostics,
        update_semantics=program.update_semantics,
    )
    return _from_dict(
        json.loads(runtime._inner._analyze_query_graph(json.dumps(graph)))
    )


def _gate(program, runtime, request):
    if program.update_semantics == "latest_relation":
        _gate_latest(program, runtime)
    if request is None:
        return
    report = program.analyze(
        runtime, options=AnalysisOptions(portability=request)
    ).query
    if report.portability != "supported":
        issue = next(
            issue for issue in report.diagnostics if issue.status != "supported"
        )
        _raise_query_issue(report, issue)


def _gate_latest(program, runtime):
    from calc_flow.errors import CompileError

    report = program.analyze(runtime).query
    if (report.schema_version, report.analysis_revision, report.update_semantics) != (
        2,
        3,
        "latest_relation",
    ):
        error = CompileError(
            "Program.update_semantics: update_semantics_unproved: "
            "latest_relation requires a matching v2 analysis"
        )
        error.query_analysis = report
        raise error
    issue = _update_failure(report, program.mode)
    if issue is not None:
        _raise_query_issue(report, issue)


def _raise_query_issue(report, issue):
    from calc_flow.errors import CompileError

    error = CompileError(f"{issue.path}: {issue.code}: {issue.message}")
    error.query_analysis = report
    raise error


def _current_mode_issue(report, issue, mode):
    paths = tuple(node.path for node in report.nodes) + tuple(
        edge.path for edge in report.edges
    )
    return any(issue.path.startswith(f"{path}.{mode}.") for path in paths) or (
        mode == "stream" and any(issue.path == edge.path for edge in report.edges)
    )


def _update_failure(report, mode):
    return next(
        (
            issue
            for issue in report.diagnostics
            if issue.status != "supported" and _current_mode_issue(report, issue, mode)
        ),
        None,
    )
