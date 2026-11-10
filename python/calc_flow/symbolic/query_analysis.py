"""Immutable query facts projected from the single Rust analysis rule set."""

from __future__ import annotations

import json
from typing import Literal

from calc_flow._compat import dataclass
from calc_flow.symbolic.types import Field

SupportStatus = Literal["supported", "unsupported", "unknown"]
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

    def __post_init__(self) -> None:
        for name in ("nodes", "edges", "diagnostics"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


def _mode(value: dict) -> QueryModeFacts:
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


def _from_dict(value: dict) -> QueryAnalysis:
    if value["schema_version"] != 1 or value["analysis_revision"] != 1:
        raise ValueError("query_analysis: unsupported revision")
    return QueryAnalysis(
        value["schema_version"],
        value["analysis_revision"],
        None if value["request"] is None else PortabilityRequest(**value["request"]),
        value["portability"],
        tuple(
            QueryNodeFacts(
                node["path"],
                None
                if node["schema"] is None
                else tuple(Field(**field) for field in node["schema"]),
                _mode(node["batch"]),
                _mode(node["stream"]),
            )
            for node in value["nodes"]
        ),
        tuple(
            QueryEdgeFacts(
                edge["path"],
                edge["source_path"],
                edge["target_path"],
                QueryEdgeModeFacts(**edge["batch"]),
                QueryEdgeModeFacts(**edge["stream"]),
            )
            for edge in value["edges"]
        ),
        tuple(QueryDiagnostic(**issue) for issue in value["diagnostics"]),
    )


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


def _roles(node):
    from calc_flow.symbolic.analyzer import _cstr_seq

    name = node.op.name
    roles = [f".{name}.args[{index}]" for index in range(len(node.args))]
    if name in ("sql", "snapshot_sql"):
        for index, alias in enumerate(_cstr_seq(node.attr("aliases"))[: len(roles)]):
            roles[index] = _segment(f".{name}.tables", alias)
    elif name in ("project", "filter", "with_columns", "attach_columns"):
        if roles:
            roles[0] = f".{name}.value"
        if name == "filter" and len(roles) > 1:
            roles[1] = ".filter.predicate"
        elif name == "attach_columns" and len(roles) > 1:
            roles[1] = ".attach_columns.array"
        elif name == "with_columns":
            for index, field in enumerate(
                _cstr_seq(node.attr("names"))[: len(roles) - 1], 1
            ):
                roles[index] = _segment("", field)
    elif name in (
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
    ):
        for index, role in enumerate(("left", "right")[: len(roles)]):
            roles[index] = f".{name}.{role}"
    elif name == "coalesce":
        roles = [f".coalesce.values[{index}]" for index in range(len(roles))]
    elif name == "where":
        for index, role in enumerate(
            ("predicate", "when_true", "when_false")[: len(roles)]
        ):
            roles[index] = f".where.{role}"
    elif (
        name
        in (
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
        )
        and roles
    ):
        roles[0] = f".{name}.value"
    return tuple(roles)


def _legacy_role(role):
    if role.startswith('["') and role.endswith("]"):
        return "." + json.loads(role[1:-1])
    return role


def _query(program, runtime, options, analyzers) -> QueryAnalysis:
    from calc_flow.symbolic.analyzer import _cstr_or_none
    from calc_flow.symbolic.nodes import CStr

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
    diagnostics, statuses = [], {}
    for analyzer in analyzers:
        for issue in analyzer.issues:
            candidates = [
                path
                for path in legacy_paths
                if issue.path == path
                or issue.path.startswith(path + ".")
                or issue.path.startswith(path + "[")
            ]
            owner = max(candidates, key=len) if candidates else None
            status = "unknown" if issue.code == "capability_mismatch" else "unsupported"
            mode = analyzer._mode
            if owner is None:
                path = f"{issue.path}.{mode}.support"
            else:
                key = legacy_paths[owner]
                path = f"{paths[key]}.{mode}{issue.path[len(owner) :] or '.support'}"
                previous = statuses.get((key, mode))
                statuses[(key, mode)] = (
                    "unsupported" if "unsupported" in (status, previous) else status
                )
            diagnostics.append(
                dict(
                    path=path,
                    code=issue.code,
                    status=status,
                    message=f"declaration analysis reported {issue.code}; "
                    "see existing analysis issues for field/type details",
                )
            )
    declarations, edges = [], []
    for key, node in nodes.items():
        facts = next(
            (
                analyzer._table_cache[key]
                for analyzer in analyzers
                if key in analyzer._table_cache
                and statuses.get((key, analyzer._mode)) != "unsupported"
            ),
            None,
        )
        schema = (
            None
            if facts is None
            else [
                dict(name=f.name, data_type=f.data_type, nullable=f.nullable)
                for f in facts.schema
            ]
        )
        identity = (
            None
            if facts is None or facts.event_time is None
            else dict(
                entity_by=list(facts.entity_by),
                event_time=facts.event_time,
                sequence_by=list(facts.sequence_by),
            )
        )
        query = node.attr("query")
        declarations.append(
            dict(
                path=paths[key],
                kind=node.op.name,
                query=query.value if isinstance(query, CStr) else None,
                schema=schema,
                ordering=dict(
                    kind="unspecified",
                    keys=[],
                    ties="unspecified",
                    temporal_identity=identity,
                ),
                batch_support=statuses.get((key, "batch")),
                stream_support=statuses.get((key, "stream")),
            )
        )
        for child, role in zip(node.args, _roles(node)):
            edges.append(
                dict(
                    path=paths[key] + role,
                    source_path=paths[child.digest],
                    target_path=paths[key],
                )
            )
    graph = dict(
        nodes=declarations,
        edges=edges,
        options=dict(portability=_request_dict(options.portability)),
        diagnostics=diagnostics,
    )
    return _from_dict(
        json.loads(runtime._inner._analyze_query_graph(json.dumps(graph)))
    )


def _gate(program, runtime, request):
    if request is None:
        return
    from calc_flow.errors import CompileError

    report = program.analyze(
        runtime, options=AnalysisOptions(portability=request)
    ).query
    if report.portability != "supported":
        issue = next(
            issue for issue in report.diagnostics if issue.status != "supported"
        )
        error = CompileError(f"{issue.path}: {issue.code}: {issue.message}")
        error.query_analysis = report
        raise error
