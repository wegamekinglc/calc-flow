//! Data-only history facts and explicit cross-lifecycle comparison checks.

mod sql;
mod updates;

use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::datatypes::{DataType, Schema, SchemaRef, TimeUnit};
use serde::{Deserialize, Serialize};

use crate::config::canonical_arrow_field_type;
use crate::{ArrowFieldSpec, CalcFlowError, NullPlacement, Result, SortDirection};

macro_rules! vocabulary {
    ($name:ident { $($variant:ident),+ $(,)? }) => {
        #[doc = "A versioned query-analysis vocabulary; serialized as `snake_case`."]
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        #[non_exhaustive]
        pub enum $name { $($variant),+ }
    };
}
vocabulary!(SupportStatus {
    Supported,
    Unsupported,
    Unknown
});
vocabulary!(HistoryScope {
    CurrentBatch,
    AcceptedHistory,
    EventWindow,
    LatestRelation,
    Unknown
});
vocabulary!(UpdateKind {
    Append,
    CompleteSnapshot,
    Unknown
});
vocabulary!(InputInterpretation {
    Append,
    CompleteReplacement,
    Unknown
});
vocabulary!(QueryFinality {
    PerBatch,
    PerRowFinal,
    GroupFinal,
    RevisableSnapshot,
    Unknown
});
vocabulary!(Comparison {
    PerBatch,
    FinalRelation
});
vocabulary!(ComparisonCutoff {
    BatchBoundary,
    AllSourcesEof,
    ClosedWindow
});
vocabulary!(ComparisonOrdering {
    Unordered,
    Declared
});
vocabulary!(QueryRowOrder {
    Unspecified,
    Ordered,
    Unknown
});
vocabulary!(QueryTieOrder {
    Unspecified,
    Total,
    Unknown
});

/// A comparison contract. It changes neither the declaration nor execution state.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "RequestFields")]
pub struct PortabilityRequest {
    comparison: Comparison,
    history: HistoryScope,
    cutoff: ComparisonCutoff,
    ordering: ComparisonOrdering,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestFields {
    comparison: Comparison,
    history: HistoryScope,
    cutoff: ComparisonCutoff,
    ordering: ComparisonOrdering,
}

impl TryFrom<RequestFields> for PortabilityRequest {
    type Error = CalcFlowError;
    fn try_from(fields: RequestFields) -> Result<Self> {
        Ok(Self::new(fields.comparison, fields.history, fields.cutoff)?
            .with_ordering(fields.ordering))
    }
}

impl PortabilityRequest {
    /// Constructs a valid comparison/history/cutoff combination.
    ///
    /// # Errors
    /// Returns an invalid-literal error for inconsistent comparison settings.
    pub fn new(
        comparison: Comparison,
        history: HistoryScope,
        cutoff: ComparisonCutoff,
    ) -> Result<Self> {
        if !matches!(
            (comparison, history, cutoff),
            (
                Comparison::PerBatch,
                HistoryScope::CurrentBatch,
                ComparisonCutoff::BatchBoundary
            ) | (
                Comparison::FinalRelation,
                HistoryScope::AcceptedHistory | HistoryScope::LatestRelation,
                ComparisonCutoff::AllSourcesEof
            ) | (
                Comparison::FinalRelation,
                HistoryScope::EventWindow,
                ComparisonCutoff::ClosedWindow
            )
        ) {
            return Err(CalcFlowError::InvalidArgument {
                field: "options.portability.cutoff".into(),
                message: "invalid_literal: inconsistent comparison, history and cutoff".into(),
            });
        }
        Ok(Self {
            comparison,
            history,
            cutoff,
            ordering: ComparisonOrdering::Unordered,
        })
    }
    /// Selects whether result row order is part of the comparison.
    #[must_use]
    pub const fn with_ordering(mut self, ordering: ComparisonOrdering) -> Self {
        self.ordering = ordering;
        self
    }
    pub const fn comparison(&self) -> Comparison {
        self.comparison
    }
    pub const fn history(&self) -> HistoryScope {
        self.history
    }
    pub const fn cutoff(&self) -> ComparisonCutoff {
        self.cutoff
    }
    pub const fn ordering(&self) -> ComparisonOrdering {
        self.ordering
    }
}

/// Optional comparison obligations for one analysis.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisOptions {
    pub portability: Option<PortabilityRequest>,
}

/// Deterministic report. It is diagnostic data, never an execution permit.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct QueryAnalysis {
    pub schema_version: u32,
    pub analysis_revision: u64,
    pub update_semantics: crate::UpdateSemantics,
    pub request: Option<PortabilityRequest>,
    pub portability: Option<SupportStatus>,
    pub nodes: Vec<QueryNodeFacts>,
    pub edges: Vec<QueryEdgeFacts>,
    pub diagnostics: Vec<QueryDiagnostic>,
}

/// Facts about one declaration node in both lifecycles.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct QueryNodeFacts {
    pub path: String,
    pub schema: Option<SchemaRef>,
    pub batch: QueryModeFacts,
    pub stream: QueryModeFacts,
}

/// Local lifecycle, history, update, finality and row-order facts.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct QueryModeFacts {
    pub support: SupportStatus,
    pub history: HistoryScope,
    pub update: UpdateKind,
    pub finality: QueryFinality,
    pub ordering: QueryOrderingFacts,
    pub reason: String,
}

/// The producer and consumer interpretation at one actual consumption edge.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct QueryEdgeFacts {
    pub path: String,
    pub source_path: String,
    pub target_path: String,
    pub batch: QueryEdgeModeFacts,
    pub stream: QueryEdgeModeFacts,
}

/// Local update interpretation at an input port.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct QueryEdgeModeFacts {
    pub support: SupportStatus,
    pub producer_update: UpdateKind,
    pub consumer_interpretation: InputInterpretation,
    pub reason: String,
}

/// Output row ordering, separate from temporal identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct QueryOrderingFacts {
    pub kind: QueryRowOrder,
    pub keys: Vec<QuerySortKey>,
    pub ties: QueryTieOrder,
    pub temporal_identity: Option<QueryTemporalIdentity>,
}

impl Default for QueryOrderingFacts {
    fn default() -> Self {
        Self {
            kind: QueryRowOrder::Unspecified,
            keys: vec![],
            ties: QueryTieOrder::Unspecified,
            temporal_identity: None,
        }
    }
}

/// A sort key referring to an output field.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct QuerySortKey {
    pub field: String,
    pub direction: SortDirection,
    pub nulls: NullPlacement,
}

/// A declaration's temporal proof; SQL output does not inherit this identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct QueryTemporalIdentity {
    pub entity_by: Vec<String>,
    pub event_time: String,
    pub sequence_by: Vec<String>,
}

/// A machine-readable finding. Message text is explanatory, not an identifier.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct QueryDiagnostic {
    pub path: String,
    pub code: String,
    pub status: SupportStatus,
    pub message: String,
}

impl std::fmt::Display for QueryAnalysis {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(issue) = self
            .diagnostics
            .iter()
            .find(|issue| issue.status != SupportStatus::Supported)
        {
            write!(
                formatter,
                "{}: {}: {}",
                issue.path, issue.code, issue.message
            )
        } else {
            write!(formatter, "query portability could not be proved")
        }
    }
}

impl QueryAnalysis {
    pub(crate) fn check_updates(self, mode: crate::CompileMode) -> Result<()> {
        if !self.update_semantics.is_legacy() && self.update_failure(mode).is_some() {
            Err(CalcFlowError::QueryUpdateSemantics {
                mode,
                report: Box::new(self),
            })
        } else {
            Ok(())
        }
    }

    pub(crate) fn update_failure_message(&self, mode: crate::CompileMode) -> String {
        self.update_failure(mode).map_or_else(
            || "latest_relation update semantics could not be proved".into(),
            |issue| format!("{}: {}: {}", issue.path, issue.code, issue.message),
        )
    }

    fn update_failure(&self, mode: crate::CompileMode) -> Option<&QueryDiagnostic> {
        self.diagnostics.iter().find(|issue| {
            issue.status != SupportStatus::Supported
                && (self.nodes.iter().any(|node| {
                    issue
                        .path
                        .starts_with(&format!("{}.{}.", node.path, mode.name()))
                }) || self.edges.iter().any(|edge| {
                    issue
                        .path
                        .starts_with(&format!("{}.{}.", edge.path, mode.name()))
                        || mode == crate::CompileMode::Stream && issue.path == edge.path
                }))
        })
    }

    pub(crate) fn check(self) -> Result<()> {
        if self.portability == Some(SupportStatus::Supported) {
            Ok(())
        } else {
            Err(CalcFlowError::QueryPortability {
                report: Box::new(self),
            })
        }
    }
}

pub(crate) fn schema_fields(schema: &SchemaRef) -> Vec<ArrowFieldSpec> {
    schema
        .fields()
        .iter()
        .map(|field| ArrowFieldSpec {
            name: field.name().clone(),
            data_type: canonical_arrow_field_type(field)
                .unwrap_or_else(|| type_name(field.data_type())),
            nullable: field.is_nullable(),
        })
        .collect()
}

fn type_name(dtype: &DataType) -> String {
    match dtype {
        DataType::Timestamp(unit, zone) => {
            let unit = match unit {
                TimeUnit::Second => "s",
                TimeUnit::Millisecond => "ms",
                TimeUnit::Microsecond => "us",
                TimeUnit::Nanosecond => "ns",
            };
            zone.as_ref().map_or_else(
                || format!("timestamp[{unit}]"),
                |zone| format!("timestamp[{unit}, {zone}]"),
            )
        }
        _ => dtype.to_string().to_ascii_lowercase(),
    }
}

impl Serialize for QueryNodeFacts {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut fields = serializer.serialize_struct("QueryNodeFacts", 4)?;
        fields.serialize_field("path", &self.path)?;
        fields.serialize_field("schema", &self.schema.as_ref().map(schema_fields))?;
        fields.serialize_field("batch", &self.batch)?;
        fields.serialize_field("stream", &self.stream)?;
        fields.end()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Declaration {
    pub path: String,
    pub kind: String,
    pub query: Option<String>,
    pub schema: Option<Vec<ArrowFieldSpec>>,
    #[serde(default)]
    pub ordering: QueryOrderingFacts,
    #[serde(default)]
    pub batch_support: Option<SupportStatus>,
    #[serde(default)]
    pub stream_support: Option<SupportStatus>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeclarationEdge {
    pub path: String,
    pub source_path: String,
    pub target_path: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeclarationGraph {
    pub nodes: Vec<Declaration>,
    pub edges: Vec<DeclarationEdge>,
    pub options: AnalysisOptions,
    #[serde(default)]
    pub diagnostics: Vec<QueryDiagnostic>,
    #[serde(default)]
    pub update_semantics: crate::UpdateSemantics,
}

/// Private data-only binding seam. Accepts no data, factories or runtime objects.
///
/// # Errors
/// Returns an error for malformed declaration data or inconsistent options.
#[doc(hidden)]
pub fn analyze_query_graph_json(declaration: &str) -> Result<String> {
    let graph: DeclarationGraph =
        serde_json::from_str(declaration).map_err(|error| CalcFlowError::InvalidArgument {
            field: "query_analysis".into(),
            message: error.to_string(),
        })?;
    let report = analyze_graph(graph)?;
    crate::canonical_json(&serde_json::to_value(report).map_err(|error| {
        CalcFlowError::Internal {
            message: error.to_string(),
        }
    })?)
}

pub(crate) fn analyze_graph(graph: DeclarationGraph) -> Result<QueryAnalysis> {
    let DeclarationGraph {
        nodes: declarations,
        edges: declared_edges,
        options,
        mut diagnostics,
        update_semantics,
    } = graph;
    let AnalyzedNodes { by_path, proofs } = analyze_nodes(
        declarations,
        &declared_edges,
        options.portability.as_ref(),
        &mut diagnostics,
        update_semantics,
    )?;
    let mut edges = analyze_edges(
        &by_path,
        &proofs,
        &declared_edges,
        options.portability.as_ref(),
        &mut diagnostics,
        update_semantics,
    );
    check_ordering(
        &by_path,
        &edges,
        options.portability.as_ref(),
        &mut diagnostics,
    );
    edges.sort_by(|a, b| a.path.cmp(&b.path));
    diagnostics.sort_by(|a, b| {
        (&a.path, &a.code, status_name(a.status)).cmp(&(&b.path, &b.code, status_name(b.status)))
    });
    diagnostics.dedup_by(|a, b| a.path == b.path && a.code == b.code && a.status == b.status);
    let portability = options
        .portability
        .as_ref()
        .map(|_| conjunction(diagnostics.iter().map(|issue| issue.status)));
    Ok(QueryAnalysis {
        schema_version: 2,
        analysis_revision: 3,
        update_semantics,
        request: options.portability,
        portability,
        nodes: by_path.into_values().collect(),
        edges,
        diagnostics,
    })
}

type NodeFacts = BTreeMap<String, QueryNodeFacts>;
type NodeProofs = BTreeMap<String, (sql::Proof, bool, String)>;
struct AnalyzedNodes {
    by_path: NodeFacts,
    proofs: NodeProofs,
}

fn analyze_nodes(
    declarations: Vec<Declaration>,
    declared_edges: &[DeclarationEdge],
    request: Option<&PortabilityRequest>,
    diagnostics: &mut Vec<QueryDiagnostic>,
    update_semantics: crate::UpdateSemantics,
) -> Result<AnalyzedNodes> {
    let mut pending = declarations
        .into_iter()
        .map(|node| (node.path.clone(), node))
        .collect::<BTreeMap<_, _>>();
    let mut by_path = BTreeMap::<String, QueryNodeFacts>::new();
    let mut proofs = BTreeMap::new();
    while !pending.is_empty() {
        let ready = pending
            .keys()
            .find(|path| {
                declared_edges
                    .iter()
                    .filter(|edge| &edge.target_path == *path)
                    .all(|edge| by_path.contains_key(&edge.source_path))
            })
            .cloned()
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: "query_analysis.edges".into(),
                message: "unresolved input or cycle".into(),
            })?;
        let node = pending.remove(&ready).expect("ready declaration exists");
        let (mut facts, mut proof, aggregate) = node_facts(&node, diagnostics, update_semantics)?;
        let inputs = declared_edges
            .iter()
            .filter(|e| e.target_path == node.path)
            .collect::<Vec<_>>();
        if !update_semantics.is_legacy() && aggregate && inputs.len() == 1 {
            let source = &by_path[&inputs[0].source_path];
            if node.schema.is_some()
                && source.schema.is_some()
                && node.query.as_ref().is_some_and(|q| {
                    sql::replacement(
                        q,
                        &schema_fields(source.schema.as_ref().expect("proved schema")),
                    )
                })
            {
                proof = sql::Proof::Replacement;
                if source.stream.update == UpdateKind::CompleteSnapshot {
                    facts.stream.history = HistoryScope::LatestRelation;
                    facts.stream.reason = "single alias replacement aggregate; runtime requires sql_replacement_state_budget".into();
                }
            }
        }
        propagate_inputs(
            &node,
            &mut facts,
            proof,
            declared_edges,
            &by_path,
            diagnostics,
        );
        if !update_semantics.is_legacy() {
            updates::node(&node, &mut facts, proof, aggregate, diagnostics);
        }
        if let Some(request) = request {
            obligations(&node, &facts, proof, aggregate, request, diagnostics);
        }
        proofs.insert(node.path.clone(), (proof, aggregate, node.kind));
        by_path.insert(node.path, facts);
    }
    Ok(AnalyzedNodes { by_path, proofs })
}

fn has_unsupported_diagnostic(path: &str, mode: &str, diagnostics: &[QueryDiagnostic]) -> bool {
    diagnostics.iter().any(|issue| {
        issue.status == SupportStatus::Unsupported
            && issue.path.starts_with(&format!("{path}.{mode}."))
    })
}

fn propagate_mode(local: &mut QueryModeFacts, proof: sql::Proof, inputs: &[&QueryModeFacts]) {
    if proof == sql::Proof::RowLocal || proof.per_batch() {
        if inputs
            .iter()
            .any(|facts| facts.update == UpdateKind::Unknown)
        {
            local.update = UpdateKind::Unknown;
            local.finality = QueryFinality::Unknown;
            local.reason = "upstream update/finality semantics are unproved".into();
        } else if inputs
            .iter()
            .any(|facts| facts.update == UpdateKind::CompleteSnapshot)
        {
            local.update = UpdateKind::CompleteSnapshot;
            local.finality = QueryFinality::RevisableSnapshot;
            local.reason = "batch transform of complete snapshot events".into();
        }
    }
    local.support =
        conjunction(std::iter::once(local.support).chain(inputs.iter().map(|facts| facts.support)));
}

fn propagate_inputs(
    node: &Declaration,
    facts: &mut QueryNodeFacts,
    proof: sql::Proof,
    declared_edges: &[DeclarationEdge],
    by_path: &NodeFacts,
    diagnostics: &[QueryDiagnostic],
) {
    for mode in ["batch", "stream"] {
        let local = if mode == "batch" {
            &mut facts.batch
        } else {
            &mut facts.stream
        };
        if has_unsupported_diagnostic(&node.path, mode, diagnostics) {
            local.support = SupportStatus::Unsupported;
        }
        let inputs = declared_edges
            .iter()
            .filter(|edge| edge.target_path == node.path)
            .map(|edge| {
                if mode == "batch" {
                    &by_path[&edge.source_path].batch
                } else {
                    &by_path[&edge.source_path].stream
                }
            })
            .collect::<Vec<_>>();
        propagate_mode(local, proof, &inputs);
    }
}

fn analyze_edges(
    by_path: &NodeFacts,
    proofs: &NodeProofs,
    declared_edges: &[DeclarationEdge],
    request: Option<&PortabilityRequest>,
    diagnostics: &mut Vec<QueryDiagnostic>,
    update_semantics: crate::UpdateSemantics,
) -> Vec<QueryEdgeFacts> {
    let mut edges = vec![];
    for edge in declared_edges {
        let edge = DeclarationEdge {
            path: edge.path.clone(),
            source_path: edge.source_path.clone(),
            target_path: edge.target_path.clone(),
        };
        let source = &by_path[&edge.source_path];
        let target = &by_path[&edge.target_path];
        let target_kind = &proofs[&edge.target_path].2;
        let consumption = if target_kind == "snapshot_sql"
            || (!update_semantics.is_legacy()
                && proofs[&edge.target_path].0 == sql::Proof::Replacement
                && source.stream.update == UpdateKind::CompleteSnapshot)
        {
            InputInterpretation::CompleteReplacement
        } else {
            InputInterpretation::Append
        };
        let edge_mode = |producer: &QueryModeFacts, consumer: &QueryModeFacts| QueryEdgeModeFacts {
            support: conjunction([producer.support, consumer.support]),
            producer_update: producer.update,
            consumer_interpretation: consumption,
            reason: if consumption == InputInterpretation::CompleteReplacement {
                "each input event replaces its alias"
            } else {
                "input events are appended"
            }
            .into(),
        };
        let batch = edge_mode(&source.batch, &target.batch);
        let mut stream = edge_mode(&source.stream, &target.stream);
        if !update_semantics.is_legacy() {
            updates::edge(
                &edge,
                &mut stream,
                &proofs[&edge.target_path],
                &target.stream.reason,
                diagnostics,
            );
        }
        if proofs[&edge.source_path].2 == "snapshot_sql" {
            stream.support = SupportStatus::Unsupported;
            diagnostic(
                diagnostics,
                &format!("{}.stream.support", edge.path),
                "unsupported_mode",
                SupportStatus::Unsupported,
                "snapshot_sql outputs must connect directly to sinks",
            );
        }
        if snapshot_append_mismatch(request, &stream, consumption, proofs[&edge.target_path].1) {
            stream.support = SupportStatus::Unsupported;
            diagnostic(
                diagnostics,
                &edge.path,
                "update_interpretation_mismatch",
                SupportStatus::Unsupported,
                "final_relation requires latest complete relations at this input; producer emits complete_snapshot and consumer appends",
            );
        }
        edges.push(QueryEdgeFacts {
            path: edge.path,
            source_path: edge.source_path,
            target_path: edge.target_path,
            batch,
            stream,
        });
    }
    edges
}

fn snapshot_append_mismatch(
    request: Option<&PortabilityRequest>,
    stream: &QueryEdgeModeFacts,
    consumption: InputInterpretation,
    aggregate: bool,
) -> bool {
    request.is_some_and(|request| request.comparison == Comparison::FinalRelation)
        && stream.producer_update == UpdateKind::CompleteSnapshot
        && consumption == InputInterpretation::Append
        && aggregate
}

fn check_ordering(
    by_path: &NodeFacts,
    edges: &[QueryEdgeFacts],
    request: Option<&PortabilityRequest>,
    diagnostics: &mut Vec<QueryDiagnostic>,
) {
    if let Some(request) = request
        && request.ordering == ComparisonOrdering::Declared
    {
        for node in by_path
            .values()
            .filter(|node| !edges.iter().any(|edge| edge.source_path == node.path))
        {
            if node.stream.ordering.kind != QueryRowOrder::Ordered
                || node.stream.ordering.ties != QueryTieOrder::Total
                || request.comparison == Comparison::FinalRelation
            {
                diagnostic(
                    diagnostics,
                    &format!("{}.stream.ordering", node.path),
                    "row_order_unproven",
                    SupportStatus::Unknown,
                    "declared row and tie order at the requested cutoff is not proved",
                );
            }
        }
    }
}

fn status_name(status: SupportStatus) -> &'static str {
    match status {
        SupportStatus::Supported => "supported",
        SupportStatus::Unsupported => "unsupported",
        SupportStatus::Unknown => "unknown",
    }
}

fn conjunction(values: impl IntoIterator<Item = SupportStatus>) -> SupportStatus {
    values
        .into_iter()
        .fold(SupportStatus::Supported, |result, next| {
            match (result, next) {
                (SupportStatus::Unsupported, _) | (_, SupportStatus::Unsupported) => {
                    SupportStatus::Unsupported
                }
                (SupportStatus::Unknown, _) | (_, SupportStatus::Unknown) => SupportStatus::Unknown,
                _ => SupportStatus::Supported,
            }
        })
}

fn diagnostic(
    issues: &mut Vec<QueryDiagnostic>,
    path: &str,
    code: &str,
    status: SupportStatus,
    message: &str,
) {
    issues.push(QueryDiagnostic {
        path: path.into(),
        code: code.into(),
        status,
        message: message.into(),
    });
}

fn node_facts(
    node: &Declaration,
    diagnostics: &mut Vec<QueryDiagnostic>,
    update_semantics: crate::UpdateSemantics,
) -> Result<(QueryNodeFacts, sql::Proof, bool)> {
    let schema = node
        .schema
        .as_ref()
        .map(|fields| {
            fields
                .iter()
                .map(crate::config::field_from_spec)
                .collect::<Result<Vec<_>>>()
                .map(|fields| Arc::new(Schema::new(fields)))
        })
        .transpose()?;
    let class = classify(node, schema.is_some(), diagnostics, update_semantics);
    let proof = class.proof;
    let aggregate = class.aggregate;
    let support = class.support;
    let stream = stream_facts(node, class);
    let batch = batch_facts(node, &stream, aggregate, support);
    for (mode, facts) in [("batch", &batch), ("stream", &stream)] {
        if facts.support == SupportStatus::Unsupported
            && !diagnostics.iter().any(|issue| {
                issue.status == SupportStatus::Unsupported
                    && issue.path.starts_with(&format!("{}.{mode}.", node.path))
            })
        {
            diagnostic(
                diagnostics,
                &format!("{}.{mode}.support", node.path),
                "unsupported_mode",
                SupportStatus::Unsupported,
                "this declaration has no supported lifecycle in the selected mode",
            );
        }
    }
    Ok((
        QueryNodeFacts {
            path: node.path.clone(),
            schema,
            batch,
            stream,
        },
        proof,
        aggregate,
    ))
}

fn batch_facts(
    node: &Declaration,
    stream: &QueryModeFacts,
    aggregate: bool,
    support: SupportStatus,
) -> QueryModeFacts {
    let stream_only = stream_only_kind(&node.kind);
    let mut batch = stream.clone();
    batch.support = node.batch_support.unwrap_or(if stream_only {
        SupportStatus::Unsupported
    } else {
        support
    });
    if aggregate && node.kind != "snapshot_sql" {
        batch.history = HistoryScope::CurrentBatch;
        batch.finality = QueryFinality::PerBatch;
        batch.update = UpdateKind::Append;
        batch.reason = "each batch execute consumes one independent complete input".into();
    }
    batch
}

struct Classification {
    proof: sql::Proof,
    aggregate: bool,
    history: HistoryScope,
    ordering: QueryOrderingFacts,
    support: SupportStatus,
}

fn row_local_kind(kind: &str) -> bool {
    matches!(
        kind,
        "table_input"
            | "project"
            | "filter"
            | "with_columns"
            | "column_ref"
            | "literal"
            | "add"
            | "sub"
            | "mul"
            | "truediv"
            | "gt"
            | "ge"
            | "lt"
            | "le"
            | "eq"
            | "ne"
            | "cast"
            | "coalesce"
            | "and"
            | "or"
            | "not"
            | "neg"
            | "is_null"
    )
}

fn classify(
    node: &Declaration,
    schema_known: bool,
    diagnostics: &mut Vec<QueryDiagnostic>,
    update_semantics: crate::UpdateSemantics,
) -> Classification {
    let row_local = row_local_kind(&node.kind);
    let mut proof = if row_local {
        sql::Proof::RowLocal
    } else {
        sql::Proof::Unknown
    };
    let mut aggregate = false;
    let mut history = if row_local {
        HistoryScope::CurrentBatch
    } else {
        HistoryScope::Unknown
    };
    let mut ordering = node.ordering.clone();
    let mut support = if row_local {
        SupportStatus::Supported
    } else {
        SupportStatus::Unknown
    };
    if let Some(query) = &node.query {
        if let Ok(sql) = sql::inspect(
            query,
            node.schema.as_deref().unwrap_or_default(),
            !update_semantics.is_legacy(),
        ) {
            proof = sql.proof;
            aggregate = sql.aggregate;
            ordering = sql.ordering;
            support = if schema_known {
                SupportStatus::Supported
            } else {
                SupportStatus::Unknown
            };
            history = sql_history(aggregate, proof);
            record_volatile_sql(node, query, diagnostics);
        } else {
            support = SupportStatus::Unsupported;
            for mode in ["batch", "stream"] {
                diagnostic(
                    diagnostics,
                    &format!("{}.{mode}.support", node.path),
                    "invalid_literal",
                    support,
                    "SQL must be one valid read-only SELECT/CTE; location is the SQL node",
                );
            }
        }
    }
    Classification {
        proof,
        aggregate,
        history,
        ordering,
        support,
    }
}

fn sql_history(aggregate: bool, proof: sql::Proof) -> HistoryScope {
    if aggregate {
        HistoryScope::AcceptedHistory
    } else if proof == sql::Proof::Unknown {
        HistoryScope::Unknown
    } else {
        HistoryScope::CurrentBatch
    }
}

fn record_volatile_sql(node: &Declaration, query: &str, diagnostics: &mut Vec<QueryDiagnostic>) {
    if crate::expression::validate_no_volatile_functions(
        &node.path,
        query,
        crate::expression::default_function_registry(),
    )
    .is_err()
    {
        diagnostic(
            diagnostics,
            &format!("{}.stream.support", node.path),
            "unsupported_mode",
            SupportStatus::Unsupported,
            "volatile or wall-clock SQL cannot satisfy deterministic stream replay",
        );
    }
}

fn stream_only_kind(kind: &str) -> bool {
    matches!(
        kind,
        "snapshot_sql"
            | "union"
            | "window_tumbling"
            | "window_hopping"
            | "window"
            | "stream_join"
            | "stream_asof_join"
            | "stream_only"
    )
}

fn stream_facts(node: &Declaration, class: Classification) -> QueryModeFacts {
    let Classification {
        proof,
        aggregate,
        mut history,
        ordering,
        mut support,
    } = class;
    let (mut update, mut finality) = ordinary_lifecycle(aggregate, proof);
    let stream_only = stream_only_kind(&node.kind);
    if stream_only {
        support = SupportStatus::Supported;
        (history, update, finality) = match node.kind.as_str() {
            "snapshot_sql" => (
                HistoryScope::LatestRelation,
                UpdateKind::CompleteSnapshot,
                QueryFinality::RevisableSnapshot,
            ),
            "window_tumbling" | "window_hopping" | "window" => (
                HistoryScope::EventWindow,
                UpdateKind::Append,
                QueryFinality::GroupFinal,
            ),
            "union" => (
                HistoryScope::CurrentBatch,
                UpdateKind::Append,
                QueryFinality::PerRowFinal,
            ),
            "stream_join" | "stream_asof_join" => (
                HistoryScope::AcceptedHistory,
                UpdateKind::Append,
                QueryFinality::Unknown,
            ),
            _ => (
                HistoryScope::Unknown,
                UpdateKind::Unknown,
                QueryFinality::Unknown,
            ),
        };
    }
    QueryModeFacts {
        support: node.stream_support.unwrap_or(support),
        history,
        update,
        finality,
        ordering,
        reason: history_reason(history).into(),
    }
}

fn ordinary_lifecycle(aggregate: bool, proof: sql::Proof) -> (UpdateKind, QueryFinality) {
    let update = if aggregate {
        UpdateKind::CompleteSnapshot
    } else if proof != sql::Proof::Unknown {
        UpdateKind::Append
    } else {
        UpdateKind::Unknown
    };
    let finality = if aggregate {
        QueryFinality::RevisableSnapshot
    } else if proof == sql::Proof::RowLocal {
        QueryFinality::PerRowFinal
    } else if proof.per_batch() {
        QueryFinality::PerBatch
    } else {
        QueryFinality::Unknown
    };
    (update, finality)
}

fn history_reason(history: HistoryScope) -> &'static str {
    match history {
        HistoryScope::AcceptedHistory => {
            "ordinary aggregates consume accepted input events and emit revisable cumulative snapshots"
        }
        HistoryScope::CurrentBatch => "this declaration reads the current batch",
        HistoryScope::EventWindow => "groups finalize only on declared event-window closure",
        HistoryScope::LatestRelation => "queries latest initialized replacement relations",
        HistoryScope::Unknown => "no rule proves this declaration's history",
    }
}

fn record_unknown_support(
    node: &Declaration,
    facts: &QueryNodeFacts,
    diagnostics: &mut Vec<QueryDiagnostic>,
) {
    for (mode, local) in [("batch", &facts.batch), ("stream", &facts.stream)] {
        if local.support == SupportStatus::Unknown {
            diagnostic(
                diagnostics,
                &format!("{}.{mode}.support", node.path),
                "query_semantics_unproven",
                SupportStatus::Unknown,
                "lifecycle or schema proof is incomplete",
            );
        }
    }
}

fn history_mismatch(
    request: &PortabilityRequest,
    facts: &QueryNodeFacts,
    proof: sql::Proof,
    aggregate: bool,
) -> bool {
    (request.comparison == Comparison::PerBatch && aggregate)
        || (request.comparison == Comparison::FinalRelation
            && proof.counterexample()
            && facts.stream.update == UpdateKind::Append)
        || (request.history == HistoryScope::EventWindow
            && facts.stream.history != HistoryScope::EventWindow)
}

fn semantics_unproven(
    request: &PortabilityRequest,
    facts: &QueryNodeFacts,
    proof: sql::Proof,
) -> bool {
    matches!(proof, sql::Proof::Unknown | sql::Proof::PerBatchUnproven)
        || (request.comparison == Comparison::PerBatch && proof == sql::Proof::Selection)
        || (request.comparison == Comparison::FinalRelation
            && proof.per_batch()
            && facts.stream.update != UpdateKind::Append)
}

fn obligations(
    node: &Declaration,
    facts: &QueryNodeFacts,
    proof: sql::Proof,
    aggregate: bool,
    request: &PortabilityRequest,
    diagnostics: &mut Vec<QueryDiagnostic>,
) {
    record_unknown_support(node, facts, diagnostics);
    if history_mismatch(request, facts, proof, aggregate) {
        diagnostic(
            diagnostics,
            &format!("{}.stream.history", node.path),
            "history_scope_mismatch",
            SupportStatus::Unsupported,
            "the observed query history cannot satisfy the requested comparison",
        );
    }
    if semantics_unproven(request, facts, proof) {
        diagnostic(
            diagnostics,
            &format!("{}.stream", node.path),
            if node.query.is_some() {
                "sql_semantics_unproven"
            } else {
                "query_semantics_unproven"
            },
            SupportStatus::Unknown,
            "no rule proves this declaration's requested comparison; location is the declaration node",
        );
    } else if aggregate && request.comparison == Comparison::FinalRelation {
        diagnostic(
            diagnostics,
            &format!("{}.stream.finality", node.path),
            "finality_unproven",
            SupportStatus::Unknown,
            "a job with no input events has no last complete snapshot; empty-history reduction is not proved",
        );
    }
}
