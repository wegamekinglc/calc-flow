//! Admission for immutable latest-relation declarations. No execution state lives here.

use super::{
    Declaration, DeclarationEdge, HistoryScope, InputInterpretation, QueryDiagnostic,
    QueryEdgeModeFacts, QueryNodeFacts, SupportStatus, UpdateKind, conjunction, diagnostic, sql,
};

pub(super) fn node(
    declaration: &Declaration,
    facts: &mut QueryNodeFacts,
    proof: sql::Proof,
    aggregate: bool,
    issues: &mut Vec<QueryDiagnostic>,
) {
    if proof == sql::Proof::RowLocal && facts.stream.update == UpdateKind::CompleteSnapshot {
        facts.stream.history = HistoryScope::LatestRelation;
        facts.stream.reason = "row-local transform of one complete latest relation event".into();
    }
    for (mode, local) in [("batch", &mut facts.batch), ("stream", &mut facts.stream)] {
        validate_mode(declaration, mode, local, issues);
    }
    if unproved_transform(declaration, facts, proof, aggregate) {
        facts.stream.support = SupportStatus::Unknown;
    }
}

fn unproved_transform(
    declaration: &Declaration,
    facts: &QueryNodeFacts,
    proof: sql::Proof,
    aggregate: bool,
) -> bool {
    !aggregate
        && proof != sql::Proof::RowLocal
        && facts.stream.update == UpdateKind::CompleteSnapshot
        && declaration.kind != "snapshot_sql"
}

fn validate_mode(
    declaration: &Declaration,
    mode: &str,
    local: &mut super::QueryModeFacts,
    issues: &mut Vec<QueryDiagnostic>,
) {
    if local.support == SupportStatus::Unsupported {
        return;
    }
    let missing_schema = missing_schema(declaration);
    if missing_schema
        || local.support == SupportStatus::Unknown
        || local.update == UpdateKind::Unknown
    {
        local.support = SupportStatus::Unknown;
        let code = if missing_schema {
            "update_event_boundary_unproved"
        } else {
            "update_semantics_unproved"
        };
        diagnostic(
            issues,
            &format!("{}.{}.update", declaration.path, mode),
            code,
            SupportStatus::Unknown,
            "latest_relation requires proved producer semantics and exact table boundaries",
        );
    }
}

fn missing_schema(declaration: &Declaration) -> bool {
    declaration.schema.is_none()
        && matches!(
            declaration.kind.as_str(),
            "table_input" | "expression" | "sql" | "project" | "filter" | "with_columns" | "union"
        )
}

pub(super) fn edge(
    edge: &DeclarationEdge,
    stream: &mut QueryEdgeModeFacts,
    target: &(sql::Proof, bool, String),
    reason: &str,
    issues: &mut Vec<QueryDiagnostic>,
) {
    if stream.producer_update != UpdateKind::CompleteSnapshot {
        return;
    }
    if matches!(target.0, sql::Proof::RowLocal | sql::Proof::Replacement)
        || target.2 == "snapshot_sql"
    {
        stream.consumer_interpretation = InputInterpretation::CompleteReplacement;
        stream.reason =
            "one complete Batch, including an empty relation, replaces this input".into();
    } else if append_only(target) {
        stream.support = SupportStatus::Unsupported;
        let reason = if target.1 {
            reason
        } else {
            "this consumer is append-only"
        };
        diagnostic(
            issues,
            &edge.path,
            "update_interpretation_mismatch",
            SupportStatus::Unsupported,
            &format!(
                "latest_relation requires complete replacement at this input; producer emits complete_snapshot but consumer appends; {reason}"
            ),
        );
    } else {
        stream.support = conjunction([stream.support, SupportStatus::Unknown]);
        stream.consumer_interpretation = InputInterpretation::Unknown;
        diagnostic(
            issues,
            &edge.path,
            "update_semantics_unproved",
            SupportStatus::Unknown,
            "complete replacement consumption is not proved for this input",
        );
    }
}

fn append_only(target: &(sql::Proof, bool, String)) -> bool {
    target.1
        || matches!(
            target.2.as_str(),
            "union"
                | "window"
                | "window_tumbling"
                | "window_hopping"
                | "rolling"
                | "cross_section"
                | "stream_join"
                | "stream_asof_join"
        )
}
