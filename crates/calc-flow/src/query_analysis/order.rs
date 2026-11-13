//! Final ordering proofs do not imply temporal identity or observed row uniqueness.
use super::{
    BTreeMap, Comparison, ComparisonCutoff, Declaration, DeclarationEdge, HistoryScope, NodeFacts,
    NodeProofs, NullPlacement, PortabilityRequest, QueryDiagnostic, QueryFinality, QueryModeFacts,
    QueryOrderingFacts, QueryRowOrder, QuerySortKey, QueryTieOrder, SortDirection, SupportStatus,
    UpdateKind, diagnostic, distinct, sql,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OrderDeclaration {
    pub by: Vec<crate::SortKey>,
    pub fetch: Option<u64>,
    pub spec: crate::SortSpec,
}
impl OrderDeclaration {
    pub(super) fn ordering(&self) -> QueryOrderingFacts {
        let mut keys = self
            .by
            .iter()
            .map(|key| QuerySortKey {
                field: key.column().into(),
                direction: key.direction(),
                nulls: key.nulls(),
            })
            .collect::<Vec<_>>();
        let stable = if let crate::SortTies::Sequence { columns } = self.spec.ties() {
            keys.extend(columns.iter().map(|field| QuerySortKey {
                field: field.clone(),
                direction: SortDirection::Ascending,
                nulls: NullPlacement::Last,
            }));
            true
        } else {
            false
        };
        QueryOrderingFacts {
            kind: QueryRowOrder::Ordered,
            keys,
            ties: if stable {
                QueryTieOrder::Total
            } else {
                QueryTieOrder::Unspecified
            },
            temporal_identity: None,
        }
    }
    pub(super) fn facts(&self, node: &Declaration) -> QueryModeFacts {
        let (history, closure) = match self.spec.history() {
            crate::SortHistory::BoundedSource => (
                HistoryScope::AcceptedHistory,
                "requires a fixed Bounded Source cut and actual normal EOF",
            ),
            crate::SortHistory::ClosedWindow { .. } => (
                HistoryScope::EventWindow,
                "requires the selected fixed window and a trustworthy watermark or bounded EOF route",
            ),
        };
        QueryModeFacts {
            support: node.stream_support.unwrap_or(SupportStatus::Supported),
            history,
            update: UpdateKind::Append,
            finality: QueryFinality::GroupFinal,
            ordering: self.ordering(),
            reason: format!(
                "bounded_order_v1/copy_visible_v2: same accepted H and cut; full-H candidates, sequence validation, sorting/output/live captures use an owned-only peak budget; borrowed ingress and other owners excluded; {closure}; stable ties require reproducible non-NULL sequence fields checked unique over H; k/offset use UInt32 indices; local policy={:?}, layout={}",
                self.spec.late_policy(),
                self.spec.state_layout()
            ),
        }
    }
    pub(super) fn obligations(
        &self,
        node: &Declaration,
        request: &PortabilityRequest,
        issues: &mut Vec<QueryDiagnostic>,
    ) {
        let domain = match self.spec.history() {
            crate::SortHistory::BoundedSource => {
                request.history == HistoryScope::AcceptedHistory
                    && request.cutoff == ComparisonCutoff::AllSourcesEof
            }
            crate::SortHistory::ClosedWindow { .. } => matches!(
                (request.history, request.cutoff),
                (HistoryScope::EventWindow, ComparisonCutoff::ClosedWindow)
                    | (
                        HistoryScope::AcceptedHistory,
                        ComparisonCutoff::AllSourcesEof
                    )
            ),
        };
        if request.comparison != Comparison::FinalRelation || !domain {
            diagnostic(
                issues,
                &format!("{}.stream.history", node.path),
                "history_scope_mismatch",
                SupportStatus::Unsupported,
                "final ordering requires its complete accepted H and real closure, not per-batch comparison",
            );
        }
        if matches!(self.spec.ties(), crate::SortTies::Unspecified)
            && (self.fetch.is_some() || self.spec.offset() != 0)
        {
            diagnostic(
                issues,
                &format!("{}.stream.ordering", node.path),
                "row_order_unproven",
                SupportStatus::Unknown,
                "top-k/offset membership among tied keys is unproved even for unordered comparison; provide explicit reproducible sequence_by",
            );
        }
    }
}
pub(super) fn invalid_fields(
    nodes: &[Declaration],
    edges: &[DeclarationEdge],
) -> std::collections::BTreeSet<String> {
    let declarations = nodes
        .iter()
        .map(|node| (node.path.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    nodes
        .iter()
        .filter_map(|node| {
            let order = node.order.as_ref()?;
            let time = match order.spec.history() {
                crate::SortHistory::BoundedSource => None,
                crate::SortHistory::ClosedWindow {
                    event_time_column, ..
                } => Some(event_time_column),
            };
            let sequence = match order.spec.ties() {
                crate::SortTies::Unspecified => &[][..],
                crate::SortTies::Sequence { columns } => columns.as_slice(),
            };
            time.into_iter()
                .chain(sequence)
                .any(|field| !distinct::time_is_input(&node.path, field, &declarations, edges))
                .then(|| node.path.clone())
        })
        .collect()
}
pub(super) fn validate(
    facts: &mut NodeFacts,
    proofs: &NodeProofs,
    edges: &[DeclarationEdge],
    invalid: &std::collections::BTreeSet<String>,
    issues: &mut Vec<QueryDiagnostic>,
) {
    use std::collections::BTreeSet;
    for (root, _) in proofs
        .iter()
        .filter(|(_, proof)| proof.2 == "bounded_order")
    {
        let mut sources = BTreeSet::new();
        let valid = distinct::walk(
            root,
            true,
            facts,
            proofs,
            edges,
            &mut BTreeSet::new(),
            &mut sources,
        ) && sources.len() == 1
            && distinct::walk(
                root,
                false,
                facts,
                proofs,
                edges,
                &mut BTreeSet::new(),
                &mut BTreeSet::new(),
            )
            && !invalid.contains(root);
        if valid {
            continue;
        }
        diagnostic(
            issues,
            &format!("{root}.sort.spec"),
            "capability_mismatch",
            SupportStatus::Unsupported,
            "ordering requires one source, original/renamed time and sequence fields, deterministic row-local transforms and no nested/stateful/snapshot/array boundary",
        );
        let mut pending = vec![root.as_str()];
        let mut seen = BTreeSet::new();
        while let Some(path) = pending.pop() {
            if !seen.insert(path) {
                continue;
            }
            let node = facts.get_mut(path).unwrap();
            node.batch.support = SupportStatus::Unsupported;
            node.stream.support = SupportStatus::Unsupported;
            pending.extend(
                edges
                    .iter()
                    .filter(|edge| edge.source_path == path)
                    .map(|edge| edge.target_path.as_str()),
            );
        }
    }
}

pub(super) fn mapped_ordering(
    node: &Declaration,
    parent: &QueryOrderingFacts,
    alias: &str,
) -> QueryOrderingFacts {
    let Some(query) = &node.query else {
        return QueryOrderingFacts::default();
    };
    let Ok(Some(columns)) = crate::sql::sql_passthrough_columns(query, alias) else {
        return QueryOrderingFacts::default();
    };
    let mapped = parent
        .keys
        .iter()
        .map(|key| {
            let field = columns
                .iter()
                .find(|(_, source)| source.as_deref() == Some(key.field.as_str()))
                .map(|(name, _)| name.clone())
                .or_else(|| columns.contains_key("*").then(|| key.field.clone()))?;
            if !node
                .schema
                .as_ref()?
                .iter()
                .any(|column| column.name == field)
            {
                return None;
            }
            Some(QuerySortKey {
                field,
                direction: key.direction,
                nulls: key.nulls,
            })
        })
        .collect::<Option<Vec<_>>>();
    mapped.map_or_else(QueryOrderingFacts::default, |keys| QueryOrderingFacts {
        keys,
        kind: parent.kind,
        ties: parent.ties,
        temporal_identity: None,
    })
}

pub(super) fn propagate(
    node: &Declaration,
    facts: &mut super::QueryNodeFacts,
    proof: sql::Proof,
    inputs: &[&DeclarationEdge],
    by_path: &NodeFacts,
    order_paths: &mut std::collections::BTreeSet<String>,
) {
    if node.order.is_some() {
        order_paths.insert(node.path.clone());
    } else if proof == sql::Proof::RowLocal
        && inputs.len() == 1
        && order_paths.contains(&inputs[0].source_path)
    {
        let parent = &by_path[&inputs[0].source_path];
        facts.stream.history = parent.stream.history;
        facts.stream.finality = parent.stream.finality;
        facts.stream.reason.clone_from(&parent.stream.reason);
        let ordering = mapped_ordering(
            node,
            &parent.stream.ordering,
            inputs[0].input_alias.as_deref().unwrap_or("input"),
        );
        facts.stream.ordering = ordering.clone();
        facts.batch.ordering = ordering;
        order_paths.insert(node.path.clone());
    }
}
