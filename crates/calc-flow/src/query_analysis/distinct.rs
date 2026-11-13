//! Structural facts accompany the runtime's mandatory DISTINCT path admission.
use super::{
    DeclarationEdge, NodeProofs, QueryDiagnostic, QueryNodeFacts, SupportStatus, diagnostic, sql,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate(
    facts: &mut BTreeMap<String, QueryNodeFacts>,
    proofs: &NodeProofs,
    edges: &[DeclarationEdge],
    invalid_times: &BTreeSet<String>,
    issues: &mut Vec<QueryDiagnostic>,
) {
    for (root, _) in proofs.iter().filter(|(_, proof)| proof.2 == "distinct") {
        let mut sources = BTreeSet::new();
        let admitted = walk(
            root,
            true,
            facts,
            proofs,
            edges,
            &mut BTreeSet::new(),
            &mut sources,
        ) && sources.len() == 1
            && walk(
                root,
                false,
                facts,
                proofs,
                edges,
                &mut BTreeSet::new(),
                &mut BTreeSet::new(),
            );
        if admitted && !invalid_times.contains(root) {
            continue;
        }
        diagnostic(
            issues,
            &issue_path(root, invalid_times),
            "capability_mismatch",
            SupportStatus::Unsupported,
            "DISTINCT requires one source, deterministic row-local upstream/downstream and no nested/stateful/snapshot/array boundary",
        );
        let mut pending = vec![root.as_str()];
        let mut seen = BTreeSet::new();
        while let Some(path) = pending.pop() {
            if !seen.insert(path) {
                continue;
            }
            let node = facts.get_mut(path).expect("analyzed path exists");
            node.batch.support = SupportStatus::Unsupported;
            node.stream.support = SupportStatus::Unsupported;
            pending.extend(
                edges
                    .iter()
                    .filter(|e| e.source_path == path)
                    .map(|e| e.target_path.as_str()),
            );
        }
    }
}

pub(super) fn invalid_times(
    nodes: &[super::Declaration],
    edges: &[DeclarationEdge],
) -> BTreeSet<String> {
    let nodes: BTreeMap<_, _> = nodes.iter().map(|n| (n.path.as_str(), n)).collect();
    nodes
        .values()
        .filter_map(|node| {
            let crate::DistinctHistory::ClosedWindow {
                event_time_column, ..
            } = node.distinct_spec.as_ref()?.history()
            else {
                return None;
            };
            (!time_is_input(&node.path, event_time_column, &nodes, edges))
                .then(|| node.path.clone())
        })
        .collect()
}

fn time_is_input<'a>(
    mut path: &'a str,
    time: &str,
    nodes: &BTreeMap<&str, &super::Declaration>,
    edges: &'a [DeclarationEdge],
) -> bool {
    let mut time = time.to_owned();
    let mut seen = BTreeSet::new();
    while seen.insert(path) {
        let incoming = edges
            .iter()
            .filter(|e| e.target_path == path)
            .collect::<Vec<_>>();
        let [edge] = incoming.as_slice() else {
            return incoming.is_empty() && nodes[path].schema.is_some();
        };
        path = &edge.source_path;
        let node = nodes[path];
        if node.query.is_none() {
            continue;
        }
        let alias = edges
            .iter()
            .find(|e| e.target_path == path)
            .and_then(|e| e.input_alias.as_deref())
            .unwrap_or("input");
        let Ok(original) =
            crate::sql::window_time_column(node.query.as_deref().unwrap(), alias, &time)
        else {
            return false;
        };
        time = original;
    }
    false
}

fn walk<'a>(
    path: &'a str,
    upstream: bool,
    facts: &BTreeMap<String, QueryNodeFacts>,
    proofs: &NodeProofs,
    edges: &'a [DeclarationEdge],
    seen: &mut BTreeSet<&'a str>,
    sources: &mut BTreeSet<&'a str>,
) -> bool {
    if !seen.insert(path) {
        return true;
    }
    let adjacent = adjacent(path, upstream, edges);
    if upstream && adjacent.is_empty() && facts[path].schema.is_some() {
        sources.insert(path);
    }
    adjacent.into_iter().all(|next| {
        proofs[next].0 == sql::Proof::RowLocal
            && !proofs[next].1
            && walk(next, upstream, facts, proofs, edges, seen, sources)
    })
}
fn issue_path(root: &str, invalid_times: &BTreeSet<String>) -> String {
    let field = if invalid_times.contains(root) {
        "spec.history"
    } else {
        "input"
    };
    format!("{root}.distinct.{field}")
}
fn adjacent<'a>(path: &str, upstream: bool, edges: &'a [DeclarationEdge]) -> Vec<&'a str> {
    edges
        .iter()
        .filter(|e| {
            if upstream {
                e.target_path == path
            } else {
                e.source_path == path
            }
        })
        .map(|e| {
            if upstream {
                e.source_path.as_str()
            } else {
                e.target_path.as_str()
            }
        })
        .collect()
}
