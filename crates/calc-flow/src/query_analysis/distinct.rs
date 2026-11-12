//! Structural facts accompany the runtime's mandatory DISTINCT path admission.
use super::{
    DeclarationEdge, NodeProofs, QueryDiagnostic, QueryNodeFacts, SupportStatus, diagnostic, sql,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate(
    facts: &mut BTreeMap<String, QueryNodeFacts>,
    proofs: &NodeProofs,
    edges: &[DeclarationEdge],
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
        if admitted {
            continue;
        }
        diagnostic(
            issues,
            &format!("{root}.distinct.input"),
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
    let adjacent = edges
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
        .collect::<Vec<_>>();
    if upstream && adjacent.is_empty() && facts[path].schema.is_some() {
        sources.insert(path);
    }
    adjacent.into_iter().all(|next| {
        proofs[next].0 == sql::Proof::RowLocal
            && !proofs[next].1
            && walk(next, upstream, facts, proofs, edges, seen, sources)
    })
}
