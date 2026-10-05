use super::*;

mod projection;

pub(super) async fn configure(
    checkpoint: Option<&OpenedCheckpointRuntime>,
    sources: &mut BTreeMap<String, SourceBinding>,
    owner: &HistoryOwner,
    plan: &mut StreamRuntimePlanParts,
) -> crate::Result<()> {
    configure_sources(checkpoint, sources, owner).await?;
    configure_replay_nodes(checkpoint, sources, plan);
    Ok(())
}

async fn configure_sources(
    checkpoint: Option<&OpenedCheckpointRuntime>,
    sources: &mut BTreeMap<String, SourceBinding>,
    owner: &HistoryOwner,
) -> crate::Result<()> {
    for (source_id, source) in sources.iter_mut() {
        let restored = checkpoint
            .and_then(|checkpoint| checkpoint.selected.as_ref())
            .and_then(|selected| selected.manifest.sources().get(source_id))
            .and_then(|entry| entry.history.as_ref());
        source.validate_history(
            source_id,
            restored,
            checkpoint.is_some_and(|checkpoint| checkpoint.selected.is_some()),
        )?;
        if source.history_spec().is_some() {
            let checkpoint = managed_checkpoint(checkpoint)?;
            source
                .install_history(
                    source_id,
                    checkpoint.transaction.clone(),
                    checkpoint.next_epoch,
                    restored.cloned(),
                )
                .await?;
            if let Some(context) = source.history_context() {
                owner.contexts.lock().insert(source_id.clone(), context);
            }
        }
    }
    Ok(())
}

fn managed_checkpoint(
    checkpoint: Option<&OpenedCheckpointRuntime>,
) -> crate::Result<&OpenedCheckpointRuntime> {
    checkpoint
        .filter(|checkpoint| checkpoint.managed || cfg!(test))
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: "source_history".into(),
            message: "immutable history requires managed checkpoint storage".into(),
        })
}

fn configure_replay_nodes(
    checkpoint: Option<&OpenedCheckpointRuntime>,
    sources: &BTreeMap<String, SourceBinding>,
    plan: &mut StreamRuntimePlanParts,
) {
    let mut configured = BTreeMap::new();
    for node in &plan.nodes {
        if !matches!(
            node.operator,
            crate::pipeline::CompiledStreamOperator::StreamAsofJoin(_)
        ) {
            continue;
        }
        let selected = checkpoint
            .and_then(|checkpoint| checkpoint.selected.as_ref())
            .and_then(|selected| selected.manifest.operators().get(node.operator_id.as_str()));
        if selected.is_some_and(|entry| !entry.inline_metadata.contains_key("source_replay")) {
            continue;
        }
        let bindings =
            ["left", "right"].map(|port| replay_binding(plan, &node.node_id, port, sources));
        let [Some(left), Some(right)] = bindings else {
            continue;
        };
        configured.insert(node.node_id.clone(), [left, right]);
    }
    for node in &mut plan.nodes {
        if let crate::pipeline::CompiledStreamOperator::StreamAsofJoin(operator) =
            &mut node.operator
            && let Some([left, right]) = configured.remove(&node.node_id)
        {
            operator.configure_source_replay(
                [left.0, right.0],
                [left.1, right.1],
                [left.2, right.2],
            );
        }
    }
}

type ReplayBinding = (
    String,
    crate::SourceHistoryContext,
    Arc<dyn crate::SourceHistoryReplayFactory>,
);

fn replay_binding(
    plan: &StreamRuntimePlanParts,
    node_id: &str,
    port: &str,
    sources: &BTreeMap<String, SourceBinding>,
) -> Option<ReplayBinding> {
    let (binding, steps) = replay_path(plan, node_id, port)?;
    let source = sources.get(&binding)?;
    Some((
        binding,
        source.history_context()?,
        projection::factory(source.history_replay_factory()?, steps),
    ))
}

fn replay_path(
    plan: &StreamRuntimePlanParts,
    node_id: &str,
    ingress: &str,
) -> Option<(String, Vec<crate::ExpressionOperator>)> {
    let mut target = (node_id, ingress);
    let mut steps = Vec::new();
    for _ in 0..=plan.nodes.len() {
        let node = plan.nodes.iter().find(|node| node.node_id == target.0)?;
        let edge = plan.edges.get(node.ingress_edges.get(target.1)?)?;
        match &edge.producer {
            crate::pipeline::RuntimeProducer::Source { binding_id } => {
                steps.reverse();
                return Some((binding_id.clone(), steps));
            }
            crate::pipeline::RuntimeProducer::Node { node_id, port } => {
                let (source_id, operator) = projection_step(plan, node_id, port)?;
                steps.push(operator);
                target = (source_id, "input");
            }
        }
    }
    None
}

fn projection_step<'a>(
    plan: &'a StreamRuntimePlanParts,
    node_id: &str,
    port: &str,
) -> Option<(&'a str, crate::ExpressionOperator)> {
    if port != "output" {
        return None;
    }
    let node = plan.nodes.iter().find(|node| node.node_id == node_id)?;
    let crate::pipeline::CompiledStreamOperator::Expression(operator) = &node.operator else {
        return None;
    };
    Some((node.node_id.as_str(), operator.source_replay_projection()?))
}

#[derive(Default)]
pub(super) struct HistoryOwner {
    contexts: Mutex<BTreeMap<String, crate::SourceHistoryContext>>,
}

impl HistoryOwner {
    pub(super) async fn drain(&self) -> Vec<Arc<RuntimeFailure>> {
        let contexts = std::mem::take(&mut *self.contexts.lock());
        let mut failures = Vec::new();
        for (binding_id, context) in contexts {
            failures.extend(context.drain().await.into_iter().map(|error| {
                Arc::new(RuntimeFailure {
                    origin: FailureOrigin::SourceClose {
                        binding_id: binding_id.clone(),
                    },
                    error,
                })
            }));
        }
        failures
    }
}
