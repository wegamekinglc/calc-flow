use super::*;

pub(super) async fn configure(
    checkpoint: Option<&OpenedCheckpointRuntime>,
    sources: &mut BTreeMap<String, SourceBinding>,
    owner: &HistoryOwner,
    plan: &mut StreamRuntimePlanParts,
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
            let checkpoint = checkpoint
                .filter(|checkpoint| checkpoint.managed || cfg!(test))
                .ok_or_else(|| CalcFlowError::InvalidArgument {
                    field: "source_history".into(),
                    message: "immutable history requires managed checkpoint storage".into(),
                })?;
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
    for node in &mut plan.nodes {
        let crate::pipeline::CompiledStreamOperator::StreamAsofJoin(operator) = &mut node.operator
        else {
            continue;
        };
        let selected = checkpoint
            .and_then(|checkpoint| checkpoint.selected.as_ref())
            .and_then(|selected| selected.manifest.operators().get(node.operator_id.as_str()));
        if selected.is_some_and(|entry| !entry.inline_metadata.contains_key("source_replay")) {
            continue;
        }
        let bindings = ["left", "right"].map(|port| {
            let route = plan
                .source_routes
                .values()
                .find(|route| route.target.node_id == node.node_id && route.target.port == port)?;
            let source = sources.get(&route.binding_id)?;
            Some((
                route.binding_id.clone(),
                source.history_context()?,
                source.history_replay_factory()?,
            ))
        });
        let [Some(left), Some(right)] = bindings else {
            continue;
        };
        operator.configure_source_replay([left.0, right.0], [left.1, right.1], [left.2, right.2]);
    }
    Ok(())
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
