use super::*;

pub(super) async fn configure(
    checkpoint: Option<&OpenedCheckpointRuntime>,
    sources: &mut BTreeMap<String, SourceBinding>,
    owner: &HistoryOwner,
) -> crate::Result<()> {
    for (source_id, source) in sources {
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
