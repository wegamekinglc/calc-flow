use super::*;

mod projection;

use crate::runtime::streaming::runner::checkpoint_v4::{encode_new_objects, encode_objects};
use crate::state::v4::{
    EncodedObject, NewObject, ObjectPayload, OpaqueObject, V4Handle, V4Manifest, V4Store,
    native::V4ObjectStore,
};
use crate::{Result, StreamJobContext};

pub(crate) struct ManagedHistoryStore {
    store: V4Store,
    job: StreamJobContext,
}

impl ManagedHistoryStore {
    pub(crate) fn new(root: std::path::PathBuf, job: &StreamJobContext) -> Self {
        Self {
            store: V4Store::new(root),
            job: job.clone(),
        }
    }
}

#[async_trait::async_trait]
impl V4ObjectStore for ManagedHistoryStore {
    async fn write_objects(
        &mut self,
        epoch: u64,
        objects: Vec<NewObject>,
    ) -> Result<Vec<EncodedObject>> {
        let encoded = encode_new_objects(
            &self.job,
            "source-history-objects",
            objects,
            self.store.root().to_owned(),
            epoch,
        )
        .await?;
        self.store.write_encoded_objects(encoded).await
    }

    async fn write_opaque(
        &mut self,
        epoch: u64,
        objects: Vec<OpaqueObject>,
    ) -> Result<Vec<EncodedObject>> {
        let encoded = encode_objects(
            &self.job,
            "source-history-opaque",
            objects,
            self.store.root().to_owned(),
            epoch,
        )
        .await?;
        self.store.write_encoded_opaque(encoded).await
    }

    fn record_carried(&mut self, count: usize) {
        self.store.record_carried(count);
    }

    fn read_epoch_manifest(&self, epoch: u64) -> Result<Option<V4Manifest>> {
        self.store.read_epoch_manifest(epoch)
    }

    fn read_object(&mut self, handle: &V4Handle) -> Result<ObjectPayload> {
        self.store.read_object(handle)
    }
}

pub(super) fn configure(
    checkpoint: Option<&OpenedCheckpointRuntime>,
    job: &StreamJobContext,
    sources: &mut BTreeMap<String, SourceBinding>,
    owner: &HistoryOwner,
    plan: &mut StreamRuntimePlanParts,
) -> Result<()> {
    if checkpoint.is_none() {
        return Ok(());
    }
    configure_sources(checkpoint, job, sources, owner)?;
    configure_replay_nodes(checkpoint, sources, plan);
    Ok(())
}

fn configure_sources(
    checkpoint: Option<&OpenedCheckpointRuntime>,
    job: &StreamJobContext,
    sources: &mut BTreeMap<String, SourceBinding>,
    owner: &HistoryOwner,
) -> Result<()> {
    for (source_id, source) in sources.iter_mut() {
        let restored = checkpoint
            .and_then(|checkpoint| checkpoint.selected_v4.as_ref())
            .and_then(|selected| selected.manifest.sources.get(source_id))
            .and_then(|entry| entry.history.as_ref());
        source.validate_history(
            source_id,
            restored,
            checkpoint.is_some_and(|checkpoint| checkpoint.selected_v4.is_some()),
        )?;
        if source.history_spec().is_some() {
            let checkpoint = managed_checkpoint(checkpoint)?;
            source.install_history(
                source_id,
                &checkpoint.manifest_root.join("lineage"),
                job,
                checkpoint.next_epoch,
                restored.cloned(),
            )?;
            if let Some(context) = source.history_context() {
                owner.contexts.lock().insert(source_id.clone(), context);
            }
        }
    }
    Ok(())
}

fn managed_checkpoint(
    checkpoint: Option<&OpenedCheckpointRuntime>,
) -> Result<&OpenedCheckpointRuntime> {
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
            .and_then(|checkpoint| checkpoint.selected_v4.as_ref())
            .and_then(|selected| selected.manifest.operators.get(node.operator_id.as_str()));
        if selected.is_some_and(|entry| {
            !entry
                .opaque_metadata()
                .is_some_and(|(metadata, _)| metadata.contains_key("source_replay"))
        }) {
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
