use std::collections::{BTreeMap, BTreeSet, btree_map::Entry};

use async_trait::async_trait;

use super::{
    StreamJobContext,
    progress::{PreparedStreamJob, StreamProgressRuntimeConfig, prepare_stream_job},
    source_task::{SourceBinding, SourceCapabilities, validate_source_capabilities},
};
use crate::{
    Batch, CalcFlowError, CheckpointManifest, DeliveryGuarantee, EdgeBudget, Epoch, JsonMap,
    Result, RetentionClass, SinkDeliveryManifest, StreamExecutionPlan,
    json::validate_portable_identifier,
    pipeline::{
        OperatorCheckpointCapability, RuntimeProducer, RuntimeStreamNode, StreamRuntimePlanParts,
    },
};

pub(crate) struct NamedSourceBinding {
    pub(crate) binding_id: String,
    pub(crate) binding: SourceBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum M2SinkDelivery {
    ProcessLocalOrdered,
    EpochIdempotent {
        mechanism: String,
        retention: RetentionClass,
    },
    Transactional,
}

impl M2SinkDelivery {
    pub(crate) fn into_manifest(self) -> SinkDeliveryManifest {
        match self {
            Self::ProcessLocalOrdered => SinkDeliveryManifest::Ordinary,
            Self::EpochIdempotent {
                mechanism,
                retention,
            } => SinkDeliveryManifest::EpochIdempotent {
                mechanism,
                retention,
            },
            Self::Transactional => SinkDeliveryManifest::Transactional,
        }
    }
}

#[async_trait]
pub(crate) trait OrdinaryStreamSink: Send {
    fn delivery_capability(&self) -> M2SinkDelivery {
        M2SinkDelivery::ProcessLocalOrdered
    }

    async fn open(&mut self) -> Result<()>;
    async fn write(&mut self, batch: &Batch) -> Result<()>;
    async fn close(&mut self) -> Result<()>;
}

#[async_trait]
pub(crate) trait TransactionalStreamSink: Send {
    async fn open(&mut self) -> Result<()>;
    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()>;
    async fn write(&mut self, batch: &Batch) -> Result<()>;
    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap>;
    async fn commit(&mut self, epoch: Epoch, state: &JsonMap) -> Result<()>;
    async fn abort(&mut self, epoch: Epoch, state: Option<&JsonMap>) -> Result<()>;
    async fn recover(&mut self, manifest: &CheckpointManifest) -> Result<()>;
    async fn close(&mut self) -> Result<()>;
}

enum SinkImplementation {
    Ordinary(Box<dyn OrdinaryStreamSink>),
    Transactional(Box<dyn TransactionalStreamSink>),
}

pub(crate) struct OrdinarySinkBinding {
    sink: SinkImplementation,
    delivery: Option<M2SinkDelivery>,
}

impl OrdinarySinkBinding {
    pub(crate) fn new(sink: Box<dyn OrdinaryStreamSink>) -> Self {
        Self {
            sink: SinkImplementation::Ordinary(sink),
            delivery: None,
        }
    }

    pub(crate) fn new_transactional(sink: Box<dyn TransactionalStreamSink>) -> Self {
        Self {
            sink: SinkImplementation::Transactional(sink),
            delivery: Some(M2SinkDelivery::Transactional),
        }
    }

    pub(crate) fn new_epoch_idempotent(
        sink: Box<dyn TransactionalStreamSink>,
        mechanism: &str,
        retention: RetentionClass,
    ) -> Result<Self> {
        validate_portable_identifier("sink.delivery.mechanism", mechanism)?;
        Ok(Self {
            sink: SinkImplementation::Transactional(sink),
            delivery: Some(M2SinkDelivery::EpochIdempotent {
                mechanism: mechanism.into(),
                retention,
            }),
        })
    }

    fn sample_delivery_once(&mut self) -> M2SinkDelivery {
        self.delivery
            .get_or_insert_with(|| match &self.sink {
                SinkImplementation::Ordinary(sink) => {
                    let _declared = sink.delivery_capability();
                    M2SinkDelivery::ProcessLocalOrdered
                }
                SinkImplementation::Transactional(_) => M2SinkDelivery::Transactional,
            })
            .clone()
    }

    pub(crate) fn delivery(&self) -> M2SinkDelivery {
        self.delivery
            .clone()
            .expect("validated sink bindings sampled delivery capability")
    }

    pub(crate) fn is_ordinary(&self) -> bool {
        matches!(self.delivery(), M2SinkDelivery::ProcessLocalOrdered)
    }

    pub(crate) async fn open(&mut self) -> Result<()> {
        match &mut self.sink {
            SinkImplementation::Ordinary(sink) => sink.open().await,
            SinkImplementation::Transactional(sink) => sink.open().await,
        }
    }

    pub(crate) async fn write(&mut self, batch: &Batch) -> Result<()> {
        match &mut self.sink {
            SinkImplementation::Ordinary(sink) => sink.write(batch).await,
            SinkImplementation::Transactional(sink) => sink.write(batch).await,
        }
    }

    pub(crate) async fn close(&mut self) -> Result<()> {
        match &mut self.sink {
            SinkImplementation::Ordinary(sink) => sink.close().await,
            SinkImplementation::Transactional(sink) => sink.close().await,
        }
    }

    pub(crate) async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()> {
        self.transactional_mut("begin_epoch")?
            .begin_epoch(epoch)
            .await
    }

    pub(crate) async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap> {
        self.transactional_mut("pre_commit")?
            .pre_commit(epoch)
            .await
    }

    pub(crate) async fn commit(&mut self, epoch: Epoch, state: &JsonMap) -> Result<()> {
        self.transactional_mut("commit")?.commit(epoch, state).await
    }

    pub(crate) async fn abort(&mut self, epoch: Epoch, state: Option<&JsonMap>) -> Result<()> {
        self.transactional_mut("abort")?.abort(epoch, state).await
    }

    pub(crate) async fn recover(&mut self, manifest: &CheckpointManifest) -> Result<()> {
        self.transactional_mut("recover")?.recover(manifest).await
    }

    fn transactional_mut(&mut self, operation: &str) -> Result<&mut dyn TransactionalStreamSink> {
        match &mut self.sink {
            SinkImplementation::Transactional(sink) => Ok(sink.as_mut()),
            SinkImplementation::Ordinary(_) => Err(CalcFlowError::InvalidArgument {
                field: "sink.delivery".into(),
                message: format!("ordinary sink does not support {operation}"),
            }),
        }
    }
}

pub(crate) struct NamedSinkBinding {
    pub(crate) output_id: String,
    pub(crate) sink_id: String,
    pub(crate) binding: OrdinarySinkBinding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum M2DeliveryMode {
    ProcessLocalOrdered,
}

pub(crate) struct ContinuousJobSpec {
    pub(crate) context: StreamJobContext,
    pub(crate) plan: StreamExecutionPlan,
    pub(crate) sources: Vec<NamedSourceBinding>,
    pub(crate) sinks: Vec<NamedSinkBinding>,
    pub(crate) edge_budget: EdgeBudget,
    pub(crate) delivery_mode: M2DeliveryMode,
}

pub(crate) struct ValidatedOrdinarySink {
    pub(crate) sink_id: String,
    pub(crate) binding: OrdinarySinkBinding,
}

pub(crate) struct ValidatedContinuousJob {
    pub(crate) context: StreamJobContext,
    pub(crate) plan: StreamRuntimePlanParts,
    pub(crate) sources: BTreeMap<String, SourceBinding>,
    pub(crate) sinks: BTreeMap<String, Vec<ValidatedOrdinarySink>>,
    pub(crate) progress: PreparedStreamJob,
    pub(crate) delivery_mode: M2DeliveryMode,
}

pub(crate) struct OutputDeliveryProof {
    pub(crate) output_id: String,
    pub(crate) requested: DeliveryGuarantee,
    pub(crate) reachable_sources: BTreeSet<String>,
    pub(crate) reachable_operators: BTreeSet<String>,
    pub(crate) sink_mechanisms: BTreeMap<String, SinkDeliveryManifest>,
}

/// Consumes and validates the complete job before any lifecycle work begins.
pub(crate) fn preflight_job(spec: ContinuousJobSpec) -> Result<ValidatedContinuousJob> {
    let ContinuousJobSpec {
        context,
        plan,
        sources,
        sinks,
        edge_budget,
        delivery_mode,
    } = spec;
    let plan = plan.into_runtime_parts(edge_budget)?;
    validate_runtime_id(&plan.name, "plan.name")?;
    validate_context_fingerprint(&context, &plan)?;
    validate_runtime_topology(&plan)?;
    let (validated_sources, progress) = validate_sources(&plan, sources)?;
    let validated_sinks = validate_sinks(&plan, sinks)?;
    validate_delivery_requirements(&plan, &validated_sources, &validated_sinks)?;

    Ok(ValidatedContinuousJob {
        context,
        plan,
        sources: validated_sources,
        sinks: validated_sinks,
        progress,
        delivery_mode,
    })
}

fn validate_context_fingerprint(
    context: &StreamJobContext,
    plan: &StreamRuntimePlanParts,
) -> Result<()> {
    if context.fingerprint() == plan.fingerprint {
        return Ok(());
    }
    Err(CalcFlowError::InvalidArgument {
        field: "context.fingerprint".into(),
        message: "must match the consumed stream plan fingerprint".into(),
    })
}

fn validate_delivery_requirements(
    plan: &StreamRuntimePlanParts,
    sources: &BTreeMap<String, SourceBinding>,
    sinks: &BTreeMap<String, Vec<ValidatedOrdinarySink>>,
) -> Result<()> {
    for (output_id, guarantee) in &plan.requirements.delivery {
        if *guarantee != DeliveryGuarantee::ExactlyOnce {
            continue;
        }
        let proof = output_delivery_proof(plan, output_id, *guarantee, sinks)?;
        let field = format!("requirements.delivery.{}", proof.output_id);
        if let Some(source_id) = proof
            .reachable_sources
            .iter()
            .find(|source_id| !sources[*source_id].sampled_capabilities().replayable)
        {
            return Err(CalcFlowError::InvalidArgument {
                field,
                message: format!("source {source_id:?} cannot replay from an exact cursor"),
            });
        }
        if let Some(operator_id) = proof.reachable_operators.iter().find(|operator_id| {
            plan.nodes
                .iter()
                .find(|node| node.node_id.as_str() == operator_id.as_str())
                .is_none_or(|node| {
                    node.checkpoint_capability != OperatorCheckpointCapability::DeterministicRestore
                })
        }) {
            return Err(CalcFlowError::InvalidArgument {
                field,
                message: format!(
                    "operator {operator_id:?} has no proven deterministic checkpoint/restore capability"
                ),
            });
        }
        for (sink_id, mechanism) in &proof.sink_mechanisms {
            match mechanism {
                SinkDeliveryManifest::Transactional
                | SinkDeliveryManifest::EpochIdempotent {
                    retention: RetentionClass::Unbounded,
                    ..
                } => {}
                SinkDeliveryManifest::EpochIdempotent {
                    mechanism,
                    retention: RetentionClass::Bounded,
                } => {
                    return Err(CalcFlowError::InvalidArgument {
                        field,
                        message: format!(
                            "sink {sink_id:?} mechanism {mechanism:?} has bounded retention"
                        ),
                    });
                }
                SinkDeliveryManifest::Ordinary => {
                    return Err(CalcFlowError::InvalidArgument {
                        field,
                        message: format!("sink {sink_id:?} is not transactional"),
                    });
                }
            }
        }
    }
    Ok(())
}

fn output_delivery_proof(
    plan: &StreamRuntimePlanParts,
    output_id: &str,
    requested: DeliveryGuarantee,
    sinks: &BTreeMap<String, Vec<ValidatedOrdinarySink>>,
) -> Result<OutputDeliveryProof> {
    let route = plan
        .sink_routes
        .get(output_id)
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("delivery proof output {output_id:?} has no runtime route"),
        })?;
    let nodes = plan
        .nodes
        .iter()
        .map(|node| (node.node_id.as_str(), node))
        .collect::<BTreeMap<_, _>>();
    let mut pending_edges = vec![route.edge_id.clone()];
    let mut visited_edges = BTreeSet::new();
    let mut reachable_sources = BTreeSet::new();
    let mut reachable_operators = BTreeSet::new();
    while let Some(edge_id) = pending_edges.pop() {
        if !visited_edges.insert(edge_id.clone()) {
            continue;
        }
        let edge = &plan.edges[&edge_id];
        match &edge.producer {
            RuntimeProducer::Source { binding_id } => {
                reachable_sources.insert(binding_id.clone());
            }
            RuntimeProducer::Node { node_id, .. } => {
                if reachable_operators.insert(node_id.clone()) {
                    pending_edges.extend(nodes[node_id.as_str()].ingress_edges.values().cloned());
                }
            }
        }
    }
    let sink_mechanisms = sinks[output_id]
        .iter()
        .map(|sink| {
            (
                sink.sink_id.clone(),
                sink.binding.delivery().into_manifest(),
            )
        })
        .collect();
    Ok(OutputDeliveryProof {
        output_id: output_id.into(),
        requested,
        reachable_sources,
        reachable_operators,
        sink_mechanisms,
    })
}

fn validate_sources(
    plan: &StreamRuntimePlanParts,
    sources: Vec<NamedSourceBinding>,
) -> Result<(BTreeMap<String, SourceBinding>, PreparedStreamJob)> {
    let mut validated = BTreeMap::new();
    for named in sources {
        let (binding_id, binding) = validate_source(plan, named)?;
        match validated.entry(binding_id) {
            Entry::Vacant(entry) => {
                entry.insert(binding);
            }
            Entry::Occupied(entry) => return Err(duplicate_source(entry.key())),
        }
    }
    if let Some(binding_id) = plan
        .source_routes
        .keys()
        .find(|binding_id| !validated.contains_key(*binding_id))
    {
        return Err(missing_source(binding_id));
    }
    let specs = plan
        .source_routes
        .keys()
        .map(|binding_id| {
            validated
                .get(binding_id)
                .expect("missing sources were rejected above")
                .progress_spec(binding_id)
        })
        .collect::<Result<Vec<_>>>()?;
    let progress = prepare_stream_job(
        &plan.fingerprint,
        &specs,
        StreamProgressRuntimeConfig::default(),
    )?;
    for prepared in progress.bindings.iter().cloned() {
        validated
            .get_mut(prepared.identity.as_str())
            .expect("prepared bindings were built from validated sources")
            .install_prepared_progress(prepared);
    }
    Ok((validated, progress))
}

fn validate_source(
    plan: &StreamRuntimePlanParts,
    mut named: NamedSourceBinding,
) -> Result<(String, SourceBinding)> {
    let field = format!("sources.{}", named.binding_id);
    validate_runtime_id(&named.binding_id, &field)?;
    let route = plan.source_routes.get(&named.binding_id).ok_or_else(|| {
        CalcFlowError::InvalidArgument {
            field,
            message: "binding does not match a compiled external input".into(),
        }
    })?;
    let edge = &plan.edges[&route.edge_id];
    let capabilities = named.binding.sample_capabilities_once();
    validate_source_budget(
        &named.binding_id,
        capabilities,
        edge.budget,
        &edge.stable_id,
    )?;
    Ok((named.binding_id, named.binding))
}

fn validate_source_budget(
    binding_id: &str,
    capabilities: SourceCapabilities,
    budget: EdgeBudget,
    edge_id: &str,
) -> Result<()> {
    validate_source_capabilities(binding_id, capabilities)?;
    if capabilities.max_batch_rows <= budget.max_rows
        && capabilities.max_batch_bytes <= budget.max_bytes
    {
        return Ok(());
    }
    Err(CalcFlowError::InvalidArgument {
        field: format!("sources.{binding_id}.capabilities"),
        message: format!(
            "maximum batch ({} rows, {} bytes) exceeds edge {:?} budget ({} rows, {} bytes)",
            capabilities.max_batch_rows,
            capabilities.max_batch_bytes,
            edge_id,
            budget.max_rows,
            budget.max_bytes
        ),
    })
}

fn duplicate_source(binding_id: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("sources.{binding_id}"),
        message: "binding is configured more than once".into(),
    }
}

fn missing_source(binding_id: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("sources.{binding_id}"),
        message: format!("missing binding for external input {binding_id}"),
    }
}

fn validate_sinks(
    plan: &StreamRuntimePlanParts,
    sinks: Vec<NamedSinkBinding>,
) -> Result<BTreeMap<String, Vec<ValidatedOrdinarySink>>> {
    let mut validated = BTreeMap::<String, Vec<ValidatedOrdinarySink>>::new();
    for named in sinks {
        let (output_id, sink) = validate_sink(plan, named)?;
        let output_sinks = validated.entry(output_id.clone()).or_default();
        if output_sinks
            .iter()
            .any(|existing| existing.sink_id == sink.sink_id)
        {
            return Err(duplicate_sink(&output_id, &sink.sink_id));
        }
        output_sinks.push(sink);
    }
    if let Some(output_id) = plan
        .sink_routes
        .keys()
        .find(|output_id| validated.get(*output_id).is_none_or(Vec::is_empty))
    {
        return Err(missing_sink(output_id));
    }
    Ok(validated)
}

fn validate_sink(
    plan: &StreamRuntimePlanParts,
    mut named: NamedSinkBinding,
) -> Result<(String, ValidatedOrdinarySink)> {
    validate_runtime_id(&named.output_id, &format!("sinks.{}", named.output_id))?;
    validate_runtime_id(
        &named.sink_id,
        &format!("sinks.{}.{}", named.output_id, named.sink_id),
    )?;
    if !plan.sink_routes.contains_key(&named.output_id) {
        return Err(CalcFlowError::InvalidArgument {
            field: format!("sinks.{}", named.output_id),
            message: "route does not match a compiled external output".into(),
        });
    }
    named.binding.sample_delivery_once();
    Ok((
        named.output_id,
        ValidatedOrdinarySink {
            sink_id: named.sink_id,
            binding: named.binding,
        },
    ))
}

fn duplicate_sink(output_id: &str, sink_id: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("sinks.{output_id}.{sink_id}"),
        message: "sink is configured more than once".into(),
    }
}

fn missing_sink(output_id: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("sinks.{output_id}"),
        message: "external output requires at least one ordinary sink".into(),
    }
}

fn validate_runtime_topology(plan: &StreamRuntimePlanParts) -> Result<()> {
    for node in &plan.nodes {
        validate_node_ingresses(plan, node)?;
        validate_node_outputs(plan, node)?;
    }
    Ok(())
}

fn validate_node_ingresses(plan: &StreamRuntimePlanParts, node: &RuntimeStreamNode) -> Result<()> {
    for ingress in node.input_ports.keys() {
        let edge_id = node
            .ingress_edges
            .get(ingress)
            .ok_or_else(|| CalcFlowError::Internal {
                message: format!(
                    "runtime node {:?} ingress {:?} has no compiled edge",
                    node.node_id, ingress
                ),
            })?;
        if !plan.edges.contains_key(edge_id) {
            return Err(CalcFlowError::Internal {
                message: format!(
                    "runtime node {:?} ingress {:?} names missing edge {:?}",
                    node.node_id, ingress, edge_id
                ),
            });
        }
    }
    Ok(())
}

fn validate_node_outputs(plan: &StreamRuntimePlanParts, node: &RuntimeStreamNode) -> Result<()> {
    for output in node.output_ports.keys() {
        let edge_ids = node
            .output_edges
            .get(output)
            .ok_or_else(|| CalcFlowError::Internal {
                message: format!(
                    "runtime node {:?} output {:?} has no compiled edge",
                    node.node_id, output
                ),
            })?;
        if edge_ids.is_empty()
            || edge_ids
                .iter()
                .any(|edge_id| !plan.edges.contains_key(edge_id))
        {
            return Err(CalcFlowError::Internal {
                message: format!(
                    "runtime node {:?} output {:?} has invalid compiled routes",
                    node.node_id, output
                ),
            });
        }
    }
    Ok(())
}

fn validate_runtime_id(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() || value.contains('\0') {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "must be non-empty, non-whitespace, and contain no NUL".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };

    use async_trait::async_trait;

    use super::{
        ContinuousJobSpec, M2DeliveryMode, M2SinkDelivery, NamedSinkBinding, NamedSourceBinding,
        OrdinarySinkBinding, OrdinaryStreamSink, TransactionalStreamSink, preflight_job,
    };
    use crate::{
        Batch, BatchKind, CalcFlowError, CancellationToken, EdgeBudget, JsonMap, PipelineBuilder,
        Port, Result, RetentionClass, StreamJobContext, StreamOperator, StreamRequirements,
        UdfRegistry, UnionOperator,
        runtime::streaming::{
            context::StreamTaskKind,
            source_task::{Cursor, SourceBinding, SourceCapabilities, SourceEvent, StreamSource},
        },
    };

    fn context(cancellation: CancellationToken) -> StreamJobContext {
        StreamJobContext::new(42, "fingerprint", JsonMap::new(), None, cancellation)
    }

    struct ProbeSink {
        capability_calls: Arc<std::sync::atomic::AtomicUsize>,
        opened: Arc<AtomicBool>,
    }

    struct ProbeTransactionalSink {
        opened: Arc<AtomicBool>,
    }

    #[async_trait]
    impl TransactionalStreamSink for ProbeTransactionalSink {
        async fn open(&mut self) -> Result<()> {
            self.opened.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
            Ok(JsonMap::new())
        }

        async fn commit(&mut self, _epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
            Ok(())
        }

        async fn abort(&mut self, _epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
            Ok(())
        }

        async fn recover(&mut self, _manifest: &crate::CheckpointManifest) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl OrdinaryStreamSink for ProbeSink {
        fn delivery_capability(&self) -> M2SinkDelivery {
            self.capability_calls.fetch_add(1, Ordering::SeqCst);
            M2SinkDelivery::ProcessLocalOrdered
        }

        async fn open(&mut self) -> Result<()> {
            self.opened.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct ClaimingOrdinarySink {
        opened: Arc<AtomicBool>,
    }

    #[async_trait]
    impl OrdinaryStreamSink for ClaimingOrdinarySink {
        fn delivery_capability(&self) -> M2SinkDelivery {
            M2SinkDelivery::Transactional
        }

        async fn open(&mut self) -> Result<()> {
            self.opened.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct CapabilityProbeSource {
        capability_calls: Arc<std::sync::atomic::AtomicUsize>,
        opened: Arc<AtomicBool>,
        capabilities: SourceCapabilities,
    }

    #[async_trait]
    impl StreamSource for CapabilityProbeSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            self.opened.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            Ok(None)
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            self.capability_calls.fetch_add(1, Ordering::SeqCst);
            self.capabilities
        }
    }

    fn union_plan_with(requirements: &StreamRequirements) -> crate::StreamExecutionPlan {
        let union = UnionOperator::new(
            "merge",
            vec![
                Port::new("left", BatchKind::Table, true, None).unwrap(),
                Port::new("right", BatchKind::Table, true, None).unwrap(),
            ],
        )
        .unwrap();
        PipelineBuilder::new("preflight")
            .unwrap()
            .add_node("merge", Box::new(union))
            .unwrap()
            .compile_stream(&UdfRegistry::new().snapshot(), requirements)
            .unwrap()
    }

    fn union_plan() -> crate::StreamExecutionPlan {
        union_plan_with(&StreamRequirements::default())
    }

    fn external_union_plan_with(requirements: &StreamRequirements) -> crate::StreamExecutionPlan {
        let union = UnionOperator::new(
            "merge",
            vec![
                Port::new("left", BatchKind::Table, true, None).unwrap(),
                Port::new("right", BatchKind::Table, true, None).unwrap(),
            ],
        )
        .unwrap();
        PipelineBuilder::new("external-preflight")
            .unwrap()
            .add_node("merge", Box::new(union) as Box<dyn StreamOperator>)
            .unwrap()
            .compile_stream(&UdfRegistry::new().snapshot(), requirements)
            .unwrap()
    }

    fn disjoint_union_plan_with(requirements: &StreamRequirements) -> crate::StreamExecutionPlan {
        let union = |name: &str| {
            UnionOperator::new(
                name,
                vec![
                    Port::new("left", BatchKind::Table, true, None).unwrap(),
                    Port::new("right", BatchKind::Table, true, None).unwrap(),
                ],
            )
            .unwrap()
        };
        PipelineBuilder::new("disjoint-preflight")
            .unwrap()
            .add_node("a", Box::new(union("a")))
            .unwrap()
            .add_node("b", Box::new(union("b")))
            .unwrap()
            .compile_stream(&UdfRegistry::new().snapshot(), requirements)
            .unwrap()
    }

    fn named_source(
        binding_id: &str,
        capabilities: SourceCapabilities,
        opened: &Arc<AtomicBool>,
    ) -> NamedSourceBinding {
        NamedSourceBinding {
            binding_id: binding_id.into(),
            binding: SourceBinding::new(
                Box::new(CapabilityProbeSource {
                    capability_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                    opened: Arc::clone(opened),
                    capabilities,
                }),
                None,
                0,
            )
            .unwrap(),
        }
    }

    fn named_sink(output_id: &str, sink_id: &str, opened: &Arc<AtomicBool>) -> NamedSinkBinding {
        NamedSinkBinding {
            output_id: output_id.into(),
            sink_id: sink_id.into(),
            binding: OrdinarySinkBinding::new(Box::new(ProbeSink {
                capability_calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                opened: Arc::clone(opened),
            })),
        }
    }

    fn preflight_spec(
        plan: crate::StreamExecutionPlan,
        sources: Vec<NamedSourceBinding>,
        sinks: Vec<NamedSinkBinding>,
    ) -> ContinuousJobSpec {
        let fingerprint = plan.fingerprint().to_owned();
        ContinuousJobSpec {
            context: StreamJobContext::new(
                42,
                fingerprint,
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources,
            sinks,
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        }
    }

    fn preflight_error(spec: ContinuousJobSpec) -> CalcFlowError {
        match preflight_job(spec) {
            Ok(_) => panic!("invalid job unexpectedly passed preflight"),
            Err(error) => error,
        }
    }

    #[test]
    fn whole_job_preflight_freezes_capabilities_and_all_boundary_routes_before_open() {
        let left_capability_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let right_capability_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let primary_sink_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let secondary_sink_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let opened = Arc::new(AtomicBool::new(false));
        let source = |binding_id: &str, capability_calls: &Arc<std::sync::atomic::AtomicUsize>| {
            NamedSourceBinding {
                binding_id: binding_id.into(),
                binding: SourceBinding::new(
                    Box::new(CapabilityProbeSource {
                        capability_calls: Arc::clone(capability_calls),
                        opened: Arc::clone(&opened),
                        capabilities: SourceCapabilities {
                            replayable: true,
                            max_batch_rows: 1,
                            max_batch_bytes: 1,
                        },
                    }),
                    None,
                    0,
                )
                .unwrap(),
            }
        };
        let plan = union_plan();
        let spec = ContinuousJobSpec {
            context: StreamJobContext::new(
                42,
                plan.fingerprint(),
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            plan,
            sources: vec![
                source("left", &left_capability_calls),
                source("right", &right_capability_calls),
            ],
            sinks: [
                ("sink-a", &primary_sink_calls),
                ("sink-b", &secondary_sink_calls),
            ]
            .into_iter()
            .map(|(sink_id, capability_calls)| NamedSinkBinding {
                output_id: "output".into(),
                sink_id: sink_id.into(),
                binding: OrdinarySinkBinding::new(Box::new(ProbeSink {
                    capability_calls: Arc::clone(capability_calls),
                    opened: Arc::clone(&opened),
                })),
            })
            .collect(),
            edge_budget: EdgeBudget {
                max_rows: 1,
                max_bytes: 1,
            },
            delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        };

        for calls in [
            &left_capability_calls,
            &right_capability_calls,
            &primary_sink_calls,
            &secondary_sink_calls,
        ] {
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        }

        let validated = preflight_job(spec).unwrap();

        assert_eq!(validated.sources.len(), 2);
        assert_eq!(validated.sinks["output"].len(), 2);
        assert_eq!(validated.plan.source_routes.len(), 2);
        assert_eq!(validated.plan.sink_routes.len(), 1);
        for calls in [
            &left_capability_calls,
            &right_capability_calls,
            &primary_sink_calls,
            &secondary_sink_calls,
        ] {
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
        assert!(!opened.load(Ordering::SeqCst));
    }

    #[test]
    fn failed_whole_job_preflight_opens_no_binding() {
        let failed_opened = Arc::new(AtomicBool::new(false));
        let capabilities = SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1,
        };
        let error = preflight_error(preflight_spec(
            union_plan(),
            vec![
                named_source("left", capabilities, &failed_opened),
                named_source("right", capabilities, &failed_opened),
            ],
            vec![named_sink(
                "unknown",
                "invalid-after-source-preflight",
                &failed_opened,
            )],
        ));
        assert!(matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. } if field == "sinks.unknown"
        ));
        assert!(!failed_opened.load(Ordering::SeqCst));
    }

    #[test]
    fn preflight_rejects_duplicate_missing_unknown_and_oversize_sources_before_open() {
        let capabilities = SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1,
        };
        for (sources, expected_field) in [
            (
                {
                    let opened = Arc::new(AtomicBool::new(false));
                    vec![
                        named_source("left", capabilities, &opened),
                        named_source("left", capabilities, &opened),
                        named_source("right", capabilities, &opened),
                    ]
                },
                "sources.left",
            ),
            (
                {
                    let opened = Arc::new(AtomicBool::new(false));
                    vec![named_source("left", capabilities, &opened)]
                },
                "sources.right",
            ),
            (
                {
                    let opened = Arc::new(AtomicBool::new(false));
                    vec![named_source("unknown", capabilities, &opened)]
                },
                "sources.unknown",
            ),
            (
                {
                    let opened = Arc::new(AtomicBool::new(false));
                    vec![
                        named_source(
                            "left",
                            SourceCapabilities {
                                max_batch_rows: 2,
                                ..capabilities
                            },
                            &opened,
                        ),
                        named_source("right", capabilities, &opened),
                    ]
                },
                "sources.left.capabilities",
            ),
        ] {
            let opened = Arc::new(AtomicBool::new(false));
            let error = preflight_error(preflight_spec(
                union_plan(),
                sources,
                vec![named_sink("output", "sink", &opened)],
            ));
            assert!(matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. } if field == expected_field
            ));
            assert!(!opened.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn preflight_rejects_missing_duplicate_and_unknown_sinks_before_open() {
        let capabilities = SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1,
        };
        for (sinks, expected_field) in [
            (Vec::new(), "sinks.output"),
            (
                {
                    let opened = Arc::new(AtomicBool::new(false));
                    vec![
                        named_sink("output", "sink", &opened),
                        named_sink("output", "sink", &opened),
                    ]
                },
                "sinks.output.sink",
            ),
            (
                {
                    let opened = Arc::new(AtomicBool::new(false));
                    vec![named_sink("unknown", "sink", &opened)]
                },
                "sinks.unknown",
            ),
        ] {
            let opened = Arc::new(AtomicBool::new(false));
            let error = preflight_error(preflight_spec(
                union_plan(),
                vec![
                    named_source("left", capabilities, &opened),
                    named_source("right", capabilities, &opened),
                ],
                sinks,
            ));
            assert!(matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. } if field == expected_field
            ));
            assert!(!opened.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn preflight_rejects_exactly_once_before_open() {
        let opened = Arc::new(AtomicBool::new(false));
        let capabilities = SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1,
        };
        let requirements = StreamRequirements {
            delivery: BTreeMap::from([("output".into(), crate::DeliveryGuarantee::ExactlyOnce)]),
        };

        let error = preflight_error(preflight_spec(
            union_plan_with(&requirements),
            vec![
                named_source("left", capabilities, &opened),
                named_source("right", capabilities, &opened),
            ],
            vec![named_sink("output", "sink", &opened)],
        ));

        assert!(matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "requirements.delivery.output"
        ));
        assert!(!opened.load(Ordering::SeqCst));
    }

    #[test]
    fn preflight_accepts_replayable_sources_and_transactional_sink_for_exactly_once() {
        let opened = Arc::new(AtomicBool::new(false));
        let capabilities = SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1,
        };
        let requirements = StreamRequirements {
            delivery: BTreeMap::from([("output".into(), crate::DeliveryGuarantee::ExactlyOnce)]),
        };
        let plan = union_plan_with(&requirements);
        let spec = preflight_spec(
            plan,
            vec![
                named_source("left", capabilities, &opened),
                named_source("right", capabilities, &opened),
            ],
            vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "transactional".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(ProbeTransactionalSink {
                    opened: Arc::clone(&opened),
                })),
            }],
        );

        let validated = preflight_job(spec).unwrap();

        assert_eq!(validated.sinks["output"].len(), 1);
        assert_eq!(
            validated.sinks["output"][0].binding.delivery(),
            M2SinkDelivery::Transactional
        );
        assert!(!opened.load(Ordering::SeqCst));
    }

    #[test]
    fn exactly_once_preflight_rejects_reachable_unproven_operator_before_open() {
        let opened = Arc::new(AtomicBool::new(false));
        let capabilities = SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1,
        };
        let requirements = StreamRequirements {
            delivery: BTreeMap::from([("output".into(), crate::DeliveryGuarantee::ExactlyOnce)]),
        };
        let error = preflight_error(preflight_spec(
            external_union_plan_with(&requirements),
            vec![
                named_source("left", capabilities, &opened),
                named_source("right", capabilities, &opened),
            ],
            vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "transactional".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(ProbeTransactionalSink {
                    opened: Arc::clone(&opened),
                })),
            }],
        ));

        assert!(matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, ref message }
                if field == "requirements.delivery.output"
                    && message.contains("operator \"merge\"")
                    && message.contains("deterministic checkpoint/restore")
        ));
        assert!(!opened.load(Ordering::SeqCst));
    }

    #[test]
    fn exactly_once_preflight_rejects_bounded_epoch_idempotent_sink_before_open() {
        let opened = Arc::new(AtomicBool::new(false));
        let capabilities = SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1,
        };
        let requirements = StreamRequirements {
            delivery: BTreeMap::from([("output".into(), crate::DeliveryGuarantee::ExactlyOnce)]),
        };
        let error = preflight_error(preflight_spec(
            union_plan_with(&requirements),
            vec![
                named_source("left", capabilities, &opened),
                named_source("right", capabilities, &opened),
            ],
            vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "idempotent".into(),
                binding: OrdinarySinkBinding::new_epoch_idempotent(
                    Box::new(ProbeTransactionalSink {
                        opened: Arc::clone(&opened),
                    }),
                    "epoch-ledger",
                    RetentionClass::Bounded,
                )
                .unwrap(),
            }],
        ));

        assert!(matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, ref message }
                if field == "requirements.delivery.output"
                    && message.contains("sink \"idempotent\"")
                    && message.contains("epoch-ledger")
                    && message.contains("bounded retention")
        ));
        assert!(!opened.load(Ordering::SeqCst));
    }

    #[test]
    fn exactly_once_preflight_rejects_transactional_claim_from_ordinary_sink() {
        let opened = Arc::new(AtomicBool::new(false));
        let capabilities = SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1,
        };
        let requirements = StreamRequirements {
            delivery: BTreeMap::from([("output".into(), crate::DeliveryGuarantee::ExactlyOnce)]),
        };
        let error = preflight_error(preflight_spec(
            union_plan_with(&requirements),
            vec![
                named_source("left", capabilities, &opened),
                named_source("right", capabilities, &opened),
            ],
            vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "claiming".into(),
                binding: OrdinarySinkBinding::new(Box::new(ClaimingOrdinarySink {
                    opened: Arc::clone(&opened),
                })),
            }],
        ));

        assert!(matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, ref message }
                if field == "requirements.delivery.output"
                    && message.contains("sink \"claiming\"")
                    && message.contains("not transactional")
        ));
        assert!(!opened.load(Ordering::SeqCst));
    }

    #[test]
    fn exactly_once_preflight_accepts_unbounded_epoch_idempotent_sink_evidence() {
        let opened = Arc::new(AtomicBool::new(false));
        let capabilities = SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1,
        };
        let requirements = StreamRequirements {
            delivery: BTreeMap::from([("output".into(), crate::DeliveryGuarantee::ExactlyOnce)]),
        };
        let validated = preflight_job(preflight_spec(
            union_plan_with(&requirements),
            vec![
                named_source("left", capabilities, &opened),
                named_source("right", capabilities, &opened),
            ],
            vec![NamedSinkBinding {
                output_id: "output".into(),
                sink_id: "idempotent".into(),
                binding: OrdinarySinkBinding::new_epoch_idempotent(
                    Box::new(ProbeTransactionalSink {
                        opened: Arc::clone(&opened),
                    }),
                    "epoch-ledger",
                    RetentionClass::Unbounded,
                )
                .unwrap(),
            }],
        ))
        .unwrap();

        assert_eq!(
            validated.sinks["output"][0].binding.delivery(),
            M2SinkDelivery::EpochIdempotent {
                mechanism: "epoch-ledger".into(),
                retention: RetentionClass::Unbounded,
            }
        );
        assert!(!opened.load(Ordering::SeqCst));
    }

    #[test]
    fn exactly_once_proof_ignores_incompatible_components_on_disjoint_output() {
        let opened = Arc::new(AtomicBool::new(false));
        let replayable = SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1,
        };
        let non_replayable = SourceCapabilities {
            replayable: false,
            ..replayable
        };
        let requirements = StreamRequirements {
            delivery: BTreeMap::from([("a.output".into(), crate::DeliveryGuarantee::ExactlyOnce)]),
        };
        let sources = [
            ("a.left", replayable),
            ("a.right", replayable),
            ("b.left", non_replayable),
            ("b.right", non_replayable),
        ]
        .into_iter()
        .map(|(id, capabilities)| named_source(id, capabilities, &opened))
        .collect();
        let sinks = vec![
            NamedSinkBinding {
                output_id: "a.output".into(),
                sink_id: "a-tx".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(ProbeTransactionalSink {
                    opened: Arc::clone(&opened),
                })),
            },
            named_sink("b.output", "b-ordinary", &opened),
        ];

        let validated = preflight_job(preflight_spec(
            disjoint_union_plan_with(&requirements),
            sources,
            sinks,
        ))
        .unwrap();

        assert_eq!(validated.sources.len(), 4);
        assert!(!opened.load(Ordering::SeqCst));
    }

    #[test]
    fn job_context_derives_validated_source_node_and_sink_scopes() {
        let context = context(CancellationToken::new());

        let source = context.for_source("input").unwrap();
        let node = context.for_node("normalize").unwrap();
        let sink = context.for_sink("output").unwrap();

        assert_eq!(source.kind(), StreamTaskKind::Source);
        assert_eq!(node.kind(), StreamTaskKind::Node);
        assert_eq!(sink.kind(), StreamTaskKind::Sink);
        assert_eq!(source.scope_id(), "input");
        assert_eq!(source.job().job_id(), context.job_id());
        assert!(context.for_source("  ").is_err());
        assert!(matches!(
            context.for_sink("  "),
            Err(CalcFlowError::InvalidArgument { ref field, .. })
                if field == "sink.output_id"
        ));
    }
}
