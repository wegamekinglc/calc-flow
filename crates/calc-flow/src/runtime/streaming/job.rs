use std::collections::BTreeMap;

use async_trait::async_trait;

use super::{StreamJobContext, source_task::SourceBinding};
use crate::{
    Batch, CalcFlowError, DeliveryGuarantee, EdgeBudget, Result, StreamExecutionPlan,
    pipeline::StreamRuntimePlanParts,
};

pub(crate) struct NamedSourceBinding {
    pub(crate) binding_id: String,
    pub(crate) binding: SourceBinding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum M2SinkDelivery {
    ProcessLocalOrdered,
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

pub(crate) struct OrdinarySinkBinding {
    pub(crate) sink: Box<dyn OrdinaryStreamSink>,
    delivery: Option<M2SinkDelivery>,
}

impl OrdinarySinkBinding {
    pub(crate) fn new(sink: Box<dyn OrdinaryStreamSink>) -> Self {
        Self {
            sink,
            delivery: None,
        }
    }

    fn sample_delivery_once(&mut self) -> M2SinkDelivery {
        *self
            .delivery
            .get_or_insert_with(|| self.sink.delivery_capability())
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
    pub(crate) delivery_mode: M2DeliveryMode,
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
    if context.fingerprint() != plan.fingerprint {
        return Err(CalcFlowError::InvalidArgument {
            field: "context.fingerprint".into(),
            message: "must match the consumed stream plan fingerprint".into(),
        });
    }
    validate_runtime_topology(&plan)?;
    for (output_id, guarantee) in &plan.requirements.delivery {
        if *guarantee == DeliveryGuarantee::ExactlyOnce {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("requirements.delivery.{output_id}"),
                message: "exactly-once delivery requires aligned checkpoints and is unavailable before M5"
                    .into(),
            });
        }
    }

    let validated_sources = validate_sources(&plan, sources)?;
    let validated_sinks = validate_sinks(&plan, sinks)?;

    Ok(ValidatedContinuousJob {
        context,
        plan,
        sources: validated_sources,
        sinks: validated_sinks,
        delivery_mode,
    })
}

fn validate_sources(
    plan: &StreamRuntimePlanParts,
    sources: Vec<NamedSourceBinding>,
) -> Result<BTreeMap<String, SourceBinding>> {
    let mut validated = BTreeMap::new();
    for mut named in sources {
        validate_runtime_id(&named.binding_id, &format!("sources.{}", named.binding_id))?;
        if !plan.source_routes.contains_key(&named.binding_id) {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sources.{}", named.binding_id),
                message: "binding does not match a compiled external input".into(),
            });
        }
        if validated.contains_key(&named.binding_id) {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sources.{}", named.binding_id),
                message: "binding is configured more than once".into(),
            });
        }
        let route = &plan.source_routes[&named.binding_id];
        let edge = &plan.edges[&route.edge_id];
        let capabilities = named.binding.sample_capabilities_once();
        if capabilities.max_batch_rows == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sources.{}.capabilities.max_batch_rows", named.binding_id),
                message: "must be greater than zero".into(),
            });
        }
        if capabilities.max_batch_bytes == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sources.{}.capabilities.max_batch_bytes", named.binding_id),
                message: "must be greater than zero".into(),
            });
        }
        if capabilities.max_batch_rows > edge.budget.max_rows
            || capabilities.max_batch_bytes > edge.budget.max_bytes
        {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sources.{}.capabilities", named.binding_id),
                message: format!(
                    "maximum batch ({} rows, {} bytes) exceeds edge {:?} budget ({} rows, {} bytes)",
                    capabilities.max_batch_rows,
                    capabilities.max_batch_bytes,
                    edge.stable_id,
                    edge.budget.max_rows,
                    edge.budget.max_bytes
                ),
            });
        }
        validated.insert(named.binding_id, named.binding);
    }
    for binding_id in plan.source_routes.keys() {
        if !validated.contains_key(binding_id) {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sources.{binding_id}"),
                message: format!("missing binding for external input {binding_id}"),
            });
        }
    }
    Ok(validated)
}

fn validate_sinks(
    plan: &StreamRuntimePlanParts,
    sinks: Vec<NamedSinkBinding>,
) -> Result<BTreeMap<String, Vec<ValidatedOrdinarySink>>> {
    let mut validated = BTreeMap::<String, Vec<ValidatedOrdinarySink>>::new();
    for mut named in sinks {
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
        let output_sinks = validated.entry(named.output_id.clone()).or_default();
        if output_sinks
            .iter()
            .any(|sink| sink.sink_id == named.sink_id)
        {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sinks.{}.{}", named.output_id, named.sink_id),
                message: "sink is configured more than once".into(),
            });
        }
        if named.binding.sample_delivery_once() != M2SinkDelivery::ProcessLocalOrdered {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sinks.{}.{}.delivery", named.output_id, named.sink_id),
                message: "ordinary M2 sinks require process-local ordered delivery".into(),
            });
        }
        output_sinks.push(ValidatedOrdinarySink {
            sink_id: named.sink_id,
            binding: named.binding,
        });
    }
    for output_id in plan.sink_routes.keys() {
        if validated.get(output_id).is_none_or(Vec::is_empty) {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sinks.{output_id}"),
                message: "external output requires at least one ordinary sink".into(),
            });
        }
    }
    Ok(validated)
}

fn validate_runtime_topology(plan: &StreamRuntimePlanParts) -> Result<()> {
    for node in &plan.nodes {
        for ingress in node.input_ports.keys() {
            let edge_id =
                node.ingress_edges
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
        for output in node.output_ports.keys() {
            let edge_ids =
                node.output_edges
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
        OrdinarySinkBinding, OrdinaryStreamSink, preflight_job,
    };
    use crate::{
        Batch, BatchKind, CalcFlowError, CancellationToken, EdgeBudget, JsonMap, PipelineBuilder,
        Port, Result, StreamJobContext, StreamRequirements, UdfRegistry, UnionOperator,
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
    }
}
