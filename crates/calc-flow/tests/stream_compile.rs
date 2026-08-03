//! RED (M1.1): `compile_stream`, dual digests, and stream graph validation.
//!
//! Every test in this file fails to compile until the v3 split exists:
//! `compile_batch`/`compile_stream`, `BatchExecutionPlan`/`StreamExecutionPlan`,
//! `StreamRequirements`/`DeliveryGuarantee`, `StreamRuntimeConfig`/`EdgeBudget`,
//! `UnionOperator`, and the `OperatorMetadata` + `BatchOperator`/`StreamOperator`
//! trait split (plan task M1.1, API note A1/A2). The expected RED reason is an
//! unresolved import of these names from `calc_flow`.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchExecutionPlan, BatchKind, BatchOperator, BatchOperatorContext, CalcFlowError,
    DeliveryGuarantee, Edge, EdgeBudget, EventTime, ExpressionOperator, JsonMap, OperatorMetadata,
    PipelineBuilder, Port, PortEndpoint, Result, SqlOperator, StreamCollector, StreamOperator,
    StreamOperatorContext, StreamRequirements, StreamRuntimeConfig, UdfKind, UdfReference,
    UdfRegistry, UnionOperator,
};
use datafusion::{
    common::ScalarValue,
    logical_expr::{ColumnarValue, ScalarUDF, Volatility, create_udf},
};

fn endpoint(node_id: &str, port: &str) -> PortEndpoint {
    PortEndpoint::new(node_id, port).unwrap()
}

fn edge(source: (&str, &str), target: (&str, &str)) -> Edge {
    Edge::new(endpoint(source.0, source.1), endpoint(target.0, target.1))
}

fn expression(name: &str) -> Box<ExpressionOperator> {
    Box::new(ExpressionOperator::new(name, "total = a + b", Vec::new(), None, Vec::new()).unwrap())
}

fn udfs() -> calc_flow::UdfRegistrySnapshot {
    UdfRegistry::new().snapshot()
}

/// A batch-only external operator: it implements `BatchOperator` but not
/// `StreamOperator`, so `compile_stream` must reject it (plan 2.2 matrix).
struct BatchOnlyOperator {
    input_ports: [Port; 1],
    output_ports: [Port; 1],
}

impl BatchOnlyOperator {
    fn boxed(name: &str) -> Box<dyn BatchOperator> {
        let _ = name;
        Box::new(Self {
            input_ports: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            output_ports: [Port::new("output", BatchKind::Table, true, None).unwrap()],
        })
    }
}

impl OperatorMetadata for BatchOnlyOperator {
    fn name(&self) -> &'static str {
        "batch_only"
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }
}

#[async_trait]
impl BatchOperator for BatchOnlyOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        Ok(BTreeMap::from([("output".into(), inputs["input"].clone())]))
    }
}

/// A stream-only external operator: `compile_batch` must reject it.
struct StreamOnlyOperator {
    input_ports: [Port; 1],
    output_ports: [Port; 1],
}

impl StreamOnlyOperator {
    fn boxed() -> Box<dyn StreamOperator> {
        Box::new(Self {
            input_ports: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            output_ports: [Port::new("output", BatchKind::Table, true, None).unwrap()],
        })
    }
}

impl OperatorMetadata for StreamOnlyOperator {
    fn name(&self) -> &'static str {
        "stream_only"
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }
}

#[async_trait]
impl StreamOperator for StreamOnlyOperator {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        _context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        assert_eq!(ingress, "input");
        output.emit("output", batch).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}

fn unary_chain() -> PipelineBuilder {
    PipelineBuilder::new("chain")
        .unwrap()
        .add_node("first", expression("first"))
        .unwrap()
        .add_node("second", expression("second"))
        .unwrap()
        .connect(edge(("first", "output"), ("second", "input")))
        .unwrap()
}

#[test]
fn compile_stream_accepts_a_unary_chain() {
    let plan = unary_chain()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();

    assert_eq!(plan.name(), "chain");
    assert_eq!(plan.topological_order(), ["first", "second"]);
    assert_eq!(plan.source_binding_ids(), ["input"]);
    assert_eq!(plan.sink_binding_ids(), ["output"]);
    assert_eq!(plan.edge_ids(), ["first.output->second.input"]);
}

#[test]
fn compile_stream_accepts_fan_out_and_independent_branches() {
    let plan = PipelineBuilder::new("branches")
        .unwrap()
        .add_node("root", expression("root"))
        .unwrap()
        .add_node("left", expression("left"))
        .unwrap()
        .add_node("right", expression("right"))
        .unwrap()
        .add_node("detached", expression("detached"))
        .unwrap()
        .connect(edge(("root", "output"), ("left", "input")))
        .unwrap()
        .connect(edge(("root", "output"), ("right", "input")))
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();

    assert_eq!(plan.source_binding_ids(), ["detached.input", "root.input"]);
    assert_eq!(
        plan.sink_binding_ids(),
        ["detached.output", "left.output", "right.output"]
    );
    assert_eq!(
        plan.edge_ids(),
        ["root.output->left.input", "root.output->right.input"]
    );
}

#[test]
fn compile_stream_accepts_a_same_schema_union() {
    let union = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap();
    let plan = PipelineBuilder::new("union")
        .unwrap()
        .add_node("first", expression("first"))
        .unwrap()
        .add_node("second", expression("second"))
        .unwrap()
        .add_node("merge", Box::new(union))
        .unwrap()
        .connect(edge(("first", "output"), ("merge", "left")))
        .unwrap()
        .connect(edge(("second", "output"), ("merge", "right")))
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();

    assert_eq!(plan.source_binding_ids(), ["first.input", "second.input"]);
    assert_eq!(plan.sink_binding_ids(), ["output"]);
}

#[test]
fn compile_stream_rejects_multi_alias_sql_naming_the_node() {
    let error = PipelineBuilder::new("multi sql")
        .unwrap()
        .add_node(
            "join",
            Box::new(
                SqlOperator::new(
                    "join",
                    "SELECT l.a FROM left_input l JOIN right_input r ON l.a = r.a",
                    vec!["left_input".into(), "right_input".into()],
                    Vec::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap_err();

    assert!(matches!(error, CalcFlowError::Compile { .. }));
    assert!(error.to_string().contains("join"));
}

#[test]
fn compile_stream_accepts_single_alias_sql() {
    let plan = PipelineBuilder::new("single sql")
        .unwrap()
        .add_node(
            "project",
            Box::new(
                SqlOperator::new(
                    "project",
                    "SELECT a + b AS total FROM events",
                    vec!["events".into()],
                    Vec::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();

    assert_eq!(plan.source_binding_ids(), ["events"]);
}

#[test]
fn compile_stream_rejects_a_batch_only_operator_naming_the_node() {
    let error = PipelineBuilder::new("batch only")
        .unwrap()
        .add_node("legacy", BatchOnlyOperator::boxed("legacy"))
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap_err();

    assert!(matches!(error, CalcFlowError::Compile { .. }));
    assert!(error.to_string().contains("legacy"));
}

#[test]
fn compile_batch_rejects_a_stream_only_operator_naming_the_node() {
    let error = PipelineBuilder::new("stream only")
        .unwrap()
        .add_node("streaming", StreamOnlyOperator::boxed())
        .unwrap()
        .compile_batch(&udfs())
        .unwrap_err();

    assert!(matches!(error, CalcFlowError::Compile { .. }));
    assert!(error.to_string().contains("streaming"));
}

#[test]
fn compile_batch_rejects_a_union_node_naming_the_node() {
    let union = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap();
    let error = PipelineBuilder::new("batch union")
        .unwrap()
        .add_node("merge", Box::new(union))
        .unwrap()
        .compile_batch(&udfs())
        .unwrap_err();

    assert!(matches!(error, CalcFlowError::Compile { .. }));
    assert!(error.to_string().contains("merge"));
}

#[test]
fn compile_batch_keeps_accepting_the_v2_batch_graphs() {
    let plan: BatchExecutionPlan = unary_chain().compile_batch(&udfs()).unwrap();

    assert_eq!(plan.name(), "chain");
    assert_eq!(plan.topological_order(), ["first", "second"]);
    assert!(plan.requires_datafusion());
}

#[test]
fn semantic_fingerprint_is_deterministic_across_insertion_order() {
    let forward = unary_chain()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();
    let reversed = PipelineBuilder::new("chain")
        .unwrap()
        .add_node("second", expression("second"))
        .unwrap()
        .add_node("first", expression("first"))
        .unwrap()
        .connect(edge(("first", "output"), ("second", "input")))
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();

    assert_eq!(forward.fingerprint(), reversed.fingerprint());
    assert_eq!(forward.edge_ids(), reversed.edge_ids());

    let batch_a = unary_chain().compile_batch(&udfs()).unwrap();
    let batch_b = unary_chain().compile_batch(&udfs()).unwrap();
    assert_eq!(batch_a.fingerprint(), batch_b.fingerprint());
}

#[test]
fn batch_and_stream_fingerprints_differ_for_the_same_topology() {
    let batch = unary_chain().compile_batch(&udfs()).unwrap();
    let stream = unary_chain()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();

    assert_ne!(batch.fingerprint(), stream.fingerprint());
}

#[test]
fn channel_capacity_and_checkpoint_interval_never_touch_the_semantic_fingerprint() {
    let plan = unary_chain()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();
    let before = plan.fingerprint().to_owned();

    let tuned = StreamRuntimeConfig {
        edge_budget: EdgeBudget::new(1, 1_048_576).unwrap(),
        checkpoint_interval: Duration::from_secs(30),
        ..Default::default()
    };

    // The semantic fingerprint is compiled into the plan; runtime tunables only
    // reach the separate runtime-config hash (spec NFR-5).
    assert_eq!(plan.fingerprint(), before);
    let default_hash = plan
        .runtime_config_hash(&StreamRuntimeConfig::default())
        .unwrap();
    let tuned_hash = plan.runtime_config_hash(&tuned).unwrap();
    assert_ne!(default_hash, tuned_hash);
}

#[test]
fn runtime_config_hash_is_deterministic_and_tracks_every_tunable() {
    let plan = unary_chain()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();
    let baseline = plan
        .runtime_config_hash(&StreamRuntimeConfig::default())
        .unwrap();
    assert_eq!(
        baseline,
        plan.runtime_config_hash(&StreamRuntimeConfig::default())
            .unwrap()
    );

    let rows = StreamRuntimeConfig {
        edge_budget: EdgeBudget::new(5_000, 64 << 20).unwrap(),
        ..Default::default()
    };
    assert_ne!(baseline, plan.runtime_config_hash(&rows).unwrap());

    let bytes = StreamRuntimeConfig {
        edge_budget: EdgeBudget::new(10_000, 32 << 20).unwrap(),
        ..Default::default()
    };
    assert_ne!(baseline, plan.runtime_config_hash(&bytes).unwrap());

    let interval = StreamRuntimeConfig {
        checkpoint_interval: Duration::from_secs(120),
        ..Default::default()
    };
    assert_ne!(baseline, plan.runtime_config_hash(&interval).unwrap());

    let timeout = StreamRuntimeConfig {
        checkpoint_timeout: Duration::from_secs(300),
        ..Default::default()
    };
    assert_ne!(baseline, plan.runtime_config_hash(&timeout).unwrap());

    let retained = StreamRuntimeConfig {
        retained_epochs: 5,
        ..Default::default()
    };
    assert_ne!(baseline, plan.runtime_config_hash(&retained).unwrap());
}

#[test]
fn runtime_config_hash_rejects_sub_microsecond_durations_naming_the_field() {
    let plan = unary_chain()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();
    let config = StreamRuntimeConfig {
        checkpoint_interval: Duration::from_nanos(1_500),
        ..Default::default()
    };

    let error = plan.runtime_config_hash(&config).unwrap_err();

    assert!(error.to_string().contains("checkpoint_interval"));
}

#[test]
fn edge_budget_rejects_zero_capacities_naming_the_field() {
    assert!(
        EdgeBudget::new(0, 1_024)
            .unwrap_err()
            .to_string()
            .contains("max_rows")
    );
    assert!(
        EdgeBudget::new(1_024, 0)
            .unwrap_err()
            .to_string()
            .contains("max_bytes")
    );
}

#[test]
fn compile_stream_rejects_delivery_requests_for_unknown_outputs() {
    let mut requirements = StreamRequirements::default();
    requirements
        .delivery
        .insert("missing".into(), DeliveryGuarantee::ExactlyOnce);

    let error = unary_chain()
        .compile_stream(&udfs(), &requirements)
        .unwrap_err();

    assert!(matches!(error, CalcFlowError::Compile { .. }));
    assert!(error.to_string().contains("missing"));
}

#[test]
fn compile_stream_records_the_delivery_requirements() {
    let mut requirements = StreamRequirements::default();
    requirements
        .delivery
        .insert("output".into(), DeliveryGuarantee::ExactlyOnce);

    let plan = unary_chain()
        .compile_stream(&udfs(), &requirements)
        .unwrap();

    assert_eq!(
        plan.requirements().delivery.get("output"),
        Some(&DeliveryGuarantee::ExactlyOnce)
    );
}

fn volatile_udf(name: &str) -> Arc<ScalarUDF> {
    Arc::new(create_udf(
        name,
        vec![],
        datafusion::arrow::datatypes::DataType::Int64,
        Volatility::Volatile,
        Arc::new(|_| Ok(ColumnarValue::Scalar(ScalarValue::Int64(Some(1))))),
    ))
}

#[test]
fn exactly_once_plan_rejects_a_volatile_udf_naming_the_node() {
    let reference =
        UdfReference::new("builtins", "unstable", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(reference.clone(), volatile_udf("unstable"), 0)
        .unwrap();
    let builder = PipelineBuilder::new("volatile")
        .unwrap()
        .add_node(
            "calc",
            Box::new(
                ExpressionOperator::new("calc", "total = a + b", Vec::new(), None, vec![reference])
                    .unwrap(),
            ),
        )
        .unwrap();

    let mut requirements = StreamRequirements::default();
    requirements
        .delivery
        .insert("output".into(), DeliveryGuarantee::ExactlyOnce);
    let error = builder
        .compile_stream(&registry.snapshot(), &requirements)
        .unwrap_err();

    assert!(matches!(error, CalcFlowError::Compile { .. }));
    assert!(error.to_string().contains("calc"));
}

#[test]
fn at_least_once_plan_accepts_a_volatile_udf() {
    let reference =
        UdfReference::new("builtins", "unstable", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(reference.clone(), volatile_udf("unstable"), 0)
        .unwrap();

    PipelineBuilder::new("volatile")
        .unwrap()
        .add_node(
            "calc",
            Box::new(
                ExpressionOperator::new("calc", "total = a + b", Vec::new(), None, vec![reference])
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(&registry.snapshot(), &StreamRequirements::default())
        .unwrap();
}

#[test]
fn union_operator_requires_at_least_two_inputs() {
    let error = UnionOperator::new(
        "merge",
        vec![Port::new("only", BatchKind::Table, true, None).unwrap()],
    )
    .unwrap_err();

    assert!(matches!(error, CalcFlowError::InvalidArgument { .. }));
}

#[test]
fn union_operator_requires_uniform_kinds_and_schemas() {
    use datafusion::arrow::datatypes::{DataType, Field};

    let kind_mismatch = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Array, true, None).unwrap(),
        ],
    )
    .unwrap_err();
    assert!(matches!(
        kind_mismatch,
        CalcFlowError::InvalidArgument { .. }
    ));

    let schema_a = vec![Field::new("value", DataType::Int64, false)];
    let schema_b = vec![Field::new("other", DataType::Int64, false)];
    let schema_mismatch = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, Some(schema_a.clone())).unwrap(),
            Port::new("right", BatchKind::Table, true, Some(schema_b)).unwrap(),
        ],
    )
    .unwrap_err();
    assert!(matches!(
        schema_mismatch,
        CalcFlowError::InvalidArgument { .. }
    ));

    let mixed_presence = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, Some(schema_a)).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap_err();
    assert!(matches!(
        mixed_presence,
        CalcFlowError::InvalidArgument { .. }
    ));
}

#[test]
fn stream_plan_compiles_stable_edge_ids_and_binding_slots() {
    let plan = PipelineBuilder::new("slots")
        .unwrap()
        .add_node("root", expression("root"))
        .unwrap()
        .add_node("left", expression("left"))
        .unwrap()
        .add_node("right", expression("right"))
        .unwrap()
        .connect(edge(("root", "output"), ("right", "input")))
        .unwrap()
        .connect(edge(("root", "output"), ("left", "input")))
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();

    assert_eq!(
        plan.edge_ids(),
        ["root.output->left.input", "root.output->right.input"]
    );
    assert_eq!(plan.source_binding_ids(), ["input"]);
    assert_eq!(plan.sink_binding_ids(), ["left.output", "right.output"]);
    assert_eq!(plan.name(), "slots");
    assert!(plan.requires_datafusion());
}
