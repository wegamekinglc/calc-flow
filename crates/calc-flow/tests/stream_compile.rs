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

/// A configurable-port operator for crafting node IDs whose formatted stable
/// edge IDs collide. It implements both execution traits so the batch and
/// stream compilers validate the same graph shape.
struct RenamableOperator {
    name: String,
    input_ports: Vec<Port>,
    output_ports: Vec<Port>,
}

impl RenamableOperator {
    fn stream(name: &str, inputs: &[&str], outputs: &[&str]) -> Box<dyn StreamOperator> {
        Box::new(Self::new(name, inputs, outputs))
    }

    fn batch(name: &str, inputs: &[&str], outputs: &[&str]) -> Box<dyn BatchOperator> {
        Box::new(Self::new(name, inputs, outputs))
    }

    fn new(name: &str, inputs: &[&str], outputs: &[&str]) -> Self {
        let ports = |names: &[&str]| {
            names
                .iter()
                .map(|name| Port::new(name, BatchKind::Table, true, None).unwrap())
                .collect()
        };
        Self {
            name: name.into(),
            input_ports: ports(inputs),
            output_ports: ports(outputs),
        }
    }
}

impl OperatorMetadata for RenamableOperator {
    fn name(&self) -> &str {
        &self.name
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
impl BatchOperator for RenamableOperator {
    async fn process(
        &mut self,
        _inputs: &BTreeMap<String, Batch>,
        _context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        Ok(BTreeMap::new())
    }
}

#[async_trait]
impl StreamOperator for RenamableOperator {
    async fn process_data(
        &mut self,
        _ingress: &str,
        batch: Batch,
        _context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let port = self.output_ports[0].name().to_owned();
        output.emit(&port, batch).await
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

/// The crafted graph: `a.b -> c.out->d.in` and `a.b->c.out -> d.in` both
/// format to the stable edge ID `a.b->c.out->d.in`.
fn colliding_stable_id_graph<O>(node: impl Fn(&str, &[&str], &[&str]) -> O) -> PipelineBuilder
where
    O: Into<calc_flow::NodeOperator>,
{
    PipelineBuilder::new("colliding")
        .unwrap()
        .add_node("a", node("a", &["in"], &["b"]))
        .unwrap()
        .add_node("c.out->d", node("c.out->d", &["in"], &["out"]))
        .unwrap()
        .add_node("a.b->c", node("a.b->c", &["in"], &["out"]))
        .unwrap()
        .add_node("d", node("d", &["in"], &["out"]))
        .unwrap()
        .connect(edge(("a", "b"), ("c.out->d", "in")))
        .unwrap()
        .connect(edge(("a.b->c", "out"), ("d", "in")))
        .unwrap()
}

#[test]
fn compile_rejects_colliding_stable_edge_ids_naming_both_edges() {
    let stream_error = colliding_stable_id_graph(RenamableOperator::stream)
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap_err();
    assert!(matches!(stream_error, CalcFlowError::Compile { .. }));
    let message = stream_error.to_string();
    assert!(message.contains("a.b->c.out->d.in"));
    assert!(message.contains("c.out->d"));
    assert!(message.contains("a.b->c"));

    let batch_error = colliding_stable_id_graph(RenamableOperator::batch)
        .compile_batch(&udfs())
        .unwrap_err();
    assert!(matches!(batch_error, CalcFlowError::Compile { .. }));
    assert!(batch_error.to_string().contains("a.b->c.out->d.in"));
}

#[test]
fn compile_accepts_separator_node_ids_while_stable_ids_stay_unique() {
    let plan = PipelineBuilder::new("separators")
        .unwrap()
        .add_node("a.b", RenamableOperator::stream("a.b", &["in"], &["out"]))
        .unwrap()
        .add_node("c->d", RenamableOperator::stream("c->d", &["in"], &["out"]))
        .unwrap()
        .connect(edge(("a.b", "out"), ("c->d", "in")))
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();

    assert_eq!(plan.edge_ids(), ["a.b.out->c->d.in"]);
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
fn union_operator_rejects_duplicate_input_port_names() {
    let error = UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("left", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap_err();

    assert!(matches!(error, CalcFlowError::InvalidArgument { .. }));
    assert!(error.to_string().contains("unique"));

    UnionOperator::new(
        "merge",
        vec![
            Port::new("left", BatchKind::Table, true, None).unwrap(),
            Port::new("right", BatchKind::Table, true, None).unwrap(),
        ],
    )
    .unwrap();
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

#[test]
fn compile_stream_accepts_a_custom_stream_operator_and_exposes_plan_accessors() {
    let plan = PipelineBuilder::new("custom")
        .unwrap()
        .add_node("custom", StreamOnlyOperator::boxed())
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();

    assert_eq!(plan.topological_order(), ["custom"]);
    assert_eq!(plan.source_binding_ids(), ["input"]);
    assert_eq!(plan.sink_binding_ids(), ["output"]);
    assert!(plan.external_inputs().contains_key("input"));
    assert!(plan.external_outputs().contains_key("output"));
    assert!(!plan.requires_datafusion());
    assert!(plan.datafusion_config().is_none());

    let debug = format!("{plan:?}");
    assert!(debug.contains("custom"));
    assert!(debug.contains(plan.fingerprint()));
}

#[test]
fn stream_plan_accessors_report_table_resources_when_datafusion_is_required() {
    let plan = unary_chain()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();

    assert!(plan.requires_datafusion());
    assert!(plan.datafusion_config().is_some());
    assert!(plan.external_inputs().contains_key("input"));
    assert!(plan.external_outputs().contains_key("output"));
}

#[test]
fn runtime_config_rejects_durations_exceeding_the_microsecond_range() {
    let config = StreamRuntimeConfig {
        checkpoint_interval: Duration::from_secs(u64::MAX),
        ..Default::default()
    };

    let error = config.validate().unwrap_err();
    assert!(error.to_string().contains("checkpoint_interval"));
    assert!(error.to_string().contains("microsecond range"));

    let plan = unary_chain()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();
    let error = plan.runtime_config_hash(&config).unwrap_err();
    assert!(error.to_string().contains("checkpoint_interval"));
}

fn late_rolling(policy: calc_flow::LatePolicySpec) -> calc_flow::RollingOperator {
    use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("key", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("x", DataType::Float64, true),
    ]));
    let spec = calc_flow::RollingSpec {
        configuration_version: 1,
        state_layout_version: 1,
        numerical_profile: calc_flow::RollingNumericalProfile::default(),
        partition_by: vec!["key".into()],
        event_time: "ts".into(),
        sequence_by: vec!["seq".into()],
        outputs: vec![calc_flow::RollingOutputSpec::Lag {
            primitive_version: 1,
            input: "x".into(),
            output: "lag".into(),
            periods: 1,
        }],
        allowed_lateness_micros: 0,
        late_policy: policy,
        value_policy: calc_flow::RollingValuePolicy::StatefulNumericV1,
    };
    calc_flow::RollingOperator::new("roll", schema, spec).unwrap()
}

fn late_policy() -> calc_flow::LatePolicySpec {
    serde_json::from_value(serde_json::json!({
        "kind": "side_output", "metrics_version": 1, "schema_version": 1
    }))
    .unwrap()
}

fn late_builder() -> PipelineBuilder {
    PipelineBuilder::new("late-contract")
        .unwrap()
        .add_node("roll", late_rolling(late_policy()))
        .unwrap()
}

#[test]
fn test_side_output_native_batch_compile_rejects_new_policy() {
    let error = late_builder()
        .compile_batch(&udfs())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("roll") && error.contains("unsupported_mode"),
        "{error}"
    );
}

fn late_path_builder(path: &str) -> (PipelineBuilder, &'static str) {
    let operator = late_rolling(late_policy());
    let input = operator.output_ports()[1].schema().unwrap().clone();
    let output = operator.input_ports()[0].schema().unwrap().clone();
    match path {
        "direct" => (late_builder(), "roll"),
        "expression" => (
            late_builder()
                .add_node(
                    "project",
                    Box::new(
                        ExpressionOperator::new(
                            "project",
                            "",
                            vec!["ts".into(), "key".into(), "seq".into(), "x".into()],
                            None,
                            vec![],
                        )
                        .unwrap()
                        .with_ports(
                            late_table_port("input", &input),
                            late_table_port("output", &output),
                        )
                        .unwrap(),
                    ),
                )
                .unwrap()
                .connect(edge(("roll", "late"), ("project", "input")))
                .unwrap(),
            "project",
        ),
        "sql" => (
            late_builder()
                .add_node(
                    "project",
                    Box::new(
                        SqlOperator::new(
                            "project",
                            "SELECT ts, key, seq, x FROM events",
                            vec!["events".into()],
                            vec![],
                        )
                        .unwrap()
                        .with_ports(
                            vec![late_table_port("events", &input)],
                            late_table_port("output", &output),
                        )
                        .unwrap(),
                    ),
                )
                .unwrap()
                .connect(edge(("roll", "late"), ("project", "events")))
                .unwrap(),
            "project",
        ),
        _ => unreachable!(),
    }
}

fn late_successor(
    successor: &str,
    input: datafusion::arrow::datatypes::SchemaRef,
) -> (calc_flow::NodeOperator, &'static str) {
    match successor {
        "rolling" => {
            let spec = late_rolling(calc_flow::LatePolicySpec::Drop { metrics_version: 1 })
                .spec()
                .clone();
            (
                calc_flow::RollingOperator::new("next", input, spec)
                    .unwrap()
                    .into(),
                "input",
            )
        }
        "cross_section" => {
            let spec = serde_json::from_value(serde_json::json!({
            "configuration_version": 1, "state_layout_version": 1,
            "event_time": "ts", "entity_by": ["key"], "partition_by": [], "sequence_by": ["seq"],
            "grouping": {"kind": "exact_time"},
            "outputs": [{"kind": "rank", "primitive_version": 1, "input": "x", "output": "rank", "direction": "ascending", "tie_method": "average", "null_placement": "exclude", "min_samples": 1}],
            "allowed_lateness_micros": 0, "late_policy": {"kind": "drop", "metrics_version": 1},
            "value_policy": "nan_exclude_preserve_v1"
        })).unwrap();
            (
                calc_flow::CrossSectionOperator::new("next", input, spec)
                    .unwrap()
                    .into(),
                "input",
            )
        }
        "window" => (
            calc_flow::WindowAggregateOperator::new(
                "next",
                input,
                calc_flow::WindowSpec::tumbling("ts", Duration::from_secs(1)).unwrap(),
            )
            .unwrap()
            .into(),
            "input",
        ),
        "union" => (
            Box::new(
                UnionOperator::new(
                    "next",
                    vec![
                        late_table_port("left", &input),
                        late_table_port("right", &input),
                    ],
                )
                .unwrap(),
            )
            .into(),
            "left",
        ),
        "external" | "array" => (
            (Box::new(StreamOnlyOperator {
                input_ports: [late_table_port("input", &input)],
                output_ports: [Port::new(
                    "output",
                    if successor == "array" {
                        BatchKind::Array
                    } else {
                        BatchKind::Table
                    },
                    true,
                    None,
                )
                .unwrap()],
            }) as Box<dyn StreamOperator>)
                .into(),
            "input",
        ),
        _ => unreachable!(),
    }
}

#[test]
fn test_side_output_native_rejects_temporal_merge_and_external_successors() {
    for path in ["direct", "expression", "sql"] {
        for successor in [
            "rolling",
            "cross_section",
            "window",
            "union",
            "external",
            "array",
        ] {
            let (builder, source) = late_path_builder(path);
            let operator = late_rolling(late_policy());
            let input = if path == "direct" {
                operator.output_ports()[1].schema()
            } else {
                operator.input_ports()[0].schema()
            }
            .unwrap()
            .clone();
            let (operator, ingress) = late_successor(successor, input);
            let port = if source == "roll" { "late" } else { "output" };
            let error = builder
                .add_node("next", operator)
                .unwrap()
                .connect(edge((source, port), ("next", ingress)))
                .unwrap()
                .compile_stream(&udfs(), &StreamRequirements::default())
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("temporal_output_unavailable"),
                "{path}/{successor}: {error}"
            );
            assert!(
                error.contains("roll.late") && error.contains("next"),
                "{error}"
            );
        }
    }
}

#[test]
fn test_side_output_native_allows_stateless_routes_and_changes_lineage() {
    for path in ["direct", "expression", "sql"] {
        let (builder, _) = late_path_builder(path);
        let plan = builder
            .compile_stream(&udfs(), &StreamRequirements::default())
            .unwrap();
        assert_eq!(plan.sink_binding_ids().len(), 2);
    }
    let old = PipelineBuilder::new("late-contract")
        .unwrap()
        .add_node(
            "roll",
            late_rolling(calc_flow::LatePolicySpec::Drop { metrics_version: 1 }),
        )
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();
    let new = late_builder()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();
    assert_ne!(old.fingerprint(), new.fingerprint());
    assert_eq!(old.sink_binding_ids(), ["output"]);
    assert_eq!(new.sink_binding_ids(), ["late", "output"]);
}

struct UnopenedLateSource(Arc<std::sync::atomic::AtomicUsize>);

#[async_trait]
impl calc_flow::StreamSource for UnopenedLateSource {
    fn capabilities(&self) -> calc_flow::SourceCapabilities {
        calc_flow::SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::Unsupported,
            delivery: calc_flow::SourceDeliveryCapability::Lossy,
            max_batch_rows: 1,
            max_batch_bytes: 4096,
            schema: calc_flow::SourceSchema::DynamicOrUnknown,
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, _cursor: Option<calc_flow::Cursor>) -> Result<()> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<calc_flow::SourceEvent>> {
        Ok(None)
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct UnopenedLateSink;

#[async_trait]
impl calc_flow::StreamSink for UnopenedLateSink {
    async fn open(&mut self) -> Result<()> {
        panic!("disabled capability must not open a sink")
    }
    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn test_side_output_runner_requires_both_bindings_and_stays_disabled() {
    for outputs in [vec!["output"], vec!["late"], vec!["late", "output"]] {
        let opens = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let plan = late_builder()
            .compile_stream(&udfs(), &StreamRequirements::default())
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let result = calc_flow::StreamingRunner::new(
            plan,
            BTreeMap::from([(
                "input".into(),
                calc_flow::SourceBinding::new(UnopenedLateSource(opens.clone())),
            )]),
            outputs
                .iter()
                .map(|name| {
                    (
                        String::from(*name),
                        vec![
                            calc_flow::SinkBinding::ordinary(
                                &format!("{name}_sink"),
                                UnopenedLateSink,
                            )
                            .unwrap(),
                        ],
                    )
                })
                .collect(),
            calc_flow::ManagedCheckpointRuntime::new(directory.path()).unwrap(),
        );
        let error = match result {
            Ok(_) => panic!("side output must stay disabled"),
            Err(error) => error.to_string(),
        };
        if outputs.len() == 1 {
            let missing = if outputs[0] == "output" {
                "late"
            } else {
                "output"
            };
            assert!(
                error.contains("missing graph output") && error.contains(missing),
                "{error}"
            );
        } else {
            assert!(
                error.contains("late side output execution is not enabled"),
                "{error}"
            );
        }
        assert_eq!(opens.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
}

#[test]
fn test_side_output_policy_migration_handles_all_variants_exhaustively() {
    fn output_count(policy: calc_flow::LatePolicySpec) -> usize {
        match policy {
            calc_flow::LatePolicySpec::Error { .. } | calc_flow::LatePolicySpec::Drop { .. } => 1,
            calc_flow::LatePolicySpec::SideOutput { .. } => 2,
        }
    }
    assert_eq!(output_count(late_policy()), 2);
    assert_eq!(
        output_count(calc_flow::LatePolicySpec::Drop { metrics_version: 1 }),
        1
    );
    assert_eq!(
        output_count(calc_flow::LatePolicySpec::Error {
            scope: calc_flow::LateErrorScope::Envelope
        }),
        1
    );
}

fn late_table_port(name: &str, schema: &datafusion::arrow::datatypes::Schema) -> Port {
    Port::new(
        name,
        BatchKind::Table,
        true,
        Some(
            schema
                .fields()
                .iter()
                .map(|field| field.as_ref().clone())
                .collect(),
        ),
    )
    .unwrap()
}

#[test]
fn test_side_output_type_erased_native_operator_keeps_contract() {
    let batch = Box::new(late_rolling(late_policy())) as Box<dyn BatchOperator>;
    let error = PipelineBuilder::new("erased")
        .unwrap()
        .add_node("roll", batch)
        .unwrap()
        .compile_batch(&udfs())
        .unwrap_err()
        .to_string();
    assert!(error.contains("unsupported_mode"), "{error}");
    let stream = Box::new(late_rolling(late_policy())) as Box<dyn StreamOperator>;
    let directory = tempfile::tempdir().unwrap();
    let opens = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let plan = PipelineBuilder::new("erased")
        .unwrap()
        .add_node("roll", stream)
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();
    let result = calc_flow::StreamingRunner::new(
        plan,
        BTreeMap::from([(
            "input".into(),
            calc_flow::SourceBinding::new(UnopenedLateSource(opens.clone())),
        )]),
        ["output", "late"]
            .into_iter()
            .map(|name| {
                (
                    name.into(),
                    vec![
                        calc_flow::SinkBinding::ordinary(&format!("{name}_sink"), UnopenedLateSink)
                            .unwrap(),
                    ],
                )
            })
            .collect(),
        calc_flow::ManagedCheckpointRuntime::new(directory.path()).unwrap(),
    );
    assert!(result.is_err());
    assert_eq!(opens.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[test]
fn test_side_output_normal_path_retains_temporal_contract() {
    let source = late_rolling(late_policy());
    let input = source.output_ports()[0].schema().unwrap().clone();
    let mut spec = source.spec().clone();
    spec.late_policy = calc_flow::LatePolicySpec::Drop { metrics_version: 1 };
    let calc_flow::RollingOutputSpec::Lag { output, .. } = &mut spec.outputs[0] else {
        unreachable!()
    };
    *output = "lag2".into();
    let next = calc_flow::RollingOperator::new("next", input, spec).unwrap();
    let plan = PipelineBuilder::new("normal-temporal")
        .unwrap()
        .add_node("roll", source)
        .unwrap()
        .add_node("next", next)
        .unwrap()
        .connect(edge(("roll", "output"), ("next", "input")))
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();
    assert_eq!(plan.sink_binding_ids(), ["late", "output"]);
}

#[test]
fn test_side_output_diagnostic_names_do_not_forge_origin() {
    let input = late_rolling(late_policy()).output_ports()[1]
        .schema()
        .unwrap()
        .clone();
    let source = ExpressionOperator::new("project", "x = x", vec![], None, vec![])
        .unwrap()
        .with_ports(
            late_table_port("input", &input),
            late_table_port("output", &input),
        )
        .unwrap();
    let spec = late_rolling(calc_flow::LatePolicySpec::Drop { metrics_version: 1 })
        .spec()
        .clone();
    let next = calc_flow::RollingOperator::new("next", input, spec).unwrap();
    let plan = PipelineBuilder::new("diagnostic-names")
        .unwrap()
        .add_node("project", Box::new(source))
        .unwrap()
        .add_node("next", next)
        .unwrap()
        .connect(edge(("project", "output"), ("next", "input")))
        .unwrap()
        .compile_stream(&udfs(), &StreamRequirements::default())
        .unwrap();
    assert_eq!(plan.sink_binding_ids(), ["output"]);
}
