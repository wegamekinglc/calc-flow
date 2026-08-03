use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchOperator, BatchOperatorContext, CalcFlowError, Edge, ExpressionOperator,
    JsonMap, OperatorMetadata, PipelineBuilder, Port, PortEndpoint, Result, UdfKind, UdfReference,
    UdfRegistry,
};
use datafusion::{
    arrow::datatypes::{DataType, Field},
    common::ScalarValue,
    logical_expr::{ColumnarValue, ScalarUDF, Volatility, create_udf},
};
use serde_json::{Value, json};

struct TestOperator {
    name: String,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    configuration: JsonMap,
    udfs: Vec<UdfReference>,
}

impl TestOperator {
    fn new(name: &str, inputs: Vec<Port>, outputs: Vec<Port>) -> Self {
        Self {
            name: name.into(),
            inputs,
            outputs,
            configuration: BTreeMap::new(),
            udfs: Vec::new(),
        }
    }

    fn with_configuration(mut self, configuration: JsonMap) -> Self {
        self.configuration = configuration;
        self
    }

    fn with_udfs(mut self, udfs: Vec<UdfReference>) -> Self {
        self.udfs = udfs;
        self
    }
}

impl OperatorMetadata for TestOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        self.configuration.clone()
    }

    fn udf_references(&self) -> Vec<UdfReference> {
        self.udfs.clone()
    }
}

#[async_trait]
impl BatchOperator for TestOperator {
    async fn process(
        &mut self,
        _inputs: &BTreeMap<String, Batch>,
        _context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        Ok(BTreeMap::new())
    }
}

fn table_port(name: &str, required: bool) -> Port {
    Port::new(name, BatchKind::Table, required, None).unwrap()
}

fn typed_table_port(name: &str, field_name: &str, data_type: DataType) -> Port {
    Port::new(
        name,
        BatchKind::Table,
        true,
        Some(vec![Field::new(field_name, data_type, false)]),
    )
    .unwrap()
}

fn array_port(name: &str) -> Port {
    Port::new(name, BatchKind::Array, true, None).unwrap()
}

fn source(name: &str) -> Box<dyn BatchOperator> {
    Box::new(TestOperator::new(
        name,
        vec![],
        vec![table_port("output", true)],
    )) as Box<dyn BatchOperator>
}

fn transform(name: &str) -> Box<dyn BatchOperator> {
    Box::new(TestOperator::new(
        name,
        vec![table_port("input", true)],
        vec![table_port("output", true)],
    )) as Box<dyn BatchOperator>
}

fn endpoint(node_id: &str, port: &str) -> PortEndpoint {
    PortEndpoint::new(node_id, port).unwrap()
}

fn edge(source_node: &str, source_port: &str, target_node: &str, target_port: &str) -> Edge {
    Edge::new(
        endpoint(source_node, source_port),
        endpoint(target_node, target_port),
    )
}

fn constant_udf(name: &str, value: i64) -> Arc<ScalarUDF> {
    Arc::new(create_udf(
        name,
        vec![],
        DataType::Int64,
        Volatility::Immutable,
        Arc::new(move |_| Ok(ColumnarValue::Scalar(ScalarValue::Int64(Some(value))))),
    ))
}

#[test]
fn endpoints_and_pipeline_names_must_not_be_empty() {
    for (node_id, port) in [("", "output"), ("source", "")] {
        assert!(matches!(
            PortEndpoint::new(node_id, port),
            Err(CalcFlowError::InvalidArgument { field, .. }) if field == "endpoint"
        ));
    }
    assert!(matches!(
        PipelineBuilder::new(""),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "pipeline.name"
    ));
}

#[test]
fn duplicate_and_empty_node_ids_are_rejected() {
    assert!(
        PipelineBuilder::new("duplicate")
            .unwrap()
            .add_node("node", source("first"))
            .unwrap()
            .add_node("node", source("second"))
            .is_err()
    );
    assert!(
        PipelineBuilder::new("empty")
            .unwrap()
            .add_node("", source("source"))
            .is_err()
    );
}

#[test]
fn connections_require_existing_nodes() {
    let builder = PipelineBuilder::new("unknown")
        .unwrap()
        .add_node("known", source("known"))
        .unwrap();
    assert!(
        builder
            .connect(edge("known", "output", "missing", "input"))
            .is_err()
    );

    let builder = PipelineBuilder::new("unknown")
        .unwrap()
        .add_node("known", transform("known"))
        .unwrap();
    assert!(
        builder
            .connect(edge("missing", "output", "known", "input"))
            .is_err()
    );
}

#[test]
fn compilation_rejects_missing_source_and_target_ports() {
    let registry = UdfRegistry::new().snapshot();
    for invalid_edge in [
        edge("source", "missing", "target", "input"),
        edge("source", "output", "target", "missing"),
    ] {
        let result = PipelineBuilder::new("ports")
            .unwrap()
            .add_node("source", source("source"))
            .unwrap()
            .add_node("target", transform("target"))
            .unwrap()
            .connect(invalid_edge)
            .unwrap()
            .compile_batch(&registry);
        assert!(matches!(result, Err(CalcFlowError::Compile { .. })));
    }
}

#[test]
fn compilation_enforces_port_direction() {
    let registry = UdfRegistry::new().snapshot();
    for invalid_edge in [
        edge("first", "input", "second", "input"),
        edge("first", "output", "second", "output"),
    ] {
        let result = PipelineBuilder::new("directions")
            .unwrap()
            .add_node("first", transform("first"))
            .unwrap()
            .add_node("second", transform("second"))
            .unwrap()
            .connect(invalid_edge)
            .unwrap()
            .compile_batch(&registry);
        assert!(matches!(result, Err(CalcFlowError::Compile { .. })));
    }
}

#[test]
fn compilation_enforces_kind_and_exact_arrow_schema_compatibility() {
    let registry = UdfRegistry::new().snapshot();
    let kind_result = PipelineBuilder::new("kind")
        .unwrap()
        .add_node(
            "source",
            Box::new(TestOperator::new(
                "source",
                vec![],
                vec![array_port("output")],
            )) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .add_node("target", transform("target"))
        .unwrap()
        .connect(edge("source", "output", "target", "input"))
        .unwrap()
        .compile_batch(&registry);
    assert!(matches!(kind_result, Err(CalcFlowError::Compile { .. })));

    let schema_result = PipelineBuilder::new("schema")
        .unwrap()
        .add_node(
            "source",
            Box::new(TestOperator::new(
                "source",
                vec![],
                vec![typed_table_port("output", "value", DataType::Int64)],
            )) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .add_node(
            "target",
            Box::new(TestOperator::new(
                "target",
                vec![typed_table_port("input", "value", DataType::Utf8)],
                vec![table_port("output", true)],
            )) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .connect(edge("source", "output", "target", "input"))
        .unwrap()
        .compile_batch(&registry);
    assert!(matches!(schema_result, Err(CalcFlowError::Compile { .. })));
}

#[test]
fn compilation_rejects_duplicate_edges_and_multiple_input_writers() {
    let registry = UdfRegistry::new().snapshot();
    let duplicate = edge("source", "output", "target", "input");
    let duplicate_result = PipelineBuilder::new("duplicate edge")
        .unwrap()
        .add_node("source", source("source"))
        .unwrap()
        .add_node("target", transform("target"))
        .unwrap()
        .connect(duplicate.clone())
        .unwrap()
        .connect(duplicate)
        .unwrap()
        .compile_batch(&registry);
    assert!(matches!(
        duplicate_result,
        Err(CalcFlowError::Compile { .. })
    ));

    let writer_result = PipelineBuilder::new("writers")
        .unwrap()
        .add_node("first", source("first"))
        .unwrap()
        .add_node("second", source("second"))
        .unwrap()
        .add_node("target", transform("target"))
        .unwrap()
        .connect(edge("first", "output", "target", "input"))
        .unwrap()
        .connect(edge("second", "output", "target", "input"))
        .unwrap()
        .compile_batch(&registry);
    assert!(matches!(writer_result, Err(CalcFlowError::Compile { .. })));
}

#[test]
fn compilation_rejects_self_and_multi_node_cycles() {
    let registry = UdfRegistry::new().snapshot();
    let self_cycle = PipelineBuilder::new("self cycle")
        .unwrap()
        .add_node("node", transform("node"))
        .unwrap()
        .connect(edge("node", "output", "node", "input"))
        .unwrap()
        .compile_batch(&registry);
    assert!(matches!(self_cycle, Err(CalcFlowError::Compile { .. })));

    let multi_cycle = PipelineBuilder::new("multi cycle")
        .unwrap()
        .add_node("first", transform("first"))
        .unwrap()
        .add_node("second", transform("second"))
        .unwrap()
        .connect(edge("first", "output", "second", "input"))
        .unwrap()
        .connect(edge("second", "output", "first", "input"))
        .unwrap()
        .compile_batch(&registry);
    assert!(matches!(multi_cycle, Err(CalcFlowError::Compile { .. })));
}

#[test]
fn kahn_topology_is_deterministic_regardless_of_insertion_order() {
    let registry = UdfRegistry::new().snapshot();
    let first = PipelineBuilder::new("topology")
        .unwrap()
        .add_node("right", transform("right"))
        .unwrap()
        .add_node("root", source("root"))
        .unwrap()
        .add_node("left", transform("left"))
        .unwrap()
        .connect(edge("root", "output", "right", "input"))
        .unwrap()
        .connect(edge("right", "output", "left", "input"))
        .unwrap()
        .compile_batch(&registry)
        .unwrap();
    let second = PipelineBuilder::new("topology")
        .unwrap()
        .add_node("left", transform("left"))
        .unwrap()
        .add_node("root", source("root"))
        .unwrap()
        .add_node("right", transform("right"))
        .unwrap()
        .connect(edge("right", "output", "left", "input"))
        .unwrap()
        .connect(edge("root", "output", "right", "input"))
        .unwrap()
        .compile_batch(&registry)
        .unwrap();

    assert_eq!(first.topological_order(), ["root", "right", "left"]);
    assert_eq!(first.topological_order(), second.topological_order());
    assert_eq!(first.fingerprint(), second.fingerprint());
}

#[test]
fn kahn_topology_counts_distinct_edges_between_the_same_nodes() {
    let plan = PipelineBuilder::new("parallel edges")
        .unwrap()
        .add_node(
            "source",
            Box::new(TestOperator::new(
                "source",
                vec![],
                vec![table_port("first", true), table_port("second", true)],
            )) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .add_node(
            "target",
            Box::new(TestOperator::new(
                "target",
                vec![table_port("first", true), table_port("second", true)],
                vec![table_port("output", true)],
            )) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .connect(edge("source", "first", "target", "first"))
        .unwrap()
        .connect(edge("source", "second", "target", "second"))
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();

    assert_eq!(plan.topological_order(), ["source", "target"]);
}

#[test]
fn external_names_are_stable_unique_and_include_required_and_optional_inputs() {
    let build = |reverse: bool| {
        let left = TestOperator::new(
            "left",
            vec![table_port("required", true), table_port("optional", false)],
            vec![table_port("output", true)],
        );
        let right = TestOperator::new(
            "right",
            vec![table_port("required", true), table_port("optional", false)],
            vec![table_port("output", true)],
        );
        let builder = PipelineBuilder::new("externals").unwrap();
        let builder = if reverse {
            builder
                .add_node("right", Box::new(right) as Box<dyn BatchOperator>)
                .unwrap()
                .add_node("left", Box::new(left) as Box<dyn BatchOperator>)
                .unwrap()
        } else {
            builder
                .add_node("left", Box::new(left) as Box<dyn BatchOperator>)
                .unwrap()
                .add_node("right", Box::new(right) as Box<dyn BatchOperator>)
                .unwrap()
        };
        builder
            .compile_batch(&UdfRegistry::new().snapshot())
            .unwrap()
    };
    let first = build(false);
    let second = build(true);

    assert_eq!(first.external_inputs(), second.external_inputs());
    assert_eq!(first.external_outputs(), second.external_outputs());
    assert_eq!(
        first
            .external_inputs()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "left.optional",
            "left.required",
            "right.optional",
            "right.required"
        ]
    );
    assert_eq!(
        first
            .external_outputs()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["left.output", "right.output"]
    );
}

#[test]
fn unique_external_port_names_stay_bare() {
    let plan = PipelineBuilder::new("single")
        .unwrap()
        .add_node("calc", transform("calc"))
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();

    assert_eq!(
        plan.external_inputs()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["input"]
    );
    assert_eq!(
        plan.external_outputs()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["output"]
    );
}

#[test]
fn compilation_requires_at_least_one_external_output() {
    let result = PipelineBuilder::new("no output")
        .unwrap()
        .add_node(
            "sink",
            Box::new(TestOperator::new(
                "sink",
                vec![table_port("input", true)],
                vec![],
            )) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot());
    assert!(matches!(result, Err(CalcFlowError::Compile { .. })));
}

fn configurable_graph(
    pipeline_name: &str,
    node_id: &str,
    configuration: Value,
    field_name: &str,
) -> String {
    PipelineBuilder::new(pipeline_name)
        .unwrap()
        .add_node(
            node_id,
            Box::new(
                TestOperator::new(
                    "operator",
                    vec![typed_table_port("input", field_name, DataType::Int64)],
                    vec![typed_table_port("output", field_name, DataType::Int64)],
                )
                .with_configuration(BTreeMap::from([("setting".into(), configuration)])),
            ) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap()
        .fingerprint()
        .into()
}

#[test]
fn fingerprints_change_with_pipeline_node_configuration_and_ports() {
    let baseline = configurable_graph("pipeline", "node", json!(1), "value");
    assert_eq!(baseline.len(), 64);
    assert!(
        baseline
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    );
    assert_eq!(
        baseline,
        configurable_graph("pipeline", "node", json!(1), "value")
    );
    assert_ne!(
        baseline,
        configurable_graph("renamed", "node", json!(1), "value")
    );
    assert_ne!(
        baseline,
        configurable_graph("pipeline", "renamed", json!(1), "value")
    );
    assert_ne!(
        baseline,
        configurable_graph("pipeline", "node", json!(2), "value")
    );
    assert_ne!(
        baseline,
        configurable_graph("pipeline", "node", json!(1), "renamed")
    );
}

#[test]
fn dictionary_schemas_compile_with_deterministic_fingerprints() {
    let build = || {
        PipelineBuilder::new("dictionary schema")
            .unwrap()
            .add_node(
                "source",
                Box::new(TestOperator::new(
                    "source",
                    vec![],
                    vec![typed_table_port(
                        "output",
                        "category",
                        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                    )],
                )) as Box<dyn BatchOperator>,
            )
            .unwrap()
            .compile_batch(&UdfRegistry::new().snapshot())
            .unwrap()
            .fingerprint()
            .to_owned()
    };

    assert_eq!(build(), build());
}

#[test]
fn fingerprint_treats_built_in_udf_references_as_a_set() {
    let first = UdfReference::new("python", "normalize", "1", UdfKind::ExternalScalar).unwrap();
    let second = UdfReference::new("python", "score", "1", UdfKind::ExternalScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry.register_external(first.clone(), 1).unwrap();
    registry.register_external(second.clone(), 1).unwrap();
    let build = |udfs| {
        PipelineBuilder::new("canonical references")
            .unwrap()
            .add_node(
                "calculate",
                Box::new(
                    ExpressionOperator::new("calculate", "value", vec![], None, udfs).unwrap(),
                ),
            )
            .unwrap()
            .compile_batch(&registry.snapshot())
            .unwrap()
            .fingerprint()
            .to_owned()
    };

    assert_eq!(
        build(vec![first.clone(), second.clone(), first.clone()]),
        build(vec![second, first])
    );
}

#[test]
fn fingerprint_preserves_arbitrary_external_configuration_array_order() {
    let build = |steps: [&str; 2]| {
        PipelineBuilder::new("ordered external configuration")
            .unwrap()
            .add_node(
                "external",
                Box::new(
                    TestOperator::new("external", vec![], vec![table_port("output", true)])
                        .with_configuration(BTreeMap::from([("udfs".into(), json!(steps))])),
                ) as Box<dyn BatchOperator>,
            )
            .unwrap()
            .compile_batch(&UdfRegistry::new().snapshot())
            .unwrap()
            .fingerprint()
            .to_owned()
    };

    assert_ne!(build(["prepare", "score"]), build(["score", "prepare"]));
}

#[test]
fn fingerprints_change_with_edges() {
    let registry = UdfRegistry::new().snapshot();
    let disconnected = PipelineBuilder::new("edges")
        .unwrap()
        .add_node("source", source("source"))
        .unwrap()
        .add_node("target", transform("target"))
        .unwrap()
        .compile_batch(&registry)
        .unwrap();
    let connected = PipelineBuilder::new("edges")
        .unwrap()
        .add_node("source", source("source"))
        .unwrap()
        .add_node("target", transform("target"))
        .unwrap()
        .connect(edge("source", "output", "target", "input"))
        .unwrap()
        .compile_batch(&registry)
        .unwrap();

    assert_ne!(disconnected.fingerprint(), connected.fingerprint());
}

#[test]
fn fingerprint_uses_only_selected_udf_catalog_entries() {
    let selected = UdfReference::new("python", "normalize", "1", UdfKind::ExternalScalar).unwrap();
    let unselected = UdfReference::new("numpy", "clip", "1", UdfKind::ExternalArray).unwrap();
    let build = |registry: &UdfRegistry| {
        PipelineBuilder::new("udfs")
            .unwrap()
            .add_node(
                "node",
                Box::new(
                    TestOperator::new("node", vec![], vec![table_port("output", true)])
                        .with_udfs(vec![selected.clone()]),
                ) as Box<dyn BatchOperator>,
            )
            .unwrap()
            .compile_batch(&registry.snapshot())
            .unwrap()
            .fingerprint()
            .to_owned()
    };

    let mut baseline_registry = UdfRegistry::new();
    baseline_registry
        .register_external(selected.clone(), 1)
        .unwrap();
    let baseline = build(&baseline_registry);

    let mut extra_registry = UdfRegistry::new();
    extra_registry
        .register_external(selected.clone(), 1)
        .unwrap();
    extra_registry.register_external(unselected, 3).unwrap();
    assert_eq!(baseline, build(&extra_registry));

    let mut changed_catalog = UdfRegistry::new();
    changed_catalog
        .register_external(selected.clone(), 2)
        .unwrap();
    assert_ne!(baseline, build(&changed_catalog));
}

#[test]
fn fingerprint_changes_with_selected_udf_references() {
    let first = UdfReference::new("python", "score", "1", UdfKind::ExternalScalar).unwrap();
    let second = UdfReference::new("python", "score", "2", UdfKind::ExternalScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry.register_external(first.clone(), 1).unwrap();
    registry.register_external(second.clone(), 1).unwrap();
    let build = |reference| {
        PipelineBuilder::new("references")
            .unwrap()
            .add_node(
                "node",
                Box::new(
                    TestOperator::new("node", vec![], vec![table_port("output", true)])
                        .with_udfs(vec![reference]),
                ) as Box<dyn BatchOperator>,
            )
            .unwrap()
            .compile_batch(&registry.snapshot())
            .unwrap()
            .fingerprint()
            .to_owned()
    };

    assert_ne!(build(first), build(second));
}

#[test]
fn unknown_referenced_udfs_are_rejected_before_plan_creation() {
    let unknown = UdfReference::new("python", "missing", "1", UdfKind::ExternalScalar).unwrap();
    let result = PipelineBuilder::new("unknown udf")
        .unwrap()
        .add_node(
            "node",
            Box::new(
                TestOperator::new("node", vec![], vec![table_port("output", true)])
                    .with_udfs(vec![unknown]),
            ) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot());
    assert!(matches!(
        result,
        Err(CalcFlowError::Compile { message }) if message.contains("unknown UDF")
    ));
}

#[test]
fn conflicting_selected_datafusion_references_are_rejected_before_plan_creation() {
    let first = UdfReference::new("rust", "score", "1", UdfKind::DataFusionScalar).unwrap();
    let second = UdfReference::new("plugin", "score", "1", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    registry
        .register_datafusion(first.clone(), constant_udf("score", 1), 0)
        .unwrap();
    registry
        .register_datafusion(second.clone(), constant_udf("score", 2), 0)
        .unwrap();

    let result = PipelineBuilder::new("conflicting udfs")
        .unwrap()
        .add_node(
            "node",
            Box::new(
                TestOperator::new("node", vec![], vec![table_port("output", true)])
                    .with_udfs(vec![first, second]),
            ) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .compile_batch(&registry.snapshot());
    assert!(matches!(
        result,
        Err(CalcFlowError::Compile { message }) if message.contains("collides")
    ));
}
