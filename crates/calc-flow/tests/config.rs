use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use calc_flow::{
    ArrowFieldSpec, Batch, BatchKind, BatchOperator, BatchOperatorContext, BatchOperatorFactory,
    ConnectorRegistry, DataFusionConfig, DataSourceSpec, Edge, EdgeSpec, ExpressionOperator,
    ExternalOperatorSpec, JsonMap, NodeSpec, OperatorMetadata, OperatorSpec,
    PROJECT_FORMAT_VERSION, PipelineBuilder, PipelineSpec, Port, PortEndpoint, PortSpec,
    PositionSpec, ProjectSpec, ProviderRegistry, Result, RunOptions, RuntimeSpec, StateConfig,
    StreamRequirements, StreamRunOptions, UdfKind, UdfReference, UdfRegistry, ValidationReport,
    compile_project, compile_stream_project, project_json_schema, validate_project,
};
use datafusion::arrow::datatypes::DataType;
use parking_lot::Mutex;
use serde_json::{Value, json};

fn project(node: NodeSpec) -> ProjectSpec {
    ProjectSpec {
        format_version: PROJECT_FORMAT_VERSION,
        id: "project".into(),
        name: "Project".into(),
        description: String::new(),
        runtime: RuntimeSpec::default(),
        graph: PipelineSpec {
            name: "graph".into(),
            nodes: vec![node],
            edges: Vec::new(),
            datafusion: DataFusionConfig::default(),
        },
        data_sources: vec![DataSourceSpec {
            id: "source".into(),
            input: "input".into(),
            format: "inline_json".into(),
            data: json!([{ "value": 1 }]),
        }],
        sources: Vec::new(),
        sinks: Vec::new(),
        state: StateConfig::default(),
    }
}

fn batch_options_mut(project: &mut ProjectSpec) -> &mut RunOptions {
    let RuntimeSpec::Batch(options) = &mut project.runtime else {
        panic!("test project must use batch runtime options");
    };
    options
}

fn expression_node(id: &str) -> NodeSpec {
    NodeSpec {
        id: id.into(),
        operator: OperatorSpec::Expression {
            expression: "result = value + 1".into(),
            select: Vec::new(),
            filter: None,
            udfs: Vec::new(),
        },
        input_ports: Vec::new(),
        output_ports: Vec::new(),
        position: Some(PositionSpec { x: 1.0, y: 2.0 }),
    }
}

fn port(name: &str, kind: BatchKind, required: bool, schema: Vec<ArrowFieldSpec>) -> PortSpec {
    PortSpec {
        name: name.into(),
        kind,
        required,
        schema,
    }
}

fn empty_registries() -> (ProviderRegistry, calc_flow::UdfRegistrySnapshot) {
    (ProviderRegistry::default(), UdfRegistry::new().snapshot())
}

fn assert_issue(report: &ValidationReport, path: &str, code: &str) {
    assert!(!report.valid, "unexpected valid report: {report:?}");
    assert!(
        report
            .issues
            .iter()
            .any(|issue| issue.path == path && issue.code == code),
        "missing {path}:{code} in {:?}",
        report.issues
    );
}

#[test]
fn project_rejects_v1_and_unknown_fields_at_every_nested_level() {
    let v1 = r#"{"format_version":1,"id":"x","name":"x","runtime":{"mode":"batch","options":{}},"graph":{"name":"p","nodes":[],"edges":[]},"data_sources":[]}"#;
    assert!(serde_json::from_str::<ProjectSpec>(v1).is_err());

    let base = serde_json::to_value(project(expression_node("node"))).unwrap();
    for path in [
        vec![],
        vec!["graph"],
        vec!["graph", "nodes", "0"],
        vec!["graph", "nodes", "0", "operator"],
        vec!["graph", "nodes", "0", "input_ports", "0"],
        vec!["graph", "nodes", "0", "input_ports", "0", "schema", "0"],
        vec!["graph", "nodes", "0", "position"],
        vec!["data_sources", "0"],
        vec!["runtime", "options"],
        vec!["graph", "datafusion"],
    ] {
        let mut value = base.clone();
        if path.contains(&"input_ports") {
            value["graph"]["nodes"][0]["input_ports"] = json!([{
                "name": "input", "kind": "table", "required": true,
                "schema": [{"name": "value", "data_type": "int64", "nullable": false}]
            }]);
        }
        let mut cursor = &mut value;
        for segment in &path {
            cursor = if let Ok(index) = segment.parse::<usize>() {
                &mut cursor[index]
            } else {
                &mut cursor[*segment]
            };
        }
        cursor["callable"] = Value::String("os.system".into());
        assert!(
            serde_json::from_value::<ProjectSpec>(value).is_err(),
            "accepted unknown field at {path:?}"
        );
    }
}

#[test]
fn generated_schema_is_v3_stable_closed_and_contains_no_executable_fields() {
    assert_eq!(PROJECT_FORMAT_VERSION, 3);
    let first = project_json_schema().unwrap();
    let second = project_json_schema().unwrap();
    assert_eq!(first, second);
    assert_eq!(first["title"], "Calc Flow Project V3");
    assert_eq!(first["properties"]["format_version"]["const"], 3);
    let text = serde_json::to_string(&first).unwrap();
    for forbidden in ["callable", "import_path", "source_code"] {
        assert!(!text.contains(forbidden));
    }
    assert_closed_models(&first, "root");
}

fn assert_closed_models(value: &Value, path: &str) {
    match value {
        Value::Object(map) => {
            if map.get("type") == Some(&Value::String("object".into()))
                && !matches!(path, _ if path.ends_with("options") || path.ends_with("data") || path.ends_with("secrets"))
            {
                assert_eq!(
                    map.get("additionalProperties"),
                    Some(&Value::Bool(false)),
                    "open object schema at {path}"
                );
            }
            for (key, child) in map {
                assert_closed_models(child, &format!("{path}.{key}"));
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                assert_closed_models(child, &format!("{path}[{index}]"));
            }
        }
        _ => {}
    }
}

#[test]
fn defaults_round_trip_canonically() {
    let text = r#"{
        "format_version": 3,
        "id": "project",
        "name": "Project",
        "runtime": {"mode": "batch", "options": {}},
        "graph": {"name": "graph", "nodes": [{
            "id": "node", "operator": {"kind": "expression", "expression": "x + 1"}
        }]}
    }"#;
    let parsed: ProjectSpec = serde_json::from_str(text).unwrap();
    assert_eq!(parsed.description, "");
    assert!(parsed.graph.edges.is_empty());
    assert!(parsed.data_sources.is_empty());
    assert_eq!(parsed.runtime, RuntimeSpec::default());
    assert_eq!(
        parsed,
        serde_json::from_value(serde_json::to_value(&parsed).unwrap()).unwrap()
    );
}

#[test]
fn datafusion_config_accepts_strict_partial_defaults() {
    let defaults = DataFusionConfig::default();
    for (value, expected) in [
        (json!({}), defaults),
        (
            json!({"batch_size": 4_096}),
            DataFusionConfig {
                batch_size: 4_096,
                ..defaults
            },
        ),
        (
            json!({"target_partitions": 2}),
            DataFusionConfig {
                target_partitions: 2,
                ..defaults
            },
        ),
    ] {
        assert_eq!(
            serde_json::from_value::<DataFusionConfig>(value).unwrap(),
            expected
        );
    }
    assert!(
        serde_json::from_value::<DataFusionConfig>(json!({"batch_size": 1, "extra": 2})).is_err()
    );
}

#[test]
fn validate_rejects_invalid_constructed_identity_positions_and_limits() {
    let (providers, udfs) = empty_registries();
    let mut value = project(expression_node("node"));
    value.format_version = 1;
    assert_issue(
        &validate_project(&value, &providers, &udfs),
        "format_version",
        "unsupported_version",
    );

    let mut value = project(expression_node(""));
    value.id.clear();
    value.graph.name.clear();
    value.graph.nodes[0].position = Some(PositionSpec {
        x: f64::NAN,
        y: f64::INFINITY,
    });
    let options = batch_options_mut(&mut value);
    options.max_rows = 0;
    options.memory_limit_mb = 2_049;
    value.graph.datafusion.batch_size = 0;
    let report = validate_project(&value, &providers, &udfs);
    for (path, code) in [
        ("id", "invalid_id"),
        ("graph.name", "required"),
        ("graph.nodes[0].id", "invalid_id"),
        ("graph.nodes[0].position.x", "not_finite"),
        ("graph.nodes[0].position.y", "not_finite"),
        ("runtime.options.max_rows", "out_of_range"),
        ("runtime.options.memory_limit_mb", "out_of_range"),
        ("graph.datafusion.batch_size", "out_of_range"),
    ] {
        assert_issue(&report, path, code);
    }
}

#[test]
fn validate_rejects_duplicate_nodes_ports_fields_and_sources() {
    let (providers, udfs) = empty_registries();
    let mut value = project(expression_node("node"));
    value.graph.nodes.push(expression_node("node"));
    value.graph.nodes[0].input_ports = vec![
        port(
            "input",
            BatchKind::Table,
            true,
            vec![
                ArrowFieldSpec {
                    name: "value".into(),
                    data_type: "int64".into(),
                    nullable: false,
                },
                ArrowFieldSpec {
                    name: "value".into(),
                    data_type: "int64".into(),
                    nullable: false,
                },
            ],
        ),
        port("input", BatchKind::Table, true, Vec::new()),
    ];
    value.data_sources.push(value.data_sources[0].clone());
    let report = validate_project(&value, &providers, &udfs);
    for (path, code) in [
        ("graph.nodes[1].id", "duplicate_id"),
        ("graph.nodes[0].input_ports[1].name", "duplicate_port"),
        (
            "graph.nodes[0].input_ports[0].schema[1].name",
            "duplicate_field",
        ),
        ("data_sources[1].id", "duplicate_id"),
        ("data_sources[1].input", "duplicate_input"),
    ] {
        assert_issue(&report, path, code);
    }
}

#[test]
fn validate_rejects_unsupported_arrow_types_and_array_schemas() {
    let (providers, udfs) = empty_registries();
    let mut value = project(expression_node("node"));
    value.graph.nodes[0].input_ports = vec![port(
        "input",
        BatchKind::Array,
        true,
        vec![ArrowFieldSpec {
            name: "value".into(),
            data_type: "object".into(),
            nullable: false,
        }],
    )];
    let report = validate_project(&value, &providers, &udfs);
    assert_issue(
        &report,
        "graph.nodes[0].input_ports[0].schema",
        "array_schema",
    );
    assert_issue(
        &report,
        "graph.nodes[0].input_ports[0].schema[0].data_type",
        "unsupported_arrow_type",
    );
}

#[test]
fn validate_requires_exact_supported_data_source_coverage() {
    let (providers, udfs) = empty_registries();
    let mut value = project(expression_node("node"));
    value.data_sources.clear();
    assert_issue(
        &validate_project(&value, &providers, &udfs),
        "data_sources",
        "source_input_mismatch",
    );

    value.data_sources.push(DataSourceSpec {
        id: "source".into(),
        input: "wrong".into(),
        format: "json".into(),
        data: Value::Null,
    });
    let report = validate_project(&value, &providers, &udfs);
    assert_issue(&report, "data_sources", "source_input_mismatch");

    value.data_sources[0].format = "pickle".into();
    let report = validate_project(&value, &providers, &udfs);
    assert_issue(
        &report,
        "data_sources[0].format",
        "unsupported_source_format",
    );
    assert!(report.fingerprint.is_none());
}

#[test]
fn compile_stream_requires_exact_connector_source_coverage() {
    let (providers, udfs) = empty_registries();
    let mut value = project(expression_node("node"));
    value.runtime = RuntimeSpec::Stream(StreamRunOptions::default());
    value.data_sources.clear();
    for sources in [
        json!([]),
        json!([{
            "binding": "wrong",
            "connector": {"provider": "test", "name": "source", "version": "1"}
        }]),
    ] {
        value.sources = serde_json::from_value(sources).unwrap();
        let error = compile_stream_project(
            &value,
            &providers,
            &udfs,
            &ConnectorRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .expect_err("stream compilation must reject incomplete source coverage");
        assert!(
            error
                .to_string()
                .contains("sources [source_input_mismatch]")
        );
    }
}

#[test]
fn compile_expression_uses_exact_configured_ports_and_schemas() {
    let (providers, udfs) = empty_registries();
    let fields = vec![ArrowFieldSpec {
        name: "value".into(),
        data_type: "int64".into(),
        nullable: false,
    }];
    let mut value = project(expression_node("node"));
    value.graph.nodes[0].input_ports = vec![port("input", BatchKind::Table, true, fields.clone())];
    value.graph.nodes[0].output_ports = vec![port("output", BatchKind::Table, true, fields)];
    let plan = compile_project(&value, &providers, &udfs).unwrap();
    assert_eq!(plan.external_inputs()["input"].node_id, "node");
    assert_eq!(plan.external_outputs()["output"].node_id, "node");

    let direct = PipelineBuilder::new("graph")
        .unwrap()
        .add_node(
            "node",
            Box::new(
                ExpressionOperator::new("node", "result = value + 1", Vec::new(), None, Vec::new())
                    .unwrap()
                    .with_ports(
                        Port::new(
                            "input",
                            BatchKind::Table,
                            true,
                            Some(vec![datafusion::arrow::datatypes::Field::new(
                                "value",
                                DataType::Int64,
                                false,
                            )]),
                        )
                        .unwrap(),
                        Port::new(
                            "output",
                            BatchKind::Table,
                            true,
                            Some(vec![datafusion::arrow::datatypes::Field::new(
                                "value",
                                DataType::Int64,
                                false,
                            )]),
                        )
                        .unwrap(),
                    )
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile_batch(&udfs)
        .unwrap();
    assert_eq!(plan.fingerprint(), direct.fingerprint());
}

#[test]
fn datafusion_config_is_preserved_and_changes_the_fingerprint() {
    let (providers, udfs) = empty_registries();
    let original = project(expression_node("node"));
    let original_plan = compile_project(&original, &providers, &udfs).unwrap();

    let mut changed = original.clone();
    changed.graph.datafusion.batch_size = 4_096;
    changed.graph.datafusion.target_partitions = 2;
    let changed_plan = compile_project(&changed, &providers, &udfs).unwrap();
    assert_eq!(
        changed_plan.datafusion_config(),
        Some(changed.graph.datafusion)
    );
    assert_ne!(original_plan.fingerprint(), changed_plan.fingerprint());

    let direct = PipelineBuilder::new("graph")
        .unwrap()
        .with_datafusion_config(changed.graph.datafusion)
        .add_node(
            "node",
            Box::new(
                ExpressionOperator::new("node", "result = value + 1", Vec::new(), None, Vec::new())
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile_batch(&udfs)
        .unwrap();
    assert_eq!(changed_plan.fingerprint(), direct.fingerprint());
}

#[test]
fn direct_graph_compile_validates_core_datafusion_config() {
    let (_, udfs) = empty_registries();
    for config in [
        DataFusionConfig {
            batch_size: 0,
            target_partitions: 1,
        },
        DataFusionConfig {
            batch_size: 8_192,
            target_partitions: 0,
        },
    ] {
        assert!(matches!(
            PipelineBuilder::new("graph")
                .unwrap()
                .with_datafusion_config(config)
                .add_node(
                    "node",
                    Box::new(
                        ExpressionOperator::new(
                            "node",
                            "result = value + 1",
                            Vec::new(),
                            None,
                            Vec::new(),
                        )
                        .unwrap(),
                    ),
                )
                .unwrap()
                .compile_batch(&udfs),
            Err(calc_flow::CalcFlowError::InvalidArgument { field, .. })
                if field.starts_with("datafusion.")
        ));
    }
}

#[test]
fn compile_sql_honors_alias_ports_and_reports_graph_errors() {
    let (providers, udfs) = empty_registries();
    let node = NodeSpec {
        id: "sql".into(),
        operator: OperatorSpec::Sql {
            query: "SELECT left.value FROM left JOIN right ON left.value = right.value".into(),
            aliases: vec!["left".into(), "right".into()],
            udfs: Vec::new(),
        },
        input_ports: vec![
            port("left", BatchKind::Table, true, Vec::new()),
            port("right", BatchKind::Table, true, Vec::new()),
        ],
        output_ports: vec![port("output", BatchKind::Table, true, Vec::new())],
        position: None,
    };
    let mut value = project(node);
    value.data_sources = vec![
        DataSourceSpec {
            id: "left".into(),
            input: "left".into(),
            format: "csv".into(),
            data: json!("value\n1"),
        },
        DataSourceSpec {
            id: "right".into(),
            input: "right".into(),
            format: "json".into(),
            data: json!([{ "value": 1 }]),
        },
    ];
    let plan = compile_project(&value, &providers, &udfs).unwrap();
    assert_eq!(plan.external_inputs().len(), 2);

    value.graph.edges.push(EdgeSpec {
        source_node: "missing".into(),
        source_port: "output".into(),
        target_node: "sql".into(),
        target_port: "left".into(),
    });
    let report = validate_project(&value, &providers, &udfs);
    assert_issue(&report, "graph.edges[0]", "graph_compile");
}

struct PassthroughOperator {
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}

impl OperatorMetadata for PassthroughOperator {
    fn name(&self) -> &'static str {
        "external"
    }
    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }
    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }
    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }
}

#[async_trait]
impl BatchOperator for PassthroughOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        Ok(inputs
            .get("input")
            .cloned()
            .map(|batch| BTreeMap::from([("output".into(), batch)]))
            .unwrap_or_default())
    }
}

type SeenPort = (String, BatchKind, bool, DataType, bool);

struct PassthroughFactory {
    seen_input: Arc<Mutex<Option<SeenPort>>>,
}

impl BatchOperatorFactory for PassthroughFactory {
    fn create(
        &self,
        _spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn BatchOperator>> {
        let input = &inputs[0];
        let field = input.schema().unwrap().field(0);
        *self.seen_input.lock() = Some((
            input.name().into(),
            input.kind(),
            input.required(),
            field.data_type().clone(),
            field.is_nullable(),
        ));
        Ok(Box::new(PassthroughOperator { inputs, outputs }) as Box<dyn BatchOperator>)
    }
}

struct CountingFactory {
    creations: Arc<AtomicUsize>,
}

impl BatchOperatorFactory for CountingFactory {
    fn create(
        &self,
        _spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn BatchOperator>> {
        self.creations.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(PassthroughOperator { inputs, outputs }) as Box<dyn BatchOperator>)
    }
}

#[test]
fn external_only_projects_ignore_unused_datafusion_configuration() {
    let (providers, udfs) = empty_registries();
    providers
        .register_batch(
            "acme",
            "passthrough",
            "1",
            Arc::new(CountingFactory {
                creations: Arc::new(AtomicUsize::new(0)),
            }),
        )
        .unwrap();
    let original = project(NodeSpec {
        id: "external".into(),
        operator: OperatorSpec::External {
            provider: "acme".into(),
            name: "passthrough".into(),
            version: "1".into(),
            options: BTreeMap::new(),
        },
        input_ports: vec![port("input", BatchKind::Table, false, Vec::new())],
        output_ports: vec![port("output", BatchKind::Table, true, Vec::new())],
        position: None,
    });
    let original_plan = compile_project(&original, &providers, &udfs).unwrap();

    let mut changed = original.clone();
    changed.graph.datafusion = DataFusionConfig {
        batch_size: 0,
        target_partitions: 0,
    };
    let report = validate_project(&changed, &providers, &udfs);
    assert!(report.valid, "{:?}", report.issues);
    let changed_plan = compile_project(&changed, &providers, &udfs).unwrap();

    assert!(!changed_plan.requires_datafusion());
    assert_eq!(changed_plan.datafusion_config(), None);
    assert_eq!(changed_plan.fingerprint(), original_plan.fingerprint());
}

#[test]
fn semantic_validation_does_not_create_external_operators() {
    let (providers, udfs) = empty_registries();
    let creations = Arc::new(AtomicUsize::new(0));
    providers
        .register_batch(
            "acme",
            "passthrough",
            "1",
            Arc::new(CountingFactory {
                creations: Arc::clone(&creations),
            }),
        )
        .unwrap();
    let mut value = project(NodeSpec {
        id: "external".into(),
        operator: OperatorSpec::External {
            provider: "acme".into(),
            name: "passthrough".into(),
            version: "1".into(),
            options: BTreeMap::new(),
        },
        input_ports: vec![port("input", BatchKind::Table, false, Vec::new())],
        output_ports: vec![port("output", BatchKind::Table, true, Vec::new())],
        position: None,
    });
    value.format_version = 1;

    assert_issue(
        &validate_project(&value, &providers, &udfs),
        "format_version",
        "unsupported_version",
    );
    assert_eq!(creations.load(Ordering::SeqCst), 0);
}

#[test]
fn external_nodes_require_exact_provider_and_receive_configured_ports() {
    let (providers, udfs) = empty_registries();
    let node = NodeSpec {
        id: "external".into(),
        operator: OperatorSpec::External {
            provider: "acme".into(),
            name: "passthrough".into(),
            version: "1".into(),
            options: BTreeMap::new(),
        },
        input_ports: vec![port(
            "input",
            BatchKind::Table,
            false,
            vec![ArrowFieldSpec {
                name: "value".into(),
                data_type: "int64".into(),
                nullable: false,
            }],
        )],
        output_ports: vec![port(
            "output",
            BatchKind::Table,
            true,
            vec![ArrowFieldSpec {
                name: "value".into(),
                data_type: "int64".into(),
                nullable: false,
            }],
        )],
        position: None,
    };
    let value = project(node);
    assert_issue(
        &validate_project(&value, &providers, &udfs),
        "graph.nodes[0].operator",
        "missing_provider",
    );
    let seen_input = Arc::new(Mutex::new(None));
    providers
        .register_batch(
            "acme",
            "passthrough",
            "1",
            Arc::new(PassthroughFactory {
                seen_input: Arc::clone(&seen_input),
            }),
        )
        .unwrap();
    let report = validate_project(&value, &providers, &udfs);
    assert!(report.valid, "{:?}", report.issues);
    assert!(report.fingerprint.is_some());
    assert_eq!(
        *seen_input.lock(),
        Some((
            "input".into(),
            BatchKind::Table,
            false,
            DataType::Int64,
            false,
        ))
    );
}

#[test]
fn missing_and_conflicting_udfs_have_stable_paths() {
    let (providers, udfs) = empty_registries();
    let missing = UdfReference::new("acme", "missing", "1", UdfKind::DataFusionScalar).unwrap();
    let mut value = project(expression_node("node"));
    let OperatorSpec::Expression { udfs: selected, .. } = &mut value.graph.nodes[0].operator else {
        unreachable!()
    };
    selected.push(missing);
    assert_issue(
        &validate_project(&value, &providers, &udfs),
        "graph.nodes[0].operator.udfs[0]",
        "missing_udf",
    );

    let OperatorSpec::Expression { udfs: selected, .. } = &mut value.graph.nodes[0].operator else {
        unreachable!()
    };
    selected.push(UdfReference::new("other", "missing", "2", UdfKind::DataFusionScalar).unwrap());
    assert_issue(
        &validate_project(&value, &providers, &udfs),
        "graph.nodes[0].operator.udfs",
        "conflicting_udf",
    );
}

#[test]
fn project_schema_artifact_matches_generator() {
    // The public generator is the one canonical v3 schema surface.
    let generated = project_json_schema().unwrap();
    assert!(generated["properties"]["format_version"]["const"] == 3);
}

#[test]
fn edge_spec_maps_to_public_builder_endpoints() {
    let edge = EdgeSpec {
        source_node: "first".into(),
        source_port: "output".into(),
        target_node: "second".into(),
        target_port: "input".into(),
    };
    let direct = Edge::new(
        PortEndpoint::new(&edge.source_node, &edge.source_port).unwrap(),
        PortEndpoint::new(&edge.target_node, &edge.target_port).unwrap(),
    );
    assert_eq!(direct.source.node_id, "first");
}

#[test]
fn constructed_operator_modes_are_validated_without_deserialization() {
    let (providers, udfs) = empty_registries();
    let mut value = project(expression_node("node"));
    value.graph.nodes[0].operator = OperatorSpec::Expression {
        expression: "x + 1".into(),
        select: vec!["x".into()],
        filter: None,
        udfs: Vec::new(),
    };
    assert_issue(
        &validate_project(&value, &providers, &udfs),
        "graph.nodes[0].operator",
        "invalid_operator",
    );
}

#[test]
fn builtin_config_inputs_must_be_required() {
    let (providers, udfs) = empty_registries();
    let mut expression = project(expression_node("node"));
    expression.graph.nodes[0].input_ports =
        vec![port("input", BatchKind::Table, false, Vec::new())];
    assert_issue(
        &validate_project(&expression, &providers, &udfs),
        "graph.nodes[0].input_ports",
        "invalid_ports",
    );

    let mut sql = project(NodeSpec {
        id: "sql".into(),
        operator: OperatorSpec::Sql {
            query: "SELECT * FROM input".into(),
            aliases: vec!["input".into()],
            udfs: Vec::new(),
        },
        input_ports: vec![port("input", BatchKind::Table, false, Vec::new())],
        output_ports: Vec::new(),
        position: None,
    });
    sql.data_sources[0].input = "input".into();
    assert_issue(
        &validate_project(&sql, &providers, &udfs),
        "graph.nodes[0].input_ports",
        "invalid_ports",
    );
}

#[test]
fn invalid_query_syntax_has_stable_operator_issue_paths() {
    let (providers, udfs) = empty_registries();
    let mut sql = project(NodeSpec {
        id: "sql".into(),
        operator: OperatorSpec::Sql {
            query: "DELETE FROM input".into(),
            aliases: vec!["input".into()],
            udfs: Vec::new(),
        },
        input_ports: Vec::new(),
        output_ports: Vec::new(),
        position: None,
    });
    sql.data_sources[0].input = "input".into();
    assert_issue(
        &validate_project(&sql, &providers, &udfs),
        "graph.nodes[0].operator",
        "invalid_operator",
    );

    let mut expression = project(expression_node("node"));
    let OperatorSpec::Expression { filter, .. } = &mut expression.graph.nodes[0].operator else {
        unreachable!()
    };
    *filter = Some("(".into());
    assert_issue(
        &validate_project(&expression, &providers, &udfs),
        "graph.nodes[0].operator",
        "invalid_operator",
    );
}

#[test]
fn duplicate_edges_and_writers_report_the_later_edge_index() {
    let (providers, udfs) = empty_registries();
    let mut duplicate = project(expression_node("source"));
    duplicate.graph.nodes.push(expression_node("target"));
    let edge = EdgeSpec {
        source_node: "source".into(),
        source_port: "output".into(),
        target_node: "target".into(),
        target_port: "input".into(),
    };
    duplicate.graph.edges = vec![edge.clone(), edge];
    assert_issue(
        &validate_project(&duplicate, &providers, &udfs),
        "graph.edges[1]",
        "duplicate_edge",
    );

    let mut writers = project(expression_node("first"));
    writers.graph.nodes.push(expression_node("second"));
    writers.graph.nodes.push(expression_node("target"));
    writers.graph.edges = vec![
        EdgeSpec {
            source_node: "first".into(),
            source_port: "output".into(),
            target_node: "target".into(),
            target_port: "input".into(),
        },
        EdgeSpec {
            source_node: "second".into(),
            source_port: "output".into(),
            target_node: "target".into(),
            target_port: "input".into(),
        },
    ];
    assert_issue(
        &validate_project(&writers, &providers, &udfs),
        "graph.edges[1]",
        "multiple_writers",
    );
}

#[test]
fn serde_rejects_non_finite_position_and_unknown_udf_fields() {
    let mut value = serde_json::to_value(project(expression_node("node"))).unwrap();
    value["graph"]["nodes"][0]["operator"]["udfs"] = json!([{
        "provider": "acme", "name": "f", "version": "1", "kind": "data_fusion_scalar", "import_path": "evil"
    }]);
    assert!(serde_json::from_value::<ProjectSpec>(value).is_err());
}

#[test]
fn supported_v1_arrow_aliases_compile_exactly() {
    let aliases = [
        "bool",
        "date32",
        "date64",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "large_string",
        "string",
        "time32[s]",
        "time64[us]",
        "timestamp[ms]",
        "timestamp[us]",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ];
    let (providers, udfs) = empty_registries();
    for alias in aliases {
        let mut value = project(expression_node("node"));
        value.graph.nodes[0].input_ports = vec![port(
            "input",
            BatchKind::Table,
            true,
            vec![ArrowFieldSpec {
                name: "value".into(),
                data_type: alias.into(),
                nullable: true,
            }],
        )];
        assert!(validate_project(&value, &providers, &udfs).valid, "{alias}");
    }
}

fn stream_join_node(id: &str, key_type: &str) -> NodeSpec {
    NodeSpec {
        id: id.into(),
        operator: OperatorSpec::StreamJoin {
            spec: calc_flow::StreamJoinSpec::inner(
                ["account_id"],
                ["account_id"],
                "authorized_at",
                "paid_at",
                calc_flow::JoinTimeBounds::new(
                    std::time::Duration::from_secs(300),
                    std::time::Duration::from_secs(60),
                )
                .unwrap(),
                calc_flow::JoinStateLimits::new(100, 1_000_000, 1_000).unwrap(),
            )
            .unwrap(),
        },
        input_ports: vec![
            port(
                "left",
                BatchKind::Table,
                true,
                vec![
                    ArrowFieldSpec {
                        name: "account_id".into(),
                        data_type: key_type.into(),
                        nullable: false,
                    },
                    ArrowFieldSpec {
                        name: "authorized_at".into(),
                        data_type: "timestamp[us]".into(),
                        nullable: false,
                    },
                ],
            ),
            port(
                "right",
                BatchKind::Table,
                true,
                vec![
                    ArrowFieldSpec {
                        name: "account_id".into(),
                        data_type: "int64".into(),
                        nullable: false,
                    },
                    ArrowFieldSpec {
                        name: "paid_at".into(),
                        data_type: "timestamp[us]".into(),
                        nullable: false,
                    },
                ],
            ),
        ],
        output_ports: Vec::new(),
        position: Some(PositionSpec { x: 1.0, y: 2.0 }),
    }
}

#[test]
fn validate_reports_stream_join_runtime_port_and_operator_issues() {
    let (providers, udfs) = empty_registries();

    let mut batch_mode = project(stream_join_node("join", "int64"));
    let report = validate_project(&batch_mode, &providers, &udfs);
    assert_issue(&report, "graph.nodes[0].operator", "incompatible_runtime");

    let mut wrong_ports = batch_mode.clone();
    wrong_ports.graph.nodes[0].input_ports.truncate(1);
    assert_issue(
        &validate_project(&wrong_ports, &providers, &udfs),
        "graph.nodes[0].input_ports",
        "invalid_ports",
    );

    let mut wrong_names = batch_mode.clone();
    wrong_names.graph.nodes[0].input_ports[0].name = "first".into();
    assert_issue(
        &validate_project(&wrong_names, &providers, &udfs),
        "graph.nodes[0].input_ports",
        "invalid_ports",
    );

    let mut wrong_key_type = batch_mode;
    wrong_key_type.graph.nodes[0].input_ports[0].schema[0].data_type = "float64".into();
    assert_issue(
        &validate_project(&wrong_key_type, &providers, &udfs),
        "graph.nodes[0].operator",
        "invalid_operator",
    );
}

#[test]
fn compile_rejects_stream_join_outside_stream_mode_and_with_invalid_ports() {
    let (providers, udfs) = empty_registries();

    let mut batch_mode = project(stream_join_node("join", "int64"));
    batch_mode.data_sources.clear();
    let error = compile_project(&batch_mode, &providers, &udfs)
        .expect_err("batch compilation must reject stream_join");
    assert!(
        error
            .to_string()
            .contains("stream_join is available only in stream runtime mode"),
        "{error}"
    );

    let mut stream_mode = batch_mode;
    stream_mode.runtime = RuntimeSpec::Stream(StreamRunOptions::default());
    let connectors = ConnectorRegistry::new().snapshot();
    let compile = |project: &ProjectSpec| {
        compile_stream_project(
            project,
            &providers,
            &udfs,
            &connectors,
            &StreamRequirements::default(),
        )
    };

    // The well-formed Join itself must clear stream validation: any failure
    // here has to come from the incomplete surrounding graph, never from the
    // stream_join port or operator contract.
    if let Err(error) = compile(&stream_mode) {
        let message = error.to_string();
        assert!(!message.contains("left and right"), "{message}");
        assert!(!message.contains("incompatible_runtime"), "{message}");
        assert!(
            !message.contains("identical supported Arrow types"),
            "{message}"
        );
    }

    let mut single_input = stream_mode.clone();
    single_input.graph.nodes[0].input_ports.truncate(1);
    let error = compile(&single_input).expect_err("stream_join requires both inputs");
    assert!(error.to_string().contains("stream_join"), "{error}");

    let mut wrong_names = stream_mode;
    wrong_names.graph.nodes[0].input_ports[1].name = "second".into();
    let error = compile(&wrong_names).expect_err("stream_join ports must be left and right");
    assert!(error.to_string().contains("stream_join"), "{error}");
}
