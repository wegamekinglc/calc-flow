//! Project-v3 connector preflight acceptance tests.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use calc_flow::{
    Batch, ConnectorCapabilities, ConnectorDescriptor, ConnectorFactories, ConnectorIdentity,
    ConnectorKind, ConnectorRegistry, ConnectorSinkFactory, ConnectorSourceFactory,
    DeliveryCapability, DeliveryGuarantee, ManagedCheckpointRuntime, NativeWatermarkCapability,
    ProjectSpec, ProviderRegistry, ReplayCapability, ReplayPositioning, Result, SecretResolver,
    SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema, StreamExecutionPlan,
    StreamRequirements, StreamSink, StreamSource, StreamingRunner, TransactionSupport, UdfRegistry,
    WatermarkSupport, compile_stream_project, compile_stream_project_graph,
};

struct TestSource;

#[async_trait]
impl StreamSource for TestSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 16,
            max_batch_bytes: 4_096,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }

    async fn open(&mut self, _cursor: Option<calc_flow::Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(None)
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct CountingSourceFactory {
    descriptor: ConnectorDescriptor,
    opens: Arc<AtomicUsize>,
}

#[async_trait]
impl ConnectorSourceFactory for CountingSourceFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        _options: &BTreeMap<String, serde_json::Value>,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(TestSource))
    }
}

fn source_descriptor() -> ConnectorDescriptor {
    ConnectorDescriptor {
        identity: ConnectorIdentity::new("test", "source", "1").unwrap(),
        kind: ConnectorKind::Source,
        capabilities: ConnectorCapabilities {
            delivery: DeliveryCapability::AtLeastOnce,
            replay: ReplayCapability::ReplayableExact,
            watermark: WatermarkSupport::GeneratedOnly,
            transaction: TransactionSupport::None,
            snapshot: true,
            polling: false,
            cdc: false,
            lookup: false,
        },
        formats: Vec::new(),
        config_schema: BTreeMap::new(),
        secret_slots: BTreeSet::from(["url".to_string()]),
        required_secret_slots: BTreeSet::from(["url".to_string()]),
    }
}

struct TestSink;

#[async_trait]
impl StreamSink for TestSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, _batch: &Batch) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct CountingSinkFactory {
    descriptor: ConnectorDescriptor,
    opens: Arc<AtomicUsize>,
}

#[async_trait]
impl ConnectorSinkFactory for CountingSinkFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        _options: &BTreeMap<String, serde_json::Value>,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSink>> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(TestSink))
    }
}

fn sink_descriptor() -> ConnectorDescriptor {
    ConnectorDescriptor {
        identity: ConnectorIdentity::new("test", "sink", "1").unwrap(),
        kind: ConnectorKind::Sink,
        capabilities: ConnectorCapabilities {
            delivery: DeliveryCapability::AtLeastOnce,
            replay: ReplayCapability::Unreplayable,
            watermark: WatermarkSupport::GeneratedOnly,
            transaction: TransactionSupport::None,
            snapshot: false,
            polling: false,
            cdc: false,
            lookup: false,
        },
        formats: Vec::new(),
        config_schema: BTreeMap::new(),
        secret_slots: BTreeSet::new(),
        required_secret_slots: BTreeSet::new(),
    }
}

fn stream_project(include_secret: bool) -> ProjectSpec {
    let secrets = include_secret
        .then(|| serde_json::json!({"url": {"resolver": "environment", "key": "UNUSED_TEST_URL"}}));
    serde_json::from_value(serde_json::json!({
        "format_version": 3,
        "id": "connector-project",
        "name": "Connector project",
        "runtime": {"mode": "stream", "options": {}},
        "graph": {
            "name": "connector-project",
            "nodes": [{
                "id": "calc",
                "operator": {"kind": "expression", "expression": "b = a + 1"}
            }]
        },
        "sources": [{
            "binding": "input",
            "connector": {"provider": "test", "name": "source", "version": "1"},
            "secrets": secrets.unwrap_or_else(|| serde_json::json!({})),
            "watermark": {"policy": "disabled"}
        }],
        "sinks": [{
            "binding": "output",
            "connector": {"provider": "test", "name": "sink", "version": "1"},
            "delivery": "at_least_once"
        }]
    }))
    .unwrap()
}

fn exactly_once_project() -> ProjectSpec {
    let mut value = serde_json::to_value(stream_project(true)).unwrap();
    value["sinks"][0]["delivery"] = serde_json::json!("exactly_once");
    serde_json::from_value(value).unwrap()
}

#[test]
fn stream_compile_validates_required_secret_slots_without_opening_factory() {
    let opens = Arc::new(AtomicUsize::new(0));
    let descriptor = source_descriptor();
    let mut connectors = ConnectorRegistry::new();
    connectors
        .register_connector(
            descriptor.clone(),
            ConnectorFactories::source_only(Arc::new(CountingSourceFactory {
                descriptor,
                opens: Arc::clone(&opens),
            })),
        )
        .unwrap();

    let error = compile_stream_project(
        &stream_project(false),
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &connectors.snapshot(),
        &StreamRequirements::default(),
    )
    .expect_err("missing required connector secret must fail preflight");

    assert!(
        error.to_string().contains("sources[0].secrets.url"),
        "{error}"
    );
    assert_eq!(opens.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn compiled_project_owns_deferred_connector_bindings_until_runner_start() {
    let source_opens = Arc::new(AtomicUsize::new(0));
    let sink_opens = Arc::new(AtomicUsize::new(0));
    let source_descriptor = source_descriptor();
    let sink_descriptor = sink_descriptor();
    let mut connectors = ConnectorRegistry::new();
    connectors
        .register_connector(
            source_descriptor.clone(),
            ConnectorFactories::source_only(Arc::new(CountingSourceFactory {
                descriptor: source_descriptor,
                opens: Arc::clone(&source_opens),
            })),
        )
        .unwrap();
    connectors
        .register_connector(
            sink_descriptor.clone(),
            ConnectorFactories::sink_only(Arc::new(CountingSinkFactory {
                descriptor: sink_descriptor,
                opens: Arc::clone(&sink_opens),
            })),
        )
        .unwrap();

    let plan = compile_stream_project(
        &stream_project(true),
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &connectors.snapshot(),
        &StreamRequirements::default(),
    )
    .unwrap();
    assert!(plan.has_project_bindings());
    assert_eq!(source_opens.load(Ordering::SeqCst), 0);
    assert_eq!(sink_opens.load(Ordering::SeqCst), 0);

    let directory = tempfile::tempdir().unwrap();
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::new(),
        BTreeMap::new(),
        ManagedCheckpointRuntime::new(directory.path()).unwrap(),
    )
    .unwrap();
    assert_eq!(source_opens.load(Ordering::SeqCst), 0);
    assert_eq!(sink_opens.load(Ordering::SeqCst), 0);

    let job = runner.start().await.unwrap();
    let _outcome = job.wait().await;
    assert_eq!(source_opens.load(Ordering::SeqCst), 1);
    assert_eq!(sink_opens.load(Ordering::SeqCst), 1);
}

#[test]
fn exactly_once_preflight_lists_every_incapable_connector_path_without_opening() {
    let source_opens = Arc::new(AtomicUsize::new(0));
    let sink_opens = Arc::new(AtomicUsize::new(0));
    let mut source_descriptor = source_descriptor();
    source_descriptor.capabilities.replay = ReplayCapability::Unreplayable;
    source_descriptor.capabilities.delivery = DeliveryCapability::BestEffort;
    let sink_descriptor = sink_descriptor();
    let mut connectors = ConnectorRegistry::new();
    connectors
        .register_connector(
            source_descriptor.clone(),
            ConnectorFactories::source_only(Arc::new(CountingSourceFactory {
                descriptor: source_descriptor,
                opens: Arc::clone(&source_opens),
            })),
        )
        .unwrap();
    connectors
        .register_connector(
            sink_descriptor.clone(),
            ConnectorFactories::sink_only(Arc::new(CountingSinkFactory {
                descriptor: sink_descriptor,
                opens: Arc::clone(&sink_opens),
            })),
        )
        .unwrap();
    let requirements = StreamRequirements {
        delivery: BTreeMap::from([("output".into(), DeliveryGuarantee::ExactlyOnce)]),
    };

    let error = compile_stream_project(
        &exactly_once_project(),
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &connectors.snapshot(),
        &requirements,
    )
    .expect_err("incapable connector path must fail exactly-once preflight");
    let message = error.to_string();
    assert!(
        message.contains("sources[0]") && message.contains("replay"),
        "{message}"
    );
    assert!(
        message.contains("sinks[0]") && message.contains("transaction"),
        "{message}"
    );
    assert_eq!(source_opens.load(Ordering::SeqCst), 0);
    assert_eq!(sink_opens.load(Ordering::SeqCst), 0);
}

fn compile_test_registry() -> ConnectorRegistry {
    let source_descriptor = source_descriptor();
    let sink_descriptor = sink_descriptor();
    let mut connectors = ConnectorRegistry::new();
    connectors
        .register_connector(
            source_descriptor.clone(),
            ConnectorFactories::source_only(Arc::new(CountingSourceFactory {
                descriptor: source_descriptor,
                opens: Arc::new(AtomicUsize::new(0)),
            })),
        )
        .unwrap();
    connectors
        .register_connector(
            sink_descriptor.clone(),
            ConnectorFactories::sink_only(Arc::new(CountingSinkFactory {
                descriptor: sink_descriptor,
                opens: Arc::new(AtomicUsize::new(0)),
            })),
        )
        .unwrap();
    connectors
}

#[test]
fn project_v3_compiles_builtin_window_and_union_nodes() {
    let connectors = compile_test_registry();
    let mut window = serde_json::to_value(stream_project(true)).unwrap();
    window["graph"]["nodes"][0]["operator"] = serde_json::json!({
        "kind": "window",
        "spec": {
            "event_time_column": "event_time",
            "group_by": ["account"],
            "geometry": {"kind": "tumbling", "size_micros": 60_000_000},
            "aggregates": [
                {"function": "sum", "column": "amount", "output": "total"}
            ]
        }
    });
    window["graph"]["nodes"][0]["input_ports"] = serde_json::json!([{
        "name": "input",
        "kind": "table",
        "required": true,
        "schema": [
            {"name": "event_time", "data_type": "timestamp[us]", "nullable": false},
            {"name": "account", "data_type": "string", "nullable": false},
            {"name": "amount", "data_type": "int64", "nullable": false}
        ]
    }]);
    let window: ProjectSpec = serde_json::from_value(window).expect("window project parses");
    compile_stream_project(
        &window,
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &connectors.snapshot(),
        &StreamRequirements::default(),
    )
    .expect("window project compiles");

    let mut union = serde_json::to_value(stream_project(true)).unwrap();
    union["graph"]["nodes"][0]["operator"] = serde_json::json!({"kind": "union"});
    union["graph"]["nodes"][0]["input_ports"] = serde_json::json!([
        {"name": "left", "kind": "table", "required": true},
        {"name": "right", "kind": "table", "required": true}
    ]);
    union["sources"][0]["binding"] = serde_json::json!("left");
    let mut right = union["sources"][0].clone();
    right["binding"] = serde_json::json!("right");
    union["sources"].as_array_mut().unwrap().push(right);
    let union: ProjectSpec = serde_json::from_value(union).expect("union project parses");
    compile_stream_project(
        &union,
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &connectors.snapshot(),
        &StreamRequirements::default(),
    )
    .expect("union project compiles");
}

#[test]
fn project_v3_rejects_window_in_batch_runtime() {
    let value = serde_json::json!({
        "format_version": 3,
        "id": "batch-window",
        "name": "Batch window",
        "runtime": {"mode": "batch", "options": {}},
        "graph": {
            "name": "batch-window",
            "nodes": [{
                "id": "window",
                "operator": {
                    "kind": "window",
                    "spec": {
                        "event_time_column": "event_time",
                        "group_by": [],
                        "geometry": {"kind": "tumbling", "size_micros": 1_000_000},
                        "aggregates": [
                            {"function": "count", "column": "value", "output": "count"}
                        ]
                    }
                },
                "input_ports": [{
                    "name": "input",
                    "kind": "table",
                    "schema": [
                        {"name": "event_time", "data_type": "timestamp[us]", "nullable": false},
                        {"name": "value", "data_type": "int64", "nullable": false}
                    ]
                }]
            }]
        }
    });
    let project: ProjectSpec = serde_json::from_value(value).expect("document parses");
    let report = calc_flow::validate_project(
        &project,
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
    );
    assert!(
        report
            .issues
            .iter()
            .any(|issue| issue.code == "incompatible_runtime"),
        "{:?}",
        report.issues
    );
}

mod static_input_declarations {
    use super::*;
    use datafusion::arrow::{
        array::{DictionaryArray, Int8Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    fn static_project(static_inputs: serde_json::Value) -> ProjectSpec {
        let mut value = serde_json::to_value(stream_project(true)).unwrap();
        value["graph"]["nodes"] = serde_json::json!([{
            "id": "merge",
            "operator": {"kind": "union"},
            "input_ports": [{"name": "main", "kind": "table"}, {"name": "weights", "kind": "table"}]
        }]);
        value["sources"][0]["binding"] = serde_json::json!("main");
        if static_inputs.is_null() {
            value.as_object_mut().unwrap().remove("static_inputs");
        } else {
            value["static_inputs"] = static_inputs;
        }
        serde_json::from_value(value).unwrap()
    }

    fn registered_connectors_with_opens(
        source_opens: Arc<AtomicUsize>,
        sink_opens: Arc<AtomicUsize>,
    ) -> calc_flow::ConnectorRegistrySnapshot {
        let mut connectors = ConnectorRegistry::new();
        let source = source_descriptor();
        connectors
            .register_connector(
                source.clone(),
                ConnectorFactories::source_only(Arc::new(CountingSourceFactory {
                    descriptor: source,
                    opens: source_opens,
                })),
            )
            .unwrap();
        let sink = sink_descriptor();
        connectors
            .register_connector(
                sink.clone(),
                ConnectorFactories::sink_only(Arc::new(CountingSinkFactory {
                    descriptor: sink,
                    opens: sink_opens,
                })),
            )
            .unwrap();
        connectors.snapshot()
    }

    fn registered_connectors() -> calc_flow::ConnectorRegistrySnapshot {
        registered_connectors_with_opens(
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        )
    }

    fn table_declaration() -> serde_json::Value {
        serde_json::json!([{
            "kind": "table",
            "name": "weights",
            "mutability": "static",
            "schema": [{"name": "factor", "data_type": "float64", "nullable": false}]
        }])
    }

    fn dictionary_declaration(ordered: bool) -> serde_json::Value {
        serde_json::json!([{
            "kind": "table",
            "name": "weights",
            "mutability": "static",
            "schema": [{
                "name": "color",
                "data_type": format!(
                    "dictionary<index=int8;value=string;ordered={ordered}>"
                ),
                "nullable": true
            }]
        }])
    }

    fn dictionary_table(ordered: bool) -> Batch {
        let dictionary = DictionaryArray::new(
            Int8Array::from(vec![Some(0_i8), None, Some(1_i8), Some(0_i8)]),
            Arc::new(StringArray::from(vec!["red", "blue"])),
        );
        let field = Field::new(
            "color",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            true,
        )
        .with_dict_is_ordered(ordered);
        Batch::table(
            vec![
                RecordBatch::try_new(
                    Arc::new(Schema::new(vec![field])),
                    vec![Arc::new(dictionary)],
                )
                .unwrap(),
            ],
            calc_flow::BatchMetadata::default(),
        )
        .unwrap()
    }

    fn compile(project: &ProjectSpec) -> Result<StreamExecutionPlan> {
        compile_stream_project(
            project,
            &ProviderRegistry::default(),
            &UdfRegistry::new().snapshot(),
            &registered_connectors(),
            &StreamRequirements::default(),
        )
    }

    fn compile_graph(project: &ProjectSpec) -> Result<StreamExecutionPlan> {
        let mut project = project.clone();
        project.sources.clear();
        project.sinks.clear();
        compile_stream_project_graph(
            &project,
            &ProviderRegistry::default(),
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
    }

    #[test]
    fn static_declarations_attach_to_the_plan_and_leave_source_bindings_alone() {
        let plan = compile(&static_project(table_declaration())).unwrap();
        assert_eq!(plan.static_input_ids(), vec!["weights"]);
        assert_eq!(plan.source_binding_ids(), vec!["main"]);
        assert_eq!(
            plan.static_inputs()["weights"].name(),
            "weights",
            "the map is keyed by the declared name"
        );
    }

    #[test]
    fn static_declarations_participate_in_the_semantic_fingerprint() {
        let float64_project = static_project(table_declaration());
        let mut int64_declaration = table_declaration();
        int64_declaration[0]["schema"][0]["data_type"] = serde_json::json!("int64");
        let int64_project = static_project(int64_declaration);

        let float64 = compile(&float64_project).unwrap();
        let int64 = compile(&int64_project).unwrap();
        assert_ne!(
            float64.fingerprint(),
            int64.fingerprint(),
            "connector-backed declarations must change the plan fingerprint"
        );

        let float64 = compile_graph(&float64_project).unwrap();
        let int64 = compile_graph(&int64_project).unwrap();
        assert_ne!(
            float64.fingerprint(),
            int64.fingerprint(),
            "graph-only declarations must change the plan fingerprint"
        );
    }

    #[test]
    fn static_declarations_reject_unknown_source_conflicting_and_duplicate_names() {
        let unknown = compile(&static_project(serde_json::json!([{
            "kind": "array", "name": "nope", "mutability": "static",
            "backend": "numpy", "dtype": "float32", "shape": [1]
        }])))
        .unwrap_err();
        assert!(unknown.to_string().contains("unknown_binding"), "{unknown}");

        let conflict = compile(&static_project(serde_json::json!([{
            "kind": "array", "name": "main", "mutability": "static",
            "backend": "numpy", "dtype": "float32", "shape": [1]
        }])))
        .unwrap_err();
        assert!(
            conflict.to_string().contains("source_binding_conflict"),
            "{conflict}"
        );

        let duplicate = compile(&static_project(serde_json::json!([
            {"kind": "array", "name": "weights", "mutability": "static",
             "backend": "numpy", "dtype": "float32", "shape": [1]},
            {"kind": "array", "name": "weights", "mutability": "static",
             "backend": "numpy", "dtype": "float32", "shape": [1]}
        ])))
        .unwrap_err();
        assert!(
            duplicate.to_string().contains("duplicate_name"),
            "{duplicate}"
        );
    }

    #[test]
    fn static_declarations_reject_unsupported_array_dtypes() {
        let error = compile(&static_project(serde_json::json!([{
            "kind": "array", "name": "weights", "mutability": "static",
            "backend": "numpy", "dtype": "complex128", "shape": [1]
        }])))
        .unwrap_err();
        assert!(error.to_string().contains("unsupported_dtype"), "{error}");
    }

    #[test]
    fn static_declarations_are_strict_about_unknown_fields() {
        let mut document = serde_json::to_value(static_project(serde_json::json!([]))).unwrap();
        document["static_inputs"] = serde_json::json!([{
            "kind": "array", "name": "weights", "mutability": "static",
            "backend": "numpy", "dtype": "float32", "shape": [1], "payload": {}
        }]);
        assert!(serde_json::from_value::<ProjectSpec>(document).is_err());
    }

    #[test]
    fn dictionary_declarations_reject_aliases_and_nested_values_at_the_frozen_path() {
        for spelling in [
            "dictionary<int8,string,false>",
            "dictionary<index=int8;value=dictionary<index=int8;value=string;ordered=false>;ordered=false>",
        ] {
            let declaration = serde_json::json!([{
                "kind": "table",
                "name": "weights",
                "mutability": "static",
                "schema": [{"name": "color", "data_type": spelling, "nullable": true}]
            }]);
            let error = compile(&static_project(declaration)).unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains("static_inputs[0].schema[0].data_type [unsupported_arrow_type]"),
                "{message}"
            );
        }
    }

    #[tokio::test]
    async fn canonical_dictionary_declaration_passes_compiled_runner_preflight() {
        let source_opens = Arc::new(AtomicUsize::new(0));
        let sink_opens = Arc::new(AtomicUsize::new(0));
        let connectors =
            registered_connectors_with_opens(Arc::clone(&source_opens), Arc::clone(&sink_opens));
        let plan = compile_stream_project(
            &static_project(dictionary_declaration(false)),
            &ProviderRegistry::default(),
            &UdfRegistry::new().snapshot(),
            &connectors,
            &StreamRequirements::default(),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let job = StreamingRunner::new(
            plan,
            BTreeMap::new(),
            BTreeMap::new(),
            ManagedCheckpointRuntime::new(directory.path()).unwrap(),
        )
        .unwrap()
        .with_static_inputs(BTreeMap::from([(
            "weights".into(),
            dictionary_table(false),
        )]))
        .unwrap()
        .start()
        .await
        .unwrap();

        let _outcome = job.wait().await;
        assert_eq!(source_opens.load(Ordering::SeqCst), 1);
        assert_eq!(sink_opens.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dictionary_ordered_mismatch_fails_before_connectors_open() {
        let source_opens = Arc::new(AtomicUsize::new(0));
        let sink_opens = Arc::new(AtomicUsize::new(0));
        let connectors =
            registered_connectors_with_opens(Arc::clone(&source_opens), Arc::clone(&sink_opens));
        let plan = compile_stream_project(
            &static_project(dictionary_declaration(true)),
            &ProviderRegistry::default(),
            &UdfRegistry::new().snapshot(),
            &connectors,
            &StreamRequirements::default(),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let error = StreamingRunner::new(
            plan,
            BTreeMap::new(),
            BTreeMap::new(),
            ManagedCheckpointRuntime::new(directory.path()).unwrap(),
        )
        .unwrap()
        .with_static_inputs(BTreeMap::from([(
            "weights".into(),
            dictionary_table(false),
        )]))
        .unwrap()
        .start()
        .await
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("static_inputs.weights.schema[0].data_type"),
            "{error}"
        );
        assert_eq!(source_opens.load(Ordering::SeqCst), 0);
        assert_eq!(sink_opens.load(Ordering::SeqCst), 0);
    }
}
