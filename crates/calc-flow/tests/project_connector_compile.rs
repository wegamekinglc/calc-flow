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
    SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema, StreamRequirements,
    StreamSink, StreamSource, StreamingRunner, TransactionSupport, UdfRegistry, WatermarkSupport,
    compile_stream_project,
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
