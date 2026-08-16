//! Integration tests for the M6.1 core connector registry, capability
//! vocabulary, secret references, format bounds, and delivery derivation.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use calc_flow::{
    CalcFlowError, ConnectorCapabilities, ConnectorDescriptor, ConnectorError, ConnectorFactories,
    ConnectorIdentity, ConnectorKind, ConnectorOperation, ConnectorRegistry,
    ConnectorRegistrySnapshot, ConnectorSinkFactory, ConnectorSourceFactory, DecodeBounds,
    DeliveryCapability, DeliveryGuarantee, DeliveryParticipant, FormatDescriptor, FormatIdentity,
    NativeWatermarkCapability, ParticipantRole, ReplayCapability, ReplayPositioning,
    RetentionClass, SecretHandle, SecretReference, SecretResolver, SecretResolverKind,
    SinkDelivery, SourceDeliveryCapability, StreamSink, StreamSource, TransactionSupport,
    WatermarkSupport, validate_connector_options, validate_delivery_guarantee,
};
use serde_json::Value;

const PROVIDER: &str = "calc-flow-connectors";

fn identity(name: &str) -> ConnectorIdentity {
    ConnectorIdentity::new(PROVIDER, name, "2.0.0").expect("valid identity")
}

fn format_identity(name: &str) -> FormatIdentity {
    FormatIdentity::new(name, "1").expect("valid format identity")
}

fn source_capabilities() -> ConnectorCapabilities {
    ConnectorCapabilities {
        delivery: DeliveryCapability::AtLeastOnce,
        replay: ReplayCapability::ReplayableExact,
        watermark: WatermarkSupport::GeneratedOnly,
        transaction: TransactionSupport::None,
        snapshot: true,
        polling: false,
        cdc: false,
        lookup: false,
    }
}

fn transactional_sink_capabilities() -> ConnectorCapabilities {
    ConnectorCapabilities {
        delivery: DeliveryCapability::ExactlyOnce,
        replay: ReplayCapability::Unreplayable,
        watermark: WatermarkSupport::GeneratedOnly,
        transaction: TransactionSupport::PreCommitCommit,
        snapshot: false,
        polling: false,
        cdc: false,
        lookup: false,
    }
}

fn descriptor(
    name: &str,
    kind: ConnectorKind,
    capabilities: ConnectorCapabilities,
) -> ConnectorDescriptor {
    ConnectorDescriptor {
        identity: identity(name),
        kind,
        capabilities,
        formats: vec![format_identity("csv")],
        config_schema: [("path".to_string(), Value::String("string".into()))]
            .into_iter()
            .collect(),
        secret_slots: ["token".to_string()].into_iter().collect(),
    }
}

struct FakeSourceFactory {
    descriptor: ConnectorDescriptor,
}

#[async_trait]
impl ConnectorSourceFactory for FakeSourceFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        _options: &calc_flow::JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> calc_flow::Result<Box<dyn StreamSource>> {
        Err(CalcFlowError::NotFound {
            resource: "source".into(),
            key: "fake".into(),
        })
    }
}

struct FakeSinkFactory {
    descriptor: ConnectorDescriptor,
}

#[async_trait]
impl ConnectorSinkFactory for FakeSinkFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        _options: &calc_flow::JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> calc_flow::Result<Box<dyn StreamSink>> {
        Err(CalcFlowError::NotFound {
            resource: "sink".into(),
            key: "fake".into(),
        })
    }
}

fn registry_with_file_connector() -> ConnectorRegistry {
    let mut registry = ConnectorRegistry::new();
    registry
        .register_connector(
            descriptor("file", ConnectorKind::Source, source_capabilities()),
            ConnectorFactories::source_only(Arc::new(FakeSourceFactory {
                descriptor: descriptor("file", ConnectorKind::Source, source_capabilities()),
            })),
        )
        .expect("first registration succeeds");
    registry
}

#[test]
fn duplicate_connector_identity_fails_atomically() {
    let mut registry = registry_with_file_connector();
    let duplicate = registry
        .register_connector(
            descriptor("file", ConnectorKind::Source, source_capabilities()),
            ConnectorFactories::source_only(Arc::new(FakeSourceFactory {
                descriptor: descriptor("file", ConnectorKind::Source, source_capabilities()),
            })),
        )
        .expect_err("duplicate identity rejected");
    match duplicate {
        CalcFlowError::Conflict { resource, key } => {
            assert!(
                resource.contains("connector"),
                "resource names connector: {resource}"
            );
            assert!(
                key.contains(PROVIDER) && key.contains("file"),
                "key carries identity: {key}"
            );
        }
        other => panic!("expected Conflict, got {other:?}"),
    }
    // The failed registration left the registry usable: a different connector registers.
    registry
        .register_connector(
            descriptor(
                "kafka",
                ConnectorKind::Sink,
                transactional_sink_capabilities(),
            ),
            ConnectorFactories::sink_only(Arc::new(FakeSinkFactory {
                descriptor: descriptor(
                    "kafka",
                    ConnectorKind::Sink,
                    transactional_sink_capabilities(),
                ),
            })),
        )
        .expect("registry unchanged after failed registration");
}

#[test]
fn registration_kind_must_match_factories() {
    let mut registry = ConnectorRegistry::new();
    let error = registry
        .register_connector(
            descriptor("file", ConnectorKind::Source, source_capabilities()),
            ConnectorFactories::sink_only(Arc::new(FakeSinkFactory {
                descriptor: descriptor("file", ConnectorKind::Source, source_capabilities()),
            })),
        )
        .expect_err("source kind without source factory rejected");
    assert!(matches!(error, CalcFlowError::InvalidArgument { .. }));
}

#[test]
fn source_kind_rejects_extra_sink_factory() {
    let mut registry = ConnectorRegistry::new();
    let error = registry
        .register_connector(
            descriptor("file", ConnectorKind::Source, source_capabilities()),
            ConnectorFactories::both(
                Arc::new(FakeSourceFactory {
                    descriptor: descriptor("file", ConnectorKind::Source, source_capabilities()),
                }),
                Arc::new(FakeSinkFactory {
                    descriptor: descriptor("file", ConnectorKind::Source, source_capabilities()),
                }),
            ),
        )
        .expect_err("source kind must not carry a sink factory");
    assert!(matches!(error, CalcFlowError::InvalidArgument { .. }));
    assert!(
        ConnectorRegistry::new()
            .snapshot()
            .resolve_source(&identity("file"))
            .is_err(),
        "failed registration leaves no connector behind"
    );
}

#[test]
fn same_connector_slot_with_different_version_conflicts() {
    let mut registry = registry_with_file_connector();
    let upgraded = ConnectorDescriptor {
        identity: ConnectorIdentity::new(PROVIDER, "file", "2.1.0").expect("valid identity"),
        ..descriptor("file", ConnectorKind::Source, source_capabilities())
    };
    let error = registry
        .register_connector(
            upgraded,
            ConnectorFactories::source_only(Arc::new(FakeSourceFactory {
                descriptor: descriptor("file", ConnectorKind::Source, source_capabilities()),
            })),
        )
        .expect_err("one slot accepts one version label");
    match error {
        CalcFlowError::Conflict { resource, key } => {
            assert!(resource.contains("connector"), "resource: {resource}");
            assert!(
                key.contains(PROVIDER) && key.contains("file"),
                "key names the occupied slot: {key}"
            );
        }
        other => panic!("expected Conflict, got {other:?}"),
    }
    let snapshot = registry.snapshot();
    assert!(
        snapshot
            .resolve_source(&ConnectorIdentity::new(PROVIDER, "file", "2.0.0").expect("identity"))
            .is_ok(),
        "the originally registered version stays resolvable"
    );
    assert!(
        registry
            .snapshot()
            .resolve_source(&ConnectorIdentity::new(PROVIDER, "file", "2.1.0").expect("identity"))
            .is_err(),
        "the rejected version never registers"
    );
}

#[test]
fn duplicate_format_identity_fails() {
    let mut registry = ConnectorRegistry::new();
    registry
        .register_format(FormatDescriptor {
            identity: format_identity("csv"),
        })
        .expect("first format registration succeeds");
    let error = registry
        .register_format(FormatDescriptor {
            identity: format_identity("csv"),
        })
        .expect_err("duplicate format rejected");
    assert!(matches!(error, CalcFlowError::Conflict { .. }));
}

#[test]
fn unknown_connector_fails_resolution_before_construction() {
    let snapshot = registry_with_file_connector().snapshot();
    let error = match snapshot.resolve_source(&identity("kafka")) {
        Ok(_) => panic!("unknown identity must fail"),
        Err(error) => error,
    };
    match error {
        CalcFlowError::NotFound { resource, key } => {
            assert!(resource.contains("connector"));
            assert!(key.contains("kafka"), "error carries identity: {key}");
        }
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[test]
fn resolving_sink_kind_through_source_fails() {
    let mut registry = registry_with_file_connector();
    registry
        .register_format(FormatDescriptor {
            identity: format_identity("csv"),
        })
        .expect("format registration succeeds");
    let snapshot = registry.snapshot();
    assert!(
        snapshot.resolve_sink(&identity("file")).is_err(),
        "source-only connector cannot resolve a sink factory"
    );
}

#[test]
fn unknown_format_fails_resolution() {
    let snapshot = ConnectorRegistry::new().snapshot();
    let error = snapshot
        .resolve_format(&format_identity("parquet"))
        .expect_err("unknown format fails");
    assert!(matches!(error, CalcFlowError::NotFound { .. }));
}

#[test]
fn snapshot_ignores_post_snapshot_registration() {
    let registry = registry_with_file_connector();
    let snapshot = registry.snapshot();
    let mut mutated = registry;
    mutated
        .register_connector(
            descriptor("http", ConnectorKind::Source, source_capabilities()),
            ConnectorFactories::source_only(Arc::new(FakeSourceFactory {
                descriptor: descriptor("http", ConnectorKind::Source, source_capabilities()),
            })),
        )
        .expect("later registration succeeds on the registry");
    assert!(
        snapshot.resolve_source(&identity("http")).is_err(),
        "snapshot never observes registrations made after capture"
    );
    assert!(
        snapshot.resolve_source(&identity("file")).is_ok(),
        "snapshot keeps its captured connectors"
    );
}

#[test]
fn snapshot_lists_identities_deterministically() {
    let mut registry = ConnectorRegistry::new();
    for name in ["websocket", "file", "http"] {
        registry
            .register_connector(
                descriptor(name, ConnectorKind::Source, source_capabilities()),
                ConnectorFactories::source_only(Arc::new(FakeSourceFactory {
                    descriptor: descriptor(name, ConnectorKind::Source, source_capabilities()),
                })),
            )
            .expect("registration succeeds");
    }
    let names: Vec<String> = registry
        .snapshot()
        .identities()
        .iter()
        .map(|identity| identity.name.to_string())
        .collect();
    assert_eq!(names, vec!["file", "http", "websocket"]);
}

#[test]
fn connector_identity_rejects_empty_components() {
    assert!(ConnectorIdentity::new("", "file", "1").is_err());
    assert!(ConnectorIdentity::new(PROVIDER, "", "1").is_err());
    assert!(ConnectorIdentity::new(PROVIDER, "file", "").is_err());
}

#[test]
fn secret_handle_redacts_debug_and_display() {
    let handle = SecretHandle::from_bytes(b"super-secret-value");
    assert_eq!(format!("{handle:?}"), "<redacted secret>");
    assert_eq!(format!("{handle}"), "<redacted secret>");
    assert_eq!(handle.expose(), b"super-secret-value");
}

#[test]
fn secret_reference_serializes_deterministically() {
    let reference = SecretReference::new(SecretResolverKind::Environment, "CALC_FLOW_TEST_TOKEN")
        .expect("valid reference");
    let encoded = serde_json::to_value(&reference).expect("reference serializes");
    assert_eq!(
        encoded,
        serde_json::json!({"key": "CALC_FLOW_TEST_TOKEN", "resolver": "environment"})
    );
    let round: SecretReference = serde_json::from_value(encoded).expect("reference round-trips");
    assert_eq!(round, reference);
    assert!(SecretReference::new(SecretResolverKind::Environment, "").is_err());
}

struct StaticResolver {
    value: Vec<u8>,
}

impl SecretResolver for StaticResolver {
    fn resolve(&self, reference: &SecretReference) -> calc_flow::Result<SecretHandle> {
        if reference.key == "known" {
            Ok(SecretHandle::from_bytes(&self.value))
        } else {
            Err(CalcFlowError::NotFound {
                resource: "secret".into(),
                key: reference.key.to_string(),
            })
        }
    }
}

#[test]
fn secret_resolver_returns_handle_or_not_found() {
    let resolver = StaticResolver {
        value: b"token-value".to_vec(),
    };
    let known =
        SecretReference::new(SecretResolverKind::Registered, "known").expect("valid reference");
    let handle = resolver.resolve(&known).expect("known reference resolves");
    assert_eq!(handle.expose(), b"token-value");
    let unknown =
        SecretReference::new(SecretResolverKind::Registered, "missing").expect("valid reference");
    assert!(resolver.resolve(&unknown).is_err());
}

#[test]
fn environment_resolver_fails_closed_for_absent_key() {
    let resolver = calc_flow::EnvironmentSecretResolver;
    let reference = SecretReference::new(
        SecretResolverKind::Environment,
        "CALC_FLOW_ABSENT_SECRET_7F3A",
    )
    .expect("valid reference");
    let error = resolver
        .resolve(&reference)
        .expect_err("absent environment secret fails");
    assert!(matches!(error, CalcFlowError::NotFound { .. }));
}

#[test]
fn connector_options_reject_secret_slot_and_unknown_keys() {
    let file_descriptor = descriptor("file", ConnectorKind::Source, source_capabilities());
    let mut options = BTreeMap::new();
    options.insert("path".to_string(), Value::String("data/".into()));
    validate_connector_options(&file_descriptor, &options).expect("known option accepted");

    let mut with_secret = options.clone();
    with_secret.insert("token".to_string(), Value::String("literal".into()));
    let error = validate_connector_options(&file_descriptor, &with_secret)
        .expect_err("secret slot must come through a reference");
    match error {
        CalcFlowError::InvalidArgument { field, message } => {
            assert!(field.contains("token"), "field names the slot: {field}");
            assert!(
                message.contains("secret"),
                "message explains the rule: {message}"
            );
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }

    let mut unknown = options.clone();
    unknown.insert("typo".to_string(), Value::Bool(true));
    assert!(validate_connector_options(&file_descriptor, &unknown).is_err());
}

#[test]
fn decode_bounds_reject_zero_limits_and_oversized_expansion() {
    assert!(DecodeBounds::new(0, 1024).is_err());
    assert!(DecodeBounds::new(10, 0).is_err());
    let bounds = DecodeBounds::new(10, 1_024).expect("valid bounds");
    bounds
        .check(&format_identity("csv"), 10, 1_024)
        .expect("at-limit expansion accepted");
    let rows_error = bounds
        .check(&format_identity("csv"), 11, 512)
        .expect_err("row overage rejected");
    let message = rows_error.to_string();
    assert!(message.contains("csv"), "error names the format: {message}");
    assert!(message.contains("rows"), "error names the bound: {message}");
    let bytes_error = bounds
        .check(&format_identity("csv"), 5, 1_025)
        .expect_err("byte overage rejected");
    assert!(bytes_error.to_string().contains("bytes"));
}

#[test]
fn capability_axes_are_independent() {
    let base = source_capabilities();
    let mut transactional = base;
    assert_ne!(base, transactional_sink_capabilities());
    transactional.transaction = TransactionSupport::LedgerIdempotent;
    assert_ne!(base, transactional, "transaction is an independent axis");
    let mut native_watermark = base;
    native_watermark.watermark = WatermarkSupport::Native;
    assert_ne!(base, native_watermark, "watermark is an independent axis");
}

#[test]
fn capabilities_project_onto_runtime_native_types() {
    let source = source_capabilities();
    assert_eq!(
        source.replay_positioning(),
        ReplayPositioning::ExactPauseReportAndSeek
    );
    assert_eq!(source.source_delivery(), SourceDeliveryCapability::Lossless);
    assert_eq!(
        source.native_watermarks(),
        NativeWatermarkCapability::NeverEmits
    );
    let mut native = source;
    native.watermark = WatermarkSupport::Native;
    assert_eq!(
        native.native_watermarks(),
        NativeWatermarkCapability::EmitsNative
    );
    let mut lossy = source;
    lossy.delivery = DeliveryCapability::BestEffort;
    lossy.replay = ReplayCapability::Unreplayable;
    assert_eq!(lossy.replay_positioning(), ReplayPositioning::Unsupported);
    assert_eq!(lossy.source_delivery(), SourceDeliveryCapability::Lossy);

    let sink = transactional_sink_capabilities();
    assert_eq!(sink.sink_delivery(), SinkDelivery::Transactional);
    let mut ledger = sink;
    ledger.transaction = TransactionSupport::LedgerIdempotent;
    assert_eq!(
        ledger.sink_delivery(),
        SinkDelivery::EpochIdempotent {
            mechanism: "ledger".into(),
            retention: RetentionClass::Bounded,
        }
    );
    let mut ordinary = sink;
    ordinary.transaction = TransactionSupport::None;
    assert_eq!(ordinary.sink_delivery(), SinkDelivery::Ordinary);
}

fn participants(
    source_caps: ConnectorCapabilities,
    sink_caps: ConnectorCapabilities,
) -> Vec<DeliveryParticipant> {
    vec![
        DeliveryParticipant {
            path: "sources/pg".into(),
            role: ParticipantRole::Source,
            capabilities: source_caps,
        },
        DeliveryParticipant {
            path: "sinks/parquet".into(),
            role: ParticipantRole::Sink,
            capabilities: sink_caps,
        },
    ]
}

#[test]
fn exactly_once_error_names_the_unmet_axis_per_participant() {
    let mut unreplayable = source_capabilities();
    unreplayable.replay = ReplayCapability::Unreplayable;
    unreplayable.delivery = DeliveryCapability::BestEffort;
    let mut ordinary_sink = transactional_sink_capabilities();
    ordinary_sink.transaction = TransactionSupport::None;
    let error = validate_delivery_guarantee(
        DeliveryGuarantee::ExactlyOnce,
        &participants(unreplayable, ordinary_sink),
    )
    .expect_err("the error reports every unmet axis");
    let message = error.to_string();
    assert!(
        message.contains("replay") && message.contains("delivery"),
        "source axes are named: {message}"
    );
    assert!(
        message.contains("transaction"),
        "sink axis is named: {message}"
    );
}

#[test]
fn exactly_once_fails_naming_the_incapable_participant() {
    let mut ordinary_sink = transactional_sink_capabilities();
    ordinary_sink.transaction = TransactionSupport::None;
    let error = validate_delivery_guarantee(
        DeliveryGuarantee::ExactlyOnce,
        &participants(source_capabilities(), ordinary_sink),
    )
    .expect_err("ordinary sink cannot join an exactly-once plan");
    let message = error.to_string();
    assert!(
        message.contains("sinks/parquet"),
        "error names the path: {message}"
    );
    assert!(
        message.contains("exactly-once"),
        "error names the axis: {message}"
    );
}

#[test]
fn exactly_once_requires_replayable_sources() {
    let mut lossy = source_capabilities();
    lossy.replay = ReplayCapability::Unreplayable;
    let error = validate_delivery_guarantee(
        DeliveryGuarantee::ExactlyOnce,
        &participants(lossy, transactional_sink_capabilities()),
    )
    .expect_err("unreplayable source cannot join an exactly-once plan");
    assert!(error.to_string().contains("sources/pg"));
}

#[test]
fn exactly_once_passes_with_capable_participants() {
    let proof = validate_delivery_guarantee(
        DeliveryGuarantee::ExactlyOnce,
        &participants(source_capabilities(), transactional_sink_capabilities()),
    )
    .expect("capable participants prove exactly-once");
    assert_eq!(proof.requested, DeliveryGuarantee::ExactlyOnce);
    assert_eq!(proof.effective, DeliveryGuarantee::ExactlyOnce);
}

#[test]
fn at_least_once_never_upgrades_silently() {
    let proof = validate_delivery_guarantee(
        DeliveryGuarantee::AtLeastOnce,
        &participants(source_capabilities(), transactional_sink_capabilities()),
    )
    .expect("at-least-once request validates");
    assert_eq!(proof.requested, DeliveryGuarantee::AtLeastOnce);
    assert_eq!(proof.effective, DeliveryGuarantee::AtLeastOnce);
}

#[test]
fn registry_snapshot_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ConnectorRegistrySnapshot>();
}

#[test]
fn connector_operation_rejects_empty_names() {
    let error = ConnectorOperation::new("").expect_err("empty operation name rejected");
    assert!(matches!(error, CalcFlowError::InvalidArgument { .. }));
    let operation = ConnectorOperation::new("open_source").expect("valid operation name");
    assert_eq!(operation.as_str(), "open_source");
}

#[test]
fn connector_error_display_carries_identity_operation_and_detail() {
    let error = ConnectorError::new(
        identity("kafka"),
        ConnectorOperation::new("open_source").expect("valid operation name"),
        "broker handshake timed out",
    );
    let message = error.to_string();
    assert!(
        message.contains(PROVIDER) && message.contains("kafka"),
        "display carries the connector identity: {message}"
    );
    assert!(
        message.contains("open_source"),
        "display carries the stable operation name: {message}"
    );
    assert!(
        message.contains("broker handshake timed out"),
        "display carries the payload-free detail: {message}"
    );

    let projected = CalcFlowError::from(error);
    match &projected {
        CalcFlowError::Connector(inner) => {
            assert_eq!(inner.identity, identity("kafka"));
            assert_eq!(inner.operation.as_str(), "open_source");
        }
        other => panic!("expected Connector, got {other:?}"),
    }
    assert!(
        projected.to_string().contains("kafka"),
        "the public error keeps identity and operation visible"
    );
}
