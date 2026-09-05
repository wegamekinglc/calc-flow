//! Integration tests for the M6.3 Kafka connector: offline configuration
//! and cursor contracts, plus broker-backed tests gated behind
//! `CALC_FLOW_CONNECTOR_CONTAINERS=1`.

#![cfg(feature = "kafka")]

use std::collections::BTreeMap;

use calc_flow::{StreamSink as _, StreamSource as _, TransactionalStreamSink as _};
use calc_flow_connectors::kafka::{
    KafkaFormat, KafkaOffsetReset, KafkaSinkConfig, KafkaSourceConfig, transactional_id,
};
use serde_json::{Value, json};

fn source_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("bootstrap_servers".to_string(), json!("localhost:9092")),
        ("topic".to_string(), json!("orders")),
        ("partitions".to_string(), json!([2, 0, 2, 1])),
        ("format".to_string(), json!("json")),
    ])
}

fn sink_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("bootstrap_servers".to_string(), json!("localhost:9092")),
        ("topic".to_string(), json!("totals")),
        ("ledger_topic".to_string(), json!("calc-flow-test-ledger")),
        ("pipeline".to_string(), json!("orders")),
        ("output".to_string(), json!("totals")),
        ("format".to_string(), json!("json")),
    ])
}

fn containers_enabled() -> bool {
    std::env::var("CALC_FLOW_CONNECTOR_CONTAINERS")
        .map(|value| value == "1")
        .unwrap_or(false)
}

fn bootstrap() -> String {
    std::env::var("CALC_FLOW_KAFKA_BOOTSTRAP").unwrap_or_else(|_| "localhost:9092".into())
}

#[test]
fn source_config_parses_and_normalizes_partitions() {
    let config = KafkaSourceConfig::from_options(&source_options()).expect("parses");
    assert_eq!(config.partitions, vec![0, 1, 2], "sorted and deduplicated");
    assert_eq!(config.auto_offset_reset, KafkaOffsetReset::Earliest);
    assert!(matches!(config.format, KafkaFormat::Json));

    let error = KafkaSourceConfig::from_options(&BTreeMap::from([(
        "bootstrap_servers".to_string(),
        json!("localhost:9092"),
    )]))
    .expect_err("topic required");
    assert!(error.to_string().contains("topic"), "{error}");

    let mut bad_format = source_options();
    bad_format.insert("format".to_string(), json!("avro"));
    let error = KafkaSourceConfig::from_options(&bad_format).expect_err("avro rejected");
    assert!(error.to_string().contains("format"), "{error}");

    let mut bad_reset = source_options();
    bad_reset.insert("auto_offset_reset".to_string(), json!("middle"));
    let error = KafkaSourceConfig::from_options(&bad_reset).expect_err("reset vocabulary");
    assert!(error.to_string().contains("auto_offset_reset"), "{error}");

    let mut empty_partitions = source_options();
    empty_partitions.insert("partitions".to_string(), json!([]));
    let error =
        KafkaSourceConfig::from_options(&empty_partitions).expect_err("empty assignment rejected");
    assert!(error.to_string().contains("partition"), "{error}");
}

#[test]
fn sink_config_derives_transactional_identity() {
    let config = KafkaSinkConfig::from_options(&sink_options()).expect("parses");
    assert_eq!(
        config.transactional_id,
        transactional_id("orders", "totals")
    );

    let mut injected = sink_options();
    injected.insert("transactional_id".into(), json!("caller-controlled"));
    let error = KafkaSinkConfig::from_options(&injected)
        .expect_err("caller-provided transactional IDs fail closed");
    assert!(error.to_string().contains("derived"), "{error}");
}

#[test]
fn transactional_ids_are_stable_and_secret_free() {
    let first = transactional_id("orders", "totals");
    let second = transactional_id("orders", "totals");
    let other = transactional_id("billing", "totals");
    assert_eq!(first, second, "stable for one identity");
    assert_ne!(first, other, "distinct per identity");
    assert!(first.starts_with("calc-flow-"), "{first}");
    assert!(
        !first.contains(':') && !first.contains('@') && !first.contains('/'),
        "the id is safe for Kafka transactional IDs: {first}"
    );
}

#[tokio::test]
async fn unreachable_broker_surfaces_as_idleness() {
    let mut options = source_options();
    options.insert("bootstrap_servers".to_string(), json!("localhost:1"));
    let config = KafkaSourceConfig::from_options(&options).expect("parses");
    let mut source =
        calc_flow_connectors::kafka::KafkaSource::new(config).expect("consumer constructs lazily");
    source
        .open(None)
        .await
        .expect("open does not require a broker");
    // The unreachable broker must surface as Idle (outage resilience),
    // never as data and never as a job-fatal poll error.
    match source.next().await.expect("poll stays healthy") {
        Some(calc_flow::SourceEvent::Idle) => {}
        _ => panic!("an unreachable broker reports idleness"),
    }
}

#[test]
#[ignore = "broker-backed; set CALC_FLOW_CONNECTOR_CONTAINERS=1 with a Kafka/Redpanda service"]
fn kafka_roundtrip_and_transactional_exactly_once() {
    if !containers_enabled() {
        return;
    }
    let bootstrap_servers = bootstrap();
    let topic = "calc-flow-kafka-it";
    let options = BTreeMap::from([
        (
            "bootstrap_servers".to_string(),
            json!(bootstrap_servers.clone()),
        ),
        ("topic".to_string(), json!(topic)),
        ("partitions".to_string(), json!([0])),
        ("format".to_string(), json!("json")),
    ]);
    let config = KafkaSourceConfig::from_options(&options).expect("parses");
    let sink_config = KafkaSinkConfig::from_options(&BTreeMap::from([
        ("bootstrap_servers".to_string(), json!(bootstrap_servers)),
        ("topic".to_string(), json!(topic)),
        (
            "ledger_topic".to_string(),
            json!("calc-flow-kafka-it-ledger"),
        ),
        ("pipeline".to_string(), json!("kafka-it")),
        ("output".to_string(), json!("records")),
        ("format".to_string(), json!("json")),
    ]))
    .expect("parses");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    rt.block_on(async move {
        provision_ledger(&bootstrap_servers, "calc-flow-kafka-it-ledger").await;

        let batch = sample_batch();
        let mut sink =
            calc_flow_connectors::kafka::TransactionalKafkaSink::new(sink_config.clone())
                .expect("transactional producer initializes");
        sink.open().await.expect("opens");
        sink.begin_epoch(calc_flow::Epoch::INITIAL)
            .await
            .expect("begins");
        sink.write(&batch).await.expect("writes");
        let evidence = sink
            .pre_commit(calc_flow::Epoch::INITIAL)
            .await
            .expect("pre");
        let segments = sink
            .pre_commit_segments(calc_flow::Epoch::INITIAL)
            .await
            .expect("prepared records");
        sink.commit(calc_flow::Epoch::INITIAL, &evidence)
            .await
            .expect("commits");
        sink.close().await.expect("first producer closes");

        let mut recovery_sink =
            calc_flow_connectors::kafka::TransactionalKafkaSink::new(sink_config)
                .expect("new producer fences stale ownership");
        recovery_sink.open().await.expect("recovery sink opens");
        recovery_sink
            .recover(
                &calc_flow::SinkRecovery::from_parts(
                    calc_flow::Epoch::INITIAL,
                    false,
                    calc_flow::SinkDelivery::EpochIdempotent {
                        mechanism: "kafka-ledger".into(),
                        retention: calc_flow::RetentionClass::Unbounded,
                    },
                    evidence,
                )
                .with_segments(segments),
            )
            .await
            .expect("committed marker suppresses replay after lost ack");
        recovery_sink
            .close()
            .await
            .expect("recovery producer closes");

        let mut source =
            calc_flow_connectors::kafka::KafkaSource::new(config).expect("source constructs");
        source.open(None).await.expect("assigns earliest offsets");
        let mut seen = 0;
        for _ in 0..64 {
            if let Some(calc_flow::SourceEvent::Data { batch, cursor }) =
                source.next().await.expect("polls")
            {
                seen += batch.num_rows();
                let offsets = cursor.payload().get("offsets").expect("offset map");
                assert!(offsets.is_object(), "cursor carries partition offsets");
                if seen >= 2 {
                    break;
                }
            }
        }
        assert_eq!(seen, 2, "recovery did not duplicate committed rows");
    });
}

async fn provision_ledger(bootstrap_servers: &str, topic: &str) {
    use rdkafka::admin::{AdminClient, AdminOptions, NewTopic, TopicReplication};
    use rdkafka::client::DefaultClientContext;

    let admin: AdminClient<DefaultClientContext> = rdkafka::config::ClientConfig::new()
        .set("bootstrap.servers", bootstrap_servers)
        .create()
        .expect("admin client");
    let ledger =
        NewTopic::new(topic, 1, TopicReplication::Fixed(1)).set("cleanup.policy", "compact");
    admin
        .create_topics(&[ledger], &AdminOptions::new())
        .await
        .expect("ledger topic request")[0]
        .as_ref()
        .expect("ledger topic is created");
}

fn sample_batch() -> calc_flow::Batch {
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use std::sync::Arc as StdArc;
    let schema = StdArc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let array = Int64Array::from(vec![1, 2]);
    let record = RecordBatch::try_new(schema, vec![StdArc::new(array)]).expect("record batch");
    calc_flow::Batch::table(
        vec![record],
        calc_flow::BatchMetadata::new("test", 1, BTreeMap::new()).unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn offline_source_reports_idle_and_replays_cursors() {
    let mut options = source_options();
    options.insert("bootstrap_servers".to_string(), json!(bootstrap()));
    let config = KafkaSourceConfig::from_options(&options).expect("parses");
    let mut source =
        calc_flow_connectors::kafka::KafkaSource::new(config).expect("offline construction");

    // A broker-less poll must surface Idle, not an error and not data.
    for _ in 0..3 {
        match source.next().await.expect("poll stays healthy") {
            Some(calc_flow::SourceEvent::Idle) => {}
            Some(calc_flow::SourceEvent::Data { .. }) => panic!("no broker can deliver data"),
            Some(calc_flow::SourceEvent::Watermark(_)) => panic!("kafka never emits watermarks"),
            None => panic!("a kafka source never ends"),
        }
    }

    // Cursor replay re-assigns partitions at the carried offsets without
    // contacting a broker.
    let offsets: serde_json::Map<String, Value> = [
        ("0".to_string(), Value::from(7_i64)),
        ("1".to_string(), Value::from(3_i64)),
        ("2".to_string(), Value::from(9_i64)),
    ]
    .into_iter()
    .collect();
    let cursor = calc_flow::Cursor::unbound(
        7_u64.to_be_bytes().to_vec(),
        BTreeMap::from([
            ("offsets".to_string(), Value::Object(offsets)),
            ("sequence".to_string(), Value::from(7_u64)),
        ]),
    )
    .expect("cursor");
    source.open(Some(cursor)).await.expect("replay assignment");

    let malformed = calc_flow::Cursor::unbound(
        8_u64.to_be_bytes().to_vec(),
        BTreeMap::from([
            ("offsets".to_string(), json!({"0": 8, "1": 4})),
            ("sequence".to_string(), Value::from(8_u64)),
        ]),
    )
    .expect("cursor shape");
    let error = source
        .open(Some(malformed))
        .await
        .expect_err("partial partition cursors fail closed");
    assert!(error.to_string().contains("partition set"), "{error}");
    source.close().await.expect("closes");
}

#[test]
fn ordinary_sink_constructs_offline() {
    let config = KafkaSinkConfig::from_options(&sink_options()).expect("parses");
    let mut options = sink_options();
    options.insert("format".to_string(), json!("csv"));
    let csv_config = KafkaSinkConfig::from_options(&options).expect("csv parses");
    assert!(matches!(csv_config.format, KafkaFormat::Csv));

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async move {
        let mut sink = calc_flow_connectors::kafka::OrdinaryKafkaSink::new(config)
            .expect("idempotent producer constructs offline");
        sink.open().await.expect("opens");
    });
}

#[test]
fn malformed_options_fail_closed() {
    let mut bad_partitions = source_options();
    bad_partitions.insert("partitions".to_string(), json!(["a"]));
    let error = KafkaSourceConfig::from_options(&bad_partitions).expect_err("string entries");
    assert!(error.to_string().contains("partitions"), "{error}");

    let mut non_array = source_options();
    non_array.insert("partitions".to_string(), json!(3));
    let error = KafkaSourceConfig::from_options(&non_array).expect_err("non-array partitions");
    assert!(error.to_string().contains("partitions"), "{error}");

    let mut bad_schema = source_options();
    bad_schema.insert("schema".to_string(), json!("nope"));
    let error = KafkaSourceConfig::from_options(&bad_schema).expect_err("schema shape");
    assert!(error.to_string().contains("schema"), "{error}");

    let mut bad_bound = source_options();
    bad_bound.insert("max_batch_rows".to_string(), json!("many"));
    let error = KafkaSourceConfig::from_options(&bad_bound).expect_err("bound type");
    assert!(error.to_string().contains("max_batch_rows"), "{error}");

    let mut missing_servers = sink_options();
    missing_servers.remove("bootstrap_servers");
    let error = KafkaSinkConfig::from_options(&missing_servers).expect_err("servers required");
    assert!(error.to_string().contains("bootstrap_servers"), "{error}");
}

#[tokio::test]
async fn factories_register_and_resolve_offline() {
    use calc_flow_connectors::register_kafka_connectors;

    let mut registry = calc_flow::ConnectorRegistry::new();
    register_kafka_connectors(&mut registry).expect("registers");
    let snapshot = registry.snapshot();
    let identity = calc_flow::ConnectorIdentity::new(
        "calc-flow-connectors",
        "kafka",
        calc_flow_connectors::KAFKA_CONNECTOR_VERSION,
    )
    .expect("identity");
    let source = snapshot.resolve_source(&identity).expect("source resolves");
    let sink = snapshot.resolve_sink(&identity).expect("sink resolves");
    assert!(!source.descriptor().capabilities.snapshot);
    assert_eq!(
        sink.descriptor().capabilities.transaction,
        calc_flow::TransactionSupport::LedgerIdempotent
    );

    // Duplicate registration conflicts atomically.
    let error = register_kafka_connectors(&mut registry).expect_err("slot occupied");
    assert!(
        matches!(error, calc_flow::CalcFlowError::Conflict { .. }),
        "{error}"
    );

    // The source factory validates options without a broker.
    let outcome = source
        .open(
            &BTreeMap::from([("format".to_string(), json!("avro"))]),
            &NoSecrets,
        )
        .await;
    assert!(outcome.is_err(), "unknown formats are rejected offline");

    // The ordinary sink factory constructs its producer offline.
    let options = BTreeMap::from([
        ("bootstrap_servers".to_string(), json!(bootstrap())),
        ("topic".to_string(), json!("calc-flow-it")),
        (
            "ledger_topic".to_string(),
            json!("calc-flow-offline-ledger"),
        ),
        ("pipeline".to_string(), json!("offline")),
        ("output".to_string(), json!("records")),
        ("format".to_string(), json!("json")),
    ]);
    let sink_result = sink.open(&options, &NoSecrets).await;
    assert!(sink_result.is_ok(), "ordinary producer constructs offline");
}

struct NoSecrets;

impl calc_flow::SecretResolver for NoSecrets {
    fn resolve(
        &self,
        reference: &calc_flow::SecretReference,
    ) -> calc_flow::Result<calc_flow::SecretHandle> {
        Err(calc_flow::CalcFlowError::NotFound {
            resource: "secret".into(),
            key: reference.key.clone(),
        })
    }
}

#[test]
fn transactional_sink_recovery_validates_identity_evidence() {
    use calc_flow_connectors::kafka::validate_recovery_evidence;

    let matching = BTreeMap::from([("transactional_id".to_string(), json!("calc-flow-test"))]);
    validate_recovery_evidence("calc-flow-test", &matching)
        .expect("matching identity evidence recovers");

    let mismatch = BTreeMap::from([("transactional_id".to_string(), json!("other-owner"))]);
    let error = validate_recovery_evidence("calc-flow-test", &mismatch)
        .expect_err("a foreign transactional ID fails closed");
    assert!(error.to_string().contains("other-owner"), "{error}");

    let missing = BTreeMap::new();
    let error = validate_recovery_evidence("calc-flow-test", &missing)
        .expect_err("missing identity evidence fails closed");
    assert!(error.to_string().contains("transactional ID"), "{error}");
}
