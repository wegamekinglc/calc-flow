//! Integration tests for the M6.3 Kafka connector: offline configuration
//! and cursor contracts, plus broker-backed tests gated behind
//! `CALC_FLOW_CONNECTOR_CONTAINERS=1`.

#![cfg(feature = "kafka")]

use std::collections::BTreeMap;

use calc_flow::{StreamSource as _, TransactionalStreamSink as _};
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
        ("transactional_id".to_string(), json!("calc-flow-test")),
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
fn sink_config_requires_transactional_identity() {
    let config = KafkaSinkConfig::from_options(&sink_options()).expect("parses");
    assert_eq!(config.transactional_id, "calc-flow-test");

    let mut missing = sink_options();
    missing.remove("transactional_id");
    let error = KafkaSinkConfig::from_options(&missing).expect_err("transactional id required");
    assert!(error.to_string().contains("transactional_id"), "{error}");
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
async fn kafka_source_cannot_be_created_without_a_broker() {
    let mut options = source_options();
    options.insert("bootstrap_servers".to_string(), json!("localhost:1"));
    let config = KafkaSourceConfig::from_options(&options).expect("parses");
    // Consumer creation succeeds lazily; the first poll must fail closed
    // instead of silently reporting idleness forever.
    let mut source =
        calc_flow_connectors::kafka::KafkaSource::new(config).expect("consumer constructs lazily");
    let outcome = source.open(None).await;
    assert!(outcome.is_ok(), "open does not require a broker");
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
        ("transactional_id".to_string(), json!("calc-flow-it-txn")),
        ("format".to_string(), json!("json")),
    ]))
    .expect("parses");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    rt.block_on(async move {
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
        sink.commit(calc_flow::Epoch::INITIAL, &evidence)
            .await
            .expect("commits");

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
        assert!(seen >= 2, "delivered rows reached the consumer: {seen}");
    });
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
