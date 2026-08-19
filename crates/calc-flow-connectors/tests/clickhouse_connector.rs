//! Integration tests for the M6.5 `ClickHouse` connector: offline option
//! and identifier contracts plus the dedup token derivation, with
//! broker-backed tests gated behind `CALC_FLOW_CONNECTOR_CONTAINERS=1`.

#![cfg(feature = "clickhouse")]

use std::collections::BTreeMap;

use calc_flow_connectors::clickhouse::{ChSourceMode, ClickHouseSourceConfig, ch_identifier};
use calc_flow_connectors::clickhouse_sink::{ClickHouseSink, ClickHouseSinkConfig, dedup_token};
use calc_flow_connectors::register_clickhouse_connectors;
use serde_json::{Value, json};

fn source_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("table".to_string(), json!("events")),
        ("mode".to_string(), json!("snapshot")),
        ("cursor_column".to_string(), json!("updated_at")),
        ("tie_breaker_column".to_string(), json!("id")),
        ("tie_breaker_unique".to_string(), json!(true)),
        ("max_batch_rows".to_string(), json!(2)),
        (
            "columns".to_string(),
            json!([
                {"name": "updated_at", "data_type": "string", "nullable": false},
                {"name": "id", "data_type": "int64", "nullable": false},
                {"name": "label", "data_type": "string", "nullable": false}
            ]),
        ),
    ])
}

fn incremental_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("table".to_string(), json!("events")),
        ("mode".to_string(), json!("incremental_query")),
        ("cursor_column".to_string(), json!("updated_at")),
        ("tie_breaker_column".to_string(), json!("id")),
        ("tie_breaker_unique".to_string(), json!(true)),
        (
            "columns".to_string(),
            json!([
                {"name": "updated_at", "data_type": "string", "nullable": false},
                {"name": "id", "data_type": "int64", "nullable": false},
                {"name": "label", "data_type": "string", "nullable": false}
            ]),
        ),
    ])
}

fn sink_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("table".to_string(), json!("events_out")),
        ("pipeline".to_string(), json!("ch_test")),
        ("output".to_string(), json!("out")),
    ])
}

struct PassUrl;

impl calc_flow::SecretResolver for PassUrl {
    fn resolve(
        &self,
        _reference: &calc_flow::SecretReference,
    ) -> calc_flow::Result<calc_flow::SecretHandle> {
        Ok(calc_flow::SecretHandle::from_bytes(
            connection_url().as_bytes(),
        ))
    }
}

fn containers_enabled() -> bool {
    std::env::var("CALC_FLOW_CONNECTOR_CONTAINERS")
        .map(|value| value == "1")
        .unwrap_or(false)
}

fn connection_url() -> String {
    std::env::var("CH_TEST_URL").unwrap_or_else(|_| "http://localhost:8123".into())
}

fn query_url() -> String {
    format!("{}/?user=default", connection_url())
}

#[test]
fn identifiers_reject_injection() {
    assert_eq!(ch_identifier("events").unwrap(), "events");
    let long_name = "a".repeat(64);
    for bad in [
        "",
        "DROP",
        "users; DROP TABLE x",
        "col-name",
        long_name.as_str(),
        "1abc",
        "db.table",
        "col with space",
    ] {
        assert!(ch_identifier(bad).is_err(), "{bad:?} must be rejected");
    }
}

#[test]
fn source_config_parses_modes_and_validates_identifiers() {
    let config = ClickHouseSourceConfig::from_options(&source_options()).expect("snapshot parses");
    assert_eq!(config.mode, ChSourceMode::Snapshot);
    assert_eq!(config.cursor_column, "updated_at");
    assert_eq!(config.tie_breaker_column, "id");

    let config =
        ClickHouseSourceConfig::from_options(&incremental_options()).expect("incremental parses");
    assert_eq!(config.mode, ChSourceMode::IncrementalQuery);

    let mut bad_table = source_options();
    bad_table.insert("table".to_string(), json!("system.users; --"));
    let error = ClickHouseSourceConfig::from_options(&bad_table).expect_err("identifier checked");
    assert!(error.to_string().contains("identifier"), "{error}");

    let mut bad_cursor = source_options();
    bad_cursor.insert("cursor_column".to_string(), json!("1 UNION SELECT 1"));
    let error = ClickHouseSourceConfig::from_options(&bad_cursor).expect_err("cursor validated");
    assert!(error.to_string().contains("identifier"), "{error}");

    let mut bad_tie = source_options();
    bad_tie.insert("tie_breaker_column".to_string(), json!(""));
    let error = ClickHouseSourceConfig::from_options(&bad_tie).expect_err("tie-breaker required");
    assert!(error.to_string().contains("identifier"), "{error}");

    let mut bad_mode = source_options();
    bad_mode.insert("mode".to_string(), json!("cdc"));
    let error = ClickHouseSourceConfig::from_options(&bad_mode).expect_err("cdc rejected");
    assert!(error.to_string().contains("mode"), "{error}");
}

#[test]
fn sink_config_parses_identity() {
    let config = ClickHouseSinkConfig::from_options(&sink_options()).expect("parses");
    assert_eq!(config.table, "events_out");
    assert_eq!(config.pipeline, "ch_test");

    let mut bad = sink_options();
    bad.insert("table".to_string(), json!("information_schema.tables"));
    let error = ClickHouseSinkConfig::from_options(&bad).expect_err("identifier");
    assert!(error.to_string().contains("identifier"), "{error}");

    let mut missing = sink_options();
    missing.remove("pipeline");
    let error = ClickHouseSinkConfig::from_options(&missing).expect_err("pipeline required");
    assert!(error.to_string().contains("pipeline"), "{error}");
}

#[test]
fn dedup_tokens_are_stable_and_distinct_per_epoch() {
    let first = dedup_token("orders", "totals", 1);
    let replay = dedup_token("orders", "totals", 1);
    let next = dedup_token("orders", "totals", 2);
    let other_output = dedup_token("orders", "archive", 1);

    assert_eq!(first, replay, "stable for one (pipeline, output, epoch)");
    assert_ne!(first, next, "distinct per epoch");
    assert_ne!(first, other_output, "distinct per output");
    assert!(first.starts_with("calc-flow-"), "{first}");
    assert!(
        !first.contains(':') && !first.contains('/'),
        "the token is safe for `ClickHouse` headers: {first}"
    );
}

#[tokio::test]
async fn pre_commit_is_side_effect_free_and_carries_the_stable_block() {
    use calc_flow::TransactionalStreamSink as _;

    let mut sink = ClickHouseSink::new(
        ClickHouseSinkConfig::from_options(&sink_options()).expect("configuration parses"),
    )
    .expect("sink builds");
    sink.begin_epoch(calc_flow::Epoch::INITIAL)
        .await
        .expect("epoch begins");
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("label", arrow::datatypes::DataType::Utf8, false),
    ]));
    let record = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![
            std::sync::Arc::new(arrow::array::Int64Array::from(vec![1, 2])),
            std::sync::Arc::new(arrow::array::StringArray::from(vec!["a", "b"])),
        ],
    )
    .expect("record batch");
    let batch = calc_flow::Batch::table(
        vec![record],
        calc_flow::BatchMetadata::new("test", 1, BTreeMap::new()).expect("metadata"),
    )
    .expect("batch");
    sink.write(&batch).await.expect("batch stages");

    // A direct sink has no endpoint. Pre-commit must still succeed because it
    // only freezes durable evidence and performs no network operation.
    let evidence = sink
        .pre_commit(calc_flow::Epoch::INITIAL)
        .await
        .expect("pre-commit is local");
    assert_eq!(evidence["rows"], json!(2));
    assert_eq!(
        evidence["token"],
        json!(dedup_token(
            "ch_test",
            "out",
            calc_flow::Epoch::INITIAL.as_u64()
        ))
    );
    assert_eq!(evidence["segment_id"], json!("insert-block"));
    let segments = sink
        .pre_commit_segments(calc_flow::Epoch::INITIAL)
        .await
        .expect("insert block becomes a state segment");
    assert_eq!(
        segments["insert-block"],
        b"{\"id\":1,\"label\":\"a\"}\n{\"id\":2,\"label\":\"b\"}"
    );

    let error = sink
        .commit(calc_flow::Epoch::INITIAL, &evidence)
        .await
        .expect_err("the side effect starts only at commit");
    assert!(error.to_string().contains("trusted factory"), "{error}");
}

#[test]
fn descriptor_exposes_retry_deduplication_without_exactly_once() {
    let mut registry = calc_flow::ConnectorRegistry::new();
    register_clickhouse_connectors(&mut registry).expect("registers");
    let identity = calc_flow::ConnectorIdentity::new(
        "calc-flow-connectors",
        "clickhouse",
        calc_flow_connectors::clickhouse::IDENTITY_VERSION,
    )
    .expect("identity");
    let sink = registry.snapshot().resolve_sink(&identity).expect("sink");

    assert_eq!(
        sink.descriptor().capabilities.transaction,
        calc_flow::TransactionSupport::None,
    );
    let mut retry_options = sink_options();
    retry_options.insert("retry_deduplicated".into(), json!(true));
    let retry_capabilities = sink
        .capabilities(&retry_options)
        .expect("retry capability validates");
    assert_eq!(
        retry_capabilities.transaction,
        calc_flow::TransactionSupport::RetryDeduplicated,
    );
    let participant = calc_flow::DeliveryParticipant {
        path: "sinks[0]".into(),
        role: calc_flow::ParticipantRole::Sink,
        capabilities: retry_capabilities,
    };
    assert!(
        calc_flow::validate_delivery_guarantee(
            calc_flow::DeliveryGuarantee::ExactlyOnce,
            &[participant],
        )
        .is_err()
    );
}

#[tokio::test]
async fn default_factory_constructs_the_declared_ordinary_sink() {
    use calc_flow::ConnectorSinkFactory as _;

    let factory = calc_flow_connectors::clickhouse::ClickHouseSinkFactory::new();
    factory
        .open(&sink_options(), &PassUrl)
        .await
        .expect("default at-least-once mode constructs an ordinary sink");
}

#[test]
fn url_only_from_secrets() {
    struct OneUrl;
    impl calc_flow::SecretResolver for OneUrl {
        fn resolve(
            &self,
            reference: &calc_flow::SecretReference,
        ) -> calc_flow::Result<calc_flow::SecretHandle> {
            if reference.key == "CH_URL" {
                Ok(calc_flow::SecretHandle::from_bytes(
                    b"http://user:pass@localhost:8123",
                ))
            } else {
                Err(calc_flow::CalcFlowError::NotFound {
                    resource: "secret".into(),
                    key: reference.key.clone(),
                })
            }
        }
    }

    let url = calc_flow_connectors::clickhouse::resolve_clickhouse_url(&OneUrl, "CH_URL")
        .expect("resolves");
    assert!(url.starts_with("http://"), "{url}");

    let error = calc_flow_connectors::clickhouse::resolve_clickhouse_url(&OneUrl, "WRONG")
        .expect_err("missing key");
    assert!(
        error.to_string().contains("could not be resolved"),
        "{error}"
    );
    assert!(!error.to_string().contains("pass"), "URL never in error");
}

#[test]
#[ignore = "broker-backed; set CALC_FLOW_CONNECTOR_CONTAINERS=1 with a `ClickHouse` service"]
#[allow(
    clippy::too_many_lines,
    reason = "one gated end-to-end flow exercising the full source and sink matrix"
)]
fn snapshot_reads_and_dedup_sink_commits() {
    use calc_flow::ConnectorSinkFactory as _;

    if !containers_enabled() {
        return;
    }
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    rt.block_on(async {
        // Create tables and seed source data via the HTTP interface.
        let client = reqwest::Client::new();
        client
            .post(query_url())
            .body("DROP TABLE IF EXISTS events")
            .send()
            .await
            .expect("drops events")
            .error_for_status()
            .expect("drop ok");
        client
            .post(query_url())
            .body("DROP TABLE IF EXISTS events_out")
            .send()
            .await
            .expect("drops output")
            .error_for_status()
            .expect("drop ok");
        client
            .post(query_url())
            .body(
                "CREATE TABLE events (id UInt64, amount Int64, label String, \
                 updated_at DateTime) ENGINE = MergeTree ORDER BY (updated_at, id)",
            )
            .send()
            .await
            .expect("creates events")
            .error_for_status()
            .expect("create ok");
        client
            .post(query_url())
            .body(
                "CREATE TABLE events_out (id UInt64, amount Int64, label String, \
                 updated_at DateTime) ENGINE = MergeTree ORDER BY (updated_at, id)",
            )
            .send()
            .await
            .expect("creates output")
            .error_for_status()
            .expect("create ok");
        client
            .post(query_url())
            .body(
                "INSERT INTO events (id, amount, label, updated_at) VALUES \
                 (1, 10, 'a', '2026-01-01 00:00:01'), \
                 (2, 20, 'b', '2026-01-01 00:00:02'), \
                 (3, 30, 'c', '2026-01-01 00:00:03')",
            )
            .send()
            .await
            .expect("seeds events")
            .error_for_status()
            .expect("seed ok");

        // Snapshot source reads all rows.
        let config =
            ClickHouseSourceConfig::from_options(&source_options()).expect("source parses");
        let mut source =
            calc_flow_connectors::clickhouse::ClickHouseSource::new(config).expect("builds");
        source
            .open_with_secrets(None, &PassUrl)
            .await
            .expect("opens");
        let first = source
            .next_with_secrets(&PassUrl)
            .await
            .expect("reads first page")
            .expect("first page exists");
        let calc_flow::SourceEvent::Data { batch, .. } = first else {
            panic!("snapshot starts with data")
        };
        assert_eq!(batch.num_rows(), 2, "configured page bound is enforced");
        let mut total_rows = batch.num_rows();

        // Both rows are outside the composite upper bound captured by open:
        // one shares its cursor with the old maximum but has a higher unique
        // tie-breaker, and the other has a later cursor.
        client
            .post(query_url())
            .body(
                "INSERT INTO events (id, amount, label, updated_at) VALUES \
                 (4, 40, 'same-cursor-late', '2026-01-01 00:00:03'), \
                 (5, 50, 'later-cursor', '2026-01-01 00:00:04')",
            )
            .send()
            .await
            .expect("writes concurrent rows")
            .error_for_status()
            .expect("concurrent insert ok");
        for _ in 0..10 {
            match source.next_with_secrets(&PassUrl).await.expect("reads") {
                Some(calc_flow::SourceEvent::Data { batch, .. }) => {
                    total_rows += batch.num_rows();
                }
                Some(calc_flow::SourceEvent::Idle) | None => break,
                Some(calc_flow::SourceEvent::Watermark(_)) => {}
            }
        }
        assert_eq!(total_rows, 3, "snapshot reads every seeded row");

        // The exact factory-resolved sink writes and commits the insert block.
        let factory = calc_flow_connectors::clickhouse::ClickHouseSinkFactory::new();
        let mut sink = factory
            .open_transactional(&sink_options(), &PassUrl)
            .await
            .expect("factory opens")
            .expect("clickhouse exposes the checkpoint-aware sink");
        sink.open().await.expect("opens");
        sink.begin_epoch(calc_flow::Epoch::INITIAL)
            .await
            .expect("begins");

        let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::UInt64, false),
            arrow::datatypes::Field::new("amount", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("label", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("updated_at", arrow::datatypes::DataType::Utf8, false),
        ]));
        let record = arrow::record_batch::RecordBatch::try_new(
            schema,
            vec![
                std::sync::Arc::new(arrow::array::UInt64Array::from(vec![100, 200])),
                std::sync::Arc::new(arrow::array::Int64Array::from(vec![5, 6])),
                std::sync::Arc::new(arrow::array::StringArray::from(vec!["x", "y"])),
                std::sync::Arc::new(arrow::array::StringArray::from(vec![
                    "2026-02-01 00:00:00",
                    "2026-02-01 00:00:01",
                ])),
            ],
        )
        .expect("record batch");
        let batch = calc_flow::Batch::table(
            vec![record],
            calc_flow::BatchMetadata::new("test", 1, BTreeMap::new()).unwrap(),
        )
        .unwrap();
        sink.write(&batch).await.expect("stages");
        let evidence = sink
            .pre_commit(calc_flow::Epoch::INITIAL)
            .await
            .expect("freezes insert block");
        let segments = sink
            .pre_commit_segments(calc_flow::Epoch::INITIAL)
            .await
            .expect("insert block becomes a state segment");
        assert_eq!(
            segments["insert-block"]
                .split(|byte| *byte == b'\n')
                .count(),
            2,
            "the exact two-row block is durable before commit"
        );
        sink.commit(calc_flow::Epoch::INITIAL, &evidence)
            .await
            .expect("commits insert block");

        // Verify the rows landed.
        let count_text: String = client
            .post(query_url())
            .body("SELECT COUNT(*) FROM events_out FORMAT TabSeparated")
            .send()
            .await
            .expect("counts")
            .text()
            .await
            .expect("body");
        let count: i64 = count_text.trim().parse().expect("numeric count");
        assert_eq!(count, 2, "dedup-token insert landed both rows");

        sink.close().await.expect("closes");
    });
}
