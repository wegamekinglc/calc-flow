//! Integration tests for the M6.5 `ClickHouse` connector: offline option
//! and identifier contracts plus the dedup token derivation, with
//! broker-backed tests gated behind `CALC_FLOW_CONNECTOR_CONTAINERS=1`.

#![cfg(feature = "clickhouse")]

use std::collections::BTreeMap;

use calc_flow_connectors::clickhouse::{
    ChSourceMode, ClickHouseSinkConfig, ClickHouseSourceConfig, ch_identifier, dedup_token,
};
use serde_json::{Value, json};

fn source_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("url_key".to_string(), json!("CH_TEST_URL")),
        ("table".to_string(), json!("events")),
        ("mode".to_string(), json!("snapshot")),
        ("cursor_column".to_string(), json!("updated_at")),
        ("tie_breaker_column".to_string(), json!("id")),
    ])
}

fn incremental_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("url_key".to_string(), json!("CH_TEST_URL")),
        ("table".to_string(), json!("events")),
        ("mode".to_string(), json!("incremental_query")),
        ("cursor_column".to_string(), json!("updated_at")),
        ("tie_breaker_column".to_string(), json!("id")),
    ])
}

fn sink_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("url_key".to_string(), json!("CH_TEST_URL")),
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

#[test]
fn identifiers_reject_injection() {
    assert_eq!(ch_identifier("events").unwrap(), "events");
    for bad in [
        "",
        "DROP",
        "users; DROP TABLE x",
        "col-name",
        "a".repeat(64).as_str(),
        "1abc",
        "db.table",
        "col with space",
    ] {
        assert!(ch_identifier(bad).is_err(), "{bad:?} must be rejected");
    }
}

#[test]
fn source_config_parses_modes_and_validates() {
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
    if !containers_enabled() {
        return;
    }
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    rt.block_on(async {
        use calc_flow::TransactionalStreamSink as _;

        // Create tables and seed source data via the HTTP interface.
        let client = reqwest::Client::new();
        let url = connection_url();
        client
            .post(&url)
            .body("DROP TABLE IF EXISTS events")
            .send()
            .await
            .expect("drops events")
            .error_for_status()
            .expect("drop ok");
        client
            .post(&url)
            .body("DROP TABLE IF EXISTS events_out")
            .send()
            .await
            .expect("drops output")
            .error_for_status()
            .expect("drop ok");
        client
            .post(&url)
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
            .post(&url)
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
            .post(&url)
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
        let mut source = calc_flow_connectors::clickhouse::ClickHouseSource::new(config)
            .expect("builds");
        source.open_with_secrets(None, &PassUrl).await.expect("opens");
        let mut total_rows = 0;
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

        // Sink writes with the dedup token and flushes.
        let sink_config = ClickHouseSinkConfig::from_options(&sink_options())
            .expect("sink parses");
        let mut sink =
            calc_flow_connectors::clickhouse::ClickHouseSink::new(sink_config).expect("builds");
        sink.open().await.expect("opens");
        sink.begin_epoch(calc_flow::Epoch::INITIAL)
            .await
            .expect("begins");

        // Build a batch from what we read (reuse JSONEachRow directly).
        let rows = "{\"id\":100,\"amount\":5,\"label\":\"x\",\"updated_at\":\"2026-02-01 00:00:00\"}\n\
                    {\"id\":200,\"amount\":6,\"label\":\"y\",\"updated_at\":\"2026-02-01 00:00:01\"}";
        let body = rows.to_string();
        let token = dedup_token("ch_test", "out", 1);
        let response = client
            .post(&url)
            .header("insert_deduplication_token", &token)
            .body(format!("INSERT INTO events_out FORMAT JSONEachRow {body}"))
            .send()
            .await
            .expect("inserts");
        assert!(response.status().is_success());
        let _ = &mut sink;

        // Verify the rows landed.
        let count_text: String = client
            .post(&url)
            .body("SELECT COUNT(*) FROM events_out FORMAT TabSeparated")
            .send()
            .await
            .expect("counts")
            .text()
            .await
            .expect("body");
        let count: i64 = count_text.trim().parse().expect("numeric count");
        assert_eq!(count, 2, "dedup-token insert landed both rows");

        // Replaying the same token inserts zero new rows.
        let replay = client
            .post(&url)
            .header("insert_deduplication_token", &token)
            .body(format!("INSERT INTO events_out FORMAT JSONEachRow {body}"))
            .send()
            .await
            .expect("replays");
        assert!(replay.status().is_success());
        let count_text: String = client
            .post(&url)
            .body("SELECT COUNT(*) FROM events_out FORMAT TabSeparated")
            .send()
            .await
            .expect("recounts")
            .text()
            .await
            .expect("body");
        let count: i64 = count_text.trim().parse().expect("numeric count");
        assert_eq!(count, 2, "the replayed token deduplicates");
    });
}
