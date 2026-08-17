//! Integration tests for the M6.4 `PostgreSQL` connector: offline option
//! and identifier contracts, plus broker-backed tests gated behind
//! `CALC_FLOW_CONNECTOR_CONTAINERS=1`.

#![cfg(feature = "postgresql")]

use std::collections::BTreeMap;

use calc_flow_connectors::database_types::{PgValue, arrow_data_type, pg_identifier};
use calc_flow_connectors::postgresql::{
    PgSinkMode, PgSourceMode, PostgresSinkConfig, PostgresSourceConfig,
};
use serde_json::{Value, json};

fn source_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("url_key".to_string(), json!("CALC_FLOW_PG_TEST_URL")),
        ("table".to_string(), json!("orders")),
        ("mode".to_string(), json!("snapshot")),
    ])
}

fn incremental_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("url_key".to_string(), json!("CALC_FLOW_PG_TEST_URL")),
        ("table".to_string(), json!("orders")),
        ("mode".to_string(), json!("incremental_query")),
        ("cursor_columns".to_string(), json!(["updated_at", "id"])),
    ])
}

fn sink_options(mode: &str) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("url_key".to_string(), json!("CALC_FLOW_PG_TEST_URL")),
        ("table".to_string(), json!("orders_out")),
        ("mode".to_string(), json!(mode)),
        ("pipeline".to_string(), json!("pg_test")),
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
    std::env::var("CALC_FLOW_PG_TEST_URL")
        .unwrap_or_else(|_| "postgresql://postgres:postgres@localhost:5432/postgres".into())
}

/// Marker for the gated container test; the URL arrives through the
/// environment the CI service exports.
#[test]
fn source_config_parses_modes_and_validates_identifiers() {
    let config = PostgresSourceConfig::from_options(&source_options()).expect("snapshot parses");
    assert_eq!(config.mode, PgSourceMode::Snapshot);

    let config =
        PostgresSourceConfig::from_options(&incremental_options()).expect("incremental parses");
    assert_eq!(config.mode, PgSourceMode::IncrementalQuery);
    assert_eq!(config.cursor_columns, vec!["updated_at", "id"]);

    let mut missing_cursor = incremental_options();
    missing_cursor.remove("cursor_columns");
    let error = PostgresSourceConfig::from_options(&missing_cursor).expect_err("cursor required");
    assert!(error.to_string().contains("cursor_columns"), "{error}");

    let mut bad_table = source_options();
    bad_table.insert("table".to_string(), json!("DROP TABLE users;--"));
    let error = PostgresSourceConfig::from_options(&bad_table).expect_err("identifier checked");
    assert!(error.to_string().contains("identifier"), "{error}");

    let mut bad_mode = source_options();
    bad_mode.insert("mode".to_string(), json!("logical_cdc"));
    let error = PostgresSourceConfig::from_options(&bad_mode).expect_err("mode vocabulary");
    assert!(error.to_string().contains("mode"), "{error}");
}

#[test]
fn sink_config_parses_modes_and_conflict_columns() {
    let config = PostgresSinkConfig::from_options(&sink_options("transactional"))
        .expect("transactional parses");
    assert_eq!(config.mode, PgSinkMode::Transactional);

    let mut upsert = sink_options("upsert");
    upsert.insert("conflict_columns".to_string(), json!(["id"]));
    let config = PostgresSinkConfig::from_options(&upsert).expect("upsert parses");
    assert_eq!(config.mode, PgSinkMode::Upsert);
    assert_eq!(config.conflict_columns, vec!["id"]);

    let mut bad_conflict = sink_options("upsert");
    bad_conflict.insert("conflict_columns".to_string(), json!([42]));
    let error = PostgresSinkConfig::from_options(&bad_conflict).expect_err("conflict type checked");
    assert!(error.to_string().contains("conflict_columns"), "{error}");
}

#[test]
fn identifiers_reject_injection() {
    assert_eq!(pg_identifier("orders").unwrap(), "orders");
    for bad in [
        "",
        "DROP",
        "users; DROP TABLE x",
        "col-name",
        "a".repeat(64).as_str(),
        "1abc",
        "table.column",
    ] {
        assert!(pg_identifier(bad).is_err(), "{bad:?} must be rejected");
    }
}

#[test]
fn type_matrix_maps_and_rejects_unknown_types() {
    use tokio_postgres::types::Type;
    assert_eq!(
        arrow_data_type(&Type::INT8).unwrap(),
        arrow::datatypes::DataType::Int64
    );
    assert_eq!(
        arrow_data_type(&Type::TEXT).unwrap(),
        arrow::datatypes::DataType::Utf8
    );
    assert_eq!(
        arrow_data_type(&Type::NUMERIC).unwrap(),
        arrow::datatypes::DataType::Utf8
    );
    let error = arrow_data_type(&Type::JSONB).expect_err("jsonb rejected");
    assert!(error.to_string().contains("matrix"), "{error}");
    let error = arrow_data_type(&Type::JSON).expect_err("json rejected");
    assert!(error.to_string().contains("matrix"), "{error}");
}

#[test]
fn pg_value_implements_tosql() {
    use tokio_postgres::types::ToSql;
    let mut buf = tokio_postgres::types::private::BytesMut::new();
    PgValue::Int64(42)
        .to_sql(&tokio_postgres::types::Type::INT8, &mut buf)
        .expect("i64 serializes");
    PgValue::Text("hello".into())
        .to_sql(&tokio_postgres::types::Type::TEXT, &mut buf)
        .expect("text serializes");
    PgValue::Null
        .to_sql(&tokio_postgres::types::Type::INT8, &mut buf)
        .expect("null serializes");
}

#[tokio::test]
async fn connection_url_only_from_secrets() {
    use calc_flow_connectors::postgresql::resolve_connection_url;

    struct OneUrl;
    impl calc_flow::SecretResolver for OneUrl {
        fn resolve(
            &self,
            reference: &calc_flow::SecretReference,
        ) -> calc_flow::Result<calc_flow::SecretHandle> {
            if reference.key == "PG_URL" {
                Ok(calc_flow::SecretHandle::from_bytes(
                    b"postgresql://user:pass@localhost/db",
                ))
            } else {
                Err(calc_flow::CalcFlowError::NotFound {
                    resource: "secret".into(),
                    key: reference.key.clone(),
                })
            }
        }
    }

    let url = resolve_connection_url(&OneUrl, "PG_URL").expect("resolves");
    assert!(url.starts_with("postgresql://"), "{url}");

    let error = resolve_connection_url(&OneUrl, "WRONG_KEY").expect_err("missing key");
    assert!(
        error.to_string().contains("could not be resolved"),
        "{error}"
    );
    assert!(
        !error.to_string().contains("pass"),
        "the URL never enters the error"
    );
}

#[test]
#[ignore = "broker-backed; set CALC_FLOW_CONNECTOR_CONTAINERS=1 with a PostgreSQL service"]
#[allow(
    clippy::too_many_lines,
    reason = "one gated end-to-end flow exercising the full source and sink matrix"
)]
fn snapshot_reads_and_transactional_sink_commits() {
    if !containers_enabled() {
        return;
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    rt.block_on(async {
        use calc_flow::{StreamSource as _, TransactionalStreamSink as _};

        // Create the test table and insert source data.
        let (client, conn) =
            tokio_postgres::connect(&connection_url(), tokio_postgres::NoTls)
                .await
                .expect("connects");
        tokio::spawn(async move {
            let _ = conn.await;
        });
        client
            .execute("DROP TABLE IF EXISTS orders", &[])
            .await
            .expect("clean orders");
        client
            .execute("DROP TABLE IF EXISTS orders_out", &[])
            .await
            .expect("clean output");
        client
            .execute("DROP TABLE IF EXISTS calc_flow_epoch_ledger", &[])
            .await
            .expect("clean ledger");
        client
            .execute(
                "CREATE TABLE orders (id BIGSERIAL PRIMARY KEY, amount BIGINT NOT NULL, label TEXT NOT NULL)",
                &[],
            )
            .await
            .expect("creates orders");
        client
            .execute(
                "CREATE TABLE orders_out (id BIGINT, amount BIGINT, label TEXT)",
                &[],
            )
            .await
            .expect("creates output");
        client
            .execute(
                "INSERT INTO orders (amount, label) VALUES (10, 'a'), (20, 'b'), (30, 'c')",
                &[],
            )
            .await
            .expect("seeds");

        // Snapshot source reads all rows.
        let config =
            PostgresSourceConfig::from_options(&source_options()).expect("source parses");
        let mut source =
            calc_flow_connectors::postgresql::PostgresSource::new(config).expect("builds");
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

        // Transactional sink writes and commits atomically.
        let sink_config = PostgresSinkConfig::from_options(&sink_options("transactional"))
            .expect("sink parses");
        let mut sink = calc_flow_connectors::postgresql::TransactionalPostgresSink::new(
            sink_config,
        )
        .expect("builds");
        sink.open_with_secrets(&PassUrl).expect("url set");
        sink.open().await.expect("connects and creates ledger");
        sink.begin_epoch(calc_flow::Epoch::INITIAL)
            .await
            .expect("begins");

        // Build a small batch to write.
        let schema = arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("amount", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("label", arrow::datatypes::DataType::Utf8, false),
        ]);
        let record = arrow::record_batch::RecordBatch::try_new(
            std::sync::Arc::new(schema),
            vec![
                std::sync::Arc::new(arrow::array::Int64Array::from(vec![100, 200])),
                std::sync::Arc::new(arrow::array::Int64Array::from(vec![5, 6])),
                std::sync::Arc::new(arrow::array::StringArray::from(vec!["x", "y"])),
            ],
        )
        .expect("record batch");
        let batch = calc_flow::Batch::table(
            vec![record],
            calc_flow::BatchMetadata::new("test", 1, BTreeMap::new()).unwrap(),
        )
        .unwrap();
        sink.write(&batch).await.expect("stages writes");
        let evidence = sink.pre_commit(calc_flow::Epoch::INITIAL).await.expect("pre");
        assert_eq!(
            evidence.get("rows").and_then(Value::as_u64),
            Some(2),
            "evidence carries the row count"
        );
        sink.commit(calc_flow::Epoch::INITIAL, &evidence)
            .await
            .expect("commits");

        let count: i64 = client
            .query_one("SELECT COUNT(*) FROM orders_out", &[])
            .await
            .expect("counts")
            .get(0);
        assert_eq!(count, 2, "committed rows landed");

        let ledger: i64 = client
            .query_one(
                "SELECT COUNT(*) FROM calc_flow_epoch_ledger WHERE pipeline = 'pg_test'",
                &[],
            )
            .await
            .expect("ledger count")
            .get(0);
        assert_eq!(ledger, 1, "the epoch ledger entry committed");

        // Replay the commit idempotently.
        sink.begin_epoch(calc_flow::Epoch::INITIAL)
            .await
            .expect("re-begins");
        let evidence2 = sink
            .pre_commit(calc_flow::Epoch::INITIAL)
            .await
            .expect("pre again");
        sink.commit(calc_flow::Epoch::INITIAL, &evidence2)
            .await
            .expect("replays");
        let count: i64 = client
            .query_one("SELECT COUNT(*) FROM orders_out", &[])
            .await
            .expect("recounts")
            .get(0);
        assert_eq!(count, 2, "replay adds no duplicates");
    });
}
