//! `MySQL` configuration, lifecycle, and service-backed recovery contracts.

#![cfg(feature = "mysql")]

use calc_flow::{ConnectorIdentity, ConnectorRegistry, JsonMap};
use calc_flow_connectors::register_mysql_connectors;
use serde_json::json;

use arrow::{
    array::{Int64Array, StringArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, ConnectorSinkFactory, ConnectorSourceFactory, Epoch, SecretHandle,
    SecretReference, SecretResolver, SinkDelivery, SinkRecovery, SourceEvent,
};
use calc_flow_connectors::{MySqlSinkFactory, MySqlSourceFactory};
use mysql_async::{Conn, prelude::Queryable};
use std::{collections::BTreeMap, sync::Arc};

fn options(table: &str, mode: &str) -> JsonMap {
    JsonMap::from([
        ("table".into(), json!(table)),
        ("mode".into(), json!(mode)),
        ("tls".into(), json!(false)),
    ])
}

fn incremental(table: &str) -> JsonMap {
    let mut options = options(table, "incremental_query");
    options.insert("cursor_columns".into(), json!(["id"]));
    options.insert("assume_monotonic_cursor".into(), json!(true));
    options.insert("poll_interval_ms".into(), json!(1));
    options.insert("max_batch_rows".into(), json!(2));
    options
}

fn transactional(table: &str) -> JsonMap {
    let mut options = options(table, "transactional");
    options.insert("pipeline".into(), json!("mysql_test"));
    options.insert("output".into(), json!(table));
    options
}

struct Url(String);
impl SecretResolver for Url {
    fn resolve(&self, reference: &SecretReference) -> calc_flow::Result<SecretHandle> {
        assert_eq!(reference.key, "url");
        Ok(SecretHandle::from_bytes(self.0.as_bytes()))
    }
}

fn test_url() -> Url {
    assert_eq!(
        std::env::var("CALC_FLOW_CONNECTOR_CONTAINERS").as_deref(),
        Ok("1"),
        "explicit opt-in required"
    );
    Url(std::env::var("CALC_FLOW_MYSQL_TEST_URL").expect("CALC_FLOW_MYSQL_TEST_URL is required"))
}

async fn admin(url: &Url) -> Conn {
    Conn::new(mysql_async::Opts::from_url(&url.0).unwrap())
        .await
        .unwrap()
}

fn batch(ids: &[i64], labels: &[&str]) -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(ids.to_vec())),
                    Arc::new(StringArray::from(labels.to_vec())),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::new("test", 1, BTreeMap::new()).unwrap(),
    )
    .unwrap()
}

#[test]
fn mysql_registry_exposes_modes_without_claiming_cdc() {
    let mut registry = ConnectorRegistry::new();
    register_mysql_connectors(&mut registry).unwrap();
    let snapshot = registry.snapshot();
    let identity = ConnectorIdentity::new("calc-flow-connectors", "mysql", "1.0.0").unwrap();
    let source = snapshot.resolve_source(&identity).unwrap();
    let sink = snapshot.resolve_sink(&identity).unwrap();
    assert!(source.descriptor().capabilities.snapshot);
    assert!(source.descriptor().capabilities.polling);
    assert!(!source.descriptor().capabilities.cdc);
    let options = JsonMap::from([("table".into(), json!("orders"))]);
    assert_eq!(
        source.capabilities(&options).unwrap().replay,
        calc_flow::ReplayCapability::Unreplayable
    );
    assert!(sink.validate(&options).is_ok());
}

#[test]
fn mysql_project_compilation_enforces_delivery_before_opening_connections() {
    let identity = json!({"provider":"calc-flow-connectors","name":"mysql","version":"1.0.0"});
    let mut document = json!({
        "format_version":3, "id":"mysql-project", "name":"MySQL project",
        "runtime":{"mode":"stream","options":{}},
        "graph":{"name":"mysql-project","nodes":[{"id":"calc","operator":{"kind":"expression","expression":"id = id + 1"}}]},
        "sources":[{"binding":"input","connector":identity,"options":options("orders","snapshot"),"secrets":{"url":{"resolver":"environment","key":"UNRESOLVED_MYSQL_TEST_URL"}},"watermark":{"policy":"disabled"}}],
        "sinks":[{"binding":"output","connector":identity,"options":options("orders_out","append"),"secrets":{"url":{"resolver":"environment","key":"UNRESOLVED_MYSQL_TEST_URL"}},"delivery":"best_effort"}]
    });
    let mut registry = ConnectorRegistry::new();
    register_mysql_connectors(&mut registry).unwrap();
    let compile = |value| {
        let project: calc_flow::ProjectSpec = serde_json::from_value(value).unwrap();
        calc_flow::compile_stream_project(
            &project,
            &calc_flow::ProviderRegistry::default(),
            &calc_flow::UdfRegistry::new().snapshot(),
            &registry.snapshot(),
            &calc_flow::StreamRequirements::default(),
        )
    };
    assert!(compile(document.clone()).unwrap().has_project_bindings());
    document["sinks"][0]["delivery"] = json!("exactly_once");
    assert!(compile(document.clone()).is_err());
    document["sources"][0]["options"] = json!(incremental("orders"));
    document["sinks"][0]["options"] = json!(transactional("orders_out"));
    assert!(compile(document.clone()).is_ok());
    document["sources"][0]["secrets"] = json!({});
    assert!(compile(document).is_err());
}

#[test]
fn invalid_mysql_options_fail_before_resolving_secrets() {
    let source = MySqlSourceFactory::default();
    let sink = MySqlSinkFactory::default();
    for (key, value) in [
        ("url", json!("mysql://secret")),
        ("mode", json!("binlog")),
        ("max_batch_rows", json!(0)),
        ("max_batch_bytes", json!(-1)),
        ("timeout_seconds", json!(3601)),
        ("table", json!("db.table")),
        ("table", json!("a`; DROP TABLE x")),
        ("tls", json!("false")),
        (
            "columns",
            json!([{"name":"x", "data_type":"unsupported", "nullable":false}]),
        ),
    ] {
        let mut config = options("orders", "snapshot");
        config.insert(key.into(), value);
        assert!(source.validate(&config).is_err(), "{key}");
    }
    let mut config = incremental("orders");
    assert!(source.validate(&config).is_ok());
    config.remove("assume_monotonic_cursor");
    assert!(source.validate(&config).is_err());
    config.insert("assume_monotonic_cursor".into(), json!(true));
    config.insert("cursor_columns".into(), json!(["id", "ID"]));
    assert!(source.validate(&config).is_err());
    assert!(sink.validate(&options("orders", "transactional")).is_err());
    assert!(
        sink.validate(&transactional("calc_flow_mysql_epoch_ledger"))
            .is_err()
    );
    assert_eq!(
        sink.capabilities(&options("orders", "upsert"))
            .unwrap()
            .transaction,
        calc_flow::TransactionSupport::None
    );
    assert_eq!(
        sink.capabilities(&transactional("orders"))
            .unwrap()
            .transaction,
        calc_flow::TransactionSupport::LedgerIdempotent
    );
}

#[tokio::test]
async fn malformed_credentials_and_inactive_lifecycles_fail_without_disclosure() {
    let factory = MySqlSourceFactory::new();
    let mut source = factory
        .open(
            &options("orders", "snapshot"),
            &Url("password-with-secret".into()),
        )
        .await
        .unwrap();
    let error = source.open(None).await.unwrap_err().to_string();
    assert!(!error.contains("password-with-secret"));
    assert!(source.next().await.is_err());
    source.close().await.unwrap();
    let mut sink = MySqlSinkFactory::new()
        .open_transactional(&transactional("orders"), &Url("invalid".into()))
        .await
        .unwrap()
        .unwrap();
    assert!(sink.begin_epoch(Epoch::INITIAL).await.is_err());
    assert!(sink.write(&batch(&[1], &["a"])).await.is_err());
    assert!(sink.pre_commit(Epoch::INITIAL).await.is_err());
    assert!(sink.commit(Epoch::INITIAL, &JsonMap::new()).await.is_err());
    assert!(sink.open().await.is_err());
    sink.abort(Epoch::INITIAL, None).await.unwrap();
    sink.close().await.unwrap();
}

#[tokio::test]
#[ignore = "requires MySQL 8.4 and CALC_FLOW_CONNECTOR_CONTAINERS=1"]
async fn mysql_snapshot_is_consistent_and_incremental_restore_preserves_unsigned_keys() {
    let url = test_url();
    let mut conn = admin(&url).await;
    conn.query_drop("DROP TABLE IF EXISTS mysql_source_test")
        .await
        .unwrap();
    conn.query_drop("CREATE TABLE mysql_source_test (id BIGINT UNSIGNED PRIMARY KEY, label TEXT NOT NULL) ENGINE=InnoDB").await.unwrap();
    conn.query_drop("INSERT INTO mysql_source_test VALUES (9007199254740993,'a'), (9007199254740994,'b'), (18446744073709551614,'c')").await.unwrap();
    let mut config = options("mysql_source_test", "snapshot");
    config.insert("max_batch_rows".into(), json!(2));
    let factory = MySqlSourceFactory::new();
    let mut snapshot = factory.open(&config, &url).await.unwrap();
    snapshot.open(None).await.unwrap();
    let Some(SourceEvent::Data {
        batch: first,
        cursor,
    }) = snapshot.next().await.unwrap()
    else {
        panic!("first snapshot page")
    };
    assert_eq!(first.num_rows(), 2);
    conn.query_drop("INSERT INTO mysql_source_test VALUES (18446744073709551615,'d')")
        .await
        .unwrap();
    let Some(SourceEvent::Data { batch: second, .. }) = snapshot.next().await.unwrap() else {
        panic!("second snapshot page")
    };
    assert_eq!(second.num_rows(), 1);
    assert!(snapshot.next().await.unwrap().is_none());
    snapshot.close().await.unwrap();
    let mut invalid_resume = factory.open(&config, &url).await.unwrap();
    assert!(invalid_resume.open(Some(cursor)).await.is_err());
    let config = incremental("mysql_source_test");
    let mut source = factory.open(&config, &url).await.unwrap();
    source.open(None).await.unwrap();
    let Some(SourceEvent::Data { cursor, .. }) = source.next().await.unwrap() else {
        panic!("incremental page")
    };
    source.close().await.unwrap();
    let mut resumed = factory.open(&config, &url).await.unwrap();
    resumed.open(Some(cursor)).await.unwrap();
    let Some(SourceEvent::Data { batch, .. }) = resumed.next().await.unwrap() else {
        panic!("resumed page")
    };
    let record = &batch.table_payload().unwrap().batches()[0];
    let ids = record
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(ids.values().as_ref(), &[u64::MAX - 1, u64::MAX]);
    assert!(matches!(
        resumed.next().await.unwrap(),
        Some(SourceEvent::Idle)
    ));
    resumed.close().await.unwrap();
    conn.disconnect().await.unwrap();
}

#[tokio::test]
#[ignore = "requires MySQL 8.4 and CALC_FLOW_CONNECTOR_CONTAINERS=1"]
async fn mysql_transaction_recovery_is_idempotent_and_rejects_changed_evidence() {
    let url = test_url();
    let mut conn = admin(&url).await;
    conn.query_drop("DROP TABLE IF EXISTS mysql_sink_test")
        .await
        .unwrap();
    conn.query_drop(
        "CREATE TABLE mysql_sink_test (id BIGINT PRIMARY KEY, label TEXT NOT NULL) ENGINE=InnoDB",
    )
    .await
    .unwrap();
    // A dedicated output identity makes this test independent of sibling tests.
    conn.query_drop("DROP TABLE IF EXISTS calc_flow_mysql_epoch_ledger")
        .await
        .unwrap();
    let factory = MySqlSinkFactory::new();
    let config = transactional("mysql_sink_test");
    let mut sink = factory
        .open_transactional(&config, &url)
        .await
        .unwrap()
        .unwrap();
    sink.open().await.unwrap();
    sink.begin_epoch(Epoch::INITIAL).await.unwrap();
    assert!(sink.begin_epoch(Epoch::INITIAL).await.is_err());
    sink.write(&batch(&[1, 2], &["a", "b"])).await.unwrap();
    let evidence = sink.pre_commit(Epoch::INITIAL).await.unwrap();
    assert!(sink.write(&batch(&[3], &["c"])).await.is_err());
    let segments = sink.pre_commit_segments(Epoch::INITIAL).await.unwrap();
    let count: Option<u64> = conn
        .query_first("SELECT COUNT(*) FROM mysql_sink_test")
        .await
        .unwrap();
    assert_eq!(count, Some(0));
    sink.close().await.unwrap();
    let recovery = SinkRecovery::from_parts(
        Epoch::INITIAL,
        false,
        SinkDelivery::Transactional,
        evidence.clone(),
    )
    .with_segments(segments.clone());
    let mut sink = factory
        .open_transactional(&config, &url)
        .await
        .unwrap()
        .unwrap();
    sink.open().await.unwrap();
    sink.recover(&recovery).await.unwrap();
    sink.recover(&recovery).await.unwrap();
    let count: Option<u64> = conn
        .query_first("SELECT COUNT(*) FROM mysql_sink_test")
        .await
        .unwrap();
    assert_eq!(count, Some(2));
    for (key, value) in [
        ("rows", json!(3)),
        ("pipeline", json!("wrong")),
        ("epoch", json!(99)),
        ("segment_sha256", json!("0".repeat(64))),
        ("schema_hash", json!("0".repeat(64))),
    ] {
        let mut bad = evidence.clone();
        bad.insert(key.into(), value);
        let recovery =
            SinkRecovery::from_parts(Epoch::INITIAL, false, SinkDelivery::Transactional, bad)
                .with_segments(segments.clone());
        assert!(sink.recover(&recovery).await.is_err(), "{key}");
    }
    // Same epoch/row count with different content must conflict with the ledger.
    sink.begin_epoch(Epoch::INITIAL).await.unwrap();
    sink.write(&batch(&[3, 4], &["c", "d"])).await.unwrap();
    let other = sink.pre_commit(Epoch::INITIAL).await.unwrap();
    assert!(
        sink.commit(Epoch::INITIAL, &other)
            .await
            .unwrap_err()
            .to_string()
            .contains("conflicts")
    );
    sink.close().await.unwrap();
    conn.disconnect().await.unwrap();
}

#[tokio::test]
#[ignore = "requires MySQL 8.4 and CALC_FLOW_CONNECTOR_CONTAINERS=1"]
async fn mysql_append_upsert_and_failed_transaction_are_atomic() {
    let url = test_url();
    let mut conn = admin(&url).await;
    conn.query_drop("DROP TABLE IF EXISTS mysql_append_test")
        .await
        .unwrap();
    conn.query_drop(
        "CREATE TABLE mysql_append_test (id BIGINT PRIMARY KEY, label TEXT NOT NULL) ENGINE=InnoDB",
    )
    .await
    .unwrap();
    let factory = MySqlSinkFactory::new();
    let mut sink = factory
        .open(&options("mysql_append_test", "append"), &url)
        .await
        .unwrap();
    sink.open().await.unwrap();
    sink.write(&batch(&[1], &["before"])).await.unwrap();
    sink.close().await.unwrap();
    let mut sink = factory
        .open(&options("mysql_append_test", "upsert"), &url)
        .await
        .unwrap();
    sink.open().await.unwrap();
    sink.write(&batch(&[1, 2], &["after", "new"]))
        .await
        .unwrap();
    sink.close().await.unwrap();
    let rows: Vec<(i64, String)> = conn
        .query("SELECT id,label FROM mysql_append_test ORDER BY id")
        .await
        .unwrap();
    assert_eq!(rows, vec![(1, "after".into()), (2, "new".into())]);
    let mut sink = factory
        .open(&options("mysql_append_test", "append"), &url)
        .await
        .unwrap();
    sink.open().await.unwrap();
    let error = sink
        .write(&batch(&[3, 1], &["rolled-back-secret", "duplicate-secret"]))
        .await
        .unwrap_err()
        .to_string();
    assert!(!error.contains("secret"));
    sink.close().await.unwrap();
    let count: Option<u64> = conn
        .query_first("SELECT COUNT(*) FROM mysql_append_test")
        .await
        .unwrap();
    assert_eq!(count, Some(2));
    conn.disconnect().await.unwrap();
}

#[tokio::test]
#[ignore = "requires MySQL 8.4 and CALC_FLOW_CONNECTOR_CONTAINERS=1"]
async fn mysql_type_matrix_roundtrips_without_numeric_or_temporal_loss() {
    use arrow::array::{Array, Date32Array, TimestampMicrosecondArray};
    let url = test_url();
    let mut conn = admin(&url).await;
    conn.query_drop("DROP TABLE IF EXISTS mysql_types_test")
        .await
        .unwrap();
    conn.query_drop("CREATE TABLE mysql_types_test (id BIGINT PRIMARY KEY, si TINYINT, ui TINYINT UNSIGNED, ss SMALLINT, us SMALLINT UNSIGNED, mi MEDIUMINT, um MEDIUMINT UNSIGNED, ii INT, uu INT UNSIGNED, bi BIGINT, ub BIGINT UNSIGNED, f FLOAT, d DOUBLE, txt TEXT, bin BLOB, deci DECIMAL(65,30), js JSON, dt DATE, ts DATETIME(6), stamp TIMESTAMP(6), tm TIME(6), yr YEAR, bits BIT(9)) ENGINE=InnoDB").await.unwrap();
    conn.query_drop("SET time_zone = '+00:00'").await.unwrap();
    conn.query_drop("INSERT INTO mysql_types_test VALUES (1,-128,255,-32768,65535,-8388608,16777215,-2147483648,4294967295,-9223372036854775808,18446744073709551615,1.25,2.5,'你好',X'00FF',12345678901234567890123456789012345.123456789012345678901234567890,JSON_OBJECT('key',1),'1969-12-31','2026-09-06 01:02:03.123456','2026-09-06 01:02:03.123456','-25:02:03.123456',2026,b'100000001')").await.unwrap();
    conn.query_drop("INSERT INTO mysql_types_test (id) VALUES (2)")
        .await
        .unwrap();
    let factory = MySqlSourceFactory::new();
    let mut source = factory
        .open(&options("mysql_types_test", "snapshot"), &url)
        .await
        .unwrap();
    source.open(None).await.unwrap();
    let Some(SourceEvent::Data { batch, .. }) = source.next().await.unwrap() else {
        panic!("typed batch")
    };
    let record = &batch.table_payload().unwrap().batches()[0];
    assert_eq!(
        record
            .column(10)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0),
        u64::MAX
    );
    assert_eq!(
        record
            .column(15)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "12345678901234567890123456789012345.123456789012345678901234567890"
    );
    assert_eq!(
        record
            .column(17)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap()
            .value(0),
        -1
    );
    assert_eq!(
        record
            .column(18)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        record
            .column(19)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0)
    );
    assert_eq!(
        record
            .column(20)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "-25:02:03.123456"
    );
    for column in record.columns().iter().skip(1) {
        assert!(column.is_null(1));
    }
    source.close().await.unwrap();
    conn.query_drop("DROP TABLE IF EXISTS mysql_types_out")
        .await
        .unwrap();
    conn.query_drop("CREATE TABLE mysql_types_out LIKE mysql_types_test")
        .await
        .unwrap();
    let mut sink = MySqlSinkFactory::new()
        .open(&options("mysql_types_out", "append"), &url)
        .await
        .unwrap();
    sink.open().await.unwrap();
    sink.write(&batch).await.unwrap();
    sink.close().await.unwrap();
    let mut source = factory
        .open(&options("mysql_types_out", "snapshot"), &url)
        .await
        .unwrap();
    source.open(None).await.unwrap();
    let Some(SourceEvent::Data { batch: copied, .. }) = source.next().await.unwrap() else {
        panic!("copied batch")
    };
    assert_eq!(record, &copied.table_payload().unwrap().batches()[0]);
    source.close().await.unwrap();
    conn.disconnect().await.unwrap();
}

#[tokio::test]
#[ignore = "requires MySQL 8.4 and CALC_FLOW_CONNECTOR_CONTAINERS=1"]
async fn mysql_source_preflight_rejects_unsafe_tables_projections_and_tls() {
    let url = test_url();
    let mut conn = admin(&url).await;
    for sql in [
        "DROP TABLE IF EXISTS mysql_preflight_test",
        "CREATE TABLE mysql_preflight_test (id BIGINT NULL, label TEXT, UNIQUE KEY (id)) ENGINE=InnoDB",
        "INSERT INTO mysql_preflight_test VALUES (1,'row')",
    ] {
        conn.query_drop(sql).await.unwrap();
    }
    let factory = MySqlSourceFactory::new();
    for config in [
        options("mysql_preflight_test", "snapshot"),
        incremental("mysql_preflight_test"),
    ] {
        let mut source = factory.open(&config, &url).await.unwrap();
        assert!(source.open(None).await.is_err());
        source.close().await.unwrap();
    }
    conn.query_drop(
        "ALTER TABLE mysql_preflight_test MODIFY id BIGINT NOT NULL, ADD PRIMARY KEY (id)",
    )
    .await
    .unwrap();
    let mut config = options("mysql_preflight_test", "snapshot");
    config.insert(
        "columns".into(),
        json!([{"name":"id", "data_type":"string", "nullable":false}]),
    );
    let mut source = factory.open(&config, &url).await.unwrap();
    assert!(source.open(None).await.is_err());
    config.remove("columns");
    config.insert("tls".into(), json!(true));
    let mut source = factory.open(&config, &url).await.unwrap();
    assert!(
        source.open(None).await.is_err(),
        "self-signed test certificate must not be trusted"
    );
    config.insert("tls".into(), json!(false));
    config.insert("max_batch_bytes".into(), json!(16));
    let mut source = factory.open(&config, &url).await.unwrap();
    source.open(None).await.unwrap();
    assert!(source.next().await.is_err());
    assert!(source.next().await.is_err());
    source.close().await.unwrap();
    conn.query_drop("ALTER TABLE mysql_preflight_test ENGINE=MyISAM")
        .await
        .unwrap();
    let mut source = factory
        .open(&incremental("mysql_preflight_test"), &url)
        .await
        .unwrap();
    assert!(source.open(None).await.is_err());
    let mut sink = MySqlSinkFactory::new()
        .open(&options("mysql_preflight_test", "append"), &url)
        .await
        .unwrap();
    assert!(sink.open().await.is_err());
    conn.disconnect().await.unwrap();
}

#[tokio::test]
#[ignore = "requires MySQL 8.4 and CALC_FLOW_CONNECTOR_CONTAINERS=1"]
async fn mysql_composite_cursor_validates_identity_and_resumes_after_ties() {
    let url = test_url();
    let mut conn = admin(&url).await;
    conn.query_drop("DROP TABLE IF EXISTS mysql_composite_test")
        .await
        .unwrap();
    conn.query_drop("CREATE TABLE mysql_composite_test (partition_id BIGINT NOT NULL, id BIGINT NOT NULL, PRIMARY KEY(partition_id,id)) ENGINE=InnoDB").await.unwrap();
    conn.query_drop("INSERT INTO mysql_composite_test VALUES (1,1),(1,2),(2,1),(2,2)")
        .await
        .unwrap();
    let factory = MySqlSourceFactory::new();
    let mut config = incremental("mysql_composite_test");
    config.insert("cursor_columns".into(), json!(["partition_id", "id"]));
    config.insert("columns".into(), json!([{"name":"id","data_type":"int64","nullable":false},{"name":"partition_id","data_type":"int64","nullable":false}]));
    let mut source = factory.open(&config, &url).await.unwrap();
    source.open(None).await.unwrap();
    let Some(SourceEvent::Data { cursor, .. }) = source.next().await.unwrap() else {
        panic!("page")
    };
    source.close().await.unwrap();
    for (key, value) in [
        ("table", json!("other")),
        ("values", json!([1])),
        ("values", json!([null, 2])),
        ("sequence", json!(0)),
        ("schema", json!("bad")),
        ("columns", json!(["id", "partition_id"])),
    ] {
        let mut payload = cursor.payload().clone();
        payload.insert(key.into(), value);
        let invalid = calc_flow::Cursor::unbound(cursor.order().to_vec(), payload).unwrap();
        let mut source = factory.open(&config, &url).await.unwrap();
        assert!(source.open(Some(invalid)).await.is_err(), "{key}");
    }
    let mut source = factory.open(&config, &url).await.unwrap();
    source.open(Some(cursor)).await.unwrap();
    let Some(SourceEvent::Data { batch, .. }) = source.next().await.unwrap() else {
        panic!("second partition")
    };
    let record = &batch.table_payload().unwrap().batches()[0];
    assert_eq!(
        record
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[1, 2]
    );
    assert_eq!(
        record
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[2, 2]
    );
    conn.query_drop("ALTER TABLE mysql_composite_test MODIFY id INT NOT NULL")
        .await
        .unwrap();
    assert!(source.next().await.is_err());
    source.close().await.unwrap();
    conn.disconnect().await.unwrap();
}
