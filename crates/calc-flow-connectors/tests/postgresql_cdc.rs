//! Container-backed gap-free `PostgreSQL` exported-snapshot to `pgoutput` test.

#![cfg(feature = "postgresql")]

use std::collections::BTreeMap;
use std::time::Duration;

use arrow::array::StringArray;
use calc_flow::{ConnectorSourceFactory as _, SourceEvent};
use calc_flow_connectors::postgresql::PostgresSourceFactory;
use serde_json::{Value, json};

struct TestUrl;

impl calc_flow::SecretResolver for TestUrl {
    fn resolve(
        &self,
        _reference: &calc_flow::SecretReference,
    ) -> calc_flow::Result<calc_flow::SecretHandle> {
        Ok(calc_flow::SecretHandle::from_bytes(
            connection_url().as_bytes(),
        ))
    }
}

fn connection_url() -> String {
    std::env::var("CALC_FLOW_PG_TEST_URL").unwrap_or_else(|_| {
        "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable".into()
    })
}

fn containers_enabled() -> bool {
    std::env::var("CALC_FLOW_CONNECTOR_CONTAINERS").as_deref() == Ok("1")
}

fn cdc_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("table".into(), json!("cdc_orders")),
        ("mode".into(), json!("logical_cdc")),
        ("slot".into(), json!("calc_flow_cdc_orders")),
        ("publication".into(), json!("calc_flow_cdc_publication")),
        ("slot_policy".into(), json!("recreate_with_snapshot")),
        ("require_before".into(), json!(true)),
        ("max_batch_rows".into(), json!(2)),
        (
            "columns".into(),
            json!([
                {"name": "id", "data_type": "int64", "nullable": false},
                {"name": "label", "data_type": "string", "nullable": false}
            ]),
        ),
    ])
}

#[tokio::test]
#[ignore = "container-backed; PostgreSQL must use wal_level=logical"]
async fn exported_snapshot_hands_off_to_pgoutput_without_a_gap() {
    if !containers_enabled() {
        return;
    }
    let (client, connection) = tokio_postgres::connect(&connection_url(), tokio_postgres::NoTls)
        .await
        .expect("connects to PostgreSQL");
    let connection = tokio::spawn(connection);
    client
        .batch_execute(
            "DROP PUBLICATION IF EXISTS calc_flow_cdc_publication; \
             DROP TABLE IF EXISTS cdc_orders; \
             CREATE TABLE cdc_orders (id BIGINT PRIMARY KEY, label TEXT NOT NULL); \
             ALTER TABLE cdc_orders REPLICA IDENTITY FULL; \
             INSERT INTO cdc_orders VALUES (1, 'one'), (2, 'two'), (3, 'three'); \
             CREATE PUBLICATION calc_flow_cdc_publication FOR TABLE cdc_orders;",
        )
        .await
        .expect("prepares CDC relation and publication");

    let factory = PostgresSourceFactory::new();
    let mut source = factory
        .open(&cdc_options(), &TestUrl)
        .await
        .expect("factory opens the CDC source");
    source
        .open(None)
        .await
        .expect("exports and imports snapshot");

    let first = next_data(&mut *source).await;
    assert_eq!(first.num_rows(), 2);
    assert_eq!(first.metadata().attributes()["snapshot"], Value::Bool(true));

    client
        .execute(
            "INSERT INTO cdc_orders VALUES \
             (4, 'four'), (5, 'five'), (6, 'six'), (7, 'seven'), (8, 'eight')",
            &[],
        )
        .await
        .expect("commits one multi-row transaction after the exported snapshot");

    let second = next_data(&mut *source).await;
    assert_eq!(second.num_rows(), 1, "snapshot excludes the later insert");
    assert_eq!(
        second.metadata().attributes()["snapshot"],
        Value::Bool(true)
    );

    let first_cdc = next_data(&mut *source).await;
    let second_cdc = next_data(&mut *source).await;
    let final_cdc = next_data(&mut *source).await;
    assert_eq!(first_cdc.num_rows(), 2);
    assert_eq!(second_cdc.num_rows(), 2);
    assert_eq!(final_cdc.num_rows(), 1);
    assert_eq!(
        first_cdc.metadata().attributes()["transaction_complete"],
        Value::Bool(false)
    );
    assert_eq!(
        final_cdc.metadata().attributes()["transaction_complete"],
        Value::Bool(true)
    );
    let record = &first_cdc.table_payload().unwrap().batches()[0];
    let operations = record
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("operation is Utf8");
    assert_eq!(operations.value(0), "insert");
    assert!(!first_cdc.metadata().attributes().contains_key("snapshot"));

    source.close().await.expect("joins the replication worker");
    client
        .batch_execute(
            "SELECT pg_drop_replication_slot('calc_flow_cdc_orders') \
             WHERE EXISTS (SELECT 1 FROM pg_replication_slots \
                           WHERE slot_name = 'calc_flow_cdc_orders'); \
             DROP PUBLICATION calc_flow_cdc_publication; \
             DROP TABLE cdc_orders;",
        )
        .await
        .expect("cleans CDC artifacts");
    drop(client);
    connection
        .await
        .expect("connection task joins")
        .expect("connection closes");
}

async fn next_data(source: &mut dyn calc_flow::StreamSource) -> calc_flow::Batch {
    let event = tokio::time::timeout(Duration::from_secs(15), source.next())
        .await
        .expect("source event does not time out")
        .expect("source read succeeds")
        .expect("continuous CDC source does not end");
    match event {
        SourceEvent::Data { batch, .. } => batch,
        SourceEvent::Idle | SourceEvent::Watermark(_) => {
            panic!("expected one data event")
        }
    }
}
