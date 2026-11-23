use std::{collections::BTreeMap, sync::Arc};

use datafusion::{
    arrow::{
        array::{ArrayRef, Int64Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
};
use serde_json::json;

use super::*;
use crate::{
    CancellationToken, EdgeCollector, Epoch, OperatorMetadata, OperatorSpec, StreamJobContext,
    StreamOperator,
};

const GROUPED: &str = "SELECT key, SUM(value) AS total FROM events GROUP BY key";

fn changelog(query: &str) -> SqlOperator {
    let mut operator =
        SqlOperator::new("changes", query, vec!["events".into()], Vec::new()).unwrap();
    operator
        .set_aggregate_output_mode(SqlAggregateOutputMode::Changelog)
        .unwrap();
    operator
}

fn input(keys: &[Option<&str>], values: &[Option<i64>]) -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(StringArray::from(keys.to_vec())) as ArrayRef,
                    Arc::new(Int64Array::from(values.to_vec())) as ArrayRef,
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "changelog",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn rows(batch: &Batch) -> Vec<Vec<ScalarValue>> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|column| ScalarValue::try_from_array(column, row).unwrap())
                    .collect::<Vec<_>>()
            })
        })
        .collect()
}

async fn process(
    operator: &mut SqlOperator,
    batch: Batch,
    context: &StreamOperatorContext<'_>,
) -> Vec<Vec<ScalarValue>> {
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", batch, context, &mut collector)
        .await
        .unwrap();
    let emitted = collector.drain("output");
    assert_eq!(emitted.len(), 1);
    rows(emitted[0].as_data().unwrap())
}

fn text(value: &str) -> ScalarValue {
    ScalarValue::Utf8(Some(value.into()))
}

fn int(value: i64) -> ScalarValue {
    ScalarValue::Int64(Some(value))
}

#[test]
fn test_sql_aggregate_output_defaults_and_project_v3_opt_in() {
    let mut operator =
        SqlOperator::new("changes", GROUPED, vec!["events".into()], Vec::new()).unwrap();
    assert_eq!(
        operator.aggregate_output_mode(),
        SqlAggregateOutputMode::Snapshot
    );
    assert!(!operator.configuration().contains_key("aggregate_output"));
    operator
        .set_aggregate_output_mode(SqlAggregateOutputMode::Changelog)
        .unwrap();
    assert_eq!(
        operator.configuration()["aggregate_output"],
        json!("changelog")
    );

    let default: OperatorSpec = serde_json::from_value(json!({
        "kind": "sql", "query": GROUPED, "aliases": ["events"], "udfs": []
    }))
    .unwrap();
    assert!(matches!(
        default,
        OperatorSpec::Sql {
            aggregate_output: SqlAggregateOutputMode::Snapshot,
            ..
        }
    ));
    assert!(
        serde_json::to_value(default)
            .unwrap()
            .get("aggregate_output")
            .is_none()
    );
    let opted_in: OperatorSpec = serde_json::from_value(json!({
        "kind": "sql",
        "query": GROUPED,
        "aliases": ["events"],
        "udfs": [],
        "aggregate_output": "changelog"
    }))
    .unwrap();
    assert!(matches!(
        opted_in,
        OperatorSpec::Sql {
            aggregate_output: SqlAggregateOutputMode::Changelog,
            ..
        }
    ));
    assert_eq!(
        serde_json::to_value(opted_in).unwrap()["aggregate_output"],
        json!("changelog")
    );
}

#[tokio::test]
async fn test_changelog_orders_group_ids_and_retracts_complete_old_rows() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "changes", None);
    let mut operator = changelog(GROUPED);
    let first = process(
        &mut operator,
        input(
            &[Some("b"), Some("a"), Some("a")],
            &[Some(2), Some(1), Some(3)],
        ),
        &context,
    )
    .await;
    assert_eq!(
        first,
        vec![
            vec![text("b"), int(2), text("upsert")],
            vec![text("a"), int(4), text("upsert")],
        ]
    );
    let second = process(&mut operator, input(&[Some("a")], &[Some(1)]), &context).await;
    assert_eq!(
        second,
        vec![
            vec![text("a"), int(4), text("retract")],
            vec![text("a"), int(5), text("upsert")],
        ]
    );
    assert!(
        process(&mut operator, input(&[Some("b")], &[None]), &context,)
            .await
            .is_empty()
    );
    let third = process(
        &mut operator,
        input(&[Some("c"), Some("b")], &[Some(7), Some(1)]),
        &context,
    )
    .await;
    assert_eq!(
        third,
        vec![
            vec![text("b"), int(2), text("retract")],
            vec![text("b"), int(3), text("upsert")],
            vec![text("c"), int(7), text("upsert")],
        ]
    );
}

#[tokio::test]
async fn test_sparse_keyed_sum_projects_only_affected_group_versions() {
    use std::sync::atomic::Ordering;

    let job = job();
    let context = StreamOperatorContext::new(&job, "changes", None);
    let mut operator = changelog(GROUPED);
    let keys = (0..256).map(|i| i.to_string()).collect::<Vec<_>>();
    let initial = keys
        .iter()
        .map(|key| Some(key.as_str()))
        .collect::<Vec<_>>();
    let first = process(
        &mut operator,
        input(&initial, &vec![Some(10); initial.len()]),
        &context,
    )
    .await;
    assert_eq!(first.len(), 256);
    let projected_batches = operator
        .incremental
        .as_ref()
        .unwrap()
        .keyed_output_batches
        .load(Ordering::Relaxed);
    assert_eq!(projected_batches, 1);
    let projected = |operator: &SqlOperator| {
        operator
            .incremental
            .as_ref()
            .unwrap()
            .keyed_output_rows
            .load(Ordering::Relaxed)
    };
    let before = projected(&operator);
    let changed = process(
        &mut operator,
        input(
            &[Some("new"), Some("197"), Some("197")],
            &[None, Some(2), Some(-1)],
        ),
        &context,
    )
    .await;
    assert_eq!(
        changed,
        vec![
            vec![text("197"), int(10), text("retract")],
            vec![text("197"), int(11), text("upsert")],
            vec![text("new"), ScalarValue::Int64(None), text("upsert")],
        ]
    );
    assert_eq!(projected(&operator) - before, 3);
}

#[tokio::test]
async fn test_direct_sum_reserves_exact_output_once_before_export() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "changes", None);
    let mut operator = changelog(GROUPED);
    let mut keys = (0..257).map(|id| format!("key-{id}")).collect::<Vec<_>>();
    keys[256] = "long-key".repeat(2_048);
    let borrowed = keys
        .iter()
        .map(|key| Some(key.as_str()))
        .collect::<Vec<_>>();
    let calls = |operator: &SqlOperator| {
        operator
            .incremental
            .as_ref()
            .unwrap()
            .changelog_output_reservations
            .swap(0, std::sync::atomic::Ordering::Relaxed)
    };
    let first = process(
        &mut operator,
        input(&borrowed, &vec![Some(1); keys.len()]),
        &context,
    )
    .await;
    assert_eq!(first.len(), keys.len());
    assert_eq!(first[256], [text(&keys[256]), int(1), text("upsert")]);
    let seed = calls(&operator);
    let unchanged = process(
        &mut operator,
        input(&borrowed, &vec![Some(0); keys.len()]),
        &context,
    )
    .await;
    assert!(unchanged.is_empty());
    let noop = calls(&operator);
    let changed = process(
        &mut operator,
        input(
            &[Some(&keys[256]), Some("new"), Some(&keys[0])],
            &[Some(5), Some(9), Some(7)],
        ),
        &context,
    )
    .await;
    assert_eq!(
        changed,
        [
            vec![text(&keys[0]), int(1), text("retract")],
            vec![text(&keys[0]), int(8), text("upsert")],
            vec![text(&keys[256]), int(1), text("retract")],
            vec![text(&keys[256]), int(6), text("upsert")],
            vec![text("new"), int(9), text("upsert")],
        ]
    );
    let update = calls(&operator);
    assert_eq!(
        [seed, noop, update],
        [1, 0, 1],
        "output reservation increments"
    );
}

#[tokio::test]
async fn test_small_changelog_operation_column_keeps_only_small_buffers() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "changes", None);
    let mut operator = changelog(GROUPED);
    for (value, count) in [(1, 1), (0, 0), (1, 2)] {
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "events",
                input(&[Some("a")], &[Some(value)]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        let emitted = collector.drain("output");
        let table = emitted[0].as_data().unwrap().table_payload().unwrap();
        let operations = table.batches()[0].column_by_name("_cf_op").unwrap();
        assert_eq!(operations.len(), count);
        let bytes = operations.get_buffer_memory_size();
        assert!(bytes <= 256, "{count} operations retain {bytes} bytes");
    }
}

#[tokio::test]
async fn test_changelog_preserves_projected_versions_across_key_morsels() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "changes", None);
    let mut operator = changelog(
        "SELECT SUM(value) AS total, key AS id, SUM(value) AS repeated FROM events GROUP BY key",
    );
    let keys = (0..4097).map(|i| format!("key-{i}")).collect::<Vec<_>>();
    let names = keys
        .iter()
        .map(|key| Some(key.as_str()))
        .collect::<Vec<_>>();
    let batch = input(&names, &vec![Some(1); keys.len()]);
    process(&mut operator, batch.clone(), &context).await;
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", batch, &context, &mut collector)
        .await
        .unwrap();
    let emitted = collector.drain("output");
    let [message] = emitted.as_slice() else {
        panic!("expected one output")
    };
    let batch = message.as_data().unwrap();
    assert_eq!(batch.table_payload().unwrap().batches().len(), 1);
    let actual = rows(batch);
    assert_eq!(actual.len(), keys.len() * 2);
    for (key, pair) in keys.iter().zip(actual.chunks_exact(2)) {
        assert_eq!(pair[0], [int(1), text(key), int(1), text("retract")]);
        assert_eq!(pair[1], [int(2), text(key), int(2), text("upsert")]);
    }
}

#[tokio::test]
async fn test_changelog_failed_emit_retries_projected_changes_in_group_id_order() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "changes", None);
    let mut operator = changelog(
        "SELECT key, SUM(value) >= 3 AS large FROM events GROUP BY key HAVING SUM(value) >= 0",
    );
    let flag = |value| ScalarValue::Boolean(Some(value));
    process(
        &mut operator,
        input(
            &[Some("a"), Some("b"), Some("c"), Some("d")],
            &[Some(1), Some(2), Some(-1), Some(4)],
        ),
        &context,
    )
    .await;
    let incoming = input(
        &[Some("new"), Some("d"), Some("c"), Some("b"), Some("a")],
        &[Some(2), Some(1), Some(1), Some(-4), Some(2)],
    );
    let mut reject_output = EdgeCollector::new(Vec::new());
    assert!(
        matches!(operator.process_data("events", incoming.clone(), &context, &mut reject_output).await,
                     Err(CalcFlowError::Compile { message }) if message.contains("unknown output port"))
    );
    assert_eq!(
        process(&mut operator, incoming, &context).await,
        vec![
            vec![text("a"), flag(false), text("retract")],
            vec![text("a"), flag(true), text("upsert")],
            vec![text("b"), flag(false), text("retract")],
            vec![text("c"), flag(false), text("upsert")],
            vec![text("new"), flag(false), text("upsert")],
        ]
    );
}

#[tokio::test]
async fn test_changelog_having_enter_exit_and_stay_hidden() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "changes", None);
    let mut operator = changelog(
        "SELECT key, SUM(value) AS total FROM events GROUP BY key HAVING SUM(value) >= 3",
    );
    assert!(
        process(&mut operator, input(&[Some("a")], &[Some(1)]), &context)
            .await
            .is_empty()
    );
    assert_eq!(
        process(&mut operator, input(&[Some("a")], &[Some(2)]), &context).await,
        vec![vec![text("a"), int(3), text("upsert")]]
    );
    assert_eq!(
        process(&mut operator, input(&[Some("a")], &[Some(-4)]), &context).await,
        vec![vec![text("a"), int(3), text("retract")]]
    );
    assert!(
        process(&mut operator, input(&[Some("a")], &[Some(0)]), &context)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn test_changelog_global_singleton_and_null_group() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "changes", None);
    let mut global = changelog("SELECT SUM(value) AS total FROM events");
    assert_eq!(
        process(&mut global, input(&[], &[]), &context).await,
        vec![vec![ScalarValue::Int64(None), text("upsert")]]
    );
    assert_eq!(
        process(&mut global, input(&[Some("ignored")], &[Some(4)]), &context,).await,
        vec![
            vec![ScalarValue::Int64(None), text("retract")],
            vec![int(4), text("upsert")],
        ]
    );

    let mut grouped = changelog(GROUPED);
    assert_eq!(
        process(&mut grouped, input(&[None], &[Some(4)]), &context).await,
        vec![vec![ScalarValue::Utf8(None), int(4), text("upsert")]]
    );
}

#[tokio::test]
async fn test_changelog_restore_has_no_bootstrap_and_diffs_from_committed_state() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "changes", None);
    let mut operator = changelog(GROUPED);
    drop(
        process(
            &mut operator,
            input(&[Some("a"), Some("b")], &[Some(1), Some(2)]),
            &context,
        )
        .await,
    );
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = changelog(GROUPED);
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    assert_eq!(
        process(&mut restored, input(&[Some("a")], &[Some(4)]), &context,).await,
        vec![
            vec![text("a"), int(1), text("retract")],
            vec![text("a"), int(5), text("upsert")],
        ]
    );
}

#[tokio::test]
async fn test_changelog_rejects_non_compact_and_invalid_visible_shapes() {
    for query in [
        "SELECT SUM(value) AS total FROM events GROUP BY key",
        "SELECT key, SUM(value) AS total FROM events GROUP BY key ORDER BY key",
        "SELECT key, SUM(value) AS total FROM events GROUP BY key LIMIT 1",
        "SELECT key AS _cf_op, SUM(value) AS total FROM events GROUP BY key",
        "SELECT key, COUNT(DISTINCT value) AS n FROM events GROUP BY key",
    ] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "changes", None);
        let mut operator = changelog(query);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let result = operator
            .process_data(
                "events",
                input(&[Some("a")], &[Some(1)]),
                &context,
                &mut collector,
            )
            .await;
        assert!(result.is_err(), "query must be rejected: {query}");
        assert!(collector.drain("output").is_empty());
    }
}

#[test]
fn test_changelog_rejects_non_aggregate_multi_alias_and_latest_relation() {
    let mut row_sql = SqlOperator::new(
        "row",
        "SELECT value FROM events",
        vec!["events".into()],
        Vec::new(),
    )
    .unwrap();
    assert!(
        row_sql
            .set_aggregate_output_mode(SqlAggregateOutputMode::Changelog)
            .is_err()
    );
    let mut multi = SqlOperator::new(
        "multi",
        "SELECT 1 AS value",
        vec!["left".into(), "right".into()],
        Vec::new(),
    )
    .unwrap();
    assert!(
        multi
            .set_aggregate_output_mode(SqlAggregateOutputMode::Changelog)
            .is_err()
    );
    let mut latest = changelog(GROUPED);
    assert!(latest.enable_replacement().is_err());
}

#[tokio::test]
async fn test_changelog_schema_appends_non_null_operation() {
    let operator = changelog(GROUPED);
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let schema = operator
        .infer_schema(&BTreeMap::from([("events".into(), input_schema)]))
        .await
        .unwrap();
    assert_eq!(schema.fields().len(), 3);
    let operation = schema.field(2);
    assert_eq!(operation.name(), "_cf_op");
    assert_eq!(operation.data_type(), &DataType::Utf8);
    assert!(!operation.is_nullable());
}

#[test]
fn test_changelog_scalar_equality_includes_float_bit_patterns() {
    assert!(!incremental::visible_scalar_eq(
        &ScalarValue::Float64(Some(-0.0)),
        &ScalarValue::Float64(Some(0.0)),
    ));
    assert!(incremental::visible_scalar_eq(
        &ScalarValue::Float32(Some(f32::from_bits(0x7fc0_0001))),
        &ScalarValue::Float32(Some(f32::from_bits(0x7fc0_0001))),
    ));
    assert!(!incremental::visible_scalar_eq(
        &ScalarValue::Float32(Some(f32::from_bits(0x7fc0_0001))),
        &ScalarValue::Float32(Some(f32::from_bits(0x7fc0_0002))),
    ));
}
