use super::*;
use crate::operator::{OperatorMetadata, StreamOperator};
use crate::{
    CancellationToken, DataFusionConfig, EdgeCollector, Epoch, JsonMap, SqlOperator,
    StreamJobContext,
};
use datafusion::arrow::array::{Int64Array, StringArray};
use std::collections::BTreeMap;

fn phase_values(unsigned: bool, phase: usize) -> Vec<Option<u64>> {
    match phase {
        0 => vec![
            Some(u64::MAX >> usize::from(!unsigned)),
            None,
            None,
            Some(5),
        ],
        1 => vec![Some(1), Some(7), None, Some(99), Some(99)],
        2 => (0..=CHUNK_ROWS)
            .map(|index| (index % 3 != 0).then_some(2))
            .collect(),
        5 => (0..=CHUNK_ROWS)
            .map(|index| (index % 4 != 0).then_some(3))
            .collect(),
        3 => vec![],
        _ => vec![None, Some(2), None, Some(99)],
    }
}

fn value_array(unsigned: bool, values: Vec<Option<u64>>) -> ArrayRef {
    if unsigned {
        Arc::new(datafusion::arrow::array::UInt64Array::from(values))
    } else {
        Arc::new(Int64Array::from(
            values
                .into_iter()
                .map(|value| value.map(|value| i64::try_from(value).unwrap()))
                .collect::<Vec<_>>(),
        ))
    }
}

fn key_array(native_key: bool, keys: &[i64]) -> ArrayRef {
    if native_key {
        Arc::new(Int64Array::from(keys.to_vec()))
    } else {
        Arc::new(StringArray::from(
            keys.iter()
                .map(|key| (*key != 2).then(|| format!("account-{key}")))
                .collect::<Vec<_>>(),
        ))
    }
}

fn enabled_array(keys: &[i64], phase: usize) -> BooleanArray {
    BooleanArray::from(
        keys.iter()
            .map(|key| match phase {
                2 => *key % 5 != 0,
                5 => *key % 7 != 0,
                _ => *key < 3 || phase == 0,
            })
            .collect::<Vec<_>>(),
    )
}

fn record(unsigned: bool, native_key: bool, phase: usize) -> RecordBatch {
    let values = phase_values(unsigned, phase);
    let keys = (0..values.len())
        .map(|index| i64::try_from(index).unwrap())
        .collect::<Vec<_>>();
    let mut columns = vec![("key", key_array(native_key, &keys))];
    if !native_key {
        columns.push((
            "stock",
            Arc::new(StringArray::from(vec![Some("stock"); keys.len()])) as ArrayRef,
        ));
    }
    columns.extend([
        ("value", value_array(unsigned, values)),
        ("enabled", Arc::new(enabled_array(&keys, phase)) as ArrayRef),
    ]);
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    RecordBatch::try_new(
        schema,
        columns.into_iter().map(|(_, array)| array).collect(),
    )
    .unwrap()
}

fn rows(records: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut rows = records
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|column| {
                        format!("{:?}", ScalarValue::try_from_array(column, row).unwrap())
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

#[tokio::test]
async fn test_grouped_integer_sum_uses_column_merge_with_exact_prefix_results() {
    for unsigned in [false, true] {
        for native_key in [false, true] {
            let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            let group = if native_key { "key" } else { "key, stock" };
            let sql = format!(
                "SELECT {group}, SUM(value) FILTER (WHERE enabled) AS total, COUNT(value) AS valid FROM events GROUP BY {group}"
            );
            let query = crate::expression::parse_select_query(&sql).unwrap();
            let first = record(unsigned, native_key, 0);
            let mut plan =
                IncrementalSql::plan(&runtime, &query, "events", first.schema(), "integer")
                    .await
                    .unwrap()
                    .unwrap();
            let job =
                StreamJobContext::new(1, "integer", JsonMap::new(), None, CancellationToken::new());
            let context = StreamOperatorContext::new(&job, "integer", None);
            let mut prefix = Vec::new();
            for phase in 0..6 {
                let record = record(unsigned, native_key, phase);
                prefix.push(record.clone());
                let batch = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
                if phase == 5 {
                    let discarded = plan.update(&batch, &context, "integer").await.unwrap();
                    drop(discarded);
                }
                let transaction = plan.update(&batch, &context, "integer").await.unwrap();
                let expected = runtime
                    .sql(
                        &sql,
                        &BTreeMap::from([(
                            "events".into(),
                            Batch::table(prefix.clone(), crate::BatchMetadata::default()).unwrap(),
                        )]),
                        Some("oracle"),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    rows(&transaction.records),
                    rows(expected.table_payload().unwrap().batches())
                );
                plan.commit(transaction);
            }
            let accumulators = plan
                .candidate_accumulators(plan.groups.first(), "integer")
                .unwrap();
            for (expression, accumulator) in plan.aggregates.iter().zip(accumulators) {
                assert_eq!(
                    accumulator.is_none(),
                    expression.fun().name() == "sum",
                    "integer SUM must merge saved states in columns instead of constructing one scalar accumulator per group"
                );
            }
            drop(plan);
            assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
        }
    }
}

#[tokio::test]
async fn test_grouped_integer_sum_pending_history_does_not_retain_scalar_buffers() {
    for native_key in [false, true] {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let group = if native_key { "key" } else { "key, stock" };
        let sql = format!(
            "SELECT {group}, SUM(value) AS integer_sum, SUM(floating) AS floating_sum FROM events GROUP BY {group}"
        );
        let query = crate::expression::parse_select_query(&sql).unwrap();
        let first = with_floating_column(&record(false, native_key, 4));
        let mut plan = IncrementalSql::plan(&runtime, &query, "events", first.schema(), "history")
            .await
            .unwrap()
            .unwrap();
        let job =
            StreamJobContext::new(1, "history", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "history", None);
        let initial = Batch::table(vec![first.clone()], crate::BatchMetadata::default()).unwrap();
        let transaction = plan.update(&initial, &context, "history").await.unwrap();
        plan.commit(transaction);
        assert!(plan.sequential.is_some());

        let mut integer_history_bytes = 0;
        let mut floating_history_bytes = 0;
        for previous in &plan.groups {
            let candidate = plan
                .candidate(Some(previous), &previous.key, None, "history")
                .unwrap();
            for (expression, state) in plan.aggregates.iter().zip(&candidate.group.states) {
                let bytes = state.capacity() * size_of::<ScalarValue>();
                if grouped_sum::integer_sum(expression) {
                    integer_history_bytes += bytes;
                } else {
                    floating_history_bytes += bytes;
                }
            }
        }
        assert!(
            floating_history_bytes > 0,
            "sequential float state still needs owned candidate history"
        );
        assert_eq!(
            integer_history_bytes, 0,
            "pending integer SUM candidates must not allocate copied historical scalar buffers"
        );

        let next = with_floating_column(&record(false, native_key, 1));
        let next_batch = Batch::table(vec![next.clone()], crate::BatchMetadata::default()).unwrap();
        let discarded = plan.update(&next_batch, &context, "history").await.unwrap();
        drop(discarded);
        let transaction = plan.update(&next_batch, &context, "history").await.unwrap();
        let expected = runtime
            .sql(
                &sql,
                &BTreeMap::from([(
                    "events".into(),
                    Batch::table(vec![first, next], crate::BatchMetadata::default()).unwrap(),
                )]),
                Some("history-oracle"),
            )
            .await
            .unwrap();
        assert_eq!(
            rows(&transaction.records),
            rows(expected.table_payload().unwrap().batches())
        );
        plan.commit(transaction);
        drop(plan);
        assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    }
}

fn with_floating_column(record: &RecordBatch) -> RecordBatch {
    let schema = record.schema();
    let values = record.column(schema.index_of("value").unwrap());
    let floating = datafusion::arrow::compute::cast(values, &DataType::Float64).unwrap();
    let mut fields = schema.fields().to_vec();
    fields.push(Arc::new(Field::new("floating", DataType::Float64, true)));
    let mut columns = record.columns().to_vec();
    columns.push(floating);
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn fixture_cell(record: &RecordBatch, column: &str, row: usize) -> ScalarValue {
    ScalarValue::try_from_array(record.column_by_name(column).unwrap(), row).unwrap()
}

#[test]
fn test_grouped_integer_fixture_preserves_phase_boundaries_and_masks() {
    let signed = record(false, false, 0);
    let unsigned = record(true, true, 0);
    assert_eq!(signed.num_rows(), 4);
    assert_eq!(
        fixture_cell(&signed, "value", 0),
        ScalarValue::Int64(Some(i64::MAX))
    );
    assert_eq!(
        fixture_cell(&unsigned, "value", 0),
        ScalarValue::UInt64(Some(u64::MAX))
    );
    assert_eq!(fixture_cell(&signed, "key", 2), ScalarValue::Utf8(None));
    assert_eq!(
        fixture_cell(&unsigned, "key", 2),
        ScalarValue::Int64(Some(2))
    );
    assert_eq!(
        fixture_cell(&signed, "enabled", 3),
        ScalarValue::Boolean(Some(true))
    );
    let next = record(false, false, 1);
    assert_eq!(next.num_rows(), 5);
    assert_eq!(fixture_cell(&next, "value", 1), ScalarValue::Int64(Some(7)));
    assert_eq!(
        fixture_cell(&next, "enabled", 3),
        ScalarValue::Boolean(Some(false))
    );
    let dense = record(false, false, 2);
    assert_eq!(dense.num_rows(), CHUNK_ROWS + 1);
    assert_eq!(fixture_cell(&dense, "value", 0), ScalarValue::Int64(None));
    assert_eq!(
        fixture_cell(&dense, "value", CHUNK_ROWS),
        ScalarValue::Int64(Some(2))
    );
    assert_eq!(
        fixture_cell(&dense, "enabled", 0),
        ScalarValue::Boolean(Some(false))
    );
    assert_eq!(
        fixture_cell(&dense, "stock", CHUNK_ROWS),
        ScalarValue::Utf8(Some("stock".into()))
    );
    let empty = record(false, false, 3);
    assert_eq!(empty.num_rows(), 0);
    assert!(
        empty
            .schema()
            .fields()
            .iter()
            .all(|field| field.is_nullable())
    );
    let sparse = record(false, false, 4);
    assert_eq!(sparse.num_rows(), 4);
    assert_eq!(fixture_cell(&sparse, "value", 0), ScalarValue::Int64(None));
    assert_eq!(
        fixture_cell(&sparse, "value", 1),
        ScalarValue::Int64(Some(2))
    );
    let updated = record(false, false, 5);
    assert_eq!(updated.num_rows(), CHUNK_ROWS + 1);
    assert_eq!(
        fixture_cell(&updated, "value", 1),
        ScalarValue::Int64(Some(3))
    );
    assert_eq!(
        fixture_cell(&updated, "value", CHUNK_ROWS),
        ScalarValue::Int64(None)
    );
    assert_eq!(
        fixture_cell(&updated, "enabled", 0),
        ScalarValue::Boolean(Some(false))
    );
}

#[tokio::test]
async fn test_utf8_key_output_avoids_per_group_string_allocations() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let input = record(false, false, 2).slice(0, 256);
    let query = crate::expression::parse_select_query(
        "SELECT key, stock, SUM(value) AS total FROM events GROUP BY key, stock",
    )
    .unwrap();
    let mut plan = IncrementalSql::plan(&runtime, &query, "events", input.schema(), "utf8-output")
        .await
        .unwrap()
        .unwrap();
    let job = StreamJobContext::new(
        1,
        "utf8-output",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "utf8-output", None);
    let batch = Batch::table(vec![input], crate::BatchMetadata::default()).unwrap();
    let transaction = plan.update(&batch, &context, "utf8-output").await.unwrap();
    plan.commit(transaction);
    assert_eq!(plan.groups.len(), 256);
    let candidates = CandidateMap::with_hasher(RandomState::new());
    let reservation = plan.scratch.new_empty();
    let charge = plan
        .output_charge(256, &candidates, &context, "utf8-output")
        .await
        .unwrap();
    reservation.try_grow(charge).unwrap();
    let mut output = None;
    let allocations = allocation_counter::measure(|| {
        output = Some(
            plan.aggregate_output_chunk(0, 256, &candidates, "utf8-output")
                .unwrap(),
        );
    });
    assert_eq!(output.as_ref().unwrap().num_rows(), 256);
    eprintln!("UTF8 output allocation evidence: {allocations:?}");
    assert!(
        allocations.count_total < 256,
        "two UTF8 key columns must not allocate a String per output group: {allocations:?}"
    );
    drop((output, reservation, plan));
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}

#[test]
fn test_utf8_output_capacity_rejects_offset_overflow_without_allocating() {
    let maximum = usize::try_from(i32::MAX).unwrap();
    assert_eq!(utf8_output_capacity(0, "capacity").unwrap(), 0);
    assert_eq!(utf8_output_capacity(maximum, "capacity").unwrap(), maximum);
    let error = utf8_output_capacity(maximum + 1, "capacity").unwrap_err();
    assert!(
        matches!(error, CalcFlowError::DataFusion { node_id: Some(name), .. } if name == "capacity")
    );
}

fn utf8_snapshot_input(keys: &[Option<&str>], values: &[Option<i64>], sequence: u64) -> Batch {
    let key = Field::new("key", DataType::Utf8, true).with_metadata(
        std::collections::HashMap::from([("origin".into(), "utf8-key".into())]),
    );
    let schema = Arc::new(Schema::new_with_metadata(
        vec![key, Field::new("value", DataType::Int64, true)],
        std::collections::HashMap::from([("source".into(), "utf8-output-test".into())]),
    ));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(keys.to_vec())),
            Arc::new(Int64Array::from(values.to_vec())),
        ],
    )
    .unwrap();
    Batch::table(
        vec![record],
        crate::BatchMetadata::new(
            "utf8-snapshot",
            sequence,
            JsonMap::from([("arrival".into(), serde_json::json!(sequence))]),
        )
        .unwrap(),
    )
    .unwrap()
}

fn utf8_snapshot_arrivals() -> Vec<Batch> {
    vec![
        utf8_snapshot_input(
            &[Some("A\0B"), Some("é"), Some(""), None, Some("e\u{301}")],
            &[Some(1), None, Some(0), Some(2), Some(3)],
            0,
        ),
        utf8_snapshot_input(
            &[Some("A\0B"), Some("é"), Some("新😀"), None],
            &[Some(2), Some(4), Some(5), None],
            1,
        ),
        utf8_snapshot_input(&[], &[], 2),
    ]
}

fn ordered_output_rows(records: &[RecordBatch]) -> Vec<Vec<String>> {
    records
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|column| {
                        format!("{:?}", ScalarValue::try_from_array(column, row).unwrap())
                    })
                    .collect()
            })
        })
        .collect()
}

#[tokio::test]
async fn test_utf8_output_preserves_held_snapshots_schema_metadata_and_order() {
    for sql in [
        "SELECT key, SUM(value) AS total FROM events GROUP BY key",
        "SELECT key, SUM(value) AS total FROM events GROUP BY key HAVING SUM(value) > 1 ORDER BY key DESC NULLS FIRST LIMIT 4",
    ] {
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let inputs = utf8_snapshot_arrivals();
        let query = crate::expression::parse_select_query(sql).unwrap();
        let mut plan = IncrementalSql::plan(
            &runtime,
            &query,
            "events",
            inputs[0].table_payload().unwrap().schema().clone(),
            "utf8-snapshots",
        )
        .await
        .unwrap()
        .unwrap();
        let job = StreamJobContext::new(
            1,
            "utf8-snapshots",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "utf8-snapshots", None);
        let mut history = Vec::new();
        let mut held = Vec::new();
        for input in inputs {
            history.extend(input.table_payload().unwrap().batches().to_vec());
            let expected = runtime
                .sql(
                    sql,
                    &BTreeMap::from([(
                        "events".into(),
                        Batch::table(history.clone(), input.metadata().clone()).unwrap(),
                    )]),
                    Some("utf8-oracle"),
                )
                .await
                .unwrap();
            let transaction = plan
                .update(&input, &context, "utf8-snapshots")
                .await
                .unwrap();
            let actual = runtime
                .incremental_output(transaction.records.clone(), input.metadata().clone())
                .unwrap();
            assert_eq!(actual.metadata(), input.metadata());
            assert_eq!(
                actual.table_payload().unwrap().schema(),
                expected.table_payload().unwrap().schema()
            );
            assert_eq!(
                rows(actual.table_payload().unwrap().batches()),
                rows(expected.table_payload().unwrap().batches())
            );
            if sql.contains("ORDER BY") {
                assert_eq!(
                    ordered_output_rows(actual.table_payload().unwrap().batches()),
                    ordered_output_rows(expected.table_payload().unwrap().batches())
                );
            }
            let original = ordered_output_rows(actual.table_payload().unwrap().batches());
            held.push((actual, original));
            plan.commit(transaction);
        }
        drop(plan);
        for (snapshot, original) in held {
            assert_eq!(
                ordered_output_rows(snapshot.table_payload().unwrap().batches()),
                original
            );
        }
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    }
}

async fn assert_utf8_output_fits_charge(input: Batch) {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let query = crate::expression::parse_select_query(
        "SELECT key, SUM(value) AS total FROM events GROUP BY key",
    )
    .unwrap();
    let mut plan = IncrementalSql::plan(
        &runtime,
        &query,
        "events",
        input.table_payload().unwrap().schema().clone(),
        "utf8-charge",
    )
    .await
    .unwrap()
    .unwrap();
    let job = StreamJobContext::new(
        1,
        "utf8-charge",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "utf8-charge", None);
    let transaction = plan.update(&input, &context, "utf8-charge").await.unwrap();
    plan.commit(transaction);
    let candidates = CandidateMap::with_hasher(RandomState::new());
    let count = plan.groups.len();
    let charge = plan
        .output_charge(count, &candidates, &context, "utf8-charge")
        .await
        .unwrap();
    let (reservation, _) = plan.input_workspace(0, "utf8-charge").unwrap();
    reservation.try_grow(charge).unwrap();
    let prepaid = reservation.size();
    let mut output = None;
    let allocations = allocation_counter::measure(|| {
        output = Some(
            plan.aggregate_output_chunk(0, count, &candidates, "utf8-charge")
                .unwrap(),
        );
    });
    assert!(
        allocations.bytes_max <= u64::try_from(prepaid).unwrap(),
        "UTF8 buffers exceed output/scratch prepayment {prepaid}: {allocations:?}"
    );
    assert_eq!(output.as_ref().unwrap().num_rows(), count);
    drop((output, reservation, plan));
    assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
}

#[tokio::test]
async fn test_utf8_output_small_null_and_mixed_buffers_fit_prepaid_charge() {
    assert_utf8_output_fits_charge(utf8_snapshot_input(&[None], &[None], 0)).await;
    assert_utf8_output_fits_charge(utf8_snapshot_input(
        &[Some(""), None, Some("长é\0😀")],
        &[Some(1), None, Some(2)],
        0,
    ))
    .await;
}

fn mixed_utf8_input(batch: &Batch) -> Batch {
    let records = batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            let schema = record.schema();
            let mut fields = schema.fields().to_vec();
            fields.push(Arc::new(Field::new("large_key", DataType::LargeUtf8, true)));
            let mut columns = record.columns().to_vec();
            columns.push(
                datafusion::arrow::compute::cast(record.column(0), &DataType::LargeUtf8).unwrap(),
            );
            let schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
            RecordBatch::try_new(schema, columns).unwrap()
        })
        .collect();
    Batch::table(records, batch.metadata().clone()).unwrap()
}

#[tokio::test]
async fn test_mixed_utf8_large_keys_restore_and_continue_exact_snapshots() {
    let sql = "SELECT key, large_key, SUM(value) AS total FROM events GROUP BY key, large_key";
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let job = StreamJobContext::new(
        1,
        "mixed-utf8",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "mixed-utf8", None);
    let mut operator = SqlOperator::new("mixed-utf8", sql, vec!["events".into()], vec![]).unwrap();
    let mut history = Vec::new();
    let mut pools = Vec::new();
    for input in utf8_snapshot_arrivals().iter().map(mixed_utf8_input) {
        history.extend(input.table_payload().unwrap().batches().to_vec());
        let expected = runtime
            .sql(
                sql,
                &BTreeMap::from([(
                    "events".into(),
                    Batch::table(history.clone(), input.metadata().clone()).unwrap(),
                )]),
                Some("mixed-utf8-oracle"),
            )
            .await
            .unwrap();
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("events", input, &context, &mut collector)
            .await
            .unwrap();
        let outputs = collector.drain("output");
        let actual = outputs[0].as_data().unwrap();
        assert_eq!(actual.metadata(), expected.metadata());
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(
            rows(actual.table_payload().unwrap().batches()),
            rows(expected.table_payload().unwrap().batches())
        );
        pools.push(
            operator
                .stream_state
                .runtime()
                .unwrap()
                .incremental_memory_pool(),
        );
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        drop(operator);
        operator = SqlOperator::new("mixed-utf8", sql, vec!["events".into()], vec![]).unwrap();
        operator.restore(&snapshot).unwrap();
    }
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pools.iter().map(|pool| pool.reserved()).sum::<usize>(), 0);
}
