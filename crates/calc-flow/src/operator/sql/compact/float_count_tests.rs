use std::{
    collections::BTreeMap,
    sync::{Arc, Weak},
};

use async_trait::async_trait;
use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Int64Array, StringArray, new_empty_array},
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
};
use serde_json::json;

use super::super::*;
use crate::{CancellationToken, EdgeCollector, StreamJobContext};

type Row = (Option<i64>, Option<f64>);
const ROWS: [Row; 8] = [
    (Some(1), None),
    (Some(1), Some(f64::NAN)),
    (None, Some(f64::INFINITY)),
    (Some(2), Some(f64::NEG_INFINITY)),
    (Some(2), Some(0.0)),
    (None, Some(-0.0)),
    (Some(3), None),
    (Some(1), Some(7.5)),
];

fn query(grouped: bool) -> &'static str {
    if grouped {
        "SELECT key, COUNT(value) AS valid, COUNT(*) AS rows FROM events GROUP BY key"
    } else {
        "SELECT COUNT(value) AS valid, COUNT(*) AS rows FROM events"
    }
}

fn schema(dtype: &DataType) -> SchemaRef {
    let fields = (0..8)
        .map(|index| match index {
            3 => Field::new("key", DataType::Int64, true),
            6 => Field::new("value", dtype.clone(), true)
                .with_metadata([("unit".into(), "count-validity".into())].into()),
            _ => Field::new(format!("unused_{index}"), DataType::Utf8, false),
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new_with_metadata(
        fields,
        [("origin".into(), "current-float-count".into())].into(),
    ))
}

fn input(dtype: &DataType, rows: &[Row], sequence: u64) -> Batch {
    let values = if rows.is_empty() {
        new_empty_array(dtype)
    } else {
        ScalarValue::iter_to_array(
            rows.iter()
                .map(|row| ScalarValue::Float64(row.1).cast_to(dtype).unwrap()),
        )
        .unwrap()
    };
    let columns = (0..8)
        .map(|index| match index {
            3 => Arc::new(Int64Array::from(
                rows.iter().map(|row| row.0).collect::<Vec<_>>(),
            )) as ArrayRef,
            6 => values.clone(),
            _ => Arc::new(StringArray::from(vec!["unused payload"; rows.len()])) as ArrayRef,
        })
        .collect::<Vec<_>>();
    Batch::table(
        vec![RecordBatch::try_new(schema(dtype), columns).unwrap()],
        BatchMetadata::new(
            "current-float-count",
            sequence,
            JsonMap::from([
                ("case".into(), json!(format!("{dtype:?}"))),
                ("prefix".into(), json!(sequence)),
            ]),
        )
        .unwrap(),
    )
    .unwrap()
}

fn operator(dtype: &DataType, query: &str) -> SqlOperator {
    SqlOperator::new("float_count", query, vec!["events".into()], vec![])
        .unwrap()
        .with_ports(
            vec![
                Port::with_schema_ref("events", BatchKind::Table, true, Some(schema(dtype)))
                    .unwrap(),
            ],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        913,
        "float-count",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn rows(batch: &Batch) -> Vec<Vec<ScalarValue>> {
    let mut rows = batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, row).unwrap())
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.partial_cmp(right).unwrap());
    rows
}

async fn process(
    state: &mut SqlOperator,
    batch: Batch,
    context: &StreamOperatorContext<'_>,
) -> Batch {
    let mut collector = EdgeCollector::new(state.output_ports().to_vec());
    state
        .process_data("events", batch, context, &mut collector)
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    output[0].as_data().unwrap().clone()
}

async fn assert_oracle(
    actual: &Batch,
    query: &str,
    dtype: &DataType,
    prefix: &[Row],
    sequence: u64,
) {
    let input = input(dtype, prefix, sequence);
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([("events".into(), input.clone())]),
            Some("oracle"),
        )
        .await
        .unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(rows(actual), rows(&expected));
    assert_eq!(actual.metadata(), input.metadata());
    assert_eq!(actual.metadata(), expected.metadata());
}

fn same_snapshot(before: &OperatorStateSnapshot, after: &OperatorStateSnapshot) {
    assert_eq!(before.inline_metadata, after.inline_metadata);
    assert_eq!(
        before.segments.keys().collect::<Vec<_>>(),
        after.segments.keys().collect::<Vec<_>>()
    );
    for (id, segment) in &before.segments {
        assert_eq!(segment.bytes(), after.segments[id].bytes());
    }
}

fn native_capture(
    state: &mut SqlOperator,
    dtype: &DataType,
    grouped: bool,
) -> OperatorStateSnapshot {
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
    assert_eq!(snapshot.inline_metadata["state_accounting"], json!(3));
    assert_eq!(
        snapshot
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["batch-metadata", "control", "group-state", "logical-schema"]
    );
    let compact = state.compact.as_ref().unwrap();
    let projection = compact.projection().unwrap();
    assert_eq!(projection.columns.logical_schema(), &schema(dtype));
    assert_eq!(
        projection.columns.ordinals(),
        if grouped { &[3, 6][..] } else { &[6][..] }
    );
    assert_eq!(
        projection.columns.physical_schema().fields().len(),
        if grouped { 2 } else { 1 }
    );
    assert!(state.retained.is_none());
    let decoded = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
    assert_eq!(
        decoded.table_payload().unwrap().schema().fields().len(),
        if grouped { 3 } else { 2 }
    );
    assert!(
        decoded
            .table_payload()
            .unwrap()
            .schema()
            .fields()
            .iter()
            .all(|field| field.data_type() == &DataType::Int64)
    );
    snapshot
}

async fn prefixes(dtype: &DataType, grouped: bool) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_count", None);
    let mut state = operator(dtype, query(grouped));
    let mut weak: Vec<Weak<dyn Array>> = Vec::new();
    for (sequence, (start, end)) in [(0, 4), (4, 7), (7, 8)].into_iter().enumerate() {
        let batch = input(dtype, &ROWS[start..end], sequence as u64);
        weak.extend(
            batch.table_payload().unwrap().batches()[0]
                .columns()
                .iter()
                .map(Arc::downgrade),
        );
        let actual = process(&mut state, batch, &context).await;
        assert_oracle(
            &actual,
            query(grouped),
            dtype,
            &ROWS[..end],
            sequence as u64,
        )
        .await;
    }
    assert_eq!(state.incremental_work, (ROWS.len(), 1));
    state.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = native_capture(&mut state, dtype, grouped);
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    let source_pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let mut restored = operator(dtype, query(grouped));
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    same_snapshot(&snapshot, &restored.checkpoint(Epoch::INITIAL).unwrap());
    let empty = process(&mut restored, input(dtype, &[], 3), &context).await;
    assert_oracle(&empty, query(grouped), dtype, &ROWS, 3).await;
    let next = [(Some(4), Some(f64::NAN)), (None, None)];
    let actual = process(&mut restored, input(dtype, &next, 4), &context).await;
    let all = ROWS.into_iter().chain(next).collect::<Vec<_>>();
    assert_oracle(&actual, query(grouped), dtype, &all, 4).await;
    let after = native_capture(&mut restored, dtype, grouped);
    let target_pool = restored
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    assert!(source_pool.reserved() > 0 && target_pool.reserved() > 0);
    drop((state, restored, snapshot, after, empty, actual));
    assert_eq!(source_pool.reserved(), 0);
    assert_eq!(target_pool.reserved(), 0);
}

#[tokio::test]
async fn test_current_float_count_global_native3_roundtrip() {
    for dtype in [DataType::Float32, DataType::Float64] {
        prefixes(&dtype, false).await;
    }
}

#[tokio::test]
async fn test_current_float_count_grouped_native3_roundtrip() {
    for dtype in [DataType::Float32, DataType::Float64] {
        prefixes(&dtype, true).await;
    }
}

struct Reject;

#[async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _: &str, _: Batch) -> Result<()> {
        Err(CalcFlowError::Operator {
            node_id: "reject-count".into(),
            message: "injected emit failure".into(),
        })
    }
}

#[tokio::test]
async fn test_current_float_count_rejected_emit_retains_capture_and_retries_once() {
    for dtype in [DataType::Float32, DataType::Float64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_count", None);
        let mut state = operator(&dtype, query(true));
        drop(process(&mut state, input(&dtype, &ROWS[..4], 0), &context).await);
        let before = state.checkpoint(Epoch::INITIAL).unwrap();
        let rejected = input(&dtype, &ROWS[4..], 1);
        let weak = rejected.table_payload().unwrap().batches()[0]
            .columns()
            .iter()
            .map(Arc::downgrade)
            .collect::<Vec<_>>();
        let failure = state
            .process_data("events", rejected, &context, &mut Reject)
            .await;
        assert!(
            matches!(failure, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-count")
        );
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
        let actual = process(&mut state, input(&dtype, &ROWS[4..], 1), &context).await;
        assert_oracle(&actual, query(true), &dtype, &ROWS, 1).await;
        let after = native_capture(&mut state, &dtype, true);
        assert_eq!(after.inline_metadata["rows"], json!(ROWS.len()));
    }
}

#[tokio::test]
async fn test_current_float_count_with_sum_keeps_raw4_fallback() {
    let query = "SELECT COUNT(value) AS valid, SUM(value) AS total FROM events";
    let finite = [(Some(1), Some(0.25)), (None, Some(0.5)), (Some(2), None)];
    for dtype in [DataType::Float32, DataType::Float64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_count", None);
        let mut state = operator(&dtype, query);
        let actual = process(&mut state, input(&dtype, &finite, 0), &context).await;
        assert_oracle(&actual, query, &dtype, &finite, 0).await;
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(snapshot.inline_metadata["state_layout"], json!(4));
        assert_eq!(snapshot.inline_metadata["state_accounting"], json!(4));
        assert!(state.incremental.is_none() && state.compact.is_none());
        assert!(snapshot.segments.contains_key("input-retained"));
    }
}
