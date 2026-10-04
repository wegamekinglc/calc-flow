use std::{collections::BTreeMap, sync::Arc};

use datafusion::{
    arrow::{
        array::{ArrayRef, Int64Array, StringArray},
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
};
use serde_json::json;

use super::*;
use crate::{BatchMetadata, CancellationToken, EdgeCollector, Epoch, JsonMap, StreamJobContext};

const NATIVE: &str = "SELECT key, SUM(value) AS total, COUNT(*) AS rows FROM events GROUP BY key";
const RETAINED: &str =
    "SELECT key, SUM(value) AS total, COUNT(*) AS rows FROM events WHERE value > 0 GROUP BY key";
const LAYOUT_REFUSAL: &str = "SQL checkpoint layout is unsupported (expected 3 or 4)";

fn schema() -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("key", DataType::Utf8, true),
            Field::new("value", DataType::Int64, true),
            Field::new("unused", DataType::Utf8, false),
        ],
        [("origin".into(), "current-checkpoint".into())].into(),
    ))
}

fn input() -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(StringArray::from(vec![
                        Some("a"),
                        Some("b"),
                        None,
                        Some("a"),
                    ])) as ArrayRef,
                    Arc::new(Int64Array::from(vec![Some(2), Some(-1), None, Some(5)])),
                    Arc::new(StringArray::from(vec!["payload"; 4])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::new(
            "current-checkpoint-input",
            9,
            JsonMap::from([("fixture".into(), json!({"current": true}))]),
        )
        .unwrap(),
    )
    .unwrap()
}

fn operator(query: &str) -> SqlOperator {
    SqlOperator::new("current_sql", query, vec!["events".into()], vec![])
        .unwrap()
        .with_ports(
            vec![Port::with_schema_ref("events", BatchKind::Table, true, Some(schema())).unwrap()],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        911,
        "current-checkpoint",
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
                    .map(|column| ScalarValue::try_from_array(column, row).unwrap())
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.partial_cmp(right).unwrap());
    rows
}

async fn capture(query: &str) -> (SqlOperator, OperatorStateSnapshot) {
    let mut operator = operator(query);
    let batch = input();
    let oracle = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([("events".into(), batch.clone())]),
            None,
        )
        .await
        .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "current_sql", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", batch, &context, &mut output)
        .await
        .unwrap();
    let emitted = output.drain("output");
    assert_eq!(emitted.len(), 1);
    let actual = emitted[0].as_data().unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        oracle.table_payload().unwrap().schema()
    );
    assert_eq!(rows(actual), rows(&oracle));
    assert_eq!(actual.metadata(), oracle.metadata());
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    (operator, snapshot)
}

#[tokio::test]
async fn test_current_native_sql_checkpoint_writes_layout3() {
    let (operator, snapshot) = capture(NATIVE).await;
    assert!(operator.incremental.is_some());
    let inline = snapshot.inline_metadata.clone();
    let segments = snapshot.segments.keys().cloned().collect::<Vec<_>>();
    drop((snapshot, operator));
    assert_eq!(inline["state_layout"], json!(3));
    assert_eq!(inline["state_accounting"], json!(3));
    assert_eq!(inline["rows"], json!(4));
    assert_eq!(
        segments,
        ["batch-metadata", "control", "group-state", "logical-schema"]
    );
}

#[tokio::test]
async fn test_current_ineligible_sql_checkpoint_writes_retained_layout4() {
    let (operator, snapshot) = capture(RETAINED).await;
    assert!(operator.incremental.is_none());
    let inline = snapshot.inline_metadata.clone();
    let segments = snapshot.segments.keys().cloned().collect::<Vec<_>>();
    drop((snapshot, operator));
    assert_eq!(inline["state_layout"], json!(4));
    assert_eq!(inline["state_accounting"], json!(4));
    assert_eq!(inline["rows"], json!(4));
    assert_eq!(
        segments,
        [
            "batch-metadata",
            "control",
            "input-retained",
            "logical-schema"
        ]
    );
}

#[test]
fn test_old_sql_layouts_are_rejected_before_input_decode() {
    for layout in [None, Some(0), Some(1), Some(2), Some(5)] {
        let mut inline_metadata = BTreeMap::new();
        if let Some(layout) = layout {
            inline_metadata.insert("state_layout".into(), json!(layout));
        }
        let snapshot = OperatorStateSnapshot {
            inline_metadata,
            segments: BTreeMap::from([(
                "input".into(),
                StateSegment::new(b"not Arrow IPC".to_vec()),
            )]),
        };
        let mut operator = operator(NATIVE);
        let result = operator.prepare_restore(&snapshot, &|| Ok(()));
        assert!(
            matches!(result, Err(CalcFlowError::Format { ref message }) if message == LAYOUT_REFUSAL),
            "layout={layout:?}"
        );
    }
}

fn current_input(full_width: bool, sequence: u64) -> Batch {
    let original = input();
    let records = original
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            if full_width {
                record.project(&[0, 1]).unwrap()
            } else {
                record.clone()
            }
        })
        .collect();
    Batch::table(
        records,
        BatchMetadata::new(
            "current-roundtrip",
            sequence,
            JsonMap::from([("phase".into(), json!(sequence))]),
        )
        .unwrap(),
    )
    .unwrap()
}

fn current_operator(query: &str, full_width: bool, declared: bool) -> SqlOperator {
    let schema = current_input(full_width, 0)
        .table_payload()
        .unwrap()
        .schema()
        .clone();
    operator(query)
        .with_ports(
            vec![
                Port::with_schema_ref("events", BatchKind::Table, true, declared.then_some(schema))
                    .unwrap(),
            ],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap()
}

async fn assert_current_prefix(
    operator: &mut SqlOperator,
    batch: Batch,
    prefix: &[Batch],
    context: &StreamOperatorContext<'_>,
) {
    let records = prefix
        .iter()
        .flat_map(|batch| batch.table_payload().unwrap().batches().iter().cloned())
        .collect();
    let expected = Batch::table(records, batch.metadata().clone()).unwrap();
    let oracle = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            &operator.query,
            &BTreeMap::from([("events".into(), expected)]),
            None,
        )
        .await
        .unwrap();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", batch, context, &mut collector)
        .await
        .unwrap();
    let emitted = collector.drain("output");
    assert_eq!(emitted.len(), 1);
    let actual = emitted[0].as_data().unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        oracle.table_payload().unwrap().schema()
    );
    assert_eq!(rows(actual), rows(&oracle));
    assert_eq!(actual.metadata(), oracle.metadata());
}

async fn current_roundtrip(query: &str, full_width: bool, declared: bool) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "current_sql", None);
    let mut state = current_operator(query, full_width, declared);
    let mut prefix = Vec::new();
    for sequence in 1..=3 {
        let batch = current_input(full_width, sequence);
        prefix.push(batch.clone());
        assert_current_prefix(&mut state, batch, &prefix, &context).await;
        state.prepare_checkpoint_async(&context).await.unwrap();
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        let layout = if query == NATIVE { 3 } else { 4 };
        assert_eq!(snapshot.inline_metadata["state_layout"], json!(layout));
        assert_eq!(snapshot.inline_metadata["state_accounting"], json!(layout));
        assert_eq!(snapshot.inline_metadata["rows"], json!(4 * sequence));
        let repeated = state.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(snapshot.inline_metadata, repeated.inline_metadata);
        assert_eq!(
            snapshot.segments.keys().collect::<Vec<_>>(),
            repeated.segments.keys().collect::<Vec<_>>()
        );
        for (name, segment) in &snapshot.segments {
            assert!(Arc::ptr_eq(
                &segment.bytes_arc(),
                &repeated.segments[name].bytes_arc()
            ));
        }
        let mut restored = current_operator(query, full_width, declared);
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        let recaptured = restored.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(recaptured.inline_metadata, snapshot.inline_metadata);
        for (name, segment) in &snapshot.segments {
            assert_eq!(segment.bytes(), recaptured.segments[name].bytes());
        }
        let cloned = restored.clone();
        assert!(cloned.retained.is_none() && cloned.compact.is_none());
        state = restored;
    }
    let logical = current_input(full_width, 4)
        .table_payload()
        .unwrap()
        .schema()
        .clone();
    let empty = Batch::table(
        vec![RecordBatch::new_empty(logical)],
        BatchMetadata::new(
            "current-roundtrip",
            4,
            JsonMap::from([("empty".into(), json!(true))]),
        )
        .unwrap(),
    )
    .unwrap();
    assert_current_prefix(&mut state, empty, &prefix, &context).await;
    let captured = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(captured.inline_metadata["rows"], json!(12));
    StreamOperator::reset(&mut state).unwrap();
    assert!(
        state
            .checkpoint(Epoch::INITIAL)
            .unwrap()
            .segments
            .is_empty()
    );
}

#[tokio::test]
async fn test_current_sql_roundtrips_continue_and_recapture_native_and_raw() {
    for query in [NATIVE, RETAINED] {
        for full_width in [false, true] {
            for declared in [false, true] {
                current_roundtrip(query, full_width, declared).await;
            }
        }
    }
}
