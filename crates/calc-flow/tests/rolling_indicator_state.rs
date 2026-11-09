use std::{collections::BTreeMap, io::Cursor, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, CancellationToken, EdgeCollector, Epoch, EventTime,
    JsonMap, OperatorMetadata, OperatorStateSnapshot, RollingOperator, RollingSpec, StateBudget,
    StateSegment, StreamJobContext, StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, BinaryArray, BooleanArray, Float64Array, StringArray,
        TimestampMicrosecondArray, UInt8Array, UInt64Array,
    },
    compute::filter_record_batch,
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    ipc::{reader::FileReader, writer::FileWriter},
    record_batch::RecordBatch,
};
use serde_json::json;

fn spec(layout: u32) -> RollingSpec {
    let outputs = match layout {
        1 => {
            json!([{"kind":"lag","primitive_version":1,"input":"price","output":"value","periods":1}])
        }
        4 => {
            json!([{"kind":"cumulative_sum","primitive_version":1,"input":"price","output":"value","min_periods":1}])
        }
        _ => json!([
            {"kind":"rsi","primitive_version":1,"input":"price","output":"value","period":1},
            {"kind":"lag","primitive_version":1,"input":"price","output":"lag","periods":1}
        ]),
    };
    serde_json::from_value(
        json!({"configuration_version":1,"state_layout_version":layout,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],"outputs":outputs,
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
        "value_policy":"stateful_numeric_v1"}),
    )
    .unwrap()
}

fn input(extra: &[(&str, bool)]) -> (SchemaRef, Batch) {
    let mut fields = vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("entity", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from(vec![1, 2, 3, 4]).with_timezone("UTC")),
        Arc::new(StringArray::from(vec!["a"; 4])),
        Arc::new(UInt64Array::from(vec![1, 2, 3, 4])),
        Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
    ];
    for (name, binary) in extra {
        let column: ArrayRef = if *binary {
            Arc::new(BinaryArray::from_iter_values(std::iter::repeat_n(
                &[7_u8][..],
                4,
            )))
        } else {
            Arc::new(Float64Array::from(vec![9.0; 4]))
        };
        fields.push(Field::new(*name, column.data_type().clone(), true));
        columns.push(column);
    }
    let schema = Arc::new(Schema::new(fields));
    let record = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    (
        schema,
        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
    )
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn capture(
    layout: u32,
    extra: &[(&str, bool)],
    finalized: i64,
) -> (SchemaRef, OperatorStateSnapshot) {
    let (schema, batch) = input(extra);
    let mut operator = RollingOperator::new("features", Arc::clone(&schema), spec(layout)).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", batch, &context, &mut output)
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(finalized), &context, &mut output)
        .await
        .unwrap();
    operator.prepare_checkpoint_async(&context).await.unwrap();
    (schema, operator.checkpoint(Epoch::new(1).unwrap()).unwrap())
}

async fn continue_snapshot(
    layout: u32,
    schema: SchemaRef,
    snapshot: &OperatorStateSnapshot,
) -> Vec<Option<f64>> {
    let mut operator = RollingOperator::new("features", schema, spec(layout)).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "features", None);
    operator.restore_async(snapshot, &context).await.unwrap();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator.on_end(&context, &mut output).await.unwrap();
    let messages = output.drain("output");
    let record = &messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    record
        .column_by_name("value")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .collect()
}

async fn assert_collision(layout: u32, extra: &[(&str, bool)], expected: f64) {
    let (schema, snapshot) = capture(layout, extra, 3).await;
    assert_eq!(
        continue_snapshot(layout, schema, &snapshot).await,
        vec![Some(expected)]
    );
}
#[tokio::test]
async fn test_layout_three_preserves_float_column_named_indicator_state() {
    assert_collision(1, &[("_indicator_state", false)], 3.0).await;
}
#[tokio::test]
async fn test_layout_four_preserves_binary_column_named_cumulative_summary() {
    assert_collision(4, &[("_cumulative_summary", true)], 10.0).await;
}
#[tokio::test]
async fn test_layout_five_preserves_both_user_tail_column_names() {
    assert_collision(
        5,
        &[("_indicator_state", false), ("_cumulative_summary", true)],
        100.0,
    )
    .await;
}

fn record(snapshot: &OperatorStateSnapshot) -> RecordBatch {
    let segment = snapshot.segments.values().next().unwrap();
    FileReader::try_new(Cursor::new(segment.bytes()), None)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
}

fn replace_record(snapshot: &OperatorStateSnapshot, record: &RecordBatch) -> OperatorStateSnapshot {
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &record.schema()).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    let segment = StateSegment::new(bytes);
    let id = snapshot.segments.keys().next().unwrap().clone();
    let mut metadata = snapshot.inline_metadata.clone();
    let handle = &mut metadata.get_mut("segment_inventory").unwrap()[0]["handle"];
    handle["byte_len"] = json!(segment.bytes().len());
    handle["sha256"] = json!(segment.sha256());
    OperatorStateSnapshot {
        inline_metadata: metadata,
        segments: BTreeMap::from([(id, segment)]),
    }
}

fn zero_count_without_indicators(
    snapshot: &OperatorStateSnapshot,
    keep_history: bool,
    keep_pending: bool,
) -> OperatorStateSnapshot {
    let record = record(snapshot);
    let kinds = record
        .column(0)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .unwrap();
    let positions = record
        .column(2)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let counters: UInt64Array = (0..record.num_rows())
        .map(|index| {
            if kinds.value(index) == 0 {
                Some(0)
            } else {
                positions.is_valid(index).then(|| positions.value(index))
            }
        })
        .collect();
    let keep: BooleanArray = kinds
        .values()
        .iter()
        .map(|kind| {
            Some(match *kind {
                5 => false,
                1 => keep_history,
                2 => keep_pending,
                _ => true,
            })
        })
        .collect();
    let mut columns = record.columns().to_vec();
    columns[2] = Arc::new(counters);
    let record = RecordBatch::try_new(record.schema(), columns).unwrap();
    replace_record(snapshot, &filter_record_batch(&record, &keep).unwrap())
}

#[tokio::test]
async fn test_layout_five_zero_count_cannot_hide_finalized_history_or_orphan_entity() {
    let (schema, snapshot) = capture(5, &[], 3).await;
    for (history, pending) in [(true, true), (true, false), (false, false)] {
        let corrupted = zero_count_without_indicators(&snapshot, history, pending);
        let mut operator = RollingOperator::new("features", Arc::clone(&schema), spec(5)).unwrap();
        let result = operator.restore(&corrupted);
        assert_state_mismatch(result.unwrap_err());
    }
    let (schema, buffer_only) = capture(5, &[], 0).await;
    assert_eq!(
        continue_snapshot(5, schema, &buffer_only).await,
        vec![None, Some(100.0), Some(100.0), Some(100.0)]
    );
}

#[tokio::test]
async fn test_layout_five_pending_infinity_is_rejected_by_restore() {
    let (schema, snapshot) = capture(5, &[], 0).await;
    let record = record(&snapshot);
    let kinds = record
        .column(0)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .unwrap();
    let prices = record
        .column(6)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let prices: Float64Array = (0..record.num_rows())
        .map(|index| {
            if kinds.value(index) == 2 {
                Some(f64::INFINITY)
            } else {
                prices.is_valid(index).then(|| prices.value(index))
            }
        })
        .collect();
    let mut columns = record.columns().to_vec();
    columns[6] = Arc::new(prices);
    let corrupted = replace_record(
        &snapshot,
        &RecordBatch::try_new(record.schema(), columns).unwrap(),
    );
    let mut operator = RollingOperator::new("features", schema, spec(5)).unwrap();
    let result = operator.restore(&corrupted);
    assert_state_mismatch(result.unwrap_err());
}

fn assert_state_mismatch(error: CalcFlowError) {
    let CalcFlowError::CheckpointMismatch { message } = error else {
        panic!("expected checkpoint mismatch, got {error}");
    };
    assert!(message.contains("features"), "{message}");
    assert!(message.contains("state.format"), "{message}");
}

#[tokio::test]
async fn test_old_only_explicit_layout_five_keeps_retained_only_budget() {
    let (schema, batch) = input(&[("_indicator_state", false)]);
    let mut declaration = spec(4);
    declaration.state_layout_version = 5;
    let mut operator =
        RollingOperator::new("features", Arc::clone(&schema), declaration.clone()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", batch, &context, &mut output)
        .await
        .unwrap();
    operator.on_end(&context, &mut output).await.unwrap();
    let budget = StateBudget::new(16, 2048).unwrap();
    operator.set_state_budget(budget).unwrap();
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert!(snapshot.segments.values().next().unwrap().bytes().len() > 2048);
    let mut restored = RollingOperator::new("features", schema, declaration).unwrap();
    restored.set_state_budget(budget).unwrap();
    restored.restore_async(&snapshot, &context).await.unwrap();
    assert!(restored.checkpoint(Epoch::new(2).unwrap()).is_ok());
}
