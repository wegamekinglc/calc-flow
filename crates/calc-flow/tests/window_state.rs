use std::{io::Cursor, sync::Arc, time::Duration};

use calc_flow::{
    AggregateFunction, Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, EventTime,
    JsonMap, LocalStateBackend, OperatorMetadata, OperatorStateSnapshot, StateBackend, StateHandle,
    StateLineageKey, StreamJobContext, StreamOperator, StreamOperatorContext,
    WindowAggregateOperator, WindowSpec,
};
use datafusion::arrow::{
    array::{
        ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    ipc::reader::FileReader,
    record_batch::RecordBatch,
};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const PIPELINE_FINGERPRINT: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn input_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("group", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]))
}

fn input_batch(event_times: Vec<i64>, groups: Vec<&str>, values: Vec<i64>) -> Batch {
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(event_times)) as ArrayRef,
            Arc::new(StringArray::from(groups)),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn nullable_input_batch(
    event_times: Vec<Option<i64>>,
    groups: Vec<&str>,
    values: Vec<i64>,
) -> Batch {
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(event_times)) as ArrayRef,
            Arc::new(StringArray::from(groups)),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        PIPELINE_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn digest(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes.as_ref()))
}

fn local_handle(
    key: &StateLineageKey,
    epoch: Epoch,
    segment_id: &str,
    bytes: &[u8],
) -> StateHandle {
    let lineage_hash = digest(format!(
        "{}\0{}",
        key.pipeline_name(),
        key.pipeline_fingerprint()
    ));
    StateHandle::new(
        "window",
        epoch,
        segment_id,
        &format!(
            "committed/{}/{}/{}-{}.arrow",
            lineage_hash,
            digest("window"),
            epoch.as_u64(),
            digest(segment_id)
        ),
        u64::try_from(bytes.len()).unwrap(),
        &digest(bytes),
    )
    .unwrap()
}

fn operator() -> WindowAggregateOperator {
    let spec = WindowSpec::tumbling("event_time", Duration::from_micros(10))
        .unwrap()
        .group_by(["group"])
        .unwrap()
        .aggregate(AggregateFunction::Sum, "value", "sum_value")
        .unwrap();
    WindowAggregateOperator::new("window", input_schema(), spec).unwrap()
}

fn aggregate_state_operator() -> WindowAggregateOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("group", DataType::Utf8, false),
        Field::new("signed", DataType::Int64, false),
        Field::new("unsigned", DataType::UInt64, false),
        Field::new("float", DataType::Float64, false),
    ]));
    let spec = WindowSpec::tumbling("event_time", Duration::from_micros(10))
        .unwrap()
        .group_by(["group"])
        .unwrap()
        .aggregate(AggregateFunction::Count, "signed", "count_value")
        .unwrap()
        .aggregate(AggregateFunction::Sum, "signed", "signed_sum")
        .unwrap()
        .aggregate(AggregateFunction::Min, "signed", "signed_min")
        .unwrap()
        .aggregate(AggregateFunction::Max, "signed", "signed_max")
        .unwrap()
        .aggregate(AggregateFunction::Avg, "signed", "signed_avg")
        .unwrap()
        .aggregate(AggregateFunction::Avg, "unsigned", "unsigned_avg")
        .unwrap()
        .aggregate(AggregateFunction::Sum, "float", "float_sum")
        .unwrap()
        .aggregate(AggregateFunction::Avg, "float", "float_avg")
        .unwrap();
    WindowAggregateOperator::new("window", schema, spec).unwrap()
}

fn aggregate_state_batch() -> Batch {
    let schema = aggregate_state_operator().input_ports()[0]
        .schema()
        .unwrap()
        .clone();
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![1, 2])) as ArrayRef,
            Arc::new(StringArray::from(vec!["g", "g"])),
            Arc::new(Int64Array::from(vec![2, 4])),
            Arc::new(UInt64Array::from(vec![4, 8])),
            Arc::new(Float64Array::from(vec![1.5, 2.5])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn first_output_sum(collector: &mut EdgeCollector) -> (u64, i64) {
    let output = collector.drain("output");
    let batch = output[0].as_data().unwrap();
    let sum = batch.table_payload().unwrap().batches()[0]
        .column_by_name("sum_value")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    (batch.metadata().sequence(), sum)
}

#[tokio::test]
async fn checkpoint_is_incremental_arrow_ipc_and_restore_replaces_live_state() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut ignored = EdgeCollector::new(operator().output_ports().to_vec());
    let mut original = operator();
    original
        .process_data(
            "input",
            input_batch(vec![1, 2], vec!["a", "a"], vec![2, 3]),
            &context,
            &mut ignored,
        )
        .await
        .unwrap();

    let first = original.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(first.segments.len(), 1);
    let first_bytes = first.segments.values().next().unwrap();
    assert!(first_bytes.starts_with(b"ARROW1"));
    let reader = FileReader::try_new(Cursor::new(first_bytes), None).unwrap();
    assert_eq!(reader.num_batches(), 1);
    assert_eq!(reader.into_iter().next().unwrap().unwrap().num_rows(), 1);

    original
        .process_data(
            "input",
            input_batch(vec![11], vec!["b"], vec![99]),
            &context,
            &mut ignored,
        )
        .await
        .unwrap();

    let mut restored = operator();
    restored.restore(&first).unwrap();
    restored
        .process_data(
            "input",
            input_batch(vec![3], vec!["a"], vec![7]),
            &context,
            &mut ignored,
        )
        .await
        .unwrap();
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    restored.on_end(&context, &mut output).await.unwrap();
    assert_eq!(first_output_sum(&mut output), (0, 12));
}

#[tokio::test]
async fn checkpointed_close_transition_and_output_sequence_survive_restore() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut original = operator();
    let mut output = EdgeCollector::new(original.output_ports().to_vec());
    original
        .process_data(
            "input",
            input_batch(vec![1], vec!["a"], vec![5]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    original
        .on_watermark(EventTime::from_micros(10), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(first_output_sum(&mut output), (0, 5));
    let snapshot = original.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut restored = operator();
    restored.restore(&snapshot).unwrap();
    restored
        .process_data(
            "input",
            input_batch(vec![11], vec!["b"], vec![7]),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    restored.on_end(&context, &mut output).await.unwrap();
    assert_eq!(first_output_sum(&mut output), (1, 7));
}

#[tokio::test]
async fn invalid_restore_has_no_side_effect_and_repeated_restore_is_idempotent() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut source = operator();
    let mut ignored = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data(
            "input",
            input_batch(vec![1], vec!["a"], vec![5]),
            &context,
            &mut ignored,
        )
        .await
        .unwrap();
    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();

    let mut target = operator();
    target.restore(&snapshot).unwrap();
    target.restore(&snapshot).unwrap();
    let mut corrupted = snapshot.clone();
    let segment = corrupted.segments.values_mut().next().unwrap();
    segment.truncate(segment.len() / 2);
    assert!(target.restore(&corrupted).is_err());

    let mut output = EdgeCollector::new(target.output_ports().to_vec());
    target.on_end(&context, &mut output).await.unwrap();
    assert_eq!(first_output_sum(&mut output), (0, 5));
}

#[tokio::test]
async fn later_delta_restores_with_the_complete_retained_segment_inventory() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut source = operator();
    let mut ignored = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data(
            "input",
            input_batch(vec![1], vec!["a"], vec![5]),
            &context,
            &mut ignored,
        )
        .await
        .unwrap();
    let first = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
    source
        .process_data(
            "input",
            input_batch(vec![2], vec!["a"], vec![7]),
            &context,
            &mut ignored,
        )
        .await
        .unwrap();
    let second = source.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert_eq!(first.segments.len(), 1);
    assert_eq!(second.segments.len(), 1);

    let mut complete = second;
    complete.segments.extend(first.segments);
    let mut restored = operator();
    restored.restore(&complete).unwrap();
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    restored.on_end(&context, &mut output).await.unwrap();
    assert_eq!(first_output_sum(&mut output), (0, 12));
}

#[tokio::test]
async fn snapshot_persists_separate_assignment_late_and_null_time_metrics() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", Some(EventTime::from_micros(15)));
    let mut source = operator();
    let mut ignored = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data(
            "input",
            nullable_input_batch(
                vec![Some(1), None, Some(16)],
                vec!["late", "null", "open"],
                vec![1, 2, 3],
            ),
            &context,
            &mut ignored,
        )
        .await
        .unwrap();

    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let metrics = snapshot.inline_metadata["metrics"].as_object().unwrap();
    assert_eq!(metrics["late_rows"], 1);
    assert_eq!(metrics["affected_batches"], 1);
    assert_eq!(metrics["max_lateness_micros"], 5);
    assert_eq!(metrics["null_event_time_rows"], 1);
    assert_eq!(metrics["null_event_time_batches"], 1);
}

#[tokio::test]
async fn delta_threshold_prepares_one_replacement_base_before_inventory_growth() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut source = operator();
    let mut ignored = EdgeCollector::new(source.output_ports().to_vec());
    let mut latest = None;
    for epoch in 1..=34 {
        source
            .process_data(
                "input",
                input_batch(vec![1], vec!["a"], vec![1]),
                &context,
                &mut ignored,
            )
            .await
            .unwrap();
        latest = Some(source.checkpoint(Epoch::new(epoch).unwrap()).unwrap());
    }
    let compacted = latest.unwrap();
    assert_eq!(compacted.segments.len(), 2);
    assert!(
        compacted
            .segments
            .keys()
            .next()
            .unwrap()
            .starts_with("base-")
    );

    let mut restored = operator();
    restored.restore(&compacted).unwrap();
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    restored.on_end(&context, &mut output).await.unwrap();
    assert_eq!(first_output_sum(&mut output), (0, 34));
}

#[tokio::test]
async fn every_accumulator_state_shape_round_trips_through_arrow_ipc() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut source = aggregate_state_operator();
    let mut ignored = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data("input", aggregate_state_batch(), &context, &mut ignored)
        .await
        .unwrap();
    let snapshot = source.checkpoint(Epoch::INITIAL).unwrap();

    let mut restored = aggregate_state_operator();
    restored.restore(&snapshot).unwrap();
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    restored.on_end(&context, &mut output).await.unwrap();
    let output = output.drain("output");
    let record = &output[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    let int_value = |name: &str| {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    };
    let float_value = |name: &str| {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0)
    };
    assert_eq!(
        record
            .column_by_name("count_value")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0),
        2
    );
    assert_eq!(int_value("signed_sum"), 6);
    assert_eq!(int_value("signed_min"), 2);
    assert_eq!(int_value("signed_max"), 4);
    assert_eq!(float_value("signed_avg").to_bits(), 3.0_f64.to_bits());
    assert_eq!(float_value("unsigned_avg").to_bits(), 6.0_f64.to_bits());
    assert_eq!(float_value("float_sum").to_bits(), 4.0_f64.to_bits());
    assert_eq!(float_value("float_avg").to_bits(), 2.0_f64.to_bits());
}

#[tokio::test]
async fn snapshot_segments_round_trip_through_the_validating_local_backend() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut source = operator();
    let mut ignored = EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data(
            "input",
            input_batch(vec![1], vec!["a"], vec![5]),
            &context,
            &mut ignored,
        )
        .await
        .unwrap();
    let snapshot = source.checkpoint(Epoch::INITIAL).unwrap();

    let directory = TempDir::new().unwrap();
    let key = StateLineageKey::new("window-test", PIPELINE_FINGERPRINT).unwrap();
    let backend = LocalStateBackend::new(directory.path()).await.unwrap();
    let lineage = backend.open_lineage(&key).await.unwrap();
    let mut loaded = std::collections::BTreeMap::new();
    for (segment_id, bytes) in snapshot.segments {
        let handle = local_handle(&key, Epoch::INITIAL, &segment_id, &bytes);
        lineage.stage_segment(&handle, &bytes).await.unwrap();
        lineage.validate_segment(&handle).await.unwrap();
        lineage.publish_segment(&handle).await.unwrap();
        loaded.insert(segment_id, lineage.load_segment(&handle).await.unwrap());
    }
    let persisted = OperatorStateSnapshot {
        inline_metadata: snapshot.inline_metadata,
        segments: loaded,
    };

    let mut restored = operator();
    restored.restore(&persisted).unwrap();
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    restored.on_end(&context, &mut output).await.unwrap();
    assert_eq!(first_output_sum(&mut output), (0, 5));
}
