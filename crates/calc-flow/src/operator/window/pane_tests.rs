use super::*;
use crate::{BatchMetadata, StateBudget};

fn hopping_spec(size_micros: u64, slide_micros: u64) -> WindowSpec {
    WindowSpec::hopping(
        "time",
        Duration::from_micros(size_micros),
        Duration::from_micros(slide_micros),
    )
    .unwrap()
    .group_by(["group"])
    .unwrap()
    .aggregate(AggregateFunction::Count, "value", "count")
    .unwrap()
    .aggregate(AggregateFunction::Min, "value", "minimum")
    .unwrap()
    .aggregate(AggregateFunction::Max, "value", "maximum")
    .unwrap()
}

fn input(
    times: &[Option<i64>],
    groups: &[Option<&str>],
    values: &[Option<f64>],
) -> (SchemaRef, Batch) {
    assert_eq!(times.len(), groups.len());
    assert_eq!(times.len(), values.len());
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("group", DataType::Utf8, true),
        Field::new("value", DataType::Float64, true),
    ]));
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(TimestampMicrosecondArray::from(times.to_vec())),
            Arc::new(StringArray::from(groups.to_vec())),
            Arc::new(Float64Array::from(values.to_vec())),
        ],
    )
    .unwrap();
    let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
    (schema, batch)
}

fn job() -> crate::StreamJobContext {
    crate::StreamJobContext::new(
        1,
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    )
}

fn output_batches(output: &mut crate::EdgeCollector) -> Vec<Batch> {
    output
        .drain("output")
        .into_iter()
        .map(|message| message.as_data().unwrap().clone())
        .collect()
}

fn only_record(batch: &Batch) -> RecordBatch {
    let batches = batch.table_payload().unwrap().batches();
    assert_eq!(batches.len(), 1);
    batches[0].clone()
}

fn float_column(record: &RecordBatch, name: &str) -> Vec<Option<u64>> {
    let values = record
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    (0..values.len())
        .map(|row| values.is_valid(row).then(|| values.value(row).to_bits()))
        .collect()
}

#[test]
fn pane_geometry_floors_negative_times_and_merges_aligned_ranges() {
    let geometry = panes::PaneGeometry::new(CompiledWindowGeometry {
        size_micros: 20,
        slide_micros: 10,
        overlap: 2,
    });
    for (micros, index, start, end) in [
        (-21, -3, -30, -20),
        (-20, -2, -20, -10),
        (-11, -2, -20, -10),
        (-10, -1, -10, 0),
        (-1, -1, -10, 0),
        (0, 0, 0, 10),
        (9, 0, 0, 10),
        (10, 1, 10, 20),
    ] {
        let index = i128::from(index);
        assert_eq!(geometry.index(EventTime::from_micros(micros)), index);
        assert_eq!(
            geometry.bounds(index).unwrap(),
            (EventTime::from_micros(start), EventTime::from_micros(end))
        );
    }
    assert_eq!(
        geometry.window_indices(EventTime::from_micros(-10)),
        (-1..1)
    );
    let dense = panes::PaneGeometry::new(CompiledWindowGeometry {
        size_micros: 1_024,
        slide_micros: 1,
        overlap: 1_024,
    });
    let indices = dense.window_indices(EventTime::from_micros(-7));
    assert_eq!(indices.end - indices.start, 1_024);
    assert_eq!((indices.start, indices.end), (-7, 1_017));
}

#[test]
fn pane_count_merge_checks_u64_overflow() {
    let mut target = AccumulatorValue::Count(u64::MAX);
    let error = panes::merge_accumulator(&mut target, &AccumulatorValue::Count(1)).unwrap_err();
    assert!(error.contains("count overflowed UInt64"));
    assert!(matches!(target, AccumulatorValue::Count(u64::MAX)));
}

#[tokio::test]
async fn hopping_panes_preaggregate_and_merge_exact_extremes_in_window_order() {
    let (schema, batch) = input(
        &[Some(-5), Some(0), Some(5), Some(10), Some(15)],
        &[Some("a"), Some("a"), Some("a"), Some("a"), Some("a")],
        &[
            Some(-0.0),
            Some(0.0),
            Some(f64::from_bits(0x7ff8_0000_0000_abcd)),
            Some(f64::INFINITY),
            None,
        ],
    );
    let mut operator =
        WindowAggregateOperator::new("window", schema, hopping_spec(20, 10)).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", batch, &context, &mut output)
        .await
        .unwrap();

    let panes = operator.state.panes.as_ref().expect("pane state");
    assert_eq!(panes.entries.len(), 3);
    assert_eq!(panes.open_windows.len(), 4);
    assert!(operator.state.accumulators.is_empty());

    operator.on_end(&context, &mut output).await.unwrap();
    let batches = output_batches(&mut output);
    assert_eq!(batches.len(), 1);
    let record = only_record(&batches[0]);
    let starts = record
        .column_by_name("window_start")
        .unwrap()
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap()
        .values()
        .as_ref();
    assert_eq!(starts, [-20, -10, 0, 10]);
    let counts = record
        .column_by_name("count")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .values()
        .as_ref();
    assert_eq!(counts, [1, 3, 3, 1]);
    assert_eq!(
        float_column(&record, "minimum"),
        vec![
            Some((-0.0_f64).to_bits()),
            Some((-0.0_f64).to_bits()),
            Some(0.0_f64.to_bits()),
            Some(f64::INFINITY.to_bits()),
        ]
    );
    assert_eq!(
        float_column(&record, "maximum"),
        vec![
            Some((-0.0_f64).to_bits()),
            Some(0x7ff8_0000_0000_abcd),
            Some(0x7ff8_0000_0000_abcd),
            Some(f64::INFINITY.to_bits()),
        ]
    );
}

#[tokio::test]
async fn valid_group_with_all_null_operands_materializes_empty_aggregates() {
    let (schema, batch) = input(&[Some(0), Some(1)], &[None, None], &[None, None]);
    let mut operator =
        WindowAggregateOperator::new("window", schema, hopping_spec(20, 10)).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", batch, &context, &mut output)
        .await
        .unwrap();
    operator.on_end(&context, &mut output).await.unwrap();
    let record = only_record(&output_batches(&mut output)[0]);
    assert_eq!(record.num_rows(), 2);
    assert!(record.column_by_name("group").unwrap().is_null(0));
    assert_eq!(
        record
            .column_by_name("count")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .as_ref(),
        [0, 0]
    );
    assert_eq!(float_column(&record, "minimum"), vec![None, None]);
    assert_eq!(float_column(&record, "maximum"), vec![None, None]);
}

#[tokio::test]
async fn mixed_late_assignment_updates_open_pane_and_retires_at_equality() {
    let (schema, batch) = input(&[Some(15)], &[Some("a")], &[Some(4.0)]);
    let mut operator =
        WindowAggregateOperator::new("window", schema, hopping_spec(20, 10)).unwrap();
    let job = job();
    let input_context =
        StreamOperatorContext::new(&job, "window", Some(EventTime::from_micros(20)));
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", batch, &input_context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.state.metrics.late_rows, 1);
    assert_eq!(operator.state.metrics.max_lateness_micros, Some(0));
    let panes = operator.state.panes.as_ref().unwrap();
    assert_eq!(panes.entries.len(), 1);
    assert_eq!(panes.open_windows.len(), 1);

    let watermark_context = StreamOperatorContext::new(&job, "window", None);
    operator
        .on_watermark(EventTime::from_micros(20), &watermark_context, &mut output)
        .await
        .unwrap();
    assert!(output_batches(&mut output).is_empty());
    assert_eq!(operator.state.panes.as_ref().unwrap().entries.len(), 1);

    operator
        .on_watermark(EventTime::from_micros(30), &watermark_context, &mut output)
        .await
        .unwrap();
    let emitted = output_batches(&mut output);
    assert_eq!(only_record(&emitted[0]).num_rows(), 1);
    let panes = operator.state.panes.as_ref().unwrap();
    assert!(panes.entries.is_empty());
    assert!(panes.open_windows.is_empty());
    assert_eq!(panes.physical_bytes, 0);
}

#[tokio::test]
async fn pane_budget_counts_logical_windows_and_rejects_atomically() {
    let (schema, batch) = input(&[Some(15)], &[Some("a")], &[Some(4.0)]);
    let mut operator =
        WindowAggregateOperator::new("window", schema, hopping_spec(60, 10)).unwrap();
    operator
        .set_state_budget(StateBudget::new(5, 1_048_576).unwrap())
        .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let error = operator
        .process_data("input", batch.clone(), &context, &mut output)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("window state budget exceeded"));
    let panes = operator.state.panes.as_ref().unwrap();
    assert!(panes.entries.is_empty());
    assert!(panes.open_windows.is_empty());
    assert_eq!(panes.physical_bytes, 0);

    operator
        .set_state_budget(StateBudget::new(6, 1_048_576).unwrap())
        .unwrap();
    operator
        .process_data("input", batch, &context, &mut output)
        .await
        .unwrap();
    let panes = operator.state.panes.as_ref().unwrap();
    assert_eq!(panes.entries.len(), 1);
    assert_eq!(panes.open_windows.len(), 6);
}

struct RejectOutput;

#[async_trait]
impl StreamCollector for RejectOutput {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Err(operator_error("window", "output rejected"))
    }
}

#[tokio::test]
async fn rejected_pane_close_restores_the_exact_checkpoint_state() {
    let (schema, batch) = input(
        &[Some(-5), Some(0), Some(5)],
        &[Some("a"), Some("a"), Some("a")],
        &[Some(1.0), Some(2.0), Some(3.0)],
    );
    let mut operator =
        WindowAggregateOperator::new("window", schema, hopping_spec(20, 10)).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", batch, &context, &mut output)
        .await
        .unwrap();
    let before = operator.checkpoint(crate::Epoch::INITIAL).unwrap();
    let error = operator
        .on_watermark(EventTime::from_micros(20), &context, &mut RejectOutput)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("output rejected"));
    let after = operator.checkpoint(crate::Epoch::new(2).unwrap()).unwrap();
    assert_eq!(before.inline_metadata, after.inline_metadata);
    assert_eq!(before.segments, after.segments);
    assert!(operator.state.last_input_watermark.is_none());
}

#[tokio::test]
async fn pane_checkpoint_restore_is_bit_identical_for_continuation() {
    let (schema, first) = input(
        &[Some(-5), Some(0), Some(5)],
        &[Some("a"), Some("a"), Some("a")],
        &[
            Some(-0.0),
            Some(0.0),
            Some(f64::from_bits(0x7ff8_0000_0000_abcd)),
        ],
    );
    let (_, second) = input(
        &[Some(10), Some(15), Some(20)],
        &[Some("a"), Some("a"), Some("a")],
        &[Some(f64::INFINITY), None, Some(7.0)],
    );
    let spec = hopping_spec(20, 10);
    let mut operator =
        WindowAggregateOperator::new("window", schema.clone(), spec.clone()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", first, &context, &mut output)
        .await
        .unwrap();
    let snapshot = operator.checkpoint(crate::Epoch::INITIAL).unwrap();
    assert_eq!(
        snapshot.inline_metadata.get("state_layout_version"),
        Some(&json!(panes::WINDOW_PANE_STATE_LAYOUT_VERSION))
    );

    let mut restored = WindowAggregateOperator::new("window", schema, spec).unwrap();
    restored.restore(&snapshot).unwrap();
    let mut resumed_output = crate::EdgeCollector::new(restored.output_ports().to_vec());
    operator
        .process_data("input", second.clone(), &context, &mut output)
        .await
        .unwrap();
    restored
        .process_data("input", second, &context, &mut resumed_output)
        .await
        .unwrap();
    let continued = operator.checkpoint(crate::Epoch::new(2).unwrap()).unwrap();
    let restored_continuation = restored.checkpoint(crate::Epoch::new(2).unwrap()).unwrap();
    assert_eq!(
        continued.inline_metadata,
        restored_continuation.inline_metadata
    );
    assert_eq!(continued.segments, restored_continuation.segments);

    operator
        .on_watermark(EventTime::from_micros(40), &context, &mut output)
        .await
        .unwrap();
    restored
        .on_watermark(EventTime::from_micros(40), &context, &mut resumed_output)
        .await
        .unwrap();
    let expected = output_batches(&mut output);
    let actual = output_batches(&mut resumed_output);
    assert_eq!(expected.len(), actual.len());
    for (expected, actual) in expected.iter().zip(&actual) {
        assert_eq!(expected.metadata(), actual.metadata());
        assert_eq!(
            expected.table_payload().unwrap().batches(),
            actual.table_payload().unwrap().batches()
        );
    }
}
