use super::*;
use crate::{CancellationToken, EdgeCollector, StreamJobContext};
use datafusion::arrow::array::{Int64Array, TimestampMicrosecondArray};

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn operator(policy: LatePolicySpec, lateness: u64) -> RollingOperator {
    let mut spec = valid_spec();
    spec.late_policy = policy;
    spec.allowed_lateness_micros = lateness;
    RollingOperator::new("roll", Arc::new(input_schema()), spec).unwrap()
}

fn side_operator(lateness: u64) -> RollingOperator {
    operator(
        LatePolicySpec::SideOutput {
            metrics_version: 1,
            schema_version: 1,
        },
        lateness,
    )
}

type InputRow<'a> = (i64, &'a str, u64, Option<f64>);

fn batch(records: &[&[InputRow<'_>]]) -> Batch {
    Batch::table(
        records
            .iter()
            .map(|rows| float64_fast_record(rows))
            .collect(),
        BatchMetadata::new("upstream", 42, JsonMap::new()).unwrap(),
    )
    .unwrap()
}

fn drain_records(output: &mut EdgeCollector, port: &str) -> Vec<RecordBatch> {
    output
        .drain(port)
        .into_iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .to_vec()
        })
        .collect()
}

struct FailingCollector {
    fail_at: usize,
    sent: Vec<Batch>,
}

#[async_trait::async_trait]
impl StreamCollector for FailingCollector {
    async fn emit(&mut self, port: &str, batch: Batch) -> Result<()> {
        assert_eq!(port, "late");
        if self.sent.len() == self.fail_at {
            return Err(operator_error("sink", "original sink failure"));
        }
        self.sent.push(batch);
        Ok(())
    }
}

#[tokio::test]
async fn test_native_late_rolling_emit_failure_preserves_state_and_forbids_live_retry() {
    for fail_at in [0, 1] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "roll", Some(EventTime::from_micros(10)))
            .with_test_output_budget(crate::EdgeBudget::new(10, 200).unwrap());
        let mut side = side_operator(0);
        let mut good = EdgeCollector::new(side.output_ports().to_vec());
        let seed = batch(&[&[(26, "a", 1, Some(26.0))]]);
        side.process_late_data(&seed, context.input_watermark(), &context, &mut good)
            .await
            .unwrap();
        let input = batch(&[&[
            (5, "a", 2, Some(5.0)),
            (6, "a", 3, Some(6.0)),
            (20, "a", 4, Some(20.0)),
        ]]);
        let mut failing = FailingCollector {
            fail_at,
            sent: Vec::new(),
        };
        let error = side
            .process_late_data(&input, context.input_watermark(), &context, &mut failing)
            .await
            .unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Operator { ref node_id, ref message } if node_id == "sink" && message == "original sink failure")
        );
        assert_eq!(failing.sent.len(), fail_at);
        assert_eq!(side.state.buffer.len(), 1);
        assert_eq!(side.state.metrics, LateMetricDelta::default());
        assert_eq!(side.state.next_late_output_sequence, 0);
        let retry = side
            .process_late_data(&input, context.input_watermark(), &context, &mut good)
            .await
            .unwrap_err();
        assert!(retry.to_string().contains("retry"), "{retry}");
        assert!(good.drain("late").is_empty());
        assert!(good.drain("output").is_empty());
    }
}

#[tokio::test]
async fn test_native_late_rolling_mixed_envelope_matches_drop_and_retains_late_occurrences() {
    let batch = batch(&[
        &[
            (12, "a", 1, Some(12.0)),
            (8, "a", 2, Some(8.0)),
            (8, "a", 2, Some(8.0)),
        ],
        &[(11, "a", 3, Some(11.0)), (7, "a", 4, Some(7.0))],
    ]);
    let original = batch.table_payload().unwrap().batches().to_vec();
    let job = job();
    let context = StreamOperatorContext::new(&job, "roll", Some(EventTime::from_micros(10)));
    let mut side = side_operator(2);
    let mut drop = operator(LatePolicySpec::Drop { metrics_version: 1 }, 2);
    let mut side_output = EdgeCollector::new(side.output_ports().to_vec());
    let mut drop_output = EdgeCollector::new(drop.output_ports().to_vec());
    side.process_late_data(
        &batch,
        context.input_watermark(),
        &context,
        &mut side_output,
    )
    .await
    .unwrap();
    drop.process_data("input", batch.clone(), &context, &mut drop_output)
        .await
        .unwrap();
    assert!(side_output.drain("output").is_empty());
    assert_eq!(side.state.metrics, drop.state.metrics);
    assert_eq!(side.state.metrics.late_rows, 3);
    let late = drain_records(&mut side_output, "late");
    let values = late
        .iter()
        .map(|record| {
            record
                .column(0)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .value(0)
        })
        .collect::<Vec<_>>();
    let indices = late
        .iter()
        .map(|record| {
            record
                .column_by_name("_cf_late_row_index")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0)
        })
        .collect::<Vec<_>>();
    let closing = late
        .iter()
        .map(|record| {
            record
                .column_by_name("_cf_late_closing_time_micros")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0)
        })
        .collect::<Vec<_>>();
    assert_eq!(values, vec![8, 8, 7]);
    assert_eq!(indices, vec![1, 2, 4]);
    assert_eq!(closing, vec![10, 10, 9]);
    let accepted = side.take_all_buffered();
    side.emit_rows(accepted, &context, &mut side_output)
        .await
        .unwrap();
    drop.on_end(&context, &mut drop_output).await.unwrap();
    assert_eq!(
        drain_records(&mut side_output, "output"),
        drain_records(&mut drop_output, "output")
    );
    assert_eq!(batch.table_payload().unwrap().batches(), original);
}

#[tokio::test]
async fn test_native_late_ordered_fast_path_and_duplicate_rollback_preserve_prior_buffer() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "roll", Some(EventTime::from_micros(10)));
    let mut spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 2)]));
    spec.late_policy = LatePolicySpec::SideOutput {
        metrics_version: 1,
        schema_version: 1,
    };
    let mut side = RollingOperator::new("roll", Arc::new(kernel_schema()), spec).unwrap();
    let mut output = EdgeCollector::new(side.output_ports().to_vec());
    let seed = batch(&[&[(11, "a", 1, Some(11.0)), (12, "a", 2, Some(12.0))]]);
    side.process_late_data(&seed, context.input_watermark(), &context, &mut output)
        .await
        .unwrap();
    assert!(
        !side.state.ordered.is_empty(),
        "wholly on-time data must keep the ordered fast path"
    );
    assert!(side.state.buffer.is_empty());
    assert!(output.drain("late").is_empty());
    let duplicate = batch(&[&[(5, "a", 9, Some(5.0)), (11, "a", 1, Some(99.0))]]);
    let error = side
        .process_late_data(&duplicate, context.input_watermark(), &context, &mut output)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("duplicate row identity"),
        "{error}"
    );
    assert!(!side.state.ordered.is_empty());
    assert!(side.state.buffer.is_empty());
    assert_eq!(side.state.metrics, LateMetricDelta::default());
    assert_eq!(side.state.next_late_output_sequence, 0);
    assert!(output.drain("output").is_empty());
    assert!(output.drain("late").is_empty());
    let accepted = batch(&[&[(5, "a", 9, Some(5.0)), (13, "a", 3, Some(13.0))]]);
    side.process_late_data(&accepted, context.input_watermark(), &context, &mut output)
        .await
        .unwrap();
    assert!(side.state.ordered.is_empty());
    assert_eq!(side.state.buffer.len(), 3);
    assert_eq!(side.state.metrics.late_rows, 1);
    assert_eq!(side.state.next_late_output_sequence, 1);
}

#[test]
fn test_late_preflight_rolling_does_not_copy_oversize_late_payload() {
    let input = batch(&[&[(5, "a", 1, Some(5.0))]]);
    let record = &input.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[5] = Arc::new(datafusion::arrow::array::StringArray::from(vec![
        "x".repeat(2 * 1024 * 1024),
    ]));
    let input = Batch::table(
        vec![RecordBatch::try_new(record.schema(), columns).unwrap()],
        input.metadata().clone(),
    )
    .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "roll", Some(EventTime::from_micros(10)))
        .with_test_output_budget(crate::EdgeBudget::new(10, 200).unwrap());
    let mut side = side_operator(0);
    let mut output = EdgeCollector::new(side.output_ports().to_vec());
    let allocations = allocation_counter::measure(|| {
        let error = futures::executor::block_on(side.process_late_data(
            &input,
            context.input_watermark(),
            &context,
            &mut output,
        ))
        .unwrap_err();
        assert!(
            error.to_string().contains("output_row_too_large"),
            "{error}"
        );
    });
    assert!(
        allocations.bytes_max < 64 * 1024,
        "late payload was scalarized: {} bytes",
        allocations.bytes_max
    );
    assert!(side.state.buffer.is_empty());
    assert_eq!(side.state.metrics, LateMetricDelta::default());
    assert_eq!(side.state.next_late_output_sequence, 0);
    assert!(output.drain("late").is_empty());
}

struct BlockingCollector;

#[async_trait::async_trait]
impl StreamCollector for BlockingCollector {
    async fn emit(&mut self, _: &str, batch: Batch) -> Result<()> {
        std::future::pending::<()>().await;
        drop(batch);
        unreachable!()
    }
}

#[tokio::test]
async fn test_native_late_rolling_aborted_emit_poison_is_retained() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "roll", Some(EventTime::from_micros(12)));
    let mut side = side_operator(2);
    let input = batch(&[&[(5, "a", 1, Some(5.0)), (20, "a", 2, Some(20.0))]]);
    let mut output = BlockingCollector;
    let mut callback =
        Box::pin(side.process_late_data(&input, context.input_watermark(), &context, &mut output));
    assert!(futures::poll!(callback.as_mut()).is_pending());
    drop(callback);
    assert!(side.state.buffer.is_empty());
    assert!(side.state.ordered.is_empty());
    assert_eq!(side.state.metrics, LateMetricDelta::default());
    assert_eq!(side.state.next_late_output_sequence, 0);
    let error = side
        .process_late_data(&input, context.input_watermark(), &context, &mut output)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("retry"));
}

fn assert_uninstalled(
    side: &RollingOperator,
    output: &mut EdgeCollector,
    metrics: LateMetricDelta,
    sequence: u64,
) {
    assert!(side.state.buffer.is_empty());
    assert!(side.state.ordered.is_empty());
    assert_eq!(side.state.metrics, metrics);
    assert_eq!(side.state.next_late_output_sequence, sequence);
    assert!(!side.state.late_output_failed);
    assert!(output.drain("output").is_empty());
    assert!(output.drain("late").is_empty());
}

fn faulty_input(case: &str) -> Batch {
    let mut input = batch(&[
        &[(5, "a", 1, Some(5.0)), (20, "a", 2, Some(20.0))],
        &[(6, "b", 3, Some(6.0))],
    ]);
    match case {
        "later_oversize" => {
            let records = input.table_payload().unwrap().batches();
            let mut columns = records[1].columns().to_vec();
            columns[1] = Arc::new(datafusion::arrow::array::StringArray::from(vec![
                "x".repeat(1024),
            ]));
            input = Batch::table(
                vec![
                    records[0].clone(),
                    RecordBatch::try_new(records[1].schema(), columns).unwrap(),
                ],
                input.metadata().clone(),
            )
            .unwrap();
        }
        "duplicate" => {
            input = batch(&[&[
                (5, "a", 1, Some(5.0)),
                (20, "a", 2, Some(20.0)),
                (20, "a", 2, None),
            ]]);
        }
        "closing" => input = batch(&[&[(5, "a", 1, Some(5.0)), (i64::MAX, "a", 2, None)]]),
        _ => {}
    }
    input
}

fn preflight_error(case: &str) -> &str {
    match case {
        "later_oversize" => "output_row_too_large",
        "scratch_rows" | "scratch_bytes" => "late scratch",
        "sequence" => "late output sequence",
        "metrics" => "window metric late_rows",
        "duplicate" => "duplicate row identity",
        "closing" => "overflowed",
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn test_native_late_rolling_preflight_errors_leave_the_whole_envelope_uninstalled() {
    for case in [
        "later_oversize",
        "scratch_rows",
        "scratch_bytes",
        "sequence",
        "metrics",
        "duplicate",
        "closing",
    ] {
        let job = job();
        let budget = match case {
            "scratch_rows" => crate::EdgeBudget::new(1, 1000).unwrap(),
            "scratch_bytes" => crate::EdgeBudget::new(10, 95).unwrap(),
            _ => crate::EdgeBudget::new(10, 200).unwrap(),
        };
        let context = StreamOperatorContext::new(&job, "roll", Some(EventTime::from_micros(12)))
            .with_test_output_budget(budget);
        let mut side = side_operator(2);
        let input = faulty_input(case);
        if case == "sequence" {
            side.state.next_late_output_sequence = u64::MAX - 1;
        }
        if case == "metrics" {
            side.state.metrics.late_rows = u64::MAX;
        }
        let expected = preflight_error(case);
        let metrics = side.state.metrics;
        let sequence = side.state.next_late_output_sequence;
        let mut output = EdgeCollector::new(side.output_ports().to_vec());
        let error = side
            .process_late_data(&input, context.input_watermark(), &context, &mut output)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{case}: {error}");
        assert_uninstalled(&side, &mut output, metrics, sequence);
    }
}

#[tokio::test]
async fn test_native_late_rolling_empty_no_watermark_and_independent_sequences() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "roll", None);
    let mut side = side_operator(2);
    let mut output = EdgeCollector::new(side.output_ports().to_vec());
    side.process_late_data(&batch(&[&[]]), None, &context, &mut output)
        .await
        .unwrap();
    assert_uninstalled(&side, &mut output, LateMetricDelta::default(), 0);
    let no_watermark = batch(&[&[(5, "a", 1, Some(5.0))]]);
    side.process_late_data(&no_watermark, None, &context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("late").is_empty());
    assert_eq!(side.state.next_late_output_sequence, 0);
    for expected in 0..2 {
        let context = StreamOperatorContext::new(&job, "roll", Some(EventTime::from_micros(12)));
        side.process_late_data(
            &no_watermark,
            context.input_watermark(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
        let messages = output.drain("late");
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].as_data().unwrap().metadata().sequence(),
            expected
        );
    }
    assert_eq!(side.state.metrics.late_rows, 2);
    assert_eq!(side.state.metrics.affected_batches, 2);
    assert_eq!(side.state.next_late_output_sequence, 2);
}

#[tokio::test]
async fn test_native_late_ordered_wholly_normal_output_matches_drop() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "roll", Some(EventTime::from_micros(10)));
    let mut spec = kernel_spec(json!([aggregate_output("mean", "price", "mean", 2)]));
    let mut drop = RollingOperator::new("roll", Arc::new(kernel_schema()), spec.clone()).unwrap();
    spec.late_policy = LatePolicySpec::SideOutput {
        metrics_version: 1,
        schema_version: 1,
    };
    let mut side = RollingOperator::new("roll", Arc::new(kernel_schema()), spec).unwrap();
    let mut side_output = EdgeCollector::new(side.output_ports().to_vec());
    let mut drop_output = EdgeCollector::new(drop.output_ports().to_vec());
    let input = batch(&[&[(11, "a", 1, Some(11.0))], &[(12, "a", 2, Some(12.0))]]);
    side.process_late_data(
        &input,
        context.input_watermark(),
        &context,
        &mut side_output,
    )
    .await
    .unwrap();
    drop.process_data("input", input, &context, &mut drop_output)
        .await
        .unwrap();
    assert!(!side.state.ordered.is_empty());
    assert!(side.state.buffer.is_empty());
    assert!(side_output.drain("late").is_empty());
    let records = side.state.ordered.take_all();
    side.emit_ordered(records, &context, &mut side_output)
        .await
        .unwrap();
    drop.on_end(&context, &mut drop_output).await.unwrap();
    assert_eq!(
        drain_records(&mut side_output, "output"),
        drain_records(&mut drop_output, "output")
    );
}

struct RejectingMetrics;

impl crate::operator::LateMetricSink for RejectingMetrics {
    fn record(&self, _: LateMetricDelta) -> Result<()> {
        unreachable!("late metrics must be staged")
    }
}

#[tokio::test]
async fn test_native_late_rolling_schema_cancellation_and_metric_sink_fail_before_emit() {
    for case in ["schema", "cancelled", "metrics"] {
        let job = job();
        let context = if case == "metrics" {
            StreamOperatorContext::for_task(
                &job,
                "roll",
                Some(EventTime::from_micros(12)),
                crate::IngressProgressSnapshot::default(),
                crate::EdgeBudget::default(),
                Arc::new(RejectingMetrics),
            )
        } else {
            StreamOperatorContext::new(&job, "roll", Some(EventTime::from_micros(12)))
        };
        let mut side = side_operator(2);
        let mut input = batch(&[&[(5, "a", 1, Some(5.0)), (20, "a", 2, Some(20.0))]]);
        if case == "schema" {
            let record = &input.table_payload().unwrap().batches()[0];
            let mut fields = record.schema().fields().to_vec();
            fields[0] = Arc::new(fields[0].as_ref().clone().with_name("wrong_time_column"));
            input = Batch::table(
                vec![
                    RecordBatch::try_new(Arc::new(Schema::new(fields)), record.columns().to_vec())
                        .unwrap(),
                ],
                input.metadata().clone(),
            )
            .unwrap();
        }
        if case == "cancelled" {
            job.cancellation().cancel();
        }
        let mut output = EdgeCollector::new(side.output_ports().to_vec());
        let error = side
            .process_late_data(&input, context.input_watermark(), &context, &mut output)
            .await
            .unwrap_err();
        let expected = match case {
            "schema" => "schema mismatch",
            "cancelled" => "cancel",
            _ => "staged updates",
        };
        assert!(error.to_string().contains(expected), "{error}");
        assert_uninstalled(&side, &mut output, LateMetricDelta::default(), 0);
    }
}

#[test]
fn test_native_late_rolling_invalid_time_and_key_are_never_diagnostics() {
    let side = side_operator(2);
    let input = batch(&[&[(5, "a", 1, Some(5.0))]]);
    let record = &input.table_payload().unwrap().batches()[0];
    for (index, array, expected) in [
        (
            side.compiled.event_time_index,
            Arc::new(Int64Array::from(vec![5])) as ArrayRef,
            "event-time value",
        ),
        (
            side.compiled.event_time_index,
            Arc::new(TimestampMicrosecondArray::from(vec![None::<i64>]).with_timezone("UTC"))
                as ArrayRef,
            "event-time value",
        ),
        (
            side.compiled.sequence_columns[0].index,
            Arc::new(UInt64Array::from(vec![None::<u64>])) as ArrayRef,
            "sequence key value is null",
        ),
    ] {
        let mut fields = record.schema().fields().to_vec();
        fields[index] = Arc::new(Field::new(
            fields[index].name(),
            array.data_type().clone(),
            true,
        ));
        let mut columns = record.columns().to_vec();
        columns[index] = array;
        let malformed = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        let error = side.read_late_identity(&malformed, 0, "roll").unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}
