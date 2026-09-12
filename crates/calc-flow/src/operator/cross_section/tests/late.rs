use super::*;
use crate::{CancellationToken, EdgeCollector, StreamJobContext};
use datafusion::arrow::array::{
    Float64Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array,
};

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn operator(policy: LatePolicySpec, grouping: CrossSectionGroupingSpec) -> CrossSectionOperator {
    let mut spec = valid_spec();
    spec.late_policy = policy;
    spec.grouping = grouping;
    spec.allowed_lateness_micros = 2;
    CrossSectionOperator::new("cross", Arc::new(input_schema()), spec).unwrap()
}

type InputRow<'a> = (i64, &'a str, u64, Option<f64>);

fn batch(records: &[&[InputRow<'_>]]) -> Batch {
    let records = records
        .iter()
        .map(|rows| {
            RecordBatch::try_new(
                Arc::new(input_schema()),
                vec![
                    Arc::new(
                        TimestampMicrosecondArray::from_iter_values(rows.iter().map(|row| row.0))
                            .with_timezone("UTC"),
                    ) as ArrayRef,
                    Arc::new(StringArray::from_iter_values(rows.iter().map(|row| row.1))),
                    Arc::new(StringArray::from(vec![Some("g"); rows.len()])),
                    Arc::new(UInt64Array::from_iter_values(rows.iter().map(|row| row.2))),
                    Arc::new(rows.iter().map(|row| row.3).collect::<Float64Array>()),
                ],
            )
            .unwrap()
        })
        .collect();
    Batch::table(
        records,
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
async fn test_native_late_cross_section_emit_failure_preserves_state_and_forbids_live_retry() {
    for fail_at in [0, 1] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "cross", Some(EventTime::from_micros(12)))
            .with_test_output_budget(crate::EdgeBudget::new(10, 200).unwrap());
        let mut side = operator(
            LatePolicySpec::SideOutput {
                metrics_version: 1,
                schema_version: 1,
            },
            CrossSectionGroupingSpec::ExactTime,
        );
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
        assert_eq!(side.state.groups.len(), 1);
        assert_eq!(side.state.identity_groups.len(), 1);
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
async fn test_native_late_cross_section_exact_and_bucket_boundaries_match_drop() {
    for (grouping, normal, late, older, closing) in [
        (CrossSectionGroupingSpec::ExactTime, 12, 10, 9, [12, 12, 11]),
        (
            CrossSectionGroupingSpec::FixedBucket { width_micros: 10 },
            10,
            9,
            -1,
            [12, 12, 2],
        ),
    ] {
        let batch = batch(&[
            &[
                (normal, "a", 1, Some(3.0)),
                (late, "d", 9, Some(9.0)),
                (late, "d", 9, Some(9.0)),
            ],
            &[(normal, "b", 2, Some(1.0)), (older, "e", 10, None)],
        ]);
        let original = batch.table_payload().unwrap().batches().to_vec();
        let job = job();
        let context = StreamOperatorContext::new(&job, "cross", Some(EventTime::from_micros(12)));
        let mut side = operator(
            LatePolicySpec::SideOutput {
                metrics_version: 1,
                schema_version: 1,
            },
            grouping,
        );
        let mut drop = operator(LatePolicySpec::Drop { metrics_version: 1 }, grouping);
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
        assert_eq!(side.state.metrics, drop.state.metrics);
        assert_eq!(side.state.metrics.late_rows, 3);
        assert!(side_output.drain("output").is_empty());
        let records = drain_records(&mut side_output, "late");
        let events = records
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
        let actual_closing = records
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
        assert_eq!(events, vec![late, late, older]);
        assert_eq!(actual_closing, closing);
        let groups = std::mem::take(&mut side.state.groups);
        side.emit_groups(groups, &context, &mut side_output)
            .await
            .unwrap();
        drop.on_end(&context, &mut drop_output).await.unwrap();
        assert_eq!(
            drain_records(&mut side_output, "output"),
            drain_records(&mut drop_output, "output")
        );
        assert_eq!(batch.table_payload().unwrap().batches(), original);
    }
}

#[test]
fn test_late_preflight_cross_section_does_not_copy_oversize_late_payload() {
    let input = batch(&[&[(5, "a", 1, Some(5.0))]]);
    let record = &input.table_payload().unwrap().batches()[0];
    let mut fields = record.schema().fields().to_vec();
    fields.push(Arc::new(Field::new("payload", DataType::Utf8, false)));
    let schema = Arc::new(Schema::new(fields));
    let mut columns = record.columns().to_vec();
    columns.push(Arc::new(StringArray::from(vec![
        "x".repeat(2 * 1024 * 1024),
    ])));
    let input = Batch::table(
        vec![RecordBatch::try_new(schema.clone(), columns).unwrap()],
        input.metadata().clone(),
    )
    .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "cross", Some(EventTime::from_micros(12)))
        .with_test_output_budget(crate::EdgeBudget::new(10, 200).unwrap());
    let mut spec = valid_spec();
    spec.late_policy = LatePolicySpec::SideOutput {
        metrics_version: 1,
        schema_version: 1,
    };
    let mut side = CrossSectionOperator::new("cross", schema, spec).unwrap();
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
    assert!(side.state.groups.is_empty());
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
async fn test_native_late_cross_section_aborted_emit_poison_is_retained() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "cross", Some(EventTime::from_micros(12)));
    let mut side = operator(
        LatePolicySpec::SideOutput {
            metrics_version: 1,
            schema_version: 1,
        },
        CrossSectionGroupingSpec::ExactTime,
    );
    let input = batch(&[&[(5, "a", 1, Some(5.0)), (20, "a", 2, Some(20.0))]]);
    let mut output = BlockingCollector;
    let mut callback =
        Box::pin(side.process_late_data(&input, context.input_watermark(), &context, &mut output));
    assert!(futures::poll!(callback.as_mut()).is_pending());
    drop(callback);
    assert!(side.state.groups.is_empty());
    assert!(side.state.identity_groups.is_empty());
    assert_eq!(side.state.metrics, LateMetricDelta::default());
    assert_eq!(side.state.next_late_output_sequence, 0);
    let error = side
        .process_late_data(&input, context.input_watermark(), &context, &mut output)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("retry"));
}

fn assert_uninstalled(
    side: &CrossSectionOperator,
    output: &mut EdgeCollector,
    metrics: LateMetricDelta,
    sequence: u64,
) {
    assert!(side.state.groups.is_empty());
    assert!(side.state.identity_groups.is_empty());
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
            columns[1] = Arc::new(StringArray::from(vec!["x".repeat(1024)]));
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
async fn test_native_late_cross_section_preflight_errors_leave_the_whole_envelope_uninstalled() {
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
        let context = StreamOperatorContext::new(&job, "cross", Some(EventTime::from_micros(12)))
            .with_test_output_budget(budget);
        let mut side = operator(
            LatePolicySpec::SideOutput {
                metrics_version: 1,
                schema_version: 1,
            },
            CrossSectionGroupingSpec::ExactTime,
        );
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
async fn test_native_late_cross_section_empty_no_watermark_and_independent_sequences() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "cross", None);
    let mut side = operator(
        LatePolicySpec::SideOutput {
            metrics_version: 1,
            schema_version: 1,
        },
        CrossSectionGroupingSpec::ExactTime,
    );
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
        let context = StreamOperatorContext::new(&job, "cross", Some(EventTime::from_micros(12)));
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

struct RejectingMetrics;

impl crate::operator::LateMetricSink for RejectingMetrics {
    fn record(&self, _: LateMetricDelta) -> Result<()> {
        unreachable!("late metrics must be staged")
    }
}

#[tokio::test]
async fn test_native_late_cross_section_schema_cancellation_and_metric_sink_fail_before_emit() {
    for case in ["schema", "cancelled", "metrics"] {
        let job = job();
        let context = if case == "metrics" {
            StreamOperatorContext::for_task(
                &job,
                "cross",
                Some(EventTime::from_micros(12)),
                crate::IngressProgressSnapshot::default(),
                crate::EdgeBudget::default(),
                Arc::new(RejectingMetrics),
            )
        } else {
            StreamOperatorContext::new(&job, "cross", Some(EventTime::from_micros(12)))
        };
        let mut side = operator(
            LatePolicySpec::SideOutput {
                metrics_version: 1,
                schema_version: 1,
            },
            CrossSectionGroupingSpec::ExactTime,
        );
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
fn test_native_late_cross_section_invalid_time_and_key_are_never_diagnostics() {
    let side = operator(
        LatePolicySpec::SideOutput {
            metrics_version: 1,
            schema_version: 1,
        },
        CrossSectionGroupingSpec::ExactTime,
    );
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
        let error = side.read_late_identity(&malformed, 0, "cross").unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[tokio::test]
async fn test_native_late_cross_section_rejects_bucket_overflow_and_cross_group_duplicates() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "cross", Some(EventTime::from_micros(i64::MAX)));
    for time in [i64::MIN, i64::MAX] {
        let mut side = operator(
            LatePolicySpec::SideOutput {
                metrics_version: 1,
                schema_version: 1,
            },
            CrossSectionGroupingSpec::FixedBucket { width_micros: 10 },
        );
        let mut output = EdgeCollector::new(side.output_ports().to_vec());
        let input = batch(&[&[(time, "a", 1, Some(5.0))]]);
        let error = side
            .process_late_data(&input, context.input_watermark(), &context, &mut output)
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("bucket") && error.to_string().contains("overflowed"),
            "{error}"
        );
        assert_uninstalled(&side, &mut output, LateMetricDelta::default(), 0);
    }
    let context = StreamOperatorContext::new(&job, "cross", Some(EventTime::from_micros(12)));
    let input = batch(&[&[
        (5, "a", 1, Some(5.0)),
        (20, "a", 2, Some(20.0)),
        (20, "a", 2, None),
    ]]);
    let record = &input.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[2] = Arc::new(StringArray::from(vec!["g", "g", "other"]));
    let input = Batch::table(
        vec![RecordBatch::try_new(record.schema(), columns).unwrap()],
        input.metadata().clone(),
    )
    .unwrap();
    let mut side = operator(
        LatePolicySpec::SideOutput {
            metrics_version: 1,
            schema_version: 1,
        },
        CrossSectionGroupingSpec::ExactTime,
    );
    let mut output = EdgeCollector::new(side.output_ports().to_vec());
    let error = side
        .process_late_data(&input, context.input_watermark(), &context, &mut output)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("duplicate row identity"));
    assert_uninstalled(&side, &mut output, LateMetricDelta::default(), 0);
}
