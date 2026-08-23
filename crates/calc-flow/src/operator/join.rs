#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc, time::Duration};

    use datafusion::arrow::{
        array::{
            BinaryArray, FixedSizeBinaryArray, Int64Array, StringArray, TimestampMicrosecondArray,
        },
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };

    use super::*;
    use crate::{
        BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, IngressProgress,
        IngressProgressSnapshot, IngressState, JsonMap, OperatorMetadata, StreamJobContext,
        StreamMessageKind, StreamOperator,
    };

    fn left_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("amount", DataType::Int64, true),
        ]))
    }

    fn right_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]))
    }

    fn spec() -> StreamJoinSpec {
        StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(100, 1_000_000, 1_000).unwrap(),
        )
        .unwrap()
        .with_prefixes("authorization", "payment")
        .unwrap()
    }

    fn left_batch(times: Vec<i64>) -> Batch {
        let rows = times.len();
        Batch::table(
            vec![
                RecordBatch::try_new(
                    left_schema(),
                    vec![
                        Arc::new(Int64Array::from(vec![7; rows])),
                        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
                        Arc::new(Int64Array::from(vec![42; rows])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    fn right_batch(times: Vec<i64>) -> Batch {
        let rows = times.len();
        Batch::table(
            vec![
                RecordBatch::try_new(
                    right_schema(),
                    vec![
                        Arc::new(Int64Array::from(vec![7; rows])),
                        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
                        Arc::new(StringArray::from(vec!["paid"; rows])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    #[test]
    fn derives_exact_prefixed_ports() {
        let operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();

        assert_eq!(
            operator
                .input_ports()
                .iter()
                .map(Port::name)
                .collect::<Vec<_>>(),
            ["left", "right"]
        );
        let output = operator.output_ports()[0].schema().unwrap();
        assert_eq!(
            output
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            [
                "authorization__account_id",
                "authorization__authorized_at",
                "authorization__amount",
                "payment__account_id",
                "payment__paid_at",
                "payment__status",
            ]
        );
        assert!(output.field(2).is_nullable());
        assert!(output.field(5).is_nullable());
    }

    #[test]
    fn rejects_values_beyond_the_json_safe_integer_domain() {
        let too_large = STREAM_JOIN_MAX_SAFE_JSON_INTEGER + 1;
        assert!(JoinStateLimits::new(too_large, 1, 1).is_err());
        assert!(JoinTimeBounds::new(Duration::from_micros(too_large), Duration::ZERO).is_err());
        assert!(JoinStateLimits::new(1, 1, 1).is_ok());
    }

    #[test]
    fn inclusive_bound_helper_accepts_both_edges() {
        let bounds =
            JoinTimeBounds::new(Duration::from_micros(10), Duration::from_micros(20)).unwrap();
        assert!(bounds.contains_pair(100, 90));
        assert!(bounds.contains_pair(100, 120));
        assert!(!bounds.contains_pair(100, 89));
        assert!(!bounds.contains_pair(100, 121));
    }

    #[test]
    fn output_frontier_uses_live_idle_and_ended_formulas() {
        let operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let snapshot = |left_state: IngressState,
                        left: Option<i64>,
                        right_state: IngressState,
                        right: Option<i64>| {
            IngressProgressSnapshot::new(BTreeMap::from([
                (
                    "left".into(),
                    IngressProgress::new(left_state, left.map(EventTime::from_micros)),
                ),
                (
                    "right".into(),
                    IngressProgress::new(right_state, right.map(EventTime::from_micros)),
                ),
            ]))
        };
        let micros = |value: Option<EventTime>| value.map(EventTime::as_micros);

        assert_eq!(
            micros(
                operator
                    .output_frontier_candidate(&snapshot(
                        IngressState::Idle,
                        Some(400_000_000),
                        IngressState::Active,
                        Some(100_000_000),
                    ))
                    .unwrap()
            ),
            Some(40_000_000)
        );
        assert_eq!(
            micros(
                operator
                    .output_frontier_candidate(&snapshot(
                        IngressState::Active,
                        Some(400_000_000),
                        IngressState::Ended,
                        Some(100_000_000),
                    ))
                    .unwrap()
            ),
            Some(100_000_000)
        );
        assert_eq!(
            micros(
                operator
                    .output_frontier_candidate(&snapshot(
                        IngressState::Ended,
                        Some(400_000_000),
                        IngressState::Active,
                        Some(100_000_000),
                    ))
                    .unwrap()
            ),
            Some(40_000_000)
        );
        assert_eq!(
            operator
                .output_frontier_candidate(&snapshot(
                    IngressState::Ended,
                    Some(400_000_000),
                    IngressState::Ended,
                    Some(100_000_000),
                ))
                .unwrap(),
            None
        );
        assert_eq!(
            operator
                .output_frontier_candidate(&snapshot(
                    IngressState::Active,
                    None,
                    IngressState::Active,
                    Some(100_000_000),
                ))
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn emits_duplicate_pairs_at_both_inclusive_boundaries() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data("left", left_batch(vec![100, 100]), &context, &mut collector)
            .await
            .unwrap();
        operator
            .process_data(
                "right",
                right_batch(vec![-299_999_900, 60_000_100, 60_000_101]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();

        let outputs = collector.drain("output");
        assert_eq!(outputs.len(), 4);
        assert!(
            outputs
                .iter()
                .all(|message| message.kind() == StreamMessageKind::Data)
        );
        assert_eq!(
            outputs
                .iter()
                .map(|message| message.as_data().unwrap().metadata().sequence())
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
    }

    fn job() -> StreamJobContext {
        StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
    }

    fn progress_context(
        job_context: &StreamJobContext,
        left: (IngressState, Option<i64>),
        right: (IngressState, Option<i64>),
    ) -> StreamOperatorContext<'_> {
        let snapshot = IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(left.0, left.1.map(EventTime::from_micros)),
            ),
            (
                "right".into(),
                IngressProgress::new(right.0, right.1.map(EventTime::from_micros)),
            ),
        ]));
        StreamOperatorContext::for_task(
            job_context,
            "match",
            None,
            snapshot,
            EdgeBudget::default(),
            Arc::new(NoopLateMetrics),
        )
    }

    struct NoopLateMetrics;

    impl crate::operator::LateMetricSink for NoopLateMetrics {
        fn record(&self, _delta: crate::operator::LateMetricDelta) -> Result<()> {
            Ok(())
        }
    }

    fn reason_of(error: &CalcFlowError) -> Option<crate::StreamingFailureReason> {
        match error {
            CalcFlowError::OperatorReason { reason_code, .. } => Some(*reason_code),
            _ => None,
        }
    }

    fn checkpoint_metadata(operator: &mut StreamJoinOperator, epoch: u64) -> JsonMap {
        operator
            .checkpoint(Epoch::new(epoch).unwrap())
            .unwrap()
            .inline_metadata
    }

    #[tokio::test]
    async fn late_rows_are_dropped_with_metrics_and_never_retained() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = progress_context(
            &job_context,
            (IngressState::Active, Some(500_000_000)),
            (IngressState::Active, None),
        );
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data(
                "left",
                left_batch(vec![100_000_000, 499_999_999]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();

        assert!(collector.drain("output").is_empty());
        let metadata = checkpoint_metadata(&mut operator, 1);
        let state = serde_json::to_string(&metadata["metrics"]["left"]).unwrap();
        assert!(state.contains("\"late_rows\":2"), "{state}");
        assert!(
            state.contains("\"late_affected_batches\":1")
                && state.contains("\"max_lateness_micros\":400000000"),
            "{state}"
        );
        assert!(
            state.contains("\"retained_rows\":0"),
            "late rows must never be retained: {state}"
        );
    }

    #[tokio::test]
    async fn null_event_time_and_null_key_rows_are_counted_not_stored() {
        let nullable_time_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            ),
            Field::new("amount", DataType::Int64, true),
        ]));
        let nullable_key_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, true),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]));
        let mut operator = StreamJoinOperator::new(
            "match",
            Arc::clone(&nullable_time_schema),
            Arc::clone(&nullable_key_schema),
            spec(),
        )
        .unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        let null_time = Batch::table(
            vec![
                RecordBatch::try_new(
                    Arc::clone(&nullable_time_schema),
                    vec![
                        Arc::new(Int64Array::from(vec![7])),
                        Arc::new(
                            TimestampMicrosecondArray::from(vec![None::<i64>]).with_timezone("UTC"),
                        ),
                        Arc::new(Int64Array::from(vec![42])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();
        let nullable_key_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, true),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]));
        let null_key = Batch::table(
            vec![
                RecordBatch::try_new(
                    Arc::clone(&nullable_key_schema),
                    vec![
                        Arc::new(Int64Array::from(vec![None::<i64>])),
                        Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
                        Arc::new(StringArray::from(vec!["paid"])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();

        operator
            .process_data("left", null_time, &context, &mut collector)
            .await
            .unwrap();
        operator
            .process_data("right", null_key, &context, &mut collector)
            .await
            .unwrap();

        assert!(collector.drain("output").is_empty());
        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["left"]["null_event_time_rows"], 1);
        assert_eq!(metadata["metrics"]["right"]["null_key_rows"], 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 0);
        assert_eq!(metadata["metrics"]["right"]["retained_rows"], 0);
    }

    #[tokio::test]
    async fn watermark_progress_evicts_expired_opposite_rows_and_end_clears_them() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "left",
                left_batch(vec![0, 100_000_000]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 2);

        let eviction = progress_context(
            &job_context,
            (IngressState::Active, Some(1_000_000)),
            (IngressState::Active, Some(150_000_000)),
        );
        operator
            .on_ingress_progress("right", &eviction)
            .await
            .unwrap();
        let metadata = checkpoint_metadata(&mut operator, 2);
        let left_metrics = serde_json::to_string(&metadata["metrics"]["left"]).unwrap();
        assert!(
            left_metrics.contains("\"retained_rows\":1")
                && left_metrics.contains("\"evicted_rows\":1"),
            "{left_metrics}"
        );

        let ended = progress_context(
            &job_context,
            (IngressState::Active, Some(1_000_000)),
            (IngressState::Ended, Some(50_000_000)),
        );
        operator.on_ingress_progress("right", &ended).await.unwrap();
        let metadata = checkpoint_metadata(&mut operator, 3);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 0);
    }

    #[tokio::test]
    async fn unknown_ingress_and_data_after_end_fail_loudly() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        let unknown = operator
            .process_data("middle", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap_err();
        assert!(unknown.to_string().contains("unknown ingress"), "{unknown}");

        operator.on_end(&context, &mut collector).await.unwrap();
        let after_end = operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap_err();
        assert!(
            after_end.to_string().contains("data after end-of-input"),
            "{after_end}"
        );

        let progress = IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
            ),
            (
                "right".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
            ),
            (
                "middle".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
            ),
        ]));
        let unknown_progress = operator
            .on_ingress_progress(
                "middle",
                &StreamOperatorContext::for_task(
                    &job_context,
                    "match",
                    None,
                    progress,
                    EdgeBudget::default(),
                    Arc::new(NoopLateMetrics),
                ),
            )
            .await
            .unwrap_err();
        assert!(
            unknown_progress.to_string().contains("unknown ingress"),
            "{unknown_progress}"
        );
    }

    #[tokio::test]
    async fn state_row_limit_failure_is_atomic_with_typed_reason() {
        let limited = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(1, 1_000_000, 1_000).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), limited).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let failure = operator
            .process_data("left", left_batch(vec![1]), &context, &mut collector)
            .await
            .unwrap_err();
        assert_eq!(
            reason_of(&failure),
            Some(crate::StreamingFailureReason::JoinStateLimitExceeded)
        );

        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["state_limit_failures"], 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 1);
        assert!(collector.drain("output").is_empty());
    }

    #[tokio::test]
    async fn match_limit_failure_is_atomic_with_typed_reason() {
        let limited = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(100, 1_000_000, 1).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), limited).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data("left", left_batch(vec![0, 0]), &context, &mut collector)
            .await
            .unwrap();
        let failure = operator
            .process_data("right", right_batch(vec![0, 0]), &context, &mut collector)
            .await
            .unwrap_err();
        assert_eq!(
            reason_of(&failure),
            Some(crate::StreamingFailureReason::JoinMatchLimitExceeded)
        );

        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["match_limit_failures"], 1);
        assert_eq!(metadata["metrics"]["right"]["retained_rows"], 0);
        assert!(collector.drain("output").is_empty());
    }

    #[tokio::test]
    async fn checkpoint_and_restore_round_trip_preserves_state_and_counters() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        operator
            .process_data("right", right_batch(vec![1]), &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");
        let snapshot = operator.checkpoint(Epoch::new(7).unwrap()).unwrap();

        let same_epoch = operator.checkpoint(Epoch::new(7).unwrap()).unwrap_err();
        assert!(
            same_epoch.to_string().contains("did not advance"),
            "{same_epoch}"
        );

        let mut restored =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        restored.restore(&snapshot).unwrap();
        let round_trip = restored.checkpoint(Epoch::new(8).unwrap()).unwrap();
        let mut expected_metadata = snapshot.inline_metadata.clone();
        expected_metadata.insert("epoch".into(), 8.into());
        assert_eq!(round_trip.inline_metadata, expected_metadata);
        assert_eq!(round_trip.segments, snapshot.segments);

        let mut collector = EdgeCollector::new(restored.output_ports().to_vec());
        restored
            .process_data("right", right_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let outputs = collector.drain("output");
        assert_eq!(outputs.len(), 1, "restored left state must still match");
    }

    #[tokio::test]
    async fn restore_rejects_tampered_checkpoints() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();

        let fresh = |snapshot: &OperatorStateSnapshot| {
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec())
                .unwrap()
                .restore(snapshot)
        };

        let mut bad_magic = snapshot.clone();
        bad_magic.segments.insert("left-v1".into(), vec![0_u8; 8]);
        assert!(fresh(&bad_magic).is_err(), "invalid magic must be rejected");

        let mut short_inventory = snapshot.clone();
        short_inventory.segments.remove("right-v1");
        assert!(
            fresh(&short_inventory).is_err(),
            "missing segment must be rejected"
        );

        let mut truncated = snapshot.clone();
        let bytes = truncated.segments.get_mut("left-v1").unwrap();
        bytes.truncate(bytes.len() - 1);
        assert!(
            fresh(&truncated).is_err(),
            "truncated segment must be rejected"
        );

        let mut wrong_layout = snapshot.clone();
        wrong_layout
            .inline_metadata
            .insert("layout_version".into(), 2.into());
        assert!(
            fresh(&wrong_layout).is_err(),
            "layout bump must be rejected"
        );

        let mut wrong_metrics = snapshot.clone();
        wrong_metrics
            .inline_metadata
            .entry("metrics".into())
            .or_default()["left"]["retained_rows"] = 99.into();
        assert!(
            fresh(&wrong_metrics).is_err(),
            "inconsistent retained metrics must be rejected"
        );

        let mut wrong_limits = snapshot.clone();
        wrong_limits
            .inline_metadata
            .entry("spec".into())
            .or_default()["limits"]["max_state_rows_per_side"] = 5.into();
        assert!(
            fresh(&wrong_limits).is_err(),
            "spec change must be rejected"
        );

        let mut bad_metadata = snapshot.clone();
        bad_metadata
            .inline_metadata
            .insert("layout_version".into(), "not-a-number".into());
        assert!(
            fresh(&bad_metadata).is_err(),
            "invalid metadata must be rejected"
        );

        assert!(fresh(&snapshot).is_ok(), "the untampered snapshot restores");
    }

    #[test]
    fn reset_clears_state_for_reuse() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        operator.state.left.push(StoredRow {
            record: left_batch(vec![0]).table_payload().unwrap().batches()[0].slice(0, 1),
            event_time: EventTime::from_micros(0),
            row_id: 0,
            charge: 64,
        });
        operator.state.metrics.left.retained_rows = 1;

        operator.reset().unwrap();

        assert!(operator.state.left.is_empty());
        assert_eq!(operator.state.metrics.left.retained_rows, 0);
    }

    #[test]
    fn metadata_exposes_data_only_configuration_and_debug() {
        let operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        assert_eq!(operator.name(), "match");
        assert_eq!(operator.input_ports().len(), 2);
        assert_eq!(operator.output_ports().len(), 1);
        let configuration = operator.configuration();
        assert_eq!(configuration["join_type"], "inner");
        assert_eq!(configuration["left_event_time"], "authorized_at");
        assert!(!configuration.contains_key("callable"));
        let debug = format!("{operator:?}");
        assert!(debug.contains("match"), "{debug}");
    }

    #[test]
    fn spec_validation_rejects_invalid_declarations() {
        let bounds =
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap();
        let limits = JoinStateLimits::new(100, 1_000_000, 1_000).unwrap();

        let empty_keys = StreamJoinSpec::inner(
            Vec::<String>::new(),
            ["account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        );
        assert!(empty_keys.is_err());

        let unequal = StreamJoinSpec::inner(
            ["a", "b"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        );
        assert!(unequal.is_err());

        let duplicate = StreamJoinSpec::inner(
            ["account_id", "account_id"],
            ["account_id", "account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        );
        assert!(duplicate.is_err());

        let empty_event_time = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "",
            "paid_at",
            bounds,
            limits,
        );
        assert!(empty_event_time.is_err());

        let valid = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        )
        .unwrap();
        assert!(valid.clone().with_prefixes("same", "same").is_err());
        assert!(valid.clone().with_prefixes("not valid", "right").is_err());

        let prefixed = valid.with_prefixes("authorization", "payment").unwrap();
        assert_eq!(prefixed.left_keys(), ["account_id"]);
        assert_eq!(prefixed.right_keys(), ["account_id"]);
        assert_eq!(prefixed.left_event_time(), "authorized_at");
        assert_eq!(prefixed.right_event_time(), "paid_at");
        assert_eq!(prefixed.left_prefix(), "authorization");
        assert_eq!(prefixed.right_prefix(), "payment");
        assert_eq!(prefixed.join_type(), StreamJoinType::Inner);
        assert_eq!(prefixed.bounds().before(), Duration::from_secs(300));
        assert_eq!(prefixed.bounds().after(), Duration::from_secs(60));
        assert_eq!(prefixed.limits().max_state_rows_per_side(), 100);
        assert_eq!(prefixed.limits().max_state_bytes_per_side(), 1_000_000);
        assert_eq!(prefixed.limits().max_matches_per_input_batch(), 1_000);
        assert!(format!("{prefixed:?}").contains("StreamJoinSpec"));
    }

    #[test]
    fn serde_round_trips_and_rejects_unknown_or_wrong_kind_fields() {
        let source = r#"{
            "join_type": "inner",
            "left_keys": ["account_id"],
            "right_keys": ["account_id"],
            "left_event_time": "authorized_at",
            "right_event_time": "paid_at",
            "bounds": {"before_micros": 300000000, "after_micros": 60000000},
            "limits": {
                "max_state_rows_per_side": 100,
                "max_state_bytes_per_side": 1000000,
                "max_matches_per_input_batch": 1000
            }
        }"#;
        let parsed: StreamJoinSpec = serde_json::from_str(source).unwrap();
        assert_eq!(parsed.left_prefix(), "left");
        assert_eq!(parsed.right_prefix(), "right");
        let encoded = serde_json::to_value(&parsed).unwrap();
        assert_eq!(encoded["bounds"]["before_micros"], 300_000_000);

        let unknown = source.replace(
            "\"join_type\": \"inner\",",
            "\"join_type\": \"inner\", \"extra\": 1,",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&unknown).is_err());

        let outer_join = source.replace("\"inner\"", "\"outer\"");
        assert!(serde_json::from_str::<StreamJoinSpec>(&outer_join).is_err());

        let unknown_bound = source.replace(
            "\"before_micros\": 300000000,",
            "\"before_micros\": 300000000, \"extra\": 1,",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&unknown_bound).is_err());

        let unknown_limit = source.replace(
            "\"max_matches_per_input_batch\": 1000",
            "\"max_matches_per_input_batch\": 1000, \"extra\": 1",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&unknown_limit).is_err());

        let zero_limit = source.replace(
            "\"max_state_rows_per_side\": 100",
            "\"max_state_rows_per_side\": 0",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&zero_limit).is_err());
    }

    #[tokio::test]
    async fn event_time_columns_accept_every_timestamp_unit_and_reject_others() {
        for (unit, _value) in [
            (TimeUnit::Second, 1_i64),
            (TimeUnit::Millisecond, 1_000),
            (TimeUnit::Microsecond, 1_000_000),
            (TimeUnit::Nanosecond, 1_000_000_000),
        ] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("account_id", DataType::Int64, false),
                Field::new(
                    "authorized_at",
                    DataType::Timestamp(unit, Some("UTC".into())),
                    false,
                ),
                Field::new("amount", DataType::Int64, true),
            ]));
            let operator = StreamJoinOperator::new("match", schema, right_schema(), spec());
            assert!(operator.is_ok(), "unit {unit:?} must be supported");
        }

        let not_a_timestamp = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new("authorized_at", DataType::Int64, false),
            Field::new("amount", DataType::Int64, true),
        ]));
        let rejected = StreamJoinOperator::new("match", not_a_timestamp, right_schema(), spec());
        assert!(
            rejected.is_err(),
            "non-timestamp event time must be rejected"
        );

        let missing_key = Arc::new(Schema::new(vec![
            Field::new("ledger_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
        ]));
        let rejected = StreamJoinOperator::new("match", missing_key, right_schema(), spec());
        assert!(rejected.is_err(), "missing key column must be rejected");

        let wrong_key_type = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Float64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("amount", DataType::Int64, true),
        ]));
        let rejected = StreamJoinOperator::new("match", wrong_key_type, right_schema(), spec());
        assert!(rejected.is_err(), "unsupported key type must be rejected");

        let zoned = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("America/New_York".into())),
                false,
            ),
            Field::new("amount", DataType::Int64, true),
        ]));
        let rejected = StreamJoinOperator::new("match", zoned, right_schema(), spec());
        assert!(rejected.is_err(), "non-UTC timezone must be rejected");
    }

    #[tokio::test]
    async fn variable_width_payloads_are_charged_and_limited_deterministically() {
        let payload_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("notes", DataType::Utf8, true),
            Field::new("blob", DataType::Binary, true),
            Field::new("tag", DataType::FixedSizeBinary(4), true),
        ]));
        let keys_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]));
        let tiny = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(100, 96, 1_000).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", Arc::clone(&payload_schema), keys_schema, tiny)
                .unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        let batch = Batch::table(
            vec![
                RecordBatch::try_new(
                    payload_schema,
                    vec![
                        Arc::new(Int64Array::from(vec![7])),
                        Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
                        Arc::new(StringArray::from(vec!["0123456789"])),
                        Arc::new(BinaryArray::from_opt_vec(vec![Some(&[0_u8; 8][..])])),
                        Arc::new(
                            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                                vec![Some([1_u8; 4])].into_iter(),
                                4,
                            )
                            .unwrap(),
                        ),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();

        let failure = operator
            .process_data("left", batch, &context, &mut collector)
            .await
            .unwrap_err();
        assert_eq!(
            reason_of(&failure),
            Some(crate::StreamingFailureReason::JoinStateLimitExceeded)
        );
        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 0);
    }
}
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    io::Cursor,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use datafusion::arrow::{
    array::{
        Array, ArrayRef, BinaryArray, BinaryViewArray, LargeBinaryArray, LargeStringArray,
        StringArray, StringViewArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray, UInt64Array,
    },
    compute::concat,
    datatypes::{DataType, Field, IntervalUnit, Schema, SchemaRef, TimeUnit},
    ipc::{reader::StreamReader, writer::StreamWriter},
    record_batch::RecordBatch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;

use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, DataFusionConfig, DataFusionRuntime, Epoch,
    EventTime, IngressProgress, JsonMap, OperatorStateSnapshot, Port, Result, StreamCollector,
    StreamOperator, StreamOperatorContext, UdfRegistrySnapshot,
};

use super::{OperatorMetadata, StreamRuntimeState, is_identifier, validate_operator_name};

/// The fixed logical bookkeeping charge for one retained Join row.
pub const STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1: u64 = 64;

/// Largest integer that round-trips exactly through ordinary JSON numbers.
pub const STREAM_JOIN_MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;

/// Supported Join semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamJoinType {
    /// Emit every pair with equal non-null keys and an event time inside the
    /// configured inclusive interval.
    Inner,
}

/// Inclusive event-time distance around one left row.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JoinTimeBounds {
    #[schemars(range(min = 0, max = 9_007_199_254_740_991_u64))]
    before_micros: u64,
    #[schemars(range(min = 0, max = 9_007_199_254_740_991_u64))]
    after_micros: u64,
}

impl JoinTimeBounds {
    /// Creates exact, non-negative microsecond bounds.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when either duration has
    /// sub-microsecond precision or exceeds the exact JSON integer domain.
    pub fn new(before: Duration, after: Duration) -> Result<Self> {
        Ok(Self {
            before_micros: exact_safe_duration_micros(before, "stream_join.bounds.before_micros")?,
            after_micros: exact_safe_duration_micros(after, "stream_join.bounds.after_micros")?,
        })
    }

    pub(crate) fn from_micros(before_micros: u64, after_micros: u64) -> Result<Self> {
        validate_safe_integer(before_micros, false, "stream_join.bounds.before_micros")?;
        validate_safe_integer(after_micros, false, "stream_join.bounds.after_micros")?;
        Ok(Self {
            before_micros,
            after_micros,
        })
    }

    /// Returns the exact preceding distance in microseconds.
    pub const fn before_micros(self) -> u64 {
        self.before_micros
    }

    /// Returns the exact following distance in microseconds.
    pub const fn after_micros(self) -> u64 {
        self.after_micros
    }

    /// Returns the preceding distance.
    pub const fn before(self) -> Duration {
        Duration::from_micros(self.before_micros)
    }

    /// Returns the following distance.
    pub const fn after(self) -> Duration {
        Duration::from_micros(self.after_micros)
    }

    pub(crate) fn contains_pair(self, left_micros: i64, right_micros: i64) -> bool {
        let left = i128::from(left_micros);
        let right = i128::from(right_micros);
        right >= left - i128::from(self.before_micros)
            && right <= left + i128::from(self.after_micros)
    }
}

impl<'de> Deserialize<'de> for JoinTimeBounds {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            before_micros: u64,
            after_micros: u64,
        }

        let fields = Fields::deserialize(deserializer)?;
        Self::from_micros(fields.before_micros, fields.after_micros).map_err(D::Error::custom)
    }
}

/// Hard logical state and per-input fan-out limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[allow(
    clippy::struct_field_names,
    reason = "the frozen public JSON field names all use the max_ limit prefix"
)]
#[serde(deny_unknown_fields)]
pub struct JoinStateLimits {
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_state_rows_per_side: u64,
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_state_bytes_per_side: u64,
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_matches_per_input_batch: u64,
}

impl JoinStateLimits {
    /// Creates required positive Join limits.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when a value is zero or is
    /// larger than [`STREAM_JOIN_MAX_SAFE_JSON_INTEGER`].
    pub fn new(
        max_state_rows_per_side: u64,
        max_state_bytes_per_side: u64,
        max_matches_per_input_batch: u64,
    ) -> Result<Self> {
        validate_safe_integer(
            max_state_rows_per_side,
            true,
            "stream_join.limits.max_state_rows_per_side",
        )?;
        validate_safe_integer(
            max_state_bytes_per_side,
            true,
            "stream_join.limits.max_state_bytes_per_side",
        )?;
        validate_safe_integer(
            max_matches_per_input_batch,
            true,
            "stream_join.limits.max_matches_per_input_batch",
        )?;
        Ok(Self {
            max_state_rows_per_side,
            max_state_bytes_per_side,
            max_matches_per_input_batch,
        })
    }

    /// Maximum retained rows on either side.
    pub const fn max_state_rows_per_side(self) -> u64 {
        self.max_state_rows_per_side
    }

    /// Maximum logical retained bytes on either side.
    pub const fn max_state_bytes_per_side(self) -> u64 {
        self.max_state_bytes_per_side
    }

    /// Maximum pairs one accepted input batch may emit.
    pub const fn max_matches_per_input_batch(self) -> u64 {
        self.max_matches_per_input_batch
    }
}

impl<'de> Deserialize<'de> for JoinStateLimits {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[allow(
            clippy::struct_field_names,
            reason = "the wire DTO must preserve the frozen max_ field names"
        )]
        #[serde(deny_unknown_fields)]
        struct Fields {
            max_state_rows_per_side: u64,
            max_state_bytes_per_side: u64,
            max_matches_per_input_batch: u64,
        }

        let fields = Fields::deserialize(deserializer)?;
        Self::new(
            fields.max_state_rows_per_side,
            fields.max_state_bytes_per_side,
            fields.max_matches_per_input_batch,
        )
        .map_err(D::Error::custom)
    }
}

/// Immutable declaration for a two-input bounded inner stream Join.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StreamJoinSpec {
    join_type: StreamJoinType,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_event_time: String,
    right_event_time: String,
    bounds: JoinTimeBounds,
    limits: JoinStateLimits,
    left_prefix: String,
    right_prefix: String,
}

impl StreamJoinSpec {
    /// Creates an inner Join with canonical `left` and `right` prefixes.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for empty, duplicate, or
    /// unequally sized key declarations and invalid event-time names.
    pub fn inner<L, R, LI, RI>(
        left_keys: L,
        right_keys: R,
        left_event_time: &str,
        right_event_time: &str,
        bounds: JoinTimeBounds,
        limits: JoinStateLimits,
    ) -> Result<Self>
    where
        L: IntoIterator<Item = LI>,
        R: IntoIterator<Item = RI>,
        LI: Into<String>,
        RI: Into<String>,
    {
        let left_keys = left_keys.into_iter().map(Into::into).collect::<Vec<_>>();
        let right_keys = right_keys.into_iter().map(Into::into).collect::<Vec<_>>();
        validate_key_names(&left_keys, &right_keys)?;
        validate_column_name(left_event_time, "stream_join.left_event_time")?;
        validate_column_name(right_event_time, "stream_join.right_event_time")?;
        Ok(Self {
            join_type: StreamJoinType::Inner,
            left_keys,
            right_keys,
            left_event_time: left_event_time.into(),
            right_event_time: right_event_time.into(),
            bounds,
            limits,
            left_prefix: "left".into(),
            right_prefix: "right".into(),
        })
    }

    /// Replaces both output prefixes without mutating the original value.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] unless both values are
    /// distinct portable identifiers.
    pub fn with_prefixes(mut self, left_prefix: &str, right_prefix: &str) -> Result<Self> {
        validate_prefixes(left_prefix, right_prefix)?;
        self.left_prefix = left_prefix.into();
        self.right_prefix = right_prefix.into();
        Ok(self)
    }

    pub const fn join_type(&self) -> StreamJoinType {
        self.join_type
    }

    pub fn left_keys(&self) -> &[String] {
        &self.left_keys
    }

    pub fn right_keys(&self) -> &[String] {
        &self.right_keys
    }

    pub fn left_event_time(&self) -> &str {
        &self.left_event_time
    }

    pub fn right_event_time(&self) -> &str {
        &self.right_event_time
    }

    pub const fn bounds(&self) -> JoinTimeBounds {
        self.bounds
    }

    pub const fn limits(&self) -> JoinStateLimits {
        self.limits
    }

    pub fn left_prefix(&self) -> &str {
        &self.left_prefix
    }

    pub fn right_prefix(&self) -> &str {
        &self.right_prefix
    }
}

impl<'de> Deserialize<'de> for StreamJoinSpec {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            join_type: StreamJoinType,
            left_keys: Vec<String>,
            right_keys: Vec<String>,
            left_event_time: String,
            right_event_time: String,
            bounds: JoinTimeBounds,
            limits: JoinStateLimits,
            #[serde(default = "default_left_prefix")]
            left_prefix: String,
            #[serde(default = "default_right_prefix")]
            right_prefix: String,
        }

        let fields = Fields::deserialize(deserializer)?;
        if fields.join_type != StreamJoinType::Inner {
            return Err(D::Error::custom("only inner stream joins are supported"));
        }
        Self::inner(
            fields.left_keys,
            fields.right_keys,
            &fields.left_event_time,
            &fields.right_event_time,
            fields.bounds,
            fields.limits,
        )
        .and_then(|spec| spec.with_prefixes(&fields.left_prefix, &fields.right_prefix))
        .map_err(D::Error::custom)
    }
}

/// Stateful two-input bounded event-time Join.
pub struct StreamJoinOperator {
    name: String,
    spec: StreamJoinSpec,
    input_ports: [Port; 2],
    output_ports: [Port; 1],
    compiled: CompiledJoin,
    runtime: StreamRuntimeState,
    state: StreamJoinState,
}

#[derive(Clone)]
struct CompiledJoin {
    left_key_indices: Vec<usize>,
    right_key_indices: Vec<usize>,
    left_event_time_index: usize,
    right_event_time_index: usize,
    equality_query: String,
}

/// Scratch-table alias holding the admitted rows of the current input batch.
const PROBE_TABLE: &str = "probe_input";
/// Scratch-table alias holding the opposite side's retained state rows.
const STATE_TABLE: &str = "state_input";
/// Renamed key column prefix shared by both scratch tables.
const KEY_COLUMN_PREFIX: &str = "__cf_join_key_";
/// Position of one admitted row inside [`PROBE_TABLE`].
const PROBE_POS_COLUMN: &str = "__cf_join_pos";
/// Retained-state row id inside [`STATE_TABLE`].
const STATE_RID_COLUMN: &str = "__cf_join_row_id";

/// One incoming row that passed null-key, null-event-time, and lateness admission.
struct AdmittedRow {
    record: RecordBatch,
    event_time: EventTime,
    row_id: u64,
    retain: bool,
}

/// Scratch accumulator for one input batch's admission pass.
struct AdmissionBundle {
    next_row_id: u64,
    metrics: SideMetrics,
    admitted: Vec<AdmittedRow>,
    had_late: bool,
}

/// Why an incoming physical row was dropped during admission.
#[derive(Clone, Copy)]
enum DropKind {
    NullEventTime,
    NullKey,
    Late(u64),
}

/// Classified disposition of one incoming physical row.
enum RowAdmission {
    Dropped(DropKind),
    Admitted(EventTime),
}

/// One time-qualified key-equal pair, ordered for emission.
struct MatchedPair {
    pos: usize,
    opposite_index: usize,
}

#[derive(Clone)]
struct StoredRow {
    record: RecordBatch,
    event_time: EventTime,
    row_id: u64,
    charge: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
struct SideMetrics {
    retained_rows: u64,
    retained_bytes: u64,
    evicted_rows: u64,
    late_rows: u64,
    late_affected_batches: u64,
    max_lateness_micros: Option<u64>,
    null_event_time_rows: u64,
    null_key_rows: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
struct JoinMetrics {
    left: SideMetrics,
    right: SideMetrics,
    emitted_match_rows: u64,
    state_limit_failures: u64,
    match_limit_failures: u64,
}

#[derive(Default)]
struct StreamJoinState {
    left: Vec<StoredRow>,
    right: Vec<StoredRow>,
    next_left_row_id: u64,
    next_right_row_id: u64,
    next_output_sequence: u64,
    metrics: JoinMetrics,
    ended: bool,
    last_checkpoint_epoch: Option<Epoch>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JoinCheckpointMetadata {
    layout_version: u32,
    spec: StreamJoinSpec,
    next_left_row_id: u64,
    next_right_row_id: u64,
    next_output_sequence: u64,
    metrics: JoinMetrics,
    ended: bool,
    epoch: u64,
}

struct PreparedJoinBatch {
    outputs: Vec<RecordBatch>,
    retained: Vec<StoredRow>,
    next_row_id: u64,
    metrics: SideMetrics,
}

impl StreamJoinOperator {
    /// Compiles one Join declaration against two exact Arrow schemas.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for declaration errors and
    /// [`CalcFlowError::Compile`] for incompatible schemas.
    pub fn new(
        name: &str,
        left_schema: SchemaRef,
        right_schema: SchemaRef,
        spec: StreamJoinSpec,
    ) -> Result<Self> {
        validate_operator_name(name)?;
        let (output_schema, compiled) = compile_schemas(&left_schema, &right_schema, &spec)?;
        Ok(Self {
            name: name.into(),
            spec,
            input_ports: [
                Port::with_schema_ref("left", BatchKind::Table, true, Some(left_schema))?,
                Port::with_schema_ref("right", BatchKind::Table, true, Some(right_schema))?,
            ],
            output_ports: [Port::with_schema_ref(
                "output",
                BatchKind::Table,
                true,
                Some(output_schema),
            )?],
            compiled,
            runtime: StreamRuntimeState::new(),
            state: StreamJoinState::default(),
        })
    }

    /// Returns the immutable Join declaration.
    pub const fn spec(&self) -> &StreamJoinSpec {
        &self.spec
    }

    pub(crate) fn set_stream_resources(
        &mut self,
        config: DataFusionConfig,
        udfs: UdfRegistrySnapshot,
    ) {
        self.runtime.set_resources(config, udfs, Vec::new());
    }

    pub(crate) const fn stream_runtime_initialized(&self) -> bool {
        self.runtime.is_initialized()
    }

    pub(crate) fn output_frontier_candidate(
        &self,
        progress: &crate::IngressProgressSnapshot,
    ) -> Result<Option<EventTime>> {
        let left = progress.get("left").ok_or_else(|| {
            operator_error(
                &self.name,
                "missing left ingress progress for output frontier",
            )
        })?;
        let right = progress.get("right").ok_or_else(|| {
            operator_error(
                &self.name,
                "missing right ingress progress for output frontier",
            )
        })?;
        let left_live = left.state() != crate::IngressState::Ended;
        let right_live = right.state() != crate::IngressState::Ended;
        let candidate = match (left_live, right_live) {
            (true, true) => match (left.watermark(), right.watermark()) {
                (Some(left), Some(right)) => Some(
                    (i128::from(left.as_micros()) - i128::from(self.spec.bounds.before_micros))
                        .min(
                            i128::from(right.as_micros())
                                - i128::from(self.spec.bounds.after_micros),
                        ),
                ),
                _ => None,
            },
            (true, false) => left.watermark().map(|left| {
                i128::from(left.as_micros()) - i128::from(self.spec.bounds.before_micros)
            }),
            (false, true) => right.watermark().map(|right| {
                i128::from(right.as_micros()) - i128::from(self.spec.bounds.after_micros)
            }),
            (false, false) => None,
        };
        candidate
            .filter(|candidate| *candidate >= i128::from(i64::MIN))
            .map(|candidate| {
                i64::try_from(candidate)
                    .map(EventTime::from_micros)
                    .map_err(|_| operator_error(&self.name, "output frontier exceeds EventTime"))
            })
            .transpose()
    }

    async fn prepare_batch(
        &mut self,
        ingress: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedJoinBatch> {
        let plan = self.begin_batch(ingress, batch)?;
        let mut bundle = self.admission_bundle(&plan);
        for record in batch.table_payload()?.batches() {
            self.admit_record(record, &plan, ingress, context, &mut bundle)?;
        }
        bundle.finish(&self.name)?;
        let outputs = self.evaluate_matches(&plan, &bundle.admitted).await?;
        self.finish_prepared(&plan, bundle, outputs)
    }

    fn finish_prepared(
        &mut self,
        plan: &SidePlan,
        bundle: AdmissionBundle,
        outputs: Vec<RecordBatch>,
    ) -> Result<PreparedJoinBatch> {
        let retained = retained_rows(&bundle.admitted, &plan.key_indices, &self.name)?;
        self.validate_state_admission(plan.incoming_is_left, &retained)?;
        Ok(PreparedJoinBatch {
            outputs,
            retained,
            next_row_id: bundle.next_row_id,
            metrics: bundle.metrics,
        })
    }

    /// Validates the ingress and port contract before any admission work.
    fn begin_batch(&self, ingress: &str, batch: &Batch) -> Result<SidePlan> {
        if self.state.ended {
            return Err(operator_error(
                &self.name,
                "received data after end-of-input",
            ));
        }
        let plan = self.side_plan(ingress)?;
        self.input_ports[plan.port_index].validate(batch, &format!("{}.{}", self.name, ingress))?;
        Ok(plan)
    }

    fn side_plan(&self, ingress: &str) -> Result<SidePlan> {
        match ingress {
            "left" => Ok(SidePlan {
                incoming_is_left: true,
                port_index: 0,
                event_time_index: self.compiled.left_event_time_index,
                key_indices: self.compiled.left_key_indices.clone(),
            }),
            "right" => Ok(SidePlan {
                incoming_is_left: false,
                port_index: 1,
                event_time_index: self.compiled.right_event_time_index,
                key_indices: self.compiled.right_key_indices.clone(),
            }),
            _ => Err(operator_error(
                &self.name,
                &format!("unknown ingress {ingress:?}; expected left or right"),
            )),
        }
    }

    fn admission_bundle(&self, plan: &SidePlan) -> AdmissionBundle {
        let (next_row_id, metrics) = if plan.incoming_is_left {
            (self.state.next_left_row_id, self.state.metrics.left.clone())
        } else {
            (
                self.state.next_right_row_id,
                self.state.metrics.right.clone(),
            )
        };
        AdmissionBundle {
            next_row_id,
            metrics,
            admitted: Vec::new(),
            had_late: false,
        }
    }

    fn admit_record(
        &self,
        record: &RecordBatch,
        plan: &SidePlan,
        ingress: &str,
        context: &StreamOperatorContext<'_>,
        bundle: &mut AdmissionBundle,
    ) -> Result<()> {
        let side_progress = context.ingress_progress().get(ingress);
        let opposite_progress = context.ingress_progress().get(if plan.incoming_is_left {
            "right"
        } else {
            "left"
        });
        for row_index in 0..record.num_rows() {
            let row_id = bundle.reserve_row_id(&self.name)?;
            match self.classify_row(record, plan, row_index, side_progress, ingress)? {
                RowAdmission::Dropped(kind) => bundle.note_dropped(kind, &self.name)?,
                RowAdmission::Admitted(event_time) => bundle.push_admitted(AdmittedRow {
                    record: record.slice(row_index, 1),
                    event_time,
                    row_id,
                    retain: should_retain(
                        plan.incoming_is_left,
                        event_time,
                        opposite_progress,
                        self.spec.bounds,
                    ),
                }),
            }
        }
        Ok(())
    }

    fn classify_row(
        &self,
        record: &RecordBatch,
        plan: &SidePlan,
        row_index: usize,
        side_progress: Option<IngressProgress>,
        ingress: &str,
    ) -> Result<RowAdmission> {
        let Some(event_time) = event_time_at(
            record,
            plan.event_time_index,
            row_index,
            &self.name,
            ingress,
        )?
        else {
            return Ok(RowAdmission::Dropped(DropKind::NullEventTime));
        };
        if plan
            .key_indices
            .iter()
            .any(|&index| record.column(index).is_null(row_index))
        {
            return Ok(RowAdmission::Dropped(DropKind::NullKey));
        }
        match late_lateness(event_time, side_progress, &self.name)? {
            Some(lateness) => Ok(RowAdmission::Dropped(DropKind::Late(lateness))),
            None => Ok(RowAdmission::Admitted(event_time)),
        }
    }

    /// Runs the batched key-equality probe and emits one output row per pair.
    ///
    /// Key equality executes as one `DataFusion` join over scratch tables per
    /// input batch; the time bound stays in checked `i128` Rust arithmetic.
    async fn evaluate_matches(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
    ) -> Result<Vec<RecordBatch>> {
        let runtime = self.runtime.runtime()?;
        let opposite = if plan.incoming_is_left {
            self.state.right.as_slice()
        } else {
            self.state.left.as_slice()
        };
        if admitted.is_empty() || opposite.is_empty() {
            return Ok(Vec::new());
        }
        let matched = matched_pairs(
            runtime,
            &self.compiled,
            &self.spec,
            plan,
            admitted,
            opposite,
            &self.name,
        )
        .await?;
        enforce_match_limit(
            matched.len(),
            &mut self.state.metrics.match_limit_failures,
            self.spec.limits.max_matches_per_input_batch,
            &self.name,
        )?;
        let output_schema = self.output_ports[0]
            .schema()
            .expect("stream Join output always has an exact schema");
        materialize_outputs(
            output_schema,
            admitted,
            opposite,
            &matched,
            plan.incoming_is_left,
            &self.name,
        )
    }

    fn validate_state_admission(
        &mut self,
        incoming_is_left: bool,
        retained: &[StoredRow],
    ) -> Result<()> {
        let current = if incoming_is_left {
            &self.state.left
        } else {
            &self.state.right
        };
        let (rows, bytes) = prospective_state_charge(current, retained, &self.name)?;
        if rows > self.spec.limits.max_state_rows_per_side
            || bytes > self.spec.limits.max_state_bytes_per_side
        {
            self.state.metrics.state_limit_failures = checked_metric(
                self.state.metrics.state_limit_failures,
                1,
                &self.name,
                "state_limit_failures",
            )?;
            return Err(operator_reason(
                &self.name,
                crate::StreamingFailureReason::JoinStateLimitExceeded,
                "retained state limit exceeded",
            ));
        }
        Ok(())
    }

    fn validate_restored_limits(&self, left: &[StoredRow], right: &[StoredRow]) -> Result<()> {
        for (side, rows) in [("left", left), ("right", right)] {
            let row_count =
                u64::try_from(rows.len()).map_err(|_| CalcFlowError::CheckpointMismatch {
                    message: format!("stream Join {:?} {side} row count is too large", self.name),
                })?;
            let byte_count = rows
                .iter()
                .try_fold(0_u64, |total, row| total.checked_add(row.charge))
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: format!("stream Join {:?} {side} byte charge overflowed", self.name),
                })?;
            if row_count > self.spec.limits.max_state_rows_per_side
                || byte_count > self.spec.limits.max_state_bytes_per_side
            {
                return Err(CalcFlowError::CheckpointMismatch {
                    message: format!(
                        "stream Join {:?} restored {side} state exceeds configured limits",
                        self.name
                    ),
                });
            }
        }
        Ok(())
    }

    async fn emit_prepared(
        &mut self,
        prepared: &PreparedJoinBatch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let sequence_count = u64::try_from(prepared.outputs.len())
            .map_err(|_| counter_overflow(&self.name, "output sequence"))?;
        self.state
            .next_output_sequence
            .checked_add(sequence_count)
            .ok_or_else(|| counter_overflow(&self.name, "output sequence"))?;
        for record in &prepared.outputs {
            let sequence = self.state.next_output_sequence;
            emit_output_row(record, sequence, context, output).await?;
            self.state.next_output_sequence += 1;
        }
        Ok(())
    }

    fn commit_prepared(&mut self, ingress: &str, prepared: PreparedJoinBatch) -> Result<()> {
        let metrics = prepared.metrics;
        if ingress == "left" {
            self.state.next_left_row_id = prepared.next_row_id;
            self.state.left.extend(prepared.retained);
            self.state.metrics.left = metrics;
            refresh_retained_metrics(&mut self.state.metrics.left, &self.state.left, &self.name)?;
        } else {
            self.state.next_right_row_id = prepared.next_row_id;
            self.state.right.extend(prepared.retained);
            self.state.metrics.right = metrics;
            refresh_retained_metrics(&mut self.state.metrics.right, &self.state.right, &self.name)?;
        }
        Ok(())
    }

    fn decode_restored_sides(
        &self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<(Vec<StoredRow>, Vec<StoredRow>)> {
        if snapshot
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != ["left-v1", "right-v1"]
        {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!("stream Join {:?} segment inventory is invalid", self.name),
            });
        }
        let left = decode_side(
            &snapshot.segments["left-v1"],
            self.input_schema(0),
            &self.name,
            "left",
        )?;
        let right = decode_side(
            &snapshot.segments["right-v1"],
            self.input_schema(1),
            &self.name,
            "right",
        )?;
        Ok((left, right))
    }

    fn input_schema(&self, port_index: usize) -> &SchemaRef {
        self.input_ports[port_index]
            .schema()
            .expect("stream Join inputs always have an exact schema")
    }

    fn validate_restored_join_rows(
        &self,
        metadata: &JoinCheckpointMetadata,
        left: &[StoredRow],
        right: &[StoredRow],
    ) -> Result<()> {
        validate_restored_rows(
            left,
            metadata.next_left_row_id,
            self.compiled.left_event_time_index,
            &self.compiled.left_key_indices,
            &self.name,
            "left",
        )?;
        validate_restored_rows(
            right,
            metadata.next_right_row_id,
            self.compiled.right_event_time_index,
            &self.compiled.right_key_indices,
            &self.name,
            "right",
        )
    }
}

impl fmt::Debug for StreamJoinOperator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamJoinOperator")
            .field("name", &self.name)
            .field("spec", &self.spec)
            .field("input_ports", &self.input_ports)
            .field("output_ports", &self.output_ports)
            .finish_non_exhaustive()
    }
}

impl OperatorMetadata for StreamJoinOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
        let value = serde_json::to_value(&self.spec)
            .expect("validated stream Join configuration remains serializable");
        let Value::Object(values) = value else {
            unreachable!("stream Join configuration serializes as an object")
        };
        values.into_iter().collect()
    }
}

#[async_trait]
impl StreamOperator for StreamJoinOperator {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        let prepared = self.prepare_batch(ingress, &batch, context).await?;
        self.emit_prepared(&prepared, context, output).await?;
        let emitted = u64::try_from(prepared.outputs.len())
            .map_err(|_| counter_overflow(&self.name, "emitted rows"))?;
        self.state.metrics.emitted_match_rows = checked_metric(
            self.state.metrics.emitted_match_rows,
            emitted,
            &self.name,
            "emitted_match_rows",
        )?;
        self.commit_prepared(ingress, prepared)?;
        Ok(())
    }

    async fn on_ingress_progress(
        &mut self,
        ingress: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let progress = context.ingress_progress().get(ingress).ok_or_else(|| {
            operator_error(
                &self.name,
                &format!("missing progress for ingress {ingress:?}"),
            )
        })?;
        match ingress {
            "left" => {
                evict_opposite(
                    &mut self.state.right,
                    progress,
                    self.spec.bounds.before_micros,
                    &mut self.state.metrics.right,
                    &self.name,
                )?;
            }
            "right" => {
                evict_opposite(
                    &mut self.state.left,
                    progress,
                    self.spec.bounds.after_micros,
                    &mut self.state.metrics.left,
                    &self.name,
                )?;
            }
            _ => {
                return Err(operator_error(
                    &self.name,
                    &format!("unknown ingress progress {ingress:?}"),
                ));
            }
        }
        Ok(())
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.state.left.clear();
        self.state.right.clear();
        self.state.metrics.left.retained_rows = 0;
        self.state.metrics.left.retained_bytes = 0;
        self.state.metrics.right.retained_rows = 0;
        self.state.metrics.right.retained_bytes = 0;
        self.state.ended = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state = StreamJoinState::default();
        Ok(())
    }

    fn checkpoint(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        if self
            .state
            .last_checkpoint_epoch
            .is_some_and(|previous| epoch <= previous)
        {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "stream Join {:?} checkpoint epoch did not advance strictly",
                    self.name
                ),
            });
        }
        let metadata = JoinCheckpointMetadata {
            layout_version: 1,
            spec: self.spec.clone(),
            next_left_row_id: self.state.next_left_row_id,
            next_right_row_id: self.state.next_right_row_id,
            next_output_sequence: self.state.next_output_sequence,
            metrics: self.state.metrics.clone(),
            ended: self.state.ended,
            epoch: epoch.as_u64(),
        };
        let Value::Object(inline_metadata) =
            serde_json::to_value(metadata).map_err(|error| CalcFlowError::Internal {
                message: format!("stream Join checkpoint metadata encoding failed: {error}"),
            })?
        else {
            unreachable!("stream Join checkpoint metadata is an object")
        };
        let segments = BTreeMap::from([
            (
                "left-v1".into(),
                encode_side(&self.state.left, &self.name, "left")?,
            ),
            (
                "right-v1".into(),
                encode_side(&self.state.right, &self.name, "right")?,
            ),
        ]);
        self.state.last_checkpoint_epoch = Some(epoch);
        Ok(OperatorStateSnapshot {
            inline_metadata: inline_metadata.into_iter().collect(),
            segments,
        })
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let metadata = decode_join_metadata(snapshot, &self.name)?;
        if !checkpoint_metadata_compatible(&metadata, &self.spec) {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "stream Join {:?} checkpoint layout or specification is incompatible",
                    self.name
                ),
            });
        }
        let (left, right) = self.decode_restored_sides(snapshot)?;
        self.validate_restored_join_rows(&metadata, &left, &right)?;
        restored_retained_metrics_match(&metadata.metrics, &left, &right, &self.name)?;
        self.validate_restored_limits(&left, &right)?;
        self.state = StreamJoinState {
            left,
            right,
            next_left_row_id: metadata.next_left_row_id,
            next_right_row_id: metadata.next_right_row_id,
            next_output_sequence: metadata.next_output_sequence,
            metrics: metadata.metrics,
            ended: metadata.ended,
            last_checkpoint_epoch: Epoch::new(metadata.epoch),
        };
        Ok(())
    }
}

/// Compile-time per-ingress lookup for one input batch.
struct SidePlan {
    incoming_is_left: bool,
    port_index: usize,
    event_time_index: usize,
    key_indices: Vec<usize>,
}

impl AdmissionBundle {
    fn reserve_row_id(&mut self, operator_id: &str) -> Result<u64> {
        let row_id = self.next_row_id;
        self.next_row_id = self
            .next_row_id
            .checked_add(1)
            .ok_or_else(|| counter_overflow(operator_id, "row_id"))?;
        Ok(row_id)
    }

    fn note_dropped(&mut self, kind: DropKind, operator_id: &str) -> Result<()> {
        match kind {
            DropKind::NullEventTime => {
                self.metrics.null_event_time_rows = checked_metric(
                    self.metrics.null_event_time_rows,
                    1,
                    operator_id,
                    "null_event_time_rows",
                )?;
            }
            DropKind::NullKey => {
                self.metrics.null_key_rows =
                    checked_metric(self.metrics.null_key_rows, 1, operator_id, "null_key_rows")?;
            }
            DropKind::Late(lateness) => {
                self.metrics.late_rows =
                    checked_metric(self.metrics.late_rows, 1, operator_id, "late_rows")?;
                self.metrics.max_lateness_micros = Some(
                    self.metrics
                        .max_lateness_micros
                        .map_or(lateness, |current| current.max(lateness)),
                );
                self.had_late = true;
            }
        }
        Ok(())
    }

    fn push_admitted(&mut self, row: AdmittedRow) {
        self.admitted.push(row);
    }

    fn finish(&mut self, operator_id: &str) -> Result<()> {
        if self.had_late {
            self.metrics.late_affected_batches = checked_metric(
                self.metrics.late_affected_batches,
                1,
                operator_id,
                "late_affected_batches",
            )?;
        }
        Ok(())
    }
}

fn checkpoint_metadata_compatible(
    metadata: &JoinCheckpointMetadata,
    spec: &StreamJoinSpec,
) -> bool {
    metadata.layout_version == 1 && metadata.spec == *spec
}

fn decode_join_metadata(
    snapshot: &OperatorStateSnapshot,
    operator_id: &str,
) -> Result<JoinCheckpointMetadata> {
    serde_json::from_value::<JoinCheckpointMetadata>(Value::Object(
        snapshot.inline_metadata.clone().into_iter().collect(),
    ))
    .map_err(|error| CalcFlowError::CheckpointMismatch {
        message: format!("stream Join {operator_id:?} metadata is invalid: {error}"),
    })
}

/// Recomputes retained charges and rejects checkpoints that disagree with them.
fn restored_retained_metrics_match(
    metrics: &JoinMetrics,
    left: &[StoredRow],
    right: &[StoredRow],
    operator_id: &str,
) -> Result<()> {
    let mut left_metrics = metrics.left.clone();
    let mut right_metrics = metrics.right.clone();
    refresh_retained_metrics(&mut left_metrics, left, operator_id)?;
    refresh_retained_metrics(&mut right_metrics, right, operator_id)?;
    if side_retained_matches(&metrics.left, &left_metrics)
        && side_retained_matches(&metrics.right, &right_metrics)
    {
        return Ok(());
    }
    Err(CalcFlowError::CheckpointMismatch {
        message: format!("stream Join {operator_id:?} restored state charge is inconsistent"),
    })
}

fn side_retained_matches(recorded: &SideMetrics, recomputed: &SideMetrics) -> bool {
    recorded.retained_rows == recomputed.retained_rows
        && recorded.retained_bytes == recomputed.retained_bytes
}

/// Prospective (rows, bytes) charge if `retained` were installed next to `current`.
fn prospective_state_charge(
    current: &[StoredRow],
    retained: &[StoredRow],
    operator_id: &str,
) -> Result<(u64, u64)> {
    let rows = state_row_count(current, operator_id)?
        .checked_add(state_row_count(retained, operator_id)?)
        .ok_or_else(|| counter_overflow(operator_id, "state rows"))?;
    let bytes = current
        .iter()
        .chain(retained)
        .try_fold(0_u64, |total, row| total.checked_add(row.charge))
        .ok_or_else(|| counter_overflow(operator_id, "state bytes"))?;
    Ok((rows, bytes))
}

fn state_row_count(rows: &[StoredRow], operator_id: &str) -> Result<u64> {
    u64::try_from(rows.len()).map_err(|_| counter_overflow(operator_id, "state rows"))
}

/// Emits one prepared output row under the effective edge byte budget.
async fn emit_output_row(
    record: &RecordBatch,
    sequence: u64,
    context: &StreamOperatorContext<'_>,
    output: &mut dyn StreamCollector,
) -> Result<()> {
    let metadata = BatchMetadata::new(context.operator_id(), sequence, BTreeMap::new())?;
    let batch = Batch::table(vec![record.clone()], metadata)?;
    if batch.estimated_bytes()? > context.output_budget().max_bytes {
        return Err(CalcFlowError::InvalidArgument {
            field: "message.bytes".into(),
            message: "one stream Join output row exceeds the effective edge byte budget".into(),
        });
    }
    output.emit("output", batch).await
}

fn late_lateness(
    event_time: EventTime,
    progress: Option<IngressProgress>,
    operator_id: &str,
) -> Result<Option<u64>> {
    let Some(watermark) = progress.and_then(IngressProgress::watermark) else {
        return Ok(None);
    };
    if event_time >= watermark {
        return Ok(None);
    }
    let lateness =
        u64::try_from(i128::from(watermark.as_micros()) - i128::from(event_time.as_micros()))
            .map_err(|_| counter_overflow(operator_id, "lateness"))?;
    Ok(Some(lateness))
}

/// Runs the batched key-equality probe and returns the time-qualified pairs
/// in emission order.
async fn matched_pairs(
    runtime: &DataFusionRuntime,
    compiled: &CompiledJoin,
    spec: &StreamJoinSpec,
    plan: &SidePlan,
    admitted: &[AdmittedRow],
    opposite: &[StoredRow],
    operator_id: &str,
) -> Result<Vec<MatchedPair>> {
    let probe = probe_key_batch(admitted, &plan.key_indices)?;
    let state_keys = state_key_batch(opposite, compiled, plan)?;
    let tables = equality_tables(probe, state_keys)?;
    let result = runtime
        .sql(&compiled.equality_query, &tables, Some(operator_id))
        .await?;
    let equal_pairs = decode_key_pairs(&result)?;
    Ok(filter_and_order_pairs(
        &spec.bounds,
        plan,
        admitted,
        opposite,
        equal_pairs,
    ))
}

fn retained_rows(
    admitted: &[AdmittedRow],
    key_indices: &[usize],
    operator_id: &str,
) -> Result<Vec<StoredRow>> {
    admitted
        .iter()
        .filter(|row| row.retain)
        .map(|row| {
            let charge = state_row_charge(&row.record, 0, key_indices, operator_id)?;
            Ok(StoredRow {
                record: row.record.clone(),
                event_time: row.event_time,
                row_id: row.row_id,
                charge,
            })
        })
        .collect()
}

fn enforce_match_limit(
    count: usize,
    failures: &mut u64,
    limit: u64,
    operator_id: &str,
) -> Result<()> {
    let count = u64::try_from(count).map_err(|_| counter_overflow(operator_id, "match_count"))?;
    if count > limit {
        *failures = checked_metric(*failures, 1, operator_id, "match_limit_failures")?;
        return Err(operator_reason(
            operator_id,
            crate::StreamingFailureReason::JoinMatchLimitExceeded,
            "input batch match limit exceeded",
        ));
    }
    Ok(())
}

/// Builds the admitted-row scratch table with renamed key columns.
fn probe_key_batch(admitted: &[AdmittedRow], key_indices: &[usize]) -> Result<RecordBatch> {
    let records = admitted.iter().map(|row| &row.record).collect::<Vec<_>>();
    let positions = UInt64Array::from_iter_values(
        (0..admitted.len()).map(|index| u64::try_from(index).expect("row count fits u64")),
    );
    key_probe_batch(&records, key_indices, PROBE_POS_COLUMN, &positions)
}

/// Builds the retained-state scratch table with renamed key columns and row ids.
fn state_key_batch(
    opposite: &[StoredRow],
    compiled: &CompiledJoin,
    plan: &SidePlan,
) -> Result<RecordBatch> {
    let key_indices = if plan.incoming_is_left {
        &compiled.right_key_indices
    } else {
        &compiled.left_key_indices
    };
    let records = opposite.iter().map(|row| &row.record).collect::<Vec<_>>();
    let row_ids = UInt64Array::from_iter_values(opposite.iter().map(|row| row.row_id));
    key_probe_batch(&records, key_indices, STATE_RID_COLUMN, &row_ids)
}

fn key_probe_batch(
    records: &[&RecordBatch],
    key_indices: &[usize],
    extra_name: &str,
    extra: &UInt64Array,
) -> Result<RecordBatch> {
    let first = records
        .first()
        .expect("join probe batches always have at least one row");
    let source_schema = first.schema();
    let mut fields = Vec::with_capacity(key_indices.len() + 1);
    let mut columns = Vec::with_capacity(key_indices.len() + 1);
    for (position, &key_index) in key_indices.iter().enumerate() {
        let source = source_schema.field(key_index);
        fields.push(Field::new(
            format!("{KEY_COLUMN_PREFIX}{position}"),
            source.data_type().clone(),
            source.is_nullable(),
        ));
        let slices = records
            .iter()
            .map(|record| record.column(key_index).as_ref())
            .collect::<Vec<_>>();
        columns.push(concat_key_column(&slices)?);
    }
    fields.push(Field::new(extra_name, DataType::UInt64, false));
    columns.push(Arc::new(extra.clone()));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|error| {
        CalcFlowError::Internal {
            message: format!("stream Join equality probe assembly failed: {error}"),
        }
    })
}

fn concat_key_column(slices: &[&dyn Array]) -> Result<ArrayRef> {
    concat(slices).map_err(|error| CalcFlowError::Internal {
        message: format!("stream Join key column concatenation failed: {error}"),
    })
}

fn equality_tables(probe: RecordBatch, state_keys: RecordBatch) -> Result<BTreeMap<String, Batch>> {
    Ok(BTreeMap::from([
        (
            PROBE_TABLE.into(),
            Batch::table(vec![probe], BatchMetadata::default())?,
        ),
        (
            STATE_TABLE.into(),
            Batch::table(vec![state_keys], BatchMetadata::default())?,
        ),
    ]))
}

fn decode_key_pairs(result: &Batch) -> Result<Vec<(u64, u64)>> {
    let mut pairs = Vec::new();
    for record in result.table_payload()?.batches() {
        let positions = u64_column(record, 0, "probe position")?;
        let row_ids = u64_column(record, 1, "state row id")?;
        for row_index in 0..record.num_rows() {
            pairs.push((positions.value(row_index), row_ids.value(row_index)));
        }
    }
    Ok(pairs)
}

fn u64_column<'a>(
    record: &'a RecordBatch,
    column_index: usize,
    field: &str,
) -> Result<&'a UInt64Array> {
    record
        .column(column_index)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("stream Join equality result is missing the {field} column"),
        })
}

fn filter_and_order_pairs(
    bounds: &JoinTimeBounds,
    plan: &SidePlan,
    admitted: &[AdmittedRow],
    opposite: &[StoredRow],
    equal_pairs: Vec<(u64, u64)>,
) -> Vec<MatchedPair> {
    let row_id_index = index_by_row_id(opposite);
    let mut matched = equal_pairs
        .into_iter()
        .filter_map(|(pos, rid)| {
            let pos = usize::try_from(pos).expect("probe positions index admitted rows");
            let opposite_index = *row_id_index
                .get(&rid)
                .expect("state row ids index retained rows");
            let incoming = &admitted[pos];
            let candidate = &opposite[opposite_index];
            let in_bounds = if plan.incoming_is_left {
                bounds.contains_pair(
                    incoming.event_time.as_micros(),
                    candidate.event_time.as_micros(),
                )
            } else {
                bounds.contains_pair(
                    candidate.event_time.as_micros(),
                    incoming.event_time.as_micros(),
                )
            };
            in_bounds.then_some(MatchedPair {
                pos,
                opposite_index,
            })
        })
        .collect::<Vec<_>>();
    matched.sort_by_key(|pair| {
        let row = &opposite[pair.opposite_index];
        (pair.pos, row.event_time, row.row_id)
    });
    matched
}

fn index_by_row_id(opposite: &[StoredRow]) -> BTreeMap<u64, usize> {
    opposite
        .iter()
        .enumerate()
        .map(|(index, row)| (row.row_id, index))
        .collect()
}

fn materialize_outputs(
    output_schema: &SchemaRef,
    admitted: &[AdmittedRow],
    opposite: &[StoredRow],
    matched: &[MatchedPair],
    incoming_is_left: bool,
    operator_id: &str,
) -> Result<Vec<RecordBatch>> {
    matched
        .iter()
        .map(|pair| {
            let incoming = &admitted[pair.pos];
            let candidate = &opposite[pair.opposite_index];
            let (left, right) = if incoming_is_left {
                (&incoming.record, &candidate.record)
            } else {
                (&candidate.record, &incoming.record)
            };
            output_record(output_schema, left, right, operator_id)
        })
        .collect()
}

fn output_record(
    output_schema: &SchemaRef,
    left: &RecordBatch,
    right: &RecordBatch,
    operator_id: &str,
) -> Result<RecordBatch> {
    let columns = left
        .columns()
        .iter()
        .chain(right.columns())
        .cloned()
        .collect::<Vec<_>>();
    RecordBatch::try_new(Arc::clone(output_schema), columns)
        .map_err(|error| operator_error(operator_id, &format!("output projection failed: {error}")))
}

fn exact_safe_duration_micros(duration: Duration, field: &str) -> Result<u64> {
    if duration.subsec_nanos() % 1_000 != 0 {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "must be an exact multiple of one microsecond".into(),
        });
    }
    let micros =
        u64::try_from(duration.as_micros()).map_err(|_| CalcFlowError::InvalidArgument {
            field: field.into(),
            message: format!("must be at most {STREAM_JOIN_MAX_SAFE_JSON_INTEGER}"),
        })?;
    validate_safe_integer(micros, false, field)?;
    Ok(micros)
}

fn validate_safe_integer(value: u64, positive: bool, field: &str) -> Result<()> {
    if (positive && value == 0) || value > STREAM_JOIN_MAX_SAFE_JSON_INTEGER {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: if positive {
                format!("must be in 1..={STREAM_JOIN_MAX_SAFE_JSON_INTEGER}")
            } else {
                format!("must be in 0..={STREAM_JOIN_MAX_SAFE_JSON_INTEGER}")
            },
        });
    }
    Ok(())
}

fn validate_key_names(left: &[String], right: &[String]) -> Result<()> {
    if left.is_empty() || left.len() != right.len() {
        return Err(CalcFlowError::InvalidArgument {
            field: "stream_join.keys".into(),
            message: "left_keys and right_keys must be non-empty and equally sized".into(),
        });
    }
    for (side, keys) in [("left", left), ("right", right)] {
        if keys.iter().any(String::is_empty)
            || keys.iter().collect::<BTreeSet<_>>().len() != keys.len()
        {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("stream_join.{side}_keys"),
                message: "must contain unique non-empty column names".into(),
            });
        }
    }
    Ok(())
}

fn validate_column_name(value: &str, field: &str) -> Result<()> {
    if value.is_empty() {
        Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "must name one column".into(),
        })
    } else {
        Ok(())
    }
}

fn validate_prefixes(left: &str, right: &str) -> Result<()> {
    if !is_identifier(left) || !is_identifier(right) || left == right {
        return Err(CalcFlowError::InvalidArgument {
            field: "stream_join.prefixes".into(),
            message: "must be distinct non-empty portable identifiers".into(),
        });
    }
    Ok(())
}

fn compile_schemas(
    left: &Schema,
    right: &Schema,
    spec: &StreamJoinSpec,
) -> Result<(SchemaRef, CompiledJoin)> {
    validate_unique_fields(left, "left")?;
    validate_unique_fields(right, "right")?;
    let (left_key_indices, right_key_indices) = compile_key_pair_indices(left, right, spec)?;
    let left_event_time_index = event_time_index(left, &spec.left_event_time, "left_event_time")?;
    let right_event_time_index =
        event_time_index(right, &spec.right_event_time, "right_event_time")?;
    let fields = prefixed_output_fields(left, right, spec);
    Ok((
        Arc::new(Schema::new(fields)),
        CompiledJoin {
            left_key_indices,
            right_key_indices,
            left_event_time_index,
            right_event_time_index,
            equality_query: equality_query(spec.left_keys.len()),
        },
    ))
}

fn compile_key_pair_indices(
    left: &Schema,
    right: &Schema,
    spec: &StreamJoinSpec,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut left_key_indices = Vec::with_capacity(spec.left_keys.len());
    let mut right_key_indices = Vec::with_capacity(spec.right_keys.len());
    for (index, (left_key, right_key)) in spec.left_keys.iter().zip(&spec.right_keys).enumerate() {
        let left_field = field_by_name(left, left_key, "left_keys", index)?;
        let right_field = field_by_name(right, right_key, "right_keys", index)?;
        validate_key_pair_types(index, left_field, right_field)?;
        left_key_indices.push(
            left.index_of(left_key)
                .expect("field lookup succeeded above"),
        );
        right_key_indices.push(
            right
                .index_of(right_key)
                .expect("field lookup succeeded above"),
        );
    }
    Ok((left_key_indices, right_key_indices))
}

fn validate_key_pair_types(index: usize, left_field: &Field, right_field: &Field) -> Result<()> {
    if left_field.data_type() != right_field.data_type()
        || !supported_key_type(left_field.data_type())
    {
        return Err(CalcFlowError::Compile {
            message: format!(
                "stream Join key pair {index} requires identical supported Arrow types; left is {} and right is {}",
                left_field.data_type(),
                right_field.data_type()
            ),
        });
    }
    Ok(())
}

fn event_time_index(schema: &Schema, name: &str, field: &str) -> Result<usize> {
    validate_event_time(schema, name, field)?;
    Ok(schema
        .index_of(name)
        .expect("event-time lookup succeeded above"))
}

fn prefixed_output_fields(left: &Schema, right: &Schema, spec: &StreamJoinSpec) -> Vec<Arc<Field>> {
    left.fields()
        .iter()
        .map(|field| {
            Arc::new(field.as_ref().clone().with_name(format!(
                "{}__{}",
                spec.left_prefix,
                field.name()
            )))
        })
        .chain(right.fields().iter().map(|field| {
            Arc::new(field.as_ref().clone().with_name(format!(
                "{}__{}",
                spec.right_prefix,
                field.name()
            )))
        }))
        .collect()
}

/// One batched key-equality probe over the per-batch scratch tables.
///
/// The query returns the admitted-row position and the retained-state row id
/// for every key-equal pair; the closed time bound is applied afterwards in
/// checked `i128` Rust arithmetic.
fn equality_query(key_count: usize) -> String {
    let equality = (0..key_count)
        .map(|position| {
            let column = quote_identifier(&format!("{KEY_COLUMN_PREFIX}{position}"));
            format!("{PROBE_TABLE}.{column} = {STATE_TABLE}.{column}")
        })
        .collect::<Vec<_>>()
        .join(" AND ");
    format!(
        "SELECT {PROBE_TABLE}.{pos}, {STATE_TABLE}.{rid} FROM {PROBE_TABLE} INNER JOIN {STATE_TABLE} ON {equality}",
        pos = quote_identifier(PROBE_POS_COLUMN),
        rid = quote_identifier(STATE_RID_COLUMN),
    )
}

fn quote_identifier(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn validate_unique_fields(schema: &Schema, side: &str) -> Result<()> {
    let mut names = BTreeSet::new();
    if schema.fields().is_empty()
        || schema
            .fields()
            .iter()
            .any(|field| !names.insert(field.name()))
    {
        return Err(CalcFlowError::Compile {
            message: format!("stream Join {side} schema must be non-empty with unique field names"),
        });
    }
    Ok(())
}

fn field_by_name<'a>(
    schema: &'a Schema,
    name: &str,
    field: &str,
    index: usize,
) -> Result<&'a Field> {
    schema
        .field_with_name(name)
        .map_err(|_| CalcFlowError::Compile {
            message: format!("stream_join.{field}[{index}] names missing column {name:?}"),
        })
}

fn validate_event_time(schema: &Schema, name: &str, field: &str) -> Result<()> {
    let column = schema
        .field_with_name(name)
        .map_err(|_| CalcFlowError::Compile {
            message: format!("stream_join.{field} names missing column {name:?}"),
        })?;
    let DataType::Timestamp(_, timezone) = column.data_type() else {
        return Err(CalcFlowError::Compile {
            message: format!("stream_join.{field} must be an Arrow timestamp"),
        });
    };
    if timezone
        .as_deref()
        .is_some_and(|timezone| timezone != "UTC")
    {
        return Err(CalcFlowError::Compile {
            message: format!("stream_join.{field} timestamp timezone must be UTC or absent"),
        });
    }
    Ok(())
}

fn supported_key_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
    )
}

const JOIN_STATE_MAGIC: &[u8; 8] = b"CFJOIN1\0";

fn encode_side(rows: &[StoredRow], operator_id: &str, side: &str) -> Result<Vec<u8>> {
    let mut ordered = rows.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|row| (row.event_time, row.row_id));
    let mut output = Vec::new();
    output.extend_from_slice(JOIN_STATE_MAGIC);
    output.extend_from_slice(
        &u64::try_from(ordered.len())
            .map_err(|_| counter_overflow(operator_id, "checkpoint rows"))?
            .to_le_bytes(),
    );
    for row in ordered {
        let mut ipc = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut ipc, row.record.schema().as_ref())
                .map_err(|error| CalcFlowError::Internal {
                    message: format!(
                        "stream Join {operator_id:?} {side} IPC writer failed: {error}"
                    ),
                })?;
            writer
                .write(&row.record)
                .and_then(|()| writer.finish())
                .map_err(|error| CalcFlowError::Internal {
                    message: format!(
                        "stream Join {operator_id:?} {side} IPC encoding failed: {error}"
                    ),
                })?;
        }
        output.extend_from_slice(&row.row_id.to_le_bytes());
        output.extend_from_slice(&row.event_time.as_micros().to_le_bytes());
        output.extend_from_slice(&row.charge.to_le_bytes());
        output.extend_from_slice(
            &u64::try_from(ipc.len())
                .map_err(|_| counter_overflow(operator_id, "IPC length"))?
                .to_le_bytes(),
        );
        output.extend_from_slice(&ipc);
    }
    Ok(output)
}

fn decode_side(
    bytes: &[u8],
    expected_schema: &SchemaRef,
    operator_id: &str,
    side: &str,
) -> Result<Vec<StoredRow>> {
    let mut offset = 0_usize;
    let row_count = decode_side_header(bytes, &mut offset, operator_id, side)?;
    let rows = decode_side_rows(
        bytes,
        &mut offset,
        row_count,
        expected_schema,
        operator_id,
        side,
    )?;
    if offset != bytes.len() {
        return Err(checkpoint_error(
            operator_id,
            side,
            "state segment has trailing bytes",
        ));
    }
    Ok(rows)
}

/// Validates the state magic and returns the declared row count.
fn decode_side_header(
    bytes: &[u8],
    offset: &mut usize,
    operator_id: &str,
    side: &str,
) -> Result<u64> {
    if take_segment_bytes(bytes, offset, JOIN_STATE_MAGIC.len())? != JOIN_STATE_MAGIC {
        return Err(checkpoint_error(
            operator_id,
            side,
            "state magic is invalid",
        ));
    }
    read_segment_u64(bytes, offset)
}

fn decode_side_rows(
    bytes: &[u8],
    offset: &mut usize,
    row_count: u64,
    expected_schema: &SchemaRef,
    operator_id: &str,
    side: &str,
) -> Result<Vec<StoredRow>> {
    let row_capacity = decode_row_capacity(row_count, bytes.len(), operator_id, side)?;
    let mut rows = Vec::with_capacity(row_capacity);
    for _ in 0..row_count {
        rows.push(decode_stored_row(
            bytes,
            offset,
            expected_schema,
            operator_id,
            side,
        )?);
    }
    Ok(rows)
}

fn decode_row_capacity(
    row_count: u64,
    segment_len: usize,
    operator_id: &str,
    side: &str,
) -> Result<usize> {
    usize::try_from(row_count)
        .ok()
        .filter(|count| *count <= segment_len)
        .ok_or_else(|| checkpoint_error(operator_id, side, "row count is invalid"))
}

fn decode_stored_row(
    bytes: &[u8],
    offset: &mut usize,
    expected_schema: &SchemaRef,
    operator_id: &str,
    side: &str,
) -> Result<StoredRow> {
    let row_id = read_segment_u64(bytes, offset)?;
    let event_time = EventTime::from_micros(read_segment_i64(bytes, offset)?);
    let charge = read_segment_u64(bytes, offset)?;
    let ipc_length = usize::try_from(read_segment_u64(bytes, offset)?)
        .map_err(|_| checkpoint_error(operator_id, side, "IPC length is invalid"))?;
    let ipc = take_segment_bytes(bytes, offset, ipc_length)?;
    let record = decode_ipc_row(ipc, expected_schema, operator_id, side)?;
    Ok(StoredRow {
        record,
        event_time,
        row_id,
        charge,
    })
}

fn decode_ipc_row(
    ipc: &[u8],
    expected_schema: &SchemaRef,
    operator_id: &str,
    side: &str,
) -> Result<RecordBatch> {
    let mut reader = StreamReader::try_new(Cursor::new(ipc), None).map_err(|error| {
        checkpoint_error(
            operator_id,
            side,
            &format!("IPC header is invalid: {error}"),
        )
    })?;
    if reader.schema().as_ref() != expected_schema.as_ref() {
        return Err(checkpoint_error(
            operator_id,
            side,
            "IPC schema is incompatible",
        ));
    }
    let record = reader
        .next()
        .transpose()
        .map_err(|error| {
            checkpoint_error(operator_id, side, &format!("IPC row is invalid: {error}"))
        })?
        .filter(|record| record.num_rows() == 1)
        .ok_or_else(|| checkpoint_error(operator_id, side, "IPC must contain one row"))?;
    if reader.next().is_some() {
        return Err(checkpoint_error(
            operator_id,
            side,
            "IPC contains extra record batches",
        ));
    }
    Ok(record)
}

fn take_segment_bytes<'a>(bytes: &'a [u8], offset: &mut usize, length: usize) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| CalcFlowError::CheckpointMismatch {
            message: "stream Join state segment offset overflowed".into(),
        })?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| CalcFlowError::CheckpointMismatch {
            message: "stream Join state segment is truncated".into(),
        })?;
    *offset = end;
    Ok(value)
}

fn read_segment_u64(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let value = take_segment_bytes(bytes, offset, 8)?;
    Ok(u64::from_le_bytes(
        value.try_into().expect("exact eight-byte segment slice"),
    ))
}

fn read_segment_i64(bytes: &[u8], offset: &mut usize) -> Result<i64> {
    let value = take_segment_bytes(bytes, offset, 8)?;
    Ok(i64::from_le_bytes(
        value.try_into().expect("exact eight-byte segment slice"),
    ))
}

fn validate_restored_rows(
    rows: &[StoredRow],
    next_row_id: u64,
    event_index: usize,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<()> {
    let mut identities = BTreeSet::new();
    for row in rows {
        validate_restored_row_identity(row, &mut identities, next_row_id, operator_id, side)?;
        validate_restored_row_payload(row, event_index, key_indices, operator_id, side)?;
    }
    Ok(())
}

fn validate_restored_row_identity(
    row: &StoredRow,
    identities: &mut BTreeSet<(EventTime, u64)>,
    next_row_id: u64,
    operator_id: &str,
    side: &str,
) -> Result<()> {
    if row.row_id >= next_row_id || !identities.insert((row.event_time, row.row_id)) {
        return Err(checkpoint_error(
            operator_id,
            side,
            "row identity is invalid",
        ));
    }
    Ok(())
}

fn validate_restored_row_payload(
    row: &StoredRow,
    event_index: usize,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<()> {
    let restored_event_time = event_time_at(&row.record, event_index, 0, operator_id, side)?
        .ok_or_else(|| checkpoint_error(operator_id, side, "stored event time is null"))?;
    let restored_charge = state_row_charge(&row.record, 0, key_indices, operator_id)?;
    if restored_event_time != row.event_time || restored_charge != row.charge {
        return Err(checkpoint_error(
            operator_id,
            side,
            "row event time or charge is inconsistent",
        ));
    }
    Ok(())
}

fn checkpoint_error(operator_id: &str, side: &str, message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("stream Join {operator_id:?} {side} {message}"),
    }
}

fn should_retain(
    incoming_is_left: bool,
    event_time: EventTime,
    opposite: Option<IngressProgress>,
    bounds: JoinTimeBounds,
) -> bool {
    let Some(opposite) = opposite else {
        return true;
    };
    if opposite.state() == crate::IngressState::Ended {
        return false;
    }
    let Some(watermark) = opposite.watermark() else {
        return true;
    };
    let extension = if incoming_is_left {
        bounds.after_micros
    } else {
        bounds.before_micros
    };
    i128::from(event_time.as_micros()) + i128::from(extension) >= i128::from(watermark.as_micros())
}

fn evict_opposite(
    rows: &mut Vec<StoredRow>,
    progress: IngressProgress,
    extension_micros: u64,
    metrics: &mut SideMetrics,
    operator_id: &str,
) -> Result<()> {
    let before = rows.len();
    if progress.state() == crate::IngressState::Ended {
        rows.clear();
    } else if let Some(watermark) = progress.watermark() {
        rows.retain(|row| {
            i128::from(row.event_time.as_micros()) + i128::from(extension_micros)
                >= i128::from(watermark.as_micros())
        });
    }
    let evicted = u64::try_from(before - rows.len())
        .map_err(|_| counter_overflow(operator_id, "evicted rows"))?;
    metrics.evicted_rows =
        checked_metric(metrics.evicted_rows, evicted, operator_id, "evicted_rows")?;
    refresh_retained_metrics(metrics, rows, operator_id)
}

fn refresh_retained_metrics(
    metrics: &mut SideMetrics,
    rows: &[StoredRow],
    operator_id: &str,
) -> Result<()> {
    metrics.retained_rows =
        u64::try_from(rows.len()).map_err(|_| counter_overflow(operator_id, "retained rows"))?;
    metrics.retained_bytes = rows
        .iter()
        .try_fold(0_u64, |total, row| total.checked_add(row.charge))
        .ok_or_else(|| counter_overflow(operator_id, "retained bytes"))?;
    Ok(())
}

fn event_time_at(
    record: &RecordBatch,
    column_index: usize,
    row_index: usize,
    operator_id: &str,
    side: &str,
) -> Result<Option<EventTime>> {
    let array = record.column(column_index).as_ref();
    if array.is_null(row_index) {
        return Ok(None);
    }
    let data_type = record.schema().field(column_index).data_type().clone();
    let value = match &data_type {
        DataType::Timestamp(TimeUnit::Second, _) => {
            downcast_timestamp::<TimestampSecondArray>(array, row_index, operator_id, side)?
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            downcast_timestamp::<TimestampMillisecondArray>(array, row_index, operator_id, side)?
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            downcast_timestamp::<TimestampMicrosecondArray>(array, row_index, operator_id, side)?
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            downcast_timestamp::<TimestampNanosecondArray>(array, row_index, operator_id, side)?
        }
        _ => {
            return Err(operator_reason(
                operator_id,
                crate::StreamingFailureReason::JoinTimeConversionFailed,
                &format!("{side} event time is not a timestamp"),
            ));
        }
    };
    EventTime::import_timestamp(value, &data_type, &format!("stream_join.{side}_event_time"))
        .map(Some)
        .map_err(|_| {
            operator_reason(
                operator_id,
                crate::StreamingFailureReason::JoinTimeConversionFailed,
                &format!("{side} event time cannot be represented"),
            )
        })
}

fn downcast_timestamp<T>(
    array: &dyn Array,
    row_index: usize,
    operator_id: &str,
    side: &str,
) -> Result<i64>
where
    T: Array + 'static,
    for<'a> &'a T: TimestampValue,
{
    let typed = array.as_any().downcast_ref::<T>().ok_or_else(|| {
        operator_reason(
            operator_id,
            crate::StreamingFailureReason::JoinTimeConversionFailed,
            &format!("{side} timestamp array type mismatch"),
        )
    })?;
    Ok(typed.timestamp_value(row_index))
}

trait TimestampValue {
    fn timestamp_value(self, row_index: usize) -> i64;
}

macro_rules! impl_timestamp_value {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl TimestampValue for &$ty {
                fn timestamp_value(self, row_index: usize) -> i64 {
                    self.value(row_index)
                }
            }
        )+
    };
}

impl_timestamp_value!(
    TimestampSecondArray,
    TimestampMillisecondArray,
    TimestampMicrosecondArray,
    TimestampNanosecondArray,
);

fn state_row_charge(
    record: &RecordBatch,
    row_index: usize,
    key_indices: &[usize],
    operator_id: &str,
) -> Result<u64> {
    let key_bytes = key_indices.iter().try_fold(0_u64, |total, &index| {
        let charge = logical_cell_charge(record.column(index).as_ref(), row_index)?;
        total
            .checked_add(charge)
            .ok_or_else(|| counter_overflow(operator_id, "encoded key length"))
    })?;
    let payload = record.columns().iter().try_fold(0_u64, |total, column| {
        let charge = logical_cell_charge(column.as_ref(), row_index)?;
        total
            .checked_add(charge)
            .ok_or_else(|| counter_overflow(operator_id, "logical payload bytes"))
    })?;
    STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1
        .checked_add(key_bytes)
        .and_then(|value| value.checked_add(16))
        .and_then(|value| value.checked_add(payload))
        .ok_or_else(|| counter_overflow(operator_id, "state row charge"))
}

/// Logical charge of one non-null cell under the frozen V1 accounting table,
/// including its validity byte.
fn logical_cell_charge(array: &dyn Array, row_index: usize) -> Result<u64> {
    if array.is_null(row_index) {
        return Ok(1);
    }
    let data_type = array.data_type();
    if let Some(value) = fixed_cell_charge(data_type) {
        return validity_wrapped(value);
    }
    let Some(value) = variable_cell_charge(array, row_index)? else {
        return sized_cell_charge(array, row_index);
    };
    validity_wrapped(value)
}

fn fixed_cell_charge(data_type: &DataType) -> Option<u64> {
    if matches!(data_type, DataType::Null) {
        return Some(0);
    }
    if matches!(
        data_type,
        DataType::Boolean | DataType::Int8 | DataType::UInt8
    ) {
        return Some(1);
    }
    if matches!(
        data_type,
        DataType::Int16 | DataType::UInt16 | DataType::Float16
    ) {
        return Some(2);
    }
    if is_four_byte_cell(data_type) {
        return Some(4);
    }
    if is_eight_byte_cell(data_type) {
        return Some(8);
    }
    fixed_wide_cell_charge(data_type)
}

fn fixed_wide_cell_charge(data_type: &DataType) -> Option<u64> {
    if is_sixteen_byte_cell(data_type) {
        return Some(16);
    }
    if matches!(data_type, DataType::Decimal256(_, _)) {
        return Some(32);
    }
    None
}

fn is_four_byte_cell(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int32
            | DataType::UInt32
            | DataType::Float32
            | DataType::Date32
            | DataType::Time32(_)
            | DataType::Interval(IntervalUnit::YearMonth)
            | DataType::Decimal32(_, _)
    )
}

fn is_eight_byte_cell(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Date64
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Interval(IntervalUnit::DayTime)
            | DataType::Decimal64(_, _)
    )
}

fn is_sixteen_byte_cell(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Interval(IntervalUnit::MonthDayNano) | DataType::Decimal128(_, _)
    )
}

fn variable_cell_charge(array: &dyn Array, row_index: usize) -> Result<Option<u64>> {
    if let DataType::FixedSizeBinary(size) = array.data_type() {
        return fixed_size_binary_charge(*size).map(Some);
    }
    string_cell_charge(array, row_index)
        .or(binary_cell_charge(array, row_index))
        .transpose()
}

fn fixed_size_binary_charge(size: i32) -> Result<u64> {
    u64::try_from(size).map_err(|_| CalcFlowError::Internal {
        message: "negative FixedSizeBinary width".into(),
    })
}

fn string_cell_charge(array: &dyn Array, row_index: usize) -> Option<Result<u64>> {
    match array.data_type() {
        DataType::Utf8 => Some(downcast_cell_charge::<StringArray>(
            array, row_index, 4, "Utf8",
        )),
        DataType::LargeUtf8 => Some(downcast_cell_charge::<LargeStringArray>(
            array,
            row_index,
            8,
            "LargeUtf8",
        )),
        DataType::Utf8View => Some(downcast_cell_charge::<StringViewArray>(
            array, row_index, 16, "Utf8View",
        )),
        _ => None,
    }
}

fn binary_cell_charge(array: &dyn Array, row_index: usize) -> Option<Result<u64>> {
    match array.data_type() {
        DataType::Binary => Some(downcast_cell_charge::<BinaryArray>(
            array, row_index, 4, "Binary",
        )),
        DataType::LargeBinary => Some(downcast_cell_charge::<LargeBinaryArray>(
            array,
            row_index,
            8,
            "LargeBinary",
        )),
        DataType::BinaryView => Some(downcast_cell_charge::<BinaryViewArray>(
            array,
            row_index,
            16,
            "BinaryView",
        )),
        _ => None,
    }
}

/// Downcasts one variable-length array kind and charges prefix plus value.
fn downcast_cell_charge<T>(
    array: &dyn Array,
    row_index: usize,
    prefix: u64,
    label: &str,
) -> Result<u64>
where
    T: Array + 'static,
    for<'a> &'a T: CellBytes,
{
    let typed = array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("{label} array type mismatch"),
        })?;
    prefix_cell_charge(prefix, typed.cell_len(row_index), label)
}

fn prefix_cell_charge(prefix: u64, len: usize, label: &str) -> Result<u64> {
    prefix
        .checked_add(u64::try_from(len).unwrap_or(u64::MAX))
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("{label} cell charge overflow"),
        })
}

trait CellBytes {
    fn cell_len(self, row_index: usize) -> usize;
}

macro_rules! impl_cell_bytes {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CellBytes for &$ty {
                fn cell_len(self, row_index: usize) -> usize {
                    self.value(row_index).len()
                }
            }
        )+
    };
}

impl_cell_bytes!(
    StringArray,
    LargeStringArray,
    StringViewArray,
    BinaryArray,
    LargeBinaryArray,
    BinaryViewArray,
);

/// Charges a variable-length cell from its raw slice memory footprint.
fn sized_cell_charge(array: &dyn Array, row_index: usize) -> Result<u64> {
    let sized = array
        .slice(row_index, 1)
        .to_data()
        .get_slice_memory_size()
        .map_err(|error| CalcFlowError::Internal {
            message: format!("logical payload charge failed: {error}"),
        })?;
    u64::try_from(sized).map_err(|_| CalcFlowError::Internal {
        message: "logical payload charge exceeds UInt64".into(),
    })
}

fn validity_wrapped(value_bytes: u64) -> Result<u64> {
    value_bytes
        .checked_add(1)
        .ok_or_else(|| CalcFlowError::Internal {
            message: "logical cell charge overflow".into(),
        })
}

fn checked_metric(current: u64, delta: u64, operator_id: &str, field: &str) -> Result<u64> {
    current
        .checked_add(delta)
        .ok_or_else(|| counter_overflow(operator_id, field))
}

fn counter_overflow(operator_id: &str, field: &str) -> CalcFlowError {
    operator_reason(
        operator_id,
        crate::StreamingFailureReason::JoinCounterOverflow,
        &format!("{field} counter overflowed"),
    )
}

fn operator_reason(
    operator_id: &str,
    reason_code: crate::StreamingFailureReason,
    message: &str,
) -> CalcFlowError {
    CalcFlowError::OperatorReason {
        node_id: operator_id.into(),
        reason_code,
        message: message.into(),
    }
}

fn operator_error(operator_id: &str, message: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: operator_id.into(),
        message: message.into(),
    }
}

fn default_left_prefix() -> String {
    "left".into()
}

fn default_right_prefix() -> String {
    "right".into()
}
