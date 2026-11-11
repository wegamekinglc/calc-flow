use super::*;

#[tokio::test]
async fn incremental_history_reclaims_deque_capacity_after_time_jump() {
    let spec = kernel_spec(json!([duration_output("mean", "price", "mean", 1_024)]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let (job, mut output) = job_output(&operator);
    let initial = (1..=128_u32)
        .map(|n| (i64::from(n), "a", u64::from(n), Some(f64::from(n))))
        .collect::<Vec<_>>();
    for (record, watermark) in [
        (float64_fast_record(&initial), 128),
        (
            float64_fast_record(&[(10_000, "a", 129, Some(1.0))]),
            10_000,
        ),
        (
            float64_fast_record(&[(10_001, "a", 130, Some(3.0))]),
            10_001,
        ),
    ] {
        let context =
            StreamOperatorContext::new(&job, "rolling", operator.state.last_input_watermark);
        operator
            .process_data(
                "input",
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        operator
            .on_watermark(EventTime::from_micros(watermark), &context, &mut output)
            .await
            .unwrap();
        output.drain("output");
    }
    let history = first_history(&operator);
    assert_eq!(history.rows.len(), 2);
    assert_eq!(history.transition_count, 130);
    assert!(
        history.rows.capacity() <= 64,
        "expired history must not leave a permanent container high-water mark: {}",
        history.rows.capacity()
    );
}

#[tokio::test]
async fn incremental_history_reuses_retained_payload_across_mixed_frames() {
    use datafusion::arrow::array::StringArray;

    let spec = kernel_spec(json!([
        duration_output("mean", "price", "time_mean", 3),
        aggregate_output("min", "price", "row_min", 5),
    ]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rolling", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    let first = float64_fast_record(&[
        (1, "a", 1, Some(1.0)),
        (2, "a", 2, Some(3.0)),
        (3, "a", 3, Some(5.0)),
    ]);
    let mut columns = first.columns().to_vec();
    columns[5] = Arc::new(StringArray::from(vec!["x".repeat(16_384); 3]));
    let first = RecordBatch::try_new(first.schema(), columns).unwrap();
    operator
        .process_data(
            "input",
            Batch::table(vec![first.clone()], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(3), &context, &mut output)
        .await
        .unwrap();
    output.drain("output");
    let history = first_history(&operator);
    let retained = history.rows.front().unwrap().as_ptr();
    let label = match &history.rows.front().unwrap()[5] {
        ScalarValue::Utf8(Some(value)) => value.as_ptr(),
        value => panic!("expected a retained string, got {value:?}"),
    };
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(3)));
    operator
        .process_data(
            "input",
            Batch::table(
                vec![float64_fast_record(&[(4, "a", 4, Some(7.0))])],
                BatchMetadata::default(),
            )
            .unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(4), &context, &mut output)
        .await
        .unwrap();
    let emitted = output.drain("output");
    let record = &emitted[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(
        record
            .column(6)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0),
        5.0
    );
    assert_eq!(
        record
            .column(7)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0),
        1.0
    );
    let history = first_history(&operator);
    assert_eq!(
        history.rows.len(),
        4,
        "the row frame still needs the oldest row"
    );
    assert_eq!(
        history.rows.front().unwrap().as_ptr(),
        retained,
        "appending one row must preserve the existing retained scalar payload"
    );
    assert!(
        matches!(&history.rows.front().unwrap()[5], ScalarValue::Utf8(Some(value)) if value.as_ptr() == label)
    );
    assert_eq!(first.num_rows(), 3);
}

async fn finalize_record(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    output: &mut crate::EdgeCollector,
    record: RecordBatch,
    watermark: i64,
) -> Vec<RecordBatch> {
    let context = StreamOperatorContext::new(job, "rolling", operator.state.last_input_watermark);
    operator
        .process_data(
            "input",
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            &context,
            output,
        )
        .await
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(watermark), &context, output)
        .await
        .unwrap();
    output
        .drain("output")
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

fn assert_live_charge(operator: &RollingOperator) {
    assert_eq!(
        operator.state.charge,
        budget::full_state_charge(
            &operator.state.buffer,
            &operator.state.ordered,
            &operator.state.histories,
            &operator.compiled,
            "rolling",
        )
        .unwrap()
    );
}

fn mixed_spec() -> RollingSpec {
    kernel_spec(json!([
        duration_output("mean", "price", "time_mean", 3),
        aggregate_output("min", "price", "row_min", 5),
    ]))
}

#[tokio::test]
async fn incremental_history_budget_rejection_preserves_history_and_retries() {
    let mut operator =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), mixed_spec()).unwrap();
    operator
        .set_state_budget(StateBudget::new(6, 1 << 20).unwrap())
        .unwrap();
    let (job, mut output) = job_output(&operator);
    finalize_record(
        &mut operator,
        &job,
        &mut output,
        float64_fast_record(&[
            (1, "a", 1, Some(1.0)),
            (2, "a", 2, Some(3.0)),
            (3, "a", 3, Some(5.0)),
        ]),
        3,
    )
    .await;
    assert_live_charge(&operator);
    let prior = format!("{:?}", operator.state.typed_kernel_state);
    let retained = first_history(&operator).rows.front().unwrap().as_ptr();
    let sequence = operator.state.next_output_sequence;
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(3)));
    operator
        .process_data(
            "input",
            Batch::table(
                vec![float64_fast_record(&[
                    (4, "a", 4, Some(7.0)),
                    (4, "b", 1, Some(9.0)),
                ])],
                BatchMetadata::default(),
            )
            .unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let error = operator
        .on_watermark(EventTime::from_micros(4), &context, &mut output)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("state budget"), "{error}");
    assert_eq!(operator.state.buffer.len(), 2);
    assert_eq!(
        operator.state.last_input_watermark,
        Some(EventTime::from_micros(3))
    );
    assert_eq!(operator.state.next_output_sequence, sequence);
    assert_eq!(format!("{:?}", operator.state.typed_kernel_state), prior);
    assert_eq!(operator.state.histories.by_entity.len(), 1);
    assert_eq!(
        first_history(&operator).rows.front().unwrap().as_ptr(),
        retained
    );
    assert!(output.drain("output").is_empty());
    assert_live_charge(&operator);
    operator
        .set_state_budget(StateBudget::new(7, 1 << 20).unwrap())
        .unwrap();
    operator
        .on_watermark(EventTime::from_micros(4), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.state.histories.by_entity.len(), 2);
    assert_eq!(first_history(&operator).transition_count, 4);
    assert!(operator.state.buffer.is_empty());
    let emitted = output.drain("output");
    let record = &emitted[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(
        record
            .column(6)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[5.0, 9.0]
    );
    assert_live_charge(&operator);
}

#[tokio::test]
async fn incremental_history_rejected_emission_keeps_history_for_manual_retry() {
    struct Reject(bool);
    #[async_trait]
    impl StreamCollector for Reject {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            if self.0 {
                Err(CalcFlowError::Cancelled { run_id: "7".into() })
            } else {
                Err(internal_error("expected history collector rejection"))
            }
        }
    }
    for cancelled in [false, true] {
        let mut operator =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), mixed_spec()).unwrap();
        let (job, mut output) = job_output(&operator);
        finalize_record(
            &mut operator,
            &job,
            &mut output,
            float64_fast_record(&[
                (1, "a", 1, Some(1.0)),
                (2, "a", 2, Some(3.0)),
                (3, "a", 3, Some(5.0)),
            ]),
            3,
        )
        .await;
        let prior = format!("{:?}", operator.state.typed_kernel_state);
        let retained = first_history(&operator).rows.front().unwrap().as_ptr();
        let sequence = operator.state.next_output_sequence;
        let next = Batch::table(
            vec![float64_fast_record(&[(4, "a", 4, Some(7.0))])],
            BatchMetadata::default(),
        )
        .unwrap();
        let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(3)));
        operator
            .process_data("input", next.clone(), &context, &mut output)
            .await
            .unwrap();
        let error = operator
            .on_watermark(EventTime::from_micros(4), &context, &mut Reject(cancelled))
            .await
            .unwrap_err();
        assert_eq!(matches!(error, CalcFlowError::Cancelled { .. }), cancelled);
        assert_eq!(
            operator.state.last_input_watermark,
            Some(EventTime::from_micros(3))
        );
        assert_eq!(operator.state.next_output_sequence, sequence);
        assert_eq!(format!("{:?}", operator.state.typed_kernel_state), prior);
        let history = first_history(&operator);
        assert_eq!(history.rows.len(), 3);
        assert_eq!(history.rows.front().unwrap().as_ptr(), retained);
        assert_eq!(history.transition_count, 3);
        assert_live_charge(&operator);
        operator
            .process_data("input", next, &context, &mut output)
            .await
            .unwrap();
        operator
            .on_watermark(EventTime::from_micros(4), &context, &mut output)
            .await
            .unwrap();
        let history = first_history(&operator);
        assert_eq!(history.transition_count, 4);
        assert_eq!(history.rows.front().unwrap().as_ptr(), retained);
        assert!(!output.drain("output").is_empty());
        assert_live_charge(&operator);
    }
}

#[tokio::test]
async fn incremental_history_matches_batch_and_checkpoint_across_mixed_boundaries() {
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        for (row_window, layout) in [(2, 1), (5, 1), (2, 3), (5, 3)] {
            let mut spec = kernel_spec(json!([
                duration_output("mean", "price", "time_mean", 3),
                aggregate_output("min", "price", "row_min", row_window),
                duration_output("max", "price", "time_max", 3),
                pair_output(
                    "covariance",
                    "price",
                    "volume",
                    "cov",
                    json!({"kind": "duration", "micros": 3}),
                    1
                ),
            ]));
            spec.numerical_profile = profile;
            let input = float64_pair_record(&[
                (0, "a", 1, Some(1.0), Some(2.0)),
                (1, "b", 1, Some(10.0), Some(20.0)),
                (2, "a", 2, Some(3.0), Some(6.0)),
                (3, "a", 3, Some(5.0), Some(10.0)),
                (4, "b", 2, Some(20.0), Some(40.0)),
                (6, "a", 4, Some(7.0), Some(14.0)),
                (7, "a", 5, Some(11.0), Some(22.0)),
                (8, "a", 6, Some(13.0), Some(26.0)),
            ]);
            let mut operator =
                RollingOperator::new("rolling", input.schema(), spec.clone()).unwrap();
            let input_batch = Batch::table(vec![input.clone()], BatchMetadata::default()).unwrap();
            let rows = read_buffered_rows(
                input_batch.table_payload().unwrap(),
                &operator.compiled,
                "rolling",
            )
            .unwrap();
            let expected = compute_output_columns(
                &rows,
                &RollingHistories::default(),
                &operator.compiled,
                "rolling",
            )
            .unwrap();
            let expected_record = build_output_record(
                &rows,
                expected.columns,
                operator.output_ports()[0].schema().unwrap(),
                "rolling",
            )
            .unwrap();
            let mut expected_histories = RollingHistories::default();
            expected_histories.apply(expected.touched);
            let (job, mut output) = job_output(&operator);
            let mut emitted =
                finalize_record(&mut operator, &job, &mut output, input.slice(0, 4), 3).await;
            assert_live_charge(&operator);
            let snapshot = checkpoint_with_layout(&mut operator, layout);
            let mut restored = RollingOperator::new("rolling", input.schema(), spec).unwrap();
            StreamOperator::restore(&mut restored, &snapshot).unwrap();
            let restored_count = first_history(&restored).transition_count;
            emitted.extend(
                finalize_record(&mut restored, &job, &mut output, input.slice(4, 4), 8).await,
            );
            assert_live_charge(&restored);
            let actual = concat_batches(&expected_record.schema(), &emitted).unwrap();
            assert_eq!(actual, expected_record);
            let history = restored
                .state
                .histories
                .by_entity
                .get(&vec![Some(KeyValue::String("a".into()))])
                .unwrap();
            assert_eq!(history.transition_count, restored_count + 3);
            let times = history
                .rows
                .iter()
                .map(|row| history_event_time(row, &restored.compiled))
                .collect::<Vec<_>>();
            assert_eq!(
                times,
                if row_window == 2 {
                    vec![6, 7, 8]
                } else {
                    vec![2, 3, 6, 7, 8]
                }
            );
            if layout == ROLLING_STATE_LAYOUT_VERSION {
                // Legacy captures omit transition counts before the checkpoint.
                for row in &rows[..4] {
                    expected_histories
                        .by_entity
                        .get_mut(&row.identity.entity)
                        .unwrap()
                        .transition_count -= 1;
                }
            }
            let expected_state = encode_state_segment(
                &expected_histories,
                &BTreeMap::new(),
                input.schema().as_ref(),
                &restored.compiled,
                TEST_FINGERPRINT,
                "rolling",
            )
            .unwrap();
            let actual_state = encode_state_segment(
                &restored.state.histories,
                &restored.state.buffer,
                input.schema().as_ref(),
                &restored.compiled,
                TEST_FINGERPRINT,
                "rolling",
            )
            .unwrap();
            assert_eq!(
                actual_state, expected_state,
                "retained state encoding must remain canonical: {profile:?}, rows={row_window}, layout={layout}"
            );
        }
    }
}

fn checkpoint_with_layout(
    operator: &mut RollingOperator,
    layout: u32,
) -> crate::OperatorStateSnapshot {
    if layout == ROLLING_STATE_LAYOUT_VERSION {
        let compiled = Arc::make_mut(&mut operator.compiled);
        compiled.state_layout_version = layout;
        compiled.state_schema_fingerprint = compiled.legacy_state_schema_fingerprint.clone();
    }
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(
        snapshot.inline_metadata["state_layout_version"],
        json!(layout)
    );
    snapshot
}

#[tokio::test]
async fn incremental_history_retention_handles_negative_time_bounds() {
    for origin in [i64::MIN + 1, -10, i64::MAX - 3] {
        let spec = kernel_spec(json!([duration_output("mean", "price", "mean", 8)]));
        let mut operator =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
        let (job, mut output) = job_output(&operator);
        finalize_record(
            &mut operator,
            &job,
            &mut output,
            float64_fast_record(&[(origin, "a", 1, Some(1.0))]),
            origin,
        )
        .await;
        let emitted = finalize_record(
            &mut operator,
            &job,
            &mut output,
            float64_fast_record(&[(origin + 2, "a", 2, Some(3.0))]),
            origin + 2,
        )
        .await;
        let record = &emitted[0];
        assert_eq!(
            record
                .column(6)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .values()
                .as_ref(),
            &[2.0]
        );
        let history = first_history(&operator);
        assert_eq!(history.rows.len(), 2);
        assert_eq!(history.transition_count, 2);
        assert_live_charge(&operator);
    }
}

fn job_output(operator: &RollingOperator) -> (crate::StreamJobContext, crate::EdgeCollector) {
    (
        crate::StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        ),
        crate::EdgeCollector::new(operator.output_ports().to_vec()),
    )
}

fn first_history(operator: &RollingOperator) -> &EntityRollingState {
    operator.state.histories.by_entity.values().next().unwrap()
}

#[tokio::test]
async fn incremental_history_expires_exact_lower_bound_after_deque_wrap() {
    let spec = kernel_spec(json!([duration_output("mean", "price", "mean", 5)]));
    let mut operator = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let (job, mut output) = job_output(&operator);
    let mut wrapped = false;
    for n in 1..=12_u32 {
        finalize_record(
            &mut operator,
            &job,
            &mut output,
            float64_fast_record(&[(i64::from(n), "a", u64::from(n), Some(f64::from(n)))]),
            i64::from(n),
        )
        .await;
        let history = first_history(&operator);
        wrapped |= !history.rows.as_slices().1.is_empty();
        let times = history
            .rows
            .iter()
            .map(|row| history_event_time(row, &operator.compiled))
            .collect::<Vec<_>>();
        assert_eq!(
            times,
            (i64::from(n.saturating_sub(4).max(1))..=i64::from(n)).collect::<Vec<_>>()
        );
        assert_live_charge(&operator);
    }
    assert!(
        wrapped,
        "the expiry path must exercise both VecDeque slices"
    );
    let survivor = first_history(&operator).rows[1].as_ptr();
    let emitted = finalize_record(
        &mut operator,
        &job,
        &mut output,
        float64_fast_record(&[(13, "a", 13, Some(13.0))]),
        13,
    )
    .await;
    let history = first_history(&operator);
    assert_eq!(history.rows.front().unwrap().as_ptr(), survivor);
    assert_eq!(history.transition_count, 13);
    assert_eq!(
        emitted[0]
            .column(6)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[11.0]
    );
    assert_live_charge(&operator);
}
