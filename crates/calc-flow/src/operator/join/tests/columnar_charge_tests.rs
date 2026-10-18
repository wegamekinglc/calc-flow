use super::*;

mod fixtures;

use fixtures::{operator, record};

#[tokio::test]
async fn test_columnar_charge_nullable_fixed_columns_skip_scalar_cells() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let record = RecordBatch::try_new(
        left_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; 70])),
            Arc::new(TimestampMicrosecondArray::from(vec![0; 70]).with_timezone("UTC")),
            Arc::new(
                (0..70)
                    .map(|row| (row % 3 != 0).then_some(42))
                    .collect::<Int64Array>(),
            ),
        ],
    )
    .unwrap()
    .slice(3, 65);
    let expected = (0..record.num_rows())
        .map(|row| state_row_charge(&record, row, &[0], "match").unwrap())
        .collect::<Vec<_>>();
    let input = Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap();
    reset_join_work();
    let prepared = operator
        .prepare_batch("left", &input, &context)
        .await
        .unwrap();
    assert_eq!(
        prepared
            .chunks
            .indexed()
            .map(|(_, row)| row)
            .map(|row| row.charge)
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(join_work().scalar_charge_cells, 0);
    assert_eq!(input.table_payload().unwrap().batches(), &[record]);
}

async fn assert_oracle(payloads: &[ArrayRef]) -> JoinWork {
    let record = record(payloads);
    let records = vec![record.slice(0, 0), record.slice(3, 65), record.slice(1, 2)];
    let expected = records
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| state_row_charge(record, row, &[0], "match").unwrap())
        })
        .collect::<Vec<_>>();
    let input = Batch::table(records.clone(), BatchMetadata::default()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut last_work = JoinWork::default();
    for side in ["left", "right"] {
        let mut operator = operator(&record);
        reset_join_work();
        let prepared = operator
            .prepare_batch(side, &input, &context)
            .await
            .unwrap();
        last_work = join_work();
        assert_eq!(
            prepared
                .chunks
                .indexed()
                .map(|(_, row)| row)
                .map(|row| row.charge)
                .collect::<Vec<_>>(),
            expected,
            "{side}: {:?}",
            record.schema()
        );
        assert_eq!(
            prepared
                .chunks
                .indexed()
                .map(|(_, row)| row)
                .map(|row| row.row_id)
                .collect::<Vec<_>>(),
            (0..67).collect::<Vec<_>>()
        );
        assert_eq!(last_work.scalar_charge_cells, 0);
        assert_eq!(last_work.charge_scalar_fallbacks, 0);
        assert_eq!(input.table_payload().unwrap().batches(), records);
    }
    last_work
}

#[tokio::test]
async fn test_columnar_charge_all_fixed_types_match_v1() {
    let payloads = fixtures::fixed();
    let work = Box::pin(assert_oracle(&payloads)).await;
    assert_eq!(work.charge_column_resolutions, (payloads.len() + 2) * 3);
    assert_eq!(work.charge_offset_reads, 0);
}

#[tokio::test]
async fn test_columnar_charge_variable_offsets_and_views_match_v1() {
    let work = Box::pin(assert_oracle(&fixtures::variable())).await;
    assert!(work.charge_offset_reads > 0);
    assert_eq!(work.charge_nested_rows, 0);
}

#[tokio::test]
async fn test_columnar_charge_nested_dictionary_and_run_payloads_match_v1() {
    let work = Box::pin(assert_oracle(&fixtures::nested())).await;
    assert!(work.charge_range_sums > 0);
    assert!(work.charge_offset_reads > 0);
}

#[tokio::test]
async fn test_columnar_charge_sparse_null_late_and_retention_coordinates() {
    let mut source = record(&[fixtures::dictionary()]);
    let mut columns = source.columns().to_vec();
    columns[0] = Arc::new(
        (0..70)
            .map(|row| (row % 5 != 0).then_some(7))
            .collect::<Int64Array>(),
    );
    columns[1] = Arc::new(
        (0..70)
            .map(|row| (row % 7 != 0).then_some(i64::from(row)))
            .collect::<TimestampMicrosecondArray>()
            .with_timezone("UTC"),
    );
    source = RecordBatch::try_new(source.schema(), columns)
        .unwrap()
        .slice(3, 65);
    let input = Batch::table(
        vec![source.clone(), source.clone()],
        BatchMetadata::default(),
    )
    .unwrap();
    let job = job();
    let context = progress_context(
        &job,
        (IngressState::Active, Some(10)),
        (IngressState::Active, Some(50)),
    );
    let mut operator = operator(&source);
    reset_join_work();
    let prepared = operator
        .prepare_batch("left", &input, &context)
        .await
        .unwrap();
    let work = join_work();
    let retained = (0..130)
        .filter(|physical| {
            let row = physical % 65;
            !source.column(0).is_null(row) && !source.column(1).is_null(row) && row + 3 >= 40
        })
        .collect::<Vec<_>>();
    assert_eq!(
        prepared
            .chunks
            .indexed()
            .map(|(_, row)| row)
            .map(|row| usize::try_from(row.row_id).unwrap())
            .collect::<Vec<_>>(),
        retained
    );
    for row in (&prepared.chunks as &dyn Rows).iter() {
        assert_eq!(
            row.charge,
            state_row_charge(
                &source,
                usize::try_from(row.row_id).unwrap() % 65,
                &[0],
                "match"
            )
            .unwrap()
        );
    }
    assert_eq!(prepared.next_row_id, 130);
    assert_eq!(prepared.metrics.null_event_time_rows, 18);
    assert_eq!(prepared.metrics.null_key_rows, 24);
    assert_eq!(prepared.metrics.late_rows, 10);
    assert_eq!(prepared.metrics.late_affected_batches, 1);
    assert_eq!(prepared.metrics.max_lateness_micros, Some(7));
    assert_eq!(work.scalar_charge_cells, 0);
    assert!(work.charge_sparse_validity_visits > 0);
    assert_eq!(
        input.table_payload().unwrap().batches(),
        &[source.clone(), source]
    );
}

#[tokio::test]
async fn test_columnar_charge_empty_and_all_rejected_skip_payloads() {
    let source = record(&[fixtures::dictionary()]);
    let job = job();
    let context = progress_context(
        &job,
        (IngressState::Active, Some(100)),
        (IngressState::Active, None),
    );
    for source in [source.slice(0, 0), source] {
        let mut operator = operator(&source);
        let input = Batch::table(vec![source], BatchMetadata::default()).unwrap();
        reset_join_work();
        let prepared = operator
            .prepare_batch("left", &input, &context)
            .await
            .unwrap();
        assert!(prepared.chunks.is_empty());
        assert_eq!(join_work().charge_column_resolutions, 0);
        assert_eq!(join_work().scalar_charge_cells, 0);
    }
}

#[tokio::test]
async fn test_columnar_charge_unsupported_dictionary_keeps_v1_error_and_null_path() {
    let source = record(&[fixtures::unsupported_dictionary()]);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let expected = state_row_charge(&source, 0, &[0], "match")
        .unwrap_err()
        .to_string();
    let mut operator = operator(&source);
    operator.spec.limits.max_state_bytes_per_side = 1;
    let input = Batch::table(vec![source.clone()], BatchMetadata::default()).unwrap();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    reset_join_work();
    let actual = operator
        .process_data("left", input, &context, &mut collector)
        .await
        .unwrap_err();
    assert_eq!(actual.to_string(), expected);
    assert_eq!(join_work().charge_scalar_fallbacks, 1);
    assert!(operator.state.left.is_empty());
    assert_eq!(operator.state.next_left_row_id, 0);
    assert_eq!(operator.status().state_limit_failures, 0);
    assert!(collector.drain("output").is_empty());
    operator.spec.limits.max_state_bytes_per_side = 10_000_000;
    let input = Batch::table(vec![source.slice(1, 1)], BatchMetadata::default()).unwrap();
    let prepared = operator
        .prepare_batch("left", &input, &context)
        .await
        .unwrap();
    assert_eq!(
        prepared.chunks.charge(0),
        state_row_charge(&source, 1, &[0], "match").unwrap()
    );
}

#[tokio::test]
async fn test_columnar_charge_byte_limit_and_overflow_reject_before_output() {
    let source = record(&[Arc::new(StringArray::from(vec!["retained payload"]))]);
    let fee = state_row_charge(&source, 0, &[0], "match").unwrap();
    let input = Batch::table(vec![source.clone()], BatchMetadata::default()).unwrap();
    for limit in [fee - 1, fee] {
        let mut operator = operator(&source);
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("right", input.clone(), &context, &mut collector)
            .await
            .unwrap();
        operator.spec.limits.max_state_bytes_per_side = limit;
        let result = operator
            .process_data("left", input.clone(), &context, &mut collector)
            .await;
        if limit < fee {
            assert_eq!(
                reason_of(&result.unwrap_err()),
                Some(crate::StreamingFailureReason::JoinStateLimitExceeded)
            );
            assert!(collector.drain("output").is_empty());
            assert_eq!(operator.state.next_left_row_id, 0);
            assert_eq!(operator.status().state_limit_failures, 1);
        } else {
            result.unwrap();
            assert_eq!(collector.drain("output").len(), 1);
            assert_eq!(operator.status().left.retained_bytes, fee);
        }
    }
    let mut operator = operator(&source);
    operator.state.metrics.left.retained_bytes = u64::MAX - fee + 1;
    let before = operator.status();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let error = operator
        .process_data("left", input, &context, &mut collector)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("state bytes"), "{error}");
    assert_eq!(operator.status(), before);
    assert!(collector.drain("output").is_empty());
}

#[tokio::test]
async fn test_columnar_charge_partial_block_cancel_and_abandon_release_workspace() {
    tokio::spawn(async {
        for cancel in [false, true] {
            let source = record(&fixtures::fixed());
            let input = Batch::table(vec![source.slice(3, 65)], BatchMetadata::default()).unwrap();
            let mut operator = operator(&source);
            let pool = operator.checkpoint_preload_test_pool().unwrap();
            let job = job();
            let context = StreamOperatorContext::new(&job, "match", None);
            let plan = operator.begin_batch("left", &input).unwrap();
            let mut bundle = operator.admission_bundle(&plan, &input).unwrap();
            operator
                .admit_record(
                    &input.table_payload().unwrap().batches()[0],
                    &plan,
                    "left",
                    &context,
                    &mut bundle,
                    0,
                )
                .await
                .unwrap();
            let rows = (&bundle.admitted as &dyn AdmittedRows)
                .groups()
                .next()
                .unwrap();
            let mut quantum = columnar::Quantum::default();
            while tokio::task::coop::has_budget_remaining() {
                tokio::task::consume_budget().await;
            }
            reset_join_work();
            let mut future = Box::pin(admission_charge::PayloadCharges::new(
                rows.block(0, 64),
                u64::MAX,
                &context,
                &mut quantum,
            ));
            assert!(futures::poll!(future.as_mut()).is_pending());
            assert_eq!(join_work().charge_column_resolutions, 1);
            if cancel {
                job.cancellation().cancel();
                assert!(matches!(
                    future.as_mut().await,
                    Err(CalcFlowError::Cancelled { .. })
                ));
            }
            drop(future);
            drop(bundle);
            assert_eq!(pool.reserved(), 0);
            assert!(operator.state.left.is_empty());
            assert_eq!(operator.state.next_left_row_id, 0);
        }
    })
    .await
    .unwrap();
}
