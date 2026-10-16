use super::{support::*, *};
use datafusion::execution::memory_pool::MemoryPool;
use std::sync::atomic::Ordering;

const WIDTH: usize = 1 << 20;

fn binary_operator() -> StreamJoinOperator {
    let mut fields = right_schema().fields()[..2].to_vec();
    fields.extend([
        Arc::new(Field::new(
            "blob",
            DataType::FixedSizeBinary(i32::try_from(WIDTH).unwrap()),
            false,
        )),
        Arc::new(Field::new("unused", DataType::Int64, false)),
    ]);
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(100_000, 100_000_000, 1_000_000).unwrap();
    let mut operator = StreamJoinOperator::new(
        "match",
        left_schema(),
        Arc::new(Schema::new(fields)),
        declaration,
    )
    .unwrap();
    operator.set_output_projection(vec![5]).unwrap();
    operator
}

async fn preload_binary(operator: &mut StreamJoinOperator, job: &StreamJobContext) {
    let values = (0..5).map(|index| Some(vec![u8::try_from(index).unwrap(); WIDTH]));
    let record = RecordBatch::try_new(
        Arc::clone(operator.input_schema(1)),
        vec![
            Arc::new(Int64Array::from(vec![7; 5])),
            Arc::new(TimestampMicrosecondArray::from(vec![0; 5]).with_timezone("UTC")),
            Arc::new(
                FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                    values,
                    i32::try_from(WIDTH).unwrap(),
                )
                .unwrap(),
            ),
            Arc::new(Int64Array::from(vec![42; 5])),
        ],
    )
    .unwrap();
    let mut sink = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            &StreamOperatorContext::new(job, "match", None),
            &mut sink,
        )
        .await
        .unwrap();
    assert!(sink.drain("output").is_empty());
    assert!(matches!(
        operator.state.right.payload(0).owner(),
        Some(columnar::RowPayload::RestoredV2 { .. })
    ));
}

fn sparse_input() -> Batch {
    left_batch(
        (0..ROWS)
            .map(|index| {
                if index % (ROWS / 4) < 20 {
                    0
                } else {
                    1_000_000_000
                }
            })
            .collect(),
    )
}

#[test]
fn test_morsel_fixed_binary_falls_back_before_materializing_preflight() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let pool = Pool::new(0);
        let mut operator = binary_operator();
        operator
            .runtime
            .initialize_benchmark_pool(pool.clone())
            .unwrap();
        let job = job().with_gather_owner(service.owner("binary-preflight".into()));
        preload_binary(&mut operator, &job).await;
        let observations = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let observed = Arc::clone(&observations);
        let observed_pool = Arc::clone(&pool);
        operator.probe_unit_test_hook = Some(Arc::new(move |phase, unit, completed| {
            if phase == data_work::ProbePhase::Fill {
                if completed {
                    observed.lock().push((
                        unit,
                        join_work().output_peak_bytes,
                        observed_pool.live.load(Ordering::SeqCst),
                        observed_pool.attempts.load(Ordering::SeqCst),
                    ));
                } else {
                    reset_join_work();
                }
            }
        }));
        let context = StreamOperatorContext::new(&job, "match", None);
        let source = sparse_input();
        let original = source.table_payload().unwrap().batches().to_vec();
        let (mut prepared, expiration) = operator
            .prepare_data_append("left", &source, &context)
            .await
            .unwrap();
        operator
            .finish_morsels(&mut prepared, &context)
            .await
            .unwrap();
        let observations = observations.lock().clone();
        println!(
            "worker (unit, peak array bytes, morsel credit, admission attempts): {observations:?}"
        );
        assert_eq!(
            observations.len(),
            4,
            "real workers must fill all four ranges"
        );
        assert!(
            observations.iter().all(|&(_, peak, _, _)| peak == 0),
            "materializing preflight must stay off unfunded morsel workers"
        );
        assert!(prepared.gathered.is_none());
        assert_eq!(prepared.output.len(), 400);
        assert_eq!(
            pool.attempts.load(Ordering::SeqCst),
            0,
            "no optional workspace admission"
        );
        let mut sink = BinarySink::default();
        operator
            .finish_data("left", prepared, expiration, &context, &mut sink)
            .await
            .unwrap();
        assert_eq!(sink.rows, 400);
        assert_eq!(operator.status().emitted_match_rows, 400);
        assert_eq!(operator.status().left.retained_rows, ROWS as u64);
        assert_eq!(source.table_payload().unwrap().batches(), &original);
        drop(operator);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[derive(Default)]
struct BinarySink {
    rows: usize,
    batches: u64,
    fail: bool,
}

#[async_trait]
impl StreamCollector for BinarySink {
    async fn emit(&mut self, _: &str, batch: Batch) -> Result<()> {
        if self.fail && self.batches > 0 {
            return Err(operator_error("match", "binary sink failure"));
        }
        assert_eq!(batch.metadata().sequence(), self.batches);
        for record in batch.table_payload()?.batches() {
            let values = record
                .column(0)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();
            for row in 0..record.num_rows() {
                let expected = u8::try_from((self.rows + row) % 5).unwrap();
                assert_eq!(values.value(row).len(), WIDTH);
                assert!(values.value(row).iter().all(|&byte| byte == expected));
            }
            self.rows += record.num_rows();
        }
        self.batches += 1;
        Ok(())
    }
}

#[test]
fn test_morsel_fixed_binary_fallback_errors_preserve_state_and_refund() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    for failure in ["match", "row", "sink"] {
        runtime.block_on(failure_case(&service, failure));
    }
    drop(runtime);
    service.shutdown();
}

async fn failure_case(service: &TestService, failure: &str) {
    let pool = Pool::new(0);
    let mut operator = binary_operator();
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    let job = job().with_gather_owner(service.owner("binary-failure".into()));
    preload_binary(&mut operator, &job).await;
    if failure == "match" {
        operator.spec.limits.max_matches_per_input_batch = 399;
        operator.spec.limits.max_state_rows_per_side = 10;
    }
    let mut context = StreamOperatorContext::new(&job, "match", None);
    if failure == "row" {
        context = context.with_output_budget(EdgeBudget::new(10_000, WIDTH - 1).unwrap());
    }
    let source = sparse_input();
    let original = source.table_payload().unwrap().batches().to_vec();
    let mut sink = BinarySink {
        fail: true,
        ..BinarySink::default()
    };
    let error = operator
        .process_data("left", source.clone(), &context, &mut sink)
        .await
        .unwrap_err();
    let expected = match failure {
        "match" => "input batch match limit exceeded",
        "row" => "byte",
        "sink" => "sink failure",
        _ => unreachable!(),
    };
    assert!(error.to_string().contains(expected), "{failure}: {error}");
    assert_eq!(
        operator.status().match_limit_failures,
        u64::from(failure == "match")
    );
    assert_eq!(operator.status().state_limit_failures, 0);
    assert_eq!(sink.batches, u64::from(failure == "sink"));
    assert_eq!(operator.state.next_output_sequence, sink.batches);
    assert_eq!(operator.state.next_left_row_id, 0);
    assert_eq!(operator.status().left.retained_rows, 0);
    assert_eq!(operator.status().right.retained_rows, 5);
    assert_eq!(operator.status().emitted_match_rows, 0);
    assert_eq!(source.table_payload().unwrap().batches(), &original);
    assert_eq!(pool.attempts.load(Ordering::SeqCst), 0);
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
