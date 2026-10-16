use super::{support::*, *};
use datafusion::arrow::array::{Array, ArrayRef, Float64Array};
use datafusion::execution::memory_pool::MemoryPool;
use std::sync::atomic::Ordering;

fn wide_operator(columns: usize) -> StreamJoinOperator {
    let mut fields = right_schema().fields()[..2].to_vec();
    fields.extend((0..columns).map(|index| {
        Arc::new(Field::new(
            format!("nullable_{index}"),
            DataType::Float64,
            true,
        ))
    }));
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(100_000, 100_000_000, 1_000_000).unwrap();
    StreamJoinOperator::new(
        "match",
        left_schema(),
        Arc::new(Schema::new(fields)),
        declaration,
    )
    .unwrap()
}

async fn restore_wide(operator: &mut StreamJoinOperator, job: &StreamJobContext) {
    let schema = operator.input_schema(1);
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![7; 5])),
        Arc::new(TimestampMicrosecondArray::from(vec![0; 5]).with_timezone("UTC")),
    ];
    columns.extend(
        (2..schema.fields().len()).map(|_| Arc::new(Float64Array::from(vec![None; 5])) as ArrayRef),
    );
    let record = RecordBatch::try_new(Arc::clone(schema), columns).unwrap();
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
    let status = operator.status();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    operator.restore(&snapshot).unwrap();
    assert_eq!(operator.status(), status);
    assert!(matches!(
        operator.state.right.payload(0).owner(),
        Some(columnar::RowPayload::Shared { .. } | columnar::RowPayload::RestoredV2 { .. })
    ));
}

#[test]
fn test_morsel_restored_wide_nullable_backing_requires_safe_fallback() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let pool = Pool::new(0);
        let mut operator = wide_operator(50);
        operator
            .runtime
            .initialize_benchmark_pool(pool.clone())
            .unwrap();
        let job = job().with_gather_owner(service.owner("nullable-result".into()));
        restore_wide(&mut operator, &job).await;
        let context = StreamOperatorContext::new(&job, "match", None);
        let source = input(true, ROWS, false);
        let (mut prepared, expiration) = operator
            .prepare_data_append("left", &source, &context)
            .await
            .unwrap();
        assert!(prepared.deferred.is_some(), "must exercise fused gather");
        operator
            .finish_morsels(&mut prepared, &context)
            .await
            .unwrap();
        if let Some(gathered) = &prepared.gathered {
            let funding = gathered.funding_for_test();
            println!("restored nullable result backing/lease per unit: {funding:?}");
            assert!(funding.iter().all(|&(backing, lease)| backing <= lease));
        }
        assert!(
            prepared.gathered.is_none(),
            "physical output exceeds 16 MiB/unit"
        );
        assert_eq!(prepared.output.len(), ROWS * 5);
        assert_eq!(
            pool.attempts.load(Ordering::SeqCst),
            1,
            "only scratch was admitted"
        );
        assert_eq!(pool.live.load(Ordering::SeqCst), 0);
        let mut sink = CheckedSink::new(pool.clone(), 0, false, false);
        operator
            .finish_data("left", prepared, expiration, &context, &mut sink)
            .await
            .unwrap();
        assert_eq!(sink.rows, ROWS * 5);
        assert_eq!(operator.status().left.retained_rows, ROWS as u64);
        drop(operator);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

struct CheckedSink {
    pool: Arc<Pool>,
    lease: usize,
    projected: bool,
    fail: bool,
    rows: usize,
    batches: u64,
}

impl CheckedSink {
    fn new(pool: Arc<Pool>, lease: usize, projected: bool, fail: bool) -> Self {
        Self {
            pool,
            lease,
            projected,
            fail,
            rows: 0,
            batches: 0,
        }
    }

    fn check_record(&self, record: &RecordBatch) {
        let amount = record
            .column(if self.projected { 1 } else { 2 })
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..record.num_rows() {
            assert_eq!(
                amount.value(row),
                i64::try_from((self.rows + row) / 5).unwrap()
            );
        }
        for column in record.columns() {
            if column.data_type() == &DataType::Float64 {
                assert_eq!(column.null_count(), record.num_rows());
            }
        }
    }
}

#[async_trait]
impl StreamCollector for CheckedSink {
    async fn emit(&mut self, _: &str, batch: Batch) -> Result<()> {
        assert_eq!(self.pool.live.load(Ordering::SeqCst), self.lease);
        tokio::task::yield_now().await;
        assert_eq!(self.pool.live.load(Ordering::SeqCst), self.lease);
        if self.fail && self.batches == 1 {
            return Err(operator_error(
                "match",
                "nullable result downstream failure",
            ));
        }
        assert_eq!(batch.metadata().sequence(), self.batches);
        for record in batch.table_payload()?.batches() {
            self.check_record(record);
            self.rows += record.num_rows();
        }
        self.batches += 1;
        Ok(())
    }
}

#[test]
fn test_morsel_restored_nullable_results_cover_backing_until_success_or_failure() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    for projected in [false, true] {
        for fail in [false, true] {
            runtime.block_on(funded_case(&service, projected, fail, 0));
        }
    }
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_morsel_restored_nullable_optional_refusals_fall_back_and_refund() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    for reject_at in [1, 2] {
        runtime.block_on(funded_case(&service, true, false, reject_at));
    }
    drop(runtime);
    service.shutdown();
}

async fn funded_case(service: &TestService, projected: bool, fail: bool, reject_at: usize) {
    let pool = Pool::new(reject_at);
    let mut operator = wide_operator(if projected { 50 } else { 2 });
    if projected {
        operator.set_output_projection(vec![54, 2, 54, 3]).unwrap();
    }
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    let job = job().with_gather_owner(service.owner("nullable-funded".into()));
    restore_wide(&mut operator, &job).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let source = input(true, ROWS, false);
    let original = source.table_payload().unwrap().batches().to_vec();
    let (mut prepared, expiration) = operator
        .prepare_data_append("left", &source, &context)
        .await
        .unwrap();
    assert!(prepared.deferred.is_some());
    operator
        .finish_morsels(&mut prepared, &context)
        .await
        .unwrap();
    let lease = check_funding(&prepared, reject_at);
    assert_eq!(
        pool.live.load(Ordering::SeqCst),
        lease,
        "scratch already refunded"
    );
    let mut sink = CheckedSink::new(pool.clone(), lease, projected, fail);
    let result = operator
        .finish_data("left", prepared, expiration, &context, &mut sink)
        .await;
    check_delivery(&operator, &sink, result, fail);
    assert_eq!(source.table_payload().unwrap().batches(), &original);
    assert_eq!(pool.live.load(Ordering::SeqCst), 0);
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn check_funding(prepared: &PreparedJoinBatch, reject_at: usize) -> usize {
    assert_eq!(prepared.gathered.is_none(), reject_at != 0);
    let Some(gathered) = &prepared.gathered else {
        return 0;
    };
    let funding = gathered.funding_for_test();
    assert_eq!(funding.len(), 4);
    assert!(
        funding
            .iter()
            .all(|&(backing, lease)| backing <= lease && lease <= 16 << 20)
    );
    println!("nullable output backing/lease per unit: {funding:?}");
    funding.iter().map(|(_, lease)| lease).sum()
}

fn check_delivery(
    operator: &StreamJoinOperator,
    sink: &CheckedSink,
    result: Result<()>,
    fail: bool,
) {
    if fail {
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("downstream failure")
        );
        assert_eq!(sink.batches, 1);
        assert_eq!(operator.status().left.retained_rows, 0);
    } else {
        result.unwrap();
        assert_eq!(sink.rows, ROWS * 5);
        assert_eq!(operator.status().left.retained_rows, ROWS as u64);
    }
    assert_eq!(operator.state.next_output_sequence, sink.batches);
}
