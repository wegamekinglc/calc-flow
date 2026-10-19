use super::*;
use crate::runtime::streaming::gather_work::TestService;
use crate::{EdgeBudget, EdgeCollector};
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryLimit, MemoryPool, MemoryReservation,
};
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Weak,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
    time::Duration,
};

const ROWS: usize = 262_144;

fn large_input(group_width: usize) -> RecordBatch {
    let values = (0..ROWS)
        .map(|row| match row % 31 {
            0 => None,
            1 => Some(f64::NAN),
            2 => Some(1e16),
            3 => Some(-1e16),
            4 if (row / group_width).is_multiple_of(7) => Some(f64::INFINITY),
            5 if (row / group_width).is_multiple_of(7) => Some(f64::NEG_INFINITY),
            _ => Some(f64::from(u32::try_from(row % 97).unwrap()) / 7.0),
        })
        .collect::<Float64Array>();
    RecordBatch::try_new(
        Arc::new(input_schema()),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(
                    (0..ROWS).map(|row| i64::try_from(row / group_width).unwrap()),
                )
                .with_timezone("UTC"),
            ),
            Arc::new(StringArray::from_iter_values((0..ROWS).map(|_| "A"))),
            Arc::new(StringArray::from(vec![None::<&str>; ROWS])),
            Arc::new(UInt64Array::from_iter_values(
                (0..ROWS).map(|row| u64::try_from(row).unwrap()),
            )),
            Arc::new(values),
        ],
    )
    .unwrap()
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn job(service: &TestService) -> crate::StreamJobContext {
    mean_job().with_gather_owner(service.owner("cross-section-mean".into()))
}

#[derive(Default)]
struct Gate {
    released: parking_lot::Mutex<bool>,
    changed: parking_lot::Condvar,
}

impl Gate {
    fn wait(&self) {
        let mut released = self.released.lock();
        while !*released {
            assert!(
                !self
                    .changed
                    .wait_for(&mut released, Duration::from_secs(10))
                    .timed_out()
            );
        }
    }
    fn release(&self) {
        *self.released.lock() = true;
        self.changed.notify_all();
    }
}

struct Release(Arc<Gate>);
impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}

fn watch(
    operator: &mut CrossSectionOperator,
    gate: &Arc<Gate>,
) -> tokio::sync::mpsc::UnboundedReceiver<usize> {
    let (entered, receiver) = tokio::sync::mpsc::unbounded_channel();
    let gate = gate.clone();
    operator.mean_test_hook = Some(Arc::new(move |ordinal, complete| {
        if !complete {
            entered.send(ordinal).unwrap();
            gate.wait();
        }
    }));
    assert!(operator.mean_test_hook.is_some());
    receiver
}

async fn started<F: Future<Output = Result<()>>>(
    flush: &mut Pin<Box<F>>,
    receiver: &mut tokio::sync::mpsc::UnboundedReceiver<usize>,
    count: usize,
) -> (Vec<usize>, Option<Result<()>>) {
    let mut units = Vec::new();
    while units.len() < count {
        tokio::select! {
            result = flush.as_mut() => return (units, Some(result)),
            ordinal = receiver.recv() => match ordinal {
                Some(ordinal) => units.push(ordinal),
                None => return (units, None),
            },
            () = tokio::time::sleep(Duration::from_secs(3)) => return (units, None),
        }
    }
    (units, None)
}

fn reference(
    input: &RecordBatch,
    operator: &CrossSectionOperator,
    context: &StreamOperatorContext<'_>,
) -> Vec<Batch> {
    let record = build_columnar_mean_record(
        input,
        &operator.compiled,
        operator.output_ports[0].schema().unwrap(),
        &operator.name,
    )
    .unwrap();
    chunk_output_record(&record, &operator.name, 0, context.output_budget()).unwrap()
}

fn verify(output: &mut EdgeCollector, expected: &[Batch]) {
    let actual = output.drain("output");
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        let actual = actual.as_data().unwrap();
        assert_eq!(actual.metadata(), expected.metadata());
        for (actual, expected) in actual
            .table_payload()
            .unwrap()
            .batches()
            .iter()
            .zip(expected.table_payload().unwrap().batches())
        {
            assert_eq!(actual.num_rows(), expected.num_rows());
            assert_eq!(actual.schema(), expected.schema());
            let mean_column = actual.num_columns() - 1;
            for column in 0..mean_column {
                let actual = actual.column(column).to_data();
                let expected = expected.column(column).to_data();
                assert_eq!(actual.offset(), expected.offset());
                assert_eq!(actual.null_count(), expected.null_count());
                for (actual, expected) in actual.buffers().iter().zip(expected.buffers()) {
                    assert_eq!(
                        actual.as_ptr(),
                        expected.as_ptr(),
                        "payload buffers stay shared"
                    );
                }
            }
            let actual = actual
                .column(mean_column)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let expected = expected
                .column(mean_column)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for row in 0..actual.len() {
                assert_eq!(actual.is_valid(row), expected.is_valid(row));
                if actual.is_valid(row) {
                    assert_eq!(actual.value(row).to_bits(), expected.value(row).to_bits());
                }
            }
        }
    }
}

#[test]
fn stream_cpu_cross_section_many_groups_overlap_and_preserve_float_bits_and_metadata() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = mean_operator();
        let job = job(&service);
        let context = StreamOperatorContext::new(&job, "features", None)
            .with_output_budget(EdgeBudget::new(7_000, 8 << 20).unwrap());
        let input = large_input(1_024);
        let expected = reference(&input, &operator, &context);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let release = Release(Arc::new(Gate::default()));
        let mut receiver = watch(&mut operator, &release.0);
        operator
            .process_data("input", mean_batch(input), &context, &mut output)
            .await
            .unwrap();
        let mut flush = Box::pin(operator.on_end(&context, &mut output));
        let (mut units, early) = started(&mut flush, &mut receiver, 4).await;
        drop(release);
        let result = match early {
            Some(result) => result,
            None => flush.as_mut().await,
        };
        drop(flush);
        result.unwrap();
        verify(&mut output, &expected);
        assert!(operator.state.ended);
        assert!(operator.state.pending_mean.is_none());
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        units.sort_unstable();
        assert_eq!(
            units,
            [0, 1, 2, 3],
            "complete groups must run concurrently on managed workers"
        );
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn stream_cpu_cross_section_single_huge_group_preserves_serial_accumulation() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = mean_operator();
        let job = job(&service);
        let context = StreamOperatorContext::new(&job, "features", None);
        let input = large_input(ROWS);
        let expected = reference(&input, &operator, &context);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        operator.mean_test_hook = Some(Arc::new(move |_, _| {
            observed.fetch_add(1, AtomicOrdering::SeqCst);
        }));
        assert!(operator.mean_test_hook.is_some());
        operator
            .process_data("input", mean_batch(input), &context, &mut output)
            .await
            .unwrap();
        operator.on_end(&context, &mut output).await.unwrap();
        verify(&mut output, &expected);
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 0);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
    });
    drop(runtime);
    service.shutdown();
}

#[derive(Debug)]
struct ObservedPool {
    inner: GreedyMemoryPool,
    attempts: AtomicUsize,
    work_live: AtomicUsize,
}

impl ObservedPool {
    fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: GreedyMemoryPool::new(limit),
            attempts: AtomicUsize::new(0),
            work_live: AtomicUsize::new(0),
        })
    }
    fn work(credit: &MemoryReservation) -> bool {
        credit
            .consumer()
            .name()
            .ends_with("cross-section-mean-work")
    }
}

impl std::fmt::Display for ObservedPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, formatter)
    }
}

impl MemoryPool for ObservedPool {
    fn name(&self) -> &'static str {
        "cross-section-mean-observer"
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn grow(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.grow(credit, bytes);
        if Self::work(credit) {
            self.work_live.fetch_add(bytes, AtomicOrdering::SeqCst);
        }
    }
    fn shrink(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.shrink(credit, bytes);
        if Self::work(credit) {
            self.work_live.fetch_sub(bytes, AtomicOrdering::SeqCst);
        }
    }
    fn try_grow(&self, credit: &MemoryReservation, bytes: usize) -> datafusion::common::Result<()> {
        if Self::work(credit) {
            self.attempts.fetch_add(1, AtomicOrdering::SeqCst);
        }
        self.inner.try_grow(credit, bytes)?;
        if Self::work(credit) {
            self.work_live.fetch_add(bytes, AtomicOrdering::SeqCst);
        }
        Ok(())
    }
}

fn pool(operator: &mut CrossSectionOperator, limit: usize) -> Arc<ObservedPool> {
    let pool = ObservedPool::new(limit);
    operator.mean_test_pool = Some(pool.clone());
    assert!(operator.mean_test_pool.is_some());
    pool
}

#[test]
fn stream_cpu_cross_section_finite_pool_refusal_preserves_serial_output_and_refunds() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    for (limit, expected_workers) in [(4 << 20, 0), (16 << 20, 4)] {
        runtime.block_on(refused_case(&service, limit, expected_workers));
    }
    drop(runtime);
    service.shutdown();
}

async fn refused_case(service: &TestService, limit: usize, expected_workers: usize) {
    let mut operator = mean_operator();
    let pool = pool(&mut operator, limit);
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    operator.mean_test_hook = Some(Arc::new(move |_, complete| {
        if !complete {
            observed.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }));
    let job = job(service);
    let context = StreamOperatorContext::new(&job, "features", None)
        .with_output_budget(EdgeBudget::new(1_000, 64 << 20).unwrap());
    let input = large_input(1_024);
    let expected = reference(&input, &operator, &context);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("input", mean_batch(input), &context, &mut output)
        .await
        .unwrap();
    operator.on_end(&context, &mut output).await.unwrap();
    verify(&mut output, &expected);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(operator);
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(calls.load(AtomicOrdering::SeqCst), expected_workers);
    assert!(
        pool.attempts.load(AtomicOrdering::SeqCst) > 0,
        "parallel group work must ask the real pool before allocation"
    );
}

#[test]
fn stream_cpu_cross_section_emitter_failure_retains_pending_groups_for_retry() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = mean_operator();
        let pool = pool(&mut operator, 1 << 30);
        let job = job(&service);
        let context = StreamOperatorContext::new(&job, "features", None);
        let input = large_input(1_024);
        let expected = reference(&input, &operator, &context);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("input", mean_batch(input), &context, &mut output)
            .await
            .unwrap();
        let charge = operator.state.charge;
        let mut rejected = RejectOutput;
        assert!(
            operator
                .on_watermark(EventTime::from_micros(1_000), &context, &mut rejected)
                .await
                .is_err()
        );
        assert!(operator.state.pending_mean.is_some());
        assert_eq!(operator.state.charge.rows, charge.rows);
        assert_eq!(operator.state.charge.bytes, charge.bytes);
        assert_eq!(operator.state.next_output_sequence, 0);
        assert_eq!(operator.state.last_input_watermark, None);
        operator.on_end(&context, &mut output).await.unwrap();
        verify(&mut output, &expected);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(operator);
        drop(job);
        assert_eq!(pool.reserved(), 0);
        assert!(
            pool.attempts.load(AtomicOrdering::SeqCst) > 0,
            "retry exercises funded group workers"
        );
    });
    drop(runtime);
    service.shutdown();
}

async fn small_snapshot() -> crate::OperatorStateSnapshot {
    let mut operator = mean_operator();
    let job = mean_job();
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            mean_batch(mean_input_record()),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    StreamOperator::checkpoint(&mut operator, Epoch::INITIAL).unwrap()
}

#[test]
fn stream_cpu_cross_section_cancel_and_abandon_keep_worker_input_funded_until_settlement() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    for cancel in [false, true] {
        runtime.block_on(stopped_case(&service, cancel));
    }
    drop(runtime);
    service.shutdown();
}

async fn stopped_case(service: &TestService, cancel: bool) {
    let snapshot = small_snapshot().await;
    let mut operator = mean_operator();
    let pool = pool(&mut operator, 1 << 30);
    let job = job(service);
    let context = StreamOperatorContext::new(&job, "features", None);
    let input = large_input(1_024);
    let weak: Weak<dyn Array> = Arc::downgrade(input.column(4));
    let backing = input.column(4).get_array_memory_size();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let release = Release(Arc::new(Gate::default()));
    let mut receiver = watch(&mut operator, &release.0);
    operator
        .process_data("input", mean_batch(input), &context, &mut output)
        .await
        .unwrap();
    let mut flush = Box::pin(operator.on_end(&context, &mut output));
    let (units, early) = started(&mut flush, &mut receiver, 1).await;
    if cancel && early.is_none() {
        job.cancellation().cancel();
        let result = tokio::time::timeout(Duration::from_secs(3), flush.as_mut())
            .await
            .unwrap();
        assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    }
    drop(flush);
    assert_eq!(
        units.len(),
        1,
        "a managed group worker must start before stopping the owner"
    );
    assert!(output.drain("output").is_empty());
    assert_eq!(operator.state.next_output_sequence, 0);
    assert!(operator.state.pending_mean.is_some());
    if cancel {
        StreamOperator::restore(&mut operator, &snapshot).unwrap();
    } else {
        StreamOperator::reset(&mut operator).unwrap();
    }
    assert!(
        weak.upgrade().is_some(),
        "paused worker still owns the shared numeric array"
    );
    assert!(
        pool.work_live.load(AtomicOrdering::SeqCst) >= backing,
        "worker input and results stay prepaid after the owner future is dropped"
    );
    drop(release);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert!(weak.upgrade().is_none());
    if cancel {
        let next_job = mean_job().with_gather_owner(service.owner("restored-mean".into()));
        let next = StreamOperatorContext::new(&next_job, "features", None);
        operator.on_end(&next, &mut output).await.unwrap();
        assert_mean_output(&mut output, &[2.0, 2.0, 5.0, 5.0]);
        assert!(next_job.gather_owner().close_and_drain().await.is_empty());
    }
    drop(operator);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

struct ChainedCollector<'a> {
    next: CrossSectionOperator,
    context: StreamOperatorContext<'a>,
    output: EdgeCollector,
    expected: Vec<Batch>,
}

#[async_trait]
impl StreamCollector for ChainedCollector<'_> {
    async fn emit(&mut self, port: &str, batch: Batch) -> Result<()> {
        assert_eq!(port, "output");
        assert!(self.expected.is_empty(), "fixture emits one upstream chunk");
        let input = &batch.table_payload()?.batches()[0];
        self.expected = reference(input, &self.next, &self.context);
        self.next
            .process_data("input", batch, &self.context, &mut self.output)
            .await?;
        self.next.on_end(&self.context, &mut self.output).await
    }
}

#[test]
fn stream_cpu_cross_section_downstream_can_use_same_job_workers_during_upstream_emit() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = mean_operator();
        let job = job(&service);
        let budget = EdgeBudget::new(ROWS, 64 << 20).unwrap();
        let context = StreamOperatorContext::new(&job, "features", None).with_output_budget(budget);
        let mut next_spec = valid_mean_spec();
        let CrossSectionOutputSpec::Mean { output, .. } = &mut next_spec.outputs[0] else {
            unreachable!();
        };
        *output = "second_mean".into();
        let mut next = CrossSectionOperator::new(
            "next",
            operator.output_ports[0].schema().unwrap().clone(),
            next_spec,
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let next_calls = Arc::new(AtomicUsize::new(0));
        for (operator, calls) in [(&mut operator, &calls), (&mut next, &next_calls)] {
            let calls = calls.clone();
            operator.mean_test_hook = Some(Arc::new(move |_, complete| {
                if !complete {
                    calls.fetch_add(1, AtomicOrdering::SeqCst);
                }
            }));
        }
        let mut collector = ChainedCollector {
            output: EdgeCollector::new(next.output_ports().to_vec()),
            next,
            context: StreamOperatorContext::new(&job, "next", None).with_output_budget(budget),
            expected: Vec::new(),
        };
        operator
            .process_data(
                "input",
                mean_batch(large_input(1_024)),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        tokio::time::timeout(
            Duration::from_secs(3),
            operator.on_end(&context, &mut collector),
        )
        .await
        .expect("upstream must release its worker attempt before awaiting downstream")
        .unwrap();
        verify(&mut collector.output, &collector.expected);
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 4);
        assert_eq!(next_calls.load(AtomicOrdering::SeqCst), 4);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
    });
    drop(runtime);
    service.shutdown();
}
