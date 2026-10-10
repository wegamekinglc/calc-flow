use super::*;
use crate::runtime::streaming::gather_work::{
    TestService,
    admission_probe::{AdmissionEvent, AdmissionProbe, AdmissionStage},
};
use datafusion::arrow::{array::NullArray, buffer::Buffer};
use datafusion::execution::memory_pool::MemoryPool;
use std::sync::Mutex;

const CHUNK_ROWS: usize = 8_192;
type Observations = Arc<Mutex<Vec<(usize, bool)>>>;

pub(super) fn run_lifetime_subject() {
    run(|runtime, service| {
        runtime
            .block_on(assert_blocked_chunk_cancellation_restores_the_last_committed_checkpoint());
        runtime.block_on(assert_lifetime_and_prefix(service));
    });
}

pub(super) fn run_refusal_subject() {
    run(|runtime, service| {
        runtime.block_on(output_gather_tests::assert_same_parent_output_takes_each_column_once_in_both_directions());
        runtime.block_on(assert_current_chunk_refusal(service));
    });
}

fn run(subject: impl FnOnce(&tokio::runtime::Runtime, &TestService)) {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    subject(&runtime, &service);
    drop(runtime);
    service.shutdown();
}

fn declaration() -> StreamJoinSpec {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(20_000, 10_000_000, 16_384).unwrap();
    declaration
}

fn observe(operator: &mut StreamJoinOperator, run: &StreamJobContext) -> Observations {
    let observations = Arc::new(Mutex::new(Vec::new()));
    let recorded = observations.clone();
    let actor = std::thread::current().id();
    let owner = run.gather_owner().clone();
    operator.materialize_unit_test_hook = Some(Arc::new(move |ordinal, complete| {
        assert_ne!(std::thread::current().id(), actor);
        assert!(owner.funding().2 > 0, "actual paid materialization attempt");
        recorded.lock().unwrap().push((ordinal, complete));
    }));
    observations
}

fn assert_chunk(batch: &Batch, sequence: u64) {
    assert_eq!(batch.metadata().sequence(), sequence);
    let record = &batch.table_payload().unwrap().batches()[0];
    assert_eq!(record.num_rows(), CHUNK_ROWS);
    assert_eq!(
        record.column(0).to_data(),
        Int64Array::from(vec![7; CHUNK_ROWS]).to_data()
    );
    assert_eq!(
        record.column(2).to_data(),
        Int64Array::from(vec![42; CHUNK_ROWS]).to_data()
    );
    assert_eq!(
        record.column(5).to_data(),
        StringArray::from(vec!["paid"; CHUNK_ROWS]).to_data()
    );
    assert_eq!(
        record.column(4).to_data(),
        TimestampMicrosecondArray::from(
            (0..CHUNK_ROWS)
                .map(|row| i64::try_from(row).unwrap())
                .collect::<Vec<_>>()
        )
        .with_timezone("UTC")
        .to_data(),
        "canonical pair order must survive ordinal fragment merge"
    );
    assert_eq!(
        record.schema().field(1).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    );
}

fn assert_settled(operator: &StreamJoinOperator, run: &StreamJobContext) {
    assert_eq!(run.gather_owner().funding().2, 0);
    assert!(operator.compaction_release.is_none());
    assert!(operator.compaction_cleanup.is_none());
}

async fn preload(operator: &mut StreamJoinOperator, context: &StreamOperatorContext<'_>) {
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            right_batch(
                (0..CHUNK_ROWS)
                    .map(|row| i64::try_from(row).unwrap())
                    .collect(),
            ),
            context,
            &mut output,
        )
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
}

async fn assert_output_lifetime(service: &TestService) {
    let run = job().with_gather_owner(service.owner("materialized-lifetime".into()));
    let context = StreamOperatorContext::new(&run, "match", None)
        .with_output_budget(EdgeBudget::new(CHUNK_ROWS, 8 << 20).unwrap());
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration()).unwrap();
    preload(&mut operator, &context).await;
    let observed = observe(&mut operator, &run);
    let takes = Arc::new(Mutex::new(Vec::new()));
    let recorded = takes.clone();
    operator.materialize_take_test_hook = Some(Arc::new(move |column, rows| {
        recorded.lock().unwrap().push((column, rows));
    }));
    let attempts_before = run.gather_owner().attempt_sequence();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![0]), &context, &mut output)
        .await
        .unwrap();
    let attempts = run.gather_owner().attempt_sequence() - attempts_before;
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    let batch = messages[0].as_data().unwrap();
    assert_chunk(batch, 0);
    let array = batch.table_payload().unwrap().batches()[0]
        .column(2)
        .clone();
    let buffer: Buffer = array.to_data().buffers()[0].clone();
    assert!(!buffer.is_empty());
    assert_eq!(operator.state.next_left_row_id, 1);
    assert_eq!(operator.state.metrics.emitted_match_rows, 8_192);
    assert_settled(&operator, &run);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((messages, output, operator, context));
    assert!(
        tokio::time::timeout(Duration::from_secs(5), run.gather_owner().close_and_drain())
            .await
            .unwrap()
            .is_empty()
    );
    drop(run);
    let held = pool.reserved();
    assert_eq!(
        array
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(CHUNK_ROWS - 1),
        42
    );
    drop(array);
    assert_eq!(
        pool.reserved(),
        held,
        "the nonempty Buffer independently owns output funding"
    );
    assert_eq!(&buffer.as_slice()[..8], &42_i64.to_ne_bytes());
    drop(buffer);
    assert_eq!(pool.reserved(), 0);
    let mut completed = observed
        .lock()
        .unwrap()
        .iter()
        .filter_map(|&(ordinal, done)| done.then_some(ordinal))
        .collect::<Vec<_>>();
    completed.sort_unstable();
    assert_eq!(
        completed,
        [0, 1],
        "one canonical chunk requires two actual paid materialization units"
    );
    assert!(
        held > 0,
        "escaping Array/Buffer clones must keep output resident credit after shutdown"
    );
    let mut actual_takes = takes.lock().unwrap().clone();
    actual_takes.sort_unstable();
    assert_eq!(
        (actual_takes, attempts),
        (
            (0..6)
                .map(|column| (column, CHUNK_ROWS))
                .collect::<Vec<_>>(),
            1
        ),
        "one canonical chunk takes every column once with one actual worker attempt"
    );
}

struct PrefixCollector {
    accepted: Vec<Batch>,
    cancel: CancellationToken,
}

#[async_trait]
impl StreamCollector for PrefixCollector {
    async fn emit(&mut self, _port: &str, batch: Batch) -> Result<()> {
        if self.accepted.is_empty() {
            self.accepted.push(batch);
            return Ok(());
        }
        self.cancel.cancel();
        std::future::pending().await
    }
}

async fn assert_cancelled_prefix(service: &TestService) {
    let run = job().with_gather_owner(service.owner("materialized-prefix".into()));
    let context = StreamOperatorContext::new(&run, "match", None)
        .with_output_budget(EdgeBudget::new(CHUNK_ROWS, 8 << 20).unwrap());
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration()).unwrap();
    preload(&mut operator, &context).await;
    let observed = observe(&mut operator, &run);
    let mut collector = PrefixCollector {
        accepted: Vec::new(),
        cancel: run.cancellation().clone(),
    };
    tokio::time::timeout(Duration::from_secs(5), async {
        tokio::select! {
            () = run.cancellation().cancelled() => {},
            result = operator.process_data("left", left_batch(vec![0; 2]), &context, &mut collector) => panic!("second canonical chunk must pause: {result:?}"),
        }
    }).await.unwrap();
    assert_eq!(collector.accepted.len(), 1);
    assert_chunk(&collector.accepted[0], 0);
    assert_eq!(operator.state.next_output_sequence, 1);
    assert_eq!(operator.state.next_left_row_id, 0);
    assert_eq!(operator.state.left.len(), 0);
    assert_eq!(operator.state.metrics.emitted_match_rows, 0);
    assert_settled(&operator, &run);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((operator, context));
    assert!(run.gather_owner().close_and_drain().await.is_empty());
    drop((collector, run));
    assert_eq!(pool.reserved(), 0);
    assert_eq!(
        observed
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, done)| *done)
            .count(),
        4
    );
}

async fn assert_bufferless_serial(service: &TestService) {
    let schema = Arc::new(Schema::new(vec![
        left_schema().field(0).clone(),
        left_schema().field(1).clone(),
        Field::new("amount", DataType::Null, true),
    ]));
    let run = job().with_gather_owner(service.owner("materialized-bufferless".into()));
    let context = StreamOperatorContext::new(&run, "match", None);
    let mut operator =
        StreamJoinOperator::new("match", schema.clone(), right_schema(), declaration()).unwrap();
    preload(&mut operator, &context).await;
    let observed = observe(&mut operator, &run);
    let input = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
            Arc::new(NullArray::new(1)),
        ],
    )
    .unwrap();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            Batch::table(vec![input], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].as_data().unwrap().num_rows(), CHUNK_ROWS);
    assert_eq!(
        messages[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0]
            .column(2)
            .data_type(),
        &DataType::Null
    );
    assert!(observed.lock().unwrap().is_empty());
    assert_settled(&operator, &run);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((messages, output, operator, context));
    assert!(run.gather_owner().close_and_drain().await.is_empty());
    drop(run);
    assert_eq!(pool.reserved(), 0);
}

async fn assert_lifetime_and_prefix(service: &TestService) {
    assert_output_lifetime(service).await;
    tokio::time::timeout(
        Duration::from_secs(5),
        assert_abandoned_materialization(service),
    )
    .await
    .unwrap();
    assert_cancelled_prefix(service).await;
    assert_bufferless_serial(service).await;
}

struct WorkerGate {
    release: Option<std::sync::mpsc::Sender<()>>,
}

impl Drop for WorkerGate {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
    }
}

fn worker_gate(
    operator: &mut StreamJoinOperator,
) -> (
    WorkerGate,
    tokio::sync::oneshot::Receiver<std::thread::ThreadId>,
) {
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let gate = Mutex::new(Some((entered, wait)));
    operator.materialize_unit_test_hook = Some(Arc::new(move |ordinal, complete| {
        if ordinal == 0 && !complete {
            let (entered, wait) = gate.lock().unwrap().take().unwrap();
            let _ = entered.send(std::thread::current().id());
            let _ = wait.recv_timeout(Duration::from_secs(10));
        }
    }));
    (
        WorkerGate {
            release: Some(release),
        },
        started,
    )
}

async fn assert_abandoned_materialization(service: &TestService) {
    let run = job().with_gather_owner(service.owner("materialized-abandoned".into()));
    let context = StreamOperatorContext::new(&run, "match", None)
        .with_output_budget(EdgeBudget::new(CHUNK_ROWS, 8 << 20).unwrap());
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration()).unwrap();
    preload(&mut operator, &context).await;
    let weak = Arc::downgrade(operator.state.right[0].record.column(0));
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let (gate, started) = worker_gate(&mut operator);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let mut process =
        Box::pin(operator.process_data("left", left_batch(vec![0]), &context, &mut output));
    let thread = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::select! {
            entered = started => entered.unwrap(),
            result = &mut process => panic!("materialization completed before worker gate: {result:?}"),
        }
    }).await.unwrap();
    assert_ne!(thread, std::thread::current().id());
    run.cancellation().cancel();
    assert!(matches!(
        process.as_mut().await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    drop(process);
    assert!(output.drain("output").is_empty());
    assert_eq!(operator.state.next_left_row_id, 0);
    assert!(operator.compaction_release.is_some());
    assert!(operator.compaction_cleanup.is_some());
    assert!(run.gather_owner().funding().2 > 0);
    operator.reset().unwrap();
    assert!(
        weak.upgrade().is_some(),
        "worker owns the parent after actor reset"
    );
    let resumed_run = job().with_gather_owner(run.gather_owner().clone());
    let resumed = StreamOperatorContext::new(&resumed_run, "match", None);
    let mut mutation =
        Box::pin(operator.process_data("right", right_batch(vec![0]), &resumed, &mut output));
    assert!(futures::poll!(mutation.as_mut()).is_pending());
    drop(mutation);
    assert_eq!(operator.state.next_right_row_id, 0);
    drop(gate);
    operator
        .process_data("right", right_batch(vec![0]), &resumed, &mut output)
        .await
        .unwrap();
    assert!(weak.upgrade().is_none());
    assert_settled(&operator, &resumed_run);
    operator
        .process_data("left", left_batch(vec![0]), &resumed, &mut output)
        .await
        .unwrap();
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].as_data().unwrap().num_rows(), 1);
    assert_eq!(messages[0].as_data().unwrap().metadata().sequence(), 0);
    assert_eq!(operator.state.next_left_row_id, 1);
    assert_eq!(operator.state.next_right_row_id, 1);
    drop((messages, output, operator, context, resumed));
    assert!(run.gather_owner().close_and_drain().await.is_empty());
    drop((run, resumed_run));
    assert_eq!(pool.reserved(), 0);
}

struct RefusalCollector {
    accepted: Vec<Batch>,
    owner: crate::runtime::streaming::gather_work::JobGatherOwner,
    pool: Arc<dyn MemoryPool>,
    probe: Option<Arc<AdmissionProbe>>,
    event: Option<AdmissionEvent>,
}

#[async_trait]
impl StreamCollector for RefusalCollector {
    async fn emit(&mut self, _port: &str, batch: Batch) -> Result<()> {
        if self.accepted.is_empty() {
            self.probe = Some(AdmissionProbe::install(
                &self.owner,
                AdmissionStage::Attempt,
                self.pool.clone(),
                1 << 30,
            ));
        } else {
            self.event = self.probe.as_ref().unwrap().take_event();
        }
        self.accepted.push(batch);
        Ok(())
    }
}

async fn assert_current_chunk_refusal(service: &TestService) {
    let run = job().with_gather_owner(service.owner("materialized-refusal".into()));
    let context = StreamOperatorContext::new(&run, "match", None)
        .with_output_budget(EdgeBudget::new(CHUNK_ROWS, 8 << 20).unwrap());
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration()).unwrap();
    preload(&mut operator, &context).await;
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let observed = observe(&mut operator, &run);
    let mut output = RefusalCollector {
        accepted: Vec::new(),
        owner: run.gather_owner().clone(),
        pool: pool.clone(),
        probe: None,
        event: None,
    };
    operator
        .process_data("left", left_batch(vec![0; 2]), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(output.accepted.len(), 2);
    for (sequence, batch) in output.accepted.iter().enumerate() {
        assert_chunk(batch, u64::try_from(sequence).unwrap());
    }
    assert_eq!(operator.state.next_output_sequence, 2);
    assert_eq!(operator.state.next_left_row_id, 2);
    assert_eq!(operator.state.left.len(), 2);
    assert_eq!(operator.state.metrics.emitted_match_rows, 16_384);
    assert_settled(&operator, &run);
    let event = output.event.take();
    let completed = observed
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, done)| *done)
        .count();
    drop((output, operator, context));
    assert!(run.gather_owner().close_and_drain().await.is_empty());
    drop(run);
    assert_eq!(pool.reserved(), 0);
    let event = event.expect("current canonical chunk must attempt paid materialization admission before serial fallback");
    assert_eq!(event.available + 1, event.fee);
    assert_eq!(event.operator, "match");
    assert_eq!(
        completed, 2,
        "only the accepted prefix used workers; refused chunk emitted once serially"
    );
}
