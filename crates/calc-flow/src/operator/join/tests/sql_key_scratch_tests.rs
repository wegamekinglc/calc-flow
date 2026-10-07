use super::*;
use crate::DataFusionRuntime;
use datafusion::{
    common::Result as DataFusionResult,
    datasource::memory::MemorySourceConfig,
    execution::TaskContext,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        RecordBatchStream, SendableRecordBatchStream, repartition::RepartitionExec,
    },
};
use futures::{Stream, StreamExt};
use parking_lot::{Condvar, Mutex};
use std::{
    fmt,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::sync::Notify;

#[derive(Debug, Default)]
struct ProducerGate {
    entered: Notify,
    dropping: Notify,
    finished: Notify,
    open: Mutex<bool>,
    release: Condvar,
}

impl ProducerGate {
    fn open(&self) {
        *self.open.lock() = true;
        self.release.notify_all();
    }

    fn hold_cleanup(&self) {
        self.dropping.notify_one();
        let mut open = self.open.lock();
        while !*open {
            self.release.wait(&mut open);
        }
    }
}

struct GateRelease(Arc<ProducerGate>);
impl Drop for GateRelease {
    fn drop(&mut self) {
        self.0.open();
    }
}

#[derive(Debug)]
struct ProducerInput {
    input: Arc<dyn ExecutionPlan>,
    gate: Arc<ProducerGate>,
    metadata_only: bool,
}

impl DisplayAs for ProducerInput {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JoinScratchProducerGate")
    }
}

impl ExecutionPlan for ProducerInput {
    fn name(&self) -> &'static str {
        "JoinScratchProducerGate"
    }

    fn downcast_delegate(&self) -> Option<&dyn ExecutionPlan> {
        Some(self.input.as_ref())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        assert_eq!(children.len(), 1);
        Ok(Arc::new(Self {
            input: children.remove(0),
            gate: Arc::clone(&self.gate),
            metadata_only: self.metadata_only,
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        Ok(Box::pin(ProducerStream {
            input: Some(self.input.execute(partition, context)?),
            held: None,
            schema: Some(self.input.schema()),
            gate: Arc::clone(&self.gate),
            metadata_only: self.metadata_only,
        }))
    }
}

struct ProducerStream {
    input: Option<SendableRecordBatchStream>,
    held: Option<RecordBatch>,
    schema: Option<SchemaRef>,
    gate: Arc<ProducerGate>,
    metadata_only: bool,
}

impl Stream for ProducerStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.held.is_none() && this.input.is_some() {
            match this.input.as_mut().unwrap().as_mut().poll_next(context) {
                Poll::Ready(Some(Ok(batch))) => {
                    if this.metadata_only {
                        drop(batch);
                        this.input = None;
                    } else {
                        this.held = Some(batch);
                    }
                    this.gate.entered.notify_one();
                }
                value => return value,
            }
        }
        Poll::Pending
    }
}

impl RecordBatchStream for ProducerStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(self.schema.as_ref().unwrap())
    }
}

impl Drop for ProducerStream {
    fn drop(&mut self) {
        self.gate.hold_cleanup();
        drop(self.input.take());
        drop(self.held.take());
        drop(self.schema.take());
        self.gate.finished.notify_one();
    }
}

async fn hold_cancelled_producer(record: RecordBatch, gate: &Arc<ProducerGate>) {
    hold_producer_kind(record, gate, false).await;
}

async fn hold_producer_kind(record: RecordBatch, gate: &Arc<ProducerGate>, metadata_only: bool) {
    let producer = tokio::spawn(gated_producer(record, Arc::clone(gate), metadata_only));
    tokio::time::timeout(Duration::from_secs(2), gate.entered.notified())
        .await
        .unwrap();
    producer.abort();
    assert!(producer.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(2), gate.dropping.notified())
        .await
        .unwrap();
}

async fn production_fallback_case(nonserial: bool) {
    let (mut operator, job) = scratch_fixture(1).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let plan = operator.side_plan("right").unwrap();
    let mut bundle = operator.admission_bundle(&plan);
    let input = right_batch(vec![0]);
    operator
        .admit_record(
            &input.table_payload().unwrap().batches()[0],
            &plan,
            "right",
            &context,
            &mut bundle,
            0,
        )
        .await
        .unwrap();
    let keys = operator.owned_state_keys(&plan, &context).await.unwrap();
    let paid_schema = Arc::downgrade(&keys.schema());
    assert!(keys.funding.is_some());
    if nonserial {
        operator.runtime.runtime = Some(
            DataFusionRuntime::new(DataFusionConfig {
                target_partitions: 2,
                min_rows_per_partition: 1,
                ..DataFusionConfig::default()
            })
            .unwrap(),
        );
    } else {
        operator.compiled.equality_query = parse_select_query(&format!(
            "{} ORDER BY {PROBE_POS_COLUMN} DESC",
            equality_query(1)
        ))
        .unwrap();
    }
    let before = operator.status();
    crate::datafusion::owned::observe_legacy_inputs();
    let (pairs, admitted) = operator
        .sql_key_pairs_owned(&plan, std::mem::take(&mut bundle.admitted), keys)
        .await;
    assert_eq!(pairs.unwrap(), vec![(0, 0)]);
    assert_eq!(operator.status(), before);
    let mut records = crate::datafusion::owned::take_legacy_inputs();
    let position = records
        .iter()
        .position(|record| record.schema().index_of(STATE_RID_COLUMN).is_ok())
        .expect("actual Legacy SQL registration was observed");
    let state = records.swap_remove(position);
    drop(records);
    let legacy_schema = Arc::downgrade(&state.schema());
    let gate = Arc::new(ProducerGate::default());
    let _release = GateRelease(Arc::clone(&gate));
    hold_producer_kind(state, &gate, true).await;
    drop(admitted);
    drop(bundle);
    drop(operator);
    assert!(
        legacy_schema.upgrade().is_some(),
        "real consumer keeps the registered Legacy schema"
    );
    assert!(
        paid_schema.upgrade().is_none(),
        "production Legacy fallback must not leak a new paid scratch schema to an unproved producer"
    );
    gate.open();
    tokio::time::timeout(Duration::from_secs(2), gate.finished.notified())
        .await
        .unwrap();
    assert!(legacy_schema.upgrade().is_none());
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_unproved_production_fallback_uses_true_legacy_schema() {
    production_fallback_case(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_nonserial_production_fallback_uses_true_legacy_schema() {
    production_fallback_case(true).await;
}

async fn collected_output_case(cancelled: bool) {
    let (mut operator, job) = scratch_fixture(1).await;
    let private_timezone = match operator.state.left[0].record.column(1).data_type() {
        DataType::Timestamp(_, Some(timezone)) => Arc::downgrade(timezone),
        _ => panic!("fixture carries a privately owned timezone"),
    };
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let input = right_batch(vec![0]);
    let before = input.table_payload().unwrap().batches()[0].clone();
    operator
        .process_data("right", input.clone(), &context, &mut collector)
        .await
        .unwrap();
    assert_eq!(input.table_payload().unwrap().batches()[0], before);
    let output = collector.drain("output")[0].as_data().unwrap().clone();
    assert_eq!(output.num_rows(), 1);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    if cancelled {
        job.cancellation().cancel();
    } else {
        operator.on_end(&context, &mut collector).await.unwrap();
        assert!(operator.status().left.ended);
    }
    drop(operator);
    drop(context);
    assert!(
        private_timezone.upgrade().is_none(),
        "caller output must use declared canonical types, not unfunded private payload controls"
    );
    assert!(
        tokio::time::timeout(Duration::from_secs(2), job.gather_owner().close_and_drain())
            .await
            .unwrap()
            .is_empty()
    );
    drop(job);
    assert_eq!(
        pool.reserved(),
        0,
        "collected independent output cannot keep managed resident funding alive"
    );
    let record = &output.table_payload().unwrap().batches()[0];
    assert_eq!(
        record
            .column(1)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        0
    );
    assert_eq!(
        record
            .column(5)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "paid"
    );
}

#[tokio::test]
async fn test_collected_output_keeps_managed_end_independent_of_private_controls() {
    collected_output_case(false).await;
}

#[tokio::test]
async fn test_collected_output_keeps_managed_cancel_independent_of_private_controls() {
    collected_output_case(true).await;
}

fn gated_plan(
    keys: RecordBatch,
    gate: Arc<ProducerGate>,
    metadata_only: bool,
) -> Arc<dyn ExecutionPlan> {
    let schema = keys.schema();
    let source = MemorySourceConfig::try_new_exec(&[vec![keys]], schema, None).unwrap();
    let input: Arc<dyn ExecutionPlan> = Arc::new(ProducerInput {
        input: source,
        gate,
        metadata_only,
    });
    Arc::new(RepartitionExec::try_new(input, Partitioning::RoundRobinBatch(2)).unwrap())
}

async fn gated_producer(keys: RecordBatch, gate: Arc<ProducerGate>, metadata_only: bool) {
    let plan = gated_plan(keys, gate, metadata_only);
    let mut stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
    let _ = stream.next().await;
}

async fn scratch_fixture(rows: usize) -> (StreamJoinOperator, StreamJobContext) {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(4_096, 4_000_000, 1_000).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            left_batch((0..i64::try_from(rows).unwrap()).collect()),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    (operator, job)
}

#[test]
fn test_serial_graph_checks_actual_types_and_every_child() {
    use datafusion::{
        datasource::memory::DataSourceExec,
        physical_plan::{ExecutionPlanProperties, coalesce_partitions::CoalescePartitionsExec},
    };
    let schema = left_schema();
    let batch = left_batch(vec![0]).table_payload().unwrap().batches()[0].clone();
    let source: Arc<dyn ExecutionPlan> =
        MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
            .unwrap();
    assert!(crate::datafusion::owned::serial_plan(&source));
    let coalesced: Arc<dyn ExecutionPlan> =
        Arc::new(CoalescePartitionsExec::new(Arc::clone(&source)));
    assert!(crate::datafusion::owned::serial_plan(&coalesced));
    let wrapped: Arc<dyn ExecutionPlan> = Arc::new(ProducerInput {
        input: Arc::clone(&source),
        gate: Arc::default(),
        metadata_only: true,
    });
    assert!(
        wrapped.is::<DataSourceExec>(),
        "DF's delegate lookup deliberately hides the wrapper"
    );
    assert!(!crate::datafusion::owned::serial_plan(&wrapped));
    let repartitioned: Arc<dyn ExecutionPlan> =
        Arc::new(RepartitionExec::try_new(source, Partitioning::RoundRobinBatch(1)).unwrap());
    assert_eq!(repartitioned.output_partitioning().partition_count(), 1);
    assert!(!crate::datafusion::owned::serial_plan(&repartitioned));
    let parallel =
        MemorySourceConfig::try_new_exec(&[vec![batch.clone()], vec![batch]], schema, None)
            .unwrap();
    let root: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(parallel));
    assert_eq!(root.output_partitioning().partition_count(), 1);
    assert!(!crate::datafusion::owned::serial_plan(&root));
}

struct InputReleaseProof {
    array: std::sync::Weak<dyn Array>,
    pool: Arc<dyn datafusion::execution::memory_pool::MemoryPool>,
    seen: Arc<std::sync::atomic::AtomicBool>,
    funding: Arc<sql_key_scratch::ScratchFunding>,
}

impl Drop for InputReleaseProof {
    fn drop(&mut self) {
        assert!(
            self.array.upgrade().is_none(),
            "actual input buffers release before the owner"
        );
        assert_eq!(self.pool.reserved(), self.funding.funded_bytes());
        self.seen.store(true, std::sync::atomic::Ordering::Release);
    }
}

async fn input_release_case(polled: bool) {
    let (mut operator, job) = scratch_fixture(1).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let plan = operator.side_plan("right").unwrap();
    let keys = operator.owned_state_keys(&plan, &context).await.unwrap();
    let array = Arc::downgrade(keys.column(0));
    let seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let runtime = operator.runtime.runtime.take().unwrap();
    let pool = runtime.incremental_memory_pool();
    let owner = InputReleaseProof {
        array,
        pool: Arc::clone(&pool),
        seen: Arc::clone(&seen),
        funding: keys.funding.unwrap(),
    };
    let tables = BTreeMap::from([(
        "cf_state".into(),
        Batch::table(vec![keys.batch], BatchMetadata::default()).unwrap(),
    )]);
    drop(operator);
    let query = parse_select_query("SELECT * FROM cf_state").unwrap();
    let _lock = runtime.lock_owned_test_query().await;
    let mut future = Box::pin(runtime.sql_equality_owned(
        &query,
        crate::datafusion::owned::Input::new(tables, owner),
        Some("match"),
    ));
    if polled {
        assert!(futures::poll!(future.as_mut()).is_pending());
    }
    drop(future);
    assert!(seen.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(pool.reserved(), 0);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

#[tokio::test]
async fn test_unpolled_owned_sql_input_releases_buffers_before_credit() {
    input_release_case(false).await;
}

#[tokio::test]
async fn test_owned_sql_query_lock_cancellation_releases_buffers_before_credit() {
    input_release_case(true).await;
}

#[tokio::test]
async fn test_background_sql_configuration_keeps_payload_and_scratch_legacy() {
    let mut operator = operator_fixture_for_sql();
    operator.set_stream_resources(
        DataFusionConfig {
            target_partitions: 2,
            min_rows_per_partition: 1,
            ..DataFusionConfig::default()
        },
        UdfRegistrySnapshot::default(),
    );
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![10]), &context, &mut collector)
        .await
        .unwrap();
    assert!(
        matches!(
            operator.state.left[0].record,
            columnar::RowPayload::Legacy(_)
        ),
        "unproved background SQL must not retain a paid Shared schema whose consumers cannot carry its lease"
    );
    let plan = operator.side_plan("right").unwrap();
    let _keys = operator.owned_state_keys(&plan, &context).await.unwrap();
    assert!(
        operator
            .retained_key_cache
            .left
            .as_ref()
            .unwrap()
            .scratch_funding
            .is_none()
    );
}

fn operator_fixture_for_sql() -> StreamJoinOperator {
    StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap()
}

#[tokio::test]
async fn test_serial_empty_sql_result_keeps_metadata_funding() {
    let (mut operator, job) = scratch_fixture(1).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let plan = operator.side_plan("right").unwrap();
    let keys = operator.owned_state_keys(&plan, &context).await.unwrap();
    let field = Arc::downgrade(&keys.schema().fields()[0]);
    let owner = keys.funding.unwrap();
    let tables = BTreeMap::from([(
        "cf_state".into(),
        Batch::table(vec![keys.batch], BatchMetadata::default()).unwrap(),
    )]);
    let query = parse_select_query("SELECT * FROM cf_state WHERE false").unwrap();
    let runtime = operator.runtime.runtime().unwrap();
    let pool = runtime.incremental_memory_pool();
    let result = runtime
        .sql_equality_owned(
            &query,
            crate::datafusion::owned::Input::new(tables, owner),
            Some("match"),
        )
        .await
        .unwrap_or_else(|failure| panic!("owned result failure: {:?}", failure.error));
    assert_eq!(result.batch().num_rows(), 0);
    assert!(Arc::ptr_eq(
        &result.batch().table_payload().unwrap().schema().fields()[0],
        &field.upgrade().unwrap()
    ));
    drop(operator);
    assert!(
        field.upgrade().is_some(),
        "the empty SQL result retains the actual scratch Field allocation"
    );
    assert!(
        pool.reserved() > 0,
        "serial metadata-only SQL output must retain its actual input funding until decode/drop"
    );
    drop(result);
    assert!(field.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_paid_key_scratch_covers_live_and_transient_allocation() {
    let (mut operator, job) = scratch_fixture(64).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let runtime = operator.runtime.runtime().unwrap();
    let pool = runtime.incremental_memory_pool();
    let baseline = pool.reserved();
    let mut future = Box::pin(sql_key_scratch::state_keys(
        runtime,
        &operator.state.left,
        &[0],
        &context,
    ));
    let waker = futures::task::noop_waker();
    let mut cx = Context::from_waker(&waker);
    let mut output = None;
    let mut retained = 0_i64;
    let mut peak = 0_i64;
    for _ in 0..128 {
        let measured = allocation_counter::measure(|| {
            if let Poll::Ready(result) = future.as_mut().poll(&mut cx) {
                output = result.unwrap();
            }
        });
        peak = peak.max(retained + i64::try_from(measured.bytes_max).unwrap());
        retained += measured.bytes_current;
        if output.is_some() {
            break;
        }
        tokio::task::yield_now().await;
    }
    drop(future);
    let output = output.expect("bounded scratch completes within fixture poll bound");
    let funded = i64::try_from(output.funded_bytes()).unwrap();
    assert!(retained > 0);
    assert!(
        funded >= retained,
        "retained={retained}, actual guard={funded}"
    );
    assert!(funded >= peak, "peak={peak}, actual guard={funded}");
    drop(output);
    assert_eq!(pool.reserved(), baseline);
    drop(operator);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_paid_key_scratch_pressure_preserves_legacy_sql_acceptance() {
    paid_key_pressure_case(128).await;
}

#[tokio::test]
async fn test_paid_key_scratch_zero_headroom_preserves_legacy_sql_acceptance() {
    paid_key_pressure_case(0).await;
}

#[tokio::test]
async fn test_closed_scratch_home_falls_back_without_spurious_cancel() {
    let (mut operator, job) = scratch_fixture(3).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let captured = operator.checkpoint(Epoch::INITIAL).unwrap();
    let before = operator.status();
    let plan = operator.side_plan("right").unwrap();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let paid = pool.reserved();
    job.gather_owner().close_admission();
    let keys = operator
        .owned_state_keys(&plan, &context)
        .await
        .expect("closed optional scratch admission must preserve a healthy context");
    assert!(keys.funding.is_none());
    assert_eq!(pool.reserved(), paid);
    drop(keys);
    let record = right_batch(vec![0]).table_payload().unwrap().batches()[0].clone();
    let admitted = vec![AdmittedRow {
        record: record.into(),
        event_time: EventTime::from_micros(0),
        row_id: 0,
        retain: true,
    }];
    let (matched, admitted) = operator.legacy_matches(&plan, admitted, &context).await;
    assert_eq!(matched.unwrap().len(), 3);
    assert_eq!(operator.status(), before);
    assert_eq!(
        operator
            .checkpoint(Epoch::new(2).unwrap())
            .unwrap()
            .segments,
        captured.segments
    );
    drop(admitted);
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_closed_scratch_home_preserves_actual_cancel_and_deadline() {
    let (mut operator, job) = scratch_fixture(1).await;
    let plan = operator.side_plan("right").unwrap();
    let before = operator.status();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let paid = pool.reserved();
    job.gather_owner().close_admission();
    for deadline in [false, true] {
        let cancellation = CancellationToken::new();
        let stopped = StreamJobContext::new(
            3,
            "scratch-stop",
            JsonMap::new(),
            deadline.then(|| chrono::Utc::now() - chrono::Duration::seconds(1)),
            cancellation.clone(),
        )
        .with_gather_owner(job.gather_owner().clone());
        if !deadline {
            cancellation.cancel();
        }
        let context = StreamOperatorContext::new(&stopped, "match", None);
        let error = operator
            .owned_state_keys(&plan, &context)
            .await
            .err()
            .unwrap();
        assert!(matches!(error, CalcFlowError::Cancelled { run_id } if run_id == "3"));
        assert_eq!(operator.status(), before);
        assert_eq!(pool.reserved(), paid);
    }
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn paid_key_pressure_case(headroom: usize) {
    let (mut operator, job) = scratch_fixture(64).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let plan = operator.side_plan("right").unwrap();
    let mut bundle = operator.admission_bundle(&plan);
    let input = right_batch(vec![0]);
    operator
        .admit_record(
            &input.table_payload().unwrap().batches()[0],
            &plan,
            "right",
            &context,
            &mut bundle,
            0,
        )
        .await
        .unwrap();
    let status = operator.status();
    let raw = state_key_batch(&operator.state.left, &operator.compiled, &plan, None).unwrap();
    let runtime = operator.runtime.runtime().unwrap();
    let pool = runtime.incremental_memory_pool();
    let copied = sql_key_scratch::state_keys(runtime, &operator.state.left, &[0], &context)
        .await
        .unwrap()
        .unwrap();
    let paid = copied.funded_bytes();
    drop(copied);
    let pressure = runtime.incremental_reservation("paid-key-pressure");
    pressure
        .try_grow((1 << 30) - pool.reserved() - paid - headroom)
        .unwrap();
    let expected = operator
        .sql_key_pairs(&plan, &bundle.admitted, raw)
        .await
        .unwrap();
    let keys = operator.owned_state_keys(&plan, &context).await.unwrap();
    assert!(keys.funding.is_some());
    drop(keys);
    assert!(
        operator
            .retained_key_cache
            .left
            .as_ref()
            .unwrap()
            .scratch_funding
            .is_some(),
        "the accepted baseline must be compared with an admitted paid copy"
    );
    crate::datafusion::owned::observe_legacy_inputs();
    let (actual, admitted) = operator
        .legacy_matches(&plan, std::mem::take(&mut bundle.admitted), &context)
        .await;
    bundle.admitted = admitted;
    let retried = crate::datafusion::owned::take_legacy_inputs();
    assert_eq!(retried.len(), if headroom == 0 { 2 } else { 0 });
    drop(retried);
    assert!(
        actual.is_ok(),
        "optional paid scratch (actual credit={paid}, remaining={headroom}) must preserve accepted SQL at the same budget: {:?}",
        actual.as_ref().err()
    );
    let actual = actual
        .unwrap()
        .iter()
        .map(|pair| {
            (
                u64::try_from(pair.pos).unwrap(),
                operator.state.left[pair.opposite_index].row_id,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert_eq!(operator.status(), status);
    drop(pressure);
    drop(bundle);
    drop(operator);
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert_eq!(pool.reserved(), home + generation);
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

async fn string_budget_case(
    stage: crate::runtime::streaming::gather_work::admission_probe::AdmissionStage,
) {
    use crate::runtime::streaming::gather_work::admission_probe::AdmissionProbe;
    let (mut operator, job) = scratch_fixture(1).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let plan = operator.side_plan("right").unwrap();
    let input = right_batch(vec![0]);
    let original = input.table_payload().unwrap().batches()[0].clone();
    let mut baseline = operator.admission_bundle(&plan);
    operator.payload_native_eligible = false;
    operator
        .admit_record(&original, &plan, "right", &context, &mut baseline, 0)
        .await
        .unwrap();
    operator.payload_native_eligible = true;
    let raw = operator.opposite_state_keys(&plan).unwrap();
    let expected = operator
        .sql_key_pairs(&plan, &baseline.admitted, raw)
        .await
        .unwrap();
    drop(baseline);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let probe = AdmissionProbe::install(job.gather_owner(), stage, Arc::clone(&pool), 1 << 30);
    let mut actual = operator.admission_bundle(&plan);
    operator
        .admit_record(&original, &plan, "right", &context, &mut actual, 0)
        .await
        .unwrap();
    let event = probe
        .take_event()
        .expect("real owned-string gather admission was reached");
    assert_eq!(event.stage, stage);
    assert_eq!(event.operator, "stream-join-owned-string");
    assert_eq!(event.available + 1, event.fee);
    assert!(matches!(
        actual.admitted[0].record,
        columnar::RowPayload::Legacy(_)
    ));
    let raw = operator.opposite_state_keys(&plan).unwrap();
    assert_eq!(
        operator
            .sql_key_pairs(&plan, &actual.admitted, raw)
            .await
            .unwrap(),
        expected
    );
    assert_eq!(input.table_payload().unwrap().batches()[0], original);
    drop(actual);
    drop(operator);
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert_eq!(pool.reserved(), home + generation);
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_string_attempt_budget_refusal_preserves_legacy_acceptance() {
    string_budget_case(
        crate::runtime::streaming::gather_work::admission_probe::AdmissionStage::Attempt,
    )
    .await;
}

#[tokio::test]
async fn test_string_generation_budget_refusal_preserves_legacy_acceptance() {
    string_budget_case(
        crate::runtime::streaming::gather_work::admission_probe::AdmissionStage::Generation,
    )
    .await;
}

#[tokio::test]
async fn test_nonbudget_sql_error_does_not_retry_legacy() {
    let (mut operator, job) = scratch_fixture(1).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let plan = operator.side_plan("right").unwrap();
    let input = right_batch(vec![0]);
    let mut bundle = operator.admission_bundle(&plan);
    operator
        .admit_record(
            &input.table_payload().unwrap().batches()[0],
            &plan,
            "right",
            &context,
            &mut bundle,
            0,
        )
        .await
        .unwrap();
    let keys = operator.owned_state_keys(&plan, &context).await.unwrap();
    operator.compiled.equality_query = parse_select_query(&format!(
        "SELECT \"missing_ResourcesExhausted\" FROM {PROBE_TABLE}"
    ))
    .unwrap();
    let before = operator.status();
    crate::datafusion::owned::observe_legacy_inputs();
    let (result, admitted) = operator
        .sql_key_pairs_owned(&plan, std::mem::take(&mut bundle.admitted), keys)
        .await;
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("ResourcesExhausted")
    );
    assert!(
        crate::datafusion::owned::take_legacy_inputs().is_empty(),
        "text containing the budget variant name is not typed ResourcesExhausted"
    );
    assert_eq!(operator.status(), before);
    drop(admitted);
    drop(bundle);
    drop(operator);
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_payload_buffers_keep_credit_until_real_consumer_cleanup() {
    let (mut operator, job) = scratch_fixture(1).await;
    let payload = operator.state.left[0].record.clone();
    let paid = payload.funded_owner().unwrap().1;
    let view = payload.view();
    let record = (*view).clone();
    let array = Arc::downgrade(record.column(0));
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let gate = Arc::new(ProducerGate::default());
    let _release = GateRelease(Arc::clone(&gate));
    let producer = tokio::spawn(gated_producer(record, Arc::clone(&gate), false));
    tokio::time::timeout(Duration::from_secs(2), gate.entered.notified())
        .await
        .unwrap();
    producer.abort();
    assert!(producer.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(2), gate.dropping.notified())
        .await
        .unwrap();
    drop(view);
    drop(payload);
    drop(operator);
    assert_eq!(
        array
            .upgrade()
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[7]
    );
    assert_eq!(
        pool.reserved(),
        paid,
        "escaped actual payload buffers keep their credit"
    );
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    assert!(futures::poll!(drain.as_mut()).is_pending());
    gate.open();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), drain)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(array.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
}

async fn string_payload_fixture(large: bool) -> (StreamJoinOperator, StreamJobContext) {
    use datafusion::arrow::array::LargeStringArray;
    let data_type = if large {
        DataType::LargeUtf8
    } else {
        DataType::Utf8
    };
    let mut fields = right_schema().fields().to_vec();
    fields[2] = Arc::new(Field::new("status", data_type, true));
    let schema = Arc::new(Schema::new(fields));
    let payload: ArrayRef = if large {
        Arc::new(LargeStringArray::from(vec!["paid"]))
    } else {
        Arc::new(StringArray::from(vec!["paid"]))
    };
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
            payload,
        ],
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("match", left_schema(), schema, spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    (operator, job)
}

async fn string_consumer_case(large: bool) {
    use datafusion::arrow::array::LargeStringArray;
    let (mut operator, job) = string_payload_fixture(large).await;
    let payload = operator.state.right[0].record.clone();
    let paid = payload.funded_owner().unwrap().1;
    let column = payload.column_view(2);
    let array = Arc::downgrade(&column);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        column.data_type().clone(),
        false,
    )]));
    let record = RecordBatch::try_new(schema, vec![column]).unwrap();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let gate = Arc::new(ProducerGate::default());
    let _release = GateRelease(Arc::clone(&gate));
    hold_cancelled_producer(record, &gate).await;
    drop(payload);
    drop(operator);
    let held = array.upgrade().unwrap();
    let value = if large {
        held.as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap()
            .value(0)
    } else {
        held.as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0)
    };
    assert_eq!(value, "paid");
    drop(held);
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(
        pool.reserved(),
        paid + home + generation + attempt,
        "a string-only actual consumer keeps its independently observed payload credit"
    );
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    assert!(futures::poll!(drain.as_mut()).is_pending());
    gate.open();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), drain)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(array.upgrade().is_none());
    let remaining = job.gather_owner().funding();
    eprintln!(
        "post-drain pool={}, actual home/generation/attempt={remaining:?}",
        pool.reserved()
    );
    assert_eq!(remaining.2, 0);
    assert_eq!(pool.reserved(), remaining.0 + remaining.1);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_payload_string_buffers_keep_credit_until_real_consumer_cleanup() {
    string_consumer_case(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_payload_large_string_buffers_keep_credit_until_real_consumer_cleanup() {
    string_consumer_case(true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_sql_producer_keeps_funding_until_real_cancel_cleanup() {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(4_096, 4_000_000, 1_000).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            left_batch((0..4_096).collect()),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator.state.left = operator.state.left[..1].to_vec().into();
    let plan = operator.side_plan("right").unwrap();
    let keys = operator.owned_state_keys(&plan, &context).await.unwrap();
    let array = Arc::downgrade(keys.column(0));
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let gate = Arc::new(ProducerGate::default());
    let _release = GateRelease(Arc::clone(&gate));
    let record = keys.batch;
    drop(keys.funding);
    let producer = tokio::spawn(gated_producer(record, Arc::clone(&gate), false));
    tokio::time::timeout(Duration::from_secs(2), gate.entered.notified())
        .await
        .unwrap();
    producer.abort();
    assert!(producer.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(2), gate.dropping.notified())
        .await
        .unwrap();
    drop(operator);
    assert_eq!(
        array
            .upgrade()
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[7]
    );
    assert!(
        pool.reserved() > 0,
        "real DF producer still owns scratch after query/operator cancellation; its funding must remain live"
    );
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    assert!(futures::poll!(drain.as_mut()).is_pending());
    gate.open();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), drain)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(array.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_naked_schema_producer_is_refused_by_owned_sql_contract() {
    let (mut operator, job) = scratch_fixture(1).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let plan = operator.side_plan("right").unwrap();
    let keys = operator.owned_state_keys(&plan, &context).await.unwrap();
    let schema = Arc::downgrade(&keys.schema());
    let array = Arc::downgrade(keys.column(0));
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let gate = Arc::new(ProducerGate::default());
    let _release = GateRelease(Arc::clone(&gate));
    let record = keys.batch;
    drop(keys.funding);
    let physical = gated_plan(record.clone(), Arc::clone(&gate), true);
    assert!(!crate::datafusion::owned::serial_plan(&physical));
    drop(physical);
    let producer = tokio::spawn(gated_producer(record, Arc::clone(&gate), true));
    tokio::time::timeout(Duration::from_secs(2), gate.entered.notified())
        .await
        .unwrap();
    producer.abort();
    assert!(producer.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(2), gate.dropping.notified())
        .await
        .unwrap();
    drop(operator);
    assert!(array.upgrade().is_none(), "this gate retains metadata only");
    assert_eq!(
        schema.upgrade().unwrap().field(0).name(),
        &format!("{KEY_COLUMN_PREFIX}0")
    );
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!(attempt, 0);
    assert_eq!(pool.reserved(), home + generation);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert!(
        schema.upgrade().is_some(),
        "the intentionally naked schema has no managed lease"
    );
    gate.open();
    tokio::time::timeout(Duration::from_secs(2), gate.finished.notified())
        .await
        .unwrap();
    assert!(schema.upgrade().is_none());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

struct ManagedPayloadSource {
    id: String,
    schema: SchemaRef,
    batch: Option<Batch>,
    release: Arc<tokio::sync::Semaphore>,
    closed: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait]
impl crate::StreamSource for ManagedPayloadSource {
    fn capabilities(&self) -> crate::SourceCapabilities {
        crate::SourceCapabilities {
            replay_positioning: crate::ReplayPositioning::Unsupported,
            delivery: crate::SourceDeliveryCapability::Lossless,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
            schema: crate::SourceSchema::Exact(Arc::clone(&self.schema)),
            native_watermarks: crate::NativeWatermarkCapability::NeverEmits,
        }
    }

    async fn open(&mut self, _: Option<crate::Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<crate::SourceEvent>> {
        if let Some(batch) = self.batch.take() {
            return Ok(Some(crate::SourceEvent::Data {
                batch,
                cursor: crate::Cursor::new(&self.id, vec![1], JsonMap::new())?,
            }));
        }
        self.release.acquire().await.unwrap().forget();
        Ok(None)
    }

    async fn close(&mut self) -> Result<()> {
        self.closed
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

struct ManagedPayloadSink {
    outputs: Arc<Mutex<Vec<Batch>>>,
    written: Arc<Notify>,
    closed: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait]
impl crate::StreamSink for ManagedPayloadSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.outputs.lock().push(batch.clone());
        self.written.notify_one();
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

fn managed_payload_sources(
    plan: &crate::StreamExecutionPlan,
    release: &Arc<tokio::sync::Semaphore>,
    closed: &Arc<std::sync::atomic::AtomicUsize>,
) -> BTreeMap<String, crate::SourceBinding> {
    plan.source_binding_ids()
        .into_iter()
        .map(|id| {
            let batch = if id.ends_with("left") {
                left_batch(vec![0])
            } else {
                right_batch(vec![0])
            };
            let source = ManagedPayloadSource {
                id: id.to_owned(),
                schema: Arc::clone(batch.table_payload().unwrap().schema()),
                batch: Some(batch),
                release: Arc::clone(release),
                closed: Arc::clone(closed),
            };
            (
                id.to_owned(),
                crate::SourceBinding::new(source)
                    .with_watermark_policy(crate::WatermarkPolicy::Disabled { idle_timeout: None }),
            )
        })
        .collect()
}

async fn managed_collected_output_case(cancelled: bool) {
    let plan = crate::PipelineBuilder::new("managed-columnar-output")
        .unwrap()
        .add_node(
            "match",
            Box::new(
                StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap(),
            ),
        )
        .unwrap()
        .compile_stream(
            &crate::UdfRegistry::new().snapshot(),
            &crate::StreamRequirements::default(),
        )
        .unwrap();
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let source_closed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let sink_closed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let sources = managed_payload_sources(&plan, &release, &source_closed);
    let outputs = Arc::new(Mutex::new(Vec::new()));
    let written = Arc::new(Notify::new());
    let sink = ManagedPayloadSink {
        outputs: Arc::clone(&outputs),
        written: Arc::clone(&written),
        closed: Arc::clone(&sink_closed),
    };
    let sinks = BTreeMap::from([(
        plan.sink_binding_ids()[0].to_owned(),
        vec![crate::SinkBinding::ordinary("kept-output", sink).unwrap()],
    )]);
    let root = tempfile::tempdir().unwrap();
    let runner = crate::StreamingRunner::new(
        plan,
        sources,
        sinks,
        crate::ManagedCheckpointRuntime::new(root.path()).unwrap(),
    )
    .unwrap();
    let (start, cleanup) = runner.start_with_cleanup();
    let job = start.await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), written.notified())
        .await
        .unwrap();
    assert_eq!(job.stream_join_status()["match"].emitted_match_rows, 1);
    let outcome = if cancelled {
        tokio::time::timeout(Duration::from_secs(5), job.cancel())
            .await
            .unwrap()
    } else {
        release.add_permits(2);
        tokio::time::timeout(Duration::from_secs(5), job.wait())
            .await
            .unwrap()
    };
    assert_eq!(
        outcome.state,
        if cancelled {
            crate::JobState::Cancelled
        } else {
            crate::JobState::Completed
        }
    );
    assert!(outcome.errors.is_empty(), "{:?}", outcome.errors);
    drop(job);
    tokio::time::timeout(Duration::from_secs(5), cleanup)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(source_closed.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(sink_closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    let kept = outputs.lock();
    assert_eq!(kept.len(), 1);
    let record = &kept[0].table_payload().unwrap().batches()[0];
    assert_eq!(
        record
            .column(1)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        0
    );
    assert_eq!(
        record
            .column(5)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "paid"
    );
}

#[tokio::test]
async fn test_managed_runner_end_releases_columnar_state_while_output_is_collected() {
    managed_collected_output_case(false).await;
}

#[tokio::test]
async fn test_managed_runner_cancel_releases_columnar_state_while_output_is_collected() {
    managed_collected_output_case(true).await;
}
