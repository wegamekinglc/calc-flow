use super::*;
use crate::operator::join::{join_work, reset_join_work};
use crate::runtime::streaming::gather_work::TestService;
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, Epoch, JoinStateLimits,
    JoinTimeBounds, JsonMap, OperatorMetadata, StreamJoinSpec, StreamOperator,
};
use datafusion::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryLimit, MemoryPool, MemoryReservation,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

const ROWS: usize = 64_000;

#[path = "batch_parallel_output_lifecycle.rs"]
mod rowed_lifecycle;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn input(dimension: bool) -> Batch {
    input_at(dimension, 0)
}

fn input_at(dimension: bool, time: i64) -> Batch {
    let length = if dimension { 64 } else { ROWS + 2 };
    let value = |index: usize| {
        if dimension {
            index
        } else {
            index.saturating_sub(1)
        }
    };
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..length).map(|index| i64::try_from(value(index) % 64).unwrap()),
            )),
            Arc::new(TimestampMicrosecondArray::from(vec![time; length]).with_timezone("UTC")),
            Arc::new(Int64Array::from_iter_values(
                (0..length).map(|index| i64::try_from(value(index)).unwrap()),
            )),
        ],
    )
    .unwrap();
    Batch::table(
        if dimension {
            vec![record]
        } else {
            vec![
                record.slice(1, ROWS / 2),
                record.slice(1 + ROWS / 2, ROWS / 2),
            ]
        },
        BatchMetadata::default(),
    )
    .unwrap()
}

fn operator() -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(100_000, 100_000_000, 1_000_000).unwrap(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("match", schema(), schema(), spec).unwrap();
    operator.set_output_projection(vec![2, 0, 5]).unwrap();
    operator
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

struct ProductionGate(usize);

impl ProductionGate {
    fn new() -> Self {
        Self(super::super::PROBE_COST_GATE.replace(super::super::PARALLEL_PROBE_MIN_VISITS))
    }
}

impl Drop for ProductionGate {
    fn drop(&mut self) {
        super::super::PROBE_COST_GATE.set(self.0);
    }
}

fn job(service: &TestService) -> StreamJobContext {
    StreamJobContext::new(
        591,
        "batch-parallel",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
    .with_gather_owner(service.owner("batch-parallel".into()))
}

async fn preload(operator: &mut StreamJoinOperator, job: &StreamJobContext) {
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            input(true),
            &StreamOperatorContext::new(job, "match", None),
            &mut output,
        )
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
}

async fn admit(operator: &mut StreamJoinOperator) -> AdmittedBatch {
    let input = input(false);
    let plan = operator.begin_batch("left", &input).unwrap();
    let mut bundle = operator.admission_bundle(&plan, &input).unwrap();
    let job = StreamJobContext::new(
        592,
        "probe-time",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "match", None);
    operator
        .admit_batch_records("left", &input, &plan, &context, &mut bundle)
        .await
        .unwrap();
    bundle.finish("match").unwrap();
    bundle.admitted
}

fn projected_probe_operator() -> StreamJoinOperator {
    let mut fields = schema().fields().to_vec();
    fields.push(Arc::new(Field::new("unused", DataType::Int64, false)));
    let mut operator = StreamJoinOperator::new(
        "match",
        schema(),
        Arc::new(Schema::new(fields)),
        operator().spec.clone(),
    )
    .unwrap();
    operator.set_output_projection(vec![2, 0, 5]).unwrap();
    assert!(operator.payload_projection.is_some());
    operator
}

async fn preload_projected_probe(operator: &mut StreamJoinOperator, job: &StreamJobContext) {
    let dimension = input(true);
    let record = &dimension.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns.push(Arc::new(Int64Array::from(vec![42; 64])));
    let right = Batch::table(
        vec![RecordBatch::try_new(Arc::clone(operator.input_schema(1)), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "right",
            right,
            &StreamOperatorContext::new(job, "match", None),
            &mut output,
        )
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
}

fn probe_only_context(job: &StreamJobContext) -> StreamOperatorContext<'_> {
    use crate::{IngressProgress, IngressProgressSnapshot, IngressState};
    use std::collections::BTreeMap;

    StreamOperatorContext::with_ingress_progress(
        job,
        "match",
        None,
        IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(IngressState::Active, None),
            ),
            (
                "right".into(),
                IngressProgress::new(IngressState::Ended, None),
            ),
        ])),
    )
}

fn verify(output: &mut EdgeCollector, max_rows: usize) {
    let mut position = 0;
    for (sequence, message) in output.drain("output").into_iter().enumerate() {
        let batch = message.as_data().unwrap();
        assert_eq!(batch.metadata().sequence(), sequence as u64);
        for record in batch.table_payload().unwrap().batches() {
            assert!(record.num_rows() <= max_rows);
            assert_eq!(record.num_columns(), 3);
            for row in 0..record.num_rows() {
                for (column, expected) in [position, position % 64, position % 64]
                    .into_iter()
                    .enumerate()
                {
                    assert_eq!(
                        record
                            .column(column)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap()
                            .value(row),
                        i64::try_from(expected).unwrap(),
                    );
                }
                position += 1;
            }
        }
    }
    assert_eq!(position, ROWS);
}

#[test]
fn batch_parallel_output_narrow_one_to_one_uses_funded_ordered_morsels() {
    let _gate = ProductionGate::new();
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = operator();
        let job = job(&service);
        preload(&mut operator, &job).await;
        let observed = Arc::new(Mutex::new(Vec::new()));
        let units = observed.clone();
        operator.probe_unit_test_hook = Some(Arc::new(move |phase, unit, complete| {
            if phase == super::super::ProbePhase::Fill {
                if complete {
                    assert!(
                        join_work().output_column_takes > 0,
                        "worker must build its output before settlement"
                    );
                    units.lock().push(unit);
                } else {
                    reset_join_work();
                }
            }
        }));
        let context = StreamOperatorContext::new(&job, "match", None)
            .with_output_budget(EdgeBudget::new(7_000, 8 << 20).unwrap());
        let source = input(false);
        let before = source.table_payload().unwrap().batches().to_vec();
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", source.clone(), &context, &mut output)
            .await
            .unwrap();
        let mut units = observed.lock().clone();
        units.sort_unstable();
        assert_eq!(
            units,
            [0, 1, 2, 3],
            "production one-to-one gate must dispatch fused row work"
        );
        verify(&mut output, 7_000);
        assert_eq!(source.table_payload().unwrap().batches(), &before);
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        let mut restored = self::operator();
        restored.restore(&snapshot).unwrap();
        assert_eq!(restored.status(), operator.status());
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((operator, restored, snapshot));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn batch_parallel_output_sealed_dimension_funds_projected_probe_only_views() {
    use crate::operator::join::{columnar::RowPayload, retained::Rows};

    let _gate = ProductionGate::new();
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = projected_probe_operator();
        let job = job(&service);
        preload_projected_probe(&mut operator, &job).await;
        assert!(matches!(
            operator.state.right.payload(0).owner(),
            Some(RowPayload::Shared { .. } | RowPayload::RestoredV2 { .. })
        ));
        let context = probe_only_context(&job);
        let source = input(false);
        let before = source.table_payload().unwrap().batches().to_vec();
        let (mut prepared, expiration) = operator
            .prepare_data_append("left", &source, &context)
            .await
            .unwrap();
        assert!(
            prepared.deferred.is_some(),
            "narrow production gate must count first"
        );
        assert!(
            prepared.retained.is_empty(),
            "sealed peer needs no new left state"
        );
        assert!(prepared.admitted.projection_is_view(0));
        assert!(prepared.admitted.projection_is_view(1));
        operator
            .finish_morsels(&mut prepared, &context)
            .await
            .unwrap();
        assert!(
            prepared.gathered.is_some(),
            "projected read-only probe views need prepaid input funding for fused workers"
        );
        let gathered = prepared.gathered.as_ref().unwrap();
        assert_eq!(gathered.rows(), ROWS);
        assert_eq!(gathered.funding_for_test().len(), 4);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .finish_data("left", prepared, expiration, &context, &mut output)
            .await
            .unwrap();
        verify(&mut output, usize::MAX);
        assert_eq!(source.table_payload().unwrap().batches(), &before);
        assert_eq!(operator.status().left.retained_rows, 0);
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(operator);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[derive(Debug)]
struct RefusingPool {
    inner: GreedyMemoryPool,
    attempts: AtomicUsize,
}

impl std::fmt::Display for RefusingPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, formatter)
    }
}

impl MemoryPool for RefusingPool {
    fn name(&self) -> &'static str {
        "batch-parallel-refusal"
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn grow(&self, reservation: &MemoryReservation, bytes: usize) {
        self.inner.grow(reservation, bytes);
    }
    fn shrink(&self, reservation: &MemoryReservation, bytes: usize) {
        self.inner.shrink(reservation, bytes);
    }
    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        bytes: usize,
    ) -> datafusion::common::Result<()> {
        if reservation.consumer().name() == "sql-incremental:stream-join-morsel" && bytes > 0 {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            return Err(datafusion::common::DataFusionError::ResourcesExhausted(
                "optional morsel refused".into(),
            ));
        }
        self.inner.try_grow(reservation, bytes)
    }
}

#[test]
fn batch_parallel_output_narrow_optional_refusal_reuses_serial_results() {
    let _gate = ProductionGate::new();
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = operator();
        let pool = Arc::new(RefusingPool {
            inner: GreedyMemoryPool::new(1 << 30),
            attempts: AtomicUsize::new(0),
        });
        operator
            .runtime
            .initialize_benchmark_pool(pool.clone())
            .unwrap();
        let job = job(&service);
        preload(&mut operator, &job).await;
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", input(false), &context, &mut output)
            .await
            .unwrap();
        assert!(
            pool.attempts.load(Ordering::SeqCst) > 0,
            "production narrow output must attempt funded morsels"
        );
        verify(&mut output, usize::MAX);
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

#[test]
fn batch_parallel_output_narrow_cancel_prevents_output_and_state_commit() {
    let _gate = ProductionGate::new();
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = operator();
        let job = job(&service);
        preload(&mut operator, &job).await;
        let token = job.cancellation().clone();
        let fills = Arc::new(AtomicUsize::new(0));
        let observed = fills.clone();
        operator.probe_unit_test_hook = Some(Arc::new(move |phase, _, complete| {
            if phase == super::super::ProbePhase::Fill && !complete {
                observed.fetch_add(1, Ordering::SeqCst);
                token.cancel();
            }
        }));
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let result = operator
            .process_data("left", input(false), &context, &mut output)
            .await;
        assert!(
            matches!(result, Err(crate::CalcFlowError::Cancelled { .. })),
            "production narrow workers must observe cancellation: {result:?}"
        );
        assert!(fills.load(Ordering::SeqCst) > 0);
        assert!(output.drain("output").is_empty());
        assert_eq!(operator.state.next_left_row_id, 0);
        assert_eq!(operator.state.next_output_sequence, 0);
        assert_eq!(operator.status().left.retained_rows, 0);
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(operator);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn batch_parallel_output_probe_time_avoids_payload_segment_lookup() {
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = operator();
        let inputs = ProbeInputs {
            admitted: admit(&mut operator).await,
            keys: None,
            slots: Vec::new(),
            opposite: None,
            index: None,
            containers: None,
            bounds: operator.spec.bounds,
            incoming_is_left: true,
            control: None,
            released: None,
            hook: None,
            unit_hook: None,
        };
        reset_join_work();
        assert_eq!(
            inputs.range(ROWS / 2),
            (
                crate::EventTime::from_micros(0),
                crate::EventTime::from_micros(0)
            )
        );
        assert_eq!(
            join_work().admission_segment_lookups,
            0,
            "probe event-time reads must not locate payload segments"
        );
    });
}

#[test]
fn batch_parallel_output_single_cpu_capacity_keeps_serial_path() {
    let _gate = ProductionGate::new();
    let service = TestService::new(1, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = operator();
        let job = job(&service);
        preload(&mut operator, &job).await;
        let fills = Arc::new(AtomicUsize::new(0));
        let observed = fills.clone();
        operator.probe_unit_test_hook = Some(Arc::new(move |phase, _, complete| {
            if phase == super::super::ProbePhase::Fill && !complete {
                observed.fetch_add(1, Ordering::SeqCst);
            }
        }));
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", input(false), &context, &mut output)
            .await
            .unwrap();
        assert_eq!(fills.load(Ordering::SeqCst), 0);
        verify(&mut output, usize::MAX);
        drop(operator);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn batch_parallel_output_potential_payload_with_no_temporal_matches_is_empty() {
    let _gate = ProductionGate::new();
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = operator();
        let job = job(&service);
        preload(&mut operator, &job).await;
        let counted = Arc::new(AtomicUsize::new(0));
        let observed = counted.clone();
        operator.probe_unit_test_hook = Some(Arc::new(move |phase, _, complete| {
            if phase == super::super::ProbePhase::Count && complete {
                observed.fetch_add(1, Ordering::SeqCst);
            }
        }));
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", input_at(false, 1), &context, &mut output)
            .await
            .unwrap();
        assert_eq!(counted.load(Ordering::SeqCst), 4);
        assert!(output.drain("output").is_empty());
        assert_eq!(operator.status().left.retained_rows, ROWS as u64);
        assert_eq!(operator.state.next_output_sequence, 0);
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(operator);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn batch_parallel_output_cursor_bounds_segment_work_for_selected_sliced_rows() {
    use crate::operator::join::{StoredRow, materialization::JoinOutput};

    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = operator();
        let admitted = admit(&mut operator).await;
        let dimension = input(true);
        let record = &dimension.table_payload().unwrap().batches()[0];
        let opposite = (0..64)
            .map(|index| StoredRow {
                record: record.slice(index, 1).into(),
                event_time: crate::EventTime::from_micros(0),
                row_id: index as u64,
                charge: 96,
                encoded_key: Arc::new(vec![].into()),
            })
            .collect::<Vec<_>>();
        let positions = (0..ROWS)
            .step_by(997)
            .flat_map(|position| [position, position])
            .chain([ROWS - 1, 0, 0, ROWS / 2])
            .collect::<Vec<_>>();
        let pairs = positions
            .iter()
            .map(|&pos| MatchedPair {
                pos,
                opposite_index: pos % 64,
            })
            .collect::<Vec<_>>();
        let projection: Arc<[usize]> = Arc::from([2, 0, 5]);
        let output = JoinOutput {
            schema: operator.output_ports[0].schema().unwrap(),
            projection: Some(&projection),
            admitted: &admitted,
            opposite: &opposite,
            matched: &pairs,
            incoming_is_left: true,
            operator_id: "cursor",
            admitted_charges: None,
        };
        reset_join_work();
        let record = output.materialize(0..pairs.len()).unwrap();
        for (row, &position) in positions.iter().enumerate() {
            for (column, expected) in [position, position % 64, position % 64]
                .into_iter()
                .enumerate()
            {
                assert_eq!(
                    record
                        .column(column)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(row),
                    i64::try_from(expected).unwrap()
                );
            }
        }
        assert!(
            join_work().admission_segment_lookups <= 12,
            "selected output must locate segments on transitions, including backward positions: {}",
            join_work().admission_segment_lookups
        );
    });
}
