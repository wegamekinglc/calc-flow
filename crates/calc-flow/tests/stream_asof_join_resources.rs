use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use calc_flow::{
    AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, CalcFlowError, CancellationToken,
    EdgeBudget, Epoch, EventTime, IngressProgress, IngressProgressSnapshot, IngressState, JsonMap,
    Result, StreamAsofJoinOperator, StreamAsofJoinSpec, StreamCollector, StreamJobContext,
    StreamOperator, StreamOperatorContext, StreamingFailureReason,
};
use datafusion::arrow::{
    array::{Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};

const KEYS: usize = 1_000;
const ROWS_PER_SIDE: usize = 50_000;
const BATCH_ROWS: usize = 256;
const STATE_BYTES: u64 = 64 * 1024 * 1024;
const TOLERANCE: i64 = 60_000_000;
const ROUND_STEP: i64 = 15_000_000;

#[derive(Clone, Copy, Debug)]
enum Trace {
    Advancing,
    Stalled,
    HotKey,
    Wide,
}

#[derive(Clone, Copy)]
struct Row {
    key: u64,
    time: i64,
    sequence: i64,
    payload_bytes: usize,
}

fn rows(trace: Trace, left: bool) -> Vec<Row> {
    (0..ROWS_PER_SIDE)
        .map(|index| {
            let round = index / KEYS;
            let slot = index % KEYS;
            let key = if matches!(trace, Trace::HotKey) && round > 0 && slot < 900 {
                0
            } else {
                slot
            };
            Row {
                key: u64::try_from(key).unwrap(),
                time: i64::try_from(round).unwrap() * ROUND_STEP + if left { 5 } else { 0 },
                sequence: i64::try_from(index).unwrap() + 1,
                payload_bytes: if matches!(trace, Trace::Wide) && round > 0 {
                    16_384
                } else {
                    8
                },
            }
        })
        .collect()
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]))
}

fn operator(schema: &SchemaRef) -> StreamAsofJoinOperator {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["sequence".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::from_micros(u64::try_from(TOLERANCE).unwrap()),
        AsofStateLimits::new(100_000, STATE_BYTES).unwrap(),
    )
    .unwrap();
    StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap()
}

fn batch(schema: &SchemaRef, rows: &[Row]) -> Batch {
    let payloads = rows
        .iter()
        .map(|row| "x".repeat(row.payload_bytes))
        .collect::<Vec<_>>();
    let record = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(
                rows.iter().map(|row| row.key).collect::<Vec<_>>(),
            )),
            Arc::new(
                TimestampMicrosecondArray::from(
                    rows.iter().map(|row| row.time).collect::<Vec<_>>(),
                )
                .with_timezone("UTC"),
            ),
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.sequence).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(payloads)),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn oracle(left: &[Row], right: &[Row]) -> BTreeMap<i64, (i64, Option<i64>)> {
    let mut by_key = BTreeMap::<u64, Vec<(i64, i64)>>::new();
    for row in right {
        by_key
            .entry(row.key)
            .or_default()
            .push((row.time, row.sequence));
    }
    for history in by_key.values_mut() {
        history.sort_unstable();
    }
    left.iter()
        .map(|row| {
            let candidate = by_key.get(&row.key).and_then(|history| {
                history[..history.partition_point(|(time, _)| *time <= row.time)]
                    .last()
                    .filter(|(time, _)| *time >= row.time - TOLERANCE)
                    .map(|(_, sequence)| *sequence)
            });
            (row.sequence, (row.time, candidate))
        })
        .collect()
}

struct OracleCollector {
    expected: BTreeMap<i64, (i64, Option<i64>)>,
    seen: BTreeSet<i64>,
    max_output_bytes: usize,
    max_output_rows: usize,
}

#[async_trait]
impl StreamCollector for OracleCollector {
    async fn emit(&mut self, port: &str, batch: Batch) -> Result<()> {
        assert_eq!(port, "output");
        self.max_output_bytes = self.max_output_bytes.max(batch.estimated_bytes()?);
        for record in batch.table_payload()?.batches() {
            self.max_output_rows = self.max_output_rows.max(record.num_rows());
            let column = |name: &str| record.column_by_name(name).unwrap();
            let left_sequence = column("left__sequence")
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let left_time = column("left__time")
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let right_sequence = column("right__sequence")
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for index in 0..record.num_rows() {
                let sequence = left_sequence.value(index);
                let selected =
                    (!right_sequence.is_null(index)).then(|| right_sequence.value(index));
                assert_eq!((left_time.value(index), selected), self.expected[&sequence]);
                assert!(
                    self.seen.insert(sequence),
                    "duplicate logical left result {sequence}"
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct Observation {
    accepted_inputs: u64,
    emitted_left: u64,
    peak_state_rows: u64,
    peak_charged_bytes: u64,
    peak_snapshot_capacity: usize,
    state_bytes_after_eof: Option<u64>,
    reset_released_bytes: i64,
    failure_reason: Option<StreamingFailureReason>,
}

impl Observation {
    fn observe(&mut self, op: &mut StreamAsofJoinOperator) {
        let status = op.status();
        assert!(status.state_rows <= 100_000);
        assert!(status.state_bytes <= STATE_BYTES);
        self.peak_state_rows = self.peak_state_rows.max(status.state_rows);
        self.peak_charged_bytes = self.peak_charged_bytes.max(status.state_bytes);
        self.accepted_inputs = status.left.accepted_rows + status.right.accepted_rows;
        self.emitted_left = status.emitted_left_rows;
        let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
        let capacity = snapshot
            .segments
            .values()
            .map(|segment| segment.bytes_arc().capacity())
            .sum::<usize>();
        assert!(u64::try_from(capacity).unwrap() <= status.state_bytes);
        self.peak_snapshot_capacity = self.peak_snapshot_capacity.max(capacity);
    }

    fn record_failure(&mut self, error: &CalcFlowError, op: &StreamAsofJoinOperator) {
        let CalcFlowError::OperatorReason { reason_code, .. } = error else {
            panic!("resource trace returned an unstructured failure: {error:?}");
        };
        assert!(
            matches!(
                reason_code,
                StreamingFailureReason::AsofStateLimitExceeded
                    | StreamingFailureReason::AsofWorkspaceLimitExceeded
            ),
            "unexpected resource reason: {reason_code:?}"
        );
        self.failure_reason = Some(*reason_code);
        let status = op.status();
        assert_eq!(
            status.state_limit_failures + status.workspace_limit_failures,
            1
        );
    }
}

fn progress(left: Option<i64>, right: Option<i64>) -> IngressProgressSnapshot {
    IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, left.map(EventTime::from_micros)),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, right.map(EventTime::from_micros)),
        ),
    ]))
}

async fn run_trace(trace: Trace, observation: &mut Observation) {
    let left = rows(trace, true);
    let right = rows(trace, false);
    assert_eq!(left.len() + right.len(), 100_000);
    assert_eq!(
        right
            .iter()
            .map(|row| row.key)
            .collect::<BTreeSet<_>>()
            .len(),
        KEYS
    );
    let schema = schema();
    let mut op = operator(&schema);
    let job = StreamJobContext::new(
        1,
        "asof-resource-trace",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let mut collector = OracleCollector {
        expected: oracle(&left, &right),
        seen: BTreeSet::new(),
        max_output_bytes: 0,
        max_output_rows: 0,
    };
    let mut watermarks = [None, None];
    'rounds: for round in 0..ROWS_PER_SIDE / KEYS {
        for (side, input) in [("right", &right), ("left", &left)] {
            for chunk in input[round * KEYS..(round + 1) * KEYS].chunks(BATCH_ROWS) {
                let cx = StreamOperatorContext::with_ingress_progress(
                    &job,
                    "asof",
                    None,
                    progress(watermarks[0], watermarks[1]),
                );
                let before = op.status();
                if let Err(error) = op
                    .process_data(side, batch(&schema, chunk), &cx, &mut collector)
                    .await
                {
                    let after = op.status();
                    assert_eq!(after.left.accepted_rows, before.left.accepted_rows);
                    assert_eq!(after.right.accepted_rows, before.right.accepted_rows);
                    assert_eq!(after.state_rows, before.state_rows);
                    assert_eq!(after.state_bytes, before.state_bytes);
                    observation.record_failure(&error, &op);
                    observation.observe(&mut op);
                    break 'rounds;
                }
                observation.observe(&mut op);
            }
        }
        let next = i64::try_from(round).unwrap() * ROUND_STEP + 6;
        watermarks[0] = Some(next);
        if round == 0 || !matches!(trace, Trace::Stalled | Trace::Wide) {
            watermarks[1] = Some(next);
        }
        let cx = StreamOperatorContext::with_ingress_progress(
            &job,
            "asof",
            None,
            progress(watermarks[0], watermarks[1]),
        );
        if let Err(error) = op
            .on_watermark(EventTime::from_micros(next), &cx, &mut collector)
            .await
        {
            observation.record_failure(&error, &op);
            observation.observe(&mut op);
            break;
        }
        observation.observe(&mut op);
        assert_retained_history(&op, trace, round);
    }
    let cx = StreamOperatorContext::with_ingress_progress(
        &job,
        "asof",
        None,
        progress(watermarks[0], watermarks[1]),
    );
    finish_trace(&mut op, &cx, &mut collector, observation).await;
}

fn assert_retained_history(op: &StreamAsofJoinOperator, trace: Trace, round: usize) {
    if matches!(trace, Trace::Advancing | Trace::HotKey) {
        assert_eq!(op.status().pending_left_rows, 0);
        assert_eq!(
            op.status().retained_right_rows,
            u64::try_from((round + 1).min(4) * KEYS).unwrap()
        );
    }
}

async fn finish_trace(
    op: &mut StreamAsofJoinOperator,
    cx: &StreamOperatorContext<'_>,
    collector: &mut OracleCollector,
    observation: &mut Observation,
) {
    if observation.failure_reason.is_none() {
        op.on_end(cx, collector).await.unwrap();
        observation.observe(op);
        assert_eq!(op.status().state_rows, 0);
        observation.state_bytes_after_eof = Some(op.status().state_bytes);
        assert_eq!(collector.seen.len(), ROWS_PER_SIDE);
        assert_eq!(
            op.status().matched_rows,
            u64::try_from(ROWS_PER_SIDE).unwrap()
        );
    } else {
        assert_eq!(collector.seen, (1..=i64::try_from(KEYS).unwrap()).collect());
        assert!(observation.accepted_inputs < 100_000);
    }
    assert_eq!(
        u64::try_from(collector.seen.len()).unwrap(),
        op.status().emitted_left_rows
    );
    assert!(collector.max_output_rows <= EdgeBudget::default().max_rows);
    assert!(collector.max_output_bytes <= EdgeBudget::default().max_bytes);
    let owned = op
        .checkpoint(Epoch::INITIAL)
        .unwrap()
        .segments
        .values()
        .map(|segment| Arc::downgrade(&segment.bytes_arc()))
        .collect::<Vec<_>>();
    let reset = allocation_counter::measure(|| op.reset().unwrap());
    observation.reset_released_bytes = -reset.bytes_current;
    assert!(owned.iter().all(|segment| segment.upgrade().is_none()));
    assert_eq!(op.status().state_rows, 0);
    assert_eq!(op.status().state_bytes, 0);
}

fn verify_trace(trace: Trace, should_fail: bool) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut observation = Observation::default();
    let allocations =
        allocation_counter::measure(|| runtime.block_on(run_trace(trace, &mut observation)));
    assert_eq!(
        observation.failure_reason.is_some(),
        should_fail,
        "{observation:?}"
    );
    assert!(observation.peak_snapshot_capacity > 0);
    assert!(observation.reset_released_bytes > 0);
    eprintln!(
        "ASOF resource {trace:?}: keys={KEYS}, generated_inputs=100000, left:right=1:1, seed=0 (formula), schema=(UInt64,UTC-us,Int64,Utf8), payload_bytes=8 (Wide after round0:16384), key_distribution=uniform (HotKey after round0:900/1000 at key0), round_step_us={ROUND_STEP}, generated_final_left_time_us=735000005, watermarks=round_time+6 (Stalled/Wide right freezes at6), tolerance_us={TOLERANCE}, batch_rows={BATCH_ROWS}, state_rows=100000, state_bytes={STATE_BYTES}, workspace_limit={STATE_BYTES}, edge={:?}; {observation:?}; current_thread_heap_peak={} (includes harness and DataFusion; not workspace or RSS)",
        EdgeBudget::default(),
        allocations.bytes_max
    );
}

#[test]
fn test_asof_resource_advancing_watermarks_release_history() {
    verify_trace(Trace::Advancing, false);
}

#[test]
fn test_asof_resource_stalled_right_fails_with_correct_output_prefix() {
    verify_trace(Trace::Stalled, true);
}

#[test]
fn test_asof_resource_hot_key_matches_oracle_with_bounded_history() {
    verify_trace(Trace::HotKey, false);
}

#[test]
fn test_asof_resource_wide_payload_fails_without_partial_admission() {
    verify_trace(Trace::Wide, true);
}

#[test]
fn test_asof_checkpoint_encoding_allocations_scale_with_retained_state() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let schema = schema();
    let mut op = operator(&schema);
    let job = StreamJobContext::new(
        2,
        "asof-encoding-allocation",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let cx = StreamOperatorContext::new(&job, "asof", None);
    let mut collector = OracleCollector {
        expected: BTreeMap::new(),
        seen: BTreeSet::new(),
        max_output_bytes: 0,
        max_output_rows: 0,
    };
    let right = rows(Trace::Advancing, false);
    runtime.block_on(async {
        for chunk in right[..KEYS].chunks(BATCH_ROWS) {
            op.process_data("right", batch(&schema, chunk), &cx, &mut collector)
                .await
                .unwrap();
        }
    });
    let charged = op.status().state_bytes;
    let empty = batch(&schema, &[]);
    let allocations = allocation_counter::measure(|| {
        runtime
            .block_on(op.process_data("right", empty, &cx, &mut collector))
            .unwrap();
    });
    assert_eq!(op.status().state_bytes, charged);
    assert!(
        allocations.bytes_total < charged * 16,
        "checkpoint encoding must not repeatedly copy its full growing buffer: charged={charged}, allocations={allocations:?}"
    );
}
