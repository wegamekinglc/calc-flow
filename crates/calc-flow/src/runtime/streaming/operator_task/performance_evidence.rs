//! Independent single-thread callback measurement, outside the CI timing gate.

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use datafusion::arrow::{
    array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    ipc::writer::StreamWriter,
    record_batch::RecordBatch,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, watch};

use super::{
    MetricsRecorder, OperatorProgress, OperatorTaskInputs, dispatch_data,
    dispatch_watermark_handler,
};
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeBudget, EdgeReceiver, Epoch, EventTime,
    IngressProgressSnapshot, JsonMap, OperatorMetadata, RollingOperator, RollingSpec,
    StreamJobContext, StreamMessage,
    operator::rolling_metrics::RollingMetricsStore,
    pipeline::{CompiledStreamOperator, OperatorCheckpointCapability},
};

#[derive(Clone, Copy, Debug)]
struct Workload {
    rows: usize,
    entities: usize,
    window: usize,
    dual: bool,
    lag1: bool,
    first_payload_bytes: usize,
}

impl Workload {
    fn identity(self) -> Value {
        let means = if self.dual { "mean5,mean20" } else { "mean" };
        json!({"rows": self.rows, "entities": self.entities, "window": self.window,
            "outputs": if self.lag1 { format!("{means},lag1") } else { means.to_owned() },
            "min_periods": "full window", "input": "Float64 exact binary fractions",
            "first_payload_bytes": self.first_payload_bytes})
    }

    /// Rolling mean outputs per row, the G2 window-row multiplier.
    const fn window_outputs(self) -> usize {
        if self.dual { 2 } else { 1 }
    }

    fn schema(self) -> Arc<Schema> {
        let mut fields = vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ];
        if self.first_payload_bytes != 0 {
            fields.push(Field::new("payload", DataType::Utf8, false));
        }
        Arc::new(Schema::new(fields))
    }

    fn batch(self, start: usize) -> Batch {
        self.rows_batch(start, self.rows)
    }

    fn rows_batch(self, start: usize, rows: usize) -> Batch {
        let end = start + rows;
        let timestamps = (start..end).map(|row| i64::try_from(row / self.entities).unwrap());
        let symbols = (0..self.entities)
            .map(|entity| format!("S{entity:04}"))
            .collect::<Vec<_>>();
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(TimestampMicrosecondArray::from_iter_values(timestamps).with_timezone("UTC")),
            Arc::new(UInt64Array::from_iter_values(
                (start..end).map(|row| u64::try_from(row).unwrap()),
            )),
            Arc::new(StringArray::from_iter_values(
                (start..end).map(|row| symbols[row % self.entities].as_str()),
            )),
            Arc::new(Float64Array::from_iter_values((start..end).map(|row| {
                100.0 + f64::from(u32::try_from(row % 257).unwrap()) / 8.0
            }))),
        ];
        if self.first_payload_bytes != 0 {
            let large = if start == 0 {
                "x".repeat(self.first_payload_bytes)
            } else {
                String::new()
            };
            columns.push(Arc::new(StringArray::from_iter_values((start..end).map(
                |row| {
                    if row == 0 { large.as_str() } else { "s" }
                },
            ))));
        }
        Batch::table(
            vec![RecordBatch::try_new(self.schema(), columns).unwrap()],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    fn spec(self) -> RollingSpec {
        let output = |window: usize, name: &str| {
            json!({
                "kind": "mean", "primitive_version": 1, "input": "price", "output": name,
                "frame": {"kind": "rows", "size": window}, "min_periods": window,
            })
        };
        let mut outputs = vec![output(self.window, "mean")];
        if self.dual {
            outputs.push(output(5, "mean5"));
        }
        if self.lag1 {
            outputs.push(json!({
                "kind": "lag", "primitive_version": 1, "input": "price", "output": "lag1",
                "periods": 1,
            }));
        }
        serde_json::from_value(json!({
            "configuration_version": 1, "state_layout_version": 1,
            "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["sequence"],
            "outputs": outputs, "allowed_lateness_micros": 0,
            "late_policy": {"kind": "error", "scope": "envelope"},
            "value_policy": "stateful_numeric_v1",
        }))
        .unwrap()
    }
}

struct Fixture {
    inputs: OperatorTaskInputs,
    receiver: EdgeReceiver,
    workload: Workload,
    previous_watermark: Option<EventTime>,
}

impl Fixture {
    fn new(workload: Workload, observed: bool) -> Self {
        Self::with_edge_rows(workload, observed, workload.rows)
    }

    /// `edge_rows` must admit the largest single dispatch's output: the
    /// fixture drains only after a whole callback returns.
    fn with_edge_rows(workload: Workload, observed: bool, edge_rows: usize) -> Self {
        let operator = RollingOperator::new("rolling", workload.schema(), workload.spec()).unwrap();
        let output_ports = operator
            .output_ports()
            .iter()
            .map(|port| (port.name().to_owned(), port.clone()))
            .collect();
        let (sender, receiver) = crate::edge_channel(
            "rolling->sink",
            EdgeBudget {
                max_rows: edge_rows,
                max_bytes: 64 << 20,
            },
        )
        .unwrap();
        let job = StreamJobContext::new(
            7,
            "callback-evidence",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let (_entry, entry_gate) = watch::channel(true);
        let (_data, data_gate) = watch::channel(true);
        let (entry_ack, _ack) = mpsc::unbounded_channel();
        Self {
            inputs: OperatorTaskInputs {
                late_output_ports: std::collections::BTreeSet::new(),
                entity_work: None,
                sql_recovery: None,
                node_id: "rolling".into(),
                operator: CompiledStreamOperator::Rolling(operator),
                checkpoint_capability: OperatorCheckpointCapability::CheckpointedStateful {
                    state_version: 1,
                },
                ingresses: BTreeMap::new(),
                outputs: BTreeMap::from([("output".into(), vec![sender.into()])]),
                output_ports,
                context: job.for_node("rolling").unwrap(),
                progress: OperatorProgress::with_optional_rolling_metrics(
                    observed.then(RollingMetricsStore::default),
                ),
                metrics: MetricsRecorder::default().operator("rolling").unwrap(),
                entry_gate,
                entry_ack,
                data_gate,
                launch_cancel: CancellationToken::new(),
                checkpoint: None,
                restore: None,
            },
            receiver,
            workload,
            previous_watermark: None,
        }
    }

    async fn dispatch(&mut self, batch: &Batch, start: usize) {
        dispatch_data(
            &mut self.inputs,
            "input",
            StreamMessage::data(batch.clone()),
            self.previous_watermark,
            IngressProgressSnapshot::default(),
        )
        .await
        .unwrap();
        let watermark = EventTime::from_micros(
            i64::try_from((start + batch.num_rows() - 1) / self.workload.entities).unwrap(),
        );
        dispatch_watermark_handler(
            &mut self.inputs,
            "input",
            watermark,
            self.previous_watermark,
            IngressProgressSnapshot::default(),
        )
        .await
        .unwrap();
        self.previous_watermark = Some(watermark);
    }

    async fn drain(&mut self) -> Vec<RecordBatch> {
        self.drain_rows(self.workload.rows).await
    }

    async fn drain_rows(&mut self, expected: usize) -> Vec<RecordBatch> {
        let mut rows = 0;
        let mut output = Vec::new();
        while rows < expected {
            let message = self.receiver.recv().await.unwrap().unwrap();
            let batch = message.as_data().unwrap();
            rows += batch.num_rows();
            output.extend_from_slice(batch.table_payload().unwrap().batches());
        }
        assert_eq!(rows, expected);
        assert_eq!(self.receiver.metrics().charged_rows, 0);
        output
    }

    /// Cumulative `numeric_update` stage time over every callback kind.
    fn numeric_update(&self) -> Duration {
        self.inputs
            .progress
            .rolling_metrics()
            .map_or(Duration::ZERO, |metrics| {
                metrics.data.numeric_update_duration
                    + metrics.watermark.numeric_update_duration
                    + metrics.end.numeric_update_duration
            })
    }
}

fn assert_same_state(left: &mut Fixture, right: &mut Fixture) {
    let epoch = Epoch::new(1).unwrap();
    let left = left.inputs.operator.checkpoint(epoch).unwrap();
    let right = right.inputs.operator.checkpoint(epoch).unwrap();
    assert_eq!(left.inline_metadata, right.inline_metadata);
    assert_eq!(left.segments, right.segments);
}

fn allocation_json(value: allocation_counter::AllocationInfo) -> Value {
    json!({"allocation_count": value.count_total, "allocated_bytes": value.bytes_total,
        "peak_net_allocation_count": value.count_max, "peak_net_bytes": value.bytes_max,
        "ending_net_allocation_count": value.count_current, "ending_net_bytes": value.bytes_current})
}

#[test]
fn recorder_presence_preserves_complete_callback_output_and_checkpoint() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for workload in [
        Workload {
            rows: 10,
            entities: 1,
            window: 20,
            dual: false,
            lag1: false,
            first_payload_bytes: 0,
        },
        Workload {
            rows: 256,
            entities: 4,
            window: 20,
            dual: true,
            lag1: false,
            first_payload_bytes: 0,
        },
    ] {
        let mut absent = Fixture::new(workload, false);
        let mut present = Fixture::new(workload, true);
        for step in 0..3 {
            let batch = workload.batch(step * workload.rows);
            runtime.block_on(absent.dispatch(&batch, step * workload.rows));
            runtime.block_on(present.dispatch(&batch, step * workload.rows));
            assert_eq!(
                runtime.block_on(absent.drain()),
                runtime.block_on(present.drain())
            );
        }
        assert_same_state(&mut absent, &mut present);
        assert!(absent.inputs.progress.rolling_metrics().is_none());
        let metrics = present.inputs.progress.rolling_metrics().unwrap();
        assert!(!metrics.overflowed);
        assert_eq!(metrics.data.started, 3);
        assert_eq!(metrics.data.succeeded, 3);
        assert_eq!(metrics.watermark.started, 3);
        assert_eq!(metrics.watermark.succeeded, 3);
        assert_eq!(
            metrics.watermark.numeric_rows,
            u64::try_from(workload.rows * 3).unwrap()
        );
    }
}

fn retention_probe(runtime: &tokio::runtime::Runtime, workload: Workload, batches: usize) -> Value {
    let mut retained = None;
    let allocations = allocation_counter::measure(|| {
        let mut fixture = Fixture::new(workload, true);
        for step in 0..batches {
            let batch = workload.batch(step * workload.rows);
            runtime.block_on(fixture.dispatch(&batch, step * workload.rows));
            drop(runtime.block_on(fixture.drain()));
        }
        retained = Some(fixture);
    });
    drop(retained);
    json!({"batches": batches, "rows_processed": batches * workload.rows,
        "scope": "fixture creation, inputs, full callbacks and output drain/drop; fixture retained at end",
        "allocations": allocation_json(allocations)})
}

fn payload_retention_probe(runtime: &tokio::runtime::Runtime, appends: usize) -> Value {
    // The large value shares the first twenty-row chunk with nineteen small
    // rows. A one-row initial chunk would disappear as a whole and miss sliced
    // array-buffer retention after the large row leaves the window.
    let first = Workload {
        rows: 20,
        entities: 1,
        window: 20,
        dual: false,
        lag1: false,
        first_payload_bytes: 4 << 20,
    };
    let mut retained = None;
    let allocations = allocation_counter::measure(|| {
        let mut fixture = Fixture::new(first, true);
        {
            let batch = first.batch(0);
            runtime.block_on(fixture.dispatch(&batch, 0));
            drop(runtime.block_on(fixture.drain()));
        }
        fixture.workload.rows = 1;
        for index in 0..appends {
            let start = first.rows + index;
            let batch = fixture.workload.batch(start);
            runtime.block_on(fixture.dispatch(&batch, start));
            drop(runtime.block_on(fixture.drain()));
        }
        retained = Some(fixture);
    });
    drop(retained);
    json!({"initial_batch_rows": 20, "append_batch_rows": 1, "appends": appends,
        "large_first_payload_bytes": 4 << 20, "entities": 1, "window": 20,
        "scope": "all input and output handles released, live operator fixture retained",
        "allocations": allocation_json(allocations)})
}

fn evidence_workloads() -> [Workload; 4] {
    [
        Workload {
            rows: 10,
            entities: 1,
            window: 20,
            dual: false,
            lag1: false,
            first_payload_bytes: 0,
        },
        Workload {
            rows: 8192,
            entities: 64,
            window: 20,
            dual: false,
            lag1: false,
            first_payload_bytes: 0,
        },
        Workload {
            rows: 8192,
            entities: 64,
            window: 20,
            dual: true,
            lag1: false,
            first_payload_bytes: 0,
        },
        Workload {
            rows: 8192,
            entities: 1,
            window: 4096,
            dual: false,
            lag1: false,
            first_payload_bytes: 0,
        },
    ]
}

#[test]
fn evidence_case_selection_rejects_unknown_empty_and_duplicate_names() {
    let selected = select_cases("6400x64-dual-lag1,64x64-dual-tick,1x1-dual").unwrap();
    assert_eq!(
        selected.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        ["6400x64-dual-lag1", "64x64-dual-tick", "1x1-dual"]
    );
    let session = select_cases("session-g0").unwrap_err();
    assert!(session.contains("unknown"), "{session}");
    let unknown = select_cases("1x1-dual,1x2-dual").unwrap_err();
    assert!(unknown.contains("`1x2-dual`"), "{unknown}");
    assert!(unknown.contains("64x64-dual-tick"), "{unknown}");
    for empty in ["", "1x1-dual,", "1x1-dual,,99x1-dual"] {
        assert!(
            select_cases(empty).unwrap_err().contains("empty"),
            "{empty}"
        );
    }
    let duplicate = select_cases("99x1-dual,99x1-dual").unwrap_err();
    assert!(duplicate.contains("more than once"), "{duplicate}");
}

#[test]
#[ignore = "dedicated performance task: release current-thread diagnostic, not a CI timing gate"]
fn native_callback_allocation_and_overhead_evidence() {
    // Keep this guard at runtime so ignored diagnostics also compile in debug builds.
    assert!(
        !std::hint::black_box(cfg!(debug_assertions)),
        "callback performance evidence requires a release test binary"
    );
    let destination =
        std::env::var("CALC_FLOW_NATIVE_EVIDENCE_OUTPUT").expect("set evidence output path");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut cases = Vec::new();
    for workload in evidence_workloads() {
        let mut arms = [Fixture::new(workload, false), Fixture::new(workload, true)];
        // Each arm has the same prefix/state. Prefix and validation are outside
        // the dispatch timer; no allocation counter is active for timings.
        let warm = workload.batch(0);
        for arm in &mut arms {
            runtime.block_on(arm.dispatch(&warm, 0));
            drop(runtime.block_on(arm.drain()));
        }
        let mut samples = Vec::new();
        for pair in 0..10 {
            let start = (pair + 1) * workload.rows;
            let batch = workload.batch(start);
            let mut seconds = [0.0; 2];
            for index in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                let began = Instant::now();
                runtime.block_on(arms[index].dispatch(&batch, start));
                seconds[index] = began.elapsed().as_secs_f64();
            }
            assert_eq!(
                runtime.block_on(arms[0].drain()),
                runtime.block_on(arms[1].drain())
            );
            samples.push(
                json!({"pair": pair, "order": if pair % 2 == 0 { "AB" } else { "BA" },
                "absent_seconds": seconds[0], "present_seconds": seconds[1]}),
            );
        }
        let (left, right) = arms.split_at_mut(1);
        assert_same_state(&mut left[0], &mut right[0]);
        let start = 11 * workload.rows;
        let batch = workload.batch(start);
        let allocations = arms
            .iter_mut()
            .map(|arm| {
                let counts =
                    allocation_counter::measure(|| runtime.block_on(arm.dispatch(&batch, start)));
                drop(runtime.block_on(arm.drain()));
                allocation_json(counts)
            })
            .collect::<Vec<_>>();
        let observations = arms[1].inputs.progress.rolling_metrics().unwrap();
        assert!(!observations.overflowed);
        let retained_memory =
            [1, 16, 128].map(|batches| retention_probe(&runtime, workload, batches));
        cases.push(
            json!({"workload": workload.identity(), "paired_samples": samples,
            "allocation_samples": {"absent": allocations[0], "present": allocations[1]},
            "complete_callback_metrics": observations,
            "retained_memory": retained_memory}),
        );
    }
    let payload_retention =
        [0, 1, 16, 128].map(|appends| payload_retention_probe(&runtime, appends));
    let evidence = json!({
        "contract": "calc-flow.native-callback-allocation-overhead/1",
        "executable": std::env::current_exe().unwrap(),
        "debug_assertions": cfg!(debug_assertions),
        "boundary": "current-thread block_on of real dispatch_data plus dispatch_watermark_handler, including rolling/collector/queue enqueue/callback publication",
        "limitations": [
            "No source/sink task scheduling, startup, shutdown, queue drain, checkpoint serialization, input creation or assertions in timer.",
            "Allocation counter is thread-local; these are not whole-process or multithreaded Tokio counts.",
            "Callback allocation ending/peak values are net changes: freeing preexisting state can lower them; they are not total live heap gauges.",
            "Retained-memory probe includes fixture/runtime diagnostics/channel state and input creation; all caller input/output handles are dropped before its endpoint.",
            "Timing uses one same-process round of ten AB/BA pairs. Two fresh-process rounds and exact build/host provenance must be attached by the controller before classification.",
            "The dual fixture computes two rolling means; the separate end-to-end driver includes spread projection.",
        ],
        "cases": cases,
        "wide_payload_retention": payload_retention,
    });
    std::fs::write(
        destination,
        serde_json::to_string_pretty(&evidence).unwrap() + "\n",
    )
    .unwrap();
}

/// Rows dispatched before warm-up so every window is full (§6.5 history).
const HISTORY_ROWS: usize = 4_096;
const WARM_DISPATCHES: usize = 3;
/// Timed dispatches per arm and case in one evidence round (spec §10.1).
const TIMED_SAMPLES: usize = 30;

const fn dual(rows: usize, entities: usize, lag1: bool) -> Workload {
    Workload {
        rows,
        entities,
        window: 20,
        dual: true,
        lag1,
        first_payload_bytes: 0,
    }
}

/// The rolling-kernel-v2 dispatch cases (spec §9, P2.1): mean(5) and mean(20).
const DISPATCH_CASES: [(&str, Workload); 5] = [
    ("1x1-dual", dual(1, 1, false)),
    ("99x1-dual", dual(99, 1, false)),
    ("6400x64-dual", dual(6_400, 64, false)),
    ("64x64-dual-tick", dual(64, 64, false)),
    ("6400x64-dual-lag1", dual(6_400, 64, true)),
];

fn known_cases() -> String {
    DISPATCH_CASES
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Parses a `CALC_FLOW_NATIVE_EVIDENCE_CASES` comma-separated selection.
fn select_cases(selection: &str) -> Result<Vec<(&'static str, Workload)>, String> {
    selection
        .split(',')
        .try_fold(Vec::new(), |mut selected, name| {
            if name.is_empty() {
                return Err(format!(
                    "empty evidence case name in `{selection}`; known cases: {}",
                    known_cases()
                ));
            }
            let case = DISPATCH_CASES
                .iter()
                .find(|(known, _)| *known == name)
                .copied()
                .ok_or_else(|| {
                    format!(
                        "unknown evidence case `{name}`; known cases: {}",
                        known_cases()
                    )
                })?;
            if selected
                .iter()
                .any(|(chosen, _): &(&str, Workload)| *chosen == name)
            {
                return Err(format!("evidence case `{name}` is selected more than once"));
            }
            selected.push(case);
            Ok(selected)
        })
}

fn dispatch_both(
    runtime: &tokio::runtime::Runtime,
    arms: &mut [Fixture; 2],
    batch: &Batch,
    start: usize,
) {
    for arm in arms.iter_mut() {
        runtime.block_on(arm.dispatch(batch, start));
    }
    let [absent, present] = arms;
    assert_eq!(
        runtime.block_on(absent.drain_rows(batch.num_rows())),
        runtime.block_on(present.drain_rows(batch.num_rows()))
    );
}

fn output_digest(records: &[RecordBatch]) -> String {
    let mut bytes = Vec::new();
    if let Some(first) = records.first() {
        let mut writer = StreamWriter::try_new(&mut bytes, first.schema()).unwrap();
        for record in records {
            writer.write(record).unwrap();
        }
        writer.finish().unwrap();
    }
    hex::encode(Sha256::digest(bytes))
}

/// One round of one dispatch case: history, warm-up, then alternating timed
/// recorder-absent and recorder-present dispatches (spec §10.1, §10.2).
fn dispatch_case(runtime: &tokio::runtime::Runtime, name: &str, workload: Workload) -> Value {
    let edge_rows = workload.rows.max(HISTORY_ROWS);
    let mut arms = [
        Fixture::with_edge_rows(workload, false, edge_rows),
        Fixture::with_edge_rows(workload, true, edge_rows),
    ];
    dispatch_both(runtime, &mut arms, &workload.rows_batch(0, HISTORY_ROWS), 0);
    let mut start = HISTORY_ROWS;
    for _ in 0..WARM_DISPATCHES {
        dispatch_both(runtime, &mut arms, &workload.batch(start), start);
        start += workload.rows;
    }
    let mut samples = Vec::with_capacity(TIMED_SAMPLES);
    for index in 0..TIMED_SAMPLES {
        let batch = workload.batch(start);
        let mut seconds = [0.0; 2];
        let numeric_before = arms[1].numeric_update();
        for side in if index % 2 == 0 { [0, 1] } else { [1, 0] } {
            let began = Instant::now();
            runtime.block_on(arms[side].dispatch(&batch, start));
            seconds[side] = began.elapsed().as_secs_f64();
        }
        let numeric = arms[1]
            .numeric_update()
            .checked_sub(numeric_before)
            .expect("cumulative stage time never decreases");
        let [absent, present] = &mut arms;
        let absent_output = runtime.block_on(absent.drain());
        let present_output = runtime.block_on(present.drain());
        assert_eq!(absent_output, present_output);
        let digest = output_digest(&absent_output);
        samples.push(
            json!({"index": index, "order": if index % 2 == 0 { "AB" } else { "BA" },
            "absent_seconds": seconds[0], "present_seconds": seconds[1],
            "present_numeric_update_seconds": numeric.as_secs_f64(),
            "output_digest": digest}),
        );
        start += workload.rows;
    }
    let (left, right) = arms.split_at_mut(1);
    assert_same_state(&mut left[0], &mut right[0]);
    let batch = workload.batch(start);
    let allocations = arms.each_mut().map(|arm| {
        let counts = allocation_counter::measure(|| runtime.block_on(arm.dispatch(&batch, start)));
        drop(runtime.block_on(arm.drain()));
        allocation_json(counts)
    });
    let metrics = arms[1].inputs.progress.rolling_metrics().unwrap();
    assert!(!metrics.overflowed);
    json!({"name": name, "kind": "dispatch", "workload": workload.identity(),
        "rows": workload.rows, "window_outputs": workload.window_outputs(),
        "history_rows": HISTORY_ROWS, "warm_dispatches": WARM_DISPATCHES, "samples": samples,
        "allocation_samples": {"absent": allocations[0], "present": allocations[1]},
        "complete_callback_metrics": metrics})
}

#[test]
#[ignore = "dedicated performance task: one release evidence round, not a CI timing gate"]
fn rolling_kernel_v2_evidence() {
    // Keep this guard at runtime so ignored diagnostics also compile in debug builds.
    assert!(
        !std::hint::black_box(cfg!(debug_assertions)),
        "rolling-kernel-v2 evidence requires a release test binary"
    );
    let destination =
        std::env::var("CALC_FLOW_NATIVE_EVIDENCE_OUTPUT").expect("set evidence output path");
    let selection = std::env::var("CALC_FLOW_NATIVE_EVIDENCE_CASES")
        .expect("set CALC_FLOW_NATIVE_EVIDENCE_CASES to comma-separated evidence cases");
    let selected = select_cases(&selection).unwrap_or_else(|error| panic!("{error}"));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let cases = selected
        .iter()
        .map(|(name, workload)| dispatch_case(&runtime, name, *workload))
        .collect::<Vec<_>>();
    let evidence = json!({
        "contract": "calc-flow.rolling-kernel-v2-evidence/1",
        "executable": std::env::current_exe().unwrap(),
        "debug_assertions": cfg!(debug_assertions),
        "selection": selected.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        "timed_samples": TIMED_SAMPLES,
        "boundaries": {
            "dispatch": "current-thread block_on of real dispatch_data plus dispatch_watermark_handler on a RollingOperator, including the rolling collector, edge enqueue and callback publication; no projection; output drain outside the timer",
        },
        "limitations": [
            "One evidence round: a controller pairs fresh-process rounds of two binaries (profile_warm_stream.py native).",
            "Allocation counts are thread-local; the dispatch fixtures run on the calling thread.",
            "Test binaries install the counting global allocator for every allocation, timed or not, in both arms.",
            "Recorder-present numeric_update time comes from the arm's cumulative rolling metrics around each timed dispatch.",
        ],
        "cases": cases,
    });
    std::fs::write(
        destination,
        serde_json::to_string_pretty(&evidence).unwrap() + "\n",
    )
    .unwrap();
}
