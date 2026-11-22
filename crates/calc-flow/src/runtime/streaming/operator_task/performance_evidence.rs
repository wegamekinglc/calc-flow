//! Independent single-thread callback measurement, outside the CI timing gate.

use std::{
    collections::BTreeMap,
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
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
    Batch, BatchKind, BatchMetadata, CalcFlowError, CancellationToken, Cursor, Edge, EdgeBudget,
    EdgeReceiver, Epoch, EventTime, ExpressionOperator, IngressProgressSnapshot, JsonMap,
    ManagedCheckpointRuntime, NativeWatermarkCapability, OperatorMetadata, PipelineBuilder, Port,
    PortEndpoint, ReplayPositioning, RollingOperator, RollingSpec, SinkBinding, SourceBinding,
    SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema, StreamExecutionPlan,
    StreamJobContext, StreamMessage, StreamRequirements, StreamRuntimeConfig, StreamSession,
    StreamSink, StreamSource, StreamingRunner, UdfRegistry, WatermarkPolicy,
    operator::rolling_metrics::RollingMetricsStore,
    pipeline::{CompiledStreamOperator, OperatorCheckpointCapability},
    state::v4::{V4Manifest, V4Store},
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
        let mut writer = StreamWriter::try_new(&mut bytes, &first.schema()).unwrap();
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

struct DirectSessionArm {
    session: StreamSession,
    input: String,
    output: String,
    entities: usize,
    _root: Option<tempfile::TempDir>,
}

fn direct_plan(workload: Workload) -> StreamExecutionPlan {
    let rolling = RollingOperator::new("rolling", workload.schema(), workload.spec()).unwrap();
    let rolled = rolling.output_ports()[0].schema().unwrap().clone();
    let projected = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        true,
    )]));
    let projection = ExpressionOperator::new(
        "project",
        "",
        vec!["mean5 - mean AS value".into()],
        None,
        Vec::new(),
    )
    .unwrap()
    .with_ports(
        Port::with_schema_ref("input", BatchKind::Table, true, Some(rolled)).unwrap(),
        Port::with_schema_ref("output", BatchKind::Table, true, Some(projected)).unwrap(),
    )
    .unwrap();
    PipelineBuilder::new("checkpoint-v4-direct")
        .unwrap()
        .add_node("rolling", rolling)
        .unwrap()
        .add_node("project", Box::new(projection))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("rolling", "output").unwrap(),
            PortEndpoint::new("project", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn direct_sessions(workload: Workload) -> [DirectSessionArm; 2] {
    [true, false].map(|checkpointing| {
        let config = StreamRuntimeConfig {
            checkpointing,
            ..StreamRuntimeConfig::default()
        };
        let mut root = None;
        let session = if checkpointing {
            let directory = tempfile::tempdir().unwrap();
            let session = StreamSession::new_with_checkpoints(
                direct_plan(workload),
                &config,
                BTreeMap::new(),
                ManagedCheckpointRuntime::new(directory.path()).unwrap(),
            )
            .unwrap();
            root = Some(directory);
            session
        } else {
            StreamSession::new(direct_plan(workload), &config, BTreeMap::new()).unwrap()
        };
        let input = session.input_ids()[0].to_owned();
        let output = session.output_ids()[0].to_owned();
        DirectSessionArm {
            session,
            input,
            output,
            entities: workload.entities,
            _root: root,
        }
    })
}

fn direct_price(row: usize) -> f64 {
    100.0 + f64::from(u32::try_from(row % 257).unwrap()) / 8.0
}

fn direct_expected(row: usize, entities: usize) -> Option<f64> {
    let tick = row / entities;
    if tick < 19 {
        return None;
    }
    let entity = row % entities;
    let mean = |window| {
        (0..window)
            .map(|offset| direct_price((tick + 1 - window + offset) * entities + entity))
            .sum::<f64>()
            / f64::from(u32::try_from(window).unwrap())
    };
    Some(mean(5) - mean(20))
}

fn validate_direct_output(records: &[RecordBatch], start: usize, entities: usize) {
    let mut row = start;
    for record in records {
        let values = record
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for offset in 0..record.num_rows() {
            let expected = direct_expected(row, entities);
            assert_eq!(values.is_null(offset), expected.is_none(), "row {row}");
            if let Some(expected) = expected {
                assert_eq!(
                    values.value(offset).to_bits(),
                    expected.to_bits(),
                    "row {row}"
                );
            }
            row += 1;
        }
    }
}

fn direct_push(arm: &mut DirectSessionArm, batch: &Batch, start: usize) -> (f64, String) {
    let watermark = EventTime::from_micros(
        i64::try_from((start + batch.num_rows() - 1) / arm.entities).unwrap(),
    );
    let began = Instant::now();
    let emitted = arm
        .session
        .push(&arm.input.clone(), batch.clone(), Some(watermark))
        .unwrap();
    let seconds = began.elapsed().as_secs_f64();
    let records = emitted
        .batches(&arm.output)
        .iter()
        .flat_map(|batch| batch.table_payload().unwrap().batches().to_vec())
        .collect::<Vec<_>>();
    validate_direct_output(&records, start, arm.entities);
    (seconds, output_digest(&records))
}

#[test]
fn direct_checkpoint_sessions_match_independent_oracle() {
    let workload = dual(1, 1, false);
    let mut arms = direct_sessions(workload);
    for start in [0, 1, 2] {
        let batch = workload.batch(start);
        let [on, off] = arms.each_mut();
        let on_output = direct_push(on, &batch, start);
        let off_output = direct_push(off, &batch, start);
        assert_eq!(on_output.1, off_output.1);
    }
}

fn direct_median(
    arm: &mut DirectSessionArm,
    workload: Workload,
    mut start: usize,
    dispatches: usize,
) -> (f64, Vec<String>) {
    let mut seconds = Vec::with_capacity(dispatches);
    let mut digests = Vec::with_capacity(dispatches);
    for _ in 0..dispatches {
        let batch = workload.batch(start);
        let (elapsed, digest) = direct_push(arm, &batch, start);
        seconds.push(elapsed);
        digests.push(digest);
        start += workload.rows;
    }
    seconds.sort_unstable_by(f64::total_cmp);
    (
        f64::midpoint(seconds[dispatches / 2 - 1], seconds[dispatches / 2]),
        digests,
    )
}

fn direct_finish(arm: &mut DirectSessionArm, start: usize) -> String {
    let emitted = arm.session.finish().unwrap();
    let records = emitted
        .batches(&arm.output)
        .iter()
        .flat_map(|batch| batch.table_payload().unwrap().batches().to_vec())
        .collect::<Vec<_>>();
    validate_direct_output(&records, start, arm.entities);
    output_digest(&records)
}

fn direct_evidence_case(
    name: &str,
    workload: Workload,
    history_rows: usize,
    pairs: usize,
    dispatches: usize,
    recorder: &str,
) -> Value {
    let mut arms = direct_sessions(workload);
    for (start, rows) in [
        (0, history_rows),
        (history_rows, workload.rows),
        (history_rows + workload.rows, workload.rows),
        (history_rows + 2 * workload.rows, workload.rows),
    ] {
        let batch = workload.rows_batch(start, rows);
        let [on, off] = arms.each_mut();
        let on = direct_push(on, &batch, start);
        let off = direct_push(off, &batch, start);
        assert_eq!(on.1, off.1);
    }
    let mut start = history_rows + 3 * workload.rows;
    let mut samples = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let mut medians = [0.0; 2];
        let mut digests = [Vec::new(), Vec::new()];
        for side in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
            let (median, output) = direct_median(&mut arms[side], workload, start, dispatches);
            medians[side] = median;
            digests[side] = output;
        }
        assert_eq!(digests[0], digests[1]);
        samples.push(json!({
            "pair": pair,
            "order": if pair % 2 == 0 { "AB" } else { "BA" },
            "dispatches_per_arm": dispatches,
            "on_median_seconds": medians[0],
            "off_median_seconds": medians[1],
            "output_digest": digests[0],
        }));
        start += dispatches * workload.rows;
    }
    let [on, off] = arms.each_mut();
    assert_eq!(direct_finish(on, start), direct_finish(off, start));
    json!({
        "case": name,
        "workload": workload.identity(),
        "history_rows": history_rows,
        "warm_dispatches": 3,
        "recorder": recorder,
        "pairs": samples,
    })
}

#[test]
fn direct_checkpoint_pair_reports_median_and_digest() {
    let evidence = direct_evidence_case("1x1-dual", dual(1, 1, false), 3, 1, 2, "absent");
    assert_eq!(evidence["pairs"][0]["dispatches_per_arm"], 2);
    assert!(evidence["pairs"][0]["on_median_seconds"].is_number());
    assert!(evidence["pairs"][0]["off_median_seconds"].is_number());
    assert!(evidence["pairs"][0]["output_digest"].is_array());
}

#[cfg(feature = "benchmark-internals")]
#[test]
#[ignore = "P3.7 direct checkpoint performance evidence, not a CI timing gate"]
fn checkpoint_direct_evidence() {
    assert!(
        !std::hint::black_box(cfg!(debug_assertions)),
        "checkpoint direct evidence requires a release test binary"
    );
    let destination = std::env::var("CALC_FLOW_CHECKPOINT_V4_DIRECT_OUTPUT")
        .expect("set direct evidence output path");
    let name =
        std::env::var("CALC_FLOW_CHECKPOINT_V4_DIRECT_CASE").expect("set direct evidence case");
    let history_rows = std::env::var("CALC_FLOW_CHECKPOINT_V4_DIRECT_HISTORY")
        .expect("set direct history rows")
        .parse::<usize>()
        .unwrap();
    let recorder =
        std::env::var("CALC_FLOW_CHECKPOINT_V4_DIRECT_RECORDER").expect("set direct recorder");
    let pairs = std::env::var("CALC_FLOW_CHECKPOINT_V4_DIRECT_PAIRS")
        .expect("set direct pair count")
        .parse::<usize>()
        .unwrap();
    let dispatches = std::env::var("CALC_FLOW_CHECKPOINT_V4_DIRECT_DISPATCHES")
        .expect("set direct dispatch count")
        .parse::<usize>()
        .unwrap();
    assert_eq!(history_rows, HISTORY_ROWS);
    assert_eq!(recorder, "absent");
    assert!(pairs > 0 && dispatches >= 2 && dispatches % 2 == 0);
    let (name, workload) = DISPATCH_CASES
        .iter()
        .find(|(candidate, _)| {
            *candidate == name && matches!(name.as_str(), "1x1-dual" | "6400x64-dual")
        })
        .copied()
        .unwrap_or_else(|| panic!("unsupported direct checkpoint case {name}"));
    let evidence = direct_evidence_case(name, workload, history_rows, pairs, dispatches, &recorder);
    let mut evidence = evidence.as_object().unwrap().clone();
    evidence.extend([
        ("contract".into(), json!(DIRECT_CONTRACT)),
        ("executable".into(), json!(std::env::current_exe().unwrap())),
        ("debug_assertions".into(), json!(cfg!(debug_assertions))),
        (
            "process".into(),
            crate::benchmark_internals::process_observations().unwrap(),
        ),
        (
            "boundary".into(),
            json!("StreamSession::push through Rolling and mean5-mean projection; input construction, warm-up, output validation and finish are outside the timer; enabled arm captures no epoch"),
        ),
    ]);
    std::fs::write(
        destination,
        serde_json::to_string_pretty(&Value::Object(evidence)).unwrap() + "\n",
    )
    .unwrap();
}

const DIRECT_CONTRACT: &str = "calc-flow.checkpoint-v4-direct/1";

struct RunnerSource {
    workload: Workload,
    position: usize,
    watermark: Option<i64>,
    cutoff: watch::Receiver<Option<(Instant, Instant)>>,
}

#[async_trait::async_trait]
impl StreamSource for RunnerSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: self.workload.rows,
            max_batch_bytes: 64 << 20,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> crate::Result<()> {
        if let Some(cursor) = cursor {
            let bytes = cursor.order();
            assert_eq!(bytes.len(), size_of::<usize>());
            self.position = usize::from_be_bytes(bytes.try_into().unwrap());
        }
        self.watermark = None;
        Ok(())
    }

    async fn next(&mut self) -> crate::Result<Option<SourceEvent>> {
        let deadline = loop {
            if let Some((_, deadline)) = *self.cutoff.borrow() {
                break deadline;
            }
            self.cutoff.changed().await.unwrap();
        };
        if Instant::now() >= deadline {
            std::future::pending::<()>().await;
        }
        if let Some(watermark) = self.watermark.take() {
            return Ok(Some(SourceEvent::Watermark(EventTime::from_micros(
                watermark,
            ))));
        }
        let batch = self.workload.batch(self.position);
        self.position += self.workload.rows;
        self.watermark = Some(i64::try_from((self.position - 1) / self.workload.entities).unwrap());
        let cursor = Cursor::unbound(self.position.to_be_bytes().to_vec(), JsonMap::new()).unwrap();
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }

    async fn close(&mut self) -> crate::Result<()> {
        Ok(())
    }
}

#[derive(Clone)]
struct RunnerSink {
    next_row: Arc<std::sync::Mutex<usize>>,
    accepted: Arc<AtomicUsize>,
    entities: usize,
}

impl RunnerSink {
    fn new(entities: usize) -> Self {
        Self {
            next_row: Arc::default(),
            accepted: Arc::default(),
            entities,
        }
    }

    fn accepted_rows(&self) -> usize {
        self.accepted.load(Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl StreamSink for RunnerSink {
    async fn open(&mut self) -> crate::Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> crate::Result<()> {
        let start = *self.next_row.lock().unwrap();
        validate_direct_output(batch.table_payload()?.batches(), start, self.entities);
        *self.next_row.lock().unwrap() = start + batch.num_rows();
        self.accepted.fetch_add(batch.num_rows(), Ordering::Relaxed);
        Ok(())
    }

    async fn close(&mut self) -> crate::Result<()> {
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "benchmark_fixture".into(),
        message: message.into(),
    }
}

fn checkpoint_objects(path: &std::path::Path) -> crate::Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let mut count = 0;
    for entry in
        std::fs::read_dir(path).map_err(|error| invalid(format!("checkpoint objects: {error}")))?
    {
        let path = entry
            .map_err(|error| invalid(format!("checkpoint object entry: {error}")))?
            .path();
        if path.is_dir() {
            count += checkpoint_objects(&path)?;
        } else {
            count += 1;
        }
    }
    Ok(count)
}

fn read_manifest_if_present(path: &std::path::Path) -> crate::Result<Option<V4Manifest>> {
    match V4Store::read_manifest(path) {
        Ok(manifest) => Ok(Some(manifest)),
        Err(CalcFlowError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn nonterminal_epochs(root: &std::path::Path) -> crate::Result<Vec<u64>> {
    let directory = root.join("manifests/lineage/manifests");
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut epochs = Vec::new();
    for entry in std::fs::read_dir(directory)
        .map_err(|error| invalid(format!("checkpoint manifest directory: {error}")))?
    {
        let path = entry
            .map_err(|error| invalid(format!("checkpoint manifest entry: {error}")))?
            .path();
        let Some(manifest) = read_manifest_if_present(&path)? else {
            continue;
        };
        if manifest.sources.values().all(|source| !source.ended) {
            epochs.push(manifest.epoch);
        }
    }
    epochs.sort_unstable();
    epochs.dedup();
    Ok(epochs)
}

#[test]
fn retention_scan_ignores_only_manifests_that_disappear() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("epoch-11.json");
    assert!(read_manifest_if_present(&path).unwrap().is_none());
    std::fs::write(&path, b"not a manifest").unwrap();
    assert!(read_manifest_if_present(&path).is_err());
}

fn timed_nonterminal_epochs(root: &std::path::Path) -> (Vec<u64>, f64) {
    let started = Instant::now();
    let epochs = nonterminal_epochs(root).unwrap();
    (epochs, started.elapsed().as_secs_f64())
}

async fn runner_arm(
    workload: Workload,
    checkpointing: bool,
    seconds: u64,
    checkpoint_ms: u64,
    minimum_epochs: usize,
) -> Value {
    assert!(seconds > 0 && checkpoint_ms > 0);
    let root = tempfile::tempdir().unwrap();
    let (cutoff_tx, cutoff_rx) = watch::channel(None);
    let source = RunnerSource {
        workload,
        position: 0,
        watermark: None,
        cutoff: cutoff_rx,
    };
    let sink = RunnerSink::new(workload.entities);
    let plan = direct_plan(workload);
    let source_id = plan.source_binding_ids()[0].to_owned();
    let sink_id = plan.sink_binding_ids()[0].to_owned();
    let source = SourceBinding::new(source).with_watermark_policy(WatermarkPolicy::SourceProvided);
    let sources = BTreeMap::from([(source_id, source)]);
    let sinks = BTreeMap::from([(
        sink_id,
        vec![SinkBinding::ordinary("checkpoint-v4-sink", sink.clone()).unwrap()],
    )]);
    let config = StreamRuntimeConfig {
        checkpointing,
        checkpoint_interval: Duration::from_millis(checkpoint_ms),
        edge_budget: EdgeBudget {
            max_rows: workload.rows,
            max_bytes: 64 << 20,
        },
        retained_epochs: minimum_epochs.max(2),
        ..StreamRuntimeConfig::default()
    };
    let runner = if checkpointing {
        StreamingRunner::new(
            plan,
            sources,
            sinks,
            ManagedCheckpointRuntime::new(root.path()).unwrap(),
        )
        .unwrap()
    } else {
        StreamingRunner::without_checkpoints(plan, sources, sinks).unwrap()
    }
    .with_runtime_config(config)
    .unwrap();
    let job = runner.start().await.unwrap();
    let started = Instant::now();
    cutoff_tx
        .send(Some((started, started + Duration::from_secs(seconds))))
        .unwrap();
    tokio::time::sleep(Duration::from_secs(seconds)).await;
    let accepted_input_rows = sink.accepted_rows();
    assert!(accepted_input_rows > 0);
    let status_at_cutoff = serde_json::to_value(job.status()).unwrap();
    let (epochs, manifest_validation_seconds) = timed_nonterminal_epochs(root.path());
    assert!(
        epochs.len() >= minimum_epochs,
        "checkpoint coverage: {epochs:?}"
    );
    let freeze_to_ack_seconds = if checkpointing {
        let began = Instant::now();
        job.trigger_checkpoint().await.unwrap();
        Some(began.elapsed().as_secs_f64())
    } else {
        None
    };
    let post_cutoff_epochs = nonterminal_epochs(root.path()).unwrap();
    let cancel_started = Instant::now();
    let outcome = job.cancel().await;
    let cancel_seconds = cancel_started.elapsed().as_secs_f64();
    assert!(outcome.errors.is_empty(), "{outcome:?}");
    let settled_rows_after_cancel = sink.accepted_rows();
    assert!(settled_rows_after_cancel >= accepted_input_rows);
    let status_after_cancel = serde_json::to_value(job.status()).unwrap();
    let checkpoint_objects = checkpoint_objects(root.path()).unwrap();
    if checkpointing {
        assert!(checkpoint_objects > 0);
    } else {
        assert_eq!(checkpoint_objects, 0);
    }
    json!({
        "checkpointing": checkpointing,
        "seconds": seconds,
        "checkpoint_interval_ms": checkpoint_ms,
        "accepted_input_rows": accepted_input_rows,
        "emitted_rows": accepted_input_rows,
        "settled_rows_after_cancel": settled_rows_after_cancel,
        "rows_per_second": f64::from(u32::try_from(accepted_input_rows).unwrap()) / f64::from(u32::try_from(seconds).unwrap()),
        "completed_nonterminal_epochs": epochs.len(),
        "post_cutoff_nonterminal_epochs": post_cutoff_epochs.len(),
        "epoch_values": epochs,
        "checkpoint_objects": checkpoint_objects,
        "freeze_to_ack_seconds": freeze_to_ack_seconds,
        "manifest_validation_seconds": manifest_validation_seconds,
        "cancel_seconds": cancel_seconds,
        "status_at_cutoff": status_at_cutoff,
        "status_after_cancel": status_after_cancel,
        "recovery_validation_seconds": Value::Null,
        "cleanup_completed": true,
    })
}

#[tokio::test]
async fn runner_checkpoint_arms_cover_epoch_and_off_cleanup() {
    let workload = dual(6_400, 64, false);
    let on = runner_arm(workload, true, 2, 10, 0).await;
    assert!(on["accepted_input_rows"].as_u64().unwrap() > 0);
    assert!(on["post_cutoff_nonterminal_epochs"].as_u64().unwrap() >= 1);
    let off = runner_arm(workload, false, 2, 10, 0).await;
    assert!(off["accepted_input_rows"].as_u64().unwrap() > 0);
    assert_eq!(off["checkpoint_objects"], 0);
}

async fn runner_round(
    workload: Workload,
    round: usize,
    seconds: u64,
    checkpoint_ms: u64,
    pairs: usize,
    warmups: usize,
    minimum_epochs: usize,
) -> Value {
    let mut warmup_evidence = Vec::new();
    for checkpointing in [true, false] {
        let mut arms = Vec::with_capacity(warmups);
        for _ in 0..warmups {
            arms.push(
                runner_arm(
                    workload,
                    checkpointing,
                    seconds,
                    checkpoint_ms,
                    minimum_epochs,
                )
                .await,
            );
        }
        warmup_evidence.push(json!({"checkpointing": checkpointing, "arms": arms}));
    }
    let mut pair_evidence = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let mut on = Value::Null;
        let mut off = Value::Null;
        for checkpointing in if pair % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        } {
            let arm = runner_arm(
                workload,
                checkpointing,
                seconds,
                checkpoint_ms,
                minimum_epochs,
            )
            .await;
            if checkpointing {
                on = arm;
            } else {
                off = arm;
            }
        }
        pair_evidence.push(json!({
            "pair": pair,
            "order": if pair % 2 == 0 { "AB" } else { "BA" },
            "on": on,
            "off": off,
        }));
    }
    json!({
        "round": round,
        "warmups": warmup_evidence,
        "pairs": pair_evidence,
    })
}

#[tokio::test]
async fn runner_round_tracks_warmups_and_alternating_pairs() {
    let evidence = runner_round(dual(6_400, 64, false), 1, 2, 10, 2, 1, 0).await;
    assert_eq!(evidence["warmups"].as_array().unwrap().len(), 2);
    assert_eq!(evidence["warmups"][0]["arms"].as_array().unwrap().len(), 1);
    assert_eq!(evidence["warmups"][1]["arms"].as_array().unwrap().len(), 1);
    assert_eq!(evidence["pairs"].as_array().unwrap().len(), 2);
    assert_eq!(evidence["pairs"][0]["order"], "AB");
    assert_eq!(evidence["pairs"][1]["order"], "BA");
    assert!(
        evidence["pairs"][0]["on"]["post_cutoff_nonterminal_epochs"]
            .as_u64()
            .unwrap()
            >= 1
    );
}

#[cfg(feature = "benchmark-internals")]
#[test]
#[ignore = "P3.7 sustained runner performance evidence, not a CI timing gate"]
fn checkpoint_runner_evidence() {
    assert!(
        !std::hint::black_box(cfg!(debug_assertions)),
        "checkpoint runner evidence requires a release test binary"
    );
    let destination = std::env::var("CALC_FLOW_CHECKPOINT_V4_RUNNER_OUTPUT")
        .expect("set runner evidence output path");
    let entities = std::env::var("CALC_FLOW_CHECKPOINT_V4_RUNNER_ENTITIES")
        .expect("set runner entities")
        .parse::<usize>()
        .unwrap();
    let batch_rows = std::env::var("CALC_FLOW_CHECKPOINT_V4_RUNNER_BATCH_ROWS")
        .expect("set runner batch rows")
        .parse::<usize>()
        .unwrap();
    let checkpoint_ms = std::env::var("CALC_FLOW_CHECKPOINT_V4_RUNNER_CHECKPOINT_MS")
        .expect("set runner checkpoint interval")
        .parse::<u64>()
        .unwrap();
    let rounds = std::env::var("CALC_FLOW_CHECKPOINT_V4_RUNNER_ROUNDS")
        .map_or(2, |value| value.parse::<usize>().unwrap());
    let pairs = std::env::var("CALC_FLOW_CHECKPOINT_V4_RUNNER_PAIRS")
        .expect("set runner pairs")
        .parse::<usize>()
        .unwrap();
    let seconds = std::env::var("CALC_FLOW_CHECKPOINT_V4_RUNNER_SECONDS")
        .expect("set runner seconds")
        .parse::<u64>()
        .unwrap();
    let warmups = std::env::var("CALC_FLOW_CHECKPOINT_V4_RUNNER_WARMUPS")
        .expect("set runner warmups")
        .parse::<usize>()
        .unwrap();
    let minimum_epochs = std::env::var("CALC_FLOW_CHECKPOINT_V4_RUNNER_MIN_EPOCHS")
        .expect("set runner minimum epochs")
        .parse::<usize>()
        .unwrap();
    assert_eq!(entities, 64);
    assert_eq!(batch_rows, 6_400);
    assert_eq!(seconds, 5);
    assert_eq!(pairs, 10);
    assert_eq!(rounds, 2);
    assert_eq!(warmups, 1);
    let workload = dual(batch_rows, entities, false);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let evidence = runtime.block_on(async {
        let mut round_evidence = Vec::with_capacity(rounds);
        for round in 0..rounds {
            round_evidence.push(
                runner_round(
                    workload,
                    round + 1,
                    seconds,
                    checkpoint_ms,
                    pairs,
                    warmups,
                    minimum_epochs,
                )
                .await,
            );
        }
        round_evidence
    });
    let evidence = json!({
        "contract": "calc-flow.checkpoint-v4-runner/1",
        "executable": std::env::current_exe().unwrap(),
        "debug_assertions": cfg!(debug_assertions),
        "process": crate::benchmark_internals::process_observations().unwrap(),
        "entities": entities,
        "batch_rows": batch_rows,
        "checkpoint_interval_ms": checkpoint_ms,
        "seconds": seconds,
        "rounds": evidence,
        "boundary": "Five seconds of source-native replay through Rolling and mean5-mean projection. Only rows accepted by the sink at cutoff count. Post-cutoff checkpoint acknowledgement, manifest validation and cancel are outside the timer.",
        "limitations": [
            "Each round warms each arm once, then runs ten adjacent AB/BA pairs with fresh runners per arm.",
            "On arms prove completed nonterminal manifests. Off arms prove zero checkpoint objects after cancellation.",
            "No recovery is performed in M3/M4; recovery_validation_seconds is null rather than fabricated.",
            "Status snapshots retain edge high-water marks and checkpoint state. Freeze-to-ack is measured separately after cutoff.",
        ],
    });
    std::fs::write(
        destination,
        serde_json::to_string_pretty(&evidence).unwrap() + "\n",
    )
    .unwrap();
}
