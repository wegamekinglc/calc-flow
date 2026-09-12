//! Independent single-thread callback measurement, outside the CI timing gate.

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use datafusion::arrow::{
    array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};
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

#[derive(Clone, Copy)]
struct Workload {
    rows: usize,
    entities: usize,
    window: usize,
    dual: bool,
    first_payload_bytes: usize,
}

impl Workload {
    fn identity(self) -> Value {
        json!({"rows": self.rows, "entities": self.entities, "window": self.window,
            "outputs": if self.dual { "mean5,mean20" } else { "mean" },
            "min_periods": "full window", "input": "Float64 exact binary fractions",
            "first_payload_bytes": self.first_payload_bytes})
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
        let end = start + self.rows;
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
        let operator = RollingOperator::new("rolling", workload.schema(), workload.spec()).unwrap();
        let output_ports = operator
            .output_ports()
            .iter()
            .map(|port| (port.name().to_owned(), port.clone()))
            .collect();
        let (sender, receiver) = crate::edge_channel(
            "rolling->sink",
            EdgeBudget {
                max_rows: workload.rows,
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
                entity_work: None,
                node_id: "rolling".into(),
                operator: CompiledStreamOperator::Rolling(operator),
                checkpoint_capability: OperatorCheckpointCapability::CheckpointedStateful {
                    state_version: 1,
                },
                ingresses: BTreeMap::new(),
                outputs: BTreeMap::from([("output".into(), vec![sender])]),
                output_ports,
                context: job.for_node("rolling").unwrap(),
                progress: OperatorProgress::with_optional_rolling_metrics(
                    observed.then(RollingMetricsStore::default),
                ),
                metrics: MetricsRecorder::default(),
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
            i64::try_from((start + self.workload.rows - 1) / self.workload.entities).unwrap(),
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
        let mut rows = 0;
        let mut output = Vec::new();
        while rows < self.workload.rows {
            let message = self.receiver.recv().await.unwrap().unwrap();
            let batch = message.as_data().unwrap();
            rows += batch.num_rows();
            output.extend_from_slice(batch.table_payload().unwrap().batches());
        }
        assert_eq!(rows, self.workload.rows);
        assert_eq!(self.receiver.metrics().charged_rows, 0);
        output
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
            first_payload_bytes: 0,
        },
        Workload {
            rows: 256,
            entities: 4,
            window: 20,
            dual: true,
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
            first_payload_bytes: 0,
        },
        Workload {
            rows: 8192,
            entities: 64,
            window: 20,
            dual: false,
            first_payload_bytes: 0,
        },
        Workload {
            rows: 8192,
            entities: 64,
            window: 20,
            dual: true,
            first_payload_bytes: 0,
        },
        Workload {
            rows: 8192,
            entities: 1,
            window: 4096,
            dual: false,
            first_payload_bytes: 0,
        },
    ]
}

#[test]
#[ignore = "dedicated performance task: release current-thread diagnostic, not a CI timing gate"]
fn native_callback_allocation_and_overhead_evidence() {
    assert!(
        !cfg!(debug_assertions),
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
