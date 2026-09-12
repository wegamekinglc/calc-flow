//! Private task-allocation diagnostic; compile/run only in an authorized quiet window.
//! The caller sets `CALC_FLOW_P7_ALLOCATION_EXECUTABLE` to its recorded absolute
//! test-binary path; that declaration does not independently observe process identity.

use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    compute::concat_batches,
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    ipc::writer::StreamWriter,
    record_batch::RecordBatch,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, watch};

use super::super::{
    JobCore, LaunchId, OperatorRegistration, create_runtime_channels, prepare_operator_task,
    take_boundary_endpoints,
};
use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, CancellationToken, Edge, EdgeBudget,
    EdgeReceiver, EdgeSender, EventTime, ExpressionOperator, JsonMap, OperatorMetadata,
    PipelineBuilder, Port, PortEndpoint, Result, RollingOperator, StreamJobContext, StreamMessage,
    StreamMessageKind, StreamRequirements, UdfRegistry,
    operator::rolling_metrics::RollingMetricsStore,
    pipeline::StreamRuntimePlanParts,
    runtime::streaming::{
        metrics::MetricsRecorder,
        operator_task::{spawn_operator_task, spawn_operator_task_pair},
        projection::StatusProjection,
        supervisor::TaskSupervisor,
    },
};

#[path = "allocation_evidence_guards.rs"]
mod guards;

const ENTITIES: usize = 64;
const WINDOW: usize = 20;
const PRELOAD_CHUNK: usize = 64_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TaskMode {
    Independent,
    Fused,
}

impl TaskMode {
    const fn name(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::Fused => "fused",
        }
    }

    const fn physical_drivers(self) -> usize {
        match self {
            Self::Independent => 2,
            Self::Fused => 1,
        }
    }
}

fn input_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
    ]))
}

fn output_schema() -> SchemaRef {
    let mut fields = input_schema().fields().to_vec();
    fields.push(Arc::new(Field::new("mean", DataType::Float64, true)));
    Arc::new(Schema::new(fields))
}

fn prepared_graph() -> StreamRuntimePlanParts {
    let spec = serde_json::from_value(json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["symbol"], "event_time": "event_time",
        "sequence_by": ["sequence"], "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1",
        "outputs": [{"kind": "mean", "primitive_version": 1,
            "input": "price", "output": "mean", "min_periods": WINDOW,
            "frame": {"kind": "rows", "size": WINDOW}}]
    }))
    .unwrap();
    let rolling = RollingOperator::new("rolling", input_schema(), spec).unwrap();
    let projection = ExpressionOperator::new(
        "project",
        "",
        ["event_time", "sequence", "symbol", "price", "mean"]
            .map(String::from)
            .to_vec(),
        None,
        vec![],
    )
    .unwrap()
    .with_ports(
        Port::with_schema_ref(
            "input",
            BatchKind::Table,
            true,
            rolling.output_ports()[0].schema().cloned(),
        )
        .unwrap(),
        Port::with_schema_ref("output", BatchKind::Table, false, Some(output_schema())).unwrap(),
    )
    .unwrap();
    let graph = PipelineBuilder::new("p7-task-allocation")
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
        .into_runtime_parts(EdgeBudget::new(PRELOAD_CHUNK, 64 << 20).unwrap())
        .unwrap();
    assert_eq!(graph.nodes.len(), 2);
    assert!(super::eligible_pair(&graph.nodes[0], &graph.nodes[1]));
    graph
}

fn price(row: usize) -> f64 {
    100.0 + f64::from(u32::try_from(row % 257).unwrap()) / 8.0
}

fn input_record(start: usize, rows: usize) -> RecordBatch {
    let symbols = (0..ENTITIES)
        .map(|entity| format!("S{entity:04}"))
        .collect::<Vec<_>>();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                (start..start + rows).map(|row| i64::try_from(row / ENTITIES).unwrap()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(UInt64Array::from_iter_values(
            (start..start + rows).map(|row| u64::try_from(row).unwrap()),
        )),
        Arc::new(StringArray::from_iter_values(
            (start..start + rows).map(|row| symbols[row % ENTITIES].as_str()),
        )),
        Arc::new(Float64Array::from_iter_values(
            (start..start + rows).map(price),
        )),
    ];
    RecordBatch::try_new(input_schema(), columns).unwrap()
}

fn reference_record(start: usize, rows: usize) -> RecordBatch {
    let input = input_record(start, rows);
    let mut columns = input.columns().to_vec();
    let means = (start..start + rows)
        .map(|row| {
            (row / ENTITIES + 1 >= WINDOW).then(|| {
                (0..WINDOW)
                    .map(|offset| price(row - offset * ENTITIES))
                    .sum::<f64>()
                    / f64::from(u32::try_from(WINDOW).unwrap())
            })
        })
        .collect::<Vec<_>>();
    columns.push(Arc::new(Float64Array::from(means)));
    RecordBatch::try_new(output_schema(), columns).unwrap()
}

struct WindowRequest {
    start: usize,
    rows: usize,
    input: Batch,
    watermark: EventTime,
}

impl WindowRequest {
    fn new(start: usize, rows: usize) -> Self {
        assert!(rows > 0 && start % ENTITIES == 0 && rows % ENTITIES == 0);
        Self {
            start,
            rows,
            input: Batch::table(
                vec![input_record(start, rows)],
                BatchMetadata::new("source", u64::try_from(start).unwrap(), BTreeMap::new())
                    .unwrap(),
            )
            .unwrap(),
            watermark: EventTime::from_micros(
                i64::try_from((start + rows - 1) / ENTITIES).unwrap(),
            ),
        }
    }
}

struct WindowOutput {
    batches: Vec<Batch>,
    watermark: EventTime,
}

#[derive(Debug, PartialEq)]
struct CheckedWindow {
    record: RecordBatch,
    metadata: Vec<BatchMetadata>,
    watermark: EventTime,
}

fn evidence_error(message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("P7 allocation evidence: {message}"),
    }
}

fn validate_window(request: &WindowRequest, output: &WindowOutput) -> Result<CheckedWindow> {
    if output.watermark != request.watermark {
        return Err(evidence_error("completed output has a different watermark"));
    }
    let records = output
        .batches
        .iter()
        .map(Batch::table_payload)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flat_map(|table| table.batches().iter().cloned())
        .collect::<Vec<_>>();
    let schema = output_schema();
    if records.iter().any(|record| record.schema() != schema) {
        return Err(evidence_error("materialized output schema changed"));
    }
    let actual =
        concat_batches(&schema, &records).map_err(|error| evidence_error(&error.to_string()))?;
    let expected = reference_record(request.start, request.rows);
    if actual.num_rows() != request.rows || actual.columns()[..4] != expected.columns()[..4] {
        return Err(evidence_error(
            "materialized input columns or row order changed",
        ));
    }
    validate_means(&actual, &expected, request.rows)?;
    Ok(CheckedWindow {
        record: actual,
        metadata: output
            .batches
            .iter()
            .map(|batch| batch.metadata().clone())
            .collect(),
        watermark: output.watermark,
    })
}

fn validate_means(actual: &RecordBatch, expected: &RecordBatch, rows: usize) -> Result<()> {
    let means = actual
        .column(4)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let oracle = expected
        .column(4)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for row in 0..rows {
        if means.is_null(row) != oracle.is_null(row)
            || (!means.is_null(row)
                && (!means.value(row).is_finite()
                    || (means.value(row) - oracle.value(row)).abs() > 1e-12))
        {
            return Err(evidence_error(
                "materialized mean differs from the independent SMA oracle",
            ));
        }
    }
    Ok(())
}

struct TaskFixture {
    supervisor: TaskSupervisor,
    source: EdgeSender,
    sink: EdgeReceiver,
    core: Arc<JobCore>,
}

impl TaskFixture {
    async fn start(graph: StreamRuntimePlanParts, mode: TaskMode) -> Self {
        let cancellation = CancellationToken::new();
        let context = StreamJobContext::new(
            7,
            &graph.fingerprint,
            JsonMap::new(),
            None,
            cancellation.clone(),
        );
        let metrics = MetricsRecorder::new(
            graph
                .edges
                .iter()
                .map(|(id, edge)| (id.clone(), edge.budget)),
            [],
            graph.nodes.iter().map(|node| node.node_id.clone()),
            [],
        );
        let (commands, _commands_rx) = mpsc::unbounded_channel();
        let core = Arc::new(JobCore::new(
            LaunchId::new(0),
            7,
            commands,
            metrics,
            StatusProjection::default(),
            false,
            graph.name.clone(),
        ));
        core.runtime_status
            .lock()
            .rolling_metrics
            .insert("rolling".into(), RollingMetricsStore::default());
        let mut supervisor =
            TaskSupervisor::new_with_terminal_arbiter(cancellation, core.terminal_arbiter.clone());
        let (entry, _) = watch::channel(false);
        let (data, _) = watch::channel(false);
        let (acks, mut ack_rx) = mpsc::unbounded_channel();
        let (mut senders, mut receivers) = create_runtime_channels(&graph, &core.metrics).unwrap();
        let mut restores = BTreeMap::new();
        let mut registration = OperatorRegistration {
            context: &context,
            core: &core,
            entry_tx: &entry,
            data_tx: &data,
            ack_tx: &acks,
            senders: &mut senders,
            receivers: &mut receivers,
            supervisor: &mut supervisor,
            metrics: &core.metrics,
            runtime_status: &core.runtime_status,
            restores: &mut restores,
            checkpoint: None,
        };
        let mut nodes = graph.nodes.into_iter();
        let Ok(first) = prepare_operator_task(nodes.next().unwrap(), &mut registration) else {
            panic!("rolling task preparation failed");
        };
        let Ok(second) = prepare_operator_task(nodes.next().unwrap(), &mut registration) else {
            panic!("projection task preparation failed");
        };
        match mode {
            TaskMode::Independent => {
                spawn_operator_task(&mut supervisor, first);
                spawn_operator_task(&mut supervisor, second);
            }
            TaskMode::Fused => {
                spawn_operator_task_pair(&mut supervisor, first, second);
            }
        }
        entry.send(true).unwrap();
        for _ in 0..2 {
            ack_rx.recv().await.unwrap().result.unwrap();
        }
        let Ok((mut source_outputs, mut sink_inputs)) = take_boundary_endpoints(
            graph.source_routes,
            graph.sink_routes,
            &mut senders,
            &mut receivers,
        ) else {
            panic!("operator boundary endpoints changed");
        };
        data.send(true).unwrap();
        Self {
            supervisor,
            core,
            source: source_outputs.pop_first().unwrap().1.pop().unwrap(),
            sink: sink_inputs.pop_first().unwrap().1,
        }
    }

    fn verify_ready(&self, mode: TaskMode) {
        assert_eq!(
            self.supervisor.physical_driver_count(),
            mode.physical_drivers()
        );
        assert_eq!(self.supervisor.task_count(), 2);
        let names = self
            .supervisor
            .registry()
            .snapshot()
            .into_values()
            .map(|task| task.task_name)
            .collect::<Vec<_>>();
        assert_eq!(names, ["operator:rolling", "operator:project"]);
        assert!(
            self.core
                .runtime_status
                .lock()
                .nodes
                .values()
                .all(|node| node.snapshot().input_batches == 0)
        );
    }

    async fn dispatch_and_drain(&mut self, request: &WindowRequest) -> WindowOutput {
        self.source
            .send(StreamMessage::data(request.input.clone()))
            .await
            .unwrap();
        self.source
            .send(StreamMessage::watermark(request.watermark))
            .await
            .unwrap();
        let mut batches = Vec::new();
        loop {
            let message = self
                .sink
                .recv()
                .await
                .unwrap()
                .expect("output closed before finality");
            match message.kind() {
                StreamMessageKind::Data => batches.push(message.as_data().unwrap().clone()),
                StreamMessageKind::Watermark => {
                    return WindowOutput {
                        batches,
                        watermark: message.as_watermark().unwrap(),
                    };
                }
                _ => panic!("unexpected control in a Data+watermark window"),
            }
        }
    }

    async fn preload(&mut self, history_rows: usize) {
        for start in (0..history_rows).step_by(PRELOAD_CHUNK) {
            let request = WindowRequest::new(start, PRELOAD_CHUNK.min(history_rows - start));
            let output = self.dispatch_and_drain(&request).await;
            validate_window(&request, &output).unwrap();
        }
        self.verify_completed_rows(history_rows);
    }

    fn verify_completed_rows(&self, rows: usize) {
        assert_eq!(self.sink.metrics().queue_depth, 0);
        assert_eq!(self.source.metrics().queue_depth, 0);
        let observed = self.core.runtime_status.lock().rolling_metrics["rolling"].snapshot();
        assert!(!observed.overflowed);
        assert_eq!(observed.data.started, observed.data.succeeded);
        assert_eq!(observed.watermark.started, observed.watermark.succeeded);
        assert_eq!(observed.data.input_rows, u64::try_from(rows).unwrap());
        assert_eq!(
            observed.watermark.numeric_rows,
            u64::try_from(rows).unwrap()
        );
        assert!(
            self.core
                .metrics
                .snapshot()
                .edges
                .values()
                .all(|edge| edge.channel.queue_depth == 0
                    && edge.channel.charged_rows == 0
                    && edge.channel.charged_bytes == 0
                    && !edge.drop_invariant_violated)
        );
    }

    async fn finish(&mut self) {
        self.source
            .send(StreamMessage::end_of_input())
            .await
            .unwrap();
        assert!(self.sink.recv().await.unwrap().unwrap().is_end_of_input());
        assert!(self.supervisor.join_all().await.errors.is_empty());
        assert_eq!(self.supervisor.task_count(), 0);
        assert!(
            self.core
                .runtime_status
                .lock()
                .nodes
                .values()
                .all(|node| node.snapshot().ended && node.snapshot().on_end_calls == 1)
        );
    }
}

fn current_thread_runtime() -> tokio::runtime::Runtime {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        tokio::spawn(async {}).await.unwrap();
    });
    runtime
}

fn allocation_json(value: allocation_counter::AllocationInfo) -> Value {
    json!({"count_total": value.count_total, "bytes_total": value.bytes_total,
        "count_max": value.count_max, "bytes_max": value.bytes_max,
        "count_current": value.count_current, "bytes_current": value.bytes_current})
}

fn window_digest(checked: &CheckedWindow) -> String {
    let mut bytes = Vec::new();
    let mut writer = StreamWriter::try_new(&mut bytes, &checked.record.schema()).unwrap();
    writer.write(&checked.record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    let mut hash = Sha256::new();
    hash.update(bytes);
    hash.update(serde_json::to_vec(&checked.metadata).unwrap());
    hash.update(checked.watermark.as_micros().to_le_bytes());
    format!("{:x}", hash.finalize())
}

fn measure_arm(
    mode: TaskMode,
    history_rows: usize,
    rows: usize,
) -> (Value, CheckedWindow, CheckedWindow) {
    let runtime = current_thread_runtime();
    let graph = prepared_graph();
    let fingerprint = graph.fingerprint.clone();
    let mut retained = None;
    let startup = allocation_counter::measure(|| {
        retained = Some(runtime.block_on(TaskFixture::start(graph, mode)));
    });
    let mut fixture = retained.unwrap();
    fixture.verify_ready(mode);
    runtime.block_on(fixture.preload(history_rows));
    let request = WindowRequest::new(history_rows, rows);
    let mut output = None;
    let warm = allocation_counter::measure(|| {
        output = Some(runtime.block_on(fixture.dispatch_and_drain(&request)));
    });
    let checked = validate_window(&request, &output.unwrap()).unwrap();
    fixture.verify_completed_rows(history_rows + rows);
    let continuation = WindowRequest::new(history_rows + rows, ENTITIES * WINDOW);
    let continued = runtime.block_on(fixture.dispatch_and_drain(&continuation));
    let checked_state = validate_window(&continuation, &continued).unwrap();
    fixture.verify_completed_rows(continuation.start + continuation.rows);
    runtime.block_on(fixture.finish());
    let evidence = json!({"mode": mode.name(), "graph_fingerprint": fingerprint,
        "physical_operator_drivers": mode.physical_drivers(), "logical_operator_tasks": 2,
        "history_rows_before_counter": history_rows, "history_rows_after_counter": history_rows + rows,
        "entities": ENTITIES, "window": WINDOW, "min_periods": WINDOW, "append_rows": rows,
        "startup_to_ready": allocation_json(startup), "warm_data_watermark_to_output": allocation_json(warm),
        "output_sha256": window_digest(&checked), "state_behavior_probe_sha256": window_digest(&checked_state),
        "final_watermark_micros": request.watermark.as_micros(),
        "state_behavior_probe_rows": ENTITIES * WINDOW, "remaining_logical_tasks_after_cleanup": 0});
    (evidence, checked, checked_state)
}

#[test]
#[ignore = "dedicated task: current-thread release allocation diagnostic, no timing claims"]
fn p7_real_task_allocation_evidence() {
    assert!(
        !cfg!(debug_assertions),
        "requires an independently identified release test binary"
    );
    let destination =
        std::env::var("CALC_FLOW_P7_ALLOCATION_OUTPUT").expect("set output artifact path");
    let mut samples = Vec::new();
    for history in [64_000, 1_024_000] {
        for rows in [64, 64_000] {
            for pair in 0..3 {
                let modes = if pair % 2 == 0 {
                    [TaskMode::Independent, TaskMode::Fused]
                } else {
                    [TaskMode::Fused, TaskMode::Independent]
                };
                let first = measure_arm(modes[0], history, rows);
                let second = measure_arm(modes[1], history, rows);
                assert_eq!(first.0["graph_fingerprint"], second.0["graph_fingerprint"]);
                assert_eq!(first.1, second.1);
                assert_eq!(first.2, second.2);
                samples.push(
                    json!({"pair": pair, "order": [modes[0].name(), modes[1].name()],
                    "arms": [first.0, second.0]}),
                );
            }
        }
    }
    let executable = std::path::PathBuf::from(
        std::env::var_os("CALC_FLOW_P7_ALLOCATION_EXECUTABLE")
            .expect("set the absolute test executable path recorded by the invoking controller"),
    );
    assert!(
        executable.is_absolute(),
        "the declared executable must be an absolute path"
    );
    assert!(
        executable.is_file(),
        "the declared executable must be a regular file"
    );
    let evidence = json!({
        "contract": "calc-flow.p7-real-task-allocation/1", "samples": samples,
        "executable": executable,
        "executable_sha256": format!("{:x}", Sha256::digest(std::fs::read(&executable).unwrap())),
        "executable_identity_source": "caller-declared absolute path; the invoking controller must independently bind the launched process to these bytes",
        "helper_sha256": format!("{:x}", Sha256::digest(include_bytes!("allocation_evidence.rs"))),
        "guards_sha256": format!("{:x}", Sha256::digest(include_bytes!("allocation_evidence_guards.rs"))),
        "runtime": "current-thread Tokio, scheduler warmed before each arm; no spawn_blocking",
        "startup_boundary": "prepared compiled graph -> actual channels, metrics, two logical operator registrations, both reset/entry acks and released data gate",
        "warm_boundary": "prebuilt immutable input -> enqueue Data and watermark, real operator callbacks/tasks/channels, collect emitted Arrow Batches through final watermark",
        "counter": "allocation-counter 0.8.1; one non-nested measure per phase; six original fields retained",
        "validation": "outside counters: schemas, values, row order, Batch metadata, watermark, empty queues, complete W20 continuation per entity, both EOF callbacks and zero tasks",
        "limitations": [
            "Private native mechanism diagnostic; not the 32-thread Python wheel heap or timing evidence.",
            "No wall-clock performance conclusion follows from these allocation counts.",
            "Graph compilation, runtime construction, preload, input creation, oracle, validation, Arrow concatenation/digests, continuation and teardown are outside both counters.",
            "Startup and warm phases are separate windows, never nested; peaks are not summed across windows.",
            "Warm output handles and operator state remain alive at the counter endpoint; current fields are signed net changes and include old-object releases.",
            "bytes_max is the maximum incremental current-thread net count in that window, not absolute live heap or peak RSS.",
            "State validation is continuation behavior, not a serialized snapshot or an operator-retained-memory measurement.",
            "Source/sink adapters, coordinator/runner driver and managed checkpoint I/O are outside this task-pair diagnostic."]
    });
    std::fs::write(destination, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
}
