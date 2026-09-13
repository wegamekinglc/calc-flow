//! Dual transactional file outputs through the public source-driven runtime.
#![cfg(feature = "file")]

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, Cursor, DecodeBounds, DeliveryGuarantee, Epoch, EventTime, FormatDecoder,
    JobState, JsonMap, ManagedCheckpointRuntime, NativeWatermarkCapability, OperatorMetadata,
    PipelineBuilder, ReplayPositioning, Result, RollingOperator, SinkBinding, SinkRecovery,
    SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema,
    StreamRequirements, StreamRuntimeConfig, StreamSource, StreamingJob, StreamingRunner,
    TransactionalStreamSink, UdfRegistry,
};
use calc_flow_connectors::{FileSinkConfig, TransactionalParquetSink, parquet::ParquetCodec};
use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    buffer::Buffer,
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::Notify;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("key", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("x", DataType::Float64, true),
    ]))
}

fn data(times: &[i64], sequence: u64) -> SourceEvent {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC"))
                as ArrayRef,
            Arc::new(StringArray::from(vec!["a"; times.len()])),
            Arc::new(UInt64Array::from(
                times
                    .iter()
                    .enumerate()
                    .map(|(row, _)| sequence * 10 + u64::try_from(row).unwrap())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                times
                    .iter()
                    .map(|time| Some(f64::from(i32::try_from(*time).unwrap())))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    SourceEvent::Data {
        batch: Batch::table(
            vec![record],
            BatchMetadata::new("source", sequence, JsonMap::new()).unwrap(),
        )
        .unwrap(),
        cursor: Cursor::unbound(vec![1], JsonMap::new()).unwrap(),
    }
}

#[derive(Default, PartialEq, Eq)]
enum LateSinkMode {
    #[default]
    Transactional,
    Ordinary,
    RequireExactlyOnce,
}

#[derive(Default)]
struct Probe {
    fanout: bool,
    cross_section: bool,
    late_sink_mode: LateSinkMode,
    ordinary_rows: Mutex<Vec<(i64, u64)>>,
    ordinary_received: Notify,
    fault: Mutex<Option<(&'static str, &'static str, usize)>>,
    paused: Notify,
    opens: Mutex<Vec<usize>>,
    source_closes: AtomicUsize,
    sink_events: Mutex<Vec<(&'static str, &'static str, u64)>>,
    sink_drops: AtomicUsize,
    blocked: Notify,
    release: Notify,
    block_sink: Mutex<Option<&'static str>>,
}

struct ScriptedSource {
    events: Vec<SourceEvent>,
    offset: usize,
    pause_at: Option<usize>,
    probe: Arc<Probe>,
}

#[async_trait]
impl StreamSource for ScriptedSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: self
                .events
                .iter()
                .filter_map(|event| match event {
                    SourceEvent::Data { batch, .. } => Some(batch.num_rows()),
                    _ => None,
                })
                .max()
                .unwrap_or(1)
                .max(1),
            max_batch_bytes: self
                .events
                .iter()
                .filter_map(|event| match event {
                    SourceEvent::Data { batch, .. } => Some(batch.estimated_bytes().unwrap()),
                    _ => None,
                })
                .max()
                .unwrap_or(1)
                .max(1),
            schema: SourceSchema::Exact(schema()),
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.offset = cursor.map_or(0, |cursor| {
            usize::try_from(cursor.payload()["offset"].as_u64().unwrap()).unwrap()
        });
        self.probe.opens.lock().unwrap().push(self.offset);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.pause_at == Some(self.offset) {
            self.probe.paused.notify_one();
            return std::future::pending().await;
        }
        let Some(event) = self.events.get(self.offset).cloned() else {
            return Ok(None);
        };
        self.offset += 1;
        Ok(Some(match event {
            SourceEvent::Data { batch, .. } => SourceEvent::Data {
                batch,
                cursor: Cursor::unbound(
                    u64::try_from(self.offset).unwrap().to_be_bytes().to_vec(),
                    BTreeMap::from([("offset".into(), json!(self.offset))]),
                )?,
            },
            control => control,
        }))
    }
    async fn close(&mut self) -> Result<()> {
        self.probe.source_closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct ObservedSink {
    writes: usize,
    sink: TransactionalParquetSink,
    name: &'static str,
    probe: Arc<Probe>,
}

impl ObservedSink {
    fn check_fault(&self, phase: &'static str, occurrence: usize) -> Result<()> {
        if *self.probe.fault.lock().unwrap() == Some((self.name, phase, occurrence)) {
            return Err(calc_flow::CalcFlowError::CheckpointMismatch {
                message: format!("injected {} {phase} failure", self.name),
            });
        }
        Ok(())
    }

    fn record(&self, event: &'static str, value: u64) {
        self.probe
            .sink_events
            .lock()
            .unwrap()
            .push((self.name, event, value));
    }
}

impl Drop for ObservedSink {
    fn drop(&mut self) {
        self.probe.sink_drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait]
impl TransactionalStreamSink for ObservedSink {
    async fn open(&mut self) -> Result<()> {
        self.record("open", 0);
        self.sink.open().await
    }
    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()> {
        self.record("begin", epoch.as_u64());
        self.sink.begin_epoch(epoch).await
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.record("write", batch.metadata().sequence());
        if self.writes == 0 && *self.probe.block_sink.lock().unwrap() == Some(self.name) {
            self.probe.blocked.notify_one();
            self.probe.release.notified().await;
        }
        self.check_fault("write", self.writes)?;
        self.writes += 1;
        self.sink.write(batch).await
    }
    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap> {
        self.record("prepare", epoch.as_u64());
        self.check_fault("prepare", 0)?;
        self.sink.pre_commit(epoch).await
    }
    async fn commit(&mut self, epoch: Epoch, evidence: &JsonMap) -> Result<()> {
        self.record("commit", epoch.as_u64());
        self.check_fault("commit", 0)?;
        self.sink.commit(epoch, evidence).await
    }
    async fn abort(&mut self, epoch: Epoch, evidence: Option<&JsonMap>) -> Result<()> {
        self.record("abort", epoch.as_u64());
        self.sink.abort(epoch, evidence).await
    }
    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        self.record("recover", recovery.epoch().as_u64());
        self.sink.recover(recovery).await
    }
    async fn close(&mut self) -> Result<()> {
        self.record("close", 0);
        self.sink.close().await
    }
}

fn runner(
    root: &Path,
    times: &[i64],
    pause_at: Option<usize>,
    probe: Arc<Probe>,
) -> StreamingRunner {
    let events = vec![
        SourceEvent::Watermark(EventTime::from_micros(10)),
        data(times, 0),
        SourceEvent::Watermark(EventTime::from_micros(30)),
        data(&[6, 40], 1),
        SourceEvent::Watermark(EventTime::from_micros(50)),
    ];
    runner_events(
        root,
        events,
        pause_at,
        probe,
        StreamRuntimeConfig {
            checkpoint_interval: Duration::from_secs(3600),
            ..StreamRuntimeConfig::default()
        },
    )
}

fn late_node(cross_section: bool) -> (calc_flow::NodeOperator, SchemaRef) {
    if cross_section {
        let operator = calc_flow::CrossSectionOperator::new("roll", schema(), serde_json::from_value(json!({
            "configuration_version":1, "state_layout_version":1,
            "entity_by":["key"], "partition_by":[], "event_time":"ts", "sequence_by":["seq"],
            "grouping":{"kind":"exact_time"},
            "outputs":[{"kind":"rank", "primitive_version":1, "input":"x", "output":"rank", "direction":"ascending", "tie_method":"average", "null_placement":"exclude", "min_samples":1}],
            "allowed_lateness_micros":0, "late_policy":{"kind":"side_output", "metrics_version":1, "schema_version":1}, "value_policy":"nan_exclude_preserve_v1"
        })).unwrap()).unwrap();
        let schema = operator.output_ports()[1].schema().unwrap().clone();
        return (operator.into(), schema);
    }
    let operator = RollingOperator::new("roll", schema(), serde_json::from_value(json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["key"], "event_time": "ts", "sequence_by": ["seq"],
        "outputs": [{"kind":"lag", "primitive_version":1, "input":"x", "output":"lag", "periods":1}],
        "allowed_lateness_micros":0,
        "late_policy":{"kind":"side_output", "metrics_version":1, "schema_version":1},
        "value_policy":"stateful_numeric_v1"
    })).unwrap()).unwrap();
    let late_schema = operator.output_ports()[1].schema().unwrap().clone();
    (operator.into(), late_schema)
}

fn late_port(name: &str, schema: &SchemaRef) -> calc_flow::Port {
    calc_flow::Port::new(
        name,
        calc_flow::BatchKind::Table,
        true,
        Some(
            schema
                .fields()
                .iter()
                .map(|field| field.as_ref().clone())
                .collect(),
        ),
    )
    .unwrap()
}

fn file_plan(
    probe: &Probe,
) -> (
    calc_flow::StreamExecutionPlan,
    Vec<(&'static str, &'static str)>,
) {
    let (operator, late_schema) = late_node(probe.cross_section);
    let mut builder = PipelineBuilder::new("late-files")
        .unwrap()
        .add_node("roll", operator)
        .unwrap();
    let bindings = if probe.fanout {
        for name in ["left", "right"] {
            let project = calc_flow::ExpressionOperator::new(
                name,
                "",
                late_schema
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect(),
                None,
                vec![],
            )
            .unwrap()
            .with_ports(
                late_port("input", &late_schema),
                late_port("output", &late_schema),
            )
            .unwrap();
            builder = builder
                .add_node(name, Box::new(project))
                .unwrap()
                .connect(calc_flow::Edge::new(
                    calc_flow::PortEndpoint::new("roll", "late").unwrap(),
                    calc_flow::PortEndpoint::new(name, "input").unwrap(),
                ))
                .unwrap();
        }
        vec![
            ("roll.output", "normal"),
            ("left.output", "left"),
            ("right.output", "right"),
        ]
    } else {
        vec![("output", "normal"), ("late", "late")]
    };
    let plan = builder
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: bindings
                    .iter()
                    .map(|(binding, _)| {
                        (
                            (*binding).into(),
                            if *binding == "late" && probe.late_sink_mode == LateSinkMode::Ordinary
                            {
                                DeliveryGuarantee::AtLeastOnce
                            } else {
                                DeliveryGuarantee::ExactlyOnce
                            },
                        )
                    })
                    .collect(),
            },
        )
        .unwrap();
    (plan, bindings)
}

fn runner_events(
    root: &Path,
    events: Vec<SourceEvent>,
    pause_at: Option<usize>,
    probe: Arc<Probe>,
    config: StreamRuntimeConfig,
) -> StreamingRunner {
    let (plan, bindings) = file_plan(&probe);
    let sinks = bindings
        .into_iter()
        .map(|(binding, name)| {
            if name == "late" && probe.late_sink_mode != LateSinkMode::Transactional {
                return (
                    binding.into(),
                    vec![SinkBinding::ordinary(name, OrdinaryLate(probe.clone())).unwrap()],
                );
            }
            let sink = ObservedSink {
                writes: 0,
                sink: TransactionalParquetSink::new(FileSinkConfig {
                    root: root.join("outputs"),
                    output: name.into(),
                })
                .unwrap(),
                name,
                probe: probe.clone(),
            };
            (
                binding.into(),
                vec![
                    SinkBinding::transactional(
                        &if probe.fanout {
                            format!("sink_{name}")
                        } else {
                            name.into()
                        },
                        sink,
                    )
                    .unwrap(),
                ],
            )
        })
        .collect();
    let source = ScriptedSource {
        events,
        offset: 0,
        pause_at,
        probe,
    };
    StreamingRunner::new(
        plan,
        BTreeMap::from([("input".into(), SourceBinding::new(source))]),
        sinks,
        ManagedCheckpointRuntime::new(root.join("state")).unwrap(),
    )
    .unwrap()
    .with_runtime_config(config)
    .unwrap()
}

fn rows(root: &Path, name: &str) -> Vec<(i64, u64)> {
    let mut rows = Vec::new();
    let Ok(epochs) = std::fs::read_dir(root.join("outputs").join(name)) else {
        return rows;
    };
    for epoch in epochs
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("epoch="))
    {
        for file in std::fs::read_dir(epoch.path())
            .unwrap()
            .map(|entry| entry.unwrap())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "parquet")
            })
        {
            let batch = ParquetCodec::new("1")
                .unwrap()
                .decode(
                    &std::fs::read(file.path()).unwrap(),
                    &DecodeBounds::new(100, 1 << 20).unwrap(),
                    &[],
                )
                .unwrap();
            for record in batch.table_payload().unwrap().batches() {
                let times = record
                    .column_by_name("ts")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap();
                let sequences = record
                    .column_by_name("seq")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                rows.extend(
                    (0..record.num_rows()).map(|row| (times.value(row), sequences.value(row))),
                );
            }
        }
    }
    rows.sort_unstable();
    rows
}

fn assert_settled(job: &StreamingJob) {
    let status = job.status();
    assert_eq!(status.task_count, 0);
    assert!(status.edges.values().all(|edge| edge.current_envelopes == 0
        && edge.current_rows == 0
        && edge.current_bytes == 0));
}

#[tokio::test]
async fn test_late_files_empty_mixed_and_all_late_epochs_recover_and_restart_terminal() {
    tokio::time::timeout(Duration::from_secs(30), async {
        for cross_section in [false, true] {
            for times in [&[20][..], &[5, 20][..], &[5, 7][..]] {
                let reference = tempfile::tempdir().unwrap();
                let reference_probe = Arc::new(Probe {
                    cross_section,
                    ..Probe::default()
                });
                let job = runner(reference.path(), times, None, reference_probe)
                    .start()
                    .await
                    .unwrap();
                assert_eq!(job.wait().await.state, JobState::Completed);
                let root = tempfile::tempdir().unwrap();
                let probe = Arc::new(Probe {
                    cross_section,
                    ..Probe::default()
                });
                let first = runner(root.path(), times, Some(2), probe.clone())
                    .start()
                    .await
                    .unwrap();
                probe.paused.notified().await;
                let epoch = first.trigger_checkpoint().await.unwrap();
                for name in ["normal", "late"] {
                    let manifest: serde_json::Value = serde_json::from_slice(
                        &std::fs::read(
                            root.path()
                                .join("outputs")
                                .join(name)
                                .join(format!("epoch={}", epoch.as_u64()))
                                .join("manifest.json"),
                        )
                        .unwrap(),
                    )
                    .unwrap();
                    assert_eq!(manifest["epoch"], epoch.as_u64());
                    assert_eq!(manifest["output"], name);
                    for event in ["prepare", "commit"] {
                        assert!(probe.sink_events.lock().unwrap().contains(&(
                            name,
                            event,
                            epoch.as_u64()
                        )));
                    }
                }
                assert_eq!(
                    rows(root.path(), "late").len(),
                    times.iter().filter(|time| **time <= 10).count()
                );
                let cancelled = first.cancel().await;
                assert_eq!(cancelled.state, JobState::Cancelled);
                assert_eq!(cancelled.completed_epoch, Some(epoch));
                assert_settled(&first);
                let second = runner(root.path(), times, None, probe.clone())
                    .start()
                    .await
                    .unwrap();
                let terminal = second.wait().await;
                assert_eq!(terminal.state, JobState::Completed, "{terminal:?}");
                assert_settled(&second);
                for name in ["normal", "late"] {
                    assert_eq!(rows(root.path(), name), rows(reference.path(), name));
                    assert!(probe.sink_events.lock().unwrap().contains(&(
                        name,
                        "recover",
                        epoch.as_u64()
                    )));
                }
                let third = runner(root.path(), times, None, probe.clone())
                    .start()
                    .await
                    .unwrap();
                assert_eq!(third.wait().await.state, JobState::Completed);
                assert_settled(&third);
                for name in ["normal", "late"] {
                    assert_eq!(rows(root.path(), name), rows(reference.path(), name));
                }
                assert_eq!(*probe.opens.lock().unwrap(), [0, 2]);
                assert_eq!(probe.source_closes.load(Ordering::SeqCst), 2);
                assert_eq!(probe.sink_drops.load(Ordering::SeqCst), 6);
            }
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_late_files_each_sink_write_prepare_and_commit_failure_settles_and_recovers() {
    tokio::time::timeout(Duration::from_secs(30), async {
        for name in ["normal", "late"] {
            for (phase, occurrence) in [("write", 0), ("write", 1), ("prepare", 0), ("commit", 0)] {
                let root = tempfile::tempdir().unwrap();
                let probe = Arc::new(Probe::default());
                *probe.fault.lock().unwrap() = Some((name, phase, occurrence));
                let first = runner(root.path(), &[5, 20], None, probe.clone())
                    .start()
                    .await
                    .unwrap();
                let failed = first.wait().await;
                assert_eq!(
                    failed.state,
                    if phase == "commit" {
                        JobState::RecoveryRequired
                    } else {
                        JobState::Failed
                    },
                    "{name}/{phase}/{occurrence}: {failed:?}"
                );
                assert_eq!(failed.errors[0].component_id(), Some(name), "{failed:?}");
                assert_settled(&first);
                assert_eq!(probe.sink_drops.load(Ordering::SeqCst), 2);
                if phase != "commit" {
                    assert!(failed.completed_epoch.is_none());
                    assert!(rows(root.path(), "late").is_empty());
                    assert!(rows(root.path(), "normal").is_empty());
                }
                *probe.fault.lock().unwrap() = None;
                let recovered = runner(root.path(), &[5, 20], None, probe.clone())
                    .start()
                    .await
                    .unwrap();
                let completed = recovered.wait().await;
                assert_eq!(
                    completed.state,
                    JobState::Completed,
                    "{name}/{phase}: {completed:?}"
                );
                assert_settled(&recovered);
                assert_eq!(
                    rows(root.path(), "late"),
                    [(5, 0), (6, 10)],
                    "{name}/{phase}/{occurrence}"
                );
                assert_eq!(
                    rows(root.path(), "normal"),
                    [(20, 1), (40, 11)],
                    "{name}/{phase}/{occurrence}; events={:?}; outcome={completed:?}",
                    probe.sink_events.lock().unwrap()
                );
                assert_eq!(probe.sink_drops.load(Ordering::SeqCst), 4);
                if phase == "commit" {
                    for output in ["normal", "late"] {
                        assert!(
                            probe
                                .sink_events
                                .lock()
                                .unwrap()
                                .iter()
                                .any(|(name, event, _)| *name == output && *event == "recover")
                        );
                    }
                }
            }
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn test_late_files_later_oversize_chunk_rejects_before_both_sinks_write() {
    let root = tempfile::tempdir().unwrap();
    let probe = Arc::new(Probe::default());
    let SourceEvent::Data { batch, cursor } = data(&[5, 6, 20], 0) else {
        unreachable!()
    };
    let record = &batch.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    let wide = "x".repeat(160);
    columns[1] = Arc::new(StringArray::from(vec!["a", &wide, "a"]));
    let batch = Batch::table(
        vec![RecordBatch::try_new(schema(), columns).unwrap()],
        batch.metadata().clone(),
    )
    .unwrap();
    let budget = calc_flow::EdgeBudget::new(3, batch.estimated_bytes().unwrap()).unwrap();
    let job = runner_events(
        root.path(),
        vec![
            SourceEvent::Watermark(EventTime::from_micros(10)),
            SourceEvent::Data { batch, cursor },
        ],
        None,
        probe.clone(),
        StreamRuntimeConfig {
            edge_budget: budget,
            ..StreamRuntimeConfig::default()
        },
    )
    .start()
    .await
    .unwrap();
    let failed = tokio::time::timeout(Duration::from_secs(5), job.wait())
        .await
        .unwrap();
    assert_eq!(failed.state, JobState::Failed, "{failed:?}");
    assert_eq!(job.status().operators["roll"].late_rows, 0);
    assert_eq!(job.status().operators["roll"].fanned_out_batches, 0);
    assert!(
        !probe
            .sink_events
            .lock()
            .unwrap()
            .iter()
            .any(|(_, event, _)| *event == "write")
    );
    assert!(rows(root.path(), "normal").is_empty());
    assert!(rows(root.path(), "late").is_empty());
    assert_settled(&job);
}

#[tokio::test]
async fn test_late_files_slow_sink_blocks_checkpoint_and_cancellation_settles() {
    let root = tempfile::tempdir().unwrap();
    let probe = Arc::new(Probe::default());
    *probe.block_sink.lock().unwrap() = Some("late");
    let job = runner_events(
        root.path(),
        vec![
            SourceEvent::Watermark(EventTime::from_micros(10)),
            data(&[5, 6, 7], 0),
            data(&[20], 1),
            SourceEvent::Watermark(EventTime::from_micros(30)),
        ],
        Some(4),
        probe.clone(),
        StreamRuntimeConfig {
            edge_budget: calc_flow::EdgeBudget::new(3, 200).unwrap(),
            ..StreamRuntimeConfig::default()
        },
    )
    .start()
    .await
    .unwrap();
    tokio::time::timeout(Duration::from_secs(5), probe.blocked.notified())
        .await
        .unwrap();
    let mut checkpoint = Box::pin(job.trigger_checkpoint());
    std::future::poll_fn(|context| {
        assert!(checkpoint.as_mut().poll(context).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    assert!(job.status().checkpoint.last_completed_epoch.is_none());
    assert!(
        !probe
            .sink_events
            .lock()
            .unwrap()
            .iter()
            .any(|(name, event, _)| *name == "normal" && *event == "write")
    );
    assert_eq!(job.cancel().await.state, JobState::Cancelled);
    assert!(checkpoint.await.is_err());
    assert_settled(&job);
    assert_eq!(probe.sink_drops.load(Ordering::SeqCst), 2);
    assert!(rows(root.path(), "late").is_empty());
}

#[tokio::test]
async fn test_late_files_blocked_barrier_times_out_without_confirming_partial_epoch() {
    let root = tempfile::tempdir().unwrap();
    let probe = Arc::new(Probe::default());
    *probe.block_sink.lock().unwrap() = Some("late");
    let job = runner_events(
        root.path(),
        vec![
            SourceEvent::Watermark(EventTime::from_micros(10)),
            data(&[5, 6, 7], 0),
        ],
        Some(2),
        probe.clone(),
        StreamRuntimeConfig {
            checkpoint_timeout: Duration::from_millis(20),
            edge_budget: calc_flow::EdgeBudget::new(3, 200).unwrap(),
            ..StreamRuntimeConfig::default()
        },
    )
    .start()
    .await
    .unwrap();
    tokio::time::timeout(Duration::from_secs(5), probe.blocked.notified())
        .await
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_secs(5), job.trigger_checkpoint())
            .await
            .unwrap()
            .is_err()
    );
    let failed = job.wait().await;
    assert_eq!(failed.state, JobState::Failed, "{failed:?}");
    assert_eq!(
        failed.errors[0].category(),
        calc_flow::StreamingErrorCategory::CheckpointTimeout
    );
    assert!(failed.completed_epoch.is_none());
    assert!(job.status().checkpoint.sink_commit_acks < 2);
    assert_settled(&job);
    assert_eq!(probe.sink_drops.load(Ordering::SeqCst), 2);
    assert!(rows(root.path(), "normal").is_empty());
    assert!(rows(root.path(), "late").is_empty());
}

fn sliced_backing_event() -> (SourceEvent, Buffer, Weak<dyn Array>) {
    let SourceEvent::Data { batch, cursor } = data(&[0, 5, 6, 7], 0) else {
        unreachable!()
    };
    let mut columns = batch.table_payload().unwrap().batches()[0]
        .columns()
        .to_vec();
    let wide = "x".repeat(2 * 1024 * 1024);
    let key = "123456789012345678901234567890";
    columns[1] = Arc::new(StringArray::from(vec![&wide, key, key, key]));
    let record = RecordBatch::try_new(schema(), columns).unwrap().slice(1, 3);
    let backing = record
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .values()
        .clone();
    let reference = Arc::downgrade(record.column(1));
    let batch = Batch::table(vec![record], batch.metadata().clone()).unwrap();
    assert!(batch.estimated_bytes().unwrap() <= 200);
    (SourceEvent::Data { batch, cursor }, backing, reference)
}

#[tokio::test]
async fn test_late_files_sliced_backing_fanout_releases_on_cancel_and_drain() {
    tokio::time::timeout(Duration::from_secs(10), async {
        for cancel in [true, false] {
            let root = tempfile::tempdir().unwrap();
            let probe = Arc::new(Probe {
                fanout: true,
                ..Probe::default()
            });
            *probe.block_sink.lock().unwrap() = Some("left");
            let (event, backing, reference) = sliced_backing_event();
            assert!(backing.len() > 2 * 1024 * 1024);
            let job = runner_events(
                root.path(),
                vec![SourceEvent::Watermark(EventTime::from_micros(10)), event],
                None,
                probe.clone(),
                StreamRuntimeConfig {
                    edge_budget: calc_flow::EdgeBudget::new(3, 200).unwrap(),
                    ..StreamRuntimeConfig::default()
                },
            )
            .start()
            .await
            .unwrap();
            probe.blocked.notified().await;
            assert!(backing.strong_count() > 1);
            let status = job.status();
            assert!(
                status
                    .edges
                    .values()
                    .all(|edge| edge.high_water_bytes <= 200 && edge.high_water_rows <= 3)
            );
            let terminal = if cancel {
                job.cancel().await
            } else {
                probe.release.notify_one();
                job.wait().await
            };
            assert_eq!(
                terminal.state,
                if cancel {
                    JobState::Cancelled
                } else {
                    JobState::Completed
                },
                "{terminal:?}"
            );
            assert_settled(&job);
            assert!(reference.upgrade().is_none());
            assert_eq!(backing.strong_count(), 1);
            assert_eq!(probe.sink_drops.load(Ordering::SeqCst), 3);
            if !cancel {
                assert_eq!(rows(root.path(), "left"), [(5, 1), (6, 2), (7, 3)]);
                assert_eq!(rows(root.path(), "right"), [(5, 1), (6, 2), (7, 3)]);
            }
        }
    })
    .await
    .unwrap();
}

struct OrdinaryLate(Arc<Probe>);

impl Drop for OrdinaryLate {
    fn drop(&mut self) {
        self.0.sink_drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait]
impl calc_flow::StreamSink for OrdinaryLate {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        for record in batch.table_payload()?.batches() {
            let times = record
                .column_by_name("ts")
                .unwrap()
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let sequences = record
                .column_by_name("seq")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            self.0
                .ordinary_rows
                .lock()
                .unwrap()
                .extend((0..record.num_rows()).map(|row| (times.value(row), sequences.value(row))));
        }
        self.0.ordinary_received.notify_one();
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

async fn wait_for_ordinary_rows(probe: &Probe, count: usize) {
    loop {
        let changed = probe.ordinary_received.notified();
        if probe.ordinary_rows.lock().unwrap().len() >= count {
            return;
        }
        changed.await;
    }
}

#[tokio::test]
async fn test_late_files_ordinary_delivery_remains_per_output_and_replays_after_durable_cut() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let root = tempfile::tempdir().unwrap();
        let invalid = Arc::new(Probe {
            late_sink_mode: LateSinkMode::RequireExactlyOnce,
            ..Probe::default()
        });
        assert!(
            runner(root.path(), &[5, 20], None, invalid.clone())
                .start()
                .await
                .is_err()
        );
        assert!(invalid.opens.lock().unwrap().is_empty());
        let probe = Arc::new(Probe {
            late_sink_mode: LateSinkMode::Ordinary,
            ..Probe::default()
        });
        let first = runner(root.path(), &[5, 20], Some(2), probe.clone())
            .start()
            .await
            .unwrap();
        probe.paused.notified().await;
        first.trigger_checkpoint().await.unwrap();
        assert_eq!(first.cancel().await.state, JobState::Cancelled);
        assert_settled(&first);
        let second = runner(root.path(), &[5, 20], Some(4), probe.clone())
            .start()
            .await
            .unwrap();
        wait_for_ordinary_rows(&probe, 2).await;
        assert_eq!(second.cancel().await.state, JobState::Cancelled);
        assert_settled(&second);
        let third = runner(root.path(), &[5, 20], None, probe.clone())
            .start()
            .await
            .unwrap();
        assert_eq!(third.wait().await.state, JobState::Completed);
        assert_settled(&third);
        assert_eq!(
            third.status().delivery["output"].effective,
            DeliveryGuarantee::ExactlyOnce
        );
        assert_eq!(
            third.status().delivery["late"].effective,
            DeliveryGuarantee::AtLeastOnce
        );
        assert_eq!(
            *probe.ordinary_rows.lock().unwrap(),
            [(5, 0), (6, 10), (6, 10)]
        );
        assert_eq!(rows(root.path(), "normal"), [(20, 1), (40, 11)]);
        assert_eq!(probe.sink_drops.load(Ordering::SeqCst), 6);
    })
    .await
    .unwrap();
}
