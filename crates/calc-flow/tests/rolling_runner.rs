//! Runner-level wiring coverage for the rolling operator: a real
//! `StreamingRunner` with scripted sources, a collecting sink, and a managed
//! checkpoint runtime drives the compiled dispatch arms, and a restart
//! against the same managed root proves checkpoint/recovery equivalence at
//! the job level (batch ≡ final stream across the restart boundary).

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
};

use calc_flow::{
    Batch, BatchMetadata, Cursor, EventTime, ExecutionOptions, JobState, JsonMap,
    ManagedCheckpointRuntime, NativeWatermarkCapability, PipelineBuilder, ReplayPositioning,
    RollingOperator, RollingSpec, SinkBinding, SourceBinding, SourceCapabilities,
    SourceDeliveryCapability, SourceEvent, SourceSchema, StreamExecutionPlan, StreamRequirements,
    StreamSink, StreamSource, StreamingRunner, UdfRegistry,
};
use datafusion::arrow::{
    array::{
        ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

fn input_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
        Field::new("volume", DataType::Int64, true),
    ]))
}

type InputRow = (i64, &'static str, u64, Option<f64>, Option<i64>);

fn input_batch(rows: &[InputRow]) -> Batch {
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from(rows.iter().map(|row| row.0).collect::<Vec<_>>())
                    .with_timezone("UTC"),
            ) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.1).collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                rows.iter().map(|row| row.2).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|row| row.3).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.4).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn rolling_spec() -> RollingSpec {
    serde_json::from_value(serde_json::json!({
        "configuration_version": 1,
        "state_layout_version": 1,
        "partition_by": ["symbol"],
        "event_time": "ts",
        "sequence_by": ["sequence"],
        "outputs": [
            {
                "kind": "lag",
                "primitive_version": 1,
                "input": "price",
                "output": "price_lag_1",
                "periods": 1
            },
            {
                "kind": "delta",
                "primitive_version": 1,
                "input": "volume",
                "output": "volume_delta_1",
                "periods": 1
            }
        ],
        "allowed_lateness_micros": 0,
        "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap()
}

fn rolling_plan() -> StreamExecutionPlan {
    let operator = RollingOperator::new("rolling", input_schema(), rolling_spec()).unwrap();
    PipelineBuilder::new("rolling_runner")
        .unwrap()
        .add_node("rolling", operator)
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

struct ScriptedSource {
    events: VecDeque<SourceEvent>,
    hold_open: bool,
}

#[async_trait::async_trait]
impl StreamSource for ScriptedSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 16,
            max_batch_bytes: 1 << 20,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> calc_flow::Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> calc_flow::Result<Option<SourceEvent>> {
        match self.events.pop_front() {
            Some(event) => Ok(Some(event)),
            None if self.hold_open => std::future::pending().await,
            None => Ok(None),
        }
    }

    async fn close(&mut self) -> calc_flow::Result<()> {
        Ok(())
    }
}

type CollectedRow = (i64, String, u64, Option<f64>, Option<i64>);

#[derive(Clone, Default)]
struct CollectedRows {
    rows: Arc<Mutex<Vec<CollectedRow>>>,
}

struct CollectingSink {
    collected: CollectedRows,
}

#[async_trait::async_trait]
impl StreamSink for CollectingSink {
    async fn open(&mut self) -> calc_flow::Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> calc_flow::Result<()> {
        for record in batch.table_payload()?.batches() {
            let ts = record
                .column_by_name("ts")
                .unwrap()
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let symbols = record
                .column_by_name("symbol")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let sequences = record
                .column_by_name("sequence")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let lags = record
                .column_by_name("price_lag_1")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let deltas = record
                .column_by_name("volume_delta_1")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let mut collected = self.collected.rows.lock().unwrap();
            for index in 0..record.num_rows() {
                collected.push((
                    ts.value(index),
                    symbols.value(index).to_owned(),
                    sequences.value(index),
                    lags.iter().nth(index).unwrap(),
                    deltas.iter().nth(index).unwrap(),
                ));
            }
        }
        Ok(())
    }

    async fn close(&mut self) -> calc_flow::Result<()> {
        Ok(())
    }
}

fn data_event(rows: &[InputRow], offset: u64) -> SourceEvent {
    SourceEvent::Data {
        batch: input_batch(rows),
        cursor: Cursor::unbound(offset.to_be_bytes().to_vec(), JsonMap::new()).unwrap(),
    }
}

fn runner(
    root: &std::path::Path,
    source: ScriptedSource,
    collected: &CollectedRows,
) -> StreamingRunner {
    let plan = rolling_plan();
    let source_id = plan.source_binding_ids()[0].to_owned();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    StreamingRunner::new(
        plan,
        BTreeMap::from([(source_id, SourceBinding::new(source))]),
        BTreeMap::from([(
            output_id,
            vec![
                SinkBinding::ordinary(
                    "sink",
                    CollectingSink {
                        collected: collected.clone(),
                    },
                )
                .unwrap(),
            ],
        )]),
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn single_job_emits_rolling_output_to_the_sink() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let rows: Vec<InputRow> = vec![
        (10, "a", 1, Some(1.0), Some(10)),
        (11, "a", 2, Some(2.0), Some(20)),
    ];
    let collected = CollectedRows::default();
    let job = runner(
        &root,
        ScriptedSource {
            events: VecDeque::from([
                data_event(&rows, 1),
                SourceEvent::Watermark(EventTime::from_micros(100)),
            ]),
            hold_open: false,
        },
        &collected,
    )
    .start()
    .await
    .unwrap();
    assert_eq!(job.wait().await.state, JobState::Completed);
    assert_eq!(collected.rows.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn runner_drives_rolling_dispatch_and_recovers_across_a_checkpoint_restart() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let rows: Vec<InputRow> = vec![
        (10, "a", 1, Some(1.0), Some(10)),
        (11, "a", 2, Some(2.0), Some(20)),
        (12, "b", 1, Some(5.0), Some(50)),
        (13, "a", 3, Some(3.0), Some(30)),
        (14, "b", 2, Some(6.0), Some(60)),
    ];

    let first_collected = CollectedRows::default();
    let first = runner(
        &root,
        ScriptedSource {
            events: VecDeque::from([data_event(&rows[..3], 1)]),
            hold_open: true,
        },
        &first_collected,
    )
    .start()
    .await
    .unwrap();
    let epoch = first.trigger_checkpoint().await.unwrap();
    assert_eq!(epoch, calc_flow::Epoch::INITIAL);
    first.cancel().await;

    let second_collected = CollectedRows::default();
    let second = runner(
        &root,
        ScriptedSource {
            events: VecDeque::from([
                data_event(&rows[3..], 2),
                SourceEvent::Watermark(EventTime::from_micros(100)),
            ]),
            hold_open: false,
        },
        &second_collected,
    )
    .start()
    .await
    .unwrap();
    assert_eq!(second.wait().await.state, JobState::Completed);

    let batch_plan = PipelineBuilder::new("rolling_runner_batch")
        .unwrap()
        .add_node(
            "rolling",
            RollingOperator::new("rolling", input_schema(), rolling_spec()).unwrap(),
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();
    let reference = batch_plan
        .execute(
            BTreeMap::from([("input".into(), input_batch(&rows))]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let reference_sink = CollectedRows::default();
    CollectingSink {
        collected: reference_sink.clone(),
    }
    .write(&reference.outputs["output"])
    .await
    .unwrap();

    assert_eq!(
        *second_collected.rows.lock().unwrap(),
        *reference_sink.rows.lock().unwrap(),
        "runner output across a checkpoint restart must equal the batch reference"
    );
}
