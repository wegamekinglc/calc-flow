mod lifecycle_tests;

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, atomic::Ordering},
};

use async_trait::async_trait;
use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    compute::concat_batches,
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use parking_lot::Mutex;
use serde_json::json;

use super::{ContinuousJobState, ContinuousRunner};
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeBudget, EventTime, JsonMap, PipelineBuilder,
    Result, RollingOperator, SessionWatermark, StreamExecutionPlan, StreamJobContext,
    StreamRequirements, StreamRuntimeConfig, StreamSession, UdfRegistry,
    runtime::streaming::{
        checkpoint::ManagedCheckpointRuntime,
        checkpoint_runtime::CheckpointRuntimeSpec,
        job::{
            ContinuousJobSpec, M2DeliveryMode, NamedSinkBinding, NamedSourceBinding,
            OrdinarySinkBinding, OrdinaryStreamSink,
        },
        source_task::{Cursor, SourceBinding, SourceCapabilities, SourceEvent, StreamSource},
    },
};

const ROWS: usize = 64_000;
const ENTITIES: usize = 64;
const PRELOAD: usize = ENTITIES * 20;
const TOTAL: usize = PRELOAD + ROWS;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("price_copy", DataType::Float64, false),
    ]))
}

fn price(row: usize) -> f64 {
    100.0 + f64::from(u32::try_from(row % 257).unwrap()) / 8.0
}

fn input() -> RecordBatch {
    let symbols = (0..ENTITIES)
        .map(|entity| format!("S{entity:03}"))
        .collect::<Vec<_>>();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                (0..TOTAL).map(|row| i64::try_from(row / ENTITIES).unwrap()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(UInt64Array::from_iter_values(
            (0..TOTAL).map(|row| u64::try_from(row).unwrap()),
        )),
        Arc::new(StringArray::from_iter_values(
            (0..TOTAL).map(|row| symbols[row % ENTITIES].as_str()),
        )),
        Arc::new(Float64Array::from_iter_values((0..TOTAL).map(price))),
        Arc::new(Float64Array::from_iter_values((0..TOTAL).map(price))),
    ];
    RecordBatch::try_new(schema(), columns).unwrap()
}

fn plan() -> StreamExecutionPlan {
    sma_plan("price_copy", &json!({"kind": "drop", "metrics_version": 1}))
}

fn sma_plan(right_input: &str, late_policy: &serde_json::Value) -> StreamExecutionPlan {
    let leaf = |size, input| {
        json!({
            "kind": "mean", "primitive_version": 1, "input": input,
            "frame": {"kind": "rows", "size": size}, "min_periods": size,
        })
    };
    let spec = serde_json::from_value(json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["symbol"], "event_time": "event_time", "sequence_by": ["sequence"],
        "allowed_lateness_micros": 0, "late_policy": late_policy,
        "value_policy": "stateful_numeric_v1",
        "outputs": [{"kind": "difference", "primitive_version": 1,
            "left": leaf(5, "price"), "right": leaf(20, right_input), "output": "spread"}],
    }))
    .unwrap();
    PipelineBuilder::new("owned-entity-lanes")
        .unwrap()
        .add_node(
            "rolling",
            RollingOperator::new("rolling", schema(), spec).unwrap(),
        )
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

struct Source(VecDeque<SourceEvent>);

#[async_trait]
impl StreamSource for Source {
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(self.0.pop_front())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: ROWS,
            max_batch_bytes: 8 << 20,
        }
    }
}

struct Sink(Arc<Mutex<Vec<RecordBatch>>>);

#[async_trait]
impl OrdinaryStreamSink for Sink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0
            .lock()
            .extend(batch.table_payload()?.batches().iter().cloned());
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

fn job_spec(
    plan: StreamExecutionPlan,
    input: &RecordBatch,
    records: Arc<Mutex<Vec<RecordBatch>>>,
) -> ContinuousJobSpec {
    let source = Source(VecDeque::from([
        SourceEvent::Data {
            batch: Batch::table(vec![input.slice(0, PRELOAD)], BatchMetadata::default()).unwrap(),
            cursor: Cursor::unbound(vec![1], JsonMap::new()).unwrap(),
        },
        SourceEvent::Watermark(EventTime::from_micros(19)),
        SourceEvent::Data {
            batch: Batch::table(vec![input.slice(PRELOAD, ROWS)], BatchMetadata::default())
                .unwrap(),
            cursor: Cursor::unbound(vec![2], JsonMap::new()).unwrap(),
        },
        SourceEvent::Watermark(EventTime::from_micros(1019)),
    ]));
    ContinuousJobSpec {
        context: StreamJobContext::new(
            7,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: vec![NamedSourceBinding {
            binding_id: "input".into(),
            binding: SourceBinding::new(Box::new(source), None, 0).unwrap(),
        }],
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "sink".into(),
            binding: OrdinarySinkBinding::new(Box::new(Sink(records))),
        }],
        edge_budget: EdgeBudget::new(ROWS, 8 << 20).unwrap(),
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

#[tokio::test]
async fn managed_native_dual_sma_registers_two_owned_morsels() {
    let input = input();
    let records = Arc::new(Mutex::new(Vec::new()));
    let spec = job_spec(plan(), &input, Arc::clone(&records));
    let mut runner = ContinuousRunner::new();
    let job = runner.start(spec).await.unwrap();
    job.core.morsels.set_worker_limit_for_test(2);
    let outcome = job.wait().await;
    assert_eq!(
        outcome.state,
        ContinuousJobState::Completed,
        "{:?}",
        outcome.errors
    );
    assert!(outcome.errors.is_empty());
    let actual = {
        let records = records.lock();
        concat_batches(&records[0].schema(), records.iter()).unwrap()
    };
    assert_eq!(actual.num_rows(), TOTAL);
    let input_width = input.num_columns();
    assert_eq!(&actual.columns()[..input_width], input.columns());
    let spread = actual
        .column(input_width)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for row in 0..TOTAL {
        assert_eq!(spread.is_null(row), row / ENTITIES < 19);
        if row / ENTITIES >= 19 {
            let mean = |window| {
                (0..window)
                    .map(|offset| price(row - offset * ENTITIES))
                    .sum::<f64>()
                    / f64::from(u32::try_from(window).unwrap())
            };
            assert!((spread.value(row) - (mean(5) - mean(20))).abs() < 1e-10);
        }
    }
    let launched = job.core.owned_lane_launches.load(Ordering::SeqCst);
    assert!(job.core.runtime_status.lock().tasks.snapshot().is_empty());
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(
        launched, 2,
        "the qualified managed route must register two owned compute morsels"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn checkpointed_shared_dual_sma_uses_owned_lanes_and_matches_caller_session() {
    let input = input();
    let plan = || sma_plan("price", &json!({"kind": "error", "scope": "envelope"}));
    let records = Arc::new(Mutex::new(Vec::new()));
    let spec = job_spec(plan(), &input, Arc::clone(&records));
    let directory = tempfile::tempdir().unwrap();
    let config = StreamRuntimeConfig {
        checkpointing: true,
        edge_budget: spec.edge_budget,
        ..StreamRuntimeConfig::default()
    };
    let checkpoint = CheckpointRuntimeSpec::managed(
        ManagedCheckpointRuntime::new(directory.path().join("runner")).unwrap(),
        config,
    )
    .unwrap();
    let mut runner = ContinuousRunner::new();
    let start = runner.start_checkpointed(spec, checkpoint);
    start
        .core
        .as_ref()
        .unwrap()
        .morsels
        .set_worker_limit_for_test(2);
    let job = start.await.unwrap();
    let outcome = job.wait().await;
    assert_eq!(
        outcome.state,
        ContinuousJobState::Completed,
        "{:?}",
        outcome.errors
    );
    let rolling = job.rolling_metrics().remove("rolling").unwrap();
    let parallel_rows = [rolling.data, rolling.watermark, rolling.end]
        .iter()
        .map(|callback| callback.numeric_parallel_rows)
        .sum::<u64>();
    let launched = job.core.owned_lane_launches.load(Ordering::SeqCst);
    drop(job);
    runner.shutdown().await.unwrap();

    let actual = {
        let records = records.lock();
        concat_batches(&records[0].schema(), records.iter()).unwrap()
    };
    let expected = std::thread::spawn(move || {
        let watermarks = BTreeMap::from([("input".to_owned(), SessionWatermark::Explicit)]);
        let mut session = StreamSession::new_with_checkpoints(
            plan(),
            &config,
            watermarks,
            crate::ManagedCheckpointRuntime::new(directory.path().join("session")).unwrap(),
        )
        .unwrap();
        let mut records = Vec::new();
        for (offset, rows, watermark) in [(0, PRELOAD, 19), (PRELOAD, ROWS, 1019)] {
            let batch =
                Batch::table(vec![input.slice(offset, rows)], BatchMetadata::default()).unwrap();
            let output = session
                .push("input", batch, Some(EventTime::from_micros(watermark)))
                .unwrap();
            for batch in output.batches("output") {
                records.extend(batch.table_payload().unwrap().batches().iter().cloned());
            }
        }
        assert!(session.finish().unwrap().is_empty());
        concat_batches(&records[0].schema(), records.iter()).unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(actual.num_rows(), TOTAL);
    assert_eq!(actual, expected);
    assert_eq!(launched, 2, "shared dual SMA should use the owned lanes");
    assert_eq!(parallel_rows, ROWS as u64);
}
