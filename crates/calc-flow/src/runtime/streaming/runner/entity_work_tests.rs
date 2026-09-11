mod lifecycle_tests;

use std::{
    collections::VecDeque,
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
    Result, RollingOperator, StreamExecutionPlan, StreamJobContext, StreamRequirements,
    UdfRegistry,
    runtime::streaming::{
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
    ];
    RecordBatch::try_new(schema(), columns).unwrap()
}

fn plan() -> StreamExecutionPlan {
    let leaf = |size| {
        json!({
            "kind": "mean", "primitive_version": 1, "input": "price",
            "frame": {"kind": "rows", "size": size}, "min_periods": size,
        })
    };
    let spec = serde_json::from_value(json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["symbol"], "event_time": "event_time", "sequence_by": ["sequence"],
        "allowed_lateness_micros": 0, "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1",
        "outputs": [{"kind": "difference", "primitive_version": 1,
            "left": leaf(5), "right": leaf(20), "output": "spread"}],
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

#[tokio::test]
async fn managed_native_dual_sma_registers_two_owned_numeric_lanes() {
    let input = input();
    let plan = plan();
    let records = Arc::new(Mutex::new(Vec::new()));
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
    let spec = ContinuousJobSpec {
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
            binding: OrdinarySinkBinding::new(Box::new(Sink(Arc::clone(&records)))),
        }],
        edge_budget: EdgeBudget::new(ROWS, 8 << 20).unwrap(),
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    };
    let mut runner = ContinuousRunner::new();
    let job = runner.start(spec).await.unwrap();
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
    assert_eq!(&actual.columns()[..4], input.columns());
    let spread = actual
        .column(4)
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
        "the qualified managed route must register two owned CPU lanes"
    );
}
