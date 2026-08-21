use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use calc_flow::{
    AggregateFunction, Batch, BatchMetadata, Cursor, EventTime, ManagedCheckpointRuntime,
    NativeWatermarkCapability, PipelineBuilder, ReplayPositioning, Result, SinkBinding,
    SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema,
    StreamExecutionPlan, StreamRequirements, StreamSink, StreamSource, StreamingRunner,
    UdfRegistry, WatermarkPolicy, WindowAggregateOperator, WindowSpec,
};
use datafusion::arrow::{
    array::{ArrayRef, Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use tokio::sync::Mutex;

#[derive(Debug, Eq, PartialEq)]
struct WindowResult {
    start_micros: i64,
    end_micros: i64,
    account: String,
    total: i64,
}

struct OrdersSource {
    step: usize,
}

#[async_trait]
impl StreamSource for OrdersSource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: 3,
            max_batch_bytes: 4096,
            schema: SourceSchema::Exact(input_schema()),
            native_watermarks: NativeWatermarkCapability::EmitsNative,
        }
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.step = cursor
            .as_ref()
            .and_then(|cursor| cursor.payload().get("step"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|step| usize::try_from(step).ok())
            .unwrap_or_default();
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let event = match self.step {
            0 => Some(SourceEvent::Data {
                batch: orders_batch()?,
                cursor: Cursor::unbound(
                    1_u64.to_be_bytes().to_vec(),
                    BTreeMap::from([("step".into(), serde_json::json!(1))]),
                )?,
            }),
            1 => Some(SourceEvent::Watermark(EventTime::from_micros(60_000_000))),
            _ => None,
        };
        self.step += 1;
        Ok(event)
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

struct CollectSink {
    windows: Arc<Mutex<Vec<WindowResult>>>,
}

#[async_trait]
impl StreamSink for CollectSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        let mut windows = self.windows.lock().await;
        for record in batch
            .table_payload()
            .expect("window output is a table")
            .batches()
        {
            let starts = record
                .column_by_name("window_start")
                .expect("window output contains window_start")
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("window_start is timestamp[us]");
            let ends = record
                .column_by_name("window_end")
                .expect("window output contains window_end")
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("window_end is timestamp[us]");
            let accounts = record
                .column_by_name("account")
                .expect("window output contains account")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("account is utf8");
            let totals = record
                .column_by_name("total")
                .expect("window output contains total")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("total is int64");
            windows.extend((0..record.num_rows()).map(|row| WindowResult {
                start_micros: starts.value(row),
                end_micros: ends.value(row),
                account: accounts.value(row).to_owned(),
                total: totals.value(row),
            }));
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

fn input_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("account", DataType::Utf8, false),
        Field::new("amount", DataType::Int64, false),
    ]))
}

fn orders_batch() -> Result<Batch> {
    let record = RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![
                5_000_000, 25_000_000, 65_000_000,
            ])) as ArrayRef,
            Arc::new(StringArray::from(vec!["retail", "retail", "retail"])),
            Arc::new(Int64Array::from(vec![5, 7, 9])),
        ],
    )
    .map_err(|error| calc_flow::CalcFlowError::InvalidArgument {
        field: "example.batch".into(),
        message: error.to_string(),
    })?;
    Batch::table(vec![record], BatchMetadata::default())
}

fn window_plan() -> Result<StreamExecutionPlan> {
    let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))?
        .group_by(["account"])?
        .aggregate(AggregateFunction::Sum, "amount", "total")?;
    let operator = WindowAggregateOperator::new("minute_totals", input_schema(), spec)?;
    PipelineBuilder::new("windowed-streaming-example")?
        .add_node("minute_totals", operator)?
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
}

#[tokio::main]
async fn main() -> Result<()> {
    let plan = window_plan()?;
    let source_id = plan.source_binding_ids()[0].to_owned();
    let output_id = plan.sink_binding_ids()[0].to_owned();
    let windows = Arc::new(Mutex::new(Vec::new()));
    let directory = tempfile::tempdir().map_err(|source| calc_flow::CalcFlowError::Io {
        path: "temporary checkpoint directory".into(),
        source,
    })?;
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source_id,
            SourceBinding::new(OrdersSource { step: 0 })
                .with_watermark_policy(WatermarkPolicy::SourceProvided),
        )]),
        BTreeMap::from([(
            output_id,
            vec![SinkBinding::ordinary(
                "window-collector",
                CollectSink {
                    windows: Arc::clone(&windows),
                },
            )?],
        )]),
        ManagedCheckpointRuntime::new(directory.path())?,
    )?;

    let outcome = runner.start().await?.wait().await;
    let actual = windows.lock().await;
    let expected = [
        WindowResult {
            start_micros: 0,
            end_micros: 60_000_000,
            account: "retail".into(),
            total: 12,
        },
        WindowResult {
            start_micros: 60_000_000,
            end_micros: 120_000_000,
            account: "retail".into(),
            total: 9,
        },
    ];
    assert_eq!(actual.as_slice(), expected);

    println!("terminal state: {:?}", outcome.state);
    println!("completed epoch: {:?}", outcome.completed_epoch);
    println!("windows: {actual:?}");
    Ok(())
}
