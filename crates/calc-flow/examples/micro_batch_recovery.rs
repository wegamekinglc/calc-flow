use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CheckpointStore, ExpressionOperator, FileCheckpointStore,
    MicroBatchRunner, PipelineBuilder, Result, RunContext, Sink, SinkRouter, Source, SourceItem,
    UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
use serde_json::{Value, json};
use tokio::sync::Mutex;

struct ReplaySource {
    items: VecDeque<SourceItem>,
}

impl ReplaySource {
    fn new() -> Result<Self> {
        Ok(Self {
            items: [source_item(1, 2, 1)?, source_item(3, 4, 2)?].into(),
        })
    }
}

#[async_trait]
impl Source for ReplaySource {
    async fn open(&mut self, cursor: Option<Value>) -> Result<()> {
        let recovered = cursor.as_ref().and_then(Value::as_u64).unwrap_or_default();
        while self
            .items
            .front()
            .is_some_and(|item| item.sequence <= recovered)
        {
            self.items.pop_front();
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceItem>> {
        Ok(self.items.pop_front())
    }
}

struct RecordingSink {
    delivered: Arc<Mutex<Vec<u64>>>,
}

#[async_trait]
impl Sink for RecordingSink {
    async fn write(&mut self, batch: &Batch, _context: &RunContext) -> Result<()> {
        self.delivered
            .lock()
            .await
            .push(batch.metadata().sequence());
        Ok(())
    }
}

fn source_item(a: i64, b: i64, sequence: u64) -> Result<SourceItem> {
    let record = RecordBatch::try_from_iter(vec![
        (
            "a",
            Arc::new(Int64Array::from(vec![a])) as Arc<dyn datafusion::arrow::array::Array>,
        ),
        ("b", Arc::new(Int64Array::from(vec![b])) as _),
    ])
    .map_err(|error| calc_flow::CalcFlowError::InvalidArgument {
        field: "example.batch".into(),
        message: error.to_string(),
    })?;
    Ok(SourceItem {
        batch: Batch::table(
            vec![record],
            BatchMetadata::new("replay-source", sequence, BTreeMap::default())?,
        )?,
        cursor: Some(json!(sequence)),
        sequence,
    })
}

fn sink_router(delivered: &Arc<Mutex<Vec<u64>>>) -> Result<SinkRouter> {
    let mut sinks = SinkRouter::new();
    sinks.add(
        "output",
        Box::new(RecordingSink {
            delivered: Arc::clone(delivered),
        }),
    )?;
    Ok(sinks)
}

#[tokio::main]
async fn main() -> Result<()> {
    let plan = Arc::new(
        PipelineBuilder::new("recovery-example")?
            .add_node(
                "calculate",
                Box::new(ExpressionOperator::new(
                    "calculate",
                    "total = a + b",
                    Vec::new(),
                    None,
                    Vec::new(),
                )?),
            )?
            .compile(&UdfRegistry::new().snapshot())?,
    );
    let directory = tempfile::tempdir().map_err(|source| calc_flow::CalcFlowError::Io {
        path: "temporary checkpoint directory".into(),
        source,
    })?;
    let store = Arc::new(FileCheckpointStore::new(directory.path()).await?);
    let delivered = Arc::new(Mutex::new(Vec::new()));

    let mut first = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(ReplaySource::new()?),
        sink_router(&delivered)?,
        Arc::clone(&store) as Arc<dyn CheckpointStore>,
        1,
    )?;
    first.next().await?.expect("the first item is available");
    drop(first);

    let mut recovered = MicroBatchRunner::new(
        Arc::clone(&plan),
        Box::new(ReplaySource::new()?),
        sink_router(&delivered)?,
        Arc::clone(&store) as Arc<dyn CheckpointStore>,
        1,
    )?;
    recovered
        .next()
        .await?
        .expect("the second item is available after recovery");
    assert!(recovered.next().await?.is_none());
    drop(recovered);

    assert_eq!(*delivered.lock().await, [1, 2]);
    assert_eq!(store.load(plan.name()).await?.unwrap().sequence, 2);
    println!("recovered deliveries: {:?}", *delivered.lock().await);
    Ok(())
}
