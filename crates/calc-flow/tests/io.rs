mod support;

use std::{
    any::Any,
    collections::BTreeMap,
    sync::{Arc, Mutex, atomic::AtomicUsize},
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, BatchingSource, CalcFlowError, ExecutionOptions, ExternalPayload, Result,
    SinkRouter, Source, SourceItem,
};
use datafusion::arrow::record_batch::RecordBatch;
use serde_json::json;
use support::{Probe, QueueSource, RecordingSink, source_item, stateful_plan, string_batch};

#[tokio::test]
async fn batching_source_coalesces_adjacent_items_and_retains_latest_position() {
    let (source, opens) = QueueSource::new(vec![
        source_item(&[1, 2], json!(2), 4),
        source_item(&[3], json!(3), 5),
        source_item(&[4, 5], json!(5), 6),
    ]);
    let mut source = BatchingSource::new(source, 3, usize::MAX).unwrap();

    source.open(Some(json!(0))).await.unwrap();
    let first = source.next().await.unwrap().unwrap();
    let second = source.next().await.unwrap().unwrap();

    assert_eq!(first.batch.num_rows(), 3);
    assert_eq!(first.cursor, Some(json!(3)));
    assert_eq!(first.sequence, 5);
    assert_eq!(second.batch.num_rows(), 2);
    assert_eq!(second.cursor, Some(json!(5)));
    assert_eq!(second.sequence, 6);
    assert!(source.next().await.unwrap().is_none());
    assert_eq!(*opens.lock().unwrap(), vec![Some(json!(0))]);
}

#[tokio::test]
async fn batching_source_uses_arrow_memory_and_emits_an_oversized_item_alone() {
    let first = source_item(&[1, 2], json!(2), 1);
    let bytes = arrow_bytes(&first.batch);
    let (source, _) = QueueSource::new(vec![
        first,
        source_item(&[3, 4], json!(4), 2),
        source_item(&[5, 6, 7, 8], json!(8), 3),
    ]);
    let mut source = BatchingSource::new(source, usize::MAX, bytes + 1).unwrap();
    source.open(None).await.unwrap();

    assert_eq!(source.next().await.unwrap().unwrap().batch.num_rows(), 2);
    assert_eq!(source.next().await.unwrap().unwrap().batch.num_rows(), 2);
    assert_eq!(source.next().await.unwrap().unwrap().batch.num_rows(), 4);
    assert!(source.next().await.unwrap().is_none());
}

#[test]
fn batching_source_rejects_zero_limits() {
    let (source, _) = QueueSource::new(Vec::new());
    assert!(matches!(
        BatchingSource::new(source, 0, 1),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "max_rows"
    ));
    let (source, _) = QueueSource::new(Vec::new());
    assert!(matches!(
        BatchingSource::new(source, 1, 0),
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "max_bytes"
    ));
}

#[tokio::test]
async fn batching_source_rejects_arrays_and_schema_mismatch_but_accepts_zero_rows() {
    let array = Batch::external(Arc::new(TestArray), BatchMetadata::default()).unwrap();
    let (array_source, _) = QueueSource::new(vec![SourceItem {
        batch: array,
        cursor: None,
        sequence: 0,
    }]);
    let mut array_source = BatchingSource::new(array_source, 10, 10_000).unwrap();
    array_source.open(None).await.unwrap();
    assert!(matches!(
        array_source.next().await,
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "source.batch"
    ));

    let empty = support::int_batch(&[]);
    let (empty_source, _) = QueueSource::new(vec![SourceItem {
        batch: empty,
        cursor: None,
        sequence: 0,
    }]);
    let mut empty_source = BatchingSource::new(empty_source, 10, 10_000).unwrap();
    empty_source.open(None).await.unwrap();
    assert_eq!(
        empty_source.next().await.unwrap().unwrap().batch.num_rows(),
        0
    );
    assert!(empty_source.next().await.unwrap().is_none());

    let (mixed, _) = QueueSource::new(vec![
        source_item(&[1], json!(1), 1),
        SourceItem {
            batch: string_batch(&["bad"]),
            cursor: Some(json!(2)),
            sequence: 2,
        },
    ]);
    let mut mixed = BatchingSource::new(mixed, 10, 10_000).unwrap();
    mixed.open(None).await.unwrap();
    assert!(matches!(
        mixed.next().await,
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "source.batch.schema"
    ));
    assert!(matches!(
        mixed.next().await,
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "source"
    ));
}

#[tokio::test]
async fn candidate_read_failure_poisons_instead_of_skipping_the_accumulated_group() {
    let mut source = BatchingSource::new(
        CandidateErrorSource {
            item: Some(source_item(&[1, 2], json!(2), 2)),
            calls: 0,
        },
        10,
        10_000,
    )
    .unwrap();
    source.open(None).await.unwrap();

    assert!(matches!(
        source.next().await,
        Err(CalcFlowError::Format { message }) if message == "candidate injected"
    ));
    assert!(matches!(
        source.next().await,
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "source"
    ));
}

#[tokio::test]
async fn sink_router_preserves_order_stops_on_failure_and_uses_run_context() {
    let plan = stateful_plan("sink order", Arc::new(Probe::default()));
    let result = plan
        .execute(
            BTreeMap::from([("input".into(), support::int_batch(&[1, 2]))]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let never = Arc::new(AtomicUsize::new(0));
    let once = Arc::new(AtomicUsize::new(1));
    let mut router = SinkRouter::new();
    router
        .add(
            "output",
            Box::new(RecordingSink::new(
                "first",
                Arc::clone(&calls),
                Arc::clone(&never),
            )),
        )
        .unwrap();
    router
        .add(
            "output",
            Box::new(RecordingSink::new(
                "second",
                Arc::clone(&calls),
                Arc::clone(&once),
            )),
        )
        .unwrap();
    router
        .add(
            "output",
            Box::new(RecordingSink::new("third", Arc::clone(&calls), never)),
        )
        .unwrap();

    assert!(router.write_all(&result).await.is_err());
    let calls = calls.lock().unwrap();
    assert_eq!(
        calls.iter().map(|call| call.0.as_str()).collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert!(calls.iter().all(|call| call.1 == result.metadata.run_id));
}

fn arrow_bytes(batch: &Batch) -> usize {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(RecordBatch::get_array_memory_size)
        .sum()
}

#[derive(Debug)]
struct TestArray;

impl ExternalPayload for TestArray {
    fn backend(&self) -> &'static str {
        "test"
    }

    fn len(&self) -> usize {
        1
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct CandidateErrorSource {
    item: Option<SourceItem>,
    calls: usize,
}

#[async_trait]
impl Source for CandidateErrorSource {
    async fn open(&mut self, _cursor: Option<serde_json::Value>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceItem>> {
        self.calls += 1;
        match self.calls {
            1 => Ok(self.item.take()),
            2 => Err(CalcFlowError::Format {
                message: "candidate injected".into(),
            }),
            _ => Ok(None),
        }
    }
}
