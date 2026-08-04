mod support;

use std::{
    any::Any,
    collections::BTreeMap,
    sync::{Arc, Mutex, atomic::AtomicUsize},
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, BatchingSource, CalcFlowError, CancellationToken, ExecutionOptions,
    ExternalPayload, Result, RunContext, Sink, SinkRouter, Source, SourceItem,
};
use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use serde_json::json;
use support::{
    GatedCandidateSource, Probe, QueueSource, RecordingSink, source_item, stateful_plan,
    string_batch,
};

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
async fn batching_source_uses_arrow_memory_and_rejects_an_oversized_item() {
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
    assert!(matches!(
        source.next().await,
        Err(CalcFlowError::InvalidArgument { field, .. }) if field == "source.batch"
    ));
}

#[tokio::test]
async fn batching_source_counts_logical_slice_bytes_across_record_batches() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    let base = Arc::new(Int64Array::from((0..64).collect::<Vec<_>>())) as ArrayRef;
    let record = |offset, len| {
        RecordBatch::try_new(Arc::clone(&schema), vec![base.slice(offset, len)]).unwrap()
    };
    let item = |records: Vec<RecordBatch>, cursor, sequence| SourceItem {
        batch: Batch::table(records, BatchMetadata::default()).unwrap(),
        cursor: Some(json!(cursor)),
        sequence,
    };
    let (source, _) = QueueSource::new(vec![
        item(vec![record(4, 2)], 2, 1),
        item(vec![record(12, 1), record(24, 1)], 4, 2),
        item(vec![record(40, 1)], 5, 3),
    ]);
    // Four logical i64 values occupy exactly 32 bytes even though every
    // record shares a much larger backing allocation.
    let mut source = BatchingSource::new(source, usize::MAX, 32).unwrap();
    source.open(None).await.unwrap();

    let first = source.next().await.unwrap().unwrap();
    assert_eq!(first.batch.num_rows(), 4);
    assert_eq!(first.cursor, Some(json!(4)));
    assert_eq!(first.sequence, 2);
    assert_eq!(source.next().await.unwrap().unwrap().batch.num_rows(), 1);
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
async fn batching_source_can_return_its_unopened_inner_source() {
    let (source, opens) = QueueSource::new(Vec::new());
    let mut source = BatchingSource::new(source, 1, 1).unwrap().into_inner();

    source.open(Some(json!(7))).await.unwrap();
    assert_eq!(*opens.lock().unwrap(), vec![Some(json!(7))]);
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
async fn candidate_read_failure_retains_the_accumulated_group_for_retry() {
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
    let retained = source.next().await.unwrap().unwrap();
    assert_eq!(retained.batch.num_rows(), 2);
    assert_eq!(retained.cursor, Some(json!(2)));
    assert_eq!(retained.sequence, 2);
}

#[tokio::test]
async fn cancelled_candidate_read_retains_the_accumulated_group_for_retry() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let source = GatedCandidateSource::new(
        vec![
            source_item(&[1], json!(1), 1),
            source_item(&[2], json!(2), 2),
        ],
        2,
        Arc::clone(&started),
        release,
    );
    let mut source = BatchingSource::new(source, 10, usize::MAX).unwrap();
    source.open(None).await.unwrap();

    let mut cancelled = Box::pin(source.next());
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("candidate gate did not suspend batching: {result:?}"),
    }
    drop(cancelled);

    let retained = source.next().await.unwrap().unwrap();
    assert_eq!(retained.batch.num_rows(), 2);
    assert_eq!(retained.cursor, Some(json!(2)));
    assert_eq!(retained.sequence, 2);
}

#[tokio::test]
async fn oversized_first_item_fails_without_awaiting_a_candidate() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let source = GatedCandidateSource::new(
        vec![
            source_item(&[1, 2], json!(2), 1),
            source_item(&[3], json!(3), 2),
        ],
        2,
        Arc::clone(&started),
        release,
    );
    let mut source = BatchingSource::new(source, 1, usize::MAX).unwrap();
    source.open(None).await.unwrap();

    tokio::select! {
        result = source.next() => assert!(
            matches!(result, Err(CalcFlowError::InvalidArgument { ref field, .. }) if field == "source.batch"),
            "oversized item must fail before emission: {result:?}"
        ),
        () = started.notified() => panic!("oversized item unnecessarily awaited a candidate"),
    }
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

#[tokio::test]
async fn sink_router_checks_cancellation_before_each_sink_write() {
    let token = CancellationToken::new();
    let plan = stateful_plan("sink cancellation", Arc::new(Probe::default()));
    let result = plan
        .execute(
            BTreeMap::from([("input".into(), support::int_batch(&[1]))]),
            ExecutionOptions {
                cancellation: token.clone(),
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap();
    let later_calls = Arc::new(AtomicUsize::new(0));
    let mut router = SinkRouter::new();
    router
        .add(
            "output",
            Box::new(CancellingSink {
                token,
                calls: Arc::new(AtomicUsize::new(0)),
            }),
        )
        .unwrap();
    router
        .add("output", Box::new(CountingSink(Arc::clone(&later_calls))))
        .unwrap();

    assert!(matches!(
        router.write_all(&result).await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(later_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
}

fn arrow_bytes(batch: &Batch) -> usize {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(RecordBatch::columns)
        .map(|column| column.to_data().get_slice_memory_size().unwrap())
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

    fn estimated_bytes(&self) -> usize {
        8
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct CandidateErrorSource {
    item: Option<SourceItem>,
    calls: usize,
}

struct CancellingSink {
    token: CancellationToken,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Sink for CancellingSink {
    async fn write(&mut self, _batch: &Batch, _context: &RunContext) -> Result<()> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.token.cancel();
        Ok(())
    }
}

struct CountingSink(Arc<AtomicUsize>);

#[async_trait]
impl Sink for CountingSink {
    async fn write(&mut self, _batch: &Batch, _context: &RunContext) -> Result<()> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
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
