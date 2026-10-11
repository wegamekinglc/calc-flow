use super::*;
use crate::StreamMessage;
use datafusion::arrow::array::ArrayData;
use std::{
    ops::Range,
    sync::atomic::{AtomicUsize, Ordering},
};

#[test]
fn test_join_column_gather_owns_chunk_buffers_and_preserves_nullable_order() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    for incoming_left in [false, true] {
        runtime.block_on(assert_owned_output(&service, incoming_left));
    }
    drop(runtime);
    service.shutdown();
}

fn inputs() -> (Batch, Batch) {
    let left = RecordBatch::try_new(
        left_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; 3])),
            Arc::new(TimestampMicrosecondArray::from(vec![3, 1, 2]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![None, Some(42), None])),
        ],
    )
    .unwrap();
    let right = right_batch(vec![5, 0]).table_payload().unwrap().batches()[0].clone();
    (
        Batch::table(
            vec![left.slice(0, 1), left.slice(1, 2)],
            BatchMetadata::default(),
        )
        .unwrap(),
        Batch::table(
            vec![right.slice(0, 1), right.slice(1, 1)],
            BatchMetadata::default(),
        )
        .unwrap(),
    )
}

async fn assert_owned_output(service: &TestService, incoming_left: bool) {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job().with_gather_owner(service.owner("owned-join-column-output".into()));
    let context = StreamOperatorContext::new(&job, "match", None)
        .with_output_budget(EdgeBudget::new(2, 8 << 20).unwrap());
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let (left, right) = inputs();
    let input_buffers = buffers(&left)
        .into_iter()
        .chain(buffers(&right))
        .collect::<Vec<_>>();
    let (first_side, first, next_side, next) = if incoming_left {
        ("right", right.clone(), "left", left.clone())
    } else {
        ("left", left.clone(), "right", right.clone())
    };
    operator
        .process_data(first_side, first, &context, &mut collector)
        .await
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let worker_calls = Arc::clone(&calls);
    let _configuration = output_gather::test_support::configure(
        true,
        Some(Arc::new(move |_, completed| {
            if !completed {
                worker_calls.fetch_add(1, Ordering::SeqCst);
            }
        })),
    );
    operator
        .process_data(next_side, next, &context, &mut collector)
        .await
        .unwrap();
    let messages = collector.drain("output");
    assert_eq!(messages.len(), 3);
    assert_eq!(calls.load(Ordering::SeqCst), 18);
    let schema = operator.output_ports[0].schema().unwrap().clone();
    let mut prior = input_buffers;
    for (sequence, message) in messages.iter().enumerate() {
        let batch = message.as_data().unwrap();
        assert_eq!(batch.metadata().sequence(), sequence as u64);
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(
            batch.table_payload().unwrap().batches()[0].schema_ref(),
            &schema
        );
        let current = buffers(batch);
        for output in &current {
            assert!(
                prior
                    .iter()
                    .all(|source| output.end <= source.start || source.end <= output.start),
                "gathered chunks must own independent buffers"
            );
        }
        prior.extend(current);
    }
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((operator, context, left, right));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert_literals(&messages, incoming_left);
}

fn buffers(batch: &Batch) -> Vec<Range<usize>> {
    let mut result = Vec::new();
    for record in batch.table_payload().unwrap().batches() {
        for column in record.columns() {
            append_buffers(&column.to_data(), &mut result);
        }
    }
    result
}

fn append_buffers(data: &ArrayData, result: &mut Vec<Range<usize>>) {
    for buffer in data.buffers() {
        if !buffer.is_empty() {
            result.push(buffer.as_ptr() as usize..buffer.as_ptr() as usize + buffer.len());
        }
    }
    if let Some(nulls) = data.nulls() {
        let bytes = nulls.inner().values();
        if !bytes.is_empty() {
            result.push(bytes.as_ptr() as usize..bytes.as_ptr() as usize + bytes.len());
        }
    }
    for child in data.child_data() {
        append_buffers(child, result);
    }
}

fn assert_literals(messages: &[StreamMessage], incoming_left: bool) {
    let records = messages.iter().map(|message| {
        &message
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0]
    });
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut amounts = Vec::new();
    for record in records {
        left.extend_from_slice(
            record
                .column(1)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .values(),
        );
        right.extend_from_slice(
            record
                .column(4)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .values(),
        );
        amounts.extend(
            record
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter(),
        );
    }
    if incoming_left {
        assert_eq!(left, [3, 3, 1, 1, 2, 2]);
        assert_eq!(right, [0, 5, 0, 5, 0, 5]);
        assert_eq!(amounts, [None, None, Some(42), Some(42), None, None]);
    } else {
        assert_eq!(left, [1, 2, 3, 1, 2, 3]);
        assert_eq!(right, [5, 5, 5, 0, 0, 0]);
        assert_eq!(amounts, [Some(42), None, None, Some(42), None, None]);
    }
}
