use super::*;
use datafusion::arrow::buffer::Buffer;
use std::sync::{
    Weak,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::mpsc;

struct Fanout {
    consumers: [mpsc::Sender<Batch>; 2],
    calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl crate::StreamCollector for Fanout {
    async fn emit(&mut self, port: &str, batch: Batch) -> Result<()> {
        assert_eq!(port, "late");
        assert_eq!(batch.num_rows(), 1);
        assert!(batch.estimated_bytes()? <= 200);
        self.calls.fetch_add(1, Ordering::SeqCst);
        for consumer in &self.consumers {
            consumer.send(batch.clone()).await.unwrap();
        }
        Ok(())
    }
}

fn sliced_input() -> (Batch, Buffer, Weak<dyn Array>) {
    let wide = "x".repeat(2 * 1024 * 1024);
    let whole = input(&[&[
        Some(&wide),
        Some("012345678901234567890123456789"),
        Some("012345678901234567890123456789"),
        Some("012345678901234567890123456789"),
    ]]);
    let record = whole.table_payload().unwrap().batches()[0].slice(1, 3);
    let probe = record
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .values()
        .clone();
    let reference = Arc::downgrade(record.column(0));
    let batch = Batch::table(
        vec![record],
        BatchMetadata::new("source", 0, crate::JsonMap::new()).unwrap(),
    )
    .unwrap();
    (batch, probe, reference)
}

async fn send(input: Batch, mut fanout: Fanout) -> Result<()> {
    let mut plan = LateOutputPlan::new(
        &input,
        "roll",
        20,
        EdgeBudget::new(3, 200).unwrap(),
        0,
        schema(input.table_payload()?.schema()),
    )?;
    for index in 0..3 {
        plan.push(0, index, u64::try_from(index).unwrap(), 1, 1)?;
    }
    assert_eq!(
        plan.scratch_usage(),
        (3, 3 * LateOutputPlan::scratch_row_bytes())
    );
    plan.prepare()?.emit(&mut fanout).await?;
    Ok(())
}

#[tokio::test]
async fn test_late_plan_cancel_releases_callback_but_last_fanout_owns_backing() {
    let (input, backing, reference) = sliced_input();
    assert!(backing.len() > 2 * 1024 * 1024);
    assert!(input.estimated_bytes().unwrap() < 200);
    assert_eq!(backing.strong_count(), 2);
    let (left, mut left_rx) = mpsc::channel(1);
    let (right, mut right_rx) = mpsc::channel(1);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut callback = Box::pin(send(
        input,
        Fanout {
            consumers: [left, right],
            calls: calls.clone(),
        },
    ));
    assert!(futures::poll!(callback.as_mut()).is_pending());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    // Probe + input slice + queued first chunk + pending second chunk; no third chunk.
    assert_eq!(backing.strong_count(), 4);
    drop(callback);
    assert!(reference.upgrade().is_none());
    assert_eq!(backing.strong_count(), 2);
    drop(left_rx.try_recv().unwrap());
    assert_eq!(backing.strong_count(), 2);
    drop(right_rx.try_recv().unwrap());
    assert_eq!(backing.strong_count(), 1);
    assert!(left_rx.try_recv().is_err());
    assert!(right_rx.try_recv().is_err());
}

#[tokio::test]
async fn test_late_plan_drain_releases_callback_and_both_fanout_references() {
    let (input, backing, reference) = sliced_input();
    let (left, left_rx) = mpsc::channel(1);
    let (right, right_rx) = mpsc::channel(1);
    let calls = Arc::new(AtomicUsize::new(0));
    let callback = send(
        input,
        Fanout {
            consumers: [left, right],
            calls: calls.clone(),
        },
    );
    let results = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        tokio::join!(callback, drain(left_rx), drain(right_rx))
    })
    .await
    .unwrap();
    results.0.unwrap();
    assert_eq!((results.1, results.2), (3, 3));
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert!(reference.upgrade().is_none());
    assert_eq!(backing.strong_count(), 1);
}

async fn drain(mut receiver: mpsc::Receiver<Batch>) -> usize {
    let mut rows = 0;
    while let Some(batch) = receiver.recv().await {
        rows += batch.num_rows();
    }
    rows
}
