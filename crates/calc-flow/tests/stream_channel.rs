//! RED (M1.4): the rows + bytes dual-limit edge channel.
//!
//! Every test in this file fails to compile until
//! `calc_flow::{edge_channel, ChannelMetrics, EdgeReceiver, EdgeSender,
//! EnvelopeCost}` exist with the frozen v3 surface (spec S10, plan task
//! M1.4). The expected RED reason is an unresolved import of those symbols.

use std::{
    any::Any,
    future::{Future, poll_fn},
    pin::Pin,
    sync::Arc,
    task::Poll,
    time::Duration,
};

use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, ChannelMetrics, EdgeBudget, EnvelopeCost, ExternalPayload,
    StreamMessage, edge_channel,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

/// An external payload with an exact, test-chosen row count and byte cost, so
/// channel budgets can be set to precise values.
#[derive(Debug)]
struct FixedCostPayload {
    rows: usize,
    bytes: usize,
}

impl ExternalPayload for FixedCostPayload {
    fn backend(&self) -> &'static str {
        "stream-channel-test"
    }

    fn len(&self) -> usize {
        self.rows
    }

    fn estimated_bytes(&self) -> usize {
        self.bytes
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn external_batch(rows: usize, bytes: usize) -> Batch {
    Batch::external(
        Arc::new(FixedCostPayload { rows, bytes }),
        BatchMetadata::default(),
    )
    .unwrap()
}

fn table_batch(rows: i64) -> Batch {
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from_iter_values(0..rows)) as _,
    )])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

/// Polls the future once, expecting it to be pending (the deterministic
/// half of a blocked-send assertion; no timing involved).
async fn assert_pending(future: &mut (impl Future + Unpin)) {
    poll_fn(|context| match Pin::new(&mut *future).poll(context) {
        Poll::Pending => Poll::Ready(()),
        Poll::Ready(_) => panic!("future unexpectedly completed"),
    })
    .await;
}

/// Polls the future once, expecting `Ready` with its output.
async fn assert_ready<T>(future: &mut (impl Future<Output = T> + Unpin)) -> T {
    poll_fn(|context| match Pin::new(&mut *future).poll(context) {
        Poll::Pending => panic!("future unexpectedly pending"),
        Poll::Ready(output) => Poll::Ready(output),
    })
    .await
}

#[test]
fn channel_rejects_a_zero_row_budget() {
    let error = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 0,
            max_bytes: 1_024,
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { ref field, .. } if field.contains("max_rows")
    ));
}

#[test]
fn channel_rejects_a_zero_byte_budget() {
    let error = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 16,
            max_bytes: 0,
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { ref field, .. } if field.contains("max_bytes")
    ));
}

#[test]
fn envelope_cost_charges_data_rows_and_bytes() {
    let data = EnvelopeCost::of_message(&StreamMessage::data(external_batch(7, 41))).unwrap();
    assert_eq!(data.messages(), 1);
    assert_eq!(data.rows(), 7);
    assert_eq!(data.bytes(), 41);
}

#[test]
fn envelope_cost_checked_add_sums_componentwise() {
    let left = EnvelopeCost::of_message(&StreamMessage::data(external_batch(2, 8))).unwrap();
    let right = EnvelopeCost::of_message(&StreamMessage::data(external_batch(3, 4))).unwrap();
    let sum = left.checked_add(&right).unwrap();
    assert_eq!(sum.messages(), 2);
    assert_eq!(sum.rows(), 5);
    assert_eq!(sum.bytes(), 12);
}

#[test]
fn envelope_cost_checked_add_overflow_is_a_typed_error() {
    let huge = EnvelopeCost::new(usize::MAX, usize::MAX, usize::MAX);
    let one = EnvelopeCost::new(1, 1, 1);
    assert!(matches!(
        huge.checked_add(&one),
        Err(CalcFlowError::InvalidArgument { .. })
    ));
}

#[test]
fn envelope_cost_checked_sub_releases_componentwise() {
    let total = EnvelopeCost::new(4, 9, 30);
    let part = EnvelopeCost::new(1, 4, 12);
    let rest = total.checked_sub(&part).unwrap();
    assert_eq!(rest.messages(), 3);
    assert_eq!(rest.rows(), 5);
    assert_eq!(rest.bytes(), 18);
}

#[test]
fn envelope_cost_checked_sub_underflow_is_a_typed_error() {
    let small = EnvelopeCost::new(1, 1, 1);
    let big = EnvelopeCost::new(2, 1, 1);
    assert!(matches!(
        small.checked_sub(&big),
        Err(CalcFlowError::Internal { .. })
    ));
}

#[tokio::test]
async fn send_receive_delivers_data_in_fifo_order() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 16,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    sender
        .send(StreamMessage::data(external_batch(1, 8)))
        .await
        .unwrap();
    sender
        .send(StreamMessage::data(external_batch(2, 16)))
        .await
        .unwrap();

    let first = receiver.recv().await.unwrap().unwrap();
    let second = receiver.recv().await.unwrap().unwrap();
    assert_eq!(first.as_data().unwrap().num_rows(), 1);
    assert_eq!(second.as_data().unwrap().num_rows(), 2);
}

#[tokio::test]
async fn send_rejects_a_single_message_larger_than_the_byte_budget() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 16,
            max_bytes: 32,
        },
    )
    .unwrap();

    let error = sender
        .send(StreamMessage::data(external_batch(1, 33)))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { ref field, .. } if field == "message.bytes"
    ));

    // The failed send changed nothing: the queue stays empty and a legal
    // message still flows (S10.3 has no "one oversize message" exception).
    let legal = StreamMessage::data(external_batch(1, 32));
    sender.send(legal).await.unwrap();
    assert_eq!(
        receiver
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_data()
            .unwrap()
            .num_rows(),
        1
    );
}

#[tokio::test]
async fn send_rejects_a_single_message_larger_than_the_row_budget() {
    let (mut sender, _receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 4,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    let error = sender
        .send(StreamMessage::data(external_batch(5, 1)))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { ref field, .. } if field == "message.rows"
    ));
}

#[tokio::test]
async fn send_accepts_a_single_message_exactly_at_the_row_budget() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 4,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    // The oversize check is strict (S10.3): a message equal to the row
    // limit still fits, and it fits without waiting.
    let mut exact = Box::pin(sender.send(StreamMessage::data(external_batch(4, 8))));
    assert_ready(&mut exact).await.unwrap();
    drop(exact);

    // Rows 4/4: one more row must wait for a release.
    let mut blocked = Box::pin(sender.send(StreamMessage::data(external_batch(1, 1))));
    assert_pending(&mut blocked).await;

    assert_eq!(
        receiver
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_data()
            .unwrap()
            .num_rows(),
        4
    );
    blocked.await.unwrap();
}

#[tokio::test]
async fn sender_blocks_when_the_row_budget_is_full_and_resumes_after_receive() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 5,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    sender
        .send(StreamMessage::data(external_batch(2, 1)))
        .await
        .unwrap();
    sender
        .send(StreamMessage::data(external_batch(3, 1)))
        .await
        .unwrap();

    // Rows 5/5: the third send cannot reserve even though bytes are free.
    let mut blocked = Box::pin(sender.send(StreamMessage::data(external_batch(1, 1))));
    assert_pending(&mut blocked).await;

    assert_eq!(
        receiver
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_data()
            .unwrap()
            .num_rows(),
        2
    );
    blocked.await.unwrap();
    assert_eq!(
        receiver
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_data()
            .unwrap()
            .num_rows(),
        3
    );
    assert_eq!(
        receiver
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_data()
            .unwrap()
            .num_rows(),
        1
    );
}

#[tokio::test]
async fn sender_blocks_when_the_byte_budget_is_full_with_few_messages() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 100,
            max_bytes: 64,
        },
    )
    .unwrap();

    sender
        .send(StreamMessage::data(external_batch(1, 64)))
        .await
        .unwrap();

    // One message and one row in flight, yet bytes 64/64 block the send.
    let mut blocked = Box::pin(sender.send(StreamMessage::data(external_batch(1, 1))));
    assert_pending(&mut blocked).await;

    assert_eq!(
        receiver
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_data()
            .unwrap()
            .num_rows(),
        1
    );
    blocked.await.unwrap();
    assert_eq!(
        receiver
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_data()
            .unwrap()
            .num_rows(),
        1
    );
}

#[tokio::test]
async fn channel_with_a_budget_of_one_hands_messages_over_one_at_a_time() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 1,
            max_bytes: 1,
        },
    )
    .unwrap();

    sender
        .send(StreamMessage::data(external_batch(1, 1)))
        .await
        .unwrap();

    // A single message saturates both limits: the next send rendezvous-waits
    // for the receiver to take the first one.
    let mut second = Box::pin(sender.send(StreamMessage::data(external_batch(1, 1))));
    assert_pending(&mut second).await;

    receiver.recv().await.unwrap().unwrap();
    second.await.unwrap();
    assert_eq!(
        receiver
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_data()
            .unwrap()
            .num_rows(),
        1
    );
}

#[tokio::test]
async fn cancelled_send_leaves_the_budget_untouched() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 2,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    sender
        .send(StreamMessage::data(external_batch(2, 8)))
        .await
        .unwrap();

    // The blocked send is polled once (so it registers as a waiter) and then
    // cancelled by dropping its future. It must release nothing and reserve
    // nothing: the budget afterwards is exactly the first message's charge.
    {
        let mut blocked = Box::pin(sender.send(StreamMessage::data(external_batch(1, 1))));
        assert_pending(&mut blocked).await;
    }

    assert_eq!(
        receiver
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_data()
            .unwrap()
            .num_rows(),
        2
    );

    // If the cancelled send had leaked a reservation, this send could not
    // complete; if it had double-released, the later drain would underflow.
    let mut resent = Box::pin(sender.send(StreamMessage::data(external_batch(2, 8))));
    assert_ready(&mut resent).await.unwrap();
    drop(resent);
    assert_eq!(
        receiver
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_data()
            .unwrap()
            .num_rows(),
        2
    );

    // The queue is now fully drained and the whole budget is free again,
    // proven by an immediately-ready full-budget send.
    let mut third = Box::pin(sender.send(StreamMessage::data(external_batch(2, 8))));
    assert_ready(&mut third).await.unwrap();
}

#[tokio::test]
async fn receiver_close_wakes_the_blocked_sender_with_edge_closed() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 1,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    sender
        .send(StreamMessage::data(external_batch(1, 8)))
        .await
        .unwrap();
    let mut blocked = Box::pin(sender.send(StreamMessage::data(external_batch(1, 8))));
    assert_pending(&mut blocked).await;

    receiver.close();
    let error = assert_ready(&mut blocked).await.unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::EdgeClosed { ref edge } if edge == "source.out->node.in"
    ));
}

#[tokio::test]
async fn send_after_receiver_close_fails_immediately() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 8,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    receiver.close();
    let error = sender
        .send(StreamMessage::data(external_batch(1, 8)))
        .await
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::EdgeClosed { .. }));
}

#[tokio::test]
async fn closed_receiver_still_drains_queued_messages_then_returns_none() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 8,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    sender
        .send(StreamMessage::data(external_batch(3, 8)))
        .await
        .unwrap();
    receiver.close();

    assert_eq!(
        receiver
            .recv()
            .await
            .unwrap()
            .unwrap()
            .as_data()
            .unwrap()
            .num_rows(),
        3
    );
    assert!(receiver.recv().await.unwrap().is_none());
}

#[tokio::test]
async fn dropping_the_receiver_wakes_the_blocked_sender_with_edge_closed() {
    let (mut sender, receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 1,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    sender
        .send(StreamMessage::data(external_batch(1, 8)))
        .await
        .unwrap();
    let mut blocked = Box::pin(sender.send(StreamMessage::data(external_batch(1, 8))));
    assert_pending(&mut blocked).await;

    drop(receiver);
    let error = assert_ready(&mut blocked).await.unwrap_err();
    assert!(matches!(error, CalcFlowError::EdgeClosed { .. }));
}

#[tokio::test]
async fn recv_waits_while_the_sender_is_alive_and_returns_none_after_its_drop() {
    let (sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 8,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    {
        let mut waiting = Box::pin(receiver.recv());
        assert_pending(&mut waiting).await;
    }

    drop(sender);
    assert!(receiver.recv().await.unwrap().is_none());
}

#[tokio::test]
async fn metrics_report_queue_depth_charges_and_high_water_marks() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 16,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    assert_eq!(sender.metrics(), ChannelMetrics::default());

    sender
        .send(StreamMessage::data(external_batch(2, 20)))
        .await
        .unwrap();
    sender
        .send(StreamMessage::data(external_batch(3, 30)))
        .await
        .unwrap();

    let metrics = receiver.metrics();
    assert_eq!(metrics.queue_depth, 2);
    assert_eq!(metrics.charged_rows, 5);
    assert_eq!(metrics.charged_bytes, 50);
    assert_eq!(metrics.high_water_depth, 2);
    assert_eq!(metrics.high_water_rows, 5);
    assert_eq!(metrics.high_water_bytes, 50);

    receiver.recv().await.unwrap();
    let metrics = sender.metrics();
    assert_eq!(metrics.queue_depth, 1);
    assert_eq!(metrics.charged_rows, 3);
    assert_eq!(metrics.charged_bytes, 30);
    // High-water marks never regress.
    assert_eq!(metrics.high_water_depth, 2);
    assert_eq!(metrics.high_water_rows, 5);
    assert_eq!(metrics.high_water_bytes, 50);
}

#[tokio::test]
async fn metrics_survive_repeated_fill_and_drain_cycles() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 4,
            max_bytes: 64,
        },
    )
    .unwrap();

    for _ in 0..3 {
        sender
            .send(StreamMessage::data(external_batch(2, 32)))
            .await
            .unwrap();
        sender
            .send(StreamMessage::data(external_batch(2, 32)))
            .await
            .unwrap();

        let metrics = sender.metrics();
        assert_eq!(metrics.queue_depth, 2);
        assert_eq!(metrics.charged_rows, 4);
        assert_eq!(metrics.charged_bytes, 64);

        receiver.recv().await.unwrap().unwrap();
        receiver.recv().await.unwrap().unwrap();

        // A full drain releases every reservation exactly once, so the next
        // cycle can refill the whole budget; the peak high-water marks
        // survive the empty queue and never regress (NFR-6).
        let metrics = receiver.metrics();
        assert_eq!(metrics.queue_depth, 0);
        assert_eq!(metrics.charged_rows, 0);
        assert_eq!(metrics.charged_bytes, 0);
        assert_eq!(metrics.high_water_depth, 2);
        assert_eq!(metrics.high_water_rows, 4);
        assert_eq!(metrics.high_water_bytes, 64);
    }
}

#[tokio::test(start_paused = true)]
async fn metrics_count_blocked_sends_and_their_blocked_duration() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 1,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    sender
        .send(StreamMessage::data(external_batch(1, 8)))
        .await
        .unwrap();
    let mut blocked = Box::pin(sender.send(StreamMessage::data(external_batch(1, 8))));
    assert_pending(&mut blocked).await;

    // The paused clock makes the wait exactly one second.
    tokio::time::advance(Duration::from_secs(1)).await;
    receiver.recv().await.unwrap();
    assert_ready(&mut blocked).await.unwrap();
    drop(blocked);

    let metrics = sender.metrics();
    assert_eq!(metrics.blocked_sends, 1);
    assert_eq!(metrics.blocked_duration, Duration::from_secs(1));

    // A send that never blocked adds neither count nor duration.
    receiver.recv().await.unwrap();
    sender
        .send(StreamMessage::data(external_batch(1, 8)))
        .await
        .unwrap();
    let metrics = receiver.metrics();
    assert_eq!(metrics.blocked_sends, 1);
    assert_eq!(metrics.blocked_duration, Duration::from_secs(1));
}

#[tokio::test]
async fn fan_out_edges_charge_independently_while_sharing_the_payload() {
    let batch = external_batch(4, 40);
    let payload = Arc::clone(batch.external_payload().unwrap());
    let message = StreamMessage::data(batch);
    let budget = EdgeBudget {
        max_rows: 16,
        max_bytes: 1_024,
    };

    let (mut sender_a, mut receiver_a) = edge_channel("root.out->a.in", budget).unwrap();
    let (mut sender_b, mut receiver_b) = edge_channel("root.out->b.in", budget).unwrap();

    sender_a.send(message.clone()).await.unwrap();
    sender_b.send(message).await.unwrap();

    // Each edge charges its own queue for the same shared payload (S3).
    assert_eq!(sender_a.metrics().charged_bytes, 40);
    assert_eq!(sender_b.metrics().charged_bytes, 40);

    // Draining one edge leaves the sibling's charge untouched.
    let drained_a = receiver_a.recv().await.unwrap().unwrap();
    assert_eq!(sender_a.metrics().charged_bytes, 0);
    assert_eq!(sender_a.metrics().queue_depth, 0);
    assert_eq!(sender_b.metrics().charged_bytes, 40);
    assert_eq!(sender_b.metrics().queue_depth, 1);

    let drained_b = receiver_b.recv().await.unwrap().unwrap();
    assert!(Arc::ptr_eq(
        drained_a.as_data().unwrap().external_payload().unwrap(),
        &payload
    ));
    assert!(Arc::ptr_eq(
        drained_b.as_data().unwrap().external_payload().unwrap(),
        &payload
    ));
}

#[tokio::test]
async fn table_batches_charge_their_arrow_slice_estimate() {
    let (mut sender, mut receiver) = edge_channel(
        "source.out->node.in",
        EdgeBudget {
            max_rows: 100,
            max_bytes: 1_024,
        },
    )
    .unwrap();

    let batch = table_batch(8);
    let expected = batch.estimated_bytes().unwrap();
    sender.send(StreamMessage::data(batch)).await.unwrap();

    let metrics = receiver.metrics();
    assert_eq!(metrics.charged_rows, 8);
    assert_eq!(metrics.charged_bytes, expected);
    assert!(metrics.charged_bytes > 0);
    receiver.recv().await.unwrap();
    assert_eq!(sender.metrics().charged_bytes, 0);
}
