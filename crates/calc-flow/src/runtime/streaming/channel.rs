//! The envelope + rows + bytes bounded channel carried by every stream edge
//! (spec S10, plan task M1.4).
//!
//! One edge owns exactly one channel with exactly one producer and exactly
//! one consumer (invariant I10): [`EdgeSender`] and [`EdgeReceiver`] are
//! deliberately not `Clone`, and both `send` and `recv` take `&mut self`, so
//! the single-producer/single-consumer contract is enforced at compile time.
//! That invariant is what makes the coordination design sound: a single
//! mutex-protected budget plus a [`tokio::sync::Notify`] wakeup performs the
//! atomic three-dimension reservation, and with at most one waiting sender a
//! release notification can never be consumed by a waiter that cannot make
//! progress (the multi-producer lost-wakeup mode the plan calls out).
//!
//! Reservation and release follow S10.1: a sender reserves all three dimensions
//! atomically with the enqueue, inside one critical section, so a blocked
//! send holds no reservation and dropping the send future leaves the budget
//! untouched; the consumer's reservation is released exactly once, when the
//! receiver dequeues the message or when the queued remainder is dropped
//! with the receiver. Closing the receiver wakes a blocked sender with
//! [`CalcFlowError::EdgeClosed`].

use std::{collections::VecDeque, fmt, sync::Arc, time::Duration};

use parking_lot::Mutex;
use tokio::sync::Notify;

use super::{
    StreamMessage,
    metrics::{EdgeTraffic, MetricsRecorder, MetricsTimer},
};
use crate::{CalcFlowError, EdgeBudget, Result, batch::checked_accumulate};

/// The logical queue charge of one edge message (spec S10.2).
///
/// Data messages charge one message, the batch row count, and the batch
/// [`Batch::estimated_bytes`](crate::Batch::estimated_bytes) estimate;
/// control messages (watermark, barrier, idle, end-of-input) charge one
/// message and zero rows/bytes. Every envelope therefore consumes one finite
/// slot even when its row and byte cost is zero. Charges are
/// logical reservations, not process RSS measurements; fan-out edges each
/// charge their own channel even though the Arrow buffers are shared (S3).
///
/// All arithmetic is checked: overflowing sums are typed errors (S10.2).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EnvelopeCost {
    messages: usize,
    rows: usize,
    bytes: usize,
}

impl EnvelopeCost {
    /// The additive identity used for an empty queue.
    pub const ZERO: Self = Self {
        messages: 0,
        rows: 0,
        bytes: 0,
    };

    /// Builds a cost from explicit components, for example in tests.
    pub const fn new(messages: usize, rows: usize, bytes: usize) -> Self {
        Self {
            messages,
            rows,
            bytes,
        }
    }

    /// Computes the charge of one message before it enters the queue.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when a table batch cannot
    /// be measured (see [`Batch::estimated_bytes`](crate::Batch::estimated_bytes)).
    pub fn of_message(message: &StreamMessage) -> Result<Self> {
        match message.as_data() {
            Some(batch) => Ok(Self {
                messages: 1,
                rows: batch.num_rows(),
                bytes: batch.estimated_bytes()?,
            }),
            None => Ok(Self {
                messages: 1,
                ..Self::ZERO
            }),
        }
    }

    pub const fn messages(&self) -> usize {
        self.messages
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// Adds two costs component-wise.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when a component sum
    /// overflows `usize` (S10.2).
    pub fn checked_add(&self, other: &Self) -> Result<Self> {
        Ok(Self {
            messages: checked_accumulate(self.messages, other.messages, "envelope_cost.messages")?,
            rows: checked_accumulate(self.rows, other.rows, "envelope_cost.rows")?,
            bytes: checked_accumulate(self.bytes, other.bytes, "envelope_cost.bytes")?,
        })
    }

    /// Releases `other` from `self` component-wise.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Internal`] when a component would underflow;
    /// releasing more than was charged violates the reservation accounting
    /// invariant and is never a caller-facing condition.
    pub fn checked_sub(&self, other: &Self) -> Result<Self> {
        let release = |charged: usize, released: usize, component: &str| {
            charged.checked_sub(released).ok_or_else(|| CalcFlowError::Internal {
                message: format!(
                    "envelope cost accounting underflowed on {component}: released {released} with only {charged} charged"
                ),
            })
        };
        Ok(Self {
            messages: release(self.messages, other.messages, "messages")?,
            rows: release(self.rows, other.rows, "rows")?,
            bytes: release(self.bytes, other.bytes, "bytes")?,
        })
    }

    /// Component-wise maximum, used for the queue high-water marks (NFR-6).
    pub(crate) fn max_components(&self, other: &Self) -> Self {
        Self {
            messages: self.messages.max(other.messages),
            rows: self.rows.max(other.rows),
            bytes: self.bytes.max(other.bytes),
        }
    }
}

#[derive(Default)]
struct ChannelState {
    queue: VecDeque<(StreamMessage, EnvelopeCost)>,
    charged: EnvelopeCost,
    receiver_closed: bool,
    sender_closed: bool,
    high_water: EnvelopeCost,
    blocked_sends: u64,
    blocked_duration: Duration,
}

struct Shared {
    edge: String,
    budget: EdgeBudget,
    metrics: MetricsRecorder,
    state: Mutex<ChannelState>,
    /// Signalled when reserved capacity is released or the receiver closes.
    capacity_available: Notify,
    /// Signalled when a message is enqueued or the sender closes.
    message_available: Notify,
}

/// A point-in-time snapshot of one edge channel's occupancy and
/// backpressure counters (spec NFR-6; the M2.5 metrics surface aggregates
/// these per edge).
///
/// `queue_depth`, `charged_rows`, and `charged_bytes` are the current
/// reservations; the `high_water_*` fields are their monotone maxima and
/// never regress. `blocked_sends` counts sends that had to await capacity at
/// least once, and `blocked_duration` accumulates the time those sends spent
/// waiting. A send that is cancelled, or that is woken by receiver
/// close/drop, while it waits still increments `blocked_sends` but
/// contributes nothing to `blocked_duration`, which only accumulates
/// when a blocked send eventually enqueues. Snapshots are consistent: every
/// field is read under one lock.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChannelMetrics {
    pub queue_depth: usize,
    pub charged_rows: usize,
    pub charged_bytes: usize,
    pub high_water_depth: usize,
    pub high_water_rows: usize,
    pub high_water_bytes: usize,
    pub blocked_sends: u64,
    pub blocked_duration: Duration,
}

impl ChannelState {
    fn metrics(&self) -> ChannelMetrics {
        ChannelMetrics {
            queue_depth: self.charged.messages(),
            charged_rows: self.charged.rows(),
            charged_bytes: self.charged.bytes(),
            high_water_depth: self.high_water.messages(),
            high_water_rows: self.high_water.rows(),
            high_water_bytes: self.high_water.bytes(),
            blocked_sends: self.blocked_sends,
            blocked_duration: self.blocked_duration,
        }
    }
}

impl fmt::Debug for Shared {
    /// Diagnostics show the edge identity and budget only; queued payloads
    /// never appear (invariant I4).
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Shared")
            .field("edge", &self.edge)
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

/// Creates the bounded channel for one edge.
///
/// `edge` is the stable edge identifier assigned at compile time (plan
/// M1.1); it names the edge in errors and metrics. `budget.max_rows`
/// independently caps queued envelopes and charged rows, while
/// `budget.max_bytes` caps charged bytes (spec S10.1). Both fields must be
/// positive.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] when `edge` is empty or either
/// budget limit is zero.
///
/// # Examples
///
/// ```
/// use calc_flow::{EdgeBudget, edge_channel};
///
/// let (_sender, _receiver) = edge_channel(
///     "source.out->node.in",
///     EdgeBudget { max_rows: 10_000, max_bytes: 64 << 20 },
/// )?;
/// # Ok::<(), calc_flow::CalcFlowError>(())
/// ```
pub fn edge_channel(
    edge: impl Into<String>,
    budget: EdgeBudget,
) -> Result<(EdgeSender, EdgeReceiver)> {
    edge_channel_with_metrics(edge, budget, MetricsRecorder::default())
}

pub(crate) fn edge_channel_with_metrics(
    edge: impl Into<String>,
    budget: EdgeBudget,
    metrics: MetricsRecorder,
) -> Result<(EdgeSender, EdgeReceiver)> {
    let edge = edge.into();
    if edge.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: "edge".into(),
            message: "must not be empty".into(),
        });
    }
    EdgeBudget::new(budget.max_rows, budget.max_bytes)?;
    let shared = Arc::new(Shared {
        edge,
        budget,
        metrics,
        state: Mutex::new(ChannelState::default()),
        capacity_available: Notify::new(),
        message_available: Notify::new(),
    });
    Ok((
        EdgeSender {
            shared: Arc::clone(&shared),
        },
        EdgeReceiver { shared },
    ))
}

/// The single producing handle of one edge channel.
///
/// Not `Clone` on purpose: every edge has exactly one producer (I10), and
/// `send` takes `&mut self`, so at most one send is ever in flight. That is
/// the precondition the budget-plus-notify coordination relies on.
pub struct EdgeSender {
    shared: Arc<Shared>,
}

impl fmt::Debug for EdgeSender {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EdgeSender")
            .field("edge", &self.shared.edge)
            .finish_non_exhaustive()
    }
}

/// The single consuming handle of one edge channel.
///
/// Not `Clone` on purpose: every edge has exactly one consumer (I10), and
/// `recv` takes `&mut self`. Dropping the receiver closes the edge for the
/// sender (a blocked send wakes with [`CalcFlowError::EdgeClosed`]) and
/// releases every queued reservation.
pub struct EdgeReceiver {
    shared: Arc<Shared>,
}

impl fmt::Debug for EdgeReceiver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EdgeReceiver")
            .field("edge", &self.shared.edge)
            .finish_non_exhaustive()
    }
}

/// Returns whether `cost` can be reserved on top of `charged` without
/// crossing any budget limit. `max_rows` independently caps both queued
/// envelopes and charged rows. A component sum that overflows `usize`
/// necessarily exceeds the budget, so it simply does not fit.
fn fits(charged: &EnvelopeCost, cost: &EnvelopeCost, budget: &EdgeBudget) -> bool {
    let Some(messages) = charged.messages().checked_add(cost.messages()) else {
        return false;
    };
    let Some(rows) = charged.rows().checked_add(cost.rows()) else {
        return false;
    };
    let Some(bytes) = charged.bytes().checked_add(cost.bytes()) else {
        return false;
    };
    messages <= budget.max_rows && rows <= budget.max_rows && bytes <= budget.max_bytes
}

impl EdgeSender {
    /// Validates that one message can fit this edge without enqueueing it.
    ///
    /// The operator collector applies this to every fan-out branch before
    /// its first send, which keeps validation failures side-effect free.
    pub(crate) fn validate_message(&self, message: &StreamMessage) -> Result<()> {
        let cost = EnvelopeCost::of_message(message)?;
        self.shared.reject_oversize(&cost)
    }

    /// Enqueues one message, awaiting capacity when any hard dimension is
    /// reached (the `Block` policy, spec S10.4).
    ///
    /// The three-dimensional reservation is atomic with the enqueue: a blocked
    /// send holds no reservation, so dropping this future while it waits
    /// leaves the budget exactly as if the send had never started (S10.1's
    /// cancelled-send rule). A single message larger than either row/byte
    /// limit fails before any wait; there is no oversize exception (S10.3).
    /// Every control or data envelope consumes one slot, including a
    /// zero-row/zero-byte data batch, and keeps its FIFO position (S1.1).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the message's cost
    /// cannot be measured or exceeds a budget limit by itself, and
    /// [`CalcFlowError::EdgeClosed`] when the receiver is closed or dropped.
    pub async fn send(&mut self, message: StreamMessage) -> Result<()> {
        let cost = EnvelopeCost::of_message(&message)?;
        self.shared.reject_oversize(&cost)?;
        let notified = self.shared.capacity_available.notified();
        tokio::pin!(notified);
        let mut blocked_since: Option<tokio::time::Instant> = None;
        let mut metrics_blocked_since: Option<MetricsTimer> = None;
        loop {
            // Register for the wakeup before re-checking the budget so a
            // release between the check and the await cannot be lost.
            notified.as_mut().enable();
            {
                let mut state = self.shared.state.lock();
                if state.receiver_closed {
                    return Err(CalcFlowError::EdgeClosed {
                        edge: self.shared.edge.clone(),
                    });
                }
                if fits(&state.charged, &cost, &self.shared.budget) {
                    let blocked_elapsed = blocked_since.map(|started| started.elapsed());
                    let metrics_blocked_elapsed = metrics_blocked_since
                        .as_ref()
                        .map(|timer| timer.elapsed(&self.shared.edge, "blocked_duration"))
                        .transpose()?;
                    self.shared.metrics.record_edge_enqueue(
                        &self.shared.edge,
                        EdgeTraffic::of_message(&message, cost)?,
                        metrics_blocked_elapsed,
                    )?;
                    state.charged = state.charged.checked_add(&cost).map_err(|error| {
                        // `fits` rejected every component sum that could
                        // overflow, so reaching this branch violates the
                        // channel's locked admission invariant.
                        CalcFlowError::Internal {
                            message: format!(
                                "edge {:?} charge overflowed after a successful capacity check: {error}",
                                self.shared.edge
                            ),
                        }
                    })?;
                    state.high_water = state.high_water.max_components(&state.charged);
                    state.queue.push_back((message, cost));
                    if let Some(elapsed) = blocked_elapsed {
                        state.blocked_duration = state
                            .blocked_duration
                            .checked_add(elapsed)
                            .ok_or_else(|| CalcFlowError::InvalidArgument {
                                field: format!(
                                    "runtime.metrics.{}.blocked_duration",
                                    self.shared.edge
                                ),
                                message: "counter overflow".into(),
                            })?;
                    }
                    drop(state);
                    self.shared.message_available.notify_one();
                    return Ok(());
                }
                if blocked_since.is_none() {
                    let next_blocked = state.blocked_sends.checked_add(1).ok_or_else(|| {
                        CalcFlowError::InvalidArgument {
                            field: format!("runtime.metrics.{}.blocked_sends", self.shared.edge),
                            message: "counter overflow".into(),
                        }
                    })?;
                    self.shared.metrics.record_edge_blocked(&self.shared.edge)?;
                    state.blocked_sends = next_blocked;
                    blocked_since = Some(tokio::time::Instant::now());
                    metrics_blocked_since = Some(self.shared.metrics.timer());
                }
            }
            // Lost-wakeup safety at this await rests on the type-level
            // single-producer invariant I10 (see the module doc): with at
            // most one waiting sender, a release notification can never be
            // consumed by a waiter that cannot make progress.
            notified.as_mut().await;
            // A woken send hands the wakeup on before re-checking the
            // budget. Under I10 this is a no-op: the sender is the only
            // task that ever waits on `capacity_available`, its consumed
            // `Notified` is no longer registered at this point, and
            // `notify_waiters` — unlike `notify_one` — stores no permit
            // for a later waiter. The call is insurance for a relaxed
            // invariant, not load-bearing today: if the channel ever
            // allowed multiple producers, a release consumed by a woken
            // send that still does not fit could otherwise strand a parked
            // peer whose message does fit the freed capacity. A genuine
            // multi-producer design would still revisit the wakeup
            // discipline; I10 is what keeps this coordination sound.
            self.shared.capacity_available.notify_waiters();
            notified.set(self.shared.capacity_available.notified());
        }
    }

    /// Returns the stable identifier of the edge this sender produces into.
    pub fn edge(&self) -> &str {
        &self.shared.edge
    }

    /// Returns a consistent snapshot of the channel's occupancy and
    /// backpressure counters.
    pub fn metrics(&self) -> ChannelMetrics {
        self.shared.state.lock().metrics()
    }

    pub(crate) fn budget(&self) -> EdgeBudget {
        self.shared.budget
    }
}

impl EdgeReceiver {
    /// Dequeues the next message in FIFO order, awaiting one while the queue
    /// is empty.
    ///
    /// Dequeueing releases the message's reservation exactly once (S10.1) and
    /// wakes a blocked sender. Returns `None` once the queue is drained and
    /// either the sender was dropped or the receiver was closed; messages
    /// enqueued before a close remain receivable (drain semantics).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Internal`] when the reservation accounting
    /// underflows, which is unreachable by construction: every dequeue
    /// releases exactly the cost reserved at enqueue.
    pub async fn recv(&mut self) -> Result<Option<StreamMessage>> {
        let notified = self.shared.message_available.notified();
        tokio::pin!(notified);
        loop {
            notified.as_mut().enable();
            {
                let mut state = self.shared.state.lock();
                if let Some((message, cost)) = state.queue.front() {
                    self.shared.metrics.record_edge_dequeue(
                        &self.shared.edge,
                        EdgeTraffic::of_message(message, *cost)?,
                    )?;
                    let (message, cost) =
                        state
                            .queue
                            .pop_front()
                            .ok_or_else(|| CalcFlowError::Internal {
                                message: format!(
                                    "edge {:?} queue front disappeared while its lock was held",
                                    self.shared.edge
                                ),
                            })?;
                    state.charged = state.charged.checked_sub(&cost)?;
                    drop(state);
                    self.shared.capacity_available.notify_one();
                    return Ok(Some(message));
                }
                if state.receiver_closed || state.sender_closed {
                    return Ok(None);
                }
            }
            notified.as_mut().await;
            notified.set(self.shared.message_available.notified());
        }
    }

    /// Returns the stable identifier of the edge this receiver consumes.
    pub fn edge(&self) -> &str {
        &self.shared.edge
    }

    /// Returns a consistent snapshot of the channel's occupancy and
    /// backpressure counters.
    pub fn metrics(&self) -> ChannelMetrics {
        self.shared.state.lock().metrics()
    }

    /// Closes the receiving side of the edge.
    ///
    /// A blocked or later send wakes/fails with [`CalcFlowError::EdgeClosed`]
    /// (S10.1). Messages already enqueued remain receivable; once the queue
    /// is drained, [`EdgeReceiver::recv`] returns `None`.
    pub fn close(&mut self) {
        let mut state = self.shared.state.lock();
        state.receiver_closed = true;
        drop(state);
        self.shared.capacity_available.notify_waiters();
    }
}

impl Drop for EdgeReceiver {
    /// Dropping the receiver closes the edge and releases every queued
    /// reservation, so a blocked sender always wakes with
    /// [`CalcFlowError::EdgeClosed`] and payload `Arc`s are released promptly
    /// instead of waiting for the sender to drop.
    fn drop(&mut self) {
        let mut state = self.shared.state.lock();
        state.receiver_closed = true;
        self.shared
            .metrics
            .record_edge_drop(&self.shared.edge, state.charged);
        state.queue.clear();
        state.charged = EnvelopeCost::ZERO;
        drop(state);
        self.shared.capacity_available.notify_waiters();
    }
}

impl Drop for EdgeSender {
    /// Dropping the sender lets a waiting receiver observe `None` once the
    /// queue is drained; stream termination itself is the explicit
    /// end-of-input message (S1.6), never channel teardown.
    fn drop(&mut self) {
        let mut state = self.shared.state.lock();
        state.sender_closed = true;
        drop(state);
        self.shared.message_available.notify_waiters();
    }
}

impl Shared {
    /// Rejects a message that can never fit within the budget, before any
    /// wait (S10.3).
    fn reject_oversize(&self, cost: &EnvelopeCost) -> Result<()> {
        if cost.rows() > self.budget.max_rows {
            return Err(CalcFlowError::InvalidArgument {
                field: "message.rows".into(),
                message: format!(
                    "{} exceeds edge {:?} row budget {}",
                    cost.rows(),
                    self.edge,
                    self.budget.max_rows
                ),
            });
        }
        if cost.bytes() > self.budget.max_bytes {
            return Err(CalcFlowError::InvalidArgument {
                field: "message.bytes".into(),
                message: format!(
                    "{} exceeds edge {:?} byte budget {}",
                    cost.bytes(),
                    self.edge,
                    self.budget.max_bytes
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, task::Poll};

    use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

    use super::*;
    use crate::{Batch, BatchMetadata, Epoch, EventTime, StreamMessageKind};

    fn data_message(values: &[i64]) -> StreamMessage {
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(values.to_vec())) as _,
        )])
        .unwrap();
        StreamMessage::data(Batch::table(vec![record], BatchMetadata::default()).unwrap())
    }

    #[tokio::test]
    async fn mixed_data_and_control_messages_keep_one_fifo_order() {
        let (mut sender, mut receiver) = edge_channel(
            "source.out->node.in",
            EdgeBudget {
                max_rows: 100,
                max_bytes: 1 << 20,
            },
        )
        .unwrap();

        sender.send(data_message(&[1])).await.unwrap();
        sender
            .send(StreamMessage::watermark(EventTime::from_micros(7)))
            .await
            .unwrap();
        sender.send(data_message(&[2])).await.unwrap();
        sender
            .send(StreamMessage::barrier(Epoch::INITIAL))
            .await
            .unwrap();
        sender.send(StreamMessage::idle()).await.unwrap();
        sender.send(StreamMessage::end_of_input()).await.unwrap();
        drop(sender);

        let mut kinds = Vec::new();
        while let Some(message) = receiver.recv().await.unwrap() {
            kinds.push(message.kind());
        }
        assert_eq!(
            kinds,
            [
                StreamMessageKind::Data,
                StreamMessageKind::Watermark,
                StreamMessageKind::Data,
                StreamMessageKind::Barrier,
                StreamMessageKind::Idle,
                StreamMessageKind::EndOfInput,
            ]
        );
    }

    #[tokio::test]
    async fn zero_cost_messages_block_at_the_slot_limit_and_resume_fifo() {
        const SLOT_LIMIT: usize = 3;
        const MESSAGE_COUNT: usize = 30;

        let message_factories: [fn(usize) -> StreamMessage; 3] = [
            |_| StreamMessage::idle(),
            |index| {
                StreamMessage::watermark(EventTime::from_micros(
                    i64::try_from(index).expect("the bounded test index fits i64"),
                ))
            },
            |_| data_message(&[]),
        ];

        for make_message in message_factories {
            let assert_message = |message: StreamMessage, index| {
                let expected = make_message(index);
                assert_eq!(message.kind(), expected.kind());
                assert_eq!(message.as_watermark(), expected.as_watermark());
            };
            let (mut sender, mut receiver) = edge_channel(
                "source.out->node.in",
                EdgeBudget {
                    max_rows: SLOT_LIMIT,
                    max_bytes: 1 << 20,
                },
            )
            .unwrap();

            for index in 0..SLOT_LIMIT {
                let message = make_message(index);
                let cost = EnvelopeCost::of_message(&message).unwrap();
                assert_eq!((cost.messages(), cost.rows(), cost.bytes()), (1, 0, 0));
                sender.send(message).await.unwrap();
            }

            for index in SLOT_LIMIT..MESSAGE_COUNT {
                let mut blocked = Box::pin(sender.send(make_message(index)));
                assert!(matches!(futures::poll!(blocked.as_mut()), Poll::Pending));

                let metrics = receiver.metrics();
                assert_eq!(metrics.queue_depth, SLOT_LIMIT);
                assert_eq!(metrics.charged_rows, 0);
                assert_eq!(metrics.charged_bytes, 0);
                assert_eq!(metrics.high_water_depth, SLOT_LIMIT);
                assert_eq!(metrics.high_water_rows, 0);
                assert_eq!(metrics.high_water_bytes, 0);
                assert_eq!(metrics.blocked_sends, (index + 1 - SLOT_LIMIT) as u64);

                let message = receiver.recv().await.unwrap().unwrap();
                assert_message(message, index - SLOT_LIMIT);
                assert!(matches!(
                    futures::poll!(blocked.as_mut()),
                    Poll::Ready(Ok(()))
                ));
                assert_eq!(receiver.metrics().queue_depth, SLOT_LIMIT);
            }

            for index in MESSAGE_COUNT - SLOT_LIMIT..MESSAGE_COUNT {
                let message = receiver.recv().await.unwrap().unwrap();
                assert_message(message, index);
            }
            let metrics = receiver.metrics();
            assert_eq!(metrics.queue_depth, 0);
            assert_eq!(metrics.charged_rows, 0);
            assert_eq!(metrics.charged_bytes, 0);
            assert_eq!(metrics.high_water_depth, SLOT_LIMIT);
            assert_eq!(metrics.blocked_sends, (MESSAGE_COUNT - SLOT_LIMIT) as u64);
        }
    }

    #[test]
    fn handle_debug_shows_the_edge_identity_without_payload() {
        let (sender, receiver) = edge_channel(
            "source.out->node.in",
            EdgeBudget {
                max_rows: 4,
                max_bytes: 64,
            },
        )
        .unwrap();
        let sender_debug = format!("{sender:?}");
        let receiver_debug = format!("{receiver:?}");
        assert!(sender_debug.contains("source.out->node.in"));
        assert!(receiver_debug.contains("source.out->node.in"));
        assert_eq!(sender.edge(), "source.out->node.in");
        assert_eq!(receiver.edge(), "source.out->node.in");
    }

    #[test]
    fn control_messages_cost_one_message_and_zero_rows_and_bytes() {
        for message in [
            StreamMessage::watermark(EventTime::from_micros(1)),
            StreamMessage::barrier(Epoch::INITIAL),
            StreamMessage::idle(),
            StreamMessage::end_of_input(),
        ] {
            let cost = EnvelopeCost::of_message(&message).unwrap();
            assert_eq!(cost.messages(), 1);
            assert_eq!(cost.rows(), 0);
            assert_eq!(cost.bytes(), 0);
        }
    }

    #[tokio::test]
    async fn runtime_metrics_linearize_at_queue_commit_and_dequeue_release() {
        let recorder = MetricsRecorder::new(
            [("edge".into(), EdgeBudget::new(4, 64).unwrap())],
            [],
            [],
            [],
        );
        let (mut sender, mut receiver) = edge_channel_with_metrics(
            "edge",
            EdgeBudget {
                max_rows: 4,
                max_bytes: 64,
            },
            recorder.clone(),
        )
        .unwrap();

        sender.send(data_message(&[1, 2])).await.unwrap();
        let enqueued = recorder.snapshot().edges.remove("edge").unwrap();
        assert_eq!(
            (
                enqueued.input_batches,
                enqueued.input_rows,
                enqueued.input_bytes,
                enqueued.output_batches,
            ),
            (1, 2, 16, 0)
        );
        assert_eq!(
            (
                enqueued.channel.queue_depth,
                enqueued.channel.charged_rows,
                enqueued.channel.high_water_rows,
            ),
            (1, 2, 2)
        );

        receiver.recv().await.unwrap().unwrap();
        let dequeued = recorder.snapshot().edges.remove("edge").unwrap();
        assert_eq!(
            (
                dequeued.output_batches,
                dequeued.output_rows,
                dequeued.output_bytes,
            ),
            (1, 2, 16)
        );
        assert_eq!(
            (
                dequeued.channel.queue_depth,
                dequeued.channel.charged_rows,
                dequeued.channel.charged_bytes,
            ),
            (0, 0, 0)
        );
    }

    #[tokio::test]
    async fn receiver_drop_surfaces_a_metrics_release_invariant_violation() {
        let recorder = MetricsRecorder::new(
            [("edge".into(), EdgeBudget::new(1, 8).unwrap())],
            [],
            [],
            [],
        );
        let (mut sender, receiver) = edge_channel_with_metrics(
            "edge",
            EdgeBudget {
                max_rows: 1,
                max_bytes: 8,
            },
            recorder.clone(),
        )
        .unwrap();
        let message = data_message(&[1]);
        let cost = EnvelopeCost::of_message(&message).unwrap();
        sender.send(message).await.unwrap();

        recorder.record_edge_drop("edge", cost);
        drop(receiver);

        let edge = &recorder.snapshot().edges["edge"];
        assert_eq!(edge.channel.queue_depth, 0);
        assert_eq!(edge.channel.charged_rows, 0);
        assert_eq!(edge.channel.charged_bytes, 0);
        assert!(edge.drop_invariant_violated);
    }
}
