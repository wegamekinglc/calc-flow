//! The rows + bytes dual-limit bounded channel carried by every stream edge
//! (spec S10, plan task M1.4).
//!
//! One edge owns exactly one channel with exactly one producer and exactly
//! one consumer (invariant I10): [`EdgeSender`] and [`EdgeReceiver`] are
//! deliberately not `Clone`, and both `send` and `recv` take `&mut self`, so
//! the single-producer/single-consumer contract is enforced at compile time.
//! That invariant is what makes the coordination design sound: a single
//! mutex-protected budget plus a [`tokio::sync::Notify`] wakeup performs the
//! atomic two-dimension reservation, and with at most one waiting sender a
//! release notification can never be consumed by a waiter that cannot make
//! progress (the multi-producer lost-wakeup mode the plan calls out).
//!
//! Reservation and release follow S10.1: a sender reserves both dimensions
//! atomically with the enqueue, inside one critical section, so a blocked
//! send holds no reservation and dropping the send future leaves the budget
//! untouched; the consumer's reservation is released exactly once, when the
//! receiver dequeues the message or when the queued remainder is dropped
//! with the receiver. Closing the receiver wakes a blocked sender with
//! [`CalcFlowError::EdgeClosed`].

use std::{collections::VecDeque, fmt, sync::Arc, time::Duration};

use parking_lot::Mutex;
use tokio::sync::Notify;

use super::StreamMessage;
use crate::{CalcFlowError, EdgeBudget, Result, batch::checked_accumulate};

/// The logical queue charge of one edge message (spec S10.2).
///
/// Data messages charge one message, the batch row count, and the batch
/// [`Batch::estimated_bytes`](crate::Batch::estimated_bytes) estimate;
/// control messages (watermark, barrier, idle, end-of-input) charge one
/// message and zero rows/bytes, so control flow is never throttled by data
/// backpressure while per-edge FIFO order is preserved (S1.1). Charges are
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
/// waiting. Snapshots are consistent: every field is read under one lock.
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
/// M1.1); it names the edge in errors and metrics. `budget` carries the two
/// hard limits (spec S10.1); both must be positive.
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
/// crossing either budget limit. A component sum that overflows `usize`
/// necessarily exceeds the budget, so it simply does not fit.
fn fits(charged: &EnvelopeCost, cost: &EnvelopeCost, budget: &EdgeBudget) -> bool {
    let Some(rows) = charged.rows().checked_add(cost.rows()) else {
        return false;
    };
    let Some(bytes) = charged.bytes().checked_add(cost.bytes()) else {
        return false;
    };
    rows <= budget.max_rows && bytes <= budget.max_bytes
}

impl EdgeSender {
    /// Enqueues one message, awaiting capacity when either hard limit is
    /// reached (the `Block` policy, spec S10.4).
    ///
    /// The two-dimensional reservation is atomic with the enqueue: a blocked
    /// send holds no reservation, so dropping this future while it waits
    /// leaves the budget exactly as if the send had never started (S10.1's
    /// cancelled-send rule). A single message larger than either limit fails
    /// before any wait; there is no oversize exception (S10.3). Control
    /// messages charge zero rows/bytes and are therefore never throttled by
    /// data backpressure, but they keep their FIFO position behind earlier
    /// data (S1.1).
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
                    state.charged = state.charged.checked_add(&cost).map_err(|error| {
                        // Unreachable: `fits` proved the sum cannot overflow.
                        CalcFlowError::Internal {
                            message: format!(
                                "edge {:?} charge overflowed after a successful capacity check: {error}",
                                self.shared.edge
                            ),
                        }
                    })?;
                    state.high_water = state.high_water.max_components(&state.charged);
                    state.queue.push_back((message, cost));
                    if let Some(since) = blocked_since.take() {
                        state.blocked_duration += since.elapsed();
                    }
                    drop(state);
                    self.shared.message_available.notify_one();
                    return Ok(());
                }
                if blocked_since.is_none() {
                    blocked_since = Some(tokio::time::Instant::now());
                    state.blocked_sends += 1;
                }
            }
            notified.as_mut().await;
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
                if let Some((message, cost)) = state.queue.pop_front() {
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
    use std::sync::Arc;

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
    async fn control_messages_charge_zero_and_are_never_throttled_by_a_full_budget() {
        let (mut sender, mut receiver) = edge_channel(
            "source.out->node.in",
            EdgeBudget {
                max_rows: 2,
                max_bytes: 1 << 20,
            },
        )
        .unwrap();

        sender.send(data_message(&[1])).await.unwrap();
        sender.send(data_message(&[2])).await.unwrap();

        // Rows 2/2: control messages still enqueue immediately, behind the
        // queued data (S1.1); they charge one message and zero rows/bytes.
        sender
            .send(StreamMessage::watermark(EventTime::from_micros(3)))
            .await
            .unwrap();
        sender
            .send(StreamMessage::barrier(Epoch::INITIAL))
            .await
            .unwrap();

        let metrics = receiver.metrics();
        assert_eq!(metrics.queue_depth, 4);
        assert_eq!(metrics.charged_rows, 2);
        assert_eq!(metrics.charged_bytes, 16);

        assert_eq!(
            receiver.recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::Data
        );
        assert_eq!(
            receiver.recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::Data
        );
        assert_eq!(
            receiver.recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::Watermark
        );
        assert_eq!(
            receiver.recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::Barrier
        );
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
}
