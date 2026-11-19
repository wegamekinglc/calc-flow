use std::{
    collections::VecDeque,
    future::poll_fn,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::Poll,
};

use futures::task::AtomicWaker;
use parking_lot::Mutex;

use super::{
    ChannelMetrics, EdgeReceiver, EdgeSender, EnvelopeCost, StreamMessage,
    edge_queue::{Dequeue, EdgeQueueCore, reject_oversize},
    metrics::MetricsRecorder,
};
use crate::{
    CalcFlowError, EdgeBudget, Result,
    operator::rolling_metrics::{RollingMetricsRecorder, RollingStage},
};

struct Shared {
    core: EdgeQueueCore,
    producer: AtomicWaker,
    consumer: AtomicWaker,
}

pub(crate) struct LocalEdgeOwner(Arc<Shared>);
pub(crate) struct LocalEdgeSender(Arc<Shared>);
pub(crate) struct LocalEdgeReceiver(Arc<Shared>);

pub(super) fn local_edge(
    edge: String,
    budget: EdgeBudget,
    metrics: MetricsRecorder,
) -> Result<(LocalEdgeSender, LocalEdgeReceiver, LocalEdgeOwner)> {
    let shared = Arc::new(Shared {
        core: EdgeQueueCore::new(edge, budget, metrics)?,
        producer: AtomicWaker::new(),
        consumer: AtomicWaker::new(),
    });
    Ok((
        LocalEdgeSender(Arc::clone(&shared)),
        LocalEdgeReceiver(Arc::clone(&shared)),
        LocalEdgeOwner(shared),
    ))
}

impl LocalEdgeSender {
    async fn send_observed(
        &mut self,
        message: StreamMessage,
        observation: Option<&RollingMetricsRecorder>,
    ) -> Result<()> {
        let cost = self.0.core.message_cost(&message)?;
        let mut message = Some(message);
        let mut blocked_since = None;
        let mut metrics_blocked_since = None;
        let mut wait = None;
        poll_fn(|context| {
            self.0.producer.register(context.waker());
            match self.0.core.try_enqueue(
                &mut message,
                cost,
                &mut blocked_since,
                &mut metrics_blocked_since,
            ) {
                Ok(true) => {
                    self.0.consumer.wake();
                    Poll::Ready(Ok(()))
                }
                Ok(false) => {
                    if wait.is_none() {
                        wait = observation.map(|recorder| recorder.stage(RollingStage::SendWait));
                    }
                    Poll::Pending
                }
                Err(error) => Poll::Ready(Err(error)),
            }
        })
        .await
    }
}

impl LocalEdgeReceiver {
    async fn recv(&mut self) -> Result<Option<StreamMessage>> {
        poll_fn(|context| {
            self.0.consumer.register(context.waker());
            match self.0.core.try_dequeue() {
                Ok(Dequeue::Message(message)) => {
                    self.0.producer.wake();
                    Poll::Ready(Ok(Some(message)))
                }
                Ok(Dequeue::Closed) => Poll::Ready(Ok(None)),
                Ok(Dequeue::Empty) => Poll::Pending,
                Err(error) => Poll::Ready(Err(error)),
            }
        })
        .await
    }
}

impl Drop for LocalEdgeReceiver {
    fn drop(&mut self) {
        self.0.core.drop_receiver();
        self.0.producer.wake();
    }
}

impl Drop for LocalEdgeSender {
    fn drop(&mut self) {
        self.0.core.close_sender();
        self.0.consumer.wake();
    }
}

impl Drop for LocalEdgeOwner {
    fn drop(&mut self) {
        self.0.core.drop_receiver();
        self.0.core.close_sender();
        self.0.producer.wake();
        self.0.consumer.wake();
        drop(self.0.producer.take());
        drop(self.0.consumer.take());
    }
}

/// One caller-drained FIFO between `StreamSession` operators. It has no
/// capacity backpressure: a session call empties every queue before it returns.
struct SessionQueue {
    edge: String,
    budget: EdgeBudget,
    producer: AtomicUsize,
    queue: Mutex<VecDeque<SessionEnvelope>>,
}

pub(crate) struct SessionEnvelope {
    pub(crate) producer: Option<usize>,
    pub(crate) message: StreamMessage,
}

pub(crate) struct SessionEdgeSender {
    queue: Arc<SessionQueue>,
}
pub(crate) struct SessionEdgeReceiver(Arc<SessionQueue>);

pub(super) fn session_edge(
    edge: String,
    budget: EdgeBudget,
) -> Result<(SessionEdgeSender, SessionEdgeReceiver)> {
    if edge.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: "edge".into(),
            message: "must not be empty".into(),
        });
    }
    EdgeBudget::new(budget.max_rows, budget.max_bytes)?;
    let shared = Arc::new(SessionQueue {
        edge,
        budget,
        producer: AtomicUsize::new(usize::MAX),
        queue: Mutex::new(VecDeque::new()),
    });
    Ok((
        SessionEdgeSender {
            queue: Arc::clone(&shared),
        },
        SessionEdgeReceiver(shared),
    ))
}

impl SessionEdgeSender {
    fn set_producer(&self, producer: usize) {
        self.queue.producer.store(producer, Ordering::Relaxed);
    }

    pub(super) fn validate_message(&self, message: &StreamMessage) -> Result<()> {
        self.check_cost(&EnvelopeCost::of_message(message)?)
    }

    /// Applies the runtime edges' oversize rule to an already computed cost.
    pub(super) fn check_cost(&self, cost: &EnvelopeCost) -> Result<()> {
        reject_oversize(&self.queue.edge, self.queue.budget, cost)
    }

    /// Appends after the runtime edges' oversize rule; never waits.
    pub(super) fn push(&self, message: StreamMessage) -> Result<()> {
        self.validate_message(&message)?;
        self.enqueue(message);
        Ok(())
    }

    /// Appends a message whose envelope the caller already admitted.
    pub(super) fn enqueue(&self, message: StreamMessage) {
        let producer = match self.queue.producer.load(Ordering::Relaxed) {
            usize::MAX => None,
            producer => Some(producer),
        };
        self.queue
            .queue
            .lock()
            .push_back(SessionEnvelope { producer, message });
    }
}

impl SessionEdgeReceiver {
    pub(super) fn try_recv_envelope(&self) -> Option<SessionEnvelope> {
        self.0.queue.lock().pop_front()
    }

    pub(super) fn try_recv(&self) -> Option<StreamMessage> {
        self.try_recv_envelope().map(|envelope| envelope.message)
    }
}

pub(crate) enum OperatorEdgeSender {
    Physical(EdgeSender),
    Local(LocalEdgeSender),
    Session(SessionEdgeSender),
}

pub(crate) enum OperatorEdgeReceiver {
    Physical(EdgeReceiver),
    Local(LocalEdgeReceiver),
    Session(SessionEdgeReceiver),
}

impl From<EdgeSender> for OperatorEdgeSender {
    fn from(sender: EdgeSender) -> Self {
        Self::Physical(sender)
    }
}

impl From<EdgeReceiver> for OperatorEdgeReceiver {
    fn from(receiver: EdgeReceiver) -> Self {
        Self::Physical(receiver)
    }
}

impl OperatorEdgeSender {
    pub(crate) fn edge(&self) -> &str {
        match self {
            Self::Physical(sender) => sender.edge(),
            Self::Local(sender) => &sender.0.core.edge,
            Self::Session(sender) => &sender.queue.edge,
        }
    }

    pub(crate) fn budget(&self) -> EdgeBudget {
        match self {
            Self::Physical(sender) => sender.budget(),
            Self::Local(sender) => sender.0.core.budget,
            Self::Session(sender) => sender.queue.budget,
        }
    }

    #[cfg(test)]
    pub(crate) fn metrics(&self) -> ChannelMetrics {
        match self {
            Self::Physical(sender) => sender.metrics(),
            Self::Local(sender) => sender.0.core.metrics(),
            Self::Session(sender) => ChannelMetrics {
                queue_depth: sender.queue.queue.lock().len(),
                ..ChannelMetrics::default()
            },
        }
    }

    pub(crate) fn validate_message(&self, message: &StreamMessage) -> Result<()> {
        match self {
            Self::Physical(sender) => sender.validate_message(message),
            Self::Local(sender) => sender.0.core.message_cost(message).map(|_| ()),
            Self::Session(sender) => sender.validate_message(message),
        }
    }

    pub(crate) fn validate_cost(&self, cost: &EnvelopeCost) -> Result<()> {
        reject_oversize(self.edge(), self.budget(), cost)
    }

    pub(crate) fn set_session_producer(&self, producer: usize) {
        if let Self::Session(sender) = self {
            sender.set_producer(producer);
        }
    }

    pub(crate) fn send_session(
        &mut self,
        message: StreamMessage,
        cost: &EnvelopeCost,
    ) -> Result<()> {
        let Self::Session(sender) = self else {
            return Err(CalcFlowError::Internal {
                message: format!("edge {:?} is not a session edge", self.edge()),
            });
        };
        sender.check_cost(cost)?;
        sender.enqueue(message);
        Ok(())
    }

    pub(crate) async fn send(&mut self, message: StreamMessage) -> Result<()> {
        self.send_observed(message, None).await
    }

    pub(crate) async fn send_observed(
        &mut self,
        message: StreamMessage,
        observation: Option<&RollingMetricsRecorder>,
    ) -> Result<()> {
        match self {
            Self::Physical(sender) => sender.send_observed(message, observation).await,
            Self::Local(sender) => sender.send_observed(message, observation).await,
            Self::Session(sender) => sender.push(message),
        }
    }

    pub(crate) fn into_physical(self) -> Result<EdgeSender> {
        let edge = self.edge().to_owned();
        match self {
            Self::Physical(sender) => Ok(sender),
            Self::Local(_) => Err(CalcFlowError::Internal {
                message: format!("local edge {edge:?} cannot be a source boundary"),
            }),
            Self::Session(_) => Err(CalcFlowError::Internal {
                message: format!("session edge {edge:?} cannot be a runtime source boundary"),
            }),
        }
    }

    pub(crate) fn into_session(self) -> Result<SessionEdgeSender> {
        match self {
            Self::Session(sender) => Ok(sender),
            other => Err(CalcFlowError::Internal {
                message: format!("edge {:?} is not a session edge", other.edge()),
            }),
        }
    }
}

impl OperatorEdgeReceiver {
    pub(crate) fn edge(&self) -> &str {
        match self {
            Self::Physical(receiver) => receiver.edge(),
            Self::Local(receiver) => &receiver.0.core.edge,
            Self::Session(receiver) => &receiver.0.edge,
        }
    }

    pub(crate) fn metrics(&self) -> ChannelMetrics {
        match self {
            Self::Physical(receiver) => receiver.metrics(),
            Self::Local(receiver) => receiver.0.core.metrics(),
            Self::Session(receiver) => ChannelMetrics {
                queue_depth: receiver.0.queue.lock().len(),
                ..ChannelMetrics::default()
            },
        }
    }

    /// A session edge never suspends: it yields its front message or `None`.
    pub(crate) async fn recv(&mut self) -> Result<Option<StreamMessage>> {
        match self {
            Self::Physical(receiver) => receiver.recv().await,
            Self::Local(receiver) => receiver.recv().await,
            Self::Session(receiver) => Ok(receiver.try_recv()),
        }
    }

    /// Synchronously takes the front message of a session edge.
    pub(crate) fn try_recv(&mut self) -> Result<Option<StreamMessage>> {
        match self {
            Self::Session(receiver) => Ok(receiver.try_recv()),
            Self::Physical(_) | Self::Local(_) => Err(CalcFlowError::Internal {
                message: format!("edge {:?} is not a session edge", self.edge()),
            }),
        }
    }

    pub(crate) fn try_recv_envelope(&mut self) -> Result<Option<SessionEnvelope>> {
        match self {
            Self::Session(receiver) => Ok(receiver.try_recv_envelope()),
            Self::Physical(_) | Self::Local(_) => Err(CalcFlowError::Internal {
                message: format!("edge {:?} is not a session edge", self.edge()),
            }),
        }
    }

    pub(crate) fn close(&mut self) {
        match self {
            Self::Physical(receiver) => receiver.close(),
            Self::Local(receiver) => {
                receiver.0.core.close_receiver();
                receiver.0.producer.wake();
            }
            Self::Session(receiver) => receiver.0.queue.lock().clear(),
        }
    }

    pub(crate) fn into_physical(mut self) -> Result<EdgeReceiver> {
        let edge = self.edge().to_owned();
        if matches!(self, Self::Local(_)) {
            self.close();
        }
        match self {
            Self::Physical(receiver) => Ok(receiver),
            Self::Local(_) => Err(CalcFlowError::Internal {
                message: format!("local edge {edge:?} cannot be a sink boundary"),
            }),
            Self::Session(_) => Err(CalcFlowError::Internal {
                message: format!("session edge {edge:?} cannot be a runtime sink boundary"),
            }),
        }
    }
}

impl std::fmt::Debug for OperatorEdgeReceiver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OperatorEdgeReceiver")
            .field("edge", &self.edge())
            .field("queue", &self.metrics())
            .finish_non_exhaustive()
    }
}
