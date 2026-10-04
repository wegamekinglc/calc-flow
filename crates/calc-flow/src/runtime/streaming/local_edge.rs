use std::{future::poll_fn, sync::Arc, task::Poll};

use futures::task::AtomicWaker;

use super::{
    ChannelMetrics, EdgeReceiver, EdgeSender, StreamMessage,
    edge_queue::{Dequeue, EdgeQueueCore},
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

pub(crate) enum OperatorEdgeSender {
    Physical(EdgeSender),
    Local(LocalEdgeSender),
}

pub(crate) enum OperatorEdgeReceiver {
    Physical(EdgeReceiver),
    Local(LocalEdgeReceiver),
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
        }
    }

    pub(crate) fn budget(&self) -> EdgeBudget {
        match self {
            Self::Physical(sender) => sender.budget(),
            Self::Local(sender) => sender.0.core.budget,
        }
    }

    pub(crate) fn validate_message(&self, message: &StreamMessage) -> Result<()> {
        match self {
            Self::Physical(sender) => sender.validate_message(message),
            Self::Local(sender) => sender.0.core.message_cost(message).map(|_| ()),
        }
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
        }
    }

    pub(crate) fn into_physical(self) -> Result<EdgeSender> {
        let edge = self.edge().to_owned();
        match self {
            Self::Physical(sender) => Ok(sender),
            Self::Local(_) => Err(CalcFlowError::Internal {
                message: format!("local edge {edge:?} cannot be a source boundary"),
            }),
        }
    }
}

impl OperatorEdgeReceiver {
    pub(crate) fn edge(&self) -> &str {
        match self {
            Self::Physical(receiver) => receiver.edge(),
            Self::Local(receiver) => &receiver.0.core.edge,
        }
    }

    pub(crate) fn metrics(&self) -> ChannelMetrics {
        match self {
            Self::Physical(receiver) => receiver.metrics(),
            Self::Local(receiver) => receiver.0.core.metrics(),
        }
    }

    pub(crate) async fn recv(&mut self) -> Result<Option<StreamMessage>> {
        match self {
            Self::Physical(receiver) => receiver.recv().await,
            Self::Local(receiver) => receiver.recv().await,
        }
    }

    pub(crate) fn close(&mut self) {
        match self {
            Self::Physical(receiver) => receiver.close(),
            Self::Local(receiver) => {
                receiver.0.core.close_receiver();
                receiver.0.producer.wake();
            }
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
