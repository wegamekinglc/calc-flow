use std::{future::poll_fn, sync::Arc, task::Poll};

use async_trait::async_trait;
use futures::task::AtomicWaker;
use parking_lot::Mutex;

use super::{
    ChannelMetrics, EdgeReceiver, EdgeSender, StreamMessage,
    edge_queue::{Dequeue, EdgeQueueCore},
    metrics::{MetricsRecorder, MetricsTimer},
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

#[async_trait]
pub(crate) trait RegionDrain: Send {
    async fn drain_one(&mut self) -> Result<bool>;
}

#[derive(Default)]
struct RegionState {
    downstream: Option<Box<dyn RegionDrain>>,
    failure: Option<CalcFlowError>,
}

pub(crate) struct RegionShared {
    core: Arc<EdgeQueueCore>,
    state: Mutex<RegionState>,
}

pub(crate) type RegionEdgeControl = Arc<RegionShared>;

enum LocalEdgeOwnerKind {
    Cooperative(Arc<Shared>),
    Region(Arc<EdgeQueueCore>),
}

pub(crate) struct LocalEdgeOwner(LocalEdgeOwnerKind);
pub(crate) struct LocalEdgeSender(Arc<Shared>);
pub(crate) struct LocalEdgeReceiver(Arc<Shared>);
pub(crate) struct RegionEdgeSender(RegionEdgeControl);
pub(crate) struct RegionEdgeReceiver(Arc<EdgeQueueCore>);

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
        LocalEdgeOwner(LocalEdgeOwnerKind::Cooperative(shared)),
    ))
}

pub(super) fn region_edge(
    edge: String,
    budget: EdgeBudget,
    metrics: MetricsRecorder,
) -> Result<(RegionEdgeSender, RegionEdgeReceiver, LocalEdgeOwner)> {
    let core = Arc::new(EdgeQueueCore::new(edge, budget, metrics)?);
    let shared = Arc::new(RegionShared {
        core: Arc::clone(&core),
        state: Mutex::new(RegionState::default()),
    });
    Ok((
        RegionEdgeSender(shared),
        RegionEdgeReceiver(Arc::clone(&core)),
        LocalEdgeOwner(LocalEdgeOwnerKind::Region(core)),
    ))
}

impl RegionShared {
    pub(crate) fn edge(&self) -> &str {
        &self.core.edge
    }

    pub(crate) fn install(&self, downstream: Box<dyn RegionDrain>) -> Result<()> {
        let mut state = self.state.lock();
        if state.downstream.is_some() {
            return Err(CalcFlowError::Internal {
                message: format!(
                    "region edge {:?} downstream was installed twice",
                    self.edge()
                ),
            });
        }
        state.downstream = Some(downstream);
        Ok(())
    }

    pub(crate) fn take_failure(&self) -> Option<CalcFlowError> {
        self.state.lock().failure.take()
    }

    pub(crate) async fn drain_all(&self) -> Result<()> {
        while self.drain_one().await? {}
        Ok(())
    }

    async fn drain_one(&self) -> Result<bool> {
        let downstream =
            self.state
                .lock()
                .downstream
                .take()
                .ok_or_else(|| CalcFlowError::Internal {
                    message: format!("region edge {:?} has no downstream", self.edge()),
                })?;
        let mut drain = RegionDrainGuard {
            shared: self,
            downstream: Some(downstream),
        };
        drain
            .downstream
            .as_mut()
            .expect("region drain owns its downstream")
            .drain_one()
            .await
    }

    async fn send_observed(
        &self,
        message: StreamMessage,
        observation: Option<&RollingMetricsRecorder>,
    ) -> Result<()> {
        let cost = self.core.message_cost(&message)?;
        let mut message = Some(message);
        let mut blocked_since = None;
        let mut metrics_blocked_since: Option<MetricsTimer> = None;
        let mut wait = None;
        loop {
            if self.core.try_enqueue(
                &mut message,
                cost,
                &mut blocked_since,
                &mut metrics_blocked_since,
            )? {
                return Ok(());
            }
            if wait.is_none() {
                wait = observation.map(|recorder| recorder.stage(RollingStage::SendWait));
            }
            match self.drain_one().await {
                Ok(true) => {}
                Ok(false) => {
                    return Err(CalcFlowError::Internal {
                        message: format!(
                            "region edge {:?} remained full with an empty downstream queue",
                            self.edge()
                        ),
                    });
                }
                Err(error) => {
                    self.state.lock().failure = Some(error);
                    return Err(CalcFlowError::Cancelled {
                        run_id: "compiled-region".into(),
                    });
                }
            }
        }
    }
}

struct RegionDrainGuard<'a> {
    shared: &'a RegionShared,
    downstream: Option<Box<dyn RegionDrain>>,
}

impl Drop for RegionDrainGuard<'_> {
    fn drop(&mut self) {
        self.shared.state.lock().downstream = self.downstream.take();
    }
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

impl RegionEdgeSender {
    async fn send_observed(
        &mut self,
        message: StreamMessage,
        observation: Option<&RollingMetricsRecorder>,
    ) -> Result<()> {
        self.0.send_observed(message, observation).await
    }
}

impl RegionEdgeReceiver {
    fn try_recv(&mut self) -> Result<Dequeue> {
        self.0.try_dequeue()
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

impl Drop for RegionEdgeReceiver {
    fn drop(&mut self) {
        self.0.drop_receiver();
    }
}

impl Drop for RegionEdgeSender {
    fn drop(&mut self) {
        self.0.core.close_sender();
    }
}

impl Drop for LocalEdgeOwner {
    fn drop(&mut self) {
        match &self.0 {
            LocalEdgeOwnerKind::Cooperative(shared) => {
                shared.core.drop_receiver();
                shared.core.close_sender();
                shared.producer.wake();
                shared.consumer.wake();
                drop(shared.producer.take());
                drop(shared.consumer.take());
            }
            LocalEdgeOwnerKind::Region(core) => {
                core.drop_receiver();
                core.close_sender();
            }
        }
    }
}

pub(crate) enum OperatorEdgeSender {
    Physical(EdgeSender),
    Local(LocalEdgeSender),
    Region(RegionEdgeSender),
}

pub(crate) enum OperatorEdgeReceiver {
    Physical(EdgeReceiver),
    Local(LocalEdgeReceiver),
    Region(RegionEdgeReceiver),
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
            Self::Region(sender) => &sender.0.core.edge,
        }
    }

    pub(crate) fn budget(&self) -> EdgeBudget {
        match self {
            Self::Physical(sender) => sender.budget(),
            Self::Local(sender) => sender.0.core.budget,
            Self::Region(sender) => sender.0.core.budget,
        }
    }

    pub(crate) fn validate_message(&self, message: &StreamMessage) -> Result<()> {
        match self {
            Self::Physical(sender) => sender.validate_message(message),
            Self::Local(sender) => sender.0.core.message_cost(message).map(|_| ()),
            Self::Region(sender) => sender.0.core.message_cost(message).map(|_| ()),
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
            Self::Region(sender) => sender.send_observed(message, observation).await,
        }
    }

    pub(crate) fn region_control(&self) -> Option<RegionEdgeControl> {
        match self {
            Self::Region(sender) => Some(Arc::clone(&sender.0)),
            _ => None,
        }
    }

    pub(crate) fn into_physical(self) -> Result<EdgeSender> {
        let edge = self.edge().to_owned();
        match self {
            Self::Physical(sender) => Ok(sender),
            Self::Local(_) | Self::Region(_) => Err(CalcFlowError::Internal {
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
            Self::Region(receiver) => &receiver.0.edge,
        }
    }

    pub(crate) fn metrics(&self) -> ChannelMetrics {
        match self {
            Self::Physical(receiver) => receiver.metrics(),
            Self::Local(receiver) => receiver.0.core.metrics(),
            Self::Region(receiver) => receiver.0.metrics(),
        }
    }

    pub(crate) async fn recv(&mut self) -> Result<Option<StreamMessage>> {
        match self {
            Self::Physical(receiver) => receiver.recv().await,
            Self::Local(receiver) => receiver.recv().await,
            Self::Region(receiver) => Err(CalcFlowError::Internal {
                message: format!(
                    "region edge {:?} cannot use cooperative receive dispatch",
                    receiver.0.edge
                ),
            }),
        }
    }

    pub(super) fn try_recv_region(&mut self) -> Result<Dequeue> {
        match self {
            Self::Region(receiver) => receiver.try_recv(),
            _ => Err(CalcFlowError::Internal {
                message: format!("edge {:?} is not a compiled region input", self.edge()),
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
            Self::Region(receiver) => receiver.0.close_receiver(),
        }
    }

    pub(crate) fn into_physical(mut self) -> Result<EdgeReceiver> {
        let edge = self.edge().to_owned();
        if matches!(self, Self::Local(_) | Self::Region(_)) {
            self.close();
        }
        match self {
            Self::Physical(receiver) => Ok(receiver),
            Self::Local(_) | Self::Region(_) => Err(CalcFlowError::Internal {
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
