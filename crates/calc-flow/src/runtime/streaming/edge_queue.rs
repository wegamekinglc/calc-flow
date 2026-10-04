use std::{collections::VecDeque, time::Duration};

use parking_lot::Mutex;

use super::{
    ChannelMetrics, EnvelopeCost, StreamMessage,
    metrics::{EdgeTraffic, MetricsRecorder, MetricsTimer},
};
use crate::{CalcFlowError, EdgeBudget, Result};

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

pub(super) struct EdgeQueueCore {
    pub(super) edge: String,
    pub(super) budget: EdgeBudget,
    metrics: MetricsRecorder,
    state: Mutex<ChannelState>,
}

pub(super) enum Dequeue {
    Message(StreamMessage),
    Empty,
    Closed,
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

impl EdgeQueueCore {
    pub(super) fn message_cost(&self, message: &StreamMessage) -> Result<EnvelopeCost> {
        let cost = EnvelopeCost::of_message(message)?;
        self.reject_oversize(&cost)?;
        Ok(cost)
    }

    fn prepare_enqueue(
        &self,
        state: &mut ChannelState,
        message: &StreamMessage,
        cost: EnvelopeCost,
        blocked_since: Option<tokio::time::Instant>,
        metrics_blocked_since: Option<&MetricsTimer>,
    ) -> Result<()> {
        let blocked_elapsed = blocked_since.map(|started| started.elapsed());
        let blocked_duration = state
            .blocked_duration
            .checked_add(blocked_elapsed.unwrap_or_default())
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: format!("runtime.metrics.{}.blocked_duration", self.edge),
                message: "counter overflow".into(),
            })?;
        let metrics_blocked_elapsed = metrics_blocked_since
            .map(|timer| timer.elapsed(&self.edge, "blocked_duration"))
            .transpose()?;
        self.metrics.record_edge_enqueue(
            &self.edge,
            EdgeTraffic::of_message(message, cost)?,
            metrics_blocked_elapsed,
        )?;
        state.charged = state.charged.checked_add(&cost).map_err(|error| {
            // The caller holds the lock after `fits` rejected every overflow.
            CalcFlowError::Internal {
                message: format!(
                    "edge {:?} charge overflowed after a successful capacity check: {error}",
                    self.edge
                ),
            }
        })?;
        state.high_water = state.high_water.max_components(&state.charged);
        state.blocked_duration = blocked_duration;
        Ok(())
    }

    fn begin_wait(
        &self,
        state: &mut ChannelState,
        blocked_since: &mut Option<tokio::time::Instant>,
        metrics_blocked_since: &mut Option<MetricsTimer>,
    ) -> Result<()> {
        if blocked_since.is_none() {
            let next_blocked = state.blocked_sends.checked_add(1).ok_or_else(|| {
                CalcFlowError::InvalidArgument {
                    field: format!("runtime.metrics.{}.blocked_sends", self.edge),
                    message: "counter overflow".into(),
                }
            })?;
            self.metrics.record_edge_blocked(&self.edge)?;
            state.blocked_sends = next_blocked;
            *blocked_since = Some(tokio::time::Instant::now());
            *metrics_blocked_since = Some(self.metrics.timer());
        }
        Ok(())
    }

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

impl EdgeQueueCore {
    pub(super) fn new(edge: String, budget: EdgeBudget, metrics: MetricsRecorder) -> Result<Self> {
        if edge.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "edge".into(),
                message: "must not be empty".into(),
            });
        }
        EdgeBudget::new(budget.max_rows, budget.max_bytes)?;
        Ok(Self {
            edge,
            budget,
            metrics,
            state: Mutex::new(ChannelState::default()),
        })
    }

    pub(super) fn try_enqueue(
        &self,
        message: &mut Option<StreamMessage>,
        cost: EnvelopeCost,
        blocked_since: &mut Option<tokio::time::Instant>,
        metrics_blocked_since: &mut Option<MetricsTimer>,
    ) -> Result<bool> {
        let mut state = self.state.lock();
        if state.receiver_closed {
            return Err(CalcFlowError::EdgeClosed {
                edge: self.edge.clone(),
            });
        }
        if !fits(&state.charged, &cost, &self.budget) {
            self.begin_wait(&mut state, blocked_since, metrics_blocked_since)?;
            return Ok(false);
        }
        self.prepare_enqueue(
            &mut state,
            message.as_ref().expect("pending send owns its message"),
            cost,
            *blocked_since,
            metrics_blocked_since.as_ref(),
        )?;
        state.queue.push_back((
            message
                .take()
                .expect("accepted send transfers its message once"),
            cost,
        ));
        Ok(true)
    }

    pub(super) fn try_dequeue(&self) -> Result<Dequeue> {
        let mut state = self.state.lock();
        if let Some((message, cost)) = state.queue.front() {
            self.metrics
                .record_edge_dequeue(&self.edge, EdgeTraffic::of_message(message, *cost)?)?;
            let charged = state.charged.checked_sub(cost)?;
            let (message, _) = state
                .queue
                .pop_front()
                .expect("locked queue still owns its front");
            state.charged = charged;
            return Ok(Dequeue::Message(message));
        }
        Ok(if state.receiver_closed || state.sender_closed {
            Dequeue::Closed
        } else {
            Dequeue::Empty
        })
    }

    #[cfg(test)]
    pub(super) fn set_blocked_duration_for_test(&self, duration: Duration) {
        self.state.lock().blocked_duration = duration;
    }

    pub(super) fn metrics(&self) -> ChannelMetrics {
        self.state.lock().metrics()
    }

    pub(super) fn close_receiver(&self) {
        self.state.lock().receiver_closed = true;
    }

    pub(super) fn drop_receiver(&self) {
        let mut state = self.state.lock();
        state.receiver_closed = true;
        self.metrics.record_edge_drop(&self.edge, state.charged);
        state.queue.clear();
        state.charged = EnvelopeCost::ZERO;
    }

    pub(super) fn close_sender(&self) {
        self.state.lock().sender_closed = true;
    }
}
