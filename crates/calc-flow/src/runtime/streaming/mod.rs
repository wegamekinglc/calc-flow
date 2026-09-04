//! The v3 streaming runtime building blocks.
//!
//! The two-field budget on [`edge_channel`] enforces three
//! independent admission predicates on every edge: queued envelopes and
//! charged rows each stay within `max_rows`, while charged bytes stay within
//! `max_bytes`.

mod channel;
#[allow(
    dead_code,
    reason = "checkpoint coordination exposes only the safe continuous facade"
)]
pub(crate) mod checkpoint;
mod context;
pub(crate) mod failure;
#[allow(
    dead_code,
    reason = "runtime completion is owned behind the safe continuous facade"
)]
pub(crate) mod job;
mod message;
#[allow(
    dead_code,
    reason = "internal metrics include fields excluded from the public status projection"
)]
mod metrics;
#[allow(
    dead_code,
    reason = "operator task internals are exercised through the continuous facade"
)]
mod operator_task;
#[allow(
    dead_code,
    reason = "progress coordination remains an internal continuous-runtime detail"
)]
pub(crate) mod progress;
#[allow(
    dead_code,
    reason = "projection includes crate-only helpers for lifecycle and manual-checkpoint errors"
)]
pub(crate) mod projection;
#[allow(
    dead_code,
    reason = "the internal runner is owned exclusively by the public one-shot facade"
)]
pub(crate) mod runner;
#[allow(
    dead_code,
    reason = "ordinary sink tasks are exercised through public sink bindings"
)]
mod sink_task;
#[cfg(test)]
mod soak;
#[allow(
    dead_code,
    reason = "source integration is exercised through public source bindings"
)]
pub(crate) mod source_task;
#[allow(
    dead_code,
    reason = "task supervision is an internal continuous-runtime detail"
)]
mod supervisor;

pub use channel::{ChannelMetrics, EdgeReceiver, EdgeSender, EnvelopeCost, edge_channel};
pub use context::StreamJobContext;
pub use message::{StreamMessage, StreamMessageKind};
