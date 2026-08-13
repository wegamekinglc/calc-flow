//! The v3 streaming runtime building blocks.
//!
//! The two-field budget on [`edge_channel`] enforces three
//! independent admission predicates on every edge: queued envelopes and
//! charged rows each stay within `max_rows`, while charged bytes stay within
//! `max_bytes`.

mod channel;
#[allow(
    dead_code,
    reason = "M5 checkpoint coordination remains crate-private until the post-M5 A6 gate"
)]
pub(crate) mod checkpoint;
mod context;
#[allow(
    dead_code,
    reason = "M2 runtime completion remains crate-private until the post-M5 public A6 gate"
)]
pub(crate) mod job;
mod message;
#[allow(
    dead_code,
    reason = "M2 metrics remain crate-private until the post-M5 public A6 gate"
)]
mod metrics;
#[allow(
    dead_code,
    reason = "M2 operator tasks remain crate-private until the post-M5 public A6 gate"
)]
mod operator_task;
#[allow(
    dead_code,
    reason = "M3 progress coordination remains crate-private until the post-M5 public A6 gate"
)]
pub(crate) mod progress;
#[allow(
    dead_code,
    reason = "A6 safe status/error projection remains crate-private until the public facade gate"
)]
pub(crate) mod projection;
#[allow(
    dead_code,
    reason = "M2 internal runner remains crate-private until the post-M5 public A6 gate"
)]
pub(crate) mod runner;
#[allow(
    dead_code,
    reason = "M2 ordinary sink tasks remain crate-private until the post-M5 public A6 gate"
)]
mod sink_task;
#[cfg(test)]
mod soak;
#[allow(
    dead_code,
    reason = "M2 source integration remains crate-private until the post-M5 public A6 gate"
)]
pub(crate) mod source_task;
#[allow(
    dead_code,
    reason = "M2 supervision remains crate-private until the post-M5 public A6 gate"
)]
mod supervisor;

pub use channel::{ChannelMetrics, EdgeReceiver, EdgeSender, EnvelopeCost, edge_channel};
pub use context::StreamJobContext;
pub use message::{StreamMessage, StreamMessageKind};
