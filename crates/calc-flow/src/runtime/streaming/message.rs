use std::fmt;

use crate::{Batch, Epoch, EventTime};

/// The kind of one edge message, exposed for inspection and routing.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StreamMessageKind {
    Data,
    Watermark,
    Barrier,
    Idle,
    EndOfInput,
}

#[derive(Clone)]
#[allow(
    dead_code,
    reason = "control variants are constructed by the M2 runtime; M1.2 unit tests pin the contract"
)]
enum MessageKind {
    Data(Batch),
    Watermark(EventTime),
    Barrier(Epoch),
    Idle,
    EndOfInput,
}

/// The single ordered message carried by one stream edge (spec S1.1).
///
/// The representation is private: `data` is the only public constructor.
/// Watermark, barrier, idle, and end-of-input messages are created only by
/// the runtime through crate-private constructors, after validation (S1.3),
/// so operators and sources can never forge, suppress, or reorder control
/// messages. Fan-out clones share the immutable `Batch` payload (S3).
#[derive(Clone)]
pub struct StreamMessage(MessageKind);

impl StreamMessage {
    /// Wraps one immutable data batch.
    pub fn data(batch: Batch) -> Self {
        Self(MessageKind::Data(batch))
    }

    #[allow(
        dead_code,
        reason = "the M2 source task constructs watermarks through this validated constructor"
    )]
    pub(crate) fn watermark(at: EventTime) -> Self {
        Self(MessageKind::Watermark(at))
    }

    #[allow(
        dead_code,
        reason = "the M5 coordinator injects barriers through this validated constructor"
    )]
    pub(crate) fn barrier(epoch: Epoch) -> Self {
        Self(MessageKind::Barrier(epoch))
    }

    #[allow(
        dead_code,
        reason = "the M3 watermark policy declares idle ingresses through this constructor"
    )]
    pub(crate) fn idle() -> Self {
        Self(MessageKind::Idle)
    }

    #[allow(
        dead_code,
        reason = "the M2 source task terminates ingresses through this constructor"
    )]
    pub(crate) fn end_of_input() -> Self {
        Self(MessageKind::EndOfInput)
    }

    /// Returns the message kind for inspection and routing.
    pub const fn kind(&self) -> StreamMessageKind {
        match self.0 {
            MessageKind::Data(_) => StreamMessageKind::Data,
            MessageKind::Watermark(_) => StreamMessageKind::Watermark,
            MessageKind::Barrier(_) => StreamMessageKind::Barrier,
            MessageKind::Idle => StreamMessageKind::Idle,
            MessageKind::EndOfInput => StreamMessageKind::EndOfInput,
        }
    }

    /// Returns the data payload, when this is a data message.
    pub const fn as_data(&self) -> Option<&Batch> {
        match &self.0 {
            MessageKind::Data(batch) => Some(batch),
            _ => None,
        }
    }

    /// Returns the watermark value, when this is a watermark message.
    pub const fn as_watermark(&self) -> Option<EventTime> {
        match self.0 {
            MessageKind::Watermark(at) => Some(at),
            _ => None,
        }
    }

    /// Returns the barrier epoch, when this is a barrier message.
    pub const fn as_barrier(&self) -> Option<Epoch> {
        match self.0 {
            MessageKind::Barrier(epoch) => Some(epoch),
            _ => None,
        }
    }

    /// Returns whether this message marks its ingress idle (S1.4).
    pub const fn is_idle(&self) -> bool {
        matches!(self.0, MessageKind::Idle)
    }

    /// Returns whether this message terminates its ingress (S1.6).
    pub const fn is_end_of_input(&self) -> bool {
        matches!(self.0, MessageKind::EndOfInput)
    }
}

impl fmt::Debug for StreamMessage {
    /// Diagnostics show kinds and typed business values only: row payloads,
    /// batch metadata, and attributes (which may carry secrets, invariant I4)
    /// never appear.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            MessageKind::Data(batch) => formatter
                .debug_struct("StreamMessage::Data")
                .field("kind", &batch.kind())
                .field("rows", &batch.num_rows())
                .finish(),
            MessageKind::Watermark(at) => formatter
                .debug_tuple("StreamMessage::Watermark")
                .field(&at.as_micros())
                .finish(),
            MessageKind::Barrier(epoch) => formatter
                .debug_tuple("StreamMessage::Barrier")
                .field(&epoch.as_u64())
                .finish(),
            MessageKind::Idle => formatter.write_str("StreamMessage::Idle"),
            MessageKind::EndOfInput => formatter.write_str("StreamMessage::EndOfInput"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

    use super::*;
    use crate::BatchMetadata;

    fn batch() -> Batch {
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as _,
        )])
        .unwrap();
        Batch::table(vec![record], BatchMetadata::default()).unwrap()
    }

    #[test]
    fn crate_private_control_constructors_carry_typed_values() {
        let watermark = StreamMessage::watermark(EventTime::from_micros(-5));
        assert_eq!(watermark.kind(), StreamMessageKind::Watermark);
        assert_eq!(watermark.as_watermark(), Some(EventTime::from_micros(-5)));
        assert!(watermark.as_data().is_none());

        let barrier = StreamMessage::barrier(Epoch::INITIAL);
        assert_eq!(barrier.kind(), StreamMessageKind::Barrier);
        assert_eq!(barrier.as_barrier(), Some(Epoch::INITIAL));

        assert!(StreamMessage::idle().is_idle());
        assert!(StreamMessage::end_of_input().is_end_of_input());
        assert_eq!(StreamMessage::idle().kind(), StreamMessageKind::Idle);
        assert_eq!(
            StreamMessage::end_of_input().kind(),
            StreamMessageKind::EndOfInput
        );
    }

    #[test]
    fn control_debug_shows_typed_values_without_payload() {
        let watermark = format!("{:?}", StreamMessage::watermark(EventTime::from_micros(-5)));
        assert!(watermark.contains("-5"));
        let barrier = format!("{:?}", StreamMessage::barrier(Epoch::INITIAL));
        assert!(barrier.contains('1'));
        assert_eq!(
            format!("{:?}", StreamMessage::idle()),
            "StreamMessage::Idle"
        );
        assert_eq!(
            format!("{:?}", StreamMessage::end_of_input()),
            "StreamMessage::EndOfInput"
        );
        let data = format!("{:?}", StreamMessage::data(batch()));
        assert!(data.contains("rows: 3"));
        assert!(!data.contains("[1, 2, 3]"));
    }
}
