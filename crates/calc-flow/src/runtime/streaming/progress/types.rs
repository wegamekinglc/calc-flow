use std::{cmp::Ordering, sync::Arc, time::Duration};

use crate::{CalcFlowError, Result};

macro_rules! semantic_counter {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub(crate) struct $name(pub(crate) u64);

        impl $name {
            pub(crate) const fn get(self) -> u64 {
                self.0
            }
        }
    };
}

semantic_counter!(LocalSequence);
semantic_counter!(GlobalSequence);
semantic_counter!(TimerSequence);
semantic_counter!(TimerGeneration);
semantic_counter!(IdleEpoch);
semantic_counter!(ReceiptSequence);
semantic_counter!(InboxSequence);
semantic_counter!(FenceSequence);
semantic_counter!(DrainEpoch);
semantic_counter!(TraceRecordOrdinal);
semantic_counter!(TracePosition);
semantic_counter!(AdmissionAttemptOrdinal);
semantic_counter!(AdmissionGateGeneration);
semantic_counter!(GateCloseOrdinal);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CheckedSemanticAllocator {
    next: u64,
    error_path: Arc<str>,
}

impl CheckedSemanticAllocator {
    pub(crate) fn new(next: u64, error_path: impl Into<Arc<str>>) -> Self {
        Self {
            next,
            error_path: error_path.into(),
        }
    }

    pub(crate) fn next(&self) -> u64 {
        self.next
    }

    pub(crate) fn checked_peek_and_successor(&self) -> Result<(u64, u64)> {
        self.next
            .checked_add(1)
            .map(|successor| (self.next, successor))
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: self.error_path.to_string(),
                message: "counter exhausted before a successor could be reserved".into(),
            })
    }

    pub(crate) fn checked_successor_after(&self, allocations: usize) -> Result<u64> {
        let allocations =
            u64::try_from(allocations).map_err(|_| CalcFlowError::InvalidArgument {
                field: self.error_path.to_string(),
                message: "allocation count exceeds the semantic counter range".into(),
            })?;
        self.next
            .checked_add(allocations)
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: self.error_path.to_string(),
                message: "counter exhausted before all successors could be reserved".into(),
            })
    }

    pub(crate) fn allocate(&mut self) -> Result<u64> {
        let (value, successor) = self.checked_peek_and_successor()?;
        self.next = successor;
        Ok(value)
    }

    pub(crate) fn set_next_for_restore(&mut self, next: u64) {
        self.next = next;
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct LogicalInstant(pub(crate) u128);

impl LogicalInstant {
    pub(crate) const ZERO: Self = Self(0);

    pub(crate) fn checked_add(self, delay: Duration) -> Result<Self> {
        self.0
            .checked_add(delay.as_nanos())
            .map(Self)
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: "runtime.progress.timers.deadline".into(),
                message: "logical timer deadline overflowed".into(),
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DriverClockCoordinate {
    pub(crate) trace: [u8; 32],
    pub(crate) instant: LogicalInstant,
}

impl DriverClockCoordinate {
    pub(crate) const fn new(trace: [u8; 32], instant: LogicalInstant) -> Self {
        Self { trace, instant }
    }
}

pub(crate) trait DriverLogicalClock: Send + Sync + 'static {
    fn coordinate(&self) -> DriverClockCoordinate;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum ReadyClass {
    InputOrControl = 0,
    WatermarkTimer = 1,
    IdleTimer = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ReadyKey {
    pub(crate) logical_instant: LogicalInstant,
    pub(crate) class: ReadyClass,
    pub(crate) binding_ordinal: super::prepare::BindingOrdinal,
    pub(crate) local_sequence: LocalSequence,
}

impl Ord for ReadyKey {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.logical_instant,
            self.class,
            self.binding_ordinal,
            self.local_sequence,
        )
            .cmp(&(
                other.logical_instant,
                other.class,
                other.binding_ordinal,
                other.local_sequence,
            ))
    }
}

impl PartialOrd for ReadyKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum TimerKind {
    Watermark,
    Idle,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct TimerIdentity {
    pub(crate) binding_ordinal: super::prepare::BindingOrdinal,
    pub(crate) kind: TimerKind,
    pub(crate) deadline: LogicalInstant,
    pub(crate) generation: TimerGeneration,
    pub(crate) timer_sequence: TimerSequence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CurrentTimer {
    pub(crate) deadline: LogicalInstant,
    pub(crate) generation: TimerGeneration,
    pub(crate) timer_sequence: TimerSequence,
    pub(crate) ready_local_sequence: LocalSequence,
}

impl CurrentTimer {
    pub(crate) fn identity(
        self,
        binding_ordinal: super::prepare::BindingOrdinal,
        kind: TimerKind,
    ) -> TimerIdentity {
        TimerIdentity {
            binding_ordinal,
            kind,
            deadline: self.deadline,
            generation: self.generation,
            timer_sequence: self.timer_sequence,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProgressFailureKind {
    InvalidPolicy,
    InvalidSchema,
    CapabilityConflict,
    ProtocolViolation,
    SnapshotBoundary,
    SnapshotMismatch,
    ReplayMismatch,
    RestoreUnsupported,
    Arithmetic,
    CounterExhaustion,
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProgressFailure {
    pub(crate) kind: ProgressFailureKind,
    pub(crate) path: Arc<str>,
    pub(crate) reason: Arc<str>,
}

impl ProgressFailure {
    pub(crate) fn new(
        kind: ProgressFailureKind,
        path: impl Into<Arc<str>>,
        reason: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            kind,
            path: path.into(),
            reason: reason.into(),
        }
    }

    pub(crate) fn protocol(path: impl Into<Arc<str>>, reason: impl Into<Arc<str>>) -> Self {
        Self::new(ProgressFailureKind::ProtocolViolation, path, reason)
    }

    pub(crate) fn counter(path: impl Into<Arc<str>>) -> Self {
        Self::new(
            ProgressFailureKind::CounterExhaustion,
            path,
            "counter exhausted before a successor could be reserved",
        )
    }

    pub(crate) fn into_existing_error(self) -> CalcFlowError {
        CalcFlowError::InvalidArgument {
            field: self.path.to_string(),
            message: self.reason.to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SelectedItemError {
    pub(crate) first_error_key: ReadyKey,
    pub(crate) failure: ProgressFailure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DriverFailurePhase {
    AdmissionAttemptAllocation,
    AdmissionDecision,
    DrainEpochAllocation,
    FenceAllocation,
    InboxFreeze,
    ReadyKeyConstruction,
    ReplayPlanValidation,
    ReplayRecordValidation,
    GateClosePlanning,
    SettlementPlanning,
    SnapshotValidation,
    TerminalCleanup,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DriverPhaseCoordinate {
    Admission {
        last_attempt: Option<AdmissionAttemptOrdinal>,
        binding: super::prepare::BindingIdentity,
        gate_generation: Option<AdmissionGateGeneration>,
    },
    Drain {
        epoch: Option<DrainEpoch>,
        binding: Option<super::prepare::BindingOrdinal>,
        fence: Option<FenceSequence>,
    },
    Counter {
        stable_path: Arc<str>,
        last_value: Option<u64>,
    },
    Snapshot {
        stable_path: Arc<str>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DriverPhaseError {
    pub(crate) phase: DriverFailurePhase,
    pub(crate) coordinate: DriverPhaseCoordinate,
    pub(crate) failure: ProgressFailure,
}

impl DriverPhaseError {
    pub(crate) fn counter(
        phase: DriverFailurePhase,
        path: impl Into<Arc<str>>,
        last_value: Option<u64>,
    ) -> Self {
        let path = path.into();
        Self {
            phase,
            coordinate: DriverPhaseCoordinate::Counter {
                stable_path: Arc::clone(&path),
                last_value,
            },
            failure: ProgressFailure::counter(path),
        }
    }

    pub(crate) fn into_existing_error(self) -> CalcFlowError {
        self.failure.into_existing_error()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{CheckedSemanticAllocator, LocalSequence, LogicalInstant, ReadyClass, ReadyKey};
    use crate::runtime::streaming::progress::prepare::BindingOrdinal;

    #[test]
    fn checked_allocator_never_returns_last_unreservable_value() {
        let mut allocator =
            CheckedSemanticAllocator::new(u64::MAX - 1, "runtime.progress.counters.test");
        assert_eq!(allocator.allocate().unwrap(), u64::MAX - 1);
        assert!(allocator.allocate().is_err());
    }

    #[test]
    fn ready_order_is_class_then_binding_then_local_sequence() {
        let instant = LogicalInstant(7);
        let key = |class, binding, local| ReadyKey {
            logical_instant: instant,
            class,
            binding_ordinal: BindingOrdinal::new(binding),
            local_sequence: LocalSequence(local),
        };
        let mut keys = vec![
            key(ReadyClass::IdleTimer, 0, 0),
            key(ReadyClass::WatermarkTimer, 1, 0),
            key(ReadyClass::InputOrControl, 1, 1),
            key(ReadyClass::InputOrControl, 0, 3),
            key(ReadyClass::WatermarkTimer, 0, 2),
        ];
        keys.sort();
        assert_eq!(
            keys.iter().map(|key| key.class).collect::<Vec<_>>(),
            [
                ReadyClass::InputOrControl,
                ReadyClass::InputOrControl,
                ReadyClass::WatermarkTimer,
                ReadyClass::WatermarkTimer,
                ReadyClass::IdleTimer,
            ]
        );
        assert_eq!(keys[0].binding_ordinal.get(), 0);
    }

    #[test]
    fn logical_deadline_addition_is_checked() {
        assert_eq!(
            LogicalInstant::ZERO
                .checked_add(Duration::from_nanos(3))
                .unwrap(),
            LogicalInstant(3)
        );
        assert!(
            LogicalInstant(u128::MAX)
                .checked_add(Duration::from_nanos(1))
                .is_err()
        );
    }
}
