//! Bounded, payload-free observations of managed Native rolling callbacks.

#[cfg(test)]
mod lease_tests;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use serde::Serialize;

use crate::{CalcFlowError, CancellationToken, Result};

/// Lifetime Native rolling observations for one operator in one running job.
///
/// Each callback class is independently accumulated. Completed stage durations
/// partition its inclusive callback duration; snapshots never expose partial
/// work from an active callback. Restored jobs begin with fresh observations.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub struct RollingMetrics {
    /// Observations of `process_data` callbacks.
    pub data: RollingCallbackMetrics,
    /// Observations of `on_watermark` callbacks.
    pub watermark: RollingCallbackMetrics,
    /// Observations of `on_end` callbacks.
    pub end: RollingCallbackMetrics,
    /// At least one observation could not be represented exactly.
    pub overflowed: bool,
}

/// Fixed counters and disjoint elapsed stages for one rolling callback class.
///
/// Work includes unsuccessful attempts. Prepared rows are not sink delivery.
/// During an active callback `started` can exceed finalized outcomes. If
/// `overflowed` is set on the enclosing observation, exclude it from evidence.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[non_exhaustive]
pub struct RollingCallbackMetrics {
    /// Entered callbacks, including callbacks still running.
    pub started: u64,
    /// Callbacks returning success after their state transition.
    pub succeeded: u64,
    /// Callbacks returning an ordinary error.
    pub failed: u64,
    /// Callbacks returning cancellation or interrupted by known cancellation.
    pub cancelled: u64,
    /// Callbacks dropped without a result or known cancellation.
    pub interrupted: u64,
    /// Inclusive elapsed time through callback completion and owned CPU drain.
    /// After forced task drop this can exceed the task processing timer.
    pub callback_duration: Duration,
    /// Input, schema, value and lateness validation.
    pub input_validation_duration: Duration,
    /// Ordering proof, duplicate detection and frontier classification.
    pub ordering_proof_duration: Duration,
    /// Entity-key encoding and row-to-entity routing.
    pub entity_resolution_duration: Duration,
    /// Preparation of private state for touched entities.
    pub state_preparation_duration: Duration,
    /// Numeric transitions and interleaved result appends.
    pub numeric_update_duration: Duration,
    /// Retained-tail selection, materialization and commit.
    pub history_maintenance_duration: Duration,
    /// Arrow finalization, assembly and required input concatenation.
    pub arrow_output_duration: Duration,
    /// Exact output accounting, chunking and envelope validation.
    pub budget_preparation_duration: Duration,
    /// Capacity awaits, including failed and cancelled waits.
    pub send_wait_duration: Duration,
    /// Callback time not assigned to another stage.
    pub other_duration: Duration,
    /// Rows presented to data callbacks, including rejected inputs.
    pub input_rows: u64,
    /// Rows examined for order proof, including retries.
    pub order_proof_rows: u64,
    /// Rows processed by entity resolution.
    pub resolved_rows: u64,
    /// Entity preparations, rather than globally distinct identities.
    pub touched_entities: u64,
    /// Actual existing-entity state copies.
    pub copied_entities: u64,
    /// Rows processed by numeric transitions.
    pub numeric_rows: u64,
    /// Actual retained-history row materializations.
    pub history_rows_materialized: u64,
    /// Actual scalar-value conversions.
    pub scalar_value_conversions: u64,
    /// Output rows prepared before emission.
    pub output_rows_prepared: u64,
    /// Output chunks prepared before emission.
    pub output_chunks_prepared: u64,
}

#[derive(Clone, Copy)]
pub(crate) enum RollingCallback {
    Data,
    Watermark,
    End,
}

impl RollingMetrics {
    fn callback_mut(&mut self, callback: RollingCallback) -> &mut RollingCallbackMetrics {
        match callback {
            RollingCallback::Data => &mut self.data,
            RollingCallback::Watermark => &mut self.watermark,
            RollingCallback::End => &mut self.end,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum RollingStage {
    InputValidation,
    OrderingProof,
    EntityResolution,
    StatePreparation,
    NumericUpdate,
    HistoryMaintenance,
    ArrowOutput,
    BudgetPreparation,
    SendWait,
    Other,
}

#[derive(Clone, Copy)]
pub(crate) enum RollingWork {
    InputRows,
    OrderProofRows,
    ResolvedRows,
    TouchedEntities,
    CopiedEntities,
    NumericRows,
    HistoryRowsMaterialized,
    ScalarValueConversions,
    OutputRowsPrepared,
    OutputChunksPrepared,
}

pub(crate) trait RollingMetricsClock: Send + Sync {
    fn now(&self) -> Duration;
}

struct MonotonicClock(Instant);

impl RollingMetricsClock for MonotonicClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
}

struct StoreInner {
    metrics: Mutex<RollingMetrics>,
    clock: Arc<dyn RollingMetricsClock>,
}

#[derive(Clone)]
pub(crate) struct RollingMetricsStore(Arc<StoreInner>);

impl Default for RollingMetricsStore {
    fn default() -> Self {
        Self::with_clock(Arc::new(MonotonicClock(Instant::now())))
    }
}

impl RollingMetricsStore {
    pub(crate) fn with_clock(clock: Arc<dyn RollingMetricsClock>) -> Self {
        Self(Arc::new(StoreInner {
            metrics: Mutex::new(RollingMetrics::default()),
            clock,
        }))
    }

    pub(crate) fn snapshot(&self) -> RollingMetrics {
        self.0.metrics.lock().clone()
    }

    pub(crate) fn begin(
        &self,
        callback: RollingCallback,
        cancellation: CancellationToken,
    ) -> RollingCallbackGuard {
        let mut snapshot = self.0.metrics.lock();
        if !snapshot.overflowed {
            let current = snapshot.callback_mut(callback);
            if let Some(started) = current.started.checked_add(1) {
                current.started = started;
            } else {
                snapshot.overflowed = true;
            }
        }
        drop(snapshot);
        let now = self.0.clock.now();
        RollingCallbackGuard {
            recorder: RollingMetricsRecorder(Arc::new(RecorderInner {
                observation: Mutex::new(Observation {
                    started: now,
                    switched: now,
                    stage: RollingStage::Other,
                    durations: [Duration::ZERO; 10],
                    work: [0; 10],
                    overflowed: false,
                    finalized: false,
                    outcome: None,
                    outstanding_work: None,
                }),
                store: self.clone(),
                callback,
            })),
            cancellation,
        }
    }
}

struct Observation {
    started: Duration,
    switched: Duration,
    stage: RollingStage,
    durations: [Duration; 10],
    work: [u64; 10],
    overflowed: bool,
    finalized: bool,
    outcome: Option<CallbackOutcome>,
    outstanding_work: Option<u64>,
}

impl Observation {
    fn switch(&mut self, stage: RollingStage, now: Duration) -> RollingStage {
        let previous = self.stage;
        if !self.finalized && !self.overflowed {
            let next = now
                .checked_sub(self.switched)
                .and_then(|elapsed| self.durations[previous as usize].checked_add(elapsed));
            if let Some(next) = next {
                self.durations[previous as usize] = next;
                self.switched = now;
                self.stage = stage;
            } else {
                self.overflowed = true;
            }
        }
        previous
    }
}

struct RecorderInner {
    observation: Mutex<Observation>,
    store: RollingMetricsStore,
    callback: RollingCallback,
}

#[derive(Clone)]
pub(crate) struct RollingMetricsRecorder(Arc<RecorderInner>);

impl RollingMetricsRecorder {
    pub(crate) fn stage(&self, stage: RollingStage) -> RollingStageGuard<'_> {
        let mut observation = self.0.observation.lock();
        let previous = if observation.outcome.is_none() {
            observation.switch(stage, self.0.store.0.clock.now())
        } else {
            observation.stage
        };
        RollingStageGuard {
            recorder: self,
            previous,
        }
    }

    pub(crate) fn add(&self, work: RollingWork, amount: usize) {
        let mut observation = self.0.observation.lock();
        if observation.outcome.is_some() || observation.overflowed {
            return;
        }
        let next = u64::try_from(amount)
            .ok()
            .and_then(|amount| observation.work[work as usize].checked_add(amount));
        if let Some(next) = next {
            observation.work[work as usize] = next;
        } else {
            observation.overflowed = true;
        }
    }

    pub(crate) fn numeric_lease(&self, work_id: u64) -> Option<NumericWorkLease> {
        let mut observation = self.0.observation.lock();
        if observation.outcome.is_some() || observation.outstanding_work.is_some() {
            return None;
        }
        observation.outstanding_work = Some(work_id);
        Some(NumericWorkLease {
            recorder: self.clone(),
            work_id,
        })
    }

    fn freeze(&self, outcome: CallbackOutcome) {
        let mut observation = self.0.observation.lock();
        if observation.outcome.is_some() {
            return;
        }
        observation.switch(RollingStage::Other, self.0.store.0.clock.now());
        observation.outcome = Some(outcome);
        self.publish_ready(&mut observation);
    }

    fn publish_ready(&self, observation: &mut Observation) {
        let Some(outcome) = observation.outcome else {
            return;
        };
        if observation.finalized || observation.outstanding_work.is_some() {
            return;
        }
        let now = self.0.store.0.clock.now();
        observation.switch(RollingStage::Other, now);
        observation.finalized = true;
        let mut snapshot = self.0.store.0.metrics.lock();
        if snapshot.overflowed {
            return;
        }
        let next = (!observation.overflowed)
            .then(|| {
                accumulate(
                    snapshot.callback_mut(self.0.callback),
                    observation,
                    now.checked_sub(observation.started)?,
                    outcome,
                )
            })
            .flatten();
        if let Some(next) = next {
            *snapshot.callback_mut(self.0.callback) = next;
        } else {
            snapshot.overflowed = true;
        }
    }
}

/// Only the owner of the joined CPU handles may settle this generation.
/// Dropping a lease does not assert that its work finished.
pub(crate) struct NumericWorkLease {
    recorder: RollingMetricsRecorder,
    work_id: u64,
}

impl NumericWorkLease {
    pub(crate) fn settle(self, numeric_rows: u64, overflowed: bool) {
        let mut observation = self.recorder.0.observation.lock();
        if observation.outstanding_work != Some(self.work_id) {
            return;
        }
        let next = observation.work[RollingWork::NumericRows as usize].checked_add(numeric_rows);
        observation.overflowed |= overflowed || next.is_none();
        if let Some(next) = next {
            observation.work[RollingWork::NumericRows as usize] = next;
        }
        observation.outstanding_work = None;
        self.recorder.publish_ready(&mut observation);
    }
}

pub(crate) struct RollingStageGuard<'a> {
    recorder: &'a RollingMetricsRecorder,
    previous: RollingStage,
}

impl Drop for RollingStageGuard<'_> {
    fn drop(&mut self) {
        let mut observation = self.recorder.0.observation.lock();
        if observation.outcome.is_none() {
            observation.switch(self.previous, self.recorder.0.store.0.clock.now());
        }
    }
}

#[derive(Clone, Copy)]
enum CallbackOutcome {
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
}

pub(crate) struct RollingCallbackGuard {
    recorder: RollingMetricsRecorder,
    cancellation: CancellationToken,
}

impl RollingCallbackGuard {
    pub(crate) fn recorder(&self) -> RollingMetricsRecorder {
        self.recorder.clone()
    }

    pub(crate) fn complete(self, result: &Result<()>) {
        self.recorder.freeze(match result {
            Ok(()) => CallbackOutcome::Succeeded,
            Err(CalcFlowError::Cancelled { .. }) => CallbackOutcome::Cancelled,
            Err(_) => CallbackOutcome::Failed,
        });
    }
}

impl Drop for RollingCallbackGuard {
    fn drop(&mut self) {
        self.recorder.freeze(if self.cancellation.is_cancelled() {
            CallbackOutcome::Cancelled
        } else {
            CallbackOutcome::Interrupted
        });
    }
}

fn accumulate(
    current: &RollingCallbackMetrics,
    observation: &Observation,
    elapsed: Duration,
    outcome: CallbackOutcome,
) -> Option<RollingCallbackMetrics> {
    let mut next = current.clone();
    let outcome_count = match outcome {
        CallbackOutcome::Succeeded => &mut next.succeeded,
        CallbackOutcome::Failed => &mut next.failed,
        CallbackOutcome::Cancelled => &mut next.cancelled,
        CallbackOutcome::Interrupted => &mut next.interrupted,
    };
    *outcome_count = outcome_count.checked_add(1)?;
    next.callback_duration = next.callback_duration.checked_add(elapsed)?;
    macro_rules! add_stages {
        ($($field:ident => $stage:ident),+ $(,)?) => {
            $(next.$field = next.$field.checked_add(observation.durations[RollingStage::$stage as usize])?;)+
        };
    }
    add_stages! {
        input_validation_duration => InputValidation,
        ordering_proof_duration => OrderingProof,
        entity_resolution_duration => EntityResolution,
        state_preparation_duration => StatePreparation,
        numeric_update_duration => NumericUpdate,
        history_maintenance_duration => HistoryMaintenance,
        arrow_output_duration => ArrowOutput,
        budget_preparation_duration => BudgetPreparation,
        send_wait_duration => SendWait,
        other_duration => Other,
    }
    macro_rules! add_work {
        ($($field:ident => $work:ident),+ $(,)?) => {
            $(next.$field = next.$field.checked_add(observation.work[RollingWork::$work as usize])?;)+
        };
    }
    add_work! {
        input_rows => InputRows,
        order_proof_rows => OrderProofRows,
        resolved_rows => ResolvedRows,
        touched_entities => TouchedEntities,
        copied_entities => CopiedEntities,
        numeric_rows => NumericRows,
        history_rows_materialized => HistoryRowsMaterialized,
        scalar_value_conversions => ScalarValueConversions,
        output_rows_prepared => OutputRowsPrepared,
        output_chunks_prepared => OutputChunksPrepared,
    }
    Some(next)
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use parking_lot::Mutex;

    use super::{
        RollingCallback, RollingMetricsClock, RollingMetricsStore, RollingStage, RollingWork,
    };
    use crate::{CalcFlowError, CancellationToken};

    #[derive(Default)]
    struct TestClock(Mutex<Duration>);

    impl TestClock {
        fn advance(&self, nanos: u64) {
            *self.0.lock() += Duration::from_nanos(nanos);
        }
    }

    impl RollingMetricsClock for TestClock {
        fn now(&self) -> Duration {
            *self.0.lock()
        }
    }

    #[test]
    fn rolling_metrics_nested_stages_partition_callback_and_snapshot_work() {
        let clock = Arc::new(TestClock::default());
        let store = RollingMetricsStore::with_clock(clock.clone());
        let callback = store.begin(RollingCallback::Data, CancellationToken::new());
        let recorder = callback.recorder();
        clock.advance(2);
        {
            let _validation = recorder.stage(RollingStage::InputValidation);
            recorder.add(RollingWork::InputRows, 7);
            clock.advance(3);
            {
                let _ordering = recorder.stage(RollingStage::OrderingProof);
                recorder.add(RollingWork::OrderProofRows, 5);
                clock.advance(5);
            }
            clock.advance(7);
        }
        clock.advance(11);
        let active = store.snapshot();
        assert_eq!(active.data.started, 1);
        assert_eq!(active.data.succeeded, 0);
        assert_eq!(active.data.input_rows, 0);
        callback.complete(&Ok(()));
        let done = store.snapshot();
        assert_eq!(done.data.started, 1);
        assert_eq!(done.data.succeeded, 1);
        assert_eq!(done.data.callback_duration, Duration::from_nanos(28));
        assert_eq!(
            done.data.input_validation_duration,
            Duration::from_nanos(10)
        );
        assert_eq!(done.data.ordering_proof_duration, Duration::from_nanos(5));
        assert_eq!(done.data.other_duration, Duration::from_nanos(13));
        assert_eq!(done.data.input_rows, 7);
        assert_eq!(done.data.order_proof_rows, 5);
        assert_eq!(active.data.succeeded, 0);
        drop(recorder);
        assert_eq!(store.snapshot(), done);
    }

    #[test]
    fn rolling_metrics_outcomes_drop_and_explicit_results_publish_once() {
        let store = RollingMetricsStore::default();
        let cancellation = CancellationToken::new();
        let failed = store.begin(RollingCallback::Watermark, cancellation.clone());
        cancellation.cancel();
        failed.complete(&Err(CalcFlowError::Internal {
            message: "expected".into(),
        }));
        let succeeded = store.begin(RollingCallback::End, cancellation.clone());
        succeeded.complete(&Ok(()));
        drop(store.begin(RollingCallback::Data, cancellation));
        drop(store.begin(RollingCallback::End, CancellationToken::new()));
        let explicit_cancel = store.begin(RollingCallback::Watermark, CancellationToken::new());
        explicit_cancel.complete(&Err(CalcFlowError::Cancelled { run_id: "1".into() }));
        let snapshot = store.snapshot();
        assert_eq!(snapshot.data.cancelled, 1);
        assert_eq!(snapshot.watermark.failed, 1);
        assert_eq!(snapshot.watermark.cancelled, 1);
        assert_eq!(snapshot.end.succeeded, 1);
        assert_eq!(snapshot.end.interrupted, 1);
        assert_eq!(snapshot.end.started, 2);
    }

    #[test]
    fn rolling_metrics_overflow_keeps_the_previous_whole_observation() {
        for duration_overflow in [false, true] {
            let store = RollingMetricsStore::default();
            if duration_overflow {
                store.0.metrics.lock().data.callback_duration = Duration::MAX;
            } else {
                store.0.metrics.lock().data.input_rows = u64::MAX;
            }
            let callback = store.begin(RollingCallback::Data, CancellationToken::new());
            callback.recorder().add(RollingWork::InputRows, 1);
            let mut previous = store.snapshot();
            callback.complete(&Ok(()));
            previous.overflowed = true;
            assert_eq!(store.snapshot(), previous);
            store
                .begin(RollingCallback::Data, CancellationToken::new())
                .complete(&Ok(()));
            assert_eq!(store.snapshot(), previous);
        }
    }

    #[test]
    fn rolling_metrics_local_overflow_or_bad_clock_invalidates_without_partial_publication() {
        for counter_overflow in [false, true] {
            let clock = Arc::new(TestClock(Mutex::new(Duration::from_nanos(5))));
            let store = RollingMetricsStore::with_clock(clock.clone());
            let callback = store.begin(RollingCallback::Watermark, CancellationToken::new());
            let recorder = callback.recorder();
            if counter_overflow {
                recorder.0.observation.lock().work[RollingWork::NumericRows as usize] = u64::MAX;
                recorder.add(RollingWork::NumericRows, 1);
            } else {
                let _numeric = recorder.stage(RollingStage::NumericUpdate);
                *clock.0.lock() = Duration::ZERO;
            }
            let mut previous = store.snapshot();
            callback.complete(&Err(CalcFlowError::Internal {
                message: "original".into(),
            }));
            previous.overflowed = true;
            assert_eq!(store.snapshot(), previous);
        }
    }

    #[test]
    fn rolling_metrics_started_overflow_and_finalized_handles_remain_inert() {
        let store = RollingMetricsStore::default();
        store.0.metrics.lock().end.started = u64::MAX;
        let callback = store.begin(RollingCallback::End, CancellationToken::new());
        let recorder = callback.recorder();
        callback.complete(&Ok(()));
        let snapshot = store.snapshot();
        assert!(snapshot.overflowed);
        assert_eq!(snapshot.end.started, u64::MAX);
        assert_eq!(snapshot.end.succeeded, 0);
        recorder.add(RollingWork::InputRows, 1);
        drop(recorder.stage(RollingStage::NumericUpdate));
        assert_eq!(store.snapshot(), snapshot);
    }

    #[test]
    fn rolling_metrics_completed_callbacks_release_their_private_recorders() {
        let store = RollingMetricsStore::default();
        for _ in 0..128 {
            let callback = store.begin(RollingCallback::Data, CancellationToken::new());
            let recorder = callback.recorder();
            recorder.add(RollingWork::NumericRows, 1);
            callback.complete(&Ok(()));
            drop(recorder);
            assert_eq!(Arc::strong_count(&store.0), 1);
        }
        assert_eq!(store.snapshot().data.succeeded, 128);
        assert_eq!(store.snapshot().data.numeric_rows, 128);
    }
}
